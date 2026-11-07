import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import torch
from torch import nn

from api import app
from generate import TextGenerator
from model import TinyLLM


class InferenceTest(unittest.TestCase):
    def test_attention_is_causal(self):
        torch.manual_seed(8)
        model = TinyLLM(width=16, layers=1, context=8).eval()
        with torch.inference_mode():
            first, _ = model(torch.tensor([[1, 2, 3, 4]]))
            second, _ = model(torch.tensor([[1, 2, 7, 8]]))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_original_checkpoint_still_loads(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            model = TinyLLM(context=8).eval()
            original = model.state_dict()
            for index in range(model.config.layers):
                prefix = f"blocks.{index}."
                original[prefix + "attn.in_proj_weight"] = original.pop(prefix + "qkv.weight")
                original[prefix + "attn.in_proj_bias"] = original.pop(prefix + "qkv.bias")
                original[prefix + "attn.out_proj.weight"] = original.pop(prefix + "proj.weight")
                original[prefix + "attn.out_proj.bias"] = original.pop(prefix + "proj.bias")
            torch.save({"model": original, "context": 8}, checkpoint)
            loaded = TextGenerator.load(checkpoint)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.model.state_dict()[key])

    def test_attention_matches_original_projection(self):
        model = TinyLLM(width=16, heads=4, layers=1, context=8).eval()
        block = model.blocks[0]
        reference = nn.MultiheadAttention(16, 4, batch_first=True).eval()
        with torch.inference_mode():
            reference.in_proj_weight.copy_(block.qkv.weight)
            reference.in_proj_bias.copy_(block.qkv.bias)
            reference.out_proj.load_state_dict(block.proj.state_dict())
            inputs = torch.randn(2, 5, 16)
            normalized = block.ln1(inputs)
            mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
            attention, _ = reference(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
            expected = inputs + attention
            expected = expected + block.ff(block.ln2(expected))
            torch.testing.assert_close(block(inputs), expected)

    def test_api_reports_missing_weights(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "SAI_CHECKPOINT": str(Path(directory) / "missing.pt"),
            "SAI_API_TOKEN": "temporary-test-token",
        }):
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 503)
                self.assertEqual(client.post("/generate", json={"prompt": "Hello"}).status_code, 401)
                response = client.post(
                    "/generate", json={"prompt": "Hello"},
                    headers={"Authorization": "Bearer temporary-test-token"},
                )
                self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
