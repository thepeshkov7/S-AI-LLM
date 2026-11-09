from dataclasses import asdict
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import torch
from torch import nn

from api import app
from generate import TextGenerator, nucleus_probabilities
from model import CopyLLM, TinyLLM
from span_chat import DIVIDE, SUBTRACT_REVERSED, calculate, integer_operands, prepare_prompt, span_value, target_span
from train import Document, save_checkpoint, train_tokenizer


class InferenceTest(unittest.TestCase):
    def test_attention_is_causal(self):
        torch.manual_seed(8)
        model = TinyLLM(width=16, layers=1, context=8).eval()
        with torch.inference_mode():
            first, _ = model(torch.tensor([[1, 2, 3, 4]]))
            second, _ = model(torch.tensor([[1, 2, 7, 8]]))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_copy_attention_is_causal_and_training_matches_generation(self):
        torch.manual_seed(13)
        model = CopyLLM(vocab_size=32, width=16, layers=1, context=8)
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        targets = torch.tensor([[-100, 2, 4, 2], [-100, 3, 1, 3]])
        scores, full_loss = model(ids, targets)
        changed = ids.clone()
        changed[:, 2:] = 7
        changed_scores, _ = model(changed)
        torch.testing.assert_close(scores[:, :2], changed_scores[:, :2])
        torch.testing.assert_close(scores.exp().sum(dim=-1), torch.ones_like(ids, dtype=torch.float))
        positions = torch.tensor([1, 3])
        torch.testing.assert_close(model.next_scores(ids, positions), scores[torch.arange(2), positions])
        masked_loss = model.answer_loss(ids, targets)
        torch.testing.assert_close(masked_loss, full_loss)
        masked_loss.backward()
        self.assertTrue(torch.isfinite(model.copy_query.weight.grad).all())

    def test_copy_checkpoint_reloads_trained_aliases(self):
        tokenizer = train_tokenizer([Document("fixture", "en", "green green red red")], 280)
        model = CopyLLM(vocab_size=tokenizer.get_vocab_size(), width=16, layers=1, context=8)
        model.set_copy_aliases(tokenizer)
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "copy.pt"
            save_checkpoint(checkpoint, model, tokenizer, {"bilingual_chat": True})
            loaded = TextGenerator.load(checkpoint)
            self.assertIsInstance(loaded.model, CopyLLM)
            self.assertEqual(loaded.chat_model, "S AI 0.5 Grounded")
            torch.testing.assert_close(loaded.model.copy_aliases, model.copy_aliases)

    def test_unversioned_copy_checkpoint_loads_strictly(self):
        tokenizer = train_tokenizer([Document("fixture", "en", "green green red red")], 280)
        model = CopyLLM(vocab_size=tokenizer.get_vocab_size(), width=16, layers=1, context=8)
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "step-010000.pt"
            torch.save({"config": asdict(model.config), "model": model.state_dict(),
                        "tokenizer": tokenizer.to_str()}, checkpoint)
            loaded = TextGenerator.load(checkpoint)
            self.assertIsInstance(loaded.model, CopyLLM)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(loaded.model.state_dict()[key], tensor)

    def test_generation_blocks_repeated_trigrams_before_top_k(self):
        model = TinyLLM(width=16, layers=1, context=8).eval()
        scores = torch.zeros(1, 256)
        scores[0, ord("A")] = 10
        scores[0, ord("B")] = 9
        with patch.object(model, "next_scores", side_effect=lambda _: scores.clone()):
            text, _ = TextGenerator(model).complete("AAAA", 4, top_k=1, no_repeat_ngram_size=3)
        self.assertEqual(text, "AAAB")

    def test_nucleus_keeps_crossing_token_and_always_keeps_first(self):
        probabilities = torch.tensor([0.5, 0.3, 0.2])
        torch.testing.assert_close(nucleus_probabilities(probabilities, 0.6), torch.tensor([0.5, 0.3, 0.0]))
        torch.testing.assert_close(nucleus_probabilities(probabilities, 0.2), torch.tensor([0.5, 0.0, 0.0]))

    def test_frequency_penalty_counts_generated_tokens(self):
        model = TinyLLM(width=16, layers=1, context=8).eval()
        scores = torch.zeros(1, 256)
        scores[0, ord("A")] = 10
        scores[0, ord("B")] = 8.5
        with patch.object(model, "next_scores", side_effect=lambda _: scores.clone()):
            output, _ = TextGenerator(model).complete("A", 4, top_k=1, frequency_penalty=1)
        self.assertEqual(output, "AABA")

    def test_vocab_loss_ignores_masked_targets_under_bf16(self):
        model = CopyLLM(vocab_size=32, width=16, heads=2, layers=1, context=8)
        ids = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, -100, 4, 5]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _ = TinyLLM.forward(model, ids)
            loss = model.vocabulary_loss(ids, targets, reduction="sum")
        reference = torch.nn.functional.cross_entropy(
            logits[targets != -100].float(), targets[targets != -100], reduction="sum"
        )
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

    def test_span_decoder_preserves_unseen_words_and_rejects_word_fragments(self):
        tokenizer = train_tokenizer([Document("fixture", "ru", "На столе лежит книга.")], 270)
        model = TinyLLM(vocab_size=tokenizer.get_vocab_size(), width=16, layers=1, context=128)
        generator = TextGenerator(model, tokenizer)
        prompt = prepare_prompt("Амина нашла тетрадь, а Ингрид нашла паспорт.", generator)
        start, end = target_span(prompt, "Ингрид")
        logits = torch.zeros(2, len(prompt.ids))
        logits[0, start] = logits[1, end] = 10
        self.assertEqual(span_value(prompt, logits), "Ингрид")
        self.assertIsNone(target_span(prompt, "нгрид"))

    def test_arithmetic_decoder_requires_two_supported_integers(self):
        numbers = integer_operands("Вычти 4 из 25.")
        self.assertEqual(numbers, (4, 25))
        self.assertEqual(calculate(SUBTRACT_REVERSED, numbers), 21)
        self.assertEqual(calculate(DIVIDE, (28, 4)), 7)
        self.assertIsNone(calculate(DIVIDE, (28, 3)))
        self.assertIsNone(calculate(DIVIDE, (28, 0)))
        for text in ("10:15 and 16:20", "Add 1.5 and 3", "8 and -2", "Add 1, 2 and 3"):
            self.assertIsNone(integer_operands(text))

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
