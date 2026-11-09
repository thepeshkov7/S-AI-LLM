"""Safely open and audit original CopyLLM checkpoints."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

from model import CopyLLM, ModelConfig, checkpoint_architecture
from packed_corpus import digest
from train import EOS

CONFIG = ModelConfig(vocab_size=6144, width=2048, heads=16, layers=20, context=256)
PARAMETERS = 1_033_003_011
RECOVERY_CONFIG = ModelConfig(vocab_size=65536, width=2048, heads=16, layers=20, context=256)
RECOVERY_PARAMETERS = 1_276_332_035


def read_checkpoint(path):
    # Requires a local ZIP-format torch.save file. Stage Drive files onto the runtime disk.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=sys.platform != "win32")


def read_tokenizer(payload):
    tokenizer = Tokenizer.from_str(payload["tokenizer"])
    config = ModelConfig(**payload["config"])
    if tokenizer.get_vocab_size() != config.vocab_size or tokenizer.token_to_id(EOS) is None:
        raise ValueError("The checkpoint tokenizer does not match its vocabulary or is missing EOS")
    return tokenizer


def build_model(payload, device, dtype=torch.float32, *, require_1b=True, recovery=False):
    config = ModelConfig(**payload["config"])
    if checkpoint_architecture(payload) != CopyLLM.architecture:
        raise ValueError("Expected a CopyLLM checkpoint")
    if recovery and config != RECOVERY_CONFIG:
        raise ValueError("Expected the original S AI 1.276B CopyLLM configuration")
    if require_1b and not recovery and config != CONFIG:
        raise ValueError("Expected a checkpoint of the original 1B copy_v1 model")
    with torch.device("meta"):
        model = CopyLLM(**asdict(config))
    parameters = sum(p.numel() for p in model.parameters())
    if recovery and parameters != RECOVERY_PARAMETERS:
        raise ValueError("Unexpected S AI 1.276B parameter count")
    if require_1b and not recovery and parameters != PARAMETERS:
        raise ValueError("Unexpected 1B parameter count")
    model.load_state_dict(payload["model"], strict=True, assign=True)
    return model.to(device=device, dtype=dtype)


def load_head(model, payload):
    if payload.get("config") != asdict(model.config):
        raise ValueError("Output head has a different model configuration")
    head = payload["output_head"]
    if not head or any(not torch.isfinite(value).all().item() for value in head.values()):
        raise ValueError("Output head has non-finite weights")
    model.output.load_state_dict(head, strict=True)


def audit(base_path, previous_path, heads):
    base = read_checkpoint(base_path)
    previous = read_checkpoint(previous_path)
    model = build_model(base, "cpu", torch.float16)
    tokenizer = read_tokenizer(base)
    counts = Counter()
    bad_tensors = []
    # Chunk to bound temporary memory for the FP16 finite-value scan.
    for name, param in model.named_parameters():
        counts[str(param.dtype)] += param.numel()
        if not all(torch.isfinite(chunk).all().item() for chunk in param.flatten().split(2**20)):
            bad_tensors.append(name)
    copied = Counter()
    for name, parameter in model.named_parameters():
        source = previous["model"].get(name)
        if source is not None and source.ndim == parameter.ndim:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, parameter.shape))
            expected = source[slices].to(parameter.dtype)
            copied["checked"] += expected.numel()
            copied["equal"] += (expected == parameter[slices]).sum().item()
    aliases = model.copy_aliases.clone()
    if not (aliases.min().item() >= 0 and aliases.max().item() < CONFIG.vocab_size):
        raise ValueError("Copy alias IDs are outside the vocabulary")
    model.set_copy_aliases(tokenizer)
    out = {
        "architecture": model.architecture, "config": base["config"],
        "parameters_by_dtype": dict(counts), "nonfinite_base_tensors": bad_tensors,
        "base_sha256": digest(base_path), "previous_sha256": digest(previous_path),
        "copied": dict(copied),
        "tokenizer_matches_previous": tokenizer.to_str() == read_tokenizer(previous).to_str(),
        "aliases_match_previous": torch.equal(aliases, previous["model"]["copy_aliases"]),
        "aliases_match_tokenizer": torch.equal(aliases, model.copy_aliases),
        "heads": {},
    }
    for path in heads:
        payload = read_checkpoint(path)
        out["heads"][path.name] = {
            "step": payload.get("step", payload.get("training_steps")),
            "best_val": payload.get("best_val"),
            "sha256": digest(path),
            "tensors": {k: {"dtype": str(v.dtype), "shape": list(v.shape),
                            "finite": torch.isfinite(v).sum().item(), "values": v.numel()}
                        for k, v in payload["output_head"].items()},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--head", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.base, args.previous, args.head)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
