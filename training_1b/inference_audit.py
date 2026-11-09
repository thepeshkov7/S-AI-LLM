"""Check an original S AI 1.276B checkpoint without changing its tensors."""

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from generate import HEADS, TextGenerator
from model import CopyLLM, checkpoint_architecture
from packed_corpus import digest
from training_1b.checkpoint import RECOVERY_CONFIG, RECOVERY_PARAMETERS, read_checkpoint
from training_1b.evaluate import DecodeCache


PROMPTS = (
    "Сегодня утром я открыл окно, и в комнату вошёл свежий воздух.",
    "In the morning, I opened the window and took a deep breath.",
)
LOSS_CALLS = {"vocabulary_loss", "answer_loss", "cross_entropy", "nll_loss", "forward", "model"}


def trainer_evidence(path):
    source = ast.parse(path.read_bytes(), filename=str(path))
    calls = []
    for node in ast.walk(source):
        if isinstance(node, ast.Call):
            target = ast.unparse(node.func)
            if target.rsplit(".", 1)[-1] in LOSS_CALLS:
                calls.append({"line": node.lineno, "call": target})
    return {"sha256": digest(path), "candidate_calls": sorted(calls, key=lambda item: item["line"])}


def weight_checks(state, vocab_size):
    aliases = state["copy_aliases"]
    if aliases.dtype != torch.long or aliases.min().item() < 0 or aliases.max().item() >= vocab_size:
        raise ValueError("copy_aliases must contain valid int64 vocabulary IDs")
    bad = [key for key, value in state.items() if value.is_floating_point()
           and not all(torch.isfinite(chunk).all().item() for chunk in value.flatten().split(2**20))]
    if bad:
        raise FloatingPointError(f"Nonfinite checkpoint tensors: {bad}")
    return {"finite_tensors": len(state), "copy_aliases_valid": True}


def tokenizer_checks(generator):
    tokenizer = generator.tokenizer
    special = {item["content"]: item["id"] for item in json.loads(tokenizer.to_str())["added_tokens"]
               if item["special"]}
    rows = []
    for prompt in PROMPTS:
        raw = tokenizer.encode(prompt, add_special_tokens=False).ids
        templated = tokenizer.encode(prompt, add_special_tokens=True)
        decoded = tokenizer.decode(raw, skip_special_tokens=False)
        rows.append({"prompt": prompt, "raw_ids": raw, "pipeline_ids": generator.encode_prompt(prompt),
                     "template_added_ids": [token for token, is_special in
                                            zip(templated.ids, templated.special_tokens_mask, strict=True) if is_special],
                     "text_round_trip": decoded == prompt,
                     "id_round_trip": tokenizer.encode(decoded, add_special_tokens=False).ids == raw})
    return {"vocab_size": tokenizer.get_vocab_size(), "eos": generator.eos, "special_tokens": special, "rows": rows}


def compare(left, right, *, atol, rtol):
    return {"max_abs": (left.float() - right.float()).abs().max().item(),
            "close": torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol),
            "same_argmax": torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))}


@torch.inference_mode()
def score_checks(generator, device, *, bf16=False):
    model = generator.model
    windows = [torch.tensor(generator.encode_prompt(prompt)[:min(size, model.context - 1)], device=device, dtype=torch.long)
               for size, prompt in zip((7, 12), PROMPTS, strict=True)]
    rows = torch.arange(len(windows), device=device)
    batch = pad_sequence(windows, batch_first=True, padding_value=generator.eos)
    positions = torch.tensor([len(window) - 1 for window in windows], device=device)
    atol, rtol = (0.1, 0.01) if bf16 else (0.0001, 0.0001)
    result = {}
    with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
        cache = DecodeCache(model, windows, max(map(len, windows)) + 1, generator.eos)
        changed = batch.clone()
        changed[:, -1] = (changed[:, -1] + 1) % model.config.vocab_size
        for head in ("vocabulary", "copy"):
            vocab = head == "vocabulary"
            full = model.vocabulary_logits if vocab else lambda ids: model(ids)[0]
            step = model.vocabulary_scores if vocab else model.next_scores
            dense = full(batch)
            result[head] = {
                "forward_vs_step": compare(dense[rows, positions], step(batch, positions), atol=atol, rtol=rtol),
                "future_token_change": compare(dense[:, :-1], full(changed)[:, :-1], atol=atol, rtol=rtol),
            }
        tokens = generator.next_scores(batch, positions).argmax(dim=-1).tolist()
        cache.advance(tokens)
        extended = [torch.cat((window, window.new_tensor([token]))) for window, token in
                    zip(windows, tokens, strict=True)]
        batch = pad_sequence(extended, batch_first=True, padding_value=generator.eos)
        for head in ("vocabulary", "copy"):
            vocab = head == "vocabulary"
            step = model.vocabulary_scores if vocab else model.next_scores
            result[head]["kv_cache_vs_step"] = compare(cache.scores(vocabulary_only=vocab),
                                                       step(batch, positions + 1), atol=atol, rtol=rtol)
    return result


@torch.inference_mode()
def decoder_checks(generator, device, limit=8):
    model = generator.model
    rows = []
    for prompt in PROMPTS:
        ids = generator.encode_prompt(prompt)
        tokens = []
        for _ in range(limit):
            inputs = torch.tensor([(ids + tokens)[-model.context:]], device=device, dtype=torch.long)
            scores = model.vocabulary_logits(inputs) if generator.vocabulary_only else model(inputs)[0]
            token = scores[0, -1].argmax().item()
            if token == generator.eos:
                break
            tokens.append(token)
        expected = generator.tokenizer.decode(tokens, skip_special_tokens=False)
        plain = {"repetition_penalty": 1.0, "frequency_penalty": 0.0, "no_repeat_ngram_size": 0,
                 "preserve_output": True}
        actual, count = generator.complete(prompt, limit, top_k=1, **plain)
        sampling = {"top_k": 40, "top_p": 0.95, "seed": 42, **plain}
        sampled = generator.complete(prompt, limit, **sampling)
        rows.append({"prompt": prompt, "reference_token_ids": tokens, "greedy": actual,
                     "matches_forward_argmax": (actual, count) == (expected, len(tokens)),
                     "sample": sampled[0], "same_seed_same_output": sampled == generator.complete(prompt, limit, **sampling)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--head", choices=HEADS, default="auto")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    if args.bf16 and args.device == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        parser.error("This CUDA device has no native BF16 support")
    torch.set_num_threads(4)
    payload = read_checkpoint(args.checkpoint)
    if checkpoint_architecture(payload) != CopyLLM.architecture or payload.get("config") != asdict(RECOVERY_CONFIG):
        raise ValueError("Expected the exact S AI 1.276B CopyLLM configuration")
    weights = weight_checks(payload["model"], RECOVERY_CONFIG.vocab_size)
    generator = TextGenerator.from_checkpoint(payload, args.device, head=args.head)
    del payload
    parameters = sum(parameter.numel() for parameter in generator.model.parameters())
    if parameters != RECOVERY_PARAMETERS:
        raise ValueError("Unexpected CopyLLM parameter count")
    report = {"checkpoint_sha256": digest(args.checkpoint), "parameters": parameters, "config": asdict(RECOVERY_CONFIG),
              "strict_state_dict": {"missing_keys": [], "unexpected_keys": [], **weights},
              "torch_version": str(torch.__version__), "device": args.device,
              "head": "vocabulary" if generator.vocabulary_only else "copy", "head_source": generator.head_source,
              "trainer": trainer_evidence(args.trainer) if args.trainer else None,
              "tokenizer": tokenizer_checks(generator), "fp32": score_checks(generator, args.device),
              "decoder": decoder_checks(generator, args.device)}
    if args.bf16:
        report["bf16_autocast"] = score_checks(generator, args.device, bf16=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Audit saved to {args.output}")


if __name__ == "__main__":
    main()
