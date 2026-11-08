"""Generate raw CopyLLM answers on bilingual probes without an external model."""
import argparse
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from generate import format_instruction
from packed_corpus import digest
from train import EOS
from training_1b.checkpoint import build_model, load_head, read_checkpoint, read_tokenizer


PROBES = Path(__file__).with_name("prompts.jsonl")


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


class DecodeCache:
    """Reuse keys and values while the probe stays within learned positions."""

    def __init__(self, model, windows, capacity, eos):
        self.model = model
        batch = pad_sequence(windows, batch_first=True, padding_value=eos)
        size, self.width = batch.shape
        self.rows = torch.arange(size, device=batch.device)
        self.positions = batch.new_tensor([len(window) - 1 for window in windows])
        self.ids = batch.new_full((size, capacity), eos)
        self.ids[:, :self.width] = batch
        self.indices = torch.arange(capacity, device=batch.device)
        hidden = model.token(batch) + model.position(self.indices[:self.width])[None]
        self.layers = []
        for block in model.blocks:
            q, k, v = block.attention_projections(hidden)
            shape = (size, k.shape[1], capacity, k.shape[-1])
            keys, values = k.new_zeros(shape), v.new_zeros(shape)
            keys[:, :, :self.width], values[:, :, :self.width] = k, v
            self.layers.append((keys, values))
            hidden = block.attention_residual(hidden, F.scaled_dot_product_attention(q, k, v, is_causal=True))
        self.hidden = hidden.new_zeros((size, capacity, hidden.shape[-1]))
        self.hidden[:, :self.width] = model.norm(hidden)

    def scores(self):
        hidden = self.hidden[:, :self.width]
        states = hidden[self.rows, self.positions]
        attention = self.model._copy_attention(states[:, None], hidden, self.positions[:, None])[:, 0]
        return self.model._mixture(states, self.ids[:, :self.width], attention)

    def advance(self, tokens):
        self.width += 1
        self.positions += 1
        ids = self.ids.new_tensor(tokens)
        self.ids[self.rows, self.positions] = ids
        hidden = (self.model.token(ids) + self.model.position(self.positions))[:, None]
        visible = self.indices[:self.width] <= self.positions[:, None]
        for block, (keys, values) in zip(self.model.blocks, self.layers, strict=True):
            q, k, v = block.attention_projections(hidden)
            keys[self.rows, :, self.positions] = k[:, :, 0]
            values[self.rows, :, self.positions] = v[:, :, 0]
            attention = F.scaled_dot_product_attention(q, keys[:, :, :self.width], values[:, :, :self.width],
                                                       attn_mask=visible[:, None, None])
            hidden = block.attention_residual(hidden, attention)
        self.hidden[self.rows, self.positions] = self.model.norm(hidden[:, 0])


@torch.inference_mode()
def generate(model, tokenizer, prompts, seeds, max_tokens, top_k):
    device = model.token.weight.device
    eos = tokenizer.token_to_id(EOS)
    sequences = [tokenizer.encode(prompt).ids for prompt in prompts]
    generated = [[] for _ in prompts]
    rngs = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    live = list(range(len(prompts)))
    capacity = max(map(len, sequences)) + max_tokens - 1
    cache = None
    if capacity <= model.context:
        cache = DecodeCache(model, [torch.tensor(ids, device=device) for ids in sequences], capacity, eos)
    for step in range(max_tokens):
        if not live:
            break
        if cache is None:
            windows = [torch.tensor(sequences[i][-model.context:], device=device) for i in live]
            positions = torch.tensor([len(window) - 1 for window in windows], device=device)
            scores = model.next_scores(pad_sequence(windows, batch_first=True, padding_value=eos), positions)
        else:
            scores = cache.scores()[live]
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("Nonfinite generation scores")
        if top_k == 1:
            tokens = scores.argmax(dim=-1).tolist()
        else:
            values, ids = (scores / 0.85).topk(min(top_k, scores.size(-1)))
            tokens = [candidates[torch.multinomial(weights.softmax(dim=-1), 1, generator=rngs[i])].item()
                      for i, weights, candidates in zip(live, values, ids, strict=True)]
        remaining = []
        advance = [eos] * len(prompts)
        for i, token in zip(live, tokens, strict=True):
            if token != eos:
                sequences[i].append(token)
                generated[i].append(token)
                advance[i] = token
                remaining.append(i)
        live = remaining
        if cache is not None and live and step + 1 < max_tokens:
            cache.advance(advance)
    return [(tokenizer.decode(ids).rstrip("\ufffd"), len(ids)) for ids in generated]


def evaluate(model, tokenizer, probes, max_tokens, seed, top_k):
    model.eval()
    histories = ["" for _ in probes]
    replies = [[] for _ in probes]
    counts = [[] for _ in probes]
    for turn in range(max(len(probe["turns"]) for probe in probes)):
        live = [i for i, probe in enumerate(probes) if len(probe["turns"]) > turn]
        prompts = [histories[i] + format_instruction(probes[i]["turns"][turn]) for i in live]
        results = generate(model, tokenizer, prompts, [seed + i * 10 + turn for i in live], max_tokens, top_k)
        for i, formatted, (answer, count) in zip(live, prompts, results, strict=True):
            replies[i].append(answer)
            counts[i].append(count)
            histories[i] = formatted + answer + "\n"
    rows = []
    for index, probe in enumerate(probes):
        expected = probe.get("expected")
        exact = replies[index][-1].strip().casefold() == expected.casefold() if expected else None
        rows.append({**probe, "answers": replies[index], "generated_tokens": counts[index], "exact": exact})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    if args.max_tokens < 1 or args.top_k < 1:
        parser.error("Token limit and top-k must be positive")
    torch.set_num_threads(6)
    payload = read_checkpoint(args.checkpoint)
    tokenizer = read_tokenizer(payload)
    precision = args.precision or ("fp16" if args.device == "cuda" else "fp32")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = build_model(payload, args.device, dtype)
    del payload
    if args.head:
        load_head(model, read_checkpoint(args.head))
    probes = [json.loads(line) for line in PROBES.read_text(encoding="utf-8").splitlines()]
    rows = evaluate(model, tokenizer, probes, args.max_tokens, args.seed, args.top_k)
    summary = {}
    for split in ("regression", "holdout"):
        summary[split] = {}
        for language in ("ru", "en"):
            items = [row for row in rows if row["set"] == split and row["language"] == language]
            scorable = [row for row in items if row["exact"] is not None]
            summary[split][language] = {"examples": len(items), "exact": sum(row["exact"] for row in scorable),
                                        "exact_total": len(scorable)}
    result = {"checkpoint_sha256": digest(args.checkpoint), "head_sha256": digest(args.head) if args.head else None,
              "probes_sha256": digest(PROBES), "device": args.device, "precision": precision,
              "seed": args.seed, "top_k": args.top_k,
              "max_new_tokens": args.max_tokens, "summary": summary, "rows": rows}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
