"""Generate raw CopyLLM answers on bilingual probes without an external model."""
import argparse
import json
from pathlib import Path
import re

import torch

from generate import TextGenerator, format_instruction
from packed_corpus import digest
from training_1b.checkpoint import build_model, load_head, read_checkpoint, read_tokenizer


PROBES = Path(__file__).with_name("prompts.jsonl")


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def evaluate(model, tokenizer, probes, max_tokens, seed, top_k):
    generator = TextGenerator(model.eval(), tokenizer)
    rows = []
    for index, probe in enumerate(probes):
        history = ""
        replies = []
        counts = []
        for turn, prompt in enumerate(probe["turns"]):
            formatted = history + format_instruction(prompt)
            answer, count = generator.complete(formatted, max_new_tokens=max_tokens, top_k=top_k,
                                               seed=seed + index * 10 + turn)
            replies.append(answer)
            counts.append(count)
            history = formatted + answer + "\n"
        expected = probe.get("expected")
        rows.append({**probe, "answers": replies, "generated_tokens": counts,
                     "exact": normalized(replies[-1]) == normalized(expected) if expected else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    torch.set_num_threads(4)
    payload = read_checkpoint(args.checkpoint)
    tokenizer = read_tokenizer(payload)
    model = build_model(payload, args.device, torch.float16 if args.device == "cuda" else torch.float32)
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
              "probes_sha256": digest(PROBES), "device": args.device, "seed": args.seed, "top_k": args.top_k,
              "max_new_tokens": args.max_tokens, "summary": summary, "rows": rows}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
