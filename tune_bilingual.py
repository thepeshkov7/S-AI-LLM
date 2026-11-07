"""Continue the dialogue checkpoint with balanced language and task sampling."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from time import monotonic

import torch
from filelock import FileLock, Timeout

from finetune import instruction_tensors
from generate import TextGenerator
from packed_corpus import LANGUAGES, SPLITS, digest
from prepare_bilingual import source_rows
from pretrain_packed import learning_rate
from train import save_checkpoint
from tune_oasst import dialogue_example, trim_batch, validate


WEIGHTS = {
    "everyday": 3,
    "dialogue": 2,
    "translation": 1,
    "arithmetic": 2,
    "extract": 1,
    "reason": 1,
    "followup": 2,
    "format": 1,
}


def load_split(folder, split, manifest, generator):
    buckets = defaultdict(list)
    for row in source_rows(folder, split, manifest):
        example = dialogue_example(row, generator)
        if example is not None:
            buckets[(example.language, row["family"])].append(example)
    return buckets


def measure(model, tensors, batch_size):
    result = validate(model, tensors, batch_size)
    result.pop("mean_loss")
    per_language = {}
    for lang in LANGUAGES:
        losses = [metrics["loss"] for (language, _), metrics in result.items() if language == lang]
        if not losses:
            raise ValueError(f"No validation categories for {lang}")
        per_language[lang] = sum(losses) / len(losses)
    return {"mean_loss": sum(per_language.values()) / len(per_language), "by_language": per_language,
            "by_family": {f"{lang}:{family}": item for (lang, family), item in result.items()}}


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(381)
    random_source = random.Random(382)
    sampler = torch.Generator().manual_seed(383)
    generator = TextGenerator.load(args.base, args.device)
    model = generator.model
    manifest_path = args.curriculum / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    examples = {split: load_split(args.curriculum, split, manifest, generator) for split in SPLITS}
    if any((lang, family) not in examples["train"] for lang in LANGUAGES for family in WEIGHTS):
        raise ValueError("Training requires examples in every language and family")
    tensors = {split: {bucket: instruction_tensors(rows, generator) for bucket, rows in group.items()}
               for split, group in examples.items()}
    del examples
    counts = {split: {f"{lang}:{family}": len(inputs) for (lang, family), (inputs, _) in group.items()}
              for split, group in tensors.items()}
    baseline = measure(model, tensors["validation"], args.batch_size)
    best, best_step = baseline, 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    stats = {
        "base_sha256": digest(args.base), "curriculum_sha256": digest(manifest_path),
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "device": args.device, "threads": args.threads, "eval_every": args.eval_every,
        "sampling_weights": WEIGHTS, "counts_that_fit": counts, "validation_before": baseline,
    }
    print(json.dumps({"step": 0, **stats}), flush=True)
    started = monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        family = random_source.choices(tuple(WEIGHTS), weights=tuple(WEIGHTS.values()))[0]
        batches = []
        for lang in LANGUAGES:
            inputs, targets = tensors["train"][(lang, family)]
            indices = torch.randint(len(inputs), (args.batch_size // len(LANGUAGES),), generator=sampler)
            batches.append((inputs[indices], targets[indices]))
        x, y = trim_batch(*(torch.cat(parts) for parts in zip(*batches, strict=True)))
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.learning_rate)
        loss = model.answer_loss(x.to(args.device), y.to(args.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every and step != args.steps:
            continue
        metrics = measure(model, tensors["validation"], args.batch_size)
        elapsed = round(monotonic() - started, 2)
        if metrics["mean_loss"] < best["mean_loss"]:
            best, best_step = metrics, step
            save_checkpoint(args.output, model, generator.tokenizer, {
                **stats, "bilingual_chat": True, "selected_step": step,
                "validation_after": metrics, "seconds": elapsed,
            })
        print(json.dumps({"step": step, "validation": metrics, "best_step": best_step, "seconds": elapsed}), flush=True)
    if best_step == 0:
        raise RuntimeError("No checkpoint improved the balanced validation loss")
    model.load_state_dict(torch.load(args.output, map_location="cpu", weights_only=True)["model"])
    test = measure(model, tensors["test"], args.batch_size)
    print(json.dumps({"finished": True, "selected_step": best_step, "test": test,
                      "seconds": round(monotonic() - started, 2)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.00075)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.threads, args.batch_size) < 1 or args.batch_size % 2:
        parser.error("Use positive limits and an even batch size")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(args.output.with_suffix(".lock"), timeout=0):
            if args.output.exists():
                parser.error("Choose a new output checkpoint path")
            train(args)
    except Timeout:
        parser.error("Another training process owns this output checkpoint")


if __name__ == "__main__":
    main()
