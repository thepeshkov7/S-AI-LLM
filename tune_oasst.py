"""Tune the bilingual base model on reviewed assistant turns."""

import argparse
import json
import math
from pathlib import Path
from time import monotonic

import torch

from finetune import instruction_tensors
from generate import TextGenerator
from instruction_data import Example
from memory_chat import chat_prompt
from packed_corpus import LANGUAGES, SPLITS, digest
from pretrain_packed import learning_rate
from train import save_checkpoint


def dialogue_example(row, generator):
    language = row["language"]
    turns = row["turns"]
    if len(turns) not in (2, 4) or language not in LANGUAGES:
        raise ValueError(f'Invalid dialogue: {row["id"]}')
    history = [(turns[0], turns[1])] if len(turns) == 4 else []
    prompt = chat_prompt(turns[-2], language, history)
    answer = turns[-1]
    length = len(generator.tokenizer.encode(prompt).ids) + len(generator.tokenizer.encode(" " + answer).ids)
    if length <= generator.model.context:
        return Example(row["id"], "prefixed_chat", language, prompt, answer)
    return None


def examples(path, generator):
    rows = {lang: [] for lang in LANGUAGES}
    with path.open(encoding="utf-8") as source:
        for line in source:
            example = dialogue_example(json.loads(line), generator)
            if example is not None:
                rows[example.language].append(example)
    if not all(rows.values()):
        raise ValueError(f"No fitting examples for one language in {path}")
    return rows


def trim_batch(x, y):
    length = int((y != -100).any(dim=0).nonzero()[-1]) + 1
    return x[:, :length].contiguous(), y[:, :length].contiguous()


@torch.inference_mode()
def validate(model, tensors, batch_size):
    model.eval()
    device = model.token.weight.device
    result = {}
    for lang, (inputs, targets) in tensors.items():
        total = 0.0
        count = 0
        for start in range(0, len(inputs), batch_size):
            x, y = trim_batch(inputs[start:start + batch_size], targets[start:start + batch_size])
            tokens = int((y != -100).sum())
            total += model.answer_loss(x.to(device), y.to(device)).item() * tokens
            count += tokens
        result[lang] = {"loss": total / count, "answer_tokens": count}
    result["mean_loss"] = sum(item["loss"] for item in result.values()) / len(result)
    return result


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(92)
    rng = torch.Generator().manual_seed(93)
    generator = TextGenerator.load(args.base)
    model = generator.model
    datasets = {}
    counts = {}
    manifest = json.loads((args.dialogues / "manifest.json").read_bytes())
    for split in SPLITS:
        path = args.dialogues / f"{split}.jsonl"
        if digest(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Dialogue split does not match its manifest: {path}")
        rows = examples(path, generator)
        counts[split] = {lang: len(items) for lang, items in rows.items()}
        datasets[split] = {lang: instruction_tensors(items, generator) for lang, items in rows.items()}
    before = validate(model, datasets["validation"], args.batch_size)
    best_metrics = before
    best_step = 0
    best_weights = None
    print(json.dumps({"step": 0, "validation": before, "examples_that_fit": counts}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    started = monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        batches = []
        for lang in LANGUAGES:
            inputs, targets = datasets["train"][lang]
            picked = torch.randint(len(inputs), (args.batch_size // len(LANGUAGES),), generator=rng)
            batches.append((inputs[picked], targets[picked]))
        x, y = trim_batch(*(torch.cat(parts) for parts in zip(*batches, strict=True)))
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.learning_rate)
        loss = model.answer_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every and step != args.steps:
            continue
        metrics = validate(model, datasets["validation"], args.batch_size)
        if metrics["mean_loss"] < best_metrics["mean_loss"]:
            best_metrics, best_step = metrics, step
            best_weights = {key: weight.detach().clone() for key, weight in model.state_dict().items()}
        print(json.dumps({"step": step, "validation": metrics, "best_step": best_step,
                          "seconds": round(monotonic() - started, 2)}), flush=True)
    if best_weights is None:
        raise RuntimeError("No tuned checkpoint improved the dialogue validation loss")
    model.load_state_dict(best_weights)
    test = validate(model, datasets["test"], args.batch_size)
    metrics = {
        "base_sha256": digest(args.base), "dialogue_manifest": digest(args.dialogues / "manifest.json"),
        "steps": args.steps, "best_step": best_step, "batch_size": args.batch_size, "seed": 92,
        "learning_rate": args.learning_rate, "eval_every": args.eval_every, "threads": args.threads,
        "examples_that_fit": counts, "initial_validation": before, "validation": best_metrics,
        "test": test, "seconds": round(monotonic() - started, 2),
    }
    save_checkpoint(args.output, model, generator.tokenizer, metrics)
    print(json.dumps({"finished": True, **metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=130)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.threads) < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error("Training limits must be positive and batch size must be even")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    train(args)


if __name__ == "__main__":
    main()
