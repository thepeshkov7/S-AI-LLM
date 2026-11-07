"""Continue a trained S AI checkpoint with bilingual instruction supervision."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from time import monotonic

import torch

from generate import TextGenerator, format_instruction
from instruction_data import build_splits
from train import TrainingWindows, encode_documents, evaluate, read_split, save_checkpoint


def instruction_tensors(examples, generator):
    tokens = generator.tokenizer
    context = generator.model.context
    inputs, targets = [], []
    for example in examples:
        text = example.prompt if example.task in {"followup", "memory_route", "prefixed_chat"} else format_instruction(example.prompt)
        prompt = tokens.encode(text).ids
        prefix = "" if example.task == "memory_route" else " "
        response = tokens.encode(prefix + example.answer).ids + [generator.eos]
        combined = prompt + response
        if len(combined) - 1 > context:
            raise ValueError(f"Instruction exceeds context: {example.group} ({example.language})")
        padding = context + 1 - len(combined)
        inputs.append(combined[:-1] + [generator.eos] * padding)
        targets.append([-100] * (len(prompt) - 1) + response + [-100] * padding)
    return torch.tensor(inputs), torch.tensor(targets)


def masked_loss(model, inputs, targets, device, batch_size):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            x = inputs[start:start + batch_size].to(device)
            y = targets[start:start + batch_size].to(device)
            _, loss = model(x, y)
            valid = int((y != -100).sum())
            total_loss += loss.item() * valid
            count += valid
    return total_loss / count


def generation_benchmark(generator, examples):
    counts = Counter()
    correct = Counter()
    predictions = []
    for example in examples:
        output, _ = generator.complete(format_instruction(example.prompt), max_new_tokens=16, top_k=1)
        passed = output.strip().casefold() == example.answer.casefold()
        for group in ("overall", example.language, example.task, f"{example.language}:{example.task}"):
            counts[group] += 1
            correct[group] += int(passed)
        predictions.append({**asdict(example), "prediction": output.strip(), "exact": passed})
    return {key: {"correct": correct[key], "total": count} for key, count in counts.items()}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--replay-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--split-seed", type=int, default=761)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.replay_every, args.threads) < 1:
        parser.error("Step counts, batch size and thread count must be positive")
    if args.replay_every == 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Replay must leave instruction batches and learning rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    splits = build_splits(args.split_seed)
    generator = TextGenerator.load(args.base, args.device)
    if generator.tokenizer is None:
        parser.error("Instruction training requires a BPE checkpoint")
    model = generator.model
    train = instruction_tensors(splits["train"], generator)
    validation = instruction_tensors(splits["validation"], generator)
    corpus_ids, corpus_hashes = set(), set()
    replay = TrainingWindows(
        encode_documents(read_split(args.data_dir / "train.jsonl", corpus_ids, corpus_hashes), generator.tokenizer),
        model.context, args.seed + 1,
    )
    literature_validation = encode_documents(
        read_split(args.data_dir / "validation.jsonl", corpus_ids, corpus_hashes), generator.tokenizer,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.learning_rate * 0.1)
    sampler = torch.Generator().manual_seed(args.seed + 2)
    tasks = ("add", "equal", "select", "extract")
    positives = {"да", "yes"}
    training_examples = splits["train"]
    task_indices = {
        task: [index for index, example in enumerate(training_examples) if example.task == task]
        for task in tasks
    }
    equal = task_indices["equal"]
    equal_indices = (
        [index for index in equal if training_examples[index].answer not in positives],
        [index for index in equal if training_examples[index].answer in positives],
    )
    initial_loss = masked_loss(model, *validation, args.device, args.batch_size)
    literature_before = evaluate(model, literature_validation, args.batch_size, args.device)
    best_loss = initial_loss
    best_step = 0
    best_weights = None
    stats = {name: len(items) for name, items in splits.items()}
    print(json.dumps({"examples": stats, "initial_instruction_validation_loss": initial_loss}), flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        if step % args.replay_every == 0:
            x, y = replay.batch(args.batch_size, args.device)
        else:
            task_picks = torch.randint(len(tasks), (args.batch_size,), generator=sampler).tolist()
            chosen = []
            for task_index in task_picks:
                task = tasks[task_index]
                pool = task_indices[task]
                if task == "equal":
                    pool = equal_indices[int(torch.randint(2, (), generator=sampler))]
                chosen.append(pool[int(torch.randint(len(pool), (), generator=sampler))])
            x, y = (tensor[chosen].to(args.device) for tensor in train)
            # Keep computation proportional to the longest instruction in the batch.
            length = int((y != -100).any(dim=0).nonzero()[-1]) + 1
            x, y = x[:, :length].contiguous(), y[:, :length].contiguous()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            heldout_loss = masked_loss(model, *validation, args.device, args.batch_size)
            print(json.dumps({"step": step, "instruction_validation_loss": heldout_loss}), flush=True)
            if heldout_loss < best_loss:
                best_loss, best_step = heldout_loss, step
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.train()
    if best_weights is None:
        raise RuntimeError("Instruction validation did not improve")
    model.load_state_dict(best_weights)
    literature_after = evaluate(model, literature_validation, args.batch_size, args.device)
    model.eval()
    training_seconds = round(monotonic() - started, 1)
    base = TextGenerator.load(args.base, args.device)
    # The test sets have not been consulted for model selection.
    comparisons, predictions = {}, {}
    for split in ("test", "new_wording"):
        before, base_predictions = generation_benchmark(base, splits[split])
        after, tuned_predictions = generation_benchmark(generator, splits[split])
        comparisons[split] = {"base": before, "tuned": after}
        predictions[split] = {"base": base_predictions, "tuned": tuned_predictions}
        print(json.dumps({"split": split, "base": before, "tuned": after}), flush=True)
    diagnostics = [
        "Сколько будет два плюс два?",
        "What is the capital of France?",
        "Почему идёт дождь?",
        "Explain gravity in one sentence.",
    ]
    for question in diagnostics:
        prompt = format_instruction(question)
        predictions.setdefault("open_diagnostics", []).append({
            "prompt": question,
            "base": base.complete(prompt, max_new_tokens=24, top_k=1)[0].strip(),
            "tuned": generator.complete(prompt, max_new_tokens=24, top_k=1)[0].strip(),
        })
    metrics = {
        "base_sha256": sha256(args.base.read_bytes()).hexdigest(),
        "examples": stats,
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "torch": str(torch.__version__),
        "training_seconds": training_seconds,
        "best_step": best_step,
        "initial_instruction_validation_loss": initial_loss,
        "best_instruction_validation_loss": best_loss,
        "literature_validation": {"base": literature_before, "tuned": literature_after},
        "exact_match": comparisons,
    }
    save_checkpoint(args.output / "s_ai_0_1_instruct.pt", model.cpu(), generator.tokenizer, metrics)
    for name, data in (("metrics.json", metrics), ("predictions.json", predictions)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"best_step": best_step, "seconds": training_seconds, "literature_validation": metrics["literature_validation"]}), flush=True)


if __name__ == "__main__":
    main()
