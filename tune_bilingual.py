"""Continue the dialogue checkpoint with balanced language and task sampling."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from time import monotonic

import torch
from filelock import FileLock, Timeout

from evaluate_assistant import score
from evaluate_chat import completions
from finetune import instruction_tensors
from generate import TextGenerator
from model import CopyLLM
from packed_corpus import LANGUAGES, SPLITS, PackedCorpus, digest
from prepare_bilingual import source_rows
from pretrain_packed import learning_rate
from train import atomic_torch_save, save_checkpoint
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


def measure(model, tensors, batch_size, selection_weights=None):
    result = validate(model, tensors, batch_size)
    result.pop("mean_loss")
    per_language = {}
    selection_weights = selection_weights or {}
    for lang in LANGUAGES:
        weighted = [(metrics["loss"], selection_weights.get(family, 1))
                    for (language, family), metrics in result.items() if language == lang]
        if not weighted:
            raise ValueError(f"No validation categories for {lang}")
        per_language[lang] = sum(loss * weight for loss, weight in weighted) / sum(weight for _, weight in weighted)
    return {"mean_loss": sum(per_language.values()) / len(per_language), "by_language": per_language,
            "by_family": {f"{lang}:{family}": item for (lang, family), item in result.items()}}


def instruction_batch(tensors, families, sampler):
    batches, input_tokens, answer_tokens = [], {}, {}
    for lang in LANGUAGES:
        selected = []
        for family, size in families.items():
            inputs, targets = tensors[(lang, family)]
            indices = torch.randint(len(inputs), (size,), generator=sampler)
            selected.append((inputs[indices], targets[indices]))
        x, y = (torch.cat(parts) for parts in zip(*selected, strict=True))
        active = y != -100
        lengths = y.size(1) - active.flip((1,)).to(torch.int8).argmax(dim=1)
        input_tokens[lang] = int(lengths.sum())
        answer_tokens[lang] = int(active.sum())
        batches.append((x, y))
    return trim_batch(*(torch.cat(parts) for parts in zip(*batches, strict=True))), input_tokens, answer_tokens


def generation_sample(buckets, count):
    rng = random.Random(8105)
    result = []
    for examples in buckets.values():
        short = [example for example in examples if len(example.answer) <= 48]
        result.extend(rng.sample(short, min(len(short), count)))
    return result


def validate_candidate(generator, tensors, batch_size, selection, samples):
    metrics = measure(generator.model, tensors, batch_size, selection)
    if samples:
        responses = completions(generator, [example.prompt for example in samples], 24, 24)
        metrics["generation"] = {"correct": sum(score(response, example.answer)
                                                 for example, response in zip(samples, responses, strict=True)),
                                 "total": len(samples)}
    return metrics


def selection_key(metrics):
    return (metrics.get("generation", {}).get("correct", 0), -metrics["mean_loss"])


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(381)
    random_source = random.Random(382)
    sampler = torch.Generator().manual_seed(383)
    generator = TextGenerator.load(args.base, args.device)
    if generator.tokenizer is None:
        raise ValueError("Bilingual tuning requires a BPE checkpoint")
    if args.with_copy_head and not isinstance(generator.model, CopyLLM):
        model = CopyLLM(**asdict(generator.model.config))
        initial_state = model.state_dict()
        initial_state.update(generator.model.state_dict())
        model.load_state_dict(initial_state)
        model.set_copy_aliases(generator.tokenizer)
        generator.model = model.to(args.device)
    model = generator.model
    replay = PackedCorpus(args.replay_corpus) if args.replay_corpus is not None else None
    if replay is not None and replay.tokenizer.to_str() != generator.tokenizer.to_str():
        raise ValueError("Replay and instruction data must use the same checkpoint vocabulary")
    replay_context = min(128, model.context)
    manifest_path = args.curriculum / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    weights = manifest.get("sampling_weights", WEIGHTS)
    selection = manifest.get("selection_weights", {})
    if not weights or any(type(weight) is not int or weight < 1 for weight in [*weights.values(), *selection.values()]):
        raise ValueError("Curriculum weights must be positive integers")
    examples = {split: load_split(args.curriculum, split, manifest, generator) for split in SPLITS}
    if any((lang, family) not in examples["train"] for lang in LANGUAGES for family in weights):
        raise ValueError("Training requires examples in every language and family")
    samples = generation_sample(examples["validation"], args.generation_checks) if args.generation_checks else []
    tensors = {split: {bucket: instruction_tensors(rows, generator) for bucket, rows in group.items()}
               for split, group in examples.items()}
    del examples
    counts = {split: {f"{lang}:{family}": len(inputs) for (lang, family), (inputs, _) in group.items()}
              for split, group in tensors.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    settings = {
        "base_sha256": digest(args.base), "curriculum_sha256": digest(manifest_path),
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "device": args.device, "threads": args.threads, "eval_every": args.eval_every,
        "sampling_weights": weights, "selection_weights": selection, "architecture": model.architecture,
        "counts_that_fit": counts, "mix_families": args.mix_families, "chat_model": args.chat_model,
        "replay_sha256": replay.fingerprint if replay is not None else None,
        "replay_every": args.replay_every if replay is not None else None,
        "replay_batch_size": 4 if replay is not None else None,
        "replay_context": replay_context if replay is not None else None,
        "torch": str(torch.__version__),
    }
    if args.equal_example_loss:
        settings["equal_example_loss"] = True
    if args.generation_checks:
        settings["generation_checks_per_group"] = args.generation_checks
    state_path = args.output.with_suffix(".training-state.pt")
    if state_path.exists() != args.resume:
        raise ValueError("Use --resume exactly when the training state exists")
    tokens = {kind: Counter({lang: 0 for lang in LANGUAGES}) for kind in ("inputs", "answers", "prose")}
    first_step, best_step, elapsed_before = 1, 0, 0.0
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["version"] != 1 or state["settings"] != settings:
            raise ValueError("Training state settings do not match this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        random_source.setstate(state["family_sampler"])
        sampler.set_state(state["example_sampler"])
        first_step = state["step"] + 1
        baseline, best, best_step = state["baseline"], state["best"], state["best_step"]
        if best_step and not args.output.is_file():
            raise ValueError("The selected inference checkpoint is missing from this resumed run")
        tokens = {kind: Counter(counts) for kind, counts in state["tokens"].items()}
        elapsed_before = state["elapsed"]
        del state
        print(json.dumps({"resumed_from": first_step - 1}), flush=True)
    else:
        baseline = best = validate_candidate(generator, tensors["validation"], args.batch_size, selection, samples)
        print(json.dumps({"step": 0, **settings, "validation_before": baseline}), flush=True)
    stop = args.stop_after or args.steps
    if stop < first_step and not (first_step == args.steps + 1 and stop == args.steps):
        raise ValueError("Requested stop is earlier than the saved training step")
    examples_per_language = args.batch_size // len(LANGUAGES)
    started = monotonic()
    for step in range(first_step, stop + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.learning_rate)
        if replay is not None and step % args.replay_every == 0:
            x, y = replay.batch(4, replay_context, sampler, args.device)
            loss = model.vocabulary_loss(x, y)
            tokens["prose"].update({lang: 2 * replay_context for lang in LANGUAGES})
        else:
            draws = random_source.choices(tuple(weights), weights=tuple(weights.values()),
                                          k=examples_per_language if args.mix_families else 1)
            families = Counter(draws) if args.mix_families else {draws[0]: examples_per_language}
            (x, y), inputs, answers = instruction_batch(tensors["train"], families, sampler)
            loss = model.answer_loss(x.to(args.device), y.to(args.device), equal_examples=args.equal_example_loss)
            tokens["inputs"].update(inputs)
            tokens["answers"].update(answers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        on_eval = step % args.eval_every == 0 or step == args.steps
        if not on_eval and step != stop:
            continue
        elapsed = round(elapsed_before + monotonic() - started, 2)
        token_counts = {kind: dict(counts) for kind, counts in tokens.items()}
        if on_eval:
            metrics = validate_candidate(generator, tensors["validation"], args.batch_size, selection, samples)
            if selection_key(metrics) > selection_key(best):
                best, best_step = metrics, step
                save_checkpoint(args.output, model, generator.tokenizer, {
                    **settings, "bilingual_chat": True, "selected_step": step,
                    "validation_before": baseline, "validation_after": metrics,
                    "token_counts": token_counts, "seconds": elapsed,
                })
            print(json.dumps({"step": step, "validation": metrics, "best_step": best_step,
                              "token_counts": token_counts, "seconds": elapsed}), flush=True)
        atomic_torch_save({"version": 1, "settings": settings, "step": step, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "family_sampler": random_source.getstate(),
                           "example_sampler": sampler.get_state(), "baseline": baseline, "best": best,
                           "best_step": best_step, "tokens": token_counts, "elapsed": elapsed}, state_path)
    if stop < args.steps:
        print(json.dumps({"paused_at": stop}), flush=True)
        return
    if best_step == 0:
        raise RuntimeError("No trained checkpoint improved the validation selection metric")
    model.load_state_dict(torch.load(args.output, map_location="cpu", weights_only=True)["model"])
    test = measure(model, tensors["test"], args.batch_size, selection)
    print(json.dumps({"finished": True, "completed_steps": args.steps, "selected_step": best_step, "test": test,
                      "token_counts": {kind: dict(counts) for kind, counts in tokens.items()},
                      "seconds": round(elapsed_before + monotonic() - started, 2)}), flush=True)


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
    parser.add_argument("--with-copy-head", action="store_true")
    parser.add_argument("--mix-families", action="store_true")
    parser.add_argument("--replay-corpus", type=Path)
    parser.add_argument("--replay-every", type=int, default=5)
    parser.add_argument("--chat-model")
    parser.add_argument("--equal-example-loss", action="store_true")
    parser.add_argument("--generation-checks", type=int, default=0, help="Validate this many short generations per family and language")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.threads, args.batch_size) < 1 or args.batch_size % 2:
        parser.error("Use positive limits and an even batch size")
    if args.replay_every < 2:
        parser.error("Prose replay must leave some steps for instructions")
    if args.generation_checks < 0:
        parser.error("Generation check count cannot be negative")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("--stop-after must be between 1 and --steps")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(args.output.with_suffix(".lock"), timeout=0):
            if args.output.exists() and not args.resume:
                parser.error("Choose a new output checkpoint path")
            train(args)
    except Timeout:
        parser.error("Another training process owns this output checkpoint")


if __name__ == "__main__":
    main()
