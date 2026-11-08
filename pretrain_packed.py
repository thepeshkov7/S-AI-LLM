"""Train or continue S AI on a packed bilingual corpus."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import monotonic

import torch
from filelock import FileLock, Timeout

from generate import TextGenerator
from model import ModelConfig, TinyLLM
from packed_corpus import LANGUAGES, PackedCorpus, digest
from train import atomic_torch_save, save_checkpoint


@torch.inference_mode()
def evaluate(model, corpus, split, batch_size, device, max_windows):
    model.eval()
    context = model.context
    result = {}
    for lang in LANGUAGES:
        tokens = corpus.splits[split][lang]
        count = (len(tokens) - 1) // context
        windows = min(count, max_windows)
        if windows < 1:
            raise ValueError(f"{split}/{lang} is shorter than the model context")
        # Non-overlapping target windows spread over the held-out stream.
        starts = torch.arange(windows) * count // windows * context
        ranges = torch.arange(context + 1)
        loss_sum = 0.0
        for start in range(0, windows, batch_size):
            batch = tokens[starts[start:start + batch_size, None] + ranges].to(device=device, dtype=torch.long)
            loss = model.vocabulary_loss(batch[:, :-1], batch[:, 1:])
            loss_sum += loss.item() * len(batch)
        loss = loss_sum / windows
        result[lang] = {"loss": loss, "perplexity": math.exp(loss), "tokens": windows * context}
    tokens = sum(result[lang]["tokens"] for lang in LANGUAGES)
    loss = sum(result[lang]["loss"] * result[lang]["tokens"] for lang in LANGUAGES) / tokens
    result["overall"] = {"loss": loss, "perplexity": math.exp(loss), "tokens": tokens}
    return result


def learning_rate(step, total, peak, warmup=None):
    if warmup is None:
        warmup = max(1, total // 20)
    if step <= warmup:
        return peak * step / warmup
    fraction = (step - warmup) / (total - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = monotonic()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.bf16 and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("--bf16 requires a CUDA device with BF16 support")
    corpus = PackedCorpus(args.corpus)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint is not None:
        generator = TextGenerator.load(init_checkpoint, device)
        if generator.tokenizer is None or generator.tokenizer.to_str() != corpus.tokenizer.to_str():
            raise ValueError("Warm-start weights and packed corpus require identical tokenizers")
        model = generator.model
        config = model.config
    else:
        config = ModelConfig(corpus.tokenizer.get_vocab_size(), args.width, args.heads, args.layers, args.context)
        model = TinyLLM(**asdict(config)).to(device)
    sequence_length = getattr(args, "sequence_length", None) or config.context
    if not 1 <= sequence_length <= config.context:
        raise ValueError("Training sequence length must fit the checkpoint context")
    if any(len(corpus.splits["train"][lang]) <= sequence_length for lang in LANGUAGES):
        raise ValueError("Each training stream needs more tokens than the model context")
    settings = {
        "corpus_sha256": corpus.fingerprint, "config": asdict(config), "steps": args.steps,
        "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed, "device": device, "bf16": args.bf16, "threads": args.threads,
        "learning_rate": args.learning_rate, "eval_every": args.eval_every,
        "eval_windows": args.eval_windows, "torch": str(torch.__version__),
    }
    if init_checkpoint is not None:
        settings["init_checkpoint_sha256"] = digest(init_checkpoint)
        settings["architecture"] = model.architecture
    if sequence_length != config.context:
        settings["sequence_length"] = sequence_length
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "training-state.pt"
    if state_path.exists() != args.resume:
        raise ValueError("Use --resume exactly when the run directory has a training state")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sampler = torch.Generator().manual_seed(args.seed + 1)
    first_step, best_step, elapsed_before = 1, 0, 0.0
    best_model = None
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["version"] != 1 or state["settings"] != settings:
            raise ValueError("Training state settings do not match the requested run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler"])
        first_step = state["step"] + 1
        initial = state["initial"]
        best_metrics = state["best_metrics"]
        best_step = state["best_step"]
        best_model = state["best_model"]
        elapsed_before = state["elapsed"]
        del state
        print(json.dumps({"resumed_from": first_step - 1}), flush=True)
    else:
        initial = evaluate(model, corpus, "validation", args.batch_size, device, args.eval_windows)
        best_metrics = initial
        print(json.dumps({"step": 0, "validation": initial, "settings": settings}), flush=True)

    stop = args.stop_after or args.steps
    if stop < first_step and not (first_step == args.steps + 1 and stop == args.steps):
        raise ValueError("Requested stop is earlier than the saved training step")
    run_steps = max(0, stop - first_step + 1)
    tokens_per_step = args.batch_size * args.gradient_accumulation * sequence_length
    for step in range(first_step, stop + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.learning_rate)
        for _ in range(args.gradient_accumulation):
            x, y = corpus.batch(args.batch_size, sequence_length, sampler, device)
            with torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext():
                loss = model.vocabulary_loss(x, y)
            (loss / args.gradient_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        on_eval = step % args.eval_every == 0 or step == args.steps
        if not on_eval and step != stop:
            continue
        if on_eval:
            metrics = evaluate(model, corpus, "validation", args.batch_size, device, args.eval_windows)
            if metrics["overall"]["loss"] < best_metrics["overall"]["loss"]:
                best_metrics, best_step = metrics, step
                best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(json.dumps({"step": step, "validation": metrics, "best_step": best_step,
                              "elapsed": round(elapsed_before + monotonic() - started, 2)}), flush=True)
        atomic_torch_save({
            "version": 1, "settings": settings, "step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "sampler": sampler.get_state(), "initial": initial,
            "best_step": best_step, "best_model": best_model, "best_metrics": best_metrics,
            "elapsed": elapsed_before + monotonic() - started,
        }, state_path)

    if stop < args.steps:
        print(json.dumps({"paused_at": stop}), flush=True)
        return
    if best_model is None:
        raise RuntimeError("No trained checkpoint improved validation over the starting weights")
    model.load_state_dict(best_model)
    test = evaluate(model, corpus, "test", args.batch_size, device, args.eval_windows)
    session_seconds = monotonic() - started
    metrics = {
        "settings": settings,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tokens_processed": args.steps * tokens_per_step,
        "tokens_by_language": {lang: args.steps * tokens_per_step // len(LANGUAGES) for lang in LANGUAGES},
        "seconds": round(elapsed_before + session_seconds, 2),
        "last_session_tokens_per_second": round(run_steps * tokens_per_step / session_seconds, 1),
        "best_step": best_step,
        "initial_validation": initial, "validation": best_metrics, "test": test,
    }
    save_checkpoint(args.output / "pretrained.pt", model.cpu(), corpus.tokenizer, metrics)
    print(json.dumps({"finished": True, **metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--init-checkpoint", type=Path, help="Continue training with this model and its original vocabulary")
    parser.add_argument("--sequence-length", type=int, help="Training window length, defaults to the model context")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("Balanced batches require a positive, even batch size")
    if min(args.steps, args.eval_every, args.eval_windows, args.gradient_accumulation, args.threads) < 1:
        parser.error("Training and evaluation limits must be positive")
    if args.sequence_length is not None and args.sequence_length < 1:
        parser.error("--sequence-length must be positive")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("--stop-after must be between 1 and --steps")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(args.output / ".training.lock", timeout=0):
            train(args)
    except Timeout:
        parser.error("Another training process owns this run directory")


if __name__ == "__main__":
    main()
