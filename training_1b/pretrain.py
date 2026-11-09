"""Train the original 1B CopyLLM and checkpoint the entire optimizer state."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import monotonic

import torch
from filelock import FileLock

from packed_corpus import LANGUAGES, PackedCorpus, digest
from pretrain_packed import learning_rate
from train import atomic_torch_save
from training_1b.checkpoint import build_model, read_checkpoint, read_tokenizer


@dataclass(frozen=True)
class Profile:
    steps: int
    sequence_length: int = 256
    micro_batch: int = 1
    accumulation: int = 4
    lr: float = 5e-5
    precision: str = "fp16"
    optimizer: str = "adamw8bit"
    gradient_checkpointing: bool = True
    last_blocks: int = 20
    objective: str = "vocabulary"
    eval_every: int = 12
    eval_windows: int = 16
    save_every: int = 12
    seed: int = 20261108
    warmup_steps: int | None = None
    one_pass: bool = False
    recovery_step: int | None = None

    def __post_init__(self):
        if min(self.steps, self.sequence_length, self.micro_batch, self.accumulation,
               self.last_blocks, self.eval_every, self.eval_windows, self.save_every) < 1:
            raise ValueError("Profile sizes and intervals must be positive")
        if self.accumulation % len(LANGUAGES):
            raise ValueError("Accumulation must give each language the same number of microbatches")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported precision")
        if self.optimizer not in {"adamw", "adamw8bit"} or self.objective not in {"vocabulary", "copy"}:
            raise ValueError("Unsupported optimizer or objective")
        if self.warmup_steps is not None and not 1 <= self.warmup_steps <= self.steps:
            raise ValueError("Warmup steps must be within the run")
        if self.recovery_step is not None and (
            self.recovery_step != 10000 or not self.one_pass or self.objective != "vocabulary"
            or self.precision != "bf16"
        ):
            raise ValueError("Step-10000 recovery requires BF16 and one-pass vocabulary training")


def configure(model, profile):
    if profile.last_blocks > len(model.blocks) or profile.sequence_length > model.context:
        raise ValueError("Profile exceeds model depth or context")
    model.gradient_checkpointing = profile.gradient_checkpointing
    full = profile.last_blocks == len(model.blocks)
    model.requires_grad_(full)
    for block in list(model.blocks)[-profile.last_blocks:]:
        block.requires_grad_(True)
    model.norm.requires_grad_(True)
    model.output.requires_grad_(True)
    if not full:
        # Learn all positional embeddings when moving from short to longer sequences.
        model.position.requires_grad_(True)
    for projection in (model.copy_query, model.copy_key, model.copy_mode):
        projection.requires_grad_(profile.objective == "copy")
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def optimizer_for(parameters, profile, device):
    if profile.optimizer == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(parameters, lr=profile.lr, betas=(0.9, 0.95), weight_decay=0.1)
    return torch.optim.AdamW(parameters, lr=profile.lr, betas=(0.9, 0.95), weight_decay=0.1,
                             fused=device == "cuda")


def autocast(profile):
    if profile.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if profile.precision == "fp16" else torch.bfloat16
    # Per-block activation checkpointing can otherwise retain an autocast weight cache.
    return torch.autocast("cuda", dtype=dtype, cache_enabled=False)


def loss_for(model, batch, profile):
    x, y = batch[:, :-1], batch[:, 1:]
    with autocast(profile):
        if profile.objective == "vocabulary":
            return model.vocabulary_loss(x, y)
        return model.answer_loss(x, y)


def make_batch(stream, starts, offsets, device):
    return stream[starts[:, None] + offsets].to(device=device, dtype=torch.long)


@torch.inference_mode()
def evaluate(model, corpus, profile, split, device):
    model.eval()
    stats = {}
    length = profile.sequence_length
    offsets = torch.arange(length + 1)
    for lang in LANGUAGES:
        stream = corpus.splits[split][lang]
        available = (len(stream) - 1) // length
        windows = min(available, profile.eval_windows)
        if not windows:
            raise ValueError(f"{split}/{lang} lacks an evaluation window")
        starts = torch.arange(windows) * available // windows * length
        total = 0.0
        for start in range(0, windows, profile.micro_batch):
            batch = make_batch(stream, starts[start:start + profile.micro_batch], offsets, device)
            total += loss_for(model, batch, profile).item() * len(batch)
        loss = total / windows
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {split} loss for {lang}")
        stats[lang] = {"loss": loss, "tokens": windows * length}
    return stats


def one_pass_windows(corpus, profile):
    """Shuffle non-overlapping target windows without reusing a token position."""
    order = torch.Generator().manual_seed(profile.seed + 2)
    needed = profile.steps * profile.accumulation * profile.micro_batch // len(LANGUAGES)
    windows = {}
    for language in LANGUAGES:
        available = (len(corpus.splits["train"][language]) - 1) // profile.sequence_length
        if available < needed:
            raise ValueError(f"One-pass schedule needs {needed} windows for {language}, but has {available}")
        windows[language] = torch.randperm(available, generator=order)[:needed] * profile.sequence_length
    return windows


def update(model, optimizer, parameters, scaler, corpus, profile, sampler, device, step, windows):
    # Retry the same examples at a lower scale after an FP16 overflow.
    state = sampler.get_state()
    offsets = torch.arange(profile.sequence_length + 1)
    for attempt in range(6):
        sampler.set_state(state)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index in range(profile.accumulation):
            lang = LANGUAGES[index % len(LANGUAGES)]
            stream = corpus.splits["train"][lang]
            if windows is None:
                starts = torch.randint(len(stream) - profile.sequence_length, (profile.micro_batch,),
                                       generator=sampler)
            else:
                turn = ((step - 1) * profile.accumulation + index) // len(LANGUAGES)
                start = turn * profile.micro_batch
                starts = windows[lang][start:start + profile.micro_batch]
            loss = loss_for(model, make_batch(stream, starts, offsets, device), profile)
            total += loss.detach().item()
            if not math.isfinite(total):
                raise FloatingPointError("Nonfinite forward loss")
            scaler.scale(loss / profile.accumulation).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if math.isfinite(norm.item()):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return total / profile.accumulation, norm.item(), attempt
        if not scaler.is_enabled():
            break
        scaler.update(new_scale=scaler.get_scale() / 2)
    raise FloatingPointError("Could not obtain finite gradients after reducing loss scale")


def runtime_stats(device):
    stats = {}
    try:
        from resource import RUSAGE_SELF, getrusage
    except ImportError:
        pass
    else:
        stats["peak_rss_gib"] = round(getrusage(RUSAGE_SELF).ru_maxrss / 2**20, 3)
    if device == "cuda":
        stats.update(
            gpu=torch.cuda.get_device_name(), total_vram_gib=round(torch.cuda.mem_get_info()[1] / 2**30, 3),
            peak_vram_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
            peak_vram_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3),
        )
    return stats


def train(corpus, profile, checkpoint, output, device, stop, *, resume=False, require_1b=True,
          deadline=None, max_session_tokens=None):
    recovery = profile.recovery_step is not None
    if recovery and (deadline is None or max_session_tokens is None):
        raise ValueError("Step-10000 recovery requires explicit time and token budgets")
    if device == "cpu" and (profile.precision != "fp32" or profile.optimizer == "adamw8bit"):
        raise ValueError("CPU requires FP32 and standard AdamW")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this runtime")
    if device == "cuda" and profile.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU has no native BF16 support")
    torch.manual_seed(profile.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sampler = torch.Generator().manual_seed(profile.seed + 1)
    payload = read_checkpoint(checkpoint)
    if recovery and not resume and (
        checkpoint.name != "step-010000.pt" or payload.get("step", 10000) != 10000
    ):
        raise ValueError("Initialize recovery from the original step-010000.pt")
    tokenizer = read_tokenizer(payload)
    if tokenizer.to_str() != corpus.tokenizer.to_str():
        raise ValueError("The training corpus and original checkpoint need the same tokenizer")
    if any(len(corpus.splits["train"][lang]) <= profile.sequence_length for lang in LANGUAGES):
        raise ValueError("Training corpus is too short")
    windows = one_pass_windows(corpus, profile) if profile.one_pass else None
    profile_settings = asdict(profile)
    if profile.warmup_steps is None:
        del profile_settings["warmup_steps"]
    if not profile.one_pass:
        del profile_settings["one_pass"]
    if not recovery:
        del profile_settings["recovery_step"]
    settings = {
        "profile": profile_settings, "corpus_sha256": corpus.fingerprint, "config": payload["config"],
        "device": device, "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    if profile.optimizer == "adamw8bit":
        from importlib.metadata import version
        settings["bitsandbytes"] = version("bitsandbytes")
    if resume and (payload.get("run_format") != 1 or payload.get("settings") != settings):
        raise ValueError("The profile, corpus or runtime differs from the saved state. Use --init for a new stage.")
    if resume and (payload["step"] >= stop or not (output / "training-state.pt").is_file()):
        raise ValueError("Resume requires an existing run and a later stop step")
    if resume and not checkpoint.samefile(output / "training-state.pt"):
        if digest(checkpoint) != digest(output / "training-state.pt"):
            raise ValueError("The staged resume file differs from the current state of this run")
    state_path = output / "training-state.pt"
    if state_path.exists() and not resume:
        raise FileExistsError("This run already has a state. Use --resume or a new output directory.")
    first = payload["step"] + 1 if resume else 1
    tokens_per_step = profile.micro_batch * profile.accumulation * profile.sequence_length
    if max_session_tokens is not None and (stop - first + 1) * tokens_per_step > max_session_tokens:
        raise ValueError("Requested optimizer steps exceed --max-session-tokens")
    if deadline is not None and monotonic() >= deadline:
        print(json.dumps({"event": "paused", "step": first - 1, "reason": "time_budget"}), flush=True)
        return payload["metrics"] if resume else None
    model = build_model(payload, device, require_1b=require_1b, recovery=recovery)
    parameters = configure(model, profile)
    # CPU mmap shares storages with the source file; clone its trainable tensors
    # so an in-place optimizer can safely update only its own parameters.
    if device == "cpu":
        with torch.no_grad():
            for parameter in parameters:
                parameter.data = parameter.detach().clone()
    optimizer = optimizer_for(parameters, profile, device)
    scaler = torch.amp.GradScaler("cuda", enabled=profile.precision == "fp16", init_scale=512.0,
                                  growth_interval=1000)
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        sampler.set_state(payload["sampler"])
        torch.random.set_rng_state(payload["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"])
        metrics = payload["metrics"]
        initial_sha = payload["initial_sha256"]
    else:
        initial_sha = digest(checkpoint)
        metrics = {
            "baseline": None if recovery else evaluate(model, corpus, profile, "validation", device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "training_seconds": 0.0, "overflow_retries": 0, "evaluations": [],
        }
        if not recovery:
            print(json.dumps({"event": "baseline", **metrics, "runtime": runtime_stats(device)}), flush=True)
    del payload
    print(json.dumps({"event": "start", "first_step": first, "stop_after": stop,
                      "settings": settings, "initial_sha256": initial_sha}), flush=True)

    def save(record):
        step = record["step"]
        metrics.update({"last": record, "runtime": runtime_stats(device)})
        atomic_torch_save({
            "run_format": 1, "format_version": 1, "architecture": model.architecture,
            "config": asdict(model.config), "tokenizer": tokenizer.to_str(), "model": model.state_dict(),
            "step": step, "initial_sha256": initial_sha, "settings": settings, "metrics": metrics,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.get_state(),
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        }, state_path)
        report = output / "metrics.json"
        staged = report.with_suffix(".tmp")
        staged.write_text(json.dumps({"settings": settings, "metrics": metrics}, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
        staged.replace(report)
        print(json.dumps({"event": "saved", "step": step, "path": str(state_path),
                          "runtime": metrics["runtime"]}), flush=True)

    record = None
    for step in range(first, stop + 1):
        if deadline is not None and monotonic() >= deadline:
            if record is not None and record["step"] != metrics.get("last", {}).get("step"):
                save(record)
            print(json.dumps({"event": "paused", "step": step - 1, "reason": "time_budget"}), flush=True)
            break
        started = monotonic()
        model.train()
        lr = learning_rate(step, profile.steps, profile.lr, profile.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step == 1 and not recovery:
            before = [block.qkv.weight[0, :32].detach().clone() for block in model.blocks]
        loss, norm, retries = update(model, optimizer, parameters, scaler, corpus, profile, sampler, device, step, windows)
        metrics["training_seconds"] += monotonic() - started
        metrics["overflow_retries"] += retries
        if step == 1 and not recovery:
            metrics["updated_transformer_blocks"] = [
                index + 1 for index, (block, start) in enumerate(zip(model.blocks, before, strict=True))
                if not torch.equal(block.qkv.weight[0, :32], start)
            ]
        record = {"step": step, "train_loss": loss, "grad_norm": norm, "lr": lr,
                  "tokens": step * tokens_per_step, "tokens_each_language": step * tokens_per_step // len(LANGUAGES),
                  "tokens_including_retries": (step + metrics["overflow_retries"]) * tokens_per_step,
                  "train_tokens_per_second": round(step * tokens_per_step / metrics["training_seconds"], 2)}
        if recovery:
            record["model_step"] = profile.recovery_step + step
        if deadline is None or monotonic() < deadline:
            if step % profile.eval_every == 0 or (not recovery and step in (1, stop)):
                record["validation"] = evaluate(model, corpus, profile, "validation", device)
                metrics["evaluations"].append(record)
            if step == profile.steps:
                record["test"] = evaluate(model, corpus, profile, "test", device)
        print(json.dumps({"event": "step", **record}), flush=True)
        if step % profile.save_every == 0 or step == stop:
            save(record)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--max-session-seconds", type=float)
    parser.add_argument("--max-session-tokens", type=int)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", type=Path)
    start.add_argument("--resume", type=Path)
    args = parser.parse_args()
    profile = Profile(**json.loads(args.profile.read_bytes()))
    stop = args.stop_after if args.stop_after is not None else profile.steps
    if not 1 <= stop <= profile.steps:
        parser.error("Stop step must be within the profile")
    seconds = args.max_session_seconds
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        parser.error("--max-session-seconds must be positive and finite")
    if args.max_session_tokens is not None and args.max_session_tokens < 1:
        parser.error("--max-session-tokens must be positive")
    if profile.recovery_step is not None and (
        args.stop_after is None or seconds is None or args.max_session_tokens is None
    ):
        parser.error("Step-10000 recovery requires --stop-after, --max-session-seconds and --max-session-tokens")
    deadline = monotonic() + seconds if seconds is not None else None
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(args.output / ".training.lock", timeout=0):
        train(PackedCorpus(args.corpus), profile, args.resume or args.init, args.output, args.device, stop,
              resume=args.resume is not None, deadline=deadline, max_session_tokens=args.max_session_tokens)


if __name__ == "__main__":
    main()
