"""Train the original 1B CopyLLM and checkpoint the entire optimizer state."""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from shutil import disk_usage
from time import monotonic, perf_counter
from uuid import uuid4

import torch
from filelock import FileLock

from packed_corpus import LANGUAGES, PackedCorpus, digest
from pretrain_packed import learning_rate
from train import atomic_torch_save
from training_1b.checkpoint import RECOVERY_PARAMETERS, build_model, read_checkpoint, read_tokenizer


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
    mask_document_transitions: bool = False
    ru_micro_batches: int | None = None

    def __post_init__(self):
        if min(self.steps, self.sequence_length, self.micro_batch, self.accumulation,
               self.last_blocks, self.eval_every, self.eval_windows, self.save_every) < 1:
            raise ValueError("Profile sizes and intervals must be positive")
        if self.ru_micro_batches is None and self.accumulation % len(LANGUAGES):
            raise ValueError("Accumulation must give each language the same number of microbatches")
        if self.ru_micro_batches is not None and (
            type(self.ru_micro_batches) is not int or not 1 <= self.ru_micro_batches < self.accumulation
        ):
            raise ValueError("Each language needs at least one microbatch per step")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported precision")
        if self.optimizer not in {"adamw", "adamw8bit"} or self.objective not in {"vocabulary", "copy"}:
            raise ValueError("Unsupported optimizer or objective")
        if self.warmup_steps is not None and not 1 <= self.warmup_steps <= self.steps:
            raise ValueError("Warmup steps must be within the run")
        if self.mask_document_transitions and self.objective != "vocabulary":
            raise ValueError("Document transition masking requires the vocabulary objective")
        if self.recovery_step is not None and (
            self.recovery_step != 10000 or not self.one_pass or self.objective != "vocabulary"
            or self.precision != "bf16"
        ):
            raise ValueError("Step-10000 recovery requires BF16 and one-pass vocabulary training")

    @property
    def language_schedule(self):
        if self.ru_micro_batches is None:
            return LANGUAGES * (self.accumulation // len(LANGUAGES))
        # Distribute RU microbatches over the step and keep the default RU/EN order at 1:1.
        ru = self.ru_micro_batches
        return tuple("ru" if (i * ru) // self.accumulation != ((i - 1) * ru) // self.accumulation else "en"
                     for i in range(self.accumulation))


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


def loss_for(model, batch, profile, eos=None):
    x, y = batch[:, :-1], batch[:, 1:]
    with autocast(profile):
        if profile.objective == "vocabulary":
            if eos is None:
                return model.vocabulary_loss(x, y)
            # EOS is learned as a target. EOS -> start-of-next-document is ignored.
            return model.vocabulary_loss(x, y.masked_fill(x == eos, -100), reduction="sum")
        return model.answer_loss(x, y)


def make_batch(stream, starts, offsets, device):
    return stream[starts[:, None] + offsets].to(device=device, dtype=torch.long)


@torch.inference_mode()
def evaluate(model, corpus, profile, split, device):
    model.eval()
    stats = {}
    length = profile.sequence_length
    offsets = torch.arange(length + 1)
    eos = corpus.eos if profile.mask_document_transitions else None
    for lang in LANGUAGES:
        stream = corpus.splits[split][lang]
        available = (len(stream) - 1) // length
        windows = min(available, profile.eval_windows)
        if not windows:
            raise ValueError(f"{split}/{lang} lacks an evaluation window")
        starts = torch.arange(windows) * available // windows * length
        total = 0.0
        targets = 0
        for start in range(0, windows, profile.micro_batch):
            batch = make_batch(stream, starts[start:start + profile.micro_batch], offsets, "cpu")
            loss = loss_for(model, batch.to(device), profile, eos).item()
            if eos is None:
                total += loss * len(batch)
                targets += len(batch) * length
            else:
                total += loss
                targets += int(batch[:, :-1].ne(eos).sum())
        if not targets:
            raise ValueError(f"{split}/{lang} has no valid next-token targets")
        loss = total / (windows if eos is None else targets)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {split} loss for {lang}")
        stats[lang] = {"loss": loss, "tokens": targets}
    return stats


def one_pass_windows(corpus, profile):
    """Shuffle non-overlapping target windows without reusing a token position."""
    order = torch.Generator().manual_seed(profile.seed + 2)
    windows = {}
    for language, micro_batches in Counter(profile.language_schedule).items():
        needed = profile.steps * micro_batches * profile.micro_batch
        available = (len(corpus.splits["train"][language]) - 1) // profile.sequence_length
        if available < needed:
            raise ValueError(f"One-pass schedule needs {needed} windows for {language}, but has {available}")
        windows[language] = torch.randperm(available, generator=order)[:needed] * profile.sequence_length
    return windows


def update(model, optimizer, parameters, scaler, corpus, profile, sampler, device, step, windows):
    offsets = torch.arange(profile.sequence_length + 1)
    schedule = profile.language_schedule
    cursor = {lang: count * (step - 1) * profile.micro_batch for lang, count in Counter(schedule).items()}
    batches = []
    for lang in schedule:
        stream = corpus.splits["train"][lang]
        if windows is None:
            starts = torch.randint(len(stream) - profile.sequence_length, (profile.micro_batch,),
                                   generator=sampler)
        else:
            start = cursor[lang]
            starts = windows[lang][start:start + profile.micro_batch]
            cursor[lang] += profile.micro_batch
        batches.append((lang, make_batch(stream, starts, offsets, "cpu")))
    eos = corpus.eos if profile.mask_document_transitions else None
    targets = Counter()
    for lang, batch in batches:
        targets[lang] += batch[:, :-1].numel() if eos is None else int(batch[:, :-1].ne(eos).sum())
    if not all(targets.values()):
        raise ValueError("Each language needs valid next-token targets in every step")
    denominator = profile.accumulation if eos is None else sum(targets.values())
    # Retry the same examples at a lower scale after an FP16 overflow.
    for attempt in range(6):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _, batch in batches:
            loss = loss_for(model, batch.to(device), profile, eos)
            losses.append(loss.detach())
            scaler.scale(loss / denominator).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        total, grad_norm = torch.stack((torch.stack(losses).sum(), norm)).tolist()
        if not math.isfinite(total):
            raise FloatingPointError("Nonfinite forward loss")
        if math.isfinite(grad_norm):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return total / denominator, grad_norm, attempt, targets
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


def emit_record(event, *, log=None, session=None, started=None, **fields):
    record = {"event": event, **fields}
    if session is not None:
        record.update(session=session, utc=datetime.now(timezone.utc).isoformat(),
                      session_seconds=round(monotonic() - started, 3))
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    if log is not None:
        print(line, file=log, flush=True)


def recovery_base_step(payload, checkpoint, resume):
    if not resume and checkpoint.name == "step-010000.pt" and payload.get("step", 10000) == 10000:
        return 10000
    if (payload.get("run_format") != 1
            or payload.get("settings", {}).get("profile", {}).get("recovery_step") != 10000):
        raise ValueError("Use the original step-010000.pt or a saved state from a recovery stage")
    base_step = payload.get("base_model_step", 10000)
    step = payload.get("step")
    if (type(base_step) is not int or base_step < 10000 or type(step) is not int or step < 1
            or payload.get("metrics", {}).get("last", {}).get("model_step") != base_step + step):
        raise ValueError("Recovery state has inconsistent step metadata")
    return base_step if resume else base_step + step


def train(corpus, profile, checkpoint, output, device, stop, *, resume=False, require_1b=True,
          deadline=None, max_session_tokens=None, emit=emit_record):
    recovery = profile.recovery_step is not None
    if recovery and (deadline is None or max_session_tokens is None):
        raise ValueError("Step-10000 recovery requires explicit time and token budgets")
    if device == "cpu" and (profile.precision != "fp32" or profile.optimizer == "adamw8bit"):
        raise ValueError("CPU requires FP32 and standard AdamW")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this runtime")
    if recovery and torch.cuda.device_count() != 1:
        raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES for step-10000 recovery")
    if device == "cuda" and profile.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU has no native BF16 support")
    torch.manual_seed(profile.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sampler = torch.Generator().manual_seed(profile.seed + 1)
    payload = read_checkpoint(checkpoint)
    base_model_step = recovery_base_step(payload, checkpoint, resume) if recovery else None
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
    if not profile.mask_document_transitions:
        del profile_settings["mask_document_transitions"]
    if profile.ru_micro_batches is None:
        del profile_settings["ru_micro_batches"]
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
        emit("paused", step=first - 1, reason="time_budget")
        return payload["metrics"] if resume else None
    if recovery and profile.optimizer == "adamw":
        # FP32 parameters and two FP32 moments. Free space excludes the previous snapshot.
        required = 12 * RECOVERY_PARAMETERS + 2**30
        if disk_usage(output).free < required:
            raise OSError(f"The run directory needs at least {math.ceil(required / 2**30)} GiB free")
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
            "baseline": None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "training_seconds": 0.0, "overflow_retries": 0, "evaluations": [],
        }
        if deadline is None or monotonic() < deadline:
            metrics["baseline"] = evaluate(model, corpus, profile, "validation", device)
            emit("baseline", **metrics, runtime=runtime_stats(device))
    del payload
    emit("start", first_step=first, stop_after=stop, settings=settings, initial_sha256=initial_sha,
         optimizer_state="restored" if resume else "new", max_session_tokens=max_session_tokens)

    nominal_targets = {lang: count * profile.micro_batch * profile.sequence_length
                       for lang, count in Counter(profile.language_schedule).items()}
    target_totals = Counter(metrics.get("targets_by_language", {}))

    def save(record):
        step = record["step"]
        metrics.update({"last": record, "runtime": runtime_stats(device)})
        state = {
            "run_format": 1, "format_version": 1, "architecture": model.architecture,
            "config": asdict(model.config), "tokenizer": tokenizer.to_str(), "model": model.state_dict(),
            "step": step, "initial_sha256": initial_sha, "settings": settings, "metrics": metrics,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.get_state(),
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        }
        if recovery:
            state["base_model_step"] = base_model_step
        atomic_torch_save(state, state_path)
        report = output / "metrics.json"
        staged = report.with_suffix(".tmp")
        staged.write_text(json.dumps({"settings": settings, "metrics": metrics}, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
        staged.replace(report)
        emit("saved", step=step, path=str(state_path), runtime=metrics["runtime"])

    record = None
    for step in range(first, stop + 1):
        if deadline is not None and monotonic() >= deadline:
            if record is not None and record["step"] != metrics.get("last", {}).get("step"):
                save(record)
            emit("paused", step=step - 1, reason="time_budget")
            break
        step_started = perf_counter()
        model.train()
        lr = learning_rate(step, profile.steps, profile.lr, profile.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step == 1 and not recovery:
            before = [block.qkv.weight[0, :32].detach().clone() for block in model.blocks]
        loss, norm, retries, targets = update(
            model, optimizer, parameters, scaler, corpus, profile, sampler, device, step, windows
        )
        metrics["training_seconds"] += perf_counter() - step_started
        metrics["overflow_retries"] += retries
        if step == 1 and not recovery:
            metrics["updated_transformer_blocks"] = [
                index + 1 for index, (block, start) in enumerate(zip(model.blocks, before, strict=True))
                if not torch.equal(block.qkv.weight[0, :32], start)
            ]
        record = {"step": step, "train_loss": loss, "grad_norm": norm, "lr": lr,
                  "tokens": step * tokens_per_step,
                  "tokens_by_language": {lang: step * count for lang, count in nominal_targets.items()},
                  "tokens_including_retries": (step + metrics["overflow_retries"]) * tokens_per_step,
                  "train_tokens_per_second": round(step * tokens_per_step / metrics["training_seconds"], 2)}
        if profile.ru_micro_batches is None:
            record["tokens_each_language"] = step * tokens_per_step // len(LANGUAGES)
        if profile.mask_document_transitions:
            target_totals.update(targets)
            metrics["targets_by_language"] = dict(target_totals)
            record["targets_by_language"] = dict(target_totals)
        if recovery:
            record["model_step"] = base_model_step + step
        if deadline is None or monotonic() < deadline:
            if step % profile.eval_every == 0 or step == stop or (not recovery and step == 1):
                record["validation"] = evaluate(model, corpus, profile, "validation", device)
                metrics["evaluations"].append(record)
            if step == profile.steps:
                record["test"] = evaluate(model, corpus, profile, "test", device)
        emit("step", **record)
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
    started = monotonic()
    deadline = started + seconds if seconds is not None else None
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(args.output / ".training.lock", timeout=0):
        with (args.output / "training-log.jsonl").open("a", encoding="utf-8") as log:
            session = uuid4().hex

            def emit(event, **fields):
                emit_record(event, log=log, session=session, started=started, **fields)

            emit("session_open", stop_after=stop, max_session_seconds=seconds,
                 max_session_tokens=args.max_session_tokens)
            try:
                corpus = PackedCorpus(args.corpus, verify_documents=profile.recovery_step is not None)
                metrics = train(corpus, profile, args.resume or args.init, args.output, args.device, stop,
                                resume=args.resume is not None, deadline=deadline,
                                max_session_tokens=args.max_session_tokens, emit=emit)
            except Exception as error:
                emit("error", type=type(error).__name__, message=str(error))
                raise
            else:
                emit("session_closed", saved_step=(metrics or {}).get("last", {}).get("step", 0))


if __name__ == "__main__":
    main()
