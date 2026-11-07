"""Train a randomly initialized S AI Transformer on disjoint JSONL document sets."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import NamedTuple

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from model import TinyLLM


EOS = "<|end_of_document|>"


class Document(NamedTuple):
    id: str
    language: str
    text: str


def read_split(path, seen_ids, seen_hashes):
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = Document(**json.loads(line))
        if not document.id or document.language not in {"ru", "en"} or not document.text.strip():
            raise ValueError(f"Invalid document in {path}")
        digest = hashlib.sha256(" ".join(document.text.split()).encode()).hexdigest()
        if document.id in seen_ids or digest in seen_hashes:
            raise ValueError(f"Duplicate document in {path}: {document.id}")
        seen_ids.add(document.id)
        seen_hashes.add(digest)
        documents.append(document)
    if {document.language for document in documents} != {"ru", "en"}:
        raise ValueError(f"{path} must contain Russian and English documents")
    return documents


def train_tokenizer(documents, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS],
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        (line for doc in documents for line in doc.text.splitlines(keepends=True)), trainer
    )
    return tokenizer


def encode_documents(documents, tokenizer):
    eos = tokenizer.token_to_id(EOS)
    return [
        (doc.language, torch.tensor([eos, *tokenizer.encode(doc.text).ids, eos]))
        for doc in documents
    ]


class TrainingWindows:
    def __init__(self, documents, context, seed, balance_languages=False):
        self.context = context
        self.generator = torch.Generator().manual_seed(seed)
        self.balance_languages = balance_languages
        if balance_languages:
            self.samplers = [
                TrainingWindows([(lang, ids) for lang, ids in documents if lang == language],
                                context, seed + index + 1)
                for index, language in enumerate(("ru", "en"))
            ]
        else:
            self.documents = [ids for _, ids in documents if len(ids) > context]
            if not self.documents:
                raise ValueError("Training requires a document longer than the context in each sampling group")
            self.ends = torch.tensor([len(ids) - context for ids in self.documents]).cumsum(0)
            self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), self.ends[:-1]])

    def batch(self, size, device):
        if self.balance_languages:
            if size < 2 or size % 2:
                raise ValueError("Balanced batches require an even batch size")
            russian, english = (sampler.batch(size // 2, device) for sampler in self.samplers)
            return torch.cat([russian[0], english[0]]), torch.cat([russian[1], english[1]])
        positions = torch.randint(int(self.ends[-1]), (size,), generator=self.generator)
        indices = torch.searchsorted(self.ends, positions, right=True)
        starts = positions - self.offsets[indices]
        tokens = torch.stack([
            self.documents[index][start:start + self.context + 1]
            for index, start in zip(indices.tolist(), starts.tolist())
        ]).to(device)
        return tokens[:, :-1], tokens[:, 1:]


@torch.inference_mode()
def evaluate(model, documents, batch_size, device):
    """Score each target once; windows never cross document boundaries."""
    model.eval()
    totals = {language: [0.0, 0] for language in ("ru", "en")}
    context = model.context
    for language, ids in documents:
        windows = []
        for start in range(0, len(ids) - 1, context):
            chunk = ids[start:start + context + 1]
            window = torch.zeros(context + 1, dtype=torch.long)
            window[:len(chunk)] = chunk
            target = window[1:].clone()
            target[len(chunk) - 1:] = -100
            windows.append((window[:-1], target, len(chunk) - 1))
        for start in range(0, len(windows), batch_size):
            inputs, targets, counts = zip(*windows[start:start + batch_size])
            _, loss = model(torch.stack(inputs).to(device), torch.stack(targets).to(device))
            count = sum(counts)
            totals[language][0] += loss.item() * count
            totals[language][1] += count
    totals["overall"] = [sum(item[index] for item in totals.values()) for index in range(2)]
    return {
        language: {"tokens": count, "loss": total / count, "perplexity": math.exp(total / count)}
        for language, (total, count) in totals.items()
    }


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        temporary_path = Path(output.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint(path, model, tokenizer, metrics):
    atomic_torch_save(
        {
            "format_version": 1,
            "config": asdict(model.config),
            "tokenizer": tokenizer.to_str(),
            "model": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--checkpoint", type=Path, default=Path("s_ai_0_1.pt"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--balance-languages", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use CUDA BF16 autocast during training")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--context", type=int, default=128)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.threads) < 1 or args.vocab_size < 257:
        parser.error("Steps, batch size, eval interval, and threads must be positive; vocabulary must be at least 257")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be positive")
    if args.balance_languages and args.batch_size % 2:
        parser.error("Balanced batches require an even batch size")
    if args.bf16 and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--bf16 requires a CUDA device with BF16 support")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = monotonic()
    seen_ids, seen_hashes = set(), set()
    paths = {name: args.data_dir / f"{name}.jsonl" for name in ("train", "validation", "test")}
    splits = {name: read_split(path, seen_ids, seen_hashes) for name, path in paths.items()}
    tokenizer = train_tokenizer(splits["train"], args.vocab_size)
    encoded = {name: encode_documents(docs, tokenizer) for name, docs in splits.items()}
    model = TinyLLM(tokenizer.get_vocab_size(), args.width, args.heads, args.layers, args.context).to(device)
    train = TrainingWindows(encoded["train"], model.context, args.seed + 1, args.balance_languages)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    stats = {
        name: {
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "documents": len(docs),
            "tokens": sum(len(ids) for _, ids in encoded[name]),
            "tokens_by_language": {
                language: sum(len(ids) for lang, ids in encoded[name] if lang == language)
                for language in ("ru", "en")
            },
        }
        for name, docs in splits.items()
    }
    print(json.dumps({"device": device, "parameters": parameter_count, "data": stats}), flush=True)
    initial_validation = evaluate(model, encoded["validation"], args.batch_size, device)
    best_validation = initial_validation
    best_step = 0
    best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    print(json.dumps({"step": 0, "validation": initial_validation}), flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            x, y = train.batch(args.batch_size, device)
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext()
            with precision:
                _, loss = model(x, y)
            (loss / args.gradient_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate(model, encoded["validation"], args.batch_size, device)
            print(json.dumps({"step": step, "train_loss": loss.item(), "validation": validation}), flush=True)
            if validation["overall"]["loss"] < best_validation["overall"]["loss"]:
                best_validation = validation
                best_step = step
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.train()
    if best_step == 0:
        raise RuntimeError("No trained checkpoint improved validation over random initialization")
    model.load_state_dict(best_weights)
    test = evaluate(model, encoded["test"], args.batch_size, device)
    metrics = {
        "parameters": parameter_count,
        "device": device,
        "seed": args.seed,
        "steps": args.steps,
        "best_step": best_step,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_every": args.eval_every,
        "threads": args.threads,
        "torch": str(torch.__version__),
        "balance_languages": args.balance_languages,
        "bf16": args.bf16,
        "gradient_accumulation": args.gradient_accumulation,
        "tokens_processed": args.steps * args.batch_size * args.context * args.gradient_accumulation,
        "data": stats,
        "initial_validation": initial_validation,
        "validation": best_validation,
        "test": test,
        "seconds": round(monotonic() - started, 1),
    }
    save_checkpoint(args.checkpoint, model.cpu(), tokenizer, metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
