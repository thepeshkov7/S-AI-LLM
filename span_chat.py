"""Train an experimental whole-span decoder over a frozen bilingual checkpoint."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

import torch
from torch import nn
import torch.nn.functional as F

from evaluate_chat import completions, evaluate
from generate import TextGenerator
from packed_corpus import LANGUAGES, digest
from prepare_bilingual import source_rows
from train import atomic_torch_save
from tune_oasst import dialogue_example


OPEN, COPY, EVENT_JSON = range(3)
ADD, SUBTRACT, SUBTRACT_REVERSED, MULTIPLY, DIVIDE = range(3, 8)
COPY_FAMILIES = {"extract", "reason", "followup", "context"}
TIME = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d")


class SpanHead(nn.Module):
    def __init__(self, width, modes=3):
        super().__init__()
        self.context = nn.Linear(width, 96)
        self.tokens = nn.Linear(width, 96)
        self.pointers = nn.Linear(96, 4)
        self.mode = nn.Sequential(nn.Linear(width, 96), nn.GELU(), nn.Linear(96, modes))

    def forward(self, hidden, ends, allowed):
        query = hidden[torch.arange(len(hidden), device=hidden.device), ends]
        features = torch.tanh(self.tokens(hidden) + self.context(query)[:, None])
        pointers = self.pointers(features).transpose(1, 2)
        return self.mode(query), pointers.masked_fill(~allowed, -1e4)


@dataclass
class Prompt:
    text: str
    ids: list
    offsets: list
    starts: list
    ends: list


def prepare_prompt(text, generator):
    encoded = generator.tokenizer.encode(text)
    context = generator.model.context
    ids = encoded.ids[-context:]
    offsets = []
    starts = []
    ends = []
    for left, right in encoded.offsets[-context:]:
        value = text[left:right]
        start = left + len(value) - len(value.lstrip())
        end = left + len(value.rstrip())
        offsets.append((start, end))
        starts.append(start < end and (start == 0 or not text[start - 1].isalnum() or not text[start].isalnum()))
        ends.append(start < end and (end == len(text) or not text[end].isalnum() or not text[end - 1].isalnum()))
    return Prompt(text, ids, offsets, starts, ends)


def target_span(prompt, value):
    for match in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", prompt.text):
        starts = [index for index, (start, _) in enumerate(prompt.offsets)
                  if start == match.start() and prompt.starts[index]]
        ends = [index for index, (_, end) in enumerate(prompt.offsets)
                if end == match.end() and prompt.ends[index]]
        if starts and ends and starts[0] <= ends[-1]:
            return starts[0], ends[-1]
    return None


def integer_operands(text):
    if re.search(r"\d[.,:]\d|(?<!\w)-\d", text):
        return None
    matches = re.findall(r"(?<!\w)\d+(?!\w)", text)
    if len(matches) != 2 or any(len(value) > 4 for value in matches):
        return None
    return tuple(map(int, matches))


def calculate(mode, operands):
    left, right = operands
    if mode == ADD:
        return left + right
    if mode == SUBTRACT:
        return left - right
    if mode == SUBTRACT_REVERSED:
        return right - left
    if mode == MULTIPLY:
        return left * right
    if mode == DIVIDE and right and left % right == 0:
        return left // right
    return None


def arithmetic_mode(item, example):
    if not item["id"].startswith(("math-", "numeracy-")):
        return OPEN
    operands = integer_operands(example.prompt)
    if operands is None:
        return OPEN
    operation = int(item["id"].rsplit("-", 1)[-1])
    modes = ((ADD,), (SUBTRACT, SUBTRACT_REVERSED), (MULTIPLY,), (DIVIDE,))[operation]
    return next((mode for mode in modes if str(calculate(mode, operands)) == example.answer), OPEN)


def read_examples(folder, split, generator, cap, with_arithmetic):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    items = source_rows(folder, split, manifest)
    random.Random(567 if split == "train" else 568).shuffle(items)
    groups = defaultdict(list)
    for item in items:
        family = item["family"]
        mode = COPY if family in COPY_FAMILIES else EVENT_JSON if family == "format" else OPEN
        example = dialogue_example(item, generator)
        if example is None:
            continue
        if family == "arithmetic" and with_arithmetic:
            mode = arithmetic_mode(item, example)
        key = item["language"], mode
        if len(groups[key]) >= cap:
            continue
        prompt = prepare_prompt(example.prompt, generator)
        labels = [-100] * 4
        if mode == COPY:
            span = target_span(prompt, example.answer)
            if span is None:
                continue
            labels[:2] = span
        elif mode == EVENT_JSON:
            values = json.loads(example.answer)
            title = target_span(prompt, values["title"])
            time = target_span(prompt, values["time"])
            if title is None or time is None:
                continue
            labels = [*title, *time]
        groups[key].append((prompt, mode, labels))
    for lang in LANGUAGES:
        for mode in range(8 if with_arithmetic else 3):
            if not groups[lang, mode]:
                raise ValueError(f"Missing training examples for {lang}/{mode}")
    return groups


def encode_batch(prompts, generator):
    size = max(len(prompt.ids) for prompt in prompts)
    inputs = torch.tensor([prompt.ids + [generator.eos] * (size - len(prompt.ids)) for prompt in prompts])
    ends = torch.tensor([len(prompt.ids) - 1 for prompt in prompts])
    masks = torch.tensor([
        [prompt.starts + [False] * (size - len(prompt.ids)), prompt.ends + [False] * (size - len(prompt.ids))]
        for prompt in prompts
    ]).repeat(1, 2, 1)
    with torch.no_grad():
        return generator.model._encode(inputs), ends, masks


def batch_loss(head, batch, generator):
    prompts, modes, labels = zip(*batch, strict=True)
    hidden, ends, masks = encode_batch(prompts, generator)
    mode, pointers = head(hidden, ends, masks)
    targets = torch.tensor(labels).flatten()
    token_loss = F.cross_entropy(pointers.flatten(0, 1), targets, reduction="sum")
    return F.cross_entropy(mode, torch.tensor(modes)) + token_loss / (targets != -100).sum().clamp_min(1)


def train(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new span checkpoint path: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(605)
    rng = random.Random(606)
    generator = TextGenerator.load(args.base)
    modes = 8 if args.with_arithmetic else 3
    head = SpanHead(generator.model.config.width, modes)
    training = read_examples(args.curriculum, "train", generator, 3400, args.with_arithmetic)
    dev = read_examples(args.curriculum, "validation", generator, 64, args.with_arithmetic)
    dev_rows = [item for group in dev.values() for item in group]
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0018)
    best = float("inf")
    print(json.dumps({"training": {f"{lang}:{mode}": len(items) for (lang, mode), items in training.items()},
                      "validation": len(dev_rows)}), flush=True)
    for step in range(1, args.steps + 1):
        head.train()
        batch = [rng.choice(group) for group in training.values() for _ in range(2 if args.with_arithmetic else 3)]
        optimizer.zero_grad(set_to_none=True)
        batch_loss(head, batch, generator).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 150 and step != args.steps:
            continue
        head.eval()
        # Dev batches keep positives so the pointer loss has supervised targets.
        rng_dev = random.Random(607)
        rng_dev.shuffle(dev_rows)
        with torch.no_grad():
            losses = [batch_loss(head, dev_rows[i:i + 48], generator).item() for i in range(0, len(dev_rows), 48)]
        value = sum(losses) / len(losses)
        print(json.dumps({"step": step, "validation_loss": value}), flush=True)
        if value < best:
            best = value
            atomic_torch_save({"state": head.state_dict(), "modes": modes, "base_sha256": digest(args.base),
                               "curriculum_sha256": digest(args.curriculum / "manifest.json"),
                               "best_step": step, "validation_loss": value}, args.output)


def span_value(prompt, pointers, pattern=None):
    start, end = pointers
    # Evaluate all whole-token boundaries so rare multi-token values can be copied intact.
    candidates = []
    for left, (begin, _) in enumerate(prompt.offsets):
        if not prompt.starts[left]:
            continue
        for right in range(left, min(left + 24, len(prompt.ids))):
            _, finish = prompt.offsets[right]
            if not prompt.ends[right] or not 0 < finish - begin <= 72:
                continue
            value = prompt.text[begin:finish]
            if "\n" in value or (pattern is not None and not pattern.fullmatch(value)):
                continue
            candidates.append((left, right, value))
    if not candidates:
        return None
    left, right = zip(*[(left, right) for left, right, _ in candidates])
    scores = start[list(left)] + end[list(right)]
    return candidates[int(scores.argmax())][2]


@torch.inference_mode()
def select_spans(generator, head, prompts, typed_only=False):
    prepared = [prepare_prompt(text, generator) for text in prompts]
    hidden, ends, allowed = encode_batch(prepared, generator)
    mode_logits, pointers = head(hidden, ends, allowed)
    confidence, modes = mode_logits.softmax(dim=-1).max(dim=-1)
    for prompt, mode, probability, scores in zip(prepared, modes, confidence, pointers, strict=True):
        if probability < 0.85 or mode == OPEN or (mode == COPY and typed_only):
            yield None, "generator"
        elif mode == COPY:
            value = span_value(prompt, scores[:2])
            yield value, "span" if value is not None else "generator"
        elif mode == EVENT_JSON:
            # Limit serialization to the schema on which this head was trained.
            words = set(re.findall(r"\b\w+\b", prompt.text.casefold()))
            title = span_value(prompt, scores[:2])
            time = span_value(prompt, scores[2:], TIME)
            if {"json", "title", "time"} <= words and title and time:
                yield json.dumps({"title": title, "time": time}, ensure_ascii=False), "json"
            else:
                yield None, "generator"
        else:
            operands = integer_operands(prompt.text)
            value = calculate(int(mode), operands) if operands is not None else None
            yield (str(value), "arithmetic") if value is not None else (None, "generator")


def score(args):
    state = torch.load(args.head, map_location="cpu", weights_only=True)
    if state["base_sha256"] != digest(args.base):
        raise ValueError("Span head requires its original frozen checkpoint")
    sources = []

    def replies(generator, prompts, max_new_tokens, batch_size):
        head = SpanHead(generator.model.config.width, state.get("modes", 3)).eval()
        head.load_state_dict(state["state"])
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            values, origins = zip(*select_spans(generator, head, chunk, args.typed_only), strict=True)
            sources.extend(origins)
            fallback = completions(generator, [text for text, value in zip(chunk, values, strict=True) if value is None],
                                   max_new_tokens, batch_size)
            yield from (next(fallback) if value is None else value for value in values)

    result = evaluate(args.base, args.benchmark, args.output, max_new_tokens=args.max_new_tokens,
                      threads=args.threads, language_prefix=True, completion_function=replies)
    report = json.loads(args.output.read_bytes())
    report["span_head_sha256"] = digest(args.head)
    report["typed_only"] = args.typed_only
    for row, source in zip(report["predictions"], sources, strict=True):
        row["answer_source"] = source
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    for name in ("train", "score"):
        command = modes.add_parser(name)
        command.add_argument("--base", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--threads", type=int, default=4)
        if name == "train":
            command.add_argument("--curriculum", type=Path, required=True)
            command.add_argument("--steps", type=int, default=900)
            command.add_argument("--with-arithmetic", action="store_true")
            command.set_defaults(run=train)
        else:
            command.add_argument("--head", type=Path, required=True)
            command.add_argument("--benchmark", type=Path, required=True)
            command.add_argument("--max-new-tokens", type=int, default=48)
            command.add_argument("--typed-only", action="store_true")
            command.set_defaults(run=score)
    args = parser.parse_args()
    if args.threads < 1 or getattr(args, "steps", 1) < 1 or getattr(args, "max_new_tokens", 1) < 1:
        parser.error("Threads, steps and output token limits must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
