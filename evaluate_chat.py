"""Run deterministic chat inference against a frozen JSONL benchmark."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

import torch

from generate import TextGenerator, format_instruction
from memory_chat import chat_prompt


def score_answer(method, response, reference):
    if method == "human":
        return None
    if method == "json":
        try:
            return json.loads(response) == reference
        except ValueError:
            return False
    if method == "time":
        return re.findall(r"(?<!\d)\d{1,2}:\d{2}(?!\d)", response) == [reference]
    if method == "exact":
        return response.casefold() == reference.casefold()
    raise ValueError(f"Unknown scoring method: {method}")


@torch.inference_mode()
def completions(generator, prompts, max_new_tokens, batch_size):
    """Greedy decoding with identical causal positions to single-request generation."""
    tokenizer = generator.tokenizer
    if tokenizer is None:
        raise ValueError("Chat benchmark requires a BPE checkpoint")
    model = generator.model
    device = model.token.weight.device
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer.encode_batch(prompts[start:start + batch_size])
        contexts = [item.ids[-model.context:] for item in encoded]
        outputs = [[] for _ in contexts]
        active = list(range(len(contexts)))
        for _ in range(max_new_tokens):
            if not active:
                break
            ids = [contexts[i] for i in active]
            width = max(map(len, ids))
            inputs = torch.tensor(
                [row + [generator.eos] * (width - len(row)) for row in ids], device=device,
            )
            last = torch.tensor([len(row) - 1 for row in ids], device=device)
            next_ids = model.next_scores(inputs, last).argmax(dim=-1).tolist()
            remaining = []
            for i, token in zip(active, next_ids):
                if token != generator.eos:
                    outputs[i].append(token)
                    contexts[i] = (contexts[i] + [token])[-model.context:]
                    remaining.append(i)
            active = remaining
        yield from (item.rstrip("\ufffd").strip() for item in tokenizer.decode_batch(outputs))


def case_prompt(case, language_prefix):
    prompt = case["prompt"]
    language = case["language"]
    if case["task"] == "followup":
        if language_prefix:
            first, rest = prompt.removeprefix("User: ").split("\nAssistant:", 1)
            return chat_prompt(first, language) + rest
        return prompt
    return chat_prompt(prompt, language) if language_prefix else format_instruction(prompt)


def evaluate(checkpoint, benchmark, output, max_new_tokens=48, batch_size=16, threads=4,
             language_prefix=False, completion_function=completions):
    torch.set_num_threads(threads)
    raw = benchmark.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line]
    generator = TextGenerator.load(checkpoint)
    prompts = [case_prompt(case, language_prefix) for case in cases]
    tallies = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []
    responses = completion_function(generator, prompts, max_new_tokens, batch_size)
    for case, response in zip(cases, responses, strict=True):
        passed = score_answer(case["score"], response, case["reference"])
        predictions.append({**case, "response": response, "passed": passed})
        if passed is None:
            continue
        for key in ("overall", case["language"], case["task"], f'{case["language"]}:{case["task"]}'):
            tallies[key]["total"] += 1
            tallies[key]["correct"] += int(passed)
    result = {
        "benchmark_sha256": sha256(raw).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "decoding": {"max_new_tokens": max_new_tokens, "greedy": True, "language_prefix": language_prefix},
        "automatic": tallies,
        "manual_cases": sum(case["score"] == "human" for case in cases),
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ("benchmark_sha256", "automatic", "manual_cases")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--language-prefix", action="store_true")
    args = parser.parse_args()
    if min(args.max_new_tokens, args.batch_size, args.threads) < 1:
        parser.error("Token, batch and thread limits must be positive")
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
