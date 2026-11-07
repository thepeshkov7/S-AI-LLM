"""Evaluate unassisted bilingual generation against a frozen 350-case challenge."""

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import unicodedata

import torch

from evaluate_chat import completions
from generate import TextGenerator
from memory_chat import chat_prompt, chat_prompt_within_context
from packed_corpus import LANGUAGES, digest


BENCHMARK = Path(__file__).parent / "data" / "assistant-eval-v1.tsv"
SECTIONS = {"questions": 60, "reading": 20, "instructions": 25, "math": 25, "translation": 15, "dialogues": 10}
MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class Case:
    id: str
    suite: str
    language: str
    prompt: str
    expected: str
    first_prompt: str | None = None
    first_expected: str | None = None


def read_benchmark(path=BENCHMARK):
    counts = Counter()
    cases = []
    section = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            section = line[1:]
            if section not in SECTIONS:
                raise ValueError(f"Unknown challenge section: {section}")
            continue
        fields = line.split("\t")
        width = {"reading": 5, "dialogues": 4}.get(section, 2)
        if section is None or len(fields) != 2 * width or not all(fields):
            raise ValueError(f"Invalid row in {section}: {line}")
        counts[section] += 1
        for index, language in enumerate(LANGUAGES):
            data = fields[index * width:(index + 1) * width]
            key = f"{section}-{counts[section]:02d}-{language}"
            if section == "reading":
                context, q1, a1, q2, a2 = data
                cases.extend([Case(key + "-1", "questions", language, context + "\n" + q1, a1),
                              Case(key + "-2", "questions", language, context + "\n" + q2, a2)])
            elif section == "dialogues":
                first, ack, followup, answer = data
                cases.append(Case(key, section, language, followup, answer, first, ack))
            else:
                prompt, answer = data
                cases.append(Case(key, section, language, prompt, answer))
    if counts != SECTIONS or len(cases) != 350:
        raise ValueError(f"Unexpected challenge distribution: {dict(counts)}")
    return cases


def normalized_answer(answer):
    text = unicodedata.normalize("NFKC", answer).casefold().replace("ё", "е")
    text = text.translate(str.maketrans({"’": "'", "−": "-", "–": "-", "—": "-"}))
    text = text.strip().strip("`\"«»“”").strip()
    if any(char.isalnum() for char in text):
        text = text.rstrip(".!?")
    return " ".join(re.sub(r",(?=\s)", "", text).split())


def unique_json_keys(items):
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("Duplicate JSON keys")
    return result


def score(response, expected):
    response = response.strip()
    if not response:
        return False
    if expected.startswith("!literal "):
        return response == expected.removeprefix("!literal ")
    if expected.startswith("!json "):
        try:
            actual = json.loads(response, object_pairs_hook=unique_json_keys)
        except ValueError:
            return False
        target = json.loads(expected.removeprefix("!json "))
        return json.dumps(actual, sort_keys=True) == json.dumps(target, sort_keys=True)
    if expected.startswith("!python "):
        try:
            actual = ast.dump(ast.parse(response, mode="eval"))
        except (SyntaxError, ValueError):
            return False
        return any(actual == ast.dump(ast.parse(target, mode="eval"))
                   for target in expected.removeprefix("!python ").split("|"))
    return normalized_answer(response) in {normalized_answer(target) for target in expected.split("|")}


def evaluate(checkpoint, benchmark, output, batch_size, threads):
    torch.set_num_threads(threads)
    generator = TextGenerator.load(checkpoint)
    cases = read_benchmark(benchmark)
    single = [case for case in cases if case.first_prompt is None]
    dialogues = [case for case in cases if case.first_prompt is not None]
    prompts = [chat_prompt(case.prompt, case.language) for case in single]
    responses = completions(generator, prompts, MAX_NEW_TOKENS, batch_size)
    predictions = [{**asdict(case), "response": response, "passed": score(response, case.expected)}
                   for case, response in zip(single, responses, strict=True)]
    first_prompts = [chat_prompt(case.first_prompt, case.language) for case in dialogues]
    first_responses = list(completions(generator, first_prompts, MAX_NEW_TOKENS, batch_size))
    second_prompts = [chat_prompt_within_context(generator, case.prompt, case.language,
                                               [(case.first_prompt, response)], MAX_NEW_TOKENS)
                      for case, response in zip(dialogues, first_responses, strict=True)]
    second_responses = completions(generator, second_prompts, MAX_NEW_TOKENS, batch_size)
    for case, first, second, prompt in zip(dialogues, first_responses, second_responses, second_prompts, strict=True):
        first_passed = score(first, case.first_expected)
        final_passed = score(second, case.expected)
        predictions.append({**asdict(case), "first_response": first, "first_passed": first_passed,
                            "context_retained": prompt != chat_prompt(case.prompt, case.language),
                            "response": second, "final_passed": final_passed, "passed": first_passed and final_passed})
    scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in predictions:
        lang, suite = item["language"], item["suite"]
        for key in ("overall", lang, suite, f"{suite}:{lang}"):
            scores[key]["total"] += 1
            scores[key]["correct"] += item["passed"]
    result = {
        "checkpoint_sha256": digest(checkpoint), "benchmark_sha256": digest(benchmark),
        "decoding": {"max_new_tokens": MAX_NEW_TOKENS, "strategy": "greedy", "external_tools": False},
        "scores": dict(scores), "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("Batch size and threads must be positive")
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
