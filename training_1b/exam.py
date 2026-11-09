"""Run the fixed bilingual exam, unseen questions, or held-out prose checks."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata

import torch

from generate import TextGenerator, format_instruction
from packed_corpus import digest


FIXTURE = Path(__file__).with_name("exam_v1.json")
SETTINGS = {
    "max_new_tokens": 64, "temperature": 0.85, "top_k": 40, "top_p": 0.95,
    "repetition_penalty": 1.2, "frequency_penalty": 0.1, "no_repeat_ngram_size": 3,
}
BASE_SEED = 42
SUITES = ("exam", "generalization", "prose")
RUBRIC = {
    "coherence": ["No understandable thought", "Partially coherent", "Coherent throughout"],
    "grammar": ["Mostly malformed", "Understandable with errors", "Natural and grammatical"],
    "correctness": ["Wrong or absent", "Partly correct", "Correct and complete"],
    "instruction_following": ["Ignores request", "Partly follows it", "Meets all requested constraints"],
    "repetition": ["Dominated by loops", "Noticeable distracting repetition", "No distracting repetition"],
}


def normalize_answer(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def repeated_fourgram_fraction(text):
    chars = re.sub(r"\s+", "", text.casefold())
    counts = Counter(chars[index:index + 4] for index in range(len(chars) - 3))
    total = sum(counts.values())
    return round((total - len(counts)) / total, 6) if total else 0.0


@torch.inference_mode()
def prose_loss(generator, text, device):
    ids = generator.tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) < 2:
        raise ValueError("A held-out prose passage must contain at least two tokens")
    context = generator.model.context
    count = len(ids) - 1
    total_nll = 0.0
    for start in range(0, count, context):
        tokens = torch.tensor([ids[start:start + context + 1]], dtype=torch.long, device=device)
        loss = generator.model.vocabulary_loss(tokens[:, :-1], tokens[:, 1:]).item()
        total_nll += loss * (tokens.shape[1] - 1)
    if not math.isfinite(total_nll):
        raise FloatingPointError("Nonfinite held-out prose loss")
    return {"mean_nll": total_nll / count, "target_tokens": count}


def run_cases(generator, cases, suite, device):
    rows = []
    for index, case in enumerate(cases):
        if suite != "all" and case["suite"] != suite:
            continue
        prose = case["suite"] == "prose"
        prompt = case["prompt"] if prose else format_instruction(case["prompt"])
        seed = BASE_SEED + index
        answer, count = generator.complete(prompt, seed=seed, preserve_output=True, **SETTINGS)
        expected = case.get("exact_answers")
        exact = None if expected is None else normalize_answer(answer) in set(map(normalize_answer, expected))
        rows.append({
            **case, "model_input": prompt, "seed": seed, "raw_answer": answer, "generated_tokens": count,
            "exact_match": exact, "repeated_character_fourgram_fraction": repeated_fourgram_fraction(answer),
            "reference_lm": prose_loss(generator, case["prompt"] + case["reference"], device) if prose else None,
            "human_review": {
                criterion: None for criterion in RUBRIC
                if not prose or criterion not in ("correctness", "instruction_following")
            },
        })
    return rows


def summarize(rows):
    result = {}
    for suite in SUITES:
        for language in ("ru", "en"):
            group = [row for row in rows if row["suite"] == suite and row["language"] == language]
            if not group:
                continue
            scorable = [row["exact_match"] for row in group if row["exact_match"] is not None]
            refs = [row["reference_lm"] for row in group if row["reference_lm"] is not None]
            ratings = {}
            for criterion in RUBRIC:
                applicable = [row["human_review"][criterion] for row in group if criterion in row["human_review"]]
                graded = [score for score in applicable if score is not None]
                if any(type(score) is not int or score not in (0, 1, 2) for score in graded):
                    raise ValueError(f"Human score for {criterion} must be null or an integer from 0 to 2")
                if applicable:
                    ratings[criterion] = {"graded": len(graded), "total": len(applicable),
                                          "mean_out_of_2": sum(graded) / len(graded) if graded else None}
            fully_graded = ratings and all(row["graded"] == row["total"] for row in ratings.values())
            result.setdefault(suite, {})[language] = {
                "examples": len(group),
                "exact_matches": sum(scorable),
                "exact_total": len(scorable),
                "mean_repeated_character_fourgram_fraction": round(
                    sum(row["repeated_character_fourgram_fraction"] for row in group) / len(group), 6
                ),
                "human_review": ratings,
                "human_score_out_of_100": round(
                    50 * sum(row["mean_out_of_2"] * row["total"] for row in ratings.values())
                    / sum(row["total"] for row in ratings.values()), 2
                ) if fully_graded else None,
            }
            if refs:
                tokens = sum(ref["target_tokens"] for ref in refs)
                nll = sum(ref["mean_nll"] * ref["target_tokens"] for ref in refs) / tokens
                result[suite][language]["reference_lm"] = {
                    "target_tokens": tokens, "mean_nll": nll, "perplexity": math.exp(nll),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--summarize", type=Path, help="Summarize a generated report with completed human_review fields")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    if args.summarize is not None:
        report = json.loads(args.summarize.read_text(encoding="utf-8"))
        print(json.dumps(summarize(report["rows"]), ensure_ascii=False, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("Choose a new --output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    checkpoint_sha256 = digest(args.checkpoint)
    generator = TextGenerator.load(args.checkpoint, args.device)
    if generator.tokenizer is None and args.suite in ("all", "prose"):
        parser.error("Held-out prose requires the model tokenizer in the checkpoint")
    rows = run_cases(generator, cases, args.suite, args.device)
    summary = summarize(rows)
    report = {
        "source": "generated_from_checkpoint",
        "checkpoint_sha256": checkpoint_sha256,
        "fixture_sha256": digest(FIXTURE),
        "torch_version": str(torch.__version__),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "dtype": str(generator.model.token.weight.dtype),
        "generation_head": "vocabulary" if generator.vocabulary_only else "checkpoint_default",
        "decoding": SETTINGS, "human_review_rubric": RUBRIC, "summary": summary, "rows": rows,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
