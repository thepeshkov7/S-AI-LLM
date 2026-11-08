"""Prepare reviewed bilingual instruction, grammar, dialogue and arithmetic data."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import pyarrow.parquet as parquet

from generate import format_instruction
from instruction_data import build_splits
from packed_corpus import LANGUAGES, SPLITS, digest
from prepare_oasst import dialogue, load_messages
from train import EOS
from training_1b.checkpoint import CONFIG, read_checkpoint, read_tokenizer
from training_1b.corpus import contains_probe, download, forbidden_prompts
from training_1b.evaluate import normalized

OASST = "OpenAssistant/oasst2"
OASST_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
GSM = "openai/gsm8k"
GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
REPO = Path(__file__).resolve().parents[1]


def split_for(group):
    bucket = int.from_bytes(sha256(group.encode()).digest()[:8], "big") % 20
    return {0: "validation", 1: "test"}.get(bucket, "train")


def sample(identifier, group, language, source, turns, split):
    return {"id": identifier, "group": group, "language": language, "source": source,
            "turns": turns, "split": split}


def openassistant(path):
    messages = load_messages(path)
    for identifier, reply in messages.items():
        tree, _, language, role, _ = reply
        if role == "assistant" and (turns := dialogue(messages, reply)):
            yield sample(f"oasst2-{identifier}", f"oasst2-{tree}", language, "oasst2", turns, split_for(tree))


def gsm8k(path, is_test):
    for batch in parquet.ParquetFile(path).iter_batches(columns=["question", "answer"], batch_size=128):
        for row in batch.to_pylist():
            question = row["question"].strip()
            explanation, separator, value = row["answer"].rpartition("####")
            if not separator or not re.fullmatch(r"-?[\d,.]+", value.strip()):
                continue
            rationale = re.sub(r"<<[^<>]+>>", "", explanation).strip()
            identifier = "gsm8k-" + sha256(normalized(question).encode()).hexdigest()[:24]
            split = "test" if is_test else ("validation" if split_for(identifier) == "validation" else "train")
            yield sample(identifier, identifier, "en", "gsm8k",
                         [question, rationale + "\nFinal answer: " + value.strip()], split)


def project_examples():
    for split, items in build_splits().items():
        if split not in SPLITS:
            continue
        for index, example in enumerate(items):
            yield sample(f"project-task-{split}-{index}", example.group, example.language, "project_tasks",
                         [example.prompt, example.answer], split)
    path = REPO / "data" / "assistant-practice-v1.tsv"
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line or line.startswith("#"):
            continue
        split, family, ru, ru_answer, en, en_answer = line.split("\t")
        if family == "translation":
            group = f"project-practice-{index}"
            yield sample(group + "-ru", group, "ru", "project_practice", [ru, ru_answer], split)
            yield sample(group + "-en", group, "en", "project_practice", [en, en_answer], split)


def arithmetic():
    for boxes in range(3, 13):
        for pencils in range(4, 19):
            for used in (2, 5, 9, 13):
                group = f"pencil-boxes-{boxes}-{pencils}-{used}"
                total = boxes * pencils
                if used > total:
                    continue
                left = total - used
                turns = {
                    "ru": [f"В каждой из коробок по {pencils} карандашей. Коробок всего {boxes}. "
                           f"Для рисунка взяли {used} карандашей. Сколько осталось? Объясни вычисления.",
                           f"Всего было {boxes} × {pencils} = {total} карандашей. После вычитания "
                           f"получаем {total} − {used} = {left}. Ответ — {left}."],
                    "en": [f"There are {boxes} boxes with {pencils} pencils in each. The class uses {used} pencils. "
                           "How many remain? Show the calculation.",
                           f"There were {boxes} × {pencils} = {total} pencils in all. Subtracting {used} "
                           f"leaves {total} − {used} = {left}. Answer: {left}."],
                }
                for lang, dialogue_turns in turns.items():
                    yield sample(group + "-" + lang, group, lang, "project_multistep", dialogue_turns,
                                 split_for(group))


def token_length(tokenizer, turns):
    prompt = "\n".join(format_instruction(user) + " " + answer for user, answer in
                       zip(turns[:-2:2], turns[1:-2:2], strict=True))
    query = format_instruction(turns[-2])
    prompt = prompt + "\n" + query if prompt else query
    return len(tokenizer.encode(prompt).ids) + len(tokenizer.encode(" " + turns[-1]).ids)


def build(output, cache, tokenizer):
    if output.exists():
        raise FileExistsError(f"Choose a new dataset output: {output}")
    assistant = download(OASST, OASST_REVISION, "2023-11-05_oasst2_ready.messages.jsonl.gz",
                         cache / "oasst2.jsonl.gz")
    gsm_train = download(GSM, GSM_REVISION, "main/train-00000-of-00001.parquet", cache / "gsm-train.parquet")
    gsm_test = download(GSM, GSM_REVISION, "main/test-00000-of-00001.parquet", cache / "gsm-test.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [*openassistant(assistant), *gsm8k(gsm_train, False), *gsm8k(gsm_test, True),
            *project_examples(), *arithmetic()]
    priority = {"test": 0, "validation": 1, "train": 2}
    rows.sort(key=lambda row: (priority[row["split"]], row["source"], row["id"]))
    prompts, groups, examples, identities = {}, {}, set(), set()
    forbidden = forbidden_prompts()
    rejected = Counter()
    selected = {split: [] for split in SPLITS}
    for row in rows:
        turns, group, split = row["turns"], row["group"], row["split"]
        queries = tuple(map(normalized, turns[::2]))
        if not all(queries) or any(EOS in text for text in turns):
            rejected["invalid"] += 1
        elif any(contains_probe(query, forbidden) for query in queries):
            rejected["probe"] += 1
        elif (row["id"] in identities or tuple(map(normalized, turns)) in examples
              or groups.get(group, split) != split or any(prompts.get(query, split) != split for query in queries)):
            rejected["duplicate_or_cross_split"] += 1
        elif token_length(tokenizer, turns) > CONFIG.context:
            rejected["context"] += 1
        else:
            groups[group] = split
            prompts.update((query, split) for query in queries)
            examples.add(tuple(map(normalized, turns)))
            identities.add(row["id"])
            selected[split].append(row)
    manifest = {
        "format_version": 1, "context": CONFIG.context, "rejected": rejected,
        "probes_sha256": digest(Path(__file__).with_name("prompts.jsonl")),
        "sources": {
            "oasst2": {"url": f"https://huggingface.co/datasets/{OASST}/tree/{OASST_REVISION}",
                       "license": "Apache-2.0", "sha256": digest(assistant)},
            "gsm8k": {"url": f"https://huggingface.co/datasets/{GSM}/tree/{GSM_REVISION}",
                      "license": "MIT", "notice": "https://github.com/openai/grade-school-math/blob/master/LICENSE",
                      "train_sha256": digest(gsm_train), "test_sha256": digest(gsm_test)},
            "project_tasks": {"source": "instruction_data.py"},
            "project_practice": {"source": "data/assistant-practice-v1.tsv"},
            "project_multistep": {"source": "training_1b/instructions.py"},
        },
        "splits": {},
    }
    with TemporaryDirectory(dir=output.parent, prefix=".instructions-") as temp:
        staged = Path(temp) / "data"
        staged.mkdir()
        for split, items in selected.items():
            path = staged / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in items:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts = Counter(row["language"] for row in items)
            if not all(counts[lang] for lang in LANGUAGES):
                raise ValueError(f"No fitting instruction examples for one language in {split}")
            manifest["splits"][split] = {
                "languages": counts,
                "sources": Counter(f'{row["source"]}/{row["language"]}' for row in items),
                "multi_turn": sum(len(row["turns"]) > 2 for row in items),
                "sha256": digest(path),
            }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = read_tokenizer(read_checkpoint(args.checkpoint))
    print(json.dumps(build(args.output, args.cache, tokenizer), indent=2))


if __name__ == "__main__":
    main()
