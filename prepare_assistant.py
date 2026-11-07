"""Filter public instruction data, retain bilingual replay and exclude evaluation prompts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

from evaluate_assistant import BENCHMARK, read_benchmark
from generate import TextGenerator
from packed_corpus import LANGUAGES, SPLITS, digest
from prepare_bilingual import DATA, row, source_rows
from prepare_grounded import bucket
from prepare_oasst import UNSUITABLE, normalized
from train import EOS
from tune_oasst import dialogue_example


AYA = "https://huggingface.co/datasets/CohereLabs/aya_dataset"
AYA_LICENSE = "https://www.apache.org/licenses/LICENSE-2.0"
PRACTICE = DATA / "assistant-practice-v1.tsv"
SAMPLING = {"everyday": 6, "dialogue": 8, "translation": 3, "arithmetic": 3,
            "extract": 3, "reason": 2, "followup": 3, "format": 3, "context": 3, "code": 3}
SELECTION = {"everyday": 4, "dialogue": 4, "translation": 2, "context": 2, "code": 2}
LETTERS = {"ru": re.compile("[А-Яа-яЁё]"), "en": re.compile("[A-Za-z]")}
OTHER_SCRIPT = re.compile(r"[^\W\d_]", re.UNICODE)


def trigrams(text):
    words = text.split()
    return set(zip(words, words[1:], words[2:]))


class BenchmarkExclusion:
    def __init__(self, path):
        self.prompts = {normalized(prompt) for case in read_benchmark(path)
                        for prompt in (case.prompt, case.first_prompt) if prompt}
        self.grams = {}
        self.index = defaultdict(set)
        for prompt in self.prompts:
            grams = trigrams(prompt)
            if len(grams) >= 4:
                self.grams[prompt] = grams
                for gram in grams:
                    self.index[gram].add(prompt)

    def matches(self, text):
        if text in self.prompts:
            return True
        grams = trigrams(text)
        if len(grams) < 4:
            return False
        candidates = set().union(*(self.index.get(gram, ()) for gram in grams))
        return any(len(grams & self.grams[prompt]) >= 0.8 * len(grams | self.grams[prompt])
                   for prompt in candidates)


def usable_text(text, language, limit):
    if not isinstance(text, str) or len(text) > limit:
        return None
    text = unicodedata.normalize("NFC", text).strip()
    if (not 3 <= len(text) <= limit or EOS in text or UNSUITABLE.search(text)
            or any(unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in text)):
        return None
    letters = len(OTHER_SCRIPT.findall(text))
    if letters < 3 or len(LETTERS[language].findall(text)) < 0.75 * letters:
        return None
    return text


def aya_rows(path):
    from pyarrow.parquet import ParquetFile

    names = {"eng": "en", "rus": "ru"}
    parquet = ParquetFile(path)
    columns = ["inputs", "targets", "language_code", "annotation_type"]
    for batch in parquet.iter_batches(batch_size=256, columns=columns):
        for entry in batch.to_pylist():
            lang = names.get(entry["language_code"])
            if lang is None or entry["annotation_type"] not in {"original-annotations", "re-annotations"}:
                continue
            prompt = usable_text(entry["inputs"], lang, 1000)
            answer = usable_text(entry["targets"], lang, 1800)
            if prompt is not None and answer is not None and normalized(prompt) != normalized(answer):
                key = normalized(prompt)
                yield bucket("aya:" + key), row("aya-" + key, lang, "dialogue", prompt, answer)


def authored_rows(path):
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        split, family, ru, ru_answer, en, en_answer = line.split("\t")
        if split not in SPLITS or family not in SAMPLING:
            raise ValueError(f"Unknown practice split or family at row {index}")
        yield split, row(f"practice-{index}-ru", "ru", family, ru, ru_answer)
        yield split, row(f"practice-{index}-en", "en", family, en, en_answer)


def prepare(previous, aya, base, output, benchmark):
    exclusion = BenchmarkExclusion(benchmark)
    generator = TextGenerator.load(base)
    previous_manifest = json.loads((previous / "manifest.json").read_bytes())
    manifest = {
        "sources": {"previous_sha256": digest(previous / "manifest.json"),
                    "aya": {"url": AYA, "license": AYA_LICENSE, "sha256": digest(aya)},
                    "authored_sha256": digest(PRACTICE)},
        "excluded_benchmark_sha256": digest(benchmark), "sampling_weights": SAMPLING,
        "selection_weights": SELECTION, "splits": {},
    }
    by_split = defaultdict(list)
    for split in SPLITS:
        by_split[split].extend(("prior", item) for item in source_rows(previous, split, previous_manifest))
    for split, item in authored_rows(PRACTICE):
        by_split[split].append(("authored", item))
    for split, item in aya_rows(aya):
        by_split[split].append(("aya", item))
    # Deduplicate complete user-turn sequences, giving priority to held-out sets.
    seen, rejected = set(), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix=".assistant-data-") as temporary:
        folder = Path(temporary) / "curriculum"
        folder.mkdir()
        for split in ("validation", "test", "train"):
            counts, sources = Counter(), Counter()
            path = folder / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as target:
                for source, item in by_split[split]:
                    turns = tuple(normalized(turn) for turn in item["turns"][::2])
                    if any(exclusion.matches(turn) for turn in turns):
                        rejected["benchmark"] += 1
                        continue
                    key = (item["language"], turns)
                    if key in seen or not all(turns):
                        rejected["duplicate"] += 1
                        continue
                    if dialogue_example(item, generator) is None:
                        rejected["too_long"] += 1
                        continue
                    seen.add(key)
                    counts[f'{item["language"]}:{item["family"]}'] += 1
                    sources[f'{source}:{item["language"]}'] += 1
                    target.write(json.dumps(item, ensure_ascii=False) + "\n")
            manifest["splits"][split] = {"rows": dict(counts), "sources": dict(sources), "sha256": digest(path)}
        manifest["rejected"] = dict(rejected)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "aya", "base", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
