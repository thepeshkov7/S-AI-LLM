"""Prepare a balanced Russian/English pilot from a dated Wikipedia text snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from time import sleep
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET = "wikimedia/wikipedia"
SNAPSHOT = "20231101"
DATASET_LICENSES = (
    "https://creativecommons.org/licenses/by-sa/3.0/",
    "https://www.gnu.org/licenses/fdl.html",
)
SOURCE_TERMS = "https://foundation.wikimedia.org/wiki/Terms_of_Use"
OFFSETS = {
    "ru": (10100, 32000, 179200, 301400, 580000, 742900),
    "en": (44100, 132000, 278000, 421400, 580100, 742900),
}
LENGTH = 50


def read_batch(cache_dir, language, offset):
    path = cache_dir / f"{language}-{offset}.json"
    if path.exists():
        return json.loads(path.read_text())["rows"]
    params = urlencode({
        "dataset": DATASET, "config": f"{SNAPSHOT}.{language}", "split": "train",
        "offset": offset, "length": LENGTH,
    })
    request = Request(
        "https://datasets-server.huggingface.co/rows?" + params,
        headers={"User-Agent": "SAI-LLM/0.2 (github.com/thepeshkov7/S-AI-LLM)"},
    )
    for retry in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                raw = response.read()
            result = json.loads(raw)
            rows = result["rows"]
            if len(rows) != LENGTH or any(row["truncated_cells"] for row in rows):
                raise ValueError(f"Incomplete Wikipedia batch for {language} at {offset}")
            path.write_bytes(raw)
            return rows
        except (OSError, TimeoutError):
            if retry == 2:
                raise
            sleep(2 ** retry)


def normalize(text, max_chars):
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    # End before editorial links and bibliographies when present.
    text = re.split(
        r"^\s*(?:References|External links|Further reading|Примечания|Литература|Ссылки)\s*$",
        text, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE,
    )[0]
    if len(text) > max_chars:
        cutoff = text.rfind("\n", max_chars * 3 // 4, max_chars)
        text = text[:cutoff if cutoff > 0 else max_chars].rsplit(" ", 1)[0]
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def prepare(output, max_chars, seed):
    cache = output / "source-cache"
    cache.mkdir(parents=True, exist_ok=True)
    requests = [(language, offset) for language, offsets in OFFSETS.items() for offset in offsets]
    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(lambda args: read_batch(cache, *args), requests))
    selected = {"ru": [], "en": []}
    seen = set()
    for (language, _), rows in zip(requests, batches):
        for item in rows:
            row = item["row"]
            text = normalize(row["text"], max_chars)
            digest = sha256(" ".join(text.split()).encode()).hexdigest()
            letters = re.findall(r"[^\W\d_]", text)
            main_alphabet = "[А-Яа-яЁё]" if language == "ru" else "[A-Za-z]"
            native = len(re.findall(main_alphabet, text))
            if len(text) < 700 or native < 0.85 * len(letters) or digest in seen:
                continue
            seen.add(digest)
            doc_id = f"wiki-{language}-{row['id']}"
            selected[language].append((
                {"id": doc_id, "language": language, "text": text},
                {"id": doc_id, "url": row["url"], "title": row["title"],
                 "language": language, "sha256": digest},
            ))
    assignments = {split: [] for split in ("train", "validation", "test")}
    rng = random.Random(seed)
    train_count = min(map(len, selected.values())) - 24
    if train_count < 40:
        raise RuntimeError("Insufficient clean documents in one or both languages")
    for items in selected.values():
        rng.shuffle(items)
        assignments["validation"].extend(items[:12])
        assignments["test"].extend(items[12:24])
        assignments["train"].extend(items[24:24 + train_count])
    manifest = {
        "dataset": DATASET, "snapshot": SNAPSHOT, "seed": seed,
        "dataset_license_metadata": DATASET_LICENSES, "article_source_terms": SOURCE_TERMS,
        "documents": [], "splits": {},
    }
    for split, items in assignments.items():
        rng.shuffle(items)
        content = "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc, _ in items).encode()
        (output / f"{split}.jsonl").write_bytes(content)
        manifest["splits"][split] = {"documents": len(items), "sha256": sha256(content).hexdigest()}
        manifest["documents"].extend({**meta, "split": split} for _, meta in items)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest["splits"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("corpus/wiki-pilot"))
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=1307)
    args = parser.parse_args()
    if args.max_chars < 700:
        parser.error("--max-chars must be at least 700")
    print(json.dumps(prepare(args.output, args.max_chars, args.seed), indent=2))


if __name__ == "__main__":
    main()
