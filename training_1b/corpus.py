"""Build a licensed RU/EN language-model corpus with disjoint document splits."""
import argparse
from array import array
from collections import Counter
from contextlib import ExitStack
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import pyarrow.parquet as parquet

from packed_corpus import LANGUAGES, SPLITS, digest, valid_document_text
from prepare_pilot import DATASET_LICENSES, SNAPSHOT, SOURCE_TERMS, normalize
from train import EOS
from training_1b.checkpoint import read_checkpoint, read_tokenizer
from training_1b.evaluate import PROBES, normalized

WIKI = "wikimedia/wikipedia"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_SHARDS = {"ru": 21, "en": 41}
ALPHABET = {"ru": re.compile(r"[А-Яа-яЁё]"), "en": re.compile(r"[A-Za-z]")}
LETTER = re.compile(r"[^\W\d_]")
WORD = re.compile(r"[^\W\d_]+")
EXAM_CASES = Path(__file__).with_name("exam_v1.json")
MIN_ARTICLE_CHARS = 700
MAX_ARTICLE_CHARS = 28_000
# Caps are upper bounds. Stop scanning once only a small tail remains.
CAP_TAIL_TOKENS = 256


def download(dataset, revision, remote, local):
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_suffix(local.suffix + ".partial")
        request = Request(f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{remote}",
                          headers={"User-Agent": "S-AI-LLM/1.0 corpus builder"})
        with urlopen(request, timeout=90) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(local)
    return local


def forbidden_prompts():
    probes = {normalized(prompt)
              for line in PROBES.read_text(encoding="utf-8").splitlines() if line
              for prompt in json.loads(line)["turns"]}
    for case in json.loads(EXAM_CASES.read_bytes()):
        probes.add(normalized(case["prompt"]))
        if case.get("reference"):
            probes.add(normalized(case["prompt"] + case["reference"]))
    return probes - {""}


def contains_probe(canonical, forbidden):
    return any(" " + phrase + " " in " " + canonical + " " for phrase in forbidden)


def selected_shards(language, wanted):
    total = WIKI_SHARDS[language]
    count = min(wanted, total) if wanted else total
    return [round((i + 0.5) * total / count - 0.5) for i in range(count)]


def assignment(article):
    bucket = int.from_bytes(sha256(article.encode()).digest()[:8], "big") % 100
    return {0: "validation", 1: "test"}.get(bucket, "train")


def pack_articles(path, language, tokenizer, streams, provenance, seen, caps, forbidden):
    stats = Counter()
    split_counts = {split: Counter(tokens=0, documents=0) for split in SPLITS}
    eos = tokenizer.token_to_id(EOS)
    columns = ("id", "url", "title", "text")
    def capped(split):
        cap = caps[split]
        return bool(cap and cap - split_counts[split]["tokens"] <= min(CAP_TAIL_TOKENS, cap // 20))

    for batch in parquet.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
        if all(capped(split) for split in SPLITS):
            break
        accepted = []
        for row in batch.to_pylist():
            stats["examined"] += 1
            article = f"wiki-{language}-{row['id']}"
            split = assignment(article)
            if capped(split):
                continue
            text = normalize(row["text"], MAX_ARTICLE_CHARS)
            if (len(text) < MIN_ARTICLE_CHARS or not valid_document_text(text)
                    or len(ALPHABET[language].findall(text)) < 0.85 * len(LETTER.findall(text))):
                stats["quality_filtered"] += 1
                continue
            words = WORD.findall(text.casefold())
            if len(words) < 65 or len(set(words)) / len(words) < 0.22:
                stats["quality_filtered"] += 1
                continue
            canonical = normalized(text)
            if contains_probe(canonical, forbidden):
                stats["probe_filtered"] += 1
                continue
            whole = sha256(canonical.encode()).digest()
            prefix = sha256(" ".join(words[:80]).encode()).digest() if len(words) >= 80 else whole
            accepted.append((article, split, row, text, whole, prefix))
        if not accepted:
            continue
        encodings = tokenizer.encode_batch([item[3] for item in accepted], add_special_tokens=False)
        for item, encoding in zip(accepted, encodings, strict=True):
            article, split, row, _, whole, prefix = item
            counts = split_counts[split]
            available = caps[split] - counts["tokens"] if caps[split] else len(encoding.ids) + 1
            if not encoding.ids or eos in encoding.ids:
                stats["quality_filtered"] += 1
                continue
            if len(encoding.ids) + 1 > available:
                stats["cap_filtered"] += 1
                continue
            inserted = seen.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (article, whole, prefix))
            if inserted.rowcount != 1:
                stats["duplicate_filtered"] += 1
                continue
            tokens = array("i", encoding.ids)
            tokens.append(eos)
            tokens.tofile(streams[(split, language)])
            counts["tokens"] += len(tokens)
            counts["documents"] += 1
            provenance.write(json.dumps({
                "id": article, "language": language, "split": split, "title": row["title"],
                "url": row["url"], "normalized_sha256": whole.hex(), "tokens": len(tokens),
                "truncated": False,
            }, ensure_ascii=False) + "\n")
        seen.commit()
    return stats, split_counts


def build(output, cache, tokenizer, shard_count, train_limit):
    if output.exists():
        raise FileExistsError(f"Corpus output already exists: {output}")
    if sys.byteorder != "little" or array("i").itemsize != 4:
        raise ValueError("Packed corpus requires little-endian 32-bit integers")
    forbidden = forbidden_prompts()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".wiki-") as work:
        staged = Path(work) / "corpus"
        staged.mkdir()
        tokenizer_file = staged / "tokenizer.json"
        tokenizer.save(str(tokenizer_file))
        provenance_file = staged / "sources.jsonl.gz"
        totals = {split: {lang: Counter(tokens=0, documents=0) for lang in LANGUAGES} for split in SPLITS}
        source = {
            "dataset": WIKI, "revision": WIKI_REVISION, "snapshot": SNAPSHOT,
            "dataset_license_metadata": DATASET_LICENSES, "article_source_terms": SOURCE_TERMS,
            "modifications": "Normalized article text, trimmed editorial sections, capped character length, tokenized.",
            "quality": {"max_chars": MAX_ARTICLE_CHARS, "min_chars": MIN_ARTICLE_CHARS,
                        "language_script_share": 0.85, "document_split": "98/1/1 by article ID hash",
                        "deduplication": "Exact normalized document and first 80 normalized words",
                        "probe_filter": "Normalized literal phrase match"},
            "probes_sha256": digest(PROBES), "exam_sha256": digest(EXAM_CASES), "shards": [],
        }
        with ExitStack() as stack:
            seen = sqlite3.connect(Path(work) / "dedupe.sqlite")
            stack.callback(seen.close)
            seen.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, body BLOB UNIQUE, prefix BLOB UNIQUE)")
            streams = {(split, lang): stack.enter_context((staged / f"{split}.{lang}.bin").open("wb"))
                       for split in SPLITS for lang in LANGUAGES}
            raw_provenance = stack.enter_context(provenance_file.open("wb"))
            compressed = gzip.GzipFile(filename="", fileobj=raw_provenance, mode="wb", mtime=0)
            provenance = stack.enter_context(TextIOWrapper(compressed, encoding="utf-8"))
            for lang in LANGUAGES:
                shards = selected_shards(lang, shard_count)
                holdout_limit = min(1_000_000, max(65_536, train_limit // 40)) if train_limit else 1_000_000
                for index, shard in enumerate(shards):
                    name = f"train-{shard:05d}-of-{WIKI_SHARDS[lang]:05d}.parquet"
                    remote = f"{SNAPSHOT}.{lang}/{name}"
                    path = download(WIKI, WIKI_REVISION, remote, cache / f"{lang}-{name}")
                    caps = {split: ((total * (index + 1)) // len(shards) - totals[split][lang]["tokens"] if total else 0)
                            for split, total in (("train", train_limit), ("validation", holdout_limit),
                                                 ("test", holdout_limit))}
                    counts, splits = pack_articles(path, lang, tokenizer, streams, provenance, seen, caps, forbidden)
                    info = {"file": remote, "sha256": digest(path), "quality_counts": dict(counts), "splits": splits}
                    source["shards"].append(info)
                    for split in SPLITS:
                        totals[split][lang].update(splits[split])
                    print(json.dumps(info), flush=True)
        if any(totals[split][lang]["tokens"] < 257 for split in SPLITS for lang in LANGUAGES):
            raise ValueError("One or more splits are too short for 256-token evaluation")
        source["attribution_sha256"] = digest(provenance_file)
        source_path = staged / "source-manifest.json"
        source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "format_version": 1, "tokenizer_sha256": digest(tokenizer_file),
            "source_manifest_sha256": digest(source_path),
            "splits": {split: {lang: {**totals[split][lang], "sha256": digest(staged / f"{split}.{lang}.bin")}
                               for lang in LANGUAGES} for split in SPLITS},
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--wiki-shards", type=int, default=2, help="Per language; 0 uses the whole snapshot")
    parser.add_argument("--limit-tokens", type=int, default=5_000_000, help="Train cap per language; 0 uses all")
    args = parser.parse_args()
    if args.wiki_shards < 0 or args.limit_tokens < 0 or 0 < args.limit_tokens < 100_000:
        parser.error("Use a nonnegative shard count and a train cap of at least 100000 or 0")
    tokenizer = read_tokenizer(read_checkpoint(args.checkpoint))
    print(json.dumps(build(args.output, args.cache, tokenizer, args.wiki_shards, args.limit_tokens), indent=2))


if __name__ == "__main__":
    main()
