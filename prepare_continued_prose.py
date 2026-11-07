"""Combine existing prose with fresh, source-disjoint Wikipedia validation and test sets."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from packed_corpus import LANGUAGES, SPLITS, digest, documents
from prepare_bilingual import source_rows
from train import Document


def fingerprint(document):
    return sha256(" ".join(document.text.split()).encode()).hexdigest()


def prepare(previous, fresh, dialogue, output):
    old = {split: list(documents(previous / f"{split}.jsonl")) for split in SPLITS}
    old_ids = {doc.id for items in old.values() for doc in items}
    old_texts = {fingerprint(doc) for items in old.values() for doc in items}
    dialogue_manifest = json.loads((dialogue / "manifest.json").read_bytes())
    dialogue_train = [Document("oasst2-" + row["id"], row["language"], "\n\n".join(row["turns"]))
                      for row in source_rows(dialogue, "train", dialogue_manifest)]
    manifest = {"sources": {name: digest(folder / "manifest.json") for name, folder in
                            (("previous_wiki", previous), ("fresh_wiki", fresh), ("oasst2", dialogue))},
                "splits": {}}
    seen_ids, seen_texts = set(), set()
    output.mkdir(parents=True, exist_ok=False)
    for split in ("validation", "test", "train"):
        incoming = [doc for doc in documents(fresh / f"{split}.jsonl")
                    if doc.id not in old_ids and fingerprint(doc) not in old_texts]
        if split == "train":
            incoming = [*old["train"], *incoming, *dialogue_train]
        path = output / f"{split}.jsonl"
        counts = Counter()
        with path.open("w", encoding="utf-8") as stream:
            for doc in incoming:
                key = fingerprint(doc)
                if doc.id in seen_ids or key in seen_texts:
                    continue
                seen_ids.add(doc.id)
                seen_texts.add(key)
                counts[doc.language] += 1
                stream.write(json.dumps(doc._asdict(), ensure_ascii=False) + "\n")
        if not all(counts[lang] for lang in LANGUAGES):
            raise ValueError(f"{split} requires data in both languages")
        manifest["splits"][split] = {"documents": dict(counts), "sha256": digest(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "fresh", "dialogue", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
