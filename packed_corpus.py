"""Pack a bilingual JSONL corpus into memory-mapped, EOS-delimited token streams."""

import argparse
from array import array
from collections import Counter
from hashlib import file_digest, sha256
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer

from train import Document, EOS, train_tokenizer


LANGUAGES = ("ru", "en")
SPLITS = ("train", "validation", "test")


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def documents(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = Document(**json.loads(line))
            if not row.id or row.language not in LANGUAGES or not row.text.strip():
                raise ValueError(f"Invalid document in {path}")
            yield row


def tokenizer_sample(paths, per_language):
    counts = Counter()
    for doc in (doc for path in paths for doc in documents(path)):
        if counts[doc.language] < per_language:
            yield doc
            counts[doc.language] += 1
        if sum(counts.values()) == len(LANGUAGES) * per_language:
            break


def pack_split(sources, output, split, tokenizer, seen_ids, seen_hashes):
    eos = tokenizer.token_to_id(EOS)
    counts = Counter()
    tokens = Counter()
    paths = {lang: output / f"{split}.{lang}.bin" for lang in LANGUAGES}
    with paths["ru"].open("wb") as russian, paths["en"].open("wb") as english:
        streams = {"ru": russian, "en": english}
        rows = (doc for source in sources for doc in documents(source))
        while batch := list(islice(rows, 64)):
            encoded = tokenizer.encode_batch([row.text for row in batch])
            for doc, sequence in zip(batch, encoded, strict=True):
                fingerprint = sha256(" ".join(doc.text.split()).encode()).digest()
                if doc.id in seen_ids or fingerprint in seen_hashes:
                    raise ValueError(f"Duplicate document: {doc.id}")
                seen_ids.add(doc.id)
                seen_hashes.add(fingerprint)
                payload = array("i", sequence.ids)
                payload.append(eos)
                payload.tofile(streams[doc.language])
                counts[doc.language] += 1
                tokens[doc.language] += len(payload)
    if not all(counts[lang] for lang in LANGUAGES):
        raise ValueError(f"{split} must contain both languages")
    return {lang: {"tokens": tokens[lang], "documents": counts[lang], "sha256": digest(paths[lang])}
            for lang in LANGUAGES}


def pack(sources, output, vocab_size, tokenizer_docs):
    if array("i").itemsize != 4 or sys.byteorder != "little":
        raise RuntimeError("Token streams require little-endian 32-bit integers")
    if output.exists():
        raise FileExistsError(f"Packed corpus already exists at {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".packing-") as temp:
        staged = Path(temp) / "corpus"
        staged.mkdir()
        paths = {split: [source / f"{split}.jsonl" for source in sources] for split in SPLITS}
        source_digests = {split: [digest(path) for path in items] for split, items in paths.items()}
        tokenizer = train_tokenizer(tokenizer_sample(paths["train"], tokenizer_docs), vocab_size)
        tokenizer_path = staged / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        ids, hashes = set(), set()
        splits = {name: pack_split(path, staged, name, tokenizer, ids, hashes) for name, path in paths.items()}
        if source_digests != {split: [digest(path) for path in items] for split, items in paths.items()}:
            raise ValueError("Source JSONL changed while packing")
        manifest = {
            "format_version": 1,
            "tokenizer_sha256": digest(tokenizer_path),
            "tokenizer_docs_per_language": tokenizer_docs,
            "source_sha256": source_digests,
            "splits": splits,
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return manifest


class PackedCorpus:
    def __init__(self, directory):
        manifest = directory / "manifest.json"
        self.manifest = json.loads(manifest.read_bytes())
        self.fingerprint = digest(manifest)
        tokenizer = directory / "tokenizer.json"
        if self.manifest["format_version"] != 1 or digest(tokenizer) != self.manifest["tokenizer_sha256"]:
            raise ValueError("Invalid packed corpus manifest or tokenizer")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.splits = {}
        for split in SPLITS:
            self.splits[split] = {}
            for lang in LANGUAGES:
                info = self.manifest["splits"][split][lang]
                path = directory / f"{split}.{lang}.bin"
                size = info["tokens"]
                if path.stat().st_size != size * 4 or digest(path) != info["sha256"]:
                    raise ValueError(f"Packed token stream does not match its manifest: {path}")
                self.splits[split][lang] = torch.from_file(str(path), shared=False, size=size, dtype=torch.int32)

    def batch(self, size, context, random_source, device):
        ranges = torch.arange(context + 1)
        rows = []
        for lang in LANGUAGES:
            tokens = self.splits["train"][lang]
            starts = torch.randint(len(tokens) - context, (size // len(LANGUAGES),), generator=random_source)
            rows.append(tokens[starts[:, None] + ranges])
        batch = torch.cat(rows).to(device=device, dtype=torch.long)
        return batch[:, :-1], batch[:, 1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=6144)
    parser.add_argument("--tokenizer-docs", type=int, default=10000)
    args = parser.parse_args()
    if args.vocab_size < 257 or args.tokenizer_docs < 1:
        parser.error("Vocabulary must be at least 257 and tokenizer document cap must be positive")
    print(json.dumps(pack(args.source, args.output, args.vocab_size, args.tokenizer_docs), indent=2))


if __name__ == "__main__":
    main()
