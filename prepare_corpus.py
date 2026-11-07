"""Fetch and normalize the fixed S AI 0.1 public-domain research corpus."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


def normalize(text):
    return unicodedata.normalize("NFC", text).replace("\u00a0", " ").replace("\u202f", " ")


def read_document(entry, cache_dir):
    source = entry["url"]
    cache = cache_dir / f'{entry["id"]}.source'
    if not cache.is_file():
        request = Request(source, headers={"User-Agent": "SAI-LLM-Research/0.1 (github.com/thepeshkov7/S-AI-LLM)"})
        with urlopen(request, timeout=30) as response:
            cache.write_bytes(response.read())
    raw = cache.read_text(encoding="utf-8-sig")
    if entry["language"] == "en":
        match = re.search(
            r"^\*\*\* START OF [^\n]+\*\*\*\s*\n(.*?)^\*\*\* END OF ",
            raw,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"Missing book boundaries at {source}")
        paragraphs = [" ".join(item.split()) for item in re.split(r"\n\s*\n", match.group(1))]
    else:
        page = BeautifulSoup(raw, "html.parser")
        content = page.select_one("#mw-content-text .mw-parser-output")
        if content is None:
            raise ValueError(f"Missing article body at {source}")
        for node in content.select("table, .ws-noexport, .reference, .references, .mw-editsection"):
            node.decompose()
        body = content.select_one("div.text") or content
        paragraphs = [" ".join(p.get_text().split()) for p in body.select("p")]
    text = normalize("\n\n".join(p for p in paragraphs if p))
    if len(text) < 1_000:
        raise ValueError(f"Extracted book is too short at {source}")
    digest = sha256(text.encode()).hexdigest()
    if digest != entry["text_sha256"]:
        raise ValueError(f"Source text changed for {entry['id']}: {digest}")
    return {"id": entry["id"], "language": entry["language"], "text": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    manifest = Path(__file__).parent / "data" / "sources.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    cache = args.output / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        documents = list(pool.map(lambda entry: read_document(entry, cache), entries))
    for split in ("train", "validation", "test"):
        lines = [json.dumps(doc, ensure_ascii=False) for entry, doc in zip(entries, documents) if entry["split"] == split]
        (args.output / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {len(lines)} documents")


if __name__ == "__main__":
    main()
