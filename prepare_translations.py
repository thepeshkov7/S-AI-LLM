"""Prepare attributed, directly linked Tatoeba pairs for instruction training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


ALLOWED_LICENSES = {
    "CC BY 2.0 FR": "https://creativecommons.org/licenses/by/2.0/fr/",
    "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}
API = "https://api.tatoeba.org/v1/sentences"


def initial_url(source, target):
    return API + "?" + urlencode({
        "lang": source, "trans:lang": target, "trans:is_direct": "yes",
        "is_unapproved": "no", "trans:is_unapproved": "no",
        "word_count": "5-15", "sort": "created", "limit": 50,
    })


def fetch_pages(cache, source, target, pages):
    url = initial_url(source, target)
    for index in range(pages):
        if not url:
            break
        location = urlsplit(url)
        if location.scheme != "https" or location.netloc != "api.tatoeba.org" or location.path != "/v1/sentences":
            raise ValueError("Unexpected Tatoeba pagination destination")
        path = cache / f"{source}-{index}.json"
        if not path.exists():
            request = Request(url, headers={"User-Agent": "SAI-LLM/0.2 (github.com/thepeshkov7/S-AI-LLM)"})
            with urlopen(request, timeout=25) as response:
                payload = response.read()
            json.loads(payload)
            path.write_bytes(payload)
        page = json.loads(path.read_bytes())
        for original in page["data"]:
            for translated in original.get("translations", []):
                if translated["lang"] == target and translated.get("is_direct") and not translated.get("is_unapproved"):
                    yield (original, translated) if source == "rus" else (translated, original)
                    break
        url = page["paging"].get("next")


def provenance(row):
    return {
        "id": row["id"], "author": row["owner"],
        "url": f"https://tatoeba.org/en/sentences/show/{row['id']}",
        "license": ALLOWED_LICENSES[row["license"]],
        "sha256": sha256(row["text"].strip().encode()).hexdigest(),
    }


def prepare(output, pages, seed):
    cache = output / "source-cache"
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(lambda langs: list(fetch_pages(cache, *langs, pages)),
                                (("eng", "rus"), ("rus", "eng"))))
    pairs = []
    seen = {"rus": set(), "eng": set()}
    for batch in batches:
        for russian, english in batch:
            items = (russian, english)
            if any(row["license"] not in ALLOWED_LICENSES or not row["owner"] or
                   not 15 <= len(row["text"].strip()) <= 160 for row in items):
                continue
            normalized = [row["text"].strip().casefold() for row in items]
            if any(text in seen[row["lang"]] for row, text in zip(items, normalized)):
                continue
            for row, text in zip(items, normalized):
                seen[row["lang"]].add(text)
            pairs.append({"ru": russian["text"].strip(), "en": english["text"].strip(),
                          "provenance": {"ru": provenance(russian), "en": provenance(english)}})
    if len(pairs) < 100:
        raise RuntimeError(f"Only {len(pairs)} verified translation pairs")
    random.Random(seed).shuffle(pairs)
    holdout = max(20, len(pairs) // 10)
    splits = {"train": pairs[holdout:], "validation": pairs[:holdout]}
    manifest = {"source": "https://tatoeba.org", "seed": seed, "splits": {}}
    for name, items in splits.items():
        payload = "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in items).encode()
        (output / f"{name}.jsonl").write_bytes(payload)
        manifest["splits"][name] = {"pairs": len(items), "sha256": sha256(payload).hexdigest(),
                                     "attribution": [pair["provenance"] for pair in items]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {name: len(items) for name, items in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("corpus/translation-pilot"))
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1807)
    args = parser.parse_args()
    if args.pages < 2:
        parser.error("--pages must be at least two")
    print(json.dumps(prepare(args.output, args.pages, args.seed)))


if __name__ == "__main__":
    main()
