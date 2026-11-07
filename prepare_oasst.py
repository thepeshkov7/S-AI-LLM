"""Prepare reviewed Russian and English OpenAssistant dialogue examples."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from packed_corpus import LANGUAGES, SPLITS, digest
from train import EOS


SOURCE = "https://huggingface.co/datasets/OpenAssistant/oasst2"
LICENSE = "https://www.apache.org/licenses/LICENSE-2.0"
EXCLUDED_LABELS = ("spam", "pii", "lang_mismatch", "not_appropriate", "hate_speech", "sexual_content")
UNSUITABLE = re.compile(
    r"https?://|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\w)\+?\d(?:[ .()\-]*\d){8,}(?!\w)|"
    r"(?im:^\s*(?:User|Assistant):)|\b(?:open\s?assistant|chatgpt)\b",
    re.IGNORECASE,
)


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def clean_text(row):
    if (row["lang"] not in LANGUAGES or row["tree_state"] != "ready_for_export"
            or row["deleted"] or row["review_result"] is not True or row["synthetic"]):
        return None
    labels = row["labels"] or {}
    if any(labels.get(name, {}).get("value", 0) > 0 for name in EXCLUDED_LABELS):
        return None
    if any(labels.get(name, {}).get("value", 0) > 0.3 for name in ("toxicity", "violence")):
        return None
    detox = row.get("detoxify") or {}
    if any(detox.get(name, 0) > 0.25 for name in ("toxicity", "threat", "sexual_explicit")):
        return None
    is_answer = row["role"] == "assistant"
    if is_answer and (row["rank"] != 0 or labels.get("quality", {}).get("value", 0) < 0.45
                      or labels.get("fails_task", {}).get("value", 0) > 0):
        return None
    text = unicodedata.normalize("NFC", row["text"]).strip()
    if not (8 <= len(text) <= (800 if is_answer else 500)) or EOS in text or UNSUITABLE.search(text):
        return None
    return text


def load_messages(source):
    messages = {}
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = clean_text(row)
            if text is not None:
                messages[row["message_id"]] = (
                    row["message_tree_id"], row["parent_id"], row["lang"], row["role"], text,
                )
    return messages


def dialogue(messages, reply):
    tree, _, lang, _, _ = reply
    turns = []
    current = reply
    expected = "assistant"
    while current is not None and len(turns) < 4:
        current_tree, parent, current_lang, role, text = current
        if (current_tree, current_lang, role) != (tree, lang, expected):
            return None
        turns.append(text)
        if parent is None:
            return list(reversed(turns)) if role == "prompter" else None
        current = messages.get(parent)
        expected = "prompter" if expected == "assistant" else "assistant"
    return None


def prepare(source, output, exclude):
    messages = load_messages(source)
    forbidden = set()
    for path in exclude:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                prompt = json.loads(line)["prompt"]
                forbidden.add(normalized(prompt))
                for user_turn in re.findall(r"(?m)^User: (.+)$", prompt):
                    forbidden.add(normalized(user_turn))
    rows = {split: [] for split in SPLITS}
    roots = {}
    seen = set()
    for message_id, reply in messages.items():
        tree, _, lang, role, _ = reply
        if role != "assistant":
            continue
        turns = dialogue(messages, reply)
        if turns is None:
            continue
        user_turns = tuple(normalized(turn) for turn in turns[::2])
        if any(not turn or turn in forbidden for turn in user_turns):
            continue
        root = user_turns[0]
        if roots.setdefault(root, tree) != tree or user_turns in seen:
            continue
        seen.add(user_turns)
        bucket = int(sha256(tree.encode()).hexdigest()[:8], 16) % 20
        split = {0: "validation", 1: "test"}.get(bucket, "train")
        rows[split].append({"id": message_id, "tree": tree, "language": lang, "turns": turns})

    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "license": LICENSE, "raw_sha256": digest(source), "splits": {}}
    for split, items in rows.items():
        counts = Counter(row["language"] for row in items)
        if not all(counts[lang] for lang in LANGUAGES):
            raise ValueError(f"{split} lacks examples in one language")
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "languages": dict(counts), "multi_turn": sum(len(row["turns"]) == 4 for row in items),
            "sha256": digest(path),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.exclude), indent=2))


if __name__ == "__main__":
    main()
