"""Expand the bilingual curriculum with reviewed dialogue and varied copying tasks."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import random

from generate import TextGenerator
from packed_corpus import LANGUAGES, SPLITS, digest
from prepare_bilingual import DATA, forbidden_prompts, row, source_rows
from prepare_oasst import LICENSE, SOURCE, clean_text, dialogue, normalized
from tune_oasst import dialogue_example


SAMPLING = {
    "everyday": 4, "dialogue": 4, "translation": 2, "arithmetic": 5,
    "extract": 3, "reason": 3, "followup": 3, "format": 3, "context": 3,
}
SELECTION = {"everyday": 3, "dialogue": 3, "translation": 2, "context": 2}
NAMES = {
    "ru": "Алина Артём Варя Валерий Гриша Дарья Денис Злата Иван Игорь Клара Леонид Люба Маша Максим Марина Матвей Наташа Оксана Пётр Светлана Сергей Таня Юлия Эмма Яков".split(),
    "en": "Alice Arthur Beth Clara Greg Daisy Dean Fred Hannah Henry Iris Jack Laura Mason Mary Max Miles Nancy Olivia Peter Robin Sara Sophie Tom Emma Victor".split(),
}
TITLES = {
    "ru": ("Летний вечер", "Первый старт", "Путь домой", "Листок", "Старый город", "Беседа", "Горизонт",
           "Белый зал", "Новая глава", "Поход", "Открытый урок", "Пять минут", "Окно", "Мастерская"),
    "en": ("Summer Evening", "First Steps", "Way Home", "Leaf", "Old Town", "Conversation", "Horizon",
           "White Hall", "New Chapter", "Hike", "Open Class", "Five Minutes", "Window", "Workshop"),
}
PLACES = {
    "ru": ("на веранде", "у сцены", "в прихожей", "на столе", "в коробке", "под креслом", "у окна", "на кухне"),
    "en": ("on the porch", "by the stage", "in the hall", "on the desk", "in the box", "under the chair",
           "by the window", "in the kitchen"),
}


def bucket(value):
    number = int(sha256(value.encode()).hexdigest()[:8], 16) % 20
    return {0: "validation", 1: "test"}.get(number, "train")


def expanded_dialogues(source, generator):
    messages = {}
    ranks = {}
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = clean_text(item, answer_ranks=(0, 1), max_answer_chars=1600)
            if text is None:
                continue
            message_id = item["message_id"]
            messages[message_id] = (item["message_tree_id"], item["parent_id"], item["lang"], item["role"], text)
            if item["role"] == "assistant":
                ranks[message_id] = item["rank"]
    selected = set()
    for identifier in sorted(ranks, key=lambda item: (ranks[item], item)):
        reply = messages[identifier]
        tree, parent, lang, _, _ = reply
        if parent in selected:
            continue
        turns = dialogue(messages, reply)
        if turns is None:
            continue
        example = row(identifier, lang, "dialogue", *turns)
        if dialogue_example(example, generator) is not None:
            selected.add(parent)
            yield bucket(tree), example


def everyday_examples():
    for number, line in enumerate((DATA / "grounded-everyday.tsv").read_text(encoding="utf-8").splitlines()):
        if not line or line.startswith("#"):
            continue
        split, ru, en, ru_answer, en_answer = line.split("\t")
        for index, (ru_prompt, en_prompt) in enumerate(zip(ru.split("|"), en.split("|"), strict=True)):
            key = f"new-topic-{number}-{index}"
            yield split, row(key + "-ru", "ru", "everyday", ru_prompt, ru_answer)
            yield split, row(key + "-en", "en", "everyday", en_prompt, en_answer)
            yield split, row(key + "-switch-ru", "ru", "everyday", "Reply in Russian. " + en_prompt, ru_answer)
            yield split, row(key + "-switch-en", "en", "everyday", "Ответь по-английски. " + ru_prompt, en_answer)


def arithmetic_examples():
    rng = random.Random(5117)
    for larger in range(4, 56):
        for smaller in range(2, min(larger, 15)):
            split = bucket(f"numbers-{larger}-{smaller}")
            a, b = larger, smaller
            for lang in LANGUAGES:
                ru = lang == "ru"
                unit = rng.choice(("груш", "ручек", "жетонов", "гостей") if ru else ("pears", "pens", "tokens", "guests"))
                prefix = rng.choice(("", "", "Для игры. " if ru else "For a game. "))
                variants = (
                    ((f"Сначала было {a} {unit}. Появилось ещё {b}. Сколько всего?",
                      f"К {a} прибавили {b}. Чему равна сумма?",
                      f"Всего {a} взрослых и {b} детей. Сколько людей вместе?"),
                     (f"Было {a} {unit}, {b} ушло. Сколько ещё есть?",
                      f"Забрали {b} из {a}. Какой остаток?",
                      f"Вычти {b} из {a}. Что получилось?"),
                     (f"Есть {a} упаковок. В каждой по {b} {unit}. Сколько в сумме?",
                      f"Взяли {a} штук по {b} рублей. Какова общая цена?",
                      f"Умножь {a} на {b}."),
                     (f"{a * b} {unit} поделили на {b} равных частей. Сколько в одной?",
                      f"Раздай {a * b} предметов поровну {b} людям. Сколько получит каждый?",
                      f"Раздели {a * b} на {b}."))
                    if ru else
                    ((f"We had {a} {unit}. We received {b} more. How many do we have altogether?",
                      f"Add {b} to {a}. What is the total?",
                      f"There are {a} adults and {b} children. How many people is that?"),
                     (f"There were {a} {unit} and {b} went away. How many are left?",
                      f"Remove {b} from a set of {a}. What's left?",
                      f"Subtract {b} from {a}."),
                     (f"There are {a} packs with {b} {unit} per pack. How many altogether?",
                      f"Each item costs {b} dollars. What is the cost for {a} items?",
                      f"Multiply {a} by {b}."),
                     (f"{a * b} {unit} are split into {b} equal shares. How many per share?",
                      f"Share {a * b} items equally among {b} people. How many does each get?",
                      f"Divide {a * b} by {b}."))
                )
                symbols = (f"{a} + {b}", f"{a} - {b}", f"{a} * {b}", f"{a * b} / {b}")
                answers = (a + b, a - b, a * b, a)
                suffix = rng.choice((" Ответь числом.", " Укажи только результат.", ""))
                if not ru:
                    suffix = rng.choice((" Give only the number.", " Return only the result.", ""))
                for index, (stories, symbol, answer) in enumerate(zip(variants, symbols, answers, strict=True)):
                    question = rng.choice((*stories, f"{symbol} = ?"))
                    yield split, row(f"numeracy-{lang}-{a}-{b}-{index}", lang, "arithmetic",
                                     prefix + question + suffix, str(answer))


def context_examples(lang, index, rng):
    ru = lang == "ru"
    actor, other = rng.sample(NAMES[lang], 2)
    item, other_item = rng.sample(TITLES[lang], 2)
    first_place, second_place = rng.sample(PLACES[lang], 2)
    hour, new_hour = rng.sample(range(7, 23), 2)
    old, new = f"{hour}:{rng.choice((0, 0, 15, 30, 45)):02}", f"{new_hour}:{rng.choice((0, 0, 10, 30, 40)):02}"
    title = item + (f" {rng.randrange(1, 125)}" if index % 2 == 0 else "")
    earlier = rng.choice((True, False))
    ask_earlier = rng.choice((True, False))
    name = actor if earlier == ask_earlier else other
    key = f"new-context-{lang}-{index}"

    if ru:
        duties = rng.choice((
            (f"На встречу в {new} зовёт {actor}. За списки отвечает {other}.", "Кто зовёт на встречу?"),
            (f"{other} принёс материалы. {actor} сообщил об изменениях в {new}.", "Кто сообщил об изменениях?"),
            (f"Время в календаре изменилось с {old} на {new}. Об этом написал {actor}.", "Кто написал об изменении?"),
            (f"Ключи принёс {other}, а свет включил {actor}.", "Кто включил свет?"),
        ))
        relation = "раньше" if earlier else "позже"
        requested = "раньше" if ask_earlier else "позже"
        ordering = rng.choice((
            f"{actor} появился {relation}, чем {other}. Кто был {requested}?",
            f"{actor} подошёл {relation} {other}. Кто подошёл {requested}?",
            f"{actor} закончил {relation}, чем {other}. Кто закончил {requested}?",
        ))
        format_prompt = rng.choice((
            f"Нужен объект JSON с полями title и time. Название — {title}. Время — {new}.",
            f"Название: {title}; время: {new}. Верни только JSON. Ключи title и time.",
            f"Заполни поля title и time в JSON. Время {new}, название {title}.",
            f"Выведи только JSON с ключами title и time. Событие {title} начинается в {new}.",
        ))
        choice = (f"Первый вариант — «{item}».", rng.choice(("Хорошо.", "Понял.", "Записал.")),
                  f"Выбор изменился на «{other_item}». Какой вариант выбран сейчас?")
        location = (f"Тетрадь лежит {first_place}. Маркер лежит {second_place}.",
                    f"Где лежит {'тетрадь' if earlier else 'маркер'}?")
    else:
        duties = rng.choice((
            (f"{actor} invites everyone to the meeting at {new}. {other} keeps the list.", "Who invites everyone?"),
            (f"{other} brought the materials. {actor} announced the changes at {new}.", "Who announced the changes?"),
            (f"The calendar time changed from {old} to {new}. {actor} sent the update.", "Who sent the update?"),
            (f"{other} brought the keys and {actor} turned the light on.", "Who turned on the light?"),
        ))
        relation = "before" if earlier else "after"
        requested = "earlier" if ask_earlier else "later"
        ordering = rng.choice((
            f"{actor} showed up {relation} {other}. Who was there {requested}?",
            f"{actor} got there {relation} {other}. Who arrived {requested}?",
            f"{actor} finished {relation} {other}. Who finished {requested}?",
        ))
        format_prompt = rng.choice((
            f"Make a JSON object with title and time. Title — {title}. Time — {new}.",
            f"Title: {title}; time: {new}. Output JSON only. Use title and time as keys.",
            f"Fill in title and time in a JSON object. The time is {new}, the title is {title}.",
            f"Output only JSON with keys title and time. The event {title} begins at {new}.",
        ))
        choice = (f"The first option is '{item}'.", rng.choice(("Okay.", "Got it.", "Noted.")),
                  f"We switched to '{other_item}'. Which option is selected now?")
        location = (f"The notebook is {first_place}. The marker is {second_place}.",
                    f"Where is the {'notebook' if earlier else 'marker'}?")
    name_only = " Ответь только именем." if ru else " Give only the name."
    yield row(key + "-extract", lang, "extract", " ".join(duties) + name_only, actor)
    yield row(key + "-order", lang, "reason", ordering + name_only, name)
    yield row(key + "-json", lang, "format", format_prompt,
              json.dumps({"title": title, "time": new}, ensure_ascii=False, separators=(",", ":")))
    yield row(key + "-change", lang, "context", *choice, other_item)
    yield row(key + "-location", lang, "context", " ".join(location), first_place if earlier else second_place)


def prepare(previous, oasst, base, output, exclude):
    generator = TextGenerator.load(base)
    prior_manifest = json.loads((previous / "manifest.json").read_bytes())
    splits = {split: source_rows(previous, split, prior_manifest) for split in SPLITS}
    prior_training = {(item["language"], tuple(map(normalized, item["turns"][::2]))) for item in splits["train"]}
    rng = random.Random(5407)
    for index in range(3100):
        split = bucket(f"scenario-{index}")
        for lang in LANGUAGES:
            splits[split].extend(context_examples(lang, index, rng))
    for split, item in chain(arithmetic_examples(), everyday_examples(), expanded_dialogues(oasst, generator)):
        splits[split].append(item)
    excluded = forbidden_prompts(exclude)
    seen = set()
    manifest = {
        "base_sha256": digest(base), "prior_curriculum": digest(previous / "manifest.json"),
        "authored": digest(DATA / "grounded-everyday.tsv"), "sampling_weights": SAMPLING,
        "selection_weights": SELECTION, "source": SOURCE, "license": LICENSE, "raw_sha256": digest(oasst),
        "excluded_benchmarks": [digest(path) for path in exclude], "splits": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    for split in ("test", "validation", "train"):
        path = output / f"{split}.jsonl"
        counts = Counter()
        with path.open("w", encoding="utf-8") as handle:
            for item in splits[split]:
                turns = tuple(map(normalized, item["turns"][::2]))
                signature = (item["language"], turns)
                if signature in seen or any(turn in excluded for turn in turns):
                    continue
                if split != "train" and signature in prior_training:
                    continue
                seen.add(signature)
                counts[f'{item["language"]}:{item["family"]}'] += 1
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": dict(counts), "sha256": digest(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--oasst", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
