"""Build a reproducible bilingual chat curriculum from local, attributed sources."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from packed_corpus import LANGUAGES, SPLITS, digest
from prepare_oasst import normalized


DATA = Path(__file__).with_name("data")
NAMES = {
    "ru": "Аня Борис Вика Галина Дима Ева Жанна Захар Илья Карина Лев Мила Никита Олег Полина Рита Семён Тамара Ульяна Фёдор Юра Яна Лиза Соня".split(),
    "en": "Amy Bob Vera Gail Dan Eva Jane Zack Eli Kara Leo Mia Nick Owen Polly Rita Sam Tara Lily Fred Joe Yana Liz Sonia".split(),
}
EVENTS = {
    "ru": ("Урок", "Концерт", "Тур", "Музей", "Встреча", "Созвон", "Секция", "Дебаты", "Ужин", "Тренировка", "Кино", "Обед"),
    "en": ("Lesson", "Concert", "Tour", "Museum", "Meeting", "Call", "Practice", "Debate", "Dinner", "Training", "Film", "Lunch"),
}


def source_rows(folder, split, manifest):
    path = folder / f"{split}.jsonl"
    if digest(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Source does not match its manifest: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def row(identifier, language, family, *turns):
    return {"id": identifier, "language": language, "family": family, "turns": turns}


def everyday_rows(split):
    for path in (DATA / "chat-topics.tsv", DATA / "bilingual-everyday.tsv"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line or line.startswith("#"):
                continue
            if path.name == "bilingual-everyday.tsv":
                assigned, ru, en, ru_answer, en_answer = line.split("\t")
            else:
                assigned, (ru, en, ru_answer, en_answer) = "train", line.split("\t")
            if assigned != split:
                continue
            for language, questions, answer in (("ru", ru, ru_answer), ("en", en, en_answer)):
                for index, question in enumerate(questions.split("|")):
                    yield row(f"{path.stem}-{number}-{language}-{index}", language, "everyday", question, answer)


def translation_rows(folder, split, manifest):
    if split == "test":
        return
    for pair in source_rows(folder, split, manifest):
        for language, original, instruction in (
            ("ru", "en", "Переведи на русский: "),
            ("en", "ru", "Translate to English: "),
        ):
            key = pair["provenance"][language]["id"]
            yield row(f"tatoeba-{language}-{key}", language, "translation",
                      instruction + pair[original], pair[language])


def arithmetic_rows(split):
    rng = random.Random(2019)
    operands = [(a, b) for a in range(2, 47) for b in range(2, min(a + 1, 16))]
    rng.shuffle(operands)
    sets = {"validation": operands[:45], "test": operands[45:90], "train": operands[90:]}
    for a, b in sets[split]:
        for lang in LANGUAGES:
            ru = lang == "ru"
            suffix = " Ответь числом." if ru else " Reply with the number."
            stories = (
                (f"У нас {a} открыток, нам подарили ещё {b}. Сколько теперь?",
                 f"Из {a} карандашей отдали {b}. Сколько осталось?",
                 f"Взяли {a} наборов по {b} наклеек. Сколько наклеек всего?",
                 f"Разложили {a * b} карточек поровну по {b} пакетам. Сколько в одном?")
                if ru else
                (f"We have {a} postcards and receive {b} more. How many do we have now?",
                 f"We gave away {b} of our {a} pencils. How many are left?",
                 f"We took {a} packs with {b} stickers in each. How many stickers is that?",
                 f"We put {a * b} cards into {b} bags equally. How many are in one bag?")
            )
            symbols = (f"{a} + {b}", f"{a} - {b}", f"{a} × {b}", f"{a * b} ÷ {b}")
            answers = (a + b, a - b, a * b, a)
            for index, (story, symbol, answer) in enumerate(zip(stories, symbols, answers, strict=True)):
                if rng.random() < 0.35:
                    story = f"Сколько будет {symbol}?" if ru else f"What is {symbol}?"
                yield row(f"math-{lang}-{a}-{b}-{index}", lang, "arithmetic", story + suffix, str(answer))


def scenario_rows(lang, number, rng):
    ru = lang == "ru"
    first, second, third = rng.sample(NAMES[lang], 3)
    event = rng.choice(EVENTS[lang])
    title = f"{event} {rng.randrange(1, 121)}"
    hour, new_hour = rng.sample(range(8, 22), 2)
    minute = rng.choice([0, 0, 0, 5, 10, 15, 20, 30, 40, 45, 50, 55])
    old, new = f"{hour}:{minute:02}", f"{new_hour}:{minute:02}"
    variant = rng.randrange(3)
    name = (first, second, third)[variant]
    direction = rng.choice(("before", "after"))
    ask_first = rng.choice((True, False))
    earlier = first if direction == "before" else second
    later = second if direction == "before" else first
    identifier = f"{lang}-{number}"

    if ru:
        scenario = f"За звонок отвечает {first}, за письмо — {second}, а за объявление — {third}."
        extract = (
            f"{scenario} Кто должен позвонить?",
            f"{scenario} Кто отправит письмо?",
            f"{scenario} Кто опубликует объявление?",
        )[variant] + " Напиши только имя."
        earlier_word = "раньше" if direction == "before" else "позже"
        who = "раньше" if ask_first else "позже"
        compare = f"{first} пришёл {earlier_word}, чем {second}. Кто пришёл {who}? Напиши имя."
        setup = f"Начало события «{title}» в {old}."
        acknowledgement = rng.choice(("Понял.", "Хорошо.", "Запомнил время."))
        update = (
            f"Время перенесли на {new}. Когда теперь начало?",
            f"Теперь оно начинается в {new}. Во сколько приходить?",
            f"Отмена старого времени. Событие будет в {new}. Во сколько теперь?",
        )[variant]
        format_request = (
            f"Составь JSON с полями title и time. Название: {title}. Время: {new}.",
            f"Время {new}, название {title}. Верни только JSON с ключами title и time.",
            f"Запиши в JSON время {new} и название {title}. Ключи title и time.",
        )[variant]
    else:
        scenario = f"{first} is making the call, {second} is sending the email, and {third} is posting the notice."
        extract = (
            f"{scenario} Who should call?",
            f"{scenario} Who will send the email?",
            f"{scenario} Who will post the notice?",
        )[variant] + " Give only the name."
        who = "earlier" if ask_first else "later"
        compare = f"{first} arrived {direction} {second}. Who arrived {who}? Give the name."
        setup = f"The event '{title}' starts at {old}."
        acknowledgement = rng.choice(("Okay.", "Got it.", "I noted the time."))
        update = (
            f"It was rescheduled to {new}. When does it start now?",
            f"It now starts at {new}. What time should I arrive?",
            f"Forget the previous time. It will take place at {new}. What is the new time?",
        )[variant]
        format_request = (
            f"Make a JSON object with title and time fields. Title: {title}. Time: {new}.",
            f"The time is {new} and the title is {title}. Return only JSON with keys title and time.",
            f"Put the time {new} and title {title} into JSON. Use the keys title and time.",
        )[variant]

    yield row(f"extract-{identifier}", lang, "extract", extract, name)
    yield row(f"reason-{identifier}", lang, "reason", compare, earlier if ask_first else later)
    yield row(f"followup-{identifier}", lang, "followup", setup, acknowledgement, update, new)
    yield row(f"format-{identifier}", lang, "format", format_request,
              json.dumps({"title": title, "time": new}, ensure_ascii=False, separators=(",", ":")))


def scenario_splits():
    splits = {split: [] for split in SPLITS}
    rng = random.Random(2503)
    for index in range(2600):
        bucket = index % 23
        split = {0: "validation", 1: "test"}.get(bucket, "train")
        for language in LANGUAGES:
            splits[split].extend(scenario_rows(language, index, rng))
    return splits


def forbidden_prompts(paths):
    prompts = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            prompt = json.loads(line)["prompt"]
            prompts.add(normalized(prompt))
            prompts.update(normalized(text.removeprefix("User: ")) for text in prompt.splitlines()
                           if text.startswith("User: "))
    return prompts


def prepare(dialogues, translations, output, exclude):
    oasst_manifest = json.loads((dialogues / "manifest.json").read_bytes())
    translations_manifest = json.loads((translations / "manifest.json").read_bytes())
    scenarios = scenario_splits()
    forbidden = forbidden_prompts(exclude)
    manifest = {"seed": [2019, 2503], "sources": {
        "oasst2_manifest": digest(dialogues / "manifest.json"),
        "tatoeba_manifest": digest(translations / "manifest.json"),
        "authored": {name: digest(DATA / name) for name in ("chat-topics.tsv", "bilingual-everyday.tsv")},
    }, "excluded_benchmarks": [digest(path) for path in exclude], "splits": {}}
    seen = set()
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        external = [
            row(item["id"], item["language"], "dialogue", *item["turns"])
            for item in source_rows(dialogues, split, oasst_manifest)
        ]
        items = [*everyday_rows(split), *external, *arithmetic_rows(split), *scenarios[split],
                 *translation_rows(translations, split, translations_manifest)]
        counts = Counter()
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as target:
            for item in items:
                user_turns = tuple(map(normalized, item["turns"][::2]))
                key = (item["language"], user_turns)
                if key in seen or any(turn in forbidden for turn in user_turns):
                    continue
                seen.add(key)
                counts[f'{item["language"]}:{item["family"]}'] += 1
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": dict(counts), "sha256": digest(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
