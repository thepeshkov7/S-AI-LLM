"""Add source-disjoint, procedurally authored short tasks to the bilingual curriculum."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from evaluate_assistant import BENCHMARK
from generate import TextGenerator
from packed_corpus import LANGUAGES, SPLITS, digest
from prepare_assistant import BenchmarkExclusion
from prepare_bilingual import row, source_rows
from prepare_oasst import normalized
from tune_oasst import dialogue_example


SEED = 107238
LABELS = {
    "ru": ("Жасмин", "Витраж", "Овраг", "Зорька", "Мелодия", "Жёлудь", "Морось", "Ветерок", "Склон", "Фарфор",
           "Лоскут", "Лаванда", "Якорь", "Галька", "Заслон", "Парус", "Гребень", "Барсук", "Утёс", "Ковыль"),
    "en": ("Jasmine", "Glasswork", "Gully", "Daybreak", "Melody", "Acorn", "Drizzle", "Breeze", "Hillside", "China",
           "Patchwork", "Lavender", "Anchor", "Pebble", "Shelter", "Sail", "Ridge", "Badger", "Crag", "Feathergrass"),
}
COLORS = {"ru": ("янтарный", "лиловый", "алый", "бежевый", "бирюзовый", "оливковый", "лимонный", "медный"),
          "en": ("amber", "mauve", "scarlet", "beige", "turquoise", "olive", "lemon", "copper")}
PEOPLE = {"ru": ("Василиса", "Галина", "Лев", "Антон", "Вера", "Андрей", "Фёдор", "Маргарита", "Кира", "Жанна"),
          "en": ("Hazel", "Gail", "Leo", "Anton", "Faith", "Andrew", "Theo", "Margot", "Kira", "Joan")}
WEIGHTS = {"reading": 9, "short_output": 7, "session": 7, "applied_math": 4, "everyday": 5,
           "dialogue": 5, "translation": 2, "arithmetic": 2, "extract": 2, "reason": 2,
           "followup": 2, "format": 2, "context": 2, "code": 2}


def read_task(lang, a, b, wanted, mode, rng):
    ru = lang == "ru"
    codes = rng.sample(range(201, 999), 2)
    clocks = [f"{h}:{rng.choice((5, 12, 18, 23, 31, 44, 56)):02d}" for h in rng.sample(range(7, 21), 2)]
    colors = rng.sample(COLORS[lang], 2)
    people = rng.sample(PEOPLE[lang], 2)
    quantities = rng.sample(range(2, 94), 2)
    places = rng.sample(LABELS[lang], 2)
    values = (codes, clocks, colors, people, quantities, places)
    texts = (
        ("У посылки «{a}» номер {x}, у «{b}» номер {y}.", "Какой номер у посылки «{q}»?"),
        ("Начало занятия «{a}» в {x}. Встреча «{b}» назначена на {y}.", "Во сколько начинается «{q}»?"),
        ("На стенде «{a}» указан цвет {x}, на стенде «{b}» указан цвет {y}.", "Какой цвет указан на «{q}»?"),
        ("За заказ «{a}» отвечает {x}. За заказ «{b}» отвечает {y}.", "Кто отвечает за «{q}»?"),
        ("Для группы «{a}» подготовили {x} билетов, для группы «{b}» — {y}.", "Сколько билетов у группы «{q}»?"),
        ("Плакат «{a}» висит в зале «{x}». Плакат «{b}» висит в зале «{y}».", "В каком зале плакат «{q}»?"),
    ) if ru else (
        ("Parcel {a} has number {x}, while {b} has number {y}.", "What number does parcel {q} have?"),
        ("The {a} class starts at {x}. The {b} meeting is at {y}.", "What time does {q} start?"),
        ("The {a} board is marked with {x}, while {b} is marked with {y}.", "What color is marked on {q}?"),
        ("{x} is in charge of order {a}; {y} is in charge of order {b}.", "Who is in charge of {q}?"),
        ("Group {a} has {x} tickets, and group {b} has {y}.", "How many tickets does group {q} have?"),
        ("The {a} poster is in room {x}. The {b} poster is in room {y}.", "Which room has the {q} poster?"),
    )
    context, question = texts[mode]
    x, y = values[mode]
    text = context.format(a=a, b=b, x=x, y=y)
    question = question.format(q=(a, b)[wanted])
    prompt = rng.choice((text + " " + question, text + "\n" + question, question + "\n" + text))
    return prompt, str((x, y)[wanted])


def output_task(lang, labels, mode, rng):
    ru = lang == "ru"
    first, second, third = labels
    if mode == 0:
        token = f"{rng.choice('ACGHKMPST')}_{rng.randrange(125, 987)}"
        return (f"Перепиши без изменений и пояснений: {token}" if ru else f"Copy this exactly and give nothing else: {token}"), token
    if mode == 1:
        pos = rng.randrange(len(labels))
        ordinal = ("первое", "второе", "третье")[pos] if ru else ("first", "second", "third")[pos]
        line = ", ".join(labels)
        return (f"Выведи {ordinal} слово из списка «{line}»." if ru else f"Output the {ordinal} word from the list “{line}”."), labels[pos]
    if mode == 2:
        return (f"Одним словом, запиши «{first.lower()}» заглавными буквами." if ru else
                f"Give “{first.lower()}” in capital letters as a single word."), first.upper()
    if mode == 3:
        return (f"Раздели слова «{first}» и «{second}» косой чертой, без пробелов." if ru else
                f"Join “{first}” and “{second}” with a slash, without spaces."), f"{first}/{second}"
    field = rng.choice(("label", "room", "tag", "group", "zone", "record"))
    number = rng.randrange(3, 88)
    prompt = (f"Сделай JSON с полем `{field}` для названия «{third}» и полем `size` для числа {number}." if ru else
              f"Return JSON using `{field}` for the name “{third}” and `size` for the number {number}.")
    return prompt, json.dumps({field: third, "size": number}, ensure_ascii=False)


def session_task(lang, topic, other, mode, rng):
    ru = lang == "ru"
    notes = (("Понял", "Да", "Принято", "Хорошо", "Записал") if ru else ("Noted", "Yes", "OK", "Got it", "Done"))
    ack = rng.choice(notes)
    code1, code2 = rng.sample(range(210, 991), 2)
    x, y = rng.sample(COLORS[lang], 2)
    variants = (
        (f"Для этой беседы код «{topic}» — {code1}, код «{other}» — {code2}. Ответь «{ack}».",
         f"Какой код у «{other}»?", str(code2)),
        (f"Сначала выбрали цвет {x}, потом заменили его на {y}. Напиши «{ack}».",
         "Какой цвет выбрали в итоге?", y),
        (f"Ключ от «{topic}» в кармане, а ключ от «{other}» на полке. Подтверди словом «{ack}».",
         f"Где ключ от «{other}»?", "на полке"),
        (f"Я назвал цвет {x} для «{topic}», размер пока неизвестен. Напиши «{ack}».",
         "Можно определить размер?", "Нет"),
    ) if ru else (
        (f"For this conversation, code {topic} is {code1} and code {other} is {code2}. Say “{ack}”.",
         f"What is the code for {other}?", str(code2)),
        (f"We first picked {x}, then switched it to {y}. Reply “{ack}”.", "Which color did we finally pick?", y),
        (f"The {topic} key is in the pocket and the {other} key is on the shelf. Acknowledge with “{ack}”.",
         f"Where is the {other} key?", "on the shelf"),
        (f"I said {topic} is {x}, but its size is unknown. Say “{ack}”.", "Can you tell what size it is?", "No"),
    )
    first, question, answer = variants[mode]
    return first, ack, question, answer


def math_task(lang, mode, rng):
    ru = lang == "ru"
    a = rng.randrange(38, 107)
    b = rng.randrange(5, 31)
    c = rng.randrange(3, 21)
    total = rng.randrange(4, 45) * 4
    minutes = rng.randrange(14, 57)
    start = rng.randrange(8 * 60, 16 * 60)
    hour, minute = divmod(start, 60)
    end_h, end_m = divmod(start + minutes, 60)
    options = (
        ((f"Было {a} листов. Убрали {b} и добавили {c}. Сколько листов получилось?" if ru else
          f"There were {a} sheets. We removed {b} and added {c}. How many sheets are there?"), a - b + c),
        ((f"Сколько составляет четверть от {total}?" if ru else f"What is a quarter of {total}?"), total // 4),
        ((f"Занятие начинается в {hour}:{minute:02d} и длится {minutes} минут. Во сколько оно закончится?" if ru else
          f"A session starts at {hour}:{minute:02d} and lasts {minutes} minutes. What time will it finish?"), f"{end_h}:{end_m:02d}"),
        ((f"У {b} коробок по {c} карточек. Сколько карточек во всех коробках?" if ru else
          f"There are {b} boxes with {c} cards in each. How many cards in all?"), b * c),
    )
    prompt, answer = options[mode]
    return prompt, str(answer)


def procedural_rows(groups):
    rng = random.Random(SEED)
    for index in range(groups):
        split = {0: "validation", 1: "test"}.get(index % 17, "train")
        for lang in LANGUAGES:
            labels = rng.sample(LABELS[lang], 3)
            key = f"short-{index}-{lang}"
            first, ack, question, answer = session_task(lang, *labels[:2], index % 4, rng)
            yield split, row(key + "-read", lang, "reading", *read_task(lang, *labels[:2], rng.randrange(2), index % 6, rng))
            yield split, row(key + "-format", lang, "short_output", *output_task(lang, labels, index % 5, rng))
            yield split, row(key + "-ack", lang, "session", first, ack)
            yield split, row(key + "-session", lang, "session", first, ack, question, answer)
            yield split, row(key + "-math", lang, "applied_math", *math_task(lang, index % 4, rng))


def prepare(previous, base, output, benchmark, groups):
    exclusion = BenchmarkExclusion(benchmark)
    generator = TextGenerator.load(base)
    prior = json.loads((previous / "manifest.json").read_bytes())
    fresh = {split: [] for split in SPLITS}
    for split, example in procedural_rows(groups):
        fresh[split].append(example)
    manifest = {"prior_sha256": digest(previous / "manifest.json"), "generator_sha256": digest(Path(__file__)),
                "seed": SEED, "groups": groups, "excluded_benchmark_sha256": digest(benchmark),
                "sampling_weights": WEIGHTS, "selection_weights": prior["selection_weights"], "splits": {}}
    seen, roots = set(), {}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix=".short-tasks-") as temporary:
        folder = Path(temporary) / "curriculum"
        folder.mkdir()
        for split in ("validation", "test", "train"):
            counts = Counter()
            path = folder / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as target:
                for example in [*source_rows(previous, split, prior), *fresh[split]]:
                    turns = tuple(normalized(turn) for turn in example["turns"][::2])
                    key = (example["language"], turns)
                    if key in seen or any(exclusion.matches(turn) for turn in turns):
                        continue
                    if example["id"].startswith("short-") and roots.setdefault((key[0], turns[0]), split) != split:
                        continue
                    if dialogue_example(example, generator) is None:
                        continue
                    seen.add(key)
                    counts[f'{example["language"]}:{example["family"]}'] += 1
                    target.write(json.dumps(example, ensure_ascii=False) + "\n")
            manifest["splits"][split] = {"rows": dict(counts), "sha256": digest(path)}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "base", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--groups", type=int, default=1800)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
