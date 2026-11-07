"""Generate original bilingual instruction examples with held-out inputs and wording."""
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
import random


WORDS = {
    "ru": "кот дом лес море река сад гора мост парк хлеб сыр чай шар рыба книга ключ лампа окно снег ветер дождь луна звезда облако яблоко ручка зонт штора стол стул дверь песок поле гроза шкаф мяч".split(),
    "en": "cat house forest sea river garden mountain bridge park bread cheese tea ball fish book key lamp window snow wind rain moon star cloud apple pen umbrella curtain table chair door sand field thunder cupboard toy".split(),
}
COLORS = {
    "ru": "синий белый зелёный красный чёрный жёлтый серый розовый фиолетовый оранжевый коричневый золотой".split(),
    "en": "blue white green red black yellow gray pink purple orange brown gold".split(),
}
SHAPES = {
    "ru": "круг квадрат ромб овал куб шар конус цилиндр прямоугольник треугольник звезда кольцо".split(),
    "en": "circle square diamond oval cube sphere cone cylinder rectangle triangle star ring".split(),
}
TEMPLATES = {
    "add": {
        "ru": ("Сложи {a} и {b}. Ответь числом.", "Сколько будет {a} + {b}?", "Вычисли сумму чисел {a} и {b}."),
        "en": ("Add {a} and {b}. Give only the number.", "What is {a} + {b}?", "Calculate the sum of {a} and {b}."),
    },
    "equal": {
        "ru": (
            "Одинаковые ли слова {a} и {b}? Ответь да или нет.",
            "Слова «{a}» и «{b}» совпадают? Напиши да или нет.",
            "Сравни {a} с {b}. Это одно и то же слово? Да или нет?",
        ),
        "en": (
            "Are {a} and {b} the same word? Answer yes or no.",
            "Do the words {a} and {b} match? Say yes or no.",
            "Compare {a} with {b}. Is it the same word? Yes or no?",
        ),
    },
    "select": {
        "ru": (
            "Из списка {a}, {b} выпиши {target} слово.",
            "Назови {target} слово из пары: {a}, {b}.",
            "Список слов: {a}, {b}. Какое из них {target}?",
        ),
        "en": (
            "Return the {target} word in the list {a}, {b}.",
            "Pick the {target} word from the pair: {a}, {b}.",
            "The words are {a}, {b}. Which one is the {target}?",
        ),
    },
    "extract": {
        "ru": (
            "Цвет: {a}. Форма: {b}. Назови {target}.",
            "У предмета цвет {a} и форма {b}. Выпиши {target}.",
            "Описание предмета: {a}, {b}. Какой указан {target}?",
        ),
        "en": (
            "Color: {a}. Shape: {b}. Give the {target}.",
            "The object has color {a} and shape {b}. Return its {target}.",
            "Object description: {a}, {b}. What is its {target}?",
        ),
    },
}


@dataclass(frozen=True)
class Example:
    group: str
    task: str
    language: str
    prompt: str
    answer: str


def render(task, pair, language, variant, choice=0, reverse=False):
    left, right = reversed(pair) if reverse else pair
    group = f"{task}:{pair[0]}:{pair[1]}"
    target = ""
    if task == "add":
        a, b, answer = str(left), str(right), str(left + right)
    elif task == "extract":
        a, b = COLORS[language][left], SHAPES[language][right]
        targets = {
            "ru": ("цвет", "форму") if variant != 2 else ("цвет", "тип формы"),
            "en": ("color", "shape"),
        }
        target = targets[language][choice]
        answer = (a, b)[choice]
    else:
        a, b = WORDS[language][left], WORDS[language][right]
        if task == "equal":
            answers = {"ru": ("нет", "да"), "en": ("no", "yes")}
            answer = answers[language][left == right]
        else:
            positions = {
                "ru": ("первое", "второе"), "en": ("first", "second"),
            }
            target = positions[language][choice]
            answer = (a, b)[choice]
    prompt = TEMPLATES[task][language][variant].format(a=a, b=b, target=target)
    return Example(group, task, language, prompt, answer)


def build_splits(seed=761):
    """Assign the semantic input before rendering templates or either language."""
    rng = random.Random(seed)
    words = range(len(WORDS["en"]))
    pairs = list(combinations_with_replacement(words, 2))
    same = [pair for pair in pairs if pair[0] == pair[1]]
    distinct = [pair for pair in pairs if pair[0] != pair[1]]
    add = list(combinations_with_replacement(range(32), 2))
    # Keep the arithmetic diagnostic outside the generated splits.
    add.remove((2, 2))
    sources = [
        ("add", add, 28),
        ("equal", same, 6),
        ("equal", distinct, 6),
        ("select", distinct.copy(), 24),
        ("extract", list(product(range(len(COLORS["en"])), range(len(SHAPES["en"])))), 18),
    ]
    splits = {name: [] for name in ("train", "validation", "test", "new_wording")}
    for task, groups, count in sources:
        rng.shuffle(groups)
        for index, pair in enumerate(groups):
            split = ("validation", "test", "new_wording")[index // count] if index < 3 * count else "train"
            for language in ("ru", "en"):
                if split == "train":
                    for variant in (0, 1):
                        choices = (0, 1) if task in {"extract", "select"} else (0,)
                        for choice in choices:
                            splits[split].append(render(task, pair, language, variant, choice))
                            if task != "extract" and pair[0] != pair[1]:
                                splits[split].append(render(task, pair, language, variant, choice, reverse=True))
                else:
                    variant = 2 if split == "new_wording" else rng.randrange(2)
                    choice = rng.randrange(2)
                    reverse = task != "extract" and rng.choice((True, False))
                    splits[split].append(render(task, pair, language, variant, choice, reverse))
    return splits
