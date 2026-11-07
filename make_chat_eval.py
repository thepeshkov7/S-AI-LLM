"""Freeze an original Russian/English chat benchmark independently of training data."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple


class Topic(NamedTuple):
    ru_term: str
    en_term: str
    ru_criteria: str
    en_criteria: str


TOPICS = [
    Topic("туман", "fog", "Мелкие капли воды у поверхности снижают видимость.", "Small water droplets near the surface reduce visibility."),
    Topic("эхо", "echo", "Звук отражается от препятствия и возвращается.", "Sound reflects off a surface and returns."),
    Topic("радуга", "a rainbow", "Капли воды преломляют, отражают и разделяют свет.", "Water droplets refract, reflect and disperse sunlight."),
    Topic("приливы", "tides", "Разница гравитационного притяжения Луны и Солнца действует на океан.", "Differences in gravitational pull from the Moon and Sun affect the oceans."),
    Topic("ржавчина", "rust", "Железо окисляется в присутствии кислорода и влаги.", "Iron oxidizes in the presence of oxygen and moisture."),
    Topic("почему мыло смывает жир", "how soap removes grease", "Мыло помогает смешать жир с водой и смыть его.", "Soap helps disperse grease in water so it can be rinsed away."),
    Topic("как работают термосы", "how a thermos works", "Изоляция замедляет теплообмен с окружающей средой.", "Insulation slows heat transfer to and from the surroundings."),
    Topic("фотосинтез", "photosynthesis", "Растение использует свет, воду и углекислый газ, чтобы получить сахара.", "Plants use light, water and carbon dioxide to make sugars."),
    Topic("почему небо голубое", "why the sky is blue", "Атмосфера сильнее рассеивает короткие волны видимого света.", "The atmosphere scatters shorter visible wavelengths more strongly."),
    Topic("инерция", "inertia", "Тело сохраняет состояние покоя или равномерного движения без результирующей силы.", "An object stays at rest or in uniform motion unless acted on by a net force."),
    Topic("компас", "a compass", "Магнитная стрелка ориентируется вдоль магнитного поля Земли.", "A magnetized needle aligns with Earth's magnetic field."),
    Topic("испарение", "evaporation", "Часть молекул покидает поверхность жидкости и переходит в газ.", "Some molecules leave a liquid's surface and enter the gas phase."),
    Topic("точка росы", "the dew point", "Температура, при которой воздух насыщается водяным паром при охлаждении.", "The temperature at which cooling air reaches saturation with water vapor."),
    Topic("зачем велосипеду тормоза", "what bicycle brakes do", "Трение замедляет вращение колеса, превращая энергию в тепло.", "Friction slows the wheel and turns motion energy into heat."),
    Topic("смена времён года", "the seasons", "Наклон земной оси меняет угол солнечного света и длину дня.", "Earth's axial tilt changes sunlight angle and day length."),
    Topic("звук", "sound", "Колебания распространяются через среду в виде волн.", "Vibrations travel through a medium as waves."),
    Topic("почему лёд плавает", "why ice floats", "Лёд менее плотный, чем жидкая вода.", "Ice is less dense than liquid water."),
    Topic("как батарейка даёт ток", "how a battery supplies current", "Химические реакции создают разность потенциалов и поддерживают ток в замкнутой цепи.", "Chemical reactions create a potential difference that drives current in a closed circuit."),
    Topic("почему линза увеличивает", "how a magnifying lens enlarges an image", "Преломление лучей даёт увеличенное виртуальное изображение близкого предмета.", "Refraction produces an enlarged virtual image of a nearby object."),
    Topic("почему Луна видна ночью", "why the Moon is visible at night", "Луна отражает солнечный свет, когда находится над горизонтом.", "The Moon reflects sunlight while it is above the horizon."),
]

TRANSLATIONS = [
    ("Мы пропустили автобус, но успели на поезд.", "We missed the bus but caught the train."),
    ("Поставь чашку рядом с чайником.", "Put the cup next to the kettle."),
    ("Собрание перенесли на пятницу.", "The meeting was moved to Friday."),
    ("К вечеру дорога высохла.", "By evening the road had dried."),
    ("Если дождь усилится, останемся дома.", "If the rain gets heavier, we'll stay home."),
    ("Я взял фонарик на всякий случай.", "I took a flashlight just in case."),
    ("Эта инструкция занимает всего страницу.", "These instructions are only one page long."),
    ("Мост закрыт до окончания ремонта.", "The bridge is closed until repairs are finished."),
    ("Она выключила музыку, чтобы услышать звонок.", "She turned off the music so she could hear the bell."),
    ("У нас осталось две попытки.", "We have two attempts left."),
    ("Скажи мне, когда закончишь.", "Tell me when you finish."),
    ("Ключи лежат под синей книгой.", "The keys are under the blue book."),
    ("Окно было открыто, хотя на улице холодно.", "The window was open even though it was cold outside."),
    ("Письмо отправили вчера утром.", "The letter was sent yesterday morning."),
    ("Он пообещал вернуть велосипед до ужина.", "He promised to return the bicycle before dinner."),
    ("Здесь можно взять воду бесплатно.", "You can get water here for free."),
    ("Я пересел ближе к двери, чтобы быстрее выйти.", "I moved closer to the door so I could get out faster."),
    ("Отнеси коробку в соседнюю комнату.", "Take the box to the next room."),
    ("Она не ожидала, что магазин уже закрылся.", "She didn't expect the store to have closed already."),
    ("Проверь расписание, прежде чем покупать билет.", "Check the schedule before buying a ticket."),
]

SETTINGS = [
    ("музей", "museum"), ("мастерская", "workshop"), ("библиотека", "library"),
    ("выставка", "exhibition"), ("кружок", "club"), ("ярмарка", "fair"),
    ("репетиция", "rehearsal"), ("экскурсия", "tour"), ("тренировка", "training session"),
    ("лекция", "lecture"), ("спектакль", "performance"), ("показ фильма", "film screening"),
    ("встреча", "meeting"), ("фестиваль", "festival"), ("прогулка", "walk"),
    ("концерт", "concert"), ("занятие", "class"), ("обсуждение", "discussion"),
    ("конкурс", "competition"), ("открытие", "opening"),
]
NAMES = ["Ася", "Лев", "Нина", "Олег", "Ира", "Павел", "Анна", "Лада", "Зоя", "Ника",
         "Тимур", "Вера", "Соня", "Марат", "Кира", "Роман", "Лина", "Илья", "Лида", "Саша"]
NAMES_EN = ["Maya", "Ben", "Nina", "Owen", "Iris", "Paul", "Anna", "Lee", "Zoe", "Nora",
            "Tom", "Vera", "Sonia", "Mark", "Kira", "Roman", "Lina", "Eli", "Lydia", "Alex"]


def cases():
    for lang in ("ru", "en"):
        ru = lang == "ru"
        for i, (setting_ru, setting_en) in enumerate(SETTINGS):
            setting = setting_ru if ru else setting_en
            names = NAMES if ru else NAMES_EN
            name, other = names[i], names[(i + 7) % len(names)]
            hour = 9 + i % 7
            delay = i % 4 + 1
            old, new = f"{hour}:00", f"{hour + delay}:00"
            topic = TOPICS[i]
            problem = i % 4
            a, b = 14 + i, 3 + i % 8
            title = f"Запись {i + 31}" if ru else f"Booking {i + 31}"
            scenario = (
                f"Событие «{setting}» перенесли с {old} на {new} из-за ремонта. {name} предупредил(а) гостей."
                if ru else f"The {setting} moved from {old} to {new} because of repairs. {name} notified the guests."
            )
            arithmetic = (
                [f"В коробке было {a} книг, {b} забрали. Сколько осталось?",
                 f"Купили {a} билетов по {b} рублей. Сколько заплатили?",
                 f"В группе {a} взрослых и {b} детей. Сколько человек в группе?",
                 f"Есть {a * b} листов. Их поровну делят между {b} людьми. Сколько получит каждый?"]
                if ru else
                [f"A box held {a} books and {b} were removed. How many remain?",
                 f"Someone bought {a} tickets at {b} dollars each. How much did they pay?",
                 f"A group has {a} adults and {b} children. How many people are in it?",
                 f"{a * b} sheets are shared equally among {b} people. How many does each get?"]
            )[problem]
            target = (a - b, a * b, a + b, a)[problem]
            followup = (
                f"User: Моя запись «{title}» на {old}.\nAssistant: Понял.\nUser: Её перенесли на {new}. "
                f"Во сколько мне прийти?\nAssistant:"
                if ru else f"User: My booking '{title}' is at {old}.\nAssistant: Understood.\n"
                f"User: It was moved to {new}. What time should I arrive?\nAssistant:"
            )
            rows = [
                ("explain", (f"Объясни простыми словами тему «{topic.ru_term}» в одном-двух предложениях."
                             if ru else f"Explain {topic.en_term} in plain language in one or two sentences."),
                 topic.ru_criteria if ru else topic.en_criteria, "human"),
                ("translate", (f"Переведи на русский, сохранив смысл. {TRANSLATIONS[i][1]}" if ru
                               else f"Translate into English while preserving the meaning. {TRANSLATIONS[i][0]}"),
                 TRANSLATIONS[i][0 if ru else 1], "human"),
                ("arithmetic", arithmetic + (" Ответь одним числом." if ru else " Give only the number."), str(target), "exact"),
                ("extract", scenario + (" Кто предупредил гостей? Напиши только имя." if ru
                                       else " Who notified the guests? Reply with only the name."), name, "exact"),
                ("summarize", scenario + (" Кратко перескажи изменение времени и причину." if ru
                                         else " Briefly summarize the time change and its reason."),
                 (f"С {old} на {new} из-за ремонта." if ru else f"From {old} to {new} because of repairs."), "human"),
                ("format", (f"Ответь только JSON с ключами title и time. Название {title}, время {new}."
                            if ru else f"Return only JSON with keys title and time. The title is {title}, and the time is {new}."),
                 {"title": title, "time": new}, "json"),
                ("followup", followup, new, "time"),
                ("reason", (f"{name} пришёл раньше, чем {other}. Кто пришёл позже? Ответь только именем." if ru
                            else f"{name} arrived before {other}. Who arrived later? Reply with only the name."), other, "exact"),
                ("writing", (f"Напиши вежливое короткое сообщение: событие «{setting}» нужно перенести с {old} на {new}."
                             if ru else f"Write a short polite message asking to reschedule the {setting} from {old} to {new}."),
                 (f"Вежливая просьба о переносе события «{setting}» с {old} на {new}."
                  if ru else f"A polite request to reschedule the {setting} from {old} to {new}."), "human"),
                ("missing_info", (f"Мне нужно узнать цену события «{setting}». В объявлении указано только время {new}. Можно назвать цену?"
                                  if ru else f"I need the price of the {setting}. The announcement only lists {new} as the time. Can you give the price?"),
                 ("Цена не указана. Не придумывать сумму." if ru else "The price is unspecified. Do not invent a price."), "human"),
            ]
            for task, prompt, reference, score in rows:
                yield {"id": f"{lang}-{task}-{i + 1:02}", "language": lang, "task": task,
                       "prompt": prompt, "reference": reference, "score": score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(cases())
    assert len(rows) == len({row["id"] for row in rows}) == 400
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise RuntimeError("Frozen benchmark differs from the generated data")
    args.output.write_bytes(payload)
    print(f"{len(rows)} cases, SHA-256 {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
