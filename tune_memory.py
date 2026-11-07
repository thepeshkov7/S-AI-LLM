"""Fine-tune the local S AI pilot for bilingual chat and neural memory selection."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from time import monotonic

import torch

from finetune import instruction_tensors, masked_loss
from generate import TextGenerator
from instruction_data import Example
from memory_chat import SLOT_CONFIDENCE, chat_prompt, routing_prompt, select_slot
from train import save_checkpoint


FIELDS = {
    "name": (
        ("name", "profile.name", "first name", "имя"),
        ("Как меня зовут?", "Напомни моё имя.", "Какое у меня имя?", "Как ты меня называешь?", "Что ты запомнил о том, как ко мне обращаться?"),
        ("What is my name?", "Remind me of my name.", "What name do I go by?", "How should you address me?", "What did you remember about what to call me?"),
    ),
    "city": (
        ("city", "profile.city", "home city", "город"),
        ("В каком городе я живу?", "Напомни мой город.", "Где я живу?", "Какой город я называю домом?", "Что записано о городе, где я живу?"),
        ("Which city do I live in?", "Remind me of my city.", "Where do I live?", "What city do I call home?", "Which city is recorded as where I live?"),
    ),
    "color": (
        ("favorite color", "preferences.favorite_color", "preferred colour", "любимый цвет"),
        ("Какой у меня любимый цвет?", "Какой цвет я люблю?", "Напомни мой любимый цвет.", "Какой оттенок мне нравится больше всего?", "Что я говорил про мой любимый цвет?"),
        ("What is my favorite color?", "Which color do I like?", "Remind me of my favourite colour.", "Which shade do I prefer most?", "What did I say about the color I like best?"),
    ),
    "food": (
        ("favorite food", "preferences.favorite_food", "favorite dish", "любимая еда"),
        ("Какая у меня любимая еда?", "Какое блюдо я люблю?", "Напомни моё любимое блюдо.", "Что я предпочитаю из еды?", "Какое блюдо я бы выбрал первым?"),
        ("What is my favorite food?", "Which dish do I like?", "Remind me of my favorite dish.", "What food do I prefer?", "Which meal would I pick first?"),
    ),
    "pet": (
        ("pet name", "profile.pet_name", "name of pet", "кличка питомца"),
        ("Как зовут моего питомца?", "Напомни кличку питомца.", "Какое имя у моего питомца?", "Под какой кличкой записан мой питомец?", "Что я говорил про имя моего питомца?"),
        ("What is my pet's name?", "Remind me what my pet is called.", "What name does my pet have?", "Which name is saved for my pet?", "What did I tell you to call my pet?"),
    ),
    "language": (
        ("preferred language", "preferences.language", "reply language", "язык общения"),
        ("На каком языке я предпочитаю общаться?", "Напомни мой язык общения.", "Какой язык для ответа мне удобнее?", "Какой язык мне нравится для разговора?", "На каком языке я обычно прошу отвечать?"),
        ("Which language do I prefer to chat in?", "Remind me of my preferred language.", "What language should replies use?", "What language do I like to talk in?", "Which language did I usually ask for?"),
    ),
    "project": (
        ("current project", "work.project", "project name", "текущий проект"),
        ("Как называется мой проект?", "Над каким проектом я работаю?", "Напомни мой текущий проект.", "Какой проект я сейчас делаю?", "Как я назвал то, над чем работаю?"),
        ("What is my project called?", "Which project am I working on?", "Remind me of my current project.", "What project am I building right now?", "What did I name the thing I am working on?"),
    ),
    "sport": (
        ("favorite sport", "preferences.sport", "preferred sport", "любимый спорт"),
        ("Какой спорт я люблю?", "Напомни мой любимый вид спорта.", "Какой мой любимый спорт?", "Каким видом спорта я больше всего интересуюсь?", "Что я предпочитаю из спорта?"),
        ("What sport do I like?", "Remind me of my favorite sport.", "Which sport is my favorite?", "What kind of sport interests me most?", "Which sport did I say I prefer?"),
    ),
}
NO_MEMORY = (
    ("Привет!", "Сколько будет два плюс три?", "Что такое облака?", "Пожелай мне удачи.", "Скажи привет моему другу.", "Почему небо голубое?"),
    ("Hello!", "What is two plus three?", "What are clouds?", "Wish me luck.", "Say hello to my friend.", "Why is the sky blue?"),
)
CHAT = (
    ("Привет!", "Привет! Чем могу помочь?", "Hello!", "Hello! How can I help?"),
    ("Доброе утро!", "Доброе утро! Как могу помочь?", "Good morning!", "Good morning! How can I help?"),
    ("Спасибо за помощь.", "Пожалуйста! Рад помочь.", "Thank you for helping.", "You're welcome! Happy to help."),
    ("На каком языке ты можешь ответить?", "Я могу отвечать на русском и английском.", "What languages can you reply in?", "I can reply in Russian and English."),
    ("Пожелай мне удачи.", "Удачи! Надеюсь, всё получится.", "Wish me luck.", "Good luck! I hope it goes well."),
    ("Сколько будет два плюс три?", "Два плюс три — пять.", "What is two plus three?", "Two plus three is five."),
    ("Кто ты?", "Я S AI, твой помощник.", "Who are you?", "I'm S AI, your assistant."),
    ("Коротко попрощайся.", "До встречи!", "Say goodbye briefly.", "See you soon!"),
)


def examples(split):
    seeds = {"train": 9031, "validation": 9032, "test": 9033, "paraphrases": 9034}
    rng = random.Random(seeds[split])
    fields = tuple(FIELDS)
    count = 2400 if split == "train" else 80
    templates = range(3) if split != "paraphrases" else range(3, 5)
    for index in range(count):
        target = fields[index % len(fields)]
        available = rng.sample(fields, rng.randint(2, 6))
        general = index % 5 == 0
        want_missing = index % 5 == 1
        if not general:
            if want_missing:
                available = [field for field in available if field != target]
            elif target not in available:
                available[rng.randrange(len(available))] = target
        aliases = [rng.choice(FIELDS[field][0]) for field in available]
        slot = available.index(target) + 1 if not general and target in available else 0
        for language_index, language in enumerate(("ru", "en")):
            if general:
                variants = NO_MEMORY[language_index]
                question = variants[rng.randrange(4) if split != "paraphrases" else rng.randrange(4, 6)]
            else:
                question = FIELDS[target][language_index + 1][rng.choice(templates)]
            yield question, language, aliases, slot


def training_rows(split):
    rows = []
    rng = random.Random(941)
    for i, (question, language, keys, slot) in enumerate(examples(split)):
        negatives = [index for index in range(1, len(keys) + 1) if index != slot]
        selected = [slot, rng.choice(negatives)] if slot else rng.sample(negatives, min(2, len(negatives)))
        rows.extend(
            Example(str(i), "memory_route", language, routing_prompt(question, language, keys[index - 1]),
                    "1" if index == slot else "0")
            for index in selected
        )
    if split == "train":
        chat = [
            Example(str(i), "prefixed_chat", language, chat_prompt(question, language), answer)
            for i, item in enumerate(CHAT)
            for language, question, answer in (("ru", item[0], item[1]), ("en", item[2], item[3]))
        ]
        rows.extend(chat * 42)
    return rows


def measure(generator, split):
    scores = {language: {"correct": 0, "total": 0, "answered": 0} for language in ("ru", "en")}
    scores["abstention"] = {"false_recall": 0, "total": 0}
    for question, language, keys, expected in examples(split):
        slot, confidence = select_slot(generator, question, language, keys)
        selected = slot if confidence >= SLOT_CONFIDENCE else 0
        scores[language]["correct"] += int(selected == expected)
        scores[language]["answered"] += int(selected > 0)
        scores[language]["total"] += 1
        if expected == 0:
            scores["abstention"]["total"] += 1
            scores["abstention"]["false_recall"] += int(selected > 0)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(6)
    torch.manual_seed(512)
    started = monotonic()
    generator = TextGenerator.load(args.base)
    model = generator.model
    train = instruction_tensors(training_rows("train"), generator)
    validation = instruction_tensors(training_rows("validation"), generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00055)
    seed = torch.Generator().manual_seed(513)
    initial = masked_loss(model, *validation, "cpu", 16)
    print(json.dumps({"step": 0, "validation_loss": initial, "train_examples": len(train[0])}), flush=True)
    best_loss, best_step, weights = initial, 0, None
    model.train()
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(train[0]), (16,), generator=seed)
        x, y = (part[ids] for part in train)
        length = int((y != -100).any(dim=0).nonzero()[-1]) + 1
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x[:, :length].contiguous(), y[:, :length].contiguous())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            value = masked_loss(model, *validation, "cpu", 16)
            print(json.dumps({"step": step, "validation_loss": value}), flush=True)
            if value < best_loss:
                best_loss, best_step = value, step
                weights = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            model.train()
    if weights is None:
        raise RuntimeError("Memory routing validation loss did not improve")
    model.load_state_dict(weights)
    model.eval()
    metrics = {
        "memory_routing": 1, "base_sha256": sha256(args.base.read_bytes()).hexdigest(),
        "steps": args.steps, "selected_step": best_step,
        "training_examples": len(train[0]), "validation_before": initial, "validation_after": best_loss,
        "threshold": SLOT_CONFIDENCE, "training_seconds": round(monotonic() - started, 1),
        "test": measure(generator, "test"), "paraphrases": measure(generator, "paraphrases"),
    }
    save_checkpoint(args.output, model, generator.tokenizer, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
