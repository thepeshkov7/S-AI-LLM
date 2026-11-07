"""Teach the pilot checkpoint to answer using translations, passages and short dialogues."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic

import torch

from finetune import instruction_tensors, masked_loss
from generate import TextGenerator
from instruction_data import Example
from train import TrainingWindows, encode_documents, read_split, save_checkpoint


DIALOGUES = [
    ("Напиши приглашение на чай.", "Пусть звучит тепло и будет на завтра.",
     "Приходи завтра на чай, буду очень рад тебя увидеть!",
     "Write an invitation for tea.", "Make it warm and invite them for tomorrow.",
     "Come over for tea tomorrow. I'd love to see you!"),
    ("Помоги ответить, что заказ готов.", "Добавь, что его можно забрать после обеда.",
     "Ваш заказ готов. Его можно забрать сегодня после обеда.",
     "Help me say the order is ready.", "Mention it can be picked up after lunch.",
     "Your order is ready and can be picked up after lunch."),
    ("Придумай подпись к снимку гор.", "Сделай её короткой и спокойной.",
     "Тишина над вершинами.",
     "Suggest a caption for a mountain photo.", "Keep it short and peaceful.",
     "Quiet above the peaks."),
    ("Что можно взять на пикник?", "Нужна одна вещь, чтобы сидеть на траве.",
     "Возьми плед, который можно расстелить на траве.",
     "What can I bring to a picnic?", "Name one thing to sit on the grass.",
     "Bring a blanket you can spread on the grass."),
    ("Напиши благодарность за помощь с переездом.", "Обращайся к другу на ты.",
     "Спасибо, что помог мне с переездом. Мне было гораздо легче с тобой.",
     "Thank someone for helping me move.", "Address a close friend.",
     "Thanks for helping me move. You made it so much easier!"),
    ("Хочу замочить фасоль.", "Зачем её заранее держат в воде?",
     "Фасоль впитывает воду и обычно быстрее варится.",
     "I want to soak some dried beans.", "Why soak them in advance?",
     "The beans absorb water, which usually helps them cook faster."),
    ("Заголовок письма должен быть про новый адрес.", "Уточни, что адрес доставки изменился.",
     "Изменился адрес доставки",
     "The email subject should be about a new address.", "Specify that the delivery address changed.",
     "Delivery address updated"),
    ("Надо описать прозрачную папку.", "Пусть это будет одна понятная фраза.",
     "Прозрачная папка защищает бумаги и позволяет видеть, что внутри.",
     "Describe a transparent folder.", "Use one clear sentence.",
     "A transparent folder protects papers while letting you see what's inside."),
    ("Помоги выбрать обувь.", "Я пойду по мокрой траве.",
     "Подойдут водостойкие ботинки с подошвой, которая хорошо держится на мокрой поверхности.",
     "Help me choose shoes.", "I'll be walking through wet grass.",
     "Choose water-resistant boots with soles that grip wet ground."),
    ("Хочу сделать рабочий стол аккуратнее.", "С чего начать, если времени мало?",
     "Убери с поверхности всё, чем сегодня не пользуешься, и оставь только нужное.",
     "I want to tidy my desk.", "Where should I start if I have little time?",
     "Remove things you won't use today and leave only what you need."),
    ("Напиши напоминание полить растения.", "Добавь, что это нужно сделать в воскресенье.",
     "В воскресенье полей растения.",
     "Write a reminder to water the plants.", "Make it for Sunday.",
     "Water the plants on Sunday."),
    ("Объясни, что такое черновик.", "Представь, что говоришь с ребёнком.",
     "Черновик — первая версия работы, которую можно исправить и улучшить.",
     "Explain what a draft is.", "Imagine you're talking to a child.",
     "A draft is a first version of your work that you can change and improve."),
]


def load_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def instruction_examples(wiki_dir, translations_dir, split):
    examples = []
    for row in load_rows(wiki_dir / f"{split}.jsonl"):
        opening = " ".join(row["text"].splitlines()[:4])
        passage = next((sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", opening)
                        if 55 <= len(sentence.strip()) <= 155), None)
        if passage is None:
            continue
        ru = row["language"] == "ru"
        prompt = (
            f"Я цитирую статью: «{passage}» Передай эту цитату без изменений."
            if ru else f"I am quoting an article: “{passage}” Repeat the quotation exactly."
        )
        examples.append(Example(row["id"], "passage", row["language"], prompt, passage))
    for pair in load_rows(translations_dir / f"{split}.jsonl"):
        group = str(pair["provenance"]["ru"]["id"])
        examples.extend((
            Example(group, "translation", "ru", f"Переведи на русский. {pair['en']}", pair["ru"]),
            Example(group, "translation", "en", f"Translate to English. {pair['ru']}", pair["en"]),
        ))
    indexed = list(enumerate(DIALOGUES))
    for index, dialogue in (indexed[:9] if split == "train" else indexed[9:]):
        for language, start in (("ru", 0), ("en", 3)):
            first, followup, answer = dialogue[start:start + 3]
            acknowledgement = "Понял." if language == "ru" else "Understood."
            prompt = f"User: {first}\nAssistant: {acknowledgement}\nUser: {followup}\nAssistant:"
            examples.append(Example(f"dialogue-{index}", "followup", language, prompt, answer))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if min(args.steps, args.threads) < 1:
        parser.error("Steps and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(321)
    rng = torch.Generator().manual_seed(322)
    started = monotonic()
    generator = TextGenerator.load(args.base)
    model = generator.model
    train_rows = instruction_examples(args.wiki, args.translations, "train")
    validation_rows = instruction_examples(args.wiki, args.translations, "validation")
    train = instruction_tensors(train_rows, generator)
    validation = instruction_tensors(validation_rows, generator)
    counts = {split: dict(Counter(row.task for row in rows))
              for split, rows in (("train", train_rows), ("validation", validation_rows))}
    replay_docs = read_split(args.wiki / "train.jsonl", set(), set())
    replay = TrainingWindows(encode_documents(replay_docs, generator.tokenizer), model.context, 323, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035)
    batch_size = 8
    initial = masked_loss(model, *validation, "cpu", batch_size)
    print(json.dumps({"step": 0, "validation_loss": initial, "data": counts}), flush=True)
    best_loss, best_step, weights = initial, 0, None
    model.train()
    for step in range(1, args.steps + 1):
        if step % 6 == 0:
            x, y = replay.batch(batch_size, "cpu")
        else:
            indices = torch.randint(len(train_rows), (batch_size,), generator=rng)
            x, y = (part[indices] for part in train)
            length = int((y != -100).any(dim=0).nonzero()[-1]) + 1
            x, y = x[:, :length].contiguous(), y[:, :length].contiguous()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 60 == 0 or step == args.steps:
            measured = masked_loss(model, *validation, "cpu", batch_size)
            print(json.dumps({"step": step, "validation_loss": measured}), flush=True)
            if measured < best_loss:
                best_loss, best_step = measured, step
                weights = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            model.train()
    if weights is None:
        raise RuntimeError("No checkpoint improved instruction validation loss")
    model.load_state_dict(weights)
    metrics = {"base_sha256": sha256(args.base.read_bytes()).hexdigest(), "steps": args.steps,
               "selected_step": best_step, "seed": 321, "data": counts,
               "initial_validation_loss": initial, "best_validation_loss": best_loss,
               "seconds": round(monotonic() - started, 1)}
    save_checkpoint(args.output, model.eval(), generator.tokenizer, metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
