"""Continue the S AI checkpoint on short Russian and English chat tasks."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import random
from time import monotonic

import torch

from finetune import instruction_tensors, masked_loss
from generate import TextGenerator, format_instruction
from instruction_data import Example
from memory_chat import chat_prompt
from train import save_checkpoint
from tune_memory import training_rows as memory_examples


TOPICS = Path(__file__).with_name("data") / "chat-topics.tsv"
NAMES = {
    "ru": "Лена Дима Оля Витя Света Глеб Юра Костя Катя Яна Саша Лида Лёша Полина Антон".split(),
    "en": "Sam Eve Luke Mia Kate Tim Noah Rose Kim Lucy Amy Max Jane Ryan Tom".split(),
}
OBJECTS = {
    "ru": ("зонт", "билет", "тетрадь", "письмо", "фото", "паспорт", "зарядку"),
    "en": ("umbrella", "ticket", "notebook", "letter", "photo", "passport", "charger"),
}


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def chat(group, language, question, answer):
    return Example(group, "prefixed_chat", language, chat_prompt(question, language), answer)


def topics(split):
    for index, line in enumerate(TOPICS.read_text(encoding="utf-8").splitlines()):
        if line.startswith("#"):
            continue
        ru, en, ru_answer, en_answer = line.split("\t")
        for language, prompts, answer in (("ru", ru, ru_answer), ("en", en, en_answer)):
            prompts = prompts.split("|")
            if split == "validation":
                prompts = prompts[-1:] if len(prompts) > 1 else []
            elif len(prompts) > 1:
                prompts = prompts[:-1]
            for prompt in prompts:
                yield chat(f"topic-{index}", language, prompt, answer)


def structured(split):
    for left in range(5, 37):
        for right in range(2, 13):
            group = f"numbers-{left}-{right}"
            if ((left * 31 + right) % 13 == 0) != (split == "validation"):
                continue
            for language in ("ru", "en"):
                if language == "ru":
                    addition = f"На полке {left} книг, поставили ещё {right}. Сколько книг стало? Ответь числом."
                    subtraction = f"Было {left + right} карандашей, забрали {right}. Сколько осталось? Ответь числом."
                    multiply = f"Взяли {left} пачек по {right} листов. Сколько всего листов? Ответь числом."
                else:
                    addition = f"There are {left} books on a shelf and we add {right}. How many are there now? Give only the number."
                    subtraction = f"We had {left + right} pencils and took away {right}. How many are left? Give only the number."
                    multiply = f"We took {left} packs with {right} sheets in each. How many sheets are there? Give only the number."
                for question, value in ((addition, left + right), (subtraction, left), (multiply, left * right)):
                    yield chat(group, language, question, str(value))

    for index in range(600):
        if (index % 13 == 0) != (split == "validation"):
            continue
        old = f"{7 + index % 10}:{(index // 4) % 4 * 15:02}"
        new = f"{9 + index % 10}:{(index // 7) % 4 * 15:02}"
        for language in ("ru", "en"):
            names = NAMES[language]
            name = names[index % len(names)]
            other = names[(index + 2) % len(names)]
            obj = OBJECTS[language][index % len(OBJECTS[language])]
            if language == "ru":
                context = f"{name} взял(а) {obj}, а {other} принёс воду. Кто взял {obj}? Ответь именем."
                first, ack, followup = f"Мы договорились созвониться в {old}.", "Хорошо.", f"Перенесли звонок на {new}. Во сколько теперь звонок?"
            else:
                context = f"{name} brought the {obj} and {other} brought water. Who brought the {obj}? Answer with the name."
                first, ack, followup = f"We agreed to have a call at {old}.", "Okay.", f"The call has moved to {new}. What time is it now?"
            group = f"context-{index}"
            yield chat(group, language, context, name)
            dialogue = chat_prompt(first, language) + f" {ack}\n" + format_instruction(followup)
            yield Example(group, "followup", language, dialogue, new)


def translations(directory, split):
    for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        group = str(row["provenance"]["ru"]["id"])
        yield chat(group, "ru", f"Переведи на русский: {row['en']}", row["ru"])
        yield chat(group, "en", f"Translate into English: {row['ru']}", row["en"])


def dataset(split, directory):
    return {
        "everyday": list(topics(split)),
        "structured": list(dict.fromkeys(structured(split))),
        "translation": list(translations(directory, split)),
        "memory": memory_examples("paraphrases" if split == "validation" else split),
    }


def train(args):
    started = monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1741)
    random_source = random.Random(1742)
    generator = TextGenerator.load(args.base)
    model = generator.model
    training = dataset("train", args.translations)
    validation = dataset("validation", args.translations)
    training_prompts = {row.prompt for rows in training.values() for row in rows}
    validation = {
        key: [row for row in dict.fromkeys(rows) if row.prompt not in training_prompts]
        for key, rows in validation.items()
    }
    if not all(validation.values()):
        raise ValueError("Every validation category needs prompts outside training")
    encoded_training = {key: instruction_tensors(rows, generator) for key, rows in training.items()}
    encoded_validation = {key: instruction_tensors(rows, generator) for key, rows in validation.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00045)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=0.00005)
    batch_size = 16
    # Memory and chat are sampled separately so neither is swamped by generated examples.
    weights = {"everyday": 3, "structured": 4, "translation": 2, "memory": 2}
    counts = {key: dict(Counter(row.language for row in rows)) for key, rows in training.items()}

    def evaluate():
        return {key: masked_loss(model, *batch, "cpu", batch_size)
                for key, batch in encoded_validation.items()}

    baseline = evaluate()
    best_score = sum(baseline.values())
    best_step = 0
    updates = Counter()
    print(json.dumps({"step": 0, "validation_loss": baseline, "training_examples": counts,
                      "validation_examples": {key: len(rows) for key, rows in validation.items()}}), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        family = random_source.choices(tuple(weights), weights=tuple(weights.values()))[0]
        inputs, targets = encoded_training[family]
        selected = random_source.choices(range(len(inputs)), k=batch_size)
        x, y = inputs[selected], targets[selected]
        length = int((y != -100).any(dim=0).nonzero()[-1]) + 1
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x[:, :length].contiguous(), y[:, :length].contiguous())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        updates[family] += 1
        if step % args.eval_every and step != args.steps:
            continue
        metrics = evaluate()
        print(json.dumps({"step": step, "validation_loss": metrics, "seconds": round(monotonic() - started, 1)}), flush=True)
        score = sum(metrics.values())
        if score < best_score:
            best_score, best_step = score, step
            save_checkpoint(args.output, model, generator.tokenizer, {
                "memory_routing": int(generator.memory_routing),
                "base_sha256": digest(args.base),
                "translations_sha256": {name: digest(args.translations / f"{name}.jsonl")
                                        for name in ("train", "validation")},
                "topics_sha256": digest(TOPICS),
                "selected_step": best_step, "planned_steps": args.steps, "updates": dict(updates),
                "training_examples": counts,
                "validation_examples": {key: len(rows) for key, rows in validation.items()},
                "validation_before": baseline, "validation_after": metrics,
            })
    if not best_step:
        raise RuntimeError("No checkpoint improved the combined validation loss")
    print(json.dumps({"selected_step": best_step, "seconds": round(monotonic() - started, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.threads) < 1:
        parser.error("Step, interval and thread counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
