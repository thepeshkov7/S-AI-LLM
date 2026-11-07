"""Neural selection of an account memory, followed by grounded retrieval."""
import re

import torch

from generate import format_instruction


MAX_SLOTS = 6
SLOT_CONFIDENCE = 0.80


def label(value):
    value = re.sub(r"[^\w .-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())[:64]


def routing_prompt(question, language, key):
    # Values are withheld from the selector. Neither key nor question can add lines.
    question = " ".join(question.split())
    return f"Memory lookup ({language})\nKey: {label(key)}\nQuestion: {question}\nMatch:"


def chat_prompt(question, language, history=()):
    language_name = "Russian" if language == "ru" else "English"
    turns = [*history, (question, None)]
    lines = []
    for index, (user, assistant) in enumerate(turns):
        if index == 0:
            user = f"Answer in {language_name}.\n{user}"
        line = format_instruction(user)
        if assistant is not None:
            line += f" {assistant}"
        lines.append(line)
    return "\n".join(lines)


def chat_prompt_within_context(generator, question, language, history, max_new_tokens):
    prompt = chat_prompt(question, language)
    budget = generator.model.context - max_new_tokens
    for count in range(1, len(history) + 1):
        candidate = chat_prompt(question, language, history[-count:])
        if len(generator.tokenizer.encode(candidate).ids) > budget:
            break
        prompt = candidate
    return prompt


@torch.inference_mode()
def select_slot(generator, question, language, keys):
    if not keys:
        return 0, 1.0
    ids = [item.ids for item in generator.tokenizer.encode_batch(
        [routing_prompt(question, language, key) for key in keys],
    )]
    width = max(map(len, ids))
    if width > generator.model.context:
        return 0, 0.0
    device = generator.model.token.weight.device
    inputs = torch.tensor([row + [generator.eos] * (width - len(row)) for row in ids], device=device)
    ends = torch.tensor([len(row) - 1 for row in ids], device=device)
    logits, _ = generator.model(inputs)
    token = generator.tokenizer.encode("1").ids
    if len(token) != 1:
        raise ValueError("The memory checkpoint must tokenize slot digits individually")
    scores = logits[torch.arange(len(ids), device=device), ends].float().softmax(dim=-1)[:, token[0]]
    confidence, position = scores.max(dim=0)
    if len(keys) > 1 and float(confidence - scores.topk(2).values[1]) < 0.12:
        return 0, 0.0
    return int(position) + 1, float(confidence)
