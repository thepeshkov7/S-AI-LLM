"""Load a local checkpoint and sample its autoregressive text continuation."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys
from zipfile import is_zipfile

import torch
from tokenizers import Tokenizer

from model import CopyLLM, TinyLLM, checkpoint_architecture


LEGACY_ATTENTION_KEYS = {
    "attn.in_proj_weight": "qkv.weight",
    "attn.in_proj_bias": "qkv.bias",
    "attn.out_proj.weight": "proj.weight",
    "attn.out_proj.bias": "proj.bias",
}


def checkpoint_key(key):
    block, attention, parameter = key.partition(".attn.")
    if attention:
        return block + "." + LEGACY_ATTENTION_KEYS["attn." + parameter]
    return key


def format_instruction(instruction):
    return f"User: {instruction.strip()}\nAssistant:"


def format_dialogue(turns):
    history = [format_instruction(user) + " " + answer
               for user, answer in zip(turns[:-1:2], turns[1::2], strict=True)]
    return "\n".join([*history, format_instruction(turns[-1])])


def nucleus_probabilities(probabilities, top_p):
    # Keep the first token that reaches the threshold, even if it is above top_p alone.
    remove = probabilities.cumsum(dim=0)[:-1] >= top_p
    return torch.cat((probabilities[:1], probabilities[1:].masked_fill(remove, 0)))


class TextGenerator:
    def __init__(self, model, tokenizer=None, memory_routing=False, chat_model=None, *, vocabulary_only=False):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_routing = memory_routing
        self.chat_model = chat_model or ("S AI 0.2 Memory" if memory_routing else None)
        self.vocabulary_only = vocabulary_only
        self.eos = tokenizer.token_to_id("<|end_of_document|>") if tokenizer else None

    @classmethod
    def load(cls, path, device="cpu"):
        mmap = sys.platform != "win32" and isinstance(path, (str, Path)) and is_zipfile(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
        state = checkpoint["model"]
        version = checkpoint.get("format_version")
        architecture = checkpoint_architecture(checkpoint)
        unversioned_copy = version is None and architecture == CopyLLM.architecture and "config" in checkpoint
        if version == 1 or unversioned_copy:
            architectures = {None: TinyLLM, CopyLLM.architecture: CopyLLM}
            if architecture not in architectures:
                raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
            with torch.device("meta"):
                model = architectures[architecture](**checkpoint["config"])
            tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
            if tokenizer.get_vocab_size() != model.config.vocab_size or tokenizer.token_to_id("<|end_of_document|>") is None:
                raise ValueError("Checkpoint tokenizer does not match the model")
        elif version is None:
            # Convert the original MultiheadAttention names to the same QKV/projection layout.
            if "context" not in checkpoint:
                raise ValueError("Unrecognized legacy checkpoint format")
            with torch.device("meta"):
                model = TinyLLM(context=checkpoint["context"])
            tokenizer = None
            state = {checkpoint_key(key): value for key, value in state.items()}
        else:
            raise ValueError(f"Unsupported checkpoint version: {version}")
        model.load_state_dict(state, strict=True, assign=True)
        metrics = checkpoint.get("metrics", {})
        chat_model = None
        if tokenizer is not None and metrics.get("bilingual_chat") is True:
            chat_model = metrics.get("chat_model")
            if not isinstance(chat_model, str) or not chat_model.strip():
                chat_model = "S AI 0.5 Grounded" if isinstance(model, CopyLLM) else "S AI 0.4 Bilingual"
        profile = checkpoint.get("settings", {}).get("profile", {})
        vocabulary_only = (unversioned_copy and model.config.vocab_size == 65536) or (
            architecture == CopyLLM.architecture and profile.get("recovery_step") == 10000
        )
        return cls(model.to(device=device, dtype=torch.float32).eval(), tokenizer,
                   metrics.get("memory_routing") == 1, chat_model, vocabulary_only=vocabulary_only)

    @torch.inference_mode()
    def complete(self, prompt, max_new_tokens=80, temperature=0.85, top_k=40, seed=None, max_bytes=None,
                 *, top_p=1.0, repetition_penalty=None, frequency_penalty=0.0, no_repeat_ngram_size=None,
                 preserve_output=False):
        if not prompt or max_new_tokens < 1 or top_k < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid generation parameters")
        if not 0 < top_p <= 1 or not math.isfinite(frequency_penalty) or frequency_penalty < 0:
            raise ValueError("Invalid nucleus or frequency penalty")
        if repetition_penalty is None:
            repetition_penalty = 1.2 if self.vocabulary_only else 1.0
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = 3 if self.vocabulary_only else 0
        if not math.isfinite(repetition_penalty) or repetition_penalty < 1 or no_repeat_ngram_size < 0:
            raise ValueError("Invalid repetition controls")
        if self.tokenizer:
            ids = (self.tokenizer.encode(prompt, add_special_tokens=False) if self.vocabulary_only
                   else self.tokenizer.encode(prompt)).ids
        else:
            ids = list(prompt.encode("utf-8"))
        if not ids:
            raise ValueError("Prompt must encode to at least one token")
        device = self.model.token.weight.device
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        output = []
        counts = Counter()
        score_tokens = self.model.vocabulary_scores if self.vocabulary_only else self.model.next_scores
        ngrams = {}
        prefix_length = max(0, no_repeat_ngram_size - 1)
        for _ in range(max_new_tokens):
            recent = torch.tensor([ids[-self.model.context:]], device=device, dtype=torch.long)
            scores = score_tokens(recent)[0].float()
            if repetition_penalty > 1 and output:
                scores[list(set(output[-128:]))] -= math.log(repetition_penalty)
            if counts:
                tokens, frequencies = zip(*counts.items())
                scores[list(tokens)] -= scores.new_tensor(frequencies) * frequency_penalty
            prefix = tuple(output[-prefix_length:]) if prefix_length else ()
            blocked = ngrams.get(prefix)
            if blocked:
                if len(blocked) == scores.size(-1):
                    break
                scores[list(blocked)] = -torch.inf
            values, indices = scores.topk(min(top_k, scores.size(-1)))
            values = (values - values[0]) / temperature
            probabilities = values.softmax(dim=0)
            if top_p < 1:
                probabilities = nucleus_probabilities(probabilities, top_p)
            selected = torch.multinomial(probabilities, 1, generator=generator)
            token = indices[selected].item()
            if token == self.eos:
                break
            if no_repeat_ngram_size and len(prefix) == prefix_length:
                ngrams.setdefault(prefix, set()).add(token)
            ids.append(token)
            output.append(token)
            if frequency_penalty:
                counts[token] += 1
        if self.tokenizer:
            completion = self.tokenizer.decode(output, skip_special_tokens=not preserve_output)
            if not preserve_output:
                completion = completion.rstrip("\ufffd")
        else:
            completion = bytes(output).decode("utf-8", errors="replace")
        if max_bytes is not None:
            completion = completion.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        return completion, len(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("s_ai_0_1.pt"))
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--instruction")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--max-new-tokens", type=int)
    length.add_argument("--max-new-bytes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    limit = args.max_new_bytes if args.max_new_bytes is not None else args.max_new_tokens
    if args.threads < 1 or (limit is not None and limit < 1):
        parser.error("Threads and output limits must be positive")
    torch.set_num_threads(args.threads)
    generator = TextGenerator.load(args.checkpoint, args.device)
    prompt_text = args.prompt or "Привет"
    if args.instruction is not None:
        if not args.instruction.strip():
            parser.error("Instruction must contain text")
        prompt_text = format_instruction(args.instruction)
    completion, _ = generator.complete(
        prompt_text, limit or 80, args.temperature, args.top_k, args.seed, args.max_new_bytes,
        top_p=args.top_p, repetition_penalty=args.repetition_penalty, frequency_penalty=args.frequency_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(completion.split("\nUser:", 1)[0].strip() if args.instruction is not None else prompt_text + completion)


if __name__ == "__main__":
    main()
