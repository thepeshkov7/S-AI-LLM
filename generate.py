"""Load a local checkpoint and sample its autoregressive text continuation."""
import argparse
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import CopyLLM, TinyLLM


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


class TextGenerator:
    def __init__(self, model, tokenizer=None, memory_routing=False, chat_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_routing = memory_routing
        self.chat_model = chat_model or ("S AI 0.2 Memory" if memory_routing else None)
        self.eos = tokenizer.token_to_id("<|end_of_document|>") if tokenizer else None

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        version = checkpoint.get("format_version")
        if version == 1:
            architecture = checkpoint.get("architecture")
            architectures = {None: TinyLLM, CopyLLM.architecture: CopyLLM}
            if architecture not in architectures:
                raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
            model = architectures[architecture](**checkpoint["config"])
            tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
            if tokenizer.get_vocab_size() != model.config.vocab_size or tokenizer.token_to_id("<|end_of_document|>") is None:
                raise ValueError("Checkpoint tokenizer does not match the model")
        elif version is None:
            # Convert the original MultiheadAttention names to the same QKV/projection layout.
            model = TinyLLM(context=checkpoint["context"])
            tokenizer = None
            state = {checkpoint_key(key): value for key, value in state.items()}
        else:
            raise ValueError(f"Unsupported checkpoint version: {version}")
        model.load_state_dict(state, strict=True)
        metrics = checkpoint.get("metrics", {})
        chat_model = None
        if tokenizer is not None and metrics.get("bilingual_chat") is True:
            chat_model = metrics.get("chat_model")
            if not isinstance(chat_model, str) or not chat_model.strip():
                chat_model = "S AI 0.5 Grounded" if isinstance(model, CopyLLM) else "S AI 0.4 Bilingual"
        return cls(model.to(device).eval(), tokenizer, metrics.get("memory_routing") == 1, chat_model)

    @torch.inference_mode()
    def complete(self, prompt, max_new_tokens=80, temperature=0.85, top_k=40, seed=None, max_bytes=None):
        if not prompt or max_new_tokens < 1 or top_k < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid generation parameters")
        ids = self.tokenizer.encode(prompt).ids if self.tokenizer else list(prompt.encode("utf-8"))
        device = self.model.token.weight.device
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        output = []
        for _ in range(max_new_tokens):
            recent = torch.tensor([ids[-self.model.context:]], device=device, dtype=torch.long)
            scores = self.model.next_scores(recent)[0]
            values, indices = (scores / temperature).topk(min(top_k, scores.size(-1)))
            selected = torch.multinomial(values.softmax(dim=0), 1, generator=generator)
            token = indices[selected].item()
            if token == self.eos:
                break
            ids.append(token)
            output.append(token)
        if self.tokenizer:
            completion = self.tokenizer.decode(output).rstrip("\ufffd")
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
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    limit = args.max_new_bytes if args.max_new_bytes is not None else args.max_new_tokens
    if args.threads < 1 or (limit is not None and limit < 1):
        parser.error("Threads and output limits must be positive")
    torch.set_num_threads(args.threads)
    generator = TextGenerator.load(args.checkpoint)
    prompt_text = args.prompt or "Привет"
    if args.instruction is not None:
        if not args.instruction.strip():
            parser.error("Instruction must contain text")
        prompt_text = format_instruction(args.instruction)
    completion, _ = generator.complete(
        prompt_text, limit or 80, args.temperature, args.top_k, args.seed, args.max_new_bytes,
    )
    print(completion.strip() if args.instruction is not None else prompt_text + completion)


if __name__ == "__main__":
    main()
