from itertools import combinations
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from finetune import instruction_tensors
from generate import TextGenerator, format_instruction
from instruction_data import build_splits
from model import TinyLLM
from train import Document, train_tokenizer
from tune_bilingual import load_split


class InstructionTest(unittest.TestCase):
    def test_disjoint_semantic_groups_and_prompts(self):
        splits = build_splits()
        self.assertEqual(splits, build_splits())
        groups = {name: {item.group for item in rows} for name, rows in splits.items()}
        prompts = {name: {item.prompt for item in rows} for name, rows in splits.items()}
        for first, second in combinations(splits, 2):
            self.assertTrue(groups[first].isdisjoint(groups[second]))
            self.assertTrue(prompts[first].isdisjoint(prompts[second]))
        for examples in splits.values():
            self.assertEqual({item.language for item in examples}, {"en", "ru"})

    def test_mask_predicts_response_and_end_token_only(self):
        examples = build_splits()["train"][:2]
        documents = [
            Document(str(index), example.language, format_instruction(example.prompt) + " " + example.answer)
            for index, example in enumerate(examples)
        ]
        tokenizer = train_tokenizer(documents, vocab_size=512)
        model = TinyLLM(vocab_size=tokenizer.get_vocab_size(), width=16, layers=1)
        generator = TextGenerator(model, tokenizer)
        inputs, targets = instruction_tensors(examples, generator)
        for example, input_ids, output_ids in zip(examples, inputs, targets):
            prefix = tokenizer.encode(format_instruction(example.prompt)).ids
            self.assertEqual(input_ids[:len(prefix)].tolist(), prefix)
            self.assertTrue(torch.all(output_ids[:len(prefix) - 1] == -100))
            active = output_ids[output_ids != -100].tolist()
            self.assertEqual(tokenizer.decode(active[:-1]).strip(), example.answer)
            self.assertEqual(active[-1], generator.eos)

    def test_reviewed_curriculum_uses_the_same_dialogue_template_as_generation(self):
        row = {"id": "example", "language": "en", "family": "instruction",
               "turns": ["Hi!", "Hello!", "What is 2 + 2?", "4"]}
        raw = json.dumps(row).encode()
        manifest = {"instruction_format": "user_assistant_v1",
                    "splits": {"train": {"sha256": sha256(raw).hexdigest()}}}
        with TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "train.jsonl").write_bytes(raw)
            example, = load_split(path, "train", manifest, None)[("en", "instruction")]
        self.assertEqual(example.prompt, "User: Hi!\nAssistant: Hello!\n" + format_instruction(row["turns"][2]))
        self.assertEqual(example.answer, "4")


if __name__ == "__main__":
    unittest.main()
