from collections import Counter
import unittest

import torch

from evaluate_chat import completions, score_answer
from generate import TextGenerator
from make_chat_eval import cases
from model import TinyLLM
from train import Document, TrainingWindows, train_tokenizer


class PilotTest(unittest.TestCase):
    def test_balanced_sampling_uses_equal_tokens_per_language(self):
        documents = [
            ("ru", torch.arange(10, 90)),
            ("en", torch.arange(100, 110)),
        ]
        balanced = TrainingWindows(documents, context=4, seed=18, balance_languages=True)
        inputs, targets = balanced.batch(6, "cpu")
        self.assertTrue((inputs[:3] < 100).all())
        self.assertTrue((inputs[3:] >= 100).all())
        torch.testing.assert_close(inputs + 1, targets)
        with self.assertRaises(ValueError):
            balanced.batch(3, "cpu")

    def test_frozen_benchmark_has_all_language_task_pairs(self):
        examples = list(cases())
        self.assertEqual(len({row["id"] for row in examples}), 400)
        self.assertEqual(len({row["prompt"] for row in examples}), 400)
        self.assertEqual(Counter(row["language"] for row in examples), {"ru": 200, "en": 200})
        self.assertEqual(set(Counter((row["task"], row["language"]) for row in examples).values()), {20})

    def test_batched_decoding_matches_single_greedy_decoding(self):
        torch.manual_seed(91)
        tokenizer = train_tokenizer(
            [Document("a", "ru", "Привет мир"), Document("b", "en", "Good morning")], 300,
        )
        model = TinyLLM(tokenizer.get_vocab_size(), width=16, layers=1, context=12).eval()
        generator = TextGenerator(model, tokenizer)
        prompts = ["Good morning", "Привет, красивый и очень длинный мир", "Good"]
        expected = [generator.complete(prompt, max_new_tokens=10, top_k=1)[0].strip() for prompt in prompts]
        self.assertEqual(list(completions(generator, prompts, 10, 2)), expected)

    def test_scoring_keeps_human_cases_separate(self):
        self.assertIsNone(score_answer("human", "any text", "reference"))
        self.assertTrue(score_answer("exact", "ЛЕВ", "Лев"))
        self.assertTrue(score_answer("json", '{"time":"12:00"}', {"time": "12:00"}))
        self.assertFalse(score_answer("json", 'Here is {"time":"12:00"}', {"time": "12:00"}))
        self.assertFalse(score_answer("time", "12:00 or 13:00", "12:00"))


if __name__ == "__main__":
    unittest.main()
