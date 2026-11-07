from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import torch

from model import TinyLLM
from packed_corpus import LANGUAGES, SPLITS, PackedCorpus, pack
from prepare_oasst import dialogue
from pretrain_packed import train


class PackedTrainingTest(unittest.TestCase):
    def test_paused_and_continuous_training_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for split in SPLITS:
                rows = [
                    {"id": f"{split}-{lang}", "language": lang, "text": phrase * 10 + split}
                    for lang, phrase in (("ru", "Слова и фразы для теста. "), ("en", "Words and sentences for testing. "))
                ]
                (raw / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            packed = root / "packed"
            pack([raw], packed, vocab_size=300, tokenizer_docs=10)
            config = {
                "corpus": packed, "device": "cpu", "bf16": False, "seed": 123, "threads": 1,
                "width": 16, "heads": 2, "layers": 1, "context": 8, "steps": 4,
                "batch_size": 2, "learning_rate": 0.001, "gradient_accumulation": 2,
                "eval_every": 2, "eval_windows": 3,
            }
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            with redirect_stdout(StringIO()):
                train(Namespace(**config, output=uninterrupted, stop_after=None, resume=False))
                train(Namespace(**config, output=resumed, stop_after=1, resume=False))
                train(Namespace(**config, output=resumed, stop_after=None, resume=True))
            one, two = [
                torch.load(path / "training-state.pt", weights_only=True)
                for path in (uninterrupted, resumed)
            ]
            for key, weights in one["model"].items():
                torch.testing.assert_close(weights, two["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(one["sampler"], two["sampler"], rtol=0, atol=0)
            self.assertEqual(one["best_metrics"], two["best_metrics"])

            corpus = PackedCorpus(packed)
            x, y = corpus.batch(4, 8, torch.Generator().manual_seed(1), "cpu")
            self.assertEqual(tuple(x.shape), (4, 8))
            torch.testing.assert_close(x[:, 1:], y[:, :-1])
            for lang, sample in zip(LANGUAGES, x.chunk(2), strict=True):
                self.assertTrue(torch.isin(sample, corpus.splits["train"][lang]).all())
            with (packed / "train.en.bin").open("r+b") as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "does not match its manifest"):
                PackedCorpus(packed)

    def test_answer_only_projection_matches_dense_masked_loss(self):
        model = TinyLLM(vocab_size=20, width=16, heads=2, layers=1, context=8)
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        y = torch.tensor([[-100, -100, 4, 0], [-100, 7, 8, 0]])
        torch.testing.assert_close(model(x, y)[1], model.answer_loss(x, y))

    def test_filtered_or_missing_ancestor_does_not_become_a_dialogue(self):
        root = ("t", None, "ru", "prompter", "Что такое туман?")
        reply = ("t", "first", "ru", "assistant", "Капли воды возле поверхности.")
        followup = ("t", "second", "ru", "prompter", "А почему он исчезает?")
        answer = ("t", "third", "ru", "assistant", "Капли могут испариться.")
        nodes = {"first": root, "second": reply, "third": followup}
        self.assertEqual(len(dialogue(nodes, answer)), 4)
        del nodes["second"]
        self.assertIsNone(dialogue(nodes, answer))


if __name__ == "__main__":
    unittest.main()
