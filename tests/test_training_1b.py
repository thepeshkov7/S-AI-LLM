import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model import CopyLLM, TinyLLM
from packed_corpus import PackedCorpus, SPLITS, pack
from train import save_checkpoint
from training_1b.checkpoint import read_checkpoint
from training_1b.evaluate import DecodeCache
from training_1b.pretrain import Profile, one_pass_windows, train


class CopyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pointer_loss_is_finite_in_fp16_and_matches_full_distribution(self):
        model = CopyLLM(vocab_size=32, width=24, heads=3, layers=2, context=16)
        ids = torch.tensor([[1, 3, 3, 7], [0, 3, 6, 2]])
        targets = torch.tensor([[-100, 3, 6, 1], [3, -100, 2, 8]])
        scores, _ = model(ids)
        expected = F.nll_loss(scores.flatten(0, 1), targets.flatten())
        torch.testing.assert_close(model.answer_loss(ids, targets), expected)
        for block in model.blocks:
            block.requires_grad_(False)
        model.gradient_checkpointing = True
        with torch.autocast("cpu", dtype=torch.float16):
            loss = model.answer_loss(ids, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertIsNotNone(model.copy_query.weight.grad)

    def test_generation_cache_respects_variable_prompt_lengths(self):
        model = CopyLLM(vocab_size=16, width=24, heads=3, layers=2, context=16).eval()
        rows = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 6, 7, 9])]
        with torch.inference_mode():
            cache = DecodeCache(model, rows, capacity=8, eos=0)
            for tokens in ([4, 7], [1, 2]):
                positions = torch.tensor([len(row) - 1 for row in rows])
                batch = pad_sequence(rows, batch_first=True)
                torch.testing.assert_close(cache.scores(), model.next_scores(batch, positions))
                torch.testing.assert_close(cache.scores(vocabulary_only=True), TinyLLM.next_scores(model, batch, positions))
                cache.advance(tokens)
                rows = [torch.cat((row, torch.tensor([token]))) for row, token in zip(rows, tokens, strict=True)]

    def test_resume_produces_same_weights_as_uninterrupted_training(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            for split in SPLITS:
                samples = [
                    {"id": f"{split}-ru", "language": "ru", "text": f"{split} Учебный текст для проверки. " * 12},
                    {"id": f"{split}-en", "language": "en", "text": f"{split} A short training verification text. " * 12},
                ]
                (root / f"{split}.jsonl").write_text("\n".join(map(json.dumps, samples)), encoding="utf-8")
            pack([root], root / "packed", 300, 8)
            corpus = PackedCorpus(root / "packed")
            model = CopyLLM(vocab_size=corpus.tokenizer.get_vocab_size(), width=16, heads=2, layers=2, context=16)
            base = root / "base.pt"
            save_checkpoint(base, model, corpus.tokenizer, {})
            profile = Profile(steps=4, sequence_length=16, last_blocks=1, precision="fp32",
                              optimizer="adamw", accumulation=2, eval_every=2, eval_windows=2, save_every=2,
                              one_pass=True)
            for starts in one_pass_windows(corpus, profile).values():
                self.assertEqual(len(starts), 4)
                self.assertTrue((starts.sort().values.diff() >= profile.sequence_length).all())
            whole, partial = root / "whole", root / "partial"
            with self.assertRaisesRegex(ValueError, "exceed --max-session-tokens"):
                train(corpus, profile, base, partial, "cpu", 2, require_1b=False, max_session_tokens=63)
            train(corpus, profile, base, whole, "cpu", 4, require_1b=False)
            train(corpus, profile, base, partial, "cpu", 2, require_1b=False, max_session_tokens=64)
            train(corpus, profile, partial / "training-state.pt", partial, "cpu", 4, resume=True, require_1b=False)
            left = read_checkpoint(whole / "training-state.pt")
            right = read_checkpoint(partial / "training-state.pt")
            self.assertEqual(right["metrics"]["updated_transformer_blocks"], [2])
            self.assertEqual(left["metrics"]["last"]["tokens"], right["metrics"]["last"]["tokens"])
            for name, tensor in left["model"].items():
                torch.testing.assert_close(tensor, right["model"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
