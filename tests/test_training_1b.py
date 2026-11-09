import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model import CopyLLM, TinyLLM
from packed_corpus import PackedCorpus, SPLITS, pack
from train import save_checkpoint
from training_1b.checkpoint import RECOVERY_CONFIG, RECOVERY_PARAMETERS, build_model, read_checkpoint
from training_1b.evaluate import DecodeCache
from training_1b.pretrain import Profile, loss_for, one_pass_windows, recovery_base_step, train


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

    def test_recovery_checkpoint_uses_exact_architecture(self):
        with torch.device("meta"):
            model = CopyLLM(**asdict(RECOVERY_CONFIG))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), RECOVERY_PARAMETERS)
        payload = {"architecture": model.architecture, "config": {**asdict(RECOVERY_CONFIG), "heads": 8}}
        with self.assertRaisesRegex(ValueError, "original S AI 1.276B"):
            build_model(payload, "cpu", recovery=True)

    def test_new_corpus_stage_keeps_absolute_model_step(self):
        original = Path("step-010000.pt")
        state_path = Path("training-state.pt")
        state = {
            "run_format": 1, "settings": {"profile": {"recovery_step": 10000}},
            "base_model_step": 10128, "step": 32, "metrics": {"last": {"model_step": 10160}},
        }
        self.assertEqual(recovery_base_step({}, original, False), 10000)
        self.assertEqual(recovery_base_step(state, state_path, False), 10160)
        self.assertEqual(recovery_base_step(state, state_path, True), 10128)
        with self.assertRaisesRegex(ValueError, "inconsistent step metadata"):
            recovery_base_step({**state, "step": 31}, state_path, True)

    def test_loss_learns_eos_without_predicting_the_next_document_from_it(self):
        model = CopyLLM(vocab_size=16, width=24, heads=3, layers=2, context=16)
        profile = Profile(steps=1, precision="fp32", optimizer="adamw", mask_document_transitions=True)
        batch = torch.tensor([[1, 3, 2, 6, 2]])
        expected = F.cross_entropy(
            model.output(model._encode(batch[:, :-1]))[0, [0, 1, 3]], torch.tensor([3, 2, 2]),
            reduction="sum",
        )
        torch.testing.assert_close(loss_for(model, batch, profile, eos=2), expected)

    def test_language_ratio_distributes_requested_microbatches(self):
        profile = Profile(steps=1, accumulation=5, ru_micro_batches=3)
        self.assertEqual(profile.language_schedule, ("ru", "en", "ru", "en", "ru"))

    def test_resume_produces_same_weights_as_uninterrupted_training(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            for split in SPLITS:
                samples = [
                    {"id": f"{split}-{language}-{i}", "language": language, "text": f"{split} {i} {text} " * 4}
                    for i in range(5)
                    for language, text in (("ru", "Учебный текст для проверки."), ("en", "A short training verification text."))
                ]
                (root / f"{split}.jsonl").write_text("\n".join(map(json.dumps, samples)), encoding="utf-8")
            pack([root], root / "packed", 300, 8)
            corpus = PackedCorpus(root / "packed")
            model = CopyLLM(vocab_size=corpus.tokenizer.get_vocab_size(), width=16, heads=2, layers=2, context=16)
            base = root / "base.pt"
            save_checkpoint(base, model, corpus.tokenizer, {})
            profile = Profile(steps=4, sequence_length=16, last_blocks=1, precision="fp32",
                              optimizer="adamw", accumulation=2, eval_every=2, eval_windows=2, save_every=2,
                              one_pass=True, mask_document_transitions=True)
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
            self.assertEqual(left["metrics"]["targets_by_language"], right["metrics"]["targets_by_language"])
            self.assertLess(sum(right["metrics"]["targets_by_language"].values()), right["metrics"]["last"]["tokens"])
            for name, tensor in left["model"].items():
                torch.testing.assert_close(tensor, right["model"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
