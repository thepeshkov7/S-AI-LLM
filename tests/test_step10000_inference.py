from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from torch.nn.utils.rnn import pad_sequence

from evaluate_chat import completions
from generate import TextGenerator, format_instruction, resolve_generation_head
from model import CopyLLM
from train import EOS
from training_1b.evaluate import DecodeCache, generate as cached_generate
from training_1b.inference_audit import decoder_checks, score_checks, tokenizer_checks


BOS = "<|begin_of_document|>"


def byte_tokenizer(*, postprocess=False):
    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([BOS, EOS])
    if postprocess:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BOS} $A {EOS}",
            special_tokens=[(token, tokenizer.token_to_id(token)) for token in (BOS, EOS)],
        )
    return tokenizer


class Step10000InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(12)
        self.tokenizer = byte_tokenizer()
        self.eos = self.tokenizer.token_to_id(EOS)
        self.model = CopyLLM(vocab_size=self.tokenizer.get_vocab_size(), width=16,
                             heads=2, layers=2, context=64).eval()

    def payload(self):
        return {"format_version": 1, "architecture": CopyLLM.architecture,
                "config": asdict(self.model.config), "tokenizer": self.tokenizer.to_str(),
                "model": self.model.state_dict()}

    def test_audit_runs_for_both_precisions_and_uses_a_forward_greedy_reference(self):
        generator = TextGenerator(self.model, self.tokenizer, vocabulary_only=True)
        for precision in (False, True):
            checks = score_checks(generator, "cpu", bf16=precision)
            self.assertTrue(all(result["close"] for head in checks.values() for result in head.values()))
        self.assertTrue(all(row["matches_forward_argmax"] and row["same_seed_same_output"]
                            for row in decoder_checks(generator, "cpu", limit=3)))
        self.assertTrue(all(row["text_round_trip"] and row["id_round_trip"]
                            for row in tokenizer_checks(generator)["rows"]))

    @torch.inference_mode()
    def test_batched_full_and_step_scores_agree_for_each_head(self):
        ids = torch.tensor([[21, 20, 37, 24, 73], [18, 11, 39, 36, 42]])
        rows = torch.arange(ids.shape[0])
        positions = torch.tensor([1, 3])
        for precision in (torch.float32, torch.bfloat16):
            with self.subTest(precision=precision), torch.autocast(
                "cpu", dtype=torch.bfloat16, enabled=precision == torch.bfloat16
            ):
                vocab_full = self.model.vocabulary_logits(ids)
                copy_full, _ = self.model(ids)
                torch.testing.assert_close(
                    vocab_full, F.linear(self.model._encode(ids), self.model.output.weight, self.model.output.bias)
                )
                for vocabulary, full in ((True, vocab_full), (False, copy_full)):
                    generator = TextGenerator(self.model, self.tokenizer, vocabulary_only=vocabulary)
                    selected = generator.next_scores(ids, positions)
                    self.assertTrue(selected.isfinite().all().item())
                    torch.testing.assert_close(selected.float(), full[rows, positions].float(), rtol=0.01, atol=0.005)
                    torch.testing.assert_close(
                        generator.next_scores(ids).float(), full[:, -1].float(), rtol=0.01, atol=0.005
                    )

    @torch.inference_mode()
    def test_prefix_is_causal_for_both_heads(self):
        ids = torch.tensor([[2, 19, 5, 29, 6]])
        changed = torch.tensor([[2, 19, 5, 89, 96]])
        torch.testing.assert_close(self.model.vocabulary_logits(ids)[:, :3],
                                   self.model.vocabulary_logits(changed)[:, :3])
        torch.testing.assert_close(self.model(ids)[0][:, :3], self.model(changed)[0][:, :3])
        torch.testing.assert_close(self.model.vocabulary_scores(ids[:, :3]),
                                   self.model.vocabulary_logits(ids)[:, 2])
        torch.testing.assert_close(self.model.next_scores(ids[:, :3]), self.model(ids)[0][:, 2])

    @torch.inference_mode()
    def test_kv_cache_agrees_with_full_forward_at_ragged_positions(self):
        windows = [torch.tensor([13, 36, 20]), torch.tensor([33, 35, 28, 24, 32])]
        cache = DecodeCache(self.model, windows, capacity=8, eos=self.eos)
        for tokens in ([82, 83], [49, 86], [10, 76]):
            padded = pad_sequence(windows, batch_first=True, padding_value=self.eos)
            positions = torch.tensor([len(window) - 1 for window in windows])
            rows = torch.arange(len(windows))
            torch.testing.assert_close(cache.scores(vocabulary_only=True),
                                       self.model.vocabulary_logits(padded)[rows, positions])
            torch.testing.assert_close(cache.scores(), self.model(padded)[0][rows, positions])
            cache.advance(tokens)
            windows = [torch.cat((window, torch.tensor([token]))) for window, token in zip(windows, tokens, strict=True)]

    @torch.inference_mode()
    def test_greedy_matches_full_forward_reference_after_context_rollover(self):
        prompt = "Привет, what comes next? " + "a" * self.model.context
        for vocabulary in (True, False):
            with self.subTest(vocabulary=vocabulary):
                generator = TextGenerator(self.model, self.tokenizer, vocabulary_only=vocabulary)
                ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
                expected = []
                for _ in range(6):
                    window = torch.tensor([ids[-self.model.context:]])
                    full = self.model.vocabulary_logits(window) if vocabulary else self.model(window)[0]
                    token = full[0, -1].argmax().item()
                    if token == self.eos:
                        break
                    expected.append(token)
                    ids.append(token)
                answer = generator.complete(prompt, max_new_tokens=6, top_k=1, repetition_penalty=1,
                                            no_repeat_ngram_size=0, preserve_output=True)
                self.assertEqual(answer, (self.tokenizer.decode(expected, skip_special_tokens=False), len(expected)))

    def test_greedy_never_samples_and_ties_select_first_id(self):
        generator = TextGenerator(self.model, self.tokenizer, vocabulary_only=True)
        tokens = sorted([self.tokenizer.token_to_id("a"), self.tokenizer.token_to_id("z")])
        scores = torch.zeros((1, self.tokenizer.get_vocab_size()))
        scores[0, tokens] = 1
        with patch.object(generator, "next_scores", return_value=scores), patch.object(
            torch, "multinomial", side_effect=AssertionError("Greedy must not sample")
        ):
            answer = generator.complete("Prompt", max_new_tokens=1, top_k=1, repetition_penalty=1,
                                        no_repeat_ngram_size=0)
        self.assertEqual(answer, (self.tokenizer.decode(tokens[:1]), 1))

    def test_bilingual_prompt_round_trips_without_injecting_bos_or_eos(self):
        tokenizer = byte_tokenizer(postprocess=True)
        generator = TextGenerator(self.model, tokenizer, vocabulary_only=True)
        prompt = format_instruction("  Привет!\nWhat is 2 + 2? 🙂  ")
        special = {tokenizer.token_to_id(BOS), tokenizer.token_to_id(EOS)}
        self.assertTrue(special.issubset(tokenizer.encode(prompt).ids))
        raw = generator.encode_prompt(prompt)
        self.assertEqual(raw, tokenizer.encode(prompt, add_special_tokens=False).ids)
        self.assertFalse(special.intersection(raw))
        self.assertTrue(all(0 <= token < self.model.config.vocab_size for token in raw))
        self.assertEqual(tokenizer.decode(raw, skip_special_tokens=False), prompt)
        restored = Tokenizer.from_str(tokenizer.to_str())
        self.assertEqual(restored.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(restored.encode(prompt, add_special_tokens=False).ids, raw)
        with patch("training_1b.evaluate.DecodeCache", wraps=DecodeCache) as create_cache:
            cached_generate(self.model, tokenizer, [prompt], [4], 1, 1, vocabulary_only=True)
        self.assertEqual(create_cache.call_args.args[1][0].tolist(), raw)

    def test_eos_stops_and_returned_text_contains_only_new_tokens(self):
        generator = TextGenerator(self.model, self.tokenizer, vocabulary_only=True)
        prompt = format_instruction("Уже видимый текст.")
        expected_prompt = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        token = self.tokenizer.token_to_id("z")
        predicted = iter((token, self.eos))
        actual_inputs = []

        def scores(ids):
            actual_inputs.append(ids[0].tolist())
            values = torch.zeros((1, self.tokenizer.get_vocab_size()))
            values[0, next(predicted)] = 1
            return values

        with patch.object(generator, "next_scores", side_effect=scores):
            result = generator.complete(prompt, max_new_tokens=6, top_k=1, repetition_penalty=1,
                                        no_repeat_ngram_size=0, preserve_output=True)
        self.assertEqual(actual_inputs, [expected_prompt, [*expected_prompt, token]])
        self.assertEqual(result, ("z", 1))

    def test_checkpoint_objective_selects_vocabulary_even_with_strong_copy_head(self):
        with torch.no_grad():
            self.model.copy_mode.weight.zero_()
            self.model.copy_mode.bias.copy_(torch.tensor([-20.0, 20.0, -20.0]))
            self.model.output.bias[self.tokenizer.token_to_id("z")] = 8
        payload = {**self.payload(), "objective": "vocabulary"}
        self.assertEqual(resolve_generation_head(payload)[0], "vocabulary")
        self.assertEqual(resolve_generation_head({**payload, "settings": {"objective": "vocabulary"}})[0],
                         "vocabulary")
        with self.assertRaises(ValueError):
            resolve_generation_head({**payload, "settings": {"profile": {"objective": "copy"}}})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            before = path.read_bytes()
            generator = TextGenerator.load(path)
            copied = TextGenerator.load(path, head="copy")
            self.assertEqual(before, path.read_bytes())
            self.assertTrue(generator.vocabulary_only)
            for name, expected in self.model.state_dict().items():
                torch.testing.assert_close(generator.model.state_dict()[name], expected)
        options = {"max_new_tokens": 1, "top_k": 1, "repetition_penalty": 1, "no_repeat_ngram_size": 0}
        self.assertEqual(generator.complete("ab", **options), ("z", 1))
        self.assertIn(copied.complete("ab", **options), (("a", 1), ("b", 1)))
        self.assertEqual(list(completions(generator, ["ab", "Привет!"], 1, 2)), ["z", "z"])

    def test_sampling_is_seeded_and_stays_within_top_k_and_top_p(self):
        generator = TextGenerator(self.model, self.tokenizer, vocabulary_only=True)
        tokens = [self.tokenizer.token_to_id(token) for token in "abc"]
        scores = torch.full((1, self.tokenizer.get_vocab_size()), -10.0)
        scores[0, tokens] = torch.tensor([2.0, 1.9, 1.0])
        options = {"max_new_tokens": 12, "top_k": 2, "top_p": 0.8, "temperature": 1.0, "seed": 9,
                   "repetition_penalty": 1, "no_repeat_ngram_size": 0}
        with patch.object(generator, "next_scores", return_value=scores):
            answer, count = generator.complete("Prompt", **options)
            self.assertEqual(generator.complete("Prompt", **options), (answer, count))
            self.assertEqual(count, options["max_new_tokens"])
            self.assertLessEqual(set(answer), {"a", "b"})
            top_nucleus, _ = generator.complete("Prompt", **{**options, "top_p": 0.1})
            self.assertEqual(top_nucleus, "a" * count)

    def test_loader_rejects_sparse_token_ids_with_matching_size(self):
        tokenizer_json = json.loads(Tokenizer(models.WordLevel({EOS: 0, "abc": 1}, unk_token=EOS)).to_str())
        tokenizer_json["model"]["vocab"]["abc"] = 40
        serialized = json.dumps(tokenizer_json)
        tokenizer = Tokenizer.from_str(serialized)
        model = CopyLLM(vocab_size=2, width=16, heads=2, layers=1, context=64).eval()
        payload = {**self.payload(), "config": asdict(model.config),
                   "tokenizer": serialized, "model": model.state_dict()}
        self.assertEqual(tokenizer.get_vocab_size(), model.config.vocab_size)
        self.assertEqual(tokenizer.encode("abc").ids, [40])
        with self.assertRaisesRegex(ValueError, "token ID"):
            TextGenerator.from_checkpoint(payload)

    def test_loader_rejects_missing_and_unexpected_weights(self):
        for bad_key in ("output.weight", "unexpected.weight"):
            with self.subTest(bad_key=bad_key), TemporaryDirectory() as directory:
                payload = self.payload()
                if bad_key in payload["model"]:
                    del payload["model"][bad_key]
                else:
                    payload["model"][bad_key] = torch.zeros(1)
                path = Path(directory) / "checkpoint.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(RuntimeError, bad_key.replace(".", r"\.")):
                    TextGenerator.load(path)


if __name__ == "__main__":
    unittest.main()
