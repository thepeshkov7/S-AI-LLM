import json
import unittest
from unittest.mock import Mock

import torch

from training_1b.exam import FIXTURE, SETTINGS, prose_loss, repeated_fourgram_fraction, run_cases, summarize


class ExamTests(unittest.TestCase):
    def test_exact_exam_has_ten_cases_and_matches_pasted_prompts(self):
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
        expected = [case for case in cases if case["suite"] == "exam"]
        baseline_path = FIXTURE.parent / "results" / "step10000-exam-pasted.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertEqual([row["language"] for row in expected], ["ru"] * 5 + ["en"] * 5)
        self.assertEqual([(row["id"], row["prompt"]) for row in expected],
                         [(row["id"], row["prompt"]) for row in baseline["rows"]])
        self.assertEqual(baseline["source"], "user_pasted_terminal_output")

    def test_generation_records_verbatim_text_and_leaves_human_scores_ungraded(self):
        answer = " \n4 dollars.\n"
        case = {"id": "exam-08", "suite": "exam", "language": "en", "prompt": "How much is left?",
                "exact_answers": ["4 dollars"]}
        generator = Mock()
        generator.complete.return_value = (answer, 6)
        rows = run_cases(generator, [case], "exam", "cpu")
        generator.complete.assert_called_once_with(
            "User: How much is left?\nAssistant:", seed=42, preserve_output=True, **SETTINGS
        )
        self.assertEqual(rows[0]["raw_answer"], answer)
        summary = summarize(rows)["exam"]["en"]
        self.assertEqual(summary["exact_matches"], 1)
        self.assertIsNone(summary["human_review"]["coherence"]["mean_out_of_2"])
        self.assertIsNone(summary["human_score_out_of_100"])
        rows[0]["human_review"]["coherence"] = 2
        self.assertEqual(summarize(rows)["exam"]["en"]["human_review"]["coherence"]["mean_out_of_2"], 2)

    def test_prose_loss_uses_next_token_shift_and_repetition_catches_joined_loops(self):
        generator = Mock()
        generator.model.context = 3
        generator.tokenizer.encode.return_value.ids = [1, 2, 3, 4]
        generator.model.vocabulary_loss.return_value = torch.tensor(1.5)
        self.assertEqual(prose_loss(generator, "A sentence.", "cpu"), {"mean_nll": 1.5, "target_tokens": 3})
        x, y = generator.model.vocabulary_loss.call_args.args
        self.assertEqual(x.tolist(), [[1, 2, 3]])
        self.assertEqual(y.tolist(), [[2, 3, 4]])
        generator.model.context = 2
        generator.model.vocabulary_loss.side_effect = [torch.tensor(1.0), torch.tensor(4.0)]
        self.assertEqual(prose_loss(generator, "A sentence.", "cpu"), {"mean_nll": 2.0, "target_tokens": 3})
        self.assertGreater(repeated_fourgram_fraction("ArtArtArtArtArt"), 0.5)
        self.assertEqual(repeated_fourgram_fraction("a b c d e f"), 0.0)


if __name__ == "__main__":
    unittest.main()
