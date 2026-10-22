"""Synthetic checks only; these do not re-run a model or the full raw archive."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transitions", ROOT / "scripts/analyze_iga_error_transitions.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TransitionTests(unittest.TestCase):
    def test_four_cells_and_wrong_label_change(self):
        r = m.partition([0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 1, 0, 1, 2])
        self.assertEqual([r[k] for k in ("stable_correct", "new_errors", "corrected_clean_errors", "persistent_errors")], [1, 1, 1, 2])
        self.assertEqual(r["changed_wrong_label"], 1)
        self.assertEqual(r["context_errors"], r["clean_errors"] + r["new_errors"] - r["corrected_clean_errors"])

    def test_invalid_classes_and_pairing(self):
        for args in (([], [], []), ([0], [0, 1], [0]), ([0], [3], [0]), ([0], [True], [0])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                m.partition(*args)

    def test_identical_models_common_subset(self):
        r = m.common_clean([0, 1, 2], [0, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1, 1])
        self.assertEqual(r, {"common_clean_correct": 2, "iga_new_errors": 1, "lora_new_errors": 1})

    def test_common_subset_empty(self):
        r = m.common_clean([0], [1], [0], [1], [0])
        self.assertEqual(r["common_clean_correct"], 0)
        self.assertIsNone(m.ratio(0, 0))

    def test_six_format_probability_average(self):
        self.assertEqual(m.semantic([[0., 0., 0.]] * 6), [1/3] * 3)
        expected = [math.exp(2)/(math.exp(2)+math.exp(1)+1), math.exp(1)/(math.exp(2)+math.exp(1)+1), 1/(math.exp(2)+math.exp(1)+1)]
        for a, b in zip(m.semantic([[2., 1., 0.]] * 6), expected):
            self.assertAlmostEqual(a, b, places=14)

    def test_shift_invariance(self):
        for a, b in zip(m.semantic([[2., 1., 0.]] * 6), m.semantic([[1002., 1001., 1000.]] * 6)):
            self.assertAlmostEqual(a, b, places=14)

    def test_bad_logits_rejected(self):
        for x in ([[0, 0, 0]] * 5, [[0, 0]] * 6, [[float("nan"), 0, 0]] * 6):
            with self.assertRaises(ValueError):
                m.semantic(x)

    def test_eighteen_models_not_ensembled(self):
        self.assertEqual(len(m.FAMILIES), 6)
        self.assertEqual(len(set(n for names in m.FAMILIES.values() for n in names)), 18)
        self.assertTrue(all(len(names) == 3 for names in m.FAMILIES.values()))

    def test_published_transition_identities(self):
        record = json.loads((ROOT / "docs/research/2026-10/error_diagnosis.json").read_text())
        for family in record["family_means_eight_backgrounds"].values():
            clean = family["mean_clean_errors"]
            persistent = family["mean_errors_in_both_conditions"]
            repaired = family["mean_clean_errors_corrected_after_backgrounds"]
            new = family["mean_new_errors_after_8_backgrounds"]
            total = family["mean_total_8_background_errors"]
            self.assertAlmostEqual(clean, persistent + repaired)
            self.assertAlmostEqual(total, persistent + new)
            self.assertAlmostEqual(family["clean_accuracy"], 1-clean/800)
            self.assertAlmostEqual(family["eight_background_accuracy"], 1-total/800)
            self.assertAlmostEqual(family["persistent_share_of_context_errors"], persistent/total)

    def test_published_common_clean_denominator(self):
        record = json.loads((ROOT / "docs/research/2026-10/error_diagnosis.json").read_text())["common_clean_correct_drawB"]
        rows = record["per_seed"]
        denominator = sum(r["common_clean_correct"] for r in rows)
        self.assertEqual(denominator, 1869)
        self.assertAlmostEqual(sum(r["iga_new_errors"] for r in rows)/denominator, record["iga_error_rate"])
        self.assertAlmostEqual(sum(r["lora_new_errors"] for r in rows)/denominator, record["lora_error_rate"])


if __name__ == "__main__":
    unittest.main()
