from __future__ import annotations

import unittest

from microtensor.scoring.extraction import micro_f1
from training.evaluate_gguf import entity_confusion_counts


class EntityConfusionCountsTests(unittest.TestCase):
    def test_matches_strict_entity_micro_f1_semantics(self) -> None:
        predictions = [
            {("alpha", "gene"), ("extra", "disease")},
            None,
            set(),
        ]
        golds = [
            {("alpha", "gene"), ("missing", "gene")},
            {("beta", "disease")},
            set(),
        ]

        true_positive, false_positive, false_negative = entity_confusion_counts(
            predictions, golds
        )

        self.assertEqual((true_positive, false_positive, false_negative), (1, 1, 2))
        expected_f1 = (
            2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
        )
        self.assertEqual(micro_f1(predictions, golds), expected_f1)

    def test_rejects_misaligned_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "align by document index"):
            entity_confusion_counts([set()], [])


if __name__ == "__main__":
    unittest.main()
