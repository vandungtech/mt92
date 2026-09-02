from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as functional

    from training.train_extract import (
        BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
        BOUNDARY_FILE_DIGEST_ALGORITHM,
        BOUNDARY_TEXT_DIGEST_ALGORITHM,
        BoundaryContrastiveCollator,
        BoundaryContrastiveDataset,
        EncodedDataset,
        EncodedItemsDataset,
        boundary_contrastive_loss,
        boundary_outer_partition,
        boundary_training_setting_fields,
        generate_boundary_corruptions,
        parse_args,
        split_boundary_encoded_refs,
        validate_boundary_contrastive_args,
        write_boundary_manifest,
    )
except ModuleNotFoundError as exc:
    torch = None
    functional = None
    TRAINING_DEPENDENCY_ERROR = str(exc)
else:
    TRAINING_DEPENDENCY_ERROR = ""


PROMPT_PREFIX = "Extract exact entities.\n\nText: "


def public_row(
    ref: str,
    text: str,
    entities: list[tuple[str, str]],
    *,
    gold: str | None = None,
) -> dict[str, object]:
    payload = {
        "entities": [
            {"text": entity_text, "type": entity_type}
            for entity_text, entity_type in entities
        ]
    }
    return {
        "ref": ref,
        "partition": "train",
        "inputs": {"text": text},
        "prompt": PROMPT_PREFIX + text,
        "gold": json.dumps(payload) if gold is None else gold,
    }


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        if tokenize or enable_thinking:
            raise AssertionError("unexpected template mode")
        prefix = f"<user>{messages[0]['content']}</user><assistant>"
        return prefix if add_generation_prompt else prefix + messages[1]["content"] + "</assistant>"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        if add_special_tokens or return_offsets_mapping:
            raise AssertionError("unexpected tokenizer mode")
        return SimpleNamespace(input_ids=[index + 1 for index in range(len(text))])


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class BoundarySplitTests(unittest.TestCase):
    def test_outer_reserve_matches_historical_shuffle_without_parsing_gold(self) -> None:
        rows = [
            {
                "ref": f"row-{index:04d}",
                "partition": "train",
                "gold": object(),
            }
            for index in range(4_816)
        ]
        expected = list(rows)
        random.Random(92).shuffle(expected)  # noqa: S311 - deterministic fixture shuffle, not cryptographic

        outer, remaining = boundary_outer_partition(rows)

        self.assertEqual([row["ref"] for row in outer], [row["ref"] for row in expected[:384]])
        self.assertEqual(
            [row["ref"] for row in remaining], [row["ref"] for row in expected[384:]]
        )
        self.assertEqual(rows[0]["ref"], "row-0000")

    def test_inner_hash_split_is_exact_deterministic_and_disjoint(self) -> None:
        encoded = [f"encoded-{index:04d}" for index in range(4_430)]
        outer = [f"outer-{index:03d}" for index in range(384)]

        first = split_boundary_encoded_refs(encoded, outer_refs=outer)
        reversed_split = split_boundary_encoded_refs(list(reversed(encoded)), outer_refs=outer)

        self.assertEqual(len(first.train_indices), 4_046)
        self.assertEqual(len(first.validation_indices), 384)
        self.assertEqual(
            first.manifest["refs_digest_algorithm"],
            BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
        )
        self.assertEqual(first.manifest["overlap_counts"], {
            "outer_inner_train": 0,
            "outer_inner_validation": 0,
            "inner_train_inner_validation": 0,
        })
        for key in (
            "outer_refs_digest",
            "encoded_refs_digest",
            "inner_train_refs_digest",
            "inner_validation_refs_digest",
        ):
            self.assertEqual(first.manifest[key], reversed_split.manifest[key])
        self.assertEqual(
            set(first.manifest["inner_train_refs"]),
            set(reversed_split.manifest["inner_train_refs"]),
        )
        self.assertEqual(
            set(first.manifest["inner_validation_refs"]),
            set(reversed_split.manifest["inner_validation_refs"]),
        )

    def test_inner_split_rejects_fold_leakage_and_duplicate_refs(self) -> None:
        encoded = ["shared", *[f"encoded-{index}" for index in range(1, 4_430)]]
        outer = ["shared", *[f"outer-{index}" for index in range(383)]]
        with self.assertRaisesRegex(ValueError, "overlaps outer reserve"):
            split_boundary_encoded_refs(encoded, outer_refs=outer)
        duplicate_encoded = [
            "duplicate", "duplicate", *[f"encoded-{index}" for index in range(4_428)]
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            split_boundary_encoded_refs(
                duplicate_encoded,
                outer_refs=[f"outer-{index}" for index in range(384)],
            )


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class BoundaryCorruptionTests(unittest.TestCase):
    def eligible_rows(self) -> list[dict[str, object]]:
        return [
            public_row(
                f"source-{index}",
                f"xDrug{index}Y causes illness{index}.",
                [(f"Drug{index}", "Chemical"), (f"illness{index}", "Disease")],
                gold=(
                    '{ "entities": [{"text": "Drug'
                    + str(index)
                    + '", "type": "Chemical"}, {"text": "illness'
                    + str(index)
                    + '", "type": "Disease"}] }'
                ),
            )
            for index in range(8)
        ]

    def test_deterministic_balanced_raw_positive_single_boundary_pairs(self) -> None:
        rows = self.eligible_rows()
        original = copy.deepcopy(rows)
        first = generate_boundary_corruptions(
            rows, heldout_refs={"outer", "inner-validation"}, seed=92, max_examples=6
        )
        second = generate_boundary_corruptions(
            list(reversed(rows)),
            heldout_refs={"inner-validation", "outer"},
            seed=92,
            max_examples=6,
        )

        self.assertEqual(first, second)
        self.assertEqual(rows, original)
        self.assertEqual(len(first.pairs), 6)
        self.assertEqual(
            first.manifest["refs_and_records_digest_algorithm"],
            BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
        )
        self.assertEqual(
            first.manifest["gold_text_digest_algorithm"],
            BOUNDARY_TEXT_DIGEST_ALGORITHM,
        )
        self.assertEqual(first.manifest["direction_counts"], {
            "expansion": 3,
            "contraction": 3,
        })
        self.assertEqual(len({pair.source_ref for pair in first.pairs}), 6)
        by_ref = {row["ref"]: row for row in rows}
        for pair in first.pairs:
            source = by_ref[pair.source_ref]
            self.assertEqual(source["gold"], original[rows.index(source)]["gold"])
            self.assertTrue(str(pair.negative_row["gold"]).startswith('{ "entities": ['))
            positive = json.loads(str(source["gold"]))
            negative = json.loads(str(pair.negative_row["gold"]))
            changed = [
                index
                for index, (gold, corrupt) in enumerate(
                    zip(positive["entities"], negative["entities"], strict=True)
                )
                if gold != corrupt
            ]
            self.assertEqual(changed, [pair.record["entity_index"]])
            index = changed[0]
            self.assertEqual(
                positive["entities"][index]["type"], negative["entities"][index]["type"]
            )
            old = positive["entities"][index]["text"]
            new = negative["entities"][index]["text"]
            self.assertEqual(abs(len(old) - len(new)), 1)
            self.assertIn(new, source["inputs"]["text"])

    def test_malformed_schema_duplicate_and_boundary_cases_fail_closed(self) -> None:
        malformed = public_row("bad", "xDrugY", [("Drug", "Chemical")], gold="not-json")
        with self.assertRaisesRegex(ValueError, "invalid gold JSON"):
            generate_boundary_corruptions(
                [malformed], heldout_refs=set(), seed=92, max_examples=2
            )
        invalid_type = public_row("type", "xDrugY", [("Drug", "Other")])
        with self.assertRaisesRegex(ValueError, "invalid gold entity"):
            generate_boundary_corruptions(
                [invalid_type], heldout_refs=set(), seed=92, max_examples=2
            )
        duplicate = public_row(
            "duplicate", "xDrugY", [("Drug", "Chemical"), ("Drug", "Chemical")]
        )
        absent = public_row("absent", "no match", [("Drug", "Chemical")])
        result = generate_boundary_corruptions(
            [duplicate, absent], heldout_refs=set(), seed=92, max_examples=2
        )
        self.assertEqual(result.pairs, ())
        self.assertEqual(result.manifest["ineligible_source_rows"], {
            "duplicate_entity": 1,
            "gold_surface_absent": 1,
        })
        duplicate_key = public_row(
            "duplicate-key",
            "xDrugY",
            [("Drug", "Chemical")],
            gold='{"entities":[{"text":"Drug","text":"Drug","type":"Chemical"}]}',
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            generate_boundary_corruptions(
                [duplicate_key], heldout_refs=set(), seed=92, max_examples=2
            )

    def test_exact_input_and_single_character_entities_balance_boundary_directions(self) -> None:
        contraction_only = public_row("contract", "Drug", [("Drug", "Chemical")])
        expansion_only = public_row("expand", "xDy", [("D", "Chemical")])

        result = generate_boundary_corruptions(
            [contraction_only, expansion_only], heldout_refs=set(), seed=92, max_examples=2
        )

        self.assertEqual(len(result.pairs), 2)
        self.assertEqual(
            {pair.source_ref: pair.record["direction"] for pair in result.pairs},
            {"contract": "contraction", "expand": "expansion"},
        )

    def test_overlap_duplicate_sources_and_invalid_caps_are_rejected(self) -> None:
        valid = self.eligible_rows()[0]
        with self.assertRaisesRegex(ValueError, "held-out refs overlap"):
            generate_boundary_corruptions(
                [valid], heldout_refs={str(valid["ref"])}, seed=92, max_examples=2
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            generate_boundary_corruptions(
                [valid, copy.deepcopy(valid)], heldout_refs=set(), seed=92, max_examples=2
            )
        collision_rows = self.eligible_rows()[:2]
        baseline = generate_boundary_corruptions(
            collision_rows, heldout_refs=set(), seed=92, max_examples=2
        )
        collision_ref = baseline.pairs[0].negative_row["ref"]
        with self.assertRaisesRegex(ValueError, "generated boundary ref collides"):
            generate_boundary_corruptions(
                collision_rows, heldout_refs={collision_ref}, seed=92, max_examples=2
            )
        for invalid in (-2, 1, 514, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generate_boundary_corruptions(
                    [valid], heldout_refs=set(), seed=92, max_examples=invalid
                )

    def test_zero_pairs_are_supported_but_mode_is_disabled_by_default(self) -> None:
        args = parse_args(["--corpus", "public.json", "--base", "base", "--out", "out"])
        self.assertFalse(args.boundary_contrastive)
        self.assertEqual(args.boundary_contrastive_examples, 512)
        result = generate_boundary_corruptions(
            self.eligible_rows(), heldout_refs=set(), seed=92, max_examples=0
        )
        self.assertEqual(result.pairs, ())
        self.assertEqual(result.manifest["direction_counts"], {
            "expansion": 0,
            "contraction": 0,
        })
        item = EncodedDataset([self.eligible_rows()[0]], CharacterTokenizer(), 512)[0]
        self.assertEqual(set(item), {"input_ids", "labels"})


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class BoundaryControlAndManifestTests(unittest.TestCase):
    @staticmethod
    def args(*extra: str):
        return parse_args(
            ["--corpus", "public.json", "--base", "base", "--out", "out", *extra]
        )

    def test_fixed_seed_and_bounded_finite_controls_fail_closed(self) -> None:
        valid = self.args("--boundary-contrastive")
        validate_boundary_contrastive_args(valid)
        self.assertEqual(
            boundary_training_setting_fields(valid),
            {
                "boundary_contrastive": True,
                "boundary_contrastive_examples": 512,
                "boundary_contrastive_lambda": 0.1,
                "boundary_contrastive_margin": 0.0,
            },
        )

        with self.assertRaisesRegex(SystemExit, "requires --seed 92"):
            validate_boundary_contrastive_args(
                self.args("--boundary-contrastive", "--seed", "93")
            )
        for flag, values, message in (
            (
                "--boundary-contrastive-lambda",
                (-0.1, 1.0001, math.inf, math.nan),
                r"finite and in \[0, 1\]",
            ),
            (
                "--boundary-contrastive-margin",
                (-0.1, 20.0001, math.inf, math.nan),
                r"finite and in \[0, 20\]",
            ),
        ):
            for value in values:
                with self.subTest(flag=flag, value=value), self.assertRaisesRegex(
                    SystemExit, message
                ):
                    validate_boundary_contrastive_args(
                        self.args("--boundary-contrastive", flag, str(value))
                    )

        disabled = self.args(
            "--seed",
            "93",
            "--boundary-contrastive-lambda",
            "999",
            "--boundary-contrastive-margin",
            "999",
        )
        validate_boundary_contrastive_args(disabled)
        self.assertEqual(boundary_training_setting_fields(disabled), {})

    def test_manifest_digest_declares_exact_pretty_printed_file_bytes(self) -> None:
        payload = {
            "schema": "microtensor.boundary_contrastive.v1",
            "embedded_digest_algorithms": {
                "refs_and_records": BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
                "gold_text": BOUNDARY_TEXT_DIGEST_ALGORITHM,
            },
            "non_ascii": "é",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boundary_contrastive_manifest.json"
            identity = write_boundary_manifest(path, payload)
            expected_bytes = (
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            expected_digest = "sha256:" + hashlib.sha256(expected_bytes).hexdigest()

            self.assertEqual(path.read_bytes(), expected_bytes)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertEqual(
                identity,
                {
                    "manifest_file": path.name,
                    "manifest_digest_algorithm": BOUNDARY_FILE_DIGEST_ALGORITHM,
                    "manifest_digest": expected_digest,
                },
            )


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class BoundaryBatchAndLossTests(unittest.TestCase):
    @staticmethod
    def encoded(values: list[int], labels: list[int]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(values, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def test_flattening_keeps_positive_negative_pair_alignment(self) -> None:
        positive_a = self.encoded([1, 2, 3], [-100, 2, 3])
        positive_b = self.encoded([4, 5], [-100, 5])
        negative_a = self.encoded([6, 7, 8, 9], [-100, 7, 8, 9])
        dataset = BoundaryContrastiveDataset(
            EncodedItemsDataset([positive_a, positive_b], ["a", "b"]),
            {"a": negative_a},
        )

        batch = BoundaryContrastiveCollator(99)([dataset[0], dataset[1]])

        self.assertEqual(batch["input_ids"].tolist(), [
            [1, 2, 3, 99], [4, 5, 99, 99], [6, 7, 8, 9]
        ])
        self.assertEqual(int(batch["boundary_positive_count"]), 2)
        self.assertEqual(batch["boundary_pair_positive_indices"].tolist(), [0])
        self.assertEqual(batch["boundary_pair_negative_indices"].tolist(), [2])

    def test_zero_pair_collation_has_empty_aligned_indices(self) -> None:
        positive = self.encoded([1, 2], [-100, 2])
        dataset = BoundaryContrastiveDataset(
            EncodedItemsDataset([positive], ["only"]), {}
        )
        batch = BoundaryContrastiveCollator(0)([dataset[0]])
        self.assertEqual(batch["input_ids"].shape[0], 1)
        self.assertEqual(batch["boundary_pair_positive_indices"].numel(), 0)
        self.assertEqual(batch["boundary_pair_negative_indices"].numel(), 0)

    def test_causal_shift_prompt_masks_and_length_normalized_pair_loss(self) -> None:
        logits = torch.tensor(
            [
                [[20.0, -20.0], [2.0, 0.0], [0.0, 2.0], [30.0, -30.0]],
                [[-20.0, 20.0], [0.5, 0.0], [30.0, -30.0], [-30.0, 30.0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[-100, -100, 0, 1], [-100, -100, 0, -100]])
        actual = boundary_contrastive_loss(
            logits,
            labels,
            positive_count=1,
            pair_positive_indices=torch.tensor([0]),
            pair_negative_indices=torch.tensor([1]),
            pairwise_lambda=0.1,
            margin=0.0,
        )
        positive_losses = functional.cross_entropy(
            logits[0, 1:3], torch.tensor([0, 1]), reduction="none"
        )
        negative_loss = functional.cross_entropy(
            logits[1, 1:2], torch.tensor([0]), reduction="none"
        )[0]
        positive_ce = positive_losses.mean()
        expected = positive_ce + 0.1 * functional.softplus(
            -negative_loss + positive_losses.mean()
        )

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(float(logits.grad[:, 0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, 3].abs().sum()), 0.0)

    def test_pair_penalty_is_sum_normalized_by_all_positives(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0], [9.0, -9.0]],
                [[0.0, 2.0], [2.0, 0.0], [-9.0, 9.0]],
                [[0.5, 0.0], [0.0, 0.5], [9.0, -9.0]],
            ]
        )
        labels = torch.tensor(
            [[-100, 0, 1], [-100, 1, 0], [-100, 0, 1]]
        )
        actual = boundary_contrastive_loss(
            logits,
            labels,
            positive_count=2,
            pair_positive_indices=torch.tensor([0]),
            pair_negative_indices=torch.tensor([2]),
            pairwise_lambda=0.1,
            margin=0.5,
        )
        positive_ce = functional.cross_entropy(
            torch.cat((logits[0, :2], logits[1, :2])),
            torch.tensor([0, 1, 1, 0]),
        )
        positive_score = -functional.cross_entropy(
            logits[0, :2], torch.tensor([0, 1])
        )
        negative_score = -functional.cross_entropy(
            logits[2, :2], torch.tensor([0, 1])
        )
        expected = positive_ce + 0.1 * functional.softplus(
            negative_score - positive_score + 0.5
        ) / 2

        torch.testing.assert_close(actual, expected)

    def test_zero_pair_loss_is_exact_positive_ce_and_alignment_fails_closed(self) -> None:
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [9.0, -9.0]]])
        labels = torch.tensor([[-100, 0, 1]])
        actual = boundary_contrastive_loss(
            logits,
            labels,
            positive_count=1,
            pair_positive_indices=torch.tensor([], dtype=torch.long),
            pair_negative_indices=torch.tensor([], dtype=torch.long),
            pairwise_lambda=0.1,
            margin=0.0,
        )
        expected = functional.cross_entropy(logits[0, :2], labels[0, 1:])
        torch.testing.assert_close(actual, expected)

        bad_logits = torch.zeros((3, 3, 2))
        bad_labels = torch.tensor([[-100, 0, 1]] * 3)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            boundary_contrastive_loss(
                bad_logits,
                bad_labels,
                positive_count=1,
                pair_positive_indices=torch.tensor([0, 0]),
                pair_negative_indices=torch.tensor([1, 2]),
                pairwise_lambda=0.1,
                margin=0.0,
            )
        for value in (-0.1, 1.0001, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                boundary_contrastive_loss(
                    logits,
                    labels,
                    positive_count=1,
                    pair_positive_indices=torch.tensor([], dtype=torch.long),
                    pair_negative_indices=torch.tensor([], dtype=torch.long),
                    pairwise_lambda=value,
                    margin=0.0,
                )
        for value in (-0.1, 20.0001, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                boundary_contrastive_loss(
                    logits,
                    labels,
                    positive_count=1,
                    pair_positive_indices=torch.tensor([], dtype=torch.long),
                    pair_negative_indices=torch.tensor([], dtype=torch.long),
                    pairwise_lambda=0.1,
                    margin=value,
                )
        nonfinite_logits = logits.clone()
        nonfinite_logits[0, 0, 0] = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite token losses"):
            boundary_contrastive_loss(
                nonfinite_logits,
                labels,
                positive_count=1,
                pair_positive_indices=torch.tensor([], dtype=torch.long),
                pair_negative_indices=torch.tensor([], dtype=torch.long),
                pairwise_lambda=0.1,
                margin=0.0,
            )


if __name__ == "__main__":
    unittest.main()
