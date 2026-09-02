from __future__ import annotations

import copy
import json
import unittest

try:
    from training.train_extract import (
        MAX_ENTITY_SUBSTITUTION_EXAMPLES,
        augment_train_fold_entity_substitutions,
        parse_args,
    )
except ModuleNotFoundError as exc:
    TRAINING_DEPENDENCY_ERROR = str(exc)
else:
    TRAINING_DEPENDENCY_ERROR = ""


PROMPT_PREFIX = (
    "Extract entities. The instructions may mention Aspirin, but must remain unchanged."
    "\n\nText: "
)


def public_row(
    ref: str,
    text: str,
    entities: list[tuple[str, str]],
    *,
    partition: str = "train",
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
        "partition": partition,
        "inputs": {"text": text, "source": "public-test"},
        "prompt": PROMPT_PREFIX + text,
        "gold": json.dumps(payload) if gold is None else gold,
    }


@unittest.skipIf(
    bool(TRAINING_DEPENDENCY_ERROR),
    f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}",
)
class EntitySubstitutionTests(unittest.TestCase):
    def test_disabled_default_is_backward_compatible_and_does_not_parse(self) -> None:
        args = parse_args(["--corpus", "public.json", "--base", "base", "--out", "out"])
        self.assertEqual(args.entity_substitution_examples, 0)

        malformed = [{"partition": "validation", "gold": "not-json"}]
        result = augment_train_fold_entity_substitutions(
            malformed, heldout_refs=set(), seed=92, max_examples=0
        )
        self.assertEqual(result.rows, ())
        self.assertFalse(result.manifest["enabled"])

    def test_deterministic_cap_same_type_and_train_only_donors(self) -> None:
        rows = [
            public_row(
                "r1",
                "Aspirin treats headache. Aspirin helps.",
                [("Aspirin", "Chemical"), ("Aspirin", "Chemical"), ("headache", "Disease")],
            ),
            public_row(
                "r2",
                "Ibuprofen treats migraine.",
                [("Ibuprofen", "Chemical"), ("migraine", "Disease")],
            ),
            public_row(
                "r3",
                "Metformin controls diabetes.",
                [("Metformin", "Chemical"), ("diabetes", "Disease")],
            ),
            public_row(
                "r4",
                "Warfarin may cause bleeding.",
                [("Warfarin", "Chemical"), ("bleeding", "Disease")],
            ),
        ]
        original = copy.deepcopy(rows)
        first = augment_train_fold_entity_substitutions(
            rows, heldout_refs={"reserved-only"}, seed=92, max_examples=3
        )
        second = augment_train_fold_entity_substitutions(
            list(reversed(rows)), heldout_refs={"reserved-only"}, seed=92, max_examples=3
        )

        self.assertEqual(first, second)
        self.assertEqual(rows, original)
        self.assertEqual(len(first.rows), 3)
        self.assertEqual(first.manifest["augmented_examples"], 3)
        source_refs = {row["ref"] for row in rows}
        source_types = {
            (row["ref"], entity["text"]): entity["type"]
            for row in rows
            for entity in json.loads(str(row["gold"]))["entities"]
        }
        for synthetic, record in zip(first.rows, first.manifest["examples"], strict=True):
            self.assertIn(record["source_ref"], source_refs)
            self.assertIn(record["donor_ref"], source_refs)
            self.assertNotEqual(record["source_ref"], record["donor_ref"])
            self.assertEqual(
                source_types[(record["source_ref"], record["source_text"])],
                record["type"],
            )
            self.assertEqual(
                source_types[(record["donor_ref"], record["donor_text"])],
                record["type"],
            )
            self.assertTrue(str(synthetic["prompt"]).endswith(synthetic["inputs"]["text"]))
            self.assertTrue(str(synthetic["prompt"]).startswith(PROMPT_PREFIX))
            self.assertNotIn("reserved-only", json.dumps(record))

    def test_replaces_every_literal_occurrence_and_matching_gold_in_public_style(self) -> None:
        rows = [
            public_row(
                "repeat",
                "Aspirin then Aspirin.",
                [("Aspirin", "Chemical"), ("Aspirin", "Chemical")],
            ),
            public_row("donor", "Ibuprofen.", [("Ibuprofen", "Chemical")]),
        ]
        result = augment_train_fold_entity_substitutions(
            rows, heldout_refs={"heldout"}, seed=7, max_examples=2
        )
        record = next(
            item for item in result.manifest["examples"] if item["source_ref"] == "repeat"
        )
        synthetic = next(row for row in result.rows if row["ref"] == record["augmented_ref"])
        donor = record["donor_text"]

        self.assertEqual(record["occurrence_count"], 2)
        self.assertEqual(synthetic["inputs"]["text"], f"{donor} then {donor}.")
        payload = json.loads(str(synthetic["gold"]))
        self.assertEqual(
            payload["entities"],
            [{"text": donor, "type": "Chemical"}, {"text": donor, "type": "Chemical"}],
        )
        self.assertTrue(str(synthetic["gold"]).startswith('{"entities": [{"text": '))
        self.assertIn("}, {", str(synthetic["gold"]))
        self.assertEqual(
            str(synthetic["prompt"])[: -len(str(synthetic["inputs"]["text"]))],
            PROMPT_PREFIX,
        )

    def test_fail_closed_schema_fold_and_holdout_overlap(self) -> None:
        valid = public_row("valid", "Aspirin.", [("Aspirin", "Chemical")])
        malformed = public_row(
            "bad", "Aspirin.", [("Aspirin", "Chemical")], gold="not-json"
        )
        with self.assertRaisesRegex(ValueError, "invalid gold JSON"):
            augment_train_fold_entity_substitutions(
                [valid, malformed], heldout_refs=set(), seed=92, max_examples=1
            )
        with self.assertRaisesRegex(ValueError, "only train-fold"):
            augment_train_fold_entity_substitutions(
                [public_row("reserved", "Aspirin.", [("Aspirin", "Chemical")], partition="test")],
                heldout_refs=set(),
                seed=92,
                max_examples=1,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            augment_train_fold_entity_substitutions(
                [valid], heldout_refs={"valid"}, seed=92, max_examples=1
            )
        for invalid_cap in (-1, True, MAX_ENTITY_SUBSTITUTION_EXAMPLES + 1):
            with self.subTest(invalid_cap=invalid_cap), self.assertRaises(ValueError):
                augment_train_fold_entity_substitutions(
                    [valid], heldout_refs=set(), seed=92, max_examples=invalid_cap
                )

    def test_nonliteral_overlapping_and_dual_typed_surfaces_are_never_repaired(self) -> None:
        rows = [
            public_row("absent", "NRA0160.", [("NRA 0160", "Chemical")]),
            public_row(
                "nested",
                "breast cancer.",
                [("breast cancer", "Disease"), ("cancer", "Disease")],
            ),
            public_row("pcp-c", "PCP.", [("PCP", "Chemical")]),
            public_row("pcp-d", "PCP.", [("PCP", "Disease")]),
            public_row("chem-a", "Aspirin.", [("Aspirin", "Chemical")]),
            public_row("chem-b", "Ibuprofen.", [("Ibuprofen", "Chemical")]),
            public_row("dis-a", "migraine.", [("migraine", "Disease")]),
            public_row("dis-b", "diabetes.", [("diabetes", "Disease")]),
        ]
        result = augment_train_fold_entity_substitutions(
            rows, heldout_refs={"heldout"}, seed=92, max_examples=4
        )

        self.assertEqual(
            result.manifest["ineligible_source_rows"]["gold_surface_absent"], 1
        )
        self.assertEqual(
            result.manifest["ineligible_source_rows"]["overlapping_entity_surfaces"], 1
        )
        self.assertEqual(result.manifest["globally_ambiguous_surfaces"], 1)
        for record in result.manifest["examples"]:
            self.assertNotIn(record["source_ref"], {"absent", "nested"})
            self.assertNotEqual(record["source_text"], "PCP")
            self.assertNotEqual(record["donor_text"], "PCP")

    def test_alphanumeric_boundary_collisions_exclude_source_and_donor(self) -> None:
        rows = [
            public_row(
                "unsafe-ascii",
                "rat and ratification.",
                [("rat", "Chemical")],
            ),
            public_row(
                "unsafe-unicode",
                "ACE and αACE.",  # noqa: RUF001 - the ambiguous character is the subject of this unsafe-unicode test
                [("ACE", "Chemical")],
            ),
            public_row("safe-a", "Aspirin.", [("Aspirin", "Chemical")]),
            public_row("safe-b", "Ibuprofen.", [("Ibuprofen", "Chemical")]),
        ]
        result = augment_train_fold_entity_substitutions(
            rows, heldout_refs={"heldout"}, seed=92, max_examples=2
        )

        self.assertEqual(
            result.manifest["ineligible_source_rows"]["alphanumeric_boundary_collision"],
            2,
        )
        unsafe_refs = {"unsafe-ascii", "unsafe-unicode"}
        for record in result.manifest["examples"]:
            self.assertNotIn(record["source_ref"], unsafe_refs)
            self.assertNotIn(record["donor_ref"], unsafe_refs)
            self.assertNotIn(record["source_text"], {"rat", "ACE"})
            self.assertNotIn(record["donor_text"], {"rat", "ACE"})

    def test_generated_ref_collision_with_heldout_ref_fails_closed(self) -> None:
        rows = [
            public_row("safe-a", "Aspirin.", [("Aspirin", "Chemical")]),
            public_row("safe-b", "Ibuprofen.", [("Ibuprofen", "Chemical")]),
        ]
        baseline = augment_train_fold_entity_substitutions(
            rows, heldout_refs=set(), seed=92, max_examples=1
        )
        collision_ref = baseline.manifest["examples"][0]["augmented_ref"]

        with self.assertRaisesRegex(ValueError, "generated augmentation ref collides"):
            augment_train_fold_entity_substitutions(
                rows, heldout_refs={collision_ref}, seed=92, max_examples=1
            )


if __name__ == "__main__":
    unittest.main()
