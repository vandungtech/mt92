from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import boundary_contrastive as boundary
from training import evaluation_selection as selection


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def synthetic_corpus(self) -> tuple[Path, bytes, list[dict[str, object]]]:
        rows: list[dict[str, object]] = [
            {"ref": f"synthetic-{index:04d}", "partition": "train"}
            for index in range(boundary.BOUNDARY_CORPUS_TRAIN_EXAMPLES)
        ]
        _, remaining = boundary.boundary_outer_partition(rows)
        for row, ref in zip(
            remaining[:2], selection.BOUNDARY_TOKENIZER_SKIPPED_REFS, strict=True
        ):
            row["ref"] = ref
        payload = json.dumps(
            {
                "tasks": rows,
                "track": "extract",
                "version": boundary.CORPUS_VERSION,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "synthetic.json"
        path.write_bytes(payload)
        return path, payload, rows

    def pinned_synthetic_corpus(self, payload: bytes):
        return mock.patch.multiple(
            boundary,
            BOUNDARY_CORPUS_BYTES=len(payload),
            BOUNDARY_CORPUS_FILE_DIGEST=boundary.digest_bytes(payload),
        )

    def test_synthetic_fold_matches_trainer_algorithms_and_order(self) -> None:
        path, payload, _ = self.synthetic_corpus()
        with self.pinned_synthetic_corpus(payload):
            first = selection.select_evaluation_rows(
                path,
                selection=selection.BOUNDARY_INNER_SELECTION,
                seed=None,
                examples=None,
                legacy_examples=64,
            )
            explicit = selection.select_evaluation_rows(
                path,
                selection=selection.BOUNDARY_INNER_SELECTION,
                seed=92,
                examples=384,
                legacy_examples=64,
            )
            parsed = boundary.parse_boundary_corpus(payload)
            outer, remaining = boundary.boundary_outer_partition(parsed)
            encoded_rows = tuple(
                row
                for row in remaining
                if row["ref"] not in selection.BOUNDARY_TOKENIZER_SKIPPED_REFS
            )
            fold = boundary.split_boundary_encoded_refs(
                [str(row["ref"]) for row in encoded_rows],
                outer_refs=[str(row["ref"]) for row in outer],
            )
            expected_refs = [
                str(encoded_rows[index]["ref"]) for index in fold.validation_indices
            ]

        first_refs = [str(row["ref"]) for row in first.rows]
        self.assertEqual(first_refs, expected_refs)
        self.assertEqual(first.rows, explicit.rows)
        self.assertEqual(first.manifest, explicit.manifest)
        self.assertEqual(len(first.rows), 384)
        self.assertTrue(
            set(first_refs).isdisjoint(selection.BOUNDARY_TOKENIZER_SKIPPED_REFS)
        )
        self.assertEqual(
            first.manifest["selected_ordered_refs_digest"],
            boundary.canonical_json_digest(first_refs),
        )
        self.assertEqual(
            first.manifest["selected_refs_digest"], boundary.refs_digest(first_refs)
        )
        self.assertEqual(
            first.manifest["skipped_refs_digest"],
            boundary.refs_digest(selection.BOUNDARY_TOKENIZER_SKIPPED_REFS),
        )
        self.assertEqual(first.manifest["tokenizer_digest"], boundary.BASE_TOKENIZER_DIGEST)
        self.assertEqual(first.manifest["max_length"], 512)
        self.assertEqual(first.manifest["outer_seed"], 92)
        self.assertEqual(first.manifest["inner_seed"], 92)
        self.assertEqual(
            {
                key: first.manifest[key]
                for key in (
                    "corpus_train_examples",
                    "outer_examples",
                    "post_outer_examples",
                    "skipped_examples",
                    "encoded_examples",
                    "inner_train_examples",
                    "inner_validation_examples",
                    "selected_examples",
                )
            },
            {
                "corpus_train_examples": 4_816,
                "outer_examples": 384,
                "post_outer_examples": 4_432,
                "skipped_examples": 2,
                "encoded_examples": 4_430,
                "inner_train_examples": 4_046,
                "inner_validation_examples": 384,
                "selected_examples": 384,
            },
        )

    def test_exact_mode_rejects_corpus_byte_tampering(self) -> None:
        path, payload, _ = self.synthetic_corpus()
        tampered = bytearray(payload)
        tampered[len(tampered) // 2] ^= 1
        path.write_bytes(tampered)

        with (
            self.pinned_synthetic_corpus(payload),
            self.assertRaisesRegex(ValueError, "pinned public corpus"),
        ):
            selection.select_evaluation_rows(
                path,
                selection=selection.BOUNDARY_INNER_SELECTION,
                seed=None,
                examples=None,
                legacy_examples=64,
            )

    def test_exact_mode_rejects_contradictory_overrides_before_file_access(self) -> None:
        missing = self.root / "missing.json"
        with self.assertRaisesRegex(ValueError, "requires --seed 92"):
            selection.select_evaluation_rows(
                missing,
                selection=selection.BOUNDARY_INNER_SELECTION,
                seed=93,
                examples=None,
                legacy_examples=64,
            )
        with self.assertRaisesRegex(ValueError, "requires --examples 384"):
            selection.select_evaluation_rows(
                missing,
                selection=selection.BOUNDARY_INNER_SELECTION,
                seed=None,
                examples=383,
                legacy_examples=64,
            )

    def test_absent_legacy_overrides_preserve_seed_and_example_defaults(self) -> None:
        path, _, rows = self.synthetic_corpus()
        result = selection.select_evaluation_rows(
            path,
            selection=None,
            seed=None,
            examples=None,
            legacy_examples=64,
        )
        expected = list(rows)
        random.Random(92).shuffle(expected)  # noqa: S311 - deterministic fixture shuffle, not cryptographic

        self.assertEqual(
            [row["ref"] for row in result.rows],
            [row["ref"] for row in expected[:64]],
        )
        self.assertEqual(result.manifest["mode"], selection.LEGACY_SELECTION)
        self.assertEqual(result.manifest["seed"], 92)
        self.assertEqual(result.manifest["requested_examples"], 64)
        self.assertEqual(result.manifest["selected_examples"], 64)

    def test_retained_corpus_replays_expected_immutable_fold(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "training"
            / "corpus"
            / "extract-fb5f1332493b1abe.json"
        )
        if not path.is_file():
            self.skipTest("retained pinned corpus is not present")

        result = selection.select_evaluation_rows(
            path,
            selection=selection.BOUNDARY_INNER_SELECTION,
            seed=None,
            examples=None,
            legacy_examples=64,
        )

        self.assertEqual(len(result.rows), 384)
        self.assertEqual(
            result.manifest,
            {
                "schema": selection.SELECTION_SCHEMA,
                "mode": selection.BOUNDARY_INNER_SELECTION,
                "corpus_version": boundary.CORPUS_VERSION,
                "corpus_file_digest_algorithm": boundary.BOUNDARY_FILE_DIGEST_ALGORITHM,
                "corpus_file_digest": boundary.BOUNDARY_CORPUS_FILE_DIGEST,
                "corpus_file_bytes": boundary.BOUNDARY_CORPUS_BYTES,
                "refs_digest_algorithm": boundary.BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
                "tokenizer_digest": boundary.BASE_TOKENIZER_DIGEST,
                "max_length": 512,
                "outer_algorithm": boundary.BOUNDARY_OUTER_SPLIT_ALGORITHM,
                "inner_algorithm": boundary.BOUNDARY_INNER_SPLIT_ALGORITHM,
                "seed": 92,
                "outer_seed": 92,
                "inner_seed": 92,
                "corpus_train_examples": 4_816,
                "outer_examples": 384,
                "post_outer_examples": 4_432,
                "skipped_examples": 2,
                "encoded_examples": 4_430,
                "inner_train_examples": 4_046,
                "inner_validation_examples": 384,
                "selected_examples": 384,
                "outer_refs_digest": (
                    "sha256:22f2abadc08586f7b1a67a4ef7a843b5"
                    "08a44ca2508bf46e88a6751009e1803c"
                ),
                "skipped_refs_digest": (
                    "sha256:9eddd043e1ea076e551bd2f22c3030b2"
                    "0319c3e2f5213f10871d9f4d1959c108"
                ),
                "selected_refs_digest": (
                    "sha256:3ae12b94d723c4ea627fa391500b64d0"
                    "516eea800c86a7b1635217082c9f7a3c"
                ),
                "selected_ordered_refs_digest": (
                    "sha256:12817194c09aaf96e1f4d67c5193863"
                    "c4c20bbfcf5a5f4e5576edb22c752035c"
                ),
            },
        )
        self.assertEqual(
            [row["ref"] for row in result.rows[:5]],
            [
                "bc5cdr-train-01396",
                "bc5cdr-train-02909",
                "bc5cdr-train-02019",
                "bc5cdr-train-05152",
                "bc5cdr-train-00747",
            ],
        )
        self.assertEqual(
            [row["ref"] for row in result.rows[-5:]],
            [
                "bc5cdr-train-00665",
                "bc5cdr-train-02637",
                "bc5cdr-train-03991",
                "bc5cdr-train-00244",
                "bc5cdr-train-04789",
            ],
        )


if __name__ == "__main__":
    unittest.main()
