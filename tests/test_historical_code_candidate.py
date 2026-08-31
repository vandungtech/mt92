from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import code_candidate
from training import historical_code_candidate as historical


def fixture_payload() -> dict[str, object]:
    return {
        "counts": dict(historical.EXPECTED_COUNTS),
        "manifest": copy.deepcopy(historical.EXPECTED_MANIFEST),
        "reference": [],
        "reference_model": "",
        "tasks": [
            {
                "gold": f"verbatim response {index} π\n",
                "max_output_tokens": 1024,
                "partition": "train",
                "prompt": f"prompt {index}",
                "ref": f"mgc-{index}",
            }
            for index in range(historical.EXPECTED_COUNTS["train"])
        ],
        "track": "code",
        "version": historical.CORPUS_VERSION,
    }


class HistoricalIdentityTests(unittest.TestCase):
    def test_official_response_identity_and_schema_are_pinned(self) -> None:
        self.assertEqual(
            historical.CORPUS_VERSION,
            "sha256:7299bd7c25056246c944ae0d38c7d0b0817b87ff1022ce331aa2bca865bc2f06",
        )
        self.assertEqual(historical.PUBLIC_CORPUS_RESPONSE_BYTES, 19_023_989)
        self.assertEqual(
            historical.PUBLIC_CORPUS_RAW_DIGEST,
            "sha256:eb76adcaabdd11c9ce0005c22e50a8530397c32127515a4461b1340e77e2d4b5",
        )
        self.assertEqual(
            historical.PUBLIC_CORPUS_CANONICAL_DIGEST,
            "sha256:18fad3468cdd409b39a4786a982c098e1378445083e913e9a215669f0acbebdc",
        )
        self.assertEqual(historical.EXPECTED_COUNTS["train"], 8_000)
        self.assertEqual(historical.DATASET_SCHEMA, "microtensor.code.prepared.v2")
        self.assertEqual(historical.TARGET_CONSTRUCTION, "gold_verbatim")
        self.assertEqual(
            historical.PREPARED_ROW_KEYS,
            frozenset({"completion", "max_output_tokens", "prompt", "ref"}),
        )


class HistoricalFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        cls.root = Path(cls.temporary.name)
        cls.payload = fixture_payload()
        cls.raw = json.dumps(
            cls.payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        canonical = historical.canonical_json_bytes(cls.payload)
        cls.identity_patch = mock.patch.multiple(
            historical,
            PUBLIC_CORPUS_RESPONSE_BYTES=len(cls.raw),
            PUBLIC_CORPUS_RAW_DIGEST=historical.digest_bytes(cls.raw),
            PUBLIC_CORPUS_CANONICAL_BYTES=len(canonical),
            PUBLIC_CORPUS_CANONICAL_DIGEST=historical.digest_bytes(canonical),
        )
        cls.identity_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.identity_patch.stop()
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.source = self.root / "source.json"
        self.source.write_bytes(self.raw)

    def tearDown(self) -> None:
        for path in self.root.iterdir():
            if path == self.source:
                path.unlink(missing_ok=True)

    def test_validates_exact_shape_without_parsing_gold(self) -> None:
        loaded, validation = historical.load_public_corpus(self.source)
        self.assertEqual(validation.task_count, 8_000)
        self.assertEqual(loaded["tasks"][0]["gold"], "verbatim response 0 π\n")

        changed = copy.deepcopy(self.payload)
        changed["tasks"][0]["unknown"] = True
        with self.assertRaisesRegex(historical.HistoricalCandidateError, "keys changed"):
            historical.validate_public_corpus_payload(
                changed,
                expected_canonical_digest=None,
            )

        changed = copy.deepcopy(self.payload)
        changed["tasks"][0]["ref"] = "bigcodebench-0"
        with self.assertRaisesRegex(historical.HistoricalCandidateError, "outside mgc"):
            historical.validate_public_corpus_payload(
                changed,
                expected_canonical_digest=None,
            )

        changed = copy.deepcopy(self.payload)
        changed["manifest"]["entry_point_style"] = "function"
        with self.assertRaisesRegex(historical.HistoricalCandidateError, "manifest changed"):
            historical.validate_public_corpus_payload(
                changed,
                expected_canonical_digest=None,
            )

    def test_source_must_be_exact_and_below_tmpfs(self) -> None:
        self.source.write_bytes(self.raw + b" ")
        with self.assertRaisesRegex(
            historical.HistoricalCandidateError,
            "raw byte count changed",
        ):
            historical.load_public_corpus(self.source)

        with tempfile.TemporaryDirectory(dir="/tmp") as persistent:
            outside = Path(persistent) / "source.json"
            outside.write_bytes(self.raw)
            with self.assertRaisesRegex(code_candidate.CandidateError, "volatile tmpfs"):
                historical.load_public_corpus(outside)

    def test_prepare_is_deterministic_verbatim_and_replayed_on_load(self) -> None:
        first_root = self.root / "first"
        second_root = self.root / "second"
        first = historical.prepare_dataset(
            self.source,
            first_root,
            holdout_examples=17,
            seed=92,
        )
        second = historical.prepare_dataset(
            self.source,
            second_root,
            holdout_examples=17,
            seed=92,
        )
        self.assertEqual(first["schema"], "microtensor.code.prepared.v2")
        self.assertEqual(first["target_construction"], "gold_verbatim")
        self.assertEqual(first["train_examples"], 7_983)
        self.assertEqual(first["holdout_examples"], 17)
        self.assertEqual(first["train_file_digest"], second["train_file_digest"])
        self.assertEqual(first["holdout_file_digest"], second["holdout_file_digest"])

        rows, loaded_manifest = historical.load_prepared_dataset(first_root, self.source)
        source_by_ref = {row["ref"]: row for row in self.payload["tasks"]}
        self.assertEqual(len(rows), 7_983)
        self.assertEqual(loaded_manifest, first)
        for row in rows[:20]:
            self.assertEqual(row["prompt"], source_by_ref[row["ref"]]["prompt"])
            self.assertEqual(row["completion"], source_by_ref[row["ref"]]["gold"])
        self.assertEqual(
            historical.replay_prepared_dataset(first_root, self.source),
            historical.source_corpus_identity(),
        )

    def test_replay_rejects_self_consistent_prepared_tampering(self) -> None:
        prepared = self.root / "tampered"
        historical.prepare_dataset(
            self.source,
            prepared,
            holdout_examples=8,
            seed=92,
        )
        train_path = prepared / "train.jsonl"
        rows = historical.load_prepared_rows(train_path, "train.jsonl")
        rows[0]["completion"] = "different but non-empty"
        train_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest_path = prepared / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["train_file_digest"] = historical.digest_file(train_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            historical.HistoricalCandidateError,
            "not an exact replay",
        ):
            historical.load_prepared_dataset(prepared, self.source)

    def test_prepared_rows_reject_duplicate_json_keys(self) -> None:
        malformed = self.root / "duplicates.jsonl"
        malformed.write_text(
            '{"completion":"x","max_output_tokens":1024,'
            '"prompt":"p","ref":"mgc-1","ref":"mgc-1"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(code_candidate.CandidateError, "repeats JSON key"):
            historical._load_prepared_rows(malformed, "duplicates.jsonl")


if __name__ == "__main__":
    unittest.main()
