from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import code_candidate as candidate


def task(index: int) -> dict[str, object]:
    code_prompt = f"def task_func(value={index}):\n"
    return {
        "ref": f"bigcodebench-{index}",
        "prompt": f"Return the value.\n\n{code_prompt}",
        "gold": "    return value\n",
        "partition": "train",
        "inputs": {
            "code_prompt": code_prompt,
            "entry_point": "task_func",
            "libraries": ["math"],
            "source": "BigCodeBench",
        },
        "max_output_tokens": 1024,
    }


def payload() -> dict[str, object]:
    return {
        "counts": dict(candidate.EXPECTED_COUNTS),
        "manifest": {
            "track": "code",
            "counts": dict(candidate.EXPECTED_COUNTS),
            "metric": "execution_pass_rate",
            "source": candidate.EXPECTED_SOURCE_MANIFEST,
            "license": candidate.EXPECTED_LICENSE,
            "ground_truth": "module",
            "digests": {
                "tasks": candidate.EXPECTED_TASKS_DIGEST,
                "tests": candidate.EXPECTED_TESTS_DIGEST,
            },
            "binding_convention": {"entry_point": "task_func", "runner": "unittest"},
            "required_libraries": ["math"],
        },
        "reference": [],
        "reference_model": "",
        "tasks": [task(index) for index in range(candidate.EXPECTED_COUNTS["train"])],
        "track": "code",
        "version": candidate.CORPUS_VERSION,
    }


class CodeCorpusTests(unittest.TestCase):
    def test_validates_complete_solution_contract(self) -> None:
        found = payload()
        validation = candidate.validate_public_corpus_payload(found, expected_canonical_digest=None)
        self.assertEqual(validation.task_count, 94)
        self.assertEqual(validation.reference_count, 0)

    def test_rejects_schema_or_reference_drift(self) -> None:
        found = payload()
        found["new_field"] = True
        with self.assertRaisesRegex(candidate.CandidateError, "keys changed"):
            candidate.validate_public_corpus_payload(found, expected_canonical_digest=None)

        found = payload()
        found["reference_model"] = "unreviewed/model@revision"
        found["reference"] = [{"ref": "bigcodebench-0", "completion": "pass"}]
        with self.assertRaisesRegex(candidate.CandidateError, "no public reference"):
            candidate.validate_public_corpus_payload(found, expected_canonical_digest=None)

    def test_rejects_body_only_or_duplicate_tasks(self) -> None:
        found = payload()
        first = found["tasks"][0]  # type: ignore[index]
        first["inputs"]["code_prompt"] = ""  # type: ignore[index]
        with self.assertRaisesRegex(candidate.CandidateError, "code_prompt"):
            candidate.validate_public_corpus_payload(found, expected_canonical_digest=None)

        found = payload()
        found["tasks"][1]["ref"] = "bigcodebench-0"  # type: ignore[index]
        with self.assertRaisesRegex(candidate.CandidateError, "repeats a task ref"):
            candidate.validate_public_corpus_payload(found, expected_canonical_digest=None)

        found = payload()
        first = found["tasks"][0]  # type: ignore[index]
        first["inputs"]["libraries"] = [{}]  # type: ignore[index]
        with self.assertRaisesRegex(candidate.CandidateError, "contains a non-string"):
            candidate.validate_public_corpus_payload(found, expected_canonical_digest=None)

    def test_digest_pin_rejects_changed_content(self) -> None:
        found = payload()
        with self.assertRaisesRegex(candidate.CandidateError, "content changed"):
            candidate.validate_public_corpus_payload(found)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(candidate.CandidateError, "repeats JSON key"):
            candidate._strict_json(b'{"track":"code","track":"extract"}', "fixture")


class TmpfsAndDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tmpfs_boundary_rejects_persistent_paths(self) -> None:
        with self.assertRaisesRegex(candidate.CandidateError, "below volatile tmpfs"):
            candidate.assert_tmpfs_path(Path("/workspace/not-volatile"))
        self.assertEqual(candidate.assert_tmpfs_path(self.root), self.root.resolve())

    def test_prepare_is_deterministic_and_constructs_complete_target(self) -> None:
        corpus = self.root / "public.json"
        corpus.write_text(json.dumps(payload(), sort_keys=True), encoding="utf-8")
        fake_validation = candidate.CorpusValidation(
            canonical_digest=candidate.PUBLIC_CORPUS_CANONICAL_DIGEST,
            canonical_bytes=1,
            task_count=94,
            reference_count=0,
            refs_digest="sha256:" + "a" * 64,
        )
        with mock.patch.object(
            candidate,
            "load_public_corpus",
            return_value=(payload(), fake_validation),
        ):
            first = candidate.prepare_dataset(
                corpus, self.root / "first", holdout_examples=16, seed=92
            )
            second = candidate.prepare_dataset(
                corpus, self.root / "second", holdout_examples=16, seed=92
            )
        self.assertEqual(first["train_refs_digest"], second["train_refs_digest"])
        self.assertEqual(first["holdout_refs_digest"], second["holdout_refs_digest"])
        self.assertEqual(first["train_examples"], 78)
        row = json.loads((self.root / "first" / "train.jsonl").read_text().splitlines()[0])
        self.assertTrue(row["completion"].startswith("def task_func"))
        compile(row["completion"], "prepared", "exec")

    def test_prepared_dataset_tampering_is_rejected(self) -> None:
        corpus = self.root / "public.json"
        corpus.write_text(json.dumps(payload(), sort_keys=True), encoding="utf-8")
        fake_validation = candidate.CorpusValidation(
            canonical_digest=candidate.PUBLIC_CORPUS_CANONICAL_DIGEST,
            canonical_bytes=1,
            task_count=94,
            reference_count=0,
            refs_digest="sha256:" + "a" * 64,
        )
        prepared = self.root / "prepared"
        with mock.patch.object(
            candidate,
            "load_public_corpus",
            return_value=(payload(), fake_validation),
        ):
            candidate.prepare_dataset(
                corpus,
                prepared,
                holdout_examples=16,
                seed=92,
            )
        rows, manifest = candidate.load_prepared_dataset(prepared)
        self.assertEqual(len(rows), 78)
        self.assertEqual(manifest["holdout_examples"], 16)
        train_path = prepared / "train.jsonl"
        train_path.write_text(
            train_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(candidate.CandidateError, "train.jsonl digest changed"):
            candidate.load_prepared_dataset(prepared)


class BaseSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def git_blob(raw: bytes) -> str:
        return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324

    def test_verifies_file_size_and_identity(self) -> None:
        raw = b"exact pinned file"
        (self.root / "config.json").write_bytes(raw)
        files = {
            "config.json": {"size": len(raw), "git_blob": self.git_blob(raw)},
        }
        with (
            mock.patch.object(candidate, "RECOMMENDED_BASE_FILES", files),
            mock.patch.object(candidate, "RECOMMENDED_BASE_REQUIRED_BYTES", len(raw)),
        ):
            identity = candidate.verify_recommended_base_snapshot(self.root)
            self.assertEqual(identity["required_bytes"], len(raw))
            (self.root / "config.json").write_bytes(b"tampered pinned file")
            with self.assertRaisesRegex(candidate.CandidateError, "bytes, expected"):
                candidate.verify_recommended_base_snapshot(self.root)


if __name__ == "__main__":
    unittest.main()
