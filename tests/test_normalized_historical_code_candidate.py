from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import historical_code_candidate as historical
from training import normalized_historical_code_candidate as normalized


def fixture_payload() -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    for index in range(normalized.EXPECTED_SOURCE_EXAMPLES):
        gold = (
            f"plain unparseable response {index}"
            if index < normalized.EXPECTED_EXCLUDED_EXAMPLES
            else (f"```python\ndef solution_{index}():\n    return {index}\n```\nexplanation")
        )
        tasks.append(
            {
                "gold": gold,
                "max_output_tokens": 1024,
                "partition": "train",
                "prompt": f"prompt {index}",
                "ref": f"mgc-{index}",
            }
        )
    return {
        "counts": dict(historical.EXPECTED_COUNTS),
        "manifest": copy.deepcopy(historical.EXPECTED_MANIFEST),
        "reference": [],
        "reference_model": "",
        "tasks": tasks,
        "track": "code",
        "version": historical.CORPUS_VERSION,
    }


class NormalizedHistoricalIdentityTests(unittest.TestCase):
    def test_public_normalization_identity_is_pinned(self) -> None:
        self.assertEqual(normalized.CORPUS_PROFILE, "historical7730-normalized-v1")
        self.assertEqual(
            normalized.DATASET_SCHEMA,
            "microtensor.code.prepared.historical-normalized.v1",
        )
        self.assertEqual(normalized.EXPECTED_SOURCE_EXAMPLES, 8_000)
        self.assertEqual(normalized.EXPECTED_TRAIN_EXAMPLES, 7_730)
        self.assertEqual(normalized.EXPECTED_EXCLUDED_EXAMPLES, 270)
        self.assertEqual(normalized.EXPECTED_TRAIN_FILE_BYTES, 15_681_824)
        self.assertEqual(
            normalized.EXPECTED_TRAIN_FILE_DIGEST,
            "sha256:10fd0cc986802fc78e5ac39384fab1f109401a95fcec1af5e2ce9c3f0efa4e03",
        )
        self.assertEqual(normalized.EXPECTED_TRAIN_REFS_CANONICAL_BYTES, 90_610)
        self.assertEqual(
            normalized.EXPECTED_TRAIN_REFS_DIGEST,
            "sha256:7ca162ac14570f270952ab4d5872b02fbdbc370ff84ea56b81793c48d89bdfc5",
        )
        self.assertEqual(normalized.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES, 3_184)
        self.assertEqual(
            normalized.EXPECTED_EXCLUDED_REFS_DIGEST,
            "sha256:03859ad7b36efe69a3a202ad203697490c50de810a2ff51e00d2abb32d96f35d",
        )
        self.assertEqual(normalized.EXPECTED_TARGET_PAIRS_CANONICAL_BYTES, 5_731_044)
        self.assertEqual(
            normalized.EXPECTED_TARGET_PAIRS_DIGEST,
            "sha256:8518c0bffc7d7c2122e2ffa896dce82b9c13a40ae0f06c0392acc0384552e79c",
        )
        self.assertEqual(normalized.EXPECTED_RAW_COMPLETION_UTF8_BYTES, 5_270_659)
        self.assertEqual(
            normalized.PYTHON311_INCOMPATIBLE_TARGETS,
            (
                (
                    "mgc-10018",
                    "sha256:709b3adf209408ef80461dda77447e71a7c5cb03a8ec3a09cd6ac2dc29407476",
                ),
                (
                    "mgc-22547",
                    "sha256:8ea9945d886739e48ee950633e7d76e8633c5a79f7c184a6ecaf229d784d0607",
                ),
                (
                    "mgc-31043",
                    "sha256:acadafb7075c4ee1f332bd13efa86b34d9e4d13eca3b16498044f538f08c39bd",
                ),
            ),
        )
        self.assertEqual(normalized.AST_FEATURE_VERSION, (3, 10))
        self.assertFalse(normalized.NORMALIZATION_CONTRACT["corpus_code_executed"])


class NormalizationAlgorithmTests(unittest.TestCase):
    def test_three_ambiguous_rows_select_parseable_nonlargest_blocks(self) -> None:
        mgc_14688 = """```python
# fib_cython.pyx
cdef int fib_cython(int n):
    return n
```
```python
# setup.py
from distutils.core import setup
setup(name="fib")
```
```python
```"""
        self.assertEqual(
            normalized.normalize_gold(mgc_14688),
            '# setup.py\nfrom distutils.core import setup\nsetup(name="fib")',
        )

        mgc_28005 = """```python
def preprocess(data):
    return data
```
```python
class DecisionTree:
    def __init__(self):
```
```python
from preprocess import preprocess
data = ...
processed = preprocess(data)
print(processed)
```"""
        self.assertEqual(
            normalized.normalize_gold(mgc_28005),
            "from preprocess import preprocess\ndata = ...\n"
            "processed = preprocess(data)\nprint(processed)",
        )

        mgc_362 = """```python
items = ["milk", "bread"]
for item in items:
    print(item)
```
```python
Frequent Itemsets:
milk -> bread
```"""
        self.assertEqual(
            normalized.normalize_gold(mgc_362),
            'items = ["milk", "bread"]\nfor item in items:\n    print(item)',
        )

    def test_longest_parseable_and_earliest_tie_are_deterministic(self) -> None:
        self.assertEqual(
            normalized.normalize_gold("```python\nx=1\n```\n```py\ny=2\n```").replace(" ", ""),
            "x=1",
        )
        self.assertEqual(
            normalized.normalize_gold("```\nx = 1\n```\n```python\nvalue = 12345\n```"),
            "value = 12345",
        )
        self.assertIsNone(normalized.normalize_gold("```python\nif True:\n```"))
        self.assertIsNone(normalized.normalize_gold("```python\n```"))
        self.assertIsNone(normalized.normalize_gold("```python\nx = 1"))

    def test_normalization_never_executes_candidate_text(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            marker = Path(temporary) / "executed"
            source = (
                "```python\n"
                f"open({str(marker)!r}, 'w').write('executed')\n"
                "raise RuntimeError('must remain text')\n"
                "```"
            )
            with (
                mock.patch.object(normalized.ast, "parse", return_value=object()) as parse,
                mock.patch("builtins.compile", side_effect=AssertionError("compile called")),
                mock.patch("builtins.eval", side_effect=AssertionError("eval called")),
                mock.patch("builtins.exec", side_effect=AssertionError("exec called")),
                mock.patch("builtins.__import__", side_effect=AssertionError("import called")),
            ):
                selected = normalized.normalize_gold(source)
            parse.assert_called_once()
            self.assertIn("raise RuntimeError", selected or "")
            self.assertFalse(marker.exists())

    def test_python311_compatibility_exclusions_precede_ast_parse(self) -> None:
        source = "```python\nvalue = f'{\"quoted\"}'\n```"
        for _ref, target_digest in normalized.PYTHON311_INCOMPATIBLE_TARGETS:
            with self.subTest(target_digest=target_digest):
                with (
                    mock.patch.object(
                        normalized,
                        "digest_bytes",
                        return_value=target_digest,
                    ) as digest,
                    mock.patch.object(
                        normalized.ast,
                        "parse",
                        side_effect=AssertionError("AST parse preceded compatibility exclusion"),
                    ),
                ):
                    self.assertIsNone(normalized.normalize_gold(source))
                digest.assert_called_once_with(b"value = f'{\"quoted\"}'")

    def test_provenance_returns_detached_normalization_contract(self) -> None:
        identity = normalized.source_corpus_identity()
        identity["normalization"]["regex_flags"].append("MUTATED")
        identity["normalization"]["python311_compatibility_exclusions"][0]["ref"] = "mutated"
        self.assertEqual(normalized.NORMALIZATION_CONTRACT["regex_flags"], ["DOTALL"])
        self.assertEqual(
            normalized.NORMALIZATION_CONTRACT["python311_compatibility_exclusions"][0]["ref"],
            "mgc-10018",
        )


class NormalizedHistoricalFixtureTests(unittest.TestCase):
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
        result = normalized._normalize_tasks(cls.payload["tasks"])
        cls.source_patch = mock.patch.multiple(
            historical,
            PUBLIC_CORPUS_RESPONSE_BYTES=len(cls.raw),
            PUBLIC_CORPUS_RAW_DIGEST=historical.digest_bytes(cls.raw),
            PUBLIC_CORPUS_CANONICAL_BYTES=len(canonical),
            PUBLIC_CORPUS_CANONICAL_DIGEST=historical.digest_bytes(canonical),
        )
        cls.normalized_patch = mock.patch.multiple(
            normalized,
            PUBLIC_CORPUS_RESPONSE_BYTES=len(cls.raw),
            PUBLIC_CORPUS_RAW_DIGEST=normalized.digest_bytes(cls.raw),
            PUBLIC_CORPUS_CANONICAL_BYTES=len(canonical),
            PUBLIC_CORPUS_CANONICAL_DIGEST=normalized.digest_bytes(canonical),
            EXPECTED_TRAIN_FILE_BYTES=len(result["train_bytes"]),
            EXPECTED_TRAIN_FILE_DIGEST=result["train_file_digest"],
            EXPECTED_TRAIN_REFS_CANONICAL_BYTES=result["train_refs_canonical_bytes"],
            EXPECTED_TRAIN_REFS_DIGEST=result["train_refs_digest"],
            EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES=len(result["excluded_bytes"]),
            EXPECTED_EXCLUDED_REFS_DIGEST=result["excluded_refs_digest"],
            EXPECTED_TARGET_PAIRS_CANONICAL_BYTES=result["target_pairs_canonical_bytes"],
            EXPECTED_TARGET_PAIRS_DIGEST=result["target_pairs_digest"],
            EXPECTED_RAW_COMPLETION_UTF8_BYTES=result["raw_completion_utf8_bytes"],
        )
        cls.source_patch.start()
        cls.normalized_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.normalized_patch.stop()
        cls.source_patch.stop()
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.source = self.root / "source.json"
        self.source.write_bytes(self.raw)

    def _new_output(self, name: str) -> Path:
        path = self.root / name
        self.assertFalse(path.exists())
        return path

    def test_prepare_is_deterministic_final_only_and_exactly_replayed(self) -> None:
        first_root = self._new_output("first")
        second_root = self._new_output("second")
        first = normalized.prepare_dataset(self.source, first_root)
        second = normalized.prepare_dataset(self.source, second_root)
        self.assertEqual(first["train_examples"], 7_730)
        self.assertEqual(first["holdout_examples"], 0)
        self.assertEqual(first["excluded_examples"], 270)
        self.assertEqual(first["train_file_digest"], second["train_file_digest"])
        self.assertEqual(first["excluded_refs_digest"], second["excluded_refs_digest"])
        self.assertEqual(
            (first_root / normalized.EXCLUDED_REFS_FILE).read_bytes(),
            (second_root / normalized.EXCLUDED_REFS_FILE).read_bytes(),
        )

        rows, loaded_manifest = normalized.load_prepared_dataset(first_root, self.source)
        self.assertEqual(len(rows), 7_730)
        self.assertEqual(loaded_manifest, first)
        self.assertTrue(all(normalized._parseable_python(row["completion"]) for row in rows))
        self.assertTrue(all("```" not in row["completion"] for row in rows))
        self.assertEqual(
            normalized.replay_prepared_dataset(first_root, self.source),
            normalized.source_corpus_identity(),
        )

        with self.assertRaisesRegex(normalized.NormalizedHistoricalCandidateError, "zero holdout"):
            normalized.prepare_dataset(
                self.source,
                self._new_output("bad-holdout"),
                holdout_examples=1,
            )
        with self.assertRaisesRegex(normalized.NormalizedHistoricalCandidateError, "seed"):
            normalized.prepare_dataset(
                self.source,
                self._new_output("bad-seed"),
                seed=93,
            )

    def test_invalid_exclusion_identity_is_refused_before_output(self) -> None:
        output = self._new_output("bad-exclusion-identity")
        with (
            mock.patch.object(
                normalized,
                "EXPECTED_EXCLUDED_REFS_DIGEST",
                "sha256:" + "0" * 64,
            ),
            self.assertRaisesRegex(
                normalized.NormalizedHistoricalCandidateError,
                "excluded refs digest changed",
            ),
        ):
            normalized.prepare_dataset(self.source, output)
        self.assertFalse(output.exists())

    def test_prepared_tamper_and_symlink_are_refused(self) -> None:
        prepared = self._new_output("tampered")
        normalized.prepare_dataset(self.source, prepared)
        excluded = prepared / normalized.EXCLUDED_REFS_FILE
        excluded.write_bytes(excluded.read_bytes() + b" ")
        with self.assertRaisesRegex(
            normalized.NormalizedHistoricalCandidateError,
            "excluded refs byte count changed",
        ):
            normalized.load_prepared_dataset(prepared, self.source)

        linked = self._new_output("linked")
        normalized.prepare_dataset(self.source, linked)
        train = linked / "train.jsonl"
        target = linked / "train-copy.jsonl"
        target.write_bytes(train.read_bytes())
        train.unlink()
        train.symlink_to(target)
        with self.assertRaisesRegex(
            normalized.NormalizedHistoricalCandidateError,
            "file set changed|regular non-symlinks",
        ):
            normalized.load_prepared_dataset(linked, self.source)

    def test_source_symlink_and_digest_change_are_refused(self) -> None:
        link = self.root / "source-link.json"
        link.symlink_to(self.source)
        with self.assertRaisesRegex(
            normalized.NormalizedHistoricalCandidateError,
            "regular non-symlink",
        ):
            normalized.load_public_corpus(link)

        self.source.write_bytes(self.raw + b" ")
        with self.assertRaisesRegex(
            normalized.NormalizedHistoricalCandidateError,
            "raw byte count changed",
        ):
            normalized.load_public_corpus(self.source)

    def test_duplicate_prepared_json_keys_are_refused(self) -> None:
        malformed = self.root / "duplicates.jsonl"
        malformed.write_text(
            '{"completion":"x=1","max_output_tokens":1024,'
            '"prompt":"p","ref":"mgc-1","ref":"mgc-1"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(normalized.candidate.CandidateError, "repeats JSON key"):
            normalized.load_prepared_rows(malformed, "duplicates.jsonl")


if __name__ == "__main__":
    unittest.main()
