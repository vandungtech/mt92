from __future__ import annotations

import ast
import copy
import hashlib
import struct
import tempfile
import unicodedata
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import code_candidate as candidate
from training import evaluate_code_gguf as evaluator
from training import historical_code_candidate as historical_candidate
from training import normalized_historical_code_candidate as normalized_historical_candidate


def _row(index: int) -> dict[str, object]:
    return {
        "completion": f"def task_func():\n    return {index}\n",
        "max_output_tokens": 1024,
        "prompt": f"raw prompt {index}",
        "ref": f"bigcodebench-{index}",
    }


class _Request:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _Decoding:
    GREEDY = object()


class _Response:
    def __init__(
        self,
        *,
        ref: str,
        output: object,
        error: str = "",
        output_pieces: int = 3,
    ) -> None:
        self.task_ref = ref
        self.output = output
        self.error = error
        self.output_tokens = output_pieces
        self.ttft_ms = 1.25
        self.total_ms = 7.5
        self.peak_rss_bytes = 8192


class _Engine:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests: list[_Request] = []

    def generate(self, request: _Request) -> _Response:
        self.requests.append(request)
        return self.response


def _gguf(file_type: int) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    return b"".join(
        (
            b"GGUF",
            struct.pack("<I", 3),
            struct.pack("<Q", 0),
            struct.pack("<Q", 2),
            string("general.architecture"),
            struct.pack("<I", 8),
            string("qwen3"),
            string("general.file_type"),
            struct.pack("<I", 4),
            struct.pack("<I", file_type),
        )
    )


def _tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), candidate.digest_file(path)))
    digest = hashlib.sha256()
    for relative, file_digest in entries:
        digest.update(unicodedata.normalize("NFC", relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


class StructuralSafetyTests(unittest.TestCase):
    def test_poison_text_is_parsed_but_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "must-not-exist"
            poison = (
                "def task_func():\n    return 1\n\n"
                f"open({str(sentinel)!r}, 'w').write('executed')\n"
            )
            fence = chr(96) * 3
            diagnostics = evaluator.structural_diagnostics(
                f"explanation\n{fence}python\n{poison}{fence}\n",
                poison,
            )
            self.assertFalse(sentinel.exists())
            self.assertFalse(diagnostics["raw_parseable_python"])
            self.assertTrue(diagnostics["scorer_extracted_parseable_python"])
            self.assertTrue(diagnostics["scorer_extracted_exact_reference_ast"])

    def test_module_has_no_dynamic_code_execution_calls(self) -> None:
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"compile", "eval", "exec"}
        }
        self.assertEqual(forbidden, set())

    def test_thinking_and_unfenced_failure_are_reported_without_rewriting(self) -> None:
        completion = "<think>draft</think> not Python"
        diagnostics = evaluator.structural_diagnostics(
            completion,
            "def task_func():\n    return 1\n",
        )
        self.assertEqual(diagnostics["scorer_extracted_output"], completion)
        self.assertTrue(diagnostics["raw_contains_thinking_markup"])
        self.assertFalse(diagnostics["raw_parseable_python"])


class ArtifactIdentityTests(unittest.TestCase):
    def test_official_tree_and_q8_header_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.gguf"
            model.write_bytes(_gguf(7))
            identity = evaluator.artifact_identity(
                root,
                entrypoint="model.gguf",
                expected_digest=_tree_digest(root),
                quantization="Q8_0",
            )
            self.assertEqual(identity["tree_digest"], _tree_digest(root))
            self.assertEqual(identity["entrypoint"]["gguf"]["architecture"], "qwen3")
            self.assertEqual(identity["entrypoint"]["gguf"]["file_type"], 7)

    def test_digest_and_quantization_mismatches_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.gguf").write_bytes(_gguf(7))
            with self.assertRaisesRegex(evaluator.EvaluationRefused, "tree digest"):
                evaluator.artifact_identity(
                    root,
                    entrypoint="model.gguf",
                    expected_digest="sha256:" + "0" * 64,
                    quantization="Q8_0",
                )
            with self.assertRaisesRegex(evaluator.EvaluationRefused, "file_type"):
                evaluator.artifact_identity(
                    root,
                    entrypoint="model.gguf",
                    expected_digest=_tree_digest(root),
                    quantization="Q4_K_M",
                )

    def test_symlink_and_parent_entrypoint_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.gguf"
            target.write_bytes(_gguf(7))
            (root / "model.gguf").symlink_to(target)
            with self.assertRaisesRegex(evaluator.EvaluationRefused, "symlink"):
                evaluator.artifact_identity(
                    root,
                    entrypoint="model.gguf",
                    expected_digest="sha256:" + "0" * 64,
                    quantization="Q8_0",
                )
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "relative POSIX"):
            evaluator._relative_entrypoint("../model.gguf")


class GenerationContractTests(unittest.TestCase):
    @mock.patch.object(
        evaluator,
        "_memory_snapshot",
        side_effect=(
            {"current_bytes": 100, "peak_bytes": 200},
            {"current_bytes": 110, "peak_bytes": 220},
        ),
    )
    def test_raw_request_and_result_are_exact(self, _memory: mock.Mock) -> None:
        response = _Response(
            ref="bigcodebench-1",
            output="def task_func():\n    return 1\n",
        )
        engine = _Engine(response)
        result = evaluator.generate_result(
            engine=engine,
            request_type=_Request,
            decoding=_Decoding,
            source=_row(1),
        )
        request = engine.requests[0]
        self.assertEqual(request.prompt, "raw prompt 1")
        self.assertEqual(request.inputs, {})
        self.assertIs(request.decoding, _Decoding.GREEDY)
        self.assertFalse(request.chat)
        self.assertEqual(request.seed, 0)
        self.assertEqual(request.nonce, evaluator.GENERATION_NONCE)
        self.assertEqual(frozenset(result), evaluator.RESULT_KEYS)
        self.assertTrue(result["ok"])
        self.assertEqual(result["engine_reported_output_pieces"], 3)
        self.assertEqual(result["peak_rss_bytes"], 8192)

    @mock.patch.object(
        evaluator,
        "_memory_snapshot",
        side_effect=(
            {"current_bytes": None, "peak_bytes": None},
            {"current_bytes": None, "peak_bytes": None},
        ),
    )
    def test_failed_response_records_empty_output_not_none_or_poison(
        self,
        _memory: mock.Mock,
    ) -> None:
        engine = _Engine(
            _Response(
                ref="bigcodebench-1",
                output="raise RuntimeError('must not be inspected as success')",
                error="generation failed",
                output_pieces=0,
            )
        )
        result = evaluator.generate_result(
            engine=engine,
            request_type=_Request,
            decoding=_Decoding,
            source=_row(1),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["raw_output"], "")
        self.assertEqual(result["error"], "generation failed")
        self.assertFalse(result["raw_parseable_python"])

    def test_sampler_contract_is_fully_neutral_and_context_bound(self) -> None:
        contract = evaluator.generation_contract(1024)
        self.assertEqual(
            contract["sampler"],
            {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "mirostat_mode": 0,
            },
        )
        self.assertFalse(contract["chat"])
        self.assertEqual(contract["threads"], 1)
        self.assertEqual(contract["gpu_layers"], 0)
        with self.assertRaises(evaluator.EvaluationRefused):
            evaluator.generation_contract(128)


class LineageAndReceiptTests(unittest.TestCase):
    def test_explicit_jsonl_must_be_the_complete_ordered_public_holdout(self) -> None:
        rows = [_row(index) for index in range(16)]
        self.assertEqual(evaluator.validate_explicit_diagnostic_rows(rows, rows), rows)
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "exact prepared holdout"):
            evaluator.validate_explicit_diagnostic_rows(list(reversed(rows)), rows)
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "incomplete"):
            evaluator.validate_explicit_diagnostic_rows(rows[:-1], rows[:-1])

    def test_v5_historical_header_is_accepted_and_wrong_schema_is_refused(self) -> None:
        manifest: dict[str, object] = {
            "seed": 92,
            "train_examples": 8000,
            "holdout_examples": 0,
        }
        manifest_digest = "sha256:" + "1" * 64
        metadata = {key: None for key in evaluator.TRAINING_METADATA_KEYS}
        metadata.update(
            {
                "schema": evaluator.TRAINING_SCHEMA_V5,
                "status": "complete",
                "run_kind": "final_all_public",
                "track": candidate.TRACK,
                "hardware_class": candidate.HARDWARE_CLASS,
                "base_model": candidate.QWEN3_BASE_MODEL,
                "corpus_version": historical_candidate.CORPUS_VERSION,
                "quality_claim": historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
                "dataset": {
                    "manifest": manifest,
                    "manifest_digest": manifest_digest,
                    "source_corpus": historical_candidate.source_corpus_identity(),
                },
            }
        )
        self.assertEqual(
            evaluator.validate_v5_receipt_header(
                metadata,
                training_manifest=manifest,
                training_manifest_digest=manifest_digest,
            ),
            metadata,
        )
        wrong = dict(metadata)
        wrong["schema"] = "microtensor.code.training.v4"
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "schema"):
            evaluator.validate_v5_receipt_header(
                wrong,
                training_manifest=manifest,
                training_manifest_digest=manifest_digest,
            )

    def test_v6_header_binds_normalized_source_counts_and_schemas(self) -> None:
        manifest: dict[str, object] = {
            "schema": normalized_historical_candidate.DATASET_SCHEMA,
            "corpus_profile": normalized_historical_candidate.CORPUS_PROFILE,
            "seed": normalized_historical_candidate.EXPECTED_SEED,
            "source_examples": normalized_historical_candidate.EXPECTED_SOURCE_EXAMPLES,
            "train_examples": normalized_historical_candidate.EXPECTED_TRAIN_EXAMPLES,
            "holdout_examples": normalized_historical_candidate.EXPECTED_HOLDOUT_EXAMPLES,
            "excluded_examples": normalized_historical_candidate.EXPECTED_EXCLUDED_EXAMPLES,
            "excluded_refs_file": normalized_historical_candidate.EXCLUDED_REFS_FILE,
            "excluded_refs_canonical_bytes": (
                normalized_historical_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES
            ),
            "excluded_refs_digest": (normalized_historical_candidate.EXPECTED_EXCLUDED_REFS_DIGEST),
            "target_construction": normalized_historical_candidate.TARGET_CONSTRUCTION,
            "normalization": normalized_historical_candidate.NORMALIZATION_CONTRACT,
            "quality_claim": normalized_historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        }
        manifest_digest = "sha256:" + "2" * 64
        metadata = {key: None for key in evaluator.TRAINING_METADATA_KEYS}
        metadata.update(
            {
                "schema": evaluator.TRAINING_SCHEMA_V6,
                "status": "complete",
                "run_kind": "final_all_public",
                "track": candidate.TRACK,
                "hardware_class": candidate.HARDWARE_CLASS,
                "base_model": candidate.QWEN3_BASE_MODEL,
                "corpus_version": normalized_historical_candidate.CORPUS_VERSION,
                "quality_claim": normalized_historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
                "dataset": {
                    "manifest": manifest,
                    "manifest_digest": manifest_digest,
                    "source_corpus": normalized_historical_candidate.source_corpus_identity(),
                },
            }
        )
        self.assertEqual(
            evaluator.validate_v6_receipt_header(
                metadata,
                training_manifest=manifest,
                training_manifest_digest=manifest_digest,
            ),
            metadata,
        )

        wrong_schema = copy.deepcopy(metadata)
        wrong_schema["schema"] = evaluator.TRAINING_SCHEMA_V5
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "schema"):
            evaluator.validate_v6_receipt_header(
                wrong_schema,
                training_manifest=manifest,
                training_manifest_digest=manifest_digest,
            )

        wrong_source = copy.deepcopy(metadata)
        wrong_source["dataset"]["source_corpus"]["raw_digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "source-corpus identity"):
            evaluator.validate_v6_receipt_header(
                wrong_source,
                training_manifest=manifest,
                training_manifest_digest=manifest_digest,
            )

        for field, changed in (
            ("seed", normalized_historical_candidate.EXPECTED_SEED + 1),
            ("train_examples", normalized_historical_candidate.EXPECTED_TRAIN_EXAMPLES - 1),
            ("holdout_examples", normalized_historical_candidate.EXPECTED_HOLDOUT_EXAMPLES + 1),
            ("excluded_examples", normalized_historical_candidate.EXPECTED_EXCLUDED_EXAMPLES + 1),
            ("excluded_refs_digest", "sha256:" + "0" * 64),
            ("schema", historical_candidate.DATASET_SCHEMA),
        ):
            with self.subTest(manifest_field=field):
                wrong_manifest = copy.deepcopy(manifest)
                wrong_manifest[field] = changed
                coordinated = copy.deepcopy(metadata)
                coordinated["dataset"]["manifest"] = wrong_manifest
                with self.assertRaisesRegex(evaluator.EvaluationRefused, field):
                    evaluator.validate_v6_receipt_header(
                        coordinated,
                        training_manifest=wrong_manifest,
                        training_manifest_digest=manifest_digest,
                    )

    def test_training_lineage_dispatches_only_from_strict_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=candidate.TMPFS_MOUNT) as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            run = root / "run"
            source = root / "source.json"
            base = root / "base"
            arguments = (run, dataset, source, base)

            dataset.joinpath("manifest.json").write_bytes(
                candidate.canonical_json_bytes(
                    {"schema": normalized_historical_candidate.DATASET_SCHEMA}
                )
            )
            v6_result = ({"schema": evaluator.TRAINING_SCHEMA_V6}, ())
            with (
                mock.patch.object(
                    evaluator,
                    "load_v6_training_lineage",
                    return_value=v6_result,
                ) as load_v6,
                mock.patch.object(evaluator, "load_v5_training_lineage") as load_v5,
            ):
                self.assertEqual(evaluator.load_training_lineage(*arguments), v6_result)
                load_v6.assert_called_once_with(*arguments)
                load_v5.assert_not_called()

            dataset.joinpath("manifest.json").write_bytes(
                candidate.canonical_json_bytes({"schema": historical_candidate.DATASET_SCHEMA})
            )
            v5_result = ({"schema": evaluator.TRAINING_SCHEMA_V5}, ())
            with (
                mock.patch.object(
                    evaluator,
                    "load_v5_training_lineage",
                    return_value=v5_result,
                ) as load_v5,
                mock.patch.object(evaluator, "load_v6_training_lineage") as load_v6,
            ):
                self.assertEqual(evaluator.load_training_lineage(*arguments), v5_result)
                load_v5.assert_called_once_with(*arguments)
                load_v6.assert_not_called()

            dataset.joinpath("manifest.json").write_bytes(
                candidate.canonical_json_bytes({"schema": "unsupported"})
            )
            with self.assertRaisesRegex(evaluator.EvaluationRefused, "schema is unsupported"):
                evaluator.load_training_lineage(*arguments)

            dataset.joinpath("manifest.json").write_bytes(b'{"schema":"first","schema":"second"}')
            with self.assertRaisesRegex(evaluator.EvaluationRefused, "repeats JSON key"):
                evaluator.load_training_lineage(*arguments)

    def test_optional_training_cli_is_all_or_none(self) -> None:
        empty = Namespace(
            training_run=None,
            training_dataset=None,
            training_source_corpus=None,
            training_base=None,
        )
        self.assertIsNone(evaluator._training_arguments(empty))
        partial = Namespace(
            training_run=Path("run"),
            training_dataset=None,
            training_source_corpus=None,
            training_base=None,
        )
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "supplied together"):
            evaluator._training_arguments(partial)


class RuntimeAndSummaryTests(unittest.TestCase):
    def test_exact_signed_runtime_markers_are_required(self) -> None:
        microtensor = SimpleNamespace(__mechanism__="0.3.0")
        constants = SimpleNamespace(RELEASE_VERSION="0.3.0", MECHANISM_VERSION="0.3.0")
        info = SimpleNamespace(name="llama-cpp", version="0.2.0", deterministic=True)
        gguf = SimpleNamespace(
            THREADS=1,
            GPU_LAYERS=0,
            SEED=0,
            DEFAULT_CONTEXT=2048,
            INFO=info,
        )
        evaluator.validate_engine_contract(
            microtensor,
            constants,
            gguf,
            llama_cpp_version="0.3.35",
        )
        with self.assertRaisesRegex(evaluator.EvaluationRefused, "llama-cpp-python"):
            evaluator.validate_engine_contract(
                microtensor,
                constants,
                gguf,
                llama_cpp_version="0.3.34",
            )

    def test_summary_is_honest_and_uses_linear_p95(self) -> None:
        reference = "def task_func():\n    return 1\n"
        base = {
            "ref": "bigcodebench-1",
            "ok": True,
            "error": "",
            "prompt_digest": "sha256:" + "1" * 64,
            "reference_digest": "sha256:" + "2" * 64,
            "max_output_tokens": 1024,
            "raw_output": reference,
            "raw_output_digest": "sha256:" + "3" * 64,
            "raw_output_utf8_bytes": len(reference.encode()),
            "engine_reported_output_pieces": 2,
            "ttft_ms": 1.0,
            "engine_total_ms": 10.0,
            "evaluator_wall_ms": 11.0,
            "evaluator_cpu_ms": 9.0,
            "rss_before_bytes": 100,
            "rss_after_bytes": 110,
            "peak_rss_bytes": 120,
            **evaluator.structural_diagnostics(reference, reference),
        }
        second = dict(base)
        second.update(
            {
                "ref": "bigcodebench-2",
                "engine_total_ms": 30.0,
                "evaluator_wall_ms": 31.0,
                "ttft_ms": 3.0,
            }
        )
        summary = evaluator.summarize_results([base, second])
        self.assertIsNone(summary["quality_score"])
        self.assertIsNone(summary["execution_pass_at_1"])
        self.assertEqual(summary["latency_ms"]["engine_total_all"]["p95_linear"], 29.0)

    def test_json_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "summary.json"
            evaluator._write_json(target, {"first": True})
            with self.assertRaises(FileExistsError):
                evaluator._write_json(target, {"second": True})


if __name__ == "__main__":
    unittest.main()
