from __future__ import annotations

# ruff: noqa: S101, S108 -- assertions and fixed /tmp identities are test fixtures.
import ast
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import code_candidate as candidate
from training import convert_code_gguf as real_converter
from training import evaluate_code_gguf as evaluator
from training import validate_code_gguf_diagnostic as validator


def _digest(raw: bytes) -> str:
    return validator._digest_bytes(raw)


def _v7_identity(character: str, size: int = 1) -> dict[str, object]:
    return {"bytes": size, "digest": "sha256:" + character * 64}


def _v7_spec_payload(
    conversion_schema: str = "microtensor.code.gguf-conversion.v4",
) -> dict[str, object]:
    commit = "a" * 40
    return validator.normalized_v7_spec_payload(
        source_root=Path("/tmp") / f"mt92-normalized-diagnostic-{commit[:7]}",
        source_commit=commit,
        source_files={
            relative: _v7_identity("1") for relative in validator.NORMALIZED_REQUIRED_SOURCE_FILES
        },
        training_receipt=_v7_identity("2"),
        merged_tree_digest="sha256:" + "3" * 64,
        conversion_schema=conversion_schema,
        conversion_receipt=_v7_identity("4"),
        calibration_receipt=(_v7_identity("5") if conversion_schema.endswith(".v5") else None),
        load_spec=_v7_identity("6"),
        artifact={
            "tree_digest": "sha256:" + "7" * 64,
            "entrypoint_bytes": 42,
            "entrypoint_digest": "sha256:" + "8" * 64,
        },
        runtime_identity=_v7_identity("9"),
    )


def _write_v7_spec(path: Path, payload: dict[str, object]) -> validator.NormalizedSpecBindings:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validator._load_normalized_v7_spec(path)


def _current94_spec_payload() -> dict[str, object]:
    commit = "c" * 40
    return validator.current94_v8_spec_payload(
        source_root=Path("/tmp") / f"mt92-current94-diagnostic-{commit[:7]}",
        source_commit=commit,
        source_files={
            relative: _v7_identity("1") for relative in validator.CURRENT94_REQUIRED_SOURCE_FILES
        },
        training_receipt=_v7_identity("2"),
        training_metrics=_v7_identity("3"),
        merged_tree_digest="sha256:" + "4" * 64,
        conversion_receipt=_v7_identity("5"),
        calibration_receipt=_v7_identity("6"),
        load_spec=_v7_identity("7"),
        artifact={
            "tree_digest": "sha256:" + "8" * 64,
            "entrypoint_bytes": 42,
            "entrypoint_digest": "sha256:" + "9" * 64,
        },
        conversion_runtime={
            "converter_interpreter": {
                "container_path": (
                    "/.uv/python_install/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
                ),
                "bytes": 21_333_768,
                "digest": (
                    "sha256:96d1b01675f2492922ec6f6ed8445791d2d3231ccae727cda521db30494b751e"
                ),
                "mode": "0o755",
            },
            "llama_cpp_runtime_closure": _v7_identity("a"),
        },
        runtime_identity=_v7_identity("b"),
    )


def _write_current94_spec(
    path: Path, payload: dict[str, object]
) -> validator.Current94SpecBindings:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validator._load_current94_v8_spec(path)


def _source_row(index: int, sentinel: Path) -> dict[str, object]:
    poison = (
        f"def task_func():\n    return {index}\n\n"
        f"open({str(sentinel)!r}, 'w').write('validator executed model output')\n"
    )
    return {
        "completion": poison,
        "max_output_tokens": 1024,
        "prompt": f"public prompt {index}",
        "ref": f"bigcodebench-{index:02d}",
    }


def _result_row(
    source: dict[str, object],
    *,
    raw_output: str | None = None,
    pieces: int = 373,
    ttft_ms: float = 5_256.0,
    engine_total_ms: float = 31_200.0,
    peak_rss_bytes: int = 1_073_741_824,
) -> dict[str, object]:
    prompt = str(source["prompt"])
    reference = str(source["completion"])
    output = reference if raw_output is None else raw_output
    output_bytes = output.encode("utf-8")
    row: dict[str, object] = {
        "ref": source["ref"],
        "ok": True,
        "error": "",
        "prompt_digest": _digest(prompt.encode("utf-8")),
        "reference_digest": _digest(reference.encode("utf-8")),
        "max_output_tokens": 1024,
        "raw_output": output,
        "raw_output_digest": _digest(output_bytes),
        "raw_output_utf8_bytes": len(output_bytes),
        "engine_reported_output_pieces": pieces,
        "ttft_ms": ttft_ms,
        "engine_total_ms": engine_total_ms,
        "evaluator_wall_ms": engine_total_ms + 1.0,
        "evaluator_cpu_ms": engine_total_ms / 2.0,
        "rss_before_bytes": 100_000,
        "rss_after_bytes": 110_000,
        "peak_rss_bytes": peak_rss_bytes,
        **evaluator.structural_diagnostics(output, reference),
    }
    if frozenset(row) != evaluator.RESULT_KEYS:
        raise AssertionError("test result fixture no longer matches evaluator.RESULT_KEYS")
    return row


def _evaluator_facade() -> SimpleNamespace:
    return SimpleNamespace(
        RESULT_KEYS=evaluator.RESULT_KEYS,
        SCHEMA=evaluator.SCHEMA,
        QUALITY_CLAIM=evaluator.QUALITY_CLAIM,
        RUNTIME_CLAIM=evaluator.RUNTIME_CLAIM,
        LINEAGE_CLAIM=evaluator.LINEAGE_CLAIM,
        structural_diagnostics=mock.Mock(wraps=evaluator.structural_diagnostics),
        summarize_results=evaluator.summarize_results,
    )


def _context(
    source_rows: list[dict[str, object]],
    *,
    static_evaluator: object | None = None,
) -> validator.ValidationContext:
    artifact = {
        "root": "/immutable/replay1/artifact",
        "tree_digest": validator.EXPECTED_ARTIFACT["tree_digest"],
        "entrypoint": {
            "bytes": validator.EXPECTED_ARTIFACT["entrypoint_bytes"],
            "digest": validator.EXPECTED_ARTIFACT["entrypoint_digest"],
            "gguf": {"version": 3, "architecture": "qwen3", "file_type": 15},
        },
    }
    configuration = {
        "generation": {"static": True, "max_input_tokens": 541},
        "load_manifest": {"format": "gguf", "quantization": "Q4_K_M"},
        "artifact_digest": artifact["tree_digest"],
        "diagnostic_refs_digest": validator.EXPECTED_LINEAGE_DIGESTS["refs"],
    }
    runtime = SimpleNamespace(
        identity={
            "python": {"version": "3.12.3"},
            "microtensor": {
                "release_version": "0.3.0",
                "mechanism_version": "0.3.0",
            },
        }
    )
    return validator.ValidationContext(
        candidate=candidate,
        evaluator=static_evaluator or _evaluator_facade(),
        rows=tuple(source_rows),
        evaluation_dataset={"public_only": True, "examples": 16},
        training_lineage={
            "status": "provided_and_validated",
            "run": {"merged": {"digest": validator.EXPECTED_LINEAGE_DIGESTS["merged_tree"]}},
        },
        runtime=runtime,
        artifact=artifact,
        configuration=configuration,
        configuration_digest=_digest(validator._canonical_json_bytes(configuration)),
    )


def _summary(
    rows: list[dict[str, object]],
    results_raw: bytes,
    context: validator.ValidationContext,
    *,
    started_at_unix_ns: int = 1_000_000,
) -> dict[str, object]:
    finished_at_unix_ns = started_at_unix_ns + 1_000_000
    return {
        "schema": evaluator.SCHEMA,
        "status": "complete",
        "track": validator.TRACK,
        "hardware_class": validator.HARDWARE_CLASS,
        "base_model": validator.BASE_MODEL,
        "quality_claim": evaluator.QUALITY_CLAIM,
        "runtime_claim": evaluator.RUNTIME_CLAIM,
        "lineage_claim": evaluator.LINEAGE_CLAIM,
        "safety_contract": copy.deepcopy(validator.SAFETY_CONTRACT),
        "configuration": copy.deepcopy(context.configuration),
        "configuration_digest": context.configuration_digest,
        "artifact": copy.deepcopy(context.artifact),
        "evaluation_dataset": copy.deepcopy(context.evaluation_dataset),
        "training_lineage": copy.deepcopy(context.training_lineage),
        "runtime": copy.deepcopy(context.runtime.identity),
        "timing": {
            "started_at_unix_ns": started_at_unix_ns,
            "finished_at_unix_ns": finished_at_unix_ns,
            "elapsed_ms": 1.0,
            "model_load_wall_ms": 10.0,
            "model_load_cpu_ms": 5.0,
        },
        "memory": {
            "initial": {"current_bytes": 100, "peak_bytes": 200},
            "model_loaded": {"current_bytes": 300, "peak_bytes": 400},
            "model_unloaded": {"current_bytes": 150, "peak_bytes": 400},
        },
        "results": {
            "file": "results.jsonl",
            "bytes": len(results_raw),
            "digest": _digest(results_raw),
            **evaluator.summarize_results(rows),
        },
    }


def _write_receipt(
    root: Path,
    rows: list[dict[str, object]],
    context: validator.ValidationContext,
    *,
    summary_rows: list[dict[str, object]] | None = None,
    started_at_unix_ns: int = 1_000_000,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    results_raw = b"".join(candidate.canonical_json_bytes(row) + b"\n" for row in rows)
    (root / "results.jsonl").write_bytes(results_raw)
    payload = _summary(
        summary_rows or rows,
        results_raw,
        context,
        started_at_unix_ns=started_at_unix_ns,
    )
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


class ReceiptFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sentinel = root.parent / "must-not-be-created"
        self.sources = [_source_row(index, self.sentinel) for index in range(16)]
        self.rows = [_result_row(source) for source in self.sources]
        self.evaluator = _evaluator_facade()
        self.context = _context(self.sources, static_evaluator=self.evaluator)

    def write(self, rows: list[dict[str, object]] | None = None) -> dict[str, object]:
        return _write_receipt(self.root, rows or self.rows, self.context)

    def validate(self, repeat: str = "r1") -> dict[str, object]:
        return validator._validate_repeat(
            repeat,
            self.root,
            context=self.context,
            gates=validator.EXPECTED_GATES,
        )


class BoundaryReceiptTests(unittest.TestCase):
    def test_exact_16_row_hard_boundaries_pass_without_executing_poison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            fixture.write()

            receipt = fixture.validate()

            self.assertEqual(receipt["gates"]["examples"], 16)
            self.assertEqual(receipt["gates"]["successful_generations"], 16)
            self.assertEqual(receipt["gates"]["failed_generations"], 0)
            self.assertEqual(receipt["gates"]["maximum_request_latency_ms"], 31_200.0)
            self.assertEqual(receipt["gates"]["p95_ttft_ms"], 5_256.0)
            self.assertEqual(receipt["gates"]["peak_rss_bytes"], 1_073_741_824)
            self.assertEqual(receipt["gates"]["maximum_stream_pieces_per_request"], 373)
            self.assertFalse(receipt["gates"]["preferred_p95_ttft_met"])
            self.assertEqual(
                evaluator.summarize_results(fixture.rows)["output"][
                    "engine_reported_stream_pieces"
                ],
                16 * 373,
            )
            self.assertEqual(fixture.evaluator.structural_diagnostics.call_count, 16)
            self.assertFalse(fixture.sentinel.exists())

    def test_ast_fields_are_recomputed_and_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            rows = copy.deepcopy(fixture.rows)
            rows[0]["scorer_extracted_top_level_task_func"] = False
            fixture.write(rows)

            with self.assertRaisesRegex(validator.ValidationRefused, "structural diagnostics"):
                fixture.validate()
            self.assertFalse(fixture.sentinel.exists())

    def test_linear_p95_is_recomputed_and_preferred_threshold_is_not_hard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            rows = copy.deepcopy(fixture.rows)
            for row, value in zip(rows, [0.0] * 14 + [5_000.0, 6_000.0], strict=True):
                row["ttft_ms"] = value
            fixture.write(rows)

            receipt = fixture.validate()

            self.assertEqual(receipt["gates"]["p95_ttft_ms"], 5_250.0)
            self.assertFalse(receipt["gates"]["preferred_p95_ttft_met"])


class StrictReceiptTests(unittest.TestCase):
    def test_strict_json_rejects_duplicates_nonfinite_values_and_bad_utf8(self) -> None:
        cases = (
            (b'{"a":1,"a":2}', "repeats JSON key"),
            (b'{"a":NaN}', "non-finite"),
            (b'{"a":1e10000}', "non-finite"),
            (b'"\\ud800"', "Unicode surrogate"),
            (b'"\xff"', "strict UTF-8 JSON"),
        )
        for raw, message in cases:
            with (
                self.subTest(raw=raw),
                self.assertRaisesRegex(validator.ValidationRefused, message),
            ):
                validator._strict_json(raw, "test JSON")

        with self.assertRaisesRegex(validator.ValidationRefused, "finite"):
            validator._number(10**10_000, "huge integer")

    def test_results_require_canonical_jsonl_and_exact_framing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            fixture.write()
            path = fixture.root / "results.jsonl"
            canonical = path.read_bytes()
            lines = canonical.splitlines()
            first = json.loads(lines[0])
            cases = {
                "noncanonical": (
                    json.dumps(first, sort_keys=True).encode("utf-8")
                    + b"\n"
                    + b"\n".join(lines[1:])
                    + b"\n"
                ),
                "missing-final-newline": canonical[:-1],
                "extra-blank-line": canonical + b"\n",
                "fifteen-lines": b"\n".join(lines[:-1]) + b"\n",
            }
            for label, malformed in cases.items():
                with self.subTest(label=label):
                    path.write_bytes(malformed)
                    with self.assertRaises(validator.ValidationRefused):
                        fixture.validate()
                    path.write_bytes(canonical)

    def test_result_schema_binding_and_digest_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            mutations = (
                ("extra-field", lambda row: row.update({"unexpected": True})),
                ("raw-digest", lambda row: row.update({"raw_output_digest": "sha256:" + "0" * 64})),
                ("prompt-digest", lambda row: row.update({"prompt_digest": "sha256:" + "1" * 64})),
                ("ref", lambda row: row.update({"ref": "wrong-ref"})),
                ("boolean-type", lambda row: row.update({"raw_parseable_python": 1})),
                ("raw-byte-count-bool", lambda row: row.update({"raw_output_utf8_bytes": True})),
                (
                    "extracted-byte-count-bool",
                    lambda row: row.update({"scorer_extracted_utf8_bytes": True}),
                ),
                (
                    "stream-piece-count-bool",
                    lambda row: row.update({"engine_reported_output_pieces": True}),
                ),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    rows = copy.deepcopy(fixture.rows)
                    mutate(rows[0])
                    if frozenset(rows[0]) == evaluator.RESULT_KEYS:
                        fixture.write(rows)
                    else:
                        fixture.write()
                        raw_lines = (fixture.root / "results.jsonl").read_bytes().splitlines()
                        raw_lines[0] = candidate.canonical_json_bytes(rows[0])
                        (fixture.root / "results.jsonl").write_bytes(b"\n".join(raw_lines) + b"\n")
                    with self.assertRaises(validator.ValidationRefused):
                        fixture.validate()

    def test_summary_schema_canonicalization_and_nested_bindings_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            original = fixture.write()
            summary_path = fixture.root / "summary.json"
            cases: list[tuple[str, dict[str, object], bool]] = []

            extra = copy.deepcopy(original)
            extra["unexpected"] = True
            cases.append(("schema", extra, True))

            safety = copy.deepcopy(original)
            safety["safety_contract"]["generated_code_executed"] = True  # type: ignore[index]
            cases.append(("safety", safety, True))

            artifact = copy.deepcopy(original)
            artifact["artifact"]["tree_digest"] = "sha256:" + "0" * 64  # type: ignore[index]
            cases.append(("artifact", artifact, True))

            results = copy.deepcopy(original)
            results["results"]["digest"] = "sha256:" + "0" * 64  # type: ignore[index]
            cases.append(("results", results, True))

            result_bytes = copy.deepcopy(original)
            result_bytes["results"]["bytes"] = True  # type: ignore[index]
            cases.append(("result-bytes-bool", result_bytes, True))

            for label, payload, canonical in cases:
                with self.subTest(label=label):
                    text = json.dumps(payload, indent=2, sort_keys=True)
                    summary_path.write_text(text + ("\n" if canonical else ""), encoding="utf-8")
                    with self.assertRaises(validator.ValidationRefused):
                        fixture.validate()

            fixture.write()
            summary_path.write_text(json.dumps(original, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationRefused, "canonical JSON"):
                fixture.validate()

    def test_tree_reader_refuses_extra_entries_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            fixture.write()
            (fixture.root / "extra").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationRefused, "entries changed"):
                fixture.validate()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("{}", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(validator.ValidationRefused, "regular non-symlink"):
                validator._stable_regular_bytes(link, "link", maximum=100)


class HardGateTests(unittest.TestCase):
    def test_every_hard_gate_rejects_one_step_beyond_its_limit(self) -> None:
        rows = [
            _result_row(
                _source_row(index, Path("/must-not-exist")),
                pieces=373,
                ttft_ms=5_256.0,
                engine_total_ms=31_200.0,
                peak_rss_bytes=1_073_741_824,
            )
            for index in range(16)
        ]
        results = evaluator.summarize_results(rows)
        metrics = validator._gate_metrics(rows, results)
        cases = (
            ("examples", {"examples": 15}, {}, "example count"),
            (
                "successful",
                {"successful_generations": 15},
                {},
                "successful generation minimum",
            ),
            ("failed", {"failed_generations": 1}, {}, "failed generation maximum"),
            ("failed-refs", {}, {"failed_refs": ["ref"]}, "failed refs"),
            (
                "parseable",
                {"scorer_extracted_parseable_python": 15},
                {},
                "parseable Python minimum",
            ),
            (
                "task-func",
                {"scorer_extracted_top_level_task_func": 15},
                {},
                "top-level task_func minimum",
            ),
            (
                "fence",
                {"scorer_extracted_residual_fences": 1},
                {},
                "residual fence maximum",
            ),
            (
                "latency",
                {"maximum_request_latency_ms": 31_200.000_001},
                {},
                "latency maximum",
            ),
            ("p95", {"p95_ttft_ms": 5_256.000_001}, {}, "p95 TTFT maximum"),
            (
                "rss",
                {"peak_rss_bytes": 1_073_741_825},
                {},
                "peak RSS maximum",
            ),
            (
                "pieces",
                {"maximum_stream_pieces_per_request": 374},
                {},
                "stream-piece maximum",
            ),
            ("quality", {}, {"quality_score": 0.0}, "quality score"),
            ("execution", {}, {"execution_pass_at_1": 0.0}, "execution pass@1"),
        )
        for label, metric_changes, result_changes, message in cases:
            with self.subTest(label=label):
                changed_metrics = dict(metrics)
                changed_metrics.update(metric_changes)
                changed_results = copy.deepcopy(results)
                changed_results.update(result_changes)
                with self.assertRaisesRegex(validator.ValidationRefused, message):
                    validator._enforce_gates(
                        changed_metrics,
                        changed_results,
                        validator.EXPECTED_GATES,
                    )

    def test_stream_piece_gate_uses_per_row_max_not_summary_sum(self) -> None:
        rows = [
            _result_row(_source_row(index, Path("/must-not-exist")), pieces=373)
            for index in range(16)
        ]
        results = evaluator.summarize_results(rows)
        metrics = validator._gate_metrics(rows, results)

        self.assertEqual(results["output"]["engine_reported_stream_pieces"], 5_968)
        self.assertEqual(metrics["maximum_stream_pieces_per_request"], 373)
        validator._enforce_gates(metrics, results, validator.EXPECTED_GATES)


def _conversion_spec(bundle1: Path, bundle2: Path) -> validator.SpecBindings:
    return validator.SpecBindings(
        path=Path("spec.json"),
        raw=b"{}",
        payload={},
        output_roots=(Path("/r1"), Path("/r2"), Path("/r3")),
        bundles=(bundle1, bundle2),
        dataset=Path("/dataset"),
        diagnostic_jsonl=Path("/diagnostic.jsonl"),
        diagnostic_source=Path("/corpus.json"),
        training_arguments=(Path("/run"), Path("/train"), Path("/source"), Path("/base")),
        source_root=validator.SOURCE_ROOT,
        gates=dict(validator.EXPECTED_GATES),
    )


def _artifact_for(root: Path) -> dict[str, object]:
    return {
        "root": str(root / "artifact"),
        "tree_digest": validator.EXPECTED_ARTIFACT["tree_digest"],
        "entrypoint": {
            "bytes": validator.EXPECTED_ARTIFACT["entrypoint_bytes"],
            "digest": validator.EXPECTED_ARTIFACT["entrypoint_digest"],
        },
    }


class ConversionBindingTests(unittest.TestCase):
    def test_successful_v6_artifact_and_both_replay_receipts_are_exactly_pinned(self) -> None:
        self.assertEqual(
            validator.EXPECTED_ARTIFACT,
            {
                "tree_digest": (
                    "sha256:3f6dc72a0cd886c74a5161ccd42feda27de56e54c914f28961e7dd89ca2917b5"
                ),
                "entrypoint_bytes": 396_704_672,
                "entrypoint_digest": (
                    "sha256:3df33a173b16af2bca9a402c335bda5d39b03e290d4ba13f4eaf5ad5c4397d5e"
                ),
            },
        )
        self.assertEqual(
            validator.EXPECTED_REPLAY_FILES,
            {
                "replay1": {
                    "load_spec": (
                        257,
                        "sha256:bbd5d02a6cb8dfc0ac9f045e86d9bf827a8bbb02eacfade684fdaff4fa77eeef",
                    ),
                    "calibration_receipt": (
                        23_855,
                        "sha256:c2700289e1cf774f1738387006ae39ff9c5e8ef31c3dfeb78519d2679a6114c6",
                    ),
                    "conversion_receipt": (
                        18_668,
                        "sha256:4f737514479942d8eac74db4f720e4d181d091c537ebd11975765465a6baa940",
                    ),
                },
                "replay2": {
                    "load_spec": (
                        257,
                        "sha256:bbd5d02a6cb8dfc0ac9f045e86d9bf827a8bbb02eacfade684fdaff4fa77eeef",
                    ),
                    "calibration_receipt": (
                        23_855,
                        "sha256:f83eaf1f255921e6f3bc6dd70994998bebdf3b99c091f92319ec62a51db5e24d",
                    ),
                    "conversion_receipt": (
                        18_668,
                        "sha256:222dea46d5f62b6e1b7b3e9473d7ffbaa105aaa72828d3ca52ab8ffb2070853a",
                    ),
                },
            },
        )

    def test_external_replay_helper_pins_order_names_artifact_and_load_spec(self) -> None:
        first_root = Path("/declared/replay1")
        second_root = Path("/declared/replay2")
        spec = _conversion_spec(first_root, second_root)
        artifact1 = _artifact_for(first_root)
        artifact2 = _artifact_for(second_root)
        load_spec = {"format": "gguf", "quantization": "Q4_K_M"}
        side_effect = (
            (artifact1, load_spec, {"root": str(first_root)}),
            (artifact2, copy.deepcopy(load_spec), {"root": str(second_root)}),
        )
        tools = validator.Toolset(candidate=SimpleNamespace(), evaluator=SimpleNamespace())

        with mock.patch.object(
            validator, "_validate_conversion_bundle", side_effect=side_effect
        ) as validate_bundle:
            bindings = validator._validate_conversion_bundles(spec, tools)

        self.assertEqual(bindings.artifact, artifact1)
        self.assertEqual(bindings.load_manifest, load_spec)
        self.assertEqual(validate_bundle.call_args_list[0].kwargs["replay"], "replay1")
        self.assertEqual(validate_bundle.call_args_list[1].kwargs["replay"], "replay2")
        self.assertEqual(validate_bundle.call_args_list[0].args[0], first_root)
        self.assertEqual(validate_bundle.call_args_list[1].args[0], second_root)

    def test_external_replay_mismatch_or_wrong_primary_root_is_refused(self) -> None:
        first_root = Path("/declared/replay1")
        second_root = Path("/declared/replay2")
        spec = _conversion_spec(first_root, second_root)
        tools = validator.Toolset(candidate=SimpleNamespace(), evaluator=SimpleNamespace())
        base1 = _artifact_for(first_root)
        base2 = _artifact_for(second_root)
        load = {"format": "gguf"}
        cases = []
        changed_artifact = copy.deepcopy(base2)
        changed_artifact["tree_digest"] = "sha256:" + "0" * 64
        cases.append(("artifact", base1, changed_artifact, load, load))
        cases.append(("load", base1, base2, load, {"format": "safetensors"}))
        wrong_root = copy.deepcopy(base1)
        wrong_root["root"] = "/wrong/artifact"
        cases.append(("root", wrong_root, base2, load, load))
        for label, artifact1, artifact2, load1, load2 in cases:
            with (
                self.subTest(label=label),
                mock.patch.object(
                    validator,
                    "_validate_conversion_bundle",
                    side_effect=(
                        (artifact1, load1, {"root": str(first_root)}),
                        (artifact2, load2, {"root": str(second_root)}),
                    ),
                ),
                self.assertRaises(validator.ValidationRefused),
            ):
                validator._validate_conversion_bundles(spec, tools)

    def test_receipt_pin_helper_checks_both_exact_size_and_digest(self) -> None:
        raw = b"immutable receipt"
        expected = (len(raw), _digest(raw))
        validator._require_expected_file(raw, expected, "receipt")
        for changed in ((len(raw) + 1, expected[1]), (len(raw), "sha256:" + "0" * 64)):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(validator.ValidationRefused, "bytes or digest"),
            ):
                validator._require_expected_file(raw, changed, "receipt")

    def test_unsupported_replay_name_fails_before_any_artifact_access(self) -> None:
        with self.assertRaisesRegex(validator.ValidationRefused, "unsupported conversion replay"):
            validator._validate_conversion_bundle(
                Path("/does/not/exist"),
                replay="replay3",
                evaluator=SimpleNamespace(),
            )


def _public_lineage() -> dict[str, object]:
    return {
        "prepared_dataset": {
            "manifest": {"digest": validator.EXPECTED_LINEAGE_DIGESTS["manifest"]},
            "holdout": {"digest": validator.EXPECTED_LINEAGE_DIGESTS["holdout"]},
            "manifest_payload": {"seed": 92, "train_examples": 78, "holdout_examples": 16},
        },
        "source_corpus": {"file": {"digest": validator.EXPECTED_LINEAGE_DIGESTS["source"]}},
        "diagnostic_jsonl": {
            "refs_digest": validator.EXPECTED_LINEAGE_DIGESTS["refs"],
            "examples": 16,
        },
        "public_only": True,
        "hidden_or_scored_tests_accessed": False,
    }


def _training_lineage() -> dict[str, object]:
    return {
        "status": "provided_and_validated",
        "schema": "microtensor.code.training.v5",
        "receipt": {"digest": validator.EXPECTED_LINEAGE_DIGESTS["training_metadata"]},
        "run": {"merged": {"digest": validator.EXPECTED_LINEAGE_DIGESTS["merged_tree"]}},
    }


class _EngineConstructionPoison:
    calls = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        type(self).calls += 1
        raise AssertionError("the validator constructed a model engine")


class _Manifest:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


class StaticContextTests(unittest.TestCase):
    def test_training_merged_digest_must_be_nested_under_run(self) -> None:
        validator._validate_training_lineage(_training_lineage())
        flattened = _training_lineage()
        flattened["run"] = {}
        flattened["merged"] = {  # type: ignore[assignment]
            "digest": validator.EXPECTED_LINEAGE_DIGESTS["merged_tree"]
        }
        with self.assertRaises(validator.ValidationRefused):
            validator._validate_training_lineage(flattened)

    def test_runtime_identity_binds_exact_interpreter_and_full_canonical_object(self) -> None:
        lexical = Path(sys.executable)
        resolved = lexical.resolve(strict=True)
        digest = "sha256:" + "1" * 64
        identity = {
            "python": {
                "version": validator.EXPECTED_PYTHON_VERSION,
                "executable": {
                    "path": str(resolved),
                    "bytes": 123,
                    "digest": digest,
                },
            },
            "microtensor": {
                "release_version": "0.3.0",
                "mechanism_version": "0.3.0",
            },
        }
        spec = SimpleNamespace(
            payload={
                "diagnostic": {
                    "signed_runtime": {
                        "python_path": str(lexical),
                        "python_resolved_path": str(resolved),
                        "python_size_bytes": 123,
                        "python_sha256": digest,
                        "python_version": "3.12.3",
                    }
                }
            }
        )
        raw = validator._canonical_json_bytes(identity)
        with (
            mock.patch.object(validator, "EXPECTED_RUNTIME_IDENTITY_BYTES", len(raw)),
            mock.patch.object(
                validator,
                "EXPECTED_RUNTIME_IDENTITY_DIGEST",
                validator._digest_bytes(raw),
            ),
        ):
            validator._validate_runtime_identity(identity, spec)
            changed = copy.deepcopy(identity)
            changed["microtensor"]["release_version"] = "0.3.1"
            with self.assertRaisesRegex(validator.ValidationRefused, "Microtensor"):
                validator._validate_runtime_identity(changed, spec)

    def test_source_import_isolated_before_any_pinned_module_executes(self) -> None:
        with (
            mock.patch.object(validator, "_validate_clean_source_root"),
            self.assertRaisesRegex(validator.ValidationRefused, "preloaded"),
        ):
            validator._load_pinned_tools(validator.SOURCE_ROOT)

        clean = (
            (validator.SOURCE_COMMIT + "\n").encode("ascii"),
            b"",
            b"",
        )
        with mock.patch.object(validator, "_git_output", side_effect=clean):
            validator._validate_clean_source_root(validator.SOURCE_ROOT)
        dirty = (clean[0], b"?? training/shadow.so\0")
        with (
            mock.patch.object(validator, "_git_output", side_effect=dirty),
            self.assertRaisesRegex(validator.ValidationRefused, "not clean"),
        ):
            validator._validate_clean_source_root(validator.SOURCE_ROOT)

    def test_prepare_context_never_constructs_fake_runtime_engine(self) -> None:
        _EngineConstructionPoison.calls = 0
        rows = [
            {
                "ref": f"ref-{index}",
                "prompt": f"prompt-{index}",
                "completion": f"def task_func():\n    return {index}\n",
            }
            for index in range(16)
        ]
        load_manifest = {
            "format": "gguf",
            "quantization": validator.QUANTIZATION,
            "entrypoint": validator.ENTRYPOINT,
            "max_input": {"tokens": validator.MAX_INPUT_TOKENS},
            "preprocessing": {"tokenizer": "tokenizer.json"},
            "base_model": validator.BASE_MODEL,
        }
        runtime = SimpleNamespace(
            engine_type=_EngineConstructionPoison,
            artifact_format=SimpleNamespace(GGUF="gguf"),
            load_manifest_type=lambda **kwargs: _Manifest(kwargs),
            identity={
                "python": {
                    "version": "3.12.3 signed",
                    "executable": {
                        "path": "/usr/bin/python3.12",
                        "bytes": 8_016_832,
                        "digest": "sha256:" + "1" * 64,
                    },
                },
                "microtensor": {
                    "release_version": "0.3.0",
                    "mechanism_version": "0.3.0",
                },
            },
        )
        fake_evaluator = SimpleNamespace(
            load_public_diagnostic=mock.Mock(return_value=(rows, _public_lineage())),
            load_v5_training_lineage=mock.Mock(return_value=(_training_lineage(), ())),
            load_signed_runtime=mock.Mock(return_value=runtime),
            generation_contract=mock.Mock(return_value={"static": True}),
        )
        spec = validator.SpecBindings(
            path=Path("spec.json"),
            raw=b"{}",
            payload={
                "diagnostic": {
                    "signed_runtime": {
                        "python_resolved_path": "/usr/bin/python3.12",
                        "python_size_bytes": 8_016_832,
                        "python_sha256": "sha256:" + "1" * 64,
                        "python_version": "3.12.3",
                    }
                }
            },
            output_roots=(Path("/r1"), Path("/r2"), Path("/r3")),
            bundles=(Path("/b1"), Path("/b2")),
            dataset=Path("/dataset"),
            diagnostic_jsonl=Path("/diagnostic"),
            diagnostic_source=Path("/source"),
            training_arguments=(Path("/run"), Path("/train"), Path("/corpus"), Path("/base")),
            source_root=validator.SOURCE_ROOT,
            gates=dict(validator.EXPECTED_GATES),
        )
        conversion = validator.ConversionBindings(
            artifact={"tree_digest": validator.EXPECTED_ARTIFACT["tree_digest"]},
            load_manifest=load_manifest,
            replay_receipts=({}, {}),
        )
        tools = validator.Toolset(candidate=SimpleNamespace(), evaluator=fake_evaluator)

        with mock.patch.object(validator, "_validate_runtime_identity") as validate_runtime:
            context = validator._prepare_context(spec, tools, conversion)

        self.assertEqual(context.configuration["load_manifest"], load_manifest)
        self.assertEqual(context.training_lineage, _training_lineage())
        self.assertEqual(_EngineConstructionPoison.calls, 0)
        validate_runtime.assert_called_once_with(runtime.identity, spec)


class CrossRepeatTests(unittest.TestCase):
    def test_cross_repeat_output_binding_and_timing_drift_fail_closed(self) -> None:
        base = {
            "bindings": {"artifact": "same", "configuration": "same"},
            "gates": {
                "successful_generations": 16,
                "failed_generations": 0,
                "scorer_extracted_parseable_python": 16,
                "scorer_extracted_top_level_task_func": 16,
                "scorer_extracted_residual_fences": 0,
                "maximum_request_latency_ms": 100.0,
                "p95_ttft_ms": 4_701.0,
                "peak_rss_bytes": 1_000,
                "maximum_stream_pieces_per_request": 10,
                "preferred_p95_ttft_met": False,
            },
            "raw_output_digests": ["a"],
            "timing": {"started_at_unix_ns": 1, "finished_at_unix_ns": 2},
        }
        second = copy.deepcopy(base)
        second["timing"] = {"started_at_unix_ns": 2, "finished_at_unix_ns": 3}
        aggregate = validator._aggregate([base, second])
        self.assertTrue(aggregate["raw_outputs_identical_across_repeats"])
        self.assertFalse(aggregate["preferred_p95_ttft_met_on_every_repeat"])
        self.assertTrue(aggregate["validated_repeat_hard_gates_passed"])
        self.assertFalse(aggregate["all_declared_local_gates_passed"])

        changed_output = copy.deepcopy(second)
        changed_output["raw_output_digests"] = ["b"]
        with self.assertRaisesRegex(validator.ValidationRefused, "raw output"):
            validator._aggregate([base, changed_output])

        changed = copy.deepcopy(second)
        changed["bindings"]["artifact"] = "different"
        with self.assertRaisesRegex(validator.ValidationRefused, "cross-repeat"):
            validator._aggregate([base, changed])
        overlapping = copy.deepcopy(second)
        overlapping["timing"] = {"started_at_unix_ns": 1, "finished_at_unix_ns": 2}
        with self.assertRaisesRegex(validator.ValidationRefused, "overlap"):
            validator._aggregate([base, overlapping])
        with self.assertRaisesRegex(validator.ValidationRefused, "zero"):
            validator._aggregate([])

    def test_full_three_repeat_validation_requires_exact_outputs_and_ordered_times(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            roots = tuple(parent / repeat for repeat in validator.REPEATS)
            fixture = ReceiptFixture(roots[0])
            rows_by_repeat = [copy.deepcopy(fixture.rows) for _ in validator.REPEATS]
            for index, (root, rows) in enumerate(zip(roots, rows_by_repeat, strict=True)):
                _write_receipt(
                    root,
                    rows,
                    fixture.context,
                    started_at_unix_ns=1_000_000 + index * 2_000_000,
                )
            spec = validator.SpecBindings(
                path=Path("spec.json"),
                raw=b"immutable spec",
                payload={},
                output_roots=roots,
                bundles=(Path("/b1"), Path("/b2")),
                dataset=Path("/dataset"),
                diagnostic_jsonl=Path("/diagnostic"),
                diagnostic_source=Path("/source"),
                training_arguments=(
                    Path("/run"),
                    Path("/training"),
                    Path("/corpus"),
                    Path("/base"),
                ),
                source_root=validator.SOURCE_ROOT,
                gates=dict(validator.EXPECTED_GATES),
            )
            conversion = validator.ConversionBindings(
                artifact=fixture.context.artifact,
                load_manifest={},
                replay_receipts=({"replay": 1}, {"replay": 2}),
            )
            tools = validator.Toolset(candidate=SimpleNamespace(), evaluator=SimpleNamespace())
            with (
                mock.patch.object(validator, "_load_spec", return_value=spec),
                mock.patch.object(
                    validator, "_validate_conversion_bundles", return_value=conversion
                ),
                mock.patch.object(validator, "_prepare_context", return_value=fixture.context),
            ):
                report = validator.validate_diagnostic(Path("ignored"), "r3", _tools=tools)

            self.assertEqual(report["aggregate"]["validated_repeats"], 3)
            self.assertTrue(report["aggregate"]["raw_outputs_identical_across_repeats"])
            self.assertTrue(report["aggregate"]["all_declared_local_gates_passed"])
            self.assertEqual(report["status"], "validated")
            self.assertEqual(report["claim"]["remaining_local_repeats"], [])
            self.assertEqual(len(report["repeats"]), 3)
            self.assertFalse(fixture.sentinel.exists())

    def test_r1_report_is_explicitly_partial_and_lists_remaining_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            roots = tuple(parent / repeat for repeat in validator.REPEATS)
            fixture = ReceiptFixture(roots[0])
            fixture.write()
            spec = validator.SpecBindings(
                path=Path("spec.json"),
                raw=b"immutable spec",
                payload={},
                output_roots=roots,
                bundles=(Path("/b1"), Path("/b2")),
                dataset=Path("/dataset"),
                diagnostic_jsonl=Path("/diagnostic"),
                diagnostic_source=Path("/source"),
                training_arguments=(
                    Path("/run"),
                    Path("/training"),
                    Path("/corpus"),
                    Path("/base"),
                ),
                source_root=validator.SOURCE_ROOT,
                gates=dict(validator.EXPECTED_GATES),
            )
            conversion = validator.ConversionBindings(
                artifact=fixture.context.artifact,
                load_manifest={},
                replay_receipts=({"replay": 1}, {"replay": 2}),
            )
            tools = validator.Toolset(candidate=SimpleNamespace(), evaluator=SimpleNamespace())
            with (
                mock.patch.object(validator, "_load_spec", return_value=spec),
                mock.patch.object(
                    validator, "_validate_conversion_bundles", return_value=conversion
                ),
                mock.patch.object(validator, "_prepare_context", return_value=fixture.context),
            ):
                report = validator.validate_diagnostic(Path("ignored"), "r1", _tools=tools)

            self.assertEqual(report["status"], "partially_validated")
            self.assertEqual(report["claim"]["remaining_local_repeats"], ["r2", "r3"])
            self.assertFalse(report["aggregate"]["all_declared_local_gates_passed"])
            self.assertTrue(report["aggregate"]["validated_repeat_hard_gates_passed"])

    def test_future_root_and_partial_staging_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            roots = tuple(parent / repeat for repeat in validator.REPEATS)
            fixture = ReceiptFixture(roots[0])
            fixture.write()
            roots[1].mkdir()
            spec = validator.SpecBindings(
                path=Path("spec.json"),
                raw=b"immutable spec",
                payload={},
                output_roots=roots,
                bundles=(Path("/b1"), Path("/b2")),
                dataset=Path("/dataset"),
                diagnostic_jsonl=Path("/diagnostic"),
                diagnostic_source=Path("/source"),
                training_arguments=(
                    Path("/run"),
                    Path("/training"),
                    Path("/corpus"),
                    Path("/base"),
                ),
                source_root=validator.SOURCE_ROOT,
                gates=dict(validator.EXPECTED_GATES),
            )
            conversion = validator.ConversionBindings(
                artifact=fixture.context.artifact,
                load_manifest={},
                replay_receipts=({"replay": 1}, {"replay": 2}),
            )
            tools = validator.Toolset(candidate=SimpleNamespace(), evaluator=SimpleNamespace())
            with (
                self.assertRaisesRegex(validator.ValidationRefused, "future diagnostic root"),
                mock.patch.object(validator, "_load_spec", return_value=spec),
                mock.patch.object(
                    validator, "_validate_conversion_bundles", return_value=conversion
                ),
                mock.patch.object(validator, "_prepare_context", return_value=fixture.context),
            ):
                validator.validate_diagnostic(Path("ignored"), "r1", _tools=tools)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            roots = tuple(parent / repeat for repeat in validator.REPEATS)
            (parent / ".r1.partial").mkdir()
            with self.assertRaisesRegex(validator.ValidationRefused, "partial staging"):
                validator._require_no_staging(roots)


class NormalizedV7ContractTests(unittest.TestCase):
    def test_spec_construction_is_deterministic_and_nonfinal_or_schema_swap_refuses(self) -> None:
        first = _v7_spec_payload()
        self.assertEqual(
            validator._canonical_json_bytes(first),
            validator._canonical_json_bytes(_v7_spec_payload()),
        )
        self.assertNotIn(b"placeholder", validator._canonical_json_bytes(first).lower())
        expected_policy = validator._normalized_artifact_use_policy(
            validator.NORMALIZED_CONVERSION_SCHEMA
        )
        self.assertEqual(first["artifact_use_policy"], expected_policy)
        self.assertEqual(expected_policy["intended_use"], "local_quality_isolation_only")
        self.assertIs(expected_policy["conversion_runtime_closure_attested"], False)
        self.assertIs(expected_policy["publication_eligible"], False)
        self.assertIs(expected_policy["submission_eligible"], False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            loaded = _write_v7_spec(path, first)
            self.assertEqual(loaded.conversion_schema, "microtensor.code.gguf-conversion.v4")
            for field, value in (
                ("status", "awaiting-training"),
                ("schema", "microtensor.code.gguf-diagnostic-experiment.v1"),
            ):
                tampered = copy.deepcopy(first)
                tampered[field] = value
                path.write_text(
                    json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(validator.ValidationRefused):
                    validator._load_normalized_v7_spec(path)
            for field, value in (
                ("publication_eligible", True),
                ("conversion_runtime_closure_attested", 0),
                ("historical_conversion_path", "/attacker"),
            ):
                tampered = copy.deepcopy(first)
                policy = tampered["artifact_use_policy"]
                assert isinstance(policy, dict)
                policy[field] = value
                path.write_text(
                    json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with (
                    self.subTest(policy_field=field),
                    self.assertRaises(validator.ValidationRefused),
                ):
                    validator._load_normalized_v7_spec(path)
            missing = copy.deepcopy(first)
            missing.pop("artifact_use_policy")
            path.write_text(
                json.dumps(missing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(validator.ValidationRefused):
                validator._load_normalized_v7_spec(path)
        with self.assertRaisesRegex(validator.ValidationRefused, "only the generic v4"):
            _v7_spec_payload("microtensor.code.gguf-conversion.v3")
        with self.assertRaisesRegex(validator.ValidationRefused, "only the generic v4"):
            _v7_spec_payload("microtensor.code.gguf-conversion.v5")

    def test_source_commit_namespace_and_git_execution_closure_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            training_root = source_root / "training"
            training_root.mkdir()
            source_files: dict[str, dict[str, object]] = {}
            source_raw: dict[str, bytes] = {}
            for relative in validator.NORMALIZED_REQUIRED_SOURCE_FILES:
                raw = (relative + "\n").encode()
                target = source_root / relative
                target.write_bytes(raw)
                source_raw[relative] = raw
                source_files[relative] = {"bytes": len(raw), "digest": _digest(raw)}
            with tempfile.TemporaryDirectory() as spec_temporary:
                declared = _write_v7_spec(
                    Path(spec_temporary) / "spec.json",
                    _v7_spec_payload(),
                )
            spec = replace(
                declared,
                source_root=source_root,
                source_commit="a" * 40,
                source_files=source_files,
            )

            def git_result(
                _root: Path,
                arguments: object,
                _label: str,
                *,
                bad_origin: bool = False,
                bad_blob: bool = False,
                ignored: bool = False,
            ) -> bytes:
                command = tuple(arguments)  # type: ignore[arg-type]
                if command == ("remote", "get-url", "origin"):
                    return (
                        b"https://attacker.invalid/repo\n"
                        if bad_origin
                        else b"https://github.com/vandungtech/mt92\n"
                    )
                if command[:2] == ("cat-file", "-t"):
                    return b"commit\n"
                if command[:3] == ("rev-parse", "--verify", "HEAD"):
                    return ("a" * 40 + "\n").encode()
                if command[:2] == ("rev-parse", "--verify"):
                    return ("b" * 40 + "\n").encode()
                if command[:2] == ("merge-base", "--is-ancestor"):
                    return b""
                if command == ("rev-parse", "--is-shallow-repository"):
                    return b"false\n"
                if command == ("replace", "-l"):
                    return b""
                if command[:2] == ("rev-parse", "--git-path"):
                    return b"/tmp/no-normalized-grafts\n"
                if command[:2] == ("status", "--porcelain=v1"):
                    return (
                        b"!! microtensor.py\0"
                        if ignored and "--ignored=matching" in command
                        else b""
                    )
                if command[:2] == ("ls-files", "-v"):
                    return f"H {command[-1]}\n".encode()
                if command[0] == "show":
                    relative = str(command[1]).split(":", 1)[1]
                    return b"changed\n" if bad_blob else source_raw[relative]
                raise AssertionError(f"unexpected Git command: {command}")

            with (
                mock.patch.object(validator, "_git_output", side_effect=git_result),
                self.assertRaisesRegex(validator.ValidationRefused, "preloaded"),
            ):
                validator._load_normalized_v7_tools(spec)
            for label, options, message in (
                ("origin", {"bad_origin": True}, "authorized repository"),
                ("commit blob", {"bad_blob": True}, "commit blob"),
                ("ignored root shadow", {"ignored": True}, "import closure"),
            ):
                with (
                    self.subTest(label=label),
                    mock.patch.object(
                        validator,
                        "_git_output",
                        side_effect=lambda root, args, git_label, options=options: git_result(
                            root, args, git_label, **options
                        ),
                    ),
                    self.assertRaisesRegex(validator.ValidationRefused, message),
                ):
                    validator._load_normalized_v7_tools(spec)

            (training_root / "__init__.py").write_text("raise AssertionError\n", encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationRefused, "initializer"):
                validator._require_namespace_training_package(source_root)
            (training_root / "__init__.py").unlink()
            (training_root / "code_candidate").mkdir()
            with self.assertRaisesRegex(validator.ValidationRefused, "shadow"):
                validator._require_no_normalized_import_shadows(source_root)
            self.assertTrue(validator._child_import_competitor(source_root))

        completed = SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        with mock.patch.object(validator.subprocess, "run", return_value=completed) as run:
            validator._git_output(Path("/tmp/source"), ("status",), "synthetic Git")
        argv = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertIn("core.fsmonitor=false", argv)
        self.assertIn("core.hooksPath=/dev/null", argv)
        self.assertEqual(environment["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(environment["GIT_NO_LAZY_FETCH"], "1")

    def test_normalized_validation_report_repeats_permanent_local_only_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            spec = _write_v7_spec(parent / "spec.json", _v7_spec_payload())
            spec = replace(
                spec,
                output_roots=tuple(parent / repeat for repeat in validator.REPEATS),
            )
            conversion = validator.ConversionBindings(
                artifact={
                    "tree_digest": "sha256:" + "1" * 64,
                    "entrypoint": {"digest": "sha256:" + "2" * 64},
                },
                load_manifest={},
                replay_receipts=({"schema": validator.NORMALIZED_CONVERSION_SCHEMA},),
            )
            with (
                mock.patch.object(validator, "_load_normalized_v7_spec", return_value=spec),
                mock.patch.object(
                    validator,
                    "_validate_normalized_conversion_bundle",
                    return_value=conversion,
                ),
                mock.patch.object(validator, "_prepare_normalized_context", return_value=object()),
                mock.patch.object(
                    validator,
                    "_validate_repeat",
                    return_value={"raw_output_digests": []},
                ),
                mock.patch.object(
                    validator,
                    "_aggregate",
                    return_value={
                        "validated_repeat_hard_gates_passed": True,
                        "all_declared_local_gates_passed": False,
                    },
                ),
            ):
                report = validator.validate_normalized_v7_diagnostic(
                    spec.path,
                    "r1",
                    _tools=validator.Toolset(
                        candidate=SimpleNamespace(), evaluator=SimpleNamespace()
                    ),
                )
        claim = report["claim"]
        self.assertEqual(
            claim["artifact_use_policy"],
            validator._normalized_artifact_use_policy(validator.NORMALIZED_CONVERSION_SCHEMA),
        )
        self.assertNotIn("authorized immutable publication", claim["remaining_external_gates"])

    def test_completed_v6_lineage_binds_receipt_dataset_exclusions_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = _write_v7_spec(Path(temporary) / "spec.json", _v7_spec_payload())
        lineage: dict[str, object] = {
            "status": "provided_and_validated",
            "schema": validator.NORMALIZED_TRAINING_SCHEMA,
            "receipt": {**spec.training_receipt, "path": "/training_metadata.json"},
            "prepared_dataset": {
                field: {**identity, "path": f"/{field}"}
                for field, identity in validator.NORMALIZED_DATASET_FILES.items()
            },
            "source_corpus": {
                "file": {
                    "bytes": validator.NORMALIZED_SOURCE_CORPUS_IDENTITY["bytes"],
                    "digest": validator.NORMALIZED_SOURCE_CORPUS_IDENTITY["digest"],
                },
                "canonical_digest": validator.NORMALIZED_SOURCE_CORPUS_IDENTITY["canonical_digest"],
            },
            "base_snapshot": {
                "base_model": validator.BASE_MODEL,
                "files": {
                    "tokenizer.json": {
                        "bytes": validator.NORMALIZED_TOKENIZER_IDENTITY["bytes"],
                        "sha256": validator.NORMALIZED_TOKENIZER_IDENTITY["digest"].removeprefix(
                            "sha256:"
                        ),
                    }
                },
            },
            "run": {"kind": "merged", "merged": {"digest": spec.merged_tree_digest}},
        }
        manifest = {
            "schema": validator.NORMALIZED_DATASET_SCHEMA,
            "corpus_profile": validator.NORMALIZED_CORPUS_PROFILE,
            "seed": 92,
            "source_examples": 8_000,
            "train_examples": 7_730,
            "holdout_examples": 0,
            "excluded_examples": 270,
            "excluded_refs_file": "excluded-refs.json",
            "excluded_refs_canonical_bytes": validator.NORMALIZED_DATASET_FILES["excluded_refs"][
                "bytes"
            ],
            "excluded_refs_digest": validator.NORMALIZED_DATASET_FILES["excluded_refs"]["digest"],
        }
        prepared = lineage["prepared_dataset"]
        assert isinstance(prepared, dict)
        prepared["manifest_payload"] = manifest
        validator._validate_normalized_training_lineage(lineage, spec)

        def receipt_tamper(item: dict[str, object]) -> None:
            receipt = item["receipt"]
            assert isinstance(receipt, dict)
            receipt["digest"] = "sha256:" + "f" * 64

        def exclusion_tamper(item: dict[str, object]) -> None:
            prepared_item = item["prepared_dataset"]
            assert isinstance(prepared_item, dict)
            excluded = prepared_item["excluded_refs"]
            assert isinstance(excluded, dict)
            excluded["bytes"] = 3_185

        def tokenizer_tamper(item: dict[str, object]) -> None:
            base = item["base_snapshot"]
            assert isinstance(base, dict)
            files = base["files"]
            assert isinstance(files, dict)
            tokenizer = files["tokenizer.json"]
            assert isinstance(tokenizer, dict)
            tokenizer["sha256"] = "f" * 64

        def merged_tamper(item: dict[str, object]) -> None:
            run = item["run"]
            assert isinstance(run, dict)
            merged = run["merged"]
            assert isinstance(merged, dict)
            merged["digest"] = "sha256:" + "e" * 64

        mutations = (
            ("running receipt", lambda item: item.update(status="running")),
            ("training receipt", receipt_tamper),
            ("excluded refs", exclusion_tamper),
            ("tokenizer", tokenizer_tamper),
            ("merged artifact lineage", merged_tamper),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                tampered = copy.deepcopy(lineage)
                mutate(tampered)
                with self.assertRaises(validator.ValidationRefused):
                    validator._validate_normalized_training_lineage(tampered, spec)

    def test_v4_bundle_is_static_and_source_schema_or_artifact_tamper_refuses(self) -> None:
        class StaticEvaluator:
            def __init__(self, artifact: dict[str, object]) -> None:
                self.artifact = artifact

            @property
            def engine_type(self) -> object:
                raise AssertionError("model engine must never be constructed")

            def artifact_identity(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return copy.deepcopy(self.artifact)

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            loaded = _write_v7_spec(parent / "spec.json", _v7_spec_payload())
            bundle = parent / "bundle"
            (bundle / "artifact").mkdir(parents=True)
            (bundle / "artifact" / "model.gguf").write_bytes(b"not-a-real-model")
            load_manifest = {
                "format": "gguf",
                "quantization": "Q4_K_M",
                "entrypoint": "model.gguf",
                "max_input": {"tokens": 541},
                "preprocessing": {"tokenizer": "tokenizer.json"},
                "base_model": validator.BASE_MODEL,
            }
            load_raw = json.dumps(load_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            (bundle / "load-spec.json").write_bytes(load_raw)
            artifact_contract = dict(loaded.artifact_contract)
            receipt = {
                "schema": "microtensor.code.gguf-conversion.v4",
                "status": "complete",
                "track": validator.TRACK,
                "hardware_class": validator.HARDWARE_CLASS,
                "base_model": validator.BASE_MODEL,
                "llama_cpp_revision": validator.LLAMA_CPP_REVISION,
                "source": {
                    "training_schema": validator.NORMALIZED_TRAINING_SCHEMA,
                    "dataset_schema": validator.NORMALIZED_DATASET_SCHEMA,
                    "corpus_profile": validator.NORMALIZED_CORPUS_PROFILE,
                    "training_metadata_digest": loaded.training_receipt["digest"],
                    "merged_tree_digest": loaded.merged_tree_digest,
                    "excluded_refs": dict(validator.NORMALIZED_DATASET_FILES["excluded_refs"]),
                },
                "conversion": {
                    "converter_digest": validator.NORMALIZED_CONVERTER_DIGEST,
                    "quantizer_digest": validator.NORMALIZED_QUANTIZER_DIGEST,
                    "commands": [
                        {
                            "name": "convert_f16",
                            "argv": [
                                str(validator.NORMALIZED_LLAMA_CPP_ROOT / "convert_hf_to_gguf.py"),
                                str(loaded.training_arguments[0] / "merged"),
                                "--outfile",
                                str(parent / ".microtensor-code-gguf-test/model-f16.gguf"),
                                "--outtype",
                                "f16",
                            ],
                            "returncode": 0,
                            "started_at_unix_ns": 1,
                            "finished_at_unix_ns": 2,
                        },
                        {
                            "name": "quantize",
                            "argv": [
                                str(
                                    validator.NORMALIZED_LLAMA_CPP_ROOT / "build/bin/llama-quantize"
                                ),
                                str(parent / ".microtensor-code-gguf-test/model-f16.gguf"),
                                str(parent / ".microtensor-code-gguf-test/artifact/model.gguf"),
                                validator.QUANTIZATION,
                            ],
                            "returncode": 0,
                            "started_at_unix_ns": 3,
                            "finished_at_unix_ns": 4,
                        },
                    ],
                },
                "artifact": {**artifact_contract, "quantization": validator.QUANTIZATION},
                "load_manifest": load_manifest,
                "calibration_receipt_digest": None,
            }

            def write_receipt(value: dict[str, object]) -> bytes:
                raw = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
                (bundle / "conversion-receipt.json").write_bytes(raw)
                return raw

            receipt_raw = write_receipt(receipt)
            spec = replace(
                loaded,
                bundle=bundle,
                load_spec={"bytes": len(load_raw), "digest": _digest(load_raw)},
                conversion_receipt={
                    "bytes": len(receipt_raw),
                    "digest": _digest(receipt_raw),
                },
            )
            artifact = {
                "root": str(bundle / "artifact"),
                "tree_digest": artifact_contract["tree_digest"],
                "entrypoint": {
                    "bytes": artifact_contract["entrypoint_bytes"],
                    "digest": artifact_contract["entrypoint_digest"],
                    "gguf": {"version": 3, "architecture": "qwen3", "file_type": 15},
                },
            }
            tools = validator.Toolset(
                candidate=SimpleNamespace(), evaluator=StaticEvaluator(artifact)
            )
            result = validator._validate_normalized_conversion_bundle(spec, tools)
            self.assertEqual(result.artifact["tree_digest"], artifact_contract["tree_digest"])

            def excluded_tamper(item: dict[str, object]) -> None:
                source = item["source"]
                assert isinstance(source, dict)
                excluded = source["excluded_refs"]
                assert isinstance(excluded, dict)
                excluded["bytes"] = 3_185

            for label, mutate in (
                ("excluded refs", excluded_tamper),
                (
                    "schema swap",
                    lambda item: item.update(schema="microtensor.code.gguf-conversion.v5"),
                ),
            ):
                with self.subTest(label=label):
                    tampered = copy.deepcopy(receipt)
                    mutate(tampered)
                    raw = write_receipt(tampered)
                    changed = replace(
                        spec,
                        conversion_receipt={"bytes": len(raw), "digest": _digest(raw)},
                    )
                    with self.assertRaises(validator.ValidationRefused):
                        validator._validate_normalized_conversion_bundle(changed, tools)
            receipt_raw = write_receipt(receipt)
            self.assertEqual(spec.conversion_receipt["digest"], _digest(receipt_raw))
            drifted_artifact = copy.deepcopy(artifact)
            drifted_artifact["tree_digest"] = "sha256:" + "f" * 64
            drifted_tools = validator.Toolset(
                candidate=SimpleNamespace(), evaluator=StaticEvaluator(drifted_artifact)
            )
            with self.assertRaisesRegex(validator.ValidationRefused, "candidate bundle"):
                validator._validate_normalized_conversion_bundle(spec, drifted_tools)

    def test_runtime_identity_is_recomputed_and_tamper_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = _write_v7_spec(Path(temporary) / "spec.json", _v7_spec_payload())
        identity = {
            "python": {
                "version": validator.EXPECTED_PYTHON_VERSION,
                "executable": {
                    "path": "/usr/bin/python3.12",
                    "bytes": 8_016_832,
                    "digest": (
                        "sha256:1319c137ea5d30f1d7599943cb0e72666648c20a94cf5932dd095364d07dafeb"
                    ),
                },
            },
            "microtensor": {"release_version": "0.3.0", "mechanism_version": "0.3.0"},
        }
        raw = validator._canonical_json_bytes(identity)
        spec = replace(
            spec,
            runtime_contract={
                **spec.runtime_contract,
                "identity": {"bytes": len(raw), "digest": _digest(raw)},
            },
        )
        with mock.patch.object(
            validator.sys,
            "executable",
            "/tmp/microtensor-v030-verify.5rMSRW/venv/bin/python",
        ):
            validator._validate_normalized_runtime_identity(identity, spec)
            tampered = copy.deepcopy(identity)
            microtensor = tampered["microtensor"]
            assert isinstance(microtensor, dict)
            microtensor["release_version"] = "0.3.1"
            with self.assertRaises(validator.ValidationRefused):
                validator._validate_normalized_runtime_identity(tampered, spec)


class Current94V8ContractTests(unittest.TestCase):
    def test_final_spec_is_deterministic_and_every_lineage_swap_refuses(self) -> None:
        payload = _current94_spec_payload()
        self.assertEqual(
            validator._canonical_json_bytes(payload),
            validator._canonical_json_bytes(_current94_spec_payload()),
        )
        self.assertEqual(payload["schema"], validator.CURRENT94_SPEC_SCHEMA)
        self.assertEqual(payload["candidate"]["gguf_architecture"], "qwen2")
        self.assertEqual(payload["conversion"]["schema"], validator.CURRENT94_CONVERSION_SCHEMA)
        self.assertEqual(
            payload["conversion"]["calibration_schema"],
            validator.CURRENT94_CALIBRATION_SCHEMA,
        )
        self.assertEqual(payload["runtime"]["release_version"], "0.3.2")
        self.assertEqual(payload["runtime"]["mechanism_version"], "0.3.0")
        self.assertEqual(payload["diagnostic"]["relationship_to_training"], "training_overlap")
        self.assertEqual(payload["training_lineage"]["run_kind"], "final_all_public")
        self.assertEqual(payload["training_lineage"]["train_examples"], 94)
        self.assertEqual(payload["training_lineage"]["holdout_examples"], 0)
        self.assertEqual(
            payload["safety_contract"], validator.CURRENT94_STATIC_VALIDATOR_SAFETY_CONTRACT
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            loaded = _write_current94_spec(path, payload)
            self.assertEqual(loaded.conversion_receipt, _v7_identity("5"))
            mutations = (
                ("status", lambda item: item.update(status="prospective")),
                (
                    "conversion schema",
                    lambda item: item["conversion"].update(
                        schema="microtensor.code.gguf-conversion.v5"
                    ),
                ),
                (
                    "calibration schema",
                    lambda item: item["conversion"].update(
                        calibration_schema="microtensor.code.imatrix-calibration.v2"
                    ),
                ),
                (
                    "architecture",
                    lambda item: item["candidate"].update(gguf_architecture="qwen3"),
                ),
                (
                    "release",
                    lambda item: item["runtime"].update(release_version="0.3.0"),
                ),
                (
                    "training split",
                    lambda item: item["training_lineage"].update(holdout_examples=16),
                ),
                (
                    "converter interpreter",
                    lambda item: item["conversion"]["runtime_receipt_content_binding"][
                        "converter_interpreter"
                    ].update(container_path="/attacker/python"),
                ),
                (
                    "execution claim",
                    lambda item: item["safety_contract"].update(
                        generated_code_executed=True
                    ),
                ),
            )
            for label, mutate in mutations:
                tampered = copy.deepcopy(payload)
                mutate(tampered)
                path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
                with self.subTest(label=label), self.assertRaises(validator.ValidationRefused):
                    validator._load_current94_v8_spec(path)

    def test_current94_v6_v3_full_filesystem_fixture_is_static_and_portable(self) -> None:
        class StaticEvaluator:
            def __init__(self, artifact: dict[str, object]) -> None:
                self.artifact = artifact
                self.SUPPORTED_QUANTIZATIONS = {validator.QUANTIZATION: 15}

            @property
            def engine_type(self) -> object:
                raise AssertionError("model engine must never be constructed")

            def file_identity(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("worker interpreter filesystem must not be inspected")

            def artifact_identity(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return copy.deepcopy(self.artifact)

        def runtime_closure_receipt() -> dict[str, object]:
            namespace_names: set[str] = set()
            for relative, _size, _digest_value in (
                real_converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
            ):
                namespace_names.add(Path(relative).name)
            for loader, target, _size, _digest_value in (
                real_converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT
            ):
                namespace_names.add(Path(loader).name)
                namespace_names.add(Path(target).name)
            for relative, target in real_converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT:
                namespace_names.add(Path(relative).name)
                namespace_names.add(target)
            return {
                "schema": real_converter.RUNTIME_LIBRARY_SCHEMA,
                "root": str(real_converter.LLAMA_CPP_ROOT),
                "directories": [
                    {"path": ".", "mode": "0755"},
                    {"path": "build", "mode": "0755"},
                    {"path": "build/bin", "mode": "0755"},
                ],
                "build_bin_namespace": [
                    f"build/bin/{name}" for name in sorted(namespace_names)
                ],
                "symlinks": [
                    {"path": relative, "target": target}
                    for relative, target in real_converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT
                ],
                "executables": [
                    {
                        "path": relative,
                        "bytes": size,
                        "digest": digest,
                        "mode": "0755",
                    }
                    for (
                        relative,
                        size,
                        digest,
                    ) in real_converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
                ],
                "libraries": [
                    {
                        "loader_path": loader,
                        "target_path": target,
                        "bytes": size,
                        "digest": digest,
                        "mode": "0755",
                    }
                    for (
                        loader,
                        target,
                        size,
                        digest,
                    ) in real_converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            loaded = _write_current94_spec(
                parent / "spec.json", _current94_spec_payload()
            )
            bundle = parent / "bundle"
            artifact_root = bundle / "artifact"
            artifact_root.mkdir(parents=True)
            artifact_raw = b"synthetic GGUF receipt fixture; never loaded"
            (artifact_root / validator.ENTRYPOINT).write_bytes(artifact_raw)
            artifact_contract = {
                "tree_digest": "sha256:" + "8" * 64,
                "entrypoint_bytes": len(artifact_raw),
                "entrypoint_digest": _digest(artifact_raw),
            }
            load_manifest = {
                "format": "gguf",
                "quantization": validator.QUANTIZATION,
                "entrypoint": validator.ENTRYPOINT,
                "max_input": {"tokens": validator.MAX_INPUT_TOKENS},
                "preprocessing": {"tokenizer": "tokenizer.json"},
                "base_model": validator.CURRENT94_BASE_MODEL,
            }
            load_raw = validator._pretty_json_bytes_for_current94(load_manifest)
            (bundle / "load-spec.json").write_bytes(load_raw)

            container_python = (
                "/.uv/python_install/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
            )
            portable_interpreter = {
                "path": container_python,
                "bytes": 21_333_768,
                "digest": (
                    "sha256:96d1b01675f2492922ec6f6ed8445791d2d3231ccae727cda521db30494b751e"
                ),
                "mode": "0o755",
            }
            interpreter_receipt = {
                "portable": portable_interpreter,
                "worker_observation": {
                    "device": 91,
                    "inode": 92,
                    "mtime_ns": 93,
                    "ctime_ns": 94,
                },
            }
            self.assertEqual(
                real_converter._converter_python_receipt_identity(
                    interpreter_receipt, "synthetic nested converter Python"
                ),
                interpreter_receipt,
            )
            runtime_closure = runtime_closure_receipt()
            closure_raw = validator._canonical_json_bytes(runtime_closure)
            llama_root = real_converter.LLAMA_CPP_ROOT
            command_argv = [
                [
                    container_python,
                    str(llama_root / "convert_hf_to_gguf.py"),
                ],
                [str(llama_root / "build/bin/llama-imatrix"), "--synthetic"],
                [str(llama_root / "build/bin/llama-quantize"), "--synthetic"],
            ]
            child_environment = real_converter._small_child_environment(single_thread=True)
            empty_digest = _digest(b"")

            def receipt_command(
                name: str,
                argv: list[str],
                *,
                cwd_role: str,
                started_at: int,
            ) -> dict[str, object]:
                stream = {
                    "bytes": 0,
                    "captured_bytes": 0,
                    "captured_digest": empty_digest,
                    "digest": empty_digest,
                    "truncated": False,
                }
                value: dict[str, object] = {
                    "name": name,
                    "argv": argv,
                    "cwd_role": cwd_role,
                    "environment": dict(child_environment),
                    "returncode": 0,
                    "started_at_unix_ns": started_at,
                    "finished_at_unix_ns": started_at + 1,
                    "stdout": copy.deepcopy(stream),
                    "stderr": copy.deepcopy(stream),
                }
                if name == "convert_f16":
                    value["launch"] = {
                        "method": "proc-self-fd",
                        "executed_object": copy.deepcopy(interpreter_receipt),
                    }
                return value

            names = ("convert_f16", "calibrate_imatrix", "quantize")
            commands = [
                receipt_command(
                    name,
                    argv,
                    cwd_role="private_staging",
                    started_at=index * 2 + 1,
                )
                for index, (name, argv) in enumerate(
                    zip(names, command_argv, strict=True)
                )
            ]
            replay_commands = [
                receipt_command(
                    name,
                    argv,
                    cwd_role="determinism_replay",
                    started_at=index * 2 + 11,
                )
                for index, (name, argv) in enumerate(
                    zip(names, command_argv, strict=True)
                )
            ]
            replay = {
                "schema": "microtensor.code.gguf-determinism-replay.v1",
                "commands": replay_commands,
                "f16_digest": _digest(b"f16"),
                "imatrix_digest": _digest(b"imatrix"),
                "entrypoint_digest": artifact_contract["entrypoint_digest"],
                "entrypoint_bytes": artifact_contract["entrypoint_bytes"],
                "artifact_tree_digest": artifact_contract["tree_digest"],
                "matches_primary": True,
            }
            source = {
                "training_schema": real_converter.CURRENT_TRAINING_SCHEMA,
                "dataset_schema": candidate.DATASET_SCHEMA,
                "corpus_profile": real_converter.CURRENT_CORPUS_PROFILE,
                "training_metadata_digest": _digest(b"training"),
                "training_metrics_digest": _digest(b"metrics"),
                "merged_tree_digest": _digest(b"merged"),
                "source_corpus": {
                    "bytes": evaluator.CURRENT94_PUBLIC_CORPUS_BYTES,
                    "digest": evaluator.CURRENT94_PUBLIC_CORPUS_RAW_DIGEST,
                    "canonical_bytes": 1,
                    "canonical_digest": candidate.PUBLIC_CORPUS_CANONICAL_DIGEST,
                    "task_count": candidate.EXPECTED_COUNTS["train"],
                    "refs_digest": _digest(b"refs"),
                },
                "prepared_dataset": {
                    "manifest_digest": _digest(b"manifest"),
                    "train_digest": _digest(b"train"),
                    "holdout_digest": _digest(b"holdout"),
                    "train_examples": candidate.EXPECTED_COUNTS["train"],
                    "holdout_examples": 0,
                },
                "base_snapshot": {
                    "base_model": candidate.QWEN25_CODER_1_5B_BASE_MODEL,
                },
            }
            expected_receipt_artifact = {
                **artifact_contract,
                "quantization": validator.QUANTIZATION,
            }
            common = {
                "status": "complete",
                "track": validator.TRACK,
                "hardware_class": validator.HARDWARE_CLASS,
                "base_model": validator.CURRENT94_BASE_MODEL,
                "llama_cpp_revision": real_converter.LLAMA_CPP_REVISION,
            }
            calibration = {
                **common,
                "schema": validator.CURRENT94_CALIBRATION_SCHEMA,
                "profile": real_converter.CALIBRATION_PROFILE,
                "source": {"synthetic": "public-calibration-source"},
                "selection": {"indices": [0]},
                "toolchain": {
                    "converter_digest": "sha256:" + "b" * 64,
                    "converter_python": copy.deepcopy(interpreter_receipt),
                    "imatrix_digest": "sha256:" + "c" * 64,
                    "quantizer_digest": "sha256:" + "d" * 64,
                    "runtime_libraries": runtime_closure,
                },
                "commands": commands,
                "determinism_replay": replay,
                "artifact": {
                    **expected_receipt_artifact,
                    "calibration_metadata": {"synthetic": True},
                },
                "load_manifest": load_manifest,
            }
            conversion = {
                **common,
                "schema": validator.CURRENT94_CONVERSION_SCHEMA,
                "source": source,
                "conversion": {
                    "converter_python": copy.deepcopy(interpreter_receipt),
                    "runtime_libraries": runtime_closure,
                    "converter_digest": "sha256:" + "b" * 64,
                    "imatrix_digest": "sha256:" + "c" * 64,
                    "quantizer_digest": "sha256:" + "d" * 64,
                    "commands": copy.deepcopy(commands),
                    "determinism_replay": copy.deepcopy(replay),
                },
                "artifact": expected_receipt_artifact,
                "load_manifest": load_manifest,
                "calibration_receipt_digest": _digest(b"replaced-when-written"),
            }
            portable_runtime = {
                "converter_interpreter": {
                    "container_path": portable_interpreter["path"],
                    "bytes": portable_interpreter["bytes"],
                    "digest": portable_interpreter["digest"],
                    "mode": portable_interpreter["mode"],
                },
                "llama_cpp_runtime_closure": {
                    "bytes": len(closure_raw),
                    "digest": _digest(closure_raw),
                },
            }
            base_spec = replace(
                loaded,
                bundle=bundle,
                load_spec={"bytes": len(load_raw), "digest": _digest(load_raw)},
                artifact_contract=artifact_contract,
                conversion_runtime=portable_runtime,
            )

            def write_receipts(
                calibration_value: dict[str, object],
                conversion_value: dict[str, object],
            ) -> validator.Current94SpecBindings:
                calibration_raw = validator._pretty_json_bytes_for_current94(
                    calibration_value
                )
                normalized_conversion = copy.deepcopy(conversion_value)
                normalized_conversion["calibration_receipt_digest"] = _digest(
                    calibration_raw
                )
                conversion_raw = validator._pretty_json_bytes_for_current94(
                    normalized_conversion
                )
                (bundle / "calibration-receipt.json").write_bytes(calibration_raw)
                (bundle / "conversion-receipt.json").write_bytes(conversion_raw)
                return replace(
                    base_spec,
                    calibration_receipt={
                        "bytes": len(calibration_raw),
                        "digest": _digest(calibration_raw),
                    },
                    conversion_receipt={
                        "bytes": len(conversion_raw),
                        "digest": _digest(conversion_raw),
                    },
                )

            artifact = {
                "root": str(artifact_root),
                "tree_digest": artifact_contract["tree_digest"],
                "entrypoint": {
                    "path": validator.ENTRYPOINT,
                    "bytes": artifact_contract["entrypoint_bytes"],
                    "digest": artifact_contract["entrypoint_digest"],
                    "gguf": {"version": 3, "architecture": "qwen2", "file_type": 15},
                },
            }
            converter = SimpleNamespace(
                CURRENT_CALIBRATED_CONVERSION_SCHEMA=(
                    real_converter.CURRENT_CALIBRATED_CONVERSION_SCHEMA
                ),
                CURRENT_CALIBRATION_SCHEMA=real_converter.CURRENT_CALIBRATION_SCHEMA,
                CURRENT_TRAINING_SCHEMA=real_converter.CURRENT_TRAINING_SCHEMA,
                QWEN25_ARCHITECTURE=real_converter.QWEN25_ARCHITECTURE,
                LLAMA_CPP_REVISION=real_converter.LLAMA_CPP_REVISION,
                CALIBRATION_PROFILE=real_converter.CALIBRATION_PROFILE,
                LLAMA_CPP_ROOT=real_converter.LLAMA_CPP_ROOT,
                _validate_current_loaded_lineage=mock.Mock(),
                _current_conversion_source=mock.Mock(return_value=source),
                _validate_calibration_material_binding=mock.Mock(),
                _converter_python_receipt_identity=(
                    real_converter._converter_python_receipt_identity
                ),
                _validate_calibrated_receipts=real_converter._validate_calibrated_receipts,
                _runtime_library_closure=mock.Mock(
                    side_effect=AssertionError("runtime binaries must not be inspected")
                ),
            )
            tools = validator.Current94Toolset(
                candidate=SimpleNamespace(),
                evaluator=StaticEvaluator(artifact),
                converter=converter,
            )
            spec = write_receipts(calibration, conversion)
            result = validator._validate_current94_conversion_bundle(spec, tools, {})
            self.assertEqual(result.artifact["tree_digest"], artifact_contract["tree_digest"])
            self.assertEqual(
                result.replay_receipts[0][
                    "converter_interpreter_portable_receipt_content"
                ],
                portable_runtime["converter_interpreter"],
            )
            converter._runtime_library_closure.assert_not_called()

            changed_conversion = copy.deepcopy(conversion)
            changed_conversion["conversion"]["converter_python"]["portable"]["digest"] = (
                "sha256:" + "0" * 64
            )
            changed_spec = write_receipts(calibration, changed_conversion)
            with self.assertRaisesRegex(
                validator.ValidationRefused,
                "different converter Python identities",
            ):
                validator._validate_current94_conversion_bundle(changed_spec, tools, {})

            changed_calibration = copy.deepcopy(calibration)
            changed_conversion = copy.deepcopy(conversion)
            changed_calibration["commands"][0]["launch"]["method"] = "direct-path"
            changed_conversion["conversion"]["commands"][0]["launch"][
                "method"
            ] = "direct-path"
            changed_spec = write_receipts(changed_calibration, changed_conversion)
            with self.assertRaisesRegex(
                validator.ValidationRefused,
                "held-fd launch identity changed",
            ):
                validator._validate_current94_conversion_bundle(changed_spec, tools, {})

            changed_calibration = copy.deepcopy(calibration)
            changed_conversion = copy.deepcopy(conversion)
            changed_calibration["determinism_replay"]["commands"][0]["launch"][
                "executed_object"
            ]["worker_observation"]["inode"] = 999
            changed_conversion["conversion"]["determinism_replay"]["commands"][0][
                "launch"
            ]["executed_object"]["worker_observation"]["inode"] = 999
            changed_spec = write_receipts(changed_calibration, changed_conversion)
            with self.assertRaisesRegex(
                validator.ValidationRefused,
                "held-fd launch identity changed",
            ):
                validator._validate_current94_conversion_bundle(changed_spec, tools, {})

            valid_spec = write_receipts(calibration, conversion)
            changed_runtime = copy.deepcopy(portable_runtime)
            changed_runtime["converter_interpreter"]["digest"] = "sha256:" + "f" * 64
            with self.assertRaisesRegex(
                validator.ValidationRefused,
                "portable converter interpreter",
            ):
                validator._validate_current94_conversion_bundle(
                    replace(valid_spec, conversion_runtime=changed_runtime),
                    tools,
                    {},
                )

    def test_signed_v032_runtime_identity_is_recomputed_and_release_swap_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = _write_current94_spec(
                Path(temporary) / "spec.json", _current94_spec_payload()
            )
        identity = {
            "python": {
                "version": validator.EXPECTED_PYTHON_VERSION,
                "executable": {
                    "path": "/usr/bin/python3.12",
                    "bytes": 8_016_832,
                    "digest": (
                        "sha256:1319c137ea5d30f1d7599943cb0e72666648c20a94cf5932dd095364d07dafeb"
                    ),
                },
            },
            "microtensor": {"release_version": "0.3.2", "mechanism_version": "0.3.0"},
        }
        raw = validator._canonical_json_bytes(identity)
        spec = replace(
            spec,
            runtime_contract={
                **spec.runtime_contract,
                "identity": {"bytes": len(raw), "digest": _digest(raw)},
            },
        )
        tools = validator.Current94Toolset(
            candidate=SimpleNamespace(),
            evaluator=SimpleNamespace(
                SIGNED_RELEASE_VERSION="0.3.2",
                SIGNED_MECHANISM_VERSION="0.3.0",
            ),
            converter=SimpleNamespace(),
        )
        with mock.patch.object(
            validator.sys,
            "executable",
            "/tmp/microtensor-v030-verify.5rMSRW/venv/bin/python",
        ):
            validator._validate_current94_runtime_identity(identity, spec, tools)
            tampered = copy.deepcopy(identity)
            tampered["microtensor"]["release_version"] = "0.3.0"
            with self.assertRaisesRegex(validator.ValidationRefused, "Microtensor"):
                validator._validate_current94_runtime_identity(tampered, spec, tools)

    def test_qwen25_v2_summary_and_overlap_claim_validate_without_running_poison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "r1")
            summary = fixture.write()
            summary["schema"] = evaluator.SCHEMA_V2
            summary["base_model"] = validator.CURRENT94_BASE_MODEL
            summary["lineage_claim"] = evaluator.CURRENT_OVERLAP_LINEAGE_CLAIM
            (fixture.root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt = validator._validate_repeat(
                "r1",
                fixture.root,
                context=fixture.context,
                gates=validator.EXPECTED_GATES,
                summary_schema=evaluator.SCHEMA_V2,
                base_model=validator.CURRENT94_BASE_MODEL,
                lineage_claim=evaluator.CURRENT_OVERLAP_LINEAGE_CLAIM,
            )
            self.assertEqual(receipt["gates"]["successful_generations"], 16)
            self.assertFalse(fixture.sentinel.exists())

    def test_current_report_repeats_non_authorizing_training_overlap_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _write_current94_spec(root / "spec.json", _current94_spec_payload())
            spec = replace(spec, output_roots=tuple(root / repeat for repeat in validator.REPEATS))
            conversion = validator.ConversionBindings(
                artifact={
                    "tree_digest": "sha256:" + "1" * 64,
                    "entrypoint": {"digest": "sha256:" + "2" * 64},
                },
                load_manifest={},
                replay_receipts=({"schema": validator.CURRENT94_CONVERSION_SCHEMA},),
            )
            tools = validator.Current94Toolset(
                candidate=SimpleNamespace(),
                evaluator=SimpleNamespace(
                    SCHEMA_V2=evaluator.SCHEMA_V2,
                    CURRENT_OVERLAP_LINEAGE_CLAIM=evaluator.CURRENT_OVERLAP_LINEAGE_CLAIM,
                ),
                converter=SimpleNamespace(),
            )
            with (
                mock.patch.object(validator, "_load_current94_v8_spec", return_value=spec),
                mock.patch.object(
                    validator,
                    "_prepare_current94_context",
                    return_value=(object(), conversion),
                ),
                mock.patch.object(
                    validator,
                    "_validate_repeat",
                    return_value={"raw_output_digests": []},
                ) as validate_repeat,
                mock.patch.object(
                    validator,
                    "_aggregate",
                    return_value={
                        "validated_repeat_hard_gates_passed": True,
                        "all_declared_local_gates_passed": False,
                    },
                ),
            ):
                report = validator.validate_current94_v8_diagnostic(
                    spec.path, "r1", _tools=tools
                )
        claim = report["claim"]
        self.assertTrue(claim["diagnostic_rows_are_training_overlap"])
        self.assertTrue(claim["conversion_v6_calibration_v3_bound"])
        self.assertFalse(claim["generated_or_corpus_code_executed_by_this_static_validator"])
        self.assertTrue(claim["conversion_runtime_receipt_content_bound"])
        self.assertTrue(claim["converter_interpreter_portable_receipt_content_bound"])
        self.assertFalse(claim["executed_interpreter_attested"])
        self.assertFalse(claim["hermetic_conversion_attested"])
        self.assertFalse(claim["conversion_runtime_execution_verified"])
        self.assertFalse(claim["execution_pass_at_1_claimed"])
        self.assertFalse(claim["quality_or_rank_claimed"])
        self.assertFalse(claim["promotion_authorized"])
        self.assertIn("hermetic containment", claim["remaining_external_gates"][0])
        self.assertEqual(validate_repeat.call_args.kwargs["summary_schema"], evaluator.SCHEMA_V2)

    def test_explicit_schema_dispatch_selects_only_current94_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            _write_current94_spec(path, _current94_spec_payload())
            expected = {"schema": validator.CURRENT94_VALIDATION_SCHEMA}
            with mock.patch.object(
                validator,
                "validate_current94_v8_diagnostic",
                return_value=expected,
            ) as current:
                self.assertEqual(validator.validate_declared_diagnostic(path, "r1"), expected)
            current.assert_called_once_with(path, "r1")


class StaticSafetyAndEntrypointTests(unittest.TestCase):
    def test_validator_has_no_dynamic_execution_or_model_engine_calls(self) -> None:
        source_path = Path(validator.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        forbidden_names: list[str] = []
        forbidden_attributes: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"compile", "eval", "exec"}:
                forbidden_names.append(node.func.id)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "engine_type",
                "generate",
                "load",
            }:
                forbidden_attributes.append(node.func.attr)
        self.assertEqual(forbidden_names, [])
        self.assertEqual(forbidden_attributes, [])

    def test_imported_module_invocation_is_refused_before_argument_or_validation_work(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(validator, "__package__", "training"),
            mock.patch.object(validator, "_parse_args") as parse_args,
            mock.patch.object(validator, "validate_declared_diagnostic") as validate,
            redirect_stderr(stderr),
        ):
            result = validator.main(["--not-even-parsed"])
        self.assertEqual(result, 2)
        self.assertIn("direct path", stderr.getvalue())
        parse_args.assert_not_called()
        validate.assert_not_called()

    def test_direct_path_main_emits_one_canonical_report_or_refuses_cleanly(self) -> None:
        report = {"status": "validated", "schema": validator.VALIDATION_SCHEMA}
        stdout = io.StringIO()
        with (
            mock.patch.object(validator, "__package__", ""),
            mock.patch.object(
                validator,
                "_parse_args",
                return_value=SimpleNamespace(experiment_spec=Path("spec"), through="r1"),
            ),
            mock.patch.object(validator, "validate_declared_diagnostic", return_value=report),
            redirect_stdout(stdout),
        ):
            result = validator.main([])
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), validator._canonical_json_bytes(report).decode() + "\n")

        stderr = io.StringIO()
        with (
            mock.patch.object(validator, "__package__", ""),
            mock.patch.object(
                validator,
                "_parse_args",
                return_value=SimpleNamespace(experiment_spec=Path("spec"), through="r1"),
            ),
            mock.patch.object(
                validator,
                "validate_declared_diagnostic",
                side_effect=validator.ValidationRefused("tampered"),
            ),
            redirect_stderr(stderr),
        ):
            result = validator.main([])
        self.assertEqual(result, 2)
        self.assertIn("refused: tampered", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
