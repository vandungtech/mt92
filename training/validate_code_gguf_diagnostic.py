#!/usr/bin/env python3
"""Read-only, fail-closed validation of the declared v6 GGUF diagnostics.

This module never constructs a model engine and never imports, compiles, or
executes generated or corpus code.  Generated and reference strings are passed
only to the pinned evaluator's AST-only structural diagnostics.  Invoke this
file by its direct path so the pinned detached source package cannot be shadowed
by the package that contains this validator.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=SyntaxWarning)

VALIDATION_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-validation.v1"
SPEC_SCHEMA: Final[str] = "microtensor.code.calibrated-quantization-experiment.v1"
SPEC_BYTES: Final[int] = 49_270
SPEC_DIGEST: Final[str] = "sha256:7dc168c55316b3cc378809d13f8fe3777bfa29824bc97dd603c215324b8bd97d"
SOURCE_ROOT: Final[Path] = Path(
    "/tmp/mt92-q4-code-c9c4eff"  # noqa: S108 - immutable experiment path
)
SOURCE_COMMIT: Final[str] = "c9c4effe77271b2d70d4eee745de654af9d1e74d"
CANDIDATE_ID: Final[str] = "qwen3-06b-historical7299-v6-q4-imatrix128-m541"
BASE_MODEL: Final[str] = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
TRACK: Final[str] = "code"
HARDWARE_CLASS: Final[str] = "mt-3g"
QUANTIZATION: Final[str] = "Q4_K_M"
ENTRYPOINT: Final[str] = "model.gguf"
MAX_INPUT_TOKENS: Final[int] = 541
LLAMA_CPP_REVISION: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
EXPECTED_PYTHON_VERSION: Final[str] = "3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]"
EXPECTED_RUNTIME_IDENTITY_BYTES: Final[int] = 21_187
EXPECTED_RUNTIME_IDENTITY_DIGEST: Final[str] = (
    "sha256:15642b2a5f91cef17283e76e812eea1c069175a3aa19fb57e3ab45e479755c11"
)
MAX_JSON_BYTES: Final[int] = 80 * 1024 * 1024
MAX_RESULT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_ERROR_BYTES: Final[int] = 4096
EXPECTED_EXAMPLES: Final[int] = 16
REPEATS: Final[tuple[str, ...]] = ("r1", "r2", "r3")
_DIGEST: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_EXECUTABLE: Final[str] = "/usr/bin/git"
PINNED_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "training",
        "training.code_candidate",
        "training.evaluate_code_gguf",
        "training.historical_code_candidate",
    }
)
PINNED_POST_CONTEXT_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        *PINNED_IMPORTS,
        "training.evaluate_code",
        "training.train_code",
    }
)
PINNED_MODULE_PATHS: Final[dict[str, str]] = {
    "training.code_candidate": "training/code_candidate.py",
    "training.evaluate_code": "training/evaluate_code.py",
    "training.evaluate_code_gguf": "training/evaluate_code_gguf.py",
    "training.historical_code_candidate": "training/historical_code_candidate.py",
    "training.train_code": "training/train_code.py",
}

EXPECTED_SOURCE_FILES: Final[dict[str, tuple[int, str]]] = {
    "training/evaluate_code_gguf.py": (
        65_040,
        "sha256:98091689311e20383ce05b58ae230ee149fd37504fb1f6edcf2d2a97594e3892",
    ),
    "training/code_candidate.py": (
        41_583,
        "sha256:220e27023ee9c01560c0810d587dc417a70c92a314cb6874b97954b6ba8e5e0c",
    ),
    "training/evaluate_code.py": (
        48_687,
        "sha256:bcc87c6641d99ed9ac3d8534040de8a881347c2dc69678bbf87496a6c2384843",
    ),
    "training/historical_code_candidate.py": (
        21_984,
        "sha256:2e8540486e53f6ae348c8d471fdce97284c1fef0c6a4254b67e35db8f141d00a",
    ),
    "training/train_code.py": (
        50_933,
        "sha256:1472e226a99da4707e5002fe860f7d543e9f1a1117dc988f4d36ba3bea7ad4b1",
    ),
    "training/publish_code_provenance.py": (
        123_572,
        "sha256:3416d44f1598ba657417ff2cb61ed93761614dabfc7fddb2b82bd3f42c37d60b",
    ),
}

SUMMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "status",
        "track",
        "hardware_class",
        "base_model",
        "quality_claim",
        "runtime_claim",
        "lineage_claim",
        "safety_contract",
        "configuration",
        "configuration_digest",
        "artifact",
        "evaluation_dataset",
        "training_lineage",
        "runtime",
        "timing",
        "memory",
        "results",
    }
)
RESULT_SUMMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "file",
        "bytes",
        "digest",
        "examples",
        "successful_generations",
        "failed_generations",
        "failed_refs",
        "quality_score",
        "execution_pass_at_1",
        "structural_diagnostics",
        "scorer_extracted_reference_text_similarity",
        "latency_ms",
        "rss",
        "output",
    }
)
SAFETY_CONTRACT: Final[dict[str, bool]] = {
    "generated_code_imported": False,
    "generated_code_executed": False,
    "generated_code_bytecode_compiled": False,
    "corpus_code_imported": False,
    "corpus_code_executed": False,
    "corpus_code_bytecode_compiled": False,
    "static_ast_parse_only": True,
    "hidden_or_scored_tests_accessed": False,
}
BOOLEAN_DIAGNOSTIC_FIELDS: Final[tuple[str, ...]] = (
    "scorer_extraction_changed",
    "raw_contains_code_fence",
    "raw_contains_thinking_markup",
    "raw_nonempty",
    "raw_parseable_python",
    "raw_top_level_task_func",
    "raw_solution_class",
    "scorer_extracted_contains_code_fence",
    "scorer_extracted_contains_thinking_markup",
    "scorer_extracted_nonempty",
    "scorer_extracted_parseable_python",
    "scorer_extracted_top_level_task_func",
    "scorer_extracted_solution_class",
    "scorer_extracted_exact_reference_text",
    "scorer_extracted_exact_reference_ast",
)

EXPECTED_GATES: Final[dict[str, int]] = {
    "successful_generations_minimum": 16,
    "failed_generations_maximum": 0,
    "scorer_extracted_parseable_python_minimum": 16,
    "scorer_extracted_top_level_task_func_minimum": 16,
    "scorer_extracted_residual_fences_maximum": 0,
    "maximum_request_latency_ms": 31_200,
    "p95_ttft_ms_maximum": 5_256,
    "p95_ttft_ms_preferred": 4_700,
    "peak_rss_bytes_maximum": 1_073_741_824,
    "maximum_stream_pieces_per_request": 373,
}

EXPECTED_LINEAGE_DIGESTS: Final[dict[str, str]] = {
    "manifest": "sha256:6af4fe8952293339773e133a867d78e817d759a83138bc7400af98e2e04898ff",
    "holdout": "sha256:8b07f781c6d160f752963b3a42f343c18d53a785d7b7cd09f472fdddbd2d7993",
    "refs": "sha256:73edc2a7674e0c718ea4ef7ea67c638b1a2c431320789b632aad5909309e01ee",
    "source": "sha256:1c37a0e212936bfac8c86f955ad61fd378f58603413b45ece88382d528ace9d5",
    "training_metadata": (
        "sha256:2c637e7e9f8e9811d5150e9c8bbd01e596b2f30908aecb10c1a1a8e2dd2cfc6c"
    ),
    "merged_tree": "sha256:52b399b89edaec39005507045b0facbc31adf654a5e2d3e801b5a52e79c3a175",
}
EXPECTED_ARTIFACT: Final[dict[str, Any]] = {
    "tree_digest": ("sha256:3f6dc72a0cd886c74a5161ccd42feda27de56e54c914f28961e7dd89ca2917b5"),
    "entrypoint_bytes": 396_704_672,
    "entrypoint_digest": (
        "sha256:3df33a173b16af2bca9a402c335bda5d39b03e290d4ba13f4eaf5ad5c4397d5e"
    ),
}
EXPECTED_REPLAY_FILES: Final[dict[str, dict[str, tuple[int, str]]]] = {
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
}


class ValidationRefused(ValueError):
    """The diagnostic or any of its declared bindings failed closed."""


@dataclass(frozen=True)
class Toolset:
    """Pinned evaluator modules used only for static replay and hashing."""

    candidate: ModuleType
    evaluator: ModuleType


_PINNED_TOOL_CACHE: Toolset | None = None


@dataclass(frozen=True)
class SpecBindings:
    """Only paths and thresholds derived from the immutable v6 declaration."""

    path: Path
    raw: bytes
    payload: dict[str, Any]
    output_roots: tuple[Path, Path, Path]
    bundles: tuple[Path, Path]
    dataset: Path
    diagnostic_jsonl: Path
    diagnostic_source: Path
    training_arguments: tuple[Path, Path, Path, Path]
    source_root: Path
    gates: dict[str, int]


@dataclass(frozen=True)
class ConversionBindings:
    """Static identities shared by all diagnostic repeats."""

    artifact: dict[str, Any]
    load_manifest: dict[str, Any]
    replay_receipts: tuple[dict[str, Any], dict[str, Any]]


@dataclass(frozen=True)
class ValidationContext:
    """Recomputed, model-free identities expected in each diagnostic summary."""

    candidate: ModuleType
    evaluator: ModuleType
    rows: tuple[dict[str, Any], ...]
    evaluation_dataset: dict[str, Any]
    training_lineage: dict[str, Any]
    runtime: Any
    artifact: dict[str, Any]
    configuration: dict[str, Any]
    configuration_digest: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationRefused(f"value is not finite canonical JSON: {exc}") from exc


def _digest_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _json_exact(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json_bytes(actual) != _canonical_json_bytes(expected):
        raise ValidationRefused(f"{label} changed")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationRefused(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValidationRefused(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value) or frozenset(value) != expected:
        raise ValidationRefused(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(value)}"
        )


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValidationRefused(f"{label} must be lowercase sha256:<64 hex>")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationRefused(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationRefused(f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationRefused(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValidationRefused(f"{label} must be finite and non-negative")
    if maximum is not None and result > maximum:
        raise ValidationRefused(f"{label} exceeds {maximum}")
    return result


def _stable_regular_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    if maximum < 0:
        raise ValidationRefused(f"{label} byte limit must be non-negative")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValidationRefused(f"{label} must be a regular non-symlink file: {path}") from exc
        raise ValidationRefused(f"{label} cannot be inspected: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValidationRefused(f"{label} must be a regular non-symlink file: {path}")
        if before.st_size > maximum:
            raise ValidationRefused(f"{label} exceeds the {maximum}-byte limit")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValidationRefused(f"{label} cannot be read: {exc}") from exc
    finally:
        os.close(descriptor)
    if len(raw) > maximum:
        raise ValidationRefused(f"{label} exceeds the {maximum}-byte limit")
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_key != after_key or len(raw) != after.st_size:
        raise ValidationRefused(f"{label} changed while it was read")
    return raw


def _strict_json(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationRefused(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidationRefused(f"{label} contains non-finite number {value}")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValidationRefused(f"{label} contains non-finite number {value}")
        return result

    def reject_surrogates(value: Any) -> None:
        if isinstance(value, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                raise ValidationRefused(f"{label} contains a Unicode surrogate")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                reject_surrogates(key)
                reject_surrogates(item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                reject_surrogates(item)

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
        reject_surrogates(payload)
        return payload
    except ValidationRefused:
        raise
    except (OverflowError, RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationRefused(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _strict_json_file(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_JSON_BYTES,
) -> tuple[Any, bytes]:
    raw = _stable_regular_bytes(path, label, maximum=maximum)
    return _strict_json(raw, label), raw


def _file_receipt(path: Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    raw = _stable_regular_bytes(path, label, maximum=maximum)
    return {"bytes": len(raw), "digest": _digest_bytes(raw)}


def _require_expected_file(
    raw: bytes,
    expected: tuple[int, str],
    label: str,
) -> None:
    expected_bytes, expected_digest = expected
    if len(raw) != expected_bytes or _digest_bytes(raw) != expected_digest:
        raise ValidationRefused(f"{label} bytes or digest changed")


def _directory_entries(path: Path, label: str) -> dict[str, int]:
    try:
        root_stat = path.lstat()
    except OSError as exc:
        raise ValidationRefused(f"{label} cannot be inspected: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValidationRefused(f"{label} must be a regular non-symlink directory: {path}")
    try:
        with os.scandir(path) as entries:
            return {entry.name: entry.stat(follow_symlinks=False).st_mode for entry in entries}
    except OSError as exc:
        raise ValidationRefused(f"{label} cannot be enumerated: {exc}") from exc


def _require_exact_tree(path: Path, expected: Mapping[str, str], label: str) -> None:
    entries = _directory_entries(path, label)
    if frozenset(entries) != frozenset(expected):
        raise ValidationRefused(
            f"{label} entries changed: expected {sorted(expected)}, got {sorted(entries)}"
        )
    for name, kind in expected.items():
        mode = entries[name]
        if kind == "file" and not stat.S_ISREG(mode):
            raise ValidationRefused(f"{label}/{name} must be a regular non-symlink file")
        if kind == "directory" and not stat.S_ISDIR(mode):
            raise ValidationRefused(f"{label}/{name} must be a regular non-symlink directory")


def _nested(payload: Mapping[str, Any], *parts: str) -> Any:
    value: Any = payload
    for part in parts:
        value = _mapping(value, ".".join(parts))
        if part not in value:
            raise ValidationRefused(f"declaration is missing {'.'.join(parts)}")
        value = value[part]
    return value


def _load_spec(path: Path) -> SpecBindings:
    payload, raw = _strict_json_file(path, "v6 experiment spec", maximum=SPEC_BYTES)
    if len(raw) != SPEC_BYTES or _digest_bytes(raw) != SPEC_DIGEST:
        raise ValidationRefused("v6 experiment spec bytes or digest changed")
    spec = dict(_mapping(payload, "v6 experiment spec"))
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValidationRefused("v6 experiment schema changed")
    if _nested(spec, "code", "commit") != SOURCE_COMMIT:
        raise ValidationRefused("v6 source commit changed")
    if _nested(spec, "code", "isolated_execution", "worktree_path") != str(SOURCE_ROOT):
        raise ValidationRefused("v6 detached source root changed")
    if _nested(spec, "candidate", "id") != CANDIDATE_ID:
        raise ValidationRefused("v6 candidate ID changed")
    if _nested(spec, "candidate", "quantization") != QUANTIZATION:
        raise ValidationRefused("v6 quantization changed")
    if _nested(spec, "candidate", "max_input_tokens") != MAX_INPUT_TOKENS:
        raise ValidationRefused("v6 max-input-tokens changed")

    diagnostic = _mapping(spec.get("diagnostic"), "v6 diagnostic declaration")
    outputs = tuple(
        Path(str(item)) for item in _sequence(diagnostic.get("output_roots"), "outputs")
    )
    if (
        len(outputs) != 3
        or len(set(outputs)) != 3
        or any(not item.is_absolute() for item in outputs)
    ):
        raise ValidationRefused("v6 diagnostic roots must be three unique absolute paths")
    bundles_payload = _mapping(
        _nested(spec, "candidate", "output_bundles"),
        "v6 conversion bundles",
    )
    if frozenset(bundles_payload) != frozenset({"replay1", "replay2"}):
        raise ValidationRefused("v6 conversion bundle declarations changed")
    bundles = (Path(str(bundles_payload["replay1"])), Path(str(bundles_payload["replay2"])))
    artifact_path = _nested(spec, "diagnostic", "artifact_binding", "artifact_path")
    if artifact_path != str(bundles[0] / "artifact"):
        raise ValidationRefused("diagnostic artifact is not bound to replay1")
    if _nested(spec, "diagnostic", "artifact_binding", "entrypoint") != ENTRYPOINT:
        raise ValidationRefused("diagnostic entrypoint changed")
    if _nested(spec, "diagnostic", "artifact_binding", "quantization") != QUANTIZATION:
        raise ValidationRefused("diagnostic artifact quantization changed")

    declared_gates = _mapping(spec.get("gates"), "v6 gates")
    gates: dict[str, int] = {}
    for field, expected in EXPECTED_GATES.items():
        value = _integer(declared_gates.get(field), f"v6 gate {field}")
        if value != expected:
            raise ValidationRefused(f"v6 gate {field} changed")
        gates[field] = value
    replay_identity_gate = declared_gates.get(
        "artifact_tree_and_entrypoint_identical_across_external_invocations"
    )
    if replay_identity_gate is not True:
        raise ValidationRefused("v6 conversion replay identity gate changed")
    if declared_gates.get("official_namespace_sandbox_selfcheck_required") is not True:
        raise ValidationRefused("official namespace selfcheck gate changed")

    declared_lineage = {
        "manifest": diagnostic.get("dataset_manifest_sha256"),
        "holdout": diagnostic.get("holdout_jsonl_sha256"),
        "refs": diagnostic.get("refs_sha256"),
        "source": diagnostic.get("source_corpus_sha256"),
        "training_metadata": _nested(spec, "lineage", "training_metadata_sha256"),
        "merged_tree": _nested(spec, "lineage", "merged_tree_sha256"),
    }
    _json_exact(declared_lineage, EXPECTED_LINEAGE_DIGESTS, "v6 lineage digests")
    if diagnostic.get("examples") != EXPECTED_EXAMPLES:
        raise ValidationRefused("v6 diagnostic example count changed")
    training = _mapping(diagnostic.get("training_lineage_arguments"), "training arguments")
    _exact_keys(
        training,
        frozenset({"training_run", "training_dataset", "source_corpus", "base"}),
        "training arguments",
    )
    return SpecBindings(
        path=path,
        raw=raw,
        payload=spec,
        output_roots=(outputs[0], outputs[1], outputs[2]),
        bundles=bundles,
        dataset=Path(str(diagnostic.get("dataset"))),
        diagnostic_jsonl=Path(str(diagnostic.get("diagnostic_jsonl"))),
        diagnostic_source=Path(str(diagnostic.get("source_corpus"))),
        training_arguments=(
            Path(str(training["training_run"])),
            Path(str(training["training_dataset"])),
            Path(str(training["source_corpus"])),
            Path(str(training["base"])),
        ),
        source_root=SOURCE_ROOT,
        gates=gates,
    )


def _source_file_identity(path: Path, label: str) -> dict[str, Any]:
    raw = _stable_regular_bytes(path, label, maximum=2 * 1024 * 1024)
    return {"bytes": len(raw), "digest": _digest_bytes(raw)}


def _git_output(source_root: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            [GIT_EXECUTABLE, "-C", str(source_root), *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=10,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationRefused(f"{label} could not be checked") from exc
    if completed.returncode != 0:
        raise ValidationRefused(f"{label} check failed")
    if len(completed.stdout) > 1024 * 1024 or len(completed.stderr) > 1024 * 1024:
        raise ValidationRefused(f"{label} check output is implausibly large")
    if completed.stderr:
        raise ValidationRefused(f"{label} check wrote stderr")
    return completed.stdout


def _validate_clean_source_root(source_root: Path) -> None:
    head = _git_output(source_root, ("rev-parse", "--verify", "HEAD"), "source HEAD")
    if head != (SOURCE_COMMIT + "\n").encode("ascii"):
        raise ValidationRefused("pinned source HEAD changed")
    status = _git_output(
        source_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        "source worktree",
    )
    if status:
        raise ValidationRefused("pinned source worktree is not clean")
    training_ignored = _git_output(
        source_root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--ignored=matching",
            "--untracked-files=all",
            "--",
            "training",
        ),
        "source import tree",
    )
    if training_ignored:
        raise ValidationRefused("pinned source import tree contains ignored shadow files")


def _load_pinned_tools(source_root: Path) -> Toolset:
    global _PINNED_TOOL_CACHE

    try:
        resolved = source_root.resolve(strict=True)
    except OSError as exc:
        raise ValidationRefused(f"pinned source root cannot be resolved: {exc}") from exc
    if resolved != SOURCE_ROOT:
        raise ValidationRefused("pinned source root resolved elsewhere")
    _validate_clean_source_root(resolved)
    for relative, (expected_bytes, expected_digest) in EXPECTED_SOURCE_FILES.items():
        identity = _source_file_identity(resolved / relative, f"pinned source {relative}")
        if identity != {"bytes": expected_bytes, "digest": expected_digest}:
            raise ValidationRefused(f"pinned source {relative} changed")

    preloaded = frozenset(
        name for name in sys.modules if name == "training" or name.startswith("training.")
    )
    if preloaded:
        if _PINNED_TOOL_CACHE is None or preloaded not in {
            PINNED_IMPORTS,
            PINNED_POST_CONTEXT_IMPORTS,
        }:
            raise ValidationRefused(f"pinned training package was preloaded: {sorted(preloaded)}")
        candidate = _PINNED_TOOL_CACHE.candidate
        evaluator = _PINNED_TOOL_CACHE.evaluator
    else:
        candidate = None
        evaluator = None

    root_text = str(resolved)
    sanitized_path = [root_text]
    for entry in sys.path:
        search_root = Path(entry or os.getcwd()).resolve(strict=False)
        if search_root == resolved:
            continue
        if (search_root / "training").is_dir():
            continue
        sanitized_path.append(entry)
    sys.path[:] = sanitized_path
    if candidate is None or evaluator is None:
        try:
            candidate = importlib.import_module("training.code_candidate")
            evaluator = importlib.import_module("training.evaluate_code_gguf")
        except ImportError as exc:
            raise ValidationRefused(f"pinned evaluator import failed: {exc}") from exc
    imported = frozenset(
        name for name in sys.modules if name == "training" or name.startswith("training.")
    )
    allowed_imports = (
        {PINNED_IMPORTS}
        if _PINNED_TOOL_CACHE is None
        else {PINNED_IMPORTS, PINNED_POST_CONTEXT_IMPORTS}
    )
    if imported not in allowed_imports:
        raise ValidationRefused(f"pinned evaluator import closure changed: got {sorted(imported)}")
    package = sys.modules["training"]
    package_paths = tuple(Path(item).resolve(strict=True) for item in package.__path__)
    if package_paths != (resolved / "training",):
        raise ValidationRefused("pinned training namespace resolved outside the source root")
    for name in sorted(imported - {"training"}):
        module = sys.modules[name]
        relative = PINNED_MODULE_PATHS.get(name)
        if relative is None:
            raise ValidationRefused(f"pinned module {name} is not declared")
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            raise ValidationRefused(f"pinned module {module.__name__} has no source path")
        try:
            if Path(module_path).resolve(strict=True) != resolved / relative:
                raise ValidationRefused(f"pinned module {module.__name__} came from another tree")
        except OSError as exc:
            raise ValidationRefused(f"pinned module source cannot be resolved: {exc}") from exc
    toolset = Toolset(candidate=candidate, evaluator=evaluator)
    if _PINNED_TOOL_CACHE is None:
        _PINNED_TOOL_CACHE = toolset
    elif (
        toolset.candidate is not _PINNED_TOOL_CACHE.candidate
        or toolset.evaluator is not _PINNED_TOOL_CACHE.evaluator
    ):
        raise ValidationRefused("pinned evaluator module objects changed")
    return _PINNED_TOOL_CACHE


def _validate_conversion_bundle(
    root: Path,
    *,
    replay: str,
    evaluator: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_files = EXPECTED_REPLAY_FILES.get(replay)
    if expected_files is None:
        raise ValidationRefused(f"unsupported conversion replay {replay!r}")
    label = f"conversion bundle {root.name}"
    _require_exact_tree(
        root,
        {
            "artifact": "directory",
            "calibration-receipt.json": "file",
            "conversion-receipt.json": "file",
            "load-spec.json": "file",
        },
        label,
    )
    _require_exact_tree(root / "artifact", {ENTRYPOINT: "file"}, f"{label} artifact")
    load_payload, load_raw = _strict_json_file(root / "load-spec.json", f"{label} load spec")
    _require_expected_file(load_raw, expected_files["load_spec"], f"{label} load spec")
    load_manifest = dict(_mapping(load_payload, f"{label} load spec"))
    expected_load_manifest = {
        "format": "gguf",
        "quantization": QUANTIZATION,
        "entrypoint": ENTRYPOINT,
        "max_input": {"tokens": MAX_INPUT_TOKENS},
        "preprocessing": {"tokenizer": "tokenizer.json"},
        "base_model": BASE_MODEL,
    }
    _json_exact(load_manifest, expected_load_manifest, f"{label} load spec")

    receipt_payload, receipt_raw = _strict_json_file(
        root / "conversion-receipt.json", f"{label} conversion receipt"
    )
    _require_expected_file(
        receipt_raw,
        expected_files["conversion_receipt"],
        f"{label} conversion receipt",
    )
    receipt = _mapping(receipt_payload, f"{label} conversion receipt")
    _exact_keys(
        receipt,
        frozenset(
            {
                "schema",
                "status",
                "track",
                "hardware_class",
                "base_model",
                "llama_cpp_revision",
                "source",
                "load_manifest",
                "artifact",
                "calibration_receipt_digest",
                "conversion",
            }
        ),
        f"{label} conversion receipt",
    )
    expected_header = {
        "schema": "microtensor.code.gguf-conversion.v3",
        "status": "complete",
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
    }
    for field, expected in expected_header.items():
        if receipt.get(field) != expected:
            raise ValidationRefused(f"{label} field {field} changed")
    _json_exact(receipt.get("load_manifest"), load_manifest, f"{label} receipt load manifest")
    _json_exact(
        receipt.get("source"),
        {
            "merged_tree_digest": EXPECTED_LINEAGE_DIGESTS["merged_tree"],
            "training_metadata_digest": EXPECTED_LINEAGE_DIGESTS["training_metadata"],
        },
        f"{label} source",
    )

    calibration_payload, calibration_raw = _strict_json_file(
        root / "calibration-receipt.json", f"{label} calibration receipt"
    )
    _require_expected_file(
        calibration_raw,
        expected_files["calibration_receipt"],
        f"{label} calibration receipt",
    )
    calibration = _mapping(calibration_payload, f"{label} calibration receipt")
    if (
        calibration.get("schema") != "microtensor.code.imatrix-calibration.v2"
        or calibration.get("status") != "complete"
        or calibration.get("track") != TRACK
        or calibration.get("hardware_class") != HARDWARE_CLASS
        or calibration.get("base_model") != BASE_MODEL
        or calibration.get("llama_cpp_revision") != LLAMA_CPP_REVISION
    ):
        raise ValidationRefused(f"{label} calibration identity changed")
    _json_exact(
        calibration.get("load_manifest"), load_manifest, f"{label} calibration load manifest"
    )
    if receipt.get("calibration_receipt_digest") != _digest_bytes(calibration_raw):
        raise ValidationRefused(f"{label} calibration receipt digest changed")

    try:
        artifact = evaluator.artifact_identity(
            root / "artifact",
            entrypoint=ENTRYPOINT,
            expected_digest=_require_digest(
                _mapping(receipt.get("artifact"), f"{label} artifact").get("tree_digest"),
                f"{label} artifact tree digest",
            ),
            quantization=QUANTIZATION,
        )
    except Exception as exc:
        if isinstance(exc, ValidationRefused):
            raise
        raise ValidationRefused(f"{label} artifact validation failed: {exc}") from exc
    entrypoint_header = _mapping(artifact.get("entrypoint"), "artifact entrypoint")
    header = _mapping(entrypoint_header.get("gguf"), "GGUF")
    if (
        header.get("version") != 3
        or header.get("architecture") != "qwen3"
        or header.get("file_type") != 15
    ):
        raise ValidationRefused(f"{label} GGUF v3/Q4_K_M identity changed")
    declared_artifact = _mapping(receipt.get("artifact"), f"{label} declared artifact")
    _exact_keys(
        declared_artifact,
        frozenset({"entrypoint_bytes", "entrypoint_digest", "quantization", "tree_digest"}),
        f"{label} declared artifact",
    )
    entrypoint = _mapping(artifact.get("entrypoint"), f"{label} actual entrypoint")
    _json_exact(
        {
            "tree_digest": artifact.get("tree_digest"),
            "entrypoint_bytes": entrypoint.get("bytes"),
            "entrypoint_digest": entrypoint.get("digest"),
        },
        EXPECTED_ARTIFACT,
        f"{label} exact successful-v6 artifact",
    )
    expected_artifact = {
        "entrypoint_bytes": entrypoint.get("bytes"),
        "entrypoint_digest": entrypoint.get("digest"),
        "quantization": QUANTIZATION,
        "tree_digest": artifact.get("tree_digest"),
    }
    _json_exact(declared_artifact, expected_artifact, f"{label} artifact receipt")
    calibration_artifact = _mapping(calibration.get("artifact"), f"{label} calibration artifact")
    for field, expected in expected_artifact.items():
        if calibration_artifact.get(field) != expected:
            raise ValidationRefused(f"{label} calibration artifact {field} changed")
    conversion = _mapping(receipt.get("conversion"), f"{label} conversion execution")
    replay = _mapping(conversion.get("determinism_replay"), f"{label} determinism replay")
    if (
        replay.get("schema") != "microtensor.code.gguf-determinism-replay.v1"
        or replay.get("matches_primary") is not True
        or replay.get("artifact_tree_digest") != artifact.get("tree_digest")
        or replay.get("entrypoint_digest") != entrypoint.get("digest")
        or replay.get("entrypoint_bytes") != entrypoint.get("bytes")
    ):
        raise ValidationRefused(f"{label} internal determinism replay changed")
    for field, expected in {
        "converter_digest": (
            "sha256:e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f"
        ),
        "imatrix_digest": (
            "sha256:3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35"
        ),
        "quantizer_digest": (
            "sha256:e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580"
        ),
    }.items():
        if conversion.get(field) != expected:
            raise ValidationRefused(f"{label} {field} changed")
    return (
        artifact,
        load_manifest,
        {
            "root": str(root),
            "conversion_receipt": {"bytes": len(receipt_raw), "digest": _digest_bytes(receipt_raw)},
            "calibration_receipt": {
                "bytes": len(calibration_raw),
                "digest": _digest_bytes(calibration_raw),
            },
            "load_spec": {"bytes": len(load_raw), "digest": _digest_bytes(load_raw)},
        },
    )


def _validate_conversion_bundles(spec: SpecBindings, tools: Toolset) -> ConversionBindings:
    first = _validate_conversion_bundle(
        spec.bundles[0],
        replay="replay1",
        evaluator=tools.evaluator,
    )
    second = _validate_conversion_bundle(
        spec.bundles[1],
        replay="replay2",
        evaluator=tools.evaluator,
    )
    first_artifact = dict(first[0])
    second_artifact = dict(second[0])
    first_artifact.pop("root", None)
    second_artifact.pop("root", None)
    _json_exact(first_artifact, second_artifact, "external replay artifact bytes and entrypoint")
    _json_exact(first[1], second[1], "external replay load specs")
    if first[0].get("root") != str(spec.bundles[0] / "artifact"):
        raise ValidationRefused("replay1 artifact resolved outside its declared bundle")
    return ConversionBindings(
        artifact=first[0],
        load_manifest=first[1],
        replay_receipts=(first[2], second[2]),
    )


def _validate_public_lineage(lineage: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    prepared = _mapping(lineage.get("prepared_dataset"), "public prepared dataset")
    manifest_file = _mapping(prepared.get("manifest"), "public manifest identity")
    holdout_file = _mapping(prepared.get("holdout"), "public holdout identity")
    source = _mapping(lineage.get("source_corpus"), "public source identity")
    diagnostic = _mapping(lineage.get("diagnostic_jsonl"), "public diagnostic identity")
    manifest = _mapping(prepared.get("manifest_payload"), "public manifest payload")
    expected = {
        "manifest": manifest_file.get("digest"),
        "holdout": holdout_file.get("digest"),
        "refs": diagnostic.get("refs_digest"),
        "source": _mapping(source.get("file"), "public source file").get("digest"),
    }
    _json_exact(
        expected,
        {key: EXPECTED_LINEAGE_DIGESTS[key] for key in ("manifest", "holdout", "refs", "source")},
        "public diagnostic lineage digests",
    )
    if (
        manifest.get("seed") != 92
        or manifest.get("train_examples") != 78
        or manifest.get("holdout_examples") != 16
        or diagnostic.get("examples") != 16
        or len(rows) != 16
        or len({row.get("ref") for row in rows}) != 16
        or lineage.get("public_only") is not True
        or lineage.get("hidden_or_scored_tests_accessed") is not False
    ):
        raise ValidationRefused("public diagnostic lineage shape changed")


def _validate_training_lineage(lineage: Mapping[str, Any]) -> None:
    if (
        lineage.get("status") != "provided_and_validated"
        or lineage.get("schema") != "microtensor.code.training.v5"
    ):
        raise ValidationRefused("v5 training lineage status changed")
    receipt = _mapping(lineage.get("receipt"), "v5 training receipt")
    run = _mapping(lineage.get("run"), "v5 training run")
    if receipt.get("digest") != EXPECTED_LINEAGE_DIGESTS["training_metadata"]:
        raise ValidationRefused("v5 training metadata digest changed")
    merged = _mapping(run.get("merged"), "v5 merged model tree")
    if merged.get("digest") != EXPECTED_LINEAGE_DIGESTS["merged_tree"]:
        raise ValidationRefused("v5 merged tree digest changed")


def _validate_runtime_identity(identity: Mapping[str, Any], spec: SpecBindings) -> None:
    python = _mapping(identity.get("python"), "signed runtime Python")
    executable = _mapping(python.get("executable"), "signed runtime Python executable")
    declared = _mapping(_nested(spec.payload, "diagnostic", "signed_runtime"), "signed runtime")
    lexical_executable = Path(sys.executable)
    try:
        resolved_executable = lexical_executable.resolve(strict=True)
    except OSError as exc:
        raise ValidationRefused("validator Python executable cannot be resolved") from exc
    if (
        str(lexical_executable) != declared.get("python_path")
        or str(resolved_executable) != declared.get("python_resolved_path")
        or executable.get("path") != declared.get("python_resolved_path")
        or executable.get("bytes") != declared.get("python_size_bytes")
        or executable.get("digest") != declared.get("python_sha256")
        or python.get("version") != EXPECTED_PYTHON_VERSION
        or declared.get("python_version") != "3.12.3"
    ):
        raise ValidationRefused("signed Python identity changed")
    microtensor = _mapping(identity.get("microtensor"), "signed Microtensor runtime")
    if (
        microtensor.get("release_version") != "0.3.0"
        or microtensor.get("mechanism_version") != "0.3.0"
    ):
        raise ValidationRefused("signed Microtensor identity changed")
    runtime_raw = _canonical_json_bytes(identity)
    if (
        len(runtime_raw) != EXPECTED_RUNTIME_IDENTITY_BYTES
        or _digest_bytes(runtime_raw) != EXPECTED_RUNTIME_IDENTITY_DIGEST
    ):
        raise ValidationRefused("full signed runtime identity changed")


def _prepare_context(
    spec: SpecBindings,
    tools: Toolset,
    conversion: ConversionBindings,
) -> ValidationContext:
    evaluator = tools.evaluator
    try:
        rows, evaluation_dataset = evaluator.load_public_diagnostic(
            spec.dataset,
            spec.diagnostic_jsonl,
            spec.diagnostic_source,
        )
        _validate_public_lineage(evaluation_dataset, rows)
        training_lineage, extra_modules = evaluator.load_v5_training_lineage(
            *spec.training_arguments
        )
        _validate_training_lineage(training_lineage)
        runtime = evaluator.load_signed_runtime(extra_tool_modules=extra_modules)
    except ValidationRefused:
        raise
    except Exception as exc:
        raise ValidationRefused(f"static lineage/runtime replay failed: {exc}") from exc
    _validate_runtime_identity(_mapping(runtime.identity, "signed runtime identity"), spec)
    try:
        manifest = runtime.load_manifest_type(
            format=runtime.artifact_format.GGUF,
            quantization=QUANTIZATION,
            entrypoint=ENTRYPOINT,
            max_input={"tokens": MAX_INPUT_TOKENS},
            preprocessing={"tokenizer": "tokenizer.json"},
            base_model=BASE_MODEL,
        ).to_dict()
    except Exception as exc:
        raise ValidationRefused(f"signed load-manifest construction failed: {exc}") from exc
    _json_exact(manifest, conversion.load_manifest, "signed load manifest")
    configuration = {
        "generation": evaluator.generation_contract(MAX_INPUT_TOKENS),
        "load_manifest": manifest,
        "artifact_digest": conversion.artifact["tree_digest"],
        "diagnostic_refs_digest": evaluation_dataset["diagnostic_jsonl"]["refs_digest"],
    }
    return ValidationContext(
        candidate=tools.candidate,
        evaluator=evaluator,
        rows=tuple(dict(row) for row in rows),
        evaluation_dataset=dict(evaluation_dataset),
        training_lineage=dict(training_lineage),
        runtime=runtime,
        artifact=dict(conversion.artifact),
        configuration=configuration,
        configuration_digest=_digest_bytes(_canonical_json_bytes(configuration)),
    )


def _load_results(
    path: Path,
    *,
    candidate: ModuleType,
    evaluator: ModuleType,
) -> tuple[list[dict[str, Any]], bytes]:
    raw = _stable_regular_bytes(path, "diagnostic results", maximum=MAX_JSON_BYTES)
    if not raw.endswith(b"\n"):
        raise ValidationRefused("diagnostic results must end in exactly one line terminator")
    lines = raw.splitlines()
    if len(lines) != EXPECTED_EXAMPLES or any(not line for line in lines):
        raise ValidationRefused("diagnostic results must contain exactly 16 nonblank lines")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        payload = _strict_json(line, f"diagnostic result row {number}")
        row = dict(_mapping(payload, f"diagnostic result row {number}"))
        _exact_keys(row, evaluator.RESULT_KEYS, f"diagnostic result row {number}")
        if candidate.canonical_json_bytes(row) != line:
            raise ValidationRefused(f"diagnostic result row {number} is not canonical JSON")
        rows.append(row)
    if b"".join(candidate.canonical_json_bytes(row) + b"\n" for row in rows) != raw:
        raise ValidationRefused("diagnostic result framing changed")
    return rows, raw


def _validate_result_row(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    number: int,
    context: ValidationContext,
) -> None:
    label = f"diagnostic result row {number}"
    ref = source.get("ref")
    prompt = source.get("prompt")
    reference = source.get("completion")
    if not isinstance(ref, str) or not isinstance(prompt, str) or not isinstance(reference, str):
        raise ValidationRefused("pinned diagnostic source row changed")
    if row.get("ref") != ref or row.get("ok") is not True or row.get("error") != "":
        raise ValidationRefused(f"{label} ref/success contract changed")
    if (
        type(row.get("error")) is not str
        or len(str(row["error"]).encode("utf-8")) > MAX_ERROR_BYTES
    ):
        raise ValidationRefused(f"{label} error contract changed")
    if type(row.get("max_output_tokens")) is not int or row.get("max_output_tokens") != 1024:
        raise ValidationRefused(f"{label} output-token budget changed")
    raw_output = row.get("raw_output")
    extracted = row.get("scorer_extracted_output")
    if not isinstance(raw_output, str) or not isinstance(extracted, str):
        raise ValidationRefused(f"{label} outputs must be strings")
    raw_bytes = raw_output.encode("utf-8")
    extracted_bytes = extracted.encode("utf-8")
    if len(raw_bytes) > MAX_RESULT_BYTES:
        raise ValidationRefused(f"{label} output exceeds the byte ceiling")
    expected_bindings = {
        "prompt_digest": _digest_bytes(prompt.encode("utf-8")),
        "reference_digest": _digest_bytes(reference.encode("utf-8")),
        "raw_output_digest": _digest_bytes(raw_bytes),
        "raw_output_utf8_bytes": len(raw_bytes),
        "scorer_extracted_digest": _digest_bytes(extracted_bytes),
        "scorer_extracted_utf8_bytes": len(extracted_bytes),
    }
    for field in ("raw_output_utf8_bytes", "scorer_extracted_utf8_bytes"):
        if type(row.get(field)) is not int:
            raise ValidationRefused(f"{label} {field} must be an integer")
    for field, expected in expected_bindings.items():
        if row.get(field) != expected or (
            field.endswith("_digest") and _DIGEST.fullmatch(str(row.get(field))) is None
        ):
            raise ValidationRefused(f"{label} {field} changed")
    _integer(row.get("engine_reported_output_pieces"), f"{label} stream pieces")
    for field in ("ttft_ms", "engine_total_ms", "evaluator_wall_ms", "evaluator_cpu_ms"):
        _number(row.get(field), f"{label} {field}")
    for field in ("rss_before_bytes", "rss_after_bytes"):
        value = row.get(field)
        if value is not None:
            _integer(value, f"{label} {field}")
    _integer(row.get("peak_rss_bytes"), f"{label} peak_rss_bytes")
    for field in BOOLEAN_DIAGNOSTIC_FIELDS:
        if type(row.get(field)) is not bool:
            raise ValidationRefused(f"{label} {field} must be boolean")
    similarity = _number(
        row.get("scorer_extracted_reference_text_similarity"),
        f"{label} reference similarity",
        maximum=1.0,
    )
    if not 0.0 <= similarity <= 1.0:
        raise ValidationRefused(f"{label} reference similarity is outside [0, 1]")
    try:
        recomputed = context.evaluator.structural_diagnostics(raw_output, reference)
    except Exception as exc:
        raise ValidationRefused(f"{label} static AST diagnostics failed: {exc}") from exc
    actual = {field: row[field] for field in recomputed}
    _json_exact(actual, recomputed, f"{label} structural diagnostics")


def _validate_timing(value: Any) -> dict[str, int]:
    timing = _mapping(value, "diagnostic timing")
    _exact_keys(
        timing,
        frozenset(
            {
                "started_at_unix_ns",
                "finished_at_unix_ns",
                "elapsed_ms",
                "model_load_wall_ms",
                "model_load_cpu_ms",
            }
        ),
        "diagnostic timing",
    )
    started = _integer(timing.get("started_at_unix_ns"), "diagnostic start")
    finished = _integer(timing.get("finished_at_unix_ns"), "diagnostic finish")
    if finished < started:
        raise ValidationRefused("diagnostic finish precedes its start")
    elapsed = _number(timing.get("elapsed_ms"), "diagnostic elapsed time")
    if elapsed != (finished - started) / 1_000_000:
        raise ValidationRefused("diagnostic elapsed time does not match its timestamps")
    _number(timing.get("model_load_wall_ms"), "model load wall time")
    _number(timing.get("model_load_cpu_ms"), "model load CPU time")
    return {
        "started_at_unix_ns": started,
        "finished_at_unix_ns": finished,
    }


def _validate_memory(value: Any) -> None:
    memory = _mapping(value, "diagnostic memory")
    _exact_keys(memory, frozenset({"initial", "model_loaded", "model_unloaded"}), "memory")
    for phase in ("initial", "model_loaded", "model_unloaded"):
        snapshot = _mapping(memory.get(phase), f"memory {phase}")
        _exact_keys(snapshot, frozenset({"current_bytes", "peak_bytes"}), f"memory {phase}")
        for field in ("current_bytes", "peak_bytes"):
            value = snapshot.get(field)
            if value is not None:
                _integer(value, f"memory {phase} {field}")


def _gate_metrics(rows: Sequence[Mapping[str, Any]], results: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping(results.get("structural_diagnostics"), "structural summaries")

    def count(field: str) -> int:
        return _integer(
            _mapping(diagnostics.get(field), f"structural summary {field}").get("count"),
            f"structural summary {field} count",
        )

    ttft = sorted(float(row["ttft_ms"]) for row in rows)
    position = 0.95 * (len(ttft) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = (
        ttft[lower]
        if lower == upper
        else ttft[lower] + (ttft[upper] - ttft[lower]) * (position - lower)
    )
    return {
        "examples": _integer(results.get("examples"), "result examples"),
        "successful_generations": _integer(
            results.get("successful_generations"), "successful generations"
        ),
        "failed_generations": _integer(results.get("failed_generations"), "failed generations"),
        "scorer_extracted_parseable_python": count("scorer_extracted_parseable_python"),
        "scorer_extracted_top_level_task_func": count("scorer_extracted_top_level_task_func"),
        "scorer_extracted_residual_fences": count("scorer_extracted_contains_code_fence"),
        "maximum_request_latency_ms": max(float(row["engine_total_ms"]) for row in rows),
        "p95_ttft_ms": p95,
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in rows),
        "maximum_stream_pieces_per_request": max(
            int(row["engine_reported_output_pieces"]) for row in rows
        ),
    }


def _enforce_gates(
    metrics: Mapping[str, Any],
    results: Mapping[str, Any],
    gates: Mapping[str, int],
) -> None:
    checks = (
        (metrics["examples"] == EXPECTED_EXAMPLES, "example count"),
        (
            metrics["successful_generations"] >= gates["successful_generations_minimum"],
            "successful generation minimum",
        ),
        (
            metrics["failed_generations"] <= gates["failed_generations_maximum"],
            "failed generation maximum",
        ),
        (
            results.get("failed_refs") == [],
            "failed refs must be empty",
        ),
        (
            metrics["scorer_extracted_parseable_python"]
            >= gates["scorer_extracted_parseable_python_minimum"],
            "parseable Python minimum",
        ),
        (
            metrics["scorer_extracted_top_level_task_func"]
            >= gates["scorer_extracted_top_level_task_func_minimum"],
            "top-level task_func minimum",
        ),
        (
            metrics["scorer_extracted_residual_fences"]
            <= gates["scorer_extracted_residual_fences_maximum"],
            "residual fence maximum",
        ),
        (
            metrics["maximum_request_latency_ms"] <= gates["maximum_request_latency_ms"],
            "per-request engine latency maximum",
        ),
        (metrics["p95_ttft_ms"] <= gates["p95_ttft_ms_maximum"], "p95 TTFT maximum"),
        (metrics["peak_rss_bytes"] <= gates["peak_rss_bytes_maximum"], "peak RSS maximum"),
        (
            metrics["maximum_stream_pieces_per_request"]
            <= gates["maximum_stream_pieces_per_request"],
            "per-request stream-piece maximum",
        ),
        (results.get("quality_score") is None, "quality score must remain null"),
        (results.get("execution_pass_at_1") is None, "execution pass@1 must remain null"),
    )
    for passed, label in checks:
        if not passed:
            raise ValidationRefused(f"diagnostic hard gate failed: {label}")


def _validate_repeat(
    repeat: str,
    root: Path,
    *,
    context: ValidationContext,
    gates: Mapping[str, int],
) -> dict[str, Any]:
    _require_exact_tree(
        root,
        {"summary.json": "file", "results.jsonl": "file"},
        f"diagnostic {repeat}",
    )
    rows, results_raw = _load_results(
        root / "results.jsonl",
        candidate=context.candidate,
        evaluator=context.evaluator,
    )
    for number, (row, source) in enumerate(zip(rows, context.rows, strict=True), 1):
        _validate_result_row(row, source, number=number, context=context)

    summary_payload, summary_raw = _strict_json_file(root / "summary.json", "diagnostic summary")
    summary = _mapping(summary_payload, "diagnostic summary")
    _exact_keys(summary, SUMMARY_KEYS, "diagnostic summary")
    expected_summary_raw = (
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if summary_raw != expected_summary_raw:
        raise ValidationRefused("diagnostic summary is not the evaluator's canonical JSON")
    expected_identity = {
        "schema": context.evaluator.SCHEMA,
        "status": "complete",
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "quality_claim": context.evaluator.QUALITY_CLAIM,
        "runtime_claim": context.evaluator.RUNTIME_CLAIM,
        "lineage_claim": context.evaluator.LINEAGE_CLAIM,
    }
    for field, expected in expected_identity.items():
        if summary.get(field) != expected:
            raise ValidationRefused(f"diagnostic summary field {field} changed")
    _json_exact(summary.get("safety_contract"), SAFETY_CONTRACT, "diagnostic safety contract")
    _json_exact(summary.get("configuration"), context.configuration, "diagnostic configuration")
    if summary.get("configuration_digest") != context.configuration_digest:
        raise ValidationRefused("diagnostic configuration digest changed")
    _json_exact(summary.get("artifact"), context.artifact, "diagnostic artifact binding")
    _json_exact(
        summary.get("evaluation_dataset"),
        context.evaluation_dataset,
        "diagnostic public lineage",
    )
    _json_exact(
        summary.get("training_lineage"), context.training_lineage, "diagnostic training lineage"
    )
    _json_exact(summary.get("runtime"), context.runtime.identity, "diagnostic signed runtime")
    validated_timing = _validate_timing(summary.get("timing"))
    _validate_memory(summary.get("memory"))

    results = _mapping(summary.get("results"), "diagnostic result summary")
    _exact_keys(results, RESULT_SUMMARY_KEYS, "diagnostic result summary")
    if (
        results.get("file") != "results.jsonl"
        or type(results.get("bytes")) is not int
        or results.get("bytes") != len(results_raw)
        or results.get("digest") != _digest_bytes(results_raw)
    ):
        raise ValidationRefused("diagnostic results file binding changed")
    recomputed_summary = context.evaluator.summarize_results(rows)
    declared_recomputed = {
        key: value for key, value in results.items() if key not in {"file", "bytes", "digest"}
    }
    _json_exact(declared_recomputed, recomputed_summary, "diagnostic result summaries")
    metrics = _gate_metrics(rows, results)
    _enforce_gates(metrics, results, gates)
    metrics["preferred_p95_ttft_met"] = metrics["p95_ttft_ms"] <= gates["p95_ttft_ms_preferred"]
    return {
        "repeat": repeat,
        "root": str(root),
        "summary": {"bytes": len(summary_raw), "digest": _digest_bytes(summary_raw)},
        "results": {"bytes": len(results_raw), "digest": _digest_bytes(results_raw)},
        "timing": validated_timing,
        "gates": metrics,
        "bindings": {
            "artifact": _digest_bytes(_canonical_json_bytes(context.artifact)),
            "configuration": context.configuration_digest,
            "evaluation_dataset": _digest_bytes(_canonical_json_bytes(context.evaluation_dataset)),
            "training_lineage": _digest_bytes(_canonical_json_bytes(context.training_lineage)),
            "runtime": _digest_bytes(_canonical_json_bytes(context.runtime.identity)),
        },
        "raw_output_digests": [row["raw_output_digest"] for row in rows],
    }


def _require_no_staging(roots: Sequence[Path]) -> None:
    for root in roots:
        parent = root.parent
        if not parent.exists():
            continue
        entries = _directory_entries(parent, f"diagnostic parent {parent}")
        prefix = f".{root.name}."
        conflicts = sorted(name for name in entries if name.startswith(prefix))
        if conflicts:
            raise ValidationRefused(
                f"diagnostic {root.name} has partial staging namespaces: {conflicts}"
            )


def _aggregate(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not repeats:
        raise ValidationRefused("cannot aggregate zero validated repeats")
    first_bindings = repeats[0]["bindings"]
    for receipt in repeats[1:]:
        _json_exact(receipt["bindings"], first_bindings, "cross-repeat static bindings")
    gates = [receipt["gates"] for receipt in repeats]
    raw = [receipt["raw_output_digests"] for receipt in repeats]
    if any(item != raw[0] for item in raw[1:]):
        raise ValidationRefused("cross-repeat raw output digests changed")
    previous_finished: int | None = None
    for receipt in repeats:
        timing = _mapping(receipt.get("timing"), "cross-repeat timing")
        started = _integer(timing.get("started_at_unix_ns"), "cross-repeat start")
        finished = _integer(timing.get("finished_at_unix_ns"), "cross-repeat finish")
        if finished < started:
            raise ValidationRefused("cross-repeat finish precedes its start")
        if previous_finished is not None and started < previous_finished:
            raise ValidationRefused("diagnostic repeats overlap or are out of order")
        previous_finished = finished
    all_repeats_complete = len(repeats) == len(REPEATS)
    return {
        "validated_repeats": len(repeats),
        "minimum_successful_generations": min(item["successful_generations"] for item in gates),
        "maximum_failed_generations": max(item["failed_generations"] for item in gates),
        "minimum_scorer_extracted_parseable_python": min(
            item["scorer_extracted_parseable_python"] for item in gates
        ),
        "minimum_scorer_extracted_top_level_task_func": min(
            item["scorer_extracted_top_level_task_func"] for item in gates
        ),
        "maximum_scorer_extracted_residual_fences": max(
            item["scorer_extracted_residual_fences"] for item in gates
        ),
        "worst_maximum_request_latency_ms": max(
            item["maximum_request_latency_ms"] for item in gates
        ),
        "worst_p95_ttft_ms": max(item["p95_ttft_ms"] for item in gates),
        "worst_peak_rss_bytes": max(item["peak_rss_bytes"] for item in gates),
        "worst_maximum_stream_pieces_per_request": max(
            item["maximum_stream_pieces_per_request"] for item in gates
        ),
        "preferred_p95_ttft_met_on_every_repeat": all(
            item["preferred_p95_ttft_met"] for item in gates
        ),
        "raw_outputs_identical_across_repeats": True,
        "validated_repeat_hard_gates_passed": True,
        "all_declared_local_gates_passed": all_repeats_complete,
    }


def validate_diagnostic(
    experiment_spec: Path,
    through: str,
    *,
    _tools: Toolset | None = None,
) -> dict[str, Any]:
    """Validate declared repeats through *through* without mutating any input.

    ``_tools`` exists solely for isolated tests; production callers must omit it.
    """

    if through not in REPEATS:
        raise ValidationRefused(f"through must be one of {REPEATS}")
    spec = _load_spec(experiment_spec)
    tools = _tools if _tools is not None else _load_pinned_tools(spec.source_root)
    conversion = _validate_conversion_bundles(spec, tools)
    context = _prepare_context(spec, tools, conversion)
    completed = REPEATS.index(through) + 1
    _require_no_staging(spec.output_roots)
    for root in spec.output_roots[completed:]:
        if os.path.lexists(root):
            raise ValidationRefused(
                f"future diagnostic root already exists before {through} validation: {root}"
            )
    receipts = [
        _validate_repeat(
            repeat,
            spec.output_roots[index],
            context=context,
            gates=spec.gates,
        )
        for index, repeat in enumerate(REPEATS[:completed])
    ]
    aggregate = _aggregate(receipts)
    for receipt in receipts:
        receipt.pop("raw_output_digests")
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "validated" if completed == len(REPEATS) else "partially_validated",
        "through": through,
        "experiment_spec": {"bytes": len(spec.raw), "digest": _digest_bytes(spec.raw)},
        "conversion": {
            "artifact_tree_digest": conversion.artifact["tree_digest"],
            "entrypoint_digest": conversion.artifact["entrypoint"]["digest"],
            "load_manifest_digest": _digest_bytes(_canonical_json_bytes(conversion.load_manifest)),
            "external_replays": list(conversion.replay_receipts),
        },
        "repeats": receipts,
        "aggregate": aggregate,
        "claim": {
            "local_structural_diagnostics_only": True,
            "quality_or_rank_claimed": False,
            "promotion_authorized": False,
            "remaining_local_repeats": list(REPEATS[completed:]),
            "remaining_external_gates": [
                "official namespace sandbox selfcheck",
                "authorized immutable publication",
                "coherent coordinator round",
                "official validator measurement and settled rank",
            ],
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    parser.add_argument("--through", choices=REPEATS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if __package__:
        print(
            "code GGUF diagnostic validation refused: invoke this validator by direct path",
            file=sys.stderr,
        )
        return 2
    args = _parse_args(argv)
    try:
        report = validate_diagnostic(args.experiment_spec, args.through)
    except (ValidationRefused, OSError, ValueError) as exc:
        print(f"code GGUF diagnostic validation refused: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
