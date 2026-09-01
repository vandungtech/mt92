#!/usr/bin/env python3
"""Fail-closed, non-executing structural diagnostics for a code GGUF.

The evaluator accepts only the exact public BigCodeBench-94 seed-92 diagnostic
split and replays that split from its pinned public source before inference. It
uses the signed Microtensor v0.3 GGUF engine with raw prompts and its exact
single-threaded CPU sampler. Generated and reference code is parsed only as
text/AST: it is never imported, executed, or compiled to Python bytecode.

No metric produced here is execution pass@1. The results are local structural
diagnostics on public training-projection data, not an official validator
certificate or a certified-device measurement.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import struct
import sys
import tempfile
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import historical_code_candidate as historical_candidate
    from training import normalized_historical_code_candidate as normalized_historical_candidate
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import historical_code_candidate as historical_candidate  # type: ignore[no-redef]
    import normalized_historical_code_candidate as normalized_historical_candidate  # type: ignore[no-redef]


SCHEMA: Final[str] = "microtensor.code.gguf-structural-evaluation.v1"
SCHEMA_V2: Final[str] = "microtensor.code.gguf-structural-evaluation.v2"
TRAINING_SCHEMA_V4: Final[str] = "microtensor.code.training.v4"
TRAINING_SCHEMA_V5: Final[str] = "microtensor.code.training.v5"
TRAINING_SCHEMA_V6: Final[str] = "microtensor.code.training.v6"
SIGNED_RELEASE_VERSION: Final[str] = "0.3.2"
SIGNED_MECHANISM_VERSION: Final[str] = "0.3.0"
ENGINE_ADAPTER_VERSION: Final[str] = "0.2.0"
LLAMA_CPP_VERSION: Final[str] = "0.3.35"
BASE_MODEL: Final[str] = candidate.QWEN3_BASE_MODEL
QWEN25_BASE_MODELS: Final[frozenset[str]] = frozenset(
    {
        candidate.RECOMMENDED_BASE_MODEL,
        candidate.QWEN25_CODER_1_5B_BASE_MODEL,
    }
)
GGUF_ARCHITECTURE_BY_BASE_MODEL: Final[dict[str, str]] = {
    candidate.RECOMMENDED_BASE_MODEL: "qwen2",
    candidate.QWEN25_CODER_1_5B_BASE_MODEL: "qwen2",
    candidate.QWEN3_BASE_MODEL: "qwen3",
}
CURRENT94_PUBLIC_CORPUS_BYTES: Final[int] = 152_605
CURRENT94_PUBLIC_CORPUS_RAW_DIGEST: Final[str] = (
    "sha256:1c37a0e212936bfac8c86f955ad61fd378f58603413b45ece88382d528ace9d5"
)
CURRENT94_FINAL_TRAINING_QUALITY_CLAIM: Final[str] = (
    "none: all 94 public examples were used for training; public code tests are withheld; "
    "no holdout or execution pass@1 was measured"
)
THREADS: Final[int] = 1
GPU_LAYERS: Final[int] = 0
SEED: Final[int] = 0
DIAGNOSTIC_SEED: Final[int] = 92
DIAGNOSTIC_TRAIN_EXAMPLES: Final[int] = 78
DIAGNOSTIC_HOLDOUT_EXAMPLES: Final[int] = 16
MIN_CONTEXT_TOKENS: Final[int] = 512
MAX_CONTEXT_TOKENS: Final[int] = 4096
MAX_RESULT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_ERROR_BYTES: Final[int] = 4096
MAX_JSON_RECEIPT_BYTES: Final[int] = 16 * 1024 * 1024
QUALITY_CLAIM: Final[str] = (
    "none: generated and reference code were never executed; public structural "
    "diagnostics are not execution pass@1"
)
RUNTIME_CLAIM: Final[str] = (
    "exact signed-v0.3 GGUF engine configuration, but a local structural diagnostic; "
    "only an official validator run on its certified device is authoritative"
)
LINEAGE_CLAIM: Final[str] = (
    "the exact current94 public holdout is a diagnostic lineage separate from optional "
    "historical8000 all-public training; no execution pass@1 is claimed"
)
NORMALIZED_LINEAGE_CLAIM: Final[str] = (
    "the exact current94 public holdout is a diagnostic lineage separate from optional "
    "normalized historical all-public training; no execution pass@1 is claimed"
)
CURRENT_OVERLAP_LINEAGE_CLAIM: Final[str] = (
    "the exact current94 public diagnostic rows are a subset of the exact current94 94/0 "
    "all-public training lineage; they are training-overlap structural and timing diagnostics "
    "only, not holdout evidence or execution pass@1"
)
NO_TRAINING_LINEAGE_CLAIM: Final[str] = (
    "no training receipt was supplied; this evaluation binds GGUF bytes and public "
    "diagnostic inputs only"
)
GENERATION_NONCE: Final[str] = "public-structural-diagnostic"
SUPPORTED_QUANTIZATIONS: Final[dict[str, int]] = {
    "Q4_K_M": 15,
    "Q5_K_M": 17,
    "Q8_0": 7,
}
PINNED_RUNTIME_SOURCE_DIGESTS: Final[dict[str, str]] = {
    "microtensor.core.constants": (
        "sha256:42b653f317667b9d264c38dc42448a04c4a3244081fe80117077e574ecd0e87b"
    ),
    "microtensor.core.hashing": (
        "sha256:59951fc1f7063b2e0b538d8eb02917d91e09e821b9a6573ffa9214cb39c66afb"
    ),
    "microtensor.core.protocol": (
        "sha256:56ebe17405757bae2c6c7f78fc292db22dbd865a9cc2da7f36bc1ab36ab90989"
    ),
    "microtensor.core.tracks": (
        "sha256:591bb0eae758d2fc37843f5f5512559532e69da2e9dbd8ba296d7dd1995793ce"
    ),
    "microtensor.harness.contract": (
        "sha256:64ef3e54b689ae5dcd0c36612cde88fb6d2a37717e71bd1ef8cfd1c2c2d03b18"
    ),
    "microtensor.harness.engines.gguf": (
        "sha256:933d0fbaf276a32d8b96608fd2efd145e985188266bd86aa41afc096cb6d009e"
    ),
}
SCORER_FENCE: Final[re.Pattern[str]] = re.compile(
    r"\x60\x60\x60(?:python|py)?\s*\n(.*?)\x60\x60\x60",
    re.DOTALL,
)
TRAINING_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "status",
        "run_kind",
        "hotkey",
        "track",
        "hardware_class",
        "base_model",
        "base_snapshot",
        "corpus_version",
        "dataset",
        "settings",
        "target",
        "token_summary",
        "runtime",
        "upstream_compatibility",
        "quality_claim",
        "selection",
        "started_at_unix",
        "holdout_diagnostics",
        "finished_at_unix",
        "elapsed_s",
        "updates",
        "metrics_digest",
        "adapter",
        "merged",
    }
)
RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ref",
        "ok",
        "error",
        "prompt_digest",
        "reference_digest",
        "max_output_tokens",
        "raw_output",
        "raw_output_digest",
        "raw_output_utf8_bytes",
        "engine_reported_output_pieces",
        "ttft_ms",
        "engine_total_ms",
        "evaluator_wall_ms",
        "evaluator_cpu_ms",
        "rss_before_bytes",
        "rss_after_bytes",
        "peak_rss_bytes",
        "scorer_extracted_output",
        "scorer_extracted_digest",
        "scorer_extracted_utf8_bytes",
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
        "scorer_extracted_reference_text_similarity",
    }
)


class EvaluationRefused(ValueError):
    """The diagnostic contract is incomplete, unsupported, or changed."""


@dataclass(frozen=True)
class RuntimeBindings:
    """Non-serializable signed-runtime objects plus their serializable identity."""

    artifact_format: Any
    decoding: Any
    engine_type: Any
    load_manifest_type: Any
    request_type: Any
    gguf_module: ModuleType
    identity: dict[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationRefused(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EvaluationRefused(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise EvaluationRefused(f"{label} contains a non-string key")
    found = frozenset(value)
    if found != expected:
        raise EvaluationRefused(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(found)}"
        )


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationRefused(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise EvaluationRefused(f"{label} must be positive")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationRefused(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvaluationRefused(f"{label} must be finite and non-negative")
    return result


def _regular_file(path: Path, label: str, *, maximum_bytes: int | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EvaluationRefused(f"{label} must be a regular non-symlink file: {path}")
    try:
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError as exc:
        raise EvaluationRefused(f"{label} cannot be inspected: {exc}") from exc
    if maximum_bytes is not None and size > maximum_bytes:
        raise EvaluationRefused(f"{label} exceeds the {maximum_bytes}-byte limit")
    return resolved


def file_identity(path: Path, label: str) -> dict[str, Any]:
    """Hash one stable regular file and refuse an in-hash mutation."""

    source = _regular_file(path, label)
    before = source.stat()
    digest = candidate.digest_file(source)
    after = source.stat()
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_key != after_key:
        raise EvaluationRefused(f"{label} changed while it was hashed")
    return {
        "path": str(source),
        "bytes": after.st_size,
        "digest": digest,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": oct(after.st_mode & 0o7777),
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }


def _same_content_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("bytes") == right.get("bytes") and left.get("digest") == right.get("digest")


def _read_strict_json(path: Path, label: str, *, maximum_bytes: int) -> tuple[Any, bytes]:
    source = _regular_file(path, label, maximum_bytes=maximum_bytes)
    try:
        raw = source.read_bytes()
        payload = candidate._strict_json(raw, str(source))
    except (OSError, candidate.CandidateError) as exc:
        raise EvaluationRefused(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    return payload, raw


def extract_scorer_code(text: str) -> str:
    """Mirror the audited fence extraction without executing the result."""

    if not isinstance(text, str):
        raise EvaluationRefused("completion must be a string")
    blocks = SCORER_FENCE.findall(text)
    return max(blocks, key=len).strip() if blocks else text.strip()


def _inspect_source(source: str) -> tuple[ast.Module | None, bool, bool]:
    if not source.strip():
        return None, False, False
    try:
        tree = ast.parse(source, filename="<static-diagnostic>", mode="exec")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None, False, False
    has_task_func = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == candidate.EXPECTED_ENTRY_POINT
        for node in tree.body
    )
    has_solution_class = any(
        isinstance(node, ast.ClassDef) and node.name == "Solution" for node in tree.body
    )
    return tree, has_task_func, has_solution_class


def structural_diagnostics(completion: str, reference: str) -> dict[str, Any]:
    """Return text and AST-only diagnostics; never import or execute either input."""

    if not isinstance(completion, str) or not isinstance(reference, str):
        raise EvaluationRefused("completion and reference must be strings")
    normalized_reference = reference.strip()
    reference_tree, _reference_task, _reference_solution = _inspect_source(normalized_reference)
    if reference_tree is None:
        raise EvaluationRefused("public diagnostic reference is not statically parseable Python")
    raw_tree, raw_task_func, raw_solution_class = _inspect_source(completion)
    extracted = extract_scorer_code(completion)
    extracted_tree, extracted_task_func, extracted_solution_class = _inspect_source(extracted)
    extracted_bytes = extracted.encode("utf-8")
    fence = chr(96) * 3
    return {
        "scorer_extracted_output": extracted,
        "scorer_extracted_digest": candidate.digest_bytes(extracted_bytes),
        "scorer_extracted_utf8_bytes": len(extracted_bytes),
        "scorer_extraction_changed": extracted != completion,
        "raw_contains_code_fence": fence in completion,
        "raw_contains_thinking_markup": any(
            marker in completion for marker in ("<think>", "</think>")
        ),
        "raw_nonempty": bool(completion.strip()),
        "raw_parseable_python": raw_tree is not None,
        "raw_top_level_task_func": raw_task_func,
        "raw_solution_class": raw_solution_class,
        "scorer_extracted_contains_code_fence": fence in extracted,
        "scorer_extracted_contains_thinking_markup": any(
            marker in extracted for marker in ("<think>", "</think>")
        ),
        "scorer_extracted_nonempty": bool(extracted),
        "scorer_extracted_parseable_python": extracted_tree is not None,
        "scorer_extracted_top_level_task_func": extracted_task_func,
        "scorer_extracted_solution_class": extracted_solution_class,
        "scorer_extracted_exact_reference_text": extracted == normalized_reference,
        "scorer_extracted_exact_reference_ast": bool(
            extracted_tree is not None
            and ast.dump(extracted_tree, include_attributes=False)
            == ast.dump(reference_tree, include_attributes=False)
        ),
        "scorer_extracted_reference_text_similarity": difflib.SequenceMatcher(
            None,
            extracted,
            normalized_reference,
            autojunk=False,
        ).ratio(),
    }


_GGUF_FIXED_WIDTH: Final[dict[int, int]] = {
    0: 1,
    1: 1,
    7: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    10: 8,
    11: 8,
    12: 8,
}
_GGUF_STRING: Final[int] = 8
_GGUF_ARRAY: Final[int] = 9
_MAX_GGUF_STRING_BYTES: Final[int] = 1 << 20
_MAX_GGUF_ARRAY_ITEMS: Final[int] = 1 << 20
_MAX_GGUF_METADATA_ITEMS: Final[int] = 100_000


def _read_exact(handle: Any, size: int, label: str) -> bytes:
    raw = handle.read(size)
    if len(raw) != size:
        raise EvaluationRefused(f"GGUF ends inside {label}")
    return raw


def _read_u32(handle: Any, label: str) -> int:
    return int(struct.unpack("<I", _read_exact(handle, 4, label))[0])


def _read_u64(handle: Any, label: str) -> int:
    return int(struct.unpack("<Q", _read_exact(handle, 8, label))[0])


def _read_gguf_string(handle: Any, label: str) -> str:
    size = _read_u64(handle, f"{label} length")
    if size > _MAX_GGUF_STRING_BYTES:
        raise EvaluationRefused(f"GGUF {label} is implausibly large")
    try:
        return _read_exact(handle, size, label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationRefused(f"GGUF {label} is not UTF-8") from exc


def _skip_gguf_bytes(handle: Any, size: int, file_size: int, label: str) -> None:
    if size < 0 or handle.tell() + size > file_size:
        raise EvaluationRefused(f"GGUF {label} extends beyond the artifact")
    handle.seek(size, os.SEEK_CUR)


def _skip_gguf_value(handle: Any, kind: int, file_size: int, label: str) -> None:
    width = _GGUF_FIXED_WIDTH.get(kind)
    if width is not None:
        _skip_gguf_bytes(handle, width, file_size, label)
        return
    if kind == _GGUF_STRING:
        _read_gguf_string(handle, label)
        return
    if kind == _GGUF_ARRAY:
        element_kind = _read_u32(handle, f"{label} array type")
        count = _read_u64(handle, f"{label} array count")
        if count > _MAX_GGUF_ARRAY_ITEMS or element_kind == _GGUF_ARRAY:
            raise EvaluationRefused(f"GGUF {label} has an unsupported array")
        if element_kind == _GGUF_STRING:
            for index in range(count):
                _read_gguf_string(handle, f"{label}[{index}]")
            return
        element_width = _GGUF_FIXED_WIDTH.get(element_kind)
        if element_width is None:
            raise EvaluationRefused(f"GGUF {label} has unknown array type {element_kind}")
        _skip_gguf_bytes(handle, element_width * count, file_size, label)
        return
    raise EvaluationRefused(f"GGUF {label} has unknown value type {kind}")


def read_gguf_identity(
    path: Path,
    *,
    expected_architecture: str = "qwen3",
) -> dict[str, Any]:
    """Read only bounded GGUF metadata required to bind architecture/quantization."""

    if expected_architecture not in frozenset(GGUF_ARCHITECTURE_BY_BASE_MODEL.values()):
        raise EvaluationRefused(
            f"unsupported expected GGUF architecture {expected_architecture!r}"
        )
    source = _regular_file(path, "GGUF entrypoint")
    file_size = source.stat().st_size
    with source.open("rb") as handle:
        if _read_exact(handle, 4, "magic") != b"GGUF":
            raise EvaluationRefused("artifact entrypoint is not a GGUF file")
        version = _read_u32(handle, "version")
        if version not in {2, 3}:
            raise EvaluationRefused(f"GGUF version {version} is unsupported")
        tensor_count = _read_u64(handle, "tensor count")
        metadata_count = _read_u64(handle, "metadata count")
        if metadata_count > _MAX_GGUF_METADATA_ITEMS:
            raise EvaluationRefused("GGUF declares too many metadata fields")
        found: dict[str, Any] = {}
        seen: set[str] = set()
        for index in range(metadata_count):
            key = _read_gguf_string(handle, f"metadata key {index}")
            if key in seen:
                raise EvaluationRefused(f"GGUF repeats metadata key {key!r}")
            seen.add(key)
            kind = _read_u32(handle, f"metadata value type for {key!r}")
            if key == "general.architecture":
                if kind != _GGUF_STRING:
                    raise EvaluationRefused("GGUF general.architecture is not a string")
                found[key] = _read_gguf_string(handle, key)
            elif key == "general.file_type":
                if kind == 4:
                    found[key] = _read_u32(handle, key)
                elif kind == 5:
                    found[key] = int(struct.unpack("<i", _read_exact(handle, 4, key))[0])
                elif kind == 10:
                    found[key] = _read_u64(handle, key)
                elif kind == 11:
                    found[key] = int(struct.unpack("<q", _read_exact(handle, 8, key))[0])
                else:
                    raise EvaluationRefused("GGUF general.file_type is not an integer")
            else:
                _skip_gguf_value(handle, kind, file_size, key)
    if found.get("general.architecture") != expected_architecture:
        raise EvaluationRefused(
            "GGUF architecture does not match the lineage-derived base architecture: "
            f"expected {expected_architecture!r}, got "
            f"{found.get('general.architecture')!r}"
        )
    if "general.file_type" not in found:
        raise EvaluationRefused("GGUF declares no general.file_type")
    return {
        "magic": "GGUF",
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "architecture": found["general.architecture"],
        "file_type": found["general.file_type"],
    }


def _relative_entrypoint(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvaluationRefused("entrypoint must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EvaluationRefused("entrypoint must be a normalized relative POSIX path")
    if unicodedata.normalize("NFC", value) != value:
        raise EvaluationRefused("entrypoint must be NFC-normalized")
    return path


def artifact_identity(
    root: Path,
    *,
    entrypoint: str,
    expected_digest: str,
    quantization: str,
    expected_architecture: str = "qwen3",
) -> dict[str, Any]:
    """Return the official tree digest plus every file identity."""

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest):
        raise EvaluationRefused("artifact digest must be lowercase sha256:<64 hex>")
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise EvaluationRefused(f"unsupported quantization {quantization!r}")
    if root.is_symlink() or not root.is_dir():
        raise EvaluationRefused("artifact must be a regular non-symlink directory")
    artifact_root = root.resolve(strict=True)
    relative_entrypoint = _relative_entrypoint(entrypoint)
    files: list[dict[str, Any]] = []
    entries: list[tuple[str, str]] = []
    normalized_names: set[str] = set()
    for path in sorted(
        artifact_root.rglob("*"),
        key=lambda item: item.relative_to(artifact_root).as_posix(),
    ):
        relative = path.relative_to(artifact_root).as_posix()
        if path.is_symlink():
            raise EvaluationRefused(f"artifact contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvaluationRefused(f"artifact contains a special file: {relative}")
        normalized = unicodedata.normalize("NFC", relative)
        if normalized in normalized_names:
            raise EvaluationRefused("artifact has colliding NFC-normalized paths")
        if normalized != relative:
            raise EvaluationRefused(f"artifact path is not NFC-normalized: {relative}")
        normalized_names.add(normalized)
        identity = file_identity(path, f"artifact file {relative}")
        files.append({"path": relative, "bytes": identity["bytes"], "digest": identity["digest"]})
        entries.append((relative, str(identity["digest"])))
    if not files:
        raise EvaluationRefused("artifact directory is empty")
    total_bytes = sum(int(item["bytes"]) for item in files)
    if total_bytes > candidate.MAX_SELECTED_ARTIFACT_BYTES:
        raise EvaluationRefused("artifact tree exceeds the local selected-artifact byte ceiling")
    official = hashlib.sha256()
    for relative, digest in sorted(entries):
        official.update(unicodedata.normalize("NFC", relative).encode("utf-8"))
        official.update(b"\0")
        official.update(digest.encode("ascii"))
        official.update(b"\0")
    tree_digest = "sha256:" + official.hexdigest()
    if tree_digest != expected_digest:
        raise EvaluationRefused(
            f"artifact tree digest changed: expected {expected_digest}, got {tree_digest}"
        )
    model_path = artifact_root.joinpath(*relative_entrypoint.parts)
    model = file_identity(model_path, "GGUF entrypoint")
    if model["bytes"] > candidate.MAX_SELECTED_ARTIFACT_BYTES:
        raise EvaluationRefused("GGUF exceeds the local selected-artifact byte ceiling")
    header = read_gguf_identity(
        model_path,
        expected_architecture=expected_architecture,
    )
    expected_file_type = SUPPORTED_QUANTIZATIONS[quantization]
    if header["file_type"] != expected_file_type:
        raise EvaluationRefused(
            "GGUF general.file_type does not match the declared quantization: "
            f"expected {expected_file_type}, got {header['file_type']}"
        )
    return {
        "root": str(artifact_root),
        "tree_algorithm": "sorted_nfc_relative_path_nul_sha256_nul_v1",
        "tree_digest": tree_digest,
        "total_bytes": total_bytes,
        "files": files,
        "entrypoint": {
            "path": relative_entrypoint.as_posix(),
            "bytes": model["bytes"],
            "digest": model["digest"],
            "gguf": header,
        },
    }


def _load_explicit_rows(path: Path) -> list[dict[str, Any]]:
    source = _regular_file(
        path,
        "public diagnostic JSONL",
        maximum_bytes=MAX_JSON_RECEIPT_BYTES,
    )
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationRefused(f"public diagnostic JSONL is unreadable: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise EvaluationRefused("public diagnostic JSONL must contain no blank lines")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            payload = candidate._strict_json(
                line.encode("utf-8"),
                f"{source}:{number}",
            )
        except candidate.CandidateError as exc:
            raise EvaluationRefused(f"public diagnostic JSONL is invalid: {exc}") from exc
        row = _mapping(payload, f"public diagnostic row {number}")
        _exact_keys(row, candidate.PREPARED_ROW_KEYS, f"public diagnostic row {number}")
        ref = row.get("ref")
        if not isinstance(ref, str) or candidate.REF_PATTERN.fullmatch(ref) is None:
            raise EvaluationRefused(f"public diagnostic row {number} has an invalid ref")
        for field in ("prompt", "completion"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise EvaluationRefused(f"public diagnostic row {number} has an invalid {field}")
        if row.get("max_output_tokens") != 1024:
            raise EvaluationRefused(
                f"public diagnostic row {number} changed its output-token budget"
            )
        rows.append(dict(row))
    return rows


def validate_explicit_diagnostic_rows(
    explicit: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require the explicit JSONL to equal the entire pinned public holdout."""

    rows = [dict(row) for row in explicit]
    if rows != [dict(row) for row in expected]:
        raise EvaluationRefused(
            "explicit public diagnostic JSONL is not the exact prepared holdout"
        )
    refs = [str(row["ref"]) for row in rows]
    if len(refs) != DIAGNOSTIC_HOLDOUT_EXAMPLES or len(set(refs)) != len(refs):
        raise EvaluationRefused("public diagnostic JSONL refs are incomplete or duplicated")
    return rows


def _project_current94(
    payload: Mapping[str, Any],
    *,
    seed: int,
    holdout_examples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project the pinned public tasks without executing their code."""

    tasks = list(_sequence(payload.get("tasks"), "public corpus tasks"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationRefused("current94 projection seed must be an integer")
    if (
        isinstance(holdout_examples, bool)
        or not isinstance(holdout_examples, int)
        or not 0 <= holdout_examples < candidate.EXPECTED_COUNTS["train"]
    ):
        raise EvaluationRefused("current94 projection holdout count is invalid")
    ranked = sorted(
        tasks,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['ref']}".encode()).hexdigest(),
            str(row["ref"]),
        ),
    )
    heldout_refs = {str(row["ref"]) for row in ranked[:holdout_examples]}
    expected_train: list[dict[str, Any]] = []
    expected_holdout: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: str(row["ref"])):
        inputs = _mapping(task.get("inputs"), f"public task {task.get('ref')} inputs")
        record = {
            "completion": str(inputs["code_prompt"]) + str(task["gold"]),
            "max_output_tokens": task["max_output_tokens"],
            "prompt": task["prompt"],
            "ref": task["ref"],
        }
        (expected_holdout if task["ref"] in heldout_refs else expected_train).append(record)
    return expected_train, expected_holdout


def _replay_current94(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = manifest.get("seed")
    holdout_examples = manifest.get("holdout_examples")
    if seed != DIAGNOSTIC_SEED or holdout_examples != DIAGNOSTIC_HOLDOUT_EXAMPLES:
        raise EvaluationRefused("diagnostic split is not the exact seed-92 78/16 split")
    return _project_current94(
        payload,
        seed=seed,
        holdout_examples=holdout_examples,
    )


def load_public_diagnostic(
    dataset: Path,
    diagnostic_jsonl: Path,
    source_corpus: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay and bind the exact public current94 diagnostic lineage."""

    dataset_root = candidate.assert_tmpfs_path(dataset, must_exist=True)
    diagnostic_path = candidate.assert_tmpfs_path(diagnostic_jsonl, must_exist=True)
    source_path = candidate.assert_tmpfs_path(source_corpus, must_exist=True)
    payload, validation = candidate.load_public_corpus(source_path)
    train_rows, manifest = candidate.load_prepared_dataset(dataset_root)
    strict_manifest, _manifest_raw = _read_strict_json(
        dataset_root / "manifest.json",
        "prepared manifest",
        maximum_bytes=MAX_JSON_RECEIPT_BYTES,
    )
    if strict_manifest != manifest:
        raise EvaluationRefused("prepared manifest strict replay changed")
    holdout_rows = candidate._load_prepared_rows(
        dataset_root / "holdout.jsonl",
        "holdout.jsonl",
    )
    if (
        manifest.get("seed"),
        manifest.get("train_examples"),
        manifest.get("holdout_examples"),
    ) != (
        DIAGNOSTIC_SEED,
        DIAGNOSTIC_TRAIN_EXAMPLES,
        DIAGNOSTIC_HOLDOUT_EXAMPLES,
    ):
        raise EvaluationRefused("diagnostic dataset is not the exact seed-92 78/16 split")
    expected_train, expected_holdout = _replay_current94(payload, manifest)
    if train_rows != expected_train or holdout_rows != expected_holdout:
        raise EvaluationRefused("prepared diagnostic dataset is not an exact source replay")
    source_identity = file_identity(source_path, "public source corpus")
    if manifest.get("source_file_digest") != source_identity["digest"]:
        raise EvaluationRefused("prepared dataset does not bind the supplied source bytes")
    explicit_identity = file_identity(diagnostic_path, "public diagnostic JSONL")
    if explicit_identity["digest"] != manifest.get("holdout_file_digest"):
        raise EvaluationRefused("explicit diagnostic JSONL digest is not the prepared holdout")
    explicit_rows = validate_explicit_diagnostic_rows(
        _load_explicit_rows(diagnostic_path),
        holdout_rows,
    )
    manifest_identity = file_identity(dataset_root / "manifest.json", "prepared manifest")
    train_identity = file_identity(dataset_root / "train.jsonl", "prepared train JSONL")
    holdout_identity = file_identity(
        dataset_root / "holdout.jsonl",
        "prepared holdout JSONL",
    )
    return explicit_rows, {
        "profile": "bigcodebench94-current-public-diagnostic",
        "source_corpus": {
            "file": source_identity,
            "corpus_version": candidate.CORPUS_VERSION,
            "canonical_bytes": validation.canonical_bytes,
            "canonical_digest": validation.canonical_digest,
            "task_count": validation.task_count,
            "refs_digest": validation.refs_digest,
        },
        "prepared_dataset": {
            "manifest": manifest_identity,
            "train": train_identity,
            "holdout": holdout_identity,
            "manifest_payload": manifest,
        },
        "diagnostic_jsonl": {
            **explicit_identity,
            "examples": len(explicit_rows),
            "refs_digest": candidate._refs_digest([str(row["ref"]) for row in explicit_rows]),
        },
        "public_only": True,
        "hidden_or_scored_tests_accessed": False,
    }


def validate_v4_receipt_header(
    payload: Any,
    *,
    training_manifest: Mapping[str, Any],
    training_manifest_digest: str,
) -> dict[str, Any]:
    """Validate an exact current94 final-training header before deep validation."""

    metadata = _mapping(payload, "training metadata")
    _exact_keys(metadata, TRAINING_METADATA_KEYS, "training metadata")
    base_model = metadata.get("base_model")
    if not isinstance(base_model, str) or base_model not in QWEN25_BASE_MODELS:
        raise EvaluationRefused("v4 training metadata does not bind a pinned Qwen2.5 base")
    required = {
        "schema": TRAINING_SCHEMA_V4,
        "status": "complete",
        "run_kind": "final_all_public",
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "corpus_version": candidate.CORPUS_VERSION,
        "quality_claim": CURRENT94_FINAL_TRAINING_QUALITY_CLAIM,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise EvaluationRefused(f"v4 training metadata field {key!r} changed")
    dataset = _mapping(metadata.get("dataset"), "v4 training dataset")
    _exact_keys(
        dataset,
        frozenset({"manifest", "manifest_digest"}),
        "v4 training dataset",
    )
    if dataset.get("manifest") != training_manifest:
        raise EvaluationRefused("v4 training metadata does not bind the prepared manifest")
    if dataset.get("manifest_digest") != training_manifest_digest:
        raise EvaluationRefused("v4 training metadata prepared-manifest digest changed")
    manifest_required = {
        "schema": candidate.DATASET_SCHEMA,
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "corpus_version": candidate.CORPUS_VERSION,
        "corpus_canonical_digest": candidate.PUBLIC_CORPUS_CANONICAL_DIGEST,
        "split_algorithm": candidate.SPLIT_ALGORITHM,
        "seed": DIAGNOSTIC_SEED,
        "train_examples": candidate.EXPECTED_COUNTS["train"],
        "holdout_examples": 0,
        "target_construction": "inputs.code_prompt + gold",
        "quality_claim": candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }
    for key, expected in manifest_required.items():
        if training_manifest.get(key) != expected:
            raise EvaluationRefused(f"v4 training manifest field {key!r} changed")
    return dict(metadata)


def load_v4_training_lineage(
    run: Path,
    dataset: Path,
    source_corpus: Path,
    base: Path,
) -> tuple[dict[str, Any], tuple[ModuleType, ...]]:
    """Replay exact current94 source bytes and deeply validate one v4 run."""

    run_root = candidate.assert_tmpfs_path(run, must_exist=True)
    dataset_root = candidate.assert_tmpfs_path(dataset, must_exist=True)
    source_root = candidate.assert_tmpfs_path(source_corpus, must_exist=True)
    base_root = candidate.assert_tmpfs_path(base, must_exist=True)
    if run_root.is_symlink() or not run_root.is_dir():
        raise EvaluationRefused("training run must be a regular non-symlink directory")
    try:
        payload, source_validation = candidate.load_public_corpus(source_root)
        train_rows, manifest = candidate.load_prepared_dataset(dataset_root)
        strict_manifest, _manifest_raw = _read_strict_json(
            dataset_root / "manifest.json",
            "v4 prepared manifest",
            maximum_bytes=MAX_JSON_RECEIPT_BYTES,
        )
        if strict_manifest != manifest:
            raise EvaluationRefused("v4 prepared manifest strict replay changed")
        holdout_rows = candidate._load_prepared_rows(
            dataset_root / "holdout.jsonl",
            "holdout.jsonl",
        )
        if (len(train_rows), len(holdout_rows)) != (
            candidate.EXPECTED_COUNTS["train"],
            0,
        ):
            raise EvaluationRefused("v4 training source replay is not exactly 94/0")
        expected_train, expected_holdout = _project_current94(
            payload,
            seed=DIAGNOSTIC_SEED,
            holdout_examples=0,
        )
        if train_rows != expected_train or holdout_rows != expected_holdout:
            raise EvaluationRefused("v4 prepared training dataset is not an exact source replay")

        manifest_identity = file_identity(
            dataset_root / "manifest.json",
            "v4 prepared manifest",
        )
        train_identity = file_identity(
            dataset_root / "train.jsonl",
            "v4 prepared train JSONL",
        )
        holdout_identity = file_identity(
            dataset_root / "holdout.jsonl",
            "v4 prepared holdout JSONL",
        )
        source_identity = file_identity(source_root, "current94 public source corpus")
        if (
            source_identity["bytes"] != CURRENT94_PUBLIC_CORPUS_BYTES
            or source_identity["digest"] != CURRENT94_PUBLIC_CORPUS_RAW_DIGEST
        ):
            raise EvaluationRefused("v4 public source corpus differs from the exact raw response")
        if manifest.get("source_file_digest") != source_identity["digest"]:
            raise EvaluationRefused("v4 prepared dataset does not bind the supplied source bytes")
        if train_identity["digest"] != manifest.get("train_file_digest"):
            raise EvaluationRefused("v4 prepared train identity differs from its manifest")
        if holdout_identity["digest"] != manifest.get("holdout_file_digest"):
            raise EvaluationRefused("v4 prepared holdout identity differs from its manifest")

        metadata_path = run_root / "training_metadata.json"
        metadata, _raw = _read_strict_json(
            metadata_path,
            "v4 training metadata",
            maximum_bytes=MAX_JSON_RECEIPT_BYTES,
        )
        validated_header = validate_v4_receipt_header(
            metadata,
            training_manifest=manifest,
            training_manifest_digest=str(manifest_identity["digest"]),
        )
        base_model = str(validated_header["base_model"])
        base_identity = candidate.verify_base_snapshot(
            base_root,
            expected_model=base_model,
        )
        evaluate_code = importlib.import_module("training.evaluate_code")
        train_code = importlib.import_module("training.train_code")
        if getattr(train_code, "SCHEMA", None) != TRAINING_SCHEMA_V4:
            raise EvaluationRefused("shared trainer no longer recognizes the v4 receipt")
        model_identity = evaluate_code._load_training_run(
            run_root / "merged",
            dataset_manifest=manifest,
            dataset_manifest_digest=str(manifest_identity["digest"]),
            base_identity=base_identity,
        )
        receipt_identity = file_identity(metadata_path, "v4 training metadata")
        deep_receipt_identity = _mapping(
            model_identity.get("training_metadata"),
            "v4 deep-validated training metadata identity",
        )
        if not _same_content_identity(receipt_identity, deep_receipt_identity):
            raise EvaluationRefused(
                "v4 training metadata identity differs from the shared deep validation"
            )
    except (
        candidate.CandidateError,
        EvaluationRefused,
        OSError,
        ValueError,
    ):
        raise
    except Exception as exc:
        raise EvaluationRefused(f"v4 training-lineage validation failed: {exc}") from exc

    return (
        {
            "status": "provided_and_validated",
            "schema": TRAINING_SCHEMA_V4,
            "receipt": receipt_identity,
            "source_corpus": {
                "file": source_identity,
                "corpus_version": candidate.CORPUS_VERSION,
                "canonical_bytes": source_validation.canonical_bytes,
                "canonical_digest": source_validation.canonical_digest,
                "task_count": source_validation.task_count,
                "refs_digest": source_validation.refs_digest,
            },
            "prepared_dataset": {
                "manifest": manifest_identity,
                "train": train_identity,
                "holdout": holdout_identity,
                "manifest_payload": manifest,
            },
            "base_snapshot": base_identity,
            "run": model_identity,
            "conversion_binding_claim": (
                "the training receipt binds the merged HF tree; this evaluator separately "
                "binds final GGUF bytes and does not claim a conversion proof"
            ),
        },
        (evaluate_code, train_code),
    )


def validate_v5_receipt_header(
    payload: Any,
    *,
    training_manifest: Mapping[str, Any],
    training_manifest_digest: str,
) -> dict[str, Any]:
    """Validate the v5-only header before the shared deep receipt validator."""

    metadata = _mapping(payload, "training metadata")
    _exact_keys(metadata, TRAINING_METADATA_KEYS, "training metadata")
    required = {
        "schema": TRAINING_SCHEMA_V5,
        "status": "complete",
        "run_kind": "final_all_public",
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "corpus_version": historical_candidate.CORPUS_VERSION,
        "quality_claim": historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise EvaluationRefused(f"v5 training metadata field {key!r} changed")
    dataset = _mapping(metadata.get("dataset"), "v5 training dataset")
    _exact_keys(
        dataset,
        frozenset({"manifest", "manifest_digest", "source_corpus"}),
        "v5 training dataset",
    )
    if dataset.get("manifest") != training_manifest:
        raise EvaluationRefused("v5 training metadata does not bind the prepared manifest")
    if dataset.get("manifest_digest") != training_manifest_digest:
        raise EvaluationRefused("v5 training metadata prepared-manifest digest changed")
    if dataset.get("source_corpus") != historical_candidate.source_corpus_identity():
        raise EvaluationRefused("v5 training metadata source-corpus identity changed")
    if (
        training_manifest.get("seed"),
        training_manifest.get("train_examples"),
        training_manifest.get("holdout_examples"),
    ) != (DIAGNOSTIC_SEED, historical_candidate.EXPECTED_COUNTS["train"], 0):
        raise EvaluationRefused("v5 training lineage is not the exact seed-92 8000/0 split")
    return dict(metadata)


def load_v5_training_lineage(
    run: Path,
    dataset: Path,
    source_corpus: Path,
    base: Path,
) -> tuple[dict[str, Any], tuple[ModuleType, ...]]:
    """Replay historical source bytes and deeply validate one optional v5 run."""

    run_root = candidate.assert_tmpfs_path(run, must_exist=True)
    dataset_root = candidate.assert_tmpfs_path(dataset, must_exist=True)
    source_root = candidate.assert_tmpfs_path(source_corpus, must_exist=True)
    base_root = candidate.assert_tmpfs_path(base, must_exist=True)
    if run_root.is_symlink() or not run_root.is_dir():
        raise EvaluationRefused("training run must be a regular non-symlink directory")
    try:
        train_rows, manifest = historical_candidate.load_prepared_dataset(
            dataset_root,
            source_root,
        )
        holdout_rows = historical_candidate.load_prepared_rows(
            dataset_root / "holdout.jsonl",
            "holdout.jsonl",
        )
        if (len(train_rows), len(holdout_rows)) != (
            historical_candidate.EXPECTED_COUNTS["train"],
            0,
        ):
            raise EvaluationRefused("v5 training source replay is not exactly 8000/0")
        manifest_identity = file_identity(
            dataset_root / "manifest.json",
            "v5 prepared manifest",
        )
        metadata_path = run_root / "training_metadata.json"
        payload, _raw = _read_strict_json(
            metadata_path,
            "v5 training metadata",
            maximum_bytes=MAX_JSON_RECEIPT_BYTES,
        )
        validate_v5_receipt_header(
            payload,
            training_manifest=manifest,
            training_manifest_digest=str(manifest_identity["digest"]),
        )
        base_identity = candidate.verify_base_snapshot(
            base_root,
            expected_model=BASE_MODEL,
        )
        evaluate_code = importlib.import_module("training.evaluate_code")
        train_code = importlib.import_module("training.train_code")
        if getattr(train_code, "HISTORICAL_SCHEMA", None) != TRAINING_SCHEMA_V5:
            raise EvaluationRefused("shared trainer no longer recognizes the v5 receipt")
        model_identity = evaluate_code._load_training_run(
            run_root / "merged",
            dataset_manifest=manifest,
            dataset_manifest_digest=str(manifest_identity["digest"]),
            base_identity=base_identity,
        )
    except (
        candidate.CandidateError,
        historical_candidate.HistoricalCandidateError,
        EvaluationRefused,
        OSError,
        ValueError,
    ):
        raise
    except Exception as exc:
        raise EvaluationRefused(f"v5 training-lineage validation failed: {exc}") from exc

    return (
        {
            "status": "provided_and_validated",
            "schema": TRAINING_SCHEMA_V5,
            "receipt": file_identity(metadata_path, "v5 training metadata"),
            "source_corpus": {
                "file": file_identity(source_root, "historical public source corpus"),
                **historical_candidate.source_corpus_identity(),
            },
            "prepared_dataset": {
                "manifest": manifest_identity,
                "train": file_identity(
                    dataset_root / "train.jsonl",
                    "v5 prepared train JSONL",
                ),
                "holdout": file_identity(
                    dataset_root / "holdout.jsonl",
                    "v5 prepared holdout JSONL",
                ),
                "manifest_payload": manifest,
            },
            "base_snapshot": base_identity,
            "run": model_identity,
            "conversion_binding_claim": (
                "the training receipt binds the merged HF tree; this evaluator separately "
                "binds final GGUF bytes and does not claim a conversion proof"
            ),
        },
        (evaluate_code, train_code),
    )


def validate_v6_receipt_header(
    payload: Any,
    *,
    training_manifest: Mapping[str, Any],
    training_manifest_digest: str,
) -> dict[str, Any]:
    """Validate the normalized v6 header before the shared deep receipt validator."""

    metadata = _mapping(payload, "training metadata")
    _exact_keys(metadata, TRAINING_METADATA_KEYS, "training metadata")
    required = {
        "schema": TRAINING_SCHEMA_V6,
        "status": "complete",
        "run_kind": "final_all_public",
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "corpus_version": normalized_historical_candidate.CORPUS_VERSION,
        "quality_claim": normalized_historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise EvaluationRefused(f"v6 training metadata field {key!r} changed")
    dataset = _mapping(metadata.get("dataset"), "v6 training dataset")
    _exact_keys(
        dataset,
        frozenset({"manifest", "manifest_digest", "source_corpus"}),
        "v6 training dataset",
    )
    if dataset.get("manifest") != training_manifest:
        raise EvaluationRefused("v6 training metadata does not bind the prepared manifest")
    if dataset.get("manifest_digest") != training_manifest_digest:
        raise EvaluationRefused("v6 training metadata prepared-manifest digest changed")
    if dataset.get("source_corpus") != normalized_historical_candidate.source_corpus_identity():
        raise EvaluationRefused("v6 training metadata source-corpus identity changed")
    manifest_required = {
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
        "excluded_refs_digest": normalized_historical_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
        "target_construction": normalized_historical_candidate.TARGET_CONSTRUCTION,
        "normalization": normalized_historical_candidate.NORMALIZATION_CONTRACT,
        "quality_claim": normalized_historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }
    for key, expected in manifest_required.items():
        if training_manifest.get(key) != expected:
            raise EvaluationRefused(f"v6 training manifest field {key!r} changed")
    return dict(metadata)


def load_v6_training_lineage(
    run: Path,
    dataset: Path,
    source_corpus: Path,
    base: Path,
) -> tuple[dict[str, Any], tuple[ModuleType, ...]]:
    """Replay normalized source bytes and deeply validate one optional v6 run."""

    run_root = candidate.assert_tmpfs_path(run, must_exist=True)
    dataset_root = candidate.assert_tmpfs_path(dataset, must_exist=True)
    source_root = candidate.assert_tmpfs_path(source_corpus, must_exist=True)
    base_root = candidate.assert_tmpfs_path(base, must_exist=True)
    if run_root.is_symlink() or not run_root.is_dir():
        raise EvaluationRefused("training run must be a regular non-symlink directory")
    try:
        train_rows, manifest = normalized_historical_candidate.load_prepared_dataset(
            dataset_root,
            source_root,
        )
        holdout_rows = normalized_historical_candidate.load_prepared_rows(
            dataset_root / "holdout.jsonl",
            "holdout.jsonl",
        )
        expected_split = (
            normalized_historical_candidate.EXPECTED_TRAIN_EXAMPLES,
            normalized_historical_candidate.EXPECTED_HOLDOUT_EXAMPLES,
        )
        if (len(train_rows), len(holdout_rows)) != expected_split:
            raise EvaluationRefused(
                f"v6 training source replay is not exactly {expected_split[0]}/{expected_split[1]}"
            )
        manifest_identity = file_identity(
            dataset_root / "manifest.json",
            "v6 prepared manifest",
        )
        train_identity = file_identity(
            dataset_root / "train.jsonl",
            "v6 prepared train JSONL",
        )
        holdout_identity = file_identity(
            dataset_root / "holdout.jsonl",
            "v6 prepared holdout JSONL",
        )
        excluded_refs_identity = file_identity(
            dataset_root / normalized_historical_candidate.EXCLUDED_REFS_FILE,
            "v6 prepared excluded refs",
        )
        source_identity = file_identity(
            source_root,
            "normalized historical public source corpus",
        )
        expected_file_identities = (
            (
                train_identity,
                manifest.get("train_file_bytes"),
                manifest.get("train_file_digest"),
                "train",
            ),
            (
                holdout_identity,
                manifest.get("holdout_file_bytes"),
                manifest.get("holdout_file_digest"),
                "holdout",
            ),
            (
                excluded_refs_identity,
                manifest.get("excluded_refs_canonical_bytes"),
                manifest.get("excluded_refs_digest"),
                "excluded-refs",
            ),
            (
                source_identity,
                normalized_historical_candidate.PUBLIC_CORPUS_RESPONSE_BYTES,
                normalized_historical_candidate.PUBLIC_CORPUS_RAW_DIGEST,
                "source-corpus",
            ),
        )
        for identity, expected_bytes, expected_digest, label in expected_file_identities:
            if identity["bytes"] != expected_bytes or identity["digest"] != expected_digest:
                raise EvaluationRefused(f"v6 {label} identity differs from its manifest contract")
        metadata_path = run_root / "training_metadata.json"
        payload, _raw = _read_strict_json(
            metadata_path,
            "v6 training metadata",
            maximum_bytes=MAX_JSON_RECEIPT_BYTES,
        )
        validate_v6_receipt_header(
            payload,
            training_manifest=manifest,
            training_manifest_digest=str(manifest_identity["digest"]),
        )
        base_identity = candidate.verify_base_snapshot(
            base_root,
            expected_model=BASE_MODEL,
        )
        evaluate_code = importlib.import_module("training.evaluate_code")
        train_code = importlib.import_module("training.train_code")
        if getattr(train_code, "NORMALIZED_HISTORICAL_SCHEMA", None) != TRAINING_SCHEMA_V6:
            raise EvaluationRefused("shared trainer no longer recognizes the v6 receipt")
        model_identity = evaluate_code._load_training_run(
            run_root / "merged",
            dataset_manifest=manifest,
            dataset_manifest_digest=str(manifest_identity["digest"]),
            base_identity=base_identity,
        )
        receipt_identity = file_identity(metadata_path, "v6 training metadata")
        deep_receipt_identity = _mapping(
            model_identity.get("training_metadata"),
            "v6 deep-validated training metadata identity",
        )
        if not _same_content_identity(receipt_identity, deep_receipt_identity):
            raise EvaluationRefused(
                "v6 training metadata identity differs from the shared deep validation"
            )
    except (
        candidate.CandidateError,
        EvaluationRefused,
        OSError,
        ValueError,
    ):
        raise
    except Exception as exc:
        raise EvaluationRefused(f"v6 training-lineage validation failed: {exc}") from exc

    return (
        {
            "status": "provided_and_validated",
            "schema": TRAINING_SCHEMA_V6,
            "receipt": receipt_identity,
            "source_corpus": {
                "file": source_identity,
                **normalized_historical_candidate.source_corpus_identity(),
            },
            "prepared_dataset": {
                "manifest": manifest_identity,
                "train": train_identity,
                "holdout": holdout_identity,
                "excluded_refs": excluded_refs_identity,
                "manifest_payload": manifest,
            },
            "base_snapshot": base_identity,
            "run": model_identity,
            "conversion_binding_claim": (
                "the training receipt binds the merged HF tree; this evaluator separately "
                "binds final GGUF bytes and does not claim a conversion proof"
            ),
        },
        (evaluate_code, train_code, normalized_historical_candidate),
    )


def load_training_lineage(
    run: Path,
    dataset: Path,
    source_corpus: Path,
    base: Path,
) -> tuple[dict[str, Any], tuple[ModuleType, ...]]:
    """Strictly dispatch an optional training lineage by prepared-manifest schema."""

    dataset_root = candidate.assert_tmpfs_path(dataset, must_exist=True)
    manifest, _raw = _read_strict_json(
        dataset_root / "manifest.json",
        "training prepared manifest",
        maximum_bytes=MAX_JSON_RECEIPT_BYTES,
    )
    schema = _mapping(manifest, "training prepared manifest").get("schema")
    if schema == candidate.DATASET_SCHEMA:
        return load_v4_training_lineage(run, dataset, source_corpus, base)
    if schema == historical_candidate.DATASET_SCHEMA:
        return load_v5_training_lineage(run, dataset, source_corpus, base)
    if schema == normalized_historical_candidate.DATASET_SCHEMA:
        return load_v6_training_lineage(run, dataset, source_corpus, base)
    raise EvaluationRefused("training prepared manifest schema is unsupported")


def lineage_evaluation_contract(training_lineage: Mapping[str, Any]) -> dict[str, str]:
    """Derive model, GGUF architecture, output schema, and claim from validated lineage."""

    status = training_lineage.get("status")
    if status == "not_provided":
        return {
            "base_model": BASE_MODEL,
            "gguf_architecture": GGUF_ARCHITECTURE_BY_BASE_MODEL[BASE_MODEL],
            "evaluation_schema": SCHEMA,
            "lineage_claim": NO_TRAINING_LINEAGE_CLAIM,
        }
    if status != "provided_and_validated":
        raise EvaluationRefused("training lineage status is unsupported")
    schema = training_lineage.get("schema")
    base_snapshot = _mapping(training_lineage.get("base_snapshot"), "training base snapshot")
    base_model = base_snapshot.get("base_model")
    if not isinstance(base_model, str):
        raise EvaluationRefused("training lineage has no exact base-model identity")
    architecture = GGUF_ARCHITECTURE_BY_BASE_MODEL.get(base_model)
    if architecture is None:
        raise EvaluationRefused("training lineage base model has no pinned GGUF architecture")
    if schema == TRAINING_SCHEMA_V4:
        if base_model not in QWEN25_BASE_MODELS or architecture != "qwen2":
            raise EvaluationRefused("v4 lineage is not a pinned Qwen2.5/qwen2 contract")
        return {
            "base_model": base_model,
            "gguf_architecture": architecture,
            "evaluation_schema": SCHEMA_V2,
            "lineage_claim": CURRENT_OVERLAP_LINEAGE_CLAIM,
        }
    if schema == TRAINING_SCHEMA_V5:
        lineage_claim = LINEAGE_CLAIM
    elif schema == TRAINING_SCHEMA_V6:
        lineage_claim = NORMALIZED_LINEAGE_CLAIM
    else:
        raise EvaluationRefused("validated training lineage schema is unsupported")
    if base_model != BASE_MODEL or architecture != "qwen3":
        raise EvaluationRefused("legacy v5/v6 lineage is not the pinned Qwen3/qwen3 contract")
    return {
        "base_model": base_model,
        "gguf_architecture": architecture,
        "evaluation_schema": SCHEMA,
        "lineage_claim": lineage_claim,
    }


def _distribution_identity(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise EvaluationRefused(f"required distribution {name!r} is unavailable") from exc
    relative_files = distribution.files
    if relative_files is None:
        raise EvaluationRefused(f"distribution {name!r} exposes no installed-file manifest")
    files: list[dict[str, Any]] = []
    for relative_item in sorted(relative_files, key=lambda item: str(item)):
        relative = str(relative_item).replace(os.sep, "/")
        path = Path(distribution.locate_file(relative_item))
        if path.is_symlink() or not path.is_file():
            raise EvaluationRefused(
                f"distribution {name!r} contains a missing or non-regular file: {relative}"
            )
        identity = file_identity(path, f"{name} distribution file {relative}")
        files.append(
            {
                "path": relative,
                "bytes": identity["bytes"],
                "digest": identity["digest"],
            }
        )
    if not files:
        raise EvaluationRefused(f"distribution {name!r} installed-file manifest is empty")
    return {
        "name": name,
        "version": distribution.version,
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "tree_digest": candidate.digest_bytes(candidate.canonical_json_bytes(files)),
    }


def _module_source_identity(
    module: ModuleType,
    label: str,
    *,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise EvaluationRefused(f"{label} exposes no source path")
    path = Path(raw_path)
    if path.suffix != ".py":
        raise EvaluationRefused(f"{label} was not imported from Python source")
    identity = file_identity(path, f"{label} source")
    if expected_digest is not None and identity["digest"] != expected_digest:
        raise EvaluationRefused(
            f"{label} source differs from signed v0.3: "
            f"expected {expected_digest}, got {identity['digest']}"
        )
    return identity


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "model name" and value.strip():
                return value.strip()
    except (OSError, UnicodeDecodeError):
        pass
    return platform.processor() or "unavailable"


def validate_engine_contract(
    microtensor_module: ModuleType,
    constants_module: ModuleType,
    gguf_module: ModuleType,
    *,
    llama_cpp_version: str,
) -> None:
    """Reject any runtime that is not the exact signed deterministic contract."""

    if getattr(constants_module, "RELEASE_VERSION", None) != SIGNED_RELEASE_VERSION:
        raise EvaluationRefused("Microtensor release is not signed v0.3.2")
    if getattr(constants_module, "MECHANISM_VERSION", None) != SIGNED_MECHANISM_VERSION:
        raise EvaluationRefused("Microtensor mechanism is not signed v0.3.0")
    if getattr(microtensor_module, "__mechanism__", None) != SIGNED_MECHANISM_VERSION:
        raise EvaluationRefused("Microtensor package mechanism marker changed")
    if llama_cpp_version != LLAMA_CPP_VERSION:
        raise EvaluationRefused(
            f"llama-cpp-python {llama_cpp_version} is not pinned {LLAMA_CPP_VERSION}"
        )
    required_constants = {
        "THREADS": THREADS,
        "GPU_LAYERS": GPU_LAYERS,
        "SEED": SEED,
        "DEFAULT_CONTEXT": 2048,
    }
    for name, expected in required_constants.items():
        if getattr(gguf_module, name, None) != expected:
            raise EvaluationRefused(f"signed GGUF engine constant {name} changed")
    info = getattr(gguf_module, "INFO", None)
    if (
        info is None
        or getattr(info, "name", None) != "llama-cpp"
        or getattr(info, "version", None) != ENGINE_ADAPTER_VERSION
        or getattr(info, "deterministic", None) is not True
    ):
        raise EvaluationRefused("signed GGUF engine identity changed")


def load_signed_runtime(
    *,
    extra_tool_modules: Sequence[ModuleType] = (),
) -> RuntimeBindings:
    """Import and fully identify the pinned signed-v0.3 CPU GGUF runtime."""

    module_names = (
        "microtensor",
        "microtensor.core.constants",
        "microtensor.core.hashing",
        "microtensor.core.protocol",
        "microtensor.core.tracks",
        "microtensor.harness.contract",
        "microtensor.harness.engines.gguf",
    )
    try:
        modules = {name: importlib.import_module(name) for name in module_names}
        llama_cpp = importlib.import_module("llama_cpp")
        llama_cpp_version = importlib.metadata.version("llama-cpp-python")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise EvaluationRefused(f"signed GGUF runtime is incomplete: {exc}") from exc
    constants_module = modules["microtensor.core.constants"]
    gguf_module = modules["microtensor.harness.engines.gguf"]
    validate_engine_contract(
        modules["microtensor"],
        constants_module,
        gguf_module,
        llama_cpp_version=llama_cpp_version,
    )

    signed_sources: dict[str, Any] = {}
    for name, expected_digest in PINNED_RUNTIME_SOURCE_DIGESTS.items():
        signed_sources[name] = _module_source_identity(
            modules[name],
            name,
            expected_digest=expected_digest,
        )
    tool_sources = {
        "evaluator": _module_source_identity(sys.modules[__name__], __name__),
        "code_candidate": _module_source_identity(candidate, candidate.__name__),
        "historical_code_candidate": _module_source_identity(
            historical_candidate,
            historical_candidate.__name__,
        ),
    }
    for module in extra_tool_modules:
        tool_sources[module.__name__] = _module_source_identity(module, module.__name__)

    try:
        python_executable = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise EvaluationRefused(f"Python executable could not be resolved: {exc}") from exc
    python_identity = file_identity(python_executable, "resolved Python executable")
    runtime_identity = {
        "microtensor": {
            "package_version": getattr(modules["microtensor"], "__version__", None),
            "release_version": constants_module.RELEASE_VERSION,
            "mechanism_version": constants_module.MECHANISM_VERSION,
            "engine": {
                "name": gguf_module.INFO.name,
                "adapter_version": gguf_module.INFO.version,
                "deterministic": gguf_module.INFO.deterministic,
                "notes": gguf_module.INFO.notes,
            },
            "signed_source_files": signed_sources,
        },
        "llama_cpp": {
            "module": _module_source_identity(llama_cpp, "llama_cpp"),
            "distribution": _distribution_identity("llama-cpp-python"),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": python_identity,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "tool_sources": tool_sources,
        "execution": {
            "threads": THREADS,
            "threads_batch": THREADS,
            "gpu_layers": GPU_LAYERS,
            "seed": SEED,
            "device": "cpu",
        },
    }
    protocol = modules["microtensor.core.protocol"]
    tracks = modules["microtensor.core.tracks"]
    contract = modules["microtensor.harness.contract"]
    return RuntimeBindings(
        artifact_format=protocol.ArtifactFormat,
        decoding=tracks.Decoding,
        engine_type=gguf_module.GgufEngine,
        load_manifest_type=protocol.LoadManifest,
        request_type=contract.Request,
        gguf_module=gguf_module,
        identity=runtime_identity,
    )


def generation_contract(max_input_tokens: int) -> dict[str, Any]:
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or not MIN_CONTEXT_TOKENS <= max_input_tokens <= MAX_CONTEXT_TOKENS
    ):
        raise EvaluationRefused(
            f"max input tokens must be in [{MIN_CONTEXT_TOKENS}, {MAX_CONTEXT_TOKENS}]"
        )
    return {
        "interface": "raw_completion",
        "chat": False,
        "decoding": "greedy",
        "context_tokens": max_input_tokens,
        "threads": THREADS,
        "threads_batch": THREADS,
        "gpu_layers": GPU_LAYERS,
        "seed": SEED,
        "reset_kv_before_each_request": True,
        "sampler": {
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
        "batch_size": 1,
    }


def _memory_snapshot() -> dict[str, int | None]:
    current: int | None = None
    proc_peak: int | None = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            parts = value.split()
            if len(parts) != 2 or parts[1] != "kB" or not parts[0].isdigit():
                continue
            if key == "VmRSS":
                current = int(parts[0]) * 1024
            elif key == "VmHWM":
                proc_peak = int(parts[0]) * 1024
    except (OSError, UnicodeDecodeError):
        pass
    usage_peak: int | None = None
    try:
        raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        usage_peak = raw_peak * 1024 if sys.platform.startswith("linux") else raw_peak
    except (OSError, ValueError):
        pass
    peaks = [value for value in (proc_peak, usage_peak) if value is not None]
    return {
        "current_bytes": current,
        "peak_bytes": max(peaks) if peaks else None,
    }


def generate_result(
    *,
    engine: Any,
    request_type: Any,
    decoding: Any,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate one raw completion and inspect only its text/AST structure."""

    ref = source.get("ref")
    prompt = source.get("prompt")
    reference = source.get("completion")
    max_output_tokens = source.get("max_output_tokens")
    if (
        not isinstance(ref, str)
        or not isinstance(prompt, str)
        or not isinstance(reference, str)
        or max_output_tokens != 1024
    ):
        raise EvaluationRefused("diagnostic row changed after lineage validation")
    request = request_type(
        task_ref=ref,
        prompt=prompt,
        inputs={},
        max_output_tokens=max_output_tokens,
        decoding=decoding.GREEDY,
        seed=SEED,
        nonce=GENERATION_NONCE,
        chat=False,
    )
    memory_before = _memory_snapshot()
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    response = engine.generate(request)
    cpu_finished = time.process_time_ns()
    wall_finished = time.perf_counter_ns()
    memory_after = _memory_snapshot()

    if getattr(response, "task_ref", None) != ref:
        raise EvaluationRefused("GGUF engine returned a mismatched task ref")
    error = getattr(response, "error", "")
    if not isinstance(error, str):
        raise EvaluationRefused("GGUF engine returned a non-string error")
    if len(error.encode("utf-8")) > MAX_ERROR_BYTES:
        raise EvaluationRefused("GGUF engine error exceeds the receipt byte limit")
    ok = not error
    raw_output = getattr(response, "output", None) if ok else ""
    if not isinstance(raw_output, str):
        raise EvaluationRefused("successful GGUF response returned a non-string output")
    output_bytes = raw_output.encode("utf-8")
    if len(output_bytes) > MAX_RESULT_BYTES:
        raise EvaluationRefused("GGUF output exceeds the receipt byte limit")
    output_pieces = _nonnegative_integer(
        getattr(response, "output_tokens", None),
        "engine-reported output pieces",
    )
    ttft_ms = _finite_number(getattr(response, "ttft_ms", None), "TTFT")
    engine_total_ms = _finite_number(
        getattr(response, "total_ms", None),
        "engine total latency",
    )
    engine_peak = _nonnegative_integer(
        getattr(response, "peak_rss_bytes", 0),
        "engine-reported peak RSS",
    )
    observed_peaks = [
        value
        for value in (
            memory_before["peak_bytes"],
            memory_after["peak_bytes"],
            engine_peak or None,
        )
        if value is not None
    ]
    row = {
        "ref": ref,
        "ok": ok,
        "error": error,
        "prompt_digest": candidate.digest_bytes(prompt.encode("utf-8")),
        "reference_digest": candidate.digest_bytes(reference.encode("utf-8")),
        "max_output_tokens": max_output_tokens,
        "raw_output": raw_output,
        "raw_output_digest": candidate.digest_bytes(output_bytes),
        "raw_output_utf8_bytes": len(output_bytes),
        # The signed adapter counts non-empty streaming pieces, not tokenizer IDs.
        "engine_reported_output_pieces": output_pieces,
        "ttft_ms": ttft_ms,
        "engine_total_ms": engine_total_ms,
        "evaluator_wall_ms": (wall_finished - wall_started) / 1_000_000,
        "evaluator_cpu_ms": (cpu_finished - cpu_started) / 1_000_000,
        "rss_before_bytes": memory_before["current_bytes"],
        "rss_after_bytes": memory_after["current_bytes"],
        "peak_rss_bytes": max(observed_peaks) if observed_peaks else None,
        **structural_diagnostics(raw_output, reference),
    }
    _exact_keys(row, RESULT_KEYS, "GGUF evaluation result")
    return row


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise EvaluationRefused("cannot take a quantile of zero values")
    if not 0.0 <= quantile <= 1.0:
        raise EvaluationRefused("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    checked = [_finite_number(value, "summary observation") for value in values]
    return {
        "total": sum(checked),
        "mean": statistics.fmean(checked),
        "median": statistics.median(checked),
        "minimum": min(checked),
        "maximum": max(checked),
        "p95_linear": _linear_quantile(checked, 0.95),
    }


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise EvaluationRefused("cannot summarize zero GGUF evaluation rows")
    for row in rows:
        _exact_keys(row, RESULT_KEYS, "GGUF evaluation result")
    count = len(rows)
    successful = [row for row in rows if row["ok"] is True]
    diagnostics: dict[str, Any] = {}
    for key in (
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
    ):
        found = sum(row.get(key) is True for row in rows)
        diagnostics[key] = {"count": found, "fraction": found / count}
    similarity = [float(row["scorer_extracted_reference_text_similarity"]) for row in rows]
    rss_values = [
        int(row["peak_rss_bytes"]) for row in rows if isinstance(row.get("peak_rss_bytes"), int)
    ]
    return {
        "examples": count,
        "successful_generations": len(successful),
        "failed_generations": count - len(successful),
        "failed_refs": [str(row["ref"]) for row in rows if row["ok"] is not True],
        "quality_score": None,
        "execution_pass_at_1": None,
        "structural_diagnostics": diagnostics,
        "scorer_extracted_reference_text_similarity": _distribution(similarity),
        "latency_ms": {
            "ttft_all": _distribution([float(row["ttft_ms"]) for row in rows]),
            "engine_total_all": _distribution([float(row["engine_total_ms"]) for row in rows]),
            "evaluator_wall_all": _distribution([float(row["evaluator_wall_ms"]) for row in rows]),
            "engine_total_successful": _distribution(
                [float(row["engine_total_ms"]) for row in successful]
            ),
        },
        "rss": {
            "sampling": (
                "Linux process VmRSS/VmHWM and getrusage snapshots before/after each "
                "request; no monitor thread was introduced"
            ),
            "peak_observed_bytes": max(rss_values) if rss_values else None,
        },
        "output": {
            "utf8_bytes": sum(int(row["raw_output_utf8_bytes"]) for row in rows),
            "engine_reported_stream_pieces": sum(
                int(row["engine_reported_output_pieces"]) for row in rows
            ),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _training_arguments(args: argparse.Namespace) -> tuple[Path, Path, Path, Path] | None:
    values = (
        args.training_run,
        args.training_dataset,
        args.training_source_corpus,
        args.training_base,
    )
    supplied = [value is not None for value in values]
    if any(supplied) and not all(supplied):
        raise EvaluationRefused(
            "--training-run, --training-dataset, --training-source-corpus, and "
            "--training-base must be supplied together"
        )
    if not any(supplied):
        return None
    run, dataset, source, base = values
    if run is None or dataset is None or source is None or base is None:
        raise EvaluationRefused("internal training-argument validation became inconsistent")
    return run, dataset, source, base


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnostic-jsonl", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument(
        "--quantization",
        choices=tuple(SUPPORTED_QUANTIZATIONS),
        required=True,
    )
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--training-source-corpus", type=Path)
    parser.add_argument("--training-base", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    try:
        training_arguments = _training_arguments(args)
        contract = generation_contract(args.max_input_tokens)
        output_root = candidate.assert_tmpfs_path(args.out)
        if output_root.exists():
            raise EvaluationRefused(f"evaluation output already exists: {output_root}")
        rows, diagnostic_lineage = load_public_diagnostic(
            args.dataset,
            args.diagnostic_jsonl,
            args.source_corpus,
        )
        extra_tool_modules: tuple[ModuleType, ...] = ()
        if training_arguments is None:
            training_lineage: dict[str, Any] = {
                "status": "not_provided",
                "claim": NO_TRAINING_LINEAGE_CLAIM,
            }
        else:
            training_lineage, extra_tool_modules = load_training_lineage(*training_arguments)
        evaluation_contract = lineage_evaluation_contract(training_lineage)
        artifact = artifact_identity(
            args.artifact,
            entrypoint=args.entrypoint,
            expected_digest=args.artifact_digest,
            quantization=args.quantization,
            expected_architecture=evaluation_contract["gguf_architecture"],
        )
        runtime = load_signed_runtime(extra_tool_modules=extra_tool_modules)
        load_manifest = runtime.load_manifest_type(
            format=runtime.artifact_format.GGUF,
            quantization=args.quantization,
            entrypoint=args.entrypoint,
            max_input={"tokens": args.max_input_tokens},
            preprocessing={"tokenizer": "tokenizer.json"},
            base_model=evaluation_contract["base_model"],
        )
        load_manifest_payload = load_manifest.to_dict()
        configuration = {
            "generation": contract,
            "load_manifest": load_manifest_payload,
            "artifact_digest": args.artifact_digest,
            "diagnostic_refs_digest": diagnostic_lineage["diagnostic_jsonl"]["refs_digest"],
        }
        if evaluation_contract["evaluation_schema"] == SCHEMA_V2:
            configuration["expected_gguf_architecture"] = evaluation_contract[
                "gguf_architecture"
            ]
        configuration_digest = candidate.digest_bytes(candidate.canonical_json_bytes(configuration))
    except (
        candidate.CandidateError,
        historical_candidate.HistoricalCandidateError,
        EvaluationRefused,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"code GGUF evaluation refused: {exc}") from exc

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    engine = runtime.engine_type(threads=THREADS, validate=True)
    started_at_unix_ns = time.time_ns()
    initial_memory = _memory_snapshot()
    try:
        load_wall_started = time.perf_counter_ns()
        load_cpu_started = time.process_time_ns()
        try:
            engine.load(Path(str(artifact["root"])), load_manifest)
        finally:
            load_cpu_finished = time.process_time_ns()
            load_wall_finished = time.perf_counter_ns()
        loaded_memory = _memory_snapshot()

        result_path = staging / "results.jsonl"
        result_rows: list[dict[str, Any]] = []
        try:
            with result_path.open("xb") as handle:
                for index, source in enumerate(rows, 1):
                    result = generate_result(
                        engine=engine,
                        request_type=runtime.request_type,
                        decoding=runtime.decoding,
                        source=source,
                    )
                    result_rows.append(result)
                    handle.write(candidate.canonical_json_bytes(result) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    if index == 1 or index % 4 == 0 or index == len(rows):
                        print(f"generated {index}/{len(rows)}", flush=True)
        finally:
            engine.unload()
        unloaded_memory = _memory_snapshot()

        after_rows, after_diagnostic_lineage = load_public_diagnostic(
            args.dataset,
            args.diagnostic_jsonl,
            args.source_corpus,
        )
        if after_rows != rows or after_diagnostic_lineage != diagnostic_lineage:
            raise EvaluationRefused("public diagnostic lineage changed during evaluation")
        if training_arguments is not None:
            after_training_lineage, after_tool_modules = load_training_lineage(*training_arguments)
            if after_training_lineage != training_lineage:
                raise EvaluationRefused("training lineage changed during evaluation")
            if lineage_evaluation_contract(after_training_lineage) != evaluation_contract:
                raise EvaluationRefused("lineage-derived evaluation contract changed")
            if tuple(module.__name__ for module in after_tool_modules) != tuple(
                module.__name__ for module in extra_tool_modules
            ):
                raise EvaluationRefused("training validation tool set changed during evaluation")
        after_artifact = artifact_identity(
            args.artifact,
            entrypoint=args.entrypoint,
            expected_digest=args.artifact_digest,
            quantization=args.quantization,
            expected_architecture=evaluation_contract["gguf_architecture"],
        )
        if after_artifact != artifact:
            raise EvaluationRefused("artifact changed during evaluation")
        after_runtime = load_signed_runtime(extra_tool_modules=extra_tool_modules)
        if after_runtime.identity != runtime.identity:
            raise EvaluationRefused("runtime or evaluator tools changed during evaluation")

        finished_at_unix_ns = time.time_ns()
        results_identity = file_identity(result_path, "GGUF results JSONL")
        summary = {
            "schema": evaluation_contract["evaluation_schema"],
            "status": (
                "complete"
                if all(row["ok"] is True for row in result_rows)
                else "complete_with_generation_failures"
            ),
            "track": candidate.TRACK,
            "hardware_class": candidate.HARDWARE_CLASS,
            "base_model": evaluation_contract["base_model"],
            "quality_claim": QUALITY_CLAIM,
            "runtime_claim": RUNTIME_CLAIM,
            "lineage_claim": evaluation_contract["lineage_claim"],
            "safety_contract": {
                "generated_code_imported": False,
                "generated_code_executed": False,
                "generated_code_bytecode_compiled": False,
                "corpus_code_imported": False,
                "corpus_code_executed": False,
                "corpus_code_bytecode_compiled": False,
                "static_ast_parse_only": True,
                "hidden_or_scored_tests_accessed": False,
            },
            "configuration": configuration,
            "configuration_digest": configuration_digest,
            "artifact": artifact,
            "evaluation_dataset": diagnostic_lineage,
            "training_lineage": training_lineage,
            "runtime": runtime.identity,
            "timing": {
                "started_at_unix_ns": started_at_unix_ns,
                "finished_at_unix_ns": finished_at_unix_ns,
                "elapsed_ms": (finished_at_unix_ns - started_at_unix_ns) / 1_000_000,
                "model_load_wall_ms": (load_wall_finished - load_wall_started) / 1_000_000,
                "model_load_cpu_ms": (load_cpu_finished - load_cpu_started) / 1_000_000,
            },
            "memory": {
                "initial": initial_memory,
                "model_loaded": loaded_memory,
                "model_unloaded": unloaded_memory,
            },
            "results": {
                "file": "results.jsonl",
                "bytes": results_identity["bytes"],
                "digest": results_identity["digest"],
                **summarize_results(result_rows),
            },
        }
        summary_path = staging / "summary.json"
        _write_json(summary_path, summary)
        if output_root.is_symlink() or output_root.exists():
            raise EvaluationRefused(f"evaluation output appeared during evaluation: {output_root}")
        os.replace(staging, output_root)
        print(
            json.dumps(
                {
                    "output": str(output_root),
                    "artifact_digest": artifact["tree_digest"],
                    "results_digest": summary["results"]["digest"],
                    "summary_digest": candidate.digest_file(output_root / "summary.json"),
                    "quality_claim": QUALITY_CLAIM,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        try:
            engine.unload()
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
