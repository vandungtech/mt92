#!/usr/bin/env python3
"""Convert one validated code training run into an atomically published GGUF bundle.

Only an exact completed source-bound v4 current, v5 historical, or v6 normalized
historical training lineage and a clean, pinned llama.cpp checkout are accepted.
Current v4/Qwen2.5 runs are accepted only through the calibrated Q4_K_M path.
Conversion happens in a unique directory on ``/dev/shm``.  The final bundle is
published with Linux ``RENAME_NOREPLACE`` only after every source, tool, and
output identity has been replayed.

This module never executes corpus or generated code.  Calibrated conversion
does run offline model forward passes through the exact reviewed
``llama-imatrix`` binary; its other children are the reviewed converter,
quantizer, and read-only Git identity checks.  Child processes receive a
deliberately small environment without credentials, wallet paths, or user
configuration.

WARNING: this converter is process supervision, not containment. It does not
attest Python's standard-library or dynamic-loader closure and must not be run
on the miner host. Current-v4 work requires an isolated external worker and a
separately verified signed execution/containment attestation before publication.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import selectors
import shutil
import stat
import struct
import subprocess
import tempfile
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import evaluate_code_gguf as gguf
    from training import historical_code_candidate as historical_candidate
    from training import normalized_historical_code_candidate as normalized_candidate
    from training import publish_code_provenance as provenance
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import evaluate_code_gguf as gguf  # type: ignore[no-redef]
    import historical_code_candidate as historical_candidate  # type: ignore[no-redef]
    import normalized_historical_code_candidate as normalized_candidate  # type: ignore[no-redef]
    import publish_code_provenance as provenance  # type: ignore[no-redef]


SCHEMA: Final[str] = provenance.CONVERSION_SCHEMA
CALIBRATED_CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v3"
NORMALIZED_CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v4"
NORMALIZED_CALIBRATED_CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v5"
CURRENT_CALIBRATED_CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v6"
CALIBRATION_SCHEMA: Final[str] = "microtensor.code.imatrix-calibration.v2"
CURRENT_CALIBRATION_SCHEMA: Final[str] = "microtensor.code.imatrix-calibration.v3"
CURRENT_TRAINING_SCHEMA: Final[str] = "microtensor.code.training.v4"
CURRENT_CORPUS_PROFILE: Final[str] = "bigcodebench94"
QWEN3_ARCHITECTURE: Final[str] = "qwen3"
QWEN25_ARCHITECTURE: Final[str] = "qwen2"
LEGACY_CONVERSION_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {"training_metadata_digest", "merged_tree_digest"}
)
NORMALIZED_CONVERSION_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "training_schema",
        "dataset_schema",
        "corpus_profile",
        "training_metadata_digest",
        "merged_tree_digest",
        "excluded_refs",
    }
)
NORMALIZED_EXCLUDED_REFS_KEYS: Final[frozenset[str]] = frozenset({"bytes", "digest"})
CURRENT_CONVERSION_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "training_schema",
        "dataset_schema",
        "corpus_profile",
        "training_metadata_digest",
        "training_metrics_digest",
        "merged_tree_digest",
        "source_corpus",
        "prepared_dataset",
        "base_snapshot",
    }
)
CURRENT_SOURCE_CORPUS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "bytes",
        "digest",
        "canonical_bytes",
        "canonical_digest",
        "task_count",
        "refs_digest",
    }
)
CURRENT_PREPARED_DATASET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "manifest_digest",
        "train_digest",
        "holdout_digest",
        "train_examples",
        "holdout_examples",
    }
)
CALIBRATION_PROFILE: Final[str] = "code-public-imatrix128-v1"
LLAMA_CPP_REVISION: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
LLAMA_CPP_ROOT: Final[Path] = Path("/tmp/llama.cpp")  # noqa: S108 - pinned RUNPATH root
ENTRYPOINT: Final[str] = "model.gguf"
LOAD_SPEC_NAME: Final[str] = "load-spec.json"
RECEIPT_NAME: Final[str] = "conversion-receipt.json"
CALIBRATION_RECEIPT_NAME: Final[str] = "calibration-receipt.json"
ARTIFACT_NAME: Final[str] = "artifact"
F16_NAME: Final[str] = "model-f16.gguf"
CALIBRATION_CORPUS_NAME: Final[str] = "calibration.txt"
IMATRIX_NAME: Final[str] = "calibration.imatrix.gguf"
SUPPORTED_QUANTIZATIONS: Final[frozenset[str]] = frozenset({"Q8_0", "Q5_K_M", "Q4_K_M"})
CALIBRATION_SEED: Final[int] = 92
CALIBRATION_CURRENT_ROWS: Final[int] = 78
CALIBRATION_DIAGNOSTIC_ROWS: Final[int] = 16
CALIBRATION_HISTORICAL_ROWS: Final[int] = 434
CALIBRATION_TOTAL_ROWS: Final[int] = 512
CALIBRATION_CHUNKS: Final[int] = 128
CALIBRATION_CONTEXT_TOKENS: Final[int] = 512
CALIBRATION_EOS_TOKEN: Final[str] = "<|im_end|>"  # noqa: S105 - tokenizer token, not a secret
CALIBRATION_EOS_TOKEN_ID: Final[int] = 151645
CALIBRATION_RENDER_SCHEMA: Final[str] = "prompt-completion-im-end-utf8-v1"
CALIBRATION_SELECTION_ALGORITHM: Final[str] = "sha256-seed-ref-ascending-v1"
CALIBRATION_MAX_BYTES: Final[int] = 16 * 1024 * 1024
CALIBRATION_MIN_FREE_BYTES: Final[int] = 6 * 1024**3
MAX_CAPTURED_LOG_BYTES: Final[int] = 1 * 1024 * 1024
_GGUF_MAX_HEADER_BYTES: Final[int] = 256 * 1024 * 1024
_GGUF_MAX_TENSORS: Final[int] = 100_000
_GGUF_MAX_METADATA: Final[int] = 100_000
_GGUF_MAX_ARRAY_ITEMS: Final[int] = 2_000_000
_GGUF_MAX_STRING_BYTES: Final[int] = 16 * 1024 * 1024
_AT_FDCWD: Final[int] = -100
_RENAME_NOREPLACE: Final[int] = 1
_STAGING_MARKER: Final[str] = ".microtensor-code-gguf-"
_CHILD_PYCACHE_PREFIX: Final[str] = ".microtensor-empty-pycache"
_MAX_ERROR_TEXT_BYTES: Final[int] = 4096
RUNTIME_LIBRARY_SCHEMA: Final[str] = "microtensor.code.llama-cpp-runtime-libraries.v1"
# (loader-relative path, final-target-relative path, bytes, SHA-256 digest).
# These ignored build outputs are part of the executable closure even though
# the pinned Git revision cannot attest to them.
# This is intentionally the in-checkout closure only. The ELF loader, libc,
# libstdc++, OpenSSL, libgomp, and other host libraries remain external runtime
# dependencies and must be disclosed separately by the experiment declaration.
LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT: Final[tuple[tuple[str, str, int, str], ...]] = (
    (
        "build/bin/libggml-base.so.0",
        "build/bin/libggml-base.so.0.22.0",
        939_528,
        "sha256:73106e6e34d4f6dcd9f4ffca57f132070c48e93c9dd409df2333eea1b7c4806f",
    ),
    (
        "build/bin/libggml-cpu.so.0",
        "build/bin/libggml-cpu.so.0.22.0",
        1_143_800,
        "sha256:196c9f2c112e51f17b79375e80b102c72bc872b3e4fc17295ab1564533812807",
    ),
    (
        "build/bin/libggml.so.0",
        "build/bin/libggml.so.0.22.0",
        56_376,
        "sha256:eaea0b8964d5acee7ce26bb4895137df772ad24387f45fbb51158495f596fa29",
    ),
    (
        "build/bin/libllama-common.so.0",
        "build/bin/libllama-common.so.0.3.0",
        5_909_928,
        "sha256:ce12dd60805687b1dfbd574033d5163089979a1b2b556cb8a6c65b85af7048f5",
    ),
    (
        "build/bin/libllama-quantize-impl.so",
        "build/bin/libllama-quantize-impl.so",
        89_792,
        "sha256:d79664774038f0f42eccee8b1d5772b2bbe0f7840181401c73e30f2986113cbb",
    ),
    (
        "build/bin/libllama.so.0",
        "build/bin/libllama.so.0.3.0",
        4_692_320,
        "sha256:8c809635a537f48c79bb058034ae9eb3c437693bc8b4fc13e0035c0be7bad8ed",
    ),
)

LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT: Final[tuple[tuple[str, int, str], ...]] = (
    (
        "build/bin/llama-imatrix",
        343_128,
        "sha256:3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35",
    ),
    (
        "build/bin/llama-quantize",
        17_928,
        "sha256:e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580",
    ),
)
LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT: Final[tuple[tuple[str, str], ...]] = (
    ("build/bin/libggml-base.so", "libggml-base.so.0"),
    ("build/bin/libggml-base.so.0", "libggml-base.so.0.22.0"),
    ("build/bin/libggml-cpu.so", "libggml-cpu.so.0"),
    ("build/bin/libggml-cpu.so.0", "libggml-cpu.so.0.22.0"),
    ("build/bin/libggml.so", "libggml.so.0"),
    ("build/bin/libggml.so.0", "libggml.so.0.22.0"),
    ("build/bin/libllama-common.so", "libllama-common.so.0"),
    ("build/bin/libllama-common.so.0", "libllama-common.so.0.3.0"),
    ("build/bin/libllama.so", "libllama.so.0"),
    ("build/bin/libllama.so.0", "libllama.so.0.3.0"),
)

if provenance.LLAMA_CPP_REVISION != LLAMA_CPP_REVISION:
    raise RuntimeError("code provenance and conversion llama.cpp revisions diverged")


class ConversionRefused(ValueError):
    """Raised before publication when any conversion contract changes."""


@dataclass(frozen=True)
class ConversionRequest:
    training_run: Path
    training_dataset: Path
    source_corpus: Path
    base: Path
    llama_cpp: Path
    converter: Path
    quantizer: Path
    output_bundle: Path
    quantization: str
    max_input_tokens: int
    calibration_profile: str | None = None
    calibration_current_dataset: Path | None = None
    calibration_current_source_corpus: Path | None = None
    imatrix_tool: Path | None = None
    converter_python: Path | None = None
    calibration_aux_dataset: Path | None = None
    calibration_aux_source_corpus: Path | None = None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConversionRefused(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected or any(not isinstance(key, str) for key in value):
        raise ConversionRefused(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(value)}"
        )


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConversionRefused(f"{label} must be a regular non-symlink file")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConversionRefused(f"{label} repeats JSON field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ConversionRefused(f"{label} contains non-finite constant {value!r}")

    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionRefused(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    return dict(_mapping(payload, label))


def _atomic_json_in_staging(path: Path, payload: Mapping[str, Any]) -> bytes:
    """Create and fsync a new JSON file; staging paths must never pre-exist."""

    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise ConversionRefused(f"short write while creating {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise ConversionRefused(f"could not create staged {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return raw


def _fresh_output_bundle(path: Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise ConversionRefused("output bundle must be an absolute path below /dev/shm")
    try:
        parent = supplied.parent.resolve(strict=True)
        tmpfs = candidate.TMPFS_MOUNT.resolve(strict=True)
    except OSError as exc:
        raise ConversionRefused(f"output parent is unavailable: {exc}") from exc
    if parent == tmpfs or tmpfs in parent.parents:
        output = parent / supplied.name
    else:
        raise ConversionRefused("output bundle must stay below /dev/shm")
    if not supplied.name or supplied.name in {".", ".."}:
        raise ConversionRefused("output bundle has an invalid final component")
    if parent.is_symlink() or not parent.is_dir():
        raise ConversionRefused("output parent must be a regular directory")
    if os.path.lexists(output):
        raise ConversionRefused("output bundle already exists; overwriting is forbidden")
    return output


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ConversionRefused(f"{label} must be a regular non-symlink directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ConversionRefused(f"{label} is unavailable: {exc}") from exc


def _regular_executable(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    if path.is_symlink():
        raise ConversionRefused(f"{label} must not be a symlink")
    try:
        before = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ConversionRefused(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ConversionRefused(f"{label} must be a regular file")
    if before.st_mode & 0o111 == 0:
        raise ConversionRefused(f"{label} must have an executable mode bit")
    if before.st_mode & 0o022:
        raise ConversionRefused(f"{label} must not be group/world writable")
    identity = gguf.file_identity(resolved, label)
    return resolved, identity


def _runtime_relative_name(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.parts[:2] != ("build", "bin") or len(path.parts) != 3:
        raise ConversionRefused(f"{label} must be one filename below build/bin")
    name = path.parts[2]
    if name in {"", ".", ".."}:
        raise ConversionRefused(f"{label} has an invalid filename")
    return name


def _runtime_symlink_target_name(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1:
        raise ConversionRefused(f"{label} must be one relative filename")
    name = path.parts[0]
    if name in {"", ".", ".."}:
        raise ConversionRefused(f"{label} has an invalid filename")
    return name


def _stable_stat_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _assert_llama_python_import_surface(root: Path) -> None:
    """Reject ignored native modules and legacy bytecode on converter import paths."""

    root = Path(root)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        before = root.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ConversionRefused(
                "llama.cpp Python import root must be a regular non-symlink directory"
            )
        descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | nofollow,
        )
        opened = os.fstat(descriptor)
        if _stable_stat_fields(opened) != _stable_stat_fields(before):
            raise ConversionRefused("llama.cpp Python import root changed while opening")

        def snapshot() -> tuple[tuple[str, tuple[int, ...]], ...]:
            entries: list[tuple[str, tuple[int, ...]]] = []
            for directory, directory_names, file_names, directory_descriptor in os.fwalk(
                ".",
                topdown=True,
                follow_symlinks=False,
                dir_fd=descriptor,
            ):
                relative_directory = Path(directory)
                if relative_directory == Path("."):
                    relative_directory = Path()
                entries.append(
                    (
                        "." if not relative_directory.parts else relative_directory.as_posix(),
                        _stable_stat_fields(os.fstat(directory_descriptor)),
                    )
                )
                all_directory_names = sorted(directory_names)
                if not relative_directory.parts:
                    directory_names[:] = [
                        name
                        for name in all_directory_names
                        if name != ".git" and name != "build" and not name.startswith("build-")
                    ]
                else:
                    directory_names.sort()
                for name in sorted((*all_directory_names, *file_names)):
                    child = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    relative = relative_directory / name
                    relative_text = relative.as_posix()
                    if name.endswith(".so"):
                        raise ConversionRefused(
                            "llama.cpp Python import surface contains an ignored "
                            f"extension-module candidate: {relative_text}"
                        )
                    if name.endswith(".pyc") and "__pycache__" not in relative.parts:
                        raise ConversionRefused(
                            "llama.cpp Python import surface contains legacy sourceless "
                            f"bytecode: {relative_text}"
                        )
                    if stat.S_ISLNK(child.st_mode):
                        raise ConversionRefused(
                            f"llama.cpp Python import surface contains a symlink: {relative_text}"
                        )
                    if not (stat.S_ISDIR(child.st_mode) or stat.S_ISREG(child.st_mode)):
                        raise ConversionRefused(
                            "llama.cpp Python import surface contains a special "
                            f"filesystem entry: {relative_text}"
                        )
                    entries.append((relative_text, _stable_stat_fields(child)))
            return tuple(entries)

        if snapshot() != snapshot():
            raise ConversionRefused("llama.cpp Python import surface changed during inspection")
        after_open = os.fstat(descriptor)
        after = root.lstat()
        if _stable_stat_fields(after_open) != _stable_stat_fields(before) or _stable_stat_fields(
            after
        ) != _stable_stat_fields(before):
            raise ConversionRefused("llama.cpp Python import root changed during inspection")
    except ConversionRefused:
        raise
    except OSError as exc:
        raise ConversionRefused(
            f"could not inspect llama.cpp Python import surface: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _attest_runtime_regular_file(
    bin_descriptor: int,
    relative: str,
    expected_bytes: int,
    expected_digest: str,
    *,
    executable: bool,
    label: str,
    nofollow: int,
) -> dict[str, Any]:
    name = _runtime_relative_name(relative, f"{label} path")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not isinstance(expected_digest, str)
        or not expected_digest.startswith("sha256:")
        or len(expected_digest) != 71
        or any(character not in "0123456789abcdef" for character in expected_digest[7:])
    ):
        raise ConversionRefused(f"{label} contract identity is malformed")
    try:
        before = os.stat(name, dir_fd=bin_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ConversionRefused(f"{label} is unavailable: {relative}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ConversionRefused(f"{label} must be a regular non-symlink file: {relative}")
    mode = stat.S_IMODE(before.st_mode)
    if mode & 0o022:
        raise ConversionRefused(f"{label} must not be group/world writable: {relative}")
    if executable and before.st_mode & 0o111 == 0:
        raise ConversionRefused(f"{label} must have an executable mode bit: {relative}")
    if before.st_size != expected_bytes:
        raise ConversionRefused(f"{label} size changed: {relative}")

    descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
            dir_fd=bin_descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stable_stat_fields(opened) != _stable_stat_fields(
            before
        ):
            raise ConversionRefused(f"{label} changed while opening: {relative}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > expected_bytes:
                raise ConversionRefused(f"{label} grew while reading: {relative}")
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    except OSError as exc:
        raise ConversionRefused(f"{label} could not be read: {relative}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if (
        total != expected_bytes
        or "sha256:" + digest.hexdigest() != expected_digest
        or _stable_stat_fields(after_open) != _stable_stat_fields(before)
    ):
        raise ConversionRefused(f"{label} bytes changed: {relative}")
    try:
        after = os.stat(name, dir_fd=bin_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ConversionRefused(f"{label} path changed: {relative}: {exc}") from exc
    if _stable_stat_fields(after) != _stable_stat_fields(before):
        raise ConversionRefused(f"{label} path changed during inspection: {relative}")
    return {
        "path": relative,
        "bytes": expected_bytes,
        "digest": expected_digest,
        "mode": f"{mode:04o}",
    }


def _runtime_library_closure(root: Path) -> dict[str, Any]:
    """Attest ignored llama.cpp shared libraries without following unsafe paths."""

    contract = LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT
    executable_contract = LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
    symlink_contract = LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT
    loader_paths = [entry[0] for entry in contract]
    executable_paths = [entry[0] for entry in executable_contract]
    symlink_paths = [entry[0] for entry in symlink_contract]
    if (
        not contract
        or loader_paths != sorted(loader_paths)
        or len(set(loader_paths)) != len(loader_paths)
    ):
        raise ConversionRefused("runtime library contract is empty, duplicated, or unsorted")
    if (
        not executable_contract
        or executable_paths != sorted(executable_paths)
        or len(set(executable_paths)) != len(executable_paths)
    ):
        raise ConversionRefused("runtime executable contract is empty, duplicated, or unsorted")
    if (
        not symlink_contract
        or symlink_paths != sorted(symlink_paths)
        or len(set(symlink_paths)) != len(symlink_paths)
    ):
        raise ConversionRefused("runtime symlink contract is empty, duplicated, or unsorted")

    namespace_names: set[str] = set()
    for relative, _expected_bytes, _expected_digest in executable_contract:
        namespace_names.add(_runtime_relative_name(relative, "runtime executable path"))
    for loader_relative, target_relative, _expected_bytes, _expected_digest in contract:
        namespace_names.add(_runtime_relative_name(loader_relative, "runtime library loader path"))
        namespace_names.add(_runtime_relative_name(target_relative, "runtime library target path"))
    for relative, target in symlink_contract:
        namespace_names.add(_runtime_relative_name(relative, "runtime symlink path"))
        namespace_names.add(_runtime_symlink_target_name(target, "runtime symlink target"))
    expected_namespace = tuple(sorted(namespace_names))

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    required_dir_fd = {os.open, os.readlink, os.stat}
    if (
        nofollow is None
        or not required_dir_fd.issubset(getattr(os, "supports_dir_fd", set()))
        or os.stat not in getattr(os, "supports_follow_symlinks", set())
    ):
        raise ConversionRefused("safe dirfd runtime-library inspection is unavailable")

    descriptors: list[int] = []
    try:
        root_descriptor = os.open(root, directory_flags)
        descriptors.append(root_descriptor)
        build_descriptor = os.open("build", directory_flags, dir_fd=root_descriptor)
        descriptors.append(build_descriptor)
        bin_descriptor = os.open("bin", directory_flags, dir_fd=build_descriptor)
        descriptors.append(bin_descriptor)
        for descriptor, label in (
            (root_descriptor, "llama.cpp root"),
            (build_descriptor, "llama.cpp build"),
            (bin_descriptor, "llama.cpp build/bin"),
        ):
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ConversionRefused(f"{label} changed from a directory")
        directory_descriptors = (
            (".", root_descriptor, "llama.cpp root"),
            ("build", build_descriptor, "llama.cpp build"),
            ("build/bin", bin_descriptor, "llama.cpp build/bin"),
        )
        directories: list[dict[str, Any]] = []
        directory_stats: list[os.stat_result] = []
        for relative, descriptor, label in directory_descriptors:
            before = os.fstat(descriptor)
            mode = stat.S_IMODE(before.st_mode)
            if mode & 0o022:
                raise ConversionRefused(
                    f"{label} must not be group/world writable for runtime closure"
                )
            directory_stats.append(before)
            directories.append({"path": relative, "mode": f"{mode:04o}"})
        actual_namespace = tuple(sorted(os.listdir(bin_descriptor)))
        if actual_namespace != expected_namespace:
            extras = sorted(set(actual_namespace) - set(expected_namespace))
            missing = sorted(set(expected_namespace) - set(actual_namespace))
            raise ConversionRefused(
                f"llama.cpp build/bin namespace changed: unexpected={extras}, missing={missing}"
            )
        build_bin_namespace = [f"build/bin/{name}" for name in expected_namespace]
    except ConversionRefused:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ConversionRefused(f"runtime library directory closure is unavailable: {exc}") from exc

    symlinks: list[dict[str, str]] = []
    symlink_states: list[tuple[str, os.stat_result, str, str]] = []
    executables: list[dict[str, Any]] = []
    libraries: list[dict[str, Any]] = []
    try:
        for relative, expected_target in symlink_contract:
            link_name = _runtime_relative_name(relative, "runtime symlink path")
            target_name = _runtime_symlink_target_name(
                expected_target,
                "runtime symlink target",
            )
            try:
                before = os.stat(
                    link_name,
                    dir_fd=bin_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ConversionRefused(
                    f"runtime symlink is unavailable: {relative}: {exc}"
                ) from exc
            if not stat.S_ISLNK(before.st_mode):
                raise ConversionRefused(f"runtime symlink changed type: {relative}")
            try:
                actual_target = os.readlink(link_name, dir_fd=bin_descriptor)
            except OSError as exc:
                raise ConversionRefused(
                    f"runtime symlink is unreadable: {relative}: {exc}"
                ) from exc
            if actual_target != target_name:
                raise ConversionRefused(f"runtime symlink escaped or was repointed: {relative}")
            symlink_states.append((link_name, before, actual_target, relative))
            symlinks.append({"path": relative, "target": actual_target})

        for relative, expected_bytes, expected_digest in executable_contract:
            executables.append(
                _attest_runtime_regular_file(
                    bin_descriptor,
                    relative,
                    expected_bytes,
                    expected_digest,
                    executable=True,
                    label="runtime executable",
                    nofollow=nofollow,
                )
            )

        for loader_relative, target_relative, expected_bytes, expected_digest in contract:
            loader_name = _runtime_relative_name(loader_relative, "runtime library loader path")
            target_name = _runtime_relative_name(target_relative, "runtime library target path")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 1
                or not isinstance(expected_digest, str)
                or not expected_digest.startswith("sha256:")
                or len(expected_digest) != 71
                or any(character not in "0123456789abcdef" for character in expected_digest[7:])
            ):
                raise ConversionRefused("runtime library contract identity is malformed")
            try:
                loader_before = os.stat(
                    loader_name,
                    dir_fd=bin_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ConversionRefused(
                    f"runtime library loader is unavailable: {loader_relative}: {exc}"
                ) from exc

            loader_target: str | None = None
            if loader_relative == target_relative:
                if not stat.S_ISREG(loader_before.st_mode):
                    raise ConversionRefused(
                        f"runtime library loader must be a regular file: {loader_relative}"
                    )
                target_before = loader_before
            else:
                if not stat.S_ISLNK(loader_before.st_mode):
                    raise ConversionRefused(
                        f"runtime library loader must be the pinned symlink: {loader_relative}"
                    )
                try:
                    loader_target = os.readlink(loader_name, dir_fd=bin_descriptor)
                except OSError as exc:
                    raise ConversionRefused(
                        f"runtime library symlink is unreadable: {loader_relative}: {exc}"
                    ) from exc
                if loader_target != target_name:
                    raise ConversionRefused(
                        f"runtime library symlink escaped or was repointed: {loader_relative}"
                    )
                try:
                    target_before = os.stat(
                        target_name,
                        dir_fd=bin_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise ConversionRefused(
                        f"runtime library target is unavailable: {target_relative}: {exc}"
                    ) from exc
                if not stat.S_ISREG(target_before.st_mode):
                    raise ConversionRefused(
                        f"runtime library final target must be regular: {target_relative}"
                    )

            mode = stat.S_IMODE(target_before.st_mode)
            if mode & 0o022:
                raise ConversionRefused(
                    f"runtime library target must not be group/world writable: {target_relative}"
                )
            if target_before.st_size != expected_bytes:
                raise ConversionRefused(f"runtime library target size changed: {target_relative}")

            target_descriptor = -1
            digest = hashlib.sha256()
            total = 0
            try:
                target_descriptor = os.open(
                    target_name,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
                    dir_fd=bin_descriptor,
                )
                opened = os.fstat(target_descriptor)
                if not stat.S_ISREG(opened.st_mode) or _stable_stat_fields(
                    opened
                ) != _stable_stat_fields(target_before):
                    raise ConversionRefused(
                        f"runtime library target changed while opening: {target_relative}"
                    )
                while True:
                    chunk = os.read(target_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_bytes:
                        raise ConversionRefused(
                            f"runtime library target grew while reading: {target_relative}"
                        )
                    digest.update(chunk)
                after_open = os.fstat(target_descriptor)
            except OSError as exc:
                raise ConversionRefused(
                    f"runtime library target could not be read: {target_relative}: {exc}"
                ) from exc
            finally:
                if target_descriptor >= 0:
                    os.close(target_descriptor)

            if (
                total != expected_bytes
                or "sha256:" + digest.hexdigest() != expected_digest
                or _stable_stat_fields(after_open) != _stable_stat_fields(target_before)
            ):
                raise ConversionRefused(f"runtime library target bytes changed: {target_relative}")
            try:
                target_after = os.stat(
                    target_name,
                    dir_fd=bin_descriptor,
                    follow_symlinks=False,
                )
                loader_after = os.stat(
                    loader_name,
                    dir_fd=bin_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ConversionRefused(
                    f"runtime library path changed after reading: {loader_relative}: {exc}"
                ) from exc
            if _stable_stat_fields(target_after) != _stable_stat_fields(
                target_before
            ) or _stable_stat_fields(loader_after) != _stable_stat_fields(loader_before):
                raise ConversionRefused(
                    f"runtime library path changed during inspection: {loader_relative}"
                )
            if loader_target is not None:
                try:
                    final_target = os.readlink(loader_name, dir_fd=bin_descriptor)
                except OSError as exc:
                    raise ConversionRefused(
                        f"runtime library symlink changed after reading: {loader_relative}: {exc}"
                    ) from exc
                if final_target != loader_target:
                    raise ConversionRefused(
                        f"runtime library symlink changed during inspection: {loader_relative}"
                    )

            libraries.append(
                {
                    "loader_path": loader_relative,
                    "target_path": target_relative,
                    "bytes": expected_bytes,
                    "digest": expected_digest,
                    "mode": f"{mode:04o}",
                }
            )
        final_namespace = tuple(sorted(os.listdir(bin_descriptor)))
        if final_namespace != actual_namespace:
            raise ConversionRefused(
                "llama.cpp build/bin namespace changed during runtime closure inspection"
            )
        final_executables = [
            _attest_runtime_regular_file(
                bin_descriptor,
                relative,
                expected_bytes,
                expected_digest,
                executable=True,
                label="runtime executable final recheck",
                nofollow=nofollow,
            )
            for relative, expected_bytes, expected_digest in executable_contract
        ]
        if final_executables != executables:
            raise ConversionRefused("runtime executable identities changed during inspection")
        for library, (
            _loader_relative,
            target_relative,
            expected_bytes,
            expected_digest,
        ) in zip(libraries, contract, strict=True):
            final_target_identity = _attest_runtime_regular_file(
                bin_descriptor,
                target_relative,
                expected_bytes,
                expected_digest,
                executable=False,
                label="runtime library target final recheck",
                nofollow=nofollow,
            )
            if final_target_identity != {
                "path": library["target_path"],
                "bytes": library["bytes"],
                "digest": library["digest"],
                "mode": library["mode"],
            }:
                raise ConversionRefused(
                    f"runtime library target identity changed during inspection: {target_relative}"
                )
        for link_name, before, target, relative in symlink_states:
            try:
                after = os.stat(
                    link_name,
                    dir_fd=bin_descriptor,
                    follow_symlinks=False,
                )
                final_target = os.readlink(link_name, dir_fd=bin_descriptor)
            except OSError as exc:
                raise ConversionRefused(
                    f"runtime symlink changed after inspection: {relative}: {exc}"
                ) from exc
            if _stable_stat_fields(after) != _stable_stat_fields(before) or final_target != target:
                raise ConversionRefused(f"runtime symlink changed during inspection: {relative}")
        if tuple(sorted(os.listdir(bin_descriptor))) != actual_namespace:
            raise ConversionRefused(
                "llama.cpp build/bin namespace changed after final runtime recheck"
            )
        for (_relative, descriptor, label), before in zip(
            directory_descriptors,
            directory_stats,
            strict=True,
        ):
            after = os.fstat(descriptor)
            if _stable_stat_fields(after) != _stable_stat_fields(before):
                raise ConversionRefused(f"{label} changed during runtime closure inspection")
            if stat.S_IMODE(after.st_mode) & 0o022:
                raise ConversionRefused(
                    f"{label} became group/world writable during runtime closure inspection"
                )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    return {
        "schema": RUNTIME_LIBRARY_SCHEMA,
        "root": str(root),
        "directories": directories,
        "build_bin_namespace": build_bin_namespace,
        "symlinks": symlinks,
        "executables": executables,
        "libraries": libraries,
    }


def _small_child_environment(*, single_thread: bool = False) -> dict[str, str]:
    """Return an offline child environment with no inherited secret values."""

    path = os.environ.get("PATH")
    if not path:
        raise ConversionRefused("PATH is required to resolve the converter's Python shebang")
    environment = {
        "PATH": path,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": _CHILD_PYCACHE_PREFIX,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "CUDA_VISIBLE_DEVICES": "",
    }
    if single_thread:
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
    return environment


def _assert_child_pycache_absent(cwd: Path, label: str) -> None:
    """Refuse stale or newly written bytecode under the private command cwd."""

    cwd = Path(cwd)
    if not cwd.is_absolute():
        raise ConversionRefused(f"{label} cwd must be absolute")
    try:
        cwd_stat = cwd.lstat()
    except OSError as exc:
        raise ConversionRefused(f"could not inspect {label} cwd: {exc}") from exc
    if stat.S_ISLNK(cwd_stat.st_mode) or not stat.S_ISDIR(cwd_stat.st_mode):
        raise ConversionRefused(f"{label} cwd must be a regular non-symlink directory")
    prefix = cwd / _CHILD_PYCACHE_PREFIX
    try:
        prefix_stat = prefix.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ConversionRefused(f"could not inspect {label} child pycache prefix: {exc}") from exc
    kind = "symlink" if stat.S_ISLNK(prefix_stat.st_mode) else "filesystem entry"
    raise ConversionRefused(f"{label} child pycache prefix must be absent; found {kind}")


def _assert_private_command_tree(cwd: Path, label: str) -> None:
    """Reject libraries, symlinks, and special files below a private command cwd."""

    cwd = Path(cwd)
    if not cwd.is_absolute():
        raise ConversionRefused(f"{label} cwd must be absolute")
    try:
        before = cwd.lstat()
    except OSError as exc:
        raise ConversionRefused(f"could not inspect {label} cwd: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ConversionRefused(f"{label} cwd must be a regular non-symlink directory")
    if stat.S_IMODE(before.st_mode) != 0o700:
        raise ConversionRefused(f"{label} cwd mode must remain 0700")

    def snapshot() -> tuple[tuple[str, tuple[int, ...]], ...]:
        entries: list[tuple[str, tuple[int, ...]]] = []
        for directory, directory_names, file_names, descriptor in os.fwalk(
            cwd,
            topdown=True,
            follow_symlinks=False,
        ):
            directory_names.sort()
            relative_directory = Path(directory).relative_to(cwd)
            for name in sorted((*directory_names, *file_names)):
                child = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                relative = (relative_directory / name).as_posix()
                if stat.S_ISLNK(child.st_mode):
                    raise ConversionRefused(f"{label} contains a symlink: {relative}")
                if not (stat.S_ISDIR(child.st_mode) or stat.S_ISREG(child.st_mode)):
                    raise ConversionRefused(
                        f"{label} contains a special filesystem entry: {relative}"
                    )
                if name.startswith("lib") and ".so" in name:
                    raise ConversionRefused(
                        f"{label} contains a loadable shared-library candidate: {relative}"
                    )
                entries.append((relative, _stable_stat_fields(child)))
        return tuple(entries)

    try:
        first = snapshot()
        second = snapshot()
    except ConversionRefused:
        raise
    except OSError as exc:
        raise ConversionRefused(f"could not inspect {label} command tree: {exc}") from exc
    if first != second:
        raise ConversionRefused(f"{label} command tree changed during inspection")

    try:
        after = cwd.lstat()
    except OSError as exc:
        raise ConversionRefused(f"{label} cwd changed during inspection: {exc}") from exc
    if _stable_stat_fields(after) != _stable_stat_fields(before):
        raise ConversionRefused(f"{label} cwd changed during inspection")


def _calibration_requested(request: ConversionRequest) -> bool:
    return any(
        value is not None
        for value in (
            request.calibration_profile,
            request.calibration_current_dataset,
            request.calibration_current_source_corpus,
            request.calibration_aux_dataset,
            request.calibration_aux_source_corpus,
            request.imatrix_tool,
        )
    )


def _read_only_command(argv: Sequence[str], *, cwd: Path) -> bytes:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=_small_child_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        error = bytes(completed.stderr)[-_MAX_ERROR_TEXT_BYTES:].decode("utf-8", "replace")
        raise ConversionRefused(f"read-only toolchain check failed: {error}")
    return bytes(completed.stdout)


def _bind_runtime_executable_identity(
    identity: Mapping[str, Any],
    closure: Mapping[str, Any],
    relative: str,
    label: str,
) -> None:
    entries = closure.get("executables")
    if not isinstance(entries, list):
        raise ConversionRefused("runtime executable closure is malformed")
    matches = [
        _mapping(entry, f"{label} runtime executable")
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("path") == relative
    ]
    if len(matches) != 1:
        raise ConversionRefused(f"{label} is absent or duplicated in runtime closure")
    expected = matches[0]
    expected_bytes = expected.get("bytes")
    identity_bytes = identity.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or isinstance(identity_bytes, bool)
        or not isinstance(identity_bytes, int)
    ):
        raise ConversionRefused(f"{label} byte identity is malformed")
    identity_mode = identity.get("mode")
    if not isinstance(identity_mode, str):
        raise ConversionRefused(f"{label} mode identity is malformed")
    try:
        normalized_mode = f"{int(identity_mode, 8):04o}"
    except ValueError as exc:
        raise ConversionRefused(f"{label} mode identity is malformed") from exc
    if {
        "path": identity.get("path"),
        "bytes": identity_bytes,
        "digest": identity.get("digest"),
        "mode": normalized_mode,
    } != {
        "path": str(LLAMA_CPP_ROOT / relative),
        "bytes": expected_bytes,
        "digest": expected.get("digest"),
        "mode": expected.get("mode"),
    }:
        raise ConversionRefused(f"{label} identity differs from runtime closure")


def _toolchain_identity(
    request: ConversionRequest,
    *,
    include_converter_python: bool = False,
) -> dict[str, Any]:
    root = _regular_directory(request.llama_cpp, "llama.cpp checkout")
    if root != LLAMA_CPP_ROOT:
        raise ConversionRefused(f"llama.cpp checkout must resolve exactly to {LLAMA_CPP_ROOT}")
    converter_python: Path | None = None
    converter_python_identity: dict[str, Any] | None = None
    if include_converter_python:
        if request.converter_python is None:
            raise ConversionRefused("current v4 conversion requires explicit converter Python")
        converter_python, converter_python_identity = _regular_executable(
            request.converter_python, "GGUF converter Python"
        )
    _assert_llama_python_import_surface(root)
    converter, converter_identity = _regular_executable(request.converter, "GGUF converter")
    quantizer, quantizer_identity = _regular_executable(request.quantizer, "GGUF quantizer")
    imatrix_identity: dict[str, Any] | None = None
    expected_converter = (root / "convert_hf_to_gguf.py").resolve(strict=True)
    expected_quantizer = (root / "build" / "bin" / "llama-quantize").resolve(strict=True)
    if converter != expected_converter:
        raise ConversionRefused("converter is not the pinned checkout's exact converter path")
    if quantizer != expected_quantizer:
        raise ConversionRefused("quantizer is not the pinned checkout's exact quantizer path")
    if _calibration_requested(request):
        if request.imatrix_tool is None:
            raise ConversionRefused("calibrated Q4 requires the exact llama-imatrix tool")
        imatrix, imatrix_identity = _regular_executable(request.imatrix_tool, "llama-imatrix")
        expected_imatrix = (root / "build" / "bin" / "llama-imatrix").resolve(strict=True)
        if imatrix != expected_imatrix:
            raise ConversionRefused("imatrix tool is not the pinned checkout's exact binary path")
    runtime_libraries = _runtime_library_closure(root)
    _bind_runtime_executable_identity(
        quantizer_identity,
        runtime_libraries,
        "build/bin/llama-quantize",
        "GGUF quantizer",
    )
    if imatrix_identity is not None:
        _bind_runtime_executable_identity(
            imatrix_identity,
            runtime_libraries,
            "build/bin/llama-imatrix",
            "llama-imatrix",
        )
    git = shutil.which("git", path=os.environ.get("PATH"))
    if git is None:
        raise ConversionRefused("git is required to validate the pinned llama.cpp checkout")
    git_path, _git_identity = _regular_executable(Path(git), "Git executable")
    top = (
        _read_only_command(
            (str(git_path), "-C", str(root), "rev-parse", "--show-toplevel"),
            cwd=root,
        )
        .decode("utf-8", "strict")
        .strip()
    )
    revision = (
        _read_only_command(
            (str(git_path), "-C", str(root), "rev-parse", "--verify", "HEAD"),
            cwd=root,
        )
        .decode("ascii", "strict")
        .strip()
    )
    dirty = _read_only_command(
        (str(git_path), "-C", str(root), "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
    )
    if Path(top).resolve(strict=True) != root:
        raise ConversionRefused("llama.cpp path is not the checkout top level")
    if revision != LLAMA_CPP_REVISION:
        raise ConversionRefused(
            f"llama.cpp revision changed: expected {LLAMA_CPP_REVISION}, got {revision}"
        )
    if dirty:
        raise ConversionRefused("llama.cpp tracked tree is dirty")
    result = {
        "root": str(root),
        "revision": revision,
        "converter": converter_identity,
        "quantizer": quantizer_identity,
        "runtime_libraries": runtime_libraries,
    }
    if converter_python is not None and converter_python_identity is not None:
        result["converter_python"] = {
            **converter_python_identity,
            "path": str(converter_python),
        }
    if imatrix_identity is not None:
        result["imatrix"] = imatrix_identity
    return result


def _legacy_conversion_source(lineage: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _mapping(lineage.get("receipt"), "training receipt identity")
    run = _mapping(lineage.get("run"), "training run identity")
    merged = _mapping(run.get("merged"), "merged HF tree identity")
    return {
        "training_metadata_digest": receipt.get("digest"),
        "merged_tree_digest": merged.get("digest"),
    }


def _normalized_conversion_source(lineage: Mapping[str, Any]) -> dict[str, Any]:
    prepared = _mapping(lineage.get("prepared_dataset"), "normalized prepared dataset identity")
    manifest = _mapping(
        prepared.get("manifest_payload"),
        "normalized prepared manifest payload",
    )
    excluded = _mapping(
        prepared.get("excluded_refs"),
        "normalized excluded refs identity",
    )
    required_manifest = {
        "schema": normalized_candidate.DATASET_SCHEMA,
        "corpus_profile": normalized_candidate.CORPUS_PROFILE,
        "source_examples": normalized_candidate.EXPECTED_SOURCE_EXAMPLES,
        "train_examples": normalized_candidate.EXPECTED_TRAIN_EXAMPLES,
        "holdout_examples": normalized_candidate.EXPECTED_HOLDOUT_EXAMPLES,
        "excluded_examples": normalized_candidate.EXPECTED_EXCLUDED_EXAMPLES,
        "excluded_refs_file": normalized_candidate.EXCLUDED_REFS_FILE,
        "excluded_refs_canonical_bytes": (
            normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES
        ),
        "excluded_refs_digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
    }
    for field, expected in required_manifest.items():
        if manifest.get(field) != expected:
            raise ConversionRefused(f"normalized training manifest field {field!r} changed")
    excluded_content = {
        "bytes": excluded.get("bytes"),
        "digest": excluded.get("digest"),
    }
    expected_excluded_content = {
        "bytes": normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
        "digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
    }
    if excluded_content != expected_excluded_content:
        raise ConversionRefused("normalized excluded refs identity changed")
    return {
        "training_schema": gguf.TRAINING_SCHEMA_V6,
        "dataset_schema": normalized_candidate.DATASET_SCHEMA,
        "corpus_profile": normalized_candidate.CORPUS_PROFILE,
        **_legacy_conversion_source(lineage),
        "excluded_refs": excluded_content,
    }


def _current_conversion_source(lineage: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _mapping(lineage.get("receipt"), "current training receipt identity")
    run = _mapping(lineage.get("run"), "current training run identity")
    metrics = _mapping(run.get("metrics"), "current training metrics identity")
    merged = _mapping(run.get("merged"), "current merged HF tree identity")
    source = _mapping(lineage.get("source_corpus"), "current source-corpus identity")
    source_file = _mapping(source.get("file"), "current source-corpus file identity")
    prepared = _mapping(lineage.get("prepared_dataset"), "current prepared dataset identity")
    manifest_identity = _mapping(prepared.get("manifest"), "current prepared manifest identity")
    train_identity = _mapping(prepared.get("train"), "current prepared train identity")
    holdout_identity = _mapping(prepared.get("holdout"), "current prepared holdout identity")
    manifest = _mapping(prepared.get("manifest_payload"), "current prepared manifest payload")
    base = _mapping(lineage.get("base_snapshot"), "current base snapshot identity")
    result = {
        "training_schema": CURRENT_TRAINING_SCHEMA,
        "dataset_schema": candidate.DATASET_SCHEMA,
        "corpus_profile": CURRENT_CORPUS_PROFILE,
        "training_metadata_digest": receipt.get("digest"),
        "training_metrics_digest": metrics.get("digest"),
        "merged_tree_digest": merged.get("digest"),
        "source_corpus": {
            "bytes": source_file.get("bytes"),
            "digest": source_file.get("digest"),
            "canonical_bytes": source.get("canonical_bytes"),
            "canonical_digest": source.get("canonical_digest"),
            "task_count": source.get("task_count"),
            "refs_digest": source.get("refs_digest"),
        },
        "prepared_dataset": {
            "manifest_digest": manifest_identity.get("digest"),
            "train_digest": train_identity.get("digest"),
            "holdout_digest": holdout_identity.get("digest"),
            "train_examples": manifest.get("train_examples"),
            "holdout_examples": manifest.get("holdout_examples"),
        },
        "base_snapshot": dict(base),
    }
    _exact_keys(result, CURRENT_CONVERSION_SOURCE_KEYS, "current conversion source")
    return result


def _lineage_model_contract(lineage: Mapping[str, Any]) -> tuple[str, str]:
    schema = lineage.get("schema")
    base = _mapping(lineage.get("base_snapshot"), "training base snapshot identity")
    model = base.get("base_model")
    if schema == CURRENT_TRAINING_SCHEMA:
        if model != candidate.QWEN25_CODER_1_5B_BASE_MODEL:
            raise ConversionRefused(
                "current v4 lineage is not bound to the pinned Qwen2.5-Coder-1.5B base"
            )
        return candidate.QWEN25_CODER_1_5B_BASE_MODEL, QWEN25_ARCHITECTURE
    if schema in {gguf.TRAINING_SCHEMA_V5, gguf.TRAINING_SCHEMA_V6}:
        if model != candidate.QWEN3_BASE_MODEL:
            raise ConversionRefused("historical lineage is not bound to the pinned Qwen3 base")
        return candidate.QWEN3_BASE_MODEL, QWEN3_ARCHITECTURE
    raise ConversionRefused("training lineage schema is not explicit v4, v5, or v6")


def _calibration_schema(lineage: Mapping[str, Any]) -> str:
    if lineage.get("schema") == CURRENT_TRAINING_SCHEMA:
        return CURRENT_CALIBRATION_SCHEMA
    if lineage.get("schema") in {gguf.TRAINING_SCHEMA_V5, gguf.TRAINING_SCHEMA_V6}:
        return CALIBRATION_SCHEMA
    raise ConversionRefused("training lineage schema is not explicit v4, v5, or v6")


def _validate_current_loaded_lineage(lineage: Mapping[str, Any]) -> None:
    receipt = _mapping(lineage.get("receipt"), "current training receipt identity")
    source = _mapping(lineage.get("source_corpus"), "current training source-corpus identity")
    _exact_keys(
        source,
        frozenset(
            {
                "file",
                "corpus_version",
                "canonical_bytes",
                "canonical_digest",
                "task_count",
                "refs_digest",
            }
        ),
        "current training source-corpus identity",
    )
    source_file = _mapping(source.get("file"), "current training source file identity")
    _exact_keys(source_file, frozenset({"bytes", "digest"}), "current training source file")
    if source_file != {
        "bytes": gguf.CURRENT94_PUBLIC_CORPUS_BYTES,
        "digest": gguf.CURRENT94_PUBLIC_CORPUS_RAW_DIGEST,
    }:
        raise ConversionRefused("current training source is not the exact current94 raw response")
    if source.get("corpus_version") != candidate.CORPUS_VERSION:
        raise ConversionRefused("current training source corpus version changed")
    if source.get("canonical_digest") != candidate.PUBLIC_CORPUS_CANONICAL_DIGEST:
        raise ConversionRefused("current training source canonical digest changed")
    if source.get("task_count") != candidate.EXPECTED_COUNTS["train"]:
        raise ConversionRefused("current training source task count changed")
    if (
        isinstance(source.get("canonical_bytes"), bool)
        or not isinstance(source.get("canonical_bytes"), int)
        or int(source["canonical_bytes"]) < 1
        or not _valid_digest(source.get("refs_digest"))
    ):
        raise ConversionRefused("current training source replay identity is malformed")

    prepared = _mapping(lineage.get("prepared_dataset"), "current prepared dataset identity")
    _exact_keys(
        prepared,
        frozenset({"manifest", "train", "holdout", "manifest_payload"}),
        "current prepared dataset identity",
    )
    manifest_identity = _mapping(prepared.get("manifest"), "current prepared manifest identity")
    train_identity = _mapping(prepared.get("train"), "current prepared train identity")
    holdout_identity = _mapping(prepared.get("holdout"), "current prepared holdout identity")
    for identity, label in (
        (manifest_identity, "current prepared manifest"),
        (train_identity, "current prepared train"),
        (holdout_identity, "current prepared holdout"),
    ):
        _exact_keys(identity, frozenset({"bytes", "digest"}), f"{label} identity")
        if (
            isinstance(identity.get("bytes"), bool)
            or not isinstance(identity.get("bytes"), int)
            or int(identity["bytes"]) < 0
            or not _valid_digest(identity.get("digest"))
        ):
            raise ConversionRefused(f"{label} identity is malformed")
    manifest = _mapping(prepared.get("manifest_payload"), "current prepared manifest payload")
    _exact_keys(manifest, candidate.PREPARED_MANIFEST_KEYS, "current prepared manifest")
    required_manifest = {
        "schema": candidate.DATASET_SCHEMA,
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "corpus_version": candidate.CORPUS_VERSION,
        "corpus_canonical_digest": candidate.PUBLIC_CORPUS_CANONICAL_DIGEST,
        "source_file_digest": gguf.CURRENT94_PUBLIC_CORPUS_RAW_DIGEST,
        "split_algorithm": candidate.SPLIT_ALGORITHM,
        "seed": gguf.DIAGNOSTIC_SEED,
        "train_examples": candidate.EXPECTED_COUNTS["train"],
        "holdout_examples": 0,
        "holdout_file_digest": candidate.digest_bytes(b""),
        "target_construction": "inputs.code_prompt + gold",
        "quality_claim": candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }
    for field, expected in required_manifest.items():
        if manifest.get(field) != expected:
            raise ConversionRefused(f"current training manifest field {field!r} changed")
    if train_identity.get("digest") != manifest.get("train_file_digest"):
        raise ConversionRefused("current prepared train identity differs from its manifest")
    if holdout_identity.get("digest") != manifest.get("holdout_file_digest"):
        raise ConversionRefused("current prepared holdout identity differs from its manifest")

    run = _mapping(lineage.get("run"), "current training run identity")
    _exact_keys(
        run,
        frozenset({"kind", "training_metadata", "metrics", "adapter", "merged"}),
        "current training run identity",
    )
    training_metadata = _mapping(
        run.get("training_metadata"),
        "current deep-validated training metadata identity",
    )
    metrics = _mapping(run.get("metrics"), "current training metrics identity")
    if dict(receipt) != dict(training_metadata):
        raise ConversionRefused("current training receipt differs from deep validation")
    if not _valid_digest(metrics.get("digest")):
        raise ConversionRefused("current training metrics digest is malformed")
    _lineage_model_contract(lineage)
    source_binding = _current_conversion_source(lineage)
    current_source = _mapping(
        source_binding.get("source_corpus"),
        "current conversion source corpus",
    )
    current_prepared = _mapping(
        source_binding.get("prepared_dataset"),
        "current conversion prepared dataset",
    )
    _exact_keys(current_source, CURRENT_SOURCE_CORPUS_KEYS, "current conversion source corpus")
    _exact_keys(
        current_prepared,
        CURRENT_PREPARED_DATASET_KEYS,
        "current conversion prepared dataset",
    )


def _conversion_source(lineage: Mapping[str, Any]) -> dict[str, Any]:
    schema = lineage.get("schema")
    if schema == CURRENT_TRAINING_SCHEMA:
        return _current_conversion_source(lineage)
    if schema == gguf.TRAINING_SCHEMA_V5:
        return _legacy_conversion_source(lineage)
    if schema == gguf.TRAINING_SCHEMA_V6:
        return _normalized_conversion_source(lineage)
    raise ConversionRefused("training lineage schema is not explicit v4, v5, or v6")


def _conversion_schema(lineage: Mapping[str, Any], *, calibrated: bool) -> str:
    schema = lineage.get("schema")
    if schema == gguf.TRAINING_SCHEMA_V5:
        return CALIBRATED_CONVERSION_SCHEMA if calibrated else SCHEMA
    if schema == gguf.TRAINING_SCHEMA_V6:
        return (
            NORMALIZED_CALIBRATED_CONVERSION_SCHEMA if calibrated else NORMALIZED_CONVERSION_SCHEMA
        )
    if schema == CURRENT_TRAINING_SCHEMA:
        if not calibrated:
            raise ConversionRefused("current v4/Qwen2.5 conversion requires calibrated Q4_K_M")
        return CURRENT_CALIBRATED_CONVERSION_SCHEMA
    raise ConversionRefused("training lineage schema is not explicit v4, v5, or v6")


def _validate_loaded_lineage(lineage: Mapping[str, Any]) -> None:
    _exact_keys(
        lineage,
        frozenset(
            {
                "status",
                "schema",
                "receipt",
                "source_corpus",
                "prepared_dataset",
                "base_snapshot",
                "run",
                "conversion_binding_claim",
            }
        ),
        "training lineage",
    )
    if lineage.get("status") != "provided_and_validated":
        raise ConversionRefused("training lineage is not completed and validated")
    schema = lineage.get("schema")
    if schema not in {
        CURRENT_TRAINING_SCHEMA,
        gguf.TRAINING_SCHEMA_V5,
        gguf.TRAINING_SCHEMA_V6,
    }:
        raise ConversionRefused("training lineage schema is not explicit v4, v5, or v6")
    receipt = _mapping(lineage.get("receipt"), "training receipt identity")
    run = _mapping(lineage.get("run"), "training run identity")
    if run.get("kind") != "merged":
        raise ConversionRefused("training lineage does not identify a completed merged run")
    merged = _mapping(run.get("merged"), "merged HF tree identity")
    for value, label in (
        (receipt.get("digest"), "training metadata digest"),
        (merged.get("digest"), "merged HF tree digest"),
    ):
        if not _valid_digest(value):
            raise ConversionRefused(f"{label} is malformed")
    _lineage_model_contract(lineage)
    source = _mapping(lineage.get("source_corpus"), "training source-corpus identity")
    source_file = _mapping(source.get("file"), "training source-corpus file identity")
    if schema == CURRENT_TRAINING_SCHEMA:
        _validate_current_loaded_lineage(lineage)
        return
    prepared = _mapping(lineage.get("prepared_dataset"), "training prepared dataset identity")
    manifest = _mapping(prepared.get("manifest_payload"), "training prepared manifest payload")
    if not _valid_digest(source_file.get("digest")):
        raise ConversionRefused("training source-corpus digest is malformed")
    if not (
        source_file.get("digest") == source.get("raw_digest") == manifest.get("source_file_digest")
    ):
        raise ConversionRefused("training lineage is not bound to one source corpus")

    if schema == gguf.TRAINING_SCHEMA_V5:
        _exact_keys(
            prepared,
            frozenset({"manifest", "train", "holdout", "manifest_payload"}),
            "v5 prepared dataset identity",
        )
        required = {
            "schema": historical_candidate.DATASET_SCHEMA,
            "seed": gguf.DIAGNOSTIC_SEED,
            "train_examples": historical_candidate.EXPECTED_COUNTS["train"],
            "holdout_examples": 0,
            "target_construction": historical_candidate.TARGET_CONSTRUCTION,
        }
        for field, expected in required.items():
            if manifest.get(field) != expected:
                raise ConversionRefused(f"v5 training manifest field {field!r} changed")
    else:
        _exact_keys(
            prepared,
            frozenset({"manifest", "train", "holdout", "excluded_refs", "manifest_payload"}),
            "v6 prepared dataset identity",
        )
        _normalized_conversion_source(lineage)


def _load_lineage(request: ConversionRequest) -> dict[str, Any]:
    try:
        lineage, _modules = gguf.load_training_lineage(
            request.training_run,
            request.training_dataset,
            request.source_corpus,
            request.base,
        )
    except Exception as exc:
        raise ConversionRefused(f"source-bound training lineage was refused: {exc}") from exc
    validated = _mapping(lineage, "training lineage")
    _validate_loaded_lineage(validated)
    return dict(validated)


def _conversion_command(
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path,
) -> dict[str, Any]:
    exact_argv = [str(item) for item in argv]
    if not exact_argv or any(not item for item in exact_argv):
        raise ConversionRefused(f"{name} argv is malformed")
    _assert_child_pycache_absent(cwd, f"{name} before launch")
    _assert_private_command_tree(cwd, f"{name} before launch")
    started = time.time_ns()
    try:
        completed = subprocess.run(
            exact_argv,
            cwd=cwd,
            env=_small_child_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise ConversionRefused(f"{name} could not start: {exc}") from exc
    finally:
        _assert_child_pycache_absent(cwd, f"{name} after exit")
        _assert_private_command_tree(cwd, f"{name} after exit")
    finished = time.time_ns()
    record = {
        "name": name,
        "argv": exact_argv,
        "returncode": completed.returncode,
        "started_at_unix_ns": started,
        "finished_at_unix_ns": finished,
    }
    if completed.returncode != 0:
        error = bytes(completed.stderr)[-_MAX_ERROR_TEXT_BYTES:].decode("utf-8", "replace")
        raise ConversionRefused(f"{name} failed with return code {completed.returncode}: {error}")
    return record


def _write_all(descriptor: int, raw: bytes, label: str) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise ConversionRefused(f"short write while capturing {label}")
        view = view[written:]


def _bounded_conversion_command(
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    log_root: Path,
    cwd_role: str = "private_staging",
    executable_path: Path | None = None,
    executable_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one calibrated command with bounded file-backed stdout/stderr capture."""

    exact_argv = [str(item) for item in argv]
    if not exact_argv or any(not item for item in exact_argv):
        raise ConversionRefused(f"{name} argv is malformed")
    if not name.replace("_", "").isalnum():
        raise ConversionRefused("calibrated command name is malformed")
    if cwd_role not in {"private_staging", "determinism_replay"}:
        raise ConversionRefused("calibrated command cwd role is malformed")
    _assert_child_pycache_absent(cwd, f"{name} before launch")
    _assert_private_command_tree(cwd, f"{name} before launch")
    environment = _small_child_environment(single_thread=True)
    descriptors: dict[str, int] = {}
    paths: dict[str, Path] = {}
    try:
        for stream_name in ("stdout", "stderr"):
            path = log_root / f"{name}.{stream_name}"
            descriptors[stream_name] = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            paths[stream_name] = path
    except OSError as exc:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise ConversionRefused(f"could not create bounded {name} logs: {exc}") from exc

    hashers = {stream_name: hashlib.sha256() for stream_name in descriptors}
    captured_hashers = {stream_name: hashlib.sha256() for stream_name in descriptors}
    totals = {stream_name: 0 for stream_name in descriptors}
    captured = {stream_name: 0 for stream_name in descriptors}
    stderr_tail = bytearray()
    process: subprocess.Popen[bytes] | None = None
    executable_fd: int | None = None
    executed_object: dict[str, Any] | None = None
    selector = selectors.DefaultSelector()
    started = time.time_ns()
    primary_error: BaseException | None = None
    try:
        popen_options: dict[str, Any] = {}
        if executable_path is not None:
            if executable_identity is None:
                raise ConversionRefused(f"{name} fd launch lacks an expected executable identity")
            executable_fd = os.open(
                executable_path,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            executed_object = _fd_executable_identity(
                executable_fd,
                executable_path,
                f"{name} held executable",
            )
            if executed_object != _converter_python_receipt_identity(
                executable_identity,
                f"{name} expected executable identity",
            ):
                raise ConversionRefused(f"{name} held executable differs from its attestation")
            popen_options = {
                "executable": f"/proc/self/fd/{executable_fd}",
                "pass_fds": (executable_fd,),
            }
        process = subprocess.Popen(
            exact_argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **popen_options,
        )
        if process.stdout is None or process.stderr is None:
            raise ConversionRefused(f"{name} did not expose output pipes")
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            for key, _mask in selector.select():
                stream_name = str(key.data)
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                hashers[stream_name].update(chunk)
                totals[stream_name] += len(chunk)
                remaining = MAX_CAPTURED_LOG_BYTES - captured[stream_name]
                if remaining > 0:
                    prefix = chunk[:remaining]
                    _write_all(descriptors[stream_name], prefix, f"{name} {stream_name}")
                    captured_hashers[stream_name].update(prefix)
                    captured[stream_name] += len(prefix)
                if stream_name == "stderr":
                    stderr_tail.extend(chunk)
                    if len(stderr_tail) > _MAX_ERROR_TEXT_BYTES:
                        del stderr_tail[:-_MAX_ERROR_TEXT_BYTES]
        returncode = process.wait()
        for descriptor in descriptors.values():
            os.fsync(descriptor)
    except OSError as exc:
        primary_error = ConversionRefused(f"{name} could not run with bounded logs: {exc}")
        primary_error.__cause__ = exc
    except BaseException as exc:
        primary_error = exc
    finally:
        finished = time.time_ns()
        cleanup_error: BaseException | None = None
        try:
            selector.close()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            for descriptor in descriptors.values():
                os.close(descriptor)
            if process is not None:
                _assert_child_pycache_absent(cwd, f"{name} after exit")
                _assert_private_command_tree(cwd, f"{name} after exit")
        except BaseException as exc:
            cleanup_error = exc
        finally:
            if executable_fd is not None:
                try:
                    after = _fd_executable_identity(
                        executable_fd,
                        executable_path,
                        f"{name} held executable after exit",
                    )
                    if executed_object != after:
                        raise ConversionRefused(
                            f"{name} held executable changed during execution"
                        )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    elif hasattr(cleanup_error, "add_note"):
                        cleanup_error.add_note(f"held executable post-check also failed: {exc}")
                finally:
                    try:
                        os.close(executable_fd)
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                        elif hasattr(cleanup_error, "add_note"):
                            cleanup_error.add_note(f"held executable close also failed: {exc}")
        if primary_error is not None:
            if cleanup_error is not None and hasattr(primary_error, "add_note"):
                primary_error.add_note(f"command cleanup also failed: {cleanup_error}")
            raise primary_error.with_traceback(primary_error.__traceback__)
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_error.__traceback__)

    record: dict[str, Any] = {
        "name": name,
        "argv": exact_argv,
        "cwd_role": cwd_role,
        "environment": environment,
        "returncode": returncode,
        "started_at_unix_ns": started,
        "finished_at_unix_ns": finished,
    }
    if executed_object is not None:
        record["launch"] = {
            "method": "proc-self-fd",
            "executed_object": executed_object,
        }
    for stream_name in ("stdout", "stderr"):
        record[stream_name] = {
            "bytes": totals[stream_name],
            "captured_bytes": captured[stream_name],
            "captured_digest": "sha256:" + captured_hashers[stream_name].hexdigest(),
            "digest": "sha256:" + hashers[stream_name].hexdigest(),
            "truncated": totals[stream_name] > captured[stream_name],
        }
    if returncode != 0:
        error = bytes(stderr_tail).decode("utf-8", "replace")
        raise ConversionRefused(f"{name} failed with return code {returncode}: {error}")
    for stream_name, path in paths.items():
        identity = gguf.file_identity(path, f"bounded {name} {stream_name} log")
        if identity["bytes"] != captured[stream_name]:
            raise ConversionRefused(f"bounded {name} {stream_name} log changed")
    return record


def _content_identity(path: Path, label: str) -> dict[str, Any]:
    identity = gguf.file_identity(path, label)
    return {"bytes": identity["bytes"], "digest": identity["digest"]}


def _dataset_identity(root: Path, manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
    schema = manifest.get("schema")
    expected_files = {"manifest.json", "train.jsonl", "holdout.jsonl"}
    if schema == normalized_candidate.DATASET_SCHEMA:
        expected_files.add(normalized_candidate.EXCLUDED_REFS_FILE)
    elif schema not in {candidate.DATASET_SCHEMA, historical_candidate.DATASET_SCHEMA}:
        raise ConversionRefused(f"{label} schema is unsupported")
    if _bundle_file_set(root) != expected_files:
        raise ConversionRefused(f"{label} contains unexpected files")
    strict_manifest = _strict_json_file(root / "manifest.json", f"{label} manifest")
    if strict_manifest != dict(manifest):
        raise ConversionRefused(f"{label} manifest changed under strict replay")
    result = {
        "tree_digest": _official_tree_digest(root),
        "manifest": strict_manifest,
        "manifest_file": _content_identity(root / "manifest.json", f"{label} manifest"),
        "train_file": _content_identity(root / "train.jsonl", f"{label} train rows"),
        "holdout_file": _content_identity(root / "holdout.jsonl", f"{label} holdout rows"),
    }
    if schema == normalized_candidate.DATASET_SCHEMA:
        excluded = _content_identity(
            root / normalized_candidate.EXCLUDED_REFS_FILE,
            f"{label} excluded refs",
        )
        expected_excluded = {
            "bytes": normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
            "digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
        }
        if excluded != expected_excluded or excluded != {
            "bytes": manifest.get("excluded_refs_canonical_bytes"),
            "digest": manifest.get("excluded_refs_digest"),
        }:
            raise ConversionRefused(f"{label} excluded refs identity changed")
        result["excluded_refs_file"] = excluded
    return result


def _strict_current_jsonl_matches(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ConversionRefused(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ConversionRefused(f"{label} exceeds the reviewed byte ceiling")
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ConversionRefused(f"{label} must be canonical LF-terminated JSONL")
    lines = raw[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise ConversionRefused(f"{label} contains a blank row")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(lines, 1):
            value = candidate._strict_json(line, f"{label} row {number}")
            row = dict(_mapping(value, f"{label} row {number}"))
            _exact_keys(row, candidate.PREPARED_ROW_KEYS, f"{label} row {number}")
            rows.append(row)
    except candidate.CandidateError as exc:
        raise ConversionRefused(f"{label} failed strict replay: {exc}") from exc
    if rows != [dict(row) for row in expected]:
        raise ConversionRefused(f"{label} differs from the validated prepared rows")


def _refs_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return candidate._refs_digest([str(row["ref"]) for row in rows])


def _load_calibration_material(
    request: ConversionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay both pinned public corpora and select only non-diagnostic rows."""

    if request.calibration_current_dataset is None:
        raise ConversionRefused("calibration current dataset is missing")
    if request.calibration_current_source_corpus is None:
        raise ConversionRefused("calibration current source corpus is missing")
    current_root = candidate.assert_tmpfs_path(
        request.calibration_current_dataset,
        must_exist=True,
    )
    current_source = candidate.assert_tmpfs_path(
        request.calibration_current_source_corpus,
        must_exist=True,
    )
    aux_dataset = request.calibration_aux_dataset
    aux_source_corpus = request.calibration_aux_source_corpus
    if (aux_dataset is None) != (aux_source_corpus is None):
        raise ConversionRefused("auxiliary calibration arguments must be supplied as a pair")
    auxiliary_pool = aux_dataset is not None
    pool_dataset = request.training_dataset if aux_dataset is None else aux_dataset
    pool_source_corpus = request.source_corpus if aux_source_corpus is None else aux_source_corpus
    historical_root = candidate.assert_tmpfs_path(pool_dataset, must_exist=True)
    historical_source = candidate.assert_tmpfs_path(pool_source_corpus, must_exist=True)
    try:
        payload, current_validation = candidate.load_public_corpus(current_source)
        current_rows, current_manifest = candidate.load_prepared_dataset(current_root)
        current_holdout = candidate._load_prepared_rows(
            current_root / "holdout.jsonl",
            "current holdout.jsonl",
        )
        expected_train, expected_holdout = gguf._replay_current94(payload, current_manifest)
        if current_rows != expected_train or current_holdout != expected_holdout:
            raise ConversionRefused("current prepared dataset is not an exact source replay")
        _strict_current_jsonl_matches(
            current_root / "train.jsonl",
            current_rows,
            "current train JSONL",
        )
        _strict_current_jsonl_matches(
            current_root / "holdout.jsonl",
            current_holdout,
            "current holdout JSONL",
        )
        historical_header = _strict_json_file(
            historical_root / "manifest.json",
            "training calibration manifest",
        )
        historical_schema = historical_header.get("schema")
        if auxiliary_pool and historical_schema != normalized_candidate.DATASET_SCHEMA:
            raise ConversionRefused(
                "auxiliary calibration pool must be the exact normalized historical dataset"
            )
        if historical_schema == historical_candidate.DATASET_SCHEMA:
            historical_rows, historical_manifest = historical_candidate.load_prepared_dataset(
                historical_root,
                historical_source,
            )
            expected_historical_split = (
                CALIBRATION_SEED,
                historical_candidate.EXPECTED_COUNTS["train"],
                0,
            )
            historical_source_identity = historical_candidate.source_corpus_identity()
        elif historical_schema == normalized_candidate.DATASET_SCHEMA:
            historical_rows, historical_manifest = normalized_candidate.load_prepared_dataset(
                historical_root,
                historical_source,
            )
            expected_historical_split = (
                normalized_candidate.EXPECTED_SEED,
                normalized_candidate.EXPECTED_TRAIN_EXAMPLES,
                normalized_candidate.EXPECTED_HOLDOUT_EXAMPLES,
            )
            historical_source_identity = normalized_candidate.source_corpus_identity()
        else:
            raise ConversionRefused("training calibration manifest schema is unsupported")
    except ConversionRefused:
        raise
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise ConversionRefused(f"public calibration lineage was refused: {exc}") from exc

    if (
        current_manifest.get("seed"),
        current_manifest.get("train_examples"),
        current_manifest.get("holdout_examples"),
    ) != (CALIBRATION_SEED, CALIBRATION_CURRENT_ROWS, CALIBRATION_DIAGNOSTIC_ROWS):
        raise ConversionRefused("current calibration split is not the exact seed-92 78/16 split")
    if (
        historical_manifest.get("seed"),
        historical_manifest.get("train_examples"),
        historical_manifest.get("holdout_examples"),
    ) != expected_historical_split:
        raise ConversionRefused("training calibration pool is not its exact final split")
    if (
        current_manifest.get("source_file_digest")
        != _content_identity(
            current_source,
            "current public source corpus",
        )["digest"]
    ):
        raise ConversionRefused("current prepared dataset does not bind the source bytes")

    current_refs = [str(row["ref"]) for row in current_rows]
    holdout_refs = [str(row["ref"]) for row in current_holdout]
    historical_refs = [str(row["ref"]) for row in historical_rows]
    if len(current_rows) != CALIBRATION_CURRENT_ROWS:
        raise ConversionRefused("current calibration training row count changed")
    if len(current_holdout) != CALIBRATION_DIAGNOSTIC_ROWS:
        raise ConversionRefused("current diagnostic holdout row count changed")
    if len(set(current_refs)) != len(current_refs):
        raise ConversionRefused("current calibration refs are duplicated")
    if len(set(holdout_refs)) != len(holdout_refs):
        raise ConversionRefused("current diagnostic refs are duplicated")
    if set(current_refs) & set(holdout_refs):
        raise ConversionRefused("current diagnostic rows leaked into calibration training rows")
    if len(historical_rows) != expected_historical_split[1]:
        raise ConversionRefused("historical calibration pool row count changed")
    if len(set(historical_refs)) != len(historical_refs):
        raise ConversionRefused("historical calibration refs are duplicated")

    ranked_historical = sorted(
        historical_rows,
        key=lambda row: (
            hashlib.sha256(f"{CALIBRATION_SEED}:{row['ref']}".encode()).hexdigest(),
            str(row["ref"]),
        ),
    )
    selected_historical = ranked_historical[:CALIBRATION_HISTORICAL_ROWS]
    ordered_current = sorted(current_rows, key=lambda row: str(row["ref"]))
    ordered_rows = [dict(row) for row in (*ordered_current, *selected_historical)]
    if len(ordered_rows) != CALIBRATION_TOTAL_ROWS:
        raise ConversionRefused("calibration selection did not produce exactly 512 rows")

    control_markers = ("<|im_start|>", CALIBRATION_EOS_TOKEN, "<|endoftext|>")
    for row in ordered_rows:
        _exact_keys(row, candidate.PREPARED_ROW_KEYS, "calibration row")
        for field in ("prompt", "completion"):
            text = row.get(field)
            if not isinstance(text, str) or not text:
                raise ConversionRefused(f"calibration row {field} is not a non-empty string")
            if any(marker in text for marker in control_markers):
                raise ConversionRefused("calibration source contains a reserved Qwen control token")

    current_source_identity = _content_identity(current_source, "current public source corpus")
    pool_source_key = "auxiliary_normalized_historical" if auxiliary_pool else "historical"
    pool_label = "auxiliary normalized historical" if auxiliary_pool else "historical"
    pool_selection = (
        {
            "auxiliary_pool_rows": len(historical_rows),
            "auxiliary_selected_rows": len(selected_historical),
            "auxiliary_selected_refs_digest": _refs_digest(selected_historical),
        }
        if auxiliary_pool
        else {
            "historical_pool_rows": len(historical_rows),
            "historical_selected_rows": len(selected_historical),
            "historical_selected_refs_digest": _refs_digest(selected_historical),
        }
    )
    snapshot = {
        "profile": CALIBRATION_PROFILE,
        "source": {
            "current": {
                "corpus": {
                    **current_source_identity,
                    "canonical_bytes": current_validation.canonical_bytes,
                    "canonical_digest": current_validation.canonical_digest,
                    "task_count": current_validation.task_count,
                    "refs_digest": current_validation.refs_digest,
                },
                "prepared_dataset": _dataset_identity(
                    current_root,
                    current_manifest,
                    "current prepared dataset",
                ),
            },
            pool_source_key: {
                "corpus": historical_source_identity,
                "prepared_dataset": _dataset_identity(
                    historical_root,
                    historical_manifest,
                    f"{pool_label} prepared dataset",
                ),
            },
        },
        "selection": {
            "algorithm": CALIBRATION_SELECTION_ALGORITHM,
            "seed": CALIBRATION_SEED,
            "current_rows": len(ordered_current),
            "current_refs_digest": _refs_digest(ordered_current),
            "diagnostic_rows_excluded": len(current_holdout),
            "diagnostic_refs_digest": _refs_digest(current_holdout),
            **pool_selection,
            "total_rows": len(ordered_rows),
        },
    }
    return ordered_rows, snapshot


def _write_calibration_corpus(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        for row in rows:
            raw = (
                str(row["prompt"]) + str(row["completion"]) + CALIBRATION_EOS_TOKEN + "\n"
            ).encode("utf-8")
            if total + len(raw) > CALIBRATION_MAX_BYTES:
                raise ConversionRefused("rendered calibration corpus exceeds 16 MiB")
            _write_all(descriptor, raw, "calibration corpus")
            digest.update(raw)
            total += len(raw)
        if len(rows) != CALIBRATION_TOTAL_ROWS or total < 1:
            raise ConversionRefused("rendered calibration corpus has an invalid row count or size")
        os.fsync(descriptor)
    except OSError as exc:
        raise ConversionRefused(f"could not write private calibration corpus: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {
        "name": CALIBRATION_CORPUS_NAME,
        "bytes": total,
        "digest": "sha256:" + digest.hexdigest(),
    }


_GGUF_SCALAR_FORMAT: Final[dict[int, str]] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_GGUF_STRING_TYPE: Final[int] = 8
_GGUF_ARRAY_TYPE: Final[int] = 9


def _gguf_exact(handle: Any, size: int, file_size: int, label: str) -> bytes:
    if size < 0 or handle.tell() + size > file_size:
        raise ConversionRefused(f"GGUF ends inside {label}")
    if handle.tell() + size > _GGUF_MAX_HEADER_BYTES:
        raise ConversionRefused("GGUF metadata/tensor header exceeds the reviewed byte ceiling")
    raw = handle.read(size)
    if len(raw) != size:
        raise ConversionRefused(f"GGUF ends inside {label}")
    return raw


def _gguf_skip(handle: Any, size: int, file_size: int, label: str) -> None:
    if size < 0 or handle.tell() + size > file_size:
        raise ConversionRefused(f"GGUF {label} extends beyond the file")
    if handle.tell() + size > _GGUF_MAX_HEADER_BYTES:
        raise ConversionRefused("GGUF metadata/tensor header exceeds the reviewed byte ceiling")
    handle.seek(size, os.SEEK_CUR)


def _gguf_u32(handle: Any, file_size: int, label: str) -> int:
    return int(struct.unpack("<I", _gguf_exact(handle, 4, file_size, label))[0])


def _gguf_u64(handle: Any, file_size: int, label: str) -> int:
    return int(struct.unpack("<Q", _gguf_exact(handle, 8, file_size, label))[0])


def _gguf_string(
    handle: Any,
    file_size: int,
    label: str,
    *,
    capture: bool,
) -> str | None:
    size = _gguf_u64(handle, file_size, f"{label} length")
    if size > _GGUF_MAX_STRING_BYTES:
        raise ConversionRefused(f"GGUF {label} exceeds the reviewed string ceiling")
    if not capture:
        _gguf_skip(handle, size, file_size, label)
        return None
    try:
        return _gguf_exact(handle, size, file_size, label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConversionRefused(f"GGUF {label} is not UTF-8") from exc


def _gguf_scalar(handle: Any, kind: int, file_size: int, label: str) -> Any:
    format_string = _GGUF_SCALAR_FORMAT.get(kind)
    if format_string is None:
        raise ConversionRefused(f"GGUF {label} has unknown scalar type {kind}")
    width = struct.calcsize(format_string)
    return struct.unpack(format_string, _gguf_exact(handle, width, file_size, label))[0]


def _gguf_value(
    handle: Any,
    kind: int,
    file_size: int,
    label: str,
    *,
    capture: bool,
) -> tuple[Any, int | None]:
    if kind in _GGUF_SCALAR_FORMAT:
        if capture:
            return _gguf_scalar(handle, kind, file_size, label), None
        _gguf_skip(handle, struct.calcsize(_GGUF_SCALAR_FORMAT[kind]), file_size, label)
        return None, None
    if kind == _GGUF_STRING_TYPE:
        return _gguf_string(handle, file_size, label, capture=capture), None
    if kind != _GGUF_ARRAY_TYPE:
        raise ConversionRefused(f"GGUF {label} has unknown value type {kind}")
    element_kind = _gguf_u32(handle, file_size, f"{label} array type")
    count = _gguf_u64(handle, file_size, f"{label} array count")
    if count > _GGUF_MAX_ARRAY_ITEMS or element_kind == _GGUF_ARRAY_TYPE:
        raise ConversionRefused(f"GGUF {label} has an unsupported array")
    if element_kind in _GGUF_SCALAR_FORMAT and not capture:
        _gguf_skip(
            handle,
            count * struct.calcsize(_GGUF_SCALAR_FORMAT[element_kind]),
            file_size,
            label,
        )
        return None, element_kind
    values: list[Any] | None = [] if capture else None
    for index in range(count):
        if element_kind == _GGUF_STRING_TYPE:
            item = _gguf_string(
                handle,
                file_size,
                f"{label}[{index}]",
                capture=capture,
            )
        elif element_kind in _GGUF_SCALAR_FORMAT:
            item = _gguf_scalar(handle, element_kind, file_size, f"{label}[{index}]")
        else:
            raise ConversionRefused(f"GGUF {label} has unknown array type {element_kind}")
        if values is not None:
            values.append(item)
    return values, element_kind


def _parse_gguf_contract(
    path: Path,
    *,
    wanted_metadata: frozenset[str],
    include_tensor_details: bool,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConversionRefused("GGUF contract input must be a regular non-symlink file")
    file_size = path.stat().st_size
    metadata: dict[str, dict[str, Any]] = {}
    metadata_keys: set[str] = set()
    tensor_details: list[dict[str, Any]] = []
    tensor_offsets: list[int] = []
    with path.open("rb") as handle:
        if _gguf_exact(handle, 4, file_size, "magic") != b"GGUF":
            raise ConversionRefused("calibration output is not a GGUF file")
        version = _gguf_u32(handle, file_size, "version")
        if version not in {2, 3}:
            raise ConversionRefused(f"GGUF version {version} is unsupported")
        tensor_count = _gguf_u64(handle, file_size, "tensor count")
        metadata_count = _gguf_u64(handle, file_size, "metadata count")
        if tensor_count > _GGUF_MAX_TENSORS:
            raise ConversionRefused("GGUF declares too many tensors")
        if metadata_count > _GGUF_MAX_METADATA:
            raise ConversionRefused("GGUF declares too many metadata fields")
        for index in range(metadata_count):
            key = _gguf_string(
                handle,
                file_size,
                f"metadata key {index}",
                capture=True,
            )
            if key is None or key in metadata_keys:
                raise ConversionRefused(f"GGUF repeats metadata key {key!r}")
            metadata_keys.add(key)
            kind = _gguf_u32(handle, file_size, f"metadata type for {key!r}")
            value, element_kind = _gguf_value(
                handle,
                kind,
                file_size,
                key,
                capture=key in wanted_metadata or key == "general.alignment",
            )
            if key in wanted_metadata or key == "general.alignment":
                metadata[key] = {
                    "type": kind,
                    "array_type": element_kind,
                    "value": value,
                }

        seen_tensors: set[str] = set()
        for index in range(tensor_count):
            name = _gguf_string(
                handle,
                file_size,
                f"tensor {index} name",
                capture=True,
            )
            if name is None or not name or name in seen_tensors:
                raise ConversionRefused(f"GGUF tensor name is empty or duplicated: {name!r}")
            seen_tensors.add(name)
            dimensions_count = _gguf_u32(handle, file_size, f"tensor {name!r} dimensions")
            if not 1 <= dimensions_count <= 4:
                raise ConversionRefused(f"GGUF tensor {name!r} has invalid dimensions")
            dimensions = tuple(
                _gguf_u64(handle, file_size, f"tensor {name!r} dimension {axis}")
                for axis in range(dimensions_count)
            )
            if any(value < 1 for value in dimensions):
                raise ConversionRefused(f"GGUF tensor {name!r} has an empty dimension")
            tensor_type = _gguf_u32(handle, file_size, f"tensor {name!r} type")
            offset = _gguf_u64(handle, file_size, f"tensor {name!r} offset")
            tensor_offsets.append(offset)
            if include_tensor_details:
                tensor_details.append(
                    {
                        "name": name,
                        "dimensions": dimensions,
                        "type": tensor_type,
                        "offset": offset,
                    }
                )
        header_end = handle.tell()

    alignment_entry = metadata.get("general.alignment")
    alignment = 32
    if alignment_entry is not None:
        if alignment_entry != {"type": 4, "array_type": None, "value": alignment_entry["value"]}:
            raise ConversionRefused("GGUF general.alignment is not a UINT32")
        alignment = int(alignment_entry["value"])
        if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise ConversionRefused("GGUF general.alignment is invalid")
    data_start = (header_end + alignment - 1) // alignment * alignment
    if data_start > file_size:
        raise ConversionRefused("GGUF tensor data begins beyond end of file")
    if any(offset % alignment or data_start + offset >= file_size for offset in tensor_offsets):
        raise ConversionRefused("GGUF tensor offset is unaligned or beyond end of file")
    return {
        "version": version,
        "file_bytes": file_size,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata_keys": frozenset(metadata_keys),
        "metadata": metadata,
        "tensors": tensor_details,
        "alignment": alignment,
        "data_start": data_start,
    }


def _required_metadata(
    parsed: Mapping[str, Any],
    key: str,
    *,
    kind: int,
    value: Any,
    array_kind: int | None = None,
) -> None:
    metadata = _mapping(parsed.get("metadata"), "GGUF selected metadata")
    found = metadata.get(key)
    expected = {"type": kind, "array_type": array_kind, "value": value}
    if found != expected:
        raise ConversionRefused(f"GGUF metadata {key!r} changed")


def _validate_imatrix_gguf(path: Path) -> dict[str, Any]:
    wanted = frozenset(
        {
            "general.type",
            "imatrix.datasets",
            "imatrix.chunk_count",
            "imatrix.chunk_size",
        }
    )
    parsed = _parse_gguf_contract(
        path,
        wanted_metadata=wanted,
        include_tensor_details=True,
    )
    if parsed["version"] != 3:
        raise ConversionRefused("imatrix GGUF version changed")
    if parsed["metadata_keys"] != wanted:
        raise ConversionRefused("imatrix GGUF metadata fields changed")
    _required_metadata(parsed, "general.type", kind=8, value="imatrix")
    _required_metadata(
        parsed,
        "imatrix.datasets",
        kind=9,
        array_kind=8,
        value=[CALIBRATION_CORPUS_NAME],
    )
    _required_metadata(parsed, "imatrix.chunk_count", kind=4, value=CALIBRATION_CHUNKS)
    _required_metadata(
        parsed,
        "imatrix.chunk_size",
        kind=4,
        value=CALIBRATION_CONTEXT_TOKENS,
    )
    tensors = list(parsed["tensors"])
    if not tensors or len(tensors) % 2:
        raise ConversionRefused("imatrix GGUF has no complete tensor pairs")
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for tensor in tensors:
        name = str(tensor["name"])
        suffix = next(
            (
                candidate_suffix
                for candidate_suffix in (".in_sum2", ".counts")
                if name.endswith(candidate_suffix)
            ),
            None,
        )
        if suffix is None:
            raise ConversionRefused("imatrix GGUF contains an unexpected tensor name")
        base_name = name[: -len(suffix)]
        pairs.setdefault(base_name, {})[suffix] = tensor
    if any(frozenset(pair) != frozenset({".in_sum2", ".counts"}) for pair in pairs.values()):
        raise ConversionRefused("imatrix GGUF has an unmatched tensor pair")
    validated_counts: list[float] = []
    tensor_ranges: list[tuple[int, int]] = []
    for pair in pairs.values():
        sums = pair[".in_sum2"]
        counts = pair[".counts"]
        if sums["type"] != 0 or counts["type"] != 0:
            raise ConversionRefused("imatrix GGUF tensor pairs must be F32")
        sums_dimensions = tuple(sums["dimensions"])
        counts_dimensions = tuple(counts["dimensions"])
        # The pinned writer constructs both tensors as 2-D, but GGUF serializes
        # them with ggml_n_dims(), which removes a trailing singleton matrix
        # axis. Dense entries therefore have canonical ranks 1/1, while expert
        # entries with more than one matrix retain canonical ranks 2/2.
        dense_dimensions = len(sums_dimensions) == 1 and counts_dimensions == (1,)
        expert_dimensions = (
            len(sums_dimensions) == 2
            and len(counts_dimensions) == 2
            and counts_dimensions[0] == 1
            and counts_dimensions[1] > 1
            and sums_dimensions[1] == counts_dimensions[1]
        )
        if not dense_dimensions and not expert_dimensions:
            raise ConversionRefused("imatrix GGUF tensor pair dimensions changed")
        for tensor in (sums, counts):
            elements = 1
            for dimension in tensor["dimensions"]:
                elements *= int(dimension)
            offset = int(tensor["offset"])
            if offset % int(parsed["alignment"]):
                raise ConversionRefused("imatrix GGUF tensor offset is not aligned")
            if int(parsed["data_start"]) + offset + elements * 4 > int(parsed["file_bytes"]):
                raise ConversionRefused("imatrix GGUF tensor data extends beyond end of file")
            start = int(parsed["data_start"]) + offset
            tensor_ranges.append((start, start + elements * 4))
            with path.open("rb") as handle:
                handle.seek(start)
                raw = handle.read(elements * 4)
            values = (value[0] for value in struct.iter_unpack("<f", raw))
            if tensor is sums:
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ConversionRefused("imatrix GGUF contains an invalid sum-of-squares value")
            else:
                for value in values:
                    if not math.isfinite(value) or value <= 0 or not value.is_integer():
                        raise ConversionRefused("imatrix GGUF contains an invalid count value")
                    validated_counts.append(value)
    if int(max(validated_counts)) // CALIBRATION_CONTEXT_TOKENS != CALIBRATION_CHUNKS:
        raise ConversionRefused("imatrix GGUF maximum count is inconsistent with chunk metadata")
    ordered_ranges = sorted(tensor_ranges)
    if any(left[1] > right[0] for left, right in pairwise(ordered_ranges)):
        raise ConversionRefused("imatrix GGUF tensor data overlaps")
    identity = _content_identity(path, "importance matrix GGUF")
    return {
        **identity,
        "version": parsed["version"],
        "tensor_count": parsed["tensor_count"],
        "entries_count": len(pairs),
        "datasets": [CALIBRATION_CORPUS_NAME],
        "chunk_count": CALIBRATION_CHUNKS,
        "chunk_size": CALIBRATION_CONTEXT_TOKENS,
    }


def _validate_calibrated_model_metadata(
    path: Path,
    *,
    architecture: str = QWEN3_ARCHITECTURE,
) -> dict[str, Any]:
    wanted = frozenset(
        {
            "general.architecture",
            "general.file_type",
            "quantize.imatrix.file",
            "quantize.imatrix.dataset",
            "quantize.imatrix.entries_count",
            "quantize.imatrix.chunks_count",
        }
    )
    parsed = _parse_gguf_contract(
        path,
        wanted_metadata=wanted,
        include_tensor_details=False,
    )
    if int(parsed["tensor_count"]) < 1:
        raise ConversionRefused("calibrated model GGUF declares no tensors")
    _required_metadata(parsed, "general.architecture", kind=8, value=architecture)
    _required_metadata(parsed, "general.file_type", kind=4, value=15)
    _required_metadata(parsed, "quantize.imatrix.file", kind=8, value=IMATRIX_NAME)
    _required_metadata(
        parsed,
        "quantize.imatrix.dataset",
        kind=8,
        value=CALIBRATION_CORPUS_NAME,
    )
    _required_metadata(
        parsed,
        "quantize.imatrix.chunks_count",
        kind=4,
        value=CALIBRATION_CHUNKS,
    )
    metadata = _mapping(parsed["metadata"], "calibrated model metadata")
    entries = _mapping(
        metadata.get("quantize.imatrix.entries_count"),
        "calibrated model entries metadata",
    )
    if entries.get("type") != 4 or entries.get("array_type") is not None:
        raise ConversionRefused("calibrated model imatrix entry count is not UINT32")
    entry_count = entries.get("value")
    if isinstance(entry_count, bool) or not isinstance(entry_count, int) or entry_count < 1:
        raise ConversionRefused("calibrated model imatrix entry count is not positive")
    return {
        "imatrix_file": IMATRIX_NAME,
        "imatrix_dataset": CALIBRATION_CORPUS_NAME,
        "imatrix_entries_count": entry_count,
        "imatrix_chunks_count": CALIBRATION_CHUNKS,
    }


def _official_tree_digest(root: Path) -> str:
    artifact_root = _regular_directory(root, "staged artifact")
    entries: list[tuple[str, str]] = []
    normalized_names: set[str] = set()
    for path in sorted(
        artifact_root.rglob("*"), key=lambda item: item.relative_to(artifact_root).as_posix()
    ):
        relative = path.relative_to(artifact_root).as_posix()
        if path.is_symlink():
            raise ConversionRefused(f"artifact contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ConversionRefused(f"artifact contains a special file: {relative}")
        normalized = unicodedata.normalize("NFC", relative)
        if normalized != relative or normalized in normalized_names:
            raise ConversionRefused("artifact paths are not uniquely NFC-normalized")
        normalized_names.add(normalized)
        identity = gguf.file_identity(path, f"artifact file {relative}")
        entries.append((relative, str(identity["digest"])))
    if not entries:
        raise ConversionRefused("artifact is empty")
    digest = hashlib.sha256()
    for relative, file_digest in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _validate_request(request: ConversionRequest) -> ConversionRequest:
    if request.quantization not in SUPPORTED_QUANTIZATIONS:
        raise ConversionRefused("quantization must be exactly Q8_0, Q5_K_M, or Q4_K_M")
    calibration_values = (
        request.calibration_profile,
        request.calibration_current_dataset,
        request.calibration_current_source_corpus,
        request.imatrix_tool,
    )
    provided = tuple(value is not None for value in calibration_values)
    if any(provided) and not all(provided):
        raise ConversionRefused("calibrated Q4 arguments must be supplied all-or-nothing")
    calibrated = all(provided)
    auxiliary = (
        request.calibration_aux_dataset,
        request.calibration_aux_source_corpus,
    )
    auxiliary_provided = tuple(value is not None for value in auxiliary)
    if any(auxiliary_provided) and not all(auxiliary_provided):
        raise ConversionRefused("auxiliary calibration arguments must be supplied as a pair")
    if any(auxiliary_provided) and not calibrated:
        raise ConversionRefused("auxiliary calibration inputs require calibrated Q4")
    if calibrated and request.calibration_profile != CALIBRATION_PROFILE:
        raise ConversionRefused(f"calibration profile must be exactly {CALIBRATION_PROFILE}")
    if calibrated and request.quantization != "Q4_K_M":
        raise ConversionRefused("importance-matrix calibration is supported only for Q4_K_M")
    tokens = request.max_input_tokens
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise ConversionRefused("max-input-tokens must be an integer")
    if not gguf.MIN_CONTEXT_TOKENS <= tokens <= gguf.MAX_CONTEXT_TOKENS:
        raise ConversionRefused(
            f"max-input-tokens must be in [{gguf.MIN_CONTEXT_TOKENS}, {gguf.MAX_CONTEXT_TOKENS}]"
        )
    output = _fresh_output_bundle(request.output_bundle)
    protected_inputs: list[tuple[Path, str]] = [
        (request.training_run, "training run"),
        (request.training_dataset, "training dataset"),
        (request.source_corpus, "training source corpus"),
        (request.base, "base snapshot"),
        (request.llama_cpp, "llama.cpp checkout"),
    ]
    if request.converter_python is not None:
        protected_inputs.append((request.converter_python, "GGUF converter Python"))
    if calibrated:
        current_dataset = request.calibration_current_dataset
        current_source = request.calibration_current_source_corpus
        if current_dataset is None or current_source is None:
            raise ConversionRefused("calibrated Q4 inputs disappeared during validation")
        protected_inputs.extend(
            (
                (current_dataset, "current calibration dataset"),
                (current_source, "current calibration source corpus"),
            )
        )
        aux_dataset = request.calibration_aux_dataset
        aux_source = request.calibration_aux_source_corpus
        if aux_dataset is not None and aux_source is not None:
            protected_inputs.extend(
                (
                    (aux_dataset, "auxiliary calibration dataset"),
                    (aux_source, "auxiliary calibration source corpus"),
                )
            )
        if shutil.disk_usage(output.parent).free < CALIBRATION_MIN_FREE_BYTES:
            raise ConversionRefused("calibrated Q4 requires at least 6 GiB free below /dev/shm")
    for source, label in protected_inputs:
        try:
            protected = Path(source).resolve(strict=True)
        except OSError as exc:
            raise ConversionRefused(f"{label} is unavailable: {exc}") from exc
        if protected == output or protected in output.parents:
            raise ConversionRefused(f"output bundle must not be inside the {label}")
    return replace(request, output_bundle=output)


def _load_manifest(
    quantization: str,
    max_input_tokens: int,
    *,
    base_model: str,
) -> dict[str, Any]:
    return {
        "format": "gguf",
        "quantization": quantization,
        "entrypoint": ENTRYPOINT,
        "max_input": {"tokens": max_input_tokens},
        "preprocessing": {"tokenizer": "tokenizer.json"},
        "base_model": base_model,
    }


def _validate_load_manifest_for_lineage(
    payload: Mapping[str, Any],
    artifact: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> None:
    if lineage.get("schema") != CURRENT_TRAINING_SCHEMA:
        provenance._validate_load_manifest(payload, artifact)
        return
    manifest = _mapping(payload, "current load manifest")
    _exact_keys(
        manifest,
        frozenset(
            {"format", "quantization", "entrypoint", "max_input", "preprocessing", "base_model"}
        ),
        "current load manifest",
    )
    maximum = _mapping(manifest.get("max_input"), "current load manifest max_input")
    _exact_keys(maximum, frozenset({"tokens"}), "current load manifest max_input")
    tokens = maximum.get("tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise ConversionRefused("current load manifest max_input tokens are malformed")
    quantization = manifest.get("quantization")
    base_model, architecture = _lineage_model_contract(lineage)
    if dict(manifest) != _load_manifest(
        str(quantization),
        tokens,
        base_model=base_model,
    ):
        raise ConversionRefused("current load manifest changed")
    entrypoint = _mapping(artifact.get("entrypoint"), "current artifact entrypoint")
    header = _mapping(entrypoint.get("gguf"), "current artifact GGUF identity")
    if (
        entrypoint.get("path") != ENTRYPOINT
        or header.get("architecture") != architecture
        or header.get("file_type") != gguf.SUPPORTED_QUANTIZATIONS.get(str(quantization))
    ):
        raise ConversionRefused("current load manifest and qwen2 artifact identity diverged")


def _receipt(
    *,
    lineage: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    converter = _mapping(toolchain["converter"], "converter identity")
    base_model, _architecture = _lineage_model_contract(lineage)
    quantizer = _mapping(toolchain["quantizer"], "quantizer identity")
    entrypoint = _mapping(artifact["entrypoint"], "artifact entrypoint")
    return {
        "schema": _conversion_schema(lineage, calibrated=False),
        "status": "complete",
        "track": provenance.TRACK,
        "hardware_class": provenance.HARDWARE_CLASS,
        "base_model": base_model,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "source": _conversion_source(lineage),
        "conversion": {
            "converter_digest": converter["digest"],
            "quantizer_digest": quantizer["digest"],
            "commands": [dict(command) for command in commands],
        },
        "artifact": {
            "tree_digest": artifact["tree_digest"],
            "entrypoint_digest": entrypoint["digest"],
            "entrypoint_bytes": entrypoint["bytes"],
            "quantization": load_manifest["quantization"],
        },
        "load_manifest": dict(load_manifest),
        "calibration_receipt_digest": None,
    }


def _calibration_receipt(
    *,
    lineage: Mapping[str, Any],
    material: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
    f16: Mapping[str, Any],
    imatrix: Mapping[str, Any],
    artifact: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
    determinism_replay: Mapping[str, Any],
) -> dict[str, Any]:
    converter_identity = _mapping(toolchain["converter"], "converter identity")
    base_model, _architecture = _lineage_model_contract(lineage)
    current = lineage.get("schema") == CURRENT_TRAINING_SCHEMA
    converter_python_identity = (
        _converter_python_receipt_identity(
            toolchain["converter_python"], "converter Python identity"
        )
        if current
        else None
    )
    imatrix_tool_identity = _mapping(toolchain["imatrix"], "imatrix tool identity")
    quantizer_identity = _mapping(toolchain["quantizer"], "quantizer identity")
    runtime_libraries = _mapping(toolchain["runtime_libraries"], "runtime library closure identity")
    entrypoint = _mapping(artifact["entrypoint"], "artifact entrypoint")
    return {
        "schema": _calibration_schema(lineage),
        "status": "complete",
        "profile": CALIBRATION_PROFILE,
        "track": provenance.TRACK,
        "hardware_class": provenance.HARDWARE_CLASS,
        "base_model": base_model,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "source": dict(_mapping(material["source"], "calibration source identity")),
        "selection": dict(_mapping(material["selection"], "calibration selection")),
        "rendering": {
            "schema": CALIBRATION_RENDER_SCHEMA,
            "encoding": "UTF-8",
            "expression": "prompt + completion + <|im_end|> + LF",
            "eos_token": CALIBRATION_EOS_TOKEN,
            "eos_token_id": CALIBRATION_EOS_TOKEN_ID,
            "rows": CALIBRATION_TOTAL_ROWS,
            "corpus": dict(corpus),
        },
        "toolchain": {
            "converter_digest": converter_identity["digest"],
            **(
                {"converter_python": converter_python_identity}
                if converter_python_identity is not None
                else {}
            ),
            "imatrix_digest": imatrix_tool_identity["digest"],
            "quantizer_digest": quantizer_identity["digest"],
            "runtime_libraries": dict(runtime_libraries),
        },
        "commands": [dict(command) for command in commands],
        "determinism_replay": dict(determinism_replay),
        "intermediate": {
            "f16": {**dict(f16), "file_type": 1},
            "imatrix": dict(imatrix),
        },
        "artifact": {
            "tree_digest": artifact["tree_digest"],
            "entrypoint_digest": entrypoint["digest"],
            "entrypoint_bytes": entrypoint["bytes"],
            "quantization": "Q4_K_M",
            "calibration_metadata": dict(model_metadata),
        },
        "load_manifest": dict(load_manifest),
    }


def _calibrated_conversion_receipt(
    *,
    lineage: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
    calibration_digest: str,
    determinism_replay: Mapping[str, Any],
) -> dict[str, Any]:
    converter_identity = _mapping(toolchain["converter"], "converter identity")
    base_model, _architecture = _lineage_model_contract(lineage)
    current = lineage.get("schema") == CURRENT_TRAINING_SCHEMA
    converter_python_identity = (
        _converter_python_receipt_identity(
            toolchain["converter_python"], "converter Python identity"
        )
        if current
        else None
    )
    imatrix_identity = _mapping(toolchain["imatrix"], "imatrix tool identity")
    quantizer_identity = _mapping(toolchain["quantizer"], "quantizer identity")
    runtime_libraries = _mapping(toolchain["runtime_libraries"], "runtime library closure identity")
    entrypoint = _mapping(artifact["entrypoint"], "artifact entrypoint")
    return {
        "schema": _conversion_schema(lineage, calibrated=True),
        "status": "complete",
        "track": provenance.TRACK,
        "hardware_class": provenance.HARDWARE_CLASS,
        "base_model": base_model,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "source": _conversion_source(lineage),
        "conversion": {
            "converter_digest": converter_identity["digest"],
            **(
                {"converter_python": converter_python_identity}
                if converter_python_identity is not None
                else {}
            ),
            "imatrix_digest": imatrix_identity["digest"],
            "quantizer_digest": quantizer_identity["digest"],
            "runtime_libraries": dict(runtime_libraries),
            "commands": [dict(command) for command in commands],
            "determinism_replay": dict(determinism_replay),
        },
        "artifact": {
            "tree_digest": artifact["tree_digest"],
            "entrypoint_digest": entrypoint["digest"],
            "entrypoint_bytes": entrypoint["bytes"],
            "quantization": "Q4_K_M",
        },
        "load_manifest": dict(load_manifest),
        "calibration_receipt_digest": calibration_digest,
    }


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


_LOCAL_CONVERTER_PYTHON_IDENTITY_FIELDS = frozenset(
    {"path", "bytes", "digest", "device", "inode", "mode", "mtime_ns", "ctime_ns"}
)
_PORTABLE_CONVERTER_PYTHON_IDENTITY_FIELDS = frozenset({"path", "bytes", "digest", "mode"})
_WORKER_OBSERVATION_FIELDS = frozenset({"device", "inode", "mtime_ns", "ctime_ns"})


def _converter_python_receipt_identity(value: Any, label: str) -> dict[str, Any]:
    identity = dict(_mapping(value, label))
    if frozenset(identity) == _LOCAL_CONVERTER_PYTHON_IDENTITY_FIELDS:
        portable = {field: identity[field] for field in _PORTABLE_CONVERTER_PYTHON_IDENTITY_FIELDS}
        worker = {field: identity[field] for field in _WORKER_OBSERVATION_FIELDS}
    else:
        _exact_keys(identity, frozenset({"portable", "worker_observation"}), label)
        portable = dict(_mapping(identity.get("portable"), f"{label} portable identity"))
        worker = dict(_mapping(identity.get("worker_observation"), f"{label} worker observation"))
        _exact_keys(
            portable,
            _PORTABLE_CONVERTER_PYTHON_IDENTITY_FIELDS,
            f"{label} portable identity",
        )
        _exact_keys(worker, _WORKER_OBSERVATION_FIELDS, f"{label} worker observation")
    path = portable.get("path")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ConversionRefused(f"{label} path must be absolute and non-empty")
    for container, field, minimum in (
        (portable, "bytes", 1),
        (worker, "device", 0),
        (worker, "inode", 1),
        (worker, "mtime_ns", 0),
        (worker, "ctime_ns", 0),
    ):
        item = container.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ConversionRefused(f"{label} {field} is malformed")
    if not _valid_digest(portable.get("digest")):
        raise ConversionRefused(f"{label} digest is malformed")
    mode = portable.get("mode")
    if not isinstance(mode, str) or not mode.startswith("0o"):
        raise ConversionRefused(f"{label} mode is malformed")
    try:
        parsed_mode = int(mode, 8)
    except ValueError as exc:
        raise ConversionRefused(f"{label} mode is malformed") from exc
    if oct(parsed_mode) != mode or parsed_mode & 0o111 == 0 or parsed_mode & 0o022:
        raise ConversionRefused(f"{label} mode is not a safe executable mode")
    return {"portable": portable, "worker_observation": worker}


def _fd_executable_identity(descriptor: int, path: Path, label: str) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ConversionRefused(f"{label} is not a regular file")
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ConversionRefused(f"{label} changed while its held descriptor was hashed")
    return _converter_python_receipt_identity(
        {
            "path": str(path),
            "bytes": after.st_size,
            "digest": "sha256:" + digest.hexdigest(),
            "device": after.st_dev,
            "inode": after.st_ino,
            "mode": oct(stat.S_IMODE(after.st_mode)),
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        },
        label,
    )


def _validate_conversion_source_shape(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    schema = receipt.get("schema")
    source = _mapping(receipt.get("source"), "conversion source")
    if schema in {SCHEMA, CALIBRATED_CONVERSION_SCHEMA}:
        _exact_keys(source, LEGACY_CONVERSION_SOURCE_KEYS, "legacy conversion source")
    elif schema in {
        NORMALIZED_CONVERSION_SCHEMA,
        NORMALIZED_CALIBRATED_CONVERSION_SCHEMA,
    }:
        _exact_keys(source, NORMALIZED_CONVERSION_SOURCE_KEYS, "normalized conversion source")
        required = {
            "training_schema": gguf.TRAINING_SCHEMA_V6,
            "dataset_schema": normalized_candidate.DATASET_SCHEMA,
            "corpus_profile": normalized_candidate.CORPUS_PROFILE,
        }
        for field, expected in required.items():
            if source.get(field) != expected:
                raise ConversionRefused(f"normalized conversion source field {field!r} changed")
        excluded = _mapping(source.get("excluded_refs"), "normalized conversion excluded refs")
        _exact_keys(
            excluded,
            NORMALIZED_EXCLUDED_REFS_KEYS,
            "normalized conversion excluded refs",
        )
        if dict(excluded) != {
            "bytes": normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
            "digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
        }:
            raise ConversionRefused("normalized conversion excluded refs identity changed")
    elif schema == CURRENT_CALIBRATED_CONVERSION_SCHEMA:
        _exact_keys(source, CURRENT_CONVERSION_SOURCE_KEYS, "current conversion source")
        required = {
            "training_schema": CURRENT_TRAINING_SCHEMA,
            "dataset_schema": candidate.DATASET_SCHEMA,
            "corpus_profile": CURRENT_CORPUS_PROFILE,
        }
        for field, expected in required.items():
            if source.get(field) != expected:
                raise ConversionRefused(f"current conversion source field {field!r} changed")
        current_source = _mapping(
            source.get("source_corpus"),
            "current conversion source-corpus identity",
        )
        _exact_keys(
            current_source,
            CURRENT_SOURCE_CORPUS_KEYS,
            "current conversion source-corpus identity",
        )
        if (
            current_source.get("bytes") != gguf.CURRENT94_PUBLIC_CORPUS_BYTES
            or current_source.get("digest") != gguf.CURRENT94_PUBLIC_CORPUS_RAW_DIGEST
            or current_source.get("canonical_digest") != candidate.PUBLIC_CORPUS_CANONICAL_DIGEST
            or current_source.get("task_count") != candidate.EXPECTED_COUNTS["train"]
            or not _valid_digest(current_source.get("refs_digest"))
        ):
            raise ConversionRefused("current conversion source-corpus identity changed")
        current_prepared = _mapping(
            source.get("prepared_dataset"),
            "current conversion prepared-dataset identity",
        )
        _exact_keys(
            current_prepared,
            CURRENT_PREPARED_DATASET_KEYS,
            "current conversion prepared-dataset identity",
        )
        if (
            current_prepared.get("train_examples") != candidate.EXPECTED_COUNTS["train"]
            or current_prepared.get("holdout_examples") != 0
        ):
            raise ConversionRefused("current conversion prepared split changed")
        for field in ("manifest_digest", "train_digest", "holdout_digest"):
            if not _valid_digest(current_prepared.get(field)):
                raise ConversionRefused(f"current conversion prepared field {field!r} is malformed")
        base = _mapping(source.get("base_snapshot"), "current conversion base snapshot")
        if base.get("base_model") != candidate.QWEN25_CODER_1_5B_BASE_MODEL:
            raise ConversionRefused("current conversion base snapshot changed")

    else:
        raise ConversionRefused("conversion receipt schema is unsupported")
    digest_fields = ["training_metadata_digest", "merged_tree_digest"]
    if schema == CURRENT_CALIBRATED_CONVERSION_SCHEMA:
        digest_fields.append("training_metrics_digest")
    for field in digest_fields:
        if not _valid_digest(source.get(field)):
            raise ConversionRefused(f"conversion source field {field!r} is malformed")
    return dict(source)


def _validate_generic_conversion_receipt(
    receipt: Mapping[str, Any],
    *,
    training_lineage: Mapping[str, Any],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
) -> None:
    expected_schema = _conversion_schema(training_lineage, calibrated=False)
    if receipt.get("schema") != expected_schema:
        raise ConversionRefused("conversion receipt schema crosses training lineages")
    source = _validate_conversion_source_shape(receipt)
    expected_source = _conversion_source(training_lineage)
    if source != expected_source:
        raise ConversionRefused("conversion receipt crosses training lineages")
    provenance._validate_generic_conversion(
        receipt,
        training_lineage=training_lineage,
        artifact=artifact,
        load_manifest=load_manifest,
        calibration_digest=None,
    )


def _safe_runtime_mode(value: Any, label: str) -> int:
    if (
        not isinstance(value, str)
        or len(value) != 4
        or any(character not in "01234567" for character in value)
    ):
        raise ConversionRefused(f"{label} mode must be four octal digits")
    mode = int(value, 8)
    if mode & 0o022:
        raise ConversionRefused(f"{label} mode must not be group/world writable")
    return mode


def _validate_runtime_library_receipt(value: Any, label: str) -> dict[str, Any]:
    closure = _mapping(value, label)
    _exact_keys(
        closure,
        frozenset(
            {
                "schema",
                "root",
                "directories",
                "build_bin_namespace",
                "symlinks",
                "executables",
                "libraries",
            }
        ),
        label,
    )
    if closure.get("schema") != RUNTIME_LIBRARY_SCHEMA:
        raise ConversionRefused(f"{label} schema changed")
    if closure.get("root") != str(LLAMA_CPP_ROOT):
        raise ConversionRefused(f"{label} root changed")

    directories = closure.get("directories")
    if not isinstance(directories, list) or len(directories) != 3:
        raise ConversionRefused(f"{label} directory closure changed")
    for entry, expected_path in zip(
        directories,
        (".", "build", "build/bin"),
        strict=True,
    ):
        directory = _mapping(entry, f"{label} directory")
        _exact_keys(directory, frozenset({"path", "mode"}), f"{label} directory")
        if directory.get("path") != expected_path:
            raise ConversionRefused(f"{label} directory path changed")
        _safe_runtime_mode(directory.get("mode"), f"{label} directory {expected_path}")

    namespace_names: set[str] = set()
    for relative, _expected_bytes, _expected_digest in LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT:
        namespace_names.add(_runtime_relative_name(relative, "runtime executable path"))
    for loader, target, _expected_bytes, _expected_digest in LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT:
        namespace_names.add(_runtime_relative_name(loader, "runtime library loader path"))
        namespace_names.add(_runtime_relative_name(target, "runtime library target path"))
    for relative, target in LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT:
        namespace_names.add(_runtime_relative_name(relative, "runtime symlink path"))
        namespace_names.add(_runtime_symlink_target_name(target, "runtime symlink target"))
    expected_namespace = [f"build/bin/{name}" for name in sorted(namespace_names)]
    namespace = closure.get("build_bin_namespace")
    if (
        not isinstance(namespace, list)
        or any(not isinstance(entry, str) for entry in namespace)
        or namespace != expected_namespace
    ):
        raise ConversionRefused(f"{label} build/bin namespace changed")

    symlinks = closure.get("symlinks")
    symlink_contract = LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT
    if not isinstance(symlinks, list) or len(symlinks) != len(symlink_contract):
        raise ConversionRefused(f"{label} symlink closure changed")
    for entry, (expected_path, expected_target) in zip(
        symlinks,
        symlink_contract,
        strict=True,
    ):
        link = _mapping(entry, f"{label} symlink")
        _exact_keys(link, frozenset({"path", "target"}), f"{label} symlink")
        if link.get("path") != expected_path or link.get("target") != expected_target:
            raise ConversionRefused(f"{label} pinned symlink edge changed")

    executables = closure.get("executables")
    executable_contract = LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
    if not isinstance(executables, list) or len(executables) != len(executable_contract):
        raise ConversionRefused(f"{label} executable closure changed")
    for entry, (expected_path, expected_bytes, expected_digest) in zip(
        executables,
        executable_contract,
        strict=True,
    ):
        executable = _mapping(entry, f"{label} executable")
        _exact_keys(
            executable,
            frozenset({"path", "bytes", "digest", "mode"}),
            f"{label} executable",
        )
        actual_bytes = executable.get("bytes")
        if isinstance(actual_bytes, bool) or not isinstance(actual_bytes, int):
            raise ConversionRefused(f"{label} executable bytes must be an integer")
        if {
            "path": executable.get("path"),
            "bytes": actual_bytes,
            "digest": executable.get("digest"),
        } != {
            "path": expected_path,
            "bytes": expected_bytes,
            "digest": expected_digest,
        }:
            raise ConversionRefused(f"{label} pinned executable identity changed")
        _safe_runtime_mode(executable.get("mode"), f"{label} executable {expected_path}")

    libraries = closure.get("libraries")
    contract = LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT
    if not isinstance(libraries, list) or len(libraries) != len(contract):
        raise ConversionRefused(f"{label} library closure changed")
    for entry, (loader, target, expected_bytes, expected_digest) in zip(
        libraries,
        contract,
        strict=True,
    ):
        library = _mapping(entry, f"{label} library")
        _exact_keys(
            library,
            frozenset({"loader_path", "target_path", "bytes", "digest", "mode"}),
            f"{label} library",
        )
        actual_bytes = library.get("bytes")
        if isinstance(actual_bytes, bool) or not isinstance(actual_bytes, int):
            raise ConversionRefused(f"{label} library bytes must be an integer")
        if {
            "loader_path": library.get("loader_path"),
            "target_path": library.get("target_path"),
            "bytes": library.get("bytes"),
            "digest": library.get("digest"),
        } != {
            "loader_path": loader,
            "target_path": target,
            "bytes": expected_bytes,
            "digest": expected_digest,
        }:
            raise ConversionRefused(f"{label} pinned library identity changed")
        _safe_runtime_mode(library.get("mode"), f"{label} library {target}")
    return dict(closure)


def _validate_calibrated_command(
    command: Mapping[str, Any],
    *,
    name: str,
    argv: Sequence[str],
    cwd_role: str,
    executed_object: Mapping[str, Any] | None = None,
) -> None:
    expected_fields = {
        "name",
        "argv",
        "cwd_role",
        "environment",
        "returncode",
        "started_at_unix_ns",
        "finished_at_unix_ns",
        "stdout",
        "stderr",
    }
    if executed_object is not None:
        expected_fields.add("launch")
    _exact_keys(
        command,
        frozenset(expected_fields),
        f"{name} command",
    )
    if executed_object is not None:
        launch = _mapping(command.get("launch"), f"{name} launch")
        _exact_keys(launch, frozenset({"method", "executed_object"}), f"{name} launch")
        if launch.get("method") != "proc-self-fd" or launch.get(
            "executed_object"
        ) != dict(executed_object):
            raise ConversionRefused(f"{name} held-fd launch identity changed")
    if command.get("name") != name or command.get("argv") != [str(item) for item in argv]:
        raise ConversionRefused(f"{name} command argv changed")
    if command.get("cwd_role") != cwd_role:
        raise ConversionRefused(f"{name} command cwd role changed")
    if command.get("environment") != _small_child_environment(single_thread=True):
        raise ConversionRefused(f"{name} command environment changed")
    started = command.get("started_at_unix_ns")
    finished = command.get("finished_at_unix_ns")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or started < 1
        or finished < started
        or command.get("returncode") != 0
    ):
        raise ConversionRefused(f"{name} command timing or return code changed")
    for stream_name in ("stdout", "stderr"):
        stream = _mapping(command.get(stream_name), f"{name} {stream_name} identity")
        _exact_keys(
            stream,
            frozenset({"bytes", "captured_bytes", "captured_digest", "digest", "truncated"}),
            f"{name} {stream_name} identity",
        )
        total = stream.get("bytes")
        captured = stream.get("captured_bytes")
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total < 0
            or isinstance(captured, bool)
            or not isinstance(captured, int)
            or captured != min(total, MAX_CAPTURED_LOG_BYTES)
            or stream.get("truncated") is not (total > captured)
            or not _valid_digest(stream.get("digest"))
            or not _valid_digest(stream.get("captured_digest"))
        ):
            raise ConversionRefused(f"{name} {stream_name} log identity changed")


def _validate_captured_logs(
    log_root: Path,
    commands: Sequence[Mapping[str, Any]],
) -> None:
    expected_files = {
        f"{command['name']}.{stream_name}"
        for command in commands
        for stream_name in ("stdout", "stderr")
    }
    if _bundle_file_set(log_root) != frozenset(expected_files):
        raise ConversionRefused("bounded command log set changed")
    for command in commands:
        for stream_name in ("stdout", "stderr"):
            stream = _mapping(command[stream_name], f"{command['name']} {stream_name} log")
            identity = _content_identity(
                log_root / f"{command['name']}.{stream_name}",
                f"bounded {command['name']} {stream_name} log",
            )
            if identity != {
                "bytes": stream["captured_bytes"],
                "digest": stream["captured_digest"],
            }:
                raise ConversionRefused("bounded command log bytes changed")


def _validate_calibrated_receipts(
    *,
    calibration_receipt: Mapping[str, Any],
    conversion_receipt: Mapping[str, Any],
    calibration_digest: str,
    expected_calibration: Mapping[str, Any],
    expected_conversion: Mapping[str, Any],
    command_argv: Sequence[tuple[str, Sequence[str]]],
    replay_command_argv: Sequence[tuple[str, Sequence[str]]],
) -> None:
    _exact_keys(
        conversion_receipt,
        frozenset(
            {
                "schema",
                "status",
                "track",
                "hardware_class",
                "base_model",
                "llama_cpp_revision",
                "source",
                "conversion",
                "artifact",
                "load_manifest",
                "calibration_receipt_digest",
            }
        ),
        "calibrated conversion receipt",
    )
    if dict(calibration_receipt) != dict(expected_calibration):
        raise ConversionRefused("calibration receipt fields changed")
    if dict(conversion_receipt) != dict(expected_conversion):
        raise ConversionRefused("calibrated conversion receipt fields changed")
    expected_calibration_schema = expected_calibration.get("schema")
    if calibration_receipt.get("schema") != expected_calibration_schema:
        raise ConversionRefused("calibration receipt schema changed")
    expected_conversion_schema = expected_conversion.get("schema")
    if expected_conversion_schema not in {
        CALIBRATED_CONVERSION_SCHEMA,
        NORMALIZED_CALIBRATED_CONVERSION_SCHEMA,
        CURRENT_CALIBRATED_CONVERSION_SCHEMA,
    }:
        raise ConversionRefused("expected calibrated conversion schema is unsupported")
    if conversion_receipt.get("schema") != expected_conversion_schema:
        raise ConversionRefused("calibrated conversion receipt schema changed")
    _validate_conversion_source_shape(conversion_receipt)
    if conversion_receipt.get("calibration_receipt_digest") != calibration_digest:
        raise ConversionRefused("conversion receipt does not bind the calibration receipt")
    calibration_commands = calibration_receipt.get("commands")
    conversion = _mapping(conversion_receipt.get("conversion"), "calibrated conversion")
    calibration_toolchain = _mapping(
        calibration_receipt.get("toolchain"),
        "calibration toolchain",
    )
    calibration_runtime = _validate_runtime_library_receipt(
        calibration_toolchain.get("runtime_libraries"),
        "calibration runtime library closure",
    )
    conversion_runtime = _validate_runtime_library_receipt(
        conversion.get("runtime_libraries"),
        "conversion runtime library closure",
    )
    if calibration_runtime != conversion_runtime:
        raise ConversionRefused("calibrated receipts bind different runtime libraries")
    calibration_interpreter: dict[str, Any] | None = None
    if expected_conversion_schema == CURRENT_CALIBRATED_CONVERSION_SCHEMA:
        calibration_interpreter = _converter_python_receipt_identity(
            calibration_toolchain.get("converter_python"),
            "calibration conversion-time converter Python identity",
        )
        conversion_interpreter = _converter_python_receipt_identity(
            conversion.get("converter_python"),
            "conversion-v6 conversion-time converter Python identity",
        )
        if calibration_interpreter != conversion_interpreter:
            raise ConversionRefused(
                "current calibrated receipts bind different converter Python identities"
            )
    conversion_commands = conversion.get("commands")
    if not isinstance(calibration_commands, list) or calibration_commands != conversion_commands:
        raise ConversionRefused("calibrated receipts do not bind the same commands")
    if len(calibration_commands) != len(command_argv):
        raise ConversionRefused("calibrated receipts do not contain exactly three commands")
    for command, (name, argv) in zip(calibration_commands, command_argv, strict=True):
        _validate_calibrated_command(
            _mapping(command, f"{name} command"),
            name=name,
            argv=argv,
            cwd_role="private_staging",
            executed_object=(calibration_interpreter if name == "convert_f16" else None),
        )
    calibration_replay = _mapping(
        calibration_receipt.get("determinism_replay"),
        "calibration determinism replay",
    )
    conversion_replay = _mapping(
        conversion.get("determinism_replay"),
        "conversion determinism replay",
    )
    if dict(calibration_replay) != dict(conversion_replay):
        raise ConversionRefused("calibrated receipts bind different determinism replays")
    _exact_keys(
        calibration_replay,
        frozenset(
            {
                "schema",
                "commands",
                "f16_digest",
                "imatrix_digest",
                "entrypoint_digest",
                "entrypoint_bytes",
                "artifact_tree_digest",
                "matches_primary",
            }
        ),
        "determinism replay proof",
    )
    if calibration_replay.get("schema") != "microtensor.code.gguf-determinism-replay.v1":
        raise ConversionRefused("determinism replay schema changed")
    for digest_field in (
        "f16_digest",
        "imatrix_digest",
        "entrypoint_digest",
        "artifact_tree_digest",
    ):
        if not _valid_digest(calibration_replay.get(digest_field)):
            raise ConversionRefused(f"determinism replay {digest_field} is malformed")
    replay_bytes = calibration_replay.get("entrypoint_bytes")
    if isinstance(replay_bytes, bool) or not isinstance(replay_bytes, int) or replay_bytes < 1:
        raise ConversionRefused("determinism replay entrypoint byte count is invalid")
    replay_commands = calibration_replay.get("commands")
    if (
        calibration_replay.get("matches_primary") is not True
        or not isinstance(replay_commands, list)
        or len(replay_commands) != len(replay_command_argv)
    ):
        raise ConversionRefused("determinism replay proof is incomplete")
    for command, (name, argv) in zip(replay_commands, replay_command_argv, strict=True):
        _validate_calibrated_command(
            _mapping(command, f"replay {name} command"),
            name=name,
            argv=argv,
            cwd_role="determinism_replay",
            executed_object=(calibration_interpreter if name == "convert_f16" else None),
        )


def _fsync_path(path: Path, *, directory: bool = False) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without the overwrite semantics of rename(2)."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ConversionRefused("Linux renameat2 is required for no-overwrite publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise ConversionRefused("output bundle appeared during conversion; overwrite refused")
        raise ConversionRefused(f"atomic bundle publication failed: {os.strerror(number)}")


def _bundle_file_set(root: Path) -> frozenset[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ConversionRefused(f"output bundle contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ConversionRefused(f"output bundle contains a special file: {relative}")
        files.add(relative)
    return frozenset(files)


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    if dict(left) != dict(right):
        raise ConversionRefused(f"{label} changed during conversion")


def _same_artifact_identity(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    left_content = {key: value for key, value in left.items() if key != "root"}
    right_content = {key: value for key, value in right.items() if key != "root"}
    _same_identity(left_content, right_content, label)


def _validate_calibration_lineage_inputs(
    request: ConversionRequest,
    lineage: Mapping[str, Any],
) -> None:
    auxiliary = (
        request.calibration_aux_dataset,
        request.calibration_aux_source_corpus,
    )
    has_auxiliary = all(value is not None for value in auxiliary)
    if any(value is not None for value in auxiliary) and not has_auxiliary:
        raise ConversionRefused("auxiliary calibration arguments must be supplied as a pair")
    if lineage.get("schema") == CURRENT_TRAINING_SCHEMA:
        if not has_auxiliary:
            raise ConversionRefused(
                "current v4/Qwen2.5 calibration requires a separate normalized-historical "
                "auxiliary dataset and source corpus"
            )
        return
    if has_auxiliary:
        raise ConversionRefused(
            "legacy v5/v6 calibration refuses auxiliary inputs that would change its schema meaning"
        )


def _validate_calibration_material_binding(
    material: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> None:
    if lineage.get("schema") != CURRENT_TRAINING_SCHEMA:
        return
    source = _mapping(material.get("source"), "current calibration source binding")
    _exact_keys(
        source,
        frozenset({"current", "auxiliary_normalized_historical"}),
        "current calibration source binding",
    )
    current = _mapping(source.get("current"), "current calibration corpus binding")
    current_corpus = _mapping(current.get("corpus"), "current calibration corpus identity")
    expected_current = _mapping(
        _current_conversion_source(lineage).get("source_corpus"),
        "current training source binding",
    )
    if dict(current_corpus) != dict(expected_current):
        raise ConversionRefused(
            "current calibration corpus differs from the current v4 training source"
        )
    auxiliary = _mapping(
        source.get("auxiliary_normalized_historical"),
        "auxiliary normalized-historical calibration binding",
    )
    auxiliary_corpus = _mapping(
        auxiliary.get("corpus"),
        "auxiliary normalized-historical corpus identity",
    )
    if dict(auxiliary_corpus) != normalized_candidate.source_corpus_identity():
        raise ConversionRefused("auxiliary calibration corpus identity changed")
    auxiliary_dataset = _mapping(
        auxiliary.get("prepared_dataset"),
        "auxiliary normalized-historical prepared dataset",
    )
    auxiliary_manifest = _mapping(
        auxiliary_dataset.get("manifest"),
        "auxiliary normalized-historical manifest",
    )
    required_manifest = {
        "schema": normalized_candidate.DATASET_SCHEMA,
        "corpus_profile": normalized_candidate.CORPUS_PROFILE,
        "seed": normalized_candidate.EXPECTED_SEED,
        "train_examples": normalized_candidate.EXPECTED_TRAIN_EXAMPLES,
        "holdout_examples": normalized_candidate.EXPECTED_HOLDOUT_EXAMPLES,
        "excluded_examples": normalized_candidate.EXPECTED_EXCLUDED_EXAMPLES,
        "excluded_refs_digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
    }
    for field, expected in required_manifest.items():
        if auxiliary_manifest.get(field) != expected:
            raise ConversionRefused(f"auxiliary calibration manifest field {field!r} changed")
    selection = _mapping(material.get("selection"), "current calibration selection")
    _exact_keys(
        selection,
        frozenset(
            {
                "algorithm",
                "seed",
                "current_rows",
                "current_refs_digest",
                "diagnostic_rows_excluded",
                "diagnostic_refs_digest",
                "auxiliary_pool_rows",
                "auxiliary_selected_rows",
                "auxiliary_selected_refs_digest",
                "total_rows",
            }
        ),
        "current calibration selection",
    )
    if (
        selection.get("algorithm") != CALIBRATION_SELECTION_ALGORITHM
        or selection.get("seed") != CALIBRATION_SEED
        or selection.get("current_rows") != CALIBRATION_CURRENT_ROWS
        or selection.get("diagnostic_rows_excluded") != CALIBRATION_DIAGNOSTIC_ROWS
        or selection.get("auxiliary_pool_rows") != normalized_candidate.EXPECTED_TRAIN_EXAMPLES
        or selection.get("auxiliary_selected_rows") != CALIBRATION_HISTORICAL_ROWS
        or selection.get("total_rows") != CALIBRATION_TOTAL_ROWS
    ):
        raise ConversionRefused("current calibration selection contract changed")


def _validate_f16_gguf(
    path: Path,
    *,
    architecture: str = QWEN3_ARCHITECTURE,
) -> dict[str, Any]:
    try:
        header = gguf.read_gguf_identity(path, expected_architecture=architecture)
    except Exception as exc:
        raise ConversionRefused(f"temporary F16 GGUF was refused: {exc}") from exc
    if (
        header.get("architecture") != architecture
        or header.get("file_type") != 1
        or int(header.get("tensor_count", 0)) < 1
    ):
        raise ConversionRefused(f"converter output is not a non-empty {architecture} F16 GGUF")
    return _content_identity(path, "temporary F16 GGUF")


def _convert_calibrated(request: ConversionRequest) -> dict[str, Any]:
    initial_lineage = _load_lineage(request)
    current = initial_lineage.get("schema") == CURRENT_TRAINING_SCHEMA
    if current and request.converter_python is None:
        raise ConversionRefused("current v4 conversion requires explicit converter Python")
    _validate_calibration_lineage_inputs(request, initial_lineage)
    base_model, architecture = _lineage_model_contract(initial_lineage)
    initial_tools = _toolchain_identity(request, include_converter_python=current)
    _mapping(initial_tools.get("imatrix"), "imatrix tool identity")
    _rows, initial_material = _load_calibration_material(request)
    _validate_calibration_material_binding(initial_material, initial_lineage)
    merged_root = Path(request.training_run).resolve(strict=True) / "merged"
    _regular_directory(merged_root, "validated merged HF tree")

    output = request.output_bundle
    staging = Path(tempfile.mkdtemp(prefix=_STAGING_MARKER, dir=output.parent)).resolve(strict=True)
    staging_stat = staging.stat()
    published = False
    try:
        os.chmod(staging, 0o700)
        staging_stat = staging.stat()
        if stat.S_IMODE(staging_stat.st_mode) != 0o700:
            raise ConversionRefused("private calibrated staging directory mode changed")
        artifact_root = staging / ARTIFACT_NAME
        artifact_root.mkdir(mode=0o700)
        log_root = staging / ".conversion-logs"
        log_root.mkdir(mode=0o700)
        f16_path = staging / F16_NAME
        corpus_path = staging / CALIBRATION_CORPUS_NAME
        imatrix_path = staging / IMATRIX_NAME
        model_path = artifact_root / ENTRYPOINT
        converter_python_path = (
            Path(str(initial_tools["converter_python"]["path"])) if current else None
        )
        converter_path = Path(str(initial_tools["converter"]["path"]))
        imatrix_tool = Path(str(initial_tools["imatrix"]["path"]))
        quantizer_path = Path(str(initial_tools["quantizer"]["path"]))

        rows, replayed_material = _load_calibration_material(request)
        _same_identity(initial_material, replayed_material, "calibration source lineage")
        corpus = _write_calibration_corpus(corpus_path, rows)
        if stat.S_IMODE(corpus_path.stat().st_mode) != 0o600:
            raise ConversionRefused("private calibration corpus mode changed")
        if _content_identity(corpus_path, "private calibration corpus") != {
            "bytes": corpus["bytes"],
            "digest": corpus["digest"],
        }:
            raise ConversionRefused("private calibration corpus bytes changed")

        convert_argv = (
            *((str(converter_python_path),) if current else ()),
            str(converter_path),
            str(merged_root),
            "--outfile",
            F16_NAME,
            "--outtype",
            "f16",
        )
        commands = [
            _bounded_conversion_command(
                "convert_f16",
                convert_argv,
                cwd=staging,
                log_root=log_root,
                executable_path=converter_python_path,
                executable_identity=(
                    _mapping(initial_tools["converter_python"], "converter Python identity")
                    if current
                    else None
                ),
            )
        ]
        _fsync_path(f16_path)
        f16 = _validate_f16_gguf(f16_path, architecture=architecture)

        imatrix_argv = (
            str(imatrix_tool),
            "--offline",
            "--model",
            F16_NAME,
            "--file",
            CALIBRATION_CORPUS_NAME,
            "--output",
            IMATRIX_NAME,
            "--output-format",
            "gguf",
            "--ctx-size",
            str(CALIBRATION_CONTEXT_TOKENS),
            "--chunks",
            str(CALIBRATION_CHUNKS),
            "--batch-size",
            "512",
            "--ubatch-size",
            "512",
            "--threads",
            "1",
            "--threads-batch",
            "1",
            "--device",
            "none",
            "--gpu-layers",
            "0",
            "--fit",
            "off",
            "--flash-attn",
            "off",
            "--no-ppl",
            "--parse-special",
            "--output-frequency",
            str(CALIBRATION_CHUNKS + 1),
            "--save-frequency",
            "0",
        )
        commands.append(
            _bounded_conversion_command(
                "calibrate_imatrix",
                imatrix_argv,
                cwd=staging,
                log_root=log_root,
            )
        )
        _fsync_path(imatrix_path)
        imatrix = _validate_imatrix_gguf(imatrix_path)
        _same_identity(
            f16, _validate_f16_gguf(f16_path, architecture=architecture), "temporary F16 GGUF"
        )
        if _content_identity(corpus_path, "private calibration corpus") != {
            "bytes": corpus["bytes"],
            "digest": corpus["digest"],
        }:
            raise ConversionRefused("private calibration corpus changed during imatrix")

        quantize_argv = (
            str(quantizer_path),
            "--imatrix",
            IMATRIX_NAME,
            F16_NAME,
            f"{ARTIFACT_NAME}/{ENTRYPOINT}",
            "Q4_K_M",
            "1",
        )
        commands.append(
            _bounded_conversion_command(
                "quantize",
                quantize_argv,
                cwd=staging,
                log_root=log_root,
            )
        )
        _fsync_path(model_path)
        model_metadata = _validate_calibrated_model_metadata(model_path, architecture=architecture)
        if model_metadata["imatrix_entries_count"] != imatrix["entries_count"]:
            raise ConversionRefused(
                "calibrated model importance-matrix entry count differs from its input"
            )
        _same_identity(
            f16, _validate_f16_gguf(f16_path, architecture=architecture), "temporary F16 GGUF"
        )
        _same_identity(imatrix, _validate_imatrix_gguf(imatrix_path), "importance matrix GGUF")
        if _content_identity(corpus_path, "private calibration corpus") != {
            "bytes": corpus["bytes"],
            "digest": corpus["digest"],
        }:
            raise ConversionRefused("private calibration corpus changed during quantization")

        tree_digest = _official_tree_digest(artifact_root)
        try:
            artifact = gguf.artifact_identity(
                artifact_root,
                entrypoint=ENTRYPOINT,
                expected_digest=tree_digest,
                quantization="Q4_K_M",
                expected_architecture=architecture,
            )
        except Exception as exc:
            raise ConversionRefused(f"calibrated model artifact was refused: {exc}") from exc

        replay_root = staging / "determinism-replay"
        replay_root.mkdir(mode=0o700)
        replay_artifact_root = replay_root / ARTIFACT_NAME
        replay_artifact_root.mkdir(mode=0o700)
        replay_log_root = replay_root / ".conversion-logs"
        replay_log_root.mkdir(mode=0o700)
        replay_f16_path = replay_root / F16_NAME
        replay_corpus_path = replay_root / CALIBRATION_CORPUS_NAME
        replay_imatrix_path = replay_root / IMATRIX_NAME
        replay_model_path = replay_artifact_root / ENTRYPOINT
        replay_rows, replay_material = _load_calibration_material(request)
        _same_identity(initial_material, replay_material, "determinism replay source lineage")
        replay_corpus = _write_calibration_corpus(replay_corpus_path, replay_rows)
        _same_identity(corpus, replay_corpus, "rendered calibration corpus")
        if stat.S_IMODE(replay_corpus_path.stat().st_mode) != 0o600:
            raise ConversionRefused("private replay calibration corpus mode changed")
        replay_commands = [
            _bounded_conversion_command(
                "convert_f16",
                convert_argv,
                cwd=replay_root,
                log_root=replay_log_root,
                cwd_role="determinism_replay",
                executable_path=converter_python_path if current else None,
                executable_identity=(
                    _mapping(initial_tools["converter_python"], "converter Python identity")
                    if current
                    else None
                ),
            )
        ]
        _fsync_path(replay_f16_path)
        replay_f16 = _validate_f16_gguf(replay_f16_path, architecture=architecture)
        _same_identity(f16, replay_f16, "determinism replay F16 GGUF")
        replay_commands.append(
            _bounded_conversion_command(
                "calibrate_imatrix",
                imatrix_argv,
                cwd=replay_root,
                log_root=replay_log_root,
                cwd_role="determinism_replay",
            )
        )
        _fsync_path(replay_imatrix_path)
        replay_imatrix = _validate_imatrix_gguf(replay_imatrix_path)
        _same_identity(imatrix, replay_imatrix, "determinism replay importance matrix")
        replay_commands.append(
            _bounded_conversion_command(
                "quantize",
                quantize_argv,
                cwd=replay_root,
                log_root=replay_log_root,
                cwd_role="determinism_replay",
            )
        )
        _fsync_path(replay_model_path)
        replay_model_metadata = _validate_calibrated_model_metadata(
            replay_model_path, architecture=architecture
        )
        if replay_model_metadata["imatrix_entries_count"] != replay_imatrix["entries_count"]:
            raise ConversionRefused(
                "determinism replay model importance-matrix entry count differs from its input"
            )
        if replay_model_metadata != model_metadata:
            raise ConversionRefused("determinism replay model metadata differs")
        replay_tree_digest = _official_tree_digest(replay_artifact_root)
        replay_artifact = gguf.artifact_identity(
            replay_artifact_root,
            entrypoint=ENTRYPOINT,
            expected_digest=replay_tree_digest,
            quantization="Q4_K_M",
            expected_architecture=architecture,
        )
        _same_artifact_identity(artifact, replay_artifact, "determinism replay artifact")
        primary_model_identity = _content_identity(model_path, "primary calibrated model")
        replay_model_identity = _content_identity(replay_model_path, "replay calibrated model")
        _same_identity(
            primary_model_identity,
            replay_model_identity,
            "determinism replay model bytes",
        )
        determinism_replay = {
            "schema": "microtensor.code.gguf-determinism-replay.v1",
            "commands": [dict(command) for command in replay_commands],
            "f16_digest": replay_f16["digest"],
            "imatrix_digest": replay_imatrix["digest"],
            "entrypoint_digest": replay_model_identity["digest"],
            "entrypoint_bytes": replay_model_identity["bytes"],
            "artifact_tree_digest": replay_tree_digest,
            "matches_primary": True,
        }
        load_manifest = _load_manifest("Q4_K_M", request.max_input_tokens, base_model=base_model)
        _validate_load_manifest_for_lineage(load_manifest, artifact, initial_lineage)
        calibration_receipt = _calibration_receipt(
            lineage=initial_lineage,
            material=initial_material,
            toolchain=initial_tools,
            commands=commands,
            corpus=corpus,
            f16=f16,
            imatrix=imatrix,
            artifact=artifact,
            model_metadata=model_metadata,
            load_manifest=load_manifest,
            determinism_replay=determinism_replay,
        )
        load_raw = _atomic_json_in_staging(staging / LOAD_SPEC_NAME, load_manifest)
        calibration_raw = _atomic_json_in_staging(
            staging / CALIBRATION_RECEIPT_NAME,
            calibration_receipt,
        )
        calibration_digest = candidate.digest_bytes(calibration_raw)
        conversion_receipt = _calibrated_conversion_receipt(
            lineage=initial_lineage,
            toolchain=initial_tools,
            commands=commands,
            artifact=artifact,
            load_manifest=load_manifest,
            calibration_digest=calibration_digest,
            determinism_replay=determinism_replay,
        )
        conversion_raw = _atomic_json_in_staging(staging / RECEIPT_NAME, conversion_receipt)
        parsed_calibration = _strict_json_file(
            staging / CALIBRATION_RECEIPT_NAME,
            "staged calibration receipt",
        )
        parsed_conversion = _strict_json_file(
            staging / RECEIPT_NAME,
            "staged calibrated conversion receipt",
        )
        command_argv = (
            ("convert_f16", convert_argv),
            ("calibrate_imatrix", imatrix_argv),
            ("quantize", quantize_argv),
        )
        _validate_calibrated_receipts(
            calibration_receipt=parsed_calibration,
            conversion_receipt=parsed_conversion,
            calibration_digest=calibration_digest,
            expected_calibration=calibration_receipt,
            expected_conversion=conversion_receipt,
            command_argv=command_argv,
            replay_command_argv=command_argv,
        )
        _validate_captured_logs(log_root, commands)
        _validate_captured_logs(replay_log_root, replay_commands)

        final_lineage = _load_lineage(request)
        final_tools = _toolchain_identity(request, include_converter_python=current)
        _final_rows, final_material = _load_calibration_material(request)
        _same_identity(initial_lineage, final_lineage, "training lineage")
        _same_identity(initial_tools, final_tools, "llama.cpp toolchain")
        _same_identity(initial_material, final_material, "calibration source lineage")
        _same_identity(
            f16, _validate_f16_gguf(f16_path, architecture=architecture), "temporary F16 GGUF"
        )
        _same_identity(imatrix, _validate_imatrix_gguf(imatrix_path), "importance matrix GGUF")
        _same_identity(
            replay_f16,
            _validate_f16_gguf(replay_f16_path, architecture=architecture),
            "determinism replay F16 GGUF",
        )
        _same_identity(
            replay_imatrix,
            _validate_imatrix_gguf(replay_imatrix_path),
            "determinism replay importance matrix",
        )
        if model_metadata != _validate_calibrated_model_metadata(
            model_path, architecture=architecture
        ):
            raise ConversionRefused("calibrated model metadata changed")
        if replay_model_metadata != _validate_calibrated_model_metadata(
            replay_model_path, architecture=architecture
        ):
            raise ConversionRefused("determinism replay model metadata changed")
        staged_artifact = gguf.artifact_identity(
            artifact_root,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization="Q4_K_M",
            expected_architecture=architecture,
        )
        _same_artifact_identity(artifact, staged_artifact, "staged artifact")
        final_replay_artifact = gguf.artifact_identity(
            replay_artifact_root,
            entrypoint=ENTRYPOINT,
            expected_digest=replay_tree_digest,
            quantization="Q4_K_M",
            expected_architecture=architecture,
        )
        _same_artifact_identity(
            replay_artifact,
            final_replay_artifact,
            "staged determinism replay artifact",
        )
        _same_identity(
            primary_model_identity,
            _content_identity(model_path, "primary calibrated model"),
            "primary calibrated model bytes",
        )
        _same_identity(
            replay_model_identity,
            _content_identity(replay_model_path, "replay calibrated model"),
            "replay calibrated model bytes",
        )
        if _content_identity(replay_corpus_path, "replay calibration corpus") != {
            "bytes": replay_corpus["bytes"],
            "digest": replay_corpus["digest"],
        }:
            raise ConversionRefused("replay calibration corpus changed during conversion")
        _same_identity(
            primary_model_identity,
            replay_model_identity,
            "determinism replay model bytes",
        )
        _validate_captured_logs(log_root, commands)
        _validate_captured_logs(replay_log_root, replay_commands)
        if (staging / LOAD_SPEC_NAME).read_bytes() != load_raw:
            raise ConversionRefused("staged load specification changed")
        if (staging / CALIBRATION_RECEIPT_NAME).read_bytes() != calibration_raw:
            raise ConversionRefused("staged calibration receipt changed")
        if (staging / RECEIPT_NAME).read_bytes() != conversion_raw:
            raise ConversionRefused("staged calibrated conversion receipt changed")

        transient_files = {
            f"{ARTIFACT_NAME}/{ENTRYPOINT}",
            F16_NAME,
            CALIBRATION_CORPUS_NAME,
            IMATRIX_NAME,
            LOAD_SPEC_NAME,
            CALIBRATION_RECEIPT_NAME,
            RECEIPT_NAME,
        }
        transient_files.update(
            f".conversion-logs/{command['name']}.{stream_name}"
            for command in commands
            for stream_name in ("stdout", "stderr")
        )
        transient_files.update(
            {
                f"determinism-replay/{F16_NAME}",
                f"determinism-replay/{CALIBRATION_CORPUS_NAME}",
                f"determinism-replay/{IMATRIX_NAME}",
                f"determinism-replay/{ARTIFACT_NAME}/{ENTRYPOINT}",
            }
        )
        transient_files.update(
            f"determinism-replay/.conversion-logs/{command['name']}.{stream_name}"
            for command in replay_commands
            for stream_name in ("stdout", "stderr")
        )
        if _bundle_file_set(staging) != frozenset(transient_files):
            raise ConversionRefused("calibrated staging contains unexpected files")
        _assert_child_pycache_absent(staging, "primary calibrated staging before cleanup")
        _assert_private_command_tree(staging, "primary calibrated staging before cleanup")
        _assert_child_pycache_absent(replay_root, "replay calibrated staging before cleanup")
        _assert_private_command_tree(replay_root, "replay calibrated staging before cleanup")
        for private_path in (f16_path, corpus_path, imatrix_path):
            if private_path.is_symlink() or not private_path.is_file():
                raise ConversionRefused("private calibrated intermediate changed before cleanup")
            private_path.unlink()
        for log_path in sorted(log_root.iterdir()):
            if log_path.is_symlink() or not log_path.is_file():
                raise ConversionRefused("bounded command log changed before cleanup")
            log_path.unlink()
        log_root.rmdir()
        for private_path in (
            replay_f16_path,
            replay_corpus_path,
            replay_imatrix_path,
            replay_model_path,
        ):
            if private_path.is_symlink() or not private_path.is_file():
                raise ConversionRefused("determinism replay intermediate changed before cleanup")
            private_path.unlink()
        for log_path in sorted(replay_log_root.iterdir()):
            if log_path.is_symlink() or not log_path.is_file():
                raise ConversionRefused("determinism replay log changed before cleanup")
            log_path.unlink()
        replay_log_root.rmdir()
        replay_artifact_root.rmdir()
        replay_root.rmdir()
        _fsync_path(artifact_root, directory=True)
        _fsync_path(staging, directory=True)

        _assert_child_pycache_absent(staging, "calibrated staging before publication")
        _assert_private_command_tree(staging, "calibrated staging before publication")
        expected_files = frozenset(
            {
                f"{ARTIFACT_NAME}/{ENTRYPOINT}",
                LOAD_SPEC_NAME,
                CALIBRATION_RECEIPT_NAME,
                RECEIPT_NAME,
            }
        )
        if _bundle_file_set(staging) != expected_files:
            raise ConversionRefused("calibrated output bundle contains unexpected files")
        prepublish_tools = _toolchain_identity(request, include_converter_python=current)
        _same_identity(
            initial_tools,
            prepublish_tools,
            "llama.cpp toolchain immediately before publication",
        )
        _publish_directory_noreplace(staging, output)
        published = True
        _fsync_path(output.parent, directory=True)

        final_artifact = gguf.artifact_identity(
            output / ARTIFACT_NAME,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization="Q4_K_M",
            expected_architecture=architecture,
        )
        _same_artifact_identity(artifact, final_artifact, "published artifact")
        if model_metadata != _validate_calibrated_model_metadata(
            output / ARTIFACT_NAME / ENTRYPOINT,
            architecture=architecture,
        ):
            raise ConversionRefused("published calibrated model metadata changed")
        if (output / LOAD_SPEC_NAME).read_bytes() != load_raw:
            raise ConversionRefused("published load specification bytes changed")
        if (output / CALIBRATION_RECEIPT_NAME).read_bytes() != calibration_raw:
            raise ConversionRefused("published calibration receipt bytes changed")
        if (output / RECEIPT_NAME).read_bytes() != conversion_raw:
            raise ConversionRefused("published calibrated conversion receipt bytes changed")
        published_load = _strict_json_file(output / LOAD_SPEC_NAME, "published load specification")
        published_calibration = _strict_json_file(
            output / CALIBRATION_RECEIPT_NAME,
            "published calibration receipt",
        )
        published_conversion = _strict_json_file(
            output / RECEIPT_NAME,
            "published calibrated conversion receipt",
        )
        if published_load != load_manifest:
            raise ConversionRefused("published load specification changed")
        _validate_calibrated_receipts(
            calibration_receipt=published_calibration,
            conversion_receipt=published_conversion,
            calibration_digest=calibration_digest,
            expected_calibration=calibration_receipt,
            expected_conversion=conversion_receipt,
            command_argv=command_argv,
            replay_command_argv=command_argv,
        )
        if _bundle_file_set(output) != expected_files:
            raise ConversionRefused("published calibrated output bundle contains unexpected files")
        return {
            "output_bundle": str(output),
            "artifact": str(output / ARTIFACT_NAME),
            "artifact_digest": tree_digest,
            "entrypoint": str(output / ARTIFACT_NAME / ENTRYPOINT),
            "entrypoint_bytes": artifact["entrypoint"]["bytes"],
            "quantization": "Q4_K_M",
            "calibration_profile": CALIBRATION_PROFILE,
            "max_input_tokens": request.max_input_tokens,
            "load_spec": str(output / LOAD_SPEC_NAME),
            "calibration_receipt": str(output / CALIBRATION_RECEIPT_NAME),
            "conversion_receipt": str(output / RECEIPT_NAME),
        }
    finally:
        if not published and os.path.lexists(staging):
            try:
                current = staging.lstat()
                same_staging = (
                    stat.S_ISDIR(current.st_mode)
                    and not staging.is_symlink()
                    and current.st_dev == staging_stat.st_dev
                    and current.st_ino == staging_stat.st_ino
                    and staging.name.startswith(_STAGING_MARKER)
                    and staging.parent == output.parent
                )
            except OSError:
                same_staging = False
            if same_staging:
                shutil.rmtree(staging)


def convert(request: ConversionRequest) -> dict[str, Any]:
    """Run conversion and atomically publish a fully replayed output bundle."""

    request = _validate_request(request)
    if _calibration_requested(request):
        return _convert_calibrated(request)
    initial_lineage = _load_lineage(request)
    _conversion_schema(initial_lineage, calibrated=False)
    base_model, architecture = _lineage_model_contract(initial_lineage)
    initial_tools = _toolchain_identity(request)
    merged_root = Path(request.training_run).resolve(strict=True) / "merged"
    _regular_directory(merged_root, "validated merged HF tree")

    output = request.output_bundle
    staging = Path(tempfile.mkdtemp(prefix=_STAGING_MARKER, dir=output.parent)).resolve(strict=True)
    staging_stat = staging.stat()
    published = False
    try:
        artifact_root = staging / ARTIFACT_NAME
        artifact_root.mkdir(mode=0o700)
        f16_path = staging / F16_NAME
        model_path = artifact_root / ENTRYPOINT
        converter = Path(str(initial_tools["converter"]["path"]))
        quantizer = Path(str(initial_tools["quantizer"]["path"]))
        convert_argv = (
            str(converter),
            str(merged_root),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        )
        commands = [_conversion_command("convert_f16", convert_argv, cwd=staging)]
        f16_identity = gguf.file_identity(f16_path, "temporary F16 GGUF")
        if int(f16_identity["bytes"]) < 1:
            raise ConversionRefused("converter produced an empty F16 GGUF")
        quantize_argv = (
            str(quantizer),
            str(f16_path),
            str(model_path),
            request.quantization,
        )
        commands.append(_conversion_command("quantize", quantize_argv, cwd=staging))
        gguf.file_identity(model_path, "quantized GGUF")
        _fsync_path(model_path)

        tree_digest = _official_tree_digest(artifact_root)
        artifact = gguf.artifact_identity(
            artifact_root,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization=request.quantization,
            expected_architecture=architecture,
        )
        load_manifest = _load_manifest(
            request.quantization, request.max_input_tokens, base_model=base_model
        )
        _validate_load_manifest_for_lineage(load_manifest, artifact, initial_lineage)
        receipt = _receipt(
            lineage=initial_lineage,
            toolchain=initial_tools,
            commands=commands,
            artifact=artifact,
            load_manifest=load_manifest,
        )
        _validate_generic_conversion_receipt(
            receipt,
            training_lineage=initial_lineage,
            artifact=artifact,
            load_manifest=load_manifest,
        )
        load_raw = _atomic_json_in_staging(staging / LOAD_SPEC_NAME, load_manifest)
        receipt_raw = _atomic_json_in_staging(staging / RECEIPT_NAME, receipt)
        f16_path.unlink()
        _fsync_path(artifact_root, directory=True)
        _fsync_path(staging, directory=True)

        final_lineage = _load_lineage(request)
        final_tools = _toolchain_identity(request)
        _same_identity(initial_lineage, final_lineage, "training lineage")
        _same_identity(initial_tools, final_tools, "llama.cpp toolchain")
        final_staged_artifact = gguf.artifact_identity(
            artifact_root,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization=request.quantization,
            expected_architecture=architecture,
        )
        _same_artifact_identity(artifact, final_staged_artifact, "staged artifact")
        if (staging / LOAD_SPEC_NAME).read_bytes() != load_raw:
            raise ConversionRefused("staged load specification changed")
        if (staging / RECEIPT_NAME).read_bytes() != receipt_raw:
            raise ConversionRefused("staged conversion receipt changed")
        _assert_child_pycache_absent(staging, "generic staging before publication")
        _assert_private_command_tree(staging, "generic staging before publication")
        expected_files = frozenset({f"{ARTIFACT_NAME}/{ENTRYPOINT}", LOAD_SPEC_NAME, RECEIPT_NAME})
        if _bundle_file_set(staging) != expected_files:
            raise ConversionRefused("staged output bundle contains unexpected files")

        prepublish_tools = _toolchain_identity(request)
        _same_identity(
            initial_tools,
            prepublish_tools,
            "llama.cpp toolchain immediately before publication",
        )

        _publish_directory_noreplace(staging, output)
        published = True
        _fsync_path(output.parent, directory=True)
        final_artifact = gguf.artifact_identity(
            output / ARTIFACT_NAME,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization=request.quantization,
            expected_architecture=architecture,
        )
        _same_artifact_identity(artifact, final_artifact, "published artifact")
        if (output / LOAD_SPEC_NAME).read_bytes() != load_raw:
            raise ConversionRefused("published load specification bytes changed")
        if (output / RECEIPT_NAME).read_bytes() != receipt_raw:
            raise ConversionRefused("published conversion receipt bytes changed")
        published_load = _strict_json_file(output / LOAD_SPEC_NAME, "published load specification")
        if published_load != load_manifest:
            raise ConversionRefused("published load specification changed")
        if _strict_json_file(output / RECEIPT_NAME, "published conversion receipt") != receipt:
            raise ConversionRefused("published conversion receipt changed")
        if _bundle_file_set(output) != expected_files:
            raise ConversionRefused("published output bundle contains unexpected files")
        return {
            "output_bundle": str(output),
            "artifact": str(output / ARTIFACT_NAME),
            "artifact_digest": tree_digest,
            "entrypoint": str(output / ARTIFACT_NAME / ENTRYPOINT),
            "entrypoint_bytes": artifact["entrypoint"]["bytes"],
            "quantization": request.quantization,
            "max_input_tokens": request.max_input_tokens,
            "load_spec": str(output / LOAD_SPEC_NAME),
            "conversion_receipt": str(output / RECEIPT_NAME),
        }
    finally:
        if not published and os.path.lexists(staging):
            try:
                current = staging.lstat()
                same_staging = (
                    stat.S_ISDIR(current.st_mode)
                    and not staging.is_symlink()
                    and current.st_dev == staging_stat.st_dev
                    and current.st_ino == staging_stat.st_ino
                    and staging.name.startswith(_STAGING_MARKER)
                    and staging.parent == output.parent
                )
            except OSError:
                same_staging = False
            if same_staging:
                shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--converter-python", type=Path)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--quantization", choices=sorted(SUPPORTED_QUANTIZATIONS), required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--calibration-profile", choices=(CALIBRATION_PROFILE,))
    parser.add_argument("--calibration-current-dataset", type=Path)
    parser.add_argument("--calibration-current-source-corpus", type=Path)
    parser.add_argument("--calibration-aux-dataset", type=Path)
    parser.add_argument("--calibration-aux-source-corpus", type=Path)
    parser.add_argument("--imatrix-tool", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert(
        ConversionRequest(
            training_run=args.training_run,
            training_dataset=args.training_dataset,
            source_corpus=args.source_corpus,
            base=args.base,
            llama_cpp=args.llama_cpp,
            converter=args.converter,
            converter_python=args.converter_python,
            quantizer=args.quantizer,
            output_bundle=args.output_bundle,
            quantization=args.quantization,
            max_input_tokens=args.max_input_tokens,
            calibration_profile=args.calibration_profile,
            calibration_current_dataset=args.calibration_current_dataset,
            calibration_current_source_corpus=args.calibration_current_source_corpus,
            calibration_aux_dataset=args.calibration_aux_dataset,
            calibration_aux_source_corpus=args.calibration_aux_source_corpus,
            imatrix_tool=args.imatrix_tool,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
