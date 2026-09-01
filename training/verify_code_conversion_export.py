#!/usr/bin/env python3
# ruff: noqa: S108
"""Inert, fail-closed verification for one external code-conversion export.

The verifier treats the complete export as hostile data.  It never imports the
GGUF, model, converter, corpus, or generated code; it performs bounded byte and
JSON parsing only.  A caller must provide a separately reviewed, offline
signature-verification hook and an exact trusted public key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

SPEC_SCHEMA: Final[str] = "microtensor.code.oci-worker-spec.current94-v8.v1"
INPUT_MANIFEST_SCHEMA: Final[str] = "microtensor.code.oci-input-manifest.current94-v8.v1"
WORKER_RECEIPT_SCHEMA: Final[str] = "microtensor.code.oci-worker-receipt.v1"
CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v6"
CALIBRATION_SCHEMA: Final[str] = "microtensor.code.imatrix-calibration.v3"
REPLAY_SCHEMA: Final[str] = "microtensor.code.gguf-determinism-replay.v1"
RUNTIME_LIBRARY_SCHEMA: Final[str] = "microtensor.code.llama-cpp-runtime-libraries.v1"
RUNNER_PREFLIGHT_SCHEMA: Final[str] = "microtensor.code.oci-runner-preflight.v1"

BASE_MODEL: Final[str] = (
    "Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a"
)
LLAMA_CPP_REVISION: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
CALIBRATION_PROFILE: Final[str] = "code-public-imatrix128-v1"
CALIBRATION_CORPUS_NAME: Final[str] = "calibration.txt"
IMATRIX_NAME: Final[str] = "calibration.imatrix.gguf"
ENTRYPOINT: Final[str] = "model.gguf"
QUANTIZATION: Final[str] = "Q4_K_M"
MAX_INPUT_TOKENS: Final[int] = 541
GGUF_FILE_TYPE: Final[int] = 15
GGUF_ARCHITECTURE: Final[str] = "qwen2"
MAX_MODEL_BYTES: Final[int] = 1_610_612_736
MAX_JSON_BYTES: Final[int] = 16 * 1024 * 1024
MAX_WORKER_RECEIPT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_LOAD_SPEC_BYTES: Final[int] = 64 * 1024
MAX_SIGNATURE_BYTES: Final[int] = 64 * 1024
MAX_GGUF_HEADER_BYTES: Final[int] = 256 * 1024 * 1024
MAX_GGUF_TENSORS: Final[int] = 100_000
MAX_GGUF_METADATA: Final[int] = 100_000
MAX_GGUF_ARRAY_ITEMS: Final[int] = 2_000_000
MAX_GGUF_STRING_BYTES: Final[int] = 16 * 1024 * 1024

EXPORT_FILES: Final[frozenset[str]] = frozenset(
    {
        "bundle/artifact/model.gguf",
        "bundle/calibration-receipt.json",
        "bundle/conversion-receipt.json",
        "bundle/load-spec.json",
        "worker-receipt.json",
        "worker-receipt.sig",
    }
)
EXPORT_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {"bundle", "bundle/artifact"}
)
BUNDLE_FILES: Final[frozenset[str]] = frozenset(
    {
        "bundle/artifact/model.gguf",
        "bundle/calibration-receipt.json",
        "bundle/conversion-receipt.json",
        "bundle/load-spec.json",
    }
)
EXPORT_CEILINGS: Final[dict[str, int]] = {
    "bundle/artifact/model.gguf": MAX_MODEL_BYTES,
    "bundle/calibration-receipt.json": MAX_JSON_BYTES,
    "bundle/conversion-receipt.json": MAX_JSON_BYTES,
    "bundle/load-spec.json": MAX_LOAD_SPEC_BYTES,
    "worker-receipt.json": MAX_WORKER_RECEIPT_BYTES,
    "worker-receipt.sig": MAX_SIGNATURE_BYTES,
}
REQUIRED_INPUT_IDS: Final[frozenset[str]] = frozenset(
    {
        "training_run",
        "training_dataset",
        "current_source_corpus",
        "base_snapshot",
        "current_calibration_dataset",
        "auxiliary_normalized_dataset",
        "auxiliary_source_corpus",
    }
)
PINNED_TOTAL_INPUT_BYTES: Final[int] = 6_392_237_563
PINNED_INPUT_FILE_COUNT: Final[int] = 35
PINNED_INPUT_AGGREGATE_DIGEST: Final[str] = (
    "sha256:323e2b10ab7aeabf0e6a09a6c4b2297a45b145b6a8dfebe7cd6075c8c8db42cb"
)
ISOLATED_CONVERTER_LAUNCHER: Final[str] = (
    "import runpy,sys,types;"
    "_p=types.ModuleType('training');"
    "_p.__path__=['/opt/microtensor-miner/training'];"
    "_p.__package__='training';"
    "sys.modules['training']=_p;"
    "runpy.run_path('/opt/microtensor-miner/training/convert_code_gguf.py',"
    "run_name='__main__')"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PLACEHOLDER_PREFIX = "UNRESOLVED:"
_GGUF_FIXED_WIDTH: Final[dict[int, int]] = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}
_GGUF_STRING: Final[int] = 8
_GGUF_ARRAY: Final[int] = 9


class VerificationRefused(ValueError):
    """The export or one of its independently supplied contracts changed."""


class SignatureVerifier(Protocol):
    """Offline detached-signature verification supplied by the operator."""

    def __call__(
        self,
        verifier_fd: int,
        message_fd: int,
        signature_fd: int,
        trusted_public_key_fd: int,
        *,
        scheme: str,
        key_id: str,
    ) -> bool: ...


@dataclass
class _BoundDirectory:
    path: Path
    entry_name: str | None
    fd: int
    stable_stat: tuple[int, ...]


@dataclass
class _BoundFile:
    path: Path
    label: str
    fd: int
    identity: dict[str, Any]
    stable_stat: tuple[int, ...]
    ancestry: tuple[_BoundDirectory, ...]


@dataclass
class _ExportBinding:
    root: Path
    expected_uid: int
    files: dict[str, _BoundFile]
    identities: dict[str, dict[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise VerificationRefused(f"{label} must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise VerificationRefused(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(value)}"
        )


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise VerificationRefused(f"{label} must be a {qualifier} integer")
    return value


def _normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationRefused(f"{label} must be non-empty relative POSIX text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise VerificationRefused(f"{label} is not a normalized relative POSIX path")
    if unicodedata.normalize("NFC", value) != value:
        raise VerificationRefused(f"{label} is not NFC-normalized")
    return value


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verification_euid() -> int:
    effective_uid = os.geteuid()
    if effective_uid == 0:
        raise VerificationRefused(
            "conversion export verification must run as a dedicated non-root account"
        )
    return effective_uid


def _open_secure_ancestry(path: Path, label: str) -> tuple[_BoundDirectory, ...]:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise VerificationRefused(f"{label} must be an absolute path without '..'")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opened: list[_BoundDirectory] = []
    pending_fd = -1
    try:
        pending_fd = os.open("/", flags)
        root_stat = os.fstat(pending_fd)
        opened.append(
            _BoundDirectory(Path("/"), None, pending_fd, _stable_stat(root_stat))
        )
        pending_fd = -1
        current = Path("/")
        for component in path.parts[1:]:
            pending_fd = os.open(component, flags, dir_fd=opened[-1].fd)
            info = os.fstat(pending_fd)
            if not stat.S_ISDIR(info.st_mode):
                raise VerificationRefused(f"{label} ancestry contains a non-directory")
            current /= component
            opened.append(
                _BoundDirectory(current, component, pending_fd, _stable_stat(info))
            )
            pending_fd = -1
        _recheck_secure_ancestry(tuple(opened), label)
        return tuple(opened)
    except BaseException:
        if pending_fd >= 0:
            with suppress(OSError):
                os.close(pending_fd)
        for directory in reversed(opened):
            with suppress(OSError):
                os.close(directory.fd)
        raise


def _recheck_secure_ancestry(
    ancestry: tuple[_BoundDirectory, ...], label: str
) -> None:
    if not ancestry:
        raise VerificationRefused(f"{label} has no held ancestry")
    for index, directory in enumerate(ancestry):
        try:
            descriptor_stat = os.fstat(directory.fd)
            if _stable_stat(descriptor_stat) != directory.stable_stat:
                raise VerificationRefused(f"{label} held ancestry changed")
            if index:
                entry_stat = os.stat(
                    str(directory.entry_name),
                    dir_fd=ancestry[index - 1].fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(entry_stat.st_mode)
                    or _stable_stat(entry_stat) != directory.stable_stat
                ):
                    raise VerificationRefused(f"{label} ancestry entry changed")
        except OSError as exc:
            raise VerificationRefused(f"{label} ancestry could not be rechecked: {exc}") from exc


def _close_secure_ancestry(ancestry: tuple[_BoundDirectory, ...]) -> None:
    for directory in reversed(ancestry):
        with suppress(OSError):
            os.close(directory.fd)


def _secure_file_identity(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    minimum_bytes: int = 0,
    allow_writable: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise VerificationRefused(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise VerificationRefused(f"{label} must be a regular non-symlink file")
    if before.st_nlink != 1:
        raise VerificationRefused(f"{label} must have exactly one hard link")
    if not allow_writable and stat.S_IMODE(before.st_mode) & 0o022:
        raise VerificationRefused(f"{label} must not be group/world writable")
    if not minimum_bytes <= before.st_size <= maximum_bytes:
        raise VerificationRefused(
            f"{label} byte size {before.st_size} is outside [{minimum_bytes}, {maximum_bytes}]"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _stable_stat(opened) != _stable_stat(before):
            raise VerificationRefused(f"{label} changed while opening")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise VerificationRefused(f"{label} grew beyond its byte ceiling")
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    except VerificationRefused:
        raise
    except OSError as exc:
        raise VerificationRefused(f"{label} could not be read: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise VerificationRefused(f"{label} disappeared after reading: {exc}") from exc
    if (
        total != before.st_size
        or _stable_stat(after_open) != _stable_stat(before)
        or _stable_stat(after) != _stable_stat(before)
    ):
        raise VerificationRefused(f"{label} changed while hashing")
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": total,
        "digest": "sha256:" + digest.hexdigest(),
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
    }


def _open_bound_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    minimum_bytes: int = 1,
    expected_mode: int,
    expected_uid: int | None = None,
) -> _BoundFile:
    path = Path(path)
    if (
        not path.is_absolute()
        or path.name in {"", ".", ".."}
        or ".." in path.parts
    ):
        raise VerificationRefused(f"{label} path is not a canonical absolute file")
    ancestry = _open_secure_ancestry(path.parent, f"{label} parent")
    try:
        parent_stat = os.fstat(ancestry[-1].fd)
    except BaseException:
        _close_secure_ancestry(ancestry)
        raise
    if stat.S_IMODE(parent_stat.st_mode) != 0o700:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} parent mode must be exactly 0700")
    if expected_uid is not None and parent_stat.st_uid != expected_uid:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} parent is not owned by verifier EUID")
    try:
        before = os.stat(path.name, dir_fd=ancestry[-1].fd, follow_symlinks=False)
    except OSError as exc:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} must be a regular non-symlink file")
    if before.st_nlink != 1:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} must have exactly one hard link")
    if stat.S_IMODE(before.st_mode) != expected_mode:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} mode must be exactly {expected_mode:04o}")
    if expected_uid is not None and before.st_uid != expected_uid:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} is not owned by verifier EUID")
    if not minimum_bytes <= before.st_size <= maximum_bytes:
        _close_secure_ancestry(ancestry)
        raise VerificationRefused(f"{label} byte size is outside its allowed range")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=ancestry[-1].fd)
        opened = os.fstat(descriptor)
        if _stable_stat(opened) != _stable_stat(before):
            raise VerificationRefused(f"{label} changed while opening")
        digest = hashlib.sha256()
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, opened.st_size - offset), offset)
            if not chunk:
                raise VerificationRefused(f"{label} ended during held-descriptor hashing")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if _stable_stat(after) != _stable_stat(opened):
            raise VerificationRefused(f"{label} changed during held-descriptor hashing")
        identity = {
            "path": str(path),
            "bytes": offset,
            "digest": "sha256:" + digest.hexdigest(),
            "mode": f"{expected_mode:04o}",
        }
        bound = _BoundFile(
            path,
            label,
            descriptor,
            identity,
            _stable_stat(opened),
            ancestry,
        )
        _recheck_bound_file(bound)
        return bound
    except Exception:
        if "descriptor" in locals():
            with suppress(OSError):
                os.close(descriptor)
        _close_secure_ancestry(ancestry)
        raise


def _bound_bytes(bound: _BoundFile, *, maximum_bytes: int) -> bytes:
    size = _positive_int(
        bound.identity["bytes"], f"{bound.label} held bytes", allow_zero=True
    )
    if size > maximum_bytes:
        raise VerificationRefused(f"{bound.label} exceeds its held byte ceiling")
    raw = os.pread(bound.fd, size + 1, 0)
    if len(raw) != size:
        raise VerificationRefused(f"{bound.label} changed during held read")
    return raw


def _recheck_bound_file(bound: _BoundFile) -> None:
    try:
        _recheck_secure_ancestry(bound.ancestry, f"{bound.label} parent")
        descriptor_stat = os.fstat(bound.fd)
        path_stat = os.stat(
            bound.path.name,
            dir_fd=bound.ancestry[-1].fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise VerificationRefused(
            f"{bound.label} changed while descriptor was held: {exc}"
        ) from exc
    if (
        _stable_stat(descriptor_stat) != bound.stable_stat
        or _stable_stat(path_stat) != bound.stable_stat
    ):
        raise VerificationRefused(f"{bound.label} changed while descriptor was held")


def _close_bound_files(*files: _BoundFile) -> None:
    for bound in files:
        with suppress(OSError):
            os.close(bound.fd)
        _close_secure_ancestry(bound.ancestry)


def _open_bound_group(
    *requests: tuple[Path, str, int, int],
    expected_uid: int | None = None,
) -> tuple[_BoundFile, ...]:
    opened: list[_BoundFile] = []
    try:
        for path, label, maximum_bytes, expected_mode in requests:
            opened.append(
                _open_bound_file(
                    path,
                    label,
                    maximum_bytes=maximum_bytes,
                    expected_mode=expected_mode,
                    expected_uid=expected_uid,
                )
            )
        return tuple(opened)
    except Exception:
        _close_bound_files(*opened)
        raise


def _read_regular_bytes(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    allow_writable: bool = False,
) -> bytes:
    identity = _secure_file_identity(
        path,
        label,
        maximum_bytes=maximum_bytes,
        allow_writable=allow_writable,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        raw = b""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise VerificationRefused(f"{label} exceeds its byte ceiling")
        raw = b"".join(chunks)
        after_open = os.fstat(descriptor)
    except VerificationRefused:
        raise
    except OSError as exc:
        raise VerificationRefused(f"{label} could not be read: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        len(raw) != identity["bytes"]
        or _digest_bytes(raw) != identity["digest"]
        or _stable_stat(opened) != _stable_stat(after_open)
    ):
        raise VerificationRefused(f"{label} changed during bounded read")
    return raw


def _strict_json(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    allow_writable: bool = False,
) -> dict[str, Any]:
    raw = _read_regular_bytes(
        path,
        label,
        maximum_bytes=maximum_bytes,
        allow_writable=allow_writable,
    )
    return _decode_strict_json(raw, label)


def _decode_strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationRefused(f"{label} repeats JSON field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise VerificationRefused(f"{label} contains non-finite constant {value!r}")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except VerificationRefused:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationRefused(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    return dict(_mapping(value, label))


def _strict_bound_json(
    bound: _BoundFile, label: str, *, maximum_bytes: int
) -> dict[str, Any]:
    return _decode_strict_json(
        _bound_bytes(bound, maximum_bytes=maximum_bytes),
        label,
    )


def _reject_placeholders(value: Any, label: str = "$", *, seen: set[int] | None = None) -> None:
    if isinstance(value, str):
        if value.startswith(_PLACEHOLDER_PREFIX):
            raise VerificationRefused(f"{label} remains unresolved: {value}")
        return
    if value is None:
        raise VerificationRefused(f"{label} contains null instead of an explicit identity")
    if isinstance(value, Mapping):
        identity = id(value)
        observed = set() if seen is None else seen
        if identity in observed:
            raise VerificationRefused(f"{label} contains a recursive object")
        observed.add(identity)
        for key, child in value.items():
            _reject_placeholders(child, f"{label}.{key}", seen=observed)
        observed.remove(identity)
        return
    if isinstance(value, list):
        observed = set() if seen is None else seen
        identity = id(value)
        if identity in observed:
            raise VerificationRefused(f"{label} contains a recursive array")
        observed.add(identity)
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{label}[{index}]", seen=observed)
        observed.remove(identity)


def _identity_projection(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"bytes": identity.get("bytes"), "digest": identity.get("digest")}


def _validate_content_identity(
    value: Any,
    label: str,
    *,
    allow_zero: bool = False,
) -> dict[str, Any]:
    identity = _mapping(value, label)
    _exact_keys(identity, frozenset({"bytes", "digest"}), label)
    size = _positive_int(identity.get("bytes"), f"{label} bytes", allow_zero=allow_zero)
    digest = identity.get("digest")
    if not _is_digest(digest):
        raise VerificationRefused(f"{label} digest is malformed")
    return {"bytes": size, "digest": digest}


def _inspect_export(
    root: Path,
    *,
    expected_uid: int | None = None,
    hash_files: bool = True,
) -> dict[str, dict[str, Any]]:
    root = Path(root)
    if not root.is_absolute():
        raise VerificationRefused("export root must be absolute")
    try:
        parent_stat = root.parent.lstat()
        root_stat = root.lstat()
    except OSError as exc:
        raise VerificationRefused(f"export root is unavailable: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise VerificationRefused("export root must be a regular non-symlink directory")
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise VerificationRefused("export parent must be a private 0700 directory")
    if expected_uid is not None and (
        parent_stat.st_uid != expected_uid or root_stat.st_uid != expected_uid
    ):
        raise VerificationRefused("export parent/root is not owned by verifier EUID")
    if stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise VerificationRefused("export root mode must be exactly 0700")
    files: set[str] = set()
    directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        relative_directory = Path(directory).relative_to(root)
        for name in directory_names:
            path = Path(directory) / name
            relative = (relative_directory / name).as_posix()
            _normalized_relative(relative, "export directory")
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise VerificationRefused(f"export directory changed type: {relative}")
            if info.st_dev != root_stat.st_dev:
                raise VerificationRefused(f"export crosses a filesystem boundary: {relative}")
            if stat.S_IMODE(info.st_mode) != 0o700:
                raise VerificationRefused(f"export directory mode is not 0700: {relative}")
            if expected_uid is not None and info.st_uid != expected_uid:
                raise VerificationRefused(
                    f"export directory is not owned by verifier EUID: {relative}"
                )
            directories.add(relative)
        for name in file_names:
            path = Path(directory) / name
            relative = (relative_directory / name).as_posix()
            _normalized_relative(relative, "export file")
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise VerificationRefused(f"export contains a non-regular file: {relative}")
            if info.st_dev != root_stat.st_dev:
                raise VerificationRefused(f"export file crosses a filesystem boundary: {relative}")
            if info.st_nlink != 1:
                raise VerificationRefused(f"export file has multiple hard links: {relative}")
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise VerificationRefused(f"export file mode is not 0600: {relative}")
            if expected_uid is not None and info.st_uid != expected_uid:
                raise VerificationRefused(
                    f"export file is not owned by verifier EUID: {relative}"
                )
            files.add(relative)
    if files != EXPORT_FILES or directories != EXPORT_DIRECTORIES:
        raise VerificationRefused(
            "export file set changed: "
            f"files={sorted(files)}, directories={sorted(directories)}"
        )
    if not hash_files:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for relative in sorted(files):
        result[relative] = _secure_file_identity(
            root / relative,
            f"export file {relative}",
            maximum_bytes=EXPORT_CEILINGS[relative],
            minimum_bytes=1,
        )
    return result


def _rehash_bound_file(bound: _BoundFile, *, maximum_bytes: int) -> None:
    before = os.fstat(bound.fd)
    if _stable_stat(before) != bound.stable_stat:
        raise VerificationRefused(f"{bound.label} changed before held rehash")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(bound.fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise VerificationRefused(f"{bound.label} ended during held rehash")
        offset += len(chunk)
        if offset > maximum_bytes:
            raise VerificationRefused(f"{bound.label} exceeds its held byte ceiling")
        digest.update(chunk)
    after = os.fstat(bound.fd)
    if (
        _stable_stat(after) != bound.stable_stat
        or offset != bound.identity["bytes"]
        or "sha256:" + digest.hexdigest() != bound.identity["digest"]
    ):
        raise VerificationRefused(f"{bound.label} changed during held rehash")
    _recheck_bound_file(bound)


def _recheck_export_binding(binding: _ExportBinding) -> None:
    directories: dict[Path, _BoundDirectory] = {}
    for relative, bound in binding.files.items():
        _rehash_bound_file(bound, maximum_bytes=EXPORT_CEILINGS[relative])
        file_stat = os.fstat(bound.fd)
        if (
            file_stat.st_uid != binding.expected_uid
            or stat.S_IMODE(file_stat.st_mode) != 0o600
        ):
            raise VerificationRefused(f"held export file ownership/mode changed: {relative}")
        for directory in bound.ancestry:
            previous = directories.setdefault(directory.path, directory)
            if previous.stable_stat != directory.stable_stat:
                raise VerificationRefused("held export ancestry identities disagree")
    required_directories = {
        binding.root.parent,
        binding.root,
        binding.root / "bundle",
        binding.root / "bundle/artifact",
    }
    if not required_directories.issubset(directories):
        raise VerificationRefused("held export ancestry is incomplete")
    for path in required_directories:
        directory = directories[path]
        info = os.fstat(directory.fd)
        if info.st_uid != binding.expected_uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise VerificationRefused(f"held export directory ownership/mode changed: {path}")
    root_fd = directories[binding.root].fd
    bundle_fd = directories[binding.root / "bundle"].fd
    artifact_fd = directories[binding.root / "bundle/artifact"].fd
    expected_names = {
        root_fd: {"bundle", "worker-receipt.json", "worker-receipt.sig"},
        bundle_fd: {
            "artifact",
            "calibration-receipt.json",
            "conversion-receipt.json",
            "load-spec.json",
        },
        artifact_fd: {"model.gguf"},
    }
    for descriptor, expected in expected_names.items():
        try:
            actual = set(os.listdir(descriptor))
        except OSError as exc:
            raise VerificationRefused(f"held export directory could not be listed: {exc}") from exc
        if actual != expected:
            raise VerificationRefused("held export directory entry set changed")


def _bind_export(root: Path, expected_uid: int) -> _ExportBinding:
    root = Path(root)
    _inspect_export(root, expected_uid=expected_uid, hash_files=False)
    requests = tuple(
        (
            root / relative,
            f"export file {relative}",
            EXPORT_CEILINGS[relative],
            0o600,
        )
        for relative in sorted(EXPORT_FILES)
    )
    bound_files = _open_bound_group(*requests, expected_uid=expected_uid)
    files = dict(zip(sorted(EXPORT_FILES), bound_files, strict=True))
    binding = _ExportBinding(
        root=root,
        expected_uid=expected_uid,
        files=files,
        identities={relative: dict(bound.identity) for relative, bound in files.items()},
    )
    try:
        _recheck_export_binding(binding)
        return binding
    except Exception:
        _close_bound_files(*bound_files)
        raise


def _close_export_binding(binding: _ExportBinding) -> None:
    _close_bound_files(*binding.files.values())


def _validate_review(value: Any, label: str) -> None:
    review = _mapping(value, label)
    _exact_keys(review, frozenset({"status", "reviewer", "review_digest"}), label)
    if review.get("status") != "accepted":
        raise VerificationRefused(f"{label} has not been accepted")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise VerificationRefused(f"{label} identifies no reviewer")
    if not _is_digest(review.get("review_digest")):
        raise VerificationRefused(f"{label} digest is malformed")


def _validate_named_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _mapping(value, label)
    _exact_keys(identity, frozenset({"name", "bytes", "digest"}), label)
    if not isinstance(identity.get("name"), str) or not identity["name"].strip():
        raise VerificationRefused(f"{label} name is empty")
    content = _validate_content_identity(
        {"bytes": identity.get("bytes"), "digest": identity.get("digest")},
        label,
    )
    return {"name": identity["name"], **content}


def _validate_signature_verifier_policy(value: Any, label: str) -> dict[str, Any]:
    identity = _mapping(value, label)
    _exact_keys(
        identity,
        frozenset({"bytes", "closure", "digest", "format", "name"}),
        label,
    )
    if (
        identity.get("format") != "static-elf-linux-amd64"
        or identity.get("closure") != "single-file-no-pt-interp-no-dynamic"
    ):
        raise VerificationRefused(f"{label} closure is not a pinned static ELF")
    named = _validate_named_identity(
        {field: identity.get(field) for field in ("name", "bytes", "digest")},
        label,
    )
    return {
        **named,
        "format": identity["format"],
        "closure": identity["closure"],
    }


def _validate_static_verifier_elf(bound: _BoundFile) -> None:
    raw = _bound_bytes(bound, maximum_bytes=64 * 1024 * 1024)
    if (
        len(raw) < 64
        or raw[:4] != b"\x7fELF"
        or raw[4:7] != b"\x02\x01\x01"
    ):
        raise VerificationRefused("signature verifier is not ELF64 little-endian")
    header = struct.unpack("<16sHHIQQQIHHHHHH", raw[:64])
    (
        _ident,
        elf_type,
        machine,
        version,
        _entry,
        phoff,
        _shoff,
        _flags,
        ehsize,
        phentsize,
        phnum,
        *_,
    ) = header
    if (
        elf_type not in {2, 3}
        or machine != 62
        or version != 1
        or ehsize != 64
        or phentsize != 56
        or not 1 <= phnum <= 128
        or phoff + phentsize * phnum > len(raw)
    ):
        raise VerificationRefused("signature verifier ELF header changed")
    has_load = False
    for index in range(phnum):
        start = phoff + index * phentsize
        program_type = struct.unpack("<I", raw[start : start + 4])[0]
        if program_type in {2, 3}:
            raise VerificationRefused("signature verifier has a dynamic/interpreter closure")
        has_load |= program_type == 1
    if not has_load:
        raise VerificationRefused("signature verifier ELF contains no loadable segment")


def _validate_worker_spec(spec: Mapping[str, Any]) -> None:
    _reject_placeholders(spec)
    _exact_keys(
        spec,
        frozenset(
            {
                "cgroup",
                "command",
                "execution_protocol",
                "expected_child_environment",
                "expected_internal_commands",
                "expected_output",
                "image",
                "independent_review",
                "input_manifest_identity",
                "mounts",
                "oci_config_identity",
                "platform",
                "receipt_signature",
                "resources",
                "runnable",
                "runner_preflight_evidence",
                "runtime_library_contract",
                "schema",
                "security",
                "status",
                "unresolved",
            }
        ),
        "worker spec",
    )
    if spec.get("schema") != SPEC_SCHEMA:
        raise VerificationRefused("worker spec schema changed")
    if spec.get("status") != "ready_and_independently_reviewed" or spec.get("runnable") is not True:
        raise VerificationRefused("worker spec is incomplete or explicitly non-runnable")
    unresolved = spec.get("unresolved")
    if unresolved != []:
        raise VerificationRefused("worker spec retains unresolved requirements")
    _validate_review(spec.get("independent_review"), "worker spec independent review")
    platform = _mapping(spec.get("platform"), "worker platform")
    if platform != {
        "architecture": "amd64",
        "os": "linux",
        "rootless_or_userns_remapped": True,
    }:
        raise VerificationRefused("worker platform contract changed")
    image = _mapping(spec.get("image"), "worker image")
    _exact_keys(
        image,
        frozenset({"digest", "reference", "sbom", "source_closure"}),
        "worker image",
    )
    image_digest = image.get("digest")
    if (
        not _is_digest(image_digest)
        or image.get("reference") != f"microtensor-converter@{image_digest}"
    ):
        raise VerificationRefused("worker image is not pinned by its exact digest")
    _validate_named_identity(image.get("sbom"), "worker image SBOM")
    source_closure = _mapping(image.get("source_closure"), "worker image source closure")
    _exact_keys(
        source_closure,
        frozenset({"files", "tree_digest", "worktree_state"}),
        "worker image source closure",
    )
    if (
        not _is_digest(source_closure.get("tree_digest"))
        or source_closure.get("worktree_state") != "clean_immutable_snapshot"
    ):
        raise VerificationRefused("worker source-closure digest is malformed")
    files = source_closure.get("files")
    if not isinstance(files, list) or len(files) < 8:
        raise VerificationRefused("worker source closure is incomplete")
    source_paths: set[str] = set()
    for index, item in enumerate(files):
        entry = _mapping(item, f"worker source file {index}")
        _exact_keys(entry, frozenset({"path", "bytes", "digest"}), f"worker source file {index}")
        source_path = _normalized_relative(
            entry.get("path"), f"worker source file {index} path"
        )
        if source_path in source_paths:
            raise VerificationRefused("worker source closure paths are duplicated")
        source_paths.add(source_path)
        _validate_content_identity(
            {"bytes": entry.get("bytes"), "digest": entry.get("digest")},
            f"worker source file {index}",
        )
    _validate_content_identity(spec.get("input_manifest_identity"), "pinned input manifest")
    _validate_content_identity(spec.get("oci_config_identity"), "pinned OCI config")
    _validate_named_identity(
        spec.get("runner_preflight_evidence"), "runner preflight evidence"
    )
    resources = _mapping(spec.get("resources"), "worker resources")
    if dict(resources) != {
        "cpu_cores": 4,
        "memory_bytes": 34_359_738_368,
        "persistent_workspace_bytes": 21_474_836_480,
        "tmpfs_bytes": 17_179_869_184,
    }:
        raise VerificationRefused("worker resource contract changed")
    _validate_mount_contract(spec.get("mounts"))
    security = _mapping(spec.get("security"), "worker security")
    required_security = {
        "capabilities": [],
        "network_mode": "none",
        "no_new_privileges": True,
        "private_namespaces": ["cgroup", "ipc", "mount", "network", "pid", "user", "uts"],
        "root_filesystem_read_only": True,
        "runtime_mode": "rootless_or_userns_remapped",
    }
    for field, expected in required_security.items():
        if security.get(field) != expected:
            raise VerificationRefused(f"worker security field {field!r} changed")
    profiles = _mapping(security.get("profiles"), "worker security profiles")
    _exact_keys(profiles, frozenset({"lsm", "seccomp"}), "worker security profiles")
    _validate_named_identity(profiles.get("seccomp"), "seccomp profile")
    _validate_named_identity(profiles.get("lsm"), "LSM profile")
    cgroup = _mapping(spec.get("cgroup"), "worker cgroup")
    expected_cgroup = {
        "cpu_quota_cores": 4,
        "memory_max_bytes": 34_359_738_368,
        "memory_swap_max_bytes": 0,
        "pids_max": 128,
    }
    if dict(cgroup) != expected_cgroup:
        raise VerificationRefused("worker cgroup limits changed")
    if spec.get("command") != _expected_top_level_command():
        raise VerificationRefused("worker top-level command changed")
    if spec.get("expected_internal_commands") != _expected_internal_commands():
        raise VerificationRefused("worker internal command argv changed")
    if spec.get("expected_child_environment") != _expected_child_environment():
        raise VerificationRefused("worker child environment changed")
    _validate_runtime_contract(spec.get("runtime_library_contract"), "worker runtime contract")
    signature = _mapping(spec.get("receipt_signature"), "worker receipt signature policy")
    _exact_keys(
        signature,
        frozenset({"key_id", "scheme", "trusted_public_key", "verifier"}),
        "worker receipt signature policy",
    )
    if not all(
        isinstance(signature.get(field), str) and signature[field]
        for field in ("scheme", "key_id")
    ):
        raise VerificationRefused("worker receipt signature policy is incomplete")
    _validate_named_identity(signature.get("trusted_public_key"), "trusted receipt public key")
    _validate_signature_verifier_policy(
        signature.get("verifier"), "receipt signature verifier"
    )
    expected_output = _mapping(spec.get("expected_output"), "expected output")
    if expected_output.get("bundle_files") != sorted(BUNDLE_FILES):
        raise VerificationRefused("expected bundle file set changed")
    identities = expected_output.get("file_identities")
    if not isinstance(identities, list) or len(identities) != len(BUNDLE_FILES):
        raise VerificationRefused("expected output identities are incomplete")
    seen: set[str] = set()
    for index, item in enumerate(identities):
        identity = _mapping(item, f"expected output identity {index}")
        _exact_keys(
            identity,
            frozenset({"bytes", "digest", "path"}),
            f"expected output identity {index}",
        )
        path = _normalized_relative(identity.get("path"), f"expected output identity {index} path")
        if path not in BUNDLE_FILES or path in seen:
            raise VerificationRefused("expected output identity paths changed")
        seen.add(path)
        maximum = MAX_MODEL_BYTES if path.endswith("model.gguf") else MAX_JSON_BYTES
        size = _positive_int(identity.get("bytes"), f"expected output identity {path} bytes")
        if size > maximum or not _is_digest(identity.get("digest")):
            raise VerificationRefused(f"expected output identity {path} is malformed")
    if seen != set(BUNDLE_FILES):
        raise VerificationRefused("expected output identities omit a bundle file")
    _validate_execution_protocol(spec.get("execution_protocol"))


def _validate_mount_contract(value: Any) -> None:
    mounts = value
    if not isinstance(mounts, list):
        raise VerificationRefused("worker mounts must be an array")
    expected = [
        {
            "destination": "/dev/shm",
            "kind": "tmpfs",
            "options": ["nodev", "noexec", "nosuid"],
            "read_only": False,
            "size_bytes": 17_179_869_184,
            "source_role": "ephemeral_memory",
        },
        *[
            {
                "destination": destination,
                "kind": "bind",
                "options": ["nodev", "noexec", "nosuid", "rbind"],
                "read_only": True,
                "source_role": identifier,
            }
            for identifier, destination in (
                ("training_run", "/dev/shm/in/training-run"),
                ("training_dataset", "/dev/shm/in/training-dataset"),
                ("current_source_corpus", "/dev/shm/in/current-corpus.json"),
                ("base_snapshot", "/dev/shm/in/base"),
                (
                    "current_calibration_dataset",
                    "/dev/shm/in/current-calibration-dataset",
                ),
                ("auxiliary_normalized_dataset", "/dev/shm/in/aux-dataset"),
                ("auxiliary_source_corpus", "/dev/shm/in/aux-corpus.json"),
            )
        ],
    ]
    if mounts != expected:
        raise VerificationRefused("worker mount contract changed")


def _validate_execution_protocol(value: Any) -> None:
    protocol = _mapping(value, "worker execution protocol")
    expected = {
        "container_removed_before_receipt": True,
        "container_stopped_before_export": True,
        "copy_only_after_cgroup_empty": True,
        "export_root": "/dev/shm/export/qwen25-current94-v8-q4-m541-v6-bundle",
        "host_persistent_mount_visible_to_container": False,
        "input_preflight_and_postflight_hashing": True,
        "private_intermediates_exported": False,
        "return_only_exact_export_file_set": True,
        "stdin": "/dev/null",
        "umask": "0077",
        "working_directory": "/opt/microtensor-miner",
    }
    if dict(protocol) != expected:
        raise VerificationRefused("worker execution/export protocol changed")


def _expected_top_level_command() -> list[str]:
    return [
        "/opt/python/bin/python3.11",
        "-I",
        "-B",
        "-c",
        ISOLATED_CONVERTER_LAUNCHER,
        "--training-run",
        "/dev/shm/in/training-run",
        "--training-dataset",
        "/dev/shm/in/training-dataset",
        "--source-corpus",
        "/dev/shm/in/current-corpus.json",
        "--base",
        "/dev/shm/in/base",
        "--llama-cpp",
        "/tmp/llama.cpp",
        "--converter",
        "/tmp/llama.cpp/convert_hf_to_gguf.py",
        "--converter-python",
        "/opt/python/bin/python3.11",
        "--quantizer",
        "/tmp/llama.cpp/build/bin/llama-quantize",
        "--output-bundle",
        "/dev/shm/export/qwen25-current94-v8-q4-m541-v6-bundle",
        "--quantization",
        QUANTIZATION,
        "--max-input-tokens",
        str(MAX_INPUT_TOKENS),
        "--calibration-profile",
        CALIBRATION_PROFILE,
        "--calibration-current-dataset",
        "/dev/shm/in/current-calibration-dataset",
        "--calibration-current-source-corpus",
        "/dev/shm/in/current-corpus.json",
        "--calibration-aux-dataset",
        "/dev/shm/in/aux-dataset",
        "--calibration-aux-source-corpus",
        "/dev/shm/in/aux-corpus.json",
        "--imatrix-tool",
        "/tmp/llama.cpp/build/bin/llama-imatrix",
    ]


def _expected_internal_commands() -> list[dict[str, Any]]:
    return [
        {
            "name": "convert_f16",
            "argv": [
                "/opt/python/bin/python3.11",
                "/tmp/llama.cpp/convert_hf_to_gguf.py",
                "/dev/shm/in/training-run/merged",
                "--outfile",
                "model-f16.gguf",
                "--outtype",
                "f16",
            ],
        },
        {
            "name": "calibrate_imatrix",
            "argv": [
                "/tmp/llama.cpp/build/bin/llama-imatrix",
                "--offline",
                "--model",
                "model-f16.gguf",
                "--file",
                "calibration.txt",
                "--output",
                "calibration.imatrix.gguf",
                "--output-format",
                "gguf",
                "--ctx-size",
                "512",
                "--chunks",
                "128",
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
                "129",
                "--save-frequency",
                "0",
            ],
        },
        {
            "name": "quantize",
            "argv": [
                "/tmp/llama.cpp/build/bin/llama-quantize",
                "--imatrix",
                "calibration.imatrix.gguf",
                "model-f16.gguf",
                "artifact/model.gguf",
                QUANTIZATION,
                "1",
            ],
        },
    ]


def _expected_child_environment() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/opt/python/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": ".microtensor-empty-pycache",
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "WANDB_MODE": "offline",
    }


def _validate_input_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _reject_placeholders(manifest)
    _exact_keys(
        manifest,
        frozenset(
            {
                "aggregate_digest",
                "file_count",
                "independent_review",
                "inputs",
                "schema",
                "sealed",
                "snapshot",
                "status",
                "total_input_bytes",
                "unresolved",
            }
        ),
        "input manifest",
    )
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise VerificationRefused("input manifest schema changed")
    if (
        manifest.get("status") != "sealed_and_independently_reviewed"
        or manifest.get("sealed") is not True
    ):
        raise VerificationRefused("input manifest is not sealed and independently reviewed")
    if manifest.get("unresolved") != []:
        raise VerificationRefused("input manifest retains unresolved requirements")
    _validate_review(manifest.get("independent_review"), "input manifest independent review")
    snapshot = _mapping(manifest.get("snapshot"), "input snapshot")
    if (
        snapshot.get("immutable_for_run") is not True
        or not isinstance(snapshot.get("technology"), str)
        or not snapshot["technology"].strip()
        or snapshot.get("host_writers_excluded") is not True
    ):
        raise VerificationRefused("input snapshot is not immutable for the complete run")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise VerificationRefused("input manifest inputs must be an array")
    identifiers: set[str] = set()
    mounts: set[str] = set()
    expected_inputs = {
        "training_run": ("directory", "/dev/shm/in/training-run", 3_258_141_232),
        "training_dataset": ("directory", "/dev/shm/in/training-dataset", 137_928),
        "current_source_corpus": ("file", "/dev/shm/in/current-corpus.json", 152_605),
        "base_snapshot": ("directory", "/dev/shm/in/base", 3_098_955_668),
        "current_calibration_dataset": (
            "directory",
            "/dev/shm/in/current-calibration-dataset",
            137_924,
        ),
        "auxiliary_normalized_dataset": (
            "directory",
            "/dev/shm/in/aux-dataset",
            15_688_217,
        ),
        "auxiliary_source_corpus": ("file", "/dev/shm/in/aux-corpus.json", 19_023_989),
    }
    file_count = 0
    total = 0
    for index, item in enumerate(inputs):
        entry = _mapping(item, f"input {index}")
        _exact_keys(
            entry,
            frozenset({"component_tree_digests", "files", "id", "kind", "mount", "total_bytes"}),
            f"input {index}",
        )
        identifier = entry.get("id")
        mount = entry.get("mount")
        if not isinstance(identifier, str) or identifier in identifiers:
            raise VerificationRefused(f"input {index} identifier is empty or duplicated")
        if not isinstance(mount, str) or not mount.startswith("/dev/shm/in/") or mount in mounts:
            raise VerificationRefused(f"input {index} mount is malformed or duplicated")
        if entry.get("kind") not in {"directory", "file"}:
            raise VerificationRefused(f"input {index} kind changed")
        expected_entry = expected_inputs.get(identifier)
        if expected_entry != (entry.get("kind"), mount, entry.get("total_bytes")):
            raise VerificationRefused(f"input {identifier!r} mount/type/size contract changed")
        identifiers.add(identifier)
        mounts.add(mount)
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise VerificationRefused(f"input {index} has no files")
        seen_paths: set[str] = set()
        subtotal = 0
        for file_index, value in enumerate(files):
            file_entry = _mapping(value, f"input {index} file {file_index}")
            _exact_keys(
                file_entry,
                frozenset({"bytes", "digest", "path"}),
                f"input {index} file {file_index}",
            )
            path = _normalized_relative(
                file_entry.get("path"), f"input {index} file {file_index} path"
            )
            if path in seen_paths:
                raise VerificationRefused(f"input {index} repeats file path {path!r}")
            seen_paths.add(path)
            size = _positive_int(
                file_entry.get("bytes"),
                f"input {index} file {path} bytes",
                allow_zero=True,
            )
            if not _is_digest(file_entry.get("digest")):
                raise VerificationRefused(f"input {index} file {path} digest is malformed")
            subtotal += size
            file_count += 1
        if entry.get("total_bytes") != subtotal:
            raise VerificationRefused(f"input {index} total byte count changed")
        component_digests = _mapping(
            entry.get("component_tree_digests"), f"input {index} component tree digests"
        )
        if any(
            not isinstance(key, str) or not _is_digest(value)
            for key, value in component_digests.items()
        ):
            raise VerificationRefused(f"input {index} component tree digests are malformed")
        total += subtotal
    if identifiers != REQUIRED_INPUT_IDS:
        raise VerificationRefused("input manifest logical input set changed")
    if total != PINNED_TOTAL_INPUT_BYTES or manifest.get("total_input_bytes") != total:
        raise VerificationRefused("input manifest total byte count changed")
    aggregate = _digest_bytes(_canonical_bytes(inputs))
    if manifest.get("aggregate_digest") != aggregate:
        raise VerificationRefused("input manifest aggregate digest changed")
    if aggregate != PINNED_INPUT_AGGREGATE_DIGEST:
        raise VerificationRefused("input manifest is not the pinned aggregate snapshot")
    if file_count != PINNED_INPUT_FILE_COUNT or manifest.get("file_count") != file_count:
        raise VerificationRefused("input manifest file count changed")
    return {"aggregate_digest": aggregate, "file_count": file_count, "total_bytes": total}


def _read_exact(handle: Any, size: int, file_size: int, label: str) -> bytes:
    if size < 0 or handle.tell() + size > file_size or handle.tell() + size > MAX_GGUF_HEADER_BYTES:
        raise VerificationRefused(f"GGUF {label} extends beyond a bounded header")
    raw = handle.read(size)
    if len(raw) != size:
        raise VerificationRefused(f"GGUF ends inside {label}")
    return raw


def _u32(handle: Any, file_size: int, label: str) -> int:
    return int(struct.unpack("<I", _read_exact(handle, 4, file_size, label))[0])


def _u64(handle: Any, file_size: int, label: str) -> int:
    return int(struct.unpack("<Q", _read_exact(handle, 8, file_size, label))[0])


def _gguf_string(handle: Any, file_size: int, label: str) -> str:
    size = _u64(handle, file_size, f"{label} length")
    if size > MAX_GGUF_STRING_BYTES:
        raise VerificationRefused(f"GGUF {label} is implausibly large")
    try:
        return _read_exact(handle, size, file_size, label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationRefused(f"GGUF {label} is not UTF-8") from exc


def _gguf_scalar(handle: Any, kind: int, file_size: int, label: str) -> Any:
    formats = {
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
    format_value = formats.get(kind)
    if format_value is None:
        raise VerificationRefused(f"GGUF {label} has unknown scalar type {kind}")
    return struct.unpack(
        format_value,
        _read_exact(handle, struct.calcsize(format_value), file_size, label),
    )[0]


def _gguf_value(handle: Any, kind: int, file_size: int, label: str, *, retain: bool) -> Any:
    if kind in _GGUF_FIXED_WIDTH:
        value = _gguf_scalar(handle, kind, file_size, label)
        return value if retain else None
    if kind == _GGUF_STRING:
        value = _gguf_string(handle, file_size, label)
        return value if retain else None
    if kind == _GGUF_ARRAY:
        element_kind = _u32(handle, file_size, f"{label} array type")
        count = _u64(handle, file_size, f"{label} array count")
        if count > MAX_GGUF_ARRAY_ITEMS or element_kind == _GGUF_ARRAY:
            raise VerificationRefused(f"GGUF {label} has an unsupported array")
        values: list[Any] | None = [] if retain else None
        for index in range(count):
            value = _gguf_value(
                handle,
                element_kind,
                file_size,
                f"{label}[{index}]",
                retain=retain,
            )
            if values is not None:
                values.append(value)
        return values
    raise VerificationRefused(f"GGUF {label} has unknown value type {kind}")


def _static_gguf_identity_bound(bound: _BoundFile) -> dict[str, Any]:
    _rehash_bound_file(bound, maximum_bytes=MAX_MODEL_BYTES)
    identity = bound.identity
    file_size = int(identity["bytes"])
    descriptor = -1
    wanted = {
        "general.alignment",
        "general.architecture",
        "general.file_type",
        "quantize.imatrix.chunks_count",
        "quantize.imatrix.dataset",
        "quantize.imatrix.entries_count",
        "quantize.imatrix.file",
    }
    try:
        descriptor = os.dup(bound.fd)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            if _read_exact(handle, 4, file_size, "magic") != b"GGUF":
                raise VerificationRefused("model is not a GGUF file")
            version = _u32(handle, file_size, "version")
            if version != 3:
                raise VerificationRefused("GGUF version must be exactly 3")
            tensor_count = _u64(handle, file_size, "tensor count")
            metadata_count = _u64(handle, file_size, "metadata count")
            if not 1 <= tensor_count <= MAX_GGUF_TENSORS:
                raise VerificationRefused("GGUF tensor count is outside the reviewed ceiling")
            if metadata_count > MAX_GGUF_METADATA:
                raise VerificationRefused("GGUF metadata count exceeds the reviewed ceiling")
            metadata: dict[str, Any] = {}
            seen_keys: set[str] = set()
            for index in range(metadata_count):
                key = _gguf_string(handle, file_size, f"metadata key {index}")
                if key in seen_keys:
                    raise VerificationRefused(f"GGUF repeats metadata key {key!r}")
                seen_keys.add(key)
                kind = _u32(handle, file_size, f"metadata type {key!r}")
                value = _gguf_value(handle, kind, file_size, key, retain=key in wanted)
                if key in wanted:
                    metadata[key] = {"type": kind, "value": value}
            offsets: list[int] = []
            tensor_names: set[str] = set()
            for index in range(tensor_count):
                name = _gguf_string(handle, file_size, f"tensor {index} name")
                if not name or name in tensor_names:
                    raise VerificationRefused("GGUF tensor name is empty or duplicated")
                tensor_names.add(name)
                dimensions_count = _u32(handle, file_size, f"tensor {name!r} dimensions")
                if not 1 <= dimensions_count <= 4:
                    raise VerificationRefused(f"GGUF tensor {name!r} dimensions changed")
                dimensions = [
                    _u64(handle, file_size, f"tensor {name!r} dimension {axis}")
                    for axis in range(dimensions_count)
                ]
                if any(value < 1 for value in dimensions):
                    raise VerificationRefused(f"GGUF tensor {name!r} has an empty dimension")
                _u32(handle, file_size, f"tensor {name!r} type")
                offsets.append(_u64(handle, file_size, f"tensor {name!r} offset"))
            header_end = handle.tell()
            if _stable_stat(os.fstat(handle.fileno())) != bound.stable_stat:
                raise VerificationRefused("GGUF model changed during held parsing")
    except VerificationRefused:
        raise
    except OSError as exc:
        raise VerificationRefused(f"GGUF model could not be parsed: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    alignment = 32
    if "general.alignment" in metadata:
        alignment_entry = metadata["general.alignment"]
        if alignment_entry["type"] != 4:
            raise VerificationRefused("GGUF general.alignment is not UINT32")
        alignment = int(alignment_entry["value"])
        if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise VerificationRefused("GGUF general.alignment is invalid")
    data_start = (header_end + alignment - 1) // alignment * alignment
    if data_start >= file_size or any(
        offset % alignment or data_start + offset >= file_size for offset in offsets
    ):
        raise VerificationRefused("GGUF tensor data offset is invalid")
    required = {
        "general.architecture": (8, GGUF_ARCHITECTURE),
        "general.file_type": (4, GGUF_FILE_TYPE),
        "quantize.imatrix.file": (8, IMATRIX_NAME),
        "quantize.imatrix.dataset": (8, CALIBRATION_CORPUS_NAME),
        "quantize.imatrix.chunks_count": (4, 128),
    }
    for key, (kind, expected) in required.items():
        if metadata.get(key) != {"type": kind, "value": expected}:
            raise VerificationRefused(f"GGUF metadata {key!r} changed")
    entries = metadata.get("quantize.imatrix.entries_count")
    if (
        not isinstance(entries, Mapping)
        or entries.get("type") != 4
        or isinstance(entries.get("value"), bool)
        or not isinstance(entries.get("value"), int)
        or int(entries["value"]) < 1
    ):
        raise VerificationRefused("GGUF imatrix entry count is not a positive UINT32")
    _rehash_bound_file(bound, maximum_bytes=MAX_MODEL_BYTES)
    return {
        **_identity_projection(identity),
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "architecture": GGUF_ARCHITECTURE,
        "file_type": GGUF_FILE_TYPE,
        "imatrix_entries_count": int(entries["value"]),
    }


def _static_gguf_identity(path: Path) -> dict[str, Any]:
    bound = _open_bound_file(
        Path(path),
        "GGUF model",
        maximum_bytes=MAX_MODEL_BYTES,
        expected_mode=0o600,
    )
    try:
        return _static_gguf_identity_bound(bound)
    finally:
        _close_bound_files(bound)


def _official_tree_digest(model_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(ENTRYPOINT.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model_digest.encode("ascii"))
    digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _expected_load_spec() -> dict[str, Any]:
    return {
        "base_model": BASE_MODEL,
        "entrypoint": ENTRYPOINT,
        "format": "gguf",
        "max_input": {"tokens": MAX_INPUT_TOKENS},
        "preprocessing": {"tokenizer": "tokenizer.json"},
        "quantization": QUANTIZATION,
    }


def _validate_stream(value: Any, label: str) -> None:
    stream = _mapping(value, label)
    _exact_keys(
        stream,
        frozenset({"bytes", "captured_bytes", "captured_digest", "digest", "truncated"}),
        label,
    )
    total = _positive_int(stream.get("bytes"), f"{label} bytes", allow_zero=True)
    captured = _positive_int(
        stream.get("captured_bytes"), f"{label} captured bytes", allow_zero=True
    )
    if captured > total or captured > 1024 * 1024:
        raise VerificationRefused(f"{label} captured byte count is invalid")
    if not _is_digest(stream.get("digest")) or not _is_digest(stream.get("captured_digest")):
        raise VerificationRefused(f"{label} digest is malformed")
    if stream.get("truncated") is not (total > captured):
        raise VerificationRefused(f"{label} truncation flag changed")


def _validate_command_record(
    value: Any,
    expected: Mapping[str, Any],
    environment: Mapping[str, Any],
    converter_python_identity: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    name = str(expected["name"])
    command = _mapping(value, f"{role} {name} command")
    command_fields = {
        "argv",
        "cwd_role",
        "environment",
        "finished_at_unix_ns",
        "name",
        "returncode",
        "started_at_unix_ns",
        "stderr",
        "stdout",
    }
    if name == "convert_f16":
        command_fields.add("launch")
    _exact_keys(command, frozenset(command_fields), f"{role} {name} command")
    expected_role = "private_staging" if role == "primary" else "determinism_replay"
    if (
        command.get("name") != name
        or command.get("argv") != expected.get("argv")
        or command.get("cwd_role") != expected_role
        or command.get("environment") != dict(environment)
        or command.get("returncode") != 0
    ):
        raise VerificationRefused(f"{role} {name} command contract changed")
    started = _positive_int(
        command.get("started_at_unix_ns"), f"{role} {name} start", allow_zero=True
    )
    finished = _positive_int(
        command.get("finished_at_unix_ns"), f"{role} {name} finish", allow_zero=True
    )
    if finished < started:
        raise VerificationRefused(f"{role} {name} timing is reversed")
    _validate_stream(command.get("stdout"), f"{role} {name} stdout")
    _validate_stream(command.get("stderr"), f"{role} {name} stderr")
    if name == "convert_f16":
        launch = _mapping(command.get("launch"), f"{role} converter launch")
        _exact_keys(
            launch,
            frozenset({"executed_object", "method"}),
            f"{role} converter launch",
        )
        executed = _mapping(
            launch.get("executed_object"),
            f"{role} converter executed object",
        )
        validated_executed = _validate_converter_python_identity(
            executed, f"{role} converter executed object"
        )
        if (
            validated_executed != dict(converter_python_identity)
            or launch.get("method") != "proc-self-fd"
        ):
            raise VerificationRefused(f"{role} converter fd-launch identity changed")
    return dict(command)


def _validate_converter_python_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _mapping(value, label)
    _exact_keys(identity, frozenset({"portable", "worker_observation"}), label)
    portable = _mapping(identity.get("portable"), f"{label} portable")
    if dict(portable) != {
        "bytes": 21_333_768,
        "digest": (
            "sha256:96d1b01675f2492922ec6f6ed8445791d2d3231ccae727cda521db30494b751e"
        ),
        "mode": "0o755",
        "path": "/opt/python/bin/python3.11",
    }:
        raise VerificationRefused(f"{label} portable identity changed")
    observation = _mapping(identity.get("worker_observation"), f"{label} worker observation")
    _exact_keys(
        observation,
        frozenset({"ctime_ns", "device", "inode", "mtime_ns"}),
        f"{label} worker observation",
    )
    for field, minimum in (
        ("ctime_ns", 0),
        ("device", 0),
        ("inode", 1),
        ("mtime_ns", 0),
    ):
        _positive_int(
            observation.get(field),
            f"{label} worker observation {field}",
            allow_zero=minimum == 0,
        )
    return {"portable": dict(portable), "worker_observation": dict(observation)}


def _runtime_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": value.get("schema"),
        "root": value.get("root"),
        "build_bin_namespace": value.get("build_bin_namespace"),
        "symlinks": value.get("symlinks"),
        "executables": value.get("executables"),
        "libraries": value.get("libraries"),
    }


def _validate_runtime_contract(value: Any, label: str) -> dict[str, Any]:
    contract = _mapping(value, label)
    _exact_keys(
        contract,
        frozenset(
            {
                "build_bin_namespace",
                "executables",
                "libraries",
                "root",
                "schema",
                "symlinks",
            }
        ),
        label,
    )
    if (
        contract.get("schema") != RUNTIME_LIBRARY_SCHEMA
        or contract.get("root") != "/tmp/llama.cpp"
    ):
        raise VerificationRefused(f"{label} identity changed")
    for collection_name in ("build_bin_namespace", "executables", "libraries", "symlinks"):
        collection = contract.get(collection_name)
        if not isinstance(collection, list) or not collection:
            raise VerificationRefused(f"{label} {collection_name} is empty")
    for item in contract["executables"]:
        entry = _mapping(item, f"{label} executable")
        if not _is_digest(entry.get("digest")) or _positive_int(
            entry.get("bytes"), f"{label} executable bytes"
        ) < 1:
            raise VerificationRefused(f"{label} executable identity is malformed")
    for item in contract["libraries"]:
        entry = _mapping(item, f"{label} library")
        if not _is_digest(entry.get("digest")) or _positive_int(
            entry.get("bytes"), f"{label} library bytes"
        ) < 1:
            raise VerificationRefused(f"{label} library identity is malformed")
    return dict(contract)


def _validate_runtime_receipt(
    value: Any,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    runtime = _mapping(value, label)
    expected_keys = frozenset(
        {
            "build_bin_namespace",
            "directories",
            "executables",
            "libraries",
            "root",
            "schema",
            "symlinks",
        }
    )
    _exact_keys(runtime, expected_keys, label)
    directories = runtime.get("directories")
    if not isinstance(directories, list) or directories != [
        {"mode": "0755", "path": "."},
        {"mode": "0755", "path": "build"},
        {"mode": "0755", "path": "build/bin"},
    ]:
        raise VerificationRefused(f"{label} directory modes changed")
    if _runtime_projection(runtime) != _runtime_projection(expected):
        raise VerificationRefused(f"{label} differs from the pinned runtime contract")
    return dict(runtime)


def _validate_replay(
    value: Any,
    spec: Mapping[str, Any],
    model: Mapping[str, Any],
    tree_digest: str,
    converter_python_identity: Mapping[str, Any],
) -> dict[str, Any]:
    replay = _mapping(value, "determinism replay")
    _exact_keys(
        replay,
        frozenset(
            {
                "artifact_tree_digest",
                "commands",
                "entrypoint_bytes",
                "entrypoint_digest",
                "f16_digest",
                "imatrix_digest",
                "matches_primary",
                "schema",
            }
        ),
        "determinism replay",
    )
    if (
        replay.get("schema") != REPLAY_SCHEMA
        or replay.get("matches_primary") is not True
        or replay.get("entrypoint_bytes") != model["bytes"]
        or replay.get("entrypoint_digest") != model["digest"]
        or replay.get("artifact_tree_digest") != tree_digest
        or not _is_digest(replay.get("f16_digest"))
        or not _is_digest(replay.get("imatrix_digest"))
    ):
        raise VerificationRefused("determinism replay does not bind the primary artifact")
    commands = replay.get("commands")
    expected_commands = spec["expected_internal_commands"]
    if not isinstance(commands, list) or len(commands) != len(expected_commands):
        raise VerificationRefused("determinism replay command count changed")
    for command, expected in zip(commands, expected_commands, strict=True):
        _validate_command_record(
            command,
            expected,
            _mapping(spec["expected_child_environment"], "expected child environment"),
            converter_python_identity,
            role="replay",
        )
    return dict(replay)


def _validate_source_binding(value: Any) -> None:
    source = _mapping(value, "conversion source")
    _exact_keys(
        source,
        frozenset(
            {
                "base_snapshot",
                "corpus_profile",
                "dataset_schema",
                "merged_tree_digest",
                "prepared_dataset",
                "source_corpus",
                "training_metadata_digest",
                "training_metrics_digest",
                "training_schema",
            }
        ),
        "conversion source",
    )
    required = {
        "training_schema": "microtensor.code.training.v4",
        "dataset_schema": "microtensor.code.prepared.v1",
        "corpus_profile": "bigcodebench94",
        "training_metadata_digest": (
            "sha256:1e983beff4f32f574a57352b61c2e4f29d9a4922d59d71b1b722902255a3ef10"
        ),
        "training_metrics_digest": (
            "sha256:1c2947a3bed290d01880698b144331ef4f148368634514ef7de396d90d67169e"
        ),
        "merged_tree_digest": (
            "sha256:5b05fe2ec5c145c5f88c28acfb5ab37a6c724816188a7022284d8581b0d356ee"
        ),
    }
    for field, expected in required.items():
        if source.get(field) != expected:
            raise VerificationRefused(f"conversion source field {field!r} changed")
    corpus = _mapping(source.get("source_corpus"), "conversion source corpus")
    if (
        corpus.get("bytes") != 152_605
        or corpus.get("digest")
        != "sha256:1c37a0e212936bfac8c86f955ad61fd378f58603413b45ece88382d528ace9d5"
        or corpus.get("canonical_digest")
        != "sha256:f126ea986aeeb45eecb3a63e850bbe2f6572c01d24142eed639b2dfbddcea4cd"
        or corpus.get("task_count") != 94
        or not _is_digest(corpus.get("refs_digest"))
    ):
        raise VerificationRefused("conversion source corpus identity changed")
    prepared = _mapping(source.get("prepared_dataset"), "conversion prepared dataset")
    required_prepared = {
        "manifest_digest": (
            "sha256:7c51718bf4728284d8fd131c16cc2f9845c6b74d4c9de71d012d4f28e71a51a2"
        ),
        "train_digest": (
            "sha256:927670027ab9a456187ebfd9779f7057e626f7eb16fc99f24e16e45d1a8e7769"
        ),
        "holdout_digest": (
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "train_examples": 94,
        "holdout_examples": 0,
    }
    if dict(prepared) != required_prepared:
        raise VerificationRefused("conversion prepared-dataset identity changed")
    base = _mapping(source.get("base_snapshot"), "conversion base snapshot")
    files = _mapping(base.get("files"), "conversion base files")
    weight = _mapping(files.get("model.safetensors"), "conversion base model weight")
    if (
        base.get("base_model") != BASE_MODEL
        or base.get("required_bytes") != 3_098_955_668
        or weight.get("bytes") != 3_087_467_144
        or weight.get("sha256")
        != "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"
    ):
        raise VerificationRefused("conversion base snapshot identity changed")


def _validate_conversion_receipts(
    conversion: Mapping[str, Any],
    calibration: Mapping[str, Any],
    load_spec: Mapping[str, Any],
    spec: Mapping[str, Any],
    model: Mapping[str, Any],
    tree_digest: str,
    calibration_raw_digest: str,
) -> None:
    expected_load = _expected_load_spec()
    if dict(load_spec) != expected_load:
        raise VerificationRefused("load specification is not exact qwen2/Q4_K_M/541")
    _exact_keys(
        conversion,
        frozenset(
            {
                "artifact",
                "base_model",
                "calibration_receipt_digest",
                "conversion",
                "hardware_class",
                "llama_cpp_revision",
                "load_manifest",
                "schema",
                "source",
                "status",
                "track",
            }
        ),
        "conversion receipt",
    )
    if (
        conversion.get("schema") != CONVERSION_SCHEMA
        or conversion.get("status") != "complete"
        or conversion.get("track") != "code"
        or conversion.get("hardware_class") != "mt-3g"
        or conversion.get("base_model") != BASE_MODEL
        or conversion.get("llama_cpp_revision") != LLAMA_CPP_REVISION
        or conversion.get("load_manifest") != expected_load
        or conversion.get("calibration_receipt_digest") != calibration_raw_digest
    ):
        raise VerificationRefused("conversion receipt identity or status changed")
    _validate_source_binding(conversion.get("source"))
    conversion_artifact = _mapping(conversion.get("artifact"), "conversion artifact")
    expected_conversion_artifact = {
        "entrypoint_bytes": model["bytes"],
        "entrypoint_digest": model["digest"],
        "quantization": QUANTIZATION,
        "tree_digest": tree_digest,
    }
    if dict(conversion_artifact) != expected_conversion_artifact:
        raise VerificationRefused("conversion receipt artifact identity changed")
    conversion_body = _mapping(conversion.get("conversion"), "conversion body")
    _exact_keys(
        conversion_body,
        frozenset(
            {
                "commands",
                "converter_digest",
                "converter_python",
                "determinism_replay",
                "imatrix_digest",
                "quantizer_digest",
                "runtime_libraries",
            }
        ),
        "conversion body",
    )
    expected_tool_digests = {
        "converter_digest": (
            "sha256:e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f"
        ),
        "imatrix_digest": (
            "sha256:3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35"
        ),
        "quantizer_digest": (
            "sha256:e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580"
        ),
    }
    for field, expected in expected_tool_digests.items():
        if conversion_body.get(field) != expected:
            raise VerificationRefused(f"conversion tool digest {field!r} changed")
    converter_python_identity = _validate_converter_python_identity(
        conversion_body.get("converter_python"), "conversion converter Python"
    )
    runtime_contract = _mapping(spec.get("runtime_library_contract"), "runtime contract")
    conversion_runtime = _validate_runtime_receipt(
        conversion_body.get("runtime_libraries"), runtime_contract, "conversion runtime libraries"
    )
    commands = conversion_body.get("commands")
    expected_commands = spec["expected_internal_commands"]
    if not isinstance(commands, list) or len(commands) != len(expected_commands):
        raise VerificationRefused("conversion command count changed")
    primary_commands = [
        _validate_command_record(
            command,
            expected,
            _mapping(spec["expected_child_environment"], "expected child environment"),
            converter_python_identity,
            role="primary",
        )
        for command, expected in zip(commands, expected_commands, strict=True)
    ]
    replay = _validate_replay(
        conversion_body.get("determinism_replay"),
        spec,
        model,
        tree_digest,
        converter_python_identity,
    )

    _exact_keys(
        calibration,
        frozenset(
            {
                "artifact",
                "base_model",
                "commands",
                "determinism_replay",
                "hardware_class",
                "intermediate",
                "llama_cpp_revision",
                "load_manifest",
                "profile",
                "rendering",
                "schema",
                "selection",
                "source",
                "status",
                "toolchain",
                "track",
            }
        ),
        "calibration receipt",
    )
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("status") != "complete"
        or calibration.get("profile") != CALIBRATION_PROFILE
        or calibration.get("track") != "code"
        or calibration.get("hardware_class") != "mt-3g"
        or calibration.get("base_model") != BASE_MODEL
        or calibration.get("llama_cpp_revision") != LLAMA_CPP_REVISION
        or calibration.get("load_manifest") != expected_load
        or calibration.get("commands") != primary_commands
        or calibration.get("determinism_replay") != replay
    ):
        raise VerificationRefused("calibration receipt identity or cross-binding changed")
    selection = _mapping(calibration.get("selection"), "calibration selection")
    selection_required = {
        "algorithm": "sha256-seed-ref-ascending-v1",
        "seed": 92,
        "current_rows": 78,
        "diagnostic_rows_excluded": 16,
        "auxiliary_pool_rows": 7_730,
        "auxiliary_selected_rows": 434,
        "total_rows": 512,
    }
    for field, expected in selection_required.items():
        if selection.get(field) != expected:
            raise VerificationRefused(f"calibration selection field {field!r} changed")
    for field in (
        "current_refs_digest",
        "diagnostic_refs_digest",
        "auxiliary_selected_refs_digest",
    ):
        if not _is_digest(selection.get(field)):
            raise VerificationRefused(f"calibration selection {field!r} is malformed")
    rendering = _mapping(calibration.get("rendering"), "calibration rendering")
    corpus = _mapping(rendering.get("corpus"), "rendered calibration corpus")
    if (
        rendering.get("schema") != "prompt-completion-im-end-utf8-v1"
        or rendering.get("encoding") != "UTF-8"
        or rendering.get("expression") != "prompt + completion + <|im_end|> + LF"
        or rendering.get("eos_token") != "<|im_end|>"
        or rendering.get("eos_token_id") != 151_645
        or rendering.get("rows") != 512
        or corpus.get("name") != CALIBRATION_CORPUS_NAME
        or not _is_digest(corpus.get("digest"))
        or _positive_int(corpus.get("bytes"), "rendered calibration corpus bytes")
        > 16 * 1024 * 1024
    ):
        raise VerificationRefused("calibration rendering contract changed")
    toolchain = _mapping(calibration.get("toolchain"), "calibration toolchain")
    _exact_keys(
        toolchain,
        frozenset(
            {
                "converter_digest",
                "converter_python",
                "imatrix_digest",
                "quantizer_digest",
                "runtime_libraries",
            }
        ),
        "calibration toolchain",
    )
    for field, expected in expected_tool_digests.items():
        calibration_field = field.removesuffix("_digest") + "_digest"
        if toolchain.get(calibration_field) != expected:
            raise VerificationRefused(f"calibration tool digest {calibration_field!r} changed")
    calibration_converter_python = _validate_converter_python_identity(
        toolchain.get("converter_python"), "calibration converter Python"
    )
    if calibration_converter_python != converter_python_identity:
        raise VerificationRefused(
            "conversion and calibration converter Python observations differ"
        )
    calibration_runtime = _validate_runtime_receipt(
        toolchain.get("runtime_libraries"), runtime_contract, "calibration runtime libraries"
    )
    if calibration_runtime != conversion_runtime:
        raise VerificationRefused("conversion and calibration runtime closures differ")
    calibration_artifact = _mapping(calibration.get("artifact"), "calibration artifact")
    metadata = _mapping(
        calibration_artifact.get("calibration_metadata"), "calibrated model metadata"
    )
    expected_metadata = {
        "imatrix_chunks_count": 128,
        "imatrix_dataset": CALIBRATION_CORPUS_NAME,
        "imatrix_entries_count": model["imatrix_entries_count"],
        "imatrix_file": IMATRIX_NAME,
    }
    if metadata != expected_metadata:
        raise VerificationRefused("calibrated model receipt metadata differs from GGUF")
    if {
        key: calibration_artifact.get(key)
        for key in ("entrypoint_bytes", "entrypoint_digest", "quantization", "tree_digest")
    } != expected_conversion_artifact:
        raise VerificationRefused("calibration receipt artifact identity changed")
    intermediate = _mapping(calibration.get("intermediate"), "calibration intermediates")
    f16 = _mapping(intermediate.get("f16"), "calibration F16 identity")
    imatrix = _mapping(intermediate.get("imatrix"), "calibration imatrix identity")
    if (
        f16.get("file_type") != 1
        or not _is_digest(f16.get("digest"))
        or _positive_int(f16.get("bytes"), "calibration F16 bytes") < 1
        or imatrix.get("entries_count") != model["imatrix_entries_count"]
        or imatrix.get("chunk_count") != 128
        or imatrix.get("chunk_size") != 512
        or imatrix.get("datasets") != [CALIBRATION_CORPUS_NAME]
        or not _is_digest(imatrix.get("digest"))
        or _positive_int(imatrix.get("bytes"), "calibration imatrix bytes") < 1
    ):
        raise VerificationRefused("calibration intermediate identities changed")


def _validate_worker_receipt(
    receipt: Mapping[str, Any],
    spec: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    spec_identity: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    input_summary: Mapping[str, Any],
    export_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    _reject_placeholders(receipt)
    _exact_keys(
        receipt,
        frozenset(
            {
                "execution",
                "image",
                "input_manifest",
                "inputs",
                "oci_config",
                "output",
                "runner_preflight_evidence",
                "schema",
                "security_profiles",
                "signature",
                "status",
                "worker_spec",
            }
        ),
        "worker receipt",
    )
    if receipt.get("schema") != WORKER_RECEIPT_SCHEMA or receipt.get("status") != "complete":
        raise VerificationRefused("worker receipt is not complete")
    if receipt.get("worker_spec") != {
        "bytes": spec_identity["bytes"],
        "digest": spec_identity["digest"],
        "schema": SPEC_SCHEMA,
    }:
        raise VerificationRefused("worker receipt does not bind the exact worker spec")
    if receipt.get("input_manifest") != {
        "bytes": manifest_identity["bytes"],
        "digest": manifest_identity["digest"],
        "schema": INPUT_MANIFEST_SCHEMA,
    }:
        raise VerificationRefused("worker receipt does not bind the exact input manifest")
    image = _mapping(spec.get("image"), "worker image")
    if receipt.get("image") != {
        "digest": image["digest"],
        "reference": image["reference"],
        "sbom": image["sbom"],
        "source_closure": image["source_closure"],
    }:
        raise VerificationRefused("worker receipt image/SBOM binding changed")
    if receipt.get("oci_config") != spec.get("oci_config_identity"):
        raise VerificationRefused("worker receipt OCI-config binding changed")
    if receipt.get("runner_preflight_evidence") != spec.get(
        "runner_preflight_evidence"
    ):
        raise VerificationRefused("worker receipt runner-preflight binding changed")
    profiles = _mapping(
        _mapping(spec.get("security"), "worker security").get("profiles"),
        "worker profiles",
    )
    if receipt.get("security_profiles") != profiles:
        raise VerificationRefused("worker receipt security-profile binding changed")
    signature_policy = _mapping(spec.get("receipt_signature"), "signature policy")
    if receipt.get("signature") != {
        "detached": True,
        "key_id": signature_policy["key_id"],
        "message_file": "worker-receipt.json",
        "scheme": signature_policy["scheme"],
        "signature_file": "worker-receipt.sig",
    }:
        raise VerificationRefused("worker receipt signature declaration changed")
    execution = _mapping(receipt.get("execution"), "worker execution")
    required_execution = {
        "capabilities": [],
        "command": spec["command"],
        "container_removed_before_receipt": True,
        "container_stopped_before_export": True,
        "exit_code": 0,
        "export_started_after_container_stop": True,
        "mounts": spec["mounts"],
        "network_mode": "none",
        "no_new_privileges": True,
        "oom_killed": False,
        "platform": spec["platform"],
        "private_namespaces": ["cgroup", "ipc", "mount", "network", "pid", "user", "uts"],
        "resources": spec["resources"],
        "root_filesystem_read_only": True,
        "runtime_mode": "rootless_or_userns_remapped",
        "timed_out": False,
    }
    for field, expected in required_execution.items():
        if execution.get(field) != expected:
            raise VerificationRefused(f"worker execution field {field!r} changed")
    cgroup = _mapping(execution.get("cgroup"), "worker execution cgroup")
    if (
        cgroup.get("limits") != spec.get("cgroup")
        or cgroup.get("empty_before_export") is not True
        or cgroup.get("remaining_processes") != 0
    ):
        raise VerificationRefused("worker cgroup was not empty before export")
    inputs = _mapping(receipt.get("inputs"), "worker input verification")
    if (
        inputs.get("aggregate_digest") != input_summary["aggregate_digest"]
        or inputs.get("snapshot_immutable") is not True
        or inputs.get("mounts_read_only") is not True
    ):
        raise VerificationRefused("worker input snapshot binding changed")
    for phase in ("preflight", "postflight"):
        evidence = _mapping(inputs.get(phase), f"worker input {phase}")
        if evidence != {
            "aggregate_digest": input_summary["aggregate_digest"],
            "files_verified": input_summary["file_count"],
            "verified": True,
        }:
            raise VerificationRefused(f"worker input {phase} did not verify every file")
    output = _mapping(receipt.get("output"), "worker output")
    if (
        output.get("exact_file_set") is not True
        or output.get("private_intermediates_absent") is not True
    ):
        raise VerificationRefused("worker did not attest the exact public output set")
    files = output.get("files")
    if not isinstance(files, list):
        raise VerificationRefused("worker output identities are absent")
    expected_files = [
        {
            "bytes": export_identities[path]["bytes"],
            "digest": export_identities[path]["digest"],
            "path": path,
        }
        for path in sorted(BUNDLE_FILES)
    ]
    if files != expected_files:
        raise VerificationRefused("worker output identities differ from exported bytes")
    expected_output = _mapping(spec.get("expected_output"), "expected output")
    if expected_output.get("file_identities") != expected_files:
        raise VerificationRefused("export differs from independently reviewed output identities")


def _validate_runner_preflight_evidence(
    evidence: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    _reject_placeholders(evidence)
    _exact_keys(
        evidence,
        frozenset(
            {
                "cgroup",
                "command",
                "expected_child_environment",
                "image",
                "independent_review",
                "mounts",
                "oci_config",
                "platform",
                "resources",
                "runtime",
                "schema",
                "security",
                "status",
            }
        ),
        "runner preflight evidence",
    )
    if evidence.get("schema") != RUNNER_PREFLIGHT_SCHEMA:
        raise VerificationRefused("runner preflight evidence schema changed")
    if evidence.get("status") != "accepted":
        raise VerificationRefused("external runner preflight was not accepted")
    _validate_review(
        evidence.get("independent_review"), "runner preflight independent review"
    )
    runtime = _mapping(evidence.get("runtime"), "runner preflight runtime")
    _exact_keys(
        runtime,
        frozenset(
            {
                "cgroup_v2_delegated_writable",
                "host_effective_uid",
                "name",
                "network_namespace_creation",
                "rootless_or_userns_remapped",
                "version",
            }
        ),
        "runner preflight runtime",
    )
    if (
        not isinstance(runtime.get("name"), str)
        or not runtime["name"].strip()
        or not isinstance(runtime.get("version"), str)
        or not runtime["version"].strip()
        or _positive_int(
            runtime.get("host_effective_uid"), "runner host effective UID"
        )
        < 1
        or runtime.get("rootless_or_userns_remapped") is not True
        or runtime.get("cgroup_v2_delegated_writable") is not True
        or runtime.get("network_namespace_creation") is not True
    ):
        raise VerificationRefused("runner preflight runtime capabilities changed")
    bindings = {
        "platform": "platform",
        "resources": "resources",
        "cgroup": "cgroup",
        "security": "security",
        "mounts": "mounts",
        "image": "image",
        "oci_config": "oci_config_identity",
        "command": "command",
        "expected_child_environment": "expected_child_environment",
    }
    for evidence_field, spec_field in bindings.items():
        if evidence.get(evidence_field) != spec.get(spec_field):
            raise VerificationRefused(
                f"runner preflight field {evidence_field!r} is not spec-bound"
            )


def _preflight_context(
    worker_spec_path: Path,
    input_manifest_path: Path,
    trusted_public_key: Path,
    signature_verifier_path: Path,
    runner_preflight_evidence_path: Path,
) -> dict[str, Any]:
    spec_path = Path(worker_spec_path)
    manifest_path = Path(input_manifest_path)
    key_path = Path(trusted_public_key)
    verifier_path = Path(signature_verifier_path)
    evidence_path = Path(runner_preflight_evidence_path)

    # Deliberately reject template placeholders before enforcing finalized-copy modes.
    initial_spec = _strict_json(
        spec_path,
        "worker spec",
        maximum_bytes=MAX_JSON_BYTES,
        allow_writable=True,
    )
    _validate_worker_spec(initial_spec)
    initial_manifest = _strict_json(
        manifest_path,
        "input manifest",
        maximum_bytes=MAX_JSON_BYTES,
        allow_writable=True,
    )
    _validate_input_manifest(initial_manifest)
    expected_uid = _verification_euid()
    trusted_paths = (
        spec_path,
        manifest_path,
        key_path,
        verifier_path,
        evidence_path,
    )
    if (
        len({path.parent for path in trusted_paths}) != 1
        or len({path.name for path in trusted_paths}) != len(trusted_paths)
    ):
        raise VerificationRefused(
            "all finalized trust files must have unique names in one private parent"
        )

    bound_files = _open_bound_group(
        (spec_path, "worker spec", MAX_JSON_BYTES, 0o600),
        (manifest_path, "input manifest", MAX_JSON_BYTES, 0o600),
        (key_path, "trusted public key", 1024 * 1024, 0o600),
        (
            verifier_path,
            "offline signature verifier",
            64 * 1024 * 1024,
            0o500,
        ),
        (evidence_path, "runner preflight evidence", MAX_JSON_BYTES, 0o600),
        expected_uid=expected_uid,
    )
    (
        spec_bound,
        manifest_bound,
        key_bound,
        verifier_bound,
        evidence_bound,
    ) = bound_files
    try:
        spec = _strict_bound_json(spec_bound, "worker spec", maximum_bytes=MAX_JSON_BYTES)
        manifest = _strict_bound_json(
            manifest_bound, "input manifest", maximum_bytes=MAX_JSON_BYTES
        )
        evidence = _strict_bound_json(
            evidence_bound,
            "runner preflight evidence",
            maximum_bytes=MAX_JSON_BYTES,
        )
        if spec != initial_spec or manifest != initial_manifest:
            raise VerificationRefused("finalized spec/manifest changed during preflight")
        _validate_worker_spec(spec)
        input_summary = _validate_input_manifest(manifest)
        if _identity_projection(manifest_bound.identity) != spec.get(
            "input_manifest_identity"
        ):
            raise VerificationRefused("worker spec does not pin the supplied input manifest")
        signature_policy = _mapping(spec.get("receipt_signature"), "signature policy")
        expected_key = _validate_named_identity(
            signature_policy.get("trusted_public_key"), "trusted receipt public key"
        )
        if key_path.name != expected_key["name"] or _identity_projection(
            key_bound.identity
        ) != _identity_projection(expected_key):
            raise VerificationRefused("trusted receipt public key identity changed")
        expected_verifier = _validate_signature_verifier_policy(
            signature_policy.get("verifier"), "receipt signature verifier"
        )
        actual_verifier = {
            "name": verifier_path.name,
            **_identity_projection(verifier_bound.identity),
            "format": "static-elf-linux-amd64",
            "closure": "single-file-no-pt-interp-no-dynamic",
        }
        if actual_verifier != expected_verifier:
            raise VerificationRefused("offline signature-verifier identity changed")
        _validate_static_verifier_elf(verifier_bound)
        expected_evidence = _validate_named_identity(
            spec.get("runner_preflight_evidence"), "runner preflight evidence"
        )
        if evidence_path.name != expected_evidence["name"] or _identity_projection(
            evidence_bound.identity
        ) != _identity_projection(expected_evidence):
            raise VerificationRefused("runner preflight evidence identity changed")
        _validate_runner_preflight_evidence(evidence, spec)
        for bound in bound_files:
            _recheck_bound_file(bound)
        return {
            "spec": spec,
            "manifest": manifest,
            "evidence": evidence,
            "input_summary": input_summary,
            "spec_identity": dict(spec_bound.identity),
            "manifest_identity": dict(manifest_bound.identity),
            "key_identity": dict(key_bound.identity),
            "verifier_identity": dict(verifier_bound.identity),
            "evidence_identity": dict(evidence_bound.identity),
            "expected_uid": expected_uid,
        }
    finally:
        _close_bound_files(*bound_files)


def preflight_only(
    worker_spec_path: Path,
    input_manifest_path: Path,
    trusted_public_key: Path,
    signature_verifier_path: Path,
    runner_preflight_evidence_path: Path,
) -> dict[str, Any]:
    """Validate finalized contracts and trusted identities without touching an export."""

    context = _preflight_context(
        worker_spec_path,
        input_manifest_path,
        trusted_public_key,
        signature_verifier_path,
        runner_preflight_evidence_path,
    )
    return {
        "schema": "microtensor.code.oci-worker-preflight-verification.v1",
        "status": "accepted",
        "input_aggregate_digest": context["input_summary"]["aggregate_digest"],
        "worker_spec_digest": context["spec_identity"]["digest"],
        "runner_preflight_evidence_digest": context["evidence_identity"]["digest"],
        "signature_verifier_digest": context["verifier_identity"]["digest"],
    }


def verify_export(
    export_root: Path,
    worker_spec_path: Path,
    input_manifest_path: Path,
    trusted_public_key: Path,
    signature_verifier_path: Path,
    runner_preflight_evidence_path: Path,
    *,
    signature_verifier: SignatureVerifier,
) -> dict[str, Any]:
    """Verify a sealed export using only inert parsing and an offline signature hook."""

    export_root = Path(export_root)
    spec_path = Path(worker_spec_path)
    manifest_path = Path(input_manifest_path)
    key_path = Path(trusted_public_key)
    verifier_path = Path(signature_verifier_path)
    evidence_path = Path(runner_preflight_evidence_path)
    context = _preflight_context(
        spec_path,
        manifest_path,
        key_path,
        verifier_path,
        evidence_path,
    )
    spec = context["spec"]
    manifest = context["manifest"]
    input_summary = context["input_summary"]
    spec_identity = context["spec_identity"]
    manifest_identity = context["manifest_identity"]
    expected_uid = context["expected_uid"]
    for path, label in (
        (spec_path, "worker spec"),
        (manifest_path, "input manifest"),
        (key_path, "trusted public key"),
        (verifier_path, "offline signature verifier"),
        (evidence_path, "runner preflight evidence"),
    ):
        try:
            path.resolve(strict=True).relative_to(export_root.resolve(strict=True))
        except ValueError:
            pass
        except OSError as exc:
            raise VerificationRefused(f"{label} could not be resolved: {exc}") from exc
        else:
            raise VerificationRefused(f"{label} must remain outside the hostile export")

    export_binding: _ExportBinding | None = None
    trusted_inputs: tuple[_BoundFile, ...] = ()
    try:
        export_binding = _bind_export(export_root, expected_uid)
        trusted_inputs = _open_bound_group(
            (spec_path, "worker spec", MAX_JSON_BYTES, 0o600),
            (manifest_path, "input manifest", MAX_JSON_BYTES, 0o600),
            (key_path, "trusted public key", 1024 * 1024, 0o600),
            (
                verifier_path,
                "offline signature verifier",
                64 * 1024 * 1024,
                0o500,
            ),
            (evidence_path, "runner preflight evidence", MAX_JSON_BYTES, 0o600),
            expected_uid=expected_uid,
        )
        (
            spec_bound,
            manifest_bound,
            key_bound,
            verifier_bound,
            evidence_bound,
        ) = trusted_inputs
        expected_trusted = (
            context["spec_identity"],
            context["manifest_identity"],
            context["key_identity"],
            context["verifier_identity"],
            context["evidence_identity"],
        )
        if any(
            _identity_projection(bound.identity)
            != _identity_projection(expected_identity)
            for bound, expected_identity in zip(
                trusted_inputs, expected_trusted, strict=True
            )
        ):
            raise VerificationRefused("a trusted input changed after preflight")
        held_spec = _strict_bound_json(
            spec_bound, "worker spec", maximum_bytes=MAX_JSON_BYTES
        )
        held_manifest = _strict_bound_json(
            manifest_bound, "input manifest", maximum_bytes=MAX_JSON_BYTES
        )
        held_evidence = _strict_bound_json(
            evidence_bound,
            "runner preflight evidence",
            maximum_bytes=MAX_JSON_BYTES,
        )
        if (
            held_spec != spec
            or held_manifest != manifest
            or held_evidence != context["evidence"]
        ):
            raise VerificationRefused("a trusted JSON contract changed after preflight")
        _validate_worker_spec(held_spec)
        held_input_summary = _validate_input_manifest(held_manifest)
        _validate_runner_preflight_evidence(held_evidence, held_spec)
        if held_input_summary != input_summary:
            raise VerificationRefused("input manifest summary changed after preflight")
        _validate_static_verifier_elf(verifier_bound)
        trusted_ceilings = (
            MAX_JSON_BYTES,
            MAX_JSON_BYTES,
            1024 * 1024,
            64 * 1024 * 1024,
            MAX_JSON_BYTES,
        )
        for bound, ceiling in zip(
            trusted_inputs, trusted_ceilings, strict=True
        ):
            _rehash_bound_file(bound, maximum_bytes=ceiling)
        _recheck_export_binding(export_binding)

        signature_policy = _mapping(
            held_spec.get("receipt_signature"), "signature policy"
        )
        message_bound = export_binding.files["worker-receipt.json"]
        signature_bound = export_binding.files["worker-receipt.sig"]
        if _identity_projection(verifier_bound.identity) != _identity_projection(
            context["verifier_identity"]
        ):
            raise VerificationRefused("signature verifier changed after preflight")
        if _identity_projection(key_bound.identity) != _identity_projection(
            context["key_identity"]
        ):
            raise VerificationRefused("trusted public key changed after preflight")
        try:
            signature_ok = signature_verifier(
                verifier_bound.fd,
                message_bound.fd,
                signature_bound.fd,
                key_bound.fd,
                scheme=str(signature_policy["scheme"]),
                key_id=str(signature_policy["key_id"]),
            )
        except Exception as exc:
            raise VerificationRefused(
                f"offline worker-receipt signature hook failed: {exc}"
            ) from exc
        for bound, ceiling in zip(
            trusted_inputs, trusted_ceilings, strict=True
        ):
            _rehash_bound_file(bound, maximum_bytes=ceiling)
        _recheck_export_binding(export_binding)
        if signature_ok is not True:
            raise VerificationRefused(
                "worker receipt detached signature was not verified"
            )

        export_identities = export_binding.identities
        receipt = _strict_bound_json(
            message_bound,
            "worker receipt",
            maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        )
        load_spec = _strict_bound_json(
            export_binding.files["bundle/load-spec.json"],
            "load specification",
            maximum_bytes=MAX_LOAD_SPEC_BYTES,
        )
        calibration = _strict_bound_json(
            export_binding.files["bundle/calibration-receipt.json"],
            "calibration receipt",
            maximum_bytes=MAX_JSON_BYTES,
        )
        conversion = _strict_bound_json(
            export_binding.files["bundle/conversion-receipt.json"],
            "conversion receipt",
            maximum_bytes=MAX_JSON_BYTES,
        )
        model = _static_gguf_identity_bound(
            export_binding.files["bundle/artifact/model.gguf"]
        )
        tree_digest = _official_tree_digest(str(model["digest"]))
        calibration_digest = export_identities[
            "bundle/calibration-receipt.json"
        ]["digest"]
        _validate_worker_receipt(
            receipt,
            held_spec,
            held_manifest,
            spec_identity,
            manifest_identity,
            held_input_summary,
            export_identities,
        )
        _validate_conversion_receipts(
            conversion,
            calibration,
            load_spec,
            held_spec,
            model,
            tree_digest,
            calibration_digest,
        )
        for bound, ceiling in zip(
            trusted_inputs, trusted_ceilings, strict=True
        ):
            _rehash_bound_file(bound, maximum_bytes=ceiling)
        _recheck_export_binding(export_binding)
        return {
            "schema": "microtensor.code.oci-export-verification.v1",
            "status": "accepted",
            "artifact": {
                "bytes": model["bytes"],
                "digest": model["digest"],
                "tree_digest": tree_digest,
                "architecture": model["architecture"],
                "file_type": model["file_type"],
            },
            "input_aggregate_digest": held_input_summary["aggregate_digest"],
            "worker_receipt_digest": export_identities[
                "worker-receipt.json"
            ]["digest"],
        }
    finally:
        _close_bound_files(*trusted_inputs)
        if export_binding is not None:
            _close_export_binding(export_binding)


class CommandSignatureVerifier:
    """Adapter for one exact, separately attested offline verification executable."""

    def __call__(
        self,
        verifier_fd: int,
        message_fd: int,
        signature_fd: int,
        trusted_public_key_fd: int,
        *,
        scheme: str,
        key_id: str,
    ) -> bool:
        descriptors = (
            verifier_fd,
            message_fd,
            signature_fd,
            trusted_public_key_fd,
        )
        if len(set(descriptors)) != 4 or any(descriptor < 0 for descriptor in descriptors):
            raise VerificationRefused("signature verifier descriptors are invalid")
        verifier = f"/proc/self/fd/{verifier_fd}"
        message = f"/proc/self/fd/{message_fd}"
        signature = f"/proc/self/fd/{signature_fd}"
        trusted_public_key = f"/proc/self/fd/{trusted_public_key_fd}"
        completed = subprocess.run(
            [
                verifier,
                "verify",
                "--key",
                trusted_public_key,
                "--signature",
                signature,
                "--message",
                message,
                "--scheme",
                scheme,
                "--key-id",
                key_id,
            ],
            cwd="/",
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            executable=verifier,
            pass_fds=descriptors,
            shell=False,
            timeout=30,
        )
        if len(completed.stdout) > 64 * 1024 or len(completed.stderr) > 64 * 1024:
            raise VerificationRefused("signature verifier exceeded its output ceiling")
        return completed.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path)
    parser.add_argument("--worker-spec", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--trusted-public-key", type=Path, required=True)
    parser.add_argument("--signature-verifier", type=Path, required=True)
    parser.add_argument("--runner-preflight-evidence", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only:
        if args.export_root is not None:
            raise VerificationRefused("--preflight-only forbids --export-root")
        result = preflight_only(
            args.worker_spec,
            args.input_manifest,
            args.trusted_public_key,
            args.signature_verifier,
            args.runner_preflight_evidence,
        )
    else:
        if args.export_root is None:
            raise VerificationRefused("--export-root is required outside --preflight-only")
        result = verify_export(
            args.export_root,
            args.worker_spec,
            args.input_manifest,
            args.trusted_public_key,
            args.signature_verifier,
            args.runner_preflight_evidence,
            signature_verifier=CommandSignatureVerifier(),
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationRefused as exc:
        raise SystemExit(f"conversion export refused: {exc}") from exc
