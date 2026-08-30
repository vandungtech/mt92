from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import ControllerConfig
from .errors import PreflightError

BINDING_SCHEMA_VERSION = 1
_CHUNK_BYTES = 8 * 1024 * 1024


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if parts in (("manifest.json",), ("artifact.enc",)):
            continue
        if any(part.startswith(".") for part in parts):
            continue
        files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def artifact_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = _artifact_files(root)
    if not files:
        raise PreflightError(f"artifact directory holds no publishable files: {root}")
    total = 0
    for path in files:
        relative = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        file_digest = _digest_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\x00")
        total += path.stat().st_size
    return "sha256:" + digest.hexdigest(), len(files), total


def load_spec(config: ControllerConfig) -> dict[str, Any]:
    return {
        "format": config.artifact_format,
        "quantization": config.quantization,
        "entrypoint": config.entrypoint,
        "max_input": {"tokens": config.max_input_tokens},
        "preprocessing": {"tokenizer": config.tokenizer},
        "base_model": config.base_model,
    }


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def expected_binding(config: ControllerConfig) -> dict[str, Any]:
    artifact, count, total = artifact_digest(config.artifact_dir)
    try:
        selfcheck_digest = _digest_file(config.selfcheck_path)
    except OSError as exc:
        raise PreflightError(f"selfcheck is unreadable: {config.selfcheck_path}: {exc}") from exc
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "artifact_digest": artifact,
        "artifact_file_count": count,
        "artifact_total_bytes": total,
        "load_spec": load_spec(config),
        "load_spec_hash": _canonical_hash(load_spec(config)),
        "selfcheck_sha256": selfcheck_digest,
    }


def _private_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PreflightError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PreflightError(f"{label} must be a regular non-symlink: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PreflightError(f"{label} must have mode 0600: {path}")
    if metadata.st_uid != os.geteuid():
        raise PreflightError(f"{label} must be owned by effective UID {os.geteuid()}: {path}")


def validate_binding(config: ControllerConfig) -> dict[str, Any]:
    path = config.selfcheck_binding_path
    _private_regular_file(path, "selfcheck binding")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"selfcheck binding is invalid: {exc}") from exc
    if not isinstance(observed, dict):
        raise PreflightError("selfcheck binding must be a JSON object")
    expected = expected_binding(config)
    if observed != expected:
        raise PreflightError(
            "selfcheck binding does not match the exact artifact, selfcheck, and GGUF load spec; "
            "rerun the pinned selfcheck and bind-selfcheck"
        )
    return expected


def write_binding(config: ControllerConfig) -> dict[str, Any]:
    payload = expected_binding(config)
    destination = config.selfcheck_binding_path
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        destination.parent.chmod(0o700)
    except OSError:
        pass
    descriptor, temporary = tempfile.mkstemp(prefix=".selfcheck-binding-", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return payload
