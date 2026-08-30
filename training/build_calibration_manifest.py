#!/usr/bin/env python3
"""Build and locally validate one fail-closed GGUF calibration manifest.

Callers supply normalized relative path strings and attest that the pinned
checkout and canonical arguments describe the recipe for the supplied bytes.
The manifest binds those bytes and that attestation; it is not a historical
execution receipt and cannot prove that an already-existing file was produced
by the claimed command. An execution-receipt schema is deliberately deferred
until the publisher can validate one without weakening this byte binding.

The successful no-clobber hard link is the install commit point. Post-commit
integrity, temporary-cleanup, and durability failures are reported in the
committed result instead of falsely reporting that no output exists.
The CLI emits that committed state as JSON and uses exit status 3 when the
installed inode's integrity could not be confirmed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import publish_provenance as provenance
else:  # pragma: no cover - covered by the direct-script help smoke test
    import publish_provenance as provenance


STANDARD_Q4_PROFILE = "q4_k_m"
ATTN_V_Q6_PROFILE = "q4_k_m_attn_v_q6"
STANDARD_Q5_PROFILE = provenance.STANDARD_Q5_PROFILE
QUANTIZATION_PROFILES = (
    STANDARD_Q4_PROFILE,
    ATTN_V_Q6_PROFILE,
    STANDARD_Q5_PROFILE,
)
ATTESTATION_SEMANTICS = (
    "caller_attests_canonical_recipe_and_clean_pinned_checkout;"
    "not_a_historical_execution_receipt"
)
POST_COMMIT_INTEGRITY_EXIT_STATUS = 3


class ManifestBuildError(ValueError):
    """Raised before installing a manifest when local evidence is invalid."""


@dataclass(frozen=True)
class ManifestRequest:
    """Explicit evidence and validation inputs for one manifest."""

    output: Path
    training_dirs: tuple[str, ...]
    source_model_dir: str
    converted_model: str
    corpus: str
    corpus_metadata: str
    imatrix: str
    quantized_artifact: str
    quantization_profile: str
    finished_block: int
    llama_cpp_dir: Path
    weight_soup_checkpoints: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BuiltCalibrationManifest:
    """The explicit integrity and durability state after no-clobber commit."""

    path: Path
    manifest: dict[str, Any]
    manifest_digest: str
    artifact_tree_digest: str
    committed: bool
    installed_device: int
    installed_inode: int
    installed_integrity_confirmed: bool
    temporary_cleanup_complete: bool
    durability_confirmed: bool
    post_commit_warnings: tuple[str, ...]
    attestation_semantics: str = ATTESTATION_SEMANTICS


@dataclass(frozen=True)
class _PreparedManifest:
    manifest: dict[str, Any]
    payload: bytes
    manifest_digest: str
    artifact_tree_digest: str
    destination: Path
    training_dirs: tuple[Path, ...]
    weight_soup_checkpoints: dict[int, Path]


@dataclass(frozen=True)
class _LinkInstallResult:
    """Post-commit integrity state and warnings from descriptor installation."""

    integrity_confirmed: bool
    warnings: tuple[str, ...]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _destination(path: Path) -> tuple[Path, Path]:
    supplied = Path(path)
    if not supplied.name or supplied.name in {".", ".."}:
        raise ManifestBuildError("manifest output must name a file")
    parent = supplied.parent if supplied.is_absolute() else Path.cwd() / supplied.parent
    try:
        details = parent.lstat()
        base = parent.resolve(strict=True)
    except OSError as exc:
        raise ManifestBuildError("manifest output parent is unavailable") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ManifestBuildError("manifest output parent must be a non-symlink directory")
    destination = base / supplied.name
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ManifestBuildError("manifest output cannot be inspected") from exc
    else:
        raise ManifestBuildError("manifest output already exists; refusing to overwrite it")
    return destination, base


def _reject_symlink_components(base: Path, path: Path, label: str) -> None:
    current = base
    for part in path.parts:
        current /= part
        try:
            details = current.lstat()
        except OSError as exc:
            raise ManifestBuildError(
                f"{label} has an unavailable path component: {current.name}"
            ) from exc
        if stat.S_ISLNK(details.st_mode):
            raise ManifestBuildError(
                f"{label} contains a symlink component: {current.name}"
            )


def _relative_path(base: Path, value: object, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestBuildError(f"{label} must be a non-empty, already stripped string")
    path = Path(value)
    if (
        path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or path.as_posix() != value
        or unicodedata.normalize("NFC", value) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ManifestBuildError(
            f"{label} must be a normalized NFC relative path without traversal"
        )
    candidate = base / path
    _reject_symlink_components(base, path, label)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as exc:
        raise ManifestBuildError(f"{label} must resolve inside the manifest directory") from exc
    return value, candidate


def _regular_file(base: Path, value: object, label: str) -> tuple[str, Path]:
    relative, path = _relative_path(base, value, label)
    try:
        details = path.lstat()
    except OSError as exc:
        raise ManifestBuildError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(details.st_mode):
        raise ManifestBuildError(f"{label} must be a non-symlink regular file")
    return relative, path


def _directory(base: Path, value: object, label: str) -> tuple[str, Path]:
    relative, path = _relative_path(base, value, label)
    try:
        details = path.lstat()
    except OSError as exc:
        raise ManifestBuildError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ManifestBuildError(f"{label} must be a non-symlink directory")
    return relative, path


def _stable_file_identity(path: Path, label: str) -> tuple[int, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ManifestBuildError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ManifestBuildError(f"{label} must be a non-symlink regular file")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestBuildError(
            f"{label} could not be opened without symlinks"
        ) from exc
    try:
        during = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ManifestBuildError(f"{label} changed during identity inspection") from exc
    identities = {
        (before.st_dev, before.st_ino),
        (during.st_dev, during.st_ino),
        (after.st_dev, after.st_ino),
    }
    if len(identities) != 1:
        raise ManifestBuildError(f"{label} changed during identity inspection")
    if not stat.S_ISREG(during.st_mode):
        raise ManifestBuildError(f"{label} must be a regular file")
    return during.st_dev, during.st_ino


def _claim(path: Path, relative: str, label: str) -> dict[str, Any]:
    size, digest = provenance._snapshot_regular_file(
        path,
        maximum=provenance.MAX_LINEAGE_ASSET_BYTES,
        label=label,
    )
    return {"path": relative, "bytes": size, "sha256": digest}


def _source_inventory(source_dir: Path) -> list[dict[str, Any]]:
    try:
        entries = sorted(source_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ManifestBuildError("source model directory cannot be inventoried") from exc
    if not entries:
        raise ManifestBuildError("source model directory is empty")
    claims: list[dict[str, Any]] = []
    for entry in entries:
        try:
            details = entry.lstat()
        except OSError as exc:
            raise ManifestBuildError(f"source model entry is unavailable: {entry.name}") from exc
        if not stat.S_ISREG(details.st_mode):
            raise ManifestBuildError(
                f"source model entries must be non-symlink regular files: {entry.name}"
            )
        claims.append(_claim(entry, entry.name, f"source model file {entry.name}"))
    names = {claim["path"] for claim in claims}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(names) or not any(name.endswith(".safetensors") for name in names):
        raise ManifestBuildError("source model inventory is incomplete")
    return claims


def _git_output(repository: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManifestBuildError("llama.cpp Git identity could not be inspected") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ManifestBuildError(f"llama.cpp Git inspection failed{suffix}")
    return completed.stdout.strip()


def _validate_llama_cpp_checkout(repository: Path) -> Path:
    supplied = Path(repository)
    try:
        details = supplied.lstat()
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ManifestBuildError("llama.cpp checkout is unavailable") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ManifestBuildError("llama.cpp checkout must be a non-symlink directory")
    revision = _git_output(resolved, ("rev-parse", "--verify", "HEAD"))
    if revision != provenance.LLAMA_CPP_REVISION:
        raise ManifestBuildError(
            f"llama.cpp revision must be exactly {provenance.LLAMA_CPP_REVISION}"
        )
    status_output = _git_output(
        resolved, ("status", "--porcelain=v1", "--untracked-files=all")
    )
    if status_output:
        raise ManifestBuildError("llama.cpp checkout must have a clean worktree and index")
    return resolved


def _index_soup_checkpoints(
    base: Path, entries: Sequence[Sequence[str]]
) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for entry in entries:
        if len(entry) != 2:
            raise ManifestBuildError("each weight-soup checkpoint requires STAGE and PATH")
        stage_text, path_text = entry
        if not isinstance(stage_text, str):
            raise ManifestBuildError("weight-soup checkpoint stage must be a string")
        try:
            stage_number = int(stage_text)
        except ValueError as exc:
            raise ManifestBuildError(
                f"weight-soup checkpoint stage is not an integer: {stage_text!r}"
            ) from exc
        if stage_number < 1 or str(stage_number) != stage_text:
            raise ManifestBuildError(
                "weight-soup checkpoint stages must be canonical positive integers"
            )
        if stage_number in indexed:
            raise ManifestBuildError(
                f"duplicate weight-soup checkpoint for stage {stage_number}"
            )
        _relative, path = _directory(
            base, path_text, f"weight-soup checkpoint for stage {stage_number}"
        )
        indexed[stage_number] = path
    return indexed


def _prepare(request: ManifestRequest) -> _PreparedManifest:
    destination, base = _destination(request.output)
    if not request.training_dirs:
        raise ManifestBuildError("at least one training directory is required")
    if request.quantization_profile not in QUANTIZATION_PROFILES:
        raise ManifestBuildError(
            f"quantization profile must be one of {QUANTIZATION_PROFILES!r}"
        )
    if (
        isinstance(request.finished_block, bool)
        or not isinstance(request.finished_block, int)
        or request.finished_block < 1
    ):
        raise ManifestBuildError("finished block must be a positive integer")
    _validate_llama_cpp_checkout(request.llama_cpp_dir)

    training_dirs = tuple(
        _directory(base, value, f"training directory {index}")[1]
        for index, value in enumerate(request.training_dirs, start=1)
    )
    soup_checkpoints = _index_soup_checkpoints(
        base, request.weight_soup_checkpoints
    )
    source_relative, source_dir = _directory(
        base, request.source_model_dir, "source model directory"
    )
    converted_relative, converted_path = _regular_file(
        base, request.converted_model, "converted F16 model"
    )
    corpus_relative, corpus_path = _regular_file(
        base, request.corpus, "calibration corpus"
    )
    metadata_relative, metadata_path = _regular_file(
        base, request.corpus_metadata, "calibration corpus metadata"
    )
    imatrix_relative, imatrix_path = _regular_file(
        base, request.imatrix, "calibration imatrix"
    )
    artifact_relative, artifact_path = _regular_file(
        base, request.quantized_artifact, "quantized artifact"
    )

    role_files = (
        (converted_path, "converted F16 model"),
        (corpus_path, "calibration corpus"),
        (metadata_path, "calibration corpus metadata"),
        (imatrix_path, "calibration imatrix"),
        (artifact_path, "quantized artifact"),
    )
    role_identities = {
        _stable_file_identity(path, label) for path, label in role_files
    }
    if len(role_identities) != 5:
        raise ManifestBuildError("calibration roles must reference five distinct files")
    artifact_root = artifact_path.parent.resolve()
    try:
        destination.relative_to(artifact_root)
    except ValueError:
        pass
    else:
        raise ManifestBuildError("manifest output must be outside the artifact tree")

    try:
        provenance._require_gguf(converted_path, "converted F16 model")
        provenance._require_gguf(imatrix_path, "calibration imatrix")
        provenance._require_gguf(artifact_path, "quantized artifact")
    except provenance.ProvenanceValidationError as exc:
        raise ManifestBuildError(str(exc)) from exc

    source_files = _source_inventory(source_dir)
    metadata_file = training_dirs[-1] / "training_metadata.json"
    try:
        training_metadata_size, training_metadata_digest = (
            provenance._snapshot_regular_file(
                metadata_file,
                maximum=provenance.MAX_METADATA_BYTES,
                label="final training metadata",
            )
        )
    except provenance.ProvenanceValidationError as exc:
        raise ManifestBuildError(str(exc)) from exc
    if training_metadata_size < 1:
        raise ManifestBuildError("final training metadata is empty")

    quantization_arguments = ["--imatrix", imatrix_relative]
    if request.quantization_profile == ATTN_V_Q6_PROFILE:
        quantization_arguments.extend(
            ("--tensor-type", provenance.ATTN_V_Q6_OVERRIDE)
        )
    quantization_type = (
        "Q5_K_M"
        if request.quantization_profile == STANDARD_Q5_PROFILE
        else "Q4_K_M"
    )
    quantization_arguments.extend(
        (converted_relative, artifact_relative, quantization_type)
    )
    try:
        artifact_tree_digest = provenance._artifact_tree_digest(
            artifact_path.parent
        )
    except provenance.ProvenanceValidationError as exc:
        raise ManifestBuildError(str(exc)) from exc

    manifest: dict[str, Any] = {
        "schema": provenance.CALIBRATION_LINEAGE_SCHEMA,
        "llama_cpp_revision": provenance.LLAMA_CPP_REVISION,
        "artifact_tree_digest": artifact_tree_digest,
        "source_model": {
            "directory": source_relative,
            "training_metadata_sha256": training_metadata_digest,
            "files": source_files,
        },
        "conversion": {
            "tool": "convert_hf_to_gguf.py",
            "arguments": [
                source_relative,
                "--outfile",
                converted_relative,
                "--outtype",
                "f16",
            ],
            "outtype": "f16",
            "output": _claim(
                converted_path, converted_relative, "converted F16 model"
            ),
        },
        "calibration": {
            "tool": "llama-imatrix",
            "arguments": [
                "--offline",
                "--model",
                converted_relative,
                "--file",
                corpus_relative,
                "--output",
                imatrix_relative,
                "--ctx-size",
                "512",
                "--chunks",
                "-1",
                "--no-ppl",
                "--parse-special",
            ],
            "corpus": _claim(
                corpus_path, corpus_relative, "calibration corpus"
            ),
            "metadata": _claim(
                metadata_path,
                metadata_relative,
                "calibration corpus metadata",
            ),
            "imatrix": _claim(
                imatrix_path, imatrix_relative, "calibration imatrix"
            ),
            "settings": {
                "offline": True,
                "ctx_size": 512,
                "chunks": -1,
                "no_ppl": True,
                "process_output": False,
                "parse_special": True,
                "output_format": "gguf",
            },
        },
        "quantization": {
            "tool": "llama-quantize",
            "arguments": quantization_arguments,
            "output": _claim(
                artifact_path, artifact_relative, "quantized artifact"
            ),
        },
    }
    if request.quantization_profile == STANDARD_Q5_PROFILE:
        manifest["quantization"]["profile"] = STANDARD_Q5_PROFILE
    payload = _json_bytes(manifest)
    if len(payload) > provenance.MAX_METADATA_BYTES:
        raise ManifestBuildError("generated calibration manifest is too large")
    return _PreparedManifest(
        manifest=manifest,
        payload=payload,
        manifest_digest=_digest_bytes(payload),
        artifact_tree_digest=artifact_tree_digest,
        destination=destination,
        training_dirs=training_dirs,
        weight_soup_checkpoints=soup_checkpoints,
    )


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest_payload(descriptor: int, payload: bytes) -> None:
    offset = 0
    try:
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("zero-byte write")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise ManifestBuildError("temporary manifest could not be written") from exc


def _manifest_stat_fingerprint(
    details: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _verify_bound_manifest(
    descriptor: int,
    temporary: Path,
    payload: bytes,
    identity: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    try:
        before = os.fstat(descriptor)
        named_before = temporary.lstat()
    except OSError as exc:
        raise ManifestBuildError(
            "temporary manifest identity could not be verified"
        ) from exc
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(named_before.st_mode):
        raise ManifestBuildError("temporary manifest is no longer a regular file")
    if (
        (before.st_dev, before.st_ino) != identity
        or (named_before.st_dev, named_before.st_ino) != identity
    ):
        raise ManifestBuildError(
            "temporary manifest path no longer names the created inode"
        )
    if before.st_size != len(payload) or named_before.st_size != len(payload):
        raise ManifestBuildError("temporary manifest size changed after validation")

    captured = bytearray()
    offset = 0
    try:
        while offset < len(payload):
            chunk = os.pread(
                descriptor,
                min(128 * 1024, len(payload) - offset),
                offset,
            )
            if not chunk:
                raise OSError("unexpected end of file")
            captured.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        named_after = temporary.lstat()
    except OSError as exc:
        raise ManifestBuildError(
            "temporary manifest content could not be verified"
        ) from exc
    if bytes(captured) != payload:
        raise ManifestBuildError("temporary manifest content changed after validation")
    fingerprint = _manifest_stat_fingerprint(before)
    if (
        _manifest_stat_fingerprint(after) != fingerprint
        or _manifest_stat_fingerprint(named_before) != fingerprint
        or _manifest_stat_fingerprint(named_after) != fingerprint
    ):
        raise ManifestBuildError("temporary manifest changed during final verification")
    return fingerprint


def _verify_installed_manifest(
    descriptor: int,
    destination_directory: int,
    destination_name: str,
    identity: tuple[int, int],
    expected_size: int,
    expected_digest: str,
) -> None:
    """Verify the committed directory entry and exact bytes through held fds."""

    try:
        before = os.fstat(descriptor)
        named_before = os.stat(
            destination_name,
            dir_fd=destination_directory,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ManifestBuildError(
            "installed manifest identity could not be verified"
        ) from exc
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(named_before.st_mode):
        raise ManifestBuildError("installed manifest is not a regular file")
    if (
        (before.st_dev, before.st_ino) != identity
        or (named_before.st_dev, named_before.st_ino) != identity
        or before.st_size != expected_size
        or named_before.st_size != expected_size
    ):
        raise ManifestBuildError("installed manifest does not name the held inode")

    digest = hashlib.sha256()
    offset = 0
    try:
        while offset < expected_size:
            chunk = os.pread(
                descriptor,
                min(128 * 1024, expected_size - offset),
                offset,
            )
            if not chunk:
                raise OSError("unexpected end of file")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(
            destination_name,
            dir_fd=destination_directory,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ManifestBuildError(
            "installed manifest content could not be verified"
        ) from exc
    if "sha256:" + digest.hexdigest() != expected_digest:
        raise ManifestBuildError("installed manifest digest does not match validated bytes")
    fingerprint = _manifest_stat_fingerprint(before)
    if (
        _manifest_stat_fingerprint(after) != fingerprint
        or _manifest_stat_fingerprint(named_before) != fingerprint
        or _manifest_stat_fingerprint(named_after) != fingerprint
    ):
        raise ManifestBuildError("installed manifest changed during verification")


def _link_bound_descriptor(
    descriptor: int,
    destination: Path,
    identity: tuple[int, int],
    expected_size: int,
    expected_fingerprint: tuple[int, int, int, int, int, int],
    expected_digest: str,
) -> _LinkInstallResult:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    proc_directory = -1
    destination_directory = -1
    committed = False
    integrity_confirmed = False
    warnings: list[str] = []
    try:
        try:
            proc_directory = os.open("/proc/self/fd", directory_flags)
            destination_directory = os.open(destination.parent, directory_flags)
            details = os.fstat(descriptor)
            proc_details = os.stat(
                str(descriptor),
                dir_fd=proc_directory,
                follow_symlinks=True,
            )
            if (
                _manifest_stat_fingerprint(details) != expected_fingerprint
                or _manifest_stat_fingerprint(proc_details) != expected_fingerprint
                or (details.st_dev, details.st_ino) != identity
                or details.st_size != expected_size
            ):
                raise ManifestBuildError(
                    "held temporary manifest changed after final verification"
                )
            os.link(
                str(descriptor),
                destination.name,
                src_dir_fd=proc_directory,
                dst_dir_fd=destination_directory,
                follow_symlinks=True,
            )
            committed = True
        except FileExistsError as exc:
            raise ManifestBuildError(
                "manifest output appeared during validation; refusing to overwrite it"
            ) from exc
        except OSError as exc:
            raise ManifestBuildError(
                "manifest could not be installed from its held descriptor"
            ) from exc

        try:
            _verify_installed_manifest(
                descriptor,
                destination_directory,
                destination.name,
                identity,
                expected_size,
                expected_digest,
            )
        except ManifestBuildError as exc:
            warnings.append(
                f"installed manifest integrity verification failed after commit: {exc}"
            )
        else:
            integrity_confirmed = True
    finally:
        for label, opened in (
            ("proc descriptor directory", proc_directory),
            ("manifest output directory", destination_directory),
        ):
            if opened < 0:
                continue
            try:
                os.close(opened)
            except OSError as exc:
                if committed:
                    warnings.append(f"{label} close failed after commit: {exc}")
    return _LinkInstallResult(
        integrity_confirmed=integrity_confirmed,
        warnings=tuple(warnings),
    )


def _unlink_bound_path(
    path: Path,
    identity: tuple[int, int],
) -> tuple[bool, str | None]:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return True, None
    except OSError as exc:
        return False, f"temporary manifest could not be inspected: {exc}"
    if (details.st_dev, details.st_ino) != identity:
        return (
            False,
            "temporary manifest path changed identity and was left untouched",
        )
    try:
        path.unlink()
    except FileNotFoundError:
        return True, None
    except OSError as exc:
        return False, f"temporary manifest cleanup failed: {exc}"
    return True, None


def _close_descriptor(descriptor: int) -> str | None:
    try:
        os.close(descriptor)
    except OSError as exc:
        return f"temporary manifest descriptor close failed: {exc}"
    return None


def build_calibration_manifest(
    request: ManifestRequest,
) -> BuiltCalibrationManifest:
    """Validate and install a manifest; hard-link success is the commit point."""

    prepared = _prepare(request)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{prepared.destination.name}.",
        suffix=".tmp",
        dir=prepared.destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        created = os.fstat(descriptor)
    except OSError as exc:
        _close_descriptor(descriptor)
        raise ManifestBuildError(
            "temporary manifest descriptor could not be inspected"
        ) from exc
    identity = (created.st_dev, created.st_ino)
    try:
        if not stat.S_ISREG(created.st_mode):
            raise ManifestBuildError("temporary manifest is not a regular file")
        try:
            os.fchmod(descriptor, 0o644)
        except OSError as exc:
            raise ManifestBuildError(
                "temporary manifest permissions could not be fixed"
            ) from exc
        _write_manifest_payload(descriptor, prepared.payload)
        try:
            publication = provenance.validate_publication(
                prepared.training_dirs,
                prepared.artifact_tree_digest,
                request.finished_block,
                calibration_manifest=temporary,
                weight_soup_checkpoints=prepared.weight_soup_checkpoints,
            )
        except provenance.ProvenanceValidationError as exc:
            raise ManifestBuildError(
                f"generated manifest failed publication validation: {exc}"
            ) from exc
        if (
            publication.artifact_digest != prepared.artifact_tree_digest
            or publication.calibration.manifest_digest
            != prepared.manifest_digest
        ):
            raise ManifestBuildError(
                "publication validator did not bind the generated manifest bytes"
            )
        _validate_llama_cpp_checkout(request.llama_cpp_dir)
        verified_fingerprint = _verify_bound_manifest(
            descriptor,
            temporary,
            prepared.payload,
            identity,
        )
        link_result = _link_bound_descriptor(
            descriptor,
            prepared.destination,
            identity,
            len(prepared.payload),
            verified_fingerprint,
            prepared.manifest_digest,
        )
    except BaseException:
        _unlink_bound_path(temporary, identity)
        _close_descriptor(descriptor)
        raise

    # The no-clobber link above is the commit point. Nothing below may turn an
    # installed output into a reported build failure.
    warnings = list(link_result.warnings)
    cleanup_complete, cleanup_warning = _unlink_bound_path(temporary, identity)
    if cleanup_warning is not None:
        warnings.append(cleanup_warning)
    try:
        _fsync_directory(prepared.destination.parent)
    except OSError as exc:
        durability_confirmed = False
        warnings.append(f"manifest directory fsync failed after commit: {exc}")
    else:
        durability_confirmed = True
    close_warning = _close_descriptor(descriptor)
    if close_warning is not None:
        warnings.append(close_warning)

    return BuiltCalibrationManifest(
        path=prepared.destination,
        manifest=prepared.manifest,
        manifest_digest=prepared.manifest_digest,
        artifact_tree_digest=prepared.artifact_tree_digest,
        committed=True,
        installed_device=identity[0],
        installed_inode=identity[1],
        installed_integrity_confirmed=link_result.integrity_confirmed,
        temporary_cleanup_complete=cleanup_complete,
        durability_confirmed=durability_confirmed,
        post_commit_warnings=tuple(warnings),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a byte-bound calibration manifest from a caller-attested "
            "canonical recipe and clean pinned llama.cpp checkout."
        ),
        epilog=(
            "This is not a historical command-execution receipt. Exit status 3 "
            "means the output was committed but installed integrity was not "
            "confirmed; inspect the emitted JSON before remediation."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-dir", action="append", required=True, metavar="PATH"
    )
    parser.add_argument("--source-model-dir", required=True, metavar="PATH")
    parser.add_argument("--converted-model", required=True, metavar="PATH")
    parser.add_argument("--corpus", required=True, metavar="PATH")
    parser.add_argument("--corpus-metadata", required=True, metavar="PATH")
    parser.add_argument("--imatrix", required=True, metavar="PATH")
    parser.add_argument("--quantized-artifact", required=True, metavar="PATH")
    parser.add_argument(
        "--quantization-profile",
        choices=QUANTIZATION_PROFILES,
        default=STANDARD_Q4_PROFILE,
    )
    parser.add_argument("--finished-block", type=int, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument(
        "--weight-soup-checkpoint",
        action="append",
        nargs=2,
        default=[],
        metavar=("STAGE", "PATH"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = ManifestRequest(
        output=args.output,
        training_dirs=tuple(args.training_dir),
        source_model_dir=args.source_model_dir,
        converted_model=args.converted_model,
        corpus=args.corpus,
        corpus_metadata=args.corpus_metadata,
        imatrix=args.imatrix,
        quantized_artifact=args.quantized_artifact,
        quantization_profile=args.quantization_profile,
        finished_block=args.finished_block,
        llama_cpp_dir=args.llama_cpp_dir,
        weight_soup_checkpoints=tuple(
            tuple(entry) for entry in args.weight_soup_checkpoint
        ),
    )
    try:
        built = build_calibration_manifest(request)
    except (
        ManifestBuildError,
        provenance.ProvenanceValidationError,
    ) as exc:
        raise SystemExit(f"calibration manifest build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "artifact_tree_digest": built.artifact_tree_digest,
                "attestation_semantics": built.attestation_semantics,
                "committed": built.committed,
                "durability_confirmed": built.durability_confirmed,
                "installed_device": built.installed_device,
                "installed_inode": built.installed_inode,
                "installed_integrity_confirmed": (
                    built.installed_integrity_confirmed
                ),
                "manifest_digest": built.manifest_digest,
                "path": str(built.path),
                "post_commit_warnings": list(built.post_commit_warnings),
                "temporary_cleanup_complete": built.temporary_cleanup_complete,
            },
            sort_keys=True,
        )
    )
    if not built.installed_integrity_confirmed:
        return POST_COMMIT_INTEGRITY_EXIT_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
