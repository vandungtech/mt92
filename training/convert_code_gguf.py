#!/usr/bin/env python3
"""Convert one validated historical code run into an atomically published GGUF bundle.

Only the exact completed v5 historical training lineage and a clean, pinned
llama.cpp checkout are accepted.  Conversion happens in a unique directory on
``/dev/shm``.  The final bundle is published with Linux ``RENAME_NOREPLACE``
only after every source, tool, and output identity has been replayed.

This module is offline and never performs model inference.  Its only child
processes are the exact reviewed converter, quantizer, and read-only Git
identity checks.  Child processes receive a deliberately small environment
without credentials, wallet paths, or user configuration.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import evaluate_code_gguf as gguf
    from training import publish_code_provenance as provenance
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import evaluate_code_gguf as gguf  # type: ignore[no-redef]
    import publish_code_provenance as provenance  # type: ignore[no-redef]


SCHEMA: Final[str] = provenance.CONVERSION_SCHEMA
LLAMA_CPP_REVISION: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
ENTRYPOINT: Final[str] = "model.gguf"
LOAD_SPEC_NAME: Final[str] = "load-spec.json"
RECEIPT_NAME: Final[str] = "conversion-receipt.json"
ARTIFACT_NAME: Final[str] = "artifact"
F16_NAME: Final[str] = "model-f16.gguf"
SUPPORTED_QUANTIZATIONS: Final[frozenset[str]] = frozenset({"Q8_0", "Q4_K_M"})
_AT_FDCWD: Final[int] = -100
_RENAME_NOREPLACE: Final[int] = 1
_STAGING_MARKER: Final[str] = ".microtensor-code-gguf-"
_MAX_ERROR_TEXT_BYTES: Final[int] = 4096

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
    identity = gguf.file_identity(resolved, label)
    return resolved, identity


def _small_child_environment() -> dict[str, str]:
    """Return an offline child environment with no inherited secret values."""

    path = os.environ.get("PATH")
    if not path:
        raise ConversionRefused("PATH is required to resolve the converter's Python shebang")
    return {
        "PATH": path,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "CUDA_VISIBLE_DEVICES": "",
    }


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


def _toolchain_identity(request: ConversionRequest) -> dict[str, Any]:
    root = _regular_directory(request.llama_cpp, "llama.cpp checkout")
    converter, converter_identity = _regular_executable(request.converter, "GGUF converter")
    quantizer, quantizer_identity = _regular_executable(request.quantizer, "GGUF quantizer")
    expected_converter = (root / "convert_hf_to_gguf.py").resolve(strict=True)
    expected_quantizer = (root / "build" / "bin" / "llama-quantize").resolve(strict=True)
    if converter != expected_converter:
        raise ConversionRefused("converter is not the pinned checkout's exact converter path")
    if quantizer != expected_quantizer:
        raise ConversionRefused("quantizer is not the pinned checkout's exact quantizer path")
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
    return {
        "root": str(root),
        "revision": revision,
        "converter": converter_identity,
        "quantizer": quantizer_identity,
    }


def _load_lineage(request: ConversionRequest) -> dict[str, Any]:
    try:
        lineage, _modules = gguf.load_v5_training_lineage(
            request.training_run,
            request.training_dataset,
            request.source_corpus,
            request.base,
        )
    except Exception as exc:
        raise ConversionRefused(f"v5 historical training lineage was refused: {exc}") from exc
    if lineage.get("schema") != gguf.TRAINING_SCHEMA_V5:
        raise ConversionRefused("training lineage is not the exact v5 schema")
    receipt = _mapping(lineage.get("receipt"), "training receipt identity")
    run = _mapping(lineage.get("run"), "training run identity")
    merged = _mapping(run.get("merged"), "merged HF tree identity")
    for value, label in (
        (receipt.get("digest"), "training metadata digest"),
        (merged.get("digest"), "merged HF tree digest"),
    ):
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise ConversionRefused(f"{label} is malformed")
    return lineage


def _conversion_command(
    name: str,
    argv: Sequence[str],
    *,
    cwd: Path,
) -> dict[str, Any]:
    exact_argv = [str(item) for item in argv]
    if not exact_argv or any(not item for item in exact_argv):
        raise ConversionRefused(f"{name} argv is malformed")
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
        raise ConversionRefused("quantization must be exactly Q8_0 or Q4_K_M")
    tokens = request.max_input_tokens
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise ConversionRefused("max-input-tokens must be an integer")
    if not gguf.MIN_CONTEXT_TOKENS <= tokens <= gguf.MAX_CONTEXT_TOKENS:
        raise ConversionRefused(
            f"max-input-tokens must be in [{gguf.MIN_CONTEXT_TOKENS}, {gguf.MAX_CONTEXT_TOKENS}]"
        )
    output = _fresh_output_bundle(request.output_bundle)
    for source, label in (
        (request.training_run, "training run"),
        (request.training_dataset, "training dataset"),
        (request.base, "base snapshot"),
        (request.llama_cpp, "llama.cpp checkout"),
    ):
        try:
            protected = Path(source).resolve(strict=True)
        except OSError as exc:
            raise ConversionRefused(f"{label} is unavailable: {exc}") from exc
        if protected == output or protected in output.parents:
            raise ConversionRefused(f"output bundle must not be inside the {label}")
    return replace(request, output_bundle=output)


def _load_manifest(quantization: str, max_input_tokens: int) -> dict[str, Any]:
    return {
        "format": "gguf",
        "quantization": quantization,
        "entrypoint": ENTRYPOINT,
        "max_input": {"tokens": max_input_tokens},
        "preprocessing": {"tokenizer": "tokenizer.json"},
        "base_model": provenance.BASE_MODEL,
    }


def _receipt(
    *,
    lineage: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_identity = _mapping(lineage["receipt"], "training receipt identity")
    run = _mapping(lineage["run"], "training run identity")
    merged = _mapping(run["merged"], "merged tree identity")
    converter = _mapping(toolchain["converter"], "converter identity")
    quantizer = _mapping(toolchain["quantizer"], "quantizer identity")
    entrypoint = _mapping(artifact["entrypoint"], "artifact entrypoint")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "track": provenance.TRACK,
        "hardware_class": provenance.HARDWARE_CLASS,
        "base_model": provenance.BASE_MODEL,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "source": {
            "training_metadata_digest": receipt_identity["digest"],
            "merged_tree_digest": merged["digest"],
        },
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


def convert(request: ConversionRequest) -> dict[str, Any]:
    """Run conversion and atomically publish a fully replayed output bundle."""

    request = _validate_request(request)
    initial_lineage = _load_lineage(request)
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
        checkout = Path(str(initial_tools["root"]))
        convert_argv = (
            str(converter),
            str(merged_root),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        )
        commands = [_conversion_command("convert_f16", convert_argv, cwd=checkout)]
        f16_identity = gguf.file_identity(f16_path, "temporary F16 GGUF")
        if int(f16_identity["bytes"]) < 1:
            raise ConversionRefused("converter produced an empty F16 GGUF")
        quantize_argv = (
            str(quantizer),
            str(f16_path),
            str(model_path),
            request.quantization,
        )
        commands.append(_conversion_command("quantize", quantize_argv, cwd=checkout))
        gguf.file_identity(model_path, "quantized GGUF")
        _fsync_path(model_path)

        tree_digest = _official_tree_digest(artifact_root)
        artifact = gguf.artifact_identity(
            artifact_root,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization=request.quantization,
        )
        load_manifest = _load_manifest(request.quantization, request.max_input_tokens)
        provenance._validate_load_manifest(load_manifest, artifact)
        receipt = _receipt(
            lineage=initial_lineage,
            toolchain=initial_tools,
            commands=commands,
            artifact=artifact,
            load_manifest=load_manifest,
        )
        provenance._validate_generic_conversion(
            receipt,
            training_lineage=initial_lineage,
            artifact=artifact,
            load_manifest=load_manifest,
            calibration_digest=None,
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
        )
        _same_artifact_identity(artifact, final_staged_artifact, "staged artifact")
        if (staging / LOAD_SPEC_NAME).read_bytes() != load_raw:
            raise ConversionRefused("staged load specification changed")
        if (staging / RECEIPT_NAME).read_bytes() != receipt_raw:
            raise ConversionRefused("staged conversion receipt changed")
        expected_files = frozenset({f"{ARTIFACT_NAME}/{ENTRYPOINT}", LOAD_SPEC_NAME, RECEIPT_NAME})
        if _bundle_file_set(staging) != expected_files:
            raise ConversionRefused("staged output bundle contains unexpected files")

        _publish_directory_noreplace(staging, output)
        published = True
        _fsync_path(output.parent, directory=True)
        final_artifact = gguf.artifact_identity(
            output / ARTIFACT_NAME,
            entrypoint=ENTRYPOINT,
            expected_digest=tree_digest,
            quantization=request.quantization,
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
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--quantization", choices=sorted(SUPPORTED_QUANTIZATIONS), required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
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
            quantizer=args.quantizer,
            output_bundle=args.output_bundle,
            quantization=args.quantization,
            max_input_tokens=args.max_input_tokens,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
