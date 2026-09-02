#!/usr/bin/env python3
"""Bootstrap the pinned llama.cpp converter under Python safe-path mode.

The wrapper imports no model or converter modules.  It validates the exact
checkout and converter identities, adds only the pinned checkout root to the
front of ``sys.path``, verifies that ``conversion`` resolves beneath that
root, and then executes the converter with ``runpy`` as ``__main__``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import stat
import sys
from pathlib import Path
from typing import Any, Final, NoReturn

sys.dont_write_bytecode = True

SCHEMA: Final[str] = "microtensor.code.pinned-llama-converter-bootstrap.v1"
LLAMA_ROOT: Final[Path] = Path("/tmp/llama.cpp")  # noqa: S108
CONVERTER: Final[Path] = LLAMA_ROOT / "convert_hf_to_gguf.py"
CONVERSION_ROOT: Final[Path] = LLAMA_ROOT / "conversion"
GGUF_ROOT: Final[Path] = LLAMA_ROOT / "gguf-py" / "gguf"
PINNED_COMMIT: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
CONVERTER_BYTES: Final[int] = 12_798
CONVERTER_SHA256: Final[str] = (
    "sha256:e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f"
)

EXACT_ENVIRONMENT: Final[dict[str, str]] = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONUNBUFFERED": "1",
    "TZ": "UTC",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "WANDB_MODE": "disabled",
}


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _directory_identity(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        _fail(f"pinned directory is not one exact absolute directory: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        _fail(f"cannot stat pinned directory {path}: {exc}")
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o002:
        _fail(f"pinned directory is absent, special, or world writable: {path}")
    return {
        "path": str(path),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": info.st_uid,
        "gid": info.st_gid,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        _fail(f"pinned file is not one exact absolute file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(f"cannot open pinned file {path}: {exc}")
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or size != before.st_size:
        _fail(f"pinned file mutated while hashing: {path}")
    if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) & 0o002:
        _fail(f"pinned file is special or world writable: {path}")
    return {
        "path": str(path),
        "bytes": size,
        "sha256": "sha256:" + digest.hexdigest(),
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
    }


def _pinned_commit() -> dict[str, Any]:
    head = LLAMA_ROOT / ".git" / "HEAD"
    reference = LLAMA_ROOT / ".git" / "refs" / "heads" / "master"
    try:
        head_text = head.read_text(encoding="ascii").strip()
        commit = reference.read_text(encoding="ascii").strip()
    except OSError as exc:
        _fail(f"cannot read pinned Git identity: {exc}")
    if head_text != "ref: refs/heads/master" or commit != PINNED_COMMIT:
        _fail(f"llama.cpp checkout is not pinned to {PINNED_COMMIT}")
    return {
        "commit": commit,
        "head": _file_identity(head),
        "reference": _file_identity(reference),
    }


def prepare_pinned_runtime() -> dict[str, Any]:
    """Validate the checkout and install its exact local import root."""

    root = _directory_identity(LLAMA_ROOT)
    converter = _file_identity(CONVERTER)
    conversion = _directory_identity(CONVERSION_ROOT)
    gguf = _directory_identity(GGUF_ROOT)
    if converter["bytes"] != CONVERTER_BYTES or converter["sha256"] != CONVERTER_SHA256:
        _fail("pinned convert_hf_to_gguf.py identity changed")
    if any(not value or not os.path.isabs(value) for value in sys.path):
        _fail("Python safe-path boundary contains an empty or relative entry")
    root_text = str(LLAMA_ROOT)
    sys.path[:] = [value for value in sys.path if value != root_text]
    sys.path.insert(0, root_text)
    spec = importlib.util.find_spec("conversion")
    if spec is None or spec.origin is None:
        _fail("pinned conversion package cannot be resolved")
    origin = Path(spec.origin).resolve()
    try:
        origin.relative_to(CONVERSION_ROOT)
    except ValueError:
        _fail(f"conversion resolved outside the pinned root: {origin}")
    return {
        "schema": SCHEMA,
        "root": root,
        "converter": converter,
        "conversion_root": conversion,
        "gguf_root": gguf,
        "git": _pinned_commit(),
        "conversion_origin": str(origin),
        "sys_path_inserted": root_text,
    }


def main(arguments: list[str] | None = None) -> None:
    if dict(os.environ) != EXACT_ENVIRONMENT:
        _fail("wrapper environment differs from the exact fixed mapping")
    forwarded = list(sys.argv[1:] if arguments is None else arguments)
    if "--remote" in forwarded:
        _fail("remote conversion is forbidden")
    evidence = prepare_pinned_runtime()
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("ascii")
    os.write(2, b"MICROTENSOR_PINNED_LLAMA=" + raw + b"\n")
    prior_argv = sys.argv
    try:
        sys.argv = [str(CONVERTER), *forwarded]
        runpy.run_path(str(CONVERTER), run_name="__main__")
    finally:
        sys.argv = prior_argv


if __name__ == "__main__":
    main()
