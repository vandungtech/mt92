#!/usr/bin/env python3
"""Hardened, model-free runtime attestation and contained command execution.

This module is intentionally independent from the training and conversion
modules.  Importing it never imports a model framework and never starts a
converter.  Callers must name an absolute Python interpreter, every requested
distribution/module tree, and every native executable involved in a command.

The implementation is Linux-only.  A command is executed without a shell,
with a fixed environment, in a new session, beneath a temporary child-subreaper
boundary.  Runtime inputs are attested immediately before and after execution;
any difference makes the attempt fail closed.  The bounded stdout/stderr
captures and the complete result are committed to an fsync'd JSON record.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import json
import os
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

sys.dont_write_bytecode = True

SCHEMA: Final[str] = "microtensor.code.conversion-command.v1"
RUNTIME_SCHEMA: Final[str] = "microtensor.code.conversion-runtime.v1"
ELF_SCHEMA: Final[str] = "microtensor.code.elf-closure.v1"
FILE_SCHEMA: Final[str] = "microtensor.code.file-identity.v1"
TREE_SCHEMA: Final[str] = "microtensor.code.module-tree.v1"
_PR_SET_CHILD_SUBREAPER: Final[int] = 36
_PR_GET_CHILD_SUBREAPER: Final[int] = 37
_PR_SET_PDEATHSIG: Final[int] = 1
_PR_GET_PDEATHSIG: Final[int] = 2
_WAIT_ALL: Final[int] = 0x40000000
_MAX_SYMLINKS: Final[int] = 40
_MAX_PROBE_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_ARG_BYTES: Final[int] = 128 * 1024
_POLL_SECONDS: Final[float] = 0.025
_RUN_LOCK: Final[threading.Lock] = threading.Lock()
_MODULE_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_]\w*\Z")
_DIST_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")

# Deliberately no PATH, HOME, user-site, credential, proxy, or dynamic-loader
# variables.  Absolute executable paths are mandatory.
FIXED_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)

DEFAULT_ELF_LIBRARY_DIRS: Final[tuple[Path, ...]] = tuple(
    Path(value)
    for value in (
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    )
)

_PROBE = r"""
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import sys

request = json.loads(sys.argv[1])
distributions = []
for name in request["distributions"]:
    distribution = metadata.distribution(name)
    distributions.append({
        "requested_name": name,
        "canonical_name": distribution.metadata.get("Name"),
        "version": distribution.version,
        "metadata_root": os.path.abspath(str(distribution._path)),
    })
modules = []
for name in request["modules"]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError("module not found: " + name)
    modules.append({
        "name": name,
        "origin": None if spec.origin in (None, "built-in", "frozen")
            else os.path.abspath(spec.origin),
        "loader": type(spec.loader).__module__ + "." + type(spec.loader).__qualname__,
        "search_locations": [] if spec.submodule_search_locations is None else
            sorted(os.path.abspath(value) for value in spec.submodule_search_locations),
    })
print(json.dumps({
    "executable": os.path.abspath(sys.executable),
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
    "version_info": list(sys.version_info[:5]),
    "hexversion": sys.hexversion,
    "cache_tag": sys.implementation.cache_tag,
    "prefix": os.path.abspath(sys.prefix),
    "base_prefix": os.path.abspath(sys.base_prefix),
    "distributions": distributions,
    "modules": modules,
}, sort_keys=True, separators=(",", ":")))
""".strip()


class RuntimeRefused(RuntimeError):
    """A runtime identity or containment precondition was not satisfied."""


class CommandExecutionError(RuntimeRefused):
    """A started command did not produce an accepted terminal record."""

    def __init__(self, message: str, record: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.record = dict(record)


@dataclass(frozen=True)
class CommandRequest:
    """Complete immutable request for one contained command attempt."""

    interpreter: Path
    argv: tuple[str, ...]
    cwd: Path
    record_path: Path
    distributions: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    executables: tuple[Path, ...] = ()
    files: tuple[Path, ...] = ()
    trees: tuple[Path, ...] = ()
    exec_target: Path | None = None
    expected_runtime_sha256: str | None = None
    elf_library_dirs: tuple[Path, ...] = ()
    timeout_seconds: float = 3600.0
    term_grace_seconds: float = 3.0
    cleanup_seconds: float = 5.0
    max_log_bytes: int = 1024 * 1024


@dataclass
class _BoundedLog:
    maximum: int
    total: int = 0
    captured: bytearray = field(default_factory=bytearray)
    digest: Any = field(default_factory=hashlib.sha256)

    def update(self, value: bytes) -> None:
        self.total += len(value)
        self.digest.update(value)
        remaining = self.maximum - len(self.captured)
        if remaining > 0:
            self.captured.extend(value[:remaining])

    def payload(self) -> dict[str, Any]:
        captured = bytes(self.captured)
        return {
            "bytes": self.total,
            "sha256": "sha256:" + self.digest.hexdigest(),
            "captured_bytes": len(captured),
            "captured_sha256": "sha256:" + hashlib.sha256(captured).hexdigest(),
            "captured_base64": base64.b64encode(captured).decode("ascii"),
            "truncated": self.total > len(captured),
        }


def _canonical_bytes(value: Any) -> bytes:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (rendered + "\n").encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _timestamp() -> tuple[str, int]:
    nanoseconds = time.time_ns()
    rendered = datetime.fromtimestamp(nanoseconds / 1_000_000_000, timezone.utc)
    return rendered.isoformat(timespec="microseconds").replace("+00:00", "Z"), nanoseconds


def _absolute(path: Path | str, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RuntimeRefused(f"{label} must be absolute: {candidate}")
    if "\x00" in os.fspath(candidate):
        raise RuntimeRefused(f"{label} contains NUL")
    return candidate


def _mode(value: int) -> str:
    return f"{stat.S_IMODE(value):04o}"


def _hash_open_file(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeRefused(f"cannot open immutable file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeRefused(f"attested path is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeRefused(f"file mutated while it was hashed: {path}")
        if size != before.st_size:
            raise RuntimeRefused(f"short or growing read while hashing {path}")
        return {
            "bytes": size,
            "sha256": "sha256:" + digest.hexdigest(),
            "mode": _mode(before.st_mode),
            "uid": before.st_uid,
            "gid": before.st_gid,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
    finally:
        os.close(descriptor)


def attest_file(path: Path | str, *, require_executable: bool = False) -> dict[str, Any]:
    """Return the symlink-chain and immutable identity of one absolute file."""

    requested = _absolute(path, "file path")
    current = requested
    links: list[dict[str, Any]] = []
    visited: set[tuple[int, int]] = set()
    for _ in range(_MAX_SYMLINKS + 1):
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise RuntimeRefused(f"cannot lstat {current}: {exc}") from exc
        if not stat.S_ISLNK(info.st_mode):
            break
        identity = (info.st_dev, info.st_ino)
        if identity in visited:
            raise RuntimeRefused(f"symlink loop while resolving {requested}")
        visited.add(identity)
        target = os.readlink(current)
        raw_target = os.fsencode(target)
        links.append(
            {
                "path": str(current),
                "target": target,
                "target_sha256": _digest_bytes(raw_target),
                "mode": _mode(info.st_mode),
                "uid": info.st_uid,
                "gid": info.st_gid,
                "device": info.st_dev,
                "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
        )
        current = Path(target) if os.path.isabs(target) else current.parent / target
        current = Path(os.path.normpath(current))
    else:
        raise RuntimeRefused(f"more than {_MAX_SYMLINKS} symlinks resolve {requested}")

    identity = _hash_open_file(current)
    mode = int(identity["mode"], 8)
    if mode & 0o002:
        raise RuntimeRefused(f"attested file is world writable: {current}")
    if require_executable and mode & 0o111 == 0:
        raise RuntimeRefused(f"attested executable has no execute bit: {current}")
    if require_executable and mode & 0o6000:
        raise RuntimeRefused(f"attested executable is setuid/setgid: {current}")
    return {
        "schema": FILE_SCHEMA,
        "requested_path": str(requested),
        "resolved_path": str(Path(os.path.realpath(requested))),
        "symlinks": links,
        "file": {"path": str(current), **identity},
        "executable_required": require_executable,
    }


def _tree_identity(path: Path, label: str) -> dict[str, Any]:
    root = _absolute(path, label)
    if root.is_symlink():
        raise RuntimeRefused(f"{label} root may not be a symlink: {root}")
    try:
        root_info = root.stat()
    except OSError as exc:
        raise RuntimeRefused(f"cannot stat {label} {root}: {exc}") from exc
    if not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeRefused(f"{label} is not a directory: {root}")
    if stat.S_IMODE(root_info.st_mode) & 0o002:
        raise RuntimeRefused(f"{label} root is world writable: {root}")
    entries: list[dict[str, Any]] = []
    native_paths: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise RuntimeRefused(f"cannot scan {label} directory {directory}: {exc}") from exc
        for child in children:
            child_path = Path(child.path)
            relative = child_path.relative_to(root).as_posix()
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeRefused(f"cannot lstat {child_path}: {exc}") from exc
            common = {
                "path": relative,
                "mode": _mode(info.st_mode),
                "uid": info.st_uid,
                "gid": info.st_gid,
                "device": info.st_dev,
                "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
            if stat.S_ISDIR(info.st_mode):
                if stat.S_IMODE(info.st_mode) & 0o002:
                    raise RuntimeRefused(f"module directory is world writable: {child_path}")
                entries.append({**common, "kind": "directory"})
                stack.append(child_path)
            elif stat.S_ISREG(info.st_mode):
                file_identity = _hash_open_file(child_path)
                if int(file_identity["mode"], 8) & 0o002:
                    raise RuntimeRefused(f"module file is world writable: {child_path}")
                entries.append(
                    {
                        "path": relative,
                        "kind": "file",
                        **{key: value for key, value in file_identity.items() if key != "path"},
                    }
                )
                if ".so" in child.name and (child.name.endswith(".so") or ".so." in child.name):
                    native_paths.append(child_path)
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(child_path)
                entries.append(
                    {
                        **common,
                        "kind": "symlink",
                        "target": target,
                        "target_sha256": _digest_bytes(os.fsencode(target)),
                    }
                )
                resolved = Path(os.path.realpath(child_path))
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise RuntimeRefused(
                        f"module symlink escapes tree: {child_path} -> {target}"
                    ) from exc
                if ".so" in child.name and (child.name.endswith(".so") or ".so." in child.name):
                    native_paths.append(child_path)
            else:
                raise RuntimeRefused(f"special file in module tree: {child_path}")
    entries.sort(key=lambda item: os.fsencode(item["path"]))
    payload = {"schema": TREE_SCHEMA, "root": str(root), "entries": entries}
    raw = _canonical_bytes(payload)
    payload["canonical_bytes"] = len(raw)
    payload["sha256"] = _digest_bytes(raw)
    payload["native_extensions"] = [str(path) for path in sorted(native_paths)]
    return payload


def _read_elf(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise RuntimeRefused(f"cannot read ELF candidate {path}: {exc}") from exc
    if not raw.startswith(b"\x7fELF"):
        return None
    if len(raw) < 64 or raw[4] != 2 or raw[5] != 1:
        raise RuntimeRefused(f"only little-endian ELF64 is supported: {path}")
    try:
        program_offset = struct.unpack_from("<Q", raw, 32)[0]
        program_size = struct.unpack_from("<H", raw, 54)[0]
        program_count = struct.unpack_from("<H", raw, 56)[0]
    except struct.error as exc:
        raise RuntimeRefused(f"truncated ELF header: {path}") from exc
    if program_size != 56 or program_count > 4096:
        raise RuntimeRefused(f"unsupported ELF program header table: {path}")
    headers: list[tuple[int, int, int, int, int, int, int, int]] = []
    try:
        for index in range(program_count):
            headers.append(struct.unpack_from("<IIQQQQQQ", raw, program_offset + index * 56))
    except struct.error as exc:
        raise RuntimeRefused(f"truncated ELF program headers: {path}") from exc
    loads = [header for header in headers if header[0] == 1]
    interpreter: str | None = None
    for header in headers:
        if header[0] == 3:
            offset, length = header[2], header[5]
            value = raw[offset : offset + length]
            if len(value) != length or not value.endswith(b"\0"):
                raise RuntimeRefused(f"invalid PT_INTERP in {path}")
            interpreter = os.fsdecode(value[:-1])
    dynamic = next((header for header in headers if header[0] == 2), None)
    if dynamic is None:
        return {"interpreter": interpreter, "needed": [], "rpath": [], "runpath": []}
    dynamic_offset, dynamic_size = dynamic[2], dynamic[5]
    if dynamic_offset + dynamic_size > len(raw) or dynamic_size % 16:
        raise RuntimeRefused(f"invalid PT_DYNAMIC in {path}")
    tags: list[tuple[int, int]] = []
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, 16):
        tag, value = struct.unpack_from("<QQ", raw, offset)
        if tag == 0:
            break
        tags.append((tag, value))
    string_addresses = [value for tag, value in tags if tag == 5]
    string_sizes = [value for tag, value in tags if tag == 10]
    if not string_addresses or not string_sizes:
        if any(tag in {1, 15, 29} for tag, _ in tags):
            raise RuntimeRefused(f"dynamic strings are unavailable in {path}")
        return {"interpreter": interpreter, "needed": [], "rpath": [], "runpath": []}
    address, length = string_addresses[0], string_sizes[0]
    string_offset: int | None = None
    for header in loads:
        file_offset, virtual, file_size = header[2], header[3], header[5]
        if virtual <= address < virtual + file_size:
            string_offset = file_offset + address - virtual
            break
    if string_offset is None or string_offset + length > len(raw):
        raise RuntimeRefused(f"ELF dynamic string table is invalid: {path}")
    strings = raw[string_offset : string_offset + length]

    def text(offset: int) -> str:
        if offset >= len(strings):
            raise RuntimeRefused(f"ELF dynamic string offset is invalid: {path}")
        end = strings.find(b"\0", offset)
        if end < 0:
            raise RuntimeRefused(f"unterminated ELF dynamic string: {path}")
        try:
            return os.fsdecode(strings[offset:end])
        except UnicodeDecodeError as exc:
            raise RuntimeRefused(f"non-decodable ELF dynamic string: {path}") from exc

    needed = [text(value) for tag, value in tags if tag == 1]
    rpath = [part for tag, value in tags if tag == 15 for part in text(value).split(":")]
    runpath = [part for tag, value in tags if tag == 29 for part in text(value).split(":")]
    return {"interpreter": interpreter, "needed": needed, "rpath": rpath, "runpath": runpath}


def _elf_search_dirs(origin: Path, values: Iterable[str]) -> tuple[Path, ...]:
    result: list[Path] = []
    for value in values:
        if not value:
            raise RuntimeRefused("empty ELF RPATH/RUNPATH component is forbidden")
        expanded = value.replace("${ORIGIN}", str(origin)).replace("$ORIGIN", str(origin))
        if "$" in expanded:
            raise RuntimeRefused(f"unsupported ELF loader token: {value}")
        candidate = Path(os.path.normpath(expanded))
        if not candidate.is_absolute():
            raise RuntimeRefused(f"relative ELF loader directory is forbidden: {value}")
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def attest_elf_closure(
    executable: Path | str,
    *,
    library_dirs: Sequence[Path | str] = (),
    require_executable: bool = True,
) -> dict[str, Any]:
    """Attest an ELF and the recursive interpreter/DT_NEEDED closure."""

    root = attest_file(executable, require_executable=require_executable)
    additional = tuple(_absolute(value, "ELF library directory") for value in library_dirs)
    for directory in additional:
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeRefused(f"ELF library directory is unavailable or a symlink: {directory}")
    queue: list[tuple[dict[str, Any], bool]] = [(root, True)]
    objects: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    while queue:
        identity, is_root = queue.pop(0)
        resolved = identity["resolved_path"]
        if resolved in objects:
            continue
        parsed = _read_elf(Path(identity["file"]["path"]))
        if parsed is None:
            if is_root:
                return {
                    "schema": ELF_SCHEMA,
                    "root": root,
                    "is_elf": False,
                    "objects": [],
                    "edges": [],
                }
            raise RuntimeRefused(f"ELF dependency is not ELF: {resolved}")
        objects[resolved] = {"identity": identity, "dynamic": parsed}
        origin = Path(resolved).parent
        if parsed["interpreter"] is not None:
            interp = parsed["interpreter"]
            if not os.path.isabs(interp):
                raise RuntimeRefused(f"relative ELF interpreter in {resolved}")
            interp_identity = attest_file(interp, require_executable=True)
            edges.append(
                {
                    "from": resolved,
                    "kind": "interpreter",
                    "needed": interp,
                    "to": interp_identity["resolved_path"],
                    "alias": interp_identity,
                }
            )
            if interp_identity["resolved_path"] not in objects:
                queue.append((interp_identity, False))
        loader_paths = _elf_search_dirs(origin, parsed["rpath"])
        loader_paths += _elf_search_dirs(origin, parsed["runpath"])
        loader_paths += additional
        loader_paths += tuple(path for path in DEFAULT_ELF_LIBRARY_DIRS if path.is_dir())
        for name in parsed["needed"]:
            if "/" in name:
                if not os.path.isabs(name):
                    raise RuntimeRefused(f"relative DT_NEEDED path in {resolved}: {name}")
                candidates = [Path(name)]
            else:
                candidates = [directory / name for directory in loader_paths]
            dependency = next(
                (candidate for candidate in candidates if os.path.lexists(candidate)),
                None,
            )
            if dependency is None:
                raise RuntimeRefused(f"cannot resolve DT_NEEDED {name!r} from {resolved}")
            dep_identity = attest_file(dependency, require_executable=False)
            edges.append(
                {
                    "from": resolved,
                    "kind": "needed",
                    "needed": name,
                    "to": dep_identity["resolved_path"],
                    "alias": dep_identity,
                }
            )
            if dep_identity["resolved_path"] not in objects:
                queue.append((dep_identity, False))
    return {
        "schema": ELF_SCHEMA,
        "root": root,
        "is_elf": True,
        "objects": [objects[key] for key in sorted(objects)],
        "edges": sorted(
            edges,
            key=lambda item: (item["from"], item["kind"], item["needed"] or ""),
        ),
    }


def _run_probe(
    interpreter: Path, distributions: Sequence[str], modules: Sequence[str]
) -> dict[str, Any]:
    request = json.dumps(
        {"distributions": list(distributions), "modules": list(modules)},
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        completed = subprocess.run(  # noqa: S603 - absolute, attested interpreter; no shell
            [str(interpreter), "-I", "-c", _PROBE, request],
            check=False,
            cwd="/",
            env=dict(FIXED_ENVIRONMENT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeRefused(f"Python identity probe could not complete: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr[:4096].decode("utf-8", errors="replace")
        raise RuntimeRefused(f"Python identity probe failed ({completed.returncode}): {detail}")
    if len(completed.stdout) > _MAX_PROBE_BYTES or len(completed.stderr) > _MAX_PROBE_BYTES:
        raise RuntimeRefused("Python identity probe exceeded its output ceiling")
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeRefused("Python identity probe did not emit one JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeRefused("Python identity probe result is not an object")
    return value


def attest_runtime(
    interpreter: Path | str,
    *,
    distributions: Sequence[str] = (),
    modules: Sequence[str] = (),
    executables: Sequence[Path | str] = (),
    files: Sequence[Path | str] = (),
    trees: Sequence[Path | str] = (),
    elf_library_dirs: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Attest the requested Python, packages, module trees, and executables."""

    if sys.platform != "linux":
        raise RuntimeRefused("conversion runtime attestation requires Linux")
    interpreter_path = _absolute(interpreter, "Python interpreter")
    distribution_names = tuple(distributions)
    module_names = tuple(modules)
    if len(set(distribution_names)) != len(distribution_names) or any(
        not _DIST_NAME.fullmatch(value) for value in distribution_names
    ):
        raise RuntimeRefused("distribution names must be unique normalized text names")
    if len(set(module_names)) != len(module_names) or any(
        not _MODULE_NAME.fullmatch(value) for value in module_names
    ):
        raise RuntimeRefused("module names must be unique top-level import names")
    interpreter_identity = attest_file(interpreter_path, require_executable=True)
    interpreter_elf = attest_elf_closure(interpreter_path, library_dirs=elf_library_dirs)
    probe = _run_probe(interpreter_path, distribution_names, module_names)
    if Path(probe.get("executable", "")).resolve() != Path(
        interpreter_identity["resolved_path"]
    ).resolve():
        raise RuntimeRefused("Python probe executable differs from the attested interpreter")
    probed_distributions = probe.pop("distributions", None)
    probed_modules = probe.pop("modules", None)
    if not isinstance(probed_distributions, list) or not isinstance(probed_modules, list):
        raise RuntimeRefused("Python probe omitted distribution/module results")
    distribution_records: list[dict[str, Any]] = []
    for item in probed_distributions:
        if not isinstance(item, dict) or item.get("requested_name") not in distribution_names:
            raise RuntimeRefused("Python probe returned an undeclared distribution")
        metadata_root = _tree_identity(Path(item["metadata_root"]), "distribution metadata")
        distribution_records.append({**item, "metadata_tree": metadata_root})
    if {item["requested_name"] for item in distribution_records} != set(distribution_names):
        raise RuntimeRefused("Python probe did not return every requested distribution")
    module_records: list[dict[str, Any]] = []
    native_extensions: set[str] = set()
    for item in probed_modules:
        if not isinstance(item, dict) or item.get("name") not in module_names:
            raise RuntimeRefused("Python probe returned an undeclared module")
        module_trees: list[dict[str, Any]] = []
        locations = item.get("search_locations")
        origin = item.get("origin")
        if not isinstance(locations, list):
            raise RuntimeRefused("Python probe module search locations are invalid")
        if locations:
            for location in locations:
                tree = _tree_identity(Path(location), "module tree")
                module_trees.append(tree)
                native_extensions.update(tree["native_extensions"])
            origin_identity = None
        elif origin is not None:
            origin_identity = attest_file(Path(origin), require_executable=False)
            if ".so" in Path(origin).name:
                native_extensions.add(str(Path(origin)))
        else:
            origin_identity = None
        module_records.append(
            {**item, "origin_identity": origin_identity, "trees": module_trees}
        )
    if {item["name"] for item in module_records} != set(module_names):
        raise RuntimeRefused("Python probe did not return every requested module")
    executable_records = [
        {
            "identity": attest_file(value, require_executable=True),
            "elf": attest_elf_closure(value, library_dirs=elf_library_dirs),
        }
        for value in executables
    ]
    file_records = [attest_file(value, require_executable=False) for value in files]
    tree_records = [_tree_identity(Path(value), "requested source tree") for value in trees]
    native_records = [
        {
            "identity": attest_file(value, require_executable=False),
            "elf": attest_elf_closure(
                value, library_dirs=elf_library_dirs, require_executable=False
            ),
        }
        for value in sorted(native_extensions)
    ]
    payload = {
        "schema": RUNTIME_SCHEMA,
        "environment": dict(FIXED_ENVIRONMENT),
        "interpreter": {
            "identity": interpreter_identity,
            "elf": interpreter_elf,
            "python": probe,
        },
        "distributions": sorted(distribution_records, key=lambda item: item["requested_name"]),
        "modules": sorted(module_records, key=lambda item: item["name"]),
        "native_extensions": native_records,
        "executables": executable_records,
        "files": file_records,
        "trees": tree_records,
    }
    raw = _canonical_bytes(payload)
    return {**payload, "canonical_bytes": len(raw), "sha256": _digest_bytes(raw)}


def _libc() -> ctypes.CDLL:
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    library.prctl.restype = ctypes.c_int
    return library


def _subreaper_state(library: ctypes.CDLL) -> int:
    value = ctypes.c_int(-1)
    ctypes.set_errno(0)
    result = library.prctl(
        _PR_GET_CHILD_SUBREAPER,
        ctypes.cast(ctypes.pointer(value), ctypes.c_void_p).value,
        0,
        0,
        0,
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise RuntimeRefused(f"cannot inspect child-subreaper state: {os.strerror(number)}")
    if value.value not in {0, 1}:
        raise RuntimeRefused(f"invalid child-subreaper state {value.value}")
    return value.value


def _set_subreaper(library: ctypes.CDLL, value: int) -> None:
    ctypes.set_errno(0)
    if library.prctl(_PR_SET_CHILD_SUBREAPER, value, 0, 0, 0) != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise RuntimeRefused(f"cannot set child-subreaper state: {os.strerror(number)}")
    if _subreaper_state(library) != value:
        raise RuntimeRefused("child-subreaper state did not take effect")


def _pdeathsig_state(library: ctypes.CDLL) -> int:
    value = ctypes.c_int(-1)
    ctypes.set_errno(0)
    result = library.prctl(
        _PR_GET_PDEATHSIG,
        ctypes.cast(ctypes.pointer(value), ctypes.c_void_p).value,
        0,
        0,
        0,
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))
    return value.value


def _set_pdeathsig(library: ctypes.CDLL, signum: int) -> None:
    ctypes.set_errno(0)
    if library.prctl(_PR_SET_PDEATHSIG, signum, 0, 0, 0) != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))
    if _pdeathsig_state(library) != signum:
        raise OSError(errno.EIO, "parent-death signal did not take effect")


def _verify_preexec_parent(expected_parent_pid: int, observed_parent_pid: int) -> None:
    if observed_parent_pid != expected_parent_pid:
        raise OSError(
            errno.ESRCH,
            "launcher parent changed while configuring the child boundary",
        )


def _child_preexec_boundary(expected_parent_pid: int) -> None:
    library = _libc()
    _set_pdeathsig(library, signal.SIGKILL)
    _verify_preexec_parent(expected_parent_pid, os.getppid())


def _task_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(item.name) for item in os.scandir("/proc/self/task")))
    except (OSError, ValueError) as exc:
        raise RuntimeRefused(f"cannot inspect launcher tasks: {exc}") from exc


def _child_pids() -> tuple[int, ...]:
    result: set[int] = set()
    for task_id in _task_ids():
        try:
            raw = Path(f"/proc/self/task/{task_id}/children").read_text(encoding="ascii")
        except OSError as exc:
            raise RuntimeRefused(f"cannot inspect launcher children: {exc}") from exc
        for value in raw.split():
            if not value.isdecimal() or int(value) < 1:
                raise RuntimeRefused("invalid PID in /proc child inventory")
            result.add(int(value))
    return tuple(sorted(result))


def _require_clean_boundary() -> dict[str, Any]:
    if sys.platform != "linux" or not Path("/proc/self/status").is_file():
        raise RuntimeRefused("Linux /proc child boundary is required")
    tasks = _task_ids()
    if tasks != (os.getpid(),):
        raise RuntimeRefused(f"launcher must have exactly one task, got {list(tasks)}")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeRefused("SIGCHLD disposition must be SIG_DFL")
    children = _child_pids()
    if children:
        raise RuntimeRefused(f"launcher has undeclared children: {list(children)}")
    while True:
        try:
            waited, status = os.waitpid(-1, os.WNOHANG | _WAIT_ALL)
        except InterruptedError:
            continue
        except ChildProcessError:
            break
        if waited == 0:
            raise RuntimeRefused("waitpid reports an undeclared live child")
        raise RuntimeRefused(f"reaped undeclared child {waited} with status {status}")
    return {
        "task_ids": list(tasks),
        "proc_children_empty": True,
        "waitpid_echild_verified": True,
        "sigchld_disposition": "SIG_DFL",
    }


def _validate_request(request: CommandRequest) -> tuple[Path, Path, Path]:
    interpreter = _absolute(request.interpreter, "Python interpreter")
    cwd = _absolute(request.cwd, "working directory")
    record = _absolute(request.record_path, "command record")
    if not cwd.is_dir() or cwd.is_symlink():
        raise RuntimeRefused(f"working directory must be a real directory: {cwd}")
    cwd_mode = stat.S_IMODE(cwd.stat().st_mode)
    if cwd_mode & 0o002:
        raise RuntimeRefused(f"working directory is world writable: {cwd}")
    if record.parent.is_symlink() or not record.parent.is_dir():
        raise RuntimeRefused("record parent must be an existing real directory")
    if stat.S_IMODE(record.parent.stat().st_mode) & 0o002:
        raise RuntimeRefused("record parent is world writable")
    if not request.argv or request.argv[0] != str(interpreter):
        raise RuntimeRefused("argv[0] must be the exact declared Python interpreter")
    if len(request.argv) < 2:
        raise RuntimeRefused("Python command must name an absolute script")
    script = _absolute(request.argv[1], "Python script")
    if script not in tuple(request.files):
        raise RuntimeRefused("the absolute Python script must be present in request.files")
    if request.exec_target is not None:
        target = _absolute(request.exec_target, "exec-wrapper target")
        if len(request.argv) < 3 or request.argv[2] != str(target):
            raise RuntimeRefused("argv[2] must be the exact exec-wrapper target")
        if target not in tuple(request.executables):
            raise RuntimeRefused("exec-wrapper target must be present in request.executables")
    if request.expected_runtime_sha256 is not None and not _SHA256.fullmatch(
        request.expected_runtime_sha256
    ):
        raise RuntimeRefused(
            "expected_runtime_sha256 must be one lowercase SHA-256 identity"
        )
    if any(not isinstance(value, str) or "\x00" in value for value in request.argv):
        raise RuntimeRefused("argv contains a non-string or NUL")
    if sum(len(os.fsencode(value)) + 1 for value in request.argv) > _MAX_ARG_BYTES:
        raise RuntimeRefused("argv exceeds its byte ceiling")
    for label, value, minimum, maximum in (
        ("timeout_seconds", request.timeout_seconds, 0.05, 7 * 24 * 3600),
        ("term_grace_seconds", request.term_grace_seconds, 0.01, 60),
        ("cleanup_seconds", request.cleanup_seconds, 0.1, 60),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not minimum <= value <= maximum
        ):
            raise RuntimeRefused(f"{label} must be in [{minimum}, {maximum}]")
    if (
        isinstance(request.max_log_bytes, bool)
        or not 0 <= request.max_log_bytes <= 64 * 1024 * 1024
    ):
        raise RuntimeRefused("max_log_bytes must be in [0, 67108864]")
    return interpreter, cwd, record


def _attest_request(request: CommandRequest) -> dict[str, Any]:
    return attest_runtime(
        request.interpreter,
        distributions=request.distributions,
        modules=request.modules,
        executables=request.executables,
        files=request.files,
        trees=request.trees,
        elf_library_dirs=request.elf_library_dirs,
    )


def _write_record(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    raw = _canonical_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeRefused(f"cannot commit command record {path}: {exc}") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeRefused("zero-byte write while committing command record")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _replace_record(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        _write_record(temporary, payload, exclusive=True)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _pidfd_open(pid: int) -> int | None:
    operation = getattr(os, "pidfd_open", None)
    if not callable(operation):
        return None
    try:
        return operation(pid, 0)
    except (OSError, ProcessLookupError):
        return None


def _signal_pidfd(descriptor: int | None, signum: signal.Signals) -> bool:
    operation = getattr(signal, "pidfd_send_signal", None)
    if descriptor is None or not callable(operation):
        return False
    try:
        operation(descriptor, signum, None, 0)
    except ProcessLookupError:
        return True
    return True


def _signal_group(pid: int, signum: signal.Signals) -> bool:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        return False
    return True


def _signal_child(pid: int, signum: signal.Signals) -> bool:
    descriptor = _pidfd_open(pid)
    try:
        if descriptor is not None:
            _signal_pidfd(descriptor, signum)
            return True
        os.kill(pid, signum)
        return False
    except ProcessLookupError:
        return descriptor is not None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _wait_status_payload(pid: int, status: int) -> dict[str, Any]:
    if os.WIFEXITED(status):
        return {"pid": pid, "kind": "exit", "returncode": os.WEXITSTATUS(status), "raw": status}
    if os.WIFSIGNALED(status):
        return {"pid": pid, "kind": "signal", "returncode": -os.WTERMSIG(status), "raw": status}
    if os.WIFSTOPPED(status):
        return {"pid": pid, "kind": "stopped", "signal": os.WSTOPSIG(status), "raw": status}
    return {"pid": pid, "kind": "unsupported", "raw": status}


def _read_pipes(
    poller: select.poll,
    streams: dict[int, tuple[str, _BoundedLog]],
    timeout_ms: int,
) -> None:
    try:
        events = poller.poll(timeout_ms)
    except InterruptedError:
        return
    for descriptor, event in events:
        if event & select.POLLNVAL:
            raise RuntimeRefused("child output descriptor became invalid")
        if not event & (select.POLLIN | select.POLLHUP | select.POLLERR):
            continue
        while True:
            try:
                value = os.read(descriptor, 64 * 1024)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            if not value:
                with suppress(OSError):
                    poller.unregister(descriptor)
                os.close(descriptor)
                streams.pop(descriptor, None)
                break
            streams[descriptor][1].update(value)


def _drain_descendants(
    evidence: dict[str, Any],
    *,
    term_grace: float,
    cleanup_seconds: float,
) -> None:
    began = time.monotonic()
    term_deadline = began + term_grace
    final_deadline = began + cleanup_seconds
    killed: set[int] = set()
    while True:
        children = _child_pids()
        if children:
            evidence["descendants_observed"] = True
        for child in children:
            evidence["observed_descendant_pids"].add(child)
            if child not in killed:
                pinned = _signal_child(child, signal.SIGTERM)
                evidence["term_signals"].append(
                    {"target": f"pid:{child}", "via": "pidfd" if pinned else "kill"}
                )
                killed.add(child)
        if time.monotonic() >= term_deadline:
            for child in _child_pids():
                evidence["observed_descendant_pids"].add(child)
                pinned = _signal_child(child, signal.SIGKILL)
                evidence["kill_signals"].append(
                    {"target": f"pid:{child}", "via": "pidfd" if pinned else "kill"}
                )
        try:
            waited, status = os.waitpid(-1, os.WNOHANG | os.WUNTRACED | _WAIT_ALL)
        except InterruptedError:
            continue
        except ChildProcessError:
            if not _child_pids():
                evidence["terminal_waitpid_echild_verified"] = True
                return
        else:
            if waited:
                evidence["descendants_observed"] = True
                evidence["observed_descendant_pids"].add(waited)
                waited_payload = _wait_status_payload(waited, status)
                evidence["reaped_descendants"].append(waited_payload)
                if waited_payload["kind"] == "stopped":
                    _signal_child(waited, signal.SIGKILL)
        if time.monotonic() >= final_deadline:
            raise RuntimeRefused("adopted descendants did not reach terminal ECHILD in time")
        time.sleep(_POLL_SECONDS)


def run_contained_command(request: CommandRequest) -> dict[str, Any]:
    """Run one attested Python script or native exec wrapper and record its outcome.

    Successful return requires return code zero, no timeout, no observed
    descendants, terminal ``waitpid(...)=ECHILD``, and byte-for-byte identical
    pre/post runtime attestations.  Otherwise ``CommandExecutionError`` is
    raised after the terminal record has been committed.
    """

    interpreter, cwd, record_path = _validate_request(request)
    with _RUN_LOCK:
        pre = _attest_request(request)
        if (
            request.expected_runtime_sha256 is not None
            and pre["sha256"] != request.expected_runtime_sha256
        ):
            raise RuntimeRefused(
                "runtime identity differs from expected_runtime_sha256: "
                f"expected {request.expected_runtime_sha256}, got {pre['sha256']}"
            )
        script_paths = {Path(item["requested_path"]) for item in pre["files"]}
        if Path(request.argv[1]) not in script_paths:
            raise RuntimeRefused("script identity is absent from the runtime attestation")
        boundary = _require_clean_boundary()
        library = _libc()
        prior_subreaper = _subreaper_state(library)
        if prior_subreaper != 0:
            raise RuntimeRefused("launcher already participates in a child-subreaper boundary")
        started_utc, started_ns = _timestamp()
        reservation = {
            "schema": SCHEMA,
            "status": "started",
            "started_at_utc": started_utc,
            "started_at_unix_ns": started_ns,
            "argv": list(request.argv),
            "cwd": str(cwd),
            "environment": dict(FIXED_ENVIRONMENT),
            "runtime_before": pre,
            "execution_kind": (
                "exec-wrapper" if request.exec_target is not None else "python"
            ),
            "exec_target": (
                None if request.exec_target is None else str(request.exec_target)
            ),
            "expected_runtime_sha256": request.expected_runtime_sha256,
        }
        _write_record(record_path, reservation, exclusive=True)
        stdout_log = _BoundedLog(request.max_log_bytes)
        stderr_log = _BoundedLog(request.max_log_bytes)
        evidence: dict[str, Any] = {
            "pre_fork_boundary": boundary,
            "prior_subreaper_state": prior_subreaper,
            "subreaper_enabled": False,
            "subreaper_restored": False,
            "direct_pid": None,
            "direct_pidfd_available": False,
            "term_signals": [],
            "kill_signals": [],
            "descendants_observed": False,
            "observed_descendant_pids": set(),
            "reaped_descendants": [],
            "terminal_waitpid_echild_verified": False,
            "immediate_preexec_boundary": None,
            "pdeathsig": {
                "signal": "SIGKILL",
                "expected_parent_pid": os.getpid(),
                "configured_and_verified_in_single_thread_preexec": False,
                "parent_race_check": "set-prctl-then-compare-getppid",
            },
        }
        process: subprocess.Popen[bytes] | None = None
        direct_pidfd: int | None = None
        direct_status: int | None = None
        timed_out = False
        failure: str | None = None
        post: dict[str, Any] | None = None
        streams: dict[int, tuple[str, _BoundedLog]] = {}
        poller = select.poll()
        subreaper_enabled = False
        try:
            # A second full attestation closes the gap between reservation and spawn.
            if _attest_request(request) != pre:
                raise RuntimeRefused("runtime mutated before child creation")
            immediate_boundary = _require_clean_boundary()
            if immediate_boundary != boundary:
                raise RuntimeRefused("launcher boundary changed before child creation")
            evidence["immediate_preexec_boundary"] = immediate_boundary
            _set_subreaper(library, 1)
            subreaper_enabled = True
            evidence["subreaper_enabled"] = True
            process = subprocess.Popen(  # noqa: S603 - exact argv and no shell
                list(request.argv),
                cwd=cwd,
                env=dict(FIXED_ENVIRONMENT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
                umask=0o077,
                preexec_fn=partial(_child_preexec_boundary, os.getpid()),
            )
            evidence["pdeathsig"][
                "configured_and_verified_in_single_thread_preexec"
            ] = True
            evidence["direct_pid"] = process.pid
            direct_pidfd = _pidfd_open(process.pid)
            evidence["direct_pidfd_available"] = direct_pidfd is not None
            if process.stdout is None or process.stderr is None:
                raise RuntimeRefused("child output pipes were not created")
            for name, handle, capture in (
                ("stdout", process.stdout, stdout_log),
                ("stderr", process.stderr, stderr_log),
            ):
                descriptor = os.dup(handle.fileno())
                handle.close()
                os.set_blocking(descriptor, False)
                streams[descriptor] = (name, capture)
                poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
            deadline = time.monotonic() + request.timeout_seconds
            term_deadline: float | None = None
            while direct_status is None:
                _read_pipes(poller, streams, int(_POLL_SECONDS * 1000))
                try:
                    waited, status = os.waitpid(
                        process.pid, os.WNOHANG | os.WUNTRACED | _WAIT_ALL
                    )
                except InterruptedError:
                    continue
                except ChildProcessError as exc:
                    raise RuntimeRefused(
                        "direct child vanished without terminal wait status"
                    ) from exc
                if waited == process.pid:
                    if os.WIFSTOPPED(status):
                        failure = f"direct child stopped with signal {os.WSTOPSIG(status)}"
                        _signal_group(process.pid, signal.SIGKILL)
                        _signal_pidfd(direct_pidfd, signal.SIGKILL)
                    elif os.WIFEXITED(status) or os.WIFSIGNALED(status):
                        direct_status = status
                        break
                    else:
                        raise RuntimeRefused("direct child returned an unsupported wait status")
                now = time.monotonic()
                if not timed_out and now >= deadline:
                    timed_out = True
                    term_deadline = now + request.term_grace_seconds
                    if _signal_group(process.pid, signal.SIGTERM):
                        evidence["term_signals"].append(
                            {"target": f"pgid:{process.pid}", "via": "killpg"}
                        )
                    if _signal_pidfd(direct_pidfd, signal.SIGTERM):
                        evidence["term_signals"].append(
                            {"target": f"pid:{process.pid}", "via": "pidfd"}
                        )
                if timed_out and term_deadline is not None and now >= term_deadline:
                    if _signal_group(process.pid, signal.SIGKILL):
                        evidence["kill_signals"].append(
                            {"target": f"pgid:{process.pid}", "via": "killpg"}
                        )
                    if _signal_pidfd(direct_pidfd, signal.SIGKILL):
                        evidence["kill_signals"].append(
                            {"target": f"pid:{process.pid}", "via": "pidfd"}
                        )
                    term_deadline = float("inf")
            if direct_status is None:
                raise RuntimeRefused("direct child has no terminal wait status")
            process.returncode = os.waitstatus_to_exitcode(direct_status)
            _drain_descendants(
                evidence,
                term_grace=request.term_grace_seconds,
                cleanup_seconds=max(request.cleanup_seconds, request.term_grace_seconds + 0.1),
            )
            while streams:
                _read_pipes(poller, streams, 0)
                if streams:
                    raise RuntimeRefused("output pipes remained open after terminal ECHILD")
        except BaseException as exc:
            failure = failure or f"{type(exc).__name__}: {exc}"
            if process is not None and direct_status is None:
                _signal_group(process.pid, signal.SIGKILL)
                _signal_pidfd(direct_pidfd, signal.SIGKILL)
                cleanup_deadline = time.monotonic() + request.cleanup_seconds
                while direct_status is None and time.monotonic() < cleanup_deadline:
                    try:
                        waited, status = os.waitpid(
                            process.pid, os.WNOHANG | os.WUNTRACED | _WAIT_ALL
                        )
                    except (InterruptedError, ChildProcessError):
                        continue
                    if waited == process.pid:
                        if os.WIFSTOPPED(status):
                            _signal_group(process.pid, signal.SIGKILL)
                        else:
                            direct_status = status
                            process.returncode = os.waitstatus_to_exitcode(status)
                    _read_pipes(poller, streams, 0)
                    time.sleep(_POLL_SECONDS)
                if direct_status is None:
                    failure += "; direct child cleanup exceeded deadline"
            if (
                process is not None
                and direct_status is not None
                and not evidence["terminal_waitpid_echild_verified"]
            ):
                try:
                    _drain_descendants(
                        evidence,
                        term_grace=request.term_grace_seconds,
                        cleanup_seconds=max(
                            request.cleanup_seconds, request.term_grace_seconds + 0.1
                        ),
                    )
                except BaseException as cleanup_exc:
                    failure += f"; descendant cleanup failed: {cleanup_exc}"
            while streams:
                try:
                    _read_pipes(poller, streams, 0)
                except BaseException as drain_exc:
                    failure += f"; log drain failed: {drain_exc}"
                    break
                if streams:
                    for descriptor in tuple(streams):
                        with suppress(OSError):
                            os.close(descriptor)
                        streams.pop(descriptor, None)
        finally:
            if direct_pidfd is not None:
                os.close(direct_pidfd)
            if subreaper_enabled:
                try:
                    _set_subreaper(library, prior_subreaper)
                    evidence["subreaper_restored"] = True
                except BaseException as exc:
                    prefix = failure + "; " if failure else ""
                    failure = prefix + f"subreaper restore failed: {exc}"
            for descriptor in tuple(streams):
                with suppress(OSError):
                    os.close(descriptor)
        try:
            post = _attest_request(request)
        except BaseException as exc:
            failure = (failure + "; " if failure else "") + f"post-attestation failed: {exc}"
        mutated = post != pre
        if mutated:
            failure = (failure + "; " if failure else "") + "runtime identity mutated"
        direct = (
            None
            if direct_status is None
            else _wait_status_payload(evidence["direct_pid"], direct_status)
        )
        if direct is not None and direct.get("returncode") != 0:
            prefix = failure + "; " if failure else ""
            failure = prefix + f"command returned {direct['returncode']}"
        if timed_out:
            failure = (failure + "; " if failure else "") + "command timed out"
        if evidence["descendants_observed"]:
            failure = (failure + "; " if failure else "") + "command left an observed descendant"
        if not evidence["terminal_waitpid_echild_verified"]:
            failure = (failure + "; " if failure else "") + "terminal ECHILD was not verified"
        finished_utc, finished_ns = _timestamp()
        evidence["observed_descendant_pids"] = sorted(evidence["observed_descendant_pids"])
        record = {
            "schema": SCHEMA,
            "status": "accepted" if failure is None else "refused",
            "started_at_utc": started_utc,
            "started_at_unix_ns": started_ns,
            "finished_at_utc": finished_utc,
            "finished_at_unix_ns": finished_ns,
            "elapsed_ns": max(0, finished_ns - started_ns),
            "argv": list(request.argv),
            "cwd": str(cwd),
            "environment": dict(FIXED_ENVIRONMENT),
            "limits": {
                "timeout_seconds": request.timeout_seconds,
                "term_grace_seconds": request.term_grace_seconds,
                "cleanup_seconds": request.cleanup_seconds,
                "max_log_bytes": request.max_log_bytes,
            },
            "runtime_before": pre,
            "runtime_after": post,
            "runtime_mutated": mutated,
            "execution_kind": (
                "exec-wrapper" if request.exec_target is not None else "python"
            ),
            "exec_target": (
                None if request.exec_target is None else str(request.exec_target)
            ),
            "expected_runtime_sha256": request.expected_runtime_sha256,
            "process": {
                "started": process is not None,
                "direct": direct,
                "timed_out": timed_out,
                "failure": failure,
            },
            "containment": evidence,
            "stdout": stdout_log.payload(),
            "stderr": stderr_log.payload(),
        }
        raw_without_identity = _canonical_bytes(record)
        record["record_payload"] = {
            "canonical_bytes": len(raw_without_identity),
            "sha256": _digest_bytes(raw_without_identity),
        }
        _replace_record(record_path, record)
        if failure is not None:
            raise CommandExecutionError(failure, record)
        return record


__all__ = [
    "CommandExecutionError",
    "CommandRequest",
    "DEFAULT_ELF_LIBRARY_DIRS",
    "ELF_SCHEMA",
    "FILE_SCHEMA",
    "FIXED_ENVIRONMENT",
    "RUNTIME_SCHEMA",
    "RuntimeRefused",
    "SCHEMA",
    "TREE_SCHEMA",
    "attest_elf_closure",
    "attest_file",
    "attest_runtime",
    "run_contained_command",
]
