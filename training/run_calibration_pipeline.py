#!/usr/bin/env python3
"""Run the pinned GGUF calibration pipeline and commit a historical receipt.

The receipt is the final commit point.  Conversion, importance-matrix creation,
and quantization write to private temporary files in their destination
directories.  Temporary inodes remain mode 0600 until their no-replace hard
links are installed, and only then are the linked inodes made mode 0644.  They
are published only after the exact inputs and tools have been checked again.  A
failed or catchably interrupted run never overwrites a destination and leaves
no final receipt.

This is an execution receipt, not a reproducibility proof.  In particular, it
cannot authenticate ignored llama.cpp build products, the Python environment,
loaded shared libraries, the kernel, or hardware.  It records their relevant
on-disk entry points and fails if those bytes change during the run.

The filesystem safety model assumes exclusive control by the effective UID.  An
effective-UID-owned ancestor with no group/other permission bits (normally mode
0700) is an exclusive trust boundary, so less restrictive descendant directory
modes are safe from different UIDs.  Protected regular files remain forbidden
when group- or world-writable.  Before that boundary, every directory must be
root/effective-UID-owned and not group/world-writable; a root-owned sticky
system anchor such as /tmp is the sole writable exception.  Every component on
a protected pathname is checked for type, ownership, and symlinks.  The private
boundary must predate
the run; previously retained directory descriptors or alternate mounts are
outside this policy.  Another process running as the same effective UID can
race pathname inspection and cleanup; POSIX does not provide an atomic "unlink
only if this inode" operation.  The llama.cpp checkout must itself be below an
effective-UID-owned private boundary; repository-local Git metadata is trusted
within that boundary.  Inherited Git redirection/configuration variables and
known config-driven helpers are disabled, but this is not a sandbox against a
malicious same-UID checkout.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA = "microtensor.calibration-execution-receipt.v1"
LLAMA_CPP_REVISION = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
TREE_ALGORITHM = "sorted_nfc_relative_path_nul_sha256_nul_bytes_nul_v1"
DEFAULT_CAPTURE_LIMIT_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_GIT_CONTROL_FILE_BYTES = 1024 * 1024
_READ_CHUNK = 4 * 1024 * 1024
_CAPTURE_JOIN_TIMEOUT_SECONDS = 30.0
_CAPTURE_KILL_JOIN_TIMEOUT_SECONDS = 5.0
_PROCESS_GROUP_GRACE_SECONDS = 2.0
_PROCESS_GROUP_POLL_SECONDS = 0.05
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
}
_GIT_ENVIRONMENT = {
    "GIT_ALLOW_PROTOCOL": "",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_PAGER": "cat",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
}
_CATCHABLE_CLEANUP_SIGNALS = frozenset(
    signum
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    if signum is not None
)


class PipelineError(RuntimeError):
    """Raised before the receipt commit point when an invariant fails."""


class PipelineInterrupted(PipelineError):
    """Raised for catchable termination signals so rollback can run."""


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    bytes: int
    sha256: str
    device: int
    inode: int
    mode: str
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class DirectorySnapshot:
    path: str
    device: int
    inode: int
    mode: str
    mtime_ns: int
    ctime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class TreeSnapshot:
    path: str
    tree_sha256: str
    total_bytes: int
    directories: tuple[DirectorySnapshot, ...]
    files: tuple[FileSnapshot, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "tree_algorithm": TREE_ALGORITHM,
            "tree_sha256": self.tree_sha256,
            "total_bytes": self.total_bytes,
            "directories": [entry.as_dict() for entry in self.directories],
            "files": [entry.as_dict() for entry in self.files],
        }


@dataclass(frozen=True)
class StreamDigest:
    bytes: int
    sha256: str
    captured_bytes: int
    capture_limit_bytes: int
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "sha256": self.sha256,
            "captured_bytes": self.captured_bytes,
            "capture_limit_bytes": self.capture_limit_bytes,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class CommandRecord:
    name: str
    argv: tuple[str, ...]
    started_at_utc: str
    started_at_unix_ns: int
    finished_at_utc: str
    finished_at_unix_ns: int
    returncode: int
    stdout: StreamDigest
    stderr: StreamDigest

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "started_at_utc": self.started_at_utc,
            "started_at_unix_ns": self.started_at_unix_ns,
            "finished_at_utc": self.finished_at_utc,
            "finished_at_unix_ns": self.finished_at_unix_ns,
            "returncode": self.returncode,
            "stdout": self.stdout.as_dict(),
            "stderr": self.stderr.as_dict(),
        }


@dataclass(frozen=True)
class PipelineRequest:
    source_model_dir: Path
    training_metadata: Path
    calibration_corpus: Path
    corpus_metadata: Path
    converted_model: Path
    imatrix: Path
    quantized_artifact: Path
    receipt: Path
    llama_cpp_dir: Path
    python_executable: Path = field(
        default_factory=lambda: Path(sys.executable).resolve()
    )
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES


@dataclass(frozen=True)
class PipelineResult:
    receipt: Path
    receipt_sha256: str
    outputs: Mapping[str, FileSnapshot]
    post_run_integrity_confirmed: bool
    durability_confirmed: bool


@dataclass
class _CaptureState:
    limit: int
    digest: Any
    captured: bytearray
    count: int = 0
    error: BaseException | None = None

    @classmethod
    def create(cls, limit: int) -> _CaptureState:
        return cls(limit=limit, digest=hashlib.sha256(), captured=bytearray())

    def result(self) -> StreamDigest:
        return StreamDigest(
            bytes=self.count,
            sha256="sha256:" + self.digest.hexdigest(),
            captured_bytes=len(self.captured),
            capture_limit_bytes=self.limit,
            truncated=self.count > len(self.captured),
        )


@dataclass
class _ProcessResult:
    returncode: int
    stdout: _CaptureState
    stderr: _CaptureState


@dataclass
class _TemporaryOutput:
    path: Path
    final: Path
    created_identity: tuple[int, int]
    descriptor: int = -1
    snapshot: FileSnapshot | None = None


CommandRunner = Callable[
    [str, Sequence[str], Mapping[str, str], int], CommandRecord
]


def _timestamp() -> tuple[str, int]:
    unix_ns = time.time_ns()
    rendered = (
        datetime.fromtimestamp(unix_ns / 1_000_000_000, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    return rendered, unix_ns


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _mode(details: os.stat_result) -> str:
    return f"0o{stat.S_IMODE(details.st_mode):04o}"


def _fingerprint(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _absolute(path: Path, label: str) -> Path:
    supplied = Path(path)
    if "\x00" in os.fspath(supplied):
        raise PipelineError(f"{label} contains a NUL byte")
    if any(part == ".." for part in supplied.parts):
        raise PipelineError(f"{label} must not contain traversal")
    return Path(os.path.abspath(os.fspath(supplied)))


def _reject_symlink_components(path: Path, label: str, *, leaf_may_be_missing: bool) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            if leaf_may_be_missing and index == len(parts) - 1:
                return
            raise PipelineError(
                f"{label} has an unavailable path component: {current}"
            ) from None
        except OSError as exc:
            raise PipelineError(f"{label} path cannot be inspected: {current}") from exc
        if stat.S_ISLNK(details.st_mode):
            raise PipelineError(f"{label} contains a symlink component: {current}")


def _validate_directory_chain(path: Path, label: str) -> bool:
    """Validate every directory and report a mode-0700 trust boundary."""

    absolute = _absolute(path, label)
    current = Path(absolute.anchor)
    directories = [current]
    for part in absolute.parts[1:]:
        current /= part
        directories.append(current)

    exclusive_boundary = False
    effective_uid = os.geteuid()
    for directory in directories:
        try:
            details = directory.lstat()
        except OSError as exc:
            raise PipelineError(
                f"{label} has an unavailable directory component: {directory}"
            ) from exc
        if stat.S_ISLNK(details.st_mode):
            raise PipelineError(f"{label} contains a symlink component: {directory}")
        if not stat.S_ISDIR(details.st_mode):
            raise PipelineError(f"{label} path component is not a directory: {directory}")
        if details.st_uid not in {0, effective_uid}:
            raise PipelineError(
                f"{label} directory component is not owned by root or the effective user: "
                f"{directory}"
            )
        permissions = stat.S_IMODE(details.st_mode)
        if details.st_uid == effective_uid and permissions & 0o077 == 0:
            exclusive_boundary = True
        root_sticky_anchor = details.st_uid == 0 and bool(
            permissions & stat.S_ISVTX
        )
        if permissions & 0o022 and not exclusive_boundary and not root_sticky_anchor:
            raise PipelineError(
                f"{label} directory component is group- or world-writable before an "
                f"exclusive trust boundary: {directory}"
            )
    return exclusive_boundary


def _existing_path(path: Path, label: str, *, directory: bool = False) -> Path:
    absolute = _absolute(path, label)
    _reject_symlink_components(absolute, label, leaf_may_be_missing=False)
    _validate_directory_chain(absolute if directory else absolute.parent, label)
    try:
        details = absolute.lstat()
    except OSError as exc:
        raise PipelineError(f"{label} is unavailable: {absolute}") from exc
    expected = stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise PipelineError(f"{label} must be a non-symlink {kind}: {absolute}")
    return absolute


def _controlled_directory(path: Path, label: str) -> Path:
    return _existing_path(path, label, directory=True)


def _destination(path: Path, label: str) -> Path:
    absolute = _absolute(path, label)
    if absolute.name in {"", ".", ".."} or (
        unicodedata.normalize("NFC", absolute.name) != absolute.name
    ):
        raise PipelineError(f"{label} must have a normalized filename")
    parent = _controlled_directory(absolute.parent, f"{label} parent")
    destination = parent / absolute.name
    try:
        destination.lstat()
    except FileNotFoundError:
        return destination
    except OSError as exc:
        raise PipelineError(f"{label} cannot be inspected") from exc
    raise PipelineError(f"{label} already exists; refusing to overwrite it")


def _directory_snapshot(path: Path, relative: str) -> DirectorySnapshot:
    _validate_directory_chain(path, f"source model directory {relative}")
    try:
        details = path.lstat()
    except OSError as exc:
        raise PipelineError(f"directory changed during inspection: {path}") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise PipelineError(f"tree entry is not a non-symlink directory: {path}")
    return DirectorySnapshot(
        path=relative,
        device=details.st_dev,
        inode=details.st_ino,
        mode=_mode(details),
        mtime_ns=details.st_mtime_ns,
        ctime_ns=details.st_ctime_ns,
    )


def _snapshot_file(
    path: Path, label: str, *, executable: bool = False, trusted: bool = False
) -> FileSnapshot:
    absolute = _existing_path(path, label)
    _validate_directory_chain(absolute.parent, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        before = absolute.lstat()
        descriptor = os.open(absolute, flags)
        opened = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(opened):
            raise PipelineError(f"{label} changed while it was opened")
        if (trusted or executable) and opened.st_uid not in {0, os.geteuid()}:
            raise PipelineError(f"{label} is not owned by root or the effective user")
        if executable and not (stat.S_IMODE(opened.st_mode) & 0o111):
            raise PipelineError(f"{label} is not executable")
        if (trusted or executable) and stat.S_IMODE(opened.st_mode) & 0o022:
            raise PipelineError(f"{label} must not be group- or world-writable")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_CHUNK):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_after = absolute.lstat()
        if (
            _fingerprint(after) != _fingerprint(opened)
            or _fingerprint(named_after) != _fingerprint(opened)
        ):
            raise PipelineError(f"{label} changed while it was hashed")
        return FileSnapshot(
            path=str(absolute),
            bytes=opened.st_size,
            sha256="sha256:" + digest.hexdigest(),
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=_mode(opened),
            mtime_ns=opened.st_mtime_ns,
            ctime_ns=opened.st_ctime_ns,
        )
    except OSError as exc:
        raise PipelineError(f"{label} could not be read safely: {absolute}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _snapshot_tree(path: Path, label: str) -> TreeSnapshot:
    root = _controlled_directory(path, label)
    directories: list[DirectorySnapshot] = []
    files: list[FileSnapshot] = []
    before_directories: dict[Path, DirectorySnapshot] = {}

    def visit(directory: Path, relative: Path) -> None:
        relative_text = "." if not relative.parts else relative.as_posix()
        if unicodedata.normalize("NFC", relative_text) != relative_text:
            raise PipelineError(f"{label} contains a non-NFC directory name")
        before = _directory_snapshot(directory, relative_text)
        before_directories[directory] = before
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise PipelineError(f"{label} cannot be inventoried: {directory}") from exc
        for entry in entries:
            entry_relative = relative / entry.name
            relative_name = entry_relative.as_posix()
            if unicodedata.normalize("NFC", relative_name) != relative_name:
                raise PipelineError(f"{label} contains a non-NFC entry name")
            try:
                details = entry.lstat()
            except OSError as exc:
                raise PipelineError(f"{label} entry is unavailable: {entry}") from exc
            if stat.S_ISLNK(details.st_mode):
                raise PipelineError(f"{label} contains a symlink: {entry}")
            if stat.S_ISDIR(details.st_mode):
                visit(entry, entry_relative)
            elif stat.S_ISREG(details.st_mode):
                snapshot = _snapshot_file(
                    entry, f"{label} file {relative_name}", trusted=True
                )
                files.append(
                    FileSnapshot(
                        path=relative_name,
                        bytes=snapshot.bytes,
                        sha256=snapshot.sha256,
                        device=snapshot.device,
                        inode=snapshot.inode,
                        mode=snapshot.mode,
                        mtime_ns=snapshot.mtime_ns,
                        ctime_ns=snapshot.ctime_ns,
                    )
                )
            else:
                raise PipelineError(f"{label} contains a non-regular entry: {entry}")

    visit(root, Path())
    for directory, before in sorted(
        before_directories.items(), key=lambda item: item[1].path
    ):
        after = _directory_snapshot(directory, before.path)
        if after != before:
            raise PipelineError(f"{label} directory changed during inventory: {directory}")
        directories.append(before)
    files.sort(key=lambda entry: entry.path)
    if not files:
        raise PipelineError(f"{label} is empty")
    top_level_names = {entry.path for entry in files if "/" not in entry.path}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(top_level_names) or not any(
        name.endswith(".safetensors") for name in top_level_names
    ):
        raise PipelineError(f"{label} does not contain a complete merged model")
    tree_digest = hashlib.sha256()
    total = 0
    for entry in files:
        total += entry.bytes
        tree_digest.update(entry.path.encode("utf-8"))
        tree_digest.update(b"\x00")
        tree_digest.update(entry.sha256.encode("ascii"))
        tree_digest.update(b"\x00")
        tree_digest.update(str(entry.bytes).encode("ascii"))
        tree_digest.update(b"\x00")
    return TreeSnapshot(
        path=str(root),
        tree_sha256="sha256:" + tree_digest.hexdigest(),
        total_bytes=total,
        directories=tuple(directories),
        files=tuple(files),
    )


def _drain(stream: BinaryIO, state: _CaptureState) -> None:
    try:
        while chunk := stream.read(64 * 1024):
            state.digest.update(chunk)
            state.count += len(chunk)
            remaining = state.limit - len(state.captured)
            if remaining > 0:
                state.captured.extend(chunk[:remaining])
    except BaseException as exc:  # surfaced on the controlling thread
        state.error = exc
    finally:
        try:
            stream.close()
        except BaseException as exc:
            if state.error is None:
                state.error = exc


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> bool:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        elif process.poll() is None:  # pragma: no cover - Linux production
            if signum == signal.SIGTERM:
                process.terminate()
            elif signum == signal.SIGKILL:
                process.kill()
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise PipelineError(
            f"could not signal command process group with signal {signum}"
        ) from exc
    return True


def _process_group_exists(process: subprocess.Popen[bytes]) -> bool:
    if os.name == "posix":
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except OSError as exc:
            raise PipelineError("could not probe command process group") from exc
        return True
    return process.poll() is None  # pragma: no cover - Linux production


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes], timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(process):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PROCESS_GROUP_POLL_SECONDS, remaining))
    return True


def _stop_process_group(process: subprocess.Popen[bytes]) -> bool:
    existed = _process_group_exists(process)
    if not existed:
        return False
    _signal_process_group(process, signal.SIGTERM)
    with suppress(OSError, subprocess.SubprocessError):
        process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
    if not _wait_for_process_group_exit(process, _PROCESS_GROUP_GRACE_SECONDS):
        _signal_process_group(process, signal.SIGKILL)
        with suppress(OSError, subprocess.SubprocessError):
            process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
        if not _wait_for_process_group_exit(process, _PROCESS_GROUP_GRACE_SECONDS):
            raise PipelineError("command process group survived SIGKILL")
    return True


def _join_capture_threads(
    threads: Sequence[threading.Thread], timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    for thread in threads:
        try:
            thread.join(max(0.0, deadline - time.monotonic()))
        except RuntimeError:
            if thread.is_alive():
                raise
    return not any(thread.is_alive() for thread in threads)


def _close_capture_fds(process: subprocess.Popen[bytes]) -> tuple[str, ...]:
    errors: list[str] = []
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        try:
            descriptor = stream.fileno()
        except (AttributeError, OSError, ValueError):
            try:
                stream.close()
            except BaseException as exc:
                errors.append(f"could not close command {name}: {exc}")
            continue
        try:
            os.close(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                errors.append(f"could not close command {name} fd: {exc}")
    return tuple(errors)


def _close_capture_stream_objects(
    process: subprocess.Popen[bytes],
) -> tuple[str, ...]:
    errors: list[str] = []
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError as exc:
            if exc.errno != errno.EBADF:
                errors.append(f"could not finalize command {name}: {exc}")
        except BaseException as exc:
            errors.append(f"could not finalize command {name}: {exc}")
    return tuple(errors)


def _finish_capture_threads(
    process: subprocess.Popen[bytes], threads: Sequence[threading.Thread]
) -> tuple[bool, tuple[str, ...]]:
    joined_without_force = _join_capture_threads(threads, _CAPTURE_JOIN_TIMEOUT_SECONDS)
    errors = list(_close_capture_fds(process))
    threads_stopped = _join_capture_threads(
        threads, _CAPTURE_KILL_JOIN_TIMEOUT_SECONDS
    )
    if not threads_stopped:
        errors.append("both command capture threads did not terminate")
    else:
        errors.extend(_close_capture_stream_objects(process))
    return joined_without_force, tuple(errors)


@contextmanager
def _blocked_cleanup_signals() -> Any:
    """Defer catchable signals across ownership-transfer instruction windows."""

    if os.name != "posix" or not hasattr(signal, "pthread_sigmask"):
        yield
        return
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK, _CATCHABLE_CLEANUP_SIGNALS
    )
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _restore_child_cleanup_signals() -> None:
    if os.name == "posix" and hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(
            signal.SIG_UNBLOCK, _CATCHABLE_CLEANUP_SIGNALS
        )


def _execute_argv(
    argv: Sequence[str],
    environment: Mapping[str, str],
    capture_limit: int,
    *,
    clear_environment_prefixes: Sequence[str] = (),
) -> _ProcessResult:
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise PipelineError("command argv must be a non-empty array of strings")
    if os.name == "posix" and threading.active_count() != 1:
        raise PipelineError(
            "command execution requires a single live Python thread at spawn"
        )
    merged_environment = {
        name: value
        for name, value in os.environ.items()
        if not any(name.startswith(prefix) for prefix in clear_environment_prefixes)
    }
    merged_environment.update(environment)
    process: subprocess.Popen[bytes] | None = None
    stdout: _CaptureState | None = None
    stderr: _CaptureState | None = None
    threads: tuple[threading.Thread, ...] = ()
    started_threads: list[threading.Thread] = []
    try:
        with _blocked_cleanup_signals():
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                start_new_session=(os.name == "posix"),
                env=merged_environment,
                preexec_fn=(
                    _restore_child_cleanup_signals if os.name == "posix" else None
                ),
            )
        if process.stdout is None or process.stderr is None:
            raise PipelineError("command capture pipes were not created")
        stdout = _CaptureState.create(capture_limit)
        stderr = _CaptureState.create(capture_limit)
        threads = (
            threading.Thread(
                target=_drain, args=(process.stdout, stdout), daemon=True
            ),
            threading.Thread(
                target=_drain, args=(process.stderr, stderr), daemon=True
            ),
        )
        for thread in threads:
            started_threads.append(thread)
            thread.start()
        returncode = process.wait()
    except BaseException as exc:
        if process is None:
            if isinstance(exc, OSError):
                raise PipelineError(f"command could not start: {argv[0]}") from exc
            raise
        cleanup_errors: list[str] = []
        try:
            _stop_process_group(process)
        except BaseException as cleanup_exc:
            cleanup_errors.append(f"process-group cleanup failed: {cleanup_exc}")
        _, thread_cleanup_errors = _finish_capture_threads(
            process, started_threads
        )
        cleanup_errors.extend(thread_cleanup_errors)
        if cleanup_errors:
            raise PipelineError(
                "command cleanup was incomplete: " + "; ".join(cleanup_errors)
            ) from exc
        raise

    if process is None or stdout is None or stderr is None:
        raise PipelineError("command process or capture state is unavailable")
    lingering_descendants = False
    cleanup_errors = []
    try:
        lingering_descendants = _process_group_exists(process)
        if lingering_descendants:
            _stop_process_group(process)
    except BaseException as cleanup_exc:
        cleanup_errors.append(f"process-group cleanup failed: {cleanup_exc}")
    capture_closed_without_force, thread_cleanup_errors = _finish_capture_threads(
        process, started_threads
    )
    cleanup_errors.extend(thread_cleanup_errors)
    if cleanup_errors:
        raise PipelineError(
            "command cleanup was incomplete: " + "; ".join(cleanup_errors)
        )
    if lingering_descendants:
        raise PipelineError(
            "command left descendant processes after its leader exited"
        )
    if not capture_closed_without_force:
        raise PipelineError(
            "command output streams did not close within the capture timeout"
        )
    if stdout.error is not None or stderr.error is not None:
        raise PipelineError("command output streams could not be captured safely")
    return _ProcessResult(returncode, stdout, stderr)


def _run_command(
    name: str,
    argv: Sequence[str],
    environment: Mapping[str, str],
    capture_limit: int,
) -> CommandRecord:
    started_at, started_ns = _timestamp()
    result = _execute_argv(argv, environment, capture_limit)
    finished_at, finished_ns = _timestamp()
    return CommandRecord(
        name=name,
        argv=tuple(argv),
        started_at_utc=started_at,
        started_at_unix_ns=started_ns,
        finished_at_utc=finished_at,
        finished_at_unix_ns=finished_ns,
        returncode=result.returncode,
        stdout=result.stdout.result(),
        stderr=result.stderr.result(),
    )


def _trusted_git_control_payload(
    path: Path, label: str, *, required: bool
) -> bytes | None:
    try:
        path.lstat()
    except FileNotFoundError as exc:
        if required:
            raise PipelineError(f"{label} is required") from exc
        return None
    except OSError as exc:
        raise PipelineError(f"{label} cannot be inspected") from exc
    before = _snapshot_file(path, label, trusted=True)
    if before.bytes > MAX_GIT_CONTROL_FILE_BYTES:
        raise PipelineError(f"{label} exceeds its byte limit")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PipelineError(f"{label} cannot be read") from exc
    after = _snapshot_file(path, label, trusted=True)
    if (
        before != after
        or len(payload) != before.bytes
        or _digest_bytes(payload) != before.sha256
    ):
        raise PipelineError(f"{label} changed while it was inspected")
    return payload


def _canonical_git_config_keys(payload: bytes, label: str) -> tuple[str, ...]:
    """Parse enough Git config grammar to fail closed on security-relevant keys."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineError(f"{label} must be UTF-8") from exc
    section: str | None = None
    keys: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        if stripped.endswith("\\"):
            raise PipelineError(
                f"{label} may not use line continuations (line {line_number})"
            )
        if stripped.startswith("["):
            closing = stripped.find("]")
            if closing < 0:
                raise PipelineError(
                    f"{label} has an invalid section at line {line_number}"
                )
            trailing = stripped[closing + 1 :].lstrip()
            if trailing and trailing[0] not in "#;":
                raise PipelineError(
                    f"{label} has trailing section data at line {line_number}"
                )
            header = stripped[1:closing].strip()
            parts = header.split(None, 1)
            base = parts[0].lower() if parts else ""
            if not base or any(
                not (
                    character.isascii()
                    and (character.isalnum() or character in ".-")
                )
                for character in base
            ):
                raise PipelineError(
                    f"{label} has an invalid section name at line {line_number}"
                )
            if len(parts) == 1:
                section = base
            else:
                subsection = parts[1].strip()
                if (
                    len(subsection) < 2
                    or subsection[0] != '"'
                    or subsection[-1] != '"'
                ):
                    raise PipelineError(
                        f"{label} has an invalid subsection at line {line_number}"
                    )
                section = f"{base}.{subsection[1:-1].lower()}"
            continue
        if section is None:
            raise PipelineError(
                f"{label} has a key outside a section at line {line_number}"
            )
        key = stripped.split("=", 1)[0].strip().lower()
        if (
            not key
            or not key[0].isascii()
            or not key[0].isalpha()
            or any(
                not (
                    character.isascii()
                    and (character.isalnum() or character == "-")
                )
                for character in key
            )
        ):
            raise PipelineError(f"{label} has an invalid key at line {line_number}")
        keys.append(f"{section}.{key}")
    return tuple(keys)


def _git_config_key_selects_helper(key: str) -> bool:
    if key.startswith(
        (
            "credential.",
            "difftool.",
            "filter.",
            "include.",
            "includeif.",
            "mergetool.",
            "pager.",
            "protocol.",
            "url.",
        )
    ):
        return True
    if key in {
        "core.alternaterefscommand",
        "core.askpass",
        "core.attributesfile",
        "core.editor",
        "core.fsmonitor",
        "core.gitproxy",
        "core.hookspath",
        "core.pager",
        "core.sshcommand",
        "diff.external",
        "gpg.program",
        "interactive.difffilter",
        "sequence.editor",
    }:
        return True
    if key.startswith("diff.") and key.endswith(".command"):
        return True
    if key.startswith("gpg.") and key.endswith(".program"):
        return True
    if key.startswith("merge.") and key.endswith(".driver"):
        return True
    return key.startswith("remote.") and key.endswith(
        (".proxy", ".receivepack", ".uploadpack", ".vcs")
    )


def _validate_git_control_plane(repository: Path) -> None:
    """Reject Git metadata paths and configuration that escape the private root."""

    metadata = _controlled_directory(repository / ".git", "llama.cpp Git metadata")
    effective_uid = os.geteuid()
    stack = [metadata]
    forbidden_redirections = {
        "commondir",
        "info/grafts",
        "objects/info/alternates",
        "objects/info/http-alternates",
    }
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            raise PipelineError(
                f"llama.cpp Git metadata cannot be inventoried: {directory}"
            ) from exc
        for entry in entries:
            relative = entry.relative_to(metadata).as_posix()
            try:
                details = entry.lstat()
            except OSError as exc:
                raise PipelineError(
                    f"llama.cpp Git metadata entry is unavailable: {entry}"
                ) from exc
            if stat.S_ISLNK(details.st_mode):
                raise PipelineError(
                    f"llama.cpp Git metadata contains a symlink: {entry}"
                )
            if details.st_uid not in {0, effective_uid}:
                raise PipelineError(
                    f"llama.cpp Git metadata entry has an untrusted owner: {entry}"
                )
            if stat.S_ISDIR(details.st_mode):
                stack.append(entry)
                continue
            if not stat.S_ISREG(details.st_mode):
                raise PipelineError(
                    f"llama.cpp Git metadata contains a special file: {entry}"
                )
            if stat.S_IMODE(details.st_mode) & 0o022:
                raise PipelineError(
                    f"llama.cpp Git metadata file must not be group- or "
                    f"world-writable: {entry}"
                )
            if relative in forbidden_redirections:
                raise PipelineError(
                    f"llama.cpp Git metadata redirection is forbidden: {entry}"
                )

    config_paths = (metadata / "config", metadata / "config.worktree")
    for index, config_path in enumerate(config_paths):
        label = f"llama.cpp Git config {config_path.name}"
        payload = _trusted_git_control_payload(
            config_path,
            label,
            required=index == 0,
        )
        if payload is None:
            continue
        if b"\x00" in payload:
            raise PipelineError(f"{label} contains a NUL byte")
        config_keys = _canonical_git_config_keys(payload, label)
        if any(
            "partialclone" in key or key.endswith(".promisor")
            for key in config_keys
        ):
            raise PipelineError(
                f"{label} may not enable partial-clone/promisor behavior"
            )
        if any(_git_config_key_selects_helper(key) for key in config_keys):
            raise PipelineError(f"{label} may not select external helpers")

    attribute_paths = {metadata / "info" / "attributes"}
    for directory, directory_names, file_names in os.walk(
        repository, followlinks=False
    ):
        if Path(directory) == repository:
            directory_names[:] = [
                name for name in directory_names if name != ".git"
            ]
        if ".gitattributes" in file_names:
            attribute_paths.add(Path(directory) / ".gitattributes")
    for attribute_path in sorted(attribute_paths):
        payload = _trusted_git_control_payload(
            attribute_path,
            f"llama.cpp Git attributes {attribute_path}",
            required=attribute_path.name == ".gitattributes",
        )
        if payload is not None and b"filter" in payload.lower():
            raise PipelineError(
                f"llama.cpp Git attributes may not select filters: {attribute_path}"
            )


def _git_output(git: Path, repository: Path, arguments: Sequence[str]) -> str:
    argv = [
        str(git),
        "--no-pager",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.askPass=",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.bare=false",
        "-c",
        f"core.worktree={repository}",
        "-c",
        "credential.helper=",
        "-c",
        "credential.interactive=never",
        "-c",
        "submodule.recurse=false",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-C",
        str(repository),
        *arguments,
    ]
    result = _execute_argv(
        argv,
        _GIT_ENVIRONMENT,
        DEFAULT_CAPTURE_LIMIT_BYTES,
        clear_environment_prefixes=("GIT_",),
    )
    if result.returncode != 0:
        raise PipelineError(f"llama.cpp Git inspection failed: {' '.join(arguments)}")
    if result.stdout.count > len(result.stdout.captured):
        raise PipelineError("llama.cpp Git inspection output exceeded its limit")
    try:
        return bytes(result.stdout.captured).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise PipelineError("llama.cpp Git inspection output is not UTF-8") from exc


def _validate_checkout(repository: Path, git: Path) -> None:
    _validate_git_control_plane(repository)
    revision = _git_output(git, repository, ("rev-parse", "--verify", "HEAD"))
    if revision != LLAMA_CPP_REVISION:
        raise PipelineError(
            f"llama.cpp revision must be exactly {LLAMA_CPP_REVISION}, got {revision!r}"
        )
    status_output = _git_output(
        git,
        repository,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=all",
        ),
    )
    if status_output:
        raise PipelineError("llama.cpp checkout must have a clean worktree and index")


def _make_temporary_output(
    final: Path,
    registry: list[_TemporaryOutput],
    *,
    hold_open: bool = False,
) -> _TemporaryOutput:
    created: tuple[int, str] | None = None
    temporary: _TemporaryOutput | None = None
    try:
        with _blocked_cleanup_signals():
            created = tempfile.mkstemp(
                prefix=f".{final.name}.", suffix=".partial", dir=final.parent
            )
            descriptor, name = created
            details = os.fstat(descriptor)
            temporary = _TemporaryOutput(
                path=Path(name),
                final=final,
                created_identity=(details.st_dev, details.st_ino),
                descriptor=descriptor,
            )
            registry.append(temporary)
        os.fchmod(descriptor, 0o600)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != temporary.created_identity:
            raise PipelineError("temporary output inode changed during creation")
        if not hold_open:
            try:
                os.close(temporary.descriptor)
            finally:
                temporary.descriptor = -1
        return temporary
    except BaseException as exc:
        if created is not None and temporary is None:
            descriptor, name = created
            try:
                details = os.fstat(descriptor)
            except OSError:
                try:
                    details = Path(name).lstat()
                except OSError as inspect_exc:
                    try:
                        os.close(descriptor)
                    except OSError as close_exc:
                        raise PipelineError(
                            "temporary creation failed and its descriptor could not close"
                        ) from close_exc
                    raise PipelineError(
                        "temporary creation failed before its inode could be registered"
                    ) from inspect_exc
            temporary = _TemporaryOutput(
                path=Path(name),
                final=final,
                created_identity=(details.st_dev, details.st_ino),
                descriptor=descriptor,
            )
            registry.append(temporary)
        cleanup_errors = (
            _cleanup_temporary_records((temporary,)) if temporary is not None else ()
        )
        if cleanup_errors:
            raise PipelineError(
                "temporary creation cleanup was incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        if isinstance(exc, OSError):
            raise PipelineError("temporary output creation failed") from exc
        raise


def _hold_completed_output(temporary: _TemporaryOutput, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(temporary.path, flags)
        details = os.fstat(descriptor)
        named = temporary.path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or (details.st_dev, details.st_ino) != temporary.created_identity
            or _fingerprint(details) != _fingerprint(named)
        ):
            raise PipelineError(f"{label} temporary inode was replaced")
        if os.pread(descriptor, 4, 0) != b"GGUF":
            raise PipelineError(f"{label} output is not a GGUF file")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < details.st_size:
            chunk = os.pread(descriptor, min(_READ_CHUNK, details.st_size - offset), offset)
            if not chunk:
                raise PipelineError(f"{label} ended while it was hashed")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        named_after = temporary.path.lstat()
        if _fingerprint(after) != _fingerprint(details) or _fingerprint(
            named_after
        ) != _fingerprint(details):
            raise PipelineError(f"{label} changed while it was hashed")
        temporary.descriptor = descriptor
        temporary.snapshot = FileSnapshot(
            path=str(temporary.final),
            bytes=details.st_size,
            sha256="sha256:" + digest.hexdigest(),
            device=details.st_dev,
            inode=details.st_ino,
            mode=_mode(details),
            mtime_ns=details.st_mtime_ns,
            ctime_ns=details.st_ctime_ns,
        )
        descriptor = -1
    except OSError as exc:
        raise PipelineError(f"{label} could not be held safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _make_held_inode_readable(temporary: _TemporaryOutput, label: str) -> None:
    if temporary.descriptor < 0:
        raise PipelineError(f"{label} held descriptor is unavailable")
    try:
        before = os.fstat(temporary.descriptor)
        if (before.st_dev, before.st_ino) != temporary.created_identity:
            raise PipelineError(f"{label} held inode changed before publication")
        os.fchmod(temporary.descriptor, 0o644)
        os.fsync(temporary.descriptor)
        after = os.fstat(temporary.descriptor)
        if (
            (after.st_dev, after.st_ino) != temporary.created_identity
            or stat.S_IMODE(after.st_mode) != 0o644
        ):
            raise PipelineError(f"{label} held inode publication failed")
    except OSError as exc:
        raise PipelineError(f"{label} held inode could not be published") from exc


def _link_descriptor_no_replace(descriptor: int, destination: Path) -> tuple[int, int]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    proc_directory = -1
    destination_directory = -1
    try:
        proc_directory = os.open("/proc/self/fd", directory_flags)
        destination_directory = os.open(destination.parent, directory_flags)
        details = os.fstat(descriptor)
        os.link(
            str(descriptor),
            destination.name,
            src_dir_fd=proc_directory,
            dst_dir_fd=destination_directory,
            follow_symlinks=True,
        )
        installed = os.stat(
            destination.name,
            dir_fd=destination_directory,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(installed.st_mode) or (
            installed.st_dev,
            installed.st_ino,
        ) != (details.st_dev, details.st_ino):
            raise PipelineError("installed path does not name the held inode")
        return details.st_dev, details.st_ino
    except FileExistsError as exc:
        raise PipelineError(
            f"destination appeared during execution; refusing to overwrite: {destination}"
        ) from exc
    except OSError as exc:
        raise PipelineError(f"could not install held inode at {destination}") from exc
    finally:
        if proc_directory >= 0:
            os.close(proc_directory)
        if destination_directory >= 0:
            os.close(destination_directory)


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (details.st_dev, details.st_ino) != identity:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _verify_output(final: Path, expected: FileSnapshot, label: str) -> FileSnapshot:
    actual = _snapshot_file(final, label)
    if (
        actual.identity != expected.identity
        or actual.bytes != expected.bytes
        or actual.sha256 != expected.sha256
        or actual.mode != "0o0644"
    ):
        raise PipelineError(f"{label} installed bytes or identity do not match")
    with final.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise PipelineError(f"{label} installed output is not GGUF")
    return actual


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


def _cleanup_temporary_records(
    records: Sequence[_TemporaryOutput],
) -> tuple[str, ...]:
    """Close, identity-unlink, and durably account for every temporary."""

    errors: list[str] = []
    parents: set[Path] = set()
    for temporary in reversed(tuple(records)):
        parents.add(temporary.path.parent)
        if temporary.descriptor >= 0:
            try:
                os.close(temporary.descriptor)
            except OSError as exc:
                errors.append(f"could not close temporary {temporary.path}: {exc}")
            finally:
                temporary.descriptor = -1
        if not _unlink_if_identity(temporary.path, temporary.created_identity):
            errors.append(
                f"could not remove temporary {temporary.path} with expected inode "
                f"{temporary.created_identity}"
            )
    for parent in sorted(parents):
        try:
            _fsync_directory(parent)
        except OSError as exc:
            errors.append(f"could not fsync temporary directory {parent}: {exc}")
    return tuple(errors)


def _rollback_destinations(
    candidates: Mapping[Path, tuple[int, int]],
) -> tuple[str, ...]:
    """Remove only this run's inodes and durably report cleanup failures."""

    errors: list[str] = []
    parents: set[Path] = set()
    for path, identity in reversed(tuple(candidates.items())):
        parents.add(path.parent)
        try:
            current = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"could not inspect rollback destination {path}: {exc}")
            continue
        if (current.st_dev, current.st_ino) != identity:
            # A foreign no-clobber winner means this run's inode was not here.
            continue
        if not _unlink_if_identity(path, identity):
            errors.append(f"could not remove {path} with expected inode {identity}")
    for parent in sorted(parents):
        try:
            _fsync_directory(parent)
        except OSError as exc:
            errors.append(f"could not fsync rollback directory {parent}: {exc}")
    return tuple(errors)


def _same_snapshot(before: FileSnapshot, after: FileSnapshot, label: str) -> None:
    if before != after:
        raise PipelineError(f"{label} changed during pipeline execution")


def _invoke(
    runner: CommandRunner,
    name: str,
    argv: Sequence[str],
    environment: Mapping[str, str],
    capture_limit: int,
) -> CommandRecord:
    record = runner(name, tuple(argv), environment, capture_limit)
    if record.name != name or record.argv != tuple(argv):
        raise PipelineError(f"{name} runner returned a mismatched execution record")
    if record.returncode != 0:
        raise PipelineError(f"{name} failed with return code {record.returncode}")
    return record


@contextmanager
def _catch_termination() -> Any:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise PipelineInterrupted(f"pipeline interrupted by signal {signum}")

    for signum in (signal.SIGHUP, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_calibration_pipeline(request: PipelineRequest) -> PipelineResult:
    """Execute, verify, and no-clobber commit outputs followed by the receipt."""

    if (
        isinstance(request.capture_limit_bytes, bool)
        or not isinstance(request.capture_limit_bytes, int)
        or request.capture_limit_bytes < 0
        or request.capture_limit_bytes > 64 * 1024 * 1024
    ):
        raise PipelineError("capture limit must be an integer from 0 through 67108864")
    started_at, started_ns = _timestamp()

    source_model = _controlled_directory(request.source_model_dir, "source model")
    training_metadata = _existing_path(request.training_metadata, "training metadata")
    corpus = _existing_path(request.calibration_corpus, "calibration corpus")
    corpus_metadata = _existing_path(request.corpus_metadata, "calibration corpus metadata")
    llama_cpp = _controlled_directory(request.llama_cpp_dir, "llama.cpp checkout")
    if not _validate_directory_chain(llama_cpp, "llama.cpp checkout"):
        raise PipelineError(
            "llama.cpp checkout must be below an effective-UID-owned exclusive "
            "private trust boundary"
        )
    _controlled_directory(llama_cpp / ".git", "llama.cpp Git metadata")
    python = _existing_path(request.python_executable, "Python executable")
    git_name = shutil.which("git")
    if git_name is None:
        raise PipelineError("git executable is unavailable")
    git = _existing_path(Path(git_name).resolve(), "git executable")
    converter = _existing_path(llama_cpp / "convert_hf_to_gguf.py", "GGUF converter")
    imatrix_tool = _existing_path(
        llama_cpp / "build/bin/llama-imatrix", "llama-imatrix executable"
    )
    quantizer = _existing_path(
        llama_cpp / "build/bin/llama-quantize", "llama-quantize executable"
    )

    destinations = {
        "converted_model": _destination(request.converted_model, "converted model"),
        "imatrix": _destination(request.imatrix, "imatrix"),
        "quantized_artifact": _destination(
            request.quantized_artifact, "quantized artifact"
        ),
        "receipt": _destination(request.receipt, "execution receipt"),
    }
    if len(set(destinations.values())) != len(destinations):
        raise PipelineError("all final outputs and the receipt must use distinct paths")
    for label, destination in destinations.items():
        try:
            destination.relative_to(source_model)
        except ValueError:
            pass
        else:
            raise PipelineError(f"{label} must not be inside the source model tree")
        try:
            destination.relative_to(llama_cpp)
        except ValueError:
            pass
        else:
            raise PipelineError(f"{label} must not be inside the llama.cpp checkout")

    source_before = _snapshot_tree(source_model, "source model")
    input_before = {
        "training_metadata": _snapshot_file(
            training_metadata, "training metadata", trusted=True
        ),
        "calibration_corpus": _snapshot_file(
            corpus, "calibration corpus", trusted=True
        ),
        "corpus_metadata": _snapshot_file(
            corpus_metadata, "calibration corpus metadata", trusted=True
        ),
    }
    if len({snapshot.identity for snapshot in input_before.values()}) != len(input_before):
        raise PipelineError(
            "training metadata, calibration corpus, and sidecar must be distinct files"
        )
    tool_paths = {
        "python": python,
        "git": git,
        "convert_hf_to_gguf.py": converter,
        "llama-imatrix": imatrix_tool,
        "llama-quantize": quantizer,
    }
    executable_tools = {"python", "git", "llama-imatrix", "llama-quantize"}
    tool_before = {
        name: _snapshot_file(
            path,
            f"tool {name}",
            executable=name in executable_tools,
            trusted=True,
        )
        for name, path in tool_paths.items()
    }
    protected_roles = [
        (f"source_model/{entry.path}", (entry.device, entry.inode))
        for entry in source_before.files
    ]
    protected_roles.extend(
        (f"input/{name}", snapshot.identity)
        for name, snapshot in input_before.items()
    )
    protected_roles.extend(
        (f"tool/{name}", snapshot.identity)
        for name, snapshot in tool_before.items()
    )
    seen_identities: dict[tuple[int, int], str] = {}
    for label, identity in protected_roles:
        previous = seen_identities.setdefault(identity, label)
        if previous != label:
            raise PipelineError(
                "source files, evidence inputs, and tool roles must use "
                f"pairwise-distinct inodes: {previous} aliases {label}"
            )
    if len({snapshot.identity for snapshot in tool_before.values()}) != len(tool_before):
        raise PipelineError("pipeline tools must reference distinct files")
    _validate_checkout(llama_cpp, git)

    temporaries: list[_TemporaryOutput] = []
    receipt_temporaries: list[_TemporaryOutput] = []
    installed: dict[Path, tuple[int, int]] = {}
    command_records: list[CommandRecord] = []
    final_snapshots: dict[str, FileSnapshot] = {}
    receipt_digest = ""

    try:
        with _catch_termination():
            converted_temp = _make_temporary_output(
                destinations["converted_model"], temporaries
            )
            imatrix_temp = _make_temporary_output(destinations["imatrix"], temporaries)
            quantized_temp = _make_temporary_output(
                destinations["quantized_artifact"], temporaries
            )

            conversion_argv = (
                str(python),
                str(converter),
                str(source_model),
                "--outfile",
                str(converted_temp.path),
                "--outtype",
                "f16",
            )
            command_records.append(
                _invoke(
                    _run_command,
                    "convert_f16",
                    conversion_argv,
                    _OFFLINE_ENVIRONMENT,
                    request.capture_limit_bytes,
                )
            )
            _hold_completed_output(converted_temp, "converted F16 model")

            imatrix_argv = (
                str(imatrix_tool),
                "--offline",
                "--model",
                str(converted_temp.path),
                "--file",
                str(corpus),
                "--output",
                str(imatrix_temp.path),
                "--ctx-size",
                "512",
                "--chunks",
                "-1",
                "--no-ppl",
                "--parse-special",
            )
            command_records.append(
                _invoke(
                    _run_command,
                    "build_imatrix",
                    imatrix_argv,
                    _OFFLINE_ENVIRONMENT,
                    request.capture_limit_bytes,
                )
            )
            _hold_completed_output(imatrix_temp, "importance matrix")

            quantization_argv = (
                str(quantizer),
                "--imatrix",
                str(imatrix_temp.path),
                str(converted_temp.path),
                str(quantized_temp.path),
                "Q4_K_M",
            )
            command_records.append(
                _invoke(
                    _run_command,
                    "quantize_q4_k_m",
                    quantization_argv,
                    _OFFLINE_ENVIRONMENT,
                    request.capture_limit_bytes,
                )
            )
            _hold_completed_output(quantized_temp, "quantized artifact")

            output_identities = {
                temporary.snapshot.identity
                for temporary in temporaries
                if temporary.snapshot is not None
            }
            protected_identities = {
                snapshot.identity for snapshot in input_before.values()
            } | {snapshot.identity for snapshot in tool_before.values()} | {
                (entry.device, entry.inode) for entry in source_before.files
            }
            if len(output_identities) != 3 or output_identities & protected_identities:
                raise PipelineError("pipeline outputs must be distinct from every input and tool")

            _validate_checkout(llama_cpp, git)
            source_after = _snapshot_tree(source_model, "source model")
            if source_after != source_before:
                raise PipelineError("source model tree changed during pipeline execution")
            input_after = {
                "training_metadata": _snapshot_file(
                    training_metadata, "training metadata", trusted=True
                ),
                "calibration_corpus": _snapshot_file(
                    corpus, "calibration corpus", trusted=True
                ),
                "corpus_metadata": _snapshot_file(
                    corpus_metadata, "calibration corpus metadata", trusted=True
                ),
            }
            for name in input_before:
                _same_snapshot(input_before[name], input_after[name], name)
            tool_after = {
                name: _snapshot_file(
                    path,
                    f"tool {name}",
                    executable=name in executable_tools,
                    trusted=True,
                )
                for name, path in tool_paths.items()
            }
            for name in tool_before:
                _same_snapshot(tool_before[name], tool_after[name], f"tool {name}")

            output_names = ("converted_model", "imatrix", "quantized_artifact")
            for temporary in temporaries:
                if temporary.snapshot is None or temporary.descriptor < 0:
                    raise PipelineError("completed output is not held for publication")
                identity = _link_descriptor_no_replace(
                    temporary.descriptor, temporary.final
                )
                installed[temporary.final] = identity
            for name, temporary in zip(output_names, temporaries, strict=True):
                _make_held_inode_readable(temporary, name)
            temporary_cleanup_errors = _cleanup_temporary_records(temporaries)
            if temporary_cleanup_errors:
                raise PipelineError(
                    "temporary output cleanup was incomplete: "
                    + "; ".join(temporary_cleanup_errors)
                )

            for name, temporary in zip(output_names, temporaries, strict=True):
                if temporary.snapshot is None:
                    raise PipelineError(f"{name} output snapshot is unavailable")
                final_snapshots[name] = _verify_output(
                    temporary.final, temporary.snapshot, name
                )

            checked_at, checked_ns = _timestamp()
            finished_at, finished_ns = _timestamp()
            receipt_document = {
                "schema": SCHEMA,
                "llama_cpp": {
                    "checkout": str(llama_cpp),
                    "revision_before": LLAMA_CPP_REVISION,
                    "revision_after": LLAMA_CPP_REVISION,
                    "clean_before": True,
                    "clean_after": True,
                },
                "execution": {
                    "started_at_utc": started_at,
                    "started_at_unix_ns": started_ns,
                    "finished_at_utc": finished_at,
                    "finished_at_unix_ns": finished_ns,
                    "shell": False,
                    "environment_overrides": dict(_OFFLINE_ENVIRONMENT),
                    "commands": [record.as_dict() for record in command_records],
                },
                "tools": {
                    name: {
                        "before": tool_before[name].as_dict(),
                        "after": tool_after[name].as_dict(),
                    }
                    for name in sorted(tool_before)
                },
                "inputs": {
                    "source_model": {
                        "before": source_before.as_dict(),
                        "after": source_after.as_dict(),
                        "training_metadata_sha256": input_before[
                            "training_metadata"
                        ].sha256,
                    },
                    **{
                        name: {
                            "before": input_before[name].as_dict(),
                            "after": input_after[name].as_dict(),
                        }
                        for name in sorted(input_before)
                    },
                },
                "outputs": {
                    name: final_snapshots[name].as_dict()
                    for name in ("converted_model", "imatrix", "quantized_artifact")
                },
                "post_run_integrity": {
                    "confirmed": True,
                    "checked_at_utc": checked_at,
                    "checked_at_unix_ns": checked_ns,
                    "source_model_unchanged": True,
                    "inputs_unchanged": True,
                    "tools_unchanged": True,
                    "llama_cpp_revision_and_cleanliness_rechecked": True,
                    "final_outputs_match_held_inodes": True,
                    "final_outputs_rechecked_immediately_before_commit": True,
                    "receipt_commit": "no_replace_hard_link_from_held_inode",
                },
            }
            receipt_payload = _json_bytes(receipt_document)
            if len(receipt_payload) > MAX_RECEIPT_BYTES:
                raise PipelineError("execution receipt exceeds its byte limit")
            receipt_digest = _digest_bytes(receipt_payload)

            receipt_temp = _make_temporary_output(
                destinations["receipt"], receipt_temporaries, hold_open=True
            )
            receipt_descriptor = receipt_temp.descriptor
            offset = 0
            while offset < len(receipt_payload):
                written = os.write(receipt_descriptor, receipt_payload[offset:])
                if written < 1:
                    raise PipelineError("execution receipt temporary write stalled")
                offset += written
            os.fsync(receipt_descriptor)
            actual_payload = bytearray()
            offset = 0
            while offset < len(receipt_payload):
                chunk = os.pread(
                    receipt_descriptor,
                    min(_READ_CHUNK, len(receipt_payload) - offset),
                    offset,
                )
                if not chunk:
                    raise PipelineError("execution receipt ended during verification")
                actual_payload.extend(chunk)
                offset += len(chunk)
            if bytes(actual_payload) != receipt_payload:
                raise PipelineError("execution receipt temporary bytes changed")
            for name, temporary in zip(output_names, temporaries, strict=True):
                if temporary.snapshot is None:
                    raise PipelineError(f"{name} output snapshot is unavailable")
                _verify_output(temporary.final, temporary.snapshot, name)
            receipt_identity = _link_descriptor_no_replace(
                receipt_descriptor, destinations["receipt"]
            )
            installed[destinations["receipt"]] = receipt_identity
            _make_held_inode_readable(receipt_temp, "execution receipt")
            installed_receipt = _snapshot_file(
                destinations["receipt"],
                "installed execution receipt",
                trusted=True,
            )
            if (
                installed_receipt.identity != receipt_temp.created_identity
                or installed_receipt.identity != receipt_identity
                or installed_receipt.bytes != len(receipt_payload)
                or installed_receipt.sha256 != receipt_digest
                or installed_receipt.mode != "0o0644"
            ):
                raise PipelineError("installed execution receipt failed verification")
            receipt_cleanup_errors = _cleanup_temporary_records(receipt_temporaries)
            if receipt_cleanup_errors:
                raise PipelineError(
                    "execution receipt temporary cleanup was incomplete: "
                    + "; ".join(receipt_cleanup_errors)
                )
    except BaseException as exc:
        all_temporaries = (*temporaries, *receipt_temporaries)
        rollback_candidates = {
            temporary.final: temporary.created_identity
            for temporary in all_temporaries
        }
        rollback_candidates.update(installed)
        rollback_errors = list(_rollback_destinations(rollback_candidates))
        temporary_cleanup_errors = list(
            _cleanup_temporary_records(all_temporaries)
        )
        cleanup_messages: list[str] = []
        if rollback_errors:
            cleanup_messages.append(
                "rollback was incomplete: " + "; ".join(rollback_errors)
            )
        if temporary_cleanup_errors:
            cleanup_messages.append(
                "temporary cleanup was incomplete: "
                + "; ".join(temporary_cleanup_errors)
            )
        if cleanup_messages:
            raise PipelineError(
                "pipeline failed and " + "; ".join(cleanup_messages)
            ) from exc
        raise

    return PipelineResult(
        receipt=destinations["receipt"],
        receipt_sha256=receipt_digest,
        outputs=final_snapshots,
        post_run_integrity_confirmed=True,
        durability_confirmed=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pinned llama.cpp F16 conversion, offline imatrix generation, "
            "and Q4_K_M quantization, then commit a historical receipt."
        )
    )
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--calibration-corpus", type=Path, required=True)
    parser.add_argument("--corpus-metadata", type=Path, required=True)
    parser.add_argument("--converted-model", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--quantized-artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable).resolve()
    )
    parser.add_argument(
        "--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = PipelineRequest(
        source_model_dir=args.source_model_dir,
        training_metadata=args.training_metadata,
        calibration_corpus=args.calibration_corpus,
        corpus_metadata=args.corpus_metadata,
        converted_model=args.converted_model,
        imatrix=args.imatrix,
        quantized_artifact=args.quantized_artifact,
        receipt=args.receipt,
        llama_cpp_dir=args.llama_cpp_dir,
        python_executable=args.python_executable,
        capture_limit_bytes=args.capture_limit_bytes,
    )
    try:
        result = run_calibration_pipeline(request)
    except PipelineError as exc:
        raise SystemExit(f"calibration pipeline failed: {exc}") from exc
    print(
        json.dumps(
            {
                "durability_confirmed": result.durability_confirmed,
                "outputs": {
                    name: snapshot.as_dict()
                    for name, snapshot in result.outputs.items()
                },
                "post_run_integrity_confirmed": result.post_run_integrity_confirmed,
                "receipt": str(result.receipt),
                "receipt_sha256": result.receipt_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
