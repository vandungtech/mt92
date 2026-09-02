from __future__ import annotations

import os
import stat
from pathlib import Path


class ProtectedFileError(RuntimeError):
    """A root-controlled service file is unavailable or has unsafe metadata."""


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate(
    metadata: os.stat_result,
    *,
    label: str,
    maximum_bytes: int,
) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProtectedFileError(f"{label} must be a regular non-symlink")
    if metadata.st_nlink != 1:
        raise ProtectedFileError(f"{label} must have exactly one hard link")
    if metadata.st_uid != 0:
        raise ProtectedFileError(f"{label} must be owned by root")
    if metadata.st_gid != os.getegid():
        raise ProtectedFileError(f"{label} group must equal the effective service group")
    if stat.S_IMODE(metadata.st_mode) != 0o640:
        raise ProtectedFileError(f"{label} mode must be exactly 0640")
    if metadata.st_size < 0 or metadata.st_size > maximum_bytes:
        raise ProtectedFileError(f"{label} exceeds the {maximum_bytes}-byte limit")


def read_root_service_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    """Read one bounded root-owned, service-group-readable file fail closed.

    The caller receives bytes only after the pathname and open descriptor have
    remained identical for the complete read. File contents are never included in
    an exception.
    """

    if not path.is_absolute():
        raise ProtectedFileError(f"{label} path must be absolute")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ProtectedFileError(f"{label} byte limit is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ProtectedFileError(f"this platform cannot securely open {label}")

    descriptor = -1
    payload = bytearray()
    try:
        path_before = path.lstat()
        _validate(path_before, label=label, maximum_bytes=maximum_bytes)
        descriptor = os.open(
            path,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        opened_before = os.fstat(descriptor)
        _validate(opened_before, label=label, maximum_bytes=maximum_bytes)
        if (path_before.st_dev, path_before.st_ino) != (
            opened_before.st_dev,
            opened_before.st_ino,
        ):
            raise ProtectedFileError(f"{label} changed while it was opened")

        expected = _fingerprint(opened_before)
        while True:
            remaining = maximum_bytes + 1 - len(payload)
            chunk = os.read(descriptor, min(16 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise ProtectedFileError(f"{label} exceeds the {maximum_bytes}-byte limit")

        opened_after = os.fstat(descriptor)
        _validate(opened_after, label=label, maximum_bytes=maximum_bytes)
        path_after = path.lstat()
        _validate(path_after, label=label, maximum_bytes=maximum_bytes)
        if (
            expected != _fingerprint(opened_after)
            or expected != _fingerprint(path_before)
            or expected != _fingerprint(path_after)
            or len(payload) != opened_before.st_size
        ):
            raise ProtectedFileError(f"{label} changed while it was read")
    except ProtectedFileError:
        raise
    except OSError:
        raise ProtectedFileError(f"{label} is unavailable or unsafe") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return bytes(payload)
