#!/usr/bin/env python3
"""Replace this trusted bootstrap with one exact absolute native executable."""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import stat
import sys
from pathlib import Path
from typing import Final, NoReturn

sys.dont_write_bytecode = True

_PR_GET_PDEATHSIG: Final[int] = 2
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


def _pdeathsig() -> int:
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    library.prctl.restype = ctypes.c_int
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
        _fail(f"cannot inspect parent-death signal: {os.strerror(number)}")
    return value.value


def main(arguments: list[str] | None = None) -> NoReturn:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if not values:
        _fail("one exact absolute executable is required")
    target = Path(values[0])
    if not target.is_absolute() or "\x00" in str(target):
        _fail("executable path must be exact, absolute, and NUL-free")
    try:
        info = target.stat()
    except OSError as exc:
        _fail(f"cannot stat executable {target}: {exc}")
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISREG(info.st_mode)
        or mode & 0o111 == 0
        or mode & 0o6002
    ):
        _fail("executable is special, non-executable, privileged, or world writable")
    if dict(os.environ) != EXACT_ENVIRONMENT:
        _fail("exec wrapper environment differs from the exact fixed mapping")
    if _pdeathsig() != signal.SIGKILL:
        _fail("SIGKILL parent-death boundary is not armed")
    argv = [str(target), *values[1:]]
    os.execve(str(target), argv, EXACT_ENVIRONMENT)  # noqa: S606 - exact path, no shell


if __name__ == "__main__":
    main()
