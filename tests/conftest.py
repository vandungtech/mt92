"""Test-session setup that must run before any test module is imported.

``training.code_conversion_runtime._require_clean_boundary`` refuses to launch a
contained command unless the launching process has exactly one task in
``/proc/self/task``. That is a real production boundary: it guarantees exact PID
accounting and that no sibling thread can interfere across the fork/exec, so it must
not be relaxed to suit a test harness.

Importing ``numpy`` eagerly starts one OpenBLAS worker thread per CPU (32 on this
host), and ``tests/test_build_weight_soup.py`` imports it immediately before
``tests/test_code_conversion_runtime.py`` in alphabetical collection order. That left
the pytest process multi-threaded and made 26 otherwise-passing tests fail in a full
run while passing in isolation.

Pinning the BLAS/OpenMP thread pools to one thread before the first import keeps the
test process single-threaded, so the suite presents exactly the launcher condition the
production check requires. These variables only affect this process; they are set with
``setdefault`` so an operator can still override them deliberately.
"""

from __future__ import annotations

import os
import signal

import pytest

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")


_RESTORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@pytest.fixture(autouse=True)
def _restore_signal_dispositions():
    """Undo process-wide signal handlers a daemon entrypoint installs under test.

    ``rank_observer.main`` and ``cli._run`` legitimately install SIGINT/SIGTERM handlers
    that set a stop event; that is correct for a long-running service but it leaks out of
    any test that calls them. ``tests/test_run_calibration_pipeline.py`` deliberately
    raises SIGINT at itself and asserts the signal is deferred, which only holds against
    the default disposition, so the leaked handler made those tests fail in a full run
    while passing in isolation.
    """

    saved = {number: signal.getsignal(number) for number in _RESTORED_SIGNALS}
    try:
        yield
    finally:
        for number, handler in saved.items():
            if handler is not None and signal.getsignal(number) is not handler:
                signal.signal(number, handler)
