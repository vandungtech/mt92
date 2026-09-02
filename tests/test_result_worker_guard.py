"""Regressions that keep MT_RESULT_WORKER out of miner configuration and deployment.

Upstream 53e4df6 added an opt-in coordinator override that collapses reconciliation to a
single named worker's report. It is unpublished and unanchored, so this miner must never
carry it. See docs/upstream-audits/53e4df648a89fad6586e1ac69916b20e747fd972.md.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

from microtensor_miner_controller.config import (
    FORBIDDEN_ENVIRONMENT_VARIABLES,
    ControllerConfig,
)
from microtensor_miner_controller.errors import ConfigError

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# Files that are allowed to name the variable, because naming it is their job.
DOCUMENTING_PATHS = frozenset(
    {
        "tests/test_result_worker_guard.py",
        "src/microtensor_miner_controller/config.py",
        "launch_todo.md",
        "docs/upstream-audits/53e4df648a89fad6586e1ac69916b20e747fd972.md",
    }
)

# Every tracked surface that could carry configuration into a running process.
SCANNED_SUFFIXES = frozenset(
    {".conf", ".env", ".example", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
)


def _minimal_environment() -> dict[str, str]:
    return {
        "MT_WALLET_PATH": "/var/lib/microtensor-miner/wallets",
        "MMC_ARTIFACT_DIR": "/var/lib/microtensor-miner/artifact",
        "MMC_ARTIFACT_COMPETITION_BINDING_PATH": (
            "/etc/microtensor-miner/artifact-competition.binding.json"
        ),
        # Compact single-colon form: the scheme is split with partition(":"), and dropping
        # "//" keeps the final on-chain commitment under its 128-byte limit.
        "MMC_SOURCE_TEMPLATE": "https:github.com/vandungtech/mt92/releases/download/r{round}",
    }


def _git_executable() -> str:
    resolved = shutil.which("git")
    if resolved is None:  # pragma: no cover - git is required to run this suite
        raise unittest.SkipTest("git is unavailable")
    return resolved


def _tracked_files() -> tuple[Path, ...]:
    # Tracked files plus untracked-but-not-ignored ones, so a newly added deployment or
    # documentation file is scanned before it is ever committed.
    # S603 is globally ignored in pyproject.toml; the argv is fixed and no shell is used.
    completed = subprocess.run(
        [
            _git_executable(),
            "-C",
            str(REPOSITORY_ROOT),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    )
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    return tuple(Path(name) for name in names)


class ForbiddenEnvironmentTests(unittest.TestCase):
    def test_mt_result_worker_is_declared_forbidden(self) -> None:
        self.assertIn("MT_RESULT_WORKER", FORBIDDEN_ENVIRONMENT_VARIABLES)

    def test_configuration_load_refuses_a_set_result_worker(self) -> None:
        environment = _minimal_environment()
        environment["MT_RESULT_WORKER"] = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"

        with self.assertRaises(ConfigError) as raised:
            ControllerConfig.from_env(environment)

        self.assertIn("MT_RESULT_WORKER", str(raised.exception))

    def test_refusal_happens_before_any_other_configuration_error(self) -> None:
        # An otherwise unusable environment must still fail on the forbidden variable,
        # proving the check runs first and cannot be bypassed by an earlier refusal.
        with self.assertRaises(ConfigError) as raised:
            ControllerConfig.from_env({"MT_RESULT_WORKER": "anything"})

        self.assertIn("MT_RESULT_WORKER", str(raised.exception))

    def test_blank_result_worker_is_tolerated(self) -> None:
        environment = _minimal_environment()
        environment["MT_RESULT_WORKER"] = "   "

        config = ControllerConfig.from_env(environment)

        self.assertEqual(config.netuid, 92)


class TrackedSurfaceScanTests(unittest.TestCase):
    """Static scan of the tracked repository for a reintroduced override."""

    def setUp(self) -> None:
        self.tracked = _tracked_files()
        # Guard against a vacuous pass: if `git ls-files` returns nothing, or the
        # documenting files are not actually tracked, the scan below proves nothing.
        self.assertGreater(len(self.tracked), 50)
        visible = {path.as_posix() for path in self.tracked}
        for name in DOCUMENTING_PATHS:
            self.assertIn(name, visible, f"{name} must be visible for this scan to mean anything")

    def test_scan_actually_detects_the_pattern(self) -> None:
        # Prove the matcher works before trusting a clean result from it.
        pattern = re.compile(r"MT_RESULT_WORKER")
        self.assertIsNotNone(pattern.search("MT_RESULT_WORKER=somehotkey"))
        self.assertIsNotNone(pattern.search('    MT_RESULT_WORKER: ${MT_RESULT_WORKER:-}'))

    def test_no_tracked_file_assigns_mt_result_worker(self) -> None:
        # An assignment, an export, a compose mapping, or a Supervisor environment entry.
        assignment = re.compile(
            r"MT_RESULT_WORKER\s*(?:=|:)|export\s+MT_RESULT_WORKER|"
            r'MT_RESULT_WORKER\s*=\s*"',
        )
        offenders: list[str] = []
        for path in self.tracked:
            name = path.as_posix()
            if name in DOCUMENTING_PATHS or path.suffix not in SCANNED_SUFFIXES:
                continue
            absolute = REPOSITORY_ROOT / path
            try:
                text = absolute.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if assignment.search(line):
                    offenders.append(f"{name}:{number}")
        self.assertEqual(offenders, [], f"MT_RESULT_WORKER assigned in: {offenders}")

    def test_documenting_files_only_describe_the_prohibition(self) -> None:
        # The audit note and launch_todo may name the variable, but must not assign it
        # in a form an operator could copy into a live environment file.
        live_assignment = re.compile(r"^\s*MT_RESULT_WORKER\s*=\s*\S")
        offenders: list[str] = []
        for name in sorted(DOCUMENTING_PATHS):
            if name.endswith(".py"):
                continue
            text = (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), 1):
                if live_assignment.search(line):
                    offenders.append(f"{name}:{number}")
        self.assertEqual(offenders, [], f"copyable MT_RESULT_WORKER assignment in: {offenders}")


if __name__ == "__main__":
    unittest.main()
