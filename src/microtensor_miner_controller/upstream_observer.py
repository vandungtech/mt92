from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .state import StateStore, utc_timestamp

log = logging.getLogger(__name__)

SCHEMA: Final[str] = "microtensor.upstream-observation.v1"
AUDITED_UPSTREAM_HEAD: Final[str] = "d77adc945de763f8b3b2d71fef8193090ede7001"
AUDITED_RELEASE_VERSION: Final[str] = "0.3.2"
AUDITED_MECHANISM_VERSION: Final[str] = "0.3.0"
EXPECTED_ORIGIN: Final[str] = "https://github.com/microtensor-io/microtensor-subnet"
CANDIDATE_REF: Final[str] = "refs/microtensor-observer/candidate-main"
DEFAULT_POLL_SECONDS: Final[int] = 300
MAX_GIT_OUTPUT_BYTES: Final[int] = 2 * 1024 * 1024
MAX_CHANGED_FILES: Final[int] = 500
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

_MINER_IMPACT_PREFIXES: Final[tuple[str, ...]] = (
    ".github/workflows/",
    "deploy/",
    "docs/",
    "microtensor/",
    "neurons/",
    "scripts/",
    "pyproject.toml",
)


class ObservationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GitResult:
    returncode: int
    output: str


GitRunner = Callable[[Path, tuple[str, ...], frozenset[int]], GitResult]


def _run_git(repo: Path, arguments: tuple[str, ...], accepted: frozenset[int]) -> GitResult:
    command = (
        "/usr/bin/git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "credential.helper=",
        "-c",
        "submodule.recurse=false",
        "-c",
        "fetch.recurseSubmodules=false",
        "-C",
        str(repo),
        *arguments,
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as output_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            deadline = time.monotonic() + 60.0
            while True:
                try:
                    returncode = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if output_file.tell() > MAX_GIT_OUTPUT_BYTES:
                        _kill_process_group(process)
                        raise ObservationError(
                            f"git {arguments[0]} output exceeded 2 MiB"
                        ) from None
                    if time.monotonic() >= deadline:
                        _kill_process_group(process)
                        raise ObservationError(
                            f"git {arguments[0]} timed out"
                        ) from None
            if output_file.tell() > MAX_GIT_OUTPUT_BYTES:
                raise ObservationError(f"git {arguments[0]} output exceeded 2 MiB")
            output_file.seek(0)
            raw = output_file.read(MAX_GIT_OUTPUT_BYTES + 1)
    except ObservationError:
        raise
    except OSError as exc:
        if process is not None and process.poll() is None:
            _kill_process_group(process)
        raise ObservationError(f"git {arguments[0]} failed: {exc}") from exc
    if len(raw) > MAX_GIT_OUTPUT_BYTES:
        raise ObservationError(f"git {arguments[0]} output exceeded 2 MiB")
    try:
        output = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ObservationError(f"git {arguments[0]} returned non-UTF-8 output") from exc
    if returncode not in accepted:
        detail = output.strip().replace("\n", " ")[:500]
        raise ObservationError(
            f"git {arguments[0]} exited {returncode}: {detail or 'no detail'}"
        )
    return GitResult(returncode, output)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def _canonical_origin(value: str) -> str:
    canonical = value.strip().removesuffix(".git").rstrip("/")
    if canonical != EXPECTED_ORIGIN:
        raise ObservationError("configured origin does not match expected public repository")
    return canonical


def _commit(value: str, label: str) -> str:
    commit = value.strip().lower()
    if _COMMIT.fullmatch(commit) is None:
        raise ObservationError(f"{label} is not a canonical 40-hex commit")
    return commit


def _advertised_commit(value: str) -> str:
    lines = [line for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ObservationError("origin advertised an ambiguous main ref")
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != "refs/heads/main":
        raise ObservationError("origin advertised a malformed main ref")
    return _commit(fields[0], "advertised origin/main")


def _static_constant(source: str, name: str, expected_type: type[str] | type[bool]) -> Any:
    try:
        module = ast.parse(source, filename="microtensor/core/constants.py", mode="exec")
    except SyntaxError as exc:
        raise ObservationError("origin constants are not valid Python syntax") from exc
    values: list[Any] = []
    for statement in module.body:
        value: ast.expr | None = None
        if (
            (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == name
            )
            or (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name
            )
        ):
            value = statement.value
        if value is not None:
            if not isinstance(value, ast.Constant) or type(value.value) is not expected_type:
                raise ObservationError(f"origin constants {name} is not a static literal")
            values.append(value.value)
    if len(values) != 1:
        raise ObservationError(f"origin constants omit or redefine {name}")
    return values[0]


def _changed_files(output: str) -> tuple[tuple[str, ...], bool]:
    if not output:
        return (), False
    if not output.endswith("\x00"):
        raise ObservationError("git diff returned malformed NUL-delimited paths")
    paths = tuple(output[:-1].split("\x00"))
    if any(not _safe_relative_git_path(path) for path in paths):
        raise ObservationError("git diff returned an unsafe path")
    return paths[:MAX_CHANGED_FILES], len(paths) > MAX_CHANGED_FILES


def _safe_relative_git_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        return False
    components = path.split("/")
    return all(component not in {"", ".", ".."} for component in components)


def _miner_impact(paths: Sequence[str]) -> bool:
    return any(
        path == prefix or path.startswith(prefix)
        for path in paths
        for prefix in _MINER_IMPACT_PREFIXES
    )


class UpstreamObserver:
    def __init__(
        self,
        repo: Path,
        state: StateStore,
        *,
        runner: GitRunner = _run_git,
    ) -> None:
        self.repo = repo
        self.state = state
        self.runner = runner
        self._history_rewrite_detected_during_attempt = False

    def _git(self, *arguments: str, accepted: frozenset[int] = frozenset({0})) -> GitResult:
        return self.runner(self.repo, tuple(arguments), accepted)

    def observe(self) -> dict[str, Any]:
        if not self.repo.is_absolute() or not self.repo.is_dir():
            raise ObservationError(f"upstream repository is unavailable: {self.repo}")
        prior = self.state.read_status()
        history_rewrite_latched = prior.get("history_rewrite_latched") is True
        self._history_rewrite_detected_during_attempt = False

        origin = _canonical_origin(self._git("remote", "get-url", "origin").output)
        local_head = _commit(self._git("rev-parse", "HEAD").output, "local HEAD")
        cached_result = self._git(
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main",
            accepted=frozenset({0, 1, 128}),
        )
        cached_before = (
            _commit(cached_result.output, "cached origin/main")
            if cached_result.returncode == 0
            else None
        )
        advertised = _advertised_commit(
            self._git(
                "ls-remote",
                "--exit-code",
                origin + ".git",
                "refs/heads/main",
            ).output
        )

        fetched = False
        history_rewrite_detected = False
        if cached_before != advertised:
            self._git(
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                "--no-recurse-submodules",
                origin + ".git",
                f"+refs/heads/main:{CANDIDATE_REF}",
            )
            fetched = True
            candidate = _commit(
                self._git("rev-parse", "--verify", CANDIDATE_REF).output,
                "fetched candidate origin/main",
            )
            if candidate != advertised:
                raise ObservationError(
                    "fetched candidate origin/main differs from the advertised commit"
                )
            if cached_before is not None:
                cached_ancestor = self._git(
                    "merge-base",
                    "--is-ancestor",
                    cached_before,
                    advertised,
                    accepted=frozenset({0, 1}),
                )
                history_rewrite_detected = cached_ancestor.returncode == 1
                self._history_rewrite_detected_during_attempt = history_rewrite_detected
                if history_rewrite_detected:
                    # Persist the fail-closed latch before mutating origin/main. A
                    # SIGKILL after update-ref must not erase the only evidence that
                    # the advertised history was not a fast-forward.
                    self.state.write(
                        "review_required",
                        ok=False,
                        message="origin/main history rewrite requires operator review",
                        details={
                            "schema": SCHEMA,
                            "observation_succeeded": False,
                            "attempted_at": utc_timestamp(),
                            "history_rewrite_detected": True,
                            "history_rewrite_latched": True,
                            "cached_origin_head_before": cached_before,
                            "origin_head": advertised,
                        },
                        preserve={},
                    )
            expected_old = cached_before or ("0" * 40)
            self._git(
                "update-ref",
                "refs/remotes/origin/main",
                advertised,
                expected_old,
            )

        cached = _commit(
            self._git(
                "rev-parse",
                "--verify",
                "refs/remotes/origin/main",
            ).output,
            "fetched origin/main",
        )
        if cached != advertised:
            raise ObservationError("fetched origin/main differs from the advertised commit")
        history_rewrite_latched = history_rewrite_latched or history_rewrite_detected

        constants = self._git(
            "show", f"{advertised}:microtensor/core/constants.py"
        ).output
        release = _static_constant(constants, "RELEASE_VERSION", str)
        mechanism = _static_constant(constants, "MECHANISM_VERSION", str)
        provenance_required = _static_constant(constants, "PROVENANCE_REQUIRED", bool)

        origin_changed = advertised != AUDITED_UPSTREAM_HEAD
        review_required = origin_changed or history_rewrite_latched
        changed: tuple[str, ...] = ()
        truncated = False
        ancestor = True
        commit_count = 0
        if origin_changed:
            ancestor_result = self._git(
                "merge-base",
                "--is-ancestor",
                AUDITED_UPSTREAM_HEAD,
                advertised,
                accepted=frozenset({0, 1}),
            )
            ancestor = ancestor_result.returncode == 0
            changed, truncated = _changed_files(
                self._git(
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--name-only",
                    "-z",
                    "--diff-filter=ACDMRTUXB",
                    AUDITED_UPSTREAM_HEAD,
                    advertised,
                ).output
            )
            if ancestor:
                raw_count = self._git(
                    "rev-list", "--count", f"{AUDITED_UPSTREAM_HEAD}..{advertised}"
                ).output.strip()
                try:
                    commit_count = int(raw_count)
                except ValueError as exc:
                    raise ObservationError("git returned an invalid upstream commit count") from exc

        miner_impact = history_rewrite_latched or (
            origin_changed and (_miner_impact(changed) or truncated or not ancestor)
        )
        observed_at_epoch = time.time()
        observed_at = utc_timestamp(observed_at_epoch)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "observation_succeeded": True,
            "origin_observed_at": observed_at_epoch,
            "checked_at": observed_at,
            "repository": str(self.repo),
            "origin": origin,
            "local_head": local_head,
            "cached_origin_head_before": cached_before,
            "origin_head": advertised,
            "audited_origin_head": AUDITED_UPSTREAM_HEAD,
            "fetched": fetched,
            "local_checkout_at_origin": local_head == advertised,
            "release_version": release,
            "mechanism_version": mechanism,
            "provenance_required": provenance_required,
            "history_rewrite_detected": history_rewrite_detected,
            "history_rewrite_latched": history_rewrite_latched,
            "review_required": review_required,
            "miner_impact_review_required": miner_impact,
            "audited_head_is_ancestor": ancestor,
            "commits_since_audit": commit_count,
            "changed_files": list(changed),
            "changed_files_truncated": truncated,
        }
        if not origin_changed and (
            release != AUDITED_RELEASE_VERSION
            or mechanism != AUDITED_MECHANISM_VERSION
            or provenance_required is not False
        ):
            raise ObservationError("audited upstream commit has unexpected release constants")
        return payload

    def poll_once(self) -> dict[str, Any]:
        prior = self.state.read_status()
        rewrite_latched = prior.get("history_rewrite_latched") is True
        try:
            payload = self.observe()
            if payload["review_required"]:
                message = (
                    f"origin/main {payload['origin_head']} requires compatibility review"
                )
                phase = "review_required"
                ok = False
            else:
                message = (
                    f"origin/main remains at audited v{payload['release_version']} "
                    f"{payload['origin_head']}"
                )
                phase = "current"
                ok = True
            return self.state.write(phase, ok=ok, message=message, details=payload)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            log.error("upstream observation failed: %s", message)
            return self.state.write(
                "check_error",
                ok=False,
                message=message[:1000],
                details={
                    "schema": SCHEMA,
                    "observation_succeeded": False,
                    "attempted_at": utc_timestamp(),
                    "history_rewrite_latched": (
                        rewrite_latched
                        or self._history_rewrite_detected_during_attempt
                    ),
                },
                preserve={},
            )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously fetch and record Microtensor origin/main drift without merging."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=_positive_int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.expanduser().absolute()
    state = StateStore(args.state_dir.expanduser().absolute())
    observer = UpstreamObserver(repo, state)
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = __import__("time").gmtime

    with state.lock():
        while not stop.is_set():
            payload = observer.poll_once()
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
            if args.once:
                return 0 if payload.get("ok") is True else 2
            stop.wait(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
