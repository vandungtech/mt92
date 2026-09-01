#!/usr/bin/env python3
"""Fail-closed launcher for the three immutable v6 GGUF diagnostics.

The evaluator is executed directly (never through a shell) with the exact
environment and argv declared by the immutable diagnostic addendum.  A
repeat is consumed before its child can start, so a failed, interrupted, or
timed-out attempt cannot be retried under the same declaration.

On Linux, the child requests ``PTRACE_TRACEME`` before ``execve``.  The kernel
then stops the new image at the exec boundary.  While that stop is held, this
launcher checks the child's ``/proc`` cmdline, environment, cwd, executable,
stdin, open descriptors, and umask.  This evidence is deliberately narrow: it
attests only the held post-exec launch boundary, not any later evaluator or
model state.  The child is allowed to continue only after every check passes.

After a zero exit, the separately pinned static validator replays all declared
identities and validates every diagnostic receipt through the current repeat.
It does not construct a model engine or execute, import, or compile generated
or corpus code.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import select
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Final

sys.dont_write_bytecode = True

SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-launch-receipt.v1"
ATTEMPT_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-launch-attempt.v1"
LEGACY_VALIDATION_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-validation.v1"
ADDENDUM_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-execution-addendum.v1"
NORMALIZED_ADDENDUM_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-execution-addendum.v2"
CURRENT94_ADDENDUM_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-execution-addendum.v3"
ADDENDUM_STATUS: Final[str] = "final"
REPOSITORY: Final[str] = "https://github.com/vandungtech/mt92"
SOURCE_COMMIT: Final[str] = "c9c4effe77271b2d70d4eee745de654af9d1e74d"
SOURCE_ROOT: Final[Path] = Path("/tmp/mt92-q4-code-c9c4eff")  # noqa: S108
SPEC_RELATIVE: Final[str] = "training/experiment_specs/code-q4-imatrix128-m541-v6.json"
SPEC_BYTES: Final[int] = 49_270
SPEC_DIGEST: Final[str] = "sha256:7dc168c55316b3cc378809d13f8fe3777bfa29824bc97dd603c215324b8bd97d"
ADDENDUM_RELATIVE: Final[str] = (
    "training/experiment_specs/code-q4-imatrix128-m541-v6-diagnostic-addendum.json"
)
NORMALIZED_SPEC_RELATIVE: Final[str] = (
    "training/experiment_specs/code-historical7730-normalized-v7-q4-m541-py311-diagnostic.json"
)
NORMALIZED_ADDENDUM_RELATIVE: Final[str] = (
    "training/experiment_specs/"
    "code-historical7730-normalized-v7-q4-m541-py311-diagnostic-addendum.json"
)
CURRENT94_SPEC_RELATIVE: Final[str] = (
    "training/experiment_specs/"
    "code-current94-qwen25-coder-15b-v8-q4-m541-v6-diagnostic.json"
)
CURRENT94_ADDENDUM_RELATIVE: Final[str] = (
    "training/experiment_specs/"
    "code-current94-qwen25-coder-15b-v8-q4-m541-v6-diagnostic-addendum.json"
)
CURRENT94_DECLARATION_LEXICAL_PATH: Final[Path] = (
    Path(os.path.abspath(__file__)).parent.parent / CURRENT94_ADDENDUM_RELATIVE
)
LAUNCHER_RELATIVE: Final[str] = "training/run_code_gguf_diagnostic.py"
VALIDATOR_RELATIVE: Final[str] = "training/validate_code_gguf_diagnostic.py"
INTERPRETER_PATH: Final[Path] = Path(
    "/tmp/microtensor-v030-verify.5rMSRW/venv/bin/python"  # noqa: S108
)
INTERPRETER_RESOLVED: Final[Path] = Path("/usr/bin/python3.12")
INTERPRETER_BYTES: Final[int] = 8_016_832
INTERPRETER_DIGEST: Final[str] = (
    "sha256:1319c137ea5d30f1d7599943cb0e72666648c20a94cf5932dd095364d07dafeb"
)
ARTIFACT_ROOT: Final[Path] = Path(
    "/dev/shm/microtensor-code/"  # noqa: S108 - exact immutable tmpfs path
    "qwen3-06b-historical7299-final-r64-e2-b1ga16-seed92-v6-q4-"
    "imatrix128-m541-replay1-bundle/artifact"
)
ARTIFACT_TREE_DIGEST: Final[str] = (
    "sha256:3f6dc72a0cd886c74a5161ccd42feda27de56e54c914f28961e7dd89ca2917b5"
)
ARTIFACT_ENTRYPOINT_BYTES: Final[int] = 396_704_672
ARTIFACT_ENTRYPOINT_DIGEST: Final[str] = (
    "sha256:3df33a173b16af2bca9a402c335bda5d39b03e290d4ba13f4eaf5ad5c4397d5e"
)
REPORT_ROOT: Final[Path] = Path(
    "/dev/shm/microtensor-code/diagnostic-launch-receipts/"  # noqa: S108
    "qwen3-06b-historical7299-b1ga16-v6-q4-imatrix128-m541-current16-signed-v030"
)
REPORT_BASE: Final[Path] = Path("/dev/shm/microtensor-code")  # noqa: S108
TIMEOUT_SECONDS: Final[int] = 900
UMASK_TEXT: Final[str] = "0077"
UMASK_VALUE: Final[int] = 0o077
REPEATS: Final[tuple[str, ...]] = ("r1", "r2", "r3")
MAX_ADDENDUM_BYTES: Final[int] = 256 * 1024
MAX_RECEIPT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_CHILD_ERROR_BYTES: Final[int] = 4096
MAX_ERROR_TEXT_BYTES: Final[int] = 4096
_DIGEST: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_PTRACE_TRACEME: Final[int] = 0
_PTRACE_DETACH: Final[int] = 17
_PR_SET_PDEATHSIG: Final[int] = 1
_PR_SET_CHILD_SUBREAPER: Final[int] = 36
_PR_GET_CHILD_SUBREAPER: Final[int] = 37
_BOUNDARY_POLL_MS: Final[int] = 20
_CLEANUP_TIMEOUT_SECONDS: Final[int] = 5
_WAIT_ALL: Final[int] = 0x40000000
_ADVERTISED_REMOTE_REF: Final[str] = "refs/remotes/origin/main"

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
    "PATH": "/tmp/microtensor-v030-verify.5rMSRW/venv/bin:/usr/bin:/bin",  # noqa: S108
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "WANDB_MODE": "disabled",
}
# The exact declared mapping has 16 keys. Earlier prose counted 15, but the
# authoritative 429-byte mapping and its SHA-256 have always included all 16.
ENVIRONMENT_CANONICAL_BYTES: Final[int] = 429
ENVIRONMENT_CANONICAL_DIGEST: Final[str] = (
    "sha256:9103f1e3ef395e681510f8044ab3f4861352748c8ff8efb928266a0b81a7ce94"
)

DATASET: Final[Path] = Path(
    "/dev/shm/microtensor-code/dataset-dev-seed92-h16"  # noqa: S108
)
DIAGNOSTIC_JSONL: Final[Path] = DATASET / "holdout.jsonl"
CURRENT_SOURCE_CORPUS: Final[Path] = Path(
    "/dev/shm/microtensor-code/public-code-corpus-v1.json"  # noqa: S108
)
TRAINING_RUN: Final[Path] = Path(
    "/dev/shm/microtensor-code/runs/"  # noqa: S108
    "qwen3-06b-historical7299-final-r64-e2-b1ga16-seed92-v5"
)
TRAINING_DATASET: Final[Path] = Path(
    "/dev/shm/microtensor-code/dataset-historical-7299-seed92-h0"  # noqa: S108
)
TRAINING_SOURCE_CORPUS: Final[Path] = Path(
    "/dev/shm/microtensor-code/public-code-corpus-7299bd7c.json"  # noqa: S108
)
TRAINING_BASE: Final[Path] = Path(
    "/dev/shm/microtensor-code/base-qwen3-06b"  # noqa: S108
)
OUTPUT_ROOTS: Final[tuple[Path, Path, Path]] = tuple(
    Path(
        "/dev/shm/microtensor-code/evaluations/"  # noqa: S108
        "qwen3-06b-historical7299-b1ga16-v6-q4-imatrix128-m541-"
        f"current16-signed-v030-{repeat}"
    )
    for repeat in REPEATS
)
NORMALIZED_NAMESPACE: Final[str] = (
    "qwen3-06b-historical7730-normalized-b1ga16-v7-q4-m541-py311-current16-signed-v030"
)
NORMALIZED_REPORT_ROOT: Final[Path] = Path(
    "/dev/shm/microtensor-code/diagnostic-launch-receipts/"  # noqa: S108
    f"{NORMALIZED_NAMESPACE}"
)
NORMALIZED_OUTPUT_ROOTS: Final[tuple[Path, Path, Path]] = tuple(
    Path("/dev/shm/microtensor-code/evaluations")  # noqa: S108
    / f"{NORMALIZED_NAMESPACE}-{repeat}"
    for repeat in REPEATS
)
NORMALIZED_BUNDLE_ROOT: Final[Path] = Path(
    "/dev/shm/microtensor-code/"  # noqa: S108
    "qwen3-06b-historical7730-normalized-final-r64-e2-b1ga16-seed92-v7-"
    "q4-m541-py311-bundle"
)
NORMALIZED_TRAINING_RUN: Final[Path] = Path(
    "/dev/shm/microtensor-code/runs/"  # noqa: S108
    "qwen3-06b-historical7730-normalized-final-r64-e2-b1ga16-seed92-v7"
)
NORMALIZED_TRAINING_DATASET: Final[Path] = Path(
    "/dev/shm/microtensor-code/"  # noqa: S108
    "dataset-historical7730-normalized-seed92-h0-v7"
)
NORMALIZED_TRAINING_SOURCE: Final[Path] = Path(
    "/dev/shm/microtensor-code/public-code-corpus-7299bd7c.json"  # noqa: S108
)
NORMALIZED_TRAINING_BASE: Final[Path] = Path(
    "/dev/shm/microtensor-code/base-qwen3-06b"  # noqa: S108
)
NORMALIZED_DATASET: Final[Path] = DATASET
NORMALIZED_DIAGNOSTIC_JSONL: Final[Path] = DIAGNOSTIC_JSONL
NORMALIZED_CURRENT_SOURCE: Final[Path] = CURRENT_SOURCE_CORPUS
NORMALIZED_SPEC_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-experiment.v2"
NORMALIZED_VALIDATION_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-validation.v2"
CURRENT94_SPEC_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-experiment.v3"
CURRENT94_VALIDATION_SCHEMA: Final[str] = "microtensor.code.gguf-diagnostic-validation.v3"

CURRENT94_CANDIDATE_ID: Final[str] = (
    "qwen25-coder-15b-current94-final-r32-e3-b4ga2-lr5e5-seed92-v8-q4-m541-v6"
)
CURRENT94_BASE_MODEL: Final[str] = (
    "Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a"
)
CURRENT94_BUNDLE_ROOT: Final[Path] = Path(
    "/dev/shm/microtensor-code/"  # noqa: S108
    "qwen25-coder-15b-current94-final-r32-e3-b4ga2-lr5e5-seed92-v8-"
    "q4-m541-v6-bundle"
)
CURRENT94_NAMESPACE: Final[str] = f"{CURRENT94_CANDIDATE_ID}-training-overlap-signed-v032"
CURRENT94_REPORT_ROOT: Final[Path] = Path(
    "/dev/shm/microtensor-code/diagnostic-launch-receipts"  # noqa: S108
) / CURRENT94_NAMESPACE
CURRENT94_OUTPUT_ROOTS: Final[tuple[Path, Path, Path]] = tuple(
    Path("/dev/shm/microtensor-code/evaluations")  # noqa: S108
    / f"{CURRENT94_NAMESPACE}-{repeat}"
    for repeat in REPEATS
)
CURRENT94_TRAINING_RUN: Final[Path] = Path(
    "/dev/shm/microtensor-code/runs/"  # noqa: S108
    "qwen25-coder-15b-current94-final-r32-e3-b4ga2-lr5e5-seed92-v8"
)
CURRENT94_TRAINING_DATASET: Final[Path] = Path(
    "/dev/shm/microtensor-code/dataset-final-seed92-h0"  # noqa: S108
)
CURRENT94_TRAINING_SOURCE: Final[Path] = Path(
    "/dev/shm/microtensor-code/public-code-corpus-v1.json"  # noqa: S108
)
CURRENT94_TRAINING_BASE: Final[Path] = Path(
    "/dev/shm/microtensor-code/base-qwen25-coder-15b"  # noqa: S108
)
CURRENT94_LIVE_BLOCKER: Final[str] = (
    "current94 live launch is disabled until a reviewed hermetic containment boundary "
    "provides reviewed proof of filesystem, network, process-tree, native-loader, and "
    "receipt integrity"
)

INSPECTION_SCOPE: Final[str] = (
    "Linux /proc fields are sampled while the PTRACE_TRACEME exec-stop is held, before "
    "this launcher detaches the declared child; this proves only that post-exec launch "
    "boundary and does not attest later evaluator or model state."
)
POST_EXEC_INSPECTION: Final[dict[str, Any]] = {
    "mechanism": "linux-ptrace-traceme-exec-stop",
    "required": True,
    "fields": ["cmdline", "environ", "cwd", "exe", "fd0", "open_fds", "umask"],
    "evidence_scope": INSPECTION_SCOPE,
    "subsequent_runtime_attested": False,
}
CONTAINMENT_SCOPE: Final[str] = (
    "The cgroup v2 mount is read-only on this host, so the launcher uses a temporary Linux "
    "child-subreaper boundary, an isolated process group, /proc adopted-child discovery, and "
    "kernel waitpid ECHILD proofs. The launcher itself must have the exact declared environment, "
    "one task, default SIGCHLD handling, and no children before consumption. It kills and reaps "
    "observed descendants, then requires waitpid(-1, WNOHANG|__WALL) to report ECHILD before "
    "local validation. The pinned Python source closure is statically rejected if it imports or "
    "calls declared process/session primitives. This does not attest native extensions or "
    "transitive native imports against undisclosed process creation, and it does not claim full "
    "descendant containment if the launcher itself is abruptly killed."
)
CONTAINMENT_CONTRACT: Final[dict[str, Any]] = {
    "mechanism": "linux-subreaper-process-group-waitpid-echild-v1",
    "cgroup_v2_used": False,
    "cgroup_v2_writable": False,
    "launcher_crash_containment": False,
    "parent_environment_exact_required": True,
    "single_launcher_task_required": True,
    "default_sigchld_disposition_required": True,
    "empty_blocked_signal_mask_required": True,
    "inspection_monotonic_deadline_alarm_required": True,
    "prior_launcher_children_must_be_empty": True,
    "direct_pid_and_process_group_killed_on_failure": True,
    "adopted_children_killed_and_terminally_reaped": True,
    "adopted_child_pidfd_signaling_used_when_observed": True,
    "terminal_waitpid_echild_proof_required": True,
    "observed_descendant_permanently_rejects_repeat": True,
    "static_python_process_escape_scan_required": True,
    "native_process_escape_attested": False,
    "evidence_scope": CONTAINMENT_SCOPE,
}
REPEAT_POLICY: Final[dict[str, Any]] = {
    "order": list(REPEATS),
    "attempts_per_repeat": 1,
    "retry_after_consumption": False,
    "stop_after_failure": True,
    "maximum_repeat": "r3",
    "previous_static_validation_required": True,
}
PREFLIGHT_CHECKS: Final[list[str]] = [
    "experiment_spec",
    "public_launcher_and_validator_source",
    "pinned_source_files",
    "pinned_source_commit_and_clean_status",
    "signed_interpreter",
    "conversion_replay_bundles",
    "artifact_tree_and_entrypoint",
    "public_diagnostic_dataset",
    "historical_training_lineage",
    "signed_runtime",
    "static_python_process_escape_scan",
]
NORMALIZED_PREFLIGHT_CHECKS: Final[list[str]] = [
    "experiment_spec",
    "public_launcher_validator_and_spec_source",
    "pinned_normalized_source_files",
    "pinned_normalized_source_commit_and_clean_status",
    "signed_interpreter",
    "completed_v6_training_lineage",
    "normalized_v4_local_quality_isolation_conversion_bundle",
    "candidate_artifact_tree_entrypoint_and_load_spec",
    "public_diagnostic_dataset",
    "signed_runtime",
    "static_python_process_escape_scan",
]
CURRENT94_PREFLIGHT_CHECKS: Final[list[str]] = [
    "final_current94_v8_experiment_spec",
    "public_launcher_validator_and_spec_source",
    "pinned_current94_source_files_commit_and_clean_status",
    "signed_evaluator_interpreter",
    "signed_microtensor_release_0_3_2_mechanism_0_3_0",
    "completed_v4_final_all_public_94_0_training_lineage",
    "qwen25_qwen2_conversion_v6_calibration_v3_bundle",
    "converter_interpreter_and_runtime_closure_receipt_content_bindings",
    "candidate_artifact_tree_entrypoint_and_load_spec",
    "public_training_overlap_diagnostic_dataset",
    "generated_and_corpus_code_no_execution_contract",
    "reviewed_hermetic_live_containment",
]


class LaunchRefused(RuntimeError):
    """The launch or any declared binding failed closed."""


class LaunchProcessRefused(LaunchRefused):
    """A launch failed after acquiring typed child-boundary evidence."""

    def __init__(
        self,
        message: str,
        *,
        process: Mapping[str, Any],
        inspection: Mapping[str, Any] | None,
        containment: Mapping[str, Any] | None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.process = dict(process)
        self.inspection = None if inspection is None else dict(inspection)
        self.containment = None if containment is None else dict(containment)
        self.original = original


@dataclass(frozen=True)
class FileIdentity:
    """Stable content identity for a regular, non-symlink file."""

    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class Invocation:
    """One exact evaluator invocation from the immutable addendum."""

    repeat: str
    argv: tuple[str, ...]
    output_root: Path
    argv_canonical_json_bytes: int
    argv_canonical_json_sha256: str


@dataclass(frozen=True)
class LaunchContract:
    """Fully checked operational subset of the immutable addendum."""

    path: Path
    raw: bytes
    digest: str
    public_commit: str
    public_git: dict[str, Any]
    repository_root: Path
    experiment_spec: Path
    validator_path: Path
    interpreter_path: Path
    interpreter_resolved: Path
    environment: dict[str, str]
    report_root: Path
    invocations: tuple[Invocation, Invocation, Invocation]
    protocol: str = "v6"
    source_root: Path = SOURCE_ROOT
    source_commit: str = SOURCE_COMMIT
    experiment_spec_identity: FileIdentity = FileIdentity(SPEC_BYTES, SPEC_DIGEST)
    artifact_tree_digest: str = ARTIFACT_TREE_DIGEST
    artifact_entrypoint_bytes: int = ARTIFACT_ENTRYPOINT_BYTES
    artifact_entrypoint_digest: str = ARTIFACT_ENTRYPOINT_DIGEST
    interpreter_identity: FileIdentity = FileIdentity(INTERPRETER_BYTES, INTERPRETER_DIGEST)
    validation_schema: str = LEGACY_VALIDATION_SCHEMA
    artifact_use_policy: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ProcessOutcome:
    """Observed result of one direct, traced evaluator exec."""

    pid: int
    returncode: int
    timed_out: bool
    inspection: dict[str, Any]
    containment: dict[str, Any]
    started_at_utc: str
    started_at_unix_ns: int
    finished_at_utc: str
    finished_at_unix_ns: int


@dataclass(frozen=True)
class Preflight:
    """Static preflight evidence plus the exact loaded validator."""

    report: dict[str, Any]
    validator: ModuleType


@dataclass(frozen=True)
class PriorReceiptBinding:
    """Canonical preflight and validation identities from one prior receipt."""

    preflight_sha256: str
    validation_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LaunchRefused(f"value is not finite canonical JSON: {exc}") from exc


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise LaunchRefused(f"value is not finite JSON: {exc}") from exc


def _digest_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _timestamp() -> tuple[str, int]:
    unix_ns = time.time_ns()
    rendered = (
        datetime.fromtimestamp(unix_ns / 1_000_000_000, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    return rendered, unix_ns


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchRefused(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LaunchRefused(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value) or frozenset(value) != expected:
        raise LaunchRefused(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(value)}"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LaunchRefused(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise LaunchRefused(f"{label} must be lowercase sha256:<64 hex>")
    return value


def _strict_json(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LaunchRefused(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise LaunchRefused(f"{label} contains non-finite value {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except LaunchRefused:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise LaunchRefused(f"{label} is not strict UTF-8 JSON: {exc}") from exc

    def validate(value: Any, location: str) -> None:
        if isinstance(value, str):
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise LaunchRefused(f"{location} contains a lone surrogate") from exc
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise LaunchRefused(f"{location} contains a non-finite number")
        elif type(value) is int:
            if not -(2**63) <= value <= 2**63 - 1:
                raise LaunchRefused(f"{location} integer is outside signed 64-bit range")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                validate(key, f"{location} key")
                validate(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                validate(item, f"{location}[{index}]")
        elif value is not None and type(value) is not bool:
            raise LaunchRefused(f"{location} has an unsupported JSON type")

    validate(payload, label)
    return payload


def _stable_regular_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be inspected: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise LaunchRefused(f"{label} must be a regular non-symlink file: {path}")
    if before.st_size > maximum:
        raise LaunchRefused(f"{label} exceeds the {maximum}-byte ceiling")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be opened safely: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise LaunchRefused(f"{label} opened as a non-regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum:
            raise LaunchRefused(f"{label} exceeds the {maximum}-byte ceiling")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be read safely: {exc}") from exc
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        any(getattr(before, field) != getattr(opened, field) for field in stable_fields)
        or any(getattr(opened, field) != getattr(after, field) for field in stable_fields)
        or len(raw) != after.st_size
    ):
        raise LaunchRefused(f"{label} changed while it was read")
    return raw


def _file_identity(path: Path, label: str, *, maximum: int = MAX_RECEIPT_BYTES) -> FileIdentity:
    raw = _stable_regular_bytes(path, label, maximum=maximum)
    return FileIdentity(bytes=len(raw), sha256=_digest_bytes(raw))


def _parse_identity(value: Any, label: str) -> FileIdentity:
    payload = _mapping(value, label)
    _exact_keys(payload, frozenset({"bytes", "sha256"}), label)
    return FileIdentity(
        bytes=_integer(payload.get("bytes"), f"{label} bytes"),
        sha256=_digest(payload.get("sha256"), f"{label} digest"),
    )


def _require_identity(actual: FileIdentity, expected: FileIdentity, label: str) -> None:
    if actual != expected:
        raise LaunchRefused(
            f"{label} identity changed: expected {expected.as_dict()}, got {actual.as_dict()}"
        )


def _expected_argv(output_root: Path) -> tuple[str, ...]:
    return (
        str(INTERPRETER_PATH),
        "-m",
        "training.evaluate_code_gguf",
        "--dataset",
        str(DATASET),
        "--diagnostic-jsonl",
        str(DIAGNOSTIC_JSONL),
        "--source-corpus",
        str(CURRENT_SOURCE_CORPUS),
        "--artifact",
        str(ARTIFACT_ROOT),
        "--artifact-digest",
        ARTIFACT_TREE_DIGEST,
        "--entrypoint",
        "model.gguf",
        "--quantization",
        "Q4_K_M",
        "--max-input-tokens",
        "541",
        "--training-run",
        str(TRAINING_RUN),
        "--training-dataset",
        str(TRAINING_DATASET),
        "--training-source-corpus",
        str(TRAINING_SOURCE_CORPUS),
        "--training-base",
        str(TRAINING_BASE),
        "--out",
        str(output_root),
    )


def expected_invocations() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the exact addendum invocation records, useful to its generator."""

    records: list[dict[str, Any]] = []
    for repeat, output_root in zip(REPEATS, OUTPUT_ROOTS, strict=True):
        argv = list(_expected_argv(output_root))
        raw = _canonical_json_bytes(argv)
        records.append(
            {
                "repeat": repeat,
                "argv": argv,
                "argv_canonical_json_bytes": len(raw),
                "argv_canonical_json_sha256": _digest_bytes(raw),
                "output_root": str(output_root),
            }
        )
    return records[0], records[1], records[2]
def _is_current94_live_request_lexically(path: Path) -> bool:
    """Recognize the declared current94 live alias without filesystem access."""

    value = os.path.normpath(os.fspath(path))
    if os.path.isabs(value):
        return value == os.path.normpath(os.fspath(CURRENT94_DECLARATION_LEXICAL_PATH))
    return value == os.path.normpath(CURRENT94_ADDENDUM_RELATIVE)


def _refuse_current94_live() -> None:
    raise LaunchRefused(CURRENT94_LIVE_BLOCKER)






def _repository_root() -> Path:
    source = Path(__file__)
    if source.is_symlink():
        raise LaunchRefused("launcher source path must not be a symlink")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(f"launcher source path cannot be resolved: {exc}") from exc
    root = resolved.parent.parent
    if root / LAUNCHER_RELATIVE != resolved:
        raise LaunchRefused("launcher source is outside the expected repository layout")
    return root


def _normalized_content_identity(value: Any, label: str) -> FileIdentity:
    payload = _mapping(value, label)
    _exact_keys(payload, frozenset({"bytes", "digest"}), label)
    return FileIdentity(
        bytes=_integer(payload.get("bytes"), f"{label} bytes", minimum=1),
        sha256=_digest(payload.get("digest"), f"{label} digest"),
    )


def _normalized_artifact_use_policy() -> dict[str, Any]:
    return {
        "intended_use": "local_quality_isolation_only",
        "historical_conversion_environment": "not_recorded",
        "historical_conversion_path": "not_recorded",
        "historical_converter_interpreter": "not_recorded",
        "historical_converter_dependencies": "not_recorded",
        "historical_quantizer_library_closure": "not_recorded",
        "conversion_runtime_closure_attested": False,
        "publication_eligible": False,
        "submission_eligible": False,
        "publication_authorized": False,
        "submission_authorized": False,
        "limitation": (
            "generic v4 conversion does not record its historical environment, PATH, "
            "Python interpreter, converter dependencies, or quantizer library closure; "
            "this artifact is permanently publication- and submission-ineligible"
        ),
    }


def _current94_artifact_use_policy() -> dict[str, Any]:
    return {
        "intended_use": "local_training_overlap_structural_and_timing_diagnostic_only",
        "training_overlap": True,
        "conversion_v6_bound": True,
        "calibration_v3_bound": True,
        "conversion_runtime_receipt_content_bound": True,
        "converter_interpreter_portable_receipt_content_bound": True,
        "executed_interpreter_attested": False,
        "hermetic_conversion_attested": False,
        "conversion_runtime_execution_verified": False,
        "generated_or_corpus_code_executed_by_this_static_validator": False,
        "execution_pass_at_1_claimed": False,
        "quality_or_rank_claimed": False,
        "publication_authorized": False,
        "submission_authorized": False,
        "transaction_authorized": False,
        "limitation": (
            "all 16 public diagnostic rows overlap the final 94/0 training lineage; local "
            "structural and timing diagnostics are not holdout evidence, execution pass@1, "
            "an official validator measurement, or a settled-rank certificate"
        ),
    }


CURRENT94_SAFETY_CONTRACT: Final[dict[str, bool]] = {
    "generated_code_imported_by_this_static_validator": False,
    "generated_code_executed_by_this_static_validator": False,
    "generated_code_bytecode_compiled_by_this_static_validator": False,
    "corpus_code_imported_by_this_static_validator": False,
    "corpus_code_executed_by_this_static_validator": False,
    "corpus_code_bytecode_compiled_by_this_static_validator": False,
    "static_ast_parse_only_by_this_static_validator": True,
    "hidden_or_scored_tests_accessed_by_this_static_validator": False,
}
CURRENT94_GATES: Final[dict[str, int]] = {
    "successful_generations_minimum": 16,
    "failed_generations_maximum": 0,
    "scorer_extracted_parseable_python_minimum": 16,
    "scorer_extracted_top_level_task_func_minimum": 16,
    "scorer_extracted_residual_fences_maximum": 0,
    "maximum_request_latency_ms": 31_200,
    "p95_ttft_ms_maximum": 5_256,
    "p95_ttft_ms_preferred": 4_700,
    "peak_rss_bytes_maximum": 1_073_741_824,
    "maximum_stream_pieces_per_request": 373,
}
CURRENT94_REQUIRED_SOURCE_FILES: Final[frozenset[str]] = frozenset(
    {
        "training/code_candidate.py",
        "training/convert_code_gguf.py",
        "training/evaluate_code.py",
        "training/evaluate_code_gguf.py",
        "training/historical_code_candidate.py",
        "training/normalized_historical_code_candidate.py",
        "training/publish_code_provenance.py",
        "training/train_code.py",
    }
)


def _normalized_spec_values(raw: bytes) -> dict[str, Any]:
    payload = _mapping(_strict_json(raw, "normalized v7 diagnostic spec"), "normalized spec")
    if raw != _pretty_json_bytes(payload):
        raise LaunchRefused("normalized v7 diagnostic spec is not canonical sorted JSON")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema",
                "status",
                "artifact_use_policy",
                "candidate",
                "source",
                "diagnostic",
                "training_lineage",
                "conversion",
                "runtime",
                "gates",
            }
        ),
        "normalized v7 diagnostic spec",
    )
    if payload.get("schema") != NORMALIZED_SPEC_SCHEMA or payload.get("status") != "final":
        raise LaunchRefused("normalized v7 diagnostic spec is not final")
    candidate = _mapping(payload.get("candidate"), "normalized candidate")
    _exact_keys(
        candidate,
        frozenset(
            {
                "id",
                "base_model",
                "bundle",
                "entrypoint",
                "quantization",
                "max_input_tokens",
                "tokenizer_json",
            }
        ),
        "normalized candidate",
    )
    if (
        candidate.get("id") != "qwen3-06b-historical7730-normalized-v7-q4-m541-py311"
        or candidate.get("base_model") != "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
        or candidate.get("bundle") != str(NORMALIZED_BUNDLE_ROOT)
        or candidate.get("entrypoint") != "model.gguf"
        or candidate.get("quantization") != "Q4_K_M"
        or candidate.get("max_input_tokens") != 541
        or candidate.get("tokenizer_json")
        != {
            "bytes": 11_422_654,
            "digest": ("sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        }
    ):
        raise LaunchRefused("normalized Qwen/tokenizer/candidate contract changed")
    source = _mapping(payload.get("source"), "normalized source")
    _exact_keys(source, frozenset({"commit", "root", "files"}), "normalized source")
    source_commit = source.get("commit")
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise LaunchRefused("normalized source commit is malformed")
    source_root = Path(str(source.get("root")))
    expected_source_root = Path("/tmp") / f"mt92-normalized-diagnostic-{source_commit[:7]}"  # noqa: S108
    if source_root != expected_source_root:
        raise LaunchRefused(f"normalized source root must be {expected_source_root}")
    source_files = _mapping(source.get("files"), "normalized source files")
    required_source_files = frozenset(
        {
            "training/code_candidate.py",
            "training/convert_code_gguf.py",
            "training/evaluate_code.py",
            "training/evaluate_code_gguf.py",
            "training/historical_code_candidate.py",
            "training/normalized_historical_code_candidate.py",
            "training/train_code.py",
        }
    )
    _exact_keys(source_files, required_source_files, "normalized source files")
    for relative, identity in source_files.items():
        _normalized_content_identity(identity, f"normalized source {relative}")

    diagnostic = _mapping(payload.get("diagnostic"), "normalized diagnostic")
    expected_diagnostic_scalars = {
        "dataset": str(NORMALIZED_DATASET),
        "diagnostic_jsonl": str(NORMALIZED_DIAGNOSTIC_JSONL),
        "source_corpus": str(NORMALIZED_CURRENT_SOURCE),
        "refs_digest": ("sha256:73edc2a7674e0c718ea4ef7ea67c638b1a2c431320789b632aad5909309e01ee"),
        "examples": 16,
        "output_roots": [str(item) for item in NORMALIZED_OUTPUT_ROOTS],
    }
    for field, expected in expected_diagnostic_scalars.items():
        if diagnostic.get(field) != expected:
            raise LaunchRefused(f"normalized diagnostic {field} changed")

    training = _mapping(payload.get("training_lineage"), "normalized training lineage")
    expected_training = {
        "schema": "microtensor.code.training.v6",
        "training_run": str(NORMALIZED_TRAINING_RUN),
        "training_dataset": str(NORMALIZED_TRAINING_DATASET),
        "source_corpus": str(NORMALIZED_TRAINING_SOURCE),
        "base": str(NORMALIZED_TRAINING_BASE),
        "dataset_schema": "microtensor.code.prepared.historical-normalized.v1",
        "corpus_profile": "historical7730-normalized-v1",
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise LaunchRefused(f"normalized training {field} changed")
    _normalized_content_identity(training.get("receipt"), "normalized training receipt")
    _digest(training.get("merged_tree_digest"), "normalized merged tree digest")

    conversion = _mapping(payload.get("conversion"), "normalized conversion")
    _exact_keys(
        conversion,
        frozenset({"schema", "receipt", "calibration_receipt", "load_spec", "artifact"}),
        "normalized conversion",
    )
    conversion_schema = conversion.get("schema")
    if conversion_schema != "microtensor.code.gguf-conversion.v4":
        raise LaunchRefused("normalized v7 accepts only the generic v4 conversion schema")
    _normalized_content_identity(conversion.get("receipt"), "normalized conversion receipt")
    _normalized_content_identity(conversion.get("load_spec"), "normalized load spec")
    calibration = conversion.get("calibration_receipt")
    if calibration is not None:
        raise LaunchRefused("normalized v4 conversion gained calibration")
    artifact = _mapping(conversion.get("artifact"), "normalized artifact")
    _exact_keys(
        artifact,
        frozenset({"tree_digest", "entrypoint_bytes", "entrypoint_digest"}),
        "normalized artifact",
    )
    artifact_values = {
        "tree_digest": _digest(artifact.get("tree_digest"), "normalized artifact tree"),
        "entrypoint_bytes": _integer(
            artifact.get("entrypoint_bytes"), "normalized artifact bytes", minimum=1
        ),
        "entrypoint_digest": _digest(
            artifact.get("entrypoint_digest"), "normalized artifact entrypoint"
        ),
    }
    runtime = _mapping(payload.get("runtime"), "normalized runtime")
    _exact_keys(runtime, frozenset({"identity", "interpreter"}), "normalized runtime")
    _normalized_content_identity(runtime.get("identity"), "normalized runtime identity")
    interpreter = _mapping(runtime.get("interpreter"), "normalized interpreter")
    if interpreter != {
        "path": str(INTERPRETER_PATH),
        "resolved_path": str(INTERPRETER_RESOLVED),
        "bytes": INTERPRETER_BYTES,
        "digest": INTERPRETER_DIGEST,
    }:
        raise LaunchRefused("normalized signed interpreter contract changed")
    artifact_use_policy = _mapping(
        payload.get("artifact_use_policy"), "normalized artifact use policy"
    )
    expected_use_policy = _normalized_artifact_use_policy()
    _exact_keys(
        artifact_use_policy,
        frozenset(expected_use_policy),
        "normalized artifact use policy",
    )
    if _canonical_json_bytes(artifact_use_policy) != _canonical_json_bytes(expected_use_policy):
        raise LaunchRefused("normalized artifact use policy changed")
    return {
        "payload": dict(payload),
        "source_root": source_root,
        "source_commit": source_commit,
        "artifact": artifact_values,
        "artifact_use_policy": dict(artifact_use_policy),
    }


def _current94_spec_values(raw: bytes) -> dict[str, Any]:
    payload = _mapping(_strict_json(raw, "current94 v8 diagnostic spec"), "current94 spec")
    if raw != _pretty_json_bytes(payload):
        raise LaunchRefused("current94 v8 diagnostic spec is not canonical sorted JSON")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema",
                "status",
                "artifact_use_policy",
                "safety_contract",
                "candidate",
                "source",
                "diagnostic",
                "training_lineage",
                "conversion",
                "runtime",
                "gates",
            }
        ),
        "current94 v8 diagnostic spec",
    )
    if payload.get("schema") != CURRENT94_SPEC_SCHEMA or payload.get("status") != "final":
        raise LaunchRefused("current94 v8 diagnostic spec is not final")
    policy = _mapping(payload.get("artifact_use_policy"), "current94 artifact use policy")
    if _canonical_json_bytes(policy) != _canonical_json_bytes(_current94_artifact_use_policy()):
        raise LaunchRefused("current94 artifact use policy changed")
    if _canonical_json_bytes(payload.get("safety_contract")) != _canonical_json_bytes(
        CURRENT94_SAFETY_CONTRACT
    ):
        raise LaunchRefused("current94 no-execution safety contract changed")

    candidate = _mapping(payload.get("candidate"), "current94 candidate")
    _exact_keys(
        candidate,
        frozenset(
            {
                "id",
                "base_model",
                "gguf_architecture",
                "bundle",
                "entrypoint",
                "quantization",
                "max_input_tokens",
            }
        ),
        "current94 candidate",
    )
    expected_candidate = {
        "id": CURRENT94_CANDIDATE_ID,
        "base_model": CURRENT94_BASE_MODEL,
        "gguf_architecture": "qwen2",
        "bundle": str(CURRENT94_BUNDLE_ROOT),
        "entrypoint": "model.gguf",
        "quantization": "Q4_K_M",
        "max_input_tokens": 541,
    }
    if dict(candidate) != expected_candidate:
        raise LaunchRefused("current94 Qwen2.5/qwen2 candidate contract changed")

    source = _mapping(payload.get("source"), "current94 source")
    _exact_keys(source, frozenset({"commit", "root", "files"}), "current94 source")
    source_commit = source.get("commit")
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise LaunchRefused("current94 source commit is malformed")
    source_root = Path(str(source.get("root")))
    expected_source_root = Path("/tmp") / f"mt92-current94-diagnostic-{source_commit[:7]}"  # noqa: S108
    if source_root != expected_source_root:
        raise LaunchRefused(f"current94 source root must be {expected_source_root}")
    source_files = _mapping(source.get("files"), "current94 source files")
    _exact_keys(source_files, CURRENT94_REQUIRED_SOURCE_FILES, "current94 source files")
    for relative, identity in source_files.items():
        _normalized_content_identity(identity, f"current94 source {relative}")

    diagnostic = _mapping(payload.get("diagnostic"), "current94 diagnostic")
    _exact_keys(
        diagnostic,
        frozenset(
            {
                "dataset",
                "diagnostic_jsonl",
                "source_corpus",
                "manifest",
                "holdout",
                "source",
                "refs_digest",
                "examples",
                "relationship_to_training",
                "output_roots",
            }
        ),
        "current94 diagnostic",
    )
    expected_diagnostic = {
        "dataset": str(DATASET),
        "diagnostic_jsonl": str(DIAGNOSTIC_JSONL),
        "source_corpus": str(CURRENT94_TRAINING_SOURCE),
        "manifest": {
            "bytes": 1_070,
            "digest": "sha256:6af4fe8952293339773e133a867d78e817d759a83138bc7400af98e2e04898ff",
        },
        "holdout": {
            "bytes": 23_390,
            "digest": "sha256:8b07f781c6d160f752963b3a42f343c18d53a785d7b7cd09f472fdddbd2d7993",
        },
        "source": {
            "bytes": 152_605,
            "digest": "sha256:1c37a0e212936bfac8c86f955ad61fd378f58603413b45ece88382d528ace9d5",
        },
        "refs_digest": "sha256:73edc2a7674e0c718ea4ef7ea67c638b1a2c431320789b632aad5909309e01ee",
        "examples": 16,
        "relationship_to_training": "training_overlap",
        "output_roots": [str(path) for path in CURRENT94_OUTPUT_ROOTS],
    }
    if _canonical_json_bytes(diagnostic) != _canonical_json_bytes(expected_diagnostic):
        raise LaunchRefused("current94 training-overlap diagnostic contract changed")

    training = _mapping(payload.get("training_lineage"), "current94 training lineage")
    _exact_keys(
        training,
        frozenset(
            {
                "schema",
                "run_kind",
                "training_run",
                "training_dataset",
                "source_corpus",
                "base",
                "receipt",
                "metrics",
                "merged_tree_digest",
                "dataset_schema",
                "corpus_profile",
                "train_examples",
                "holdout_examples",
                "quality_claim",
            }
        ),
        "current94 training lineage",
    )
    expected_training = {
        "schema": "microtensor.code.training.v4",
        "run_kind": "final_all_public",
        "training_run": str(CURRENT94_TRAINING_RUN),
        "training_dataset": str(CURRENT94_TRAINING_DATASET),
        "source_corpus": str(CURRENT94_TRAINING_SOURCE),
        "base": str(CURRENT94_TRAINING_BASE),
        "dataset_schema": "microtensor.code.prepared.v1",
        "corpus_profile": "bigcodebench94",
        "train_examples": 94,
        "holdout_examples": 0,
        "quality_claim": (
            "none: all 94 public examples were used for training; public code tests are "
            "withheld; no holdout or execution pass@1 was measured"
        ),
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise LaunchRefused(f"current94 training field {field} changed")
    _normalized_content_identity(training.get("receipt"), "current94 training receipt")
    _normalized_content_identity(training.get("metrics"), "current94 training metrics")
    _digest(training.get("merged_tree_digest"), "current94 merged tree digest")

    conversion = _mapping(payload.get("conversion"), "current94 conversion")
    _exact_keys(
        conversion,
        frozenset(
            {
                "schema",
                "calibration_schema",
                "receipt",
                "calibration_receipt",
                "load_spec",
                "artifact",
                "runtime_receipt_content_binding",
            }
        ),
        "current94 conversion",
    )
    if (
        conversion.get("schema") != "microtensor.code.gguf-conversion.v6"
        or conversion.get("calibration_schema")
        != "microtensor.code.imatrix-calibration.v3"
    ):
        raise LaunchRefused("current94 conversion-v6/calibration-v3 binding changed")
    for field in ("receipt", "calibration_receipt", "load_spec"):
        _normalized_content_identity(conversion.get(field), f"current94 conversion {field}")
    artifact = _mapping(conversion.get("artifact"), "current94 artifact")
    _exact_keys(
        artifact,
        frozenset({"tree_digest", "entrypoint_bytes", "entrypoint_digest"}),
        "current94 artifact",
    )
    artifact_values = {
        "tree_digest": _digest(artifact.get("tree_digest"), "current94 artifact tree"),
        "entrypoint_bytes": _integer(
            artifact.get("entrypoint_bytes"), "current94 artifact bytes", minimum=1
        ),
        "entrypoint_digest": _digest(
            artifact.get("entrypoint_digest"), "current94 artifact entrypoint"
        ),
    }
    conversion_runtime = _mapping(
        conversion.get("runtime_receipt_content_binding"), "current94 conversion runtime"
    )
    _exact_keys(
        conversion_runtime,
        frozenset({"converter_interpreter", "llama_cpp_runtime_closure"}),
        "current94 conversion runtime",
    )
    converter_interpreter = _mapping(
        conversion_runtime.get("converter_interpreter"),
        "current94 converter interpreter",
    )
    _exact_keys(
        converter_interpreter,
        frozenset({"container_path", "bytes", "digest", "mode"}),
        "current94 converter interpreter",
    )
    if (
        converter_interpreter.get("container_path")
        != "/.uv/python_install/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
        or converter_interpreter.get("mode") != "0o755"
    ):
        raise LaunchRefused("current94 converter interpreter container path or mode changed")
    _integer(converter_interpreter.get("bytes"), "current94 converter Python bytes", minimum=1)
    _digest(converter_interpreter.get("digest"), "current94 converter Python digest")
    _normalized_content_identity(
        conversion_runtime.get("llama_cpp_runtime_closure"),
        "current94 llama.cpp runtime closure",
    )

    runtime = _mapping(payload.get("runtime"), "current94 runtime")
    _exact_keys(
        runtime,
        frozenset({"release_version", "mechanism_version", "identity", "interpreter"}),
        "current94 runtime",
    )
    if (
        runtime.get("release_version") != "0.3.2"
        or runtime.get("mechanism_version") != "0.3.0"
    ):
        raise LaunchRefused("current94 signed release/mechanism changed")
    _normalized_content_identity(runtime.get("identity"), "current94 signed runtime identity")
    if _mapping(runtime.get("interpreter"), "current94 signed interpreter") != {
        "path": str(INTERPRETER_PATH),
        "resolved_path": str(INTERPRETER_RESOLVED),
        "bytes": INTERPRETER_BYTES,
        "digest": INTERPRETER_DIGEST,
    }:
        raise LaunchRefused("current94 signed evaluator interpreter changed")
    if _mapping(payload.get("gates"), "current94 gates") != CURRENT94_GATES:
        raise LaunchRefused("current94 diagnostic gates changed")
    return {
        "payload": dict(payload),
        "source_root": source_root,
        "source_commit": source_commit,
        "artifact": artifact_values,
        "artifact_use_policy": dict(policy),
        "conversion_runtime": dict(conversion_runtime),
    }


def _normalized_v7_expected_invocations(
    spec_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    values = _normalized_spec_values(_pretty_json_bytes(spec_payload))
    artifact = _mapping(values["artifact"], "normalized artifact")
    records: list[dict[str, Any]] = []
    for repeat, output_root in zip(REPEATS, NORMALIZED_OUTPUT_ROOTS, strict=True):
        argv = [
            str(INTERPRETER_PATH),
            "-m",
            "training.evaluate_code_gguf",
            "--dataset",
            str(NORMALIZED_DATASET),
            "--diagnostic-jsonl",
            str(NORMALIZED_DIAGNOSTIC_JSONL),
            "--source-corpus",
            str(NORMALIZED_CURRENT_SOURCE),
            "--artifact",
            str(NORMALIZED_BUNDLE_ROOT / "artifact"),
            "--artifact-digest",
            str(artifact["tree_digest"]),
            "--entrypoint",
            "model.gguf",
            "--quantization",
            "Q4_K_M",
            "--max-input-tokens",
            "541",
            "--training-run",
            str(NORMALIZED_TRAINING_RUN),
            "--training-dataset",
            str(NORMALIZED_TRAINING_DATASET),
            "--training-source-corpus",
            str(NORMALIZED_TRAINING_SOURCE),
            "--training-base",
            str(NORMALIZED_TRAINING_BASE),
            "--out",
            str(output_root),
        ]
        argv_raw = _canonical_json_bytes(argv)
        records.append(
            {
                "repeat": repeat,
                "argv": argv,
                "argv_canonical_json_bytes": len(argv_raw),
                "argv_canonical_json_sha256": _digest_bytes(argv_raw),
                "output_root": str(output_root),
            }
        )
    return records[0], records[1], records[2]


def _current94_expected_invocations(
    spec_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    values = _current94_spec_values(_pretty_json_bytes(spec_payload))
    artifact = _mapping(values["artifact"], "current94 artifact")
    records: list[dict[str, Any]] = []
    for repeat, output_root in zip(REPEATS, CURRENT94_OUTPUT_ROOTS, strict=True):
        argv = [
            str(INTERPRETER_PATH),
            "-m",
            "training.evaluate_code_gguf",
            "--dataset",
            str(DATASET),
            "--diagnostic-jsonl",
            str(DIAGNOSTIC_JSONL),
            "--source-corpus",
            str(CURRENT94_TRAINING_SOURCE),
            "--artifact",
            str(CURRENT94_BUNDLE_ROOT / "artifact"),
            "--artifact-digest",
            str(artifact["tree_digest"]),
            "--entrypoint",
            "model.gguf",
            "--quantization",
            "Q4_K_M",
            "--max-input-tokens",
            "541",
            "--training-run",
            str(CURRENT94_TRAINING_RUN),
            "--training-dataset",
            str(CURRENT94_TRAINING_DATASET),
            "--training-source-corpus",
            str(CURRENT94_TRAINING_SOURCE),
            "--training-base",
            str(CURRENT94_TRAINING_BASE),
            "--out",
            str(output_root),
        ]
        argv_raw = _canonical_json_bytes(argv)
        records.append(
            {
                "repeat": repeat,
                "argv": argv,
                "argv_canonical_json_bytes": len(argv_raw),
                "argv_canonical_json_sha256": _digest_bytes(argv_raw),
                "output_root": str(output_root),
            }
        )
    return records[0], records[1], records[2]


def current94_v8_static_addendum_payload(
    *,
    experiment_spec_raw: bytes,
    public_commit: str,
    public_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a non-executable current94 declaration from final observed identities."""

    values = _current94_spec_values(experiment_spec_raw)
    if _COMMIT.fullmatch(public_commit) is None:
        raise LaunchRefused("current94 public commit must be 40 lowercase hex characters")
    required_public_files = frozenset(
        {LAUNCHER_RELATIVE, VALIDATOR_RELATIVE, CURRENT94_SPEC_RELATIVE}
    )
    if frozenset(public_files) != required_public_files:
        raise LaunchRefused("current94 public source file closure changed")
    normalized_files = {
        relative: _parse_identity(public_files[relative], f"current94 public {relative}").as_dict()
        for relative in sorted(public_files)
    }
    spec_identity = FileIdentity(len(experiment_spec_raw), _digest_bytes(experiment_spec_raw))
    if normalized_files[CURRENT94_SPEC_RELATIVE] != spec_identity.as_dict():
        raise LaunchRefused("current94 public spec identity differs from supplied bytes")
    artifact = _mapping(values["artifact"], "current94 artifact")
    return {
        "schema": CURRENT94_ADDENDUM_SCHEMA,
        "status": "blocked",
        "blocker": CURRENT94_LIVE_BLOCKER,
        "public_source": {
            "repository": REPOSITORY,
            "commit": public_commit,
            "identity_scope": "declared_content_identities_for_offline_static_inspection_only",
            "files": normalized_files,
        },
        "experiment_spec": {
            "path": CURRENT94_SPEC_RELATIVE,
            **spec_identity.as_dict(),
        },
        "artifact_use_policy": dict(values["artifact_use_policy"]),
        "interpreter": {
            "path": str(INTERPRETER_PATH),
            "resolved_path": str(INTERPRETER_RESOLVED),
            "bytes": INTERPRETER_BYTES,
            "sha256": INTERPRETER_DIGEST,
        },
        "static_preflight": {
            "validator_path": VALIDATOR_RELATIVE,
            "validator_mode": "in_process_static_only",
            "model_engine_construction_permitted": False,
            "live_launch_permitted": False,
            "required_checks": list(CURRENT94_PREFLIGHT_CHECKS),
            "source_commit": values["source_commit"],
            "artifact_tree_sha256": artifact["tree_digest"],
            "artifact_entrypoint_bytes": artifact["entrypoint_bytes"],
            "artifact_entrypoint_sha256": artifact["entrypoint_digest"],
            "conversion_runtime_receipt_content_binding": dict(values["conversion_runtime"]),
        },
        "planned_invocations": list(_current94_expected_invocations(values["payload"])),
    }


def normalized_v7_addendum_payload(
    *,
    experiment_spec_raw: bytes,
    public_commit: str,
    public_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a final v7 addendum deterministically; never invent missing hashes."""

    values = _normalized_spec_values(experiment_spec_raw)
    if _COMMIT.fullmatch(public_commit) is None:
        raise LaunchRefused("normalized public commit must be 40 lowercase hex characters")
    required_public_files = frozenset(
        {LAUNCHER_RELATIVE, VALIDATOR_RELATIVE, NORMALIZED_SPEC_RELATIVE}
    )
    if frozenset(public_files) != required_public_files:
        raise LaunchRefused("normalized public source file closure changed")
    normalized_files = {
        relative: _parse_identity(public_files[relative], f"normalized public {relative}").as_dict()
        for relative in sorted(public_files)
    }
    spec_identity = FileIdentity(len(experiment_spec_raw), _digest_bytes(experiment_spec_raw))
    if normalized_files[NORMALIZED_SPEC_RELATIVE] != spec_identity.as_dict():
        raise LaunchRefused("normalized public spec identity differs from supplied bytes")
    artifact = _mapping(values["artifact"], "normalized artifact")
    artifact_use_policy = _mapping(values["artifact_use_policy"], "normalized artifact use policy")
    source_root = Path(str(values["source_root"]))
    source_commit = str(values["source_commit"])
    return {
        "schema": NORMALIZED_ADDENDUM_SCHEMA,
        "status": ADDENDUM_STATUS,
        "public_source": {
            "repository": REPOSITORY,
            "commit": public_commit,
            "raw_readback_verified": True,
            "files": normalized_files,
        },
        "experiment_spec": {
            "path": NORMALIZED_SPEC_RELATIVE,
            **spec_identity.as_dict(),
        },
        "artifact_use_policy": dict(artifact_use_policy),
        "interpreter": {
            "path": str(INTERPRETER_PATH),
            "resolved_path": str(INTERPRETER_RESOLVED),
            "bytes": INTERPRETER_BYTES,
            "sha256": INTERPRETER_DIGEST,
        },
        "execution": {
            "cwd": str(source_root),
            "shell": False,
            "stdin": "/dev/null",
            "umask": UMASK_TEXT,
            "close_fds": True,
            "new_session": True,
            "parent_death_signal": "SIGKILL",
            "timeout_seconds": TIMEOUT_SECONDS,
            "environment_exact": dict(EXACT_ENVIRONMENT),
            "environment_canonical_json_bytes": ENVIRONMENT_CANONICAL_BYTES,
            "environment_canonical_json_sha256": ENVIRONMENT_CANONICAL_DIGEST,
            "parent_environment_exact": True,
            "report_root": str(NORMALIZED_REPORT_ROOT),
            "post_exec_inspection": dict(POST_EXEC_INSPECTION),
            "containment": dict(CONTAINMENT_CONTRACT),
            "repeat_policy": dict(REPEAT_POLICY),
        },
        "preflight": {
            "validator_path": VALIDATOR_RELATIVE,
            "validator_mode": "in_process_static_only",
            "model_engine_construction_permitted": False,
            "required_checks": list(NORMALIZED_PREFLIGHT_CHECKS),
            "source_commit": source_commit,
            "artifact_tree_sha256": artifact["tree_digest"],
            "artifact_entrypoint_bytes": artifact["entrypoint_bytes"],
            "artifact_entrypoint_sha256": artifact["entrypoint_digest"],
        },
        "invocations": list(_normalized_v7_expected_invocations(values["payload"])),
    }


def _load_normalized_v7_contract(path: Path) -> LaunchContract:
    repository_root = _repository_root()
    expected_path = repository_root / NORMALIZED_ADDENDUM_RELATIVE
    try:
        if path.resolve(strict=True) != expected_path:
            raise LaunchRefused(f"normalized addendum must be repository file {expected_path}")
    except OSError as exc:
        raise LaunchRefused(f"normalized diagnostic addendum cannot be resolved: {exc}") from exc
    raw = _stable_regular_bytes(path, "normalized diagnostic addendum", maximum=MAX_ADDENDUM_BYTES)
    payload = _mapping(_strict_json(raw, "normalized diagnostic addendum"), "addendum")
    if raw != _pretty_json_bytes(payload):
        raise LaunchRefused("normalized diagnostic addendum is not canonical sorted JSON")
    public_source = _mapping(payload.get("public_source"), "normalized public source")
    public_commit = public_source.get("commit")
    if not isinstance(public_commit, str):
        raise LaunchRefused("normalized public commit is absent")
    public_files = _mapping(public_source.get("files"), "normalized public files")
    spec_path = repository_root / NORMALIZED_SPEC_RELATIVE
    spec_raw = _stable_regular_bytes(
        spec_path,
        "normalized diagnostic spec",
        maximum=MAX_ADDENDUM_BYTES,
    )
    expected_payload = normalized_v7_addendum_payload(
        experiment_spec_raw=spec_raw,
        public_commit=public_commit,
        public_files=public_files,
    )
    if _canonical_json_bytes(payload) != _canonical_json_bytes(expected_payload):
        raise LaunchRefused("normalized diagnostic addendum contract changed")
    declared_identities = {
        relative: _parse_identity(public_files[relative], f"normalized public {relative}")
        for relative in public_files
    }
    for relative, expected in declared_identities.items():
        _require_identity(
            _file_identity(repository_root / relative, f"normalized public {relative}"),
            expected,
            f"normalized public {relative}",
        )
    public_git = _validate_public_git_binding(
        repository_root,
        public_commit,
        declared_identities,
    )
    values = _normalized_spec_values(spec_raw)
    artifact = _mapping(values["artifact"], "normalized artifact")
    invocations: list[Invocation] = []
    for record in _normalized_v7_expected_invocations(values["payload"]):
        invocations.append(
            Invocation(
                repeat=str(record["repeat"]),
                argv=tuple(str(item) for item in record["argv"]),
                output_root=Path(str(record["output_root"])),
                argv_canonical_json_bytes=int(record["argv_canonical_json_bytes"]),
                argv_canonical_json_sha256=str(record["argv_canonical_json_sha256"]),
            )
        )
    spec_identity = FileIdentity(len(spec_raw), _digest_bytes(spec_raw))
    return LaunchContract(
        path=expected_path,
        raw=raw,
        digest=_digest_bytes(raw),
        public_commit=public_commit,
        public_git=public_git,
        repository_root=repository_root,
        experiment_spec=spec_path,
        validator_path=repository_root / VALIDATOR_RELATIVE,
        interpreter_path=INTERPRETER_PATH,
        interpreter_resolved=INTERPRETER_RESOLVED,
        environment=dict(EXACT_ENVIRONMENT),
        report_root=NORMALIZED_REPORT_ROOT,
        invocations=(invocations[0], invocations[1], invocations[2]),
        protocol="normalized-v7",
        source_root=Path(str(values["source_root"])),
        source_commit=str(values["source_commit"]),
        experiment_spec_identity=spec_identity,
        artifact_tree_digest=str(artifact["tree_digest"]),
        artifact_entrypoint_bytes=int(artifact["entrypoint_bytes"]),
        artifact_entrypoint_digest=str(artifact["entrypoint_digest"]),
        interpreter_identity=FileIdentity(INTERPRETER_BYTES, INTERPRETER_DIGEST),
        validation_schema=NORMALIZED_VALIDATION_SCHEMA,
        artifact_use_policy=tuple(sorted(values["artifact_use_policy"].items())),
    )


def inspect_current94_static_declaration_offline(path: Path) -> dict[str, Any]:
    """Inspect declared local bytes and Git state; never authorize or dispatch a launch."""

    repository_root = _repository_root()
    expected_path = repository_root / CURRENT94_ADDENDUM_RELATIVE
    try:
        if path.resolve(strict=True) != expected_path:
            raise LaunchRefused(f"current94 declaration must be repository file {expected_path}")
    except OSError as exc:
        raise LaunchRefused(f"current94 diagnostic declaration cannot be resolved: {exc}") from exc
    raw = _stable_regular_bytes(
        path,
        "current94 diagnostic declaration",
        maximum=MAX_ADDENDUM_BYTES,
    )
    payload = _mapping(_strict_json(raw, "current94 diagnostic declaration"), "declaration")
    if raw != _pretty_json_bytes(payload):
        raise LaunchRefused("current94 diagnostic declaration is not canonical sorted JSON")
    public_source = _mapping(payload.get("public_source"), "current94 public source")
    public_commit = public_source.get("commit")
    if not isinstance(public_commit, str):
        raise LaunchRefused("current94 public commit is absent")
    public_files = _mapping(public_source.get("files"), "current94 public files")
    spec_path = repository_root / CURRENT94_SPEC_RELATIVE
    spec_raw = _stable_regular_bytes(
        spec_path,
        "current94 diagnostic spec",
        maximum=MAX_ADDENDUM_BYTES,
    )
    expected_payload = current94_v8_static_addendum_payload(
        experiment_spec_raw=spec_raw,
        public_commit=public_commit,
        public_files=public_files,
    )
    if _canonical_json_bytes(payload) != _canonical_json_bytes(expected_payload):
        raise LaunchRefused("current94 diagnostic declaration contract changed")
    declared = {
        relative: _parse_identity(public_files[relative], f"current94 public {relative}")
        for relative in public_files
    }
    for relative, expected in declared.items():
        _require_identity(
            _file_identity(repository_root / relative, f"current94 public {relative}"),
            expected,
            f"current94 public {relative}",
        )
    public_git = _validate_public_git_binding(repository_root, public_commit, declared)
    return {
        "schema": "microtensor.code.gguf-diagnostic-static-declaration-inspection.v1",
        "status": "verified_offline_static_only",
        "live_launch_permitted": False,
        "blocker": CURRENT94_LIVE_BLOCKER,
        "declaration": FileIdentity(len(raw), _digest_bytes(raw)).as_dict(),
        "experiment_spec": FileIdentity(len(spec_raw), _digest_bytes(spec_raw)).as_dict(),
        "local_declared_file_content_verified": True,
        "public_git": public_git,
    }


def _load_contract(path: Path) -> LaunchContract:
    if _is_current94_live_request_lexically(path):
        _refuse_current94_live()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(f"diagnostic addendum cannot be resolved: {exc}") from exc
    if resolved == CURRENT94_DECLARATION_LEXICAL_PATH:
        _refuse_current94_live()
    repository_root = _repository_root()
    if resolved == repository_root / CURRENT94_ADDENDUM_RELATIVE:
        _refuse_current94_live()
    if resolved == repository_root / ADDENDUM_RELATIVE:
        return _load_v6_contract(path)
    if resolved == repository_root / NORMALIZED_ADDENDUM_RELATIVE:
        return _load_normalized_v7_contract(path)
    raise LaunchRefused("diagnostic addendum is outside the three declared immutable protocols")


def _load_v6_contract(path: Path) -> LaunchContract:
    repository_root = _repository_root()
    expected_path = repository_root / ADDENDUM_RELATIVE
    try:
        if path.resolve(strict=True) != expected_path:
            raise LaunchRefused(f"addendum must be the repository file {expected_path}")
    except OSError as exc:
        raise LaunchRefused(f"diagnostic addendum cannot be resolved: {exc}") from exc
    raw = _stable_regular_bytes(path, "diagnostic execution addendum", maximum=MAX_ADDENDUM_BYTES)
    payload = _mapping(_strict_json(raw, "diagnostic execution addendum"), "addendum")
    if raw != _pretty_json_bytes(payload):
        raise LaunchRefused("diagnostic execution addendum is not canonical sorted JSON")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema",
                "status",
                "public_source",
                "experiment_spec",
                "interpreter",
                "execution",
                "preflight",
                "invocations",
            }
        ),
        "addendum",
    )
    if payload.get("schema") != ADDENDUM_SCHEMA or payload.get("status") != ADDENDUM_STATUS:
        raise LaunchRefused("diagnostic execution addendum schema or status changed")

    public_source = _mapping(payload.get("public_source"), "public source")
    _exact_keys(
        public_source,
        frozenset({"repository", "commit", "raw_readback_verified", "files"}),
        "public source",
    )
    commit = public_source.get("commit")
    if (
        public_source.get("repository") != REPOSITORY
        or not isinstance(commit, str)
        or _COMMIT.fullmatch(commit) is None
        or public_source.get("raw_readback_verified") is not True
    ):
        raise LaunchRefused("public source repository, commit, or raw readback changed")
    source_files = _mapping(public_source.get("files"), "public source files")
    expected_source_names = frozenset({LAUNCHER_RELATIVE, VALIDATOR_RELATIVE})
    _exact_keys(source_files, expected_source_names, "public source files")
    declared_source_identities: dict[str, FileIdentity] = {}
    for relative in sorted(expected_source_names):
        expected = _parse_identity(source_files.get(relative), f"public source {relative}")
        declared_source_identities[relative] = expected
        actual = _file_identity(repository_root / relative, f"public source {relative}")
        _require_identity(actual, expected, f"public source {relative}")
    public_git = _validate_public_git_binding(
        repository_root,
        commit,
        declared_source_identities,
    )

    experiment = _mapping(payload.get("experiment_spec"), "experiment spec identity")
    _exact_keys(
        experiment,
        frozenset({"path", "bytes", "sha256"}),
        "experiment spec identity",
    )
    if experiment.get("path") != SPEC_RELATIVE:
        raise LaunchRefused("experiment spec path changed")
    expected_spec = FileIdentity(
        bytes=_integer(experiment.get("bytes"), "experiment spec bytes"),
        sha256=_digest(experiment.get("sha256"), "experiment spec digest"),
    )
    pinned_spec = FileIdentity(SPEC_BYTES, SPEC_DIGEST)
    if expected_spec != pinned_spec:
        raise LaunchRefused("experiment spec declaration changed")
    _require_identity(
        _file_identity(repository_root / SPEC_RELATIVE, "experiment spec"),
        pinned_spec,
        "experiment spec",
    )

    interpreter = _mapping(payload.get("interpreter"), "signed interpreter")
    _exact_keys(
        interpreter,
        frozenset({"path", "resolved_path", "bytes", "sha256"}),
        "signed interpreter",
    )
    if (
        interpreter.get("path") != str(INTERPRETER_PATH)
        or interpreter.get("resolved_path") != str(INTERPRETER_RESOLVED)
        or _integer(interpreter.get("bytes"), "interpreter bytes") != INTERPRETER_BYTES
        or _digest(interpreter.get("sha256"), "interpreter digest") != INTERPRETER_DIGEST
    ):
        raise LaunchRefused("signed interpreter declaration changed")

    execution = _mapping(payload.get("execution"), "execution contract")
    _exact_keys(
        execution,
        frozenset(
            {
                "cwd",
                "shell",
                "stdin",
                "umask",
                "close_fds",
                "new_session",
                "parent_death_signal",
                "timeout_seconds",
                "environment_exact",
                "environment_canonical_json_bytes",
                "environment_canonical_json_sha256",
                "parent_environment_exact",
                "report_root",
                "post_exec_inspection",
                "containment",
                "repeat_policy",
            }
        ),
        "execution contract",
    )
    exact_scalars = {
        "cwd": str(SOURCE_ROOT),
        "shell": False,
        "stdin": "/dev/null",
        "umask": UMASK_TEXT,
        "close_fds": True,
        "new_session": True,
        "parent_death_signal": "SIGKILL",
        "timeout_seconds": TIMEOUT_SECONDS,
        "environment_canonical_json_bytes": ENVIRONMENT_CANONICAL_BYTES,
        "environment_canonical_json_sha256": ENVIRONMENT_CANONICAL_DIGEST,
        "parent_environment_exact": True,
        "report_root": str(REPORT_ROOT),
    }
    for field, expected in exact_scalars.items():
        if execution.get(field) != expected or type(execution.get(field)) is not type(expected):
            raise LaunchRefused(f"execution contract {field} changed")
    environment = _mapping(execution.get("environment_exact"), "exact child environment")
    if dict(environment) != EXACT_ENVIRONMENT or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        raise LaunchRefused("exact child environment changed or gained inherited fields")
    environment_raw = _canonical_json_bytes(environment)
    if (
        len(environment) != 16
        or len(environment_raw) != ENVIRONMENT_CANONICAL_BYTES
        or _digest_bytes(environment_raw) != ENVIRONMENT_CANONICAL_DIGEST
    ):
        raise LaunchRefused("exact 16-key child environment binding changed")
    if _canonical_json_bytes(execution.get("post_exec_inspection")) != _canonical_json_bytes(
        POST_EXEC_INSPECTION
    ):
        raise LaunchRefused("post-exec inspection contract changed")
    if _canonical_json_bytes(execution.get("containment")) != _canonical_json_bytes(
        CONTAINMENT_CONTRACT
    ):
        raise LaunchRefused("process containment contract changed")
    if _canonical_json_bytes(execution.get("repeat_policy")) != _canonical_json_bytes(
        REPEAT_POLICY
    ):
        raise LaunchRefused("repeat policy changed")

    preflight = _mapping(payload.get("preflight"), "preflight contract")
    _exact_keys(
        preflight,
        frozenset(
            {
                "validator_path",
                "validator_mode",
                "model_engine_construction_permitted",
                "required_checks",
                "source_commit",
                "artifact_tree_sha256",
                "artifact_entrypoint_bytes",
                "artifact_entrypoint_sha256",
            }
        ),
        "preflight contract",
    )
    if (
        preflight.get("validator_path") != VALIDATOR_RELATIVE
        or preflight.get("validator_mode") != "in_process_static_only"
        or preflight.get("model_engine_construction_permitted") is not False
        or preflight.get("required_checks") != PREFLIGHT_CHECKS
        or preflight.get("source_commit") != SOURCE_COMMIT
        or preflight.get("artifact_tree_sha256") != ARTIFACT_TREE_DIGEST
        or preflight.get("artifact_entrypoint_bytes") != ARTIFACT_ENTRYPOINT_BYTES
        or preflight.get("artifact_entrypoint_sha256") != ARTIFACT_ENTRYPOINT_DIGEST
    ):
        raise LaunchRefused("static preflight contract changed")

    invocation_values = _sequence(payload.get("invocations"), "diagnostic invocations")
    expected_records = expected_invocations()
    if len(invocation_values) != len(expected_records):
        raise LaunchRefused("diagnostic invocation count changed")
    invocations: list[Invocation] = []
    invocation_keys = frozenset(
        {
            "repeat",
            "argv",
            "argv_canonical_json_bytes",
            "argv_canonical_json_sha256",
            "output_root",
        }
    )
    for index, (value, expected) in enumerate(
        zip(invocation_values, expected_records, strict=True)
    ):
        declared = _mapping(value, f"diagnostic invocation {index + 1}")
        _exact_keys(declared, invocation_keys, f"diagnostic invocation {index + 1}")
        if _canonical_json_bytes(declared) != _canonical_json_bytes(expected):
            raise LaunchRefused(f"diagnostic invocation {index + 1} changed")
        argv_values = _sequence(declared.get("argv"), f"invocation {index + 1} argv")
        if any(not isinstance(item, str) or not item for item in argv_values):
            raise LaunchRefused(f"diagnostic invocation {index + 1} argv is invalid")
        invocations.append(
            Invocation(
                repeat=str(declared["repeat"]),
                argv=tuple(argv_values),
                output_root=Path(str(declared["output_root"])),
                argv_canonical_json_bytes=int(declared["argv_canonical_json_bytes"]),
                argv_canonical_json_sha256=str(declared["argv_canonical_json_sha256"]),
            )
        )

    return LaunchContract(
        path=expected_path,
        raw=raw,
        digest=_digest_bytes(raw),
        public_commit=commit,
        public_git=public_git,
        repository_root=repository_root,
        experiment_spec=repository_root / SPEC_RELATIVE,
        validator_path=repository_root / VALIDATOR_RELATIVE,
        interpreter_path=INTERPRETER_PATH,
        interpreter_resolved=INTERPRETER_RESOLVED,
        environment=dict(environment),
        report_root=REPORT_ROOT,
        invocations=(invocations[0], invocations[1], invocations[2]),
    )


def _signed_interpreter_identity(contract: LaunchContract) -> dict[str, Any]:
    try:
        path_stat_before = contract.interpreter_path.lstat()
        resolved = contract.interpreter_path.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(f"signed interpreter cannot be resolved: {exc}") from exc
    if not (stat.S_ISREG(path_stat_before.st_mode) or stat.S_ISLNK(path_stat_before.st_mode)):
        raise LaunchRefused("signed interpreter path is neither a regular file nor a symlink")
    if resolved != contract.interpreter_resolved:
        raise LaunchRefused(
            f"signed interpreter resolved elsewhere: expected {contract.interpreter_resolved}, "
            f"got {resolved}"
        )
    identity = _file_identity(resolved, "signed interpreter", maximum=16 * 1024 * 1024)
    _require_identity(identity, contract.interpreter_identity, "signed interpreter")
    try:
        path_stat_after = contract.interpreter_path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"signed interpreter path changed during inspection: {exc}") from exc
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(path_stat_before, key) != getattr(path_stat_after, key) for key in stable):
        raise LaunchRefused("signed interpreter path changed during inspection")
    if resolved.stat().st_mode & 0o111 == 0:
        raise LaunchRefused("signed interpreter is not executable")
    return {
        "path": str(contract.interpreter_path),
        "resolved_path": str(resolved),
        **identity.as_dict(),
    }


_GIT_ENVIRONMENT: Final[dict[str, str]] = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _git_output(
    root: Path,
    arguments: Sequence[str],
    label: str,
    *,
    maximum: int = MAX_RECEIPT_BYTES,
) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=_GIT_ENVIRONMENT,
            shell=False,
            close_fds=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchRefused(f"{label} Git inspection failed: {exc}") from exc
    if completed.returncode != 0 or completed.stderr:
        raise LaunchRefused(f"{label} Git inspection refused")
    if len(completed.stdout) > maximum:
        raise LaunchRefused(f"{label} Git output exceeds its byte ceiling")
    return completed.stdout


def _normalized_repository_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    return normalized[:-4] if normalized.endswith(".git") else normalized


def _validate_public_git_binding(
    repository_root: Path,
    commit: str,
    declared_files: Mapping[str, FileIdentity],
) -> dict[str, Any]:
    if _COMMIT.fullmatch(commit) is None:
        raise LaunchRefused("public source commit is not a full lowercase Git commit")
    try:
        origin = (
            _git_output(
                repository_root,
                ("remote", "get-url", "origin"),
                "public source origin",
                maximum=4096,
            )
            .decode("utf-8", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise LaunchRefused("public source origin is not UTF-8") from exc
    if _normalized_repository_url(origin) != REPOSITORY:
        raise LaunchRefused(f"public source origin changed: {origin!r}")
    object_type = _git_output(
        repository_root,
        ("cat-file", "-t", commit),
        "public source commit",
        maximum=64,
    )
    if object_type != b"commit\n":
        raise LaunchRefused("declared public source object is not a local Git commit")

    current_head = (
        _git_output(
            repository_root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            "current public repository HEAD",
            maximum=64,
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if _COMMIT.fullmatch(current_head) is None:
        raise LaunchRefused("current public repository HEAD is not a full commit")
    _git_output(
        repository_root,
        ("merge-base", "--is-ancestor", commit, current_head),
        "public source HEAD ancestry",
        maximum=1,
    )
    advertised_head = (
        _git_output(
            repository_root,
            ("rev-parse", "--verify", f"{_ADVERTISED_REMOTE_REF}^{{commit}}"),
            "advertised public remote branch",
            maximum=64,
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if _COMMIT.fullmatch(advertised_head) is None:
        raise LaunchRefused("advertised public remote branch is not a full commit")
    _git_output(
        repository_root,
        ("merge-base", "--is-ancestor", commit, advertised_head),
        "advertised public source ancestry",
        maximum=1,
    )

    blobs: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(declared_files.items()):
        size_raw = _git_output(
            repository_root,
            ("cat-file", "-s", f"{commit}:{relative}"),
            f"public source blob size {relative}",
            maximum=64,
        )
        try:
            blob_size = int(size_raw.decode("ascii", errors="strict").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise LaunchRefused(f"public source blob size is invalid for {relative}") from exc
        if blob_size != expected.bytes:
            raise LaunchRefused(
                f"public source commit blob {relative} byte count changed: "
                f"expected {expected.bytes}, got {blob_size}"
            )
        blob = _git_output(
            repository_root,
            ("show", f"{commit}:{relative}"),
            f"public source blob {relative}",
            maximum=max(expected.bytes, 1),
        )
        actual = FileIdentity(len(blob), _digest_bytes(blob))
        _require_identity(actual, expected, f"public source commit blob {relative}")
        blobs[relative] = actual.as_dict()
    return {
        "origin": origin,
        "normalized_origin": _normalized_repository_url(origin),
        "commit": commit,
        "commit_object_verified_locally": True,
        "current_head": current_head,
        "commit_ancestor_of_current_head_verified_locally": True,
        "advertised_remote_ref": _ADVERTISED_REMOTE_REF,
        "advertised_remote_ref_head": advertised_head,
        "commit_ancestor_of_advertised_remote_ref_verified_locally": True,
        "commit_blobs_match_worktree_bytes": True,
        "blobs": blobs,
        "raw_github_readback_performed_by_launcher": False,
    }


def _git_source_identity(contract: LaunchContract) -> dict[str, Any]:
    try:
        resolved = contract.source_root.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(f"pinned source root cannot be resolved: {exc}") from exc
    if resolved != contract.source_root:
        raise LaunchRefused("pinned source root resolved elsewhere")
    head = (
        _git_output(
            contract.source_root,
            ("rev-parse", "--verify", "HEAD"),
            "pinned source HEAD",
            maximum=64,
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if head != contract.source_commit:
        raise LaunchRefused(
            f"pinned source commit changed: expected {contract.source_commit}, got {head}"
        )
    status = _git_output(
        contract.source_root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        "pinned source status",
    )
    if status != b"":
        raise LaunchRefused("pinned source worktree is not clean")
    return {"root": str(resolved), "commit": head, "status_empty": True}


def _load_validator(contract: LaunchContract) -> ModuleType:
    identity = _file_identity(contract.validator_path, "static validator source")
    module_name = "_microtensor_pinned_code_gguf_diagnostic_validator_" + identity.sha256[7:23]
    specification = importlib.util.spec_from_file_location(module_name, contract.validator_path)
    if specification is None or specification.loader is None:
        raise LaunchRefused("static validator import specification is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise LaunchRefused(f"static validator could not be loaded: {exc}") from exc
    schema_names = {
        "v6": "VALIDATION_SCHEMA",
        "normalized-v7": "NORMALIZED_VALIDATION_SCHEMA",
        "current94-v8": "CURRENT94_VALIDATION_SCHEMA",
    }
    schema_name = schema_names.get(contract.protocol)
    if schema_name is None:
        raise LaunchRefused(f"static validator protocol is unsupported: {contract.protocol}")
    if getattr(module, schema_name, None) != contract.validation_schema:
        raise LaunchRefused("static validator schema changed")
    return module


_FORBIDDEN_PROCESS_MODULES: Final[frozenset[str]] = frozenset(
    {"multiprocessing", "pty", "subprocess"}
)
_FORBIDDEN_OS_PRIMITIVES: Final[frozenset[str]] = frozenset(
    {
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "forkpty",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "setpgid",
        "setsid",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
        "unshare",
    }
)


def _runtime_source_records(context: Any) -> dict[str, Mapping[str, Any]]:
    identity = _mapping(context.runtime.identity, "signed runtime identity")
    microtensor = _mapping(identity.get("microtensor"), "signed Microtensor identity")
    signed = _mapping(
        microtensor.get("signed_source_files"),
        "signed Microtensor source files",
    )
    tools = _mapping(identity.get("tool_sources"), "signed tool source files")
    llama = _mapping(identity.get("llama_cpp"), "signed llama.cpp identity")
    llama_module = _mapping(llama.get("module"), "signed llama.cpp module source")
    records = {f"runtime:{name}": _mapping(value, name) for name, value in signed.items()}
    records.update({f"tool:{name}": _mapping(value, name) for name, value in tools.items()})
    records["runtime:llama_cpp"] = llama_module
    return records


def _static_process_escape_scan(context: Any) -> dict[str, Any]:
    scanned: list[dict[str, Any]] = []
    for label, declared in sorted(_runtime_source_records(context).items()):
        path_value = declared.get("path")
        bytes_value = declared.get("bytes")
        digest_value = declared.get("digest")
        if (
            not isinstance(path_value, str)
            or type(bytes_value) is not int
            or not isinstance(digest_value, str)
            or _DIGEST.fullmatch(digest_value) is None
        ):
            raise LaunchRefused(f"{label} source identity is incomplete")
        path = Path(path_value)
        raw = _stable_regular_bytes(path, f"{label} process scan", maximum=2 * 1024 * 1024)
        if len(raw) != bytes_value or _digest_bytes(raw) != digest_value:
            raise LaunchRefused(f"{label} changed before process-escape scanning")
        try:
            source = raw.decode("utf-8", errors="strict")
            tree = ast.parse(source, filename=str(path))
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise LaunchRefused(f"{label} cannot be statically scanned: {exc}") from exc

        os_aliases: set[str] = set()
        direct_os_primitives: set[str] = set()
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".", 1)[0]
                    if base in _FORBIDDEN_PROCESS_MODULES:
                        violations.append(f"line {node.lineno}: import {alias.name}")
                    if alias.name == "os":
                        os_aliases.add(alias.asname or "os")
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or "").split(".", 1)[0]
                if base in _FORBIDDEN_PROCESS_MODULES:
                    violations.append(f"line {node.lineno}: from {node.module} import")
                if node.module == "concurrent.futures" and any(
                    alias.name == "ProcessPoolExecutor" for alias in node.names
                ):
                    violations.append(
                        f"line {node.lineno}: from concurrent.futures import ProcessPoolExecutor"
                    )
                if node.module == "os":
                    for alias in node.names:
                        if alias.name in _FORBIDDEN_OS_PRIMITIVES:
                            violations.append(f"line {node.lineno}: from os import {alias.name}")
                            direct_os_primitives.add(alias.asname or alias.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in direct_os_primitives:
                    violations.append(f"line {node.lineno}: direct os.{node.func.id} call")
                elif (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in os_aliases
                    and node.func.attr in _FORBIDDEN_OS_PRIMITIVES
                ):
                    violations.append(
                        f"line {node.lineno}: {node.func.value.id}.{node.func.attr} call"
                    )
        if violations:
            raise LaunchRefused(f"{label} contains process/session primitives: {violations}")
        scanned.append(
            {
                "label": label,
                "path": str(path),
                "bytes": len(raw),
                "sha256": _digest_bytes(raw),
            }
        )
    if not scanned:
        raise LaunchRefused("static process-escape scan received no source files")
    return {
        "schema": "microtensor.code.python-process-escape-scan.v1",
        "status": "passed",
        "files": scanned,
        "forbidden_modules": sorted(_FORBIDDEN_PROCESS_MODULES),
        "forbidden_os_primitives": sorted(_FORBIDDEN_OS_PRIMITIVES),
        "python_process_or_session_primitives_found": False,
        "native_extensions_or_transitive_imports_attested": False,
        "scope": CONTAINMENT_SCOPE,
    }


def _preflight_checks(contract: LaunchContract) -> list[str]:
    if contract.protocol == "v6":
        return list(PREFLIGHT_CHECKS)
    if contract.protocol == "normalized-v7":
        return list(NORMALIZED_PREFLIGHT_CHECKS)
    if contract.protocol == "current94-v8":
        return list(CURRENT94_PREFLIGHT_CHECKS)
    raise LaunchRefused(f"static preflight protocol is unsupported: {contract.protocol}")


def _static_preflight(contract: LaunchContract) -> Preflight:
    source = _git_source_identity(contract)
    interpreter = _signed_interpreter_identity(contract)
    validator = _load_validator(contract)
    try:
        if contract.protocol == "v6":
            spec = validator._load_spec(contract.experiment_spec)
            tools = validator._load_pinned_tools(spec.source_root)
            conversion = validator._validate_conversion_bundles(spec, tools)
            context = validator._prepare_context(spec, tools, conversion)
        elif contract.protocol == "normalized-v7":
            spec = validator._load_normalized_v7_spec(contract.experiment_spec)
            tools = validator._load_normalized_v7_tools(spec)
            conversion = validator._validate_normalized_conversion_bundle(spec, tools)
            context = validator._prepare_normalized_context(spec, tools, conversion)
        elif contract.protocol == "current94-v8":
            spec = validator._load_current94_v8_spec(contract.experiment_spec)
            tools = validator._load_current94_v8_tools(spec)
            context, conversion = validator._prepare_current94_context(spec, tools)
        else:
            raise LaunchRefused(f"static preflight protocol is unsupported: {contract.protocol}")
        process_escape_scan = _static_process_escape_scan(context)
        canonical = validator._canonical_json_bytes
        digest = validator._digest_bytes
    except Exception as exc:
        raise LaunchRefused(f"pinned static preflight refused: {exc}") from exc
    entrypoint = _mapping(conversion.artifact.get("entrypoint"), "preflight entrypoint")
    actual_artifact = {
        "tree_sha256": conversion.artifact.get("tree_digest"),
        "entrypoint_bytes": entrypoint.get("bytes"),
        "entrypoint_sha256": entrypoint.get("digest"),
    }
    expected_artifact = {
        "tree_sha256": contract.artifact_tree_digest,
        "entrypoint_bytes": contract.artifact_entrypoint_bytes,
        "entrypoint_sha256": contract.artifact_entrypoint_digest,
    }
    if _canonical_json_bytes(actual_artifact) != _canonical_json_bytes(expected_artifact):
        raise LaunchRefused("static preflight artifact identity changed")
    report = {
        "source": source,
        "interpreter": interpreter,
        "experiment_spec": contract.experiment_spec_identity.as_dict(),
        "artifact": actual_artifact,
        "configuration_sha256": context.configuration_digest,
        "evaluation_dataset_sha256": digest(canonical(context.evaluation_dataset)),
        "training_lineage_sha256": digest(canonical(context.training_lineage)),
        "runtime_sha256": digest(canonical(context.runtime.identity)),
        "conversion_replays_sha256": digest(canonical(list(conversion.replay_receipts))),
        "process_escape_scan": process_escape_scan,
        "checks": _preflight_checks(contract),
        "model_engine_constructed": False,
    }
    return Preflight(report=report, validator=validator)


def _validate_through(
    validator: ModuleType,
    contract: LaunchContract,
    repeat: str,
) -> tuple[dict[str, Any], str]:
    try:
        if contract.protocol == "v6":
            validation = validator.validate_diagnostic
        elif contract.protocol == "normalized-v7":
            validation = validator.validate_normalized_v7_diagnostic
        elif contract.protocol == "current94-v8":
            validation = validator.validate_current94_v8_diagnostic
        else:
            raise LaunchRefused(f"validation protocol is unsupported: {contract.protocol}")
        report = validation(contract.experiment_spec, repeat)
        raw = validator._canonical_json_bytes(report)
    except Exception as exc:
        raise LaunchRefused(
            f"static diagnostic validation through {repeat} refused: {exc}"
        ) from exc
    completed = REPEATS.index(repeat) + 1
    all_repeats_complete = completed == len(REPEATS)
    expected_status = "validated" if all_repeats_complete else "partially_validated"
    expected_remaining = list(REPEATS[completed:])
    aggregate = _mapping(report.get("aggregate"), "validation aggregate")
    claim = _mapping(report.get("claim"), "validation claim")
    if contract.protocol == "normalized-v7":
        expected_policy = dict(contract.artifact_use_policy)
        expected_claim_keys = frozenset(
            {
                "local_structural_diagnostics_only",
                "completed_v6_training_lineage_bound",
                "normalized_conversion_schema_bound",
                "artifact_use_policy",
                "quality_or_rank_claimed",
                "promotion_authorized",
                "remaining_local_repeats",
                "remaining_external_gates",
            }
        )
        _exact_keys(claim, expected_claim_keys, "normalized validation claim")
        policy = _mapping(claim.get("artifact_use_policy"), "validation artifact use policy")
        expected_external = [
            (
                "a fresh strengthened conversion with exact runtime closure and a fresh "
                "diagnostic namespace is required for any publication candidate"
            ),
            "official validator measurement and settled rank remain external",
        ]
        if (
            _canonical_json_bytes(policy) != _canonical_json_bytes(expected_policy)
            or claim.get("local_structural_diagnostics_only") is not True
            or claim.get("completed_v6_training_lineage_bound") is not True
            or claim.get("normalized_conversion_schema_bound") is not True
            or claim.get("quality_or_rank_claimed") is not False
            or claim.get("promotion_authorized") is not False
            or claim.get("remaining_external_gates") != expected_external
        ):
            raise LaunchRefused("normalized validation artifact-use claim changed")
    if contract.protocol == "current94-v8":
        expected_policy = dict(contract.artifact_use_policy)
        expected_claim_keys = frozenset(
            {
                "local_structural_and_timing_diagnostics_only",
                "generated_or_corpus_code_executed_by_this_static_validator",
                "completed_v4_final_all_public_training_lineage_bound",
                "diagnostic_rows_are_training_overlap",
                "qwen25_qwen2_contract_bound",
                "conversion_v6_calibration_v3_bound",
                "conversion_runtime_receipt_content_bound",
                "converter_interpreter_portable_receipt_content_bound",
                "executed_interpreter_attested",
                "hermetic_conversion_attested",
                "conversion_runtime_execution_verified",
                "signed_release_v032_mechanism_v030_bound",
                "artifact_use_policy",
                "execution_pass_at_1_claimed",
                "quality_or_rank_claimed",
                "promotion_authorized",
                "remaining_local_repeats",
                "remaining_external_gates",
            }
        )
        _exact_keys(claim, expected_claim_keys, "current94 validation claim")
        policy = _mapping(claim.get("artifact_use_policy"), "current94 artifact use policy")
        if (
            _canonical_json_bytes(policy) != _canonical_json_bytes(expected_policy)
            or claim.get("local_structural_and_timing_diagnostics_only") is not True
            or claim.get("generated_or_corpus_code_executed_by_this_static_validator") is not False
            or claim.get("completed_v4_final_all_public_training_lineage_bound") is not True
            or claim.get("diagnostic_rows_are_training_overlap") is not True
            or claim.get("qwen25_qwen2_contract_bound") is not True
            or claim.get("conversion_v6_calibration_v3_bound") is not True
            or claim.get("conversion_runtime_receipt_content_bound") is not True
            or claim.get("converter_interpreter_portable_receipt_content_bound") is not True
            or claim.get("executed_interpreter_attested") is not False
            or claim.get("hermetic_conversion_attested") is not False
            or claim.get("conversion_runtime_execution_verified") is not False
            or claim.get("signed_release_v032_mechanism_v030_bound") is not True
            or claim.get("execution_pass_at_1_claimed") is not False
            or claim.get("quality_or_rank_claimed") is not False
            or claim.get("promotion_authorized") is not False
        ):
            raise LaunchRefused("current94 validation claim changed")
    if (
        report.get("schema") != contract.validation_schema
        or report.get("status") != expected_status
        or report.get("through") != repeat
        or aggregate.get("validated_repeat_hard_gates_passed") is not True
        or aggregate.get("all_declared_local_gates_passed") is not all_repeats_complete
        or claim.get("remaining_local_repeats") != expected_remaining
    ):
        raise LaunchRefused(f"static diagnostic validation through {repeat} was incomplete")
    return report, _digest_bytes(raw)


def _launch_receipt_claim(contract: LaunchContract) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "local_structural_diagnostic_only": True,
        "generated_or_corpus_code_executed_by_validator": False,
        "official_quality_or_rank_claimed": False,
        "publication_authorized_by_receipt": False,
        "submission_authorized_by_receipt": False,
        "transaction_authorized_by_receipt": False,
    }
    if contract.protocol == "normalized-v7":
        claim["artifact_use_policy"] = dict(contract.artifact_use_policy)
    if contract.protocol == "current94-v8":
        claim["artifact_use_policy"] = dict(contract.artifact_use_policy)
        claim["live_launch_supported"] = False
        claim["live_launch_blocker"] = CURRENT94_LIVE_BLOCKER
    return claim


def _require_directory(path: Path, label: str, *, mode: int | None = None) -> os.stat_result:
    try:
        current = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be inspected: {exc}") from exc
    if not stat.S_ISDIR(current.st_mode):
        raise LaunchRefused(f"{label} must be a regular non-symlink directory")
    if current.st_uid != os.geteuid():
        raise LaunchRefused(f"{label} must be owned by the effective UID")
    if mode is not None and stat.S_IMODE(current.st_mode) != mode:
        raise LaunchRefused(f"{label} mode must be exactly {mode:04o}")
    return current


def _ensure_report_root(path: Path = REPORT_ROOT, base: Path = REPORT_BASE) -> None:
    if not path.is_absolute() or path.parent.parent != base:
        raise LaunchRefused("report root is outside its predeclared tmpfs namespace")
    _require_directory(base, "diagnostic tmpfs base")
    parent = path.parent
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise LaunchRefused(f"diagnostic receipt parent could not be created: {exc}") from exc
    _require_directory(parent, "diagnostic receipt parent", mode=0o700)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise LaunchRefused(f"diagnostic report root could not be created: {exc}") from exc
    _require_directory(path, "diagnostic report root", mode=0o700)


def _directory_names(path: Path, label: str) -> frozenset[str]:
    try:
        root = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be inspected: {exc}") from exc
    if not stat.S_ISDIR(root.st_mode):
        raise LaunchRefused(f"{label} must be a regular non-symlink directory")
    try:
        with os.scandir(path) as entries:
            return frozenset(entry.name for entry in entries)
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be enumerated: {exc}") from exc


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _marker_path(contract: LaunchContract, repeat: str) -> Path:
    return contract.report_root / f"{repeat}-attempt.json"


def _receipt_path(contract: LaunchContract, repeat: str) -> Path:
    return contract.report_root / f"{repeat}-launch-receipt.json"


def _validate_namespace_state(
    contract: LaunchContract, repeat: str
) -> dict[str, PriorReceiptBinding]:
    index = REPEATS.index(repeat)
    for position, invocation in enumerate(contract.invocations):
        root_exists = _path_exists(invocation.output_root)
        if position < index and not root_exists:
            raise LaunchRefused(f"prior diagnostic root is absent: {invocation.output_root}")
        if position >= index and root_exists:
            raise LaunchRefused(
                f"current or future diagnostic root already exists: {invocation.output_root}"
            )
        parent = invocation.output_root.parent
        if parent.exists():
            names = _directory_names(parent, f"diagnostic output parent {parent}")
            prefix = f".{invocation.output_root.name}."
            conflicts = sorted(name for name in names if name.startswith(prefix))
            if conflicts:
                raise LaunchRefused(
                    "diagnostic output staging namespace exists for "
                    f"{invocation.repeat}: {conflicts}"
                )

    report_names = _directory_names(contract.report_root, "diagnostic report root")
    expected_report_names = frozenset(
        name
        for prior in REPEATS[:index]
        for name in (
            _marker_path(contract, prior).name,
            _receipt_path(contract, prior).name,
        )
    )
    if report_names != expected_report_names:
        raise LaunchRefused(
            "diagnostic report inventory changed: expected "
            f"{sorted(expected_report_names)}, got {sorted(report_names)}"
        )
    prior_bindings: dict[str, PriorReceiptBinding] = {}
    for position, prior in enumerate(REPEATS):
        marker = _marker_path(contract, prior)
        receipt = _receipt_path(contract, prior)
        if position < index:
            prior_bindings[prior] = _validate_previous_record(
                contract,
                prior,
                marker,
                receipt,
                expected_previous_validation_digest=(
                    None
                    if position == 0
                    else prior_bindings[REPEATS[position - 1]].validation_sha256
                ),
            )
        else:
            if _path_exists(marker) or _path_exists(receipt):
                raise LaunchRefused(f"{prior} was already consumed or recorded")
    return prior_bindings


def _read_canonical_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _stable_regular_bytes(path, label, maximum=MAX_RECEIPT_BYTES)
    payload = _mapping(_strict_json(raw, label), label)
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise LaunchRefused(f"{label} is not canonical compact JSON")
    return payload, raw


def _public_source_payload(contract: LaunchContract) -> dict[str, Any]:
    return {
        "repository": REPOSITORY,
        "commit": contract.public_commit,
        "addendum_raw_github_readback_declared_verified": True,
        "launcher_local_git_verification": contract.public_git,
        "launcher_raw_github_readback_performed": False,
    }


def _validate_boundary_record(value: Any, label: str) -> None:
    boundary = _mapping(value, label)
    _exact_keys(
        boundary,
        frozenset(
            {
                "schema",
                "status",
                "parent_environment",
                "task_ids",
                "task_count",
                "sigchld_disposition",
                "sigalrm_disposition",
                "realtime_timer_clear",
                "blocked_signal_mask_empty",
                "child_subreaper_state",
                "proc_children_empty",
                "waitpid_echild_verified",
            }
        ),
        label,
    )
    parent = _mapping(boundary.get("parent_environment"), f"{label} parent environment")
    _exact_keys(
        parent,
        frozenset({"exact", "keys", "canonical_json_bytes", "canonical_json_sha256"}),
        f"{label} parent environment",
    )
    tasks = _sequence(boundary.get("task_ids"), f"{label} task IDs")
    if (
        boundary.get("schema") != "microtensor.code.launcher-child-boundary.v1"
        or boundary.get("status") != "ready"
        or parent.get("exact") is not True
        or parent.get("keys") != sorted(EXACT_ENVIRONMENT)
        or parent.get("canonical_json_bytes") != ENVIRONMENT_CANONICAL_BYTES
        or parent.get("canonical_json_sha256") != ENVIRONMENT_CANONICAL_DIGEST
        or len(tasks) != 1
        or type(tasks[0]) is not int
        or tasks[0] < 1
        or boundary.get("task_count") != 1
        or boundary.get("sigchld_disposition") != "SIG_DFL"
        or boundary.get("sigalrm_disposition") != "SIG_DFL"
        or boundary.get("realtime_timer_clear") is not True
        or boundary.get("blocked_signal_mask_empty") is not True
        or boundary.get("child_subreaper_state") != 0
        or boundary.get("proc_children_empty") is not True
        or boundary.get("waitpid_echild_verified") is not True
    ):
        raise LaunchRefused(f"{label} is not an exact ready boundary")


def _validate_prior_process(
    contract: LaunchContract,
    repeat: str,
    receipt: Mapping[str, Any],
) -> None:
    process = _mapping(receipt.get("process"), "prior process")
    _exact_keys(
        process,
        frozenset(
            {
                "started",
                "pid",
                "stage",
                "returncode",
                "timed_out",
                "started_at_utc",
                "started_at_unix_ns",
                "finished_at_utc",
                "finished_at_unix_ns",
            }
        ),
        "prior process",
    )
    pid = _integer(process.get("pid"), "prior process PID", minimum=1)
    started_ns = _integer(process.get("started_at_unix_ns"), "prior process start")
    finished_ns = _integer(process.get("finished_at_unix_ns"), "prior process finish")
    if (
        process.get("started") is not True
        or process.get("stage") != "terminal"
        or process.get("returncode") != 0
        or process.get("timed_out") is not False
        or not isinstance(process.get("started_at_utc"), str)
        or not isinstance(process.get("finished_at_utc"), str)
        or finished_ns < started_ns
    ):
        raise LaunchRefused(f"{repeat} prior process evidence changed")

    inspection = _mapping(receipt.get("inspection"), "prior post-exec inspection")
    inspection_keys = frozenset(POST_EXEC_INSPECTION) | frozenset(
        {
            "pid",
            "ptrace_stop_signal",
            "cmdline",
            "environ",
            "cwd",
            "exe",
            "fd0",
            "open_fds",
            "umask",
            "process_group",
        }
    )
    _exact_keys(inspection, inspection_keys, "prior post-exec inspection")
    invocation = contract.invocations[REPEATS.index(repeat)]
    for key, expected in POST_EXEC_INSPECTION.items():
        if inspection.get(key) != expected:
            raise LaunchRefused(f"prior post-exec inspection field {key} changed")
    cmdline = _mapping(inspection.get("cmdline"), "prior inspected cmdline")
    environ = _mapping(inspection.get("environ"), "prior inspected environment")
    cwd = _mapping(inspection.get("cwd"), "prior inspected cwd")
    executable = _mapping(inspection.get("exe"), "prior inspected executable")
    fd0 = _mapping(inspection.get("fd0"), "prior inspected stdin")
    _exact_keys(cmdline, frozenset({"matched", "canonical_json_sha256"}), "cmdline")
    _exact_keys(
        environ,
        frozenset({"matched", "keys", "canonical_json_bytes", "canonical_json_sha256"}),
        "environment",
    )
    _exact_keys(cwd, frozenset({"matched", "path"}), "cwd")
    _exact_keys(executable, frozenset({"matched", "path"}), "executable")
    _exact_keys(fd0, frozenset({"matched_dev_null", "proc_link"}), "stdin")
    if (
        inspection.get("pid") != pid
        or inspection.get("ptrace_stop_signal") != signal.SIGTRAP
        or cmdline.get("matched") is not True
        or cmdline.get("canonical_json_sha256") != invocation.argv_canonical_json_sha256
        or environ.get("matched") is not True
        or environ.get("keys") != sorted(EXACT_ENVIRONMENT)
        or environ.get("canonical_json_bytes") != ENVIRONMENT_CANONICAL_BYTES
        or environ.get("canonical_json_sha256") != ENVIRONMENT_CANONICAL_DIGEST
        or cwd != {"matched": True, "path": str(contract.source_root)}
        or executable != {"matched": True, "path": str(contract.interpreter_resolved)}
        or fd0.get("matched_dev_null") is not True
        or inspection.get("open_fds") != [0, 1, 2]
        or inspection.get("umask") != UMASK_TEXT
        or inspection.get("process_group") != pid
    ):
        raise LaunchRefused(f"{repeat} prior held exec-stop evidence changed")


def _validate_prior_containment(receipt: Mapping[str, Any], process_pid: int) -> None:
    containment = _mapping(receipt.get("containment"), "prior containment")
    _exact_keys(
        containment,
        frozenset({"pre_consumption_boundary", "process", "post_process_boundary"}),
        "prior containment",
    )
    _validate_boundary_record(
        containment.get("pre_consumption_boundary"),
        "prior pre-consumption boundary",
    )
    _validate_boundary_record(
        containment.get("post_process_boundary"),
        "prior post-process boundary",
    )
    process = _mapping(containment.get("process"), "prior process containment")
    evidence_keys = frozenset(
        {
            "pre_fork_boundary",
            "prior_subreaper_state",
            "subreaper_enabled",
            "direct_pid",
            "direct_sigkill_attempted",
            "direct_terminal_status_observed",
            "direct_returncode",
            "process_group_sigkill_attempted",
            "process_group_absent",
            "descendants_observed",
            "observed_descendant_pids",
            "pidfd_signaled_descendant_pids",
            "terminal_waitpid_echild_verified",
            "subreaper_restored",
        }
    )
    _exact_keys(
        process,
        frozenset(CONTAINMENT_CONTRACT) | evidence_keys,
        "prior process containment",
    )
    for key, expected in CONTAINMENT_CONTRACT.items():
        if process.get(key) != expected:
            raise LaunchRefused(f"prior containment contract field {key} changed")
    _validate_boundary_record(process.get("pre_fork_boundary"), "prior pre-fork boundary")
    if not (
        containment.get("pre_consumption_boundary")
        == process.get("pre_fork_boundary")
        == containment.get("post_process_boundary")
    ):
        raise LaunchRefused("prior launcher boundary evidence changed across the process")
    if (
        process.get("prior_subreaper_state") != 0
        or process.get("subreaper_enabled") is not True
        or process.get("direct_pid") != process_pid
        or process.get("direct_sigkill_attempted") is not False
        or process.get("direct_terminal_status_observed") is not True
        or process.get("direct_returncode") != 0
        or process.get("process_group_sigkill_attempted") is not True
        or process.get("process_group_absent") is not True
        or process.get("descendants_observed") is not False
        or process.get("observed_descendant_pids") != []
        or process.get("pidfd_signaled_descendant_pids") != []
        or process.get("terminal_waitpid_echild_verified") is not True
        or process.get("subreaper_restored") is not True
    ):
        raise LaunchRefused("prior process containment evidence was incomplete")


def _validate_prior_preflight(contract: LaunchContract, value: Any) -> None:
    preflight = _mapping(value, "prior preflight")
    _exact_keys(
        preflight,
        frozenset(
            {
                "source",
                "interpreter",
                "experiment_spec",
                "artifact",
                "configuration_sha256",
                "evaluation_dataset_sha256",
                "training_lineage_sha256",
                "runtime_sha256",
                "conversion_replays_sha256",
                "process_escape_scan",
                "checks",
                "model_engine_constructed",
            }
        ),
        "prior preflight",
    )
    source = _mapping(preflight.get("source"), "prior preflight source")
    interpreter = _mapping(preflight.get("interpreter"), "prior preflight interpreter")
    _exact_keys(source, frozenset({"root", "commit", "status_empty"}), "preflight source")
    _exact_keys(
        interpreter,
        frozenset({"path", "resolved_path", "bytes", "sha256"}),
        "preflight interpreter",
    )
    _integer(interpreter.get("bytes"), "prior preflight interpreter bytes", minimum=1)
    _digest(interpreter.get("sha256"), "prior preflight interpreter digest")
    for field in (
        "configuration_sha256",
        "evaluation_dataset_sha256",
        "training_lineage_sha256",
        "runtime_sha256",
        "conversion_replays_sha256",
    ):
        _digest(preflight.get(field), f"prior preflight {field}")
    scan = _mapping(preflight.get("process_escape_scan"), "prior process escape scan")
    _exact_keys(
        scan,
        frozenset(
            {
                "schema",
                "status",
                "files",
                "forbidden_modules",
                "forbidden_os_primitives",
                "python_process_or_session_primitives_found",
                "native_extensions_or_transitive_imports_attested",
                "scope",
            }
        ),
        "prior process escape scan",
    )
    files = _sequence(scan.get("files"), "prior process escape scan files")
    if not files:
        raise LaunchRefused("prior process escape scan did not cover any files")
    for index, item in enumerate(files):
        record = _mapping(item, f"prior process escape scan file {index}")
        _exact_keys(
            record,
            frozenset({"label", "path", "bytes", "sha256"}),
            f"prior process escape scan file {index}",
        )
        _integer(record.get("bytes"), f"prior scan file {index} bytes")
        _digest(record.get("sha256"), f"prior scan file {index} digest")
    if (
        source
        != {
            "root": str(contract.source_root),
            "commit": contract.source_commit,
            "status_empty": True,
        }
        or interpreter.get("path") != str(contract.interpreter_path)
        or interpreter.get("resolved_path") != str(contract.interpreter_resolved)
        or preflight.get("experiment_spec") != contract.experiment_spec_identity.as_dict()
        or preflight.get("artifact")
        != {
            "tree_sha256": contract.artifact_tree_digest,
            "entrypoint_bytes": contract.artifact_entrypoint_bytes,
            "entrypoint_sha256": contract.artifact_entrypoint_digest,
        }
        or preflight.get("checks") != _preflight_checks(contract)
        or preflight.get("model_engine_constructed") is not False
        or scan.get("schema") != "microtensor.code.python-process-escape-scan.v1"
        or scan.get("status") != "passed"
        or scan.get("forbidden_modules") != sorted(_FORBIDDEN_PROCESS_MODULES)
        or scan.get("forbidden_os_primitives") != sorted(_FORBIDDEN_OS_PRIMITIVES)
        or scan.get("python_process_or_session_primitives_found") is not False
        or scan.get("native_extensions_or_transitive_imports_attested") is not False
        or scan.get("scope") != CONTAINMENT_SCOPE
    ):
        raise LaunchRefused("prior static preflight evidence changed")


def _validate_prior_receipt_details(
    contract: LaunchContract,
    repeat: str,
    receipt: Mapping[str, Any],
    expected_previous_validation_digest: str | None,
) -> None:
    if receipt.get("public_source") != _public_source_payload(contract):
        raise LaunchRefused("prior public source evidence changed")
    invocation = contract.invocations[REPEATS.index(repeat)]
    expected_execution = {
        "argv": list(invocation.argv),
        "argv_canonical_json_bytes": invocation.argv_canonical_json_bytes,
        "argv_canonical_json_sha256": invocation.argv_canonical_json_sha256,
        "cwd": str(contract.source_root),
        "shell": False,
        "stdin": "/dev/null",
        "umask": UMASK_TEXT,
        "close_fds": True,
        "new_session": True,
        "parent_death_signal": "SIGKILL",
        "timeout_seconds": TIMEOUT_SECONDS,
        "environment_keys": sorted(contract.environment),
        "environment_canonical_json_bytes": ENVIRONMENT_CANONICAL_BYTES,
        "environment_canonical_json_sha256": ENVIRONMENT_CANONICAL_DIGEST,
        "parent_environment_exact": True,
        "containment_contract": CONTAINMENT_CONTRACT,
        "output_root": str(invocation.output_root),
    }
    if receipt.get("execution") != expected_execution:
        raise LaunchRefused("prior execution evidence changed")
    _validate_prior_preflight(contract, receipt.get("preflight"))
    _validate_prior_process(contract, repeat, receipt)
    process = _mapping(receipt.get("process"), "prior process")
    _validate_prior_containment(receipt, int(process["pid"]))

    post = _mapping(receipt.get("post_control"), "prior post-run control")
    _exact_keys(
        post,
        frozenset(
            {
                "source",
                "interpreter",
                "public_git",
                "addendum_unchanged",
                "public_source_unchanged",
            }
        ),
        "prior post-run control",
    )
    if (
        _mapping(post.get("source"), "prior post-run source")
        != {
            "root": str(contract.source_root),
            "commit": contract.source_commit,
            "status_empty": True,
        }
        or _mapping(post.get("interpreter"), "prior post-run interpreter")
        != _mapping(receipt.get("preflight"), "prior preflight").get("interpreter")
        or post.get("public_git") != contract.public_git
        or post.get("addendum_unchanged") is not True
        or post.get("public_source_unchanged") is not True
    ):
        raise LaunchRefused("prior post-run public control changed")

    index = REPEATS.index(repeat)
    previous = receipt.get("previous_validation")
    if index == 0:
        if previous is not None:
            raise LaunchRefused("r1 prior receipt unexpectedly has previous validation")
    else:
        previous_map = _mapping(previous, "prior receipt previous validation")
        _exact_keys(
            previous_map,
            frozenset(
                {
                    "status",
                    "through",
                    "report_sha256",
                    "validated_repeat_hard_gates_passed",
                    "all_declared_local_gates_passed",
                    "remaining_local_repeats",
                }
            ),
            "prior receipt previous validation",
        )
        previous_repeat = REPEATS[index - 1]
        _digest(previous_map.get("report_sha256"), "previous validation report digest")
        if (
            previous_map.get("status") != "partially_validated"
            or previous_map.get("through") != previous_repeat
            or previous_map.get("validated_repeat_hard_gates_passed") is not True
            or previous_map.get("all_declared_local_gates_passed") is not False
            or previous_map.get("remaining_local_repeats") != list(REPEATS[index:])
            or previous_map.get("report_sha256") != expected_previous_validation_digest
        ):
            raise LaunchRefused("prior receipt previous validation evidence changed")

    claim = _mapping(receipt.get("claim"), "prior receipt claim")
    expected_claim = _launch_receipt_claim(contract)
    _exact_keys(claim, frozenset(expected_claim), "prior receipt claim")
    if claim != expected_claim:
        raise LaunchRefused("prior receipt claim changed")


def _validate_previous_record(
    contract: LaunchContract,
    repeat: str,
    marker_path: Path,
    receipt_path: Path,
    *,
    expected_previous_validation_digest: str | None,
) -> PriorReceiptBinding:
    marker, marker_raw = _read_canonical_json(marker_path, f"{repeat} attempt marker")
    _exact_keys(
        marker,
        frozenset(
            {
                "schema",
                "status",
                "repeat",
                "addendum",
                "public_commit",
                "consumed_at_utc",
                "consumed_at_unix_ns",
            }
        ),
        f"{repeat} attempt marker",
    )
    expected_addendum = {"bytes": len(contract.raw), "sha256": contract.digest}
    if (
        marker.get("schema") != ATTEMPT_SCHEMA
        or marker.get("status") != "consumed"
        or marker.get("repeat") != repeat
        or marker.get("addendum") != expected_addendum
        or marker.get("public_commit") != contract.public_commit
    ):
        raise LaunchRefused(f"{repeat} attempt marker binding changed")
    marker_identity = FileIdentity(len(marker_raw), _digest_bytes(marker_raw))
    receipt, _ = _read_canonical_json(receipt_path, f"{repeat} launch receipt")
    _exact_keys(
        receipt,
        frozenset(
            {
                "schema",
                "status",
                "repeat",
                "outcome",
                "error",
                "addendum",
                "public_source",
                "attempt",
                "execution",
                "preflight",
                "previous_validation",
                "process",
                "inspection",
                "containment",
                "post_control",
                "validation",
                "claim",
            }
        ),
        f"{repeat} launch receipt",
    )
    completed = REPEATS.index(repeat) + 1
    all_repeats_complete = completed == len(REPEATS)
    expected_status = "validated" if all_repeats_complete else "partially_validated"
    expected_remaining = list(REPEATS[completed:])
    validation = _mapping(receipt.get("validation"), "prior validation")
    _exact_keys(
        validation,
        frozenset(
            {
                "status",
                "through",
                "report_sha256",
                "validated_repeat_hard_gates_passed",
                "all_declared_local_gates_passed",
                "remaining_local_repeats",
            }
        ),
        "prior validation",
    )
    validation_digest = _digest(
        validation.get("report_sha256"),
        "prior validation report digest",
    )
    _require_identity(
        _parse_identity(receipt.get("attempt"), "prior receipt attempt identity"),
        marker_identity,
        "prior receipt attempt marker",
    )
    _validate_prior_receipt_details(
        contract,
        repeat,
        receipt,
        expected_previous_validation_digest,
    )
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != expected_status
        or receipt.get("repeat") != repeat
        or receipt.get("outcome") != "validated_repeat_hard_gates_passed"
        or receipt.get("error") is not None
        or receipt.get("addendum") != expected_addendum
        or validation.get("status") != expected_status
        or validation.get("through") != repeat
        or validation.get("validated_repeat_hard_gates_passed") is not True
        or validation.get("all_declared_local_gates_passed") is not all_repeats_complete
        or validation.get("remaining_local_repeats") != expected_remaining
    ):
        raise LaunchRefused(f"{repeat} prior launch was not exactly validated")
    preflight_digest = _digest_bytes(
        _canonical_json_bytes(_mapping(receipt.get("preflight"), "prior preflight"))
    )
    return PriorReceiptBinding(
        preflight_sha256=preflight_digest,
        validation_sha256=validation_digest,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_noreplace(path: Path, payload: Mapping[str, Any]) -> FileIdentity:
    raw = _canonical_json_bytes(payload) + b"\n"
    parent = path.parent
    _require_directory(parent, "atomic receipt directory", mode=0o700)
    staging = parent / f".{path.name}.{os.getpid()}.{time.time_ns()}"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise LaunchRefused("short write while staging launch receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(staging, path, follow_symlinks=False)
        linked = True
        _fsync_directory(parent)
    except FileExistsError as exc:
        raise LaunchRefused(f"launch record already exists: {path}") from exc
    except OSError as exc:
        raise LaunchRefused(f"atomic launch record publication failed for {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            if linked:
                raise LaunchRefused(f"launch receipt staging cleanup failed: {exc}") from exc
    try:
        os.chmod(path, 0o600, follow_symlinks=False)
        _fsync_directory(parent)
    except OSError as exc:
        raise LaunchRefused(f"launch record durability finalization failed: {exc}") from exc
    return FileIdentity(len(raw), _digest_bytes(raw))


def _parse_nul_fields(raw: bytes, label: str) -> list[bytes]:
    if not raw.endswith(b"\0"):
        raise LaunchRefused(f"{label} is not NUL-terminated")
    fields = raw[:-1].split(b"\0")
    if not fields or any(field == b"" for field in fields):
        raise LaunchRefused(f"{label} contains an empty field")
    return fields


def _read_proc_file(path: Path, label: str, *, maximum: int = 1024 * 1024) -> bytes:
    if type(maximum) is not int or maximum < 1:
        raise LaunchRefused(f"{label} byte ceiling is invalid")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be opened safely: {exc}") from exc
    try:
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise LaunchRefused(f"{label} cannot be read: {exc}") from exc
    finally:
        os.close(descriptor)
    if len(raw) > maximum:
        raise LaunchRefused(f"{label} exceeds its byte ceiling")
    return raw


def _inspect_stopped_child(
    pid: int,
    invocation: Invocation,
    contract: LaunchContract,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    process_root = proc_root / str(pid)
    cmdline_raw = _read_proc_file(process_root / "cmdline", "child cmdline")
    try:
        cmdline = [field.decode("utf-8") for field in _parse_nul_fields(cmdline_raw, "cmdline")]
    except UnicodeDecodeError as exc:
        raise LaunchRefused(f"child cmdline is not UTF-8: {exc}") from exc
    if cmdline != list(invocation.argv):
        raise LaunchRefused("post-exec child cmdline differs from the immutable argv")

    environ_raw = _read_proc_file(process_root / "environ", "child environment")
    environment: dict[str, str] = {}
    try:
        fields = _parse_nul_fields(environ_raw, "environment")
        for field in fields:
            key_raw, separator, value_raw = field.partition(b"=")
            if not separator or not key_raw:
                raise LaunchRefused("child environment contains an invalid entry")
            key = key_raw.decode("utf-8")
            value = value_raw.decode("utf-8")
            if key in environment:
                raise LaunchRefused(f"child environment repeats {key!r}")
            environment[key] = value
    except UnicodeDecodeError as exc:
        raise LaunchRefused(f"child environment is not UTF-8: {exc}") from exc
    if environment != contract.environment:
        raise LaunchRefused("post-exec child environment differs or contains inherited fields")

    try:
        cwd = (process_root / "cwd").resolve(strict=True)
        executable = (process_root / "exe").resolve(strict=True)
    except OSError as exc:
        raise LaunchRefused(f"post-exec cwd or executable cannot be resolved: {exc}") from exc
    if cwd != contract.source_root:
        raise LaunchRefused(f"post-exec child cwd changed: {cwd}")
    if executable != contract.interpreter_resolved:
        raise LaunchRefused(f"post-exec child executable changed: {executable}")

    fd_root = process_root / "fd"
    try:
        descriptors = sorted(int(item.name) for item in fd_root.iterdir())
    except (OSError, ValueError) as exc:
        raise LaunchRefused(f"post-exec child descriptors cannot be enumerated: {exc}") from exc
    if descriptors != [0, 1, 2]:
        raise LaunchRefused(f"post-exec child has unrelated open descriptors: {descriptors}")
    try:
        fd0 = (fd_root / "0").stat()
        devnull = Path("/dev/null").stat()
        fd0_link = os.readlink(fd_root / "0")
    except OSError as exc:
        raise LaunchRefused(f"post-exec child stdin cannot be inspected: {exc}") from exc
    if not stat.S_ISCHR(fd0.st_mode) or (fd0.st_dev, fd0.st_ino, fd0.st_rdev) != (
        devnull.st_dev,
        devnull.st_ino,
        devnull.st_rdev,
    ):
        raise LaunchRefused("post-exec child stdin is not /dev/null")

    status_raw = _read_proc_file(process_root / "status", "child status")
    try:
        status_lines = status_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LaunchRefused(f"child status is not UTF-8: {exc}") from exc
    masks = [line.split(":", 1)[1].strip() for line in status_lines if line.startswith("Umask:")]
    if masks != [UMASK_TEXT]:
        raise LaunchRefused(f"post-exec child umask changed: {masks}")

    try:
        process_group = os.getpgid(pid)
    except OSError as exc:
        raise LaunchRefused(f"post-exec process group cannot be inspected: {exc}") from exc
    if process_group != pid:
        raise LaunchRefused("post-exec child is not the leader of its isolated process group")
    return {
        **POST_EXEC_INSPECTION,
        "pid": pid,
        "ptrace_stop_signal": signal.SIGTRAP,
        "cmdline": {
            "matched": True,
            "canonical_json_sha256": _digest_bytes(_canonical_json_bytes(cmdline)),
        },
        "environ": {
            "matched": True,
            "keys": sorted(environment),
            "canonical_json_bytes": len(_canonical_json_bytes(environment)),
            "canonical_json_sha256": _digest_bytes(_canonical_json_bytes(environment)),
        },
        "cwd": {"matched": True, "path": str(cwd)},
        "exe": {"matched": True, "path": str(executable)},
        "fd0": {"matched_dev_null": True, "proc_link": fd0_link},
        "open_fds": descriptors,
        "umask": masks[0],
        "process_group": process_group,
    }


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _ptrace(library: ctypes.CDLL, request: int, pid: int) -> None:
    ctypes.set_errno(0)
    result = library.ptrace(
        ctypes.c_ulong(request),
        ctypes.c_ulong(pid),
        ctypes.c_void_p(),
        ctypes.c_void_p(),
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))


def _prctl_parent_death(library: ctypes.CDLL) -> None:
    ctypes.set_errno(0)
    result = library.prctl(
        ctypes.c_int(_PR_SET_PDEATHSIG),
        ctypes.c_ulong(signal.SIGKILL),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))


def _child_error(descriptor: int, stage: str, exc: BaseException) -> None:
    number = exc.errno if isinstance(exc, OSError) and exc.errno is not None else 0
    payload = f"{stage}:{type(exc).__name__}:{number}".encode("ascii", errors="replace")
    with suppress(OSError):
        os.write(descriptor, payload[:MAX_CHILD_ERROR_BYTES])


def _maximum_fd() -> int:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        try:
            soft = int(os.sysconf("SC_OPEN_MAX"))
        except (OSError, ValueError):
            soft = 65_536
    return max(4, min(int(soft), 1 << 20))


def _child_exec(
    invocation: Invocation,
    contract: LaunchContract,
    error_read: int,
    error_write: int,
    parent_pid: int,
    library: ctypes.CDLL,
) -> None:
    stage = "child_setup"
    try:
        os.close(error_read)
        stage = "parent_death_signal"
        _prctl_parent_death(library)
        if os.getppid() != parent_pid:
            raise OSError(errno.ESRCH, "launcher parent changed before child setup")
        stage = "new_session"
        os.setsid()
        stage = "cwd"
        os.chdir(contract.source_root)
        stage = "umask"
        os.umask(UMASK_VALUE)
        stage = "stdin"
        null_descriptor = os.open(
            "/dev/null",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.dup2(null_descriptor, 0, inheritable=True)
        finally:
            if null_descriptor not in {0, error_write}:
                os.close(null_descriptor)
        stage = "close_fds"
        if error_write <= 2:
            raise OSError(errno.EBADF, "exec-error descriptor overlaps standard descriptors")
        os.closerange(3, error_write)
        os.closerange(error_write + 1, _maximum_fd())
        stage = "ptrace_traceme"
        _ptrace(library, _PTRACE_TRACEME, 0)
        stage = "execve"
        os.execve(  # noqa: S606 - exact direct exec; no shell is involved
            contract.interpreter_path,
            list(invocation.argv),
            contract.environment,
        )
    except BaseException as exc:
        _child_error(error_write, stage, exc)
        os._exit(127)


def _prctl_child_subreaper(library: ctypes.CDLL) -> int:
    value = ctypes.c_int(-1)
    ctypes.set_errno(0)
    result = library.prctl(
        ctypes.c_int(_PR_GET_CHILD_SUBREAPER),
        ctypes.byref(value),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))
    if value.value not in {0, 1}:
        raise LaunchRefused(f"kernel returned invalid child-subreaper state {value.value}")
    return value.value


def _set_child_subreaper(library: ctypes.CDLL, value: int) -> None:
    if value not in {0, 1}:
        raise LaunchRefused("child-subreaper state must be zero or one")
    ctypes.set_errno(0)
    result = library.prctl(
        ctypes.c_int(_PR_SET_CHILD_SUBREAPER),
        ctypes.c_ulong(value),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))
    if _prctl_child_subreaper(library) != value:
        raise LaunchRefused("child-subreaper state did not take effect")


def _launcher_task_ids(proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    task_root = proc_root / "self" / "task"
    try:
        with os.scandir(task_root) as entries:
            task_ids = tuple(sorted(int(entry.name) for entry in entries if entry.name.isdecimal()))
    except (OSError, ValueError) as exc:
        raise LaunchRefused(f"launcher task inventory cannot be read: {exc}") from exc
    if not task_ids:
        raise LaunchRefused("launcher task inventory is empty")
    return task_ids


def _launcher_child_pids(proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    children: set[int] = set()
    for task_id in _launcher_task_ids(proc_root):
        raw = _read_proc_file(
            proc_root / "self" / "task" / str(task_id) / "children",
            f"launcher task {task_id} children",
            maximum=64 * 1024,
        )
        try:
            fields = raw.decode("ascii", errors="strict").split()
        except UnicodeDecodeError as exc:
            raise LaunchRefused("launcher child inventory is not ASCII") from exc
        for field in fields:
            if not field.isdecimal() or int(field) < 1:
                raise LaunchRefused(f"launcher child inventory contains invalid PID {field!r}")
            children.add(int(field))
    return tuple(sorted(children))


def _require_waitpid_echild() -> None:
    while True:
        try:
            waited, status = os.waitpid(-1, os.WNOHANG | _WAIT_ALL)
        except InterruptedError:
            continue
        except ChildProcessError:
            return
        if waited == 0:
            raise LaunchRefused("launcher has a live child according to waitpid")
        raise LaunchRefused(
            f"launcher reaped undeclared child {waited} with status {status} during boundary proof"
        )


def _validate_parent_environment() -> dict[str, Any]:
    environment = dict(os.environ)
    if environment != EXACT_ENVIRONMENT:
        raise LaunchRefused(
            "launcher parent environment must be exactly the declared 16-key mapping"
        )
    raw = _canonical_json_bytes(environment)
    if (
        len(raw) != ENVIRONMENT_CANONICAL_BYTES
        or _digest_bytes(raw) != ENVIRONMENT_CANONICAL_DIGEST
    ):
        raise LaunchRefused("launcher parent environment canonical identity changed")
    return {
        "exact": True,
        "keys": sorted(environment),
        "canonical_json_bytes": len(raw),
        "canonical_json_sha256": _digest_bytes(raw),
    }


def _validate_launcher_boundary_ready(*, expected_subreaper_state: int = 0) -> dict[str, Any]:
    if expected_subreaper_state not in {0, 1}:
        raise LaunchRefused("expected child-subreaper state must be zero or one")
    if sys.platform != "linux" or not Path("/proc/self/status").is_file():
        raise LaunchRefused("required Linux child-boundary evidence is unavailable")
    parent_environment = _validate_parent_environment()
    if not callable(getattr(os, "pidfd_open", None)) or not callable(
        getattr(signal, "pidfd_send_signal", None)
    ):
        raise LaunchRefused("required Linux pidfd descendant signaling is unavailable")
    try:
        sigchld = signal.getsignal(signal.SIGCHLD)
    except (OSError, ValueError) as exc:
        raise LaunchRefused(f"SIGCHLD disposition cannot be inspected: {exc}") from exc
    if sigchld != signal.SIG_DFL:
        raise LaunchRefused("launcher SIGCHLD disposition must be SIG_DFL")
    try:
        sigalrm = signal.getsignal(signal.SIGALRM)
        realtime_timer = signal.getitimer(signal.ITIMER_REAL)
    except (AttributeError, OSError, ValueError) as exc:
        raise LaunchRefused(f"SIGALRM deadline state cannot be inspected: {exc}") from exc
    if sigalrm != signal.SIG_DFL or realtime_timer != (0.0, 0.0):
        raise LaunchRefused("launcher SIGALRM disposition/timer must be clear")
    try:
        blocked_signals = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    except (AttributeError, OSError, ValueError) as exc:
        raise LaunchRefused(f"launcher blocked-signal mask cannot be inspected: {exc}") from exc
    if blocked_signals:
        raise LaunchRefused("launcher blocked-signal mask must be empty")
    try:
        child_subreaper_state = _prctl_child_subreaper(_libc())
    except OSError as exc:
        raise LaunchRefused(f"child-subreaper state cannot be inspected: {exc}") from exc
    if child_subreaper_state != expected_subreaper_state:
        raise LaunchRefused(
            "launcher child-subreaper state must be "
            f"{expected_subreaper_state}, got {child_subreaper_state}"
        )
    task_ids = _launcher_task_ids()
    if task_ids != (os.getpid(),):
        raise LaunchRefused(f"launcher must have exactly its main task, got {list(task_ids)}")
    children = _launcher_child_pids()
    if children:
        raise LaunchRefused(f"launcher has undeclared children before fork: {list(children)}")
    _require_waitpid_echild()
    if _launcher_child_pids():
        raise LaunchRefused("launcher child inventory changed during the ECHILD proof")
    return {
        "schema": "microtensor.code.launcher-child-boundary.v1",
        "status": "ready",
        "parent_environment": parent_environment,
        "task_ids": list(task_ids),
        "task_count": 1,
        "sigchld_disposition": "SIG_DFL",
        "sigalrm_disposition": "SIG_DFL",
        "realtime_timer_clear": True,
        "blocked_signal_mask_empty": True,
        "child_subreaper_state": child_subreaper_state,
        "proc_children_empty": True,
        "waitpid_echild_verified": True,
    }


def _kill_pid_and_group(pid: int) -> None:
    errors: list[OSError] = []
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(exc)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(exc)
    if errors:
        raise errors[0]


def _kill_process_group(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def _kill_discovered_pid_and_group(pid: int) -> bool:
    descriptor = -1
    pinned = False
    errors: list[OSError] = []
    try:
        descriptor = os.pidfd_open(pid, 0)
        pinned = True
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(exc)
    if descriptor >= 0:
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL, None, 0)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(exc)
        finally:
            os.close(descriptor)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(exc)
    if errors:
        raise errors[0]
    return pinned


def _process_group_absent(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _wait_direct_terminal(pid: int, deadline: float) -> int:
    while True:
        try:
            waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED | _WAIT_ALL)
        except InterruptedError:
            continue
        except ChildProcessError as exc:
            raise LaunchRefused("direct child disappeared without a terminal wait status") from exc
        if waited == 0:
            if time.monotonic() >= deadline:
                raise LaunchRefused("direct child did not terminate during bounded cleanup")
            time.sleep(_BOUNDARY_POLL_MS / 1000)
            continue
        if waited != pid:
            raise LaunchRefused(f"waitpid returned unexpected direct child {waited}")
        if os.WIFSTOPPED(status):
            _kill_pid_and_group(pid)
            continue
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            return status
        raise LaunchRefused("direct child produced an unsupported wait status")


def _wait_exec_stop(
    pid: int,
    error_read: int,
    deadline: float,
) -> tuple[int, bool, bytes]:
    os.set_blocking(error_read, False)
    poller = select.poll()
    poller.register(error_read, select.POLLIN | select.POLLHUP | select.POLLERR)
    child_error = bytearray()
    pipe_eof = False
    stopped_status: int | None = None
    while True:
        while not pipe_eof:
            try:
                chunk = os.read(error_read, MAX_CHILD_ERROR_BYTES + 1 - len(child_error))
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            if not chunk:
                pipe_eof = True
                break
            child_error.extend(chunk)
            if len(child_error) > MAX_CHILD_ERROR_BYTES:
                raise LaunchRefused("child setup error exceeded its byte ceiling")

        try:
            waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED | _WAIT_ALL)
        except InterruptedError:
            continue
        except ChildProcessError as exc:
            raise LaunchRefused("child disappeared before the exec-stop proof") from exc
        if waited == pid:
            if os.WIFSTOPPED(status):
                if os.WSTOPSIG(status) != signal.SIGTRAP:
                    raise LaunchRefused(
                        f"child stopped with signal {os.WSTOPSIG(status)} before exec proof"
                    )
                stopped_status = status
            elif os.WIFEXITED(status) or os.WIFSIGNALED(status):
                return status, True, bytes(child_error)
            else:
                raise LaunchRefused("child produced an unsupported pre-exec wait status")
        elif waited != 0:
            raise LaunchRefused(f"waitpid returned unexpected pre-exec child {waited}")

        if stopped_status is not None and pipe_eof:
            return stopped_status, False, bytes(child_error)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("child did not reach the verified exec-stop before its deadline")
        timeout_ms = max(1, min(_BOUNDARY_POLL_MS, math.ceil(remaining * 1000)))
        try:
            events = poller.poll(timeout_ms)
        except InterruptedError:
            continue
        if any(mask & select.POLLNVAL for _, mask in events):
            raise LaunchRefused("child setup error pipe became invalid")


def _inspect_with_deadline(
    runner: Any,
    pid: int,
    invocation: Invocation,
    contract: LaunchContract,
    deadline: float,
) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("child inspection deadline expired before inspection")
    if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
        raise LaunchRefused("SIGALRM disposition changed before child inspection")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise LaunchRefused("real-time interval timer changed before child inspection")

    def deadline_expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError("child inspection exceeded the shared launch deadline")

    previous = signal.signal(signal.SIGALRM, deadline_expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, remaining)
        result = dict(_mapping(runner(pid, invocation, contract), "child inspection result"))
        if time.monotonic() >= deadline:
            raise TimeoutError("child inspection exceeded the shared launch deadline")
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _containment_evidence(boundary: Mapping[str, Any], prior_subreaper: int) -> dict[str, Any]:
    return {
        **CONTAINMENT_CONTRACT,
        "pre_fork_boundary": dict(boundary),
        "prior_subreaper_state": prior_subreaper,
        "subreaper_enabled": False,
        "direct_pid": None,
        "direct_sigkill_attempted": False,
        "direct_terminal_status_observed": False,
        "direct_returncode": None,
        "process_group_sigkill_attempted": False,
        "process_group_absent": False,
        "descendants_observed": False,
        "observed_descendant_pids": [],
        "pidfd_signaled_descendant_pids": [],
        "terminal_waitpid_echild_verified": False,
        "subreaper_restored": False,
    }


def _drain_adopted_children(
    direct_pid: int,
    evidence: dict[str, Any],
    deadline: float,
) -> None:
    evidence["process_group_sigkill_attempted"] = True
    _kill_process_group(direct_pid)
    observed: set[int] = set(evidence["observed_descendant_pids"])
    pidfd_signaled: set[int] = set(evidence["pidfd_signaled_descendant_pids"])
    while True:
        children = _launcher_child_pids()
        if children:
            evidence["descendants_observed"] = True
        for child_pid in children:
            observed.add(child_pid)
            if _kill_discovered_pid_and_group(child_pid):
                pidfd_signaled.add(child_pid)
        evidence["observed_descendant_pids"] = sorted(observed)
        evidence["pidfd_signaled_descendant_pids"] = sorted(pidfd_signaled)

        try:
            waited, status = os.waitpid(-1, os.WNOHANG | os.WUNTRACED | _WAIT_ALL)
        except InterruptedError:
            continue
        except ChildProcessError:
            if _launcher_child_pids():
                if time.monotonic() >= deadline:
                    raise LaunchRefused(
                        "/proc child inventory remained after waitpid ECHILD"
                    ) from None
                continue
            _kill_process_group(direct_pid)
            if _process_group_absent(direct_pid):
                evidence["process_group_absent"] = True
                evidence["terminal_waitpid_echild_verified"] = True
                return
            evidence["descendants_observed"] = True
        else:
            if waited == 0:
                evidence["descendants_observed"] = True
            else:
                observed.add(waited)
                evidence["descendants_observed"] = True
                evidence["observed_descendant_pids"] = sorted(observed)
                if os.WIFSTOPPED(status):
                    if _kill_discovered_pid_and_group(waited):
                        pidfd_signaled.add(waited)
                        evidence["pidfd_signaled_descendant_pids"] = sorted(pidfd_signaled)
                elif not (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
                    raise LaunchRefused("adopted child produced an unsupported wait status")
        if time.monotonic() >= deadline:
            raise LaunchRefused("descendant/process-group cleanup exceeded its bounded deadline")
        time.sleep(_BOUNDARY_POLL_MS / 1000)


def _returncode(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    raise LaunchRefused("child wait status is neither exited nor signaled")


def _run_traced_process(
    invocation: Invocation,
    contract: LaunchContract,
    *,
    timeout_seconds: int = TIMEOUT_SECONDS,
    _child_target: Any = _child_exec,
    _inspection_runner: Any = _inspect_stopped_child,
) -> ProcessOutcome:
    if type(timeout_seconds) is not int or timeout_seconds < 1 or timeout_seconds > TIMEOUT_SECONDS:
        raise LaunchRefused(f"timeout must be an integer in [1, {TIMEOUT_SECONDS}]")
    if invocation.argv[0] != str(contract.interpreter_path):
        raise LaunchRefused("direct exec argv[0] differs from the signed interpreter")
    environment = dict(contract.environment)
    environment_raw = _canonical_json_bytes(environment)
    if (
        environment != EXACT_ENVIRONMENT
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
        or len(environment) != 16
        or len(environment_raw) != ENVIRONMENT_CANONICAL_BYTES
        or _digest_bytes(environment_raw) != ENVIRONMENT_CANONICAL_DIGEST
    ):
        raise LaunchRefused("direct exec environment changed after contract loading")
    for descriptor in (1, 2):
        try:
            os.fstat(descriptor)
        except OSError as exc:
            raise LaunchRefused(
                f"inherited output descriptor {descriptor} is unavailable: {exc}"
            ) from exc

    boundary = _validate_launcher_boundary_ready()
    library = _libc()
    try:
        prior_subreaper = _prctl_child_subreaper(library)
    except OSError as exc:
        raise LaunchRefused(f"child-subreaper state cannot be inspected: {exc}") from exc
    if prior_subreaper != 0:
        raise LaunchRefused("launcher entered with an existing child-subreaper boundary")
    containment = _containment_evidence(boundary, prior_subreaper)
    started_at_utc, started_at_unix_ns = _timestamp()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + timeout_seconds
    process: dict[str, Any] = {
        "started": False,
        "pid": None,
        "stage": "pre_fork",
        "returncode": None,
        "timed_out": False,
        "started_at_utc": started_at_utc,
        "started_at_unix_ns": started_at_unix_ns,
        "finished_at_utc": None,
        "finished_at_unix_ns": None,
    }
    error_read = -1
    error_write = -1
    pid = -1
    reaped = False
    status: int | None = None
    inspection: dict[str, Any] | None = None
    subreaper_enabled = False
    try:
        _set_child_subreaper(library, 1)
        subreaper_enabled = True
        containment["subreaper_enabled"] = True
        boundary_after_subreaper = _validate_launcher_boundary_ready(expected_subreaper_state=1)
        expected_enabled_boundary = {**boundary, "child_subreaper_state": 1}
        if boundary_after_subreaper != expected_enabled_boundary:
            raise LaunchRefused("launcher child boundary changed while enabling subreaper")
        try:
            error_read, error_write = os.pipe2(os.O_CLOEXEC)
        except (AttributeError, OSError) as exc:
            raise LaunchRefused(f"close-on-exec child error pipe is unavailable: {exc}") from exc
        parent_pid = os.getpid()
        pid = os.fork()
        if pid == 0:
            _child_target(
                invocation,
                contract,
                error_read,
                error_write,
                parent_pid,
                library,
            )
            os._exit(127)
        process.update({"started": True, "pid": pid, "stage": "waiting_for_exec_stop"})
        containment["direct_pid"] = pid
        os.close(error_write)
        error_write = -1
        status, terminal, child_error = _wait_exec_stop(pid, error_read, deadline)
        os.close(error_read)
        error_read = -1
        if terminal:
            reaped = True
            containment["direct_terminal_status_observed"] = True
            containment["direct_returncode"] = _returncode(status)
            process["returncode"] = _returncode(status)
        if child_error:
            rendered = bytes(child_error).decode("ascii", errors="replace")
            raise LaunchRefused(f"child refused before exec: {rendered}")
        if terminal:
            raise LaunchRefused("child terminated before the required post-exec stop")
        process["stage"] = "exec_stop_inspection"
        inspection = _inspect_with_deadline(
            _inspection_runner,
            pid,
            invocation,
            contract,
            deadline,
        )
        process["stage"] = "exec_stop_verified"
        try:
            _ptrace(library, _PTRACE_DETACH, pid)
        except OSError as exc:
            raise LaunchRefused(f"could not detach the verified post-exec child: {exc}") from exc

        process["stage"] = "running"
        timed_out = False
        while True:
            try:
                waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED | _WAIT_ALL)
            except InterruptedError:
                continue
            except ChildProcessError as exc:
                raise LaunchRefused("direct child disappeared during execution") from exc
            if waited == pid:
                if os.WIFSTOPPED(status):
                    raise LaunchRefused(
                        f"verified child stopped unexpectedly with signal {os.WSTOPSIG(status)}"
                    )
                if not (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
                    raise LaunchRefused("verified child produced an unsupported wait status")
                reaped = True
                break
            if waited != 0:
                raise LaunchRefused(f"waitpid returned unexpected running child {waited}")
            if time.monotonic() >= deadline:
                timed_out = True
                process["timed_out"] = True
                process["stage"] = "runtime_timeout_cleanup"
                containment["direct_sigkill_attempted"] = True
                containment["process_group_sigkill_attempted"] = True
                _kill_pid_and_group(pid)
                status = _wait_direct_terminal(
                    pid,
                    time.monotonic() + _CLEANUP_TIMEOUT_SECONDS,
                )
                reaped = True
                break
            time.sleep(_BOUNDARY_POLL_MS / 1000)
        if status is None:
            raise LaunchRefused("direct child has no terminal wait status")
        returncode = _returncode(status)
        containment["direct_terminal_status_observed"] = True
        containment["direct_returncode"] = returncode
        process["returncode"] = returncode
        process["stage"] = "containment_drain"
        _drain_adopted_children(
            pid,
            containment,
            time.monotonic() + _CLEANUP_TIMEOUT_SECONDS,
        )
        _set_child_subreaper(library, prior_subreaper)
        subreaper_enabled = False
        containment["subreaper_restored"] = True
        finished_at_utc, finished_at_unix_ns = _timestamp()
        process.update(
            {
                "stage": "terminal",
                "finished_at_utc": finished_at_utc,
                "finished_at_unix_ns": finished_at_unix_ns,
            }
        )
        if containment["descendants_observed"]:
            process["stage"] = "containment_rejected"
            raise LaunchProcessRefused(
                "evaluator created an observed descendant; repeat is permanently rejected",
                process=process,
                inspection=inspection,
                containment=containment,
            )
        return ProcessOutcome(
            pid=pid,
            returncode=returncode,
            timed_out=timed_out,
            inspection=inspection,
            containment=containment,
            started_at_utc=started_at_utc,
            started_at_unix_ns=started_at_unix_ns,
            finished_at_utc=finished_at_utc,
            finished_at_unix_ns=finished_at_unix_ns,
        )
    except BaseException as exc:
        if isinstance(exc, LaunchProcessRefused):
            raise
        cleanup_failure: BaseException | None = None
        if pid > 0:
            if process["stage"] == "waiting_for_exec_stop" and isinstance(exc, TimeoutError):
                process["stage"] = "pre_exec_timeout_cleanup"
                process["timed_out"] = True
            elif process["stage"] == "exec_stop_inspection" and isinstance(exc, TimeoutError):
                process["stage"] = "exec_stop_inspection_timeout_cleanup"
                process["timed_out"] = True
            elif process["stage"] == "exec_stop_inspection":
                process["stage"] = "exec_stop_inspection_failed"
            else:
                process["stage"] = f"{process['stage']}_failed"
            try:
                if not reaped:
                    containment["direct_sigkill_attempted"] = True
                    containment["process_group_sigkill_attempted"] = True
                    _kill_pid_and_group(pid)
                    status = _wait_direct_terminal(
                        pid,
                        time.monotonic() + _CLEANUP_TIMEOUT_SECONDS,
                    )
                    reaped = True
                if status is not None:
                    returncode = _returncode(status)
                    containment["direct_terminal_status_observed"] = True
                    containment["direct_returncode"] = returncode
                    process["returncode"] = returncode
                _drain_adopted_children(
                    pid,
                    containment,
                    time.monotonic() + _CLEANUP_TIMEOUT_SECONDS,
                )
            except BaseException as cleanup_exc:
                cleanup_failure = cleanup_exc
            try:
                if subreaper_enabled:
                    _set_child_subreaper(library, prior_subreaper)
                    subreaper_enabled = False
                    containment["subreaper_restored"] = True
            except BaseException as restore_exc:
                if cleanup_failure is None:
                    cleanup_failure = restore_exc
            finished_at_utc, finished_at_unix_ns = _timestamp()
            process["finished_at_utc"] = finished_at_utc
            process["finished_at_unix_ns"] = finished_at_unix_ns
            message = str(exc)
            if cleanup_failure is not None:
                message += (
                    f"; bounded containment cleanup failed: {_bounded_error(cleanup_failure)}"
                )
            raise LaunchProcessRefused(
                message,
                process=process,
                inspection=inspection,
                containment=containment,
                original=exc,
            ) from exc
        if subreaper_enabled:
            try:
                _set_child_subreaper(library, prior_subreaper)
            except BaseException as restore_exc:
                raise LaunchRefused(
                    f"{exc}; child-subreaper restoration failed: {restore_exc}"
                ) from restore_exc
        raise
    finally:
        if error_read >= 0:
            os.close(error_read)
        if error_write >= 0:
            os.close(error_write)


def _bounded_error(exc: BaseException) -> str:
    rendered = f"{type(exc).__name__}: {exc}"
    raw = rendered.encode("utf-8", errors="replace")[:MAX_ERROR_TEXT_BYTES]
    return raw.decode("utf-8", errors="replace")


def _attempt_payload(contract: LaunchContract, repeat: str) -> dict[str, Any]:
    consumed_at_utc, consumed_at_unix_ns = _timestamp()
    return {
        "schema": ATTEMPT_SCHEMA,
        "status": "consumed",
        "repeat": repeat,
        "addendum": {"bytes": len(contract.raw), "sha256": contract.digest},
        "public_commit": contract.public_commit,
        "consumed_at_utc": consumed_at_utc,
        "consumed_at_unix_ns": consumed_at_unix_ns,
    }


def _process_payload(
    outcome: ProcessOutcome | None,
    process_failure: LaunchProcessRefused | None = None,
) -> dict[str, Any]:
    if process_failure is not None:
        return dict(process_failure.process)
    if outcome is None:
        return {
            "started": False,
            "pid": None,
            "stage": "not_started",
            "returncode": None,
            "timed_out": False,
            "started_at_utc": None,
            "started_at_unix_ns": None,
            "finished_at_utc": None,
            "finished_at_unix_ns": None,
        }
    return {
        "started": True,
        "pid": outcome.pid,
        "stage": "terminal",
        "returncode": outcome.returncode,
        "timed_out": outcome.timed_out,
        "started_at_utc": outcome.started_at_utc,
        "started_at_unix_ns": outcome.started_at_unix_ns,
        "finished_at_utc": outcome.finished_at_utc,
        "finished_at_unix_ns": outcome.finished_at_unix_ns,
    }


def _launch_repeat(
    contract: LaunchContract,
    repeat: str,
    *,
    process_runner: Any = _run_traced_process,
    preflight_runner: Any = _static_preflight,
    validation_runner: Any = _validate_through,
) -> dict[str, Any]:
    if repeat not in REPEATS:
        raise LaunchRefused(f"repeat must be one of {REPEATS}; no r4 exists")
    if contract.protocol == "current94-v8":
        raise LaunchRefused(CURRENT94_LIVE_BLOCKER)
    _validate_launcher_boundary_ready()
    _ensure_report_root(contract.report_root, REPORT_BASE)
    prior_bindings = _validate_namespace_state(contract, repeat)
    preflight = preflight_runner(contract)
    index = REPEATS.index(repeat)
    preflight_digest = _digest_bytes(_canonical_json_bytes(preflight.report))
    for prior_repeat, binding in prior_bindings.items():
        if binding.preflight_sha256 != preflight_digest:
            raise LaunchRefused(
                f"fresh preflight digest differs from the {prior_repeat} prior receipt"
            )

    fresh_prior_validations: dict[str, tuple[dict[str, Any], str]] = {}
    for prior_repeat in REPEATS[:index]:
        prior_report, prior_digest = validation_runner(
            preflight.validator,
            contract,
            prior_repeat,
        )
        if prior_digest != prior_bindings[prior_repeat].validation_sha256:
            raise LaunchRefused(
                f"fresh {prior_repeat} validation digest differs from its prior receipt"
            )
        fresh_prior_validations[prior_repeat] = (prior_report, prior_digest)

    previous_validation: dict[str, Any] | None = None
    if index:
        previous_repeat = REPEATS[index - 1]
        previous_report, previous_digest = fresh_prior_validations[previous_repeat]
        previous_validation = {
            "status": previous_report["status"],
            "through": previous_repeat,
            "report_sha256": previous_digest,
            "validated_repeat_hard_gates_passed": _mapping(
                previous_report.get("aggregate"), "previous validation aggregate"
            ).get("validated_repeat_hard_gates_passed"),
            "all_declared_local_gates_passed": _mapping(
                previous_report.get("aggregate"), "previous validation aggregate"
            ).get("all_declared_local_gates_passed"),
            "remaining_local_repeats": _mapping(
                previous_report.get("claim"), "previous validation claim"
            ).get("remaining_local_repeats"),
        }

    # Recheck the complete addendum-derived contract and once-only roots after
    # potentially expensive preflight/validation, before consuming the repeat.
    reloaded_contract = _load_contract(contract.path)
    if reloaded_contract != contract:
        raise LaunchRefused("diagnostic execution contract changed before consumption")
    contract = reloaded_contract
    rechecked_bindings = _validate_namespace_state(contract, repeat)
    if rechecked_bindings != prior_bindings:
        raise LaunchRefused("prior validation receipt bindings changed before consumption")
    pre_consumption_boundary = _validate_launcher_boundary_ready()
    invocation = contract.invocations[index]
    attempt_identity = _atomic_publish_noreplace(
        _marker_path(contract, repeat),
        _attempt_payload(contract, repeat),
    )
    outcome: ProcessOutcome | None = None
    validation_payload: dict[str, Any] = {
        "status": "not_run",
        "through": None,
        "report_sha256": None,
        "validated_repeat_hard_gates_passed": False,
        "all_declared_local_gates_passed": False,
        "remaining_local_repeats": list(REPEATS[index:]),
    }
    post_control: dict[str, Any] | None = None
    post_process_boundary: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        outcome = process_runner(invocation, contract)
        if outcome.timed_out:
            raise LaunchRefused(f"{repeat} exceeded the immutable {TIMEOUT_SECONDS}s timeout")
        if outcome.returncode != 0:
            raise LaunchRefused(f"{repeat} evaluator exited {outcome.returncode}")
        post_process_boundary = _validate_launcher_boundary_ready()
        post_contract = _load_contract(contract.path)
        if (
            post_contract.digest != contract.digest
            or post_contract.public_commit != contract.public_commit
            or post_contract.public_git != contract.public_git
        ):
            raise LaunchRefused(
                "addendum or public source binding changed during the evaluator run"
            )
        post_control = {
            "source": _git_source_identity(post_contract),
            "interpreter": _signed_interpreter_identity(post_contract),
            "public_git": post_contract.public_git,
            "addendum_unchanged": True,
            "public_source_unchanged": True,
        }
        validation_report, validation_digest = validation_runner(
            preflight.validator,
            post_contract,
            repeat,
        )
        aggregate = _mapping(validation_report.get("aggregate"), "validation aggregate")
        claim = _mapping(validation_report.get("claim"), "validation claim")
        validation_payload = {
            "status": validation_report["status"],
            "through": repeat,
            "report_sha256": validation_digest,
            "validated_repeat_hard_gates_passed": aggregate.get(
                "validated_repeat_hard_gates_passed"
            ),
            "all_declared_local_gates_passed": aggregate.get("all_declared_local_gates_passed"),
            "remaining_local_repeats": claim.get("remaining_local_repeats"),
        }
    except BaseException as exc:
        failure = exc

    status = validation_payload["status"] if failure is None else "rejected"
    process_failure = failure if isinstance(failure, LaunchProcessRefused) else None
    inspection = (
        outcome.inspection
        if outcome is not None
        else None
        if process_failure is None
        else process_failure.inspection
    )
    process_containment = (
        outcome.containment
        if outcome is not None
        else None
        if process_failure is None
        else process_failure.containment
    )
    receipt = {
        "schema": SCHEMA,
        "status": status,
        "repeat": repeat,
        "outcome": (
            "validated_repeat_hard_gates_passed" if failure is None else "permanently_rejected"
        ),
        "error": None if failure is None else _bounded_error(failure),
        "addendum": {"bytes": len(contract.raw), "sha256": contract.digest},
        "public_source": _public_source_payload(contract),
        "attempt": attempt_identity.as_dict(),
        "execution": {
            "argv": list(invocation.argv),
            "argv_canonical_json_bytes": invocation.argv_canonical_json_bytes,
            "argv_canonical_json_sha256": invocation.argv_canonical_json_sha256,
            "cwd": str(contract.source_root),
            "shell": False,
            "stdin": "/dev/null",
            "umask": UMASK_TEXT,
            "close_fds": True,
            "new_session": True,
            "parent_death_signal": "SIGKILL",
            "timeout_seconds": TIMEOUT_SECONDS,
            "environment_keys": sorted(contract.environment),
            "environment_canonical_json_bytes": ENVIRONMENT_CANONICAL_BYTES,
            "environment_canonical_json_sha256": ENVIRONMENT_CANONICAL_DIGEST,
            "parent_environment_exact": True,
            "containment_contract": CONTAINMENT_CONTRACT,
            "output_root": str(invocation.output_root),
        },
        "preflight": preflight.report,
        "previous_validation": previous_validation,
        "process": _process_payload(outcome, process_failure),
        "inspection": inspection,
        "containment": {
            "pre_consumption_boundary": pre_consumption_boundary,
            "process": process_containment,
            "post_process_boundary": post_process_boundary,
        },
        "post_control": post_control,
        "validation": validation_payload,
        "claim": _launch_receipt_claim(contract),
    }
    try:
        _atomic_publish_noreplace(_receipt_path(contract, repeat), receipt)
    except BaseException as receipt_exc:
        if failure is not None:
            raise LaunchRefused(
                f"{_bounded_error(failure)}; launch receipt publication also failed: "
                f"{_bounded_error(receipt_exc)}"
            ) from receipt_exc
        raise
    if failure is not None:
        original = process_failure.original if process_failure is not None else failure
        if isinstance(original, (KeyboardInterrupt, SystemExit)):
            raise original
        raise LaunchRefused(str(failure)) from failure
    return receipt


def launch_diagnostic(addendum: Path, repeat: str) -> dict[str, Any]:
    """Load the immutable contract and run exactly one eligible repeat."""

    if _is_current94_live_request_lexically(addendum):
        _refuse_current94_live()
    contract = _load_contract(addendum)
    return _launch_repeat(contract, repeat)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--repeat", choices=REPEATS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if __package__:
        print(
            "code GGUF diagnostic launch refused: invoke this launcher by direct path",
            file=sys.stderr,
        )
        return 2
    args = _parse_args(argv)
    try:
        receipt = launch_diagnostic(args.addendum, args.repeat)
    except (LaunchRefused, OSError, ValueError) as exc:
        print(f"code GGUF diagnostic launch refused: {exc}", file=sys.stderr)
        return 2
    print((_canonical_json_bytes(receipt) + b"\n").decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
