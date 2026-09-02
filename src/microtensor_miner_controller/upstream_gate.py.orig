from __future__ import annotations

import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any

OBSERVER_SCHEMA = "microtensor.upstream-observation.v1"
OBSERVER_SCHEMA_VERSION = 1
AUDITED_ORIGIN_HEAD = "d77adc945de763f8b3b2d71fef8193090ede7001"
AUDITED_RELEASE_VERSION = "0.3.2"
AUDITED_MECHANISM_VERSION = "0.3.0"
EXPECTED_ORIGIN = "https://github.com/microtensor-io/microtensor-subnet"
DEFAULT_MAX_AGE_SECONDS = 900
MIN_MAX_AGE_SECONDS = 600
MAX_FUTURE_SKEW_SECONDS = 30
MAX_STATUS_BYTES = 64 * 1024


class UpstreamGateError(RuntimeError):
    """The observer status cannot prove the exact audited upstream state."""


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _validate_metadata(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise UpstreamGateError("upstream observer status must be a regular non-symlink")
    if metadata.st_nlink != 1:
        raise UpstreamGateError("upstream observer status must have exactly one hard link")
    if metadata.st_uid != os.geteuid():
        raise UpstreamGateError("upstream observer status must be owned by the effective user")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise UpstreamGateError("upstream observer status mode must be exactly 0600")
    if metadata.st_size < 0 or metadata.st_size > MAX_STATUS_BYTES:
        raise UpstreamGateError("upstream observer status exceeds the 64 KiB limit")


def _read_status(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise UpstreamGateError("upstream observer status path must be absolute")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise UpstreamGateError("this platform cannot securely open observer status files")

    descriptor = -1
    try:
        path_before = path.lstat()
        _validate_metadata(path_before)
        flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        opened_before = os.fstat(descriptor)
        _validate_metadata(opened_before)
        if (path_before.st_dev, path_before.st_ino) != (
            opened_before.st_dev,
            opened_before.st_ino,
        ):
            raise UpstreamGateError("upstream observer status changed while it was opened")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(16 * 1024, MAX_STATUS_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_STATUS_BYTES:
                raise UpstreamGateError("upstream observer status exceeds the 64 KiB limit")
        opened_after = os.fstat(descriptor)
        _validate_metadata(opened_after)
        path_after = path.lstat()
        _validate_metadata(path_after)
    except UpstreamGateError:
        raise
    except OSError:
        raise UpstreamGateError("upstream observer status is unavailable or unsafe") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    expected_identity = _identity(opened_before)
    if (
        expected_identity != _identity(opened_after)
        or expected_identity != _identity(path_before)
        or expected_identity != _identity(path_after)
        or total != opened_before.st_size
    ):
        raise UpstreamGateError("upstream observer status changed while it was read")

    raw = b"".join(chunks)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise UpstreamGateError("upstream observer status is not valid UTF-8") from None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite number {value!r}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError):
        raise UpstreamGateError("upstream observer status is not strict JSON") from None
    if not isinstance(payload, dict):
        raise UpstreamGateError("upstream observer status must be a JSON mapping")
    return payload


def _require_exact(payload: dict[str, Any], key: str, expected: Any) -> None:
    if key not in payload or type(payload[key]) is not type(expected) or payload[key] != expected:
        raise UpstreamGateError(f"upstream observer status has invalid {key}")


def _require_fresh_timestamp(
    payload: dict[str, Any],
    key: str,
    *,
    now: float,
    max_age_seconds: int,
) -> None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamGateError(f"upstream observer status has invalid {key}")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise UpstreamGateError(f"upstream observer status has invalid {key}")
    if timestamp > now + MAX_FUTURE_SKEW_SECONDS:
        raise UpstreamGateError(f"upstream observer status {key} is in the future")
    if now - timestamp > max_age_seconds:
        raise UpstreamGateError(f"upstream observer status {key} is stale")


def verify_upstream_observer_status(
    path: Path,
    *,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    """Read and validate the observer's current audited-upstream attestation.

    This consumes inert JSON only. It never imports the observer or any Microtensor
    checkout, and every invocation reopens and revalidates the status file.
    """

    if (
        type(max_age_seconds) is not int
        or max_age_seconds < MIN_MAX_AGE_SECONDS
        or max_age_seconds > DEFAULT_MAX_AGE_SECONDS
    ):
        raise UpstreamGateError(
            "observer maximum age must be an integer between "
            f"{MIN_MAX_AGE_SECONDS} and {DEFAULT_MAX_AGE_SECONDS} seconds"
        )
    observed_now = time.time() if now is None else now
    if isinstance(observed_now, bool) or not isinstance(observed_now, (int, float)):
        raise UpstreamGateError("observer verification time is invalid")
    observed_now = float(observed_now)
    if not math.isfinite(observed_now) or observed_now <= 0:
        raise UpstreamGateError("observer verification time is invalid")

    payload = _read_status(path)
    exact_fields = {
        "schema": OBSERVER_SCHEMA,
        "schema_version": OBSERVER_SCHEMA_VERSION,
        "observation_succeeded": True,
        "phase": "current",
        "ok": True,
        "origin": EXPECTED_ORIGIN,
        "origin_head": AUDITED_ORIGIN_HEAD,
        "audited_origin_head": AUDITED_ORIGIN_HEAD,
        "release_version": AUDITED_RELEASE_VERSION,
        "mechanism_version": AUDITED_MECHANISM_VERSION,
        "provenance_required": False,
        "local_checkout_at_origin": True,
        "audited_head_is_ancestor": True,
        "review_required": False,
        "miner_impact_review_required": False,
        "history_rewrite_detected": False,
        "history_rewrite_latched": False,
        "changed_files_truncated": False,
        "commits_since_audit": 0,
    }
    for key, expected in exact_fields.items():
        _require_exact(payload, key, expected)
    changed_files = payload.get("changed_files")
    if type(changed_files) is not list or changed_files:
        raise UpstreamGateError("upstream observer status has invalid changed_files")

    _require_fresh_timestamp(
        payload,
        "updated_at_epoch",
        now=observed_now,
        max_age_seconds=max_age_seconds,
    )
    _require_fresh_timestamp(
        payload,
        "origin_observed_at",
        now=observed_now,
        max_age_seconds=max_age_seconds,
    )
    return payload
