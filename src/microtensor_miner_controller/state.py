from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import ControllerError
from .redaction import sanitize

SCHEMA_VERSION = 1


class AlreadyRunning(ControllerError):
    pass


def utc_timestamp(now: float | None = None) -> str:
    value = time.time() if now is None else now
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


class StateStore:
    def __init__(self, root: Path, secrets: Sequence[str] = ()) -> None:
        self.root = root
        self.status_path = root / "status.json"
        self.health_path = root / "health.json"
        self.rank_path = root / "rank.json"
        self.authorization_path = root / "authorization-refusal.json"
        self.submission_pending_path = root / "submission-pending.json"
        self.lock_path = root / "controller.lock"
        self.secrets = tuple(secrets)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            self.root.chmod(0o700)

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self.ensure()
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AlreadyRunning(f"another controller holds {self.lock_path}") from exc
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"{os.getpid()}\n".encode())
            os.fsync(descriptor)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def read_status(self) -> dict[str, Any]:
        return self._read(self.status_path)

    def read_health(self) -> dict[str, Any]:
        return self._read(self.health_path)

    def read_rank(self) -> dict[str, Any]:
        return self._read(self.rank_path)

    def read_authorization_refusal(self) -> dict[str, Any]:
        invalid = {
            "schema_version": SCHEMA_VERSION,
            "authorization_latched": True,
            "authorization_reason": (
                "authorization refusal marker is unreadable or invalid; operator review is required"
            ),
            "marker_invalid": True,
        }
        marker = self._read_marker(self.authorization_path, invalid=invalid)
        if not marker or marker.get("marker_invalid") is True:
            return marker
        if (
            marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("authorization_latched") is not True
            or not isinstance(marker.get("authorization_reason"), str)
            or not marker["authorization_reason"]
        ):
            return invalid
        return marker

    def latch_authorization_refusal(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        existing = self.read_authorization_refusal()
        if existing and existing.get("marker_invalid") is not True:
            return existing
        timestamp = time.time() if now is None else now
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "authorization_latched": True,
            "authorization_reason": message,
            "latched_at": utc_timestamp(timestamp),
            "latched_at_epoch": timestamp,
        }
        if details:
            payload.update(dict(details))
        payload["schema_version"] = SCHEMA_VERSION
        payload["authorization_latched"] = True
        payload["authorization_reason"] = message
        payload = sanitize(payload, self.secrets)
        self._atomic_json(self.authorization_path, payload)
        return payload

    def read_submission_pending(self) -> dict[str, Any]:
        invalid = {
            "schema_version": SCHEMA_VERSION,
            "submission_pending": True,
            "reason": (
                "submission pending marker is unreadable or invalid; operator review is required"
            ),
            "marker_invalid": True,
        }
        marker = self._read_marker(self.submission_pending_path, invalid=invalid)
        if not marker or marker.get("marker_invalid") is True:
            return marker
        if (
            marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("submission_pending") is not True
            or type(marker.get("round")) is not int
            or marker["round"] < 0
            or not isinstance(marker.get("source"), str)
            or not marker["source"]
            or not isinstance(marker.get("hotkey"), str)
            or not marker["hotkey"]
            or not self._valid_fingerprint(marker.get("commitment_fingerprint"))
        ):
            return invalid
        return marker

    def mark_submission_pending(
        self,
        *,
        round_index: int,
        source: str,
        hotkey: str,
        commitment_fingerprint: str,
        details: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        if (
            type(round_index) is not int
            or round_index < 0
            or not source
            or not hotkey
            or not self._valid_fingerprint(commitment_fingerprint)
        ):
            raise ControllerError("submission pending marker fields are invalid")
        if self.read_submission_pending():
            raise ControllerError(
                "a submission pending marker already exists; read-only reconciliation is required"
            )
        timestamp = time.time() if now is None else now
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "submission_pending": True,
            "round": round_index,
            "source": source,
            "hotkey": hotkey,
            "commitment_fingerprint": commitment_fingerprint,
            "created_at": utc_timestamp(timestamp),
            "created_at_epoch": timestamp,
        }
        if details:
            payload.update(dict(details))
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "submission_pending": True,
                "round": round_index,
                "source": source,
                "hotkey": hotkey,
                "commitment_fingerprint": commitment_fingerprint,
            }
        )
        payload = sanitize(payload, self.secrets)
        self._atomic_json(self.submission_pending_path, payload)
        return payload

    def clear_submission_pending(self, commitment_fingerprint: str) -> None:
        pending = self.read_submission_pending()
        if not pending:
            raise ControllerError("submission pending marker disappeared before reconciliation")
        if pending.get("marker_invalid") is True:
            raise ControllerError(
                "invalid submission pending marker cannot be cleared automatically"
            )
        if pending.get("commitment_fingerprint") != commitment_fingerprint:
            raise ControllerError(
                "submission pending marker does not match the verified commitment"
            )
        try:
            self.submission_pending_path.unlink()
        except OSError as exc:
            raise ControllerError(
                "verified submission pending marker could not be removed"
            ) from exc
        self._fsync_root()

    def write_rank(self, payload: Mapping[str, Any]) -> None:
        """Atomically persist observer-only rank state without touching health."""
        self._atomic_json(self.rank_path, sanitize(dict(payload), self.secrets))

    def write(
        self,
        phase: str,
        *,
        ok: bool,
        message: str,
        now: float | None = None,
        details: Mapping[str, Any] | None = None,
        preserve: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = time.time() if now is None else now
        previous = dict(self.read_status() if preserve is None else preserve)
        reserved = {
            "schema_version",
            "updated_at",
            "updated_at_epoch",
            "pid",
            "phase",
            "ok",
            "message",
        }
        payload: dict[str, Any] = {
            key: value
            for key, value in previous.items()
            if key
            not in {
                "schema_version",
                "updated_at",
                "updated_at_epoch",
                "pid",
                "phase",
                "ok",
                "message",
            }
        }
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_timestamp(timestamp),
                "updated_at_epoch": timestamp,
                "pid": os.getpid(),
                "phase": phase,
                "ok": bool(ok),
                "message": message,
            }
        )
        if details:
            payload.update({key: value for key, value in details.items() if key not in reserved})
        payload = sanitize(payload, self.secrets)

        health = {
            "schema_version": SCHEMA_VERSION,
            "ok": bool(payload.get("ok", False)),
            "phase": str(payload.get("phase", "unknown")),
            "message": str(payload.get("message", "")),
            "round": payload.get("round"),
            "last_success_round": payload.get("last_success_round"),
            "updated_at": payload["updated_at"],
            "updated_at_epoch": payload["updated_at_epoch"],
        }
        self._atomic_json(self.status_path, payload)
        self._atomic_json(self.health_path, health)
        return payload

    def mark_verified(
        self,
        *,
        round_index: int,
        message: str,
        details: Mapping[str, Any],
        now: float | None = None,
    ) -> dict[str, Any]:
        timestamp = time.time() if now is None else now
        merged = dict(details)
        merged.update(
            {
                "round": round_index,
                "last_success_round": round_index,
                "last_success_at": utc_timestamp(timestamp),
                "last_success_at_epoch": timestamp,
                "last_verified_at": utc_timestamp(timestamp),
                "last_verified_at_epoch": timestamp,
                "consecutive_failures": 0,
            }
        )
        return self.write("verified", ok=True, message=message, now=timestamp, details=merged)

    def heartbeat(self, *, now: float | None = None) -> dict[str, Any]:
        previous = self.read_status()
        if not previous:
            return self.write(
                "starting", ok=False, message="controller has not completed preflight"
            )
        details = {
            key: value
            for key, value in previous.items()
            if key
            not in {
                "schema_version",
                "updated_at",
                "updated_at_epoch",
                "pid",
                "phase",
                "ok",
                "message",
            }
        }
        return self.write(
            str(previous.get("phase", "unknown")),
            ok=bool(previous.get("ok", False)),
            message=str(previous.get("message", "")),
            now=now,
            details=details,
            preserve=previous,
        )

    def health(
        self, max_age_seconds: int, *, now: float | None = None
    ) -> tuple[bool, dict[str, Any]]:
        current = self.read_health()
        if not current:
            return False, {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "phase": "missing",
                "message": f"no health state at {self.health_path}",
            }
        timestamp = time.time() if now is None else now
        try:
            age = max(0.0, timestamp - float(current["updated_at_epoch"]))
        except (KeyError, TypeError, ValueError):
            age = float("inf")
        current["age_seconds"] = None if age == float("inf") else round(age, 3)
        if age > max_age_seconds:
            current["ok"] = False
            current["message"] = f"health state is stale ({age:.0f}s old)"
        return bool(current.get("ok", False)), current

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _valid_fingerprint(value: Any) -> bool:
        if not isinstance(value, str) or not value.startswith("sha256:"):
            return False
        digest = value.removeprefix("sha256:")
        return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)

    @staticmethod
    def _read_marker(path: Path, *, invalid: Mapping[str, Any]) -> dict[str, Any]:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return {}
        except OSError:
            return dict(invalid)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return dict(invalid)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(invalid)
        if not isinstance(value, dict) or not value:
            return dict(invalid)
        return dict(value)

    def _fsync_root(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.root, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.ensure()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            os.fchmod(descriptor, 0o600)
            encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            with contextlib.suppress(OSError):
                path.chmod(0o600)
            self._fsync_root()
        except Exception:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
