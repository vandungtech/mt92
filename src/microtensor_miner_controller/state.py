from __future__ import annotations

import contextlib
import fcntl
import json
import os
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
        previous = dict(preserve or self.read_status())
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
        payload.update({
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_timestamp(timestamp),
            "updated_at_epoch": timestamp,
            "pid": os.getpid(),
            "phase": phase,
            "ok": bool(ok),
            "message": message,
        })
        if details:
            payload.update(
                {key: value for key, value in details.items() if key not in reserved}
            )
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
            return self.write("starting", ok=False, message="controller has not completed preflight")
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

    def health(self, max_age_seconds: int, *, now: float | None = None) -> tuple[bool, dict[str, Any]]:
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
        except Exception:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
