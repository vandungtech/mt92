from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .redaction import redact_text
from .state import StateStore, utc_timestamp

log = logging.getLogger(__name__)

PUBLIC_API_URL = "https://api.microtensor.cloud"
POLL_SECONDS = 300
TIMEOUT_SECONDS = 10
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

Fetcher = Callable[[str, int], Mapping[str, Any]]


def fetch_leaderboard(url: str, timeout: int = TIMEOUT_SECONDS) -> Mapping[str, Any]:
    """Fetch public leaderboard JSON without credentials or side effects."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("the public leaderboard URL must use HTTPS")
    request = urllib.request.Request(  # noqa: S310 - HTTPS is checked above
        url,
        headers={"accept": "application/json", "user-agent": "microtensor-rank-monitor/1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        if urllib.parse.urlparse(response.geturl()).scheme != "https":
            raise ValueError("the public leaderboard redirected away from HTTPS")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("the public leaderboard response is too large")
    payload = json.loads(raw or b"null")
    if not isinstance(payload, dict):
        raise ValueError("the public leaderboard response must be a JSON object")
    return payload


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"leaderboard {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"leaderboard {field} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"leaderboard {field} must be finite and non-negative")
    return number


def _rank(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("leaderboard rank must be a positive integer")
    try:
        rank = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("leaderboard rank must be a positive integer") from exc
    if rank < 1 or str(value).strip() not in {str(rank), f"{rank}.0"}:
        raise ValueError("leaderboard rank must be a positive integer")
    return rank


def _round_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("leaderboard round_index must be a non-negative integer")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("leaderboard round_index must be a non-negative integer") from exc
    if index < 0 or str(value).strip() not in {str(index), f"{index}.0"}:
        raise ValueError("leaderboard round_index must be a non-negative integer")
    return index


def _standing(row: Mapping[str, Any]) -> dict[str, Any]:
    hotkey = str(row.get("hotkey", "")).strip()
    if not hotkey:
        raise ValueError("leaderboard system is missing its hotkey")
    if not isinstance(row.get("on_frontier"), bool):
        raise ValueError("leaderboard on_frontier must be boolean")
    return {
        "hotkey": hotkey,
        "rank": _rank(row.get("rank")),
        "quality": _number(row.get("quality"), "quality"),
        "cost": _number(row.get("expected_ms"), "expected_ms"),
        "frontier": bool(row["on_frontier"]),
        "share": _number(row.get("exclusive_hv"), "exclusive_hv"),
    }


def parse_leaderboard(
    payload: Mapping[str, Any], *, track: str, hardware_class: str, hotkey: str
) -> dict[str, Any]:
    """Return one hotkey's rank plus the current leader from a public board."""
    if payload.get("track") != track or payload.get("class") != hardware_class:
        raise ValueError("the public leaderboard names a different competition")
    raw_systems = payload.get("systems")
    if not isinstance(raw_systems, list):
        raise ValueError("the public leaderboard systems field must be a list")

    standings: list[dict[str, Any]] = []
    for raw in raw_systems:
        if not isinstance(raw, dict):
            raise ValueError("the public leaderboard contains a malformed system")
        standings.append(_standing(raw))

    ranks = [row["rank"] for row in standings]
    if len(set(ranks)) != len(ranks):
        raise ValueError("the public leaderboard contains duplicate ranks")
    leader = min(standings, key=lambda row: row["rank"], default=None)
    if leader is not None and leader["rank"] != 1:
        raise ValueError("the public leaderboard has systems but no rank-1 leader")

    matches = [row for row in standings if row["hotkey"] == hotkey]
    if len(matches) > 1:
        raise ValueError("the public leaderboard contains duplicate rows for the hotkey")
    mine = matches[0] if matches else None
    return {
        "round": _round_index(payload.get("round_index")),
        "found": mine is not None,
        "rank": mine["rank"] if mine else None,
        "quality": mine["quality"] if mine else None,
        "cost": mine["cost"] if mine else None,
        "frontier": mine["frontier"] if mine else False,
        "share": mine["share"] if mine else None,
        "leader": leader,
        "goal_achieved": bool(mine and mine["rank"] == 1),
        "reachability": True,
        "systems": len(standings),
    }


class RankMonitor:
    """Best-effort observer that is deliberately disconnected from submission state."""

    def __init__(
        self,
        state: StateStore,
        *,
        track: str,
        hardware_class: str,
        hotkey: str,
        base_url: str = PUBLIC_API_URL,
        poll_seconds: int = POLL_SECONDS,
        timeout_seconds: int = TIMEOUT_SECONDS,
        fetcher: Fetcher = fetch_leaderboard,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.state = state
        self.track = track
        self.hardware_class = hardware_class
        self.hotkey = hotkey
        self.url = (
            f"{base_url.rstrip('/')}/v1/arenas/{track}/{hardware_class}/leaderboard"
        )
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.fetcher = fetcher
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> dict[str, Any]:
        now = self.clock() if self.clock is not None else time.time()
        common: dict[str, Any] = {
            "schema_version": 1,
            "updated_at": utc_timestamp(now),
            "updated_at_epoch": now,
            "competition": f"{self.track}/{self.hardware_class}",
            "hotkey": self.hotkey,
            "endpoint": self.url,
        }
        try:
            parsed = parse_leaderboard(
                self.fetcher(self.url, self.timeout_seconds),
                track=self.track,
                hardware_class=self.hardware_class,
                hotkey=self.hotkey,
            )
            payload = {**common, **parsed}
        except Exception as exc:
            message = redact_text(str(exc) or exc.__class__.__name__, self.state.secrets)
            payload = {
                **common,
                "found": False,
                "rank": None,
                "quality": None,
                "cost": None,
                "frontier": False,
                "share": None,
                "leader": None,
                "goal_achieved": False,
                "reachability": False,
                "error": message,
            }
            log.warning("public rank unavailable: %s", message)
        self.state.write_rank(payload)
        return payload

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="microtensor-rank-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                message = redact_text(str(exc) or exc.__class__.__name__, self.state.secrets)
                log.warning("rank observer could not persist its state: %s", message)
            self._stop.wait(self.poll_seconds)
