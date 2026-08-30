from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .errors import RoundRefused
from .models import RoundWindow

MAX_RESPONSE_BYTES = 1024 * 1024


def _canonical_hash(config: Mapping[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def fetch_current_round(base_url: str, timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/round/current"
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "microtensor-miner-controller/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RoundRefused(f"coordinator unavailable: {exc}") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RoundRefused("coordinator round response exceeds 1 MiB")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RoundRefused(f"coordinator returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise RoundRefused("coordinator returned no current round")
    return payload


def _strict_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoundRefused(f"coordinator round has no integer {key}")
    return value


def validate_served_round(
    payload: Mapping[str, Any],
    *,
    chain_head: int,
    track: str,
    hardware_class: str,
    require_anchored: bool,
    previous_status: Mapping[str, Any] | None = None,
    base_model: str = "",
) -> RoundWindow:
    index = _strict_int(payload, "round")
    start = _strict_int(payload, "start_block")
    close = _strict_int(payload, "close_block")
    end = _strict_int(payload, "end_block")
    phase = str(payload.get("phase", ""))
    if index < 0 or not (start <= close < end):
        raise RoundRefused("coordinator returned invalid round bounds")
    if not start <= chain_head < end:
        raise RoundRefused(
            f"coordinator round {index} is stale for chain head {chain_head} "
            f"(window {start}..{end - 1})"
        )
    if phase not in {"submissions", "evaluation"}:
        raise RoundRefused(f"coordinator returned unsupported phase {phase!r}")
    if phase == "submissions" and chain_head >= close:
        raise RoundRefused("coordinator says submissions while the chain is at/after close")
    if phase == "evaluation" and chain_head < close:
        raise RoundRefused("coordinator says evaluation before the close block")

    anchored = payload.get("anchored") is True
    if require_anchored and not anchored:
        raise RoundRefused("coordinator round is not anchored on chain")
    config = payload.get("config")
    config_hash = str(payload.get("config_hash", ""))
    if not isinstance(config, dict) or not config_hash:
        raise RoundRefused("coordinator round has no served config/config hash")
    if _canonical_hash(config) != config_hash:
        raise RoundRefused("coordinator served config does not match its config hash")

    competitions = config.get("competitions")
    wanted = [track, hardware_class]
    if not isinstance(competitions, list) or wanted not in competitions:
        raise RoundRefused(f"coordinator config does not open {track}/{hardware_class}")
    if base_model:
        arenas = config.get("arenas")
        arena = arenas.get(f"{track}/{hardware_class}") if isinstance(arenas, dict) else None
        allowed = arena.get("allowed_base_models") if isinstance(arena, dict) else None
        if not isinstance(allowed, list) or not allowed:
            raise RoundRefused(
                f"coordinator config publishes no base-model allowlist for {track}/{hardware_class}"
            )
        if base_model not in allowed:
            raise RoundRefused(f"base model {base_model} is not allowed in the served arena")

    arenas = config.get("arenas")
    arena = arenas.get(f"{track}/{hardware_class}") if isinstance(arenas, dict) else None
    if not isinstance(arena, dict):
        raise RoundRefused(f"coordinator config has no arena rules for {track}/{hardware_class}")
    overrides = arena.get("ceilings")
    classes = config.get("classes")
    class_rules = classes.get(hardware_class) if isinstance(classes, dict) else None
    overrides = overrides if isinstance(overrides, dict) else {}
    class_rules = class_rules if isinstance(class_rules, dict) else {}

    def ceiling(name: str) -> int:
        raw = overrides.get(name, class_rules.get(name))
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RoundRefused(
                f"anchored config has no positive {name} for {track}/{hardware_class}"
            )
        return raw

    max_size_bytes = ceiling("max_size_bytes")
    max_rss_bytes = ceiling("max_rss_bytes")
    max_p95_ms = ceiling("max_p95_ms")

    previous = dict(previous_status or {})
    previous_index = previous.get("last_coordinator_round")
    if isinstance(previous_index, int) and index < previous_index:
        raise RoundRefused(
            f"coordinator rolled back from observed round {previous_index} to {index}"
        )
    old_window = previous.get("last_coordinator_window")
    if isinstance(old_window, dict) and previous_index == index:
        expected = {
            "start_block": start,
            "close_block": close,
            "end_block": end,
            "config_hash": config_hash,
        }
        observed = {key: old_window.get(key) for key in expected}
        if observed != expected:
            raise RoundRefused(f"coordinator changed the already observed window for round {index}")

    return RoundWindow(
        index=index,
        start_block=start,
        close_block=close,
        end_block=end,
        phase=phase,
        source="coordinator",
        config_hash=config_hash,
        max_size_bytes=max_size_bytes,
        max_rss_bytes=max_rss_bytes,
        max_p95_ms=max_p95_ms,
    )


def resolve_round(
    *,
    base_url: str,
    timeout: int,
    chain_head: int,
    chain_round: RoundWindow,
    track: str,
    hardware_class: str,
    require_anchored: bool,
    allow_chain_fallback: bool,
    previous_status: Mapping[str, Any] | None = None,
    fetcher: Callable[[str, int], dict[str, Any]] = fetch_current_round,
    base_model: str = "",
) -> RoundWindow:
    try:
        payload = fetcher(base_url, timeout)
        return validate_served_round(
            payload,
            chain_head=chain_head,
            track=track,
            hardware_class=hardware_class,
            require_anchored=require_anchored,
            previous_status=previous_status,
            base_model=base_model,
        )
    except RoundRefused:
        if not allow_chain_fallback:
            raise
    return RoundWindow(
        index=chain_round.index,
        start_block=chain_round.start_block,
        close_block=chain_round.close_block,
        end_block=chain_round.end_block,
        phase="submissions" if chain_head < chain_round.close_block else "evaluation",
        source="chain-fallback",
        config_hash="",
    )
