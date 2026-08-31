from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .config import (
    SIGNED_V030_CONFIG_VERSION,
    SIGNED_V030_CORPUS_VERSION,
    SIGNED_V030_EMISSION_SHARE,
    SIGNED_V030_EVALUATION_BLOCKS,
    SIGNED_V030_MECHANISM_VERSION,
    SIGNED_V030_METRIC,
    SIGNED_V030_SUBMISSION_BLOCKS,
)
from .errors import RoundNotOpen, RoundRefused
from .models import RoundWindow

MAX_RESPONSE_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BLOCK_HASH = re.compile(r"^0x[0-9a-f]{64}$")
_ENVIRONMENT_DIGEST = re.compile(r"^env:[0-9a-f]{16}$")


def _canonical_hash(config: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise RoundRefused("coordinator served a non-canonical config") from exc
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise ValueError(f"duplicate JSON key {key!r}")
        found[key] = value
    return found


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def fetch_current_round(base_url: str, timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/round/current"
    request = urllib.request.Request(  # noqa: S310 -- config requires HTTPS
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
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RoundRefused(f"coordinator returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise RoundNotOpen("coordinator returned no current round")
    return payload


def _strict_int(payload: Mapping[str, Any], key: str, *, context: str = "round") -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoundRefused(f"coordinator {context} has no integer {key}")
    return value


def _positive_int(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    value = _strict_int(payload, key, context=context)
    if value <= 0:
        raise RoundRefused(f"coordinator {context} has no positive {key}")
    return value


def _validate_v030_config(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    track: str,
    hardware_class: str,
    base_model: str,
    arena: Mapping[str, Any],
) -> dict[str, Any]:
    version = config.get("version")
    if type(version) is not int or version != SIGNED_V030_CONFIG_VERSION:
        raise RoundRefused(
            f"coordinator config version is {version!r}, expected {SIGNED_V030_CONFIG_VERSION}"
        )
    mechanism = config.get("mechanism_version")
    if mechanism != SIGNED_V030_MECHANISM_VERSION:
        raise RoundRefused(
            f"coordinator mechanism is {mechanism!r}, expected {SIGNED_V030_MECHANISM_VERSION!r}"
        )
    corpus_version = config.get("corpus_version")
    if corpus_version != SIGNED_V030_CORPUS_VERSION:
        raise RoundRefused(
            f"coordinator corpus version is {corpus_version!r}, expected "
            f"{SIGNED_V030_CORPUS_VERSION!r}"
        )

    tracks = config.get("tracks")
    rules = tracks.get(track) if isinstance(tracks, dict) else None
    if not isinstance(rules, dict):
        raise RoundRefused(f"coordinator config has no track rules for {track}")
    metric = rules.get("metric")
    if metric != SIGNED_V030_METRIC:
        raise RoundRefused(f"coordinator metric is {metric!r}, expected {SIGNED_V030_METRIC!r}")
    emission = rules.get("emission_share")
    if (
        isinstance(emission, bool)
        or not isinstance(emission, (int, float))
        or float(emission) != SIGNED_V030_EMISSION_SHARE
    ):
        raise RoundRefused(
            f"coordinator emission share is {emission!r}, expected {SIGNED_V030_EMISSION_SHARE}"
        )

    weights = config.get("class_weights")
    class_weight = weights.get(hardware_class) if isinstance(weights, dict) else None
    if (
        isinstance(class_weight, bool)
        or not isinstance(class_weight, (int, float))
        or float(class_weight) != 1.0
    ):
        raise RoundRefused(f"coordinator config has no unit class weight for {hardware_class}")

    class_rules = config.get("classes")
    class_rule = class_rules.get(hardware_class) if isinstance(class_rules, dict) else None
    if not isinstance(class_rule, dict):
        raise RoundRefused(f"coordinator config has no class rules for {hardware_class}")
    for name in ("max_size_bytes", "max_rss_bytes", "max_p95_ms"):
        _positive_int(class_rule, name, context=f"class {hardware_class}")

    allowed = arena.get("allowed_base_models")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(item, str) or not item for item in allowed)
        or allowed != sorted(set(allowed))
    ):
        raise RoundRefused(
            f"coordinator config publishes no canonical base-model allowlist for "
            f"{track}/{hardware_class}"
        )
    if not base_model:
        raise RoundRefused("signed v0.3 activation has no selected base model")
    if base_model not in allowed:
        raise RoundRefused(f"base model {base_model} is not allowed in the served arena")

    cpu_seconds = _positive_int(
        arena,
        "cpu_seconds_per_artifact",
        context=f"arena {track}/{hardware_class}",
    )
    tasks = _positive_int(
        arena,
        "tasks_per_round",
        context=f"arena {track}/{hardware_class}",
    )
    _positive_int(config, "tasks_per_round", context="config")

    environment = arena.get("environment_digest")
    if not isinstance(environment, str) or _ENVIRONMENT_DIGEST.fullmatch(environment) is None:
        raise RoundRefused(
            f"coordinator config has no valid environment digest for {track}/{hardware_class}"
        )
    corpus_digest = payload.get("corpus_digest")
    if not isinstance(corpus_digest, str) or _SHA256.fullmatch(corpus_digest) is None:
        raise RoundRefused("coordinator round has no valid corpus digest")

    seed = _strict_int(payload, "seed_block")
    close = _strict_int(payload, "close_block")
    phase = payload.get("phase")
    block_hash = payload.get("block_hash")
    if seed != close:
        raise RoundRefused("coordinator round seed block does not equal its close block")
    if phase == "submissions" and block_hash != "":
        raise RoundRefused("coordinator submissions phase must not reveal a seed block hash")
    if phase == "evaluation" and (
        not isinstance(block_hash, str) or _BLOCK_HASH.fullmatch(block_hash) is None
    ):
        raise RoundRefused("coordinator evaluation phase has no canonical seed block hash")

    return {
        "seed_block": seed,
        "block_hash": str(block_hash),
        "mechanism_version": str(mechanism),
        "corpus_version": str(corpus_version),
        "corpus_digest": corpus_digest,
        "metric": str(metric),
        "emission_share": float(emission),
        "cpu_seconds_per_artifact": cpu_seconds,
        "tasks_per_round": tasks,
        "environment_digest": environment,
    }


def validate_served_round(
    payload: Mapping[str, Any],
    *,
    chain_head: int,
    track: str,
    hardware_class: str,
    require_anchored: bool,
    previous_status: Mapping[str, Any] | None = None,
    base_model: str = "",
    strict_v030: bool = False,
) -> RoundWindow:
    phase_value = payload.get("phase")
    state_value = payload.get("state")
    if phase_value == "scheduled" or state_value == "scheduled":
        raise RoundNotOpen("coordinator round is scheduled but not open")

    index = _strict_int(payload, "round")
    start = _strict_int(payload, "start_block")
    close = _strict_int(payload, "close_block")
    end = _strict_int(payload, "end_block")
    phase = str(phase_value or "")
    if index < 0 or not (start <= close < end):
        raise RoundRefused("coordinator returned invalid round bounds")
    if chain_head < start:
        raise RoundNotOpen(
            f"coordinator round {index} starts at {start}; chain head is {chain_head}"
        )
    if chain_head >= end:
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
    if strict_v030 and (
        close - start != SIGNED_V030_SUBMISSION_BLOCKS
        or end - close != SIGNED_V030_EVALUATION_BLOCKS
    ):
        raise RoundRefused(
            "signed v0.3 round must have exact 7200-block submission and evaluation windows"
        )

    anchored = payload.get("anchored") is True
    if require_anchored and not anchored:
        raise RoundRefused("coordinator round is not anchored on chain")
    config = payload.get("config")
    config_hash = payload.get("config_hash")
    if (
        not isinstance(config, dict)
        or not isinstance(config_hash, str)
        or _SHA256.fullmatch(config_hash) is None
    ):
        raise RoundRefused("coordinator round has no served config/config hash")
    if _canonical_hash(config) != config_hash:
        raise RoundRefused("coordinator served config does not match its config hash")

    competitions = config.get("competitions")
    wanted = [track, hardware_class]
    if not isinstance(competitions, list) or wanted not in competitions:
        raise RoundRefused(f"coordinator config does not open {track}/{hardware_class}")

    arenas = config.get("arenas")
    arena = arenas.get(f"{track}/{hardware_class}") if isinstance(arenas, dict) else None
    if not isinstance(arena, dict):
        raise RoundRefused(f"coordinator config has no arena rules for {track}/{hardware_class}")

    v030: dict[str, Any] = {}
    if strict_v030:
        v030 = _validate_v030_config(
            payload,
            config,
            track=track,
            hardware_class=hardware_class,
            base_model=base_model,
            arena=arena,
        )
    elif base_model:
        allowed = arena.get("allowed_base_models")
        if not isinstance(allowed, list) or not allowed:
            raise RoundRefused(
                f"coordinator config publishes no base-model allowlist for {track}/{hardware_class}"
            )
        if base_model not in allowed:
            raise RoundRefused(f"base model {base_model} is not allowed in the served arena")

    overrides = arena.get("ceilings")
    classes = config.get("classes")
    class_rules = classes.get(hardware_class) if isinstance(classes, dict) else None
    overrides = overrides if isinstance(overrides, dict) else {}
    class_rules = class_rules if isinstance(class_rules, dict) else {}

    def ceiling(name: str) -> int:
        raw = overrides.get(name) if strict_v030 else overrides.get(name, class_rules.get(name))
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
    if previous_index == index:
        if not isinstance(old_window, dict):
            raise RoundRefused(
                f"coordinator round {index} has no complete previously observed window"
            )
        if strict_v030:
            expected = {
                "round": index,
                "start_block": start,
                "seed_block": v030["seed_block"],
                "close_block": close,
                "end_block": end,
                "config_hash": config_hash,
                "mechanism_version": v030["mechanism_version"],
                "corpus_version": v030["corpus_version"],
                "corpus_digest": v030["corpus_digest"],
            }
            observed = {key: old_window.get(key) for key in expected}
            if observed != expected:
                raise RoundRefused(
                    f"coordinator changed the already observed identity for round {index}"
                )
            old_phase = old_window.get("phase")
            old_hash = old_window.get("block_hash")
            new_hash = v030["block_hash"]
            if old_phase == "submissions":
                if old_hash != "" or phase not in {"submissions", "evaluation"}:
                    raise RoundRefused(
                        f"coordinator changed the observed seed state for round {index}"
                    )
            elif old_phase == "evaluation":
                if phase != "evaluation" or old_hash != new_hash:
                    raise RoundRefused(
                        f"coordinator regressed or changed evaluation for round {index}"
                    )
            else:
                raise RoundRefused(
                    f"coordinator round {index} has an invalid previously observed phase"
                )
        else:
            expected = {
                "start_block": start,
                "close_block": close,
                "end_block": end,
                "config_hash": config_hash,
            }
            observed = {key: old_window.get(key) for key in expected}
            if observed != expected:
                raise RoundRefused(
                    f"coordinator changed the already observed window for round {index}"
                )

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
        **v030,
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
    strict_v030: bool = False,
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
            strict_v030=strict_v030,
        )
    except RoundRefused:
        if strict_v030 or not allow_chain_fallback:
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
