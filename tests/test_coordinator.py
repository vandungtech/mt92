from __future__ import annotations

import hashlib
import json
import unittest

from helpers import V030_TEST_BASE_MODEL, coordinator_payload, v030_coordinator_payload

from microtensor_miner_controller.coordinator import resolve_round, validate_served_round
from microtensor_miner_controller.errors import RoundNotOpen, RoundRefused
from microtensor_miner_controller.models import RoundWindow


def _dictionary(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("test fixture value must be a dictionary")
    return value


def _rehash(payload: dict[str, object]) -> None:
    config = _dictionary(payload["config"])
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    payload["config_hash"] = "sha256:" + hashlib.sha256(raw).hexdigest()


def _observed(window: RoundWindow) -> dict[str, object]:
    return {
        "round": window.index,
        "start_block": window.start_block,
        "seed_block": window.seed_block,
        "close_block": window.close_block,
        "end_block": window.end_block,
        "phase": window.phase,
        "block_hash": window.block_hash,
        "config_hash": window.config_hash,
        "mechanism_version": window.mechanism_version,
        "corpus_version": window.corpus_version,
        "corpus_digest": window.corpus_digest,
    }


def _validate_v030(
    payload: dict[str, object],
    *,
    chain_head: int,
    previous: dict[str, object] | None = None,
) -> RoundWindow:
    return validate_served_round(
        payload,
        chain_head=chain_head,
        track="code",
        hardware_class="mt-3g",
        require_anchored=True,
        previous_status=previous,
        base_model=V030_TEST_BASE_MODEL,
        strict_v030=True,
    )


class CoordinatorTests(unittest.TestCase):
    def test_v030_seed_hash_is_phase_specific(self) -> None:
        submitted = _validate_v030(v030_coordinator_payload(), chain_head=150)
        self.assertEqual(submitted.seed_block, submitted.close_block)
        self.assertEqual(submitted.block_hash, "")

        with self.assertRaisesRegex(RoundRefused, "must not reveal"):
            _validate_v030(
                v030_coordinator_payload(block_hash="0x" + "c" * 64),
                chain_head=150,
            )

        evaluated = _validate_v030(
            v030_coordinator_payload(phase="evaluation"),
            chain_head=7_400,
        )
        self.assertEqual(evaluated.block_hash, "0x" + "c" * 64)
        with self.assertRaisesRegex(RoundRefused, "canonical seed block hash"):
            _validate_v030(
                v030_coordinator_payload(phase="evaluation", block_hash=""),
                chain_head=7_400,
            )

    def test_v032_accepts_control_plane_phase_split(self) -> None:
        found = _validate_v030(
            v030_coordinator_payload(start=100, close=8_990, end=14_500),
            chain_head=150,
        )
        self.assertEqual(found.close_block - found.start_block, 8_890)
        self.assertEqual(found.end_block - found.close_block, 5_510)

    def test_v032_requires_two_positive_phase_windows(self) -> None:
        for start, close, end, phase, head in (
            (100, 100, 14_500, "evaluation", 100),
            (100, 14_500, 14_500, "submissions", 150),
        ):
            with (
                self.subTest(start=start, close=close, end=end),
                self.assertRaisesRegex(RoundRefused, "invalid round bounds"),
            ):
                _validate_v030(
                    v030_coordinator_payload(
                        start=start,
                        close=close,
                        end=end,
                        phase=phase,
                    ),
                    chain_head=head,
                )

    def test_v030_same_round_transition_is_monotonic_and_identity_bound(self) -> None:
        submitted = _validate_v030(v030_coordinator_payload(), chain_head=150)
        previous = {
            "last_coordinator_round": submitted.index,
            "last_coordinator_window": _observed(submitted),
        }
        _validate_v030(v030_coordinator_payload(), chain_head=150, previous=previous)
        evaluated = _validate_v030(
            v030_coordinator_payload(phase="evaluation"),
            chain_head=7_400,
            previous=previous,
        )
        evaluated_previous = {
            "last_coordinator_round": evaluated.index,
            "last_coordinator_window": _observed(evaluated),
        }
        _validate_v030(
            v030_coordinator_payload(phase="evaluation"),
            chain_head=7_400,
            previous=evaluated_previous,
        )
        with self.assertRaisesRegex(RoundRefused, "regressed"):
            _validate_v030(
                v030_coordinator_payload(),
                chain_head=150,
                previous=evaluated_previous,
            )
        with self.assertRaisesRegex(RoundRefused, "changed evaluation"):
            _validate_v030(
                v030_coordinator_payload(phase="evaluation", block_hash="0x" + "d" * 64),
                chain_head=7_400,
                previous=evaluated_previous,
            )

        stable = _observed(submitted)
        for key in (
            "round",
            "start_block",
            "seed_block",
            "close_block",
            "end_block",
            "config_hash",
            "mechanism_version",
            "corpus_version",
            "corpus_digest",
        ):
            changed = dict(stable)
            value = changed[key]
            changed[key] = value + 1 if isinstance(value, int) else f"{value}-changed"
            prior = {
                "last_coordinator_round": submitted.index,
                "last_coordinator_window": changed,
            }
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RoundRefused, "changed the already observed identity"),
            ):
                _validate_v030(
                    v030_coordinator_payload(),
                    chain_head=150,
                    previous=prior,
                )

    def test_accepts_strict_signed_v030_round_and_arena_bindings(self) -> None:
        found = validate_served_round(
            v030_coordinator_payload(),
            chain_head=150,
            track="code",
            hardware_class="mt-3g",
            require_anchored=True,
            base_model=V030_TEST_BASE_MODEL,
            strict_v030=True,
        )
        self.assertEqual(found.mechanism_version, "0.3.0")
        self.assertEqual(found.corpus_version, "2026.1")
        self.assertEqual(found.metric, "execution_pass_rate")
        self.assertEqual(found.emission_share, 1.0)
        self.assertEqual(found.tasks_per_round, 60)
        self.assertEqual(found.environment_digest, "env:a9b6b17587d8aaea")

    def test_scheduled_v030_round_is_not_open(self) -> None:
        with self.assertRaisesRegex(RoundNotOpen, "scheduled"):
            validate_served_round(
                v030_coordinator_payload(phase="scheduled"),
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

    def test_v030_round_refuses_unanchored_or_incoherent_seed(self) -> None:
        payload = v030_coordinator_payload(anchored=False)
        with self.assertRaisesRegex(RoundRefused, "not anchored"):
            validate_served_round(
                payload,
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

        payload = v030_coordinator_payload()
        payload["seed_block"] = 199
        with self.assertRaisesRegex(RoundRefused, "seed block"):
            validate_served_round(
                payload,
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

    def test_v030_round_refuses_missing_arena_and_emission_mismatch(self) -> None:
        missing = v030_coordinator_payload()
        config = _dictionary(missing["config"])
        config["arenas"] = {}
        _rehash(missing)
        with self.assertRaisesRegex(RoundRefused, "no arena rules"):
            validate_served_round(
                missing,
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

        mismatch = v030_coordinator_payload()
        config = _dictionary(mismatch["config"])
        tracks = _dictionary(config["tracks"])
        code = _dictionary(tracks["code"])
        code["emission_share"] = 0.0
        _rehash(mismatch)
        with self.assertRaisesRegex(RoundRefused, "emission share"):
            validate_served_round(
                mismatch,
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

    def test_v030_round_refuses_changed_config_and_missing_environment(self) -> None:
        changed = v030_coordinator_payload()
        config = _dictionary(changed["config"])
        config["corpus_version"] = "other"
        with self.assertRaisesRegex(RoundRefused, "does not match"):
            validate_served_round(
                changed,
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

        missing = v030_coordinator_payload()
        config = _dictionary(missing["config"])
        arenas = _dictionary(config["arenas"])
        arena = _dictionary(arenas["code/mt-3g"])
        arena.pop("environment_digest")
        _rehash(missing)
        with self.assertRaisesRegex(RoundRefused, "environment digest"):
            validate_served_round(
                missing,
                chain_head=150,
                track="code",
                hardware_class="mt-3g",
                require_anchored=True,
                base_model=V030_TEST_BASE_MODEL,
                strict_v030=True,
            )

    def test_accepts_coherent_anchored_round(self) -> None:
        found = validate_served_round(
            coordinator_payload(),
            chain_head=150,
            track="extract",
            hardware_class="mt-3g",
            require_anchored=True,
        )
        self.assertEqual(found.index, 7)
        self.assertEqual(found.source, "coordinator")

    def test_stale_bounds_are_refused(self) -> None:
        with self.assertRaisesRegex(RoundRefused, "stale"):
            validate_served_round(
                coordinator_payload(),
                chain_head=301,
                track="extract",
                hardware_class="mt-3g",
                require_anchored=True,
            )

    def test_phase_mismatch_is_refused(self) -> None:
        with self.assertRaisesRegex(RoundRefused, "evaluation before"):
            validate_served_round(
                coordinator_payload(phase="evaluation"),
                chain_head=150,
                track="extract",
                hardware_class="mt-3g",
                require_anchored=True,
            )

    def test_changed_observed_window_is_refused(self) -> None:
        previous = {
            "last_coordinator_round": 7,
            "last_coordinator_window": {
                "start_block": 100,
                "close_block": 199,
                "end_block": 300,
                "config_hash": coordinator_payload()["config_hash"],
            },
        }
        with self.assertRaisesRegex(RoundRefused, "changed"):
            validate_served_round(
                coordinator_payload(),
                chain_head=150,
                track="extract",
                hardware_class="mt-3g",
                require_anchored=True,
                previous_status=previous,
            )

    def test_fallback_requires_explicit_switch(self) -> None:
        chain = RoundWindow(99, 100, 200, 300, "submissions", "chain")

        def unavailable(url: str, timeout: int) -> dict[str, object]:
            del url, timeout
            raise RoundRefused("offline")

        with self.assertRaisesRegex(RoundRefused, "offline"):
            resolve_round(
                base_url="https://coordinator.example",
                timeout=1,
                chain_head=150,
                chain_round=chain,
                track="extract",
                hardware_class="mt-3g",
                require_anchored=True,
                allow_chain_fallback=False,
                fetcher=unavailable,
            )
        fallback = resolve_round(
            base_url="https://coordinator.example",
            timeout=1,
            chain_head=150,
            chain_round=chain,
            track="extract",
            hardware_class="mt-3g",
            require_anchored=True,
            allow_chain_fallback=True,
            fetcher=unavailable,
        )
        self.assertEqual(fallback.source, "chain-fallback")
        self.assertEqual(fallback.index, 99)
