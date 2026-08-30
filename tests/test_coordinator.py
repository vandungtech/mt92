from __future__ import annotations

import unittest

from microtensor_miner_controller.coordinator import resolve_round, validate_served_round
from microtensor_miner_controller.errors import RoundRefused
from microtensor_miner_controller.models import RoundWindow

from helpers import coordinator_payload


class CoordinatorTests(unittest.TestCase):
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
