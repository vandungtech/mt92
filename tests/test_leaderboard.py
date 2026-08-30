from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from microtensor_miner_controller.leaderboard import RankMonitor, parse_leaderboard
from microtensor_miner_controller.state import StateStore


def board() -> dict[str, object]:
    return {
        "track": "extract",
        "class": "mt-3g",
        "round_index": 1237,
        "systems": [
            {
                "hotkey": "5Mine",
                "rank": 1,
                "quality": 0.95,
                "expected_ms": 3550,
                "on_frontier": True,
                "exclusive_hv": 0.7,
            },
            {
                "hotkey": "5Other",
                "rank": 2,
                "quality": 0.47,
                "expected_ms": 1351,
                "on_frontier": True,
                "exclusive_hv": 0.12,
            },
        ],
    }


class LeaderboardTests(unittest.TestCase):
    def test_parses_server_rank_and_rank_one_goal(self) -> None:
        parsed = parse_leaderboard(
            board(), track="extract", hardware_class="mt-3g", hotkey="5Mine"
        )
        self.assertEqual(parsed["rank"], 1)
        self.assertEqual(parsed["quality"], 0.95)
        self.assertEqual(parsed["cost"], 3550.0)
        self.assertTrue(parsed["frontier"])
        self.assertEqual(parsed["share"], 0.7)
        self.assertTrue(parsed["goal_achieved"])
        self.assertTrue(parsed["reachability"])
        self.assertEqual(parsed["leader"]["hotkey"], "5Mine")  # type: ignore[index]

    def test_reachable_board_can_report_hotkey_absent(self) -> None:
        parsed = parse_leaderboard(
            board(), track="extract", hardware_class="mt-3g", hotkey="5Absent"
        )
        self.assertFalse(parsed["found"])
        self.assertIsNone(parsed["rank"])
        self.assertFalse(parsed["goal_achieved"])
        self.assertTrue(parsed["reachability"])
        self.assertEqual(parsed["leader"]["rank"], 1)  # type: ignore[index]

    def test_api_failure_only_changes_private_rank_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary), secrets=("top-secret",))
            state.write("dry_run", ok=False, message="submission state")
            status_before = state.read_status()
            health_before = state.read_health()

            def unavailable(url: str, timeout: int) -> dict[str, object]:
                del url, timeout
                raise TimeoutError("token=top-secret")

            monitor = RankMonitor(
                state,
                track="extract",
                hardware_class="mt-3g",
                hotkey="5Mine",
                fetcher=unavailable,
                clock=lambda: 1_000.0,
            )
            result = monitor.poll_once()

            self.assertFalse(result["reachability"])
            self.assertFalse(result["goal_achieved"])
            self.assertIsNone(result["rank"])
            self.assertNotIn("top-secret", str(result))
            self.assertEqual(state.read_status(), status_before)
            self.assertEqual(state.read_health(), health_before)
            self.assertEqual(state.read_rank(), result)
            self.assertEqual(os.stat(state.rank_path).st_mode & 0o777, 0o600)

    def test_background_monitor_polls_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            called = threading.Event()

            def available(url: str, timeout: int) -> dict[str, object]:
                del url, timeout
                called.set()
                return board()

            state = StateStore(Path(temporary))
            monitor = RankMonitor(
                state,
                track="extract",
                hardware_class="mt-3g",
                hotkey="5Mine",
                poll_seconds=60,
                fetcher=available,
            )
            monitor.start()
            self.assertTrue(called.wait(1.0))
            monitor.stop()
            self.assertTrue(state.read_rank()["goal_achieved"])

    def test_wrong_competition_is_unreachable_not_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = board()
            payload["class"] = "mt-1g"
            state = StateStore(Path(temporary))
            monitor = RankMonitor(
                state,
                track="extract",
                hardware_class="mt-3g",
                hotkey="5Mine",
                fetcher=lambda url, timeout: payload,
            )
            result = monitor.poll_once()
            self.assertFalse(result["reachability"])
            self.assertFalse(result["goal_achieved"])


if __name__ == "__main__":
    unittest.main()
