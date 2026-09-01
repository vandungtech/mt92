from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import ClassVar

from microtensor_miner_controller import rank_observer


class FakeMonitor:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, state: object, **kwargs: object) -> None:
        self.calls.append({"state": state, **kwargs})

    def poll_once(self) -> dict[str, object]:
        return {"competition": "code/mt-3g", "goal_achieved": False, "rank": None}


class RankObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeMonitor.calls.clear()

    def test_once_polls_exact_public_identity_and_prints_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = StringIO()
            with redirect_stdout(output):
                result = rank_observer.main(
                    [
                        "--track",
                        "code",
                        "--hardware-class",
                        "mt-3g",
                        "--hotkey",
                        "5Mine",
                        "--state-dir",
                        temporary,
                        "--poll-seconds",
                        "17",
                        "--once",
                    ],
                    monitor_factory=FakeMonitor,
                )

            self.assertEqual(result, 0)
            self.assertEqual(len(FakeMonitor.calls), 1)
            call = FakeMonitor.calls[0]
            self.assertEqual(call["track"], "code")
            self.assertEqual(call["hardware_class"], "mt-3g")
            self.assertEqual(call["hotkey"], "5Mine")
            self.assertEqual(call["poll_seconds"], 17)
            self.assertEqual(call["state"].root, Path(temporary).absolute())
            self.assertEqual(
                json.loads(output.getvalue()),
                {"competition": "code/mt-3g", "goal_achieved": False, "rank": None},
            )

    def test_poll_interval_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            rank_observer.main(
                [
                    "--track",
                    "code",
                    "--hardware-class",
                    "mt-3g",
                    "--hotkey",
                    "5Mine",
                    "--state-dir",
                    "unused-state",
                    "--poll-seconds",
                    "0",
                    "--once",
                ],
                monitor_factory=FakeMonitor,
            )
        self.assertEqual(FakeMonitor.calls, [])


if __name__ == "__main__":
    unittest.main()
