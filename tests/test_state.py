from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from microtensor_miner_controller.state import StateStore


class StateTests(unittest.TestCase):
    def test_verified_state_and_health_are_atomic_private_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary), ("top-secret",))
            store.mark_verified(
                round_index=4,
                message="verified token=top-secret",
                details={
                    "proofs": {
                        "source": True,
                        "source_full": True,
                        "provenance": True,
                        "on_chain": True,
                    }
                },
                now=100.0,
            )
            status = store.read_status()
            ok, health = store.health(10, now=105.0)
            self.assertTrue(ok)
            self.assertEqual(status["last_success_round"], 4)
            self.assertNotIn("top-secret", status["message"])
            self.assertTrue(health["ok"])
            self.assertEqual(os.stat(store.status_path).st_mode & 0o777, 0o600)

    def test_stale_health_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary))
            store.write("verified", ok=True, message="ok", now=100.0)
            ok, health = store.health(10, now=111.0)
            self.assertFalse(ok)
            self.assertIn("stale", health["message"])
