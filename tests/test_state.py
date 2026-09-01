from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from microtensor_miner_controller.errors import ControllerError
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

    def test_explicit_empty_preserve_discards_prior_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary))
            store.write(
                "current",
                ok=True,
                message="observed",
                details={"origin_head": "a" * 40, "review_required": False},
                now=100.0,
            )
            payload = store.write(
                "check_error",
                ok=False,
                message="failed",
                details={"observation_succeeded": False},
                preserve={},
                now=101.0,
            )
            self.assertNotIn("origin_head", payload)
            self.assertNotIn("review_required", payload)
            self.assertFalse(payload["observation_succeeded"])

    def test_stale_health_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary))
            store.write("verified", ok=True, message="ok", now=100.0)
            ok, health = store.health(10, now=111.0)
            self.assertFalse(ok)
            self.assertIn("stale", health["message"])

    def test_authorization_refusal_marker_is_private_and_preserves_first_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary), ("top-secret",))
            marker = store.latch_authorization_refusal(
                "fee top-secret was not authorized",
                details={"authorization_latched": False},
                now=100.0,
            )
            self.assertTrue(marker["authorization_latched"])
            self.assertNotIn("top-secret", marker["authorization_reason"])
            self.assertEqual(
                os.stat(store.authorization_path).st_mode & 0o777,
                0o600,
            )

            preserved = store.latch_authorization_refusal("different reason", now=200.0)
            self.assertEqual(preserved["latched_at_epoch"], 100.0)
            self.assertEqual(
                preserved["authorization_reason"],
                marker["authorization_reason"],
            )

    def test_submission_pending_marker_lifecycle_requires_exact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary))
            fingerprint = "sha256:" + ("a" * 64)
            marker = store.mark_submission_pending(
                round_index=7,
                source="s3:public/uid-32/round-7",
                hotkey="5Hotkey",
                commitment_fingerprint=fingerprint,
                details={"provenance_block": 150},
                now=100.0,
            )
            self.assertTrue(marker["submission_pending"])
            self.assertEqual(store.read_submission_pending(), marker)
            self.assertEqual(
                os.stat(store.submission_pending_path).st_mode & 0o777,
                0o600,
            )

            with self.assertRaises(ControllerError):
                store.clear_submission_pending("sha256:" + ("b" * 64))
            self.assertTrue(store.submission_pending_path.exists())

            store.clear_submission_pending(fingerprint)
            self.assertEqual(store.read_submission_pending(), {})
            self.assertFalse(store.submission_pending_path.exists())

    def test_corrupt_and_structurally_invalid_markers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary))
            store.ensure()

            store.authorization_path.write_text("{", encoding="utf-8")
            authorization = store.read_authorization_refusal()
            self.assertTrue(authorization["authorization_latched"])
            self.assertTrue(authorization["marker_invalid"])

            store.authorization_path.write_text(
                '{"schema_version": 1, "authorization_latched": false}',
                encoding="utf-8",
            )
            self.assertTrue(store.read_authorization_refusal()["marker_invalid"])

            store.submission_pending_path.write_text('{"x": 1}', encoding="utf-8")
            pending = store.read_submission_pending()
            self.assertTrue(pending["submission_pending"])
            self.assertTrue(pending["marker_invalid"])
            with self.assertRaises(ControllerError):
                store.clear_submission_pending("sha256:" + ("a" * 64))


if __name__ == "__main__":
    unittest.main()
