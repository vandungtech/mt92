from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from helpers import base_env, coordinator_payload

from microtensor_miner_controller.config import UPSTREAM_COMMIT, UPSTREAM_RELEASE, ControllerConfig
from microtensor_miner_controller.controller import Controller
from microtensor_miner_controller.errors import AuthorizationRefused, VerificationError
from microtensor_miner_controller.models import (
    PackagedArtifact,
    PreflightSnapshot,
    PublishReceipt,
    RoundWindow,
)
from microtensor_miner_controller.state import StateStore


class FakeBackend:
    def __init__(
        self,
        *,
        fail_source: bool = False,
        refuse_authorization: bool = False,
        publish_error: Exception | None = None,
        registration_error: Exception | None = None,
        onchain_payload: str = "mt1|payload",
    ) -> None:
        self.operations: list[str] = []
        self.published = False
        self.fail_source = fail_source
        self.refuse_authorization = refuse_authorization
        self.publish_error = publish_error
        self.registration_error = registration_error
        self.onchain_payload = onchain_payload
        self.provenance_blocks: list[int] = []
        self.local: PackagedArtifact | None = None

    def preflight(self) -> PreflightSnapshot:
        self.operations.append("preflight")
        return PreflightSnapshot("5Hotkey", 32, 150, UPSTREAM_RELEASE, UPSTREAM_COMMIT)

    def block(self) -> int:
        self.operations.append("block")
        return 150

    def chain_round(self, block: int) -> RoundWindow:
        self.operations.append("chain_round")
        return RoundWindow(7, 100, 200, 300, "submissions", "chain")

    def verify_round_anchor(self, window: RoundWindow) -> None:
        self.operations.append("anchor")

    def validate_round(self, window: RoundWindow) -> None:
        self.operations.append("arena")

    def assert_registered(self) -> None:
        self.operations.append("registered")
        if self.registration_error is not None:
            raise self.registration_error

    def load_local(self) -> PackagedArtifact | None:
        self.operations.append("load_local")
        return self.local

    def package(self, round_index: int, source: str) -> PackagedArtifact:
        self.operations.append("package")
        self.local = PackagedArtifact(
            round_index, source, "5Hotkey", "sha256:manifest", "sha256:artifact", 2, 10
        )
        return self.local

    def upload(self, packaged: PackagedArtifact) -> None:
        self.operations.append("upload")

    def validate_commitment(self, packaged: PackagedArtifact) -> str:
        self.operations.append("validate_commitment")
        return "mt1|payload"

    def verify_source(self, packaged: PackagedArtifact, *, full: bool) -> None:
        self.operations.append("verify_source_full" if full else "verify_source_manifest")
        if self.fail_source:
            raise VerificationError("remote mismatch")

    def verify_provenance(self, packaged: PackagedArtifact, block: int) -> None:
        self.operations.append("provenance")
        self.provenance_blocks.append(block)

    def publish(self, packaged: PackagedArtifact) -> PublishReceipt:
        self.operations.append("publish")
        if self.refuse_authorization:
            raise AuthorizationRefused("estimated transaction fee is 1 rao")
        if self.publish_error is not None:
            raise self.publish_error
        self.published = True
        return PublishReceipt(packaged.round_index, "mt1|payload")

    def verify_on_chain(self, packaged: PackagedArtifact) -> str:
        self.operations.append("on_chain")
        if not self.published:
            raise VerificationError("not present")
        return self.onchain_payload

    def close(self) -> None:
        self.operations.append("close")


class ControllerTests(unittest.TestCase):
    def _build(
        self, root: Path, *, dry_run: bool, backend: FakeBackend
    ) -> tuple[Controller, StateStore]:
        config = ControllerConfig.from_env(base_env(root, dry_run=True))
        config = replace(config, dry_run=dry_run)
        state = StateStore(config.state_dir)
        payload = coordinator_payload()
        controller = Controller(
            config,
            backend,
            state,
            clock=lambda: 1_000.0,
            round_fetcher=lambda url, timeout: dict(payload),
        )
        return controller, state

    def test_dry_run_does_not_mutate_or_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            controller, state = self._build(Path(temporary), dry_run=True, backend=backend)
            self.assertEqual(controller.run(once=True), 0)
            self.assertEqual(state.read_status()["phase"], "dry_run")
            self.assertFalse(state.read_health()["ok"])
            self.assertNotIn("package", backend.operations)
            self.assertNotIn("publish", backend.operations)

    def test_live_success_requires_all_proofs_and_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 0)
            status = state.read_status()
            self.assertTrue(status["ok"])
            self.assertEqual(status["phase"], "verified")
            self.assertTrue(all(status["proofs"].values()))
            self.assertLess(
                backend.operations.index("validate_commitment"), backend.operations.index("upload")
            )
            self.assertLess(
                backend.operations.index("provenance"), backend.operations.index("upload")
            )
            self.assertLess(
                backend.operations.index("verify_source_full"), backend.operations.index("publish")
            )
            self.assertLess(
                backend.operations.index("provenance"), backend.operations.index("publish")
            )
            self.assertGreater(backend.operations.count("on_chain"), 1)

    def test_source_failure_never_publishes_or_claims_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(fail_source=True)
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 2)
            self.assertNotIn("publish", backend.operations)
            self.assertFalse(state.read_health()["ok"])
            self.assertNotEqual(state.read_status()["phase"], "verified")

    def test_periodic_provenance_uses_validator_close_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            controller, _ = self._build(Path(temporary), dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 0)
            snapshot = controller.preflight_snapshot
            self.assertIsNotNone(snapshot)
            controller.clock = lambda: 2_000.0
            self.assertEqual(controller.cycle(snapshot), "verified")  # type: ignore[arg-type]
            self.assertEqual(backend.provenance_blocks, [150, 200])

    def test_authorization_refusal_latches_and_prevents_restart_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend(refuse_authorization=True)
            controller, state = self._build(root, dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 3)
            status = state.read_status()
            self.assertEqual(status["phase"], "authorization_refused")
            self.assertTrue(status["authorization_latched"])
            self.assertEqual(backend.operations.count("publish"), 1)

            restarted_backend = FakeBackend()
            restarted, _ = self._build(root, dry_run=False, backend=restarted_backend)
            self.assertEqual(restarted.run(once=True), 3)
            self.assertNotIn("preflight", restarted_backend.operations)
            self.assertNotIn("publish", restarted_backend.operations)

    def test_ambiguous_publish_leaves_pending_marker_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(publish_error=RuntimeError("connection dropped"))
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 3)
            self.assertEqual(backend.operations.count("publish"), 1)
            self.assertTrue(state.read_submission_pending()["submission_pending"])
            self.assertTrue(state.read_authorization_refusal()["authorization_latched"])

    def test_pending_submission_blocks_resubmit_when_readback_is_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            source = controller.config.source_for(7, "5Hotkey")
            backend.local = PackagedArtifact(
                7, source, "5Hotkey", "sha256:manifest", "sha256:artifact", 2, 10
            )
            fingerprint = controller._fingerprint("mt1|payload")
            state.mark_submission_pending(
                round_index=7,
                source=source,
                hotkey="5Hotkey",
                commitment_fingerprint=fingerprint,
                details={"provenance_block": 150},
                now=900.0,
            )

            self.assertEqual(controller.run(once=True), 3)
            self.assertNotIn("publish", backend.operations)
            self.assertTrue(state.read_submission_pending()["submission_pending"])
            self.assertTrue(state.read_authorization_refusal()["authorization_latched"])

    def test_pending_exact_readback_clears_marker_without_resubmitting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            backend.published = True
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            source = controller.config.source_for(7, "5Hotkey")
            backend.local = PackagedArtifact(
                7, source, "5Hotkey", "sha256:manifest", "sha256:artifact", 2, 10
            )
            fingerprint = controller._fingerprint("mt1|payload")
            state.mark_submission_pending(
                round_index=7,
                source=source,
                hotkey="5Hotkey",
                commitment_fingerprint=fingerprint,
                details={"provenance_block": 150},
                now=900.0,
            )

            self.assertEqual(controller.run(once=True), 0)
            self.assertNotIn("publish", backend.operations)
            self.assertEqual(state.read_submission_pending(), {})
            self.assertEqual(state.read_status()["phase"], "verified")
            self.assertIn("on_chain", backend.operations)

    def test_registration_refusal_stops_before_pending_marker_or_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(
                registration_error=AuthorizationRefused("hotkey is no longer registered")
            )
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 3)
            self.assertNotIn("publish", backend.operations)
            self.assertEqual(state.read_submission_pending(), {})
            self.assertTrue(state.read_authorization_refusal()["authorization_latched"])

    def test_authorization_exit_code_survives_both_persistence_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(registration_error=AuthorizationRefused("registration changed"))
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            real_write = state.write

            def fail_only_authorization_status(
                phase: str,
                **kwargs: object,
            ) -> dict[str, object]:
                if phase == "authorization_refused":
                    raise OSError("status filesystem unavailable")
                return real_write(phase, **kwargs)

            with (
                mock.patch.object(
                    state,
                    "latch_authorization_refusal",
                    side_effect=OSError("latch filesystem unavailable"),
                ),
                mock.patch.object(
                    state,
                    "write",
                    side_effect=fail_only_authorization_status,
                ),
            ):
                self.assertEqual(controller.run(once=True), 3)
            self.assertNotIn("publish", backend.operations)

    def test_corrupt_authorization_marker_stops_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            state.ensure()
            state.authorization_path.write_text('{"x": 1}', encoding="utf-8")

            self.assertEqual(controller.run(once=True), 3)
            self.assertNotIn("preflight", backend.operations)
            self.assertNotIn("publish", backend.operations)


if __name__ == "__main__":
    unittest.main()
