from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from microtensor_miner_controller.config import ControllerConfig, UPSTREAM_COMMIT, UPSTREAM_RELEASE
from microtensor_miner_controller.controller import Controller
from microtensor_miner_controller.errors import VerificationError
from microtensor_miner_controller.models import (
    PackagedArtifact,
    PreflightSnapshot,
    PublishReceipt,
    RoundWindow,
)
from microtensor_miner_controller.state import StateStore

from helpers import base_env, coordinator_payload


class FakeBackend:
    def __init__(self, *, fail_source: bool = False) -> None:
        self.operations: list[str] = []
        self.published = False
        self.fail_source = fail_source
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

    def publish(self, packaged: PackagedArtifact) -> PublishReceipt:
        self.operations.append("publish")
        self.published = True
        return PublishReceipt(packaged.round_index, "mt1|payload")

    def verify_on_chain(self, packaged: PackagedArtifact) -> str:
        self.operations.append("on_chain")
        if not self.published:
            raise VerificationError("not present")
        return "mt1|payload"

    def close(self) -> None:
        self.operations.append("close")


class ControllerTests(unittest.TestCase):
    def _build(self, root: Path, *, dry_run: bool, backend: FakeBackend) -> tuple[Controller, StateStore]:
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
            self.assertLess(backend.operations.index("validate_commitment"), backend.operations.index("upload"))
            self.assertLess(backend.operations.index("provenance"), backend.operations.index("upload"))
            self.assertLess(backend.operations.index("verify_source_full"), backend.operations.index("publish"))
            self.assertLess(backend.operations.index("provenance"), backend.operations.index("publish"))
            self.assertGreater(backend.operations.count("on_chain"), 1)

    def test_source_failure_never_publishes_or_claims_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(fail_source=True)
            controller, state = self._build(Path(temporary), dry_run=False, backend=backend)
            self.assertEqual(controller.run(once=True), 2)
            self.assertNotIn("publish", backend.operations)
            self.assertFalse(state.read_health()["ok"])
            self.assertNotEqual(state.read_status()["phase"], "verified")


if __name__ == "__main__":
    unittest.main()
