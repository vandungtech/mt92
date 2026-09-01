from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from helpers import base_env, v030_env

from microtensor_miner_controller.config import (
    AUTHORIZED_HOTKEY_SS58,
    SIGNED_V030_COORDINATOR_URL,
    ControllerConfig,
)
from microtensor_miner_controller.errors import (
    AuthorizationRefused,
    PreflightError,
    VerificationError,
)
from microtensor_miner_controller.models import PackagedArtifact
from microtensor_miner_controller.upstream_gate import (
    AUDITED_MECHANISM_VERSION,
    AUDITED_ORIGIN_HEAD,
    AUDITED_RELEASE_VERSION,
    OBSERVER_SCHEMA,
    UpstreamGateError,
    verify_upstream_observer_status,
)


def _valid_status(now: float) -> dict[str, object]:
    return {
        "schema": OBSERVER_SCHEMA,
        "schema_version": 1,
        "updated_at_epoch": now,
        "origin_observed_at": now,
        "observation_succeeded": True,
        "phase": "current",
        "ok": True,
        "origin": "https://github.com/microtensor-io/microtensor-subnet",
        "origin_head": AUDITED_ORIGIN_HEAD,
        "audited_origin_head": AUDITED_ORIGIN_HEAD,
        "release_version": AUDITED_RELEASE_VERSION,
        "mechanism_version": AUDITED_MECHANISM_VERSION,
        "provenance_required": False,
        "local_checkout_at_origin": True,
        "audited_head_is_ancestor": True,
        "review_required": False,
        "miner_impact_review_required": False,
        "history_rewrite_detected": False,
        "history_rewrite_latched": False,
        "changed_files": [],
        "changed_files_truncated": False,
        "commits_since_audit": 0,
    }


def _write_status(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


class UpstreamGateTests(unittest.TestCase):
    def test_gate_is_pinned_to_exact_compatibility_reviewed_head(self) -> None:
        self.assertEqual(
            AUDITED_ORIGIN_HEAD,
            "d77adc945de763f8b3b2d71fef8193090ede7001",
        )

    def test_accepts_exact_current_fresh_audited_status(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            payload = _valid_status(now)
            _write_status(path, payload)
            observed = verify_upstream_observer_status(path, now=now)
        self.assertEqual(observed, payload)

    def test_rejects_stale_future_and_non_numeric_timestamps(self) -> None:
        now = 2_000_000_000.0
        cases = (
            ("updated_at_epoch", now - 901, "stale"),
            ("origin_observed_at", now - 901, "stale"),
            ("updated_at_epoch", now + 31, "future"),
            ("origin_observed_at", now + 31, "future"),
            ("updated_at_epoch", True, "invalid"),
            ("origin_observed_at", "2000000000", "invalid"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "status.json"
                payload = _valid_status(now)
                payload[key] = value
                _write_status(path, payload)
                with self.assertRaisesRegex(UpstreamGateError, message):
                    verify_upstream_observer_status(path, now=now)

    def test_rejects_drift_review_latches_and_error_without_stale_verdict(self) -> None:
        now = time.time()
        cases: tuple[tuple[str, dict[str, object]], ...] = (
            ("origin_head", {"origin_head": "a" * 40}),
            ("review_required", {"review_required": True}),
            ("miner_impact_review_required", {"miner_impact_review_required": True}),
            ("history_rewrite_latched", {"history_rewrite_latched": True}),
            (
                "observation_succeeded",
                {
                    "observation_succeeded": False,
                    "phase": "check_error",
                    "ok": False,
                },
            ),
        )
        for message, changes in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "status.json"
                payload = _valid_status(now)
                payload.update(changes)
                _write_status(path, payload)
                with self.assertRaisesRegex(UpstreamGateError, message):
                    verify_upstream_observer_status(path, now=now)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            _write_status(
                path,
                {
                    "schema": OBSERVER_SCHEMA,
                    "schema_version": 1,
                    "observation_succeeded": False,
                    "phase": "check_error",
                    "ok": False,
                    "updated_at_epoch": now,
                },
            )
            with self.assertRaisesRegex(UpstreamGateError, "observation_succeeded"):
                verify_upstream_observer_status(path, now=now)

    def test_rejects_missing_or_wrongly_typed_required_fields(self) -> None:
        now = time.time()
        cases: tuple[tuple[str, object], ...] = (
            ("schema_version", True),
            ("ok", 1),
            ("local_checkout_at_origin", 1),
            ("release_version", 0.3),
            ("changed_files", "not-a-list"),
        )
        for key, wrong in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "status.json"
                payload = _valid_status(now)
                payload[key] = wrong
                _write_status(path, payload)
                with self.assertRaisesRegex(UpstreamGateError, key):
                    verify_upstream_observer_status(path, now=now)

            with self.subTest(key=key, missing=True), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "status.json"
                payload = _valid_status(now)
                del payload[key]
                _write_status(path, payload)
                with self.assertRaisesRegex(UpstreamGateError, key):
                    verify_upstream_observer_status(path, now=now)

    def test_rejects_link_mode_owner_size_and_mutation(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real.json"
            link = root / "status.json"
            _write_status(real, _valid_status(now))
            link.symlink_to(real)
            with self.assertRaisesRegex(UpstreamGateError, "regular non-symlink"):
                verify_upstream_observer_status(link, now=now)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            _write_status(path, _valid_status(now))
            path.chmod(0o640)
            with self.assertRaisesRegex(UpstreamGateError, "exactly 0600"):
                verify_upstream_observer_status(path, now=now)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "status.json"
            second = root / "second.json"
            _write_status(path, _valid_status(now))
            os.link(path, second)
            with self.assertRaisesRegex(UpstreamGateError, "exactly one hard link"):
                verify_upstream_observer_status(path, now=now)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            _write_status(path, _valid_status(now))
            with (
                mock.patch(
                    "microtensor_miner_controller.upstream_gate.os.geteuid",
                    return_value=os.geteuid() + 1,
                ),
                self.assertRaisesRegex(UpstreamGateError, "effective user"),
            ):
                verify_upstream_observer_status(path, now=now)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            path.write_bytes(b" " * (64 * 1024 + 1))
            path.chmod(0o600)
            with self.assertRaisesRegex(UpstreamGateError, "64 KiB"):
                verify_upstream_observer_status(path, now=now)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            _write_status(path, _valid_status(now))
            real_fstat = os.fstat
            calls = 0

            def changed_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
                nonlocal calls
                calls += 1
                result = real_fstat(descriptor)
                if calls != 2:
                    return result
                values = {
                    name: getattr(result, name)
                    for name in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_uid",
                        "st_gid",
                        "st_nlink",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
                values["st_mtime_ns"] += 1
                return SimpleNamespace(**values)

            with (
                mock.patch(
                    "microtensor_miner_controller.upstream_gate.os.fstat",
                    side_effect=changed_fstat,
                ),
                self.assertRaisesRegex(UpstreamGateError, "changed while it was read"),
            ):
                verify_upstream_observer_status(path, now=now)

    def test_rejects_non_utf8_duplicate_keys_and_non_mapping_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            path.write_bytes(b"\xff")
            path.chmod(0o600)
            with self.assertRaisesRegex(UpstreamGateError, "UTF-8"):
                verify_upstream_observer_status(path)

        for raw, message in ((b'{"ok":true,"ok":true}', "strict JSON"), (b"[]", "mapping")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "status.json"
                path.write_bytes(raw)
                path.chmod(0o600)
                with self.assertRaisesRegex(UpstreamGateError, message):
                    verify_upstream_observer_status(path)

    def test_direct_gate_call_cannot_relax_freshness_bounds(self) -> None:
        for max_age in (599, 901, True):
            with (
                self.subTest(max_age=max_age),
                self.assertRaisesRegex(UpstreamGateError, "between 600 and 900"),
            ):
                verify_upstream_observer_status(
                    Path("/status-must-not-be-read.json"),
                    max_age_seconds=max_age,
                )

    def test_backend_rechecks_gate_at_preflight_registration_and_live_policy(self) -> None:
        from microtensor_miner_controller.backend import MicrotensorBackend

        with tempfile.TemporaryDirectory() as temporary:
            backend = MicrotensorBackend(ControllerConfig.from_env(v030_env(Path(temporary))))
            with (
                mock.patch.object(backend, "_verify_upstream_observer_status") as gate,
                mock.patch.object(
                    backend,
                    "_verify_transaction_dependencies",
                    side_effect=PreflightError("stop after gate"),
                ),
                self.assertRaisesRegex(PreflightError, "stop after gate"),
            ):
                backend.preflight()
            gate.assert_called_once_with()

            backend._hotkey = AUTHORIZED_HOTKEY_SS58
            backend._client = SimpleNamespace(
                snapshot=lambda refresh: SimpleNamespace(
                    uid_of=lambda hotkey: 32 if hotkey == AUTHORIZED_HOTKEY_SS58 else None
                )
            )
            with mock.patch.object(backend, "_verify_upstream_observer_status") as gate:
                backend.assert_registered()
            gate.assert_called_once_with()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = v030_env(root, dry_run=False)
            env.update(
                {
                    "MMC_COORDINATOR_URL": SIGNED_V030_COORDINATOR_URL,
                    "MMC_SOURCE_TEMPLATE": (
                        "https:github.com/vandungtech/mt92/releases/download/r{round}"
                    ),
                    "MMC_GITHUB_TOKEN_FILE": str(root / "github.token"),
                }
            )
            backend = MicrotensorBackend(ControllerConfig.from_env(env))
            backend._hotkey = AUTHORIZED_HOTKEY_SS58
            with (
                mock.patch.object(backend, "_verify_upstream_observer_status") as gate,
                mock.patch.object(backend, "_verify_transaction_dependencies"),
            ):
                backend._assert_live_transaction_policy()
            gate.assert_called_once_with()

            with (
                mock.patch.object(
                    backend,
                    "_verify_upstream_observer_status",
                    side_effect=UpstreamGateError("drift"),
                ),
                self.assertRaisesRegex(AuthorizationRefused, "observer gate failed: drift"),
            ):
                backend._assert_live_transaction_policy()

            packaged = PackagedArtifact(
                round_index=7,
                source="https:github.com/vandungtech/mt92/releases/download/r7",
                hotkey=AUTHORIZED_HOTKEY_SS58,
                manifest_digest="sha256:" + "1" * 64,
                artifact_digest="sha256:" + "2" * 64,
                file_count=1,
                total_bytes=1,
            )
            with (
                mock.patch.object(backend, "_verify_upstream_observer_status") as gate,
                mock.patch.object(backend, "_upload_github") as upload,
            ):
                backend.upload(packaged)
            gate.assert_called_once_with()
            upload.assert_called_once_with(packaged)

            with (
                mock.patch.object(
                    backend,
                    "_verify_upstream_observer_status",
                    side_effect=UpstreamGateError("upload drift"),
                ),
                mock.patch.object(backend, "_upload_github") as upload,
                self.assertRaisesRegex(VerificationError, "observer gate failed: upload drift"),
            ):
                backend.upload(packaged)
            upload.assert_not_called()

    def test_registration_converts_gate_failure_to_verification_error(self) -> None:
        from microtensor_miner_controller.backend import MicrotensorBackend

        with tempfile.TemporaryDirectory() as temporary:
            backend = MicrotensorBackend(ControllerConfig.from_env(v030_env(Path(temporary))))
            with (
                mock.patch.object(
                    backend,
                    "_verify_upstream_observer_status",
                    side_effect=UpstreamGateError("stale"),
                ),
                self.assertRaisesRegex(VerificationError, "observer gate failed: stale"),
            ):
                backend.assert_registered()

    def test_legacy_preflight_does_not_require_observer_status(self) -> None:
        from microtensor_miner_controller.backend import MicrotensorBackend

        with tempfile.TemporaryDirectory() as temporary:
            backend = MicrotensorBackend(ControllerConfig.from_env(base_env(Path(temporary))))
            with (
                mock.patch.object(backend, "_verify_upstream_observer_status") as gate,
                mock.patch.object(
                    backend,
                    "_verify_transaction_dependencies",
                    side_effect=PreflightError("stop"),
                ),
                self.assertRaisesRegex(PreflightError, "stop"),
            ):
                backend.preflight()
            gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
