from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from helpers import base_env

from microtensor_miner_controller.artifact_competition import (
    MAX_ARTIFACT_COMPETITION_BINDING_BYTES,
    validate_artifact_competition_binding,
)
from microtensor_miner_controller.binding import artifact_digest
from microtensor_miner_controller.config import ControllerConfig
from microtensor_miner_controller.errors import ArtifactCompetitionBindingError


class ArtifactCompetitionBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata_patchers = []
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = ControllerConfig.from_env(base_env(self.root))
        self.config.artifact_dir.mkdir()
        (self.config.artifact_dir / "model.gguf").write_bytes(b"fixture-model")
        digest, _count, _total = artifact_digest(self.config.artifact_dir)
        self.payload: dict[str, object] = {
            "artifact_digest": digest,
            "hardware_class": "mt-3g",
            "schema_version": 1,
            "track": "extract",
        }
        self._write(json.dumps(self.payload, sort_keys=True).encode())

        if os.geteuid() != 0 or os.getegid() != 0:
            real_lstat = Path.lstat
            real_fstat = os.fstat

            def root_owned(metadata: os.stat_result) -> os.stat_result:
                fields = list(metadata)
                fields[4] = 0
                fields[5] = 0
                return os.stat_result(fields)

            self.metadata_patchers = [
                mock.patch.object(Path, "lstat", lambda path: root_owned(real_lstat(path))),
                mock.patch.object(
                    os,
                    "fstat",
                    lambda descriptor: root_owned(real_fstat(descriptor)),
                ),
            ]
            for patcher in self.metadata_patchers:
                patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.metadata_patchers):
            patcher.stop()
        self.temporary.cleanup()

    def _write(self, payload: bytes, *, mode: int = 0o600) -> None:
        self.config.artifact_competition_binding_path.write_bytes(payload)
        self.config.artifact_competition_binding_path.chmod(mode)

    def test_exact_binding_is_accepted(self) -> None:
        self.assertEqual(validate_artifact_competition_binding(self.config), self.payload)

    def test_missing_empty_oversized_and_non_utf8_bindings_are_refused(self) -> None:
        cases = (
            ("empty", b"", "must not be empty"),
            (
                "oversized",
                b"x" * (MAX_ARTIFACT_COMPETITION_BINDING_BYTES + 1),
                "4096-byte limit",
            ),
            ("non-UTF-8", b"\xff", "strict UTF-8"),
        )
        for label, payload, message in cases:
            with self.subTest(label=label):
                self._write(payload)
                with self.assertRaisesRegex(ArtifactCompetitionBindingError, message):
                    validate_artifact_competition_binding(self.config)

        self.config.artifact_competition_binding_path.unlink()
        with self.assertRaisesRegex(ArtifactCompetitionBindingError, "unavailable or unsafe"):
            validate_artifact_competition_binding(self.config)

    def test_strict_json_rejects_duplicate_nonfinite_and_nonobject_values(self) -> None:
        digest = self.payload["artifact_digest"]
        cases = (
            (
                "duplicate",
                (
                    '{"schema_version":1,"schema_version":1,'
                    f'"artifact_digest":"{digest}","track":"extract",'
                    '"hardware_class":"mt-3g"}'
                ).encode(),
            ),
            (
                "nonfinite",
                (
                    f'{{"schema_version":NaN,"artifact_digest":"{digest}",'
                    '"track":"extract","hardware_class":"mt-3g"}'
                ).encode(),
            ),
            ("nonobject", b"[]"),
        )
        for label, payload in cases:
            with self.subTest(label=label):
                self._write(payload)
                with self.assertRaisesRegex(ArtifactCompetitionBindingError, "strict JSON"):
                    validate_artifact_competition_binding(self.config)

    def test_extra_fields_bool_schema_and_noncanonical_digest_are_refused(self) -> None:
        cases = (
            ({**self.payload, "unexpected": True}, "missing or extra"),
            ({**self.payload, "schema_version": True}, "schema version"),
            (
                {
                    **self.payload,
                    "artifact_digest": str(self.payload["artifact_digest"]).upper(),
                },
                "non-canonical",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                self._write(json.dumps(payload, sort_keys=True).encode())
                with self.assertRaisesRegex(ArtifactCompetitionBindingError, message):
                    validate_artifact_competition_binding(self.config)

    def test_symlink_hardlink_and_mode_are_enforced(self) -> None:
        binding = self.config.artifact_competition_binding_path

        symlink = self.root / "binding-symlink.json"
        symlink.symlink_to(binding)
        with self.assertRaisesRegex(ArtifactCompetitionBindingError, "non-symlink"):
            validate_artifact_competition_binding(
                replace(self.config, artifact_competition_binding_path=symlink)
            )

        hardlink = self.root / "binding-hardlink.json"
        hardlink.hardlink_to(binding)
        with self.assertRaisesRegex(ArtifactCompetitionBindingError, "exactly one hard link"):
            validate_artifact_competition_binding(self.config)
        hardlink.unlink()

        binding.chmod(0o640)
        with self.assertRaisesRegex(ArtifactCompetitionBindingError, "exactly 0600"):
            validate_artifact_competition_binding(self.config)
        binding.chmod(0o600)

    @unittest.skipUnless(
        os.geteuid() == 0 and os.getegid() == 0,
        "exact root:root ownership mutation requires a root test process",
    )
    def test_exact_root_user_and_group_ownership_are_enforced(self) -> None:
        binding = self.config.artifact_competition_binding_path
        try:
            for uid, gid in ((1, 0), (0, 1)):
                with self.subTest(uid=uid, gid=gid):
                    os.chown(binding, uid, gid)
                    with self.assertRaisesRegex(
                        ArtifactCompetitionBindingError, "root:root"
                    ):
                        validate_artifact_competition_binding(self.config)
        finally:
            os.chown(binding, 0, 0)

    def test_competition_mismatch_refuses_before_artifact_tree_hashing(self) -> None:
        config = replace(self.config, track="code")
        with (
            mock.patch(
                "microtensor_miner_controller.artifact_competition.artifact_digest"
            ) as digest,
            self.assertRaisesRegex(
                ArtifactCompetitionBindingError,
                "targets extract/mt-3g.*targets code/mt-3g",
            ),
        ):
            validate_artifact_competition_binding(config)
        digest.assert_not_called()

    def test_current_artifact_digest_mismatch_is_refused(self) -> None:
        self.payload["artifact_digest"] = "sha256:" + "0" * 64
        self._write(json.dumps(self.payload, sort_keys=True).encode())
        with self.assertRaisesRegex(ArtifactCompetitionBindingError, "digest does not match"):
            validate_artifact_competition_binding(self.config)
