from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from microtensor_miner_controller.config import ControllerConfig, UPSTREAM_COMMIT
from microtensor_miner_controller.errors import ConfigError

from helpers import base_env


class ConfigTests(unittest.TestCase):
    def test_expected_identity_and_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = ControllerConfig.from_env(base_env(Path(temporary)))
        self.assertEqual(config.netuid, 92)
        self.assertEqual(config.wallet_name, "you-cold")
        self.assertEqual(config.wallet_hotkey, "you-hot1")
        self.assertEqual(config.expected_uid, 32)
        self.assertEqual(config.source_for(41, "5Hotkey"), "s3:public/uid-32/round-41")
        self.assertEqual(len(UPSTREAM_COMMIT), 40)

    def test_source_requires_round_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_SOURCE_TEMPLATE"] = "s3:bucket/static"
            with self.assertRaisesRegex(ConfigError, "must contain"):
                ControllerConfig.from_env(env)

    def test_live_cannot_bypass_upstream_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary), dry_run=False)
            env["MMC_ALLOW_UNVERIFIED_UPSTREAM"] = "true"
            with self.assertRaisesRegex(ConfigError, "only with MMC_DRY_RUN"):
                ControllerConfig.from_env(env)

    def test_hf_is_refused_for_automatic_round_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_SOURCE_TEMPLATE"] = "hf:org/repo@{round}"
            with self.assertRaisesRegex(ConfigError, "only per-round s3"):
                ControllerConfig.from_env(env)

    def test_identity_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_EXPECTED_UID"] = "33"
            with self.assertRaisesRegex(ConfigError, "must remain 32"):
                ControllerConfig.from_env(env)
