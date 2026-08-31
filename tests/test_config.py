from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import base_env, v030_env

from microtensor_miner_controller.config import (
    SIGNED_V030_MECHANISM_VERSION,
    SIGNED_V030_RELEASE,
    SIGNED_V030_RELEASE_SIGNING_KEY,
    SIGNED_V030_WHEEL_SHA256,
    TRANSACTION_AUTHORIZATION,
    UPSTREAM_COMMIT,
    ControllerConfig,
)
from microtensor_miner_controller.errors import ConfigError


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

    def test_signed_v030_identity_and_explicit_activation_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = ControllerConfig.from_env(v030_env(Path(temporary)))
        self.assertTrue(config.uses_signed_v030)
        self.assertEqual(config.v030_activation_block, 100)
        self.assertEqual(config.upstream_release, SIGNED_V030_RELEASE)
        self.assertEqual(SIGNED_V030_MECHANISM_VERSION, "0.3.0")
        self.assertEqual(len(SIGNED_V030_WHEEL_SHA256), 64)
        self.assertEqual(len(SIGNED_V030_RELEASE_SIGNING_KEY), 66)
        self.assertEqual((config.track, config.hardware_class), ("code", "mt-3g"))

    def test_code_profile_without_activation_block_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_TRACK"] = "code"
            with self.assertRaisesRegex(ConfigError, "ACTIVATION_BLOCK"):
                ControllerConfig.from_env(env)

    def test_v030_activation_does_not_select_a_model_or_load_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_V030_ACTIVATION_BLOCK"] = "100"
            env["MMC_TRACK"] = "code"
            for key in (
                "MMC_ENTRYPOINT",
                "MMC_ARTIFACT_FORMAT",
                "MMC_QUANTIZATION",
                "MMC_MAX_INPUT_TOKENS",
                "MMC_TOKENIZER",
                "MMC_BASE_MODEL",
            ):
                env.pop(key)
            with self.assertRaisesRegex(ConfigError, "explicit final model/load spec"):
                ControllerConfig.from_env(env)

    def test_live_v030_requires_exact_official_coordinator_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = v030_env(Path(temporary), dry_run=False)
            env["MMC_COORDINATOR_URL"] = "https://coordinator.example"
            with self.assertRaisesRegex(ConfigError, "official coordinator URL"):
                ControllerConfig.from_env(env)

    def test_v030_never_allows_unverified_upstream_or_schedule_fallback(self) -> None:
        for key in ("MMC_ALLOW_UNVERIFIED_UPSTREAM", "MMC_ALLOW_CHAIN_SCHEDULE_FALLBACK"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                env = v030_env(Path(temporary))
                env[key] = "true"
                with self.assertRaisesRegex(ConfigError, "signed v0.3 activation never"):
                    ControllerConfig.from_env(env)

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
            with self.assertRaisesRegex(ConfigError, "supervised uploads"):
                ControllerConfig.from_env(env)

    def test_dry_run_github_source_may_omit_token_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_SOURCE_TEMPLATE"] = (
                "https:github.com/vandungtech/mt92/releases/download/r{round}"
            )
            config = ControllerConfig.from_env(env)
        self.assertIsNone(config.github_token_file)
        self.assertEqual(
            config.source_for(41, "5Hotkey"),
            "https:github.com/vandungtech/mt92/releases/download/r41",
        )
        self.assertEqual(
            config.github_release_coordinates(config.source_for(41, "5Hotkey")),
            ("vandungtech", "mt92", "r41"),
        )

    def test_live_github_source_requires_token_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = base_env(root, dry_run=False)
            env["MMC_SOURCE_TEMPLATE"] = (
                "https:github.com/vandungtech/mt92/releases/download/r{round}"
            )
            with self.assertRaisesRegex(ConfigError, "MMC_GITHUB_TOKEN_FILE"):
                ControllerConfig.from_env(env)
            env["MMC_GITHUB_TOKEN_FILE"] = str(root / "github.token")
            config = ControllerConfig.from_env(env)
        self.assertEqual(config.github_token_file, root / "github.token")

    def test_live_s3_and_r2_sources_remain_refused(self) -> None:
        for scheme in ("s3", "r2"):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as temporary:
                env = base_env(Path(temporary), dry_run=False)
                env["MMC_SOURCE_TEMPLATE"] = f"{scheme}:public/uid-{{uid}}/round-{{round}}"
                with self.assertRaisesRegex(ConfigError, "live S3/R2 activation is refused"):
                    ControllerConfig.from_env(env)

    def test_live_requires_exact_transaction_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = base_env(root, dry_run=False)
            env["MMC_SOURCE_TEMPLATE"] = (
                "https:github.com/vandungtech/mt92/releases/download/r{round}"
            )
            env["MMC_GITHUB_TOKEN_FILE"] = str(root / "github.token")
            env.pop("MMC_TRANSACTION_AUTHORIZATION")
            with self.assertRaisesRegex(ConfigError, "exact zero-cost"):
                ControllerConfig.from_env(env)
            env["MMC_TRANSACTION_AUTHORIZATION"] = TRANSACTION_AUTHORIZATION
            self.assertFalse(ControllerConfig.from_env(env).dry_run)

    def test_other_github_repository_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_SOURCE_TEMPLATE"] = "https:github.com/example/other/releases/download/r{round}"
            with self.assertRaisesRegex(ConfigError, "authorized vandungtech/mt92"):
                ControllerConfig.from_env(env)

    def test_invalid_github_release_tag_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_SOURCE_TEMPLATE"] = (
                "https:github.com/vandungtech/mt92/releases/download/r{round}.lock"
            )
            with self.assertRaisesRegex(ConfigError, "GitHub release source is invalid"):
                ControllerConfig.from_env(env)

    def test_token_file_setting_is_valid_only_for_github_https(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_GITHUB_TOKEN_FILE"] = str(Path(temporary) / "github.token")
            with self.assertRaisesRegex(ConfigError, "valid only with"):
                ControllerConfig.from_env(env)

    def test_arbitrary_https_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_SOURCE_TEMPLATE"] = "https:artifacts.example/mt92/releases/download/r{round}"
            with self.assertRaisesRegex(ConfigError, "must be github.com"):
                ControllerConfig.from_env(env)

    def test_identity_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = base_env(Path(temporary))
            env["MMC_EXPECTED_UID"] = "33"
            with self.assertRaisesRegex(ConfigError, "must remain 32"):
                ControllerConfig.from_env(env)
