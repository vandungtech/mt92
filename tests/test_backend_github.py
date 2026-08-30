from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from microtensor_miner_controller.backend import MicrotensorBackend
from microtensor_miner_controller.config import ControllerConfig, UPSTREAM_COMMIT, UPSTREAM_RELEASE
from microtensor_miner_controller.errors import PreflightError, VerificationError
from microtensor_miner_controller.github_release import ReleasePublishError
from microtensor_miner_controller.models import PackagedArtifact

from helpers import base_env

TOKEN = "github_pat_backend_test_secret"  # noqa: S105
SOURCE_TEMPLATE = "https:github.com/miner-artifacts/public/releases/download/r{round}"


class GitHubBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.token_file = self.root / "github.token"
        self.token_file.write_text(TOKEN + "\n", encoding="ascii")
        self.token_file.chmod(0o600)
        artifact_dir = self.root / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "model.gguf").write_bytes(b"model")
        (artifact_dir / "manifest.json").write_bytes(b"manifest")

        env = base_env(self.root, dry_run=False)
        env["MMC_SOURCE_TEMPLATE"] = SOURCE_TEMPLATE
        env["MMC_GITHUB_TOKEN_FILE"] = str(self.token_file)
        self.config = ControllerConfig.from_env(env)
        self.source = self.config.source_for(7, "5Hotkey")
        self.native = object()
        self.packaged = PackagedArtifact(
            round_index=7,
            source=self.source,
            hotkey="5Hotkey",
            manifest_digest="sha256:manifest",
            artifact_digest="sha256:artifact",
            file_count=2,
            total_bytes=13,
            native=self.native,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_secure_token_file_accepts_optional_final_newline(self) -> None:
        for suffix in (b"", b"\n"):
            with self.subTest(final_newline=bool(suffix)):
                self.token_file.write_bytes(TOKEN.encode("ascii") + suffix)
                self.token_file.chmod(0o600)
                backend = MicrotensorBackend(self.config)

                self.assertEqual(backend._read_github_token(), TOKEN)

    def test_malformed_token_file_content_is_refused_without_echo(self) -> None:
        cases = (
            ("empty", b"", "invalid token"),
            ("crlf", TOKEN.encode("ascii") + b"\r\n", "exactly one token"),
            ("multiple", TOKEN.encode("ascii") + b"\nsecond", "exactly one token"),
            ("oversized", b"x" * 4097, "size limit"),
            ("non-ascii", b"\xff" * 8, "ASCII"),
            ("whitespace", b" " + TOKEN.encode("ascii"), "invalid token"),
        )
        for label, payload, message in cases:
            with self.subTest(label=label):
                self.token_file.write_bytes(payload)
                self.token_file.chmod(0o600)
                with self.assertRaisesRegex(PreflightError, message) as raised:
                    MicrotensorBackend(self.config)._read_github_token()
                self.assertNotIn(TOKEN, str(raised.exception))

    def test_missing_and_nonregular_token_paths_are_refused(self) -> None:
        directory = self.root / "token-directory"
        directory.mkdir()
        missing = self.root / "missing.token"
        for path, message in (
            (directory, "regular non-symlink"),
            (missing, "unavailable or unsafe"),
        ):
            with self.subTest(path=path.name):
                config = replace(self.config, github_token_file=path)
                with self.assertRaisesRegex(PreflightError, message):
                    MicrotensorBackend(config)._read_github_token()

    def test_token_file_must_be_exact_mode_0600(self) -> None:
        for mode in (0o400, 0o640, 0o644):
            with self.subTest(mode=oct(mode)):
                self.token_file.chmod(mode)
                backend = MicrotensorBackend(self.config)
                with self.assertRaisesRegex(PreflightError, "exactly 0600"):
                    backend._read_github_token()

    def test_token_file_symlink_is_refused(self) -> None:
        link = self.root / "github-link.token"
        link.symlink_to(self.token_file)
        config = replace(self.config, github_token_file=link)

        with self.assertRaisesRegex(PreflightError, "non-symlink"):
            MicrotensorBackend(config)._read_github_token()

    def test_token_file_hard_link_is_refused(self) -> None:
        link = self.root / "github-hard-link.token"
        link.hardlink_to(self.token_file)

        with self.assertRaisesRegex(PreflightError, "exactly one hard link"):
            MicrotensorBackend(self.config)._read_github_token()

    def test_live_preflight_checks_token_before_wallet_or_network(self) -> None:
        self.token_file.chmod(0o640)
        backend = MicrotensorBackend(self.config)

        with (
            patch.object(
                backend,
                "_verify_upstream",
                return_value=(UPSTREAM_COMMIT, UPSTREAM_RELEASE),
            ),
            self.assertRaisesRegex(PreflightError, "exactly 0600"),
        ):
            backend.preflight()

    def test_upload_dispatches_exact_publishable_files_to_github(self) -> None:
        backend = MicrotensorBackend(self.config)
        publisher = Mock()
        publisher.publish.return_value = SimpleNamespace(
            source=self.source,
            assets=(
                SimpleNamespace(name="model.gguf"),
                SimpleNamespace(name="manifest.json"),
            ),
        )

        with (
            patch(
                "microtensor_miner_controller.backend._publishable_files",
                return_value=["model.gguf", "manifest.json"],
            ),
            patch(
                "microtensor_miner_controller.github_release.GitHubReleasePublisher",
                return_value=publisher,
            ) as publisher_class,
        ):
            backend.upload(self.packaged)

        publisher_class.assert_called_once_with(
            owner="miner-artifacts",
            repo="public",
            tag="r7",
            token=TOKEN,
        )
        assets = publisher.publish.call_args.args[0]
        self.assertEqual(
            assets,
            {
                "model.gguf": self.config.artifact_dir / "model.gguf",
                "manifest.json": self.config.artifact_dir / "manifest.json",
            },
        )

    def test_publisher_secret_error_is_wrapped_generically(self) -> None:
        backend = MicrotensorBackend(self.config)
        publisher = Mock()
        publisher.publish.side_effect = ReleasePublishError(TOKEN)

        with (
            patch(
                "microtensor_miner_controller.backend._publishable_files",
                return_value=["model.gguf", "manifest.json"],
            ),
            patch(
                "microtensor_miner_controller.github_release.GitHubReleasePublisher",
                return_value=publisher,
            ),
            self.assertRaises(VerificationError) as raised,
        ):
            backend.upload(self.packaged)

        self.assertEqual(str(raised.exception), "GitHub immutable release upload failed")
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_dry_run_and_non_https_uploads_are_refused_before_dispatch(self) -> None:
        for config, source, message in (
            (replace(self.config, dry_run=True), self.source, "MMC_DRY_RUN=true"),
            (
                replace(self.config, github_token_file=None),
                "s3:public/uid-32/round-7",
                "S3/R2 are refused",
            ),
            (
                replace(self.config, github_token_file=None),
                "r2:public/uid-32/round-7",
                "S3/R2 are refused",
            ),
        ):
            with self.subTest(source=source):
                packaged = replace(self.packaged, source=source)
                with (
                    patch(
                        "microtensor_miner_controller.github_release.GitHubReleasePublisher"
                    ) as publisher_class,
                    patch(
                        "microtensor_miner_controller.backend._publishable_files"
                    ) as file_resolver,
                    self.assertRaisesRegex(VerificationError, message),
                ):
                    MicrotensorBackend(config).upload(packaged)
                publisher_class.assert_not_called()
                file_resolver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
