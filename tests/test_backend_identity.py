from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from helpers import v030_env

from microtensor_miner_controller.backend import MicrotensorBackend, _signed_installation_tree
from microtensor_miner_controller.config import (
    SIGNED_V030_INSTALLED_TREE_BYTES,
    SIGNED_V030_INSTALLED_TREE_FILES,
    SIGNED_V030_INSTALLED_TREE_SHA256,
    SIGNED_V030_MECHANISM_VERSION,
    SIGNED_V030_RELEASE,
    SIGNED_V030_RELEASE_SIGNING_KEY,
    SIGNED_V030_WHEEL_SHA256,
    ControllerConfig,
)
from microtensor_miner_controller.errors import PreflightError, VerificationError
from microtensor_miner_controller.models import PackagedArtifact, RoundWindow


class FakeDistribution:
    def __init__(self, direct: object, *, version: str = SIGNED_V030_RELEASE) -> None:
        self.version = version
        self.direct = direct

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json" or self.direct is None:
            return None
        return json.dumps(self.direct)

    def locate_file(self, path: str) -> Path:
        del path
        return Path("/")


def _direct(digest: str = SIGNED_V030_WHEEL_SHA256) -> dict[str, object]:
    return {
        "url": "file:///verified/microtensor_subnet-0.3.2-py3-none-any.whl",
        "archive_info": {
            "hash": f"sha256={digest}",
            "hashes": {"sha256": digest},
        },
    }


def _constants(**overrides: object) -> SimpleNamespace:
    values = {
        "RELEASE_VERSION": SIGNED_V030_RELEASE,
        "MECHANISM_VERSION": SIGNED_V030_MECHANISM_VERSION,
        "PROVENANCE_REQUIRED": False,
        "RELEASE_SIGNING_KEY": SIGNED_V030_RELEASE_SIGNING_KEY,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BackendIdentityTests(unittest.TestCase):
    def _backend(self, root: Path) -> MicrotensorBackend:
        return MicrotensorBackend(ControllerConfig.from_env(v030_env(root)))

    def test_accepts_exact_signed_v030_wheel_and_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = self._backend(Path(temporary))
            with (
                mock.patch(
                    "microtensor_miner_controller.backend._signed_installation_tree",
                    return_value=(
                        SIGNED_V030_INSTALLED_TREE_FILES,
                        SIGNED_V030_INSTALLED_TREE_BYTES,
                        SIGNED_V030_INSTALLED_TREE_SHA256,
                    ),
                ),
                mock.patch(
                    "microtensor_miner_controller.backend.importlib.metadata.distribution",
                    return_value=FakeDistribution(_direct()),
                ),
                mock.patch(
                    "microtensor_miner_controller.backend.importlib.import_module",
                    return_value=_constants(),
                ),
            ):
                identity, version = backend._verify_upstream()
        self.assertEqual(identity, f"sha256:{SIGNED_V030_WHEEL_SHA256}")
        self.assertEqual(version, SIGNED_V030_RELEASE)

    def test_evaluation_seed_hash_is_independently_read_from_chain(self) -> None:
        seed_hash = "0x" + "c" * 64
        with tempfile.TemporaryDirectory() as temporary:
            backend = self._backend(Path(temporary))
            requested: list[int] = []

            def block_hash(block: int) -> str:
                requested.append(block)
                return seed_hash

            backend._client = SimpleNamespace(block_hash=block_hash)
            evaluation = RoundWindow(
                7,
                100,
                7_300,
                14_500,
                "evaluation",
                "coordinator",
                seed_block=7_300,
                block_hash=seed_hash,
            )
            backend._verify_v030_seed_hash(evaluation)
            self.assertEqual(requested, [7_300])

            mismatch = RoundWindow(
                7,
                100,
                7_300,
                14_500,
                "evaluation",
                "coordinator",
                seed_block=7_300,
                block_hash="0x" + "d" * 64,
            )
            with self.assertRaisesRegex(VerificationError, "independent chain block hash"):
                backend._verify_v030_seed_hash(mismatch)

            disclosed = RoundWindow(
                7,
                100,
                7_300,
                14_500,
                "submissions",
                "coordinator",
                seed_block=7_300,
                block_hash=seed_hash,
            )
            with self.assertRaisesRegex(VerificationError, "future seed"):
                backend._verify_v030_seed_hash(disclosed)

    def test_refuses_wrong_or_malformed_archive_identity(self) -> None:
        cases = (
            (FakeDistribution(_direct("0" * 64)), "wheel hash"),
            (FakeDistribution({"archive_info": {}}), "malformed"),
            (FakeDistribution(None), "no PEP 610"),
            (FakeDistribution(_direct(), version="0.3.1"), "expected signed release"),
        )
        for distribution, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                backend = self._backend(Path(temporary))
                with (
                    mock.patch(
                        "microtensor_miner_controller.backend.importlib.metadata.distribution",
                        return_value=distribution,
                    ),
                    self.assertRaisesRegex(PreflightError, message),
                ):
                    backend._verify_upstream()

    def test_refuses_mechanism_or_release_signing_key_mismatch(self) -> None:
        cases = (
            (_constants(MECHANISM_VERSION="0.3.1"), "mechanism identity"),
            (_constants(PROVENANCE_REQUIRED=True), "provenance gate"),
            (_constants(RELEASE_SIGNING_KEY="0x" + "00" * 32), "signing key"),
            (_constants(RELEASE_VERSION="0.3.1"), "release identity"),
        )
        for constants, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                backend = self._backend(Path(temporary))
                with (
                    mock.patch(
                        "microtensor_miner_controller.backend._signed_installation_tree",
                        return_value=(
                            SIGNED_V030_INSTALLED_TREE_FILES,
                            SIGNED_V030_INSTALLED_TREE_BYTES,
                            SIGNED_V030_INSTALLED_TREE_SHA256,
                        ),
                    ),
                    mock.patch(
                        "microtensor_miner_controller.backend.importlib.metadata.distribution",
                        return_value=FakeDistribution(_direct()),
                    ),
                    mock.patch(
                        "microtensor_miner_controller.backend.importlib.import_module",
                        return_value=constants,
                    ),
                    self.assertRaisesRegex(PreflightError, message),
                ):
                    backend._verify_upstream()

    def test_signed_installation_tree_detects_extra_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (
                root / "microtensor",
                root / "neurons",
                root / "microtensor_subnet-0.3.2.dist-info",
            ):
                directory.mkdir()
            (root / "microtensor" / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / "neurons" / "miner.py").write_text("value = 2\n", encoding="utf-8")
            (root / "microtensor_subnet-0.3.2.dist-info" / "METADATA").write_text(
                "Version: 0.3.2\n", encoding="utf-8"
            )
            original = _signed_installation_tree(root)
            (root / "microtensor" / "unexpected.py").write_text(
                "value = 3\n", encoding="utf-8"
            )
            changed = _signed_installation_tree(root)
            self.assertNotEqual(original, changed)
            (root / "microtensor" / "link.py").symlink_to("module.py")
            with self.assertRaisesRegex(PreflightError, "regular file"):
                _signed_installation_tree(root)

    def test_v032_disabled_provenance_gate_never_imports_or_contacts_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = self._backend(Path(temporary))
            packaged = PackagedArtifact(
                round_index=7,
                source="https://github.com/example/repository/releases/download/round-7/artifact",
                hotkey="5Hotkey",
                manifest_digest="sha256:" + "1" * 64,
                artifact_digest="sha256:" + "2" * 64,
                file_count=1,
                total_bytes=1,
            )
            with mock.patch(
                "builtins.__import__",
                side_effect=AssertionError("unexpected import or network client load"),
            ):
                backend.verify_provenance(packaged, 100)


if __name__ == "__main__":
    unittest.main()
