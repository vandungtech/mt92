from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from helpers import v030_env

from microtensor_miner_controller.backend import MicrotensorBackend
from microtensor_miner_controller.config import (
    SIGNED_V030_MECHANISM_VERSION,
    SIGNED_V030_RELEASE,
    SIGNED_V030_RELEASE_SIGNING_KEY,
    SIGNED_V030_WHEEL_SHA256,
    ControllerConfig,
)
from microtensor_miner_controller.errors import PreflightError, VerificationError
from microtensor_miner_controller.models import RoundWindow


class FakeDistribution:
    def __init__(self, direct: object, *, version: str = SIGNED_V030_RELEASE) -> None:
        self.version = version
        self.direct = direct

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json" or self.direct is None:
            return None
        return json.dumps(self.direct)


def _direct(digest: str = SIGNED_V030_WHEEL_SHA256) -> dict[str, object]:
    return {
        "url": "file:///verified/microtensor_subnet-0.3.0-py3-none-any.whl",
        "archive_info": {
            "hash": f"sha256={digest}",
            "hashes": {"sha256": digest},
        },
    }


def _constants(**overrides: str) -> SimpleNamespace:
    values = {
        "RELEASE_VERSION": SIGNED_V030_RELEASE,
        "MECHANISM_VERSION": SIGNED_V030_MECHANISM_VERSION,
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
            (_constants(RELEASE_SIGNING_KEY="0x" + "00" * 32), "signing key"),
            (_constants(RELEASE_VERSION="0.3.1"), "release identity"),
        )
        for constants, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                backend = self._backend(Path(temporary))
                with (
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


if __name__ == "__main__":
    unittest.main()
