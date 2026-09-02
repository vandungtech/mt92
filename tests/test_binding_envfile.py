from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from helpers import base_env

from microtensor_miner_controller.binding import validate_binding, write_binding
from microtensor_miner_controller.config import ControllerConfig
from microtensor_miner_controller.envfile import load_env_file
from microtensor_miner_controller.errors import ConfigError


class EnvFileTests(unittest.TestCase):
    def test_root_service_mode_0640_data_file_loads_without_shell_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "miner.env"
            path.write_text("MMC_DRY_RUN=true\nMMC_SOURCE_TEMPLATE=s3:bucket/round-{round}\n")
            path.chmod(0o640)
            destination: dict[str, str] = {}
            loaded = load_env_file(path, destination)
            self.assertEqual(destination["MMC_DRY_RUN"], "true")
            self.assertIn("MMC_SOURCE_TEMPLATE", loaded)

    def test_env_file_rejects_permissions_and_shell_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "miner.env"
            path.write_text("TOKEN=$(id)\n")
            path.chmod(0o640)
            with self.assertRaisesRegex(ConfigError, "forbidden"):
                load_env_file(path, {})
            path.write_text("TOKEN=value\n")
            path.chmod(0o644)
            with self.assertRaisesRegex(ConfigError, "0640"):
                load_env_file(path, {})


class BindingTests(unittest.TestCase):
    def test_binding_tracks_exact_artifact_selfcheck_and_load_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = base_env(root)
            config = ControllerConfig.from_env(env)
            config.artifact_dir.mkdir(parents=True)
            config.upstream_home.mkdir(parents=True)
            (config.artifact_dir / "model.gguf").write_bytes(b"GGUF-model")
            config.selfcheck_path.write_text(
                json.dumps(
                    {"size_bytes": 10, "peak_rss_bytes": 20, "p95_latency_ms": 30}
                )
            )

            written = write_binding(config)
            self.assertEqual(validate_binding(config), written)
            self.assertEqual(os.stat(config.selfcheck_binding_path).st_mode & 0o777, 0o600)

            (config.artifact_dir / "model.gguf").write_bytes(b"changed")
            with self.assertRaisesRegex(Exception, "does not match"):
                validate_binding(config)


if __name__ == "__main__":
    unittest.main()
