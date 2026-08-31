from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import publish_code_provenance as provenance


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, object] = {}


class FakeWandb:
    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_calls: list[dict[str, object]] = []
        self.logs: list[tuple[dict[str, object], int]] = []
        self.finished = False

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls.append(kwargs)
        return self.run

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.logs.append((payload, step))

    def finish(self) -> None:
        self.finished = True


class PublishCodeProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.dataset = self.root / "dataset"
        self.base = self.root / "base"
        self.artifact = self.root / "artifact"
        for directory in (self.run, self.dataset, self.base, self.artifact):
            directory.mkdir()
        self.source = self.root / "source.json"
        self.source.write_text("{}", encoding="utf-8")

        self.selection = {
            "policy": "final_epoch_no_holdout",
            "metric": None,
            "terminal_epoch": 2,
            "terminal_loss": None,
            "best_epoch": None,
            "best_loss": None,
            "exported_epoch": 2,
            "exported_step": 2,
        }
        self.metrics = [
            {
                "step": step,
                "epoch": step,
                "loss": 1.0 / step,
                "loss_mass": 10.0,
                "supervised_tokens": 10,
                "terminal_eos_tokens": 1,
                "terminal_eos_loss_weight": 1.0,
                "microbatches": 1,
                "gradient_norm": 0.5,
                "learning_rate": 0.0002,
                "elapsed_s": float(step),
            }
            for step in (1, 2)
        ]
        self.metrics.append({"event": "export_selection", **self.selection})
        metrics_raw = b"".join(
            json.dumps(row, sort_keys=True).encode() + b"\n" for row in self.metrics
        )
        (self.run / "metrics.jsonl").write_bytes(metrics_raw)
        self.metadata = {
            "schema": provenance.TRAINING_SCHEMA,
            "updates": 2,
            "selection": self.selection,
            "metrics_digest": digest(metrics_raw),
        }
        write_json(self.run / "training_metadata.json", self.metadata)

        self.artifact_digest = "sha256:" + "a" * 64
        self.artifact_identity = {
            "root": str(self.artifact.resolve()),
            "tree_algorithm": "sorted_nfc_relative_path_nul_sha256_nul_v1",
            "tree_digest": self.artifact_digest,
            "total_bytes": 99,
            "files": [{"path": "model.gguf", "bytes": 99, "digest": "sha256:" + "b" * 64}],
            "entrypoint": {
                "path": "model.gguf",
                "bytes": 99,
                "digest": "sha256:" + "b" * 64,
                "gguf": {"file_type": 15},
            },
        }
        self.load = {
            "format": "gguf",
            "quantization": "Q4_K_M",
            "entrypoint": "model.gguf",
            "max_input": {"tokens": 1024},
            "preprocessing": {"tokenizer": "tokenizer.json"},
            "base_model": provenance.BASE_MODEL,
        }
        self.load_path = self.root / "load.json"
        write_json(self.load_path, self.load)
        self.training_lineage = {
            "receipt": {"digest": digest((self.run / "training_metadata.json").read_bytes())},
            "run": {"merged": {"digest": "sha256:" + "c" * 64, "files": []}},
        }
        self.conversion = {
            "schema": provenance.CONVERSION_SCHEMA,
            "status": "complete",
            "track": provenance.TRACK,
            "hardware_class": provenance.HARDWARE_CLASS,
            "base_model": provenance.BASE_MODEL,
            "llama_cpp_revision": provenance.LLAMA_CPP_REVISION,
            "source": {
                "training_metadata_digest": self.training_lineage["receipt"]["digest"],
                "merged_tree_digest": self.training_lineage["run"]["merged"]["digest"],
            },
            "conversion": {
                "converter_digest": "sha256:" + "d" * 64,
                "quantizer_digest": "sha256:" + "e" * 64,
                "commands": [
                    {
                        "name": name,
                        "argv": [name, "input", "output"],
                        "returncode": 0,
                        "started_at_unix_ns": index,
                        "finished_at_unix_ns": index + 1,
                    }
                    for index, name in enumerate(("convert_f16", "quantize"), 1)
                ],
            },
            "artifact": {
                "tree_digest": self.artifact_digest,
                "entrypoint_digest": self.artifact_identity["entrypoint"]["digest"],
                "entrypoint_bytes": 99,
                "quantization": "Q4_K_M",
            },
            "load_manifest": self.load,
            "calibration_receipt_digest": None,
        }
        self.conversion_path = self.root / "conversion.json"
        write_json(self.conversion_path, self.conversion)
        self.request = provenance.PublicationRequest(
            training_run=self.run,
            training_dataset=self.dataset,
            source_corpus=self.source,
            base=self.base,
            artifact=self.artifact,
            artifact_digest=self.artifact_digest,
            load_spec=self.load_path,
            conversion_receipt=self.conversion_path,
            finished_block=123,
        )

    def patches(self):
        return (
            mock.patch.object(
                provenance.gguf,
                "load_v5_training_lineage",
                return_value=(self.training_lineage, ()),
            ),
            mock.patch.object(
                provenance.gguf,
                "artifact_identity",
                return_value=self.artifact_identity,
            ),
        )

    def validate(self) -> provenance.Publication:
        first, second = self.patches()
        with first, second:
            return provenance.validate_publication(self.request)

    def test_exact_official_config_and_metrics_are_published_through_injected_client(self) -> None:
        first, second = self.patches()
        fake = FakeWandb()
        with (
            first,
            second,
            mock.patch.object(
                provenance.importlib,
                "import_module",
                side_effect=AssertionError("credentials/network module must not be consulted"),
            ),
        ):
            publication = provenance.validate_publication(self.request)
            provenance.publish(publication, fake)
        self.assertEqual(len(fake.init_calls), 1)
        call = fake.init_calls[0]
        self.assertEqual(
            (call["entity"], call["project"], call["name"]),
            (provenance.ENTITY, provenance.PROJECT, provenance.HOTKEY),
        )
        config = call["config"]
        self.assertEqual(config["mt_track"], provenance.TRACK)
        self.assertEqual(config["mt_class"], provenance.HARDWARE_CLASS)
        self.assertEqual(config["mt_base_model"], provenance.BASE_MODEL)
        self.assertEqual(config["mt_artifact_digest"], self.artifact_digest)
        self.assertEqual(config["mt_calibration_claim"], provenance.NO_CALIBRATION_CLAIM)
        self.assertEqual(fake.logs, [(row, index) for index, row in enumerate(self.metrics, 1)])
        self.assertEqual(fake.run.summary["mt_artifact_digest"], self.artifact_digest)
        self.assertEqual(fake.run.summary["mt_finished_at"], 123)
        self.assertTrue(fake.finished)

    def test_mutation_after_validation_fails_before_wandb_init(self) -> None:
        publication = self.validate()
        (self.run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        fake = FakeWandb()
        first, second = self.patches()
        with first, second, self.assertRaises(provenance.CodeProvenanceError):
            provenance.publish(publication, fake)
        self.assertEqual(fake.init_calls, [])

    def test_symlinked_receipt_is_rejected(self) -> None:
        target = self.conversion_path.with_name("actual-conversion.json")
        self.conversion_path.replace(target)
        self.conversion_path.symlink_to(target)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "regular non-symlink"):
            self.validate()

    def test_cross_lineage_conversion_is_rejected(self) -> None:
        self.conversion["source"]["merged_tree_digest"] = "sha256:" + "f" * 64
        write_json(self.conversion_path, self.conversion)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "crosses training lineages"):
            self.validate()

    def test_nonfinite_metric_is_rejected(self) -> None:
        raw = (
            (self.run / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .replace('"loss": 1.0', '"loss": NaN', 1)
        )
        (self.run / "metrics.jsonl").write_text(raw, encoding="utf-8")
        self.metadata["metrics_digest"] = digest(raw.encode())
        write_json(self.run / "training_metadata.json", self.metadata)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "non-finite JSON"):
            self.validate()

    def test_unvalidated_calibration_claim_is_rejected(self) -> None:
        self.conversion["calibration_receipt_digest"] = "sha256:" + "1" * 64
        write_json(self.conversion_path, self.conversion)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "calibration binding"):
            self.validate()

    def test_finished_block_is_strict(self) -> None:
        for value in (0, True, -1):
            request = provenance.PublicationRequest(
                **{**self.request.__dict__, "finished_block": value}
            )
            first, second = self.patches()
            with (
                first,
                second,
                self.assertRaisesRegex(provenance.CodeProvenanceError, "finished block"),
            ):
                provenance.validate_publication(request)


if __name__ == "__main__":
    unittest.main()
