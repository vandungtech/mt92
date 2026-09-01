from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest import mock

from training import historical_code_candidate as historical_candidate
from training import normalized_historical_code_candidate as normalized_candidate
from training import publish_code_provenance as provenance


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FakeSettings:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        for key, value in provenance._wandb_identity_defaults().items():
            setattr(self, key, value)
        for key, value in values.items():
            setattr(self, key, value)


class FakeSetup:
    def __init__(self, settings: FakeSettings) -> None:
        self.settings = settings


class FakeRun:
    def __init__(self, client: FakeWandb) -> None:
        self.summary: dict[str, object] = {}
        self.metric_definitions: list[dict[str, object]] = []
        self.logs: list[tuple[dict[str, object], int]] = []
        self.finished = False
        self.client = client
        self.events = client.events
        self.id: str | None = None
        self.name: str | None = None
        self.config: dict[str, object] = {}
        self.state = "running"
        self.finish_exit_codes: list[int] = []

    def define_metric(self, **kwargs: object) -> None:
        self.events.append("define_metric")
        self.metric_definitions.append(kwargs)

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.events.append(f"log:{step}")
        if self.client.fail_log_step == step:
            raise RuntimeError("simulated log failure")
        self.logs.append((payload, step))
        summaries_disabled = any(
            definition.get("name") == "*" and definition.get("summary") == "none"
            for definition in self.metric_definitions
        )
        if not summaries_disabled:
            self.summary.update(payload)

    def finish(self, *, exit_code: int) -> None:
        self.events.append(f"finish:{exit_code}")
        self.finish_exit_codes.append(exit_code)
        self.summary.update(
            {
                "_runtime": 0,
                "_step": len(self.logs),
                "_timestamp": 0,
                "_wandb": {"runtime": 0},
            }
        )
        self.finished = True
        self.state = "finished" if exit_code == 0 else "failed"


class FakeRemoteRun:
    def __init__(self, client: FakeWandb) -> None:
        self.client = client
        self.local = client.created_run
        self.id = self.local.id
        self.entity = provenance.ENTITY
        self.project = provenance.PROJECT
        self.name = self.local.name
        self.state = client.remote_state_override or self.local.state
        self.config = dict(client.remote_config_override or self.local.config)
        self.rawconfig = dict(
            client.remote_rawconfig_override
            or {**self.config, "_wandb": {"m": [], "start_time": 0, "t": {}}}
        )
        self.summary = dict(self.local.summary)

    def scan_history(self, **kwargs: object) -> list[dict[str, object]]:
        self.client.events.append("remote_scan")
        self.client.scan_history_calls.append(kwargs)
        return [
            {
                **payload,
                "_runtime": 0,
                "_step": step,
                "_timestamp": 0,
            }
            for payload, step in self.local.logs
        ]

    def update(self) -> None:
        self.client.events.append("remote_update")
        self.client.remote_update_names.append(str(self.name))
        if self.client.remote_update_exception is not None:
            if self.client.remote_update_applies_before_exception:
                self.local.name = self.name
            raise self.client.remote_update_exception
        self.local.name = self.name


class FakeApi:
    def __init__(self, client: FakeWandb) -> None:
        self.client = client

    def run(self, path: str) -> FakeRemoteRun:
        self.client.events.append("api_run")
        self.client.api_run_paths.append(path)
        call = len(self.client.api_run_paths)
        if call in self.client.api_run_failure_calls:
            raise RuntimeError(f"simulated API readback failure {call}")
        remote = FakeRemoteRun(self.client)
        self.client.readback_names.append(remote.name)
        return remote


class FakeWandb:
    __version__ = provenance.WANDB_SDK_VERSION

    def __init__(self) -> None:
        self.events: list[str] = []
        self.run = None
        self.settings_calls: list[dict[str, object]] = []
        self.settings_observations: list[dict[str, object]] = []
        self.setup_calls: list[FakeSettings] = []
        self.teardown_calls = 0
        self.init_calls: list[dict[str, object]] = []
        self.api_calls: list[dict[str, object]] = []
        self.api_run_paths: list[str] = []
        self.scan_history_calls: list[dict[str, object]] = []
        self.remote_update_names: list[str] = []
        self.readback_names: list[str | None] = []
        self.api_run_failure_calls: set[int] = set()
        self.remote_update_exception: BaseException | None = None
        self.remote_update_applies_before_exception = False
        self.teardown_exception: BaseException | None = None
        self.fail_log_step: int | None = None
        self.remote_state_override: str | None = None
        self.remote_config_override: dict[str, object] | None = None
        self.remote_rawconfig_override: dict[str, object] | None = None
        self.created_run = FakeRun(self)

    @property
    def logs(self) -> list[tuple[dict[str, object], int]]:
        return self.created_run.logs

    @property
    def finished(self) -> bool:
        return self.created_run.finished

    def Settings(self, **kwargs: object) -> FakeSettings:
        self.events.append("settings")
        self.settings_calls.append(kwargs)
        root = Path(str(kwargs["root_dir"]))
        self.settings_observations.append(
            {
                "cwd": Path.cwd(),
                "root_mode": stat.S_IMODE(root.stat().st_mode),
                "system_settings_exists": Path(str(kwargs["settings_system"])).exists(),
                "credentials_file_exists": Path(str(kwargs["credentials_file"])).exists(),
                "error_reporting": os.environ.get(provenance.WANDB_ERROR_REPORTING_VARIABLE),
            }
        )
        return FakeSettings(kwargs)

    def setup(self, *, settings: FakeSettings) -> FakeSetup:
        self.events.append("setup")
        self.setup_calls.append(settings)
        os.environ["WANDB_SERVICE"] = "fake-service"
        return FakeSetup(settings)

    def teardown(self) -> None:
        self.events.append("teardown")
        self.teardown_calls += 1
        if self.teardown_exception is not None:
            raise self.teardown_exception

    def init(self, **kwargs: object) -> FakeRun:
        self.events.append("init")
        self.init_calls.append(kwargs)
        self.created_run.id = str(kwargs["id"])
        self.created_run.name = str(kwargs["name"])
        self.created_run.config = dict(kwargs["config"])
        return self.created_run

    def Api(self, **kwargs: object) -> FakeApi:
        self.events.append("api")
        self.api_calls.append(kwargs)
        return FakeApi(self)

    def log(self, _payload: dict[str, object], *, step: int) -> None:
        raise AssertionError(f"module-level W&B log is forbidden at step {step}")

    def finish(self, *, exit_code: int) -> None:
        raise AssertionError(f"module-level W&B finish is forbidden: {exit_code}")


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
        self.base_identity = {
            "base_model": provenance.BASE_MODEL,
            "required_bytes": 1,
            "files": {
                "config.json": {
                    "bytes": 1,
                    "sha256": "f" * 64,
                }
            },
        }
        self.manifest = {
            "schema": historical_candidate.DATASET_SCHEMA,
            "track": provenance.TRACK,
            "hardware_class": provenance.HARDWARE_CLASS,
            "corpus_version": historical_candidate.CORPUS_VERSION,
            "source_file_digest": historical_candidate.PUBLIC_CORPUS_RAW_DIGEST,
            "seed": 92,
            "train_examples": historical_candidate.EXPECTED_COUNTS["train"],
            "holdout_examples": 0,
            "train_file_digest": "sha256:" + "1" * 64,
            "holdout_file_digest": "sha256:" + "2" * 64,
            "target_construction": historical_candidate.TARGET_CONSTRUCTION,
            "quality_claim": historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        }
        self.manifest_identity = {
            "bytes": 1,
            "digest": "sha256:" + "3" * 64,
        }
        self.metadata = {
            "schema": provenance.TRAINING_SCHEMA,
            "status": "complete",
            "run_kind": "final_all_public",
            "hotkey": provenance.HOTKEY,
            "track": provenance.TRACK,
            "hardware_class": provenance.HARDWARE_CLASS,
            "base_model": provenance.BASE_MODEL,
            "base_snapshot": self.base_identity,
            "corpus_version": historical_candidate.CORPUS_VERSION,
            "dataset": {
                "manifest": self.manifest,
                "manifest_digest": self.manifest_identity["digest"],
                "source_corpus": historical_candidate.source_corpus_identity(),
            },
            "quality_claim": historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
            "updates": 2,
            "selection": self.selection,
            "metrics_digest": digest(metrics_raw),
        }
        write_json(self.run / "training_metadata.json", self.metadata)
        metadata_raw = (self.run / "training_metadata.json").read_bytes()
        metadata_identity = {
            "bytes": len(metadata_raw),
            "digest": digest(metadata_raw),
        }

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
            "quantization": "Q8_0",
            "entrypoint": "model.gguf",
            "max_input": {"tokens": 1024},
            "preprocessing": {"tokenizer": "tokenizer.json"},
            "base_model": provenance.BASE_MODEL,
        }
        self.load_path = self.root / "load.json"
        write_json(self.load_path, self.load)
        self.training_lineage = {
            "status": "provided_and_validated",
            "schema": provenance.TRAINING_SCHEMA,
            "receipt": {**metadata_identity},
            "source_corpus": {
                "file": {
                    "bytes": historical_candidate.PUBLIC_CORPUS_RESPONSE_BYTES,
                    "digest": historical_candidate.PUBLIC_CORPUS_RAW_DIGEST,
                },
                **historical_candidate.source_corpus_identity(),
            },
            "prepared_dataset": {
                "manifest": self.manifest_identity,
                "train": {"bytes": 1, "digest": self.manifest["train_file_digest"]},
                "holdout": {
                    "bytes": 0,
                    "digest": self.manifest["holdout_file_digest"],
                },
                "manifest_payload": self.manifest,
            },
            "base_snapshot": self.base_identity,
            "run": {
                "kind": "merged",
                "training_metadata": {**metadata_identity},
                "metrics": {
                    "bytes": len(metrics_raw),
                    "digest": digest(metrics_raw),
                },
                "adapter": {
                    "digest": "sha256:" + "4" * 64,
                    "files": [{"path": "adapter.safetensors", "bytes": 1}],
                    "total_bytes": 1,
                },
                "merged": {
                    "digest": "sha256:" + "c" * 64,
                    "files": [{"path": "model.safetensors", "bytes": 1}],
                    "total_bytes": 1,
                },
            },
            "conversion_binding_claim": "fixture validated conversion binding",
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
                "quantization": "Q8_0",
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
                "load_training_lineage",
                return_value=(self.training_lineage, ()),
            ),
            mock.patch.object(
                provenance.gguf,
                "artifact_identity",
                return_value=self.artifact_identity,
            ),
        )

    def rewrite_training_metadata(self) -> None:
        write_json(self.run / "training_metadata.json", self.metadata)
        raw = (self.run / "training_metadata.json").read_bytes()
        identity = {"bytes": len(raw), "digest": digest(raw)}
        self.training_lineage["receipt"] = copy.deepcopy(identity)
        self.training_lineage["run"]["training_metadata"] = copy.deepcopy(identity)

    def configure_normalized_v6(self) -> None:
        self.manifest = {
            "schema": normalized_candidate.DATASET_SCHEMA,
            "track": provenance.TRACK,
            "hardware_class": provenance.HARDWARE_CLASS,
            "corpus_profile": normalized_candidate.CORPUS_PROFILE,
            "corpus_version": normalized_candidate.CORPUS_VERSION,
            "source_file_digest": normalized_candidate.PUBLIC_CORPUS_RAW_DIGEST,
            "seed": normalized_candidate.EXPECTED_SEED,
            "source_examples": normalized_candidate.EXPECTED_SOURCE_EXAMPLES,
            "train_examples": normalized_candidate.EXPECTED_TRAIN_EXAMPLES,
            "holdout_examples": normalized_candidate.EXPECTED_HOLDOUT_EXAMPLES,
            "excluded_examples": normalized_candidate.EXPECTED_EXCLUDED_EXAMPLES,
            "train_file_bytes": normalized_candidate.EXPECTED_TRAIN_FILE_BYTES,
            "train_file_digest": normalized_candidate.EXPECTED_TRAIN_FILE_DIGEST,
            "holdout_file_bytes": normalized_candidate.EXPECTED_HOLDOUT_FILE_BYTES,
            "holdout_file_digest": normalized_candidate.EXPECTED_HOLDOUT_FILE_DIGEST,
            "excluded_refs_file": normalized_candidate.EXCLUDED_REFS_FILE,
            "excluded_refs_canonical_bytes": (
                normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES
            ),
            "excluded_refs_digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
            "target_construction": normalized_candidate.TARGET_CONSTRUCTION,
            "normalization": normalized_candidate.NORMALIZATION_CONTRACT,
            "quality_claim": normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        }
        self.training_lineage.update(
            {
                "schema": provenance.NORMALIZED_TRAINING_SCHEMA,
                "source_corpus": {
                    "file": {
                        "bytes": normalized_candidate.PUBLIC_CORPUS_RESPONSE_BYTES,
                        "digest": normalized_candidate.PUBLIC_CORPUS_RAW_DIGEST,
                    },
                    **normalized_candidate.source_corpus_identity(),
                },
                "prepared_dataset": {
                    "manifest": self.manifest_identity,
                    "train": {
                        "bytes": normalized_candidate.EXPECTED_TRAIN_FILE_BYTES,
                        "digest": normalized_candidate.EXPECTED_TRAIN_FILE_DIGEST,
                    },
                    "holdout": {
                        "bytes": normalized_candidate.EXPECTED_HOLDOUT_FILE_BYTES,
                        "digest": normalized_candidate.EXPECTED_HOLDOUT_FILE_DIGEST,
                    },
                    "excluded_refs": {
                        "bytes": (normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES),
                        "digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
                    },
                    "manifest_payload": self.manifest,
                },
            }
        )
        self.metadata.update(
            {
                "schema": provenance.NORMALIZED_TRAINING_SCHEMA,
                "corpus_version": normalized_candidate.CORPUS_VERSION,
                "dataset": {
                    "manifest": self.manifest,
                    "manifest_digest": self.manifest_identity["digest"],
                    "source_corpus": normalized_candidate.source_corpus_identity(),
                },
                "quality_claim": normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
            }
        )
        self.rewrite_training_metadata()
        self.conversion["schema"] = provenance.NORMALIZED_CONVERSION_SCHEMA
        self.conversion["source"] = provenance._expected_conversion_source(self.training_lineage)
        write_json(self.conversion_path, self.conversion)

    def validate(self) -> provenance.Publication:
        first, second = self.patches()
        with first, second:
            return provenance.validate_publication(self.request)

    def export(self, publication: provenance.Publication, path: Path) -> tuple[int, str]:
        first, second = self.patches()
        with first, second:
            return provenance.export_payload(publication, path)

    def publish_fake(
        self,
        publication: provenance.Publication,
        payload_path: Path,
        payload_digest: str,
        fake: FakeWandb,
    ) -> provenance.WandbPublicationOutcome:
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
        ):
            return provenance.publish(publication, payload_path, payload_digest, fake)

    def argv(self, action: str, *extra: str) -> list[str]:
        return [
            action,
            "--training-run",
            str(self.run),
            "--training-dataset",
            str(self.dataset),
            "--source-corpus",
            str(self.source),
            "--base",
            str(self.base),
            "--artifact",
            str(self.artifact),
            "--artifact-digest",
            self.artifact_digest,
            "--load-spec",
            str(self.load_path),
            "--conversion-receipt",
            str(self.conversion_path),
            "--finished-block",
            "123",
            *extra,
        ]

    def test_export_is_canonical_deterministic_and_exact(self) -> None:
        publication = self.validate()
        first_path = self.root / "wandb-one.json"
        second_path = self.root / "wandb-two.json"
        first_size, first_digest = self.export(publication, first_path)
        second_size, second_digest = self.export(publication, second_path)
        first_bytes = first_path.read_bytes()
        second_bytes = second_path.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual((first_size, first_digest), (second_size, second_digest))
        self.assertEqual(first_size, len(first_bytes))
        self.assertEqual(first_digest, digest(first_bytes))
        self.assertTrue(first_bytes.endswith(b"\n"))
        self.assertFalse(first_bytes.endswith(b"\n\n"))
        self.assertEqual(stat.S_IMODE(first_path.stat().st_mode), 0o600)

        payload = json.loads(first_bytes)
        canonical = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        self.assertEqual(first_bytes, canonical)
        self.assertEqual(
            set(payload),
            {"schema", "destination", "controls", "config", "logs", "summary"},
        )
        self.assertEqual(payload["schema"], provenance.WANDB_PAYLOAD_SCHEMA)
        run_id = payload["destination"]["id"]
        self.assertRegex(run_id, r"^mt92-[0-9a-f]{40}$")
        self.assertEqual(run_id, provenance._wandb_run_id(publication))
        controls_variant = json.loads(json.dumps(payload["controls"]))
        controls_variant["settings"]["quiet"] = False
        self.assertNotEqual(
            run_id,
            provenance._wandb_run_id_from_envelope(
                controls_variant,
                payload["config"],
                payload["logs"],
                payload["summary"],
            ),
        )
        self.assertEqual(
            payload["destination"],
            {
                "entity": provenance.ENTITY,
                "project": provenance.PROJECT,
                "name": provenance.HOTKEY,
                "id": run_id,
                "pending_name": provenance._wandb_pending_name(run_id),
            },
        )
        self.assertEqual(payload["controls"], provenance._wandb_controls())
        service_policy = payload["controls"]["service_metadata_policy"]
        self.assertEqual(
            service_policy["application_summary_fields"],
            list(provenance.APPLICATION_SUMMARY_FIELDS),
        )
        self.assertEqual(
            set(payload["summary"]),
            set(service_policy["application_summary_fields"]),
        )
        self.assertEqual(
            set(service_policy["reserved_config"]),
            {"_wandb", "wandb_version"},
        )
        self.assertEqual(
            service_policy["reserved_config"]["_wandb"]["producer"],
            "W&B SDK and service",
        )
        self.assertEqual(
            service_policy["automatic_history_fields"],
            ["_runtime", "_step", "_timestamp"],
        )
        self.assertEqual(
            service_policy["automatic_summary_paths"],
            ["_runtime", "_step", "_timestamp", "_wandb.runtime"],
        )
        self.assertIn("platform_and_architecture", service_policy["telemetry_fields"])
        self.assertIn(
            "third_party_error_reporting", service_policy["disabled_automatic_collectors"]
        )
        self.assertNotIn("environment", service_policy["disabled_automatic_collectors"])
        self.assertEqual(
            service_policy["allowed_sdk_transaction_records"],
            ["exit", "header", "history", "metric", "run", "summary", "telemetry"],
        )
        self.assertEqual(
            payload["controls"]["environment"]["forced_process_variables"],
            {provenance.WANDB_ERROR_REPORTING_VARIABLE: "false"},
        )
        self.assertEqual(payload["controls"]["run_identity"]["resume"], "never")
        self.assertEqual(
            payload["controls"]["settings"]["host"],
            provenance.WANDB_REDACTED_HOST,
        )
        self.assertEqual(
            payload["controls"]["isolation"]["effective_identity_defaults"],
            provenance._wandb_identity_defaults(),
        )
        self.assertEqual(
            service_policy["authorization_scope"],
            "exact-application-envelope-plus-disclosed-bounded-service-metadata",
        )
        self.assertFalse(service_policy["wire_transcript_authorized_exactly"])
        lifecycle = payload["controls"]["publication_lifecycle"]
        self.assertEqual(
            lifecycle["update_acknowledgement"], "marks-committed-before-second-readback"
        )
        self.assertIn("outcome_uncertain", lifecycle["terminal_states"])
        self.assertFalse(payload["controls"]["environment"]["credential_values_in_envelope"])
        self.assertEqual(payload["config"], provenance._wandb_config(publication))
        self.assertEqual(
            payload["logs"],
            [{"step": index, "payload": row} for index, row in enumerate(self.metrics, 1)],
        )
        self.assertEqual(
            payload["summary"],
            {
                "mt_artifact_digest": self.artifact_digest,
                "mt_finished_at": 123,
                "mt_training_records": len(self.metrics),
                "mt_training_schema": provenance.TRAINING_SCHEMA,
                "mt_conversion_receipt_digest": digest(self.conversion_path.read_bytes()),
            },
        )

    def test_payload_contains_all_1001_explicit_steps(self) -> None:
        publication = self.validate()
        metrics = tuple({"value": index} for index in range(1_001))
        payload = provenance.publication_payload(replace(publication, metrics=metrics))
        self.assertEqual(len(payload["logs"]), 1_001)
        self.assertEqual(
            [record["step"] for record in payload["logs"]],
            list(range(1, 1_002)),
        )
        self.assertEqual(
            [record["payload"] for record in payload["logs"]],
            list(metrics),
        )

    def test_exact_authorized_payload_is_published_through_injected_client(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            mock.patch.object(
                provenance.importlib,
                "import_module",
                side_effect=AssertionError("credentials/network module must not be consulted"),
            ),
        ):
            outcome = provenance.publish(publication, payload_path, payload_digest, fake)
            self.assertNotIn("WANDB_SERVICE", os.environ)
            self.assertNotIn(provenance.WANDB_ERROR_REPORTING_VARIABLE, os.environ)
            self.assertEqual(os.environ["WANDB_API_KEY"], "test-key")

        self.assertEqual(outcome.state, provenance.WandbPublicationState.COMMITTED)
        self.assertEqual(outcome.run_id, payload["destination"]["id"])
        self.assertEqual(outcome.resolution, "update_ack")
        self.assertTrue(outcome.retry_forbidden)
        self.assertEqual(fake.events[:4], ["settings", "setup", "init", "define_metric"])
        self.assertEqual(fake.events[-1], "teardown")
        self.assertEqual(fake.created_run.finish_exit_codes, [0])
        self.assertEqual(fake.remote_update_names, [provenance.HOTKEY])
        self.assertEqual(
            fake.readback_names,
            [payload["destination"]["pending_name"], provenance.HOTKEY],
        )
        self.assertLess(fake.events.index("remote_scan"), fake.events.index("remote_update"))
        self.assertEqual(len(fake.api_calls), 2)
        self.assertEqual(len(fake.settings_calls), 1)
        self.assertEqual(fake.setup_calls, [fake.init_calls[0]["settings"]])
        self.assertEqual(fake.teardown_calls, 1)
        runtime_settings = fake.settings_calls[0]
        for key, value in payload["controls"]["settings"].items():
            self.assertEqual(runtime_settings[key], value)
        private_root = Path(str(runtime_settings["root_dir"]))
        self.assertEqual(
            Path(str(runtime_settings["settings_system"])),
            private_root / "system-settings",
        )
        self.assertEqual(
            Path(str(runtime_settings["credentials_file"])),
            private_root / "credentials.json",
        )
        self.assertEqual(
            fake.settings_observations,
            [
                {
                    "cwd": private_root,
                    "root_mode": 0o700,
                    "system_settings_exists": False,
                    "credentials_file_exists": False,
                    "error_reporting": "false",
                }
            ],
        )
        self.assertFalse(private_root.exists())

        self.assertEqual(len(fake.init_calls), 1)
        call = fake.init_calls[0]
        self.assertEqual(
            (call["entity"], call["project"], call["name"]),
            (
                provenance.ENTITY,
                provenance.PROJECT,
                payload["destination"]["pending_name"],
            ),
        )
        self.assertEqual(call["id"], payload["destination"]["id"])
        self.assertEqual(call["settings"].values, runtime_settings)
        self.assertEqual(
            fake.created_run.metric_definitions,
            [payload["controls"]["metric_definition"]],
        )
        config = call["config"]
        self.assertEqual(config["mt_track"], provenance.TRACK)
        self.assertEqual(config["mt_class"], provenance.HARDWARE_CLASS)
        self.assertEqual(config["mt_base_model"], provenance.BASE_MODEL)
        self.assertEqual(config["mt_artifact_digest"], self.artifact_digest)
        self.assertEqual(config["mt_calibration_claim"], provenance.NO_CALIBRATION_CLAIM)
        self.assertEqual(fake.logs, [(row, index) for index, row in enumerate(self.metrics, 1)])

        application_summary = {
            key: value for key, value in fake.created_run.summary.items() if not key.startswith("_")
        }
        reserved_summary = set(fake.created_run.summary) - set(application_summary)
        self.assertEqual(set(application_summary), set(provenance.APPLICATION_SUMMARY_FIELDS))
        self.assertEqual(
            reserved_summary,
            {"_runtime", "_step", "_timestamp", "_wandb"},
        )
        self.assertEqual(application_summary["mt_artifact_digest"], self.artifact_digest)
        self.assertEqual(application_summary["mt_finished_at"], 123)
        self.assertTrue(fake.finished)

    def test_effective_global_identity_mutation_fails_before_init_and_cleans_up(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()
        original_setup = fake.setup

        def contaminated_setup(*, settings: FakeSettings) -> FakeSetup:
            setup_state = original_setup(settings=settings)
            setup_state.settings.sweep_id = "ambient-sweep"
            return setup_state

        fake.setup = contaminated_setup  # type: ignore[method-assign]
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
        ):
            with self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "effective global W&B identity setting",
            ):
                provenance.publish(publication, payload_path, payload_digest, fake)
            self.assertNotIn("WANDB_SERVICE", os.environ)
            self.assertNotIn(provenance.WANDB_ERROR_REPORTING_VARIABLE, os.environ)
            self.assertEqual(os.environ["WANDB_API_KEY"], "test-key")

        self.assertEqual(fake.init_calls, [])
        self.assertEqual(fake.teardown_calls, 1)

    def test_mutation_after_validation_fails_before_wandb_init(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        (self.run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        fake = FakeWandb()
        first, second = self.patches()
        with first, second, self.assertRaises(provenance.CodeProvenanceError):
            provenance.publish(publication, payload_path, payload_digest, fake)
        self.assertEqual(fake.settings_calls, [])
        self.assertEqual(fake.init_calls, [])

    def test_digest_or_destination_mutation_fails_before_wandb_init(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)

        wrong_digest_client = FakeWandb()
        first, second = self.patches()
        with (
            first,
            second,
            self.assertRaisesRegex(provenance.CodeProvenanceError, "digest mismatch"),
        ):
            provenance.publish(
                publication,
                payload_path,
                "sha256:" + "0" * 64,
                wrong_digest_client,
            )
        self.assertEqual(wrong_digest_client.settings_calls, [])
        self.assertEqual(wrong_digest_client.init_calls, [])

        mutated = json.loads(payload_path.read_bytes())
        mutated["destination"]["entity"] = "different-entity"
        payload_path.write_bytes(
            json.dumps(
                mutated,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        destination_client = FakeWandb()
        first, second = self.patches()
        with (
            first,
            second,
            self.assertRaisesRegex(provenance.CodeProvenanceError, "does not exactly match"),
        ):
            provenance.publish(
                publication,
                payload_path,
                payload_digest,
                destination_client,
            )
        self.assertEqual(destination_client.settings_calls, [])
        self.assertEqual(destination_client.init_calls, [])

    def test_wrong_sdk_version_fails_before_settings_or_init(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()
        fake.__version__ = "different"
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            self.assertRaisesRegex(provenance.CodeProvenanceError, "SDK version changed"),
        ):
            provenance.publish(publication, payload_path, payload_digest, fake)
        self.assertEqual(fake.settings_calls, [])
        self.assertEqual(fake.init_calls, [])

    def test_api_key_and_ambient_wandb_environment_are_fail_closed(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        cases = (
            ({}, "nonempty WANDB_API_KEY"),
            ({"WANDB_API_KEY": "   "}, "nonempty WANDB_API_KEY"),
            (
                {"WANDB_API_KEY": "test-key", "WANDB_SWEEP_ID": "ambient"},
                "ambient W&B variables",
            ),
            (
                {"WANDB_API_KEY": "test-key", "WANDB_RUN_ID": "ambient"},
                "ambient W&B variables",
            ),
            (
                {"WANDB_API_KEY": "test-key", "WANDB_TAGS": "ambient"},
                "ambient W&B variables",
            ),
            (
                {"WANDB_API_KEY": "test-key", "WANDB_CONFIG_PATHS": "ambient"},
                "ambient W&B variables",
            ),
            (
                {"WANDB_API_KEY": "test-key", "WANDB_ERROR_REPORTING": "false"},
                "ambient W&B variables",
            ),
        )
        for environment, message in cases:
            with self.subTest(environment=sorted(environment)):
                fake = FakeWandb()
                first, second = self.patches()
                with (
                    first,
                    second,
                    mock.patch.dict(os.environ, environment, clear=True),
                    self.assertRaisesRegex(provenance.CodeProvenanceError, message),
                ):
                    provenance.publish(publication, payload_path, payload_digest, fake)
                self.assertEqual(fake.settings_calls, [])
                self.assertEqual(fake.init_calls, [])

    def test_ambient_wandb_environment_fails_before_import(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(
                os.environ,
                {"WANDB_API_KEY": "test-key", "WANDB_SWEEP_ID": "ambient"},
                clear=True,
            ),
            mock.patch.object(
                provenance.importlib,
                "import_module",
                side_effect=AssertionError("W&B must not be imported"),
            ),
            self.assertRaisesRegex(SystemExit, "ambient W&B variables"),
        ):
            provenance.main(
                self.argv(
                    "publish",
                    "--payload-file",
                    str(payload_path),
                    "--authorized-payload-digest",
                    payload_digest,
                )
            )

    def test_validate_cli_prints_canonical_success_json(self) -> None:
        stdout = io.StringIO()
        first, second = self.patches()
        with first, second, mock.patch.object(sys, "stdout", stdout):
            self.assertEqual(provenance.main(self.argv("validate")), 0)

        result = json.loads(stdout.getvalue())
        self.assertEqual(stdout.getvalue(), json.dumps(result, sort_keys=True) + "\n")
        self.assertEqual(result["action"], "validate")
        self.assertNotIn("payload", result)
        self.assertNotIn("publication", result)

    def test_export_cli_prints_exact_payload_identity(self) -> None:
        payload_path = self.root / "cli-export.json"
        stdout = io.StringIO()
        first, second = self.patches()
        with first, second, mock.patch.object(sys, "stdout", stdout):
            self.assertEqual(
                provenance.main(self.argv("export", "--payload-file", str(payload_path))),
                0,
            )

        result = json.loads(stdout.getvalue())
        self.assertEqual(stdout.getvalue(), json.dumps(result, sort_keys=True) + "\n")
        self.assertEqual(result["action"], "export")
        self.assertEqual(result["payload"]["path"], str(payload_path))
        self.assertEqual(result["payload"]["bytes"], len(payload_path.read_bytes()))
        self.assertEqual(result["payload"]["digest"], digest(payload_path.read_bytes()))

    def test_publish_cli_prints_committed_terminal_state(self) -> None:
        publication = self.validate()
        payload_path = self.root / "cli-publish.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        stdout = io.StringIO()
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            mock.patch.object(sys, "stdout", stdout),
        ):
            self.assertEqual(
                provenance.main(
                    self.argv(
                        "publish",
                        "--payload-file",
                        str(payload_path),
                        "--authorized-payload-digest",
                        payload_digest,
                    ),
                    wandb_client=fake,
                ),
                0,
            )

        result = json.loads(stdout.getvalue())
        self.assertEqual(stdout.getvalue(), json.dumps(result, sort_keys=True) + "\n")
        self.assertEqual(
            result["publication"],
            {
                "state": "committed",
                "run_id": payload["destination"]["id"],
                "resolution": "update_ack",
                "retry_forbidden": True,
            },
        )

    def test_publish_cli_failure_preserves_outcome_uncertain_state(self) -> None:
        publication = self.validate()
        payload_path = self.root / "cli-uncertain.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()
        fake.remote_update_exception = RuntimeError("simulated ambiguous update")
        fake.api_run_failure_calls = {2}
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            self.assertRaisesRegex(SystemExit, "never retry or delete") as caught,
        ):
            provenance.main(
                self.argv(
                    "publish",
                    "--payload-file",
                    str(payload_path),
                    "--authorized-payload-digest",
                    payload_digest,
                ),
                wandb_client=fake,
            )

        cause = caught.exception.__cause__
        self.assertIsInstance(cause, provenance.WandbOutcomeUncertainError)
        self.assertEqual(cause.state, provenance.WandbPublicationState.OUTCOME_UNCERTAIN)
        self.assertTrue(cause.retry_forbidden)

    def test_missing_cli_authorization_fails_before_wandb_import(self) -> None:
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(
                provenance.importlib,
                "import_module",
                side_effect=AssertionError("W&B must not be imported"),
            ),
            self.assertRaisesRegex(SystemExit, "requires --payload-file"),
        ):
            provenance.main(self.argv("publish"))

    def test_export_refuses_existing_regular_file_and_symlink(self) -> None:
        publication = self.validate()
        existing = self.root / "existing.json"
        existing.write_bytes(b"keep")
        first, second = self.patches()
        with (
            first,
            second,
            self.assertRaisesRegex(provenance.CodeProvenanceError, "already exists"),
        ):
            provenance.export_payload(publication, existing)
        self.assertEqual(existing.read_bytes(), b"keep")

        target = self.root / "target.json"
        target.write_bytes(b"target")
        linked = self.root / "linked.json"
        linked.symlink_to(target)
        first, second = self.patches()
        with (
            first,
            second,
            self.assertRaisesRegex(provenance.CodeProvenanceError, "already exists"),
        ):
            provenance.export_payload(publication, linked)
        self.assertEqual(target.read_bytes(), b"target")

    def test_python_equal_numeric_mutation_cannot_bypass_exact_bytes(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        self.assertEqual(publication.metrics[0]["loss"], 1.0)
        publication.metrics[0]["loss"] = 1
        fake = FakeWandb()
        first, second = self.patches()
        with (
            first,
            second,
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "publication inputs changed after validation",
            ),
        ):
            provenance.publish(publication, payload_path, payload_digest, fake)
        self.assertEqual(fake.settings_calls, [])
        self.assertEqual(fake.init_calls, [])

    def test_training_metadata_is_byte_bound_to_loaded_lineage(self) -> None:
        self.training_lineage["receipt"]["digest"] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(
            provenance.CodeProvenanceError,
            "training metadata bytes differ from the validated lineage",
        ):
            self.validate()

    def test_training_metadata_is_bound_to_deep_validated_run_bytes(self) -> None:
        self.training_lineage["run"]["training_metadata"]["digest"] = "sha256:" + "e" * 64
        with self.assertRaisesRegex(
            provenance.CodeProvenanceError,
            "training metadata bytes differ from the validated lineage",
        ):
            self.validate()

    def test_training_metrics_are_bound_to_deep_validated_run_bytes(self) -> None:
        self.training_lineage["run"]["metrics"]["bytes"] += 1
        with self.assertRaisesRegex(
            provenance.CodeProvenanceError,
            "training metrics bytes differ from the validated lineage",
        ):
            self.validate()

    def test_read_regular_rejects_inode_swap_between_check_and_open(self) -> None:
        victim = self.root / "victim.json"
        replacement = self.root / "replacement.json"
        victim.write_bytes(b"{}")
        replacement.write_bytes(b"0123456789abcdef")
        real_open = provenance.os.open
        swapped = False

        def swap_then_open(path: object, flags: int, *args: object) -> int:
            nonlocal swapped
            if Path(path) == victim and not swapped:
                swapped = True
                victim.unlink()
                replacement.replace(victim)
            return real_open(path, flags, *args)

        with (
            mock.patch.object(provenance.os, "open", side_effect=swap_then_open),
            self.assertRaisesRegex(provenance.CodeProvenanceError, "changed while being opened"),
        ):
            provenance._read_regular(victim, "swapped file", maximum=4)

    def test_failed_export_removes_only_partial_output(self) -> None:
        publication = self.validate()
        output = self.root / "partial.json"
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(provenance.os, "write", side_effect=OSError("simulated")),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "could not be published safely",
            ),
        ):
            provenance.export_payload(publication, output)
        self.assertFalse(output.exists())

    def test_atomic_export_never_unlinks_a_racing_destination(self) -> None:
        publication = self.validate()
        output = self.root / "racing.json"
        real_link = provenance.os.link

        def replace_before_link(
            source: object,
            destination: object,
            *,
            follow_symlinks: bool,
        ) -> None:
            Path(destination).write_bytes(b"replacement")
            real_link(source, destination, follow_symlinks=follow_symlinks)

        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(provenance.os, "link", side_effect=replace_before_link),
            self.assertRaisesRegex(provenance.CodeProvenanceError, "already exists"),
        ):
            provenance.export_payload(publication, output)
        self.assertEqual(output.read_bytes(), b"replacement")
        self.assertEqual(
            [entry.name for entry in self.root.iterdir() if ".staging-" in entry.name],
            [],
        )

    def test_post_link_verification_failure_preserves_destination_as_uncertain(self) -> None:
        publication = self.validate()
        output = self.root / "post-link-verification.json"
        real_link = provenance.os.link

        def link_then_change_mode(
            source: object,
            destination: object,
            *,
            follow_symlinks: bool,
        ) -> None:
            real_link(source, destination, follow_symlinks=follow_symlinks)
            Path(destination).chmod(0o640)

        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(provenance.os, "link", side_effect=link_then_change_mode),
            self.assertRaisesRegex(
                provenance.PayloadExportOutcomeUncertainError,
                "destination was intentionally preserved",
            ),
        ):
            provenance.export_payload(publication, output)
        self.assertTrue(output.exists())
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)

    def test_post_link_parent_fsync_failure_preserves_destination_as_uncertain(self) -> None:
        publication = self.validate()
        output = self.root / "post-link-fsync.json"
        real_fsync = provenance.os.fsync
        directory_calls = 0

        def fail_first_directory_fsync(descriptor: int) -> None:
            nonlocal directory_calls
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                directory_calls += 1
                if directory_calls == 1:
                    raise OSError("simulated parent fsync failure")
            real_fsync(descriptor)

        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(provenance.os, "fsync", side_effect=fail_first_directory_fsync),
            self.assertRaisesRegex(
                provenance.PayloadExportOutcomeUncertainError,
                "destination was intentionally preserved",
            ),
        ):
            provenance.export_payload(publication, output)
        self.assertEqual(directory_calls, 1)
        self.assertTrue(output.exists())

    def test_durable_export_staging_cleanup_failure_is_committed_nonretryable(self) -> None:
        publication = self.validate()
        output = self.root / "durable-cleanup-failure.json"
        real_temporary_directory = provenance.tempfile.TemporaryDirectory

        class CleanupFailure:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.inner = real_temporary_directory(*args, **kwargs)

            def __enter__(self) -> str:
                return self.inner.__enter__()

            def __exit__(self, *args: object) -> None:
                self.inner.__exit__(*args)
                raise RuntimeError("simulated staging cleanup failure")

        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(
                provenance.tempfile,
                "TemporaryDirectory",
                CleanupFailure,
            ),
            self.assertRaises(provenance.PayloadExportPostCommitError) as caught,
        ):
            provenance.export_payload(publication, output)

        self.assertEqual(caught.exception.state, "committed")
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertTrue(output.exists())
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(json.loads(output.read_bytes())["schema"], provenance.WANDB_PAYLOAD_SCHEMA)

    def test_post_link_failure_has_no_lstat_then_unlink_replacement_window(self) -> None:
        publication = self.validate()
        output = self.root / "post-link-no-unlink.json"
        replacement = self.root / "post-link-racing-replacement.json"
        replacement.write_bytes(b"replacement")
        real_link = provenance.os.link
        real_unlink = Path.unlink
        replacement_race_ran = False

        def link_then_change_mode(
            source: object,
            destination: object,
            *,
            follow_symlinks: bool,
        ) -> None:
            real_link(source, destination, follow_symlinks=follow_symlinks)
            Path(destination).chmod(0o640)

        def replace_between_lstat_and_unlink(
            candidate: Path, *args: object, **kwargs: object
        ) -> None:
            nonlocal replacement_race_ran
            if candidate == output:
                replacement_race_ran = True
                os.replace(replacement, output)
            real_unlink(candidate, *args, **kwargs)

        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(provenance.os, "link", side_effect=link_then_change_mode),
            mock.patch.object(Path, "unlink", new=replace_between_lstat_and_unlink),
            self.assertRaises(provenance.PayloadExportOutcomeUncertainError),
        ):
            provenance.export_payload(publication, output)

        self.assertFalse(replacement_race_ran)
        self.assertTrue(output.exists())
        self.assertEqual(replacement.read_bytes(), b"replacement")

    def test_post_link_replacement_race_is_preserved_and_outcome_is_uncertain(self) -> None:
        publication = self.validate()
        output = self.root / "post-link-replacement.json"

        def replace_before_cleanup(_parent: Path) -> None:
            output.unlink()
            output.write_bytes(b"replacement")
            raise OSError("simulated post-link durability failure")

        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.object(
                provenance,
                "_fsync_directory",
                side_effect=replace_before_cleanup,
            ),
            self.assertRaises(provenance.PayloadExportOutcomeUncertainError) as caught,
        ):
            provenance.export_payload(publication, output)
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(caught.exception.state, "outcome_uncertain")
        self.assertEqual(output.read_bytes(), b"replacement")

    def test_failed_pending_run_is_not_discoverable_or_admissible_by_official_store(self) -> None:
        from microtensor.provenance import record as official_record
        from microtensor.provenance.wandb_store import WandbStore

        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.fail_log_step = 2
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            self.assertRaisesRegex(
                provenance.WandbPendingFailedError,
                "deterministic run ID",
            ) as caught,
        ):
            provenance.publish(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.PENDING_FAILED)
        self.assertEqual(caught.exception.run_id, payload["destination"]["id"])
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(fake.created_run.finish_exit_codes, [1])
        self.assertEqual(fake.created_run.state, "failed")
        self.assertEqual(fake.created_run.name, payload["destination"]["pending_name"])
        self.assertNotEqual(fake.created_run.name, provenance.HOTKEY)
        self.assertEqual(fake.remote_update_names, [])

        class OfficialApi:
            def runs(self, _path: str, *, filters: dict[str, str], per_page: int) -> list[object]:
                del per_page
                return (
                    [fake.created_run] if filters["display_name"] == fake.created_run.name else []
                )

        candidates = WandbStore(api=OfficialApi()).candidates(provenance.HOTKEY)
        self.assertEqual(candidates, [])
        verdict = official_record.best_verdict(
            candidates,
            hotkey=provenance.HOTKEY,
            artifact_digest=publication.artifact["tree_digest"],
            track=provenance.TRACK,
            hardware_class=provenance.HARDWARE_CLASS,
            allowed_base_models=frozenset({provenance.BASE_MODEL}),
        )
        self.assertFalse(verdict.admissible)

    def test_final_name_is_not_committed_until_remote_readback_succeeds(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.remote_config_override = {"not": "authorized"}
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "application config differs",
            ),
        ):
            provenance.publish(publication, payload_path, payload_digest, fake)
        self.assertEqual(fake.created_run.finish_exit_codes, [0])
        self.assertEqual(fake.created_run.name, payload["destination"]["pending_name"])
        self.assertEqual(fake.remote_update_names, [])

    def test_final_name_is_not_committed_when_raw_application_config_differs(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.remote_rawconfig_override = {
            **payload["config"],
            "mt_track": "different",
            "_wandb": {"m": [], "start_time": 0, "t": {}},
        }
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "raw application config differs",
            ),
        ):
            provenance.publish(publication, payload_path, payload_digest, fake)
        self.assertEqual(fake.created_run.finish_exit_codes, [0])
        self.assertEqual(fake.created_run.name, payload["destination"]["pending_name"])
        self.assertEqual(fake.remote_update_names, [])

    def test_update_exception_with_exact_final_readback_is_committed(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.remote_update_exception = RuntimeError("simulated lost update acknowledgement")
        fake.remote_update_applies_before_exception = True

        outcome = self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(outcome.state, provenance.WandbPublicationState.COMMITTED)
        self.assertEqual(outcome.resolution, "exact_readback_after_update_exception")
        self.assertEqual(outcome.run_id, payload["destination"]["id"])
        self.assertEqual(
            fake.readback_names,
            [payload["destination"]["pending_name"], provenance.HOTKEY],
        )

    def test_update_exception_with_exact_pending_readback_is_retry_forbidden(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.remote_update_exception = RuntimeError("simulated rejected update")

        with self.assertRaises(provenance.WandbPendingFailedError) as caught:
            self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.PENDING_FAILED)
        self.assertEqual(caught.exception.run_id, payload["destination"]["id"])
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(fake.created_run.name, payload["destination"]["pending_name"])

    def test_update_exception_with_failed_resolution_is_outcome_uncertain(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.remote_update_exception = RuntimeError("simulated ambiguous update")
        fake.api_run_failure_calls = {2}

        with self.assertRaises(provenance.WandbOutcomeUncertainError) as caught:
            self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.OUTCOME_UNCERTAIN)
        self.assertEqual(caught.exception.run_id, payload["destination"]["id"])
        self.assertTrue(caught.exception.retry_forbidden)

    def test_update_ack_then_second_readback_failure_remains_committed(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        payload = json.loads(payload_path.read_bytes())
        fake = FakeWandb()
        fake.api_run_failure_calls = {2}

        with self.assertRaises(provenance.WandbPostCommitError) as caught:
            self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.COMMITTED)
        self.assertEqual(caught.exception.run_id, payload["destination"]["id"])
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(fake.created_run.name, provenance.HOTKEY)

    def test_postcommit_teardown_failure_remains_committed(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()
        fake.teardown_exception = RuntimeError("simulated teardown failure")

        with self.assertRaises(provenance.WandbPostCommitError) as caught:
            self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.COMMITTED)
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(fake.created_run.name, provenance.HOTKEY)

    def test_postcommit_temporary_root_cleanup_failure_remains_committed(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()
        real_temporary_directory = provenance.tempfile.TemporaryDirectory

        class CleanupFailure:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.inner = real_temporary_directory(*args, **kwargs)

            def __enter__(self) -> str:
                return self.inner.__enter__()

            def __exit__(self, *args: object) -> None:
                self.inner.__exit__(*args)
                raise RuntimeError("simulated temporary-root cleanup failure")

        with (
            mock.patch.object(
                provenance.tempfile,
                "TemporaryDirectory",
                CleanupFailure,
            ),
            self.assertRaises(provenance.WandbPostCommitError) as caught,
        ):
            self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.COMMITTED)
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(fake.created_run.name, provenance.HOTKEY)

    def test_postcommit_environment_cleanup_failure_remains_committed(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()

        with (
            mock.patch.object(
                provenance,
                "_restore_wandb_environment",
                side_effect=RuntimeError("simulated environment cleanup failure"),
            ),
            self.assertRaises(provenance.WandbPostCommitError) as caught,
        ):
            self.publish_fake(publication, payload_path, payload_digest, fake)

        self.assertEqual(caught.exception.state, provenance.WandbPublicationState.COMMITTED)
        self.assertTrue(caught.exception.retry_forbidden)
        self.assertEqual(fake.created_run.name, provenance.HOTKEY)

    def test_preimported_real_wandb_client_is_rejected(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()
        fake.__name__ = "wandb"
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "pre-imported real W&B client",
            ),
        ):
            provenance.publish(publication, payload_path, payload_digest, fake)
        self.assertEqual(fake.settings_calls, [])
        self.assertEqual(fake.init_calls, [])

    def test_real_sdk_distribution_preflight_precedes_import(self) -> None:
        with (
            mock.patch.object(
                provenance.importlib.metadata,
                "version",
                return_value="different",
            ),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "distribution version changed",
            ),
        ):
            provenance._preflight_real_wandb_import()

    def test_preimported_sentry_sdk_is_rejected_before_real_wandb_import(self) -> None:
        with (
            mock.patch.dict(provenance.sys.modules, {"sentry_sdk": mock.Mock()}),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "Sentry was imported before isolated W&B publication",
            ),
        ):
            provenance._preflight_real_wandb_import()

    def test_real_sdk_error_reporting_must_be_disabled(self) -> None:
        fake_real = mock.Mock(__name__="wandb")
        fake_env = mock.Mock(error_reporting_enabled=lambda: True)
        fake_analytics = mock.Mock(get_sentry=lambda: mock.Mock(_enabled=True))
        with (
            mock.patch.object(
                provenance.importlib,
                "import_module",
                side_effect=(fake_env, fake_analytics),
            ),
            self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "third-party error reporting is not disabled",
            ),
        ):
            provenance._require_disabled_wandb_error_reporting(fake_real)

    def test_environment_is_restored_when_settings_factory_fails(self) -> None:
        publication = self.validate()
        payload_path = self.root / "wandb.json"
        _size, payload_digest = self.export(publication, payload_path)
        fake = FakeWandb()

        def failing_settings(**_kwargs: object) -> FakeSettings:
            os.environ["WANDB_FACTORY_LEAK"] = "leak"
            raise RuntimeError("simulated")

        fake.Settings = failing_settings  # type: ignore[method-assign]
        first, second = self.patches()
        with (
            first,
            second,
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=True),
        ):
            with self.assertRaisesRegex(
                provenance.CodeProvenanceError,
                "privacy settings were refused",
            ):
                provenance.publish(publication, payload_path, payload_digest, fake)
            self.assertEqual(dict(os.environ), {"WANDB_API_KEY": "test-key"})

    def prepare_calibrated_v3(self) -> provenance.PublicationRequest:
        from training import convert_code_gguf as converter

        self.load = {**self.load, "quantization": "Q4_K_M"}
        merged = self.run / "merged"
        merged.mkdir()
        bundle = self.root / "calibrated-bundle"
        bundle.mkdir()
        artifact = bundle / converter.ARTIFACT_NAME
        artifact.mkdir()
        (artifact / converter.ENTRYPOINT).write_bytes(b"fake calibrated gguf")
        load_path = bundle / converter.LOAD_SPEC_NAME
        write_json(load_path, self.load)

        llama_cpp = self.root / "llama.cpp"
        bin_root = llama_cpp / "build" / "bin"
        bin_root.mkdir(parents=True)
        converter_path = llama_cpp / "convert_hf_to_gguf.py"
        imatrix_path = bin_root / "llama-imatrix"
        quantizer_path = bin_root / "llama-quantize"
        for path in (converter_path, imatrix_path, quantizer_path):
            path.write_bytes(path.name.encode())
            path.chmod(0o755)
        current_dataset = self.root / "current-dataset"
        current_dataset.mkdir()
        current_source = self.root / "current-source.json"
        current_source.write_text("{}", encoding="utf-8")

        self.v3_rows = [
            {
                "ref": f"row-{index:03d}",
                "prompt": f"private-calibration-prompt-{index:03d}",
                "completion": f"private-calibration-completion-{index:03d}",
                "max_output_tokens": 1,
            }
            for index in range(converter.CALIBRATION_TOTAL_ROWS)
        ]
        self.v3_selection = {
            "algorithm": converter.CALIBRATION_SELECTION_ALGORITHM,
            "seed": converter.CALIBRATION_SEED,
            "current_rows": converter.CALIBRATION_CURRENT_ROWS,
            "current_refs_digest": "sha256:" + "1" * 64,
            "diagnostic_rows_excluded": converter.CALIBRATION_DIAGNOSTIC_ROWS,
            "diagnostic_refs_digest": "sha256:" + "2" * 64,
            "historical_pool_rows": 8_000,
            "historical_selected_rows": converter.CALIBRATION_HISTORICAL_ROWS,
            "historical_selected_refs_digest": "sha256:" + "3" * 64,
            "total_rows": converter.CALIBRATION_TOTAL_ROWS,
        }
        self.v3_source = {
            "current": {
                "corpus": {
                    "bytes": 2,
                    "digest": digest(b"{}"),
                    "canonical_bytes": 2,
                    "canonical_digest": digest(b"{}"),
                    "task_count": 94,
                    "refs_digest": "sha256:" + "4" * 64,
                },
                "prepared_dataset": {"tree_digest": "sha256:" + "5" * 64},
            },
            "historical": {
                "corpus": {"raw_digest": "sha256:" + "6" * 64},
                "prepared_dataset": {"tree_digest": "sha256:" + "7" * 64},
            },
        }
        self.v3_material = {
            "profile": converter.CALIBRATION_PROFILE,
            "source": self.v3_source,
            "selection": self.v3_selection,
        }
        runtime_namespace: set[str] = set()
        for path, _byte_count, _tool_digest in converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT:
            runtime_namespace.add(path)
        for (
            loader,
            target,
            _byte_count,
            _library_digest,
        ) in converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT:
            runtime_namespace.update((loader, target))
        for path, target in converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT:
            runtime_namespace.add(path)
            runtime_namespace.add(f"build/bin/{target}")
        runtime_libraries = {
            "schema": converter.RUNTIME_LIBRARY_SCHEMA,
            "root": str(llama_cpp.resolve()),
            "directories": [{"path": path, "mode": "0755"} for path in (".", "build", "build/bin")],
            "build_bin_namespace": sorted(runtime_namespace),
            "symlinks": [
                {"path": path, "target": target}
                for path, target in converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT
            ],
            "executables": [
                {
                    "path": path,
                    "bytes": byte_count,
                    "digest": executable_digest,
                    "mode": "0755",
                }
                for path, byte_count, executable_digest in (
                    converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
                )
            ],
            "libraries": [
                {
                    "loader_path": loader,
                    "target_path": target,
                    "bytes": byte_count,
                    "digest": library_digest,
                    "mode": "0755",
                }
                for loader, target, byte_count, library_digest in (
                    converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT
                )
            ],
        }
        self.v3_toolchain = {
            "root": str(llama_cpp.resolve()),
            "revision": provenance.LLAMA_CPP_REVISION,
            "converter": {
                "path": str(converter_path.resolve()),
                "digest": "sha256:" + "8" * 64,
            },
            "imatrix": {
                "path": str(imatrix_path.resolve()),
                "digest": "sha256:" + "9" * 64,
            },
            "quantizer": {
                "path": str(quantizer_path.resolve()),
                "digest": "sha256:" + "a" * 64,
            },
            "runtime_libraries": runtime_libraries,
        }
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": ".microtensor-empty-pycache",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        empty_stream = {
            "bytes": 0,
            "captured_bytes": 0,
            "captured_digest": digest(b""),
            "digest": digest(b""),
            "truncated": False,
        }
        command_argv = (
            (
                "convert_f16",
                [
                    str(converter_path.resolve()),
                    str(merged.resolve()),
                    "--outfile",
                    converter.F16_NAME,
                    "--outtype",
                    "f16",
                ],
            ),
            (
                "calibrate_imatrix",
                [
                    str(imatrix_path.resolve()),
                    "--offline",
                    "--model",
                    converter.F16_NAME,
                    "--file",
                    converter.CALIBRATION_CORPUS_NAME,
                    "--output",
                    converter.IMATRIX_NAME,
                    "--output-format",
                    "gguf",
                    "--ctx-size",
                    str(converter.CALIBRATION_CONTEXT_TOKENS),
                    "--chunks",
                    str(converter.CALIBRATION_CHUNKS),
                    "--batch-size",
                    "512",
                    "--ubatch-size",
                    "512",
                    "--threads",
                    "1",
                    "--threads-batch",
                    "1",
                    "--device",
                    "none",
                    "--gpu-layers",
                    "0",
                    "--fit",
                    "off",
                    "--flash-attn",
                    "off",
                    "--no-ppl",
                    "--parse-special",
                    "--output-frequency",
                    str(converter.CALIBRATION_CHUNKS + 1),
                    "--save-frequency",
                    "0",
                ],
            ),
            (
                "quantize",
                [
                    str(quantizer_path.resolve()),
                    "--imatrix",
                    converter.IMATRIX_NAME,
                    converter.F16_NAME,
                    f"{converter.ARTIFACT_NAME}/{converter.ENTRYPOINT}",
                    "Q4_K_M",
                    "1",
                ],
            ),
        )

        def commands(role: str, first_start: int) -> list[dict[str, object]]:
            return [
                {
                    "name": name,
                    "argv": argv,
                    "cwd_role": role,
                    "environment": copy.deepcopy(environment),
                    "returncode": 0,
                    "started_at_unix_ns": first_start + index * 2,
                    "finished_at_unix_ns": first_start + index * 2 + 1,
                    "stdout": copy.deepcopy(empty_stream),
                    "stderr": copy.deepcopy(empty_stream),
                }
                for index, (name, argv) in enumerate(command_argv)
            ]

        primary_commands = commands("private_staging", 1)
        replay_commands = commands("determinism_replay", 7)
        f16_digest = "sha256:" + "b" * 64
        imatrix_digest = "sha256:" + "c" * 64
        model_metadata = {
            "imatrix_file": converter.IMATRIX_NAME,
            "imatrix_dataset": converter.CALIBRATION_CORPUS_NAME,
            "imatrix_entries_count": 64,
            "imatrix_chunks_count": converter.CALIBRATION_CHUNKS,
        }
        self.v3_model_metadata = model_metadata
        replay = {
            "schema": provenance.DETERMINISM_REPLAY_SCHEMA,
            "commands": replay_commands,
            "f16_digest": f16_digest,
            "imatrix_digest": imatrix_digest,
            "entrypoint_digest": self.artifact_identity["entrypoint"]["digest"],
            "entrypoint_bytes": self.artifact_identity["entrypoint"]["bytes"],
            "artifact_tree_digest": self.artifact_identity["tree_digest"],
            "matches_primary": True,
        }
        corpus_identity = provenance._rendered_calibration_identity(self.v3_rows, converter)
        self.v3_calibration = {
            "schema": converter.CALIBRATION_SCHEMA,
            "status": "complete",
            "profile": converter.CALIBRATION_PROFILE,
            "track": provenance.TRACK,
            "hardware_class": provenance.HARDWARE_CLASS,
            "base_model": provenance.BASE_MODEL,
            "llama_cpp_revision": provenance.LLAMA_CPP_REVISION,
            "source": copy.deepcopy(self.v3_source),
            "selection": copy.deepcopy(self.v3_selection),
            "rendering": {
                "schema": converter.CALIBRATION_RENDER_SCHEMA,
                "encoding": "UTF-8",
                "expression": "prompt + completion + <|im_end|> + LF",
                "eos_token": converter.CALIBRATION_EOS_TOKEN,
                "eos_token_id": converter.CALIBRATION_EOS_TOKEN_ID,
                "rows": converter.CALIBRATION_TOTAL_ROWS,
                "corpus": corpus_identity,
            },
            "toolchain": {
                "converter_digest": self.v3_toolchain["converter"]["digest"],
                "imatrix_digest": self.v3_toolchain["imatrix"]["digest"],
                "quantizer_digest": self.v3_toolchain["quantizer"]["digest"],
                "runtime_libraries": copy.deepcopy(runtime_libraries),
            },
            "commands": copy.deepcopy(primary_commands),
            "determinism_replay": copy.deepcopy(replay),
            "intermediate": {
                "f16": {"bytes": 100, "digest": f16_digest, "file_type": 1},
                "imatrix": {
                    "bytes": 200,
                    "digest": imatrix_digest,
                    "version": 3,
                    "tensor_count": 128,
                    "entries_count": 64,
                    "datasets": [converter.CALIBRATION_CORPUS_NAME],
                    "chunk_count": converter.CALIBRATION_CHUNKS,
                    "chunk_size": converter.CALIBRATION_CONTEXT_TOKENS,
                },
            },
            "artifact": {
                "tree_digest": self.artifact_identity["tree_digest"],
                "entrypoint_digest": self.artifact_identity["entrypoint"]["digest"],
                "entrypoint_bytes": self.artifact_identity["entrypoint"]["bytes"],
                "quantization": "Q4_K_M",
                "calibration_metadata": copy.deepcopy(model_metadata),
            },
            "load_manifest": copy.deepcopy(self.load),
        }
        self.v3_calibration_path = bundle / converter.CALIBRATION_RECEIPT_NAME
        write_json(self.v3_calibration_path, self.v3_calibration)
        self.v3_conversion = {
            "schema": converter.CALIBRATED_CONVERSION_SCHEMA,
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
                "converter_digest": self.v3_toolchain["converter"]["digest"],
                "imatrix_digest": self.v3_toolchain["imatrix"]["digest"],
                "quantizer_digest": self.v3_toolchain["quantizer"]["digest"],
                "runtime_libraries": copy.deepcopy(runtime_libraries),
                "commands": copy.deepcopy(primary_commands),
                "determinism_replay": copy.deepcopy(replay),
            },
            "artifact": {
                "tree_digest": self.artifact_identity["tree_digest"],
                "entrypoint_digest": self.artifact_identity["entrypoint"]["digest"],
                "entrypoint_bytes": self.artifact_identity["entrypoint"]["bytes"],
                "quantization": "Q4_K_M",
            },
            "load_manifest": copy.deepcopy(self.load),
            "calibration_receipt_digest": digest(self.v3_calibration_path.read_bytes()),
        }
        self.v3_conversion_path = bundle / converter.RECEIPT_NAME
        write_json(self.v3_conversion_path, self.v3_conversion)
        self.artifact_identity = copy.deepcopy(self.artifact_identity)
        self.artifact_identity["root"] = str(artifact.resolve())
        self.v3_request = replace(
            self.request,
            artifact=artifact,
            load_spec=load_path,
            conversion_receipt=self.v3_conversion_path,
            calibration_receipt=self.v3_calibration_path,
            llama_cpp=llama_cpp,
            calibration_current_dataset=current_dataset,
            calibration_current_source_corpus=current_source,
        )
        return self.v3_request

    def prepare_calibrated_v5(self) -> provenance.PublicationRequest:
        self.configure_normalized_v6()
        request = self.prepare_calibrated_v3()
        self.v3_selection["historical_pool_rows"] = normalized_candidate.EXPECTED_TRAIN_EXAMPLES
        self.v3_source["historical"] = {
            "corpus": normalized_candidate.source_corpus_identity(),
            "prepared_dataset": {
                "tree_digest": "sha256:" + "7" * 64,
                "manifest": copy.deepcopy(self.manifest),
                "excluded_refs_file": {
                    "bytes": normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
                    "digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
                },
            },
        }
        self.v3_calibration["source"] = copy.deepcopy(self.v3_source)
        self.v3_calibration["selection"] = copy.deepcopy(self.v3_selection)
        self.v3_conversion["schema"] = provenance.NORMALIZED_CALIBRATED_CONVERSION_SCHEMA
        self.v3_conversion["source"] = provenance._expected_conversion_source(self.training_lineage)
        self.rewrite_v3_receipts()
        return request

    def rewrite_v3_receipts(self) -> None:
        write_json(self.v3_calibration_path, self.v3_calibration)
        self.v3_conversion["calibration_receipt_digest"] = digest(
            self.v3_calibration_path.read_bytes()
        )
        write_json(self.v3_conversion_path, self.v3_conversion)

    def validate_v3(self, *, pinned_root: Path | None = None) -> provenance.Publication:
        from training import convert_code_gguf as converter

        if pinned_root is None:
            pinned_root = self.v3_request.llama_cpp.resolve(strict=True)
        patches = (
            *self.patches(),
            mock.patch.object(provenance, "LLAMA_CPP_ROOT", pinned_root),
            mock.patch.object(converter, "LLAMA_CPP_ROOT", pinned_root),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                return_value=self.v3_toolchain,
            ),
            mock.patch.object(
                converter,
                "_load_calibration_material",
                return_value=(self.v3_rows, self.v3_material),
            ),
            mock.patch.object(
                converter,
                "_validate_calibrated_model_metadata",
                return_value=self.v3_model_metadata,
            ),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return provenance.validate_publication(self.v3_request)

    def test_calibrated_v3_is_fully_bound_without_raw_calibration_text(self) -> None:
        self.prepare_calibrated_v3()
        publication = self.validate_v3()
        self.assertEqual(publication.conversion["schema"], provenance.CALIBRATED_CONVERSION_SCHEMA)
        self.assertEqual(publication.calibration["schema"], provenance.IMATRIX_CALIBRATION_SCHEMA)
        self.assertEqual(
            publication.calibration["receipt"]["selection"]["historical_selected_rows"],
            434,
        )
        public_config = json.dumps(provenance._wandb_config(publication), sort_keys=True)
        self.assertNotIn("private-calibration-prompt-000", public_config)
        self.assertNotIn("private-calibration-completion-000", public_config)
        self.assertNotIn('"prompt":', public_config)
        self.assertNotIn('"completion":', public_config)
        self.assertEqual(
            publication.conversion["conversion"]["runtime_libraries"],
            self.v3_toolchain["runtime_libraries"],
        )

    def test_normalized_calibrated_v5_binds_profile_exclusions_and_pool(self) -> None:
        self.prepare_calibrated_v5()
        publication = self.validate_v3()
        self.assertEqual(
            publication.conversion["schema"],
            provenance.NORMALIZED_CALIBRATED_CONVERSION_SCHEMA,
        )
        self.assertEqual(
            publication.conversion["source"],
            provenance._expected_conversion_source(self.training_lineage),
        )
        self.assertEqual(
            publication.calibration["receipt"]["selection"]["historical_pool_rows"],
            normalized_candidate.EXPECTED_TRAIN_EXAMPLES,
        )
        self.assertEqual(
            provenance._wandb_config(publication)["mt_quality_claim"],
            normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        )

    def test_normalized_calibrated_v5_rejects_legacy_v3_schema_swap(self) -> None:
        self.prepare_calibrated_v5()
        self.v3_conversion["schema"] = provenance.CALIBRATED_CONVERSION_SCHEMA
        self.v3_conversion["source"] = {
            "training_metadata_digest": self.training_lineage["receipt"]["digest"],
            "merged_tree_digest": self.training_lineage["run"]["merged"]["digest"],
        }
        self.rewrite_v3_receipts()
        with self.assertRaisesRegex(
            provenance.CodeProvenanceError,
            "schema crosses training lineages",
        ):
            self.validate_v3()

    def test_obsolete_calibrated_v2_is_ineligible(self) -> None:
        self.prepare_calibrated_v3()
        self.v3_conversion["schema"] = provenance.OBSOLETE_CALIBRATED_CONVERSION_SCHEMA
        write_json(self.v3_conversion_path, self.v3_conversion)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "lacks the pinned"):
            self.validate_v3()

    def test_calibrated_v3_rejects_source_selection_and_corpus_changes(self) -> None:
        self.prepare_calibrated_v3()
        original_calibration = copy.deepcopy(self.v3_calibration)
        original_conversion = copy.deepcopy(self.v3_conversion)

        def source_change(calibration: dict[str, object], _conversion: dict[str, object]) -> None:
            calibration["source"]["current"]["corpus"]["digest"] = "sha256:" + "d" * 64

        def selection_change(
            calibration: dict[str, object], _conversion: dict[str, object]
        ) -> None:
            calibration["selection"]["current_rows"] = 77

        def corpus_change(calibration: dict[str, object], _conversion: dict[str, object]) -> None:
            calibration["rendering"]["corpus"]["digest"] = "sha256:" + "e" * 64

        for label, mutate in (
            ("source", source_change),
            ("selection", selection_change),
            ("corpus", corpus_change),
        ):
            with self.subTest(label=label):
                self.v3_calibration = copy.deepcopy(original_calibration)
                self.v3_conversion = copy.deepcopy(original_conversion)
                mutate(self.v3_calibration, self.v3_conversion)
                self.rewrite_v3_receipts()
                with self.assertRaises(provenance.CodeProvenanceError):
                    self.validate_v3()

    def test_calibrated_v3_rejects_tool_command_log_and_numeric_mutations(self) -> None:
        self.prepare_calibrated_v3()
        original_calibration = copy.deepcopy(self.v3_calibration)
        original_conversion = copy.deepcopy(self.v3_conversion)

        def tool_change(calibration: dict[str, object], _conversion: dict[str, object]) -> None:
            calibration["toolchain"]["imatrix_digest"] = "sha256:" + "d" * 64

        def argv_change(calibration: dict[str, object], _conversion: dict[str, object]) -> None:
            calibration["commands"][1]["argv"][-1] = "unexpected"

        def log_change(calibration: dict[str, object], _conversion: dict[str, object]) -> None:
            calibration["commands"][0]["stdout"]["captured_digest"] = "sha256:" + "d" * 64

        def numeric_change(calibration: dict[str, object], conversion: dict[str, object]) -> None:
            calibration["commands"][0]["started_at_unix_ns"] = 1.0
            conversion["conversion"]["commands"][0]["started_at_unix_ns"] = 1.0

        for label, mutate in (
            ("tool", tool_change),
            ("argv", argv_change),
            ("log", log_change),
            ("integer-to-float", numeric_change),
        ):
            with self.subTest(label=label):
                self.v3_calibration = copy.deepcopy(original_calibration)
                self.v3_conversion = copy.deepcopy(original_conversion)
                mutate(self.v3_calibration, self.v3_conversion)
                self.rewrite_v3_receipts()
                with self.assertRaises(provenance.CodeProvenanceError):
                    self.validate_v3()

    def test_calibrated_v3_rejects_every_runtime_closure_mutation(self) -> None:
        self.prepare_calibrated_v3()
        original_calibration = copy.deepcopy(self.v3_calibration)
        original_conversion = copy.deepcopy(self.v3_conversion)
        for mutation in (
            "root",
            "extra-backend",
            "symlink",
            "executable-digest",
            "executable-float-bytes",
            "executable-mode",
            "library-digest",
            "library-float-bytes",
            "library-mode",
        ):
            with self.subTest(mutation=mutation):
                self.v3_calibration = copy.deepcopy(original_calibration)
                self.v3_conversion = copy.deepcopy(original_conversion)
                for container in (
                    self.v3_calibration["toolchain"],
                    self.v3_conversion["conversion"],
                ):
                    closure = container["runtime_libraries"]
                    if mutation == "root":
                        closure["root"] = str(self.root / "different-llama.cpp")
                    elif mutation == "extra-backend":
                        closure["build_bin_namespace"].append("build/bin/libggml-cuda.so")
                    elif mutation == "symlink":
                        closure["symlinks"][0]["target"] = "../escaped.so"
                    elif mutation == "executable-digest":
                        closure["executables"][0]["digest"] = "sha256:" + "d" * 64
                    elif mutation == "executable-float-bytes":
                        closure["executables"][0]["bytes"] = float(
                            closure["executables"][0]["bytes"]
                        )
                    elif mutation == "executable-mode":
                        closure["executables"][0]["mode"] = "0644"
                    elif mutation == "library-digest":
                        closure["libraries"][0]["digest"] = "sha256:" + "d" * 64
                    elif mutation == "library-float-bytes":
                        closure["libraries"][0]["bytes"] = float(closure["libraries"][0]["bytes"])
                    else:
                        closure["libraries"][0]["mode"] = "0777"
                self.rewrite_v3_receipts()
                with self.assertRaisesRegex(provenance.CodeProvenanceError, "runtime"):
                    self.validate_v3()

    def test_calibrated_v3_rejects_replay_metadata_digest_and_bundle_changes(self) -> None:
        self.prepare_calibrated_v3()
        self.v3_conversion["conversion"]["determinism_replay"]["commands"][-1][
            "finished_at_unix_ns"
        ] += 1
        self.rewrite_v3_receipts()
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "cross-receipt"):
            self.validate_v3()

    def test_calibrated_v3_rejects_model_imatrix_count_mismatch(self) -> None:
        self.prepare_calibrated_v3()
        self.v3_calibration["intermediate"]["imatrix"]["entries_count"] += 1
        self.rewrite_v3_receipts()
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "metadata bindings"):
            self.validate_v3()

    def test_calibrated_v3_rejects_wrong_calibration_digest(self) -> None:
        self.prepare_calibrated_v3()
        self.v3_conversion["calibration_receipt_digest"] = "sha256:" + "d" * 64
        write_json(self.v3_conversion_path, self.v3_conversion)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "calibration receipt binding"):
            self.validate_v3()

    def test_calibrated_v3_rejects_extra_bundle_file(self) -> None:
        self.prepare_calibrated_v3()
        (self.v3_conversion_path.parent / "unexpected.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "exactly model"):
            self.validate_v3()

    def test_calibrated_v3_requires_all_replay_inputs(self) -> None:
        self.prepare_calibrated_v3()
        self.v3_request = replace(self.v3_request, calibration_current_dataset=None)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "requires --llama-cpp"):
            self.validate_v3()

    def test_calibrated_v3_requires_the_reviewed_resolved_llama_cpp_root(self) -> None:
        self.prepare_calibrated_v3()
        reviewed_root = self.v3_request.llama_cpp.resolve(strict=True)
        different_root = self.root / "different-llama.cpp"
        different_root.mkdir()
        self.v3_request = replace(self.v3_request, llama_cpp=different_root)
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "resolve exactly"):
            self.validate_v3(pinned_root=reviewed_root)

    def test_calibrated_v3_requires_exact_q4_load_manifest(self) -> None:
        self.prepare_calibrated_v3()
        self.load["quantization"] = "Q8_0"
        write_json(self.v3_request.load_spec, self.load)
        self.v3_calibration["load_manifest"] = copy.deepcopy(self.load)
        self.v3_conversion["load_manifest"] = copy.deepcopy(self.load)
        self.rewrite_v3_receipts()
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "exact Q4_K_M"):
            self.validate_v3()

    def test_calibrated_v3_never_serializes_raw_calibration_fields(self) -> None:
        self.prepare_calibrated_v3()
        replay_current = self.v3_material["source"]["current"]
        receipt_current = self.v3_calibration["source"]["current"]
        replay_current["prompt"] = "private prompt"
        receipt_current["prompt"] = "private prompt"
        self.rewrite_v3_receipts()
        with self.assertRaisesRegex(provenance.CodeProvenanceError, "raw calibration text"):
            self.validate_v3()

    def test_calibrated_converter_import_orders_are_acyclic(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for modules in (
            ("training.publish_code_provenance", "training.convert_code_gguf"),
            ("training.convert_code_gguf", "training.publish_code_provenance"),
        ):
            with self.subTest(first=modules[0]):
                script = (
                    "import importlib, sys; "
                    f"importlib.import_module({modules[0]!r}); "
                    f"importlib.import_module({modules[1]!r}); "
                    "publisher=importlib.import_module('training.publish_code_provenance'); "
                    "converter=importlib.import_module('training.convert_code_gguf'); "
                    "assert publisher._calibrated_converter_module() is converter; "
                    "assert 'wandb' not in sys.modules"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=repository,
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONNOUSERSITE": "1"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_generic_conversion_v1_rejects_q4_without_calibrated_receipts(self) -> None:
        self.load["quantization"] = "Q4_K_M"
        self.conversion["artifact"]["quantization"] = "Q4_K_M"
        self.conversion["load_manifest"] = copy.deepcopy(self.load)
        write_json(self.load_path, self.load)
        write_json(self.conversion_path, self.conversion)
        with self.assertRaisesRegex(
            provenance.CodeProvenanceError,
            "Q4_K_M publication requires calibrated conversion-v3",
        ):
            self.validate()

    def test_generic_conversion_v1_retains_q5_compatibility(self) -> None:
        self.load["quantization"] = "Q5_K_M"
        self.conversion["artifact"]["quantization"] = "Q5_K_M"
        self.conversion["load_manifest"] = copy.deepcopy(self.load)
        write_json(self.load_path, self.load)
        write_json(self.conversion_path, self.conversion)
        publication = self.validate()
        self.assertEqual(publication.load_manifest["quantization"], "Q5_K_M")

    def test_normalized_v6_generic_v4_is_exactly_bound_and_dynamic(self) -> None:
        self.configure_normalized_v6()
        publication = self.validate()
        self.assertEqual(
            publication.training_lineage["schema"],
            provenance.NORMALIZED_TRAINING_SCHEMA,
        )
        self.assertEqual(publication.conversion["schema"], provenance.NORMALIZED_CONVERSION_SCHEMA)
        self.assertEqual(
            publication.conversion["source"],
            {
                "training_schema": provenance.NORMALIZED_TRAINING_SCHEMA,
                "dataset_schema": normalized_candidate.DATASET_SCHEMA,
                "corpus_profile": normalized_candidate.CORPUS_PROFILE,
                "training_metadata_digest": publication.training_lineage["receipt"]["digest"],
                "merged_tree_digest": publication.training_lineage["run"]["merged"]["digest"],
                "excluded_refs": {
                    "bytes": normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
                    "digest": normalized_candidate.EXPECTED_EXCLUDED_REFS_DIGEST,
                },
            },
        )
        config = provenance._wandb_config(publication)
        summary = provenance._wandb_summary(publication)
        self.assertEqual(config["mt_training_schema"], provenance.NORMALIZED_TRAINING_SCHEMA)
        self.assertEqual(
            config["mt_quality_claim"],
            normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        )
        self.assertEqual(config["mt_corpus_version"], normalized_candidate.CORPUS_VERSION)
        self.assertEqual(summary["mt_training_schema"], provenance.NORMALIZED_TRAINING_SCHEMA)

    def test_normalized_v6_rejects_schema_source_manifest_and_lineage_tampering(self) -> None:
        self.configure_normalized_v6()
        baseline_lineage = copy.deepcopy(self.training_lineage)
        baseline_metadata = copy.deepcopy(self.metadata)
        baseline_conversion = copy.deepcopy(self.conversion)

        def legacy_schema() -> None:
            self.conversion["schema"] = provenance.CONVERSION_SCHEMA
            self.conversion["source"] = {
                "training_metadata_digest": self.training_lineage["receipt"]["digest"],
                "merged_tree_digest": self.training_lineage["run"]["merged"]["digest"],
            }

        def training_schema() -> None:
            self.conversion["source"]["training_schema"] = provenance.TRAINING_SCHEMA

        def dataset_schema() -> None:
            self.conversion["source"]["dataset_schema"] = historical_candidate.DATASET_SCHEMA

        def corpus_profile() -> None:
            self.conversion["source"]["corpus_profile"] = historical_candidate.CORPUS_PROFILE

        def conversion_exclusion() -> None:
            self.conversion["source"]["excluded_refs"]["bytes"] += 1

        def conversion_exclusion_type() -> None:
            self.conversion["source"]["excluded_refs"]["bytes"] = float(
                normalized_candidate.EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES
            )

        def training_metadata_digest() -> None:
            self.conversion["source"]["training_metadata_digest"] = "sha256:" + "0" * 64

        def merged_tree_digest() -> None:
            self.conversion["source"]["merged_tree_digest"] = "sha256:" + "0" * 64

        def source_corpus() -> None:
            self.training_lineage["source_corpus"]["raw_digest"] = "sha256:" + "0" * 64

        def manifest_exclusion() -> None:
            self.training_lineage["prepared_dataset"]["manifest_payload"][
                "excluded_refs_digest"
            ] = "sha256:" + "0" * 64

        def prepared_exclusion() -> None:
            self.training_lineage["prepared_dataset"]["excluded_refs"]["digest"] = (
                "sha256:" + "0" * 64
            )

        def non_qwen_base() -> None:
            self.training_lineage["base_snapshot"]["base_model"] = "other/model"

        def incomplete_run() -> None:
            del self.training_lineage["run"]["adapter"]

        cases = (
            ("legacy schema swap", legacy_schema),
            ("training schema", training_schema),
            ("dataset schema", dataset_schema),
            ("corpus profile", corpus_profile),
            ("conversion exclusion", conversion_exclusion),
            ("conversion exclusion numeric type", conversion_exclusion_type),
            ("training metadata digest", training_metadata_digest),
            ("merged tree digest", merged_tree_digest),
            ("source corpus", source_corpus),
            ("manifest exclusion", manifest_exclusion),
            ("prepared exclusion", prepared_exclusion),
            ("non-Qwen3 base", non_qwen_base),
            ("incomplete run", incomplete_run),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                self.training_lineage = copy.deepcopy(baseline_lineage)
                self.metadata = copy.deepcopy(baseline_metadata)
                self.conversion = copy.deepcopy(baseline_conversion)
                mutate()
                write_json(self.conversion_path, self.conversion)
                with self.assertRaises(provenance.CodeProvenanceError):
                    self.validate()

    def test_normalized_generic_v4_rejects_q4_without_calibrated_v5(self) -> None:
        self.configure_normalized_v6()
        self.load["quantization"] = "Q4_K_M"
        self.conversion["artifact"]["quantization"] = "Q4_K_M"
        self.conversion["load_manifest"] = copy.deepcopy(self.load)
        write_json(self.load_path, self.load)
        write_json(self.conversion_path, self.conversion)
        with self.assertRaisesRegex(
            provenance.CodeProvenanceError,
            "Q4_K_M publication requires calibrated conversion-v5",
        ):
            self.validate()

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
        metadata_raw = (self.run / "training_metadata.json").read_bytes()
        metadata_identity = {
            "bytes": len(metadata_raw),
            "digest": digest(metadata_raw),
        }
        self.training_lineage["receipt"].update(metadata_identity)
        self.training_lineage["run"]["training_metadata"] = metadata_identity
        self.training_lineage["run"]["metrics"] = {
            "bytes": len(raw.encode()),
            "digest": digest(raw.encode()),
        }
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
