from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from training import publish_provenance as provenance


ARTIFACT_DIGEST = "sha256:" + "a" * 64
CORPUS_FILE_DIGEST = "sha256:" + "b" * 64


def target_settings(
    *, epochs: int = 1, canonicalization: object = "first", weight: object = 3.0
) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "gold_canonicalization": canonicalization,
        "entity_text_token_weight": weight,
    }


def target_controls(
    *, canonicalization: object = "first", weight: object = 3.0
) -> dict[str, Any]:
    return {
        "entity_match": "exact_text_and_type_set",
        "gold_canonicalization": canonicalization,
        "entity_text_token_weight": weight,
        "validation_loss": "ordinary_unweighted_causal_lm",
    }


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_tree_digest(files: dict[str, bytes]) -> str:
    tree = hashlib.sha256()
    for name, payload in sorted(files.items()):
        tree.update(name.encode("utf-8"))
        tree.update(b"\x00")
        tree.update(digest(payload).encode("ascii"))
        tree.update(b"\x00")
    return "sha256:" + tree.hexdigest()


def encoded_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def metrics_for(epochs: int, updates_per_epoch: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    step = 0
    elapsed = 0.0
    for epoch in range(1, epochs + 1):
        for _update in range(updates_per_epoch):
            step += 1
            elapsed += 0.5
            records.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": 1.0 / step,
                    "learning_rate": 0.001 / step,
                    "elapsed_s": elapsed,
                }
            )
        elapsed += 0.25
        validation_loss = 0.2 + epoch / 10
        records.append(
            {
                "step": step,
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_perplexity": math.exp(min(20.0, validation_loss)),
                "elapsed_s": elapsed,
            }
        )
    return records


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}


class FakeWandb:
    """In-memory W&B module double; it has no credential or network behavior."""

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_calls: list[dict[str, Any]] = []
        self.log_calls: list[tuple[dict[str, Any], int]] = []
        self.finish_calls = 0

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_calls.append(copy.deepcopy(kwargs))
        return self.run

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        self.log_calls.append((copy.deepcopy(payload), step))

    def finish(self) -> None:
        self.finish_calls += 1


class FakeSoupValidationError(ValueError):
    pass


class PublishProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_stage(
        self,
        name: str,
        *,
        parent_digest: str | None = None,
        epochs: int = 1,
        updates_per_epoch: int = 2,
        started: int = 1_000,
        metadata_changes: dict[str, Any] | None = None,
        input_changes: dict[str, Any] | None = None,
        metric_records: list[dict[str, Any]] | None = None,
    ) -> tuple[Path, dict[str, Any], list[dict[str, Any]], bytes]:
        directory = self.root / name
        directory.mkdir()
        records = (
            copy.deepcopy(metric_records)
            if metric_records is not None
            else metrics_for(epochs, updates_per_epoch)
        )
        elapsed = max(float(record["elapsed_s"]) for record in records) + 0.5
        training_input: dict[str, Any]
        if parent_digest is None:
            training_input = {
                "kind": "huggingface_snapshot",
                "revision": provenance.BASE_REVISION,
                "weights_digest": provenance.BASE_WEIGHTS_DIGEST,
                "tokenizer_digest": provenance.BASE_TOKENIZER_DIGEST,
            }
        else:
            training_input = {
                "kind": "derived_model",
                "parent_metadata_digest": parent_digest,
                "weights_digest": "sha256:" + "c" * 64,
                "tokenizer_digest": "sha256:" + "d" * 64,
            }
        training_input.update(input_changes or {})
        metadata = {
            "hotkey": provenance.HOTKEY,
            "track": provenance.TRACK,
            "hardware_class": provenance.HARDWARE_CLASS,
            "base_model": provenance.PINNED_BASE_MODEL,
            "training_input": training_input,
            "corpus_version": provenance.CORPUS_VERSION,
            "corpus_file_digest": CORPUS_FILE_DIGEST,
            "settings": {"epochs": epochs},
            "started_at_unix": started,
            "finished_at_unix": started + math.ceil(elapsed) + 1,
            "elapsed_s": elapsed,
            "updates": epochs * updates_per_epoch,
        }
        metadata.update(metadata_changes or {})
        metadata_bytes = encoded_json(metadata)
        (directory / "training_metadata.json").write_bytes(metadata_bytes)
        (directory / "metrics.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        return directory, metadata, records, metadata_bytes

    def write_augmented_stage(
        self, name: str, *, enabled: bool = True
    ) -> tuple[Path, dict[str, Any], bytes, Path | None]:
        directory, metadata, _records, _original = self.write_stage(name)
        requested = 1 if enabled else 0
        summary: dict[str, Any] = {
            "algorithm": provenance.ENTITY_SUBSTITUTION_ALGORITHM,
            "enabled": enabled,
            "seed": 92,
            "requested_examples": requested,
            "augmented_examples": requested,
            "replacement_count": requested,
        }
        manifest_path: Path | None = None
        if enabled:
            summary.update(
                {
                    "source_training_rows": 2,
                    "eligible_source_rows": 2,
                    "ineligible_source_rows": {},
                    "globally_ambiguous_surfaces": 0,
                    "no_compatible_donor_rows": 0,
                    "donor_entity_counts": {"Chemical": 1, "Disease": 1},
                    "source_training_refs_digest": "sha256:" + "1" * 64,
                    "heldout_refs_digest": "sha256:" + "2" * 64,
                    "donor_pool_digest": "sha256:" + "3" * 64,
                }
            )
            manifest = {
                **summary,
                "examples": [
                    {
                        "source_ref": "source-1",
                        "augmented_ref": (
                            "source-1::entity-substitution::0123456789abcdef"
                        ),
                        "donor_ref": "donor-1",
                        "type": "Chemical",
                        "source_text": "Aspirin",
                        "donor_text": "Ibuprofen",
                        "occurrence_count": 1,
                    }
                ],
            }
            manifest_bytes = encoded_json(manifest)
            manifest_path = directory / "entity_substitution_manifest.json"
            manifest_path.write_bytes(manifest_bytes)
            summary.update(
                {
                    "manifest_file": manifest_path.name,
                    "manifest_digest": digest(manifest_bytes),
                }
            )

        metadata.update(
            {
                "settings": {
                    **metadata["settings"],
                    "seed": 92,
                    "entity_substitution_examples": requested,
                },
                "augmentation": {
                    "entity_substitution": {
                        **summary,
                        "composition": provenance.ENTITY_SUBSTITUTION_COMPOSITION,
                    }
                },
                "source_training_examples": 2,
                "disease_extra_examples": 0,
                "training_examples": 2 + requested,
                "skipped_training_examples": 0,
            }
        )
        metadata_bytes = encoded_json(metadata)
        (directory / "training_metadata.json").write_bytes(metadata_bytes)
        return directory, metadata, metadata_bytes, manifest_path

    def write_soup_checkpoint(
        self, name: str
    ) -> tuple[Path, dict[str, Any], SimpleNamespace]:
        checkpoint = self.root / name
        checkpoint.mkdir()
        metadata = {
            "schema": provenance.WEIGHT_SOUP_SCHEMA,
            "base": {"identity": provenance.PINNED_BASE_MODEL},
            "algorithm": {"formula": "base + weighted deltas"},
            "sources": [
                {
                    "position": 1,
                    "parent_training_metadata": {
                        "sha256": digest(f"{name}-parent".encode())
                    },
                }
            ],
            "runtime": {"python": "fixture"},
            "output": {"fixture": name},
        }
        metadata_bytes = encoded_json(metadata)
        (checkpoint / provenance.WEIGHT_SOUP_METADATA_FILENAME).write_bytes(
            metadata_bytes
        )
        (checkpoint / "fixture.safetensors").write_bytes(b"valid-shard")
        validated = SimpleNamespace(
            metadata_digest=digest(metadata_bytes),
            output_manifest_digest=digest(f"{name}-manifest".encode()),
            index_digest=digest(f"{name}-index".encode()),
            tokenizer_digest=provenance.BASE_TOKENIZER_DIGEST,
        )
        return checkpoint, metadata, validated

    def write_soup_stage(
        self, name: str, validated: SimpleNamespace
    ) -> tuple[Path, dict[str, Any], bytes]:
        directory, metadata, _records, _metadata_bytes = self.write_stage(name)
        metadata["training_input"] = {
            "kind": "deterministic_weight_soup",
            "soup_schema": provenance.WEIGHT_SOUP_SCHEMA,
            "soup_metadata_digest": validated.metadata_digest,
            "output_manifest_digest": validated.output_manifest_digest,
            "index_digest": validated.index_digest,
            "tokenizer_digest": validated.tokenizer_digest,
        }
        metadata_bytes = encoded_json(metadata)
        (directory / "training_metadata.json").write_bytes(metadata_bytes)
        return directory, metadata, metadata_bytes

    @staticmethod
    def fake_soup_module(
        results: dict[Path, SimpleNamespace | Exception],
    ) -> SimpleNamespace:
        def validate(path: Path) -> SimpleNamespace:
            result = results[path]
            if isinstance(result, Exception):
                raise result
            return result

        return SimpleNamespace(
            SCHEMA=provenance.WEIGHT_SOUP_SCHEMA,
            METADATA_FILENAME=provenance.WEIGHT_SOUP_METADATA_FILENAME,
            SoupValidationError=FakeSoupValidationError,
            validate_weight_soup_checkpoint=mock.Mock(side_effect=validate),
        )

    def write_calibration_manifest(
        self, directory: Path, training_metadata_bytes: bytes
    ) -> tuple[Path, dict[str, Any], dict[str, Path]]:
        merged = directory / "merged"
        merged.mkdir()
        source_payloads = {
            "config.json": b'{"model_type":"qwen3"}\n',
            "model.safetensors": b"test-model-weights",
            "tokenizer.json": b'{"version":"1.0"}\n',
            "tokenizer_config.json": b'{"chat_template":"inline"}\n',
        }
        for name, payload in source_payloads.items():
            (merged / name).write_bytes(payload)

        assets = self.root / f"{directory.name}-calibration-assets"
        assets.mkdir()
        corpus_path = assets / "calibration.txt"
        record_payloads = [b"record-one\n", b"record-two\n"]
        corpus_payload = b"".join(record_payloads)
        corpus_path.write_bytes(corpus_payload)
        imatrix_path = assets / "calibration.imatrix.gguf"
        imatrix_path.write_bytes(b"GGUF-imatrix")
        converted_path = assets / "model-f16.gguf"
        converted_path.write_bytes(b"GGUF-f16")
        artifact_dir = assets / "artifact"
        artifact_dir.mkdir()
        artifact_path = artifact_dir / "model.gguf"
        artifact_payload = b"GGUF-model"
        artifact_path.write_bytes(artifact_payload)

        tokenizer_artifacts = {
            name: {"bytes": len(source_payloads[name]), "sha256": digest(source_payloads[name])}
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        }
        sidecar = {
            "schema": provenance.IMATRIX_CORPUS_SCHEMA,
            "source": {
                "corpus_bytes": 123,
                "corpus_file_sha256": CORPUS_FILE_DIGEST,
                "corpus_version": provenance.CORPUS_VERSION,
                "loader": "training.train_extract.load_rows",
                "public_train_rows": 2,
            },
            "tokenizer": {
                "artifacts": tokenizer_artifacts,
                "chat_template_sha256": "sha256:" + "4" * 64,
                "chat_template_source": "tokenizer_config.json:chat_template",
                "loader": "transformers.AutoTokenizer.from_pretrained",
                "local_files_only": True,
                "runtime_class": "Qwen2Tokenizer",
                "trust_remote_code": False,
                "model_type": "qwen3",
                "tokenizer_class": "Qwen2Tokenizer",
            },
            "selection": {
                "algorithm": "random.Random(seed).shuffle(rows)",
                "eligible_examples": 2,
                "included_examples": 2,
                "included_refs": ["row-1", "row-2"],
                "max_examples": 2,
                "omitted_after_cap": 0,
                "rejected_rows": [],
                "reserve_examples": 0,
                "reserved_refs": [],
                "seed": 92,
            },
            "rendering": {
                "add_generation_prompt": False,
                "canonical_gold": {
                    "deduplicate": "exact_text_type_pair",
                    "entity_order": "unicode_text_then_type",
                    "json": "utf8_compact_sorted_keys",
                    "substring_scope": "inputs.text",
                },
                "enable_thinking": False,
                "record_separator": (
                    "none; each Qwen rendering ends with <|im_end|>\\n"
                ),
                "tokenize": False,
            },
            "output": {
                "bytes": len(corpus_payload),
                "records": len(record_payloads),
                "sha256": digest(corpus_payload),
            },
            "records": [
                {
                    "bytes": len(payload),
                    "ref": f"row-{index}",
                    "sha256": digest(payload),
                }
                for index, payload in enumerate(record_payloads, start=1)
            ],
            "runtime": {"python": "3.11.14"},
        }
        metadata_path = assets / "calibration.txt.metadata.json"
        metadata_path.write_bytes(encoded_json(sidecar))

        def claim(path: Path) -> dict[str, Any]:
            payload = path.read_bytes()
            return {
                "path": path.relative_to(self.root).as_posix(),
                "bytes": len(payload),
                "sha256": digest(payload),
            }

        corpus_claim = claim(corpus_path)
        metadata_claim = claim(metadata_path)
        imatrix_claim = claim(imatrix_path)
        converted_claim = claim(converted_path)
        artifact_claim = claim(artifact_path)
        manifest = {
            "schema": provenance.CALIBRATION_LINEAGE_SCHEMA,
            "llama_cpp_revision": provenance.LLAMA_CPP_REVISION,
            "artifact_tree_digest": artifact_tree_digest(
                {"model.gguf": artifact_payload}
            ),
            "source_model": {
                "directory": merged.relative_to(self.root).as_posix(),
                "training_metadata_sha256": digest(training_metadata_bytes),
                "files": [
                    {
                        "path": name,
                        "bytes": len(payload),
                        "sha256": digest(payload),
                    }
                    for name, payload in sorted(source_payloads.items())
                ],
            },
            "conversion": {
                "tool": "convert_hf_to_gguf.py",
                "outtype": "f16",
                "output": converted_claim,
                "arguments": [
                    merged.relative_to(self.root).as_posix(),
                    "--outfile",
                    converted_claim["path"],
                    "--outtype",
                    "f16",
                ],
            },
            "calibration": {
                "tool": "llama-imatrix",
                "corpus": corpus_claim,
                "metadata": metadata_claim,
                "imatrix": imatrix_claim,
                "settings": {
                    "offline": True,
                    "ctx_size": 512,
                    "chunks": -1,
                    "no_ppl": True,
                    "process_output": False,
                    "parse_special": True,
                    "output_format": "gguf",
                },
                "arguments": [
                    "--offline",
                    "--model",
                    converted_claim["path"],
                    "--file",
                    corpus_claim["path"],
                    "--output",
                    imatrix_claim["path"],
                    "--ctx-size",
                    "512",
                    "--chunks",
                    "-1",
                    "--no-ppl",
                    "--parse-special",
                ],
            },
            "quantization": {
                "tool": "llama-quantize",
                "arguments": [
                    "--imatrix",
                    imatrix_claim["path"],
                    "--tensor-type",
                    provenance.ATTN_V_Q6_OVERRIDE,
                    converted_claim["path"],
                    artifact_claim["path"],
                    "Q4_K_M",
                ],
                "output": artifact_claim,
            },
        }
        manifest_path = self.root / f"{directory.name}-calibration-lineage.json"
        manifest_path.write_bytes(encoded_json(manifest))
        return manifest_path, manifest, {
            "artifact": artifact_path,
            "converted": converted_path,
            "corpus": corpus_path,
            "metadata": metadata_path,
            "imatrix": imatrix_path,
        }

    def invoke_main(
        self,
        directories: list[Path],
        fake: FakeWandb,
        *,
        calibration_manifest: Path | None = None,
        artifact_digest: str = ARTIFACT_DIGEST,
        weight_soup_checkpoints: dict[int, Path] | None = None,
    ) -> int:
        arguments: list[str] = []
        for directory in directories:
            arguments.extend(("--training-dir", str(directory)))
        arguments.extend(
            ("--artifact-digest", artifact_digest, "--finished-block", "8955436")
        )
        for stage_number, checkpoint in sorted((weight_soup_checkpoints or {}).items()):
            arguments.extend(
                ("--weight-soup-checkpoint", str(stage_number), str(checkpoint))
            )
        arguments.extend(
            (
                "--calibration-manifest",
                str(calibration_manifest or self.root / "missing-calibration-lineage.json"),
            )
        )
        with (
            mock.patch.dict(provenance.os.environ, {}, clear=True),
            mock.patch.object(
                provenance.importlib,
                "import_module",
                side_effect=AssertionError("wandb import was not lazy"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            return provenance.main(arguments, wandb_client=fake)

    def assert_invalid(self, directories: list[Path], pattern: str) -> None:
        with self.assertRaisesRegex(provenance.ProvenanceValidationError, pattern):
            provenance.validate_publication(directories, ARTIFACT_DIGEST, 8955436)

    def test_valid_single_stage_keeps_cli_and_config_compatibility(self) -> None:
        directory, metadata, metrics, metadata_bytes = self.write_stage("stage-one")
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )
        fake = FakeWandb()

        self.assertEqual(
            self.invoke_main(
                [directory],
                fake,
                calibration_manifest=calibration_path,
                artifact_digest=calibration["artifact_tree_digest"],
            ),
            0,
        )

        self.assertEqual(len(fake.init_calls), 1)
        call = fake.init_calls[0]
        self.assertEqual(call["entity"], provenance.ENTITY)
        self.assertEqual(call["project"], provenance.PROJECT)
        self.assertEqual(call["name"], provenance.HOTKEY)
        config = call["config"]
        self.assertEqual(config["hotkey"], metadata["hotkey"])
        self.assertEqual(config["mt_track"], provenance.TRACK)
        self.assertEqual(config["mt_class"], provenance.HARDWARE_CLASS)
        self.assertEqual(config["mt_base_model"], provenance.PINNED_BASE_MODEL)
        self.assertNotIn("mt_target_controls_semantics", config)
        stage_config = config["mt_stage_metadata"]["stage_1"]
        self.assertEqual(stage_config["metadata"], metadata)
        self.assertEqual(stage_config["metadata_digest"], digest(metadata_bytes))
        self.assertEqual(
            config["mt_calibration_lineage"],
            {
                "manifest": calibration,
                "manifest_digest": digest(encoded_json(calibration)),
            },
        )

        self.assertEqual([step for _payload, step in fake.log_calls], [1, 2, 3])
        for source, (logged, _step) in zip(metrics, fake.log_calls, strict=True):
            self.assertEqual({key: logged[key] for key in source}, source)
            self.assertEqual(logged["mt_stage"], 1)
            self.assertEqual(logged["mt_stage_epoch"], source["epoch"])
            self.assertEqual(logged["mt_stage_step"], source["step"])
        self.assertEqual(
            fake.run.summary,
            {
                "mt_artifact_digest": calibration["artifact_tree_digest"],
                "mt_calibration_manifest_digest": digest(encoded_json(calibration)),
                "mt_finished_at": 8955436,
                "mt_training_records": 3,
                "mt_training_stages": 1,
            },
        )
        self.assertEqual(fake.finish_calls, 1)

    def test_target_controls_are_validated_and_publish_exact_semantics(self) -> None:
        controls = target_controls(canonicalization="sorted", weight=2.5)
        directory, metadata, _metrics, metadata_bytes = self.write_stage(
            "controlled",
            metadata_changes={
                "settings": target_settings(canonicalization="sorted", weight=2.5),
                "target_controls": controls,
            },
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )
        fake = FakeWandb()

        self.assertEqual(
            self.invoke_main(
                [directory],
                fake,
                calibration_manifest=calibration_path,
                artifact_digest=calibration["artifact_tree_digest"],
            ),
            0,
        )

        config = fake.init_calls[0]["config"]
        self.assertEqual(config["target_controls"], controls)
        self.assertEqual(
            config["mt_target_controls_semantics"],
            {
                "schema": provenance.TARGET_CONTROLS_SCHEMA,
                "canonicalization": {
                    "none": "preserve_raw_gold_target",
                    "first": "strict_exact_pair_deduplication_in_first_occurrence_order",
                    "sorted": "strict_exact_pair_deduplication_in_lexicographic_order",
                },
                "malformed_gold": "reject_without_repair",
                "entity_text_token_binding": provenance.ENTITY_TEXT_TOKEN_BINDING,
                "entity_text_weighting_loss": provenance.WEIGHTED_TRAINING_LOSS,
                "weighting_active_when": "entity_text_token_weight_greater_than_one",
                "validation_loss": "ordinary_unweighted_causal_lm",
            },
        )
        self.assertEqual(
            config["mt_stage_metadata"]["stage_1"]["metadata"], metadata
        )

    def test_target_controls_require_exact_schema_and_settings_consistency(self) -> None:
        valid_settings = target_settings()
        valid_controls = target_controls()
        cases: list[tuple[str, dict[str, Any], str]] = [
            (
                "not-object",
                {"settings": valid_settings, "target_controls": []},
                "target_controls must be an object",
            ),
            (
                "missing-field",
                {
                    "settings": valid_settings,
                    "target_controls": {
                        key: value
                        for key, value in valid_controls.items()
                        if key != "validation_loss"
                    },
                },
                "must contain exactly",
            ),
            (
                "extra-field",
                {
                    "settings": valid_settings,
                    "target_controls": {**valid_controls, "repair_malformed": False},
                },
                "must contain exactly",
            ),
            (
                "canonicalization-mismatch",
                {
                    "settings": valid_settings,
                    "target_controls": target_controls(canonicalization="sorted"),
                },
                "gold_canonicalization must exactly match settings",
            ),
            (
                "weight-mismatch",
                {
                    "settings": valid_settings,
                    "target_controls": target_controls(weight=4.0),
                },
                "entity_text_token_weight must exactly match settings",
            ),
            (
                "unknown-match",
                {
                    "settings": valid_settings,
                    "target_controls": {**valid_controls, "entity_match": "casefolded"},
                },
                "entity_match has unknown semantics",
            ),
            (
                "unknown-validation-loss",
                {
                    "settings": valid_settings,
                    "target_controls": {
                        **valid_controls,
                        "validation_loss": "weighted_causal_lm",
                    },
                },
                "validation_loss has unknown semantics",
            ),
            (
                "invalid-canonicalization",
                {
                    "settings": target_settings(canonicalization="casefolded"),
                    "target_controls": target_controls(canonicalization="casefolded"),
                },
                "settings.gold_canonicalization",
            ),
            (
                "invalid-weight",
                {
                    "settings": target_settings(weight=0.5),
                    "target_controls": target_controls(weight=0.5),
                },
                "settings.entity_text_token_weight",
            ),
            (
                "partial-settings",
                {
                    "settings": {"epochs": 1, "gold_canonicalization": "first"},
                    "target_controls": valid_controls,
                },
                "both target-control settings",
            ),
            (
                "weighted-without-canonicalization",
                {
                    "settings": target_settings(canonicalization="none", weight=2.0),
                    "target_controls": target_controls(canonicalization="none", weight=2.0),
                },
                "requires canonicalized gold",
            ),
        ]
        for name, changes, pattern in cases:
            with self.subTest(case=name):
                directory, _metadata, _metrics, _bytes = self.write_stage(
                    f"controls-{name}", metadata_changes=changes
                )
                self.assert_invalid([directory], pattern)

    def test_historical_metadata_may_omit_controls_but_new_settings_may_not(self) -> None:
        historical, _metadata, _metrics, metadata_bytes = self.write_stage("historical")
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            historical, metadata_bytes
        )
        provenance.validate_publication(
            [historical],
            calibration["artifact_tree_digest"],
            8955436,
            calibration_manifest=calibration_path,
        )

        stripped, _metadata, _metrics, _bytes = self.write_stage(
            "stripped-controls",
            metadata_changes={"settings": target_settings()},
        )
        self.assert_invalid([stripped], "target_controls is required")

    def test_valid_two_stage_lineage_uses_unique_global_record_steps(self) -> None:
        first, first_metadata, first_metrics, first_bytes = self.write_stage(
            "oldest", epochs=2, started=1_000
        )
        second, second_metadata, second_metrics, second_bytes = self.write_stage(
            "newest",
            parent_digest=digest(first_bytes),
            started=2_000,
            metadata_changes={"pipeline_stages": 2},
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            second, second_bytes
        )
        fake = FakeWandb()

        self.assertEqual(
            self.invoke_main(
                [first, second],
                fake,
                calibration_manifest=calibration_path,
                artifact_digest=calibration["artifact_tree_digest"],
            ),
            0,
        )

        all_metrics = [*first_metrics, *second_metrics]
        self.assertEqual(
            [global_step for _payload, global_step in fake.log_calls],
            list(range(1, len(all_metrics) + 1)),
        )
        for source, (logged, _global_step) in zip(all_metrics, fake.log_calls, strict=True):
            self.assertEqual({key: logged[key] for key in source}, source)
        second_first = fake.log_calls[len(first_metrics)][0]
        self.assertEqual(second_first["step"], 1)
        self.assertEqual(second_first["epoch"], 1)
        self.assertEqual(second_first["mt_stage"], 2)
        self.assertEqual(second_first["mt_stage_step"], 1)

        config = fake.init_calls[0]["config"]
        self.assertEqual(config["settings"], second_metadata["settings"])
        self.assertEqual(config["mt_training_stages"], 2)
        self.assertEqual(
            config["mt_stage_metadata"]["stage_1"]["metadata_digest"],
            digest(first_bytes),
        )
        self.assertEqual(
            config["mt_stage_metadata"]["stage_2"]["metadata_digest"],
            digest(second_bytes),
        )
        self.assertEqual(fake.run.summary["mt_training_records"], len(all_metrics))
        self.assertEqual(fake.run.summary["mt_training_stages"], 2)
        self.assertEqual(fake.finish_calls, 1)
        self.assertEqual(first_metadata["training_input"]["kind"], "huggingface_snapshot")

    def test_broken_or_reversed_lineage_fails_before_wandb_init(self) -> None:
        first, _metadata, _metrics, first_bytes = self.write_stage("first")
        second, _metadata, _metrics, _bytes = self.write_stage(
            "second",
            parent_digest="sha256:" + "0" * 64,
            started=2_000,
        )
        fake = FakeWandb()
        with self.assertRaisesRegex(SystemExit, "parent metadata digest"):
            self.invoke_main([first, second], fake)
        self.assertEqual(fake.init_calls, [])

        correct_second, _metadata, _metrics, _bytes = self.write_stage(
            "correct-second",
            parent_digest=digest(first_bytes),
            started=2_000,
        )
        self.assert_invalid([correct_second, first], "stage 1 training_input.kind")

    def test_parent_digest_hashes_exact_metadata_bytes(self) -> None:
        first, _metadata, _metrics, first_bytes = self.write_stage("first")
        second, _metadata, _metrics, _bytes = self.write_stage(
            "second",
            parent_digest=digest(first_bytes),
            started=2_000,
        )
        (first / "training_metadata.json").write_bytes(b" \n" + first_bytes)

        self.assert_invalid([first, second], "parent metadata digest")

    def test_each_expected_identity_is_exact(self) -> None:
        mismatches = {
            "hotkey": "other-hotkey",
            "track": "other-track",
            "hardware_class": "other-class",
            "base_model": "Qwen/Qwen3-0.6B@main",
            "corpus_version": "sha256:" + "0" * 64,
        }
        for index, (field, value) in enumerate(mismatches.items()):
            with self.subTest(field=field):
                directory, _metadata, _metrics, _bytes = self.write_stage(
                    f"identity-{index}",
                    metadata_changes={field: value},
                )
                self.assert_invalid([directory], field)

    def test_corpus_file_identity_is_consistent_across_stages(self) -> None:
        first, _metadata, _metrics, first_bytes = self.write_stage("corpus-first")
        second, _metadata, _metrics, _bytes = self.write_stage(
            "corpus-second",
            parent_digest=digest(first_bytes),
            started=2_000,
            metadata_changes={"corpus_file_digest": "sha256:" + "e" * 64},
        )

        self.assert_invalid(
            [first, second], "corpus_file_digest differs from stage 1"
        )

    def test_declared_stage_count_must_match_supplied_lineage(self) -> None:
        first, _metadata, _metrics, first_bytes = self.write_stage("count-first")
        second, _metadata, _metrics, _bytes = self.write_stage(
            "count-second",
            parent_digest=digest(first_bytes),
            started=2_000,
            metadata_changes={"pipeline_stages": 3},
        )

        self.assert_invalid([first, second], "pipeline_stages")

    def test_stage_one_snapshot_identity_is_allowlisted(self) -> None:
        mismatches = {
            "kind": "derived_model",
            "revision": "main",
            "weights_digest": "sha256:" + "0" * 64,
            "tokenizer_digest": "sha256:" + "0" * 64,
        }
        for index, (field, value) in enumerate(mismatches.items()):
            with self.subTest(field=field):
                directory, _metadata, _metrics, _bytes = self.write_stage(
                    f"snapshot-{index}",
                    input_changes={field: value},
                )
                self.assert_invalid([directory], f"{field}|base {field.split('_')[0]}")

    def test_artifact_digest_and_finished_block_are_strict(self) -> None:
        invalid_digests = [
            "sha256:abc",
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
            ARTIFACT_DIGEST + " ",
        ]
        for value in invalid_digests:
            with self.subTest(digest=value), self.assertRaisesRegex(
                provenance.ProvenanceValidationError, "artifact digest"
            ):
                provenance.validate_publication([self.root], value, 1)

        for value in (0, -1, True):
            with self.subTest(block=value), self.assertRaisesRegex(
                provenance.ProvenanceValidationError, "finished block"
            ):
                provenance.validate_publication([self.root], ARTIFACT_DIGEST, value)


    def test_entity_substitution_disabled_and_enabled_metadata_is_bound(self) -> None:
        disabled, disabled_metadata, metadata_bytes, augmentation_path = self.write_augmented_stage(
            "augmentation-disabled", enabled=False
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            disabled, metadata_bytes
        )
        publication = provenance.validate_publication(
            [disabled],
            calibration["artifact_tree_digest"],
            8955436,
            calibration_manifest=calibration_path,
        )
        self.assertIsNone(augmentation_path)
        self.assertEqual(
            publication.stages[0].metadata["augmentation"],
            disabled_metadata["augmentation"],
        )

        enabled, enabled_metadata, metadata_bytes, augmentation_path = self.write_augmented_stage(
            "augmentation-enabled"
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            enabled, metadata_bytes
        )
        publication = provenance.validate_publication(
            [enabled],
            calibration["artifact_tree_digest"],
            8955436,
            calibration_manifest=calibration_path,
        )
        self.assertIsNotNone(augmentation_path)
        self.assertEqual(
            publication.stages[0].metadata["augmentation"],
            enabled_metadata["augmentation"],
        )

    def test_entity_substitution_claims_fail_closed(self) -> None:
        stripped, metadata, _bytes, _manifest = self.write_augmented_stage(
            "augmentation-stripped", enabled=False
        )
        del metadata["augmentation"]
        (stripped / "training_metadata.json").write_bytes(encoded_json(metadata))
        self.assert_invalid([stripped], "augmentation is required")

        tampered, _metadata, _bytes, manifest_path = self.write_augmented_stage(
            "augmentation-tampered"
        )
        if manifest_path is None:
            self.fail("enabled augmentation fixture did not write its manifest")
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        self.assert_invalid([tampered], "manifest_digest does not match")

        mismatched, metadata, _bytes, _manifest = self.write_augmented_stage(
            "augmentation-summary-mismatch"
        )
        metadata["augmentation"]["entity_substitution"]["donor_pool_digest"] = (
            "sha256:" + "9" * 64
        )
        (mismatched / "training_metadata.json").write_bytes(encoded_json(metadata))
        self.assert_invalid([mismatched], "manifest does not match")

    def test_weight_soup_checkpoint_is_revalidated_and_published(self) -> None:
        checkpoint, soup_metadata, validated = self.write_soup_checkpoint(
            "published-soup"
        )
        directory, stage_metadata, metadata_bytes = self.write_soup_stage(
            "soup-continuation", validated
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )
        soup_module = self.fake_soup_module({checkpoint: validated})
        fake = FakeWandb()

        with mock.patch.object(
            provenance, "_load_weight_soup_module", return_value=soup_module
        ):
            self.assertEqual(
                self.invoke_main(
                    [directory],
                    fake,
                    calibration_manifest=calibration_path,
                    artifact_digest=calibration["artifact_tree_digest"],
                    weight_soup_checkpoints={1: checkpoint},
                ),
                0,
            )

        soup_module.validate_weight_soup_checkpoint.assert_called_once_with(checkpoint)
        self.assertEqual(
            fake.init_calls[0]["config"]["training_input"],
            stage_metadata["training_input"],
        )
        self.assertEqual(
            fake.init_calls[0]["config"]["mt_weight_soup_lineage"],
            {
                "stage_1": {
                    "schema": provenance.WEIGHT_SOUP_SCHEMA,
                    "metadata": soup_metadata,
                    "metadata_digest": validated.metadata_digest,
                    "output_manifest_digest": validated.output_manifest_digest,
                    "index_digest": validated.index_digest,
                    "tokenizer_digest": validated.tokenizer_digest,
                }
            },
        )

    def test_weight_soup_checkpoint_mapping_is_exact_and_not_positional(self) -> None:
        checkpoint, _metadata, validated = self.write_soup_checkpoint("mapped-soup")
        soup_stage, _stage_metadata, _bytes = self.write_soup_stage(
            "mapped-stage", validated
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, r"missing=\[1\]"
        ):
            provenance.validate_publication([soup_stage], ARTIFACT_DIGEST, 8955436)

        snapshot, _metadata, _records, _bytes = self.write_stage("mapped-snapshot")
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, r"extra=\[1\]"
        ):
            provenance.validate_publication(
                [snapshot],
                ARTIFACT_DIGEST,
                8955436,
                weight_soup_checkpoints={1: checkpoint},
            )

        other_checkpoint, _other_metadata, other_validated = self.write_soup_checkpoint(
            "other-soup"
        )
        soup_module = self.fake_soup_module(
            {checkpoint: validated, other_checkpoint: other_validated}
        )
        with (
            mock.patch.object(
                provenance, "_load_weight_soup_module", return_value=soup_module
            ),
            self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "soup_metadata_digest does not match",
            ),
        ):
            provenance.validate_publication(
                [soup_stage],
                ARTIFACT_DIGEST,
                8955436,
                weight_soup_checkpoints={1: other_checkpoint},
            )

        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "duplicate weight-soup checkpoint"
        ):
            provenance._index_weight_soup_checkpoints(
                (("1", str(checkpoint)), ("1", str(other_checkpoint)))
            )

    def test_weight_soup_root_can_precede_a_derived_stage(self) -> None:
        checkpoint, _soup_metadata, validated = self.write_soup_checkpoint(
            "continued-soup"
        )
        soup_stage, _stage_metadata, soup_stage_bytes = self.write_soup_stage(
            "continued-stage-1", validated
        )
        derived_stage, _derived_metadata, _records, derived_bytes = self.write_stage(
            "continued-stage-2",
            parent_digest=digest(soup_stage_bytes),
            started=1_100,
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            derived_stage, derived_bytes
        )
        soup_module = self.fake_soup_module({checkpoint: validated})

        with mock.patch.object(
            provenance, "_load_weight_soup_module", return_value=soup_module
        ):
            publication = provenance.validate_publication(
                [soup_stage, derived_stage],
                calibration["artifact_tree_digest"],
                8_955_436,
                calibration_manifest=calibration_path,
                weight_soup_checkpoints={1: checkpoint},
            )

        self.assertEqual(len(publication.stages), 2)
        self.assertEqual(publication.weight_soups[0].stage_number, 1)

    def test_weight_soup_cannot_replace_a_derived_stage(self) -> None:
        checkpoint, _soup_metadata, validated = self.write_soup_checkpoint(
            "late-soup"
        )
        first_stage, _metadata, _records, first_bytes = self.write_stage("first")
        late_stage, metadata, _records, _bytes = self.write_stage(
            "late-stage",
            parent_digest=digest(first_bytes),
            started=1_100,
        )
        metadata["training_input"] = {
            "kind": "deterministic_weight_soup",
            "soup_schema": provenance.WEIGHT_SOUP_SCHEMA,
            "soup_metadata_digest": validated.metadata_digest,
            "output_manifest_digest": validated.output_manifest_digest,
            "index_digest": validated.index_digest,
            "tokenizer_digest": validated.tokenizer_digest,
        }
        (late_stage / "training_metadata.json").write_bytes(encoded_json(metadata))

        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "must start the supplied lineage"
        ):
            provenance.validate_publication(
                [first_stage, late_stage],
                ARTIFACT_DIGEST,
                8_955_436,
                weight_soup_checkpoints={2: checkpoint},
            )

    def test_weight_soup_checkpoint_mapping_rejects_invalid_api_and_cli_keys(self) -> None:
        checkpoint, _metadata, validated = self.write_soup_checkpoint("keyed-soup")
        stage, _stage_metadata, _bytes = self.write_soup_stage("keyed-stage", validated)
        invalid_mappings: tuple[object, ...] = (
            {True: checkpoint},
            {"1": checkpoint},
            {0: checkpoint},
            {1: str(checkpoint)},
            ((1, checkpoint),),
        )
        for mapping in invalid_mappings:
            with (
                self.subTest(mapping=mapping),
                self.assertRaises(provenance.ProvenanceValidationError),
            ):
                provenance.validate_publication(
                    [stage],
                    ARTIFACT_DIGEST,
                    8_955_436,
                    weight_soup_checkpoints=mapping,  # type: ignore[arg-type]
                )

        for stage_text in ("0", "01", "+1", " 1"):
            with (
                self.subTest(stage_text=stage_text),
                self.assertRaises(provenance.ProvenanceValidationError),
            ):
                provenance._index_weight_soup_checkpoints(
                    ((stage_text, str(checkpoint)),)
                )

    def test_weight_soup_schema_and_all_digest_claims_fail_closed(self) -> None:
        checkpoint, _soup_metadata, validated = self.write_soup_checkpoint(
            "claims-soup"
        )
        soup_module = self.fake_soup_module({checkpoint: validated})

        schema_stage, metadata, _bytes = self.write_soup_stage(
            "soup-schema", validated
        )
        metadata["training_input"]["soup_schema"] = "microtensor.unknown-soup.v1"
        (schema_stage / "training_metadata.json").write_bytes(encoded_json(metadata))
        self.assert_invalid([schema_stage], "soup_schema is not supported")

        extra_stage, metadata, _bytes = self.write_soup_stage("soup-extra", validated)
        metadata["training_input"]["weights_digest"] = "sha256:" + "8" * 64
        (extra_stage / "training_metadata.json").write_bytes(encoded_json(metadata))
        self.assert_invalid([extra_stage], "must contain exactly")

        for field in (
            "soup_metadata_digest",
            "output_manifest_digest",
            "index_digest",
            "tokenizer_digest",
        ):
            with self.subTest(field=field):
                stage, metadata, _bytes = self.write_soup_stage(
                    f"soup-digest-{field}", validated
                )
                metadata["training_input"][field] = "sha256:" + "9" * 64
                (stage / "training_metadata.json").write_bytes(encoded_json(metadata))
                pattern = (
                    "soup tokenizer digest"
                    if field == "tokenizer_digest"
                    else f"{field} does not match"
                )
                with (
                    mock.patch.object(
                        provenance,
                        "_load_weight_soup_module",
                        return_value=soup_module,
                    ),
                    self.assertRaisesRegex(provenance.ProvenanceValidationError, pattern),
                ):
                    provenance.validate_publication(
                        [stage],
                        ARTIFACT_DIGEST,
                        8955436,
                        weight_soup_checkpoints={1: checkpoint},
                    )

    def test_weight_soup_shard_failure_is_wrapped_before_publication(self) -> None:
        checkpoint, _metadata, validated = self.write_soup_checkpoint("shard-soup")
        stage, _stage_metadata, _bytes = self.write_soup_stage("shard-stage", validated)
        with (checkpoint / "fixture.safetensors").open("ab") as handle:
            handle.write(b"tampered")
        soup_module = self.fake_soup_module(
            {checkpoint: FakeSoupValidationError("weight shard digest mismatch")}
        )
        with (
            mock.patch.object(
                provenance, "_load_weight_soup_module", return_value=soup_module
            ),
            self.assertRaisesRegex(
                provenance.ProvenanceValidationError, "weight shard digest mismatch"
            ),
        ):
            provenance.validate_publication(
                [stage],
                ARTIFACT_DIGEST,
                8955436,
                weight_soup_checkpoints={1: checkpoint},
            )

    def test_weight_soup_metadata_cannot_change_after_checkpoint_validation(self) -> None:
        checkpoint, _metadata, validated = self.write_soup_checkpoint("moving-soup")
        stage, _stage_metadata, _bytes = self.write_soup_stage("moving-stage", validated)
        soup_module = self.fake_soup_module({checkpoint: validated})

        def mutate_after_validation(path: Path) -> SimpleNamespace:
            self.assertEqual(path, checkpoint)
            (checkpoint / provenance.WEIGHT_SOUP_METADATA_FILENAME).write_bytes(
                encoded_json({"schema": provenance.WEIGHT_SOUP_SCHEMA})
            )
            return validated

        soup_module.validate_weight_soup_checkpoint.side_effect = mutate_after_validation
        with (
            mock.patch.object(
                provenance, "_load_weight_soup_module", return_value=soup_module
            ),
            self.assertRaisesRegex(
                provenance.ProvenanceValidationError, "changed after validation"
            ),
        ):
            provenance.validate_publication(
                [stage],
                ARTIFACT_DIGEST,
                8_955_436,
                weight_soup_checkpoints={1: checkpoint},
            )

    def test_weight_soup_validator_loads_in_direct_script_mode(self) -> None:
        soup_module = self.fake_soup_module({})
        with (
            mock.patch.object(provenance, "__package__", ""),
            mock.patch.object(
                provenance.importlib, "import_module", return_value=soup_module
            ) as importer,
        ):
            self.assertIs(provenance._load_weight_soup_module(), soup_module)
        importer.assert_called_once_with("build_weight_soup")

    def test_calibration_lineage_is_published_with_exact_manifest_digest(self) -> None:
        directory, _metadata, _records, metadata_bytes = self.write_stage(
            "calibrated"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )

        fake = FakeWandb()
        self.assertEqual(
            self.invoke_main(
                [directory],
                fake,
                calibration_manifest=manifest_path,
                artifact_digest=manifest["artifact_tree_digest"],
            ),
            0,
        )

        config = fake.init_calls[0]["config"]["mt_calibration_lineage"]
        self.assertEqual(config["manifest"], manifest)
        self.assertEqual(config["manifest_digest"], digest(encoded_json(manifest)))
        self.assertEqual(
            fake.run.summary["mt_calibration_manifest_digest"],
            digest(encoded_json(manifest)),
        )

        del manifest["quantization"]["arguments"][2:4]
        manifest_path.write_bytes(encoded_json(manifest))
        publication = provenance.validate_publication(
            [directory],
            manifest["artifact_tree_digest"],
            8955436,
            calibration_manifest=manifest_path,
        )
        self.assertIsNotNone(publication.calibration)

    def test_every_artifact_requires_calibration_manifest(self) -> None:
        directory, _metadata, _records, _metadata_bytes = self.write_stage(
            "calibration-required"
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "calibration manifest is required"
        ):
            provenance.validate_publication([directory], ARTIFACT_DIGEST, 8955436)

    def test_calibration_lineage_files_and_settings_fail_closed(self) -> None:
        tampered, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-tampered-artifact"
        )
        manifest_path, manifest, assets = self.write_calibration_manifest(
            tampered, metadata_bytes
        )
        assets["artifact"].write_bytes(b"GGUF-other")
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "sha256 does not match"
        ):
            provenance.validate_publication(
                [tampered],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        unsafe, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-unsafe-path"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            unsafe, metadata_bytes
        )
        manifest["calibration"]["corpus"]["path"] = "../calibration.txt"
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "normalized relative path"
        ):
            provenance.validate_publication(
                [unsafe],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        wrong_conversion, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-wrong-conversion-input"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            wrong_conversion, metadata_bytes
        )
        manifest["conversion"]["arguments"][0] = manifest["calibration"]["corpus"][
            "path"
        ]
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "conversion ordered arguments"
        ):
            provenance.validate_publication(
                [wrong_conversion],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        wrong_imatrix, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-wrong-imatrix-model"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            wrong_imatrix, metadata_bytes
        )
        manifest["calibration"]["arguments"][2] = manifest["calibration"]["corpus"][
            "path"
        ]
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "imatrix ordered arguments"
        ):
            provenance.validate_publication(
                [wrong_imatrix],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        wrong_settings, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-wrong-settings"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            wrong_settings, metadata_bytes
        )
        manifest["calibration"]["settings"]["process_output"] = True
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "settings are not allowlisted"
        ):
            provenance.validate_publication(
                [wrong_settings],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        wrong_arguments, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-wrong-arguments"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            wrong_arguments, metadata_bytes
        )
        manifest["quantization"]["arguments"][3] = (
            r"^blk\.[0-9]+\.attn_v\.weight$=Q5_K"
        )
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "ordered arguments"
        ):
            provenance.validate_publication(
                [wrong_arguments],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        bad_sidecar, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-bad-sidecar"
        )
        manifest_path, manifest, assets = self.write_calibration_manifest(
            bad_sidecar, metadata_bytes
        )
        sidecar = json.loads(assets["metadata"].read_text(encoding="utf-8"))
        sidecar["output"]["sha256"] = "sha256:" + "0" * 64
        sidecar_bytes = encoded_json(sidecar)
        assets["metadata"].write_bytes(sidecar_bytes)
        manifest["calibration"]["metadata"].update(
            {"bytes": len(sidecar_bytes), "sha256": digest(sidecar_bytes)}
        )
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "output does not match"
        ):
            provenance.validate_publication(
                [bad_sidecar],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

        incomplete, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-incomplete-source"
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            incomplete, metadata_bytes
        )
        manifest["source_model"]["files"].pop()
        manifest_path.write_bytes(encoded_json(manifest))
        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "exact directory inventory"
        ):
            provenance.validate_publication(
                [incomplete],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

    def test_training_metadata_cannot_inject_reserved_wandb_namespace(self) -> None:
        directory, _metadata, _records, metadata_bytes = self.write_stage(
            "reserved-wandb-injection",
            metadata_changes={
                "mt_boundary_contrastive_lineage": {
                    "stage_999": {"schema": "fabricated"}
                }
            },
        )
        manifest_path, manifest, _assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )

        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError,
            "publisher-reserved metadata fields",
        ):
            provenance.validate_publication(
                [directory],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

    def test_calibration_roles_reject_hardlink_aliases(self) -> None:
        directory, _metadata, _records, metadata_bytes = self.write_stage(
            "calibration-hardlink-alias"
        )
        manifest_path, manifest, assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )
        assets["imatrix"].unlink()
        os.link(assets["converted"], assets["imatrix"])
        aliased_payload = assets["imatrix"].read_bytes()
        manifest["calibration"]["imatrix"].update(
            {
                "bytes": len(aliased_payload),
                "sha256": digest(aliased_payload),
            }
        )
        manifest_path.write_bytes(encoded_json(manifest))

        with self.assertRaisesRegex(
            provenance.ProvenanceValidationError, "distinct files and inodes"
        ):
            provenance.validate_publication(
                [directory],
                manifest["artifact_tree_digest"],
                8955436,
                calibration_manifest=manifest_path,
            )

    def test_malformed_metrics_are_rejected(self) -> None:
        cases: list[tuple[str, str, str]] = []

        valid = metrics_for(1, 2)
        nan_records = copy.deepcopy(valid)
        nan_records[0]["loss"] = math.nan
        cases.append(
            (
                "nan",
                "".join(json.dumps(record) + "\n" for record in nan_records),
                "strict JSON",
            )
        )

        string_step = copy.deepcopy(valid)
        string_step[0]["step"] = "1"
        cases.append(
            (
                "string-step",
                "".join(json.dumps(record) + "\n" for record in string_step),
                "step must be",
            )
        )

        duplicate_step = copy.deepcopy(valid)
        duplicate_step[1]["step"] = 1
        cases.append(
            (
                "duplicate-step",
                "".join(json.dumps(record) + "\n" for record in duplicate_step),
                "steps must be exactly",
            )
        )

        negative_rate = copy.deepcopy(valid)
        negative_rate[0]["learning_rate"] = -0.1
        cases.append(
            (
                "negative-rate",
                "".join(json.dumps(record) + "\n" for record in negative_rate),
                "learning_rate",
            )
        )
        cases.append(("syntax", "{not-json}\n", "strict JSON"))
        cases.append(
            (
                "missing-validation",
                "".join(json.dumps(record) + "\n" for record in valid[:-1]),
                "end with exactly one validation",
            )
        )

        for name, contents, pattern in cases:
            with self.subTest(case=name):
                directory, _metadata, _metrics, _bytes = self.write_stage(f"metric-{name}")
                (directory / "metrics.jsonl").write_text(contents, encoding="utf-8")
                self.assert_invalid([directory], pattern)

    def test_timestamp_update_and_file_bounds_are_enforced(self) -> None:
        bad_time, _metadata, _metrics, _bytes = self.write_stage(
            "bad-time",
            metadata_changes={"finished_at_unix": 999},
        )
        self.assert_invalid([bad_time], "precedes")

        bad_updates, _metadata, _metrics, _bytes = self.write_stage(
            "bad-updates",
            metadata_changes={"updates": 3},
        )
        self.assert_invalid([bad_updates], "steps must be exactly")

        oversized, _metadata, _metrics, metadata_bytes = self.write_stage("oversized")
        with mock.patch.object(provenance, "MAX_METADATA_BYTES", len(metadata_bytes) - 1):
            self.assert_invalid([oversized], "exceeds")

        irregular, _metadata, _metrics, _bytes = self.write_stage("irregular")
        metrics_path = irregular / "metrics.jsonl"
        metrics_path.unlink()
        metrics_path.mkdir()
        self.assert_invalid([irregular], "not a regular file")

    def test_real_client_path_requires_credentials_only_after_validation(self) -> None:
        directory, _metadata, _metrics, metadata_bytes = self.write_stage(
            "credential-check"
        )
        calibration_path, calibration, _assets = self.write_calibration_manifest(
            directory, metadata_bytes
        )
        arguments = [
            "--training-dir",
            str(directory),
            "--artifact-digest",
            calibration["artifact_tree_digest"],
            "--finished-block",
            "1",
            "--calibration-manifest",
            str(calibration_path),
        ]
        with (
            mock.patch.dict(provenance.os.environ, {}, clear=True),
            mock.patch.object(provenance.importlib, "import_module") as import_module,
            self.assertRaisesRegex(SystemExit, "WANDB_API_KEY"),
        ):
            provenance.main(arguments)
        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
