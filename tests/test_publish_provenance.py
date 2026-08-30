from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
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

    def invoke_main(self, directories: list[Path], fake: FakeWandb) -> int:
        arguments: list[str] = []
        for directory in directories:
            arguments.extend(("--training-dir", str(directory)))
        arguments.extend(("--artifact-digest", ARTIFACT_DIGEST, "--finished-block", "8955436"))
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
        fake = FakeWandb()

        self.assertEqual(self.invoke_main([directory], fake), 0)

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

        self.assertEqual([step for _payload, step in fake.log_calls], [1, 2, 3])
        for source, (logged, _step) in zip(metrics, fake.log_calls, strict=True):
            self.assertEqual({key: logged[key] for key in source}, source)
            self.assertEqual(logged["mt_stage"], 1)
            self.assertEqual(logged["mt_stage_epoch"], source["epoch"])
            self.assertEqual(logged["mt_stage_step"], source["step"])
        self.assertEqual(
            fake.run.summary,
            {
                "mt_artifact_digest": ARTIFACT_DIGEST,
                "mt_finished_at": 8955436,
                "mt_training_records": 3,
                "mt_training_stages": 1,
            },
        )
        self.assertEqual(fake.finish_calls, 1)

    def test_target_controls_are_validated_and_publish_exact_semantics(self) -> None:
        controls = target_controls(canonicalization="sorted", weight=2.5)
        directory, metadata, _metrics, _bytes = self.write_stage(
            "controlled",
            metadata_changes={
                "settings": target_settings(canonicalization="sorted", weight=2.5),
                "target_controls": controls,
            },
        )
        fake = FakeWandb()

        self.assertEqual(self.invoke_main([directory], fake), 0)

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
        historical, _metadata, _metrics, _bytes = self.write_stage("historical")
        provenance.validate_publication([historical], ARTIFACT_DIGEST, 8955436)

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
        fake = FakeWandb()

        self.assertEqual(self.invoke_main([first, second], fake), 0)

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
        directory, _metadata, _metrics, _bytes = self.write_stage("credential-check")
        arguments = [
            "--training-dir",
            str(directory),
            "--artifact-digest",
            ARTIFACT_DIGEST,
            "--finished-block",
            "1",
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
