from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest import mock

from training import boundary_contrastive as boundary
from training import publish_provenance as provenance


def encoded_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}


class FakeWandb:
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


class BoundaryPublisherReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture_index = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def row(index: int) -> dict[str, Any]:
        text = f"xDrug{index}Y caused illness{index}."
        gold = {
            "entities": [
                {"text": f"Drug{index}", "type": "Chemical"},
                {"text": f"illness{index}", "type": "Disease"},
            ]
        }
        return {
            "ref": f"row-{index:04d}",
            "partition": "train",
            "inputs": {"text": text},
            "prompt": "Extract exact entities.\n\nText: " + text,
            "gold": json.dumps(gold),
        }

    @contextmanager
    def fixture(self) -> Iterator[dict[str, Any]]:
        self.fixture_index += 1
        rows = [self.row(index) for index in range(boundary.BOUNDARY_CORPUS_TRAIN_EXAMPLES)]
        corpus_payload = json.dumps(
            {
                "version": boundary.CORPUS_VERSION,
                "track": "extract",
                "tasks": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        corpus_path = self.root / f"public-corpus-{self.fixture_index}.json"
        corpus_path.write_bytes(corpus_payload)
        _outer, remaining = boundary.boundary_outer_partition(rows)
        skipped_refs = tuple(sorted(row["ref"] for row in remaining[:2]))
        corpus_digest = boundary.digest_bytes(corpus_payload)

        with mock.patch.multiple(
            boundary,
            BOUNDARY_CORPUS_BYTES=len(corpus_payload),
            BOUNDARY_CORPUS_FILE_DIGEST=corpus_digest,
            BOUNDARY_SKIPPED_ENCODED_REFS=skipped_refs,
        ):
            manifest, _fold, corruption = boundary.build_boundary_manifest(
                boundary.parse_boundary_corpus(corpus_payload),
                skipped_refs=skipped_refs,
                requested_examples=512,
                pairwise_lambda=0.1,
                margin=0.0,
            )
            stage_directory = self.root / f"stage-{self.fixture_index}"
            stage_directory.mkdir()
            manifest_path = stage_directory / "boundary_contrastive_manifest.json"
            manifest_payload = encoded_json(manifest)
            manifest_path.write_bytes(manifest_payload)
            manifest_digest = boundary.digest_bytes(manifest_payload)

            settings = {
                "epochs": 1,
                "seed": 92,
                "max_length": 512,
                "validation_examples": 384,
                "disease_row_weight": 1.0,
                "gold_canonicalization": "none",
                "entity_text_token_weight": 1.0,
                "entity_substitution_examples": 0,
                "boundary_contrastive": True,
                "boundary_contrastive_examples": 512,
                "boundary_contrastive_lambda": 0.1,
                "boundary_contrastive_margin": 0.0,
            }
            metadata = {
                "hotkey": provenance.HOTKEY,
                "track": provenance.TRACK,
                "hardware_class": provenance.HARDWARE_CLASS,
                "base_model": provenance.PINNED_BASE_MODEL,
                "training_input": {
                    "kind": "huggingface_snapshot",
                    "revision": provenance.BASE_REVISION,
                    "weights_digest": provenance.BASE_WEIGHTS_DIGEST,
                    "tokenizer_digest": provenance.BASE_TOKENIZER_DIGEST,
                },
                "corpus_version": provenance.CORPUS_VERSION,
                "corpus_file_digest": corpus_digest,
                "settings": settings,
                "target_controls": {
                    "entity_match": "exact_text_and_type_set",
                    "gold_canonicalization": "none",
                    "entity_text_token_weight": 1.0,
                    "validation_loss": "ordinary_unweighted_causal_lm",
                },
                "augmentation": {
                    "entity_substitution": {
                        "algorithm": provenance.ENTITY_SUBSTITUTION_ALGORITHM,
                        "enabled": False,
                        "seed": 92,
                        "requested_examples": 0,
                        "augmented_examples": 0,
                        "replacement_count": 0,
                        "composition": "disabled_for_boundary_contrastive",
                    }
                },
                "boundary_contrastive": boundary.boundary_metadata_summary(
                    manifest, manifest_digest=manifest_digest
                ),
                "source_training_examples": 4_046,
                "training_examples": 4_046,
                "validation_examples": 384,
                "skipped_training_examples": 0,
                "skipped_validation_examples": 0,
                "disease_source_examples": 4_046,
                "disease_extra_examples": 0,
                "started_at_unix": 1_000,
                "finished_at_unix": 1_003,
                "elapsed_s": 2.0,
                "updates": 1,
            }
            metadata_payload = encoded_json(metadata)
            (stage_directory / "training_metadata.json").write_bytes(metadata_payload)
            validation_loss = 0.3
            records = [
                {
                    "step": 1,
                    "epoch": 1,
                    "loss": 0.5,
                    "learning_rate": 0.0001,
                    "elapsed_s": 1.0,
                },
                {
                    "step": 1,
                    "epoch": 1,
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(validation_loss),
                    "elapsed_s": 1.5,
                },
            ]
            (stage_directory / "metrics.jsonl").write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            yield {
                "corpus_path": corpus_path,
                "corpus_payload": corpus_payload,
                "corpus_digest": corpus_digest,
                "stage_directory": stage_directory,
                "manifest": manifest,
                "manifest_path": manifest_path,
                "manifest_digest": manifest_digest,
                "metadata": metadata,
                "metadata_payload": metadata_payload,
                "corruption": corruption,
            }

    def validate_fixture(self, fixture: dict[str, Any]) -> provenance.Publication:
        calibration = provenance.CalibrationLineage(
            manifest={"schema": "fixture"},
            manifest_digest="sha256:" + "c" * 64,
        )
        with mock.patch.object(
            provenance, "_validate_calibration_lineage", return_value=calibration
        ):
            return provenance.validate_publication(
                [fixture["stage_directory"]],
                "sha256:" + "a" * 64,
                8_955_436,
                calibration_manifest=self.root / "fixture-calibration.json",
                boundary_corpora={1: fixture["corpus_path"]},
            )

    def test_full_replay_and_exact_wandb_config(self) -> None:
        with self.fixture() as fixture:
            publication = self.validate_fixture(fixture)

            self.assertEqual(len(publication.boundary_contrastive), 1)
            lineage = publication.boundary_contrastive[0]
            self.assertEqual(
                fixture["corruption"].manifest["direction_counts"],
                {"expansion": 256, "contraction": 256},
            )
            self.assertEqual(lineage.manifest, fixture["manifest"])
            self.assertEqual(lineage.manifest_digest, fixture["manifest_digest"])
            self.assertEqual(lineage.corpus_digest, fixture["corpus_digest"])

            fake = FakeWandb()
            provenance.publish(publication, fake)
            self.assertEqual(
                fake.init_calls[0]["config"]["mt_boundary_contrastive_lineage"],
                {
                    "stage_1": {
                        "schema": boundary.BOUNDARY_SCHEMA,
                        "manifest_digest": fixture["manifest_digest"],
                        "corpus_digest": fixture["corpus_digest"],
                        "manifest": fixture["manifest"],
                    }
                },
            )

            legacy = provenance.Publication(
                stages=publication.stages,
                artifact_digest=publication.artifact_digest,
                finished_block=publication.finished_block,
                calibration=publication.calibration,
                weight_soups=(),
                boundary_contrastive=(),
            )
            legacy_fake = FakeWandb()
            provenance.publish(legacy, legacy_fake)
            self.assertNotIn(
                "mt_boundary_contrastive_lineage",
                legacy_fake.init_calls[0]["config"],
            )

    def test_mapping_must_exactly_match_boundary_stages(self) -> None:
        with self.fixture() as fixture:
            calibration = provenance.CalibrationLineage(
                manifest={"schema": "fixture"},
                manifest_digest="sha256:" + "c" * 64,
            )
            with (
                mock.patch.object(
                    provenance,
                    "_validate_calibration_lineage",
                    return_value=calibration,
                ),
                self.assertRaisesRegex(
                    provenance.ProvenanceValidationError,
                    r"missing=\[1\], extra=\[2\]",
                ),
            ):
                provenance.validate_publication(
                    [fixture["stage_directory"]],
                    "sha256:" + "a" * 64,
                    8_955_436,
                    calibration_manifest=self.root / "fixture-calibration.json",
                    boundary_corpora={2: fixture["corpus_path"]},
                )

    def test_manifest_and_metadata_tampering_fail_replay(self) -> None:
        with self.fixture() as fixture:
            tampered = copy.deepcopy(fixture["manifest"])
            record = tampered["corruption"]["examples"][0]
            record["direction"] = (
                "expansion" if record["direction"] == "contraction" else "contraction"
            )
            tampered_payload = encoded_json(tampered)
            fixture["manifest_path"].write_bytes(tampered_payload)
            fixture["metadata"]["boundary_contrastive"]["manifest_digest"] = (
                boundary.digest_bytes(tampered_payload)
            )
            (fixture["stage_directory"] / "training_metadata.json").write_bytes(
                encoded_json(fixture["metadata"])
            )
            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "manifest does not match independent corpus replay",
            ):
                self.validate_fixture(fixture)

        with self.fixture() as fixture:
            fixture["metadata"]["boundary_contrastive"]["split"][
                "inner_train_examples"
            ] = 4_045
            (fixture["stage_directory"] / "training_metadata.json").write_bytes(
                encoded_json(fixture["metadata"])
            )
            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "metadata summary does not match",
            ):
                self.validate_fixture(fixture)

        with self.fixture() as fixture:
            fixture["metadata"]["disease_source_examples"] = 4_045
            (fixture["stage_directory"] / "training_metadata.json").write_bytes(
                encoded_json(fixture["metadata"])
            )
            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "disease_source_examples must be exactly 4046",
            ):
                self.validate_fixture(fixture)

    def test_exact_tokenizer_and_corpus_identity_are_required(self) -> None:
        with self.fixture() as fixture:
            fixture["metadata"]["training_input"]["tokenizer_digest"] = (
                "sha256:" + "0" * 64
            )
            (fixture["stage_directory"] / "training_metadata.json").write_bytes(
                encoded_json(fixture["metadata"])
            )
            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "tokenizer digest is not allowlisted",
            ):
                self.validate_fixture(fixture)

        with self.fixture() as fixture:
            fixture["corpus_path"].write_bytes(fixture["corpus_payload"] + b" ")
            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "pinned public corpus|exactly|exceeds",
            ):
                self.validate_fixture(fixture)

    def test_symlinked_corpus_ancestor_is_rejected(self) -> None:
        with self.fixture() as fixture:
            actual_directory = self.root / "actual-evidence"
            actual_directory.mkdir()
            actual_corpus = actual_directory / "corpus.json"
            actual_corpus.write_bytes(fixture["corpus_payload"])
            linked_directory = self.root / "linked-evidence"
            os.symlink(actual_directory, linked_directory)
            fixture["corpus_path"] = linked_directory / "corpus.json"

            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "symlink, unsafe component, or unreadable",
            ):
                self.validate_fixture(fixture)

    def test_symlinked_boundary_manifest_ancestor_is_rejected(self) -> None:
        with self.fixture() as fixture:
            linked_stage = self.root / "linked-stage"
            os.symlink(fixture["stage_directory"], linked_stage)
            fixture["stage_directory"] = linked_stage

            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "symlink, unsafe component, or unreadable",
            ):
                self.validate_fixture(fixture)

    def test_boundary_corpus_and_manifest_reject_hardlink_alias(self) -> None:
        with self.fixture() as fixture:
            fixture["manifest_path"].unlink()
            os.link(fixture["corpus_path"], fixture["manifest_path"])

            with self.assertRaisesRegex(
                provenance.ProvenanceValidationError,
                "corpus and manifest must be distinct files",
            ):
                self.validate_fixture(fixture)

    def test_main_passes_stage_indexed_boundary_corpus(self) -> None:
        publication = SimpleNamespace(record_count=0, stages=(object(),))
        fake_client = object()
        with (
            mock.patch.object(
                provenance, "validate_publication", return_value=publication
            ) as validate,
            mock.patch.object(provenance, "publish") as publish,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                provenance.main(
                    [
                        "--training-dir",
                        "stage",
                        "--artifact-digest",
                        "sha256:" + "a" * 64,
                        "--finished-block",
                        "8955436",
                        "--calibration-manifest",
                        "calibration.json",
                        "--boundary-corpus",
                        "1",
                        "public.json",
                    ],
                    wandb_client=fake_client,
                ),
                0,
            )
        self.assertEqual(
            validate.call_args.kwargs["boundary_corpora"], {1: Path("public.json")}
        )
        publish.assert_called_once_with(publication, fake_client)

    def test_boundary_cli_mapping_is_canonical_and_unique(self) -> None:
        self.assertEqual(
            provenance._index_boundary_corpora(
                [["1", "first.json"], ["2", "second.json"]]
            ),
            {1: Path("first.json"), 2: Path("second.json")},
        )
        for entries, pattern in (
            ([["01", "first.json"]], "canonical positive"),
            ([["0", "first.json"]], "canonical positive"),
            (
                [["1", "first.json"], ["1", "second.json"]],
                "duplicate boundary corpus",
            ),
            ([["1", " first.json"]], "path for stage 1 is invalid"),
        ):
            with self.subTest(entries=entries), self.assertRaisesRegex(
                provenance.ProvenanceValidationError, pattern
            ):
                provenance._index_boundary_corpora(entries)


if __name__ == "__main__":
    unittest.main()
