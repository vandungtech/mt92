from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from training import train_extract as train
except ModuleNotFoundError as exc:
    train = None
    TRAINING_DEPENDENCY_ERROR = str(exc)
else:
    TRAINING_DEPENDENCY_ERROR = ""


@unittest.skipIf(
    train is None,
    f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}",
)
class TrainingInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_existing_snapshot_and_single_file_derived_inputs_are_unchanged(self) -> None:
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        weights = snapshot / "model.safetensors"
        tokenizer = snapshot / "tokenizer.json"
        weights.write_bytes(b"pinned-weights")
        tokenizer.write_bytes(b"pinned-tokenizer")
        marker = (
            snapshot
            / ".cache"
            / "huggingface"
            / "trees"
            / f"{train.BASE_REVISION}.json"
        )
        marker.parent.mkdir(parents=True)
        marker.write_text("{}\n")
        with (
            mock.patch.object(train, "BASE_WEIGHTS_DIGEST", train.sha256(weights)),
            mock.patch.object(train, "BASE_TOKENIZER_DIGEST", train.sha256(tokenizer)),
        ):
            self.assertEqual(
                train.verify_training_input(snapshot),
                {
                    "kind": "huggingface_snapshot",
                    "revision": train.BASE_REVISION,
                    "weights_digest": train.sha256(weights),
                    "tokenizer_digest": train.sha256(tokenizer),
                },
            )

        run = self.root / "derived-run"
        merged = run / "merged"
        merged.mkdir(parents=True)
        derived_weights = merged / "model.safetensors"
        derived_tokenizer = merged / "tokenizer.json"
        derived_weights.write_bytes(b"derived-weights")
        derived_tokenizer.write_bytes(b"derived-tokenizer")
        parent = run / "training_metadata.json"
        parent.write_text(
            json.dumps({"base_model": f"{train.BASE_MODEL}@{train.BASE_REVISION}"}) + "\n"
        )
        self.assertEqual(
            train.verify_training_input(merged),
            {
                "kind": "derived_model",
                "parent_metadata_digest": train.sha256(parent),
                "weights_digest": train.sha256(derived_weights),
                "tokenizer_digest": train.sha256(derived_tokenizer),
            },
        )

    def test_soup_dispatch_records_all_validated_lineage_digests(self) -> None:
        model_dir = self.root / "soup"
        model_dir.mkdir()
        (model_dir / train.weight_soup.METADATA_FILENAME).write_text("{}\n")
        validated = train.weight_soup.ValidatedSoupCheckpoint(
            metadata_digest="sha256:" + "a" * 64,
            output_manifest_digest="sha256:" + "b" * 64,
            index_digest="sha256:" + "c" * 64,
            tokenizer_digest="sha256:" + "d" * 64,
        )
        with mock.patch.object(
            train.weight_soup,
            "validate_weight_soup_checkpoint",
            return_value=validated,
        ) as validator:
            identity = train.verify_training_input(model_dir)

        validator.assert_called_once_with(model_dir)
        self.assertEqual(
            identity,
            {
                "kind": "deterministic_weight_soup",
                "soup_schema": train.weight_soup.SCHEMA,
                "soup_metadata_digest": validated.metadata_digest,
                "output_manifest_digest": validated.output_manifest_digest,
                "index_digest": validated.index_digest,
                "tokenizer_digest": validated.tokenizer_digest,
            },
        )

    def test_partial_soup_marker_fails_closed_instead_of_using_single_file_fallback(self) -> None:
        run = self.root / "partial-run"
        model_dir = run / "merged"
        model_dir.mkdir(parents=True)
        (model_dir / train.weight_soup.INDEX_FILENAME).write_text("{}\n")
        (model_dir / "model.safetensors").write_bytes(b"otherwise-derived")
        (model_dir / "tokenizer.json").write_text("{}\n")
        (run / "training_metadata.json").write_text(
            json.dumps({"base_model": f"{train.BASE_MODEL}@{train.BASE_REVISION}"}) + "\n"
        )
        error = train.weight_soup.SoupValidationError("partial soup rejected")
        with (
            mock.patch.object(
                train.weight_soup,
                "validate_weight_soup_checkpoint",
                side_effect=error,
            ),
            self.assertRaisesRegex(train.weight_soup.SoupValidationError, "partial soup rejected"),
        ):
            train.verify_training_input(model_dir)


if __name__ == "__main__":
    unittest.main()

