from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from training import build_weight_soup as soup
except ModuleNotFoundError as exc:
    torch = None
    SOUP_DEPENDENCY_ERROR = str(exc)
else:
    SOUP_DEPENDENCY_ERROR = ""


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {SOUP_DEPENDENCY_ERROR}"
)
class BuildWeightSoupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.base_config = {
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 10,
            "eos_token_id": 11,
            "head_dim": 2,
            "hidden_act": "silu",
            "hidden_size": 2,
            "initializer_range": 0.02,
            "intermediate_size": 4,
            "max_position_embeddings": 32,
            "max_window_layers": 1,
            "model_type": "qwen3",
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_scaling": None,
            "rope_theta": 1_000_000,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "fixture-old",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 2,
        }
        (self.base / "config.json").write_text(json.dumps(self.base_config))
        support = {
            "generation_config.json": b'{"fixture":true}\n',
            "merges.txt": b"a b\n",
            "tokenizer.json": b'{"tokenizer":"base"}\n',
            "tokenizer_config.json": b'{"chat_template":"fixture"}\n',
            "vocab.json": b'{"a":0}\n',
        }
        for name, payload in support.items():
            (self.base / name).write_bytes(payload)
        self.base_tensors = {
            "lm_head.weight": torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
            "model.embed_tokens.weight": torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
            "model.layers.0.weight": torch.tensor([[10, 20], [30, 40]], dtype=torch.bfloat16),
        }
        save_file(
            self.base_tensors,
            self.base / soup.MODEL_FILENAME,
            metadata={"format": "pt"},
        )
        base_files = {path.name: digest(path) for path in self.base.iterdir() if path.is_file()}
        base_specs, _metadata = soup._inspect_safetensors(
            self.base / soup.MODEL_FILENAME,
            "fixture base",
        )
        effective_specs = {
            name: spec for name, spec in base_specs.items() if name != "lm_head.weight"
        }
        self.allowlist = soup.BaseAllowlist(
            model="Fixture/Qwen3",
            revision="fixture-revision",
            files=base_files,
            architecture_digest=soup._manifest_digest(
                soup._normalize_config(self.base_config, "fixture base")
            ),
            tensor_schema_digest=soup._manifest_digest(soup._schema_records(effective_specs)),
            copy_files=tuple(sorted(name for name in base_files if name != soup.MODEL_FILENAME)),
            tied_aliases=(("lm_head.weight", "model.embed_tokens.weight"),),
        )
        self.source1 = self.write_source(
            "run-one",
            {
                "model.embed_tokens.weight": torch.tensor([[5, 6], [7, 8]], dtype=torch.bfloat16),
                "model.layers.0.weight": torch.tensor([[14, 24], [34, 44]], dtype=torch.bfloat16),
            },
        )
        self.source2 = self.write_source(
            "run-two",
            {
                "model.embed_tokens.weight": torch.tensor(
                    [[9, 10], [11, 12]], dtype=torch.bfloat16
                ),
                "model.layers.0.weight": torch.tensor([[18, 28], [38, 48]], dtype=torch.bfloat16),
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_config(self) -> dict[str, object]:
        config = copy.deepcopy(self.base_config)
        config.pop("torch_dtype")
        config.pop("rope_scaling")
        config.pop("rope_theta")
        config["dtype"] = "bfloat16"
        config["rope_parameters"] = {"rope_theta": 1_000_000, "rope_type": "default"}
        config["layer_types"] = ["full_attention"]
        config["pad_token_id"] = None
        config["transformers_version"] = "fixture-new"
        return config

    def write_source(self, name: str, tensors: dict[str, torch.Tensor]) -> Path:
        run = self.root / name
        merged = run / "merged"
        merged.mkdir(parents=True)
        save_file(tensors, merged / soup.MODEL_FILENAME, metadata={"format": "pt"})
        (merged / "config.json").write_text(json.dumps(self.source_config()))
        (merged / "tokenizer.json").write_text(json.dumps({"source": name}))
        metadata = {
            "base_model": self.allowlist.identity,
            "elapsed_s": 1.5,
            "finished_at_unix": 100,
            "training_input": {
                "kind": "huggingface_snapshot",
                "revision": self.allowlist.revision,
                "tokenizer_digest": self.allowlist.files["tokenizer.json"],
                "weights_digest": self.allowlist.files[soup.MODEL_FILENAME],
            },
            "updates": 2,
        }
        (run / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        return merged

    def build(self, output: str = "soup") -> tuple[Path, dict[str, object]]:
        destination = self.root / output
        metadata = soup.build_weight_soup(
            base_dir=self.base,
            sources=(
                soup.SourceInput(self.source1, "1"),
                soup.SourceInput(self.source2, "3"),
            ),
            output_dir=destination,
            chunk_bytes=4,
            allowlist=self.allowlist,
        )
        return destination, metadata

    @staticmethod
    def read_output(path: Path) -> dict[str, torch.Tensor]:
        index = json.loads((path / soup.INDEX_FILENAME).read_text())
        tensors: dict[str, torch.Tensor] = {}
        for name, shard_name in index["weight_map"].items():
            with safe_open(path / shard_name, framework="pt", device="cpu") as handle:
                tensors[name] = handle.get_tensor(name)
        return tensors

    def test_builds_exact_delta_average_and_preserves_base_identity_files(self) -> None:
        destination, metadata = self.build()
        actual = self.read_output(destination)

        self.assertNotIn("lm_head.weight", actual)
        expected_embed = torch.tensor([[8, 9], [10, 11]], dtype=torch.bfloat16)
        expected_layer = torch.tensor([[17, 27], [37, 47]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual["model.embed_tokens.weight"], expected_embed))
        self.assertTrue(torch.equal(actual["model.layers.0.weight"], expected_layer))
        for name in self.allowlist.copy_files:
            self.assertEqual((destination / name).read_bytes(), (self.base / name).read_bytes())
        self.assertEqual(
            [source["normalized_weight_decimal"] for source in metadata["sources"]],
            ["0.25", "0.75"],
        )
        self.assertEqual(
            math.fsum(
                float.fromhex(source["normalized_weight_float_hex"])
                for source in metadata["sources"]
            ),
            1.0,
        )

    def test_metadata_binds_sources_parents_settings_and_every_output_hash(self) -> None:
        destination, metadata = self.build()
        first = metadata["sources"][0]
        self.assertEqual(
            first["parent_training_metadata"]["sha256"],
            digest(self.source1.parent / "training_metadata.json"),
        )
        self.assertEqual(
            first["files"][soup.MODEL_FILENAME]["sha256"],
            digest(self.source1 / soup.MODEL_FILENAME),
        )
        self.assertEqual(metadata["algorithm"]["accumulation_dtype"], "float32")
        records = metadata["output"]["files"]
        self.assertEqual(metadata["output"]["manifest_sha256"], soup._manifest_digest(records))
        for record in records:
            self.assertEqual(record["sha256"], digest(destination / record["path"]))
        on_disk = json.loads((destination / soup.METADATA_FILENAME).read_text())
        self.assertEqual(on_disk, metadata)

    def test_repeated_build_is_byte_deterministic(self) -> None:
        first, first_metadata = self.build("first")
        second, second_metadata = self.build("second")
        self.assertEqual(first_metadata, second_metadata)
        first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
        self.assertEqual(first_files, second_files)

    def test_refuses_architecture_key_shape_dtype_and_tied_alias_mismatch(self) -> None:
        cases: list[tuple[str, dict[str, torch.Tensor] | None, str]] = [
            (
                "missing-key",
                {"model.embed_tokens.weight": self.base_tensors["model.embed_tokens.weight"]},
                "tensor keys mismatch",
            ),
            (
                "shape",
                {
                    "model.embed_tokens.weight": torch.ones((3, 2), dtype=torch.bfloat16),
                    "model.layers.0.weight": self.base_tensors["model.layers.0.weight"],
                },
                "schema mismatch",
            ),
            (
                "dtype",
                {
                    "model.embed_tokens.weight": self.base_tensors[
                        "model.embed_tokens.weight"
                    ].float(),
                    "model.layers.0.weight": self.base_tensors["model.layers.0.weight"],
                },
                "schema mismatch",
            ),
            (
                "tied",
                {
                    **self.base_tensors,
                    "lm_head.weight": self.base_tensors["lm_head.weight"] + 1,
                },
                "differ in value",
            ),
        ]
        for name, tensors, pattern in cases:
            with self.subTest(name=name):
                source = self.write_source(name, tensors or {})
                with self.assertRaisesRegex(soup.SoupValidationError, pattern):
                    soup.build_weight_soup(
                        base_dir=self.base,
                        sources=(
                            soup.SourceInput(source, "1"),
                            soup.SourceInput(self.source2, "1"),
                        ),
                        output_dir=self.root / f"out-{name}",
                        chunk_bytes=4,
                        allowlist=self.allowlist,
                    )

        bad_config = self.source1 / "config.json"
        config = json.loads(bad_config.read_text())
        config["hidden_size"] = 3
        bad_config.write_text(json.dumps(config))
        with self.assertRaisesRegex(soup.SoupValidationError, "architecture"):
            self.build("bad-architecture")

    def test_refuses_wrong_base_or_parent_identity_and_nonfinite_tensor(self) -> None:
        vocab = self.base / "vocab.json"
        vocab.write_text("changed")
        with self.assertRaisesRegex(soup.SoupValidationError, "allowlisted snapshot"):
            self.build("bad-base")
        vocab.write_bytes(b'{"a":0}\n')

        parent = self.source1.parent / "training_metadata.json"
        metadata = json.loads(parent.read_text())
        metadata["base_model"] = "wrong/base"
        parent.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(soup.SoupValidationError, "allowlisted base"):
            self.build("bad-parent")
        metadata["base_model"] = self.allowlist.identity
        parent.write_text(json.dumps(metadata))

        tensors = {
            "model.embed_tokens.weight": self.base_tensors["model.embed_tokens.weight"].clone(),
            "model.layers.0.weight": self.base_tensors["model.layers.0.weight"].clone(),
        }
        tensors["model.layers.0.weight"][0, 0] = float("nan")
        bad = self.write_source("nonfinite", tensors)
        with self.assertRaisesRegex(soup.SoupValidationError, "non-finite"):
            soup.build_weight_soup(
                base_dir=self.base,
                sources=(soup.SourceInput(bad, "0"), soup.SourceInput(self.source2, "1")),
                output_dir=self.root / "bad-finite",
                chunk_bytes=4,
                allowlist=self.allowlist,
            )

    def test_weights_and_atomic_destination_fail_closed(self) -> None:
        invalid = ("nan", "inf", "-1")
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(soup.SoupValidationError):
                soup.normalize_weights(
                    (soup.SourceInput(self.source1, value), soup.SourceInput(self.source2, "1"))
                )
        with self.assertRaisesRegex(soup.SoupValidationError, "positive"):
            soup.normalize_weights(
                (soup.SourceInput(self.source1, "0"), soup.SourceInput(self.source2, "0"))
            )
        destination = self.root / "existing"
        destination.mkdir()
        sentinel = destination / "sentinel"
        sentinel.write_text("untouched")
        with self.assertRaisesRegex(soup.SoupValidationError, "already exists"):
            soup.build_weight_soup(
                base_dir=self.base,
                sources=(
                    soup.SourceInput(self.source1, "1"),
                    soup.SourceInput(self.source2, "1"),
                ),
                output_dir=destination,
                allowlist=self.allowlist,
            )
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertEqual(list(self.root.glob(".failed.staging-*")), [])

    def test_rejects_symlink_source_and_duplicate_source(self) -> None:
        linked = self.root / "linked"
        os.symlink(self.source1, linked)
        with self.assertRaisesRegex(soup.SoupValidationError, "non-symlink directory"):
            soup.build_weight_soup(
                base_dir=self.base,
                sources=(soup.SourceInput(linked, "1"), soup.SourceInput(self.source2, "1")),
                output_dir=self.root / "linked-output",
                allowlist=self.allowlist,
            )
        with self.assertRaisesRegex(soup.SoupValidationError, "distinct"):
            soup.build_weight_soup(
                base_dir=self.base,
                sources=(
                    soup.SourceInput(self.source1, "1"),
                    soup.SourceInput(self.source1, "1"),
                ),
                output_dir=self.root / "duplicate-output",
                allowlist=self.allowlist,
            )

    def refresh_output_record(
        self,
        destination: Path,
        metadata: dict[str, object],
        filename: str,
    ) -> None:
        records = metadata["output"]["files"]
        record = next(item for item in records if item["path"] == filename)
        record["bytes"] = (destination / filename).stat().st_size
        record["sha256"] = digest(destination / filename)
        metadata["output"]["manifest_sha256"] = soup._manifest_digest(records)
        (destination / soup.METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    def test_validates_published_checkpoint_and_ignores_nonweight_sidecars(self) -> None:
        destination, metadata = self.build("validated")
        (destination / "heldout-results.jsonl").write_text("{}\n")
        (destination / "model-q4-imatrix.gguf").write_bytes(b"evaluation-sidecar")

        validated = soup.validate_weight_soup_checkpoint(destination, self.allowlist)

        self.assertEqual(validated.metadata_digest, digest(destination / soup.METADATA_FILENAME))
        self.assertEqual(
            validated.output_manifest_digest,
            metadata["output"]["manifest_sha256"],
        )
        self.assertEqual(validated.index_digest, digest(destination / soup.INDEX_FILENAME))
        self.assertEqual(validated.tokenizer_digest, digest(destination / "tokenizer.json"))

    def test_checkpoint_validation_rejects_missing_and_unlisted_weight_shards(self) -> None:
        missing, missing_metadata = self.build("missing-shard")
        missing_shard = next(
            record["path"]
            for record in missing_metadata["output"]["files"]
            if record["path"].endswith(".safetensors")
        )
        (missing / missing_shard).unlink()
        with self.assertRaisesRegex(soup.SoupValidationError, "readable regular file"):
            soup.validate_weight_soup_checkpoint(missing, self.allowlist)

        extra, _extra_metadata = self.build("extra-shard")
        (extra / "extra.safetensors").write_bytes(b"undeclared")
        with self.assertRaisesRegex(soup.SoupValidationError, "unlisted weight shards"):
            soup.validate_weight_soup_checkpoint(extra, self.allowlist)

    def test_checkpoint_validation_derives_schema_and_finiteness_from_shards(self) -> None:
        forged, forged_metadata = self.build("forged-schema")
        index = json.loads((forged / soup.INDEX_FILENAME).read_text())
        tensor_name, shard_name = next(iter(index["weight_map"].items()))
        del index["weight_map"][tensor_name]
        index["weight_map"]["forged.weight"] = shard_name
        save_file(
            {"forged.weight": torch.zeros((2, 2), dtype=torch.bfloat16)},
            forged / shard_name,
            metadata={"format": "pt"},
        )
        (forged / soup.INDEX_FILENAME).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        self.refresh_output_record(forged, forged_metadata, shard_name)
        self.refresh_output_record(forged, forged_metadata, soup.INDEX_FILENAME)
        with self.assertRaisesRegex(soup.SoupValidationError, "tensor schema"):
            soup.validate_weight_soup_checkpoint(forged, self.allowlist)

        nonfinite, nonfinite_metadata = self.build("nonfinite-checkpoint")
        nonfinite_index = json.loads((nonfinite / soup.INDEX_FILENAME).read_text())
        tensor_name, shard_name = next(iter(nonfinite_index["weight_map"].items()))
        bad_tensor = torch.zeros((2, 2), dtype=torch.bfloat16)
        bad_tensor[0, 0] = float("nan")
        save_file(
            {tensor_name: bad_tensor},
            nonfinite / shard_name,
            metadata={"format": "pt"},
        )
        self.refresh_output_record(nonfinite, nonfinite_metadata, shard_name)
        with self.assertRaisesRegex(soup.SoupValidationError, "non-finite"):
            soup.validate_weight_soup_checkpoint(nonfinite, self.allowlist)


    def test_checkpoint_validation_rejects_tampered_files_index_base_and_parent(self) -> None:
        corrupt, corrupt_metadata = self.build("corrupt-output")
        shard = next(
            record["path"]
            for record in corrupt_metadata["output"]["files"]
            if record["path"].endswith(".safetensors")
        )
        with (corrupt / shard).open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(soup.SoupValidationError, "does not match its record"):
            soup.validate_weight_soup_checkpoint(corrupt, self.allowlist)

        bad_index, index_metadata = self.build("bad-index")
        index = json.loads((bad_index / soup.INDEX_FILENAME).read_text())
        index["weight_map"].pop(next(iter(index["weight_map"])))
        (bad_index / soup.INDEX_FILENAME).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        self.refresh_output_record(bad_index, index_metadata, soup.INDEX_FILENAME)
        with self.assertRaisesRegex(soup.SoupValidationError, "index summary"):
            soup.validate_weight_soup_checkpoint(bad_index, self.allowlist)

        bad_base, base_metadata = self.build("bad-checkpoint-base")
        base_metadata["base"]["identity"] = "Wrong/Base@revision"
        (bad_base / soup.METADATA_FILENAME).write_text(
            json.dumps(base_metadata, indent=2, sort_keys=True) + "\n"
        )
        with self.assertRaisesRegex(soup.SoupValidationError, "allowlisted base"):
            soup.validate_weight_soup_checkpoint(bad_base, self.allowlist)

        bad_parent, parent_metadata = self.build("bad-checkpoint-parent")
        parent_metadata["sources"][0]["parent_training_metadata"].pop("sha256")
        (bad_parent / soup.METADATA_FILENAME).write_text(
            json.dumps(parent_metadata, indent=2, sort_keys=True) + "\n"
        )
        with self.assertRaisesRegex(soup.SoupValidationError, "binding is incomplete"):
            soup.validate_weight_soup_checkpoint(bad_parent, self.allowlist)


if __name__ == "__main__":
    unittest.main()
