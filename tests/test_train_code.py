from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training import code_candidate as candidate
from training import historical_code_candidate as historical_candidate
from training import train_code


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("the raw-completion contract must not add special tokens")
        return {"input_ids": [1, 2] if text == "prompt" else [3, 4, 5]}


class TrainCodeContractTests(unittest.TestCase):
    def test_encode_masks_prompt_and_supervises_complete_module(self) -> None:
        row = {
            "completion": "def task_func():\n    return 1\n",
            "max_output_tokens": 1024,
            "prompt": "prompt",
            "ref": "bigcodebench-1",
        }
        encoded = train_code.encode_record(row, _Tokenizer(), max_length=16)
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5, 99])
        self.assertEqual(encoded["labels"], [-100, -100, 3, 4, 5, 99])
        self.assertEqual(encoded["attention_mask"], [1] * 6)

    def test_encode_refuses_overlength_or_changed_schema(self) -> None:
        row = {
            "completion": "def task_func():\n    return 1\n",
            "max_output_tokens": 1024,
            "prompt": "prompt",
            "ref": "bigcodebench-1",
        }
        with self.assertRaisesRegex(train_code.TrainingRefused, "above max_length"):
            train_code.encode_record(row, _Tokenizer(), max_length=5)
        with self.assertRaisesRegex(train_code.TrainingRefused, "does not match pinned target"):
            train_code.encode_record(
                row,
                _Tokenizer(),
                max_length=16,
                target_eos_token_id=98,
            )
        row["extra"] = True
        with self.assertRaisesRegex(train_code.TrainingRefused, "fields changed"):
            train_code.encode_record(row, _Tokenizer(), max_length=16)

    def test_historical_record_uses_explicit_ref_contract_without_truncation(self) -> None:
        row = {
            "completion": "prose before a fenced solution",
            "max_output_tokens": 1024,
            "prompt": "prompt",
            "ref": "mgc-7",
        }
        encoded = train_code.encode_record(
            row,
            _Tokenizer(),
            max_length=16,
            ref_pattern=historical_candidate.REF_PATTERN,
        )
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5, 99])
        self.assertEqual(encoded["labels"], [-100, -100, 3, 4, 5, 99])
        with self.assertRaisesRegex(train_code.TrainingRefused, "invalid ref"):
            train_code.encode_record(row, _Tokenizer(), max_length=16)
        with self.assertRaisesRegex(train_code.TrainingRefused, "above max_length"):
            train_code.encode_record(
                row,
                _Tokenizer(),
                max_length=5,
                ref_pattern=historical_candidate.REF_PATTERN,
            )

    def test_settings_and_distribution_versions_fail_closed(self) -> None:
        train_code.validate_settings(train_code.Settings())
        with self.assertRaisesRegex(train_code.TrainingRefused, "epochs"):
            train_code.validate_settings(train_code.Settings(epochs=0))
        observed = dict(train_code.EXPECTED_DISTRIBUTIONS)
        for weight in (0.99, 128.01, float("nan"), True):
            with (
                self.subTest(terminal_eos_loss_weight=weight),
                self.assertRaisesRegex(
                    train_code.TrainingRefused,
                    "terminal_eos_loss_weight",
                ),
            ):
                train_code.validate_settings(train_code.Settings(terminal_eos_loss_weight=weight))
        observed["torch"] = "2.13.0+cu130"
        self.assertEqual(
            train_code.validate_distribution_versions(observed)["torch"],
            "2.13.0+cu130",
        )
        observed["transformers"] = "5.16.2"
        with self.assertRaisesRegex(train_code.TrainingRefused, "does not match pinned"):
            train_code.validate_distribution_versions(observed)

    def test_argument_contract_has_one_lora_rank_option(self) -> None:
        args = train_code._parse_args(
            [
                "--dataset",
                str(candidate.TMPFS_MOUNT / "dataset"),
                "--base",
                str(candidate.TMPFS_MOUNT / "base"),
                "--out",
                str(candidate.TMPFS_MOUNT / "out"),
                "--lora-rank",
                "32",
                "--terminal-eos-loss-weight",
                "16",
            ]
        )
        self.assertEqual(args.lora_rank, 32)
        self.assertEqual(args.terminal_eos_loss_weight, 16.0)
        self.assertIsNone(args.base_model)
        self.assertFalse(args.final_all_public)
        final_args = train_code._parse_args(
            [
                "--dataset",
                str(candidate.TMPFS_MOUNT / "dataset"),
                "--base",
                str(candidate.TMPFS_MOUNT / "base"),
                "--out",
                str(candidate.TMPFS_MOUNT / "out"),
                "--base-model",
                candidate.QWEN3_BASE_MODEL,
                "--final-all-public",
            ]
        )
        self.assertEqual(final_args.base_model, candidate.QWEN3_BASE_MODEL)
        self.assertTrue(final_args.final_all_public)
        historical_args = train_code._parse_args(
            [
                "--dataset",
                str(candidate.TMPFS_MOUNT / "historical"),
                "--dataset-profile",
                train_code.HISTORICAL_CORPUS_PROFILE,
                "--source-corpus",
                str(candidate.TMPFS_MOUNT / "historical.json"),
                "--base",
                str(candidate.TMPFS_MOUNT / "base"),
                "--out",
                str(candidate.TMPFS_MOUNT / "out"),
            ]
        )
        self.assertEqual(historical_args.dataset_profile, "historical8000")
        self.assertEqual(historical_args.source_corpus, candidate.TMPFS_MOUNT / "historical.json")

    def test_run_kind_requires_explicit_exact_all_public_split(self) -> None:
        development_manifest = {"train_examples": 78, "holdout_examples": 16}
        development_train = [{} for _ in range(78)]
        development_holdout = [{} for _ in range(16)]
        self.assertEqual(
            train_code.validate_run_kind(
                final_all_public=False,
                dataset_manifest=development_manifest,
                train_rows=development_train,
                holdout_rows=development_holdout,
            ),
            train_code.DEVELOPMENT_RUN_KIND,
        )
        final_manifest = {"train_examples": 94, "holdout_examples": 0}
        final_train = [{} for _ in range(94)]
        with self.assertRaisesRegex(train_code.TrainingRefused, "explicit"):
            train_code.validate_run_kind(
                final_all_public=False,
                dataset_manifest=final_manifest,
                train_rows=final_train,
                holdout_rows=[],
            )
        self.assertEqual(
            train_code.validate_run_kind(
                final_all_public=True,
                dataset_manifest=final_manifest,
                train_rows=final_train,
                holdout_rows=[],
            ),
            train_code.FINAL_ALL_PUBLIC_RUN_KIND,
        )
        with self.assertRaisesRegex(train_code.TrainingRefused, "exactly 94"):
            train_code.validate_run_kind(
                final_all_public=True,
                dataset_manifest=development_manifest,
                train_rows=development_train,
                holdout_rows=development_holdout,
            )
        self.assertEqual(
            train_code.no_holdout_diagnostics(),
            {
                "status": "not_run",
                "examples": 0,
                "claim": train_code.FINAL_ALL_PUBLIC_HOLDOUT_CLAIM,
            },
        )
        historical_manifest = {"train_examples": 8_000, "holdout_examples": 0}
        historical_rows = [{}] * 8_000
        self.assertEqual(
            train_code.validate_run_kind(
                final_all_public=True,
                dataset_manifest=historical_manifest,
                train_rows=historical_rows,
                holdout_rows=[],
                corpus_profile=train_code.HISTORICAL_CORPUS_PROFILE,
            ),
            train_code.FINAL_ALL_PUBLIC_RUN_KIND,
        )
        self.assertEqual(
            train_code.no_holdout_diagnostics(train_code.HISTORICAL_CORPUS_PROFILE)["claim"],
            historical_candidate.FINAL_ALL_PUBLIC_HOLDOUT_CLAIM,
        )

    def test_requested_step_plan_is_exactly_sixty_updates(self) -> None:
        self.assertEqual(
            train_code._optimization_plan(
                20,
                gradient_accumulation=2,
                epochs=6,
                warmup_ratio=0.1,
            ),
            (10, 60, 6),
        )
        self.assertEqual(
            train_code._optimization_plan(
                20,
                gradient_accumulation=2,
                epochs=1,
                warmup_ratio=0.0,
            ),
            (10, 10, 0),
        )
        self.assertEqual(
            train_code._optimization_plan(
                2_000,
                gradient_accumulation=4,
                epochs=2,
                warmup_ratio=0.05,
            ),
            (500, 1_000, 50),
        )

    def test_scheduler_keeps_every_applied_rate_positive(self) -> None:
        multipliers = [
            train_code._cosine_multiplier(step, warmup_steps=6, total_steps=60)
            for step in range(61)
        ]
        self.assertTrue(all(value > 0.0 for value in multipliers[:60]))
        self.assertEqual(multipliers[0], 1 / 6)
        self.assertEqual(multipliers[5], 1.0)
        self.assertLess(multipliers[6], 1.0)
        self.assertGreater(multipliers[59], 0.0)
        self.assertEqual(multipliers[60], 0.0)
        with self.assertRaisesRegex(train_code.TrainingRefused, "scheduler step bounds"):
            train_code._cosine_multiplier(0, warmup_steps=7, total_steps=6)

    def test_accumulation_is_weighted_by_actual_supervised_tokens(self) -> None:
        weights = train_code._accumulation_weights([2, 6])
        self.assertEqual(weights, [0.25, 0.75])
        self.assertEqual((1.0 * weights[0]) + (3.0 * weights[1]), 2.5)
        self.assertEqual(train_code._accumulation_weights([2.5, 7.5]), [0.25, 0.75])
        self.assertEqual(train_code._accumulation_weights([5]), [1.0])
        for malformed in ([], [0], [True], [1, -1], [float("nan")]):
            with (
                self.subTest(malformed=malformed),
                self.assertRaises(train_code.TrainingRefused),
            ):
                train_code._accumulation_weights(malformed)

    def test_terminal_eos_weighted_loss_matches_exact_weighted_mean(self) -> None:
        import torch

        labels = torch.tensor(
            [
                [-100, 0, 2, -100],
                [-100, 1, 0, 2],
            ],
            dtype=torch.long,
        )
        logits = (torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10).requires_grad_()
        shifted_labels = labels[..., 1:]
        token_losses = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].reshape(-1, 3),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shifted_labels)
        raw_loss = token_losses[shifted_labels.ne(-100)].mean()
        outputs = SimpleNamespace(loss=raw_loss, logits=logits)

        default_loss, supervised, terminal, default_mass = train_code._terminal_eos_weighted_loss(
            torch,
            outputs,
            labels,
            eos_token_id=2,
            terminal_eos_loss_weight=1.0,
        )
        self.assertIs(default_loss, raw_loss)
        self.assertEqual((supervised, terminal, default_mass), (5, 2, 5.0))

        weighted_loss, supervised, terminal, weighted_mass = train_code._terminal_eos_weighted_loss(
            torch,
            outputs,
            labels,
            eos_token_id=2,
            terminal_eos_loss_weight=3.0,
        )
        terminal_nll = token_losses[0, 1] + token_losses[1, 2]
        expected = (token_losses.sum() + (2.0 * terminal_nll)) / 9.0
        torch.testing.assert_close(weighted_loss, expected)
        self.assertEqual((supervised, terminal, weighted_mass), (5, 2, 9.0))
        weighted_loss.backward()
        self.assertIsNotNone(logits.grad)

        changed = labels.clone()
        changed[1, -1] = 1
        with self.assertRaisesRegex(train_code.TrainingRefused, "pinned EOS"):
            train_code._terminal_eos_weighted_loss(
                torch,
                outputs,
                changed,
                eos_token_id=2,
                terminal_eos_loss_weight=3.0,
            )

    def test_nonfinite_values_refuse_and_first_exact_tie_wins(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(train_code.TrainingRefused, "not finite"),
            ):
                train_code._finite_float(value, "test value")
        self.assertTrue(train_code._strictly_better(0.8, None))
        self.assertTrue(train_code._strictly_better(0.8, 1.0))
        self.assertFalse(train_code._strictly_better(0.8, 0.8))
        self.assertFalse(train_code._strictly_better(0.9, 0.8))

    def test_current_schema_preserves_previous_receipt_identifier(self) -> None:
        self.assertEqual(train_code.LEGACY_SCHEMA, "microtensor.code.training.v1")
        self.assertEqual(train_code.PREVIOUS_SCHEMA, "microtensor.code.training.v2")
        self.assertEqual(train_code.BEST_HOLDOUT_SCHEMA, "microtensor.code.training.v3")
        self.assertEqual(train_code.SCHEMA, "microtensor.code.training.v4")

    def test_tree_identity_binds_exact_regular_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=candidate.TMPFS_MOUNT) as temporary:
            root = Path(temporary)
            (root / "weights.bin").write_bytes(b"weights")
            identity = train_code.tree_identity(root)
            self.assertEqual(identity["total_bytes"], 7)
            self.assertEqual(identity["files"][0]["path"], "weights.bin")
            self.assertEqual(
                identity["files"][0]["digest"], candidate.digest_file(root / "weights.bin")
            )


if __name__ == "__main__":
    unittest.main()
