from __future__ import annotations

import contextlib
import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from training import code_candidate as candidate
from training import evaluate_code, train_code
from training import historical_code_candidate as historical_candidate
from training import normalized_historical_code_candidate as normalized_candidate


class _Tensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def to(self, _device: object) -> _Tensor:
        return self

    def __getitem__(self, key: tuple[int, slice]) -> _Tensor:
        row, selected = key
        if row != 0:
            raise IndexError(row)
        return _Tensor(self.values[selected])

    def tolist(self) -> list[int]:
        return list(self.values)


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.decode_call: tuple[_Tensor, dict[str, object]] | None = None

    def __call__(self, text: str, **kwargs: object) -> dict[str, _Tensor]:
        self.calls.append((text, kwargs))
        return {
            "input_ids": _Tensor([10, 11]),
            "attention_mask": _Tensor([1, 1]),
        }

    def decode(self, ids: _Tensor, **kwargs: object) -> str:
        self.decode_call = (ids, kwargs)
        return "def task_func():\n    return 1\n"


class _Model:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> _Tensor:
        self.kwargs = kwargs
        return _Tensor([10, 11, 20, 99])


class _Cuda:
    def synchronize(self) -> None:
        return None


class _Torch:
    cuda = _Cuda()

    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


TEST_BASE_CONTRACT = candidate.BaseSnapshotContract(
    model="Fixture/Raw@revision",
    files={},
    required_bytes=0,
    repository_bytes=0,
    target_eos_token_id=99,
    pad_token_id=0,
    generation_stop_token_ids=(99, 98),
    thinking_token_ids=(77, 78),
)


def _training_metadata(
    *,
    manifest: dict[str, object],
    manifest_digest: str,
    base_identity: dict[str, object],
    adapter_identity: dict[str, object],
    merged_identity: dict[str, object],
    metrics_digest: str,
    run_kind: str = train_code.DEVELOPMENT_RUN_KIND,
    schema: str = train_code.SCHEMA,
) -> dict[str, object]:
    final_run = run_kind == train_code.FINAL_ALL_PUBLIC_RUN_KIND
    historical = schema == train_code.HISTORICAL_SCHEMA
    normalized = schema == train_code.NORMALIZED_HISTORICAL_SCHEMA
    source_bound = historical or normalized
    corpus_profile = (
        train_code.NORMALIZED_HISTORICAL_CORPUS_PROFILE
        if normalized
        else train_code.HISTORICAL_CORPUS_PROFILE
        if historical
        else train_code.DEFAULT_CORPUS_PROFILE
    )
    current = schema in {
        train_code.BEST_HOLDOUT_SCHEMA,
        train_code.SCHEMA,
        train_code.HISTORICAL_SCHEMA,
        train_code.NORMALIZED_HISTORICAL_SCHEMA,
    }
    weighted = schema in {
        train_code.SCHEMA,
        train_code.HISTORICAL_SCHEMA,
        train_code.NORMALIZED_HISTORICAL_SCHEMA,
    }
    settings = train_code.Settings()
    base_contract = candidate.contract_for_identity(base_identity)
    settings_payload = asdict(settings)
    if not weighted:
        settings_payload.pop("terminal_eos_loss_weight")
    target = (
        {
            "construction": (
                normalized_candidate.TRAINING_TARGET_CONSTRUCTION
                if normalized
                else historical_candidate.TRAINING_TARGET_CONSTRUCTION
                if historical
                else "raw prompt -> complete importable task_func module"
            ),
            "loss": train_code.TERMINAL_EOS_LOSS_CONTRACT,
            "chat_template": False,
            "ordinary_target_token_weight": 1.0,
            "terminal_eos_token_id": base_contract.target_eos_token_id,
            "terminal_eos_token_weight": settings.terminal_eos_loss_weight,
        }
        if weighted
        else {
            "construction": "raw prompt -> complete importable task_func module",
            "loss": "causal cross entropy on completion tokens only",
            "chat_template": False,
        }
    )
    train_examples = int(manifest["train_examples"])
    batches_per_epoch = (train_examples + settings.batch_size - 1) // settings.batch_size
    updates_per_epoch, updates, _ = train_code._optimization_plan(
        batches_per_epoch,
        gradient_accumulation=settings.gradient_accumulation,
        epochs=settings.epochs,
        warmup_ratio=settings.warmup_ratio,
    )
    quality_claim = (
        normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM
        if normalized
        else historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM
        if historical and final_run
        else historical_candidate.DEVELOPMENT_QUALITY_CLAIM
        if historical
        else evaluate_code.FINAL_ALL_PUBLIC_TRAINING_QUALITY_CLAIM
        if final_run
        else evaluate_code.TRAINING_QUALITY_CLAIM
    )
    holdout_diagnostics = (
        train_code.no_holdout_diagnostics(corpus_profile)
        if final_run
        else (
            {
                "baseline_loss": 2.0,
                "terminal_loss": 1.7,
                "best_loss": 1.5,
                "loss_change": -0.5,
                "claim": train_code.DEVELOPMENT_HOLDOUT_CLAIM,
            }
            if current
            else {
                "baseline_loss": 2.0,
                "final_loss": 1.5,
                "loss_change": -0.5,
                "claim": train_code.DEVELOPMENT_HOLDOUT_CLAIM,
            }
        )
    )
    metadata: dict[str, object] = {
        "schema": schema,
        "status": "complete",
        "run_kind": run_kind,
        "hotkey": train_code.HOTKEY,
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "base_model": base_identity["base_model"],
        "base_snapshot": base_identity,
        "corpus_version": (
            historical_candidate.CORPUS_VERSION if source_bound else candidate.CORPUS_VERSION
        ),
        "dataset": {
            "manifest": manifest,
            "manifest_digest": manifest_digest,
            **(
                {
                    "source_corpus": (
                        normalized_candidate.source_corpus_identity()
                        if normalized
                        else historical_candidate.source_corpus_identity()
                    )
                }
                if source_bound
                else {}
            ),
        },
        "settings": settings_payload,
        "target": target,
        "token_summary": {
            "maximum_sequence_tokens": 100,
            "maximum_target_tokens": 50,
            "train_target_tokens": 1000,
            "holdout_target_tokens": 0 if final_run else 200,
        },
        "runtime": {
            "distributions": dict(train_code.EXPECTED_DISTRIBUTIONS),
            "cuda": "13.0",
            "gpu": "fixture GPU",
            "capability": [12, 0],
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "upstream_compatibility": {
            "commit": candidate.AUDITED_UNSIGNED_UPSTREAM_COMMIT,
            "mechanism_version": candidate.MECHANISM_VERSION,
            "signed_release": False,
            "activation_blocked": True,
        },
        "quality_claim": quality_claim,
        "started_at_unix": 1,
        "holdout_diagnostics": holdout_diagnostics,
        "finished_at_unix": 2,
        "elapsed_s": 1.0,
        "updates": updates,
        "metrics_digest": metrics_digest,
        "adapter": adapter_identity,
        "merged": merged_identity,
    }
    if current:
        metadata["selection"] = (
            {
                "policy": train_code.FINAL_EPOCH_SELECTION_POLICY,
                "metric": None,
                "terminal_epoch": settings.epochs,
                "terminal_loss": None,
                "best_epoch": None,
                "best_loss": None,
                "exported_epoch": settings.epochs,
                "exported_step": updates,
            }
            if final_run
            else {
                "policy": train_code.BEST_HOLDOUT_SELECTION_POLICY,
                "metric": "holdout_loss",
                "terminal_epoch": settings.epochs,
                "terminal_loss": 1.7,
                "best_epoch": 1,
                "best_loss": 1.5,
                "exported_epoch": 1,
                "exported_step": updates_per_epoch,
            }
        )
    if schema == train_code.LEGACY_SCHEMA:
        metadata.pop("run_kind")
    return metadata


class StructuralDiagnosticTests(unittest.TestCase):
    def test_mirrors_largest_fenced_block_and_never_executes_it(self) -> None:
        fence = chr(96) * 3
        reference = "def task_func():\n    return 1\n\nraise RuntimeError('must not run')\n"
        completion = (
            f"explanation\n{fence}py\nx = 1\n{fence}\n"
            f"{fence}python\n{reference}{fence}\ntrailing prose"
        )
        diagnostics = evaluate_code.structural_diagnostics(completion, reference)
        self.assertFalse(diagnostics["raw_parseable_python"])
        self.assertTrue(diagnostics["scorer_extracted_parseable_python"])
        self.assertTrue(diagnostics["scorer_extracted_top_level_task_func"])
        self.assertTrue(diagnostics["scorer_extracted_exact_reference_text"])
        self.assertTrue(diagnostics["scorer_extracted_exact_reference_ast"])
        self.assertEqual(
            diagnostics["scorer_extracted_completion"],
            reference.strip(),
        )

    def test_thinking_markup_is_retained_and_reported(self) -> None:
        completion = "<think>private draft</think>\ndef task_func():\n    return 1\n"
        diagnostics = evaluate_code.structural_diagnostics(
            completion,
            "def task_func():\n    return 1\n",
        )
        self.assertIn("<think>", diagnostics["scorer_extracted_completion"])
        self.assertTrue(diagnostics["raw_contains_thinking_markup"])
        self.assertTrue(diagnostics["scorer_extracted_contains_thinking_markup"])

    def test_unfenced_invalid_output_remains_an_honest_failure(self) -> None:
        diagnostics = evaluate_code.structural_diagnostics(
            "Here is your answer: not Python",
            "def task_func():\n    return 1\n",
        )
        self.assertTrue(diagnostics["raw_nonempty"])
        self.assertFalse(diagnostics["raw_parseable_python"])
        self.assertFalse(diagnostics["scorer_extracted_parseable_python"])
        self.assertFalse(diagnostics["scorer_extracted_top_level_task_func"])
        self.assertFalse(diagnostics["scorer_extracted_exact_reference_ast"])

    def test_zero_holdout_is_refused_as_evaluation_evidence(self) -> None:
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "no holdout"):
            evaluate_code.require_evaluation_holdout([])
        self.assertIsNone(evaluate_code.require_evaluation_holdout([{"ref": "heldout"}]))


class GenerationContractTests(unittest.TestCase):
    def test_generation_is_raw_greedy_and_decodes_only_new_tokens(self) -> None:
        tokenizer = _Tokenizer()
        model = _Model()
        with mock.patch.object(
            evaluate_code.time,
            "perf_counter_ns",
            side_effect=(1_000_000, 6_000_000),
        ):
            result = evaluate_code.generate_raw_completion(
                model=model,
                tokenizer=tokenizer,
                torch=_Torch(),
                device="cuda:0",
                prompt="raw prompt",
                max_new_tokens=1024,
                base_contract=TEST_BASE_CONTRACT,
            )
        self.assertEqual(
            tokenizer.calls,
            [("raw prompt", {"add_special_tokens": False, "return_tensors": "pt"})],
        )
        self.assertIs(model.kwargs["do_sample"], False)
        self.assertEqual(model.kwargs["num_beams"], 1)
        self.assertEqual(
            model.kwargs["repetition_penalty"],
            evaluate_code.NEUTRAL_REPETITION_PENALTY,
        )
        self.assertEqual(model.kwargs["max_new_tokens"], 1024)
        self.assertEqual(model.kwargs["eos_token_id"], [99, 98])
        self.assertEqual(tokenizer.decode_call[0].tolist(), [20, 99])  # type: ignore[index]
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertEqual(result["generated_tokens"], 2)
        self.assertTrue(result["eos_reached"])
        self.assertEqual(result["stop_token_id"], 99)
        self.assertEqual(result["latency_ms"], 5.0)


class TrainingLineageTests(unittest.TestCase):
    def test_historical_lineage_requires_and_replays_exact_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=candidate.TMPFS_MOUNT) as temporary:
            dataset_root = Path(temporary)
            (dataset_root / "manifest.json").write_text(
                json.dumps({"schema": historical_candidate.DATASET_SCHEMA}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                evaluate_code.EvaluationRefused,
                "--training-source-corpus",
            ):
                evaluate_code._prepared_training_lineage(dataset_root, None)

            wrong_source = dataset_root / "wrong-source.json"
            wrong_source.write_bytes(b"{}")
            refusal = historical_candidate.HistoricalCandidateError(
                "historical public corpus raw digest changed"
            )
            with (
                mock.patch.object(
                    historical_candidate,
                    "load_prepared_dataset",
                    side_effect=refusal,
                ) as replay,
                self.assertRaisesRegex(
                    historical_candidate.HistoricalCandidateError,
                    "raw digest changed",
                ),
            ):
                evaluate_code._prepared_training_lineage(dataset_root, wrong_source)
            replay.assert_called_once_with(dataset_root, wrong_source)

    def test_normalized_lineage_requires_and_replays_exact_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=candidate.TMPFS_MOUNT) as temporary:
            dataset_root = Path(temporary)
            (dataset_root / "manifest.json").write_text(
                json.dumps({"schema": normalized_candidate.DATASET_SCHEMA}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                evaluate_code.EvaluationRefused,
                "--training-source-corpus",
            ):
                evaluate_code._prepared_training_lineage(dataset_root, None)

            wrong_source = dataset_root / "wrong-source.json"
            wrong_source.write_bytes(b"{}")
            refusal = normalized_candidate.NormalizedHistoricalCandidateError(
                "historical public corpus raw digest changed"
            )
            with (
                mock.patch.object(
                    normalized_candidate,
                    "load_prepared_dataset",
                    side_effect=refusal,
                ) as replay,
                self.assertRaisesRegex(
                    normalized_candidate.NormalizedHistoricalCandidateError,
                    "raw digest changed",
                ),
            ):
                evaluate_code._prepared_training_lineage(dataset_root, wrong_source)
            replay.assert_called_once_with(dataset_root, wrong_source)

    def test_current_lineage_rejects_historical_source_cross_use(self) -> None:
        with tempfile.TemporaryDirectory(dir=candidate.TMPFS_MOUNT) as temporary:
            dataset_root = Path(temporary)
            (dataset_root / "manifest.json").write_text(
                json.dumps({"schema": candidate.DATASET_SCHEMA}),
                encoding="utf-8",
            )
            source = dataset_root / "source.json"
            source.write_bytes(b"{}")
            with self.assertRaisesRegex(
                evaluate_code.EvaluationRefused,
                "current94 training lineage refuses",
            ):
                evaluate_code._prepared_training_lineage(dataset_root, source)

    def test_cli_binds_separate_training_and_source_lineages(self) -> None:
        current_dataset = candidate.TMPFS_MOUNT / "current94"
        training_dataset = candidate.TMPFS_MOUNT / "historical8000"
        training_source = candidate.TMPFS_MOUNT / "historical.json"
        base = candidate.TMPFS_MOUNT / "base"
        model = candidate.TMPFS_MOUNT / "run" / "merged"
        output = candidate.TMPFS_MOUNT / "evaluation"
        args = evaluate_code._parse_args(
            [
                "--kind",
                "merged",
                "--dataset",
                str(current_dataset),
                "--training-dataset",
                str(training_dataset),
                "--training-source-corpus",
                str(training_source),
                "--base",
                str(base),
                "--model",
                str(model),
                "--out",
                str(output),
            ]
        )
        self.assertEqual(args.training_dataset, training_dataset)
        self.assertEqual(args.training_source_corpus, training_source)


class TrainingMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest: dict[str, object] = {
            "schema": candidate.DATASET_SCHEMA,
            "train_examples": 78,
            "holdout_examples": 16,
        }
        self.manifest_digest = "sha256:" + "1" * 64
        self.base_identity: dict[str, object] = {
            "base_model": candidate.RECOMMENDED_BASE_MODEL,
            "required_bytes": candidate.RECOMMENDED_BASE_REQUIRED_BYTES,
            "files": {},
        }
        self.adapter_identity: dict[str, object] = {"digest": "adapter"}
        self.merged_identity: dict[str, object] = {"digest": "merged"}
        self.metrics_digest = "sha256:" + "2" * 64
        self.metadata = _training_metadata(
            manifest=self.manifest,
            manifest_digest=self.manifest_digest,
            base_identity=self.base_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
        )

    def validate(self, payload: object) -> dict[str, object]:
        return evaluate_code.validate_training_metadata(
            payload,
            dataset_manifest=self.manifest,
            dataset_manifest_digest=self.manifest_digest,
            base_identity=self.base_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
        )

    def test_complete_exact_activation_blocked_lineage_is_accepted(self) -> None:
        self.assertEqual(self.validate(self.metadata), self.metadata)

    def test_qwen3_v4_receipt_is_accepted_but_cross_base_substitution_is_not(self) -> None:
        qwen3_identity: dict[str, object] = {
            "base_model": candidate.QWEN3_BASE_MODEL,
            "required_bytes": candidate.QWEN3_BASE_REQUIRED_BYTES,
            "files": {},
        }
        metadata = _training_metadata(
            manifest=self.manifest,
            manifest_digest=self.manifest_digest,
            base_identity=qwen3_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
        )
        self.assertEqual(
            evaluate_code.validate_training_metadata(
                metadata,
                dataset_manifest=self.manifest,
                dataset_manifest_digest=self.manifest_digest,
                base_identity=qwen3_identity,
                adapter_identity=self.adapter_identity,
                merged_identity=self.merged_identity,
                metrics_digest=self.metrics_digest,
            ),
            metadata,
        )
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "base_model"):
            self.validate(metadata)

    def test_legacy_v1_development_receipt_remains_accepted(self) -> None:
        legacy = _training_metadata(
            manifest=self.manifest,
            manifest_digest=self.manifest_digest,
            base_identity=self.base_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
            schema=train_code.LEGACY_SCHEMA,
        )
        self.assertEqual(self.validate(legacy), legacy)

    def test_previous_v2_development_receipt_remains_accepted(self) -> None:
        previous = _training_metadata(
            manifest=self.manifest,
            manifest_digest=self.manifest_digest,
            base_identity=self.base_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
            schema=train_code.PREVIOUS_SCHEMA,
        )
        self.assertEqual(self.validate(previous), previous)

    def test_best_holdout_v3_development_receipt_remains_accepted(self) -> None:
        previous = _training_metadata(
            manifest=self.manifest,
            manifest_digest=self.manifest_digest,
            base_identity=self.base_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
            schema=train_code.BEST_HOLDOUT_SCHEMA,
        )
        self.assertEqual(self.validate(previous), previous)

    def test_final_v4_receipt_is_truthful_and_mode_bound(self) -> None:
        final_manifest: dict[str, object] = {
            "schema": candidate.DATASET_SCHEMA,
            "train_examples": 94,
            "holdout_examples": 0,
        }
        metadata = _training_metadata(
            manifest=final_manifest,
            manifest_digest=self.manifest_digest,
            base_identity=self.base_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
            run_kind=train_code.FINAL_ALL_PUBLIC_RUN_KIND,
        )

        def validate_final(payload: object) -> dict[str, object]:
            return evaluate_code.validate_training_metadata(
                payload,
                dataset_manifest=final_manifest,
                dataset_manifest_digest=self.manifest_digest,
                base_identity=self.base_identity,
                adapter_identity=self.adapter_identity,
                merged_identity=self.merged_identity,
                metrics_digest=self.metrics_digest,
            )

        self.assertEqual(validate_final(metadata), metadata)
        wrong_kind = copy.deepcopy(metadata)
        wrong_kind["run_kind"] = train_code.DEVELOPMENT_RUN_KIND
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "quality claim"):
            validate_final(wrong_kind)

    def test_historical_v5_receipt_binds_source_and_rejects_cross_swap(self) -> None:
        historical_manifest: dict[str, object] = {
            "schema": historical_candidate.DATASET_SCHEMA,
            "train_examples": 8_000,
            "holdout_examples": 0,
        }
        qwen3_identity: dict[str, object] = {
            "base_model": candidate.QWEN3_BASE_MODEL,
            "required_bytes": candidate.QWEN3_BASE_REQUIRED_BYTES,
            "files": {},
        }
        metadata = _training_metadata(
            manifest=historical_manifest,
            manifest_digest=self.manifest_digest,
            base_identity=qwen3_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
            run_kind=train_code.FINAL_ALL_PUBLIC_RUN_KIND,
            schema=train_code.HISTORICAL_SCHEMA,
        )

        def validate_historical(payload: object) -> dict[str, object]:
            return evaluate_code.validate_training_metadata(
                payload,
                dataset_manifest=historical_manifest,
                dataset_manifest_digest=self.manifest_digest,
                base_identity=qwen3_identity,
                adapter_identity=self.adapter_identity,
                merged_identity=self.merged_identity,
                metrics_digest=self.metrics_digest,
            )

        self.assertEqual(validate_historical(metadata), metadata)
        self.assertEqual(
            metadata["quality_claim"],
            historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        )
        wrong_source = copy.deepcopy(metadata)
        wrong_source["dataset"]["source_corpus"]["raw_digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(
            evaluate_code.EvaluationRefused,
            "source-corpus identity",
        ):
            validate_historical(wrong_source)

        with self.assertRaisesRegex(
            evaluate_code.EvaluationRefused,
            "cross-swapped",
        ):
            evaluate_code.validate_training_metadata(
                metadata,
                dataset_manifest=self.manifest,
                dataset_manifest_digest=self.manifest_digest,
                base_identity=qwen3_identity,
                adapter_identity=self.adapter_identity,
                merged_identity=self.merged_identity,
                metrics_digest=self.metrics_digest,
            )

    def test_normalized_v6_receipt_binds_projection_and_rejects_cross_swap(self) -> None:
        normalized_manifest: dict[str, object] = {
            "schema": normalized_candidate.DATASET_SCHEMA,
            "train_examples": normalized_candidate.EXPECTED_TRAIN_EXAMPLES,
            "holdout_examples": 0,
        }
        qwen3_identity: dict[str, object] = {
            "base_model": candidate.QWEN3_BASE_MODEL,
            "required_bytes": candidate.QWEN3_BASE_REQUIRED_BYTES,
            "files": {},
        }
        metadata = _training_metadata(
            manifest=normalized_manifest,
            manifest_digest=self.manifest_digest,
            base_identity=qwen3_identity,
            adapter_identity=self.adapter_identity,
            merged_identity=self.merged_identity,
            metrics_digest=self.metrics_digest,
            run_kind=train_code.FINAL_ALL_PUBLIC_RUN_KIND,
            schema=train_code.NORMALIZED_HISTORICAL_SCHEMA,
        )

        def validate_normalized(payload: object) -> dict[str, object]:
            return evaluate_code.validate_training_metadata(
                payload,
                dataset_manifest=normalized_manifest,
                dataset_manifest_digest=self.manifest_digest,
                base_identity=qwen3_identity,
                adapter_identity=self.adapter_identity,
                merged_identity=self.merged_identity,
                metrics_digest=self.metrics_digest,
            )

        self.assertEqual(validate_normalized(metadata), metadata)
        self.assertEqual(
            metadata["quality_claim"],
            normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        )
        wrong_source = copy.deepcopy(metadata)
        wrong_source["dataset"]["source_corpus"]["excluded_refs_digest"] = (  # type: ignore[index]
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(
            evaluate_code.EvaluationRefused,
            "source-corpus identity",
        ):
            validate_normalized(wrong_source)

        historical_manifest: dict[str, object] = {
            "schema": historical_candidate.DATASET_SCHEMA,
            "train_examples": historical_candidate.EXPECTED_COUNTS["train"],
            "holdout_examples": 0,
        }
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "cross-swapped"):
            evaluate_code.validate_training_metadata(
                metadata,
                dataset_manifest=historical_manifest,
                dataset_manifest_digest=self.manifest_digest,
                base_identity=qwen3_identity,
                adapter_identity=self.adapter_identity,
                merged_identity=self.merged_identity,
                metrics_digest=self.metrics_digest,
            )

    def test_v4_selection_and_update_tampering_is_refused(self) -> None:
        wrong_export = copy.deepcopy(self.metadata)
        wrong_export["selection"]["exported_epoch"] = 2  # type: ignore[index]
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "not the best epoch"):
            self.validate(wrong_export)

        wrong_loss = copy.deepcopy(self.metadata)
        wrong_loss["selection"]["terminal_loss"] = 1.8  # type: ignore[index]
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "terminal loss changed"):
            self.validate(wrong_loss)

        wrong_updates = copy.deepcopy(self.metadata)
        wrong_updates["updates"] = 9
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "update count"):
            self.validate(wrong_updates)

    def test_metadata_drift_or_activation_enablement_is_refused(self) -> None:
        enabled = copy.deepcopy(self.metadata)
        enabled["upstream_compatibility"]["activation_blocked"] = False  # type: ignore[index]
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "activation-blocked"):
            self.validate(enabled)

        extra = copy.deepcopy(self.metadata)
        extra["unreviewed"] = True
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "fields changed"):
            self.validate(extra)

        changed_model = copy.deepcopy(self.metadata)
        changed_model["merged"] = {"digest": "different"}
        with self.assertRaisesRegex(evaluate_code.EvaluationRefused, "merged tree"):
            self.validate(changed_model)


if __name__ == "__main__":
    unittest.main()
