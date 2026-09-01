#!/usr/bin/env python3
"""Offline structural evaluation for a Microtensor code candidate.

This evaluator never imports or executes generated or reference code. It
validates the exact prepared corpus and model lineage, performs raw greedy
completion without a chat template, and records syntax/structure diagnostics.
Those diagnostics are not execution pass@1.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import historical_code_candidate as historical_candidate
    from training import normalized_historical_code_candidate as normalized_candidate
    from training import train_code
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import historical_code_candidate as historical_candidate  # type: ignore[no-redef]
    import normalized_historical_code_candidate as normalized_candidate  # type: ignore[no-redef]
    import train_code  # type: ignore[no-redef]


LEGACY_SCHEMA: Final[str] = "microtensor.code.structural-evaluation.v1"
PREVIOUS_SCHEMA: Final[str] = "microtensor.code.structural-evaluation.v3"
SCHEMA: Final[str] = "microtensor.code.structural-evaluation.v4"
QUALITY_CLAIM: Final[str] = (
    "none: generated and reference code were never executed; structural diagnostics "
    "are not execution pass@1"
)
HF_RUNTIME_CLAIM: Final[str] = (
    "non-authoritative approximation: Transformers bfloat16 CUDA generation is a local "
    "diagnostic; validator GGUF/llama.cpp generation is authoritative"
)
SEPARATE_LINEAGE_CLAIM: Final[str] = (
    "the evaluation dataset is a structural diagnostic lineage separate from model training; "
    "no execution pass@1 is claimed"
)
NEUTRAL_REPETITION_PENALTY: Final[float] = 1.0
TRAINING_QUALITY_CLAIM: Final[str] = train_code.DEVELOPMENT_QUALITY_CLAIM
FINAL_ALL_PUBLIC_TRAINING_QUALITY_CLAIM: Final[str] = train_code.FINAL_ALL_PUBLIC_QUALITY_CLAIM
LEGACY_TRAINING_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "status",
        "hotkey",
        "track",
        "hardware_class",
        "base_model",
        "base_snapshot",
        "corpus_version",
        "dataset",
        "settings",
        "target",
        "token_summary",
        "runtime",
        "upstream_compatibility",
        "quality_claim",
        "started_at_unix",
        "holdout_diagnostics",
        "finished_at_unix",
        "elapsed_s",
        "updates",
        "metrics_digest",
        "adapter",
        "merged",
    }
)
V2_TRAINING_METADATA_KEYS: Final[frozenset[str]] = LEGACY_TRAINING_METADATA_KEYS | {"run_kind"}
TRAINING_METADATA_KEYS: Final[frozenset[str]] = V2_TRAINING_METADATA_KEYS | {"selection"}
TRAINING_RUNTIME_KEYS: Final[frozenset[str]] = frozenset(
    {"distributions", "cuda", "gpu", "capability", "deterministic_algorithms", "tf32"}
)
TOKEN_SUMMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "maximum_sequence_tokens",
        "maximum_target_tokens",
        "train_target_tokens",
        "holdout_target_tokens",
    }
)
LEGACY_DEVELOPMENT_HOLDOUT_DIAGNOSTIC_KEYS: Final[frozenset[str]] = frozenset(
    {"baseline_loss", "final_loss", "loss_change", "claim"}
)
DEVELOPMENT_HOLDOUT_DIAGNOSTIC_KEYS: Final[frozenset[str]] = frozenset(
    {"baseline_loss", "terminal_loss", "best_loss", "loss_change", "claim"}
)
FINAL_ALL_PUBLIC_HOLDOUT_DIAGNOSTIC_KEYS: Final[frozenset[str]] = frozenset(
    {"status", "examples", "claim"}
)
SELECTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "policy",
        "metric",
        "terminal_epoch",
        "terminal_loss",
        "best_epoch",
        "best_loss",
        "exported_epoch",
        "exported_step",
    }
)
CURRENT_TRAINING_SETTING_KEYS: Final[frozenset[str]] = frozenset(asdict(train_code.Settings()))
PRE_TERMINAL_EOS_SETTING_KEYS: Final[frozenset[str]] = CURRENT_TRAINING_SETTING_KEYS - {
    "terminal_eos_loss_weight"
}
RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ref",
        "prompt_digest",
        "reference_digest",
        "completion",
        "completion_digest",
        "completion_utf8_bytes",
        "scorer_extracted_completion",
        "scorer_extracted_digest",
        "scorer_extracted_utf8_bytes",
        "scorer_extraction_changed",
        "prompt_tokens",
        "generated_tokens",
        "max_new_tokens",
        "eos_reached",
        "stop_token_id",
        "latency_ms",
        "raw_contains_thinking_markup",
        "raw_nonempty",
        "raw_parseable_python",
        "raw_top_level_task_func",
        "scorer_extracted_contains_thinking_markup",
        "scorer_extracted_nonempty",
        "scorer_extracted_parseable_python",
        "scorer_extracted_top_level_task_func",
        "scorer_extracted_exact_reference_text",
        "scorer_extracted_exact_reference_ast",
        "scorer_extracted_reference_text_similarity",
    }
)
SCORER_FENCE: Final[re.Pattern[str]] = re.compile(
    r"\x60\x60\x60(?:python|py)?\s*\n(.*?)\x60\x60\x60",
    re.DOTALL,
)


class EvaluationRefused(ValueError):
    """The local structural-evaluation contract is incomplete or changed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationRefused(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise EvaluationRefused(f"{label} contains a non-string key")
    found = frozenset(value)
    if found != expected:
        raise EvaluationRefused(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(found)}"
        )


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationRefused(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise EvaluationRefused(f"{label} must be finite and at least {minimum}")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationRefused(f"{label} must be a non-negative integer")
    return value


def extract_scorer_code(text: str) -> str:
    """Mirror the audited scorer's fenced-code extraction without executing code."""

    if not isinstance(text, str):
        raise EvaluationRefused("completion must be a string")
    blocks = SCORER_FENCE.findall(text)
    return max(blocks, key=len).strip() if blocks else text.strip()


def _inspect_source(source: str) -> tuple[ast.Module | None, bool]:
    try:
        tree = ast.parse(source, filename="<generated>", mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return None, False
    has_entry_point = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == candidate.EXPECTED_ENTRY_POINT
        for node in tree.body
    )
    return tree, has_entry_point


def structural_diagnostics(completion: str, reference: str) -> dict[str, Any]:
    """Inspect raw and scorer-extracted text with ast.parse; never execute it."""

    if not isinstance(completion, str) or not isinstance(reference, str):
        raise EvaluationRefused("completion and reference must be strings")
    normalized_reference = reference.strip()
    reference_tree = ast.parse(normalized_reference, filename="<reference>", mode="exec")
    raw_tree, raw_entry_point = _inspect_source(completion)
    extracted = extract_scorer_code(completion)
    extracted_tree, extracted_entry_point = _inspect_source(extracted)
    extracted_bytes = extracted.encode("utf-8")
    return {
        "scorer_extracted_completion": extracted,
        "scorer_extracted_digest": candidate.digest_bytes(extracted_bytes),
        "scorer_extracted_utf8_bytes": len(extracted_bytes),
        "scorer_extraction_changed": extracted != completion,
        "raw_contains_thinking_markup": any(
            marker in completion for marker in ("<think>", "</think>")
        ),
        "raw_nonempty": bool(completion.strip()),
        "raw_parseable_python": raw_tree is not None,
        "raw_top_level_task_func": raw_entry_point,
        "scorer_extracted_contains_thinking_markup": any(
            marker in extracted for marker in ("<think>", "</think>")
        ),
        "scorer_extracted_nonempty": bool(extracted),
        "scorer_extracted_parseable_python": extracted_tree is not None,
        "scorer_extracted_top_level_task_func": extracted_entry_point,
        "scorer_extracted_exact_reference_text": extracted == normalized_reference,
        "scorer_extracted_exact_reference_ast": bool(
            extracted_tree is not None
            and ast.dump(extracted_tree, include_attributes=False)
            == ast.dump(reference_tree, include_attributes=False)
        ),
        "scorer_extracted_reference_text_similarity": difflib.SequenceMatcher(
            None,
            extracted,
            normalized_reference,
            autojunk=False,
        ).ratio(),
    }


def validate_training_metadata(
    payload: Any,
    *,
    dataset_manifest: Mapping[str, Any],
    dataset_manifest_digest: str,
    base_identity: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
    merged_identity: Mapping[str, Any],
    metrics_digest: str,
) -> dict[str, Any]:
    """Validate the complete trainer receipt and every locally checkable binding."""

    metadata = _mapping(payload, "training metadata")
    schema = metadata.get("schema")
    if schema == train_code.LEGACY_SCHEMA:
        _exact_keys(metadata, LEGACY_TRAINING_METADATA_KEYS, "training metadata")
        run_kind = train_code.DEVELOPMENT_RUN_KIND
    elif schema == train_code.PREVIOUS_SCHEMA:
        _exact_keys(metadata, V2_TRAINING_METADATA_KEYS, "training metadata")
        run_kind = metadata.get("run_kind")
    elif schema in {
        train_code.BEST_HOLDOUT_SCHEMA,
        train_code.SCHEMA,
        train_code.HISTORICAL_SCHEMA,
        train_code.NORMALIZED_HISTORICAL_SCHEMA,
    }:
        _exact_keys(metadata, TRAINING_METADATA_KEYS, "training metadata")
        run_kind = metadata.get("run_kind")
    else:
        raise EvaluationRefused("training metadata schema is unsupported")
    if run_kind not in {
        train_code.DEVELOPMENT_RUN_KIND,
        train_code.FINAL_ALL_PUBLIC_RUN_KIND,
    }:
        raise EvaluationRefused("training metadata has an invalid run_kind")

    historical_profile = schema == train_code.HISTORICAL_SCHEMA
    normalized_profile = schema == train_code.NORMALIZED_HISTORICAL_SCHEMA
    source_bound_profile = historical_profile or normalized_profile
    expected_dataset_schema = (
        normalized_candidate.DATASET_SCHEMA
        if normalized_profile
        else historical_candidate.DATASET_SCHEMA
        if historical_profile
        else candidate.DATASET_SCHEMA
    )
    if dataset_manifest.get("schema") != expected_dataset_schema:
        raise EvaluationRefused("training receipt and prepared-corpus schemas were cross-swapped")
    if normalized_profile:
        expected_quality_claim = normalized_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM
    elif historical_profile:
        expected_quality_claim = (
            historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM
            if run_kind == train_code.FINAL_ALL_PUBLIC_RUN_KIND
            else historical_candidate.DEVELOPMENT_QUALITY_CLAIM
        )
    else:
        expected_quality_claim = (
            FINAL_ALL_PUBLIC_TRAINING_QUALITY_CLAIM
            if run_kind == train_code.FINAL_ALL_PUBLIC_RUN_KIND
            else TRAINING_QUALITY_CLAIM
        )
    required = {
        "status": "complete",
        "hotkey": train_code.HOTKEY,
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "base_model": base_identity.get("base_model"),
        "corpus_version": (
            historical_candidate.CORPUS_VERSION
            if source_bound_profile
            else candidate.CORPUS_VERSION
        ),
    }
    try:
        base_contract = candidate.contract_for_identity(base_identity)
    except candidate.CandidateError as exc:
        raise EvaluationRefused(f"base identity is unsupported: {exc}") from exc
    if historical_profile:
        if run_kind != train_code.FINAL_ALL_PUBLIC_RUN_KIND:
            raise EvaluationRefused("historical v5 receipt must describe final_all_public training")
        if base_contract.model != candidate.QWEN3_BASE_MODEL:
            raise EvaluationRefused("historical v5 receipt must bind the pinned Qwen3-0.6B base")
    if normalized_profile:
        if run_kind != train_code.FINAL_ALL_PUBLIC_RUN_KIND:
            raise EvaluationRefused("normalized v6 receipt must describe final_all_public training")
        if base_contract.model != candidate.QWEN3_BASE_MODEL:
            raise EvaluationRefused("normalized v6 receipt must bind the pinned Qwen3-0.6B base")
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise EvaluationRefused(f"training metadata field {key!r} changed")
    if metadata.get("quality_claim") != expected_quality_claim:
        raise EvaluationRefused("training metadata quality claim does not match its run_kind")
    if metadata.get("base_snapshot") != base_identity:
        raise EvaluationRefused("training metadata does not bind the exact base snapshot")

    dataset = _mapping(metadata.get("dataset"), "training metadata dataset")
    expected_dataset_keys = (
        frozenset({"manifest", "manifest_digest", "source_corpus"})
        if source_bound_profile
        else frozenset({"manifest", "manifest_digest"})
    )
    _exact_keys(dataset, expected_dataset_keys, "training metadata dataset")
    if dataset.get("manifest") != dataset_manifest:
        raise EvaluationRefused("training metadata does not bind the prepared manifest")
    if dataset.get("manifest_digest") != dataset_manifest_digest:
        raise EvaluationRefused("training metadata prepared-manifest digest changed")
    expected_source_identity = (
        normalized_candidate.source_corpus_identity()
        if normalized_profile
        else historical_candidate.source_corpus_identity()
        if historical_profile
        else None
    )
    if source_bound_profile and dataset.get("source_corpus") != expected_source_identity:
        raise EvaluationRefused("source-bound training metadata source-corpus identity changed")

    train_examples = _nonnegative_integer(
        dataset_manifest.get("train_examples"), "prepared train examples"
    )
    holdout_examples = _nonnegative_integer(
        dataset_manifest.get("holdout_examples"), "prepared holdout examples"
    )
    expected_examples = (
        normalized_candidate.EXPECTED_TRAIN_EXAMPLES
        if normalized_profile
        else historical_candidate.EXPECTED_COUNTS["train"]
        if historical_profile
        else candidate.EXPECTED_COUNTS["train"]
    )
    if train_examples + holdout_examples != expected_examples:
        raise EvaluationRefused(
            f"prepared split does not contain all {expected_examples} public examples"
        )
    if run_kind == train_code.FINAL_ALL_PUBLIC_RUN_KIND:
        if (train_examples, holdout_examples) != (expected_examples, 0):
            raise EvaluationRefused(
                f"final_all_public metadata does not bind a {expected_examples}/0 split"
            )
    elif holdout_examples == 0:
        raise EvaluationRefused("development metadata must bind a non-empty deterministic holdout")

    target = _mapping(metadata.get("target"), "training metadata target")
    if schema in {
        train_code.SCHEMA,
        train_code.HISTORICAL_SCHEMA,
        train_code.NORMALIZED_HISTORICAL_SCHEMA,
    }:
        raw_settings = _mapping(metadata.get("settings"), "training settings")
        expected_target = {
            "construction": (
                normalized_candidate.TRAINING_TARGET_CONSTRUCTION
                if normalized_profile
                else historical_candidate.TRAINING_TARGET_CONSTRUCTION
                if historical_profile
                else "raw prompt -> complete importable task_func module"
            ),
            "loss": train_code.TERMINAL_EOS_LOSS_CONTRACT,
            "chat_template": False,
            "ordinary_target_token_weight": 1.0,
            "terminal_eos_token_id": base_contract.target_eos_token_id,
            "terminal_eos_token_weight": raw_settings.get("terminal_eos_loss_weight"),
        }
    else:
        expected_target = {
            "construction": "raw prompt -> complete importable task_func module",
            "loss": "causal cross entropy on completion tokens only",
            "chat_template": False,
        }
    if target != expected_target:
        raise EvaluationRefused("training target contract changed")

    upstream = _mapping(metadata.get("upstream_compatibility"), "upstream compatibility")
    if upstream != {
        "commit": candidate.AUDITED_UNSIGNED_UPSTREAM_COMMIT,
        "mechanism_version": candidate.MECHANISM_VERSION,
        "signed_release": False,
        "activation_blocked": True,
    }:
        raise EvaluationRefused("training metadata is not activation-blocked at the audit bound")

    settings = _mapping(metadata.get("settings"), "training settings")
    expected_setting_keys = (
        CURRENT_TRAINING_SETTING_KEYS
        if schema
        in {
            train_code.SCHEMA,
            train_code.HISTORICAL_SCHEMA,
            train_code.NORMALIZED_HISTORICAL_SCHEMA,
        }
        else PRE_TERMINAL_EOS_SETTING_KEYS
    )
    _exact_keys(settings, expected_setting_keys, "training settings")
    try:
        parsed_settings = train_code.Settings(**settings)
        train_code.validate_settings(parsed_settings)
    except (TypeError, train_code.TrainingRefused) as exc:
        raise EvaluationRefused(f"training settings are invalid: {exc}") from exc

    token_summary = _mapping(metadata.get("token_summary"), "training token summary")
    _exact_keys(token_summary, TOKEN_SUMMARY_KEYS, "training token summary")
    for key, value in token_summary.items():
        _nonnegative_integer(value, f"training token summary {key}")

    runtime = _mapping(metadata.get("runtime"), "training runtime")
    _exact_keys(runtime, TRAINING_RUNTIME_KEYS, "training runtime")
    distributions = _mapping(runtime.get("distributions"), "training distributions")
    try:
        train_code.validate_distribution_versions(distributions)
    except train_code.TrainingRefused as exc:
        raise EvaluationRefused(f"training runtime changed: {exc}") from exc
    cuda = runtime.get("cuda")
    gpu = runtime.get("gpu")
    capability = runtime.get("capability")
    if not isinstance(cuda, str) or cuda.partition(".")[0] != "13":
        raise EvaluationRefused("training metadata does not record CUDA 13")
    if not isinstance(gpu, str) or not gpu:
        raise EvaluationRefused("training metadata has no GPU identity")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
    ):
        raise EvaluationRefused("training metadata has an invalid CUDA capability")
    if runtime.get("deterministic_algorithms") is not True or runtime.get("tf32") is not False:
        raise EvaluationRefused("training runtime determinism contract changed")

    diagnostics = _mapping(metadata.get("holdout_diagnostics"), "holdout diagnostics")
    if run_kind == train_code.FINAL_ALL_PUBLIC_RUN_KIND:
        _exact_keys(
            diagnostics,
            FINAL_ALL_PUBLIC_HOLDOUT_DIAGNOSTIC_KEYS,
            "holdout diagnostics",
        )
        diagnostic_profile = (
            train_code.NORMALIZED_HISTORICAL_CORPUS_PROFILE
            if normalized_profile
            else train_code.HISTORICAL_CORPUS_PROFILE
            if historical_profile
            else train_code.DEFAULT_CORPUS_PROFILE
        )
        if dict(diagnostics) != train_code.no_holdout_diagnostics(diagnostic_profile):
            raise EvaluationRefused("final_all_public holdout diagnostics changed")
        if token_summary.get("holdout_target_tokens") != 0:
            raise EvaluationRefused("final_all_public metadata reports holdout target tokens")
    else:
        if schema in {
            train_code.BEST_HOLDOUT_SCHEMA,
            train_code.SCHEMA,
            train_code.HISTORICAL_SCHEMA,
            train_code.NORMALIZED_HISTORICAL_SCHEMA,
        }:
            _exact_keys(
                diagnostics,
                DEVELOPMENT_HOLDOUT_DIAGNOSTIC_KEYS,
                "holdout diagnostics",
            )
            baseline = _finite_number(diagnostics.get("baseline_loss"), "baseline holdout loss")
            terminal = _finite_number(diagnostics.get("terminal_loss"), "terminal holdout loss")
            best = _finite_number(diagnostics.get("best_loss"), "best holdout loss")
        else:
            _exact_keys(
                diagnostics,
                LEGACY_DEVELOPMENT_HOLDOUT_DIAGNOSTIC_KEYS,
                "holdout diagnostics",
            )
            baseline = _finite_number(diagnostics.get("baseline_loss"), "baseline holdout loss")
            terminal = _finite_number(diagnostics.get("final_loss"), "final holdout loss")
            best = terminal
        expected_loss_change = best - baseline
        change = diagnostics.get("loss_change")
        if isinstance(change, bool) or not isinstance(change, (int, float)):
            raise EvaluationRefused("holdout loss change must be numeric")
        if not math.isfinite(float(change)) or not math.isclose(
            float(change),
            expected_loss_change,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvaluationRefused("holdout loss change is inconsistent")
        if diagnostics.get("claim") != train_code.DEVELOPMENT_HOLDOUT_CLAIM:
            raise EvaluationRefused("holdout diagnostic claim changed")

    for key in ("started_at_unix", "finished_at_unix"):
        _nonnegative_integer(metadata.get(key), f"training metadata {key}")
    updates = _nonnegative_integer(metadata.get("updates"), "training metadata updates")
    if schema in {
        train_code.BEST_HOLDOUT_SCHEMA,
        train_code.SCHEMA,
        train_code.HISTORICAL_SCHEMA,
        train_code.NORMALIZED_HISTORICAL_SCHEMA,
    }:
        selection = _mapping(metadata.get("selection"), "training selection")
        _exact_keys(selection, SELECTION_KEYS, "training selection")
        batches_per_epoch = math.ceil(train_examples / parsed_settings.batch_size)
        updates_per_epoch = math.ceil(batches_per_epoch / parsed_settings.gradient_accumulation)
        expected_updates = updates_per_epoch * parsed_settings.epochs
        if updates != expected_updates:
            raise EvaluationRefused("training update count does not match settings")
        terminal_epoch = _nonnegative_integer(
            selection.get("terminal_epoch"), "selection terminal epoch"
        )
        exported_epoch = _nonnegative_integer(
            selection.get("exported_epoch"), "selection exported epoch"
        )
        exported_step = _nonnegative_integer(
            selection.get("exported_step"), "selection exported step"
        )
        if terminal_epoch != parsed_settings.epochs:
            raise EvaluationRefused("selection terminal epoch does not match settings")
        if run_kind == train_code.FINAL_ALL_PUBLIC_RUN_KIND:
            if selection.get("policy") != train_code.FINAL_EPOCH_SELECTION_POLICY:
                raise EvaluationRefused("final training selection policy changed")
            if selection.get("metric") is not None:
                raise EvaluationRefused("final all-public selection cannot name a metric")
            if any(
                selection.get(key) is not None
                for key in ("terminal_loss", "best_epoch", "best_loss")
            ):
                raise EvaluationRefused("final all-public selection claims holdout evidence")
            if exported_epoch != parsed_settings.epochs or exported_step != updates:
                raise EvaluationRefused("final all-public export is not the terminal checkpoint")
        else:
            if selection.get("policy") != train_code.BEST_HOLDOUT_SELECTION_POLICY:
                raise EvaluationRefused("development training selection policy changed")
            if selection.get("metric") != "holdout_loss":
                raise EvaluationRefused("development selection metric changed")
            selected_terminal = _finite_number(
                selection.get("terminal_loss"), "selection terminal loss"
            )
            selected_best = _finite_number(selection.get("best_loss"), "selection best loss")
            selected_best_epoch = _nonnegative_integer(
                selection.get("best_epoch"), "selection best epoch"
            )
            if not 1 <= selected_best_epoch <= parsed_settings.epochs:
                raise EvaluationRefused("selection best epoch is out of range")
            if exported_epoch != selected_best_epoch:
                raise EvaluationRefused("development export is not the best epoch")
            if exported_step != selected_best_epoch * updates_per_epoch:
                raise EvaluationRefused("development exported step is inconsistent")
            if not math.isclose(selected_terminal, terminal, rel_tol=1e-12, abs_tol=1e-12):
                raise EvaluationRefused("selection terminal loss changed")
            if not math.isclose(selected_best, best, rel_tol=1e-12, abs_tol=1e-12):
                raise EvaluationRefused("selection best loss changed")
            if selected_best > selected_terminal:
                raise EvaluationRefused("selection best loss exceeds terminal loss")

    _finite_number(metadata.get("elapsed_s"), "training elapsed seconds")
    if metadata.get("metrics_digest") != metrics_digest:
        raise EvaluationRefused("training metrics digest changed")
    if metadata.get("adapter") != adapter_identity:
        raise EvaluationRefused("training metadata does not bind the adapter tree")
    if metadata.get("merged") != merged_identity:
        raise EvaluationRefused("training metadata does not bind the merged tree")
    return dict(metadata)


def require_evaluation_holdout(holdout: Sequence[Mapping[str, Any]]) -> None:
    """Refuse to misrepresent an all-public training set as holdout evidence."""

    if not holdout:
        raise EvaluationRefused(
            "prepared dataset has no holdout; final all-public training cannot be evaluated "
            "as holdout evidence"
        )


def _prepared_holdout(
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _, manifest = candidate.load_prepared_dataset(dataset_root)
    holdout = candidate._load_prepared_rows(dataset_root / "holdout.jsonl", "holdout.jsonl")
    identity = {
        "manifest": {
            "bytes": (dataset_root / "manifest.json").stat().st_size,
            "digest": candidate.digest_file(dataset_root / "manifest.json"),
        },
        "train": {
            "bytes": (dataset_root / "train.jsonl").stat().st_size,
            "digest": candidate.digest_file(dataset_root / "train.jsonl"),
        },
        "holdout": {
            "bytes": (dataset_root / "holdout.jsonl").stat().st_size,
            "digest": candidate.digest_file(dataset_root / "holdout.jsonl"),
        },
        "holdout_examples": len(holdout),
        "holdout_refs_digest": manifest["holdout_refs_digest"],
        "corpus_version": candidate.CORPUS_VERSION,
    }
    return holdout, manifest, identity


def _prepared_training_lineage(
    dataset_root: Path,
    source_corpus_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvaluationRefused("training prepared manifest must be a regular non-symlink file")
    payload = candidate._strict_json(
        manifest_path.read_bytes(),
        str(manifest_path),
    )
    manifest_probe = _mapping(payload, "training prepared manifest")
    schema = manifest_probe.get("schema")
    normalized_profile = schema == normalized_candidate.DATASET_SCHEMA
    if normalized_profile:
        if source_corpus_root is None:
            raise EvaluationRefused(
                "normalized historical training lineage requires --training-source-corpus"
            )
        _rows, manifest = normalized_candidate.load_prepared_dataset(
            dataset_root,
            source_corpus_root,
        )
        holdout = normalized_candidate.load_prepared_rows(
            dataset_root / "holdout.jsonl", "holdout.jsonl"
        )
    elif schema == historical_candidate.DATASET_SCHEMA:
        if source_corpus_root is None:
            raise EvaluationRefused("historical training lineage requires --training-source-corpus")
        _rows, manifest = historical_candidate.load_prepared_dataset(
            dataset_root,
            source_corpus_root,
        )
        holdout = historical_candidate.load_prepared_rows(
            dataset_root / "holdout.jsonl", "holdout.jsonl"
        )
    elif schema == candidate.DATASET_SCHEMA:
        if source_corpus_root is not None:
            raise EvaluationRefused("current94 training lineage refuses a historical source corpus")
        _rows, manifest = candidate.load_prepared_dataset(dataset_root)
        holdout = candidate._load_prepared_rows(dataset_root / "holdout.jsonl", "holdout.jsonl")
    else:
        raise EvaluationRefused("training prepared dataset schema is unsupported")
    identity = {
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "digest": candidate.digest_file(manifest_path),
        },
        "train": {
            "bytes": (dataset_root / "train.jsonl").stat().st_size,
            "digest": candidate.digest_file(dataset_root / "train.jsonl"),
        },
        "holdout": {
            "bytes": (dataset_root / "holdout.jsonl").stat().st_size,
            "digest": candidate.digest_file(dataset_root / "holdout.jsonl"),
        },
        "holdout_examples": len(holdout),
        "holdout_refs_digest": manifest["holdout_refs_digest"],
        "corpus_version": manifest["corpus_version"],
        **(
            {
                "excluded_refs": {
                    "bytes": (dataset_root / normalized_candidate.EXCLUDED_REFS_FILE)
                    .stat()
                    .st_size,
                    "digest": candidate.digest_file(
                        dataset_root / normalized_candidate.EXCLUDED_REFS_FILE
                    ),
                }
            }
            if normalized_profile
            else {}
        ),
    }
    return manifest, identity


def _load_training_run(
    model_root: Path,
    *,
    dataset_manifest: Mapping[str, Any],
    dataset_manifest_digest: str,
    base_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if model_root.name != "merged":
        raise EvaluationRefused("a merged model must be the trainer run's merged directory")
    run_root = model_root.parent
    metadata_path = run_root / "training_metadata.json"
    metrics_path = run_root / "metrics.jsonl"
    adapter_root = run_root / "adapter"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise EvaluationRefused("training metadata must be a regular non-symlink file")
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise EvaluationRefused("training metrics must be a regular non-symlink file")
    raw = metadata_path.read_bytes()
    payload = candidate._strict_json(raw, str(metadata_path))
    adapter_identity = train_code.tree_identity(adapter_root)
    merged_identity = train_code.tree_identity(model_root)
    metrics_digest = candidate.digest_file(metrics_path)
    validate_training_metadata(
        payload,
        dataset_manifest=dataset_manifest,
        dataset_manifest_digest=dataset_manifest_digest,
        base_identity=base_identity,
        adapter_identity=adapter_identity,
        merged_identity=merged_identity,
        metrics_digest=metrics_digest,
    )
    return {
        "kind": "merged",
        "training_metadata": {
            "bytes": len(raw),
            "digest": candidate.digest_bytes(raw),
        },
        "metrics": {
            "bytes": metrics_path.stat().st_size,
            "digest": metrics_digest,
        },
        "adapter": adapter_identity,
        "merged": merged_identity,
    }


def _load_runtime() -> tuple[Any, Any, Any, dict[str, str]]:
    versions = train_code.installed_distribution_versions()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise EvaluationRefused(f"pinned evaluation runtime is incomplete: {exc}") from exc
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise EvaluationRefused("CUDA is required for representative generation diagnostics")
    if str(torch.version.cuda).partition(".")[0] != "13":
        raise EvaluationRefused(f"CUDA {torch.version.cuda} is not the pinned CUDA 13 runtime")
    return torch, AutoModelForCausalLM, AutoTokenizer, versions


def generate_raw_completion(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    prompt: str,
    max_new_tokens: int,
    base_contract: candidate.BaseSnapshotContract | None = None,
) -> dict[str, Any]:
    """Generate one non-authoritative HF diagnostic without a chat template."""

    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise EvaluationRefused("tokenizer returned malformed raw prompt tensors")
    inputs: dict[str, Any] = {}
    for key, value in encoded.items():
        if not isinstance(key, str) or not hasattr(value, "to"):
            raise EvaluationRefused("tokenizer returned malformed raw prompt tensors")
        inputs[key] = value.to(device)
    input_ids = inputs["input_ids"]
    try:
        prompt_tokens = int(input_ids.shape[-1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise EvaluationRefused("tokenizer returned malformed input_ids shape") from exc
    if prompt_tokens < 1:
        raise EvaluationRefused("tokenizer returned an empty raw prompt")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise EvaluationRefused("tokenizer has no valid eos_token_id")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        pad_token_id = eos_token_id
    if base_contract is None:
        stop_token_ids = (eos_token_id,)
    else:
        try:
            candidate.validate_tokenizer_contract(tokenizer, base_contract)
        except candidate.CandidateError as exc:
            raise EvaluationRefused(f"model tokenizer contract changed: {exc}") from exc
        pad_token_id = base_contract.pad_token_id
        stop_token_ids = base_contract.generation_stop_token_ids

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            # Transformers inherits active logit processors from the model's
            # generation_config. Both pinned Qwen2.5 snapshots declare a
            # non-neutral repetition penalty, while the validator's signed
            # GGUF sampler fixes repeat_penalty=1.0. Override that inherited
            # setting; sampling-only knobs are deliberately omitted because
            # do_sample=False ignores them.
            repetition_penalty=NEUTRAL_REPETITION_PENALTY,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=list(stop_token_ids),
            use_cache=True,
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    try:
        generated_ids = sequences[0, prompt_tokens:]
        token_ids = generated_ids.tolist()
    except (AttributeError, IndexError, TypeError) as exc:
        raise EvaluationRefused("model returned malformed generated token ids") from exc
    if not isinstance(token_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in token_ids
    ):
        raise EvaluationRefused("model returned malformed generated token ids")
    completion = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(completion, str):
        raise EvaluationRefused("tokenizer returned a non-string completion")
    stop_token_id = next((token for token in token_ids if token in stop_token_ids), None)
    return {
        "completion": completion,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(token_ids),
        "eos_reached": stop_token_id is not None,
        "stop_token_id": stop_token_id,
        "latency_ms": latency_ms,
    }


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise EvaluationRefused("cannot summarize zero evaluation rows")
    for row in rows:
        _exact_keys(row, RESULT_KEYS, "evaluation result")
    count = len(rows)
    diagnostics = {}
    for key in (
        "raw_contains_thinking_markup",
        "raw_nonempty",
        "raw_parseable_python",
        "raw_top_level_task_func",
        "scorer_extracted_contains_thinking_markup",
        "scorer_extracted_nonempty",
        "scorer_extracted_parseable_python",
        "scorer_extracted_top_level_task_func",
        "scorer_extracted_exact_reference_text",
        "scorer_extracted_exact_reference_ast",
    ):
        found = sum(value.get(key) is True for value in rows)
        diagnostics[key] = {"count": found, "fraction": found / count}
    similarities = [float(row["scorer_extracted_reference_text_similarity"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "examples": count,
        "structural_diagnostics": diagnostics,
        "scorer_extracted_reference_text_similarity": {
            "mean": statistics.fmean(similarities),
            "median": statistics.median(similarities),
            "minimum": min(similarities),
            "maximum": max(similarities),
        },
        "latency_ms": {
            "total": sum(latencies),
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
        },
        "tokens": {
            "prompt": sum(int(row["prompt_tokens"]) for row in rows),
            "generated": sum(int(row["generated_tokens"]) for row in rows),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("base", "merged"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=None,
        help="separate prepared lineage used to train a merged model",
    )
    parser.add_argument(
        "--training-source-corpus",
        type=Path,
        default=None,
        help="exact raw source replay required for historical training lineage",
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default=None,
        help="require this exact pinned base identity; otherwise auto-detect by snapshot",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        dataset_root = candidate.assert_tmpfs_path(args.dataset, must_exist=True)
        base_root = candidate.assert_tmpfs_path(args.base, must_exist=True)
        model_root = candidate.assert_tmpfs_path(args.model, must_exist=True)
        output_root = candidate.assert_tmpfs_path(args.out)
        if output_root.exists():
            raise EvaluationRefused(f"evaluation output already exists: {output_root}")
        holdout, dataset_manifest, dataset_identity = _prepared_holdout(dataset_root)
        require_evaluation_holdout(holdout)
        base_identity = candidate.verify_base_snapshot(
            base_root,
            expected_model=args.base_model,
        )
        base_contract = candidate.contract_for_identity(base_identity)
        if (
            dataset_manifest.get("train_examples"),
            dataset_manifest.get("holdout_examples"),
            dataset_manifest.get("seed"),
        ) != (78, 16, 92):
            raise EvaluationRefused("evaluation requires the exact current94 seed-92 78/16 split")
        training_manifest = dataset_manifest
        training_dataset_identity: dict[str, Any] = dataset_identity
        training_root = dataset_root
        training_source_root: Path | None = None
        if args.training_source_corpus is not None:
            if args.training_dataset is None:
                raise EvaluationRefused(
                    "--training-source-corpus requires an explicit --training-dataset"
                )
            training_source_root = candidate.assert_tmpfs_path(
                args.training_source_corpus,
                must_exist=True,
            )
        if args.training_dataset is not None:
            if args.kind == "base":
                raise EvaluationRefused("base evaluation has no training-dataset lineage")
            training_root = candidate.assert_tmpfs_path(
                args.training_dataset,
                must_exist=True,
            )
            training_manifest, training_dataset_identity = _prepared_training_lineage(
                training_root,
                training_source_root,
            )
        training_manifest_digest = training_dataset_identity["manifest"]["digest"]
        if args.kind == "base":
            if model_root != base_root:
                raise EvaluationRefused("base evaluation requires --model and --base to match")
            model_identity = {"kind": "base", "base_snapshot": base_identity}
        else:
            model_identity = _load_training_run(
                model_root,
                dataset_manifest=training_manifest,
                dataset_manifest_digest=training_manifest_digest,
                base_identity=base_identity,
            )
        torch, AutoModelForCausalLM, AutoTokenizer, versions = _load_runtime()
    except (
        candidate.CandidateError,
        train_code.TrainingRefused,
        EvaluationRefused,
        OSError,
    ) as exc:
        raise SystemExit(f"code evaluation refused: {exc}") from exc

    torch.manual_seed(int(dataset_manifest["seed"]))
    torch.cuda.manual_seed_all(int(dataset_manifest["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    try:
        candidate.validate_tokenizer_contract(tokenizer, base_contract)
    except candidate.CandidateError as exc:
        raise SystemExit(f"code evaluation refused: {exc}") from exc
    model = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        result_path = staging / "results.jsonl"
        rows: list[dict[str, Any]] = []
        with result_path.open("xb") as handle:
            for source in holdout:
                generated = generate_raw_completion(
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    device=device,
                    prompt=str(source["prompt"]),
                    max_new_tokens=int(source["max_output_tokens"]),
                    base_contract=base_contract,
                )
                completion = str(generated["completion"])
                row = {
                    "ref": source["ref"],
                    "prompt_digest": candidate.digest_bytes(str(source["prompt"]).encode("utf-8")),
                    "reference_digest": candidate.digest_bytes(
                        str(source["completion"]).encode("utf-8")
                    ),
                    "completion": completion,
                    "completion_digest": candidate.digest_bytes(completion.encode("utf-8")),
                    "completion_utf8_bytes": len(completion.encode("utf-8")),
                    "prompt_tokens": generated["prompt_tokens"],
                    "generated_tokens": generated["generated_tokens"],
                    "max_new_tokens": source["max_output_tokens"],
                    "eos_reached": generated["eos_reached"],
                    "stop_token_id": generated["stop_token_id"],
                    "latency_ms": generated["latency_ms"],
                    **structural_diagnostics(completion, str(source["completion"])),
                }
                _exact_keys(row, RESULT_KEYS, "evaluation result")
                rows.append(row)
                handle.write(candidate.canonical_json_bytes(row) + b"\n")
                handle.flush()
            os.fsync(handle.fileno())

        summary = {
            "schema": SCHEMA,
            "status": "complete",
            "track": candidate.TRACK,
            "hardware_class": candidate.HARDWARE_CLASS,
            "generation_contract": {
                "prompt": "raw",
                "chat_template": False,
                "decoding": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": NEUTRAL_REPETITION_PENALTY,
                "batch_size": 1,
                "target_eos_token_id": base_contract.target_eos_token_id,
                "accepted_stop_token_ids": list(base_contract.generation_stop_token_ids),
                "pad_token_id": base_contract.pad_token_id,
                "thinking_markup": "retained and reported; never stripped",
            },
            "quality_claim": QUALITY_CLAIM,
            "runtime_claim": HF_RUNTIME_CLAIM,
            "upstream_compatibility": {
                "commit": candidate.AUDITED_UNSIGNED_UPSTREAM_COMMIT,
                "mechanism_version": candidate.MECHANISM_VERSION,
                "signed_release": False,
                "activation_blocked": True,
            },
            "evaluation_dataset": dataset_identity,
            "training_dataset": training_dataset_identity if args.kind == "merged" else None,
            "lineage_claim": SEPARATE_LINEAGE_CLAIM,
            "dataset": dataset_identity,
            "model": model_identity,
            "runtime": {
                "distributions": versions,
                "cuda": str(torch.version.cuda),
                "gpu": torch.cuda.get_device_name(0),
                "deterministic_algorithms": True,
                "tf32": False,
            },
            "results": {
                "file": "results.jsonl",
                "bytes": result_path.stat().st_size,
                "digest": candidate.digest_file(result_path),
                **summarize_results(rows),
            },
        }

        after_holdout, after_manifest, after_dataset_identity = _prepared_holdout(dataset_root)
        if after_manifest != dataset_manifest or after_dataset_identity != dataset_identity:
            raise EvaluationRefused("prepared dataset changed during evaluation")
        if [row["ref"] for row in after_holdout] != [row["ref"] for row in holdout]:
            raise EvaluationRefused("prepared holdout changed during evaluation")
        if (
            candidate.verify_base_snapshot(base_root, expected_model=base_contract.model)
            != base_identity
        ):
            raise EvaluationRefused("base snapshot changed during evaluation")
        if args.kind == "merged":
            after_training_manifest, after_training_identity = _prepared_training_lineage(
                training_root,
                training_source_root,
            )
            if (
                after_training_manifest != training_manifest
                or after_training_identity != training_dataset_identity
            ):
                raise EvaluationRefused("prepared training dataset changed during evaluation")
            after_model_identity = _load_training_run(
                model_root,
                dataset_manifest=training_manifest,
                dataset_manifest_digest=training_manifest_digest,
                base_identity=base_identity,
            )
            if after_model_identity != model_identity:
                raise EvaluationRefused("merged training run changed during evaluation")

        summary_path = staging / "summary.json"
        _write_json(summary_path, summary)
        os.replace(staging, output_root)
        print(
            json.dumps(
                {
                    "output": str(output_root),
                    "results_digest": summary["results"]["digest"],
                    "summary_digest": candidate.digest_file(output_root / "summary.json"),
                    "quality_claim": QUALITY_CLAIM,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
