#!/usr/bin/env python3
"""Train a separate, truthful Microtensor ``code/mt-3g`` LoRA candidate.

The trainer is deliberately offline.  Corpus preparation and base-model
download are separate reviewed steps; this process accepts only the exact
prepared corpus and exact supported base snapshot validated by
``training.code_candidate``.  All outputs must stay in ``/dev/shm``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import historical_code_candidate as historical_candidate
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import historical_code_candidate as historical_candidate  # type: ignore[no-redef]


LEGACY_SCHEMA: Final[str] = "microtensor.code.training.v1"
PREVIOUS_SCHEMA: Final[str] = "microtensor.code.training.v2"
BEST_HOLDOUT_SCHEMA: Final[str] = "microtensor.code.training.v3"
SCHEMA: Final[str] = "microtensor.code.training.v4"
HISTORICAL_SCHEMA: Final[str] = "microtensor.code.training.v5"
HOTKEY: Final[str] = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"
DEFAULT_CORPUS_PROFILE: Final[str] = "bigcodebench94"
HISTORICAL_CORPUS_PROFILE: Final[str] = historical_candidate.CORPUS_PROFILE
DEVELOPMENT_RUN_KIND: Final[str] = "development_holdout"
FINAL_ALL_PUBLIC_RUN_KIND: Final[str] = "final_all_public"
DEVELOPMENT_QUALITY_CLAIM: Final[str] = (
    "none: public code tests are withheld; holdout validation loss is a training "
    "diagnostic, not execution pass@1"
)
FINAL_ALL_PUBLIC_QUALITY_CLAIM: Final[str] = (
    "none: all 94 public examples were used for training; public code tests are withheld; "
    "no holdout or execution pass@1 was measured"
)
DEVELOPMENT_HOLDOUT_CLAIM: Final[str] = (
    "directional validation-loss evidence only; not execution pass@1"
)
FINAL_ALL_PUBLIC_HOLDOUT_CLAIM: Final[str] = (
    "all 94 public examples were training inputs; no holdout or execution pass@1 was measured"
)
BEST_HOLDOUT_SELECTION_POLICY: Final[str] = "minimum_holdout_loss_first_epoch"
FINAL_EPOCH_SELECTION_POLICY: Final[str] = "final_epoch_no_holdout"
TERMINAL_EOS_LOSS_CONTRACT: Final[str] = (
    "weighted mean causal cross entropy on completion tokens only; ordinary target "
    "tokens have weight 1 and each example's terminal EOS target has the configured weight"
)
TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPECTED_DISTRIBUTIONS: Final[dict[str, str]] = {
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.16.1",
}


class TrainingRefused(ValueError):
    """The local training contract is incomplete or has changed."""


@dataclass(frozen=True)
class Settings:
    seed: int = 92
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 2048
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    max_grad_norm: float = 1.0
    terminal_eos_loss_weight: float = 1.0


def validate_run_kind(
    *,
    final_all_public: bool,
    dataset_manifest: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    corpus_profile: str = DEFAULT_CORPUS_PROFILE,
) -> str:
    """Bind an explicit training purpose to the exact prepared split."""

    if not isinstance(final_all_public, bool):
        raise TrainingRefused("final_all_public must be boolean")
    declared: dict[str, int] = {}
    for key, rows in (("train_examples", train_rows), ("holdout_examples", holdout_rows)):
        count = dataset_manifest.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TrainingRefused(f"prepared manifest {key} must be a non-negative integer")
        if count != len(rows):
            raise TrainingRefused(f"prepared manifest {key} does not match loaded rows")
        declared[key] = count
    if corpus_profile == DEFAULT_CORPUS_PROFILE:
        expected_total = candidate.EXPECTED_COUNTS["train"]
    elif corpus_profile == HISTORICAL_CORPUS_PROFILE:
        expected_total = historical_candidate.EXPECTED_COUNTS["train"]
    else:
        raise TrainingRefused(f"unsupported training corpus profile {corpus_profile!r}")
    total = declared["train_examples"] + declared["holdout_examples"]
    if total != expected_total:
        raise TrainingRefused(f"prepared split must contain all {expected_total} public examples")
    if final_all_public:
        if declared != {"train_examples": expected_total, "holdout_examples": 0}:
            raise TrainingRefused(
                "--final-all-public requires exactly "
                f"{expected_total} training examples and zero holdout examples"
            )
        return FINAL_ALL_PUBLIC_RUN_KIND
    if declared["holdout_examples"] == 0:
        raise TrainingRefused(
            "zero-holdout training requires the explicit --final-all-public acknowledgement"
        )
    return DEVELOPMENT_RUN_KIND


def no_holdout_diagnostics(
    corpus_profile: str = DEFAULT_CORPUS_PROFILE,
) -> dict[str, Any]:
    """Return the truthful diagnostic receipt for an all-public final fit."""

    if corpus_profile == DEFAULT_CORPUS_PROFILE:
        claim = FINAL_ALL_PUBLIC_HOLDOUT_CLAIM
    elif corpus_profile == HISTORICAL_CORPUS_PROFILE:
        claim = historical_candidate.FINAL_ALL_PUBLIC_HOLDOUT_CLAIM
    else:
        raise TrainingRefused(f"unsupported training corpus profile {corpus_profile!r}")
    return {
        "status": "not_run",
        "examples": 0,
        "claim": claim,
    }


def validate_settings(settings: Settings) -> None:
    integer_bounds = {
        "seed": (settings.seed, 0, 2**31 - 1),
        "epochs": (settings.epochs, 1, 20),
        "batch_size": (settings.batch_size, 1, 32),
        "gradient_accumulation": (settings.gradient_accumulation, 1, 128),
        "max_length": (settings.max_length, 512, 4096),
        "lora_rank": (settings.lora_rank, 1, 256),
        "lora_alpha": (settings.lora_alpha, 1, 1024),
    }
    for name, (value, minimum, maximum) in integer_bounds.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TrainingRefused(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise TrainingRefused(f"{name} must be in [{minimum}, {maximum}]")
    finite_bounds = {
        "learning_rate": (settings.learning_rate, 0.0, 0.01, False),
        "weight_decay": (settings.weight_decay, 0.0, 1.0, True),
        "warmup_ratio": (settings.warmup_ratio, 0.0, 0.5, True),
        "lora_dropout": (settings.lora_dropout, 0.0, 0.5, True),
        "max_grad_norm": (settings.max_grad_norm, 0.0, 100.0, False),
        "terminal_eos_loss_weight": (
            settings.terminal_eos_loss_weight,
            1.0,
            128.0,
            True,
        ),
    }
    for name, (value, minimum, maximum, include_minimum) in finite_bounds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingRefused(f"{name} must be numeric")
        number = float(value)
        lower_ok = number >= minimum if include_minimum else number > minimum
        if not math.isfinite(number) or not lower_ok or number > maximum:
            boundary = "[" if include_minimum else "("
            raise TrainingRefused(f"{name} must be finite and in {boundary}{minimum}, {maximum}]")


def _token_ids(tokenizer: Any, text: str, field: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    raw = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded.input_ids
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TrainingRefused(f"tokenizer returned malformed {field} ids")
    ids = list(raw)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
        raise TrainingRefused(f"tokenizer returned malformed {field} ids")
    if not ids:
        raise TrainingRefused(f"tokenizer returned no {field} ids")
    return ids


def encode_record(
    row: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    *,
    target_eos_token_id: int | None = None,
    ref_pattern: Any = candidate.REF_PATTERN,
) -> dict[str, Any]:
    if frozenset(row) != candidate.PREPARED_ROW_KEYS:
        raise TrainingRefused("prepared row fields changed")
    ref = row.get("ref")
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(ref, str) or ref_pattern.fullmatch(ref) is None:
        raise TrainingRefused("prepared row has an invalid ref")
    if not isinstance(prompt, str) or not prompt:
        raise TrainingRefused(f"prepared row {ref!r} has no prompt")
    if not isinstance(completion, str) or not completion:
        raise TrainingRefused(f"prepared row {ref!r} has no completion")
    if row.get("max_output_tokens") != 1024:
        raise TrainingRefused(f"prepared row {ref!r} changed its output budget")
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise TrainingRefused("tokenizer has no valid eos_token_id")
    if target_eos_token_id is not None and eos != target_eos_token_id:
        raise TrainingRefused(
            f"tokenizer eos_token_id {eos} does not match pinned target {target_eos_token_id}"
        )
    target_eos = eos if target_eos_token_id is None else target_eos_token_id
    prompt_ids = _token_ids(tokenizer, prompt, "prompt")
    target_ids = [*_token_ids(tokenizer, completion, "completion"), target_eos]
    input_ids = [*prompt_ids, *target_ids]
    if len(input_ids) > max_length:
        raise TrainingRefused(
            f"prepared row {ref!r} needs {len(input_ids)} tokens, above max_length {max_length}"
        )
    return {
        "attention_mask": [1] * len(input_ids),
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "ref": ref,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
    }


def validate_distribution_versions(observed: Mapping[str, str]) -> dict[str, str]:
    exact: dict[str, str] = {}
    for name, wanted in EXPECTED_DISTRIBUTIONS.items():
        found = observed.get(name)
        if not isinstance(found, str) or not found:
            raise TrainingRefused(f"required distribution {name} is unavailable")
        public = found.partition("+")[0]
        if public != wanted:
            raise TrainingRefused(f"{name} version {found} does not match pinned {wanted}")
        exact[name] = found
    return exact


def installed_distribution_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name in EXPECTED_DISTRIBUTIONS:
        try:
            observed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return validate_distribution_versions(observed)


def tree_identity(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise TrainingRefused(f"output tree is not a regular directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise TrainingRefused(f"output tree contains a symlink: {path}")
        if not path.is_file():
            continue
        files.append(
            {
                "bytes": path.stat().st_size,
                "digest": candidate.digest_file(path),
                "path": path.relative_to(root).as_posix(),
            }
        )
    if not files:
        raise TrainingRefused(f"output tree is empty: {root}")
    return {
        "digest": candidate.digest_bytes(candidate.canonical_json_bytes(files)),
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_metric(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps < 1 or not 0 <= warmup_steps <= total_steps:
        raise TrainingRefused("scheduler step bounds are invalid")
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    # LambdaLR evaluates index zero before the first optimizer step. Sample the
    # cosine on (0, 1) for the remaining optimizer steps, then reach zero only
    # after the final step has already consumed a strictly positive rate.
    progress = (step - warmup_steps + 1) / max(1, total_steps - warmup_steps + 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _accumulation_weights(loss_masses: Sequence[int | float]) -> list[float]:
    """Weight microbatch mean losses as one exact weighted-target mean."""

    masses: list[float] = []
    for value in loss_masses:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingRefused("every microbatch must have numeric loss mass")
        mass = float(value)
        if not math.isfinite(mass) or mass <= 0.0:
            raise TrainingRefused("every microbatch must have finite positive loss mass")
        masses.append(mass)
    if not masses:
        raise TrainingRefused("cannot accumulate an empty microbatch group")
    total = sum(masses)
    return [value / total for value in masses]


def _terminal_eos_supervision(labels: Any, eos_token_id: int) -> tuple[int, int]:
    """Count shifted targets and verify every sequence ends in the pinned EOS target."""

    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise TrainingRefused("terminal EOS token id is invalid")
    if getattr(labels, "ndim", None) != 2 or labels.shape[0] < 1 or labels.shape[1] < 2:
        raise TrainingRefused("labels must be a non-empty rank-two causal batch")
    shifted_labels = labels[..., 1:]
    supervised = shifted_labels.ne(-100)
    per_sequence = supervised.sum(dim=-1)
    if bool(per_sequence.le(0).any().item()):
        raise TrainingRefused("every sequence must contain supervised target tokens")
    positions = labels.new_tensor(range(shifted_labels.shape[-1])).unsqueeze(0)
    positions = positions.expand_as(shifted_labels)
    terminal_positions = positions.masked_fill(~supervised, -1).max(dim=-1).values
    row_indices = labels.new_tensor(range(labels.shape[0]))
    terminal_labels = shifted_labels[row_indices, terminal_positions]
    if bool(terminal_labels.ne(eos_token_id).any().item()):
        raise TrainingRefused("every sequence must end supervision at the pinned EOS token")
    return int(supervised.sum().item()), int(labels.shape[0])


def _weighted_loss_mass(
    supervised_tokens: int,
    terminal_tokens: int,
    terminal_eos_loss_weight: float,
) -> float:
    if (
        isinstance(supervised_tokens, bool)
        or not isinstance(supervised_tokens, int)
        or supervised_tokens < 1
    ):
        raise TrainingRefused("supervised token count must be a positive integer")
    if (
        isinstance(terminal_tokens, bool)
        or not isinstance(terminal_tokens, int)
        or not 1 <= terminal_tokens <= supervised_tokens
    ):
        raise TrainingRefused("terminal token count is invalid")
    if (
        isinstance(terminal_eos_loss_weight, bool)
        or not isinstance(terminal_eos_loss_weight, (int, float))
        or not math.isfinite(float(terminal_eos_loss_weight))
        or not 1.0 <= float(terminal_eos_loss_weight) <= 128.0
    ):
        raise TrainingRefused("terminal EOS loss weight must be finite and in [1.0, 128.0]")
    return supervised_tokens + ((float(terminal_eos_loss_weight) - 1.0) * terminal_tokens)


def _terminal_eos_weighted_loss(
    torch: Any,
    outputs: Any,
    labels: Any,
    *,
    eos_token_id: int,
    terminal_eos_loss_weight: float,
) -> tuple[Any, int, int, float]:
    """Return the exact completion-token CE with an explicit terminal-EOS multiplier."""

    raw_loss = getattr(outputs, "loss", None)
    logits = getattr(outputs, "logits", None)
    if raw_loss is None or logits is None:
        raise TrainingRefused("causal model output must contain loss and logits")
    if getattr(logits, "ndim", None) != 3 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise TrainingRefused("causal model logits do not align with labels")
    supervised_tokens, terminal_tokens = _terminal_eos_supervision(labels, eos_token_id)
    loss_mass = _weighted_loss_mass(
        supervised_tokens,
        terminal_tokens,
        terminal_eos_loss_weight,
    )
    if float(terminal_eos_loss_weight) == 1.0:
        return raw_loss, supervised_tokens, terminal_tokens, loss_mass

    shifted_labels = labels[..., 1:]
    supervised = shifted_labels.ne(-100)
    positions = labels.new_tensor(range(shifted_labels.shape[-1])).unsqueeze(0)
    positions = positions.expand_as(shifted_labels)
    terminal_positions = positions.masked_fill(~supervised, -1).max(dim=-1).values
    row_indices = labels.new_tensor(range(labels.shape[0]))
    terminal_labels = shifted_labels[row_indices, terminal_positions]
    terminal_logits = logits[..., :-1, :][row_indices, terminal_positions, :]
    terminal_nll = torch.nn.functional.cross_entropy(
        terminal_logits.float(),
        terminal_labels,
        reduction="sum",
    )
    extra_weight = float(terminal_eos_loss_weight) - 1.0
    weighted_loss = ((raw_loss * supervised_tokens) + (terminal_nll * extra_weight)) / loss_mass
    return weighted_loss, supervised_tokens, terminal_tokens, loss_mass


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingRefused(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise TrainingRefused(f"{label} is not finite")
    return number


def _strictly_better(candidate_loss: float, best_loss: float | None) -> bool:
    """Keep the earliest epoch when two finite diagnostics tie exactly."""

    loss = _finite_float(candidate_loss, "holdout loss")
    return best_loss is None or loss < _finite_float(best_loss, "best holdout loss")


def _optimization_plan(
    batches_per_epoch: int,
    *,
    gradient_accumulation: int,
    epochs: int,
    warmup_ratio: float,
) -> tuple[int, int, int]:
    if batches_per_epoch < 1 or gradient_accumulation < 1 or epochs < 1:
        raise TrainingRefused("optimization plan counts must be positive")
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio <= 0.5:
        raise TrainingRefused("optimization warmup ratio is invalid")
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation)
    total_updates = updates_per_epoch * epochs
    warmup_steps = math.floor(total_updates * warmup_ratio)
    return updates_per_epoch, total_updates, warmup_steps


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset-profile",
        choices=(DEFAULT_CORPUS_PROFILE, HISTORICAL_CORPUS_PROFILE),
        default=DEFAULT_CORPUS_PROFILE,
        help="explicit prepared-corpus contract; BigCodeBench-94 remains the default",
    )
    parser.add_argument(
        "--source-corpus",
        type=Path,
        default=None,
        help="exact raw public response; required and replayed for historical8000",
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default=None,
        help="require this exact pinned base identity; otherwise auto-detect by snapshot",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--final-all-public",
        action="store_true",
        help="explicitly train on every row in the selected public corpus with no holdout",
    )
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--terminal-eos-loss-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def _load_training_runtime() -> tuple[Any, Any, Any, dict[str, str]]:
    versions = installed_distribution_versions()
    try:
        import torch
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TrainingRefused(f"pinned training runtime is incomplete: {exc}") from exc
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise TrainingRefused("the pinned CUDA training runtime is required; CPU torch is refused")
    if str(torch.version.cuda).partition(".")[0] != "13":
        raise TrainingRefused(f"CUDA {torch.version.cuda} is not the pinned CUDA 13 runtime")
    if not torch.cuda.is_bf16_supported():
        raise TrainingRefused("the selected CUDA device does not support native BF16 training")
    peft_runtime = (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    return torch, peft_runtime, (AutoModelForCausalLM, AutoTokenizer), versions


def _collator(torch: Any, pad_token_id: int) -> Any:
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise TrainingRefused("cannot collate an empty batch")
        width = max(len(item["input_ids"]) for item in items)
        input_ids = torch.full((len(items), width), pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        attention = torch.zeros((len(items), width), dtype=torch.long)
        for index, item in enumerate(items):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[index, :length] = torch.tensor(item["labels"], dtype=torch.long)
            attention[index, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}

    return collate


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # Set these before importing torch/Transformers or touching CUDA.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    settings = Settings(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_grad_norm=args.max_grad_norm,
        terminal_eos_loss_weight=args.terminal_eos_loss_weight,
    )
    try:
        validate_settings(settings)
        dataset_root = candidate.assert_tmpfs_path(args.dataset, must_exist=True)
        base_root = candidate.assert_tmpfs_path(args.base, must_exist=True)
        output_root = candidate.assert_tmpfs_path(args.out)
        if output_root.exists():
            raise TrainingRefused(f"training output already exists: {output_root}")
        if shutil.disk_usage(candidate.TMPFS_MOUNT).free < candidate.MIN_TRAINING_TMPFS_FREE_BYTES:
            raise TrainingRefused("less than the required 12 GiB is free in /dev/shm")
        if args.dataset_profile == HISTORICAL_CORPUS_PROFILE:
            if args.source_corpus is None:
                raise TrainingRefused(
                    "historical8000 requires --source-corpus for exact replay before training"
                )
            source_root = candidate.assert_tmpfs_path(args.source_corpus, must_exist=True)
            train_rows, dataset_manifest = historical_candidate.load_prepared_dataset(
                dataset_root,
                source_root,
            )
            holdout_rows = historical_candidate.load_prepared_rows(
                dataset_root / "holdout.jsonl", "holdout.jsonl"
            )
            source_corpus_identity = historical_candidate.source_corpus_identity()
            corpus_version = historical_candidate.CORPUS_VERSION
            ref_pattern = historical_candidate.REF_PATTERN
            receipt_schema = HISTORICAL_SCHEMA
            target_construction = historical_candidate.TRAINING_TARGET_CONSTRUCTION
        else:
            if args.source_corpus is not None:
                raise TrainingRefused(
                    "--source-corpus is reserved for the explicit historical8000 profile"
                )
            train_rows, dataset_manifest = candidate.load_prepared_dataset(dataset_root)
            holdout_rows = candidate._load_prepared_rows(
                dataset_root / "holdout.jsonl", "holdout.jsonl"
            )
            source_corpus_identity = None
            corpus_version = candidate.CORPUS_VERSION
            ref_pattern = candidate.REF_PATTERN
            receipt_schema = SCHEMA
            target_construction = "raw prompt -> complete importable task_func module"
        run_kind = validate_run_kind(
            final_all_public=args.final_all_public,
            dataset_manifest=dataset_manifest,
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            corpus_profile=args.dataset_profile,
        )
        base_identity = candidate.verify_base_snapshot(
            base_root,
            expected_model=args.base_model,
        )
        base_contract = candidate.contract_for_identity(base_identity)
        torch, peft_runtime, auto_classes, runtime_versions = _load_training_runtime()
        if args.dataset_profile == HISTORICAL_CORPUS_PROFILE:
            if run_kind != FINAL_ALL_PUBLIC_RUN_KIND:
                raise TrainingRefused(
                    "historical8000 training is approved only for the exact 8000/0 final split"
                )
            if base_contract.model != candidate.QWEN3_BASE_MODEL:
                raise TrainingRefused(
                    "historical8000 final training requires the pinned Qwen3-0.6B base"
                )
        LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict = (
            peft_runtime
        )
        AutoModelForCausalLM, AutoTokenizer = auto_classes
    except (candidate.CandidateError, TrainingRefused, OSError) as exc:
        raise SystemExit(f"code training refused: {exc}") from exc

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    try:
        candidate.validate_tokenizer_contract(tokenizer, base_contract)
    except candidate.CandidateError as exc:
        raise SystemExit(f"code training refused: {exc}") from exc
    tokenizer.padding_side = "right"
    encoded_train = [
        encode_record(
            row,
            tokenizer,
            settings.max_length,
            target_eos_token_id=base_contract.target_eos_token_id,
            ref_pattern=ref_pattern,
        )
        for row in train_rows
    ]
    encoded_holdout = [
        encode_record(
            row,
            tokenizer,
            settings.max_length,
            target_eos_token_id=base_contract.target_eos_token_id,
            ref_pattern=ref_pattern,
        )
        for row in holdout_rows
    ]
    token_summary = {
        "maximum_sequence_tokens": max(
            len(item["input_ids"]) for item in [*encoded_train, *encoded_holdout]
        ),
        "maximum_target_tokens": max(
            int(item["target_tokens"]) for item in [*encoded_train, *encoded_holdout]
        ),
        "train_target_tokens": sum(int(item["target_tokens"]) for item in encoded_train),
        "holdout_target_tokens": sum(int(item["target_tokens"]) for item in encoded_holdout),
    }

    output_root.mkdir(parents=True, exist_ok=False)
    metrics_path = output_root / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)
    metadata_path = output_root / "training_metadata.json"
    started_at = int(time.time())
    if args.dataset_profile == HISTORICAL_CORPUS_PROFILE:
        quality_claim = (
            historical_candidate.FINAL_ALL_PUBLIC_QUALITY_CLAIM
            if run_kind == FINAL_ALL_PUBLIC_RUN_KIND
            else historical_candidate.DEVELOPMENT_QUALITY_CLAIM
        )
    else:
        quality_claim = (
            FINAL_ALL_PUBLIC_QUALITY_CLAIM
            if run_kind == FINAL_ALL_PUBLIC_RUN_KIND
            else DEVELOPMENT_QUALITY_CLAIM
        )
    metadata: dict[str, Any] = {
        "schema": receipt_schema,
        "status": "running",
        "run_kind": run_kind,
        "hotkey": HOTKEY,
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "base_model": base_contract.model,
        "base_snapshot": base_identity,
        "corpus_version": corpus_version,
        "dataset": {
            "manifest": dataset_manifest,
            "manifest_digest": candidate.digest_file(dataset_root / "manifest.json"),
            **(
                {"source_corpus": source_corpus_identity}
                if args.dataset_profile == HISTORICAL_CORPUS_PROFILE
                else {}
            ),
        },
        "settings": asdict(settings),
        "target": {
            "construction": target_construction,
            "loss": TERMINAL_EOS_LOSS_CONTRACT,
            "chat_template": False,
            "ordinary_target_token_weight": 1.0,
            "terminal_eos_token_id": base_contract.target_eos_token_id,
            "terminal_eos_token_weight": settings.terminal_eos_loss_weight,
        },
        "token_summary": token_summary,
        "runtime": {
            "distributions": runtime_versions,
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
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
        "selection": {
            "policy": (
                FINAL_EPOCH_SELECTION_POLICY
                if run_kind == FINAL_ALL_PUBLIC_RUN_KIND
                else BEST_HOLDOUT_SELECTION_POLICY
            ),
            "metric": None if run_kind == FINAL_ALL_PUBLIC_RUN_KIND else "holdout_loss",
            "terminal_epoch": settings.epochs,
            "terminal_loss": None,
            "best_epoch": None,
            "best_loss": None,
            "exported_epoch": None,
            "exported_step": None,
        },
        "started_at_unix": started_at,
    }
    if run_kind == FINAL_ALL_PUBLIC_RUN_KIND:
        metadata["holdout_diagnostics"] = no_holdout_diagnostics(args.dataset_profile)
    _atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        if (
            candidate.verify_base_snapshot(base_root, expected_model=base_contract.model)
            != base_identity
        ):
            raise TrainingRefused("base snapshot changed while the model was loading")
        model.config.use_cache = False
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.lora_rank,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(TARGET_MODULES),
            ),
        )
        device = torch.device("cuda:0")
        model.to(device)

        collate = _collator(torch, int(tokenizer.pad_token_id))
        generator = torch.Generator().manual_seed(settings.seed)
        train_loader = torch.utils.data.DataLoader(
            encoded_train,
            batch_size=settings.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=0,
            pin_memory=True,
        )
        holdout_loader = None
        if encoded_holdout:
            holdout_loader = torch.utils.data.DataLoader(
                encoded_holdout,
                batch_size=settings.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=0,
                pin_memory=True,
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise TrainingRefused("LoRA exposed no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            fused=False,
        )
        batches_per_epoch = len(train_loader)
        updates_per_epoch, total_updates, warmup_steps = _optimization_plan(
            batches_per_epoch,
            gradient_accumulation=settings.gradient_accumulation,
            epochs=settings.epochs,
            warmup_ratio=settings.warmup_ratio,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _cosine_multiplier(
                step, warmup_steps=warmup_steps, total_steps=total_updates
            ),
        )

        def validation_loss() -> float:
            if holdout_loader is None:
                raise TrainingRefused("holdout validation was called without holdout examples")
            model.eval()
            total_weighted_nll = 0.0
            supervised_tokens = 0
            terminal_tokens = 0
            loss_mass = 0.0
            with torch.no_grad():
                for batch in holdout_loader:
                    batch = {
                        key: value.to(device, non_blocking=True) for key, value in batch.items()
                    }
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(**batch)
                        loss, batch_tokens, batch_terminal_tokens, batch_loss_mass = (
                            _terminal_eos_weighted_loss(
                                torch,
                                outputs,
                                batch["labels"],
                                eos_token_id=base_contract.target_eos_token_id,
                                terminal_eos_loss_weight=settings.terminal_eos_loss_weight,
                            )
                        )
                    loss_value = _finite_float(loss.detach(), "validation loss")
                    total_weighted_nll += loss_value * batch_loss_mass
                    supervised_tokens += batch_tokens
                    terminal_tokens += batch_terminal_tokens
                    loss_mass += batch_loss_mass
            model.train()
            expected_mass = _weighted_loss_mass(
                supervised_tokens,
                terminal_tokens,
                settings.terminal_eos_loss_weight,
            )
            if not math.isclose(loss_mass, expected_mass, rel_tol=0.0, abs_tol=1e-12):
                raise TrainingRefused("holdout weighted loss mass became inconsistent")
            return _finite_float(
                total_weighted_nll / loss_mass,
                "aggregate validation loss",
            )

        started = time.monotonic()
        update = 0
        baseline_holdout: float | None = None
        terminal_holdout: float | None = None
        best_holdout: float | None = None
        best_epoch: int | None = None
        best_adapter_state: dict[str, Any] | None = None
        if holdout_loader is not None:
            baseline_holdout = validation_loss()
            baseline_metric = {
                "step": 0,
                "epoch": 0,
                "holdout_loss": baseline_holdout,
                "holdout_perplexity": math.exp(min(20.0, baseline_holdout)),
                "quality_claim": "training diagnostic only; no public execution tests",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            _append_metric(metrics_path, baseline_metric)
            print(json.dumps(baseline_metric, sort_keys=True), flush=True)
            metadata["holdout_diagnostics"] = {"baseline_loss": baseline_holdout}
            _atomic_json(metadata_path, metadata)
        for epoch in range(1, settings.epochs + 1):
            model.train()
            batches = list(train_loader)
            for offset in range(0, len(batches), settings.gradient_accumulation):
                group = batches[offset : offset + settings.gradient_accumulation]
                supervision = [
                    _terminal_eos_supervision(
                        batch["labels"],
                        base_contract.target_eos_token_id,
                    )
                    for batch in group
                ]
                loss_masses = [
                    _weighted_loss_mass(
                        supervised_tokens,
                        terminal_tokens,
                        settings.terminal_eos_loss_weight,
                    )
                    for supervised_tokens, terminal_tokens in supervision
                ]
                loss_weights = _accumulation_weights(loss_masses)
                token_counts = [item[0] for item in supervision]
                terminal_counts = [item[1] for item in supervision]
                group_tokens = sum(token_counts)
                group_terminal_tokens = sum(terminal_counts)
                group_loss_mass = sum(loss_masses)
                optimizer.zero_grad(set_to_none=True)
                group_weighted_nll = 0.0
                for microbatch_index, (batch, loss_weight) in enumerate(
                    zip(group, loss_weights, strict=True)
                ):
                    batch = {
                        key: value.to(device, non_blocking=True) for key, value in batch.items()
                    }
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(**batch)
                        objective_loss, tokens, terminal_tokens, loss_mass = (
                            _terminal_eos_weighted_loss(
                                torch,
                                outputs,
                                batch["labels"],
                                eos_token_id=base_contract.target_eos_token_id,
                                terminal_eos_loss_weight=settings.terminal_eos_loss_weight,
                            )
                        )
                    counts_changed = (tokens, terminal_tokens) != supervision[microbatch_index]
                    mass_changed = not math.isclose(
                        loss_mass,
                        loss_masses[microbatch_index],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    if counts_changed or mass_changed:
                        raise TrainingRefused("microbatch weighted loss mass changed on device")
                    objective_loss_value = _finite_float(
                        objective_loss.detach(),
                        "training loss",
                    )
                    scaled_loss = objective_loss * loss_weight
                    scaled_loss.backward()
                    group_weighted_nll += objective_loss_value * loss_mass
                if not any(parameter.grad is not None for parameter in trainable):
                    raise TrainingRefused("training produced no gradients")
                try:
                    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        settings.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise TrainingRefused(
                        f"non-finite gradient norm at epoch {epoch}, step {update + 1}"
                    ) from exc
                gradient_norm = _finite_float(gradient_norm_tensor, "gradient norm")
                learning_rate_used = _finite_float(
                    optimizer.param_groups[0]["lr"], "optimizer learning rate"
                )
                if learning_rate_used <= 0.0:
                    raise TrainingRefused("optimizer learning rate must remain positive")
                optimizer.step()
                scheduler.step()
                update += 1
                metric = {
                    "step": update,
                    "epoch": epoch,
                    "loss": group_weighted_nll / group_loss_mass,
                    "loss_mass": group_loss_mass,
                    "supervised_tokens": group_tokens,
                    "terminal_eos_tokens": group_terminal_tokens,
                    "terminal_eos_loss_weight": settings.terminal_eos_loss_weight,
                    "microbatches": len(group),
                    "gradient_norm": gradient_norm,
                    "learning_rate": learning_rate_used,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
                _append_metric(metrics_path, metric)
                print(json.dumps(metric, sort_keys=True), flush=True)
            if holdout_loader is not None:
                heldout = validation_loss()
                terminal_holdout = heldout
                selected_as_best = _strictly_better(heldout, best_holdout)
                if selected_as_best:
                    live_adapter_state = get_peft_model_state_dict(
                        model,
                        adapter_name="default",
                        save_embedding_layers=False,
                    )
                    best_adapter_state = {
                        key: value.detach().to(device="cpu", copy=True)
                        for key, value in live_adapter_state.items()
                    }
                    if not best_adapter_state:
                        raise TrainingRefused("best checkpoint contains no adapter tensors")
                    best_holdout = heldout
                    best_epoch = epoch
                metric = {
                    "step": update,
                    "epoch": epoch,
                    "holdout_loss": heldout,
                    "holdout_perplexity": math.exp(min(20.0, heldout)),
                    "selected_as_best": selected_as_best,
                    "best_epoch": best_epoch,
                    "best_loss": best_holdout,
                    "quality_claim": "training diagnostic only; no public execution tests",
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
                _append_metric(metrics_path, metric)
                print(json.dumps(metric, sort_keys=True), flush=True)
                metadata["selection"].update(
                    {
                        "terminal_loss": terminal_holdout,
                        "best_epoch": best_epoch,
                        "best_loss": best_holdout,
                    }
                )
                _atomic_json(metadata_path, metadata)

        if update != total_updates:
            raise TrainingRefused(f"completed {update} optimizer updates, expected {total_updates}")
        if (baseline_holdout is None) != (terminal_holdout is None):
            raise TrainingRefused("holdout diagnostic state became inconsistent")
        if baseline_holdout is not None:
            if (
                terminal_holdout is None
                or best_holdout is None
                or best_epoch is None
                or best_adapter_state is None
            ):
                raise TrainingRefused("development run has no best holdout checkpoint")
            if not 1 <= best_epoch <= settings.epochs:
                raise TrainingRefused("best holdout epoch is out of range")
            set_peft_model_state_dict(
                model,
                dict(best_adapter_state),
                adapter_name="default",
                ignore_mismatched_sizes=False,
                low_cpu_mem_usage=False,
            )
            restored_adapter_state = get_peft_model_state_dict(
                model,
                adapter_name="default",
                save_embedding_layers=False,
            )
            if frozenset(restored_adapter_state) != frozenset(best_adapter_state):
                raise TrainingRefused("restored adapter checkpoint keys changed")
            for key, wanted in best_adapter_state.items():
                found = restored_adapter_state[key].detach().to(device="cpu")
                if not torch.equal(found, wanted):
                    raise TrainingRefused(f"restored adapter tensor {key!r} changed")

        if baseline_holdout is None:
            holdout_diagnostics = no_holdout_diagnostics(args.dataset_profile)
            exported_epoch = settings.epochs
            exported_step = update
        else:
            exported_epoch = best_epoch
            exported_step = best_epoch * updates_per_epoch
            holdout_diagnostics = {
                "baseline_loss": baseline_holdout,
                "terminal_loss": terminal_holdout,
                "best_loss": best_holdout,
                "loss_change": best_holdout - baseline_holdout,
                "claim": DEVELOPMENT_HOLDOUT_CLAIM,
            }

        selection_receipt = {
            "policy": (
                FINAL_EPOCH_SELECTION_POLICY
                if baseline_holdout is None
                else BEST_HOLDOUT_SELECTION_POLICY
            ),
            "metric": None if baseline_holdout is None else "holdout_loss",
            "terminal_epoch": settings.epochs,
            "terminal_loss": terminal_holdout,
            "best_epoch": best_epoch,
            "best_loss": best_holdout,
            "exported_epoch": exported_epoch,
            "exported_step": exported_step,
        }
        metadata["selection"] = selection_receipt
        selection_metric = {"event": "export_selection", **selection_receipt}
        _append_metric(metrics_path, selection_metric)
        print(json.dumps(selection_metric, sort_keys=True), flush=True)
        best_adapter_state = None

        adapter_dir = output_root / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        model.config.use_cache = True
        merged = model.merge_and_unload(safe_merge=True)
        merged_dir = output_root / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(merged_dir)
        if (
            candidate.verify_base_snapshot(base_root, expected_model=base_contract.model)
            != base_identity
        ):
            raise TrainingRefused("base snapshot changed during training")

        metadata.update(
            {
                "status": "complete",
                "finished_at_unix": int(time.time()),
                "elapsed_s": round(time.monotonic() - started, 3),
                "updates": update,
                "metrics_digest": candidate.digest_file(metrics_path),
                "holdout_diagnostics": holdout_diagnostics,
                "adapter": tree_identity(adapter_dir),
                "merged": tree_identity(merged_dir),
            }
        )
        _atomic_json(metadata_path, metadata)
        print(f"complete merged code candidate: {merged_dir}", flush=True)
        return 0
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failure_type"] = exc.__class__.__name__
        metadata["finished_at_unix"] = int(time.time())
        _atomic_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
