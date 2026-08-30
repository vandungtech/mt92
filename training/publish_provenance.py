#!/usr/bin/env python3
"""Validate and publish one complete local training lineage to W&B."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENTITY = "microtensor"
PROJECT = "training-runs"
HOTKEY = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"
TRACK = "extract"
HARDWARE_CLASS = "mt-3g"
BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PINNED_BASE_MODEL = f"{BASE_MODEL}@{BASE_REVISION}"
CORPUS_VERSION = "sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8"
BASE_WEIGHTS_DIGEST = "sha256:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
BASE_TOKENIZER_DIGEST = (
    "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)

MAX_METADATA_BYTES = 1024 * 1024
MAX_METRICS_BYTES = 64 * 1024 * 1024
MAX_METRIC_RECORDS = 1_000_000
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CORE_METRIC_FIELDS = frozenset({"step", "epoch", "elapsed_s"})
_TRAIN_METRIC_FIELDS = frozenset({"loss", "learning_rate"})
_VALIDATION_METRIC_FIELDS = frozenset({"validation_loss", "validation_perplexity"})


class ProvenanceValidationError(ValueError):
    """Raised before any publication when a local provenance trail is invalid."""


@dataclass(frozen=True)
class TrainingStage:
    """A validated training stage and its exact metadata-file digest."""

    number: int
    directory: Path
    metadata: dict[str, Any]
    metadata_digest: str
    metrics: tuple[dict[str, Any], ...]
    epochs: int
    updates: int


@dataclass(frozen=True)
class Publication:
    """Everything needed to publish, after local validation has completed."""

    stages: tuple[TrainingStage, ...]
    artifact_digest: str
    finished_block: int

    @property
    def record_count(self) -> int:
        return sum(len(stage.metrics) for stage in self.stages)


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ProvenanceValidationError(f"{label} must be sha256: followed by 64 lowercase hex")
    return value


def _require_int(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProvenanceValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProvenanceValidationError(f"{label} must be a finite number >= {minimum}")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ProvenanceValidationError(
            f"{label} must be a finite number >= {minimum}"
        ) from exc
    if not math.isfinite(numeric) or numeric < minimum:
        raise ProvenanceValidationError(f"{label} must be a finite number >= {minimum}")
    return numeric


def _read_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    """Read a stable, non-symlink regular file without exceeding the byte limit."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise ProvenanceValidationError(f"{label} is not a readable regular file: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ProvenanceValidationError(f"{label} is not a regular file: {path}")
    if before.st_size > maximum:
        raise ProvenanceValidationError(f"{label} exceeds the {maximum}-byte limit: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise ProvenanceValidationError(f"{label} changed before it could be read: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ProvenanceValidationError(f"{label} is not a readable regular file: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(payload) > maximum:
        raise ProvenanceValidationError(f"{label} exceeds the {maximum}-byte limit: {path}")
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or len(payload) != after.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ProvenanceValidationError(f"{label} changed while it was being read: {path}")
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        parsed = json.loads(
            decoded,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except ValueError as exc:
        raise ProvenanceValidationError(f"{label} is not valid strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProvenanceValidationError(f"{label} must contain a JSON object")
    return parsed


def _parse_metrics(payload: bytes, path: Path) -> tuple[dict[str, Any], ...]:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceValidationError(f"metrics file is not UTF-8: {path}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        if not line.strip():
            raise ProvenanceValidationError(f"metrics line {line_number} is blank: {path}")
        record = _parse_json_object(
            line.encode("utf-8"),
            f"metrics line {line_number} in {path}",
        )
        records.append(record)
        if len(records) > MAX_METRIC_RECORDS:
            raise ProvenanceValidationError(
                f"metrics file exceeds the {MAX_METRIC_RECORDS}-record limit: {path}"
            )
    if not records:
        raise ProvenanceValidationError(f"metrics file contains no records: {path}")
    return tuple(records)


def _validate_identity(metadata: Mapping[str, Any], stage_number: int) -> str:
    expected = {
        "hotkey": HOTKEY,
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "base_model": PINNED_BASE_MODEL,
        "corpus_version": CORPUS_VERSION,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ProvenanceValidationError(
                f"stage {stage_number} {field} must be exactly {value!r}"
            )
    return _require_digest(
        metadata.get("corpus_file_digest"),
        f"stage {stage_number} corpus_file_digest",
    )


def _validate_training_input(
    metadata: Mapping[str, Any],
    *,
    stage_number: int,
    parent_metadata_digest: str | None,
) -> None:
    training_input = metadata.get("training_input")
    if not isinstance(training_input, dict):
        raise ProvenanceValidationError(f"stage {stage_number} training_input must be an object")

    weights_digest = _require_digest(
        training_input.get("weights_digest"),
        f"stage {stage_number} training_input.weights_digest",
    )
    tokenizer_digest = _require_digest(
        training_input.get("tokenizer_digest"),
        f"stage {stage_number} training_input.tokenizer_digest",
    )

    if parent_metadata_digest is None:
        if training_input.get("kind") != "huggingface_snapshot":
            raise ProvenanceValidationError(
                "stage 1 training_input.kind must be exactly 'huggingface_snapshot'"
            )
        if training_input.get("revision") != BASE_REVISION:
            raise ProvenanceValidationError(
                f"stage 1 training_input.revision must be exactly {BASE_REVISION!r}"
            )
        if weights_digest != BASE_WEIGHTS_DIGEST:
            raise ProvenanceValidationError("stage 1 base weights digest is not allowlisted")
        if tokenizer_digest != BASE_TOKENIZER_DIGEST:
            raise ProvenanceValidationError("stage 1 base tokenizer digest is not allowlisted")
        return

    if training_input.get("kind") != "derived_model":
        raise ProvenanceValidationError(
            f"stage {stage_number} training_input.kind must be exactly 'derived_model'"
        )
    claimed_parent = _require_digest(
        training_input.get("parent_metadata_digest"),
        f"stage {stage_number} training_input.parent_metadata_digest",
    )
    if claimed_parent != parent_metadata_digest:
        raise ProvenanceValidationError(
            f"stage {stage_number} parent metadata digest does not match stage {stage_number - 1}"
        )


def _validate_metadata_times_and_counts(
    metadata: Mapping[str, Any], stage_number: int
) -> tuple[int, int, int, int, float]:
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ProvenanceValidationError(f"stage {stage_number} settings must be an object")
    epochs = _require_int(settings.get("epochs"), f"stage {stage_number} settings.epochs")
    updates = _require_int(metadata.get("updates"), f"stage {stage_number} updates")
    if epochs > updates:
        raise ProvenanceValidationError(
            f"stage {stage_number} cannot have more epochs than optimizer updates"
        )

    started = _require_int(
        metadata.get("started_at_unix"), f"stage {stage_number} started_at_unix"
    )
    finished = _require_int(
        metadata.get("finished_at_unix"), f"stage {stage_number} finished_at_unix"
    )
    if finished < started:
        raise ProvenanceValidationError(
            f"stage {stage_number} finished_at_unix precedes started_at_unix"
        )
    elapsed = _require_number(metadata.get("elapsed_s"), f"stage {stage_number} elapsed_s")
    if elapsed > finished - started + 2.0:
        raise ProvenanceValidationError(
            f"stage {stage_number} elapsed_s exceeds its wall-clock interval"
        )
    return epochs, updates, started, finished, elapsed


def _metric_kind(record: Mapping[str, Any], label: str) -> str:
    fields = frozenset(record)
    training_fields = _CORE_METRIC_FIELDS | _TRAIN_METRIC_FIELDS
    validation_fields = _CORE_METRIC_FIELDS | _VALIDATION_METRIC_FIELDS
    if fields == training_fields:
        return "training"
    if fields == validation_fields:
        return "validation"
    raise ProvenanceValidationError(
        f"{label} must have exactly one complete training or validation metric schema"
    )


def _validate_metrics(
    records: tuple[dict[str, Any], ...],
    *,
    stage_number: int,
    epochs: int,
    updates: int,
    metadata_elapsed: float,
) -> None:
    previous_step = 0
    previous_epoch = 1
    previous_elapsed = 0.0
    training_steps: list[int] = []
    records_by_epoch: dict[int, list[tuple[str, dict[str, Any]]]] = {}

    for record_number, record in enumerate(records, start=1):
        label = f"stage {stage_number} metric {record_number}"
        kind = _metric_kind(record, label)
        step = _require_int(record.get("step"), f"{label} step")
        epoch = _require_int(record.get("epoch"), f"{label} epoch")
        elapsed = _require_number(record.get("elapsed_s"), f"{label} elapsed_s")
        if step < previous_step:
            raise ProvenanceValidationError(f"{label} step regresses")
        if epoch < previous_epoch:
            raise ProvenanceValidationError(f"{label} epoch regresses")
        if epoch > epochs:
            raise ProvenanceValidationError(f"{label} epoch exceeds configured epochs")
        if elapsed < previous_elapsed:
            raise ProvenanceValidationError(f"{label} elapsed_s regresses")
        if elapsed > metadata_elapsed + 0.001:
            raise ProvenanceValidationError(f"{label} elapsed_s exceeds metadata elapsed_s")

        if kind == "training":
            _require_number(record.get("loss"), f"{label} loss")
            _require_number(record.get("learning_rate"), f"{label} learning_rate")
            training_steps.append(step)
        else:
            validation_loss = _require_number(
                record.get("validation_loss"), f"{label} validation_loss"
            )
            perplexity = _require_number(
                record.get("validation_perplexity"),
                f"{label} validation_perplexity",
                minimum=1.0,
            )
            expected_perplexity = math.exp(min(20.0, validation_loss))
            if not math.isclose(perplexity, expected_perplexity, rel_tol=1e-9, abs_tol=1e-12):
                raise ProvenanceValidationError(
                    f"{label} validation_perplexity does not match validation_loss"
                )

        records_by_epoch.setdefault(epoch, []).append((kind, record))
        previous_step = step
        previous_epoch = epoch
        previous_elapsed = elapsed

    expected_steps = list(range(1, updates + 1))
    if training_steps != expected_steps:
        raise ProvenanceValidationError(
            f"stage {stage_number} training metric steps must be exactly 1..{updates}"
        )
    if sorted(records_by_epoch) != list(range(1, epochs + 1)):
        raise ProvenanceValidationError(
            f"stage {stage_number} metrics must cover every configured epoch"
        )

    for epoch in range(1, epochs + 1):
        epoch_records = records_by_epoch[epoch]
        validation_positions = [
            index for index, (kind, _record) in enumerate(epoch_records) if kind == "validation"
        ]
        if validation_positions != [len(epoch_records) - 1]:
            raise ProvenanceValidationError(
                f"stage {stage_number} epoch {epoch} must end with exactly one validation metric"
            )
        training_records = [record for kind, record in epoch_records if kind == "training"]
        if not training_records:
            raise ProvenanceValidationError(
                f"stage {stage_number} epoch {epoch} contains no training metrics"
            )
        validation_record = epoch_records[-1][1]
        if validation_record["step"] != training_records[-1]["step"]:
            raise ProvenanceValidationError(
                f"stage {stage_number} epoch {epoch} validation step must repeat its final update"
            )


def validate_publication(
    training_dirs: Sequence[Path],
    artifact_digest: str,
    finished_block: int,
) -> Publication:
    """Load and validate a complete oldest-to-newest local training lineage."""

    _require_digest(artifact_digest, "artifact digest")
    _require_int(finished_block, "finished block")
    if not training_dirs:
        raise ProvenanceValidationError("at least one training directory is required")

    stages: list[TrainingStage] = []
    previous_digest: str | None = None
    previous_finished: int | None = None
    corpus_file_digest: str | None = None
    for stage_number, raw_directory in enumerate(training_dirs, start=1):
        directory = Path(raw_directory)
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "metrics.jsonl"
        metadata_bytes = _read_regular_file(
            metadata_path,
            maximum=MAX_METADATA_BYTES,
            label=f"stage {stage_number} metadata file",
        )
        metrics_bytes = _read_regular_file(
            metrics_path,
            maximum=MAX_METRICS_BYTES,
            label=f"stage {stage_number} metrics file",
        )
        metadata = _parse_json_object(
            metadata_bytes, f"stage {stage_number} metadata file {metadata_path}"
        )
        metrics = _parse_metrics(metrics_bytes, metrics_path)
        metadata_digest = _digest_bytes(metadata_bytes)

        current_corpus_file_digest = _validate_identity(metadata, stage_number)
        if corpus_file_digest is not None and current_corpus_file_digest != corpus_file_digest:
            raise ProvenanceValidationError(
                f"stage {stage_number} corpus_file_digest differs from stage 1"
            )
        corpus_file_digest = current_corpus_file_digest
        _validate_training_input(
            metadata,
            stage_number=stage_number,
            parent_metadata_digest=previous_digest,
        )
        epochs, updates, started, finished, elapsed = _validate_metadata_times_and_counts(
            metadata, stage_number
        )
        if previous_finished is not None and started < previous_finished:
            raise ProvenanceValidationError(
                f"stage {stage_number} starts before stage {stage_number - 1} finished"
            )
        _validate_metrics(
            metrics,
            stage_number=stage_number,
            epochs=epochs,
            updates=updates,
            metadata_elapsed=elapsed,
        )
        stages.append(
            TrainingStage(
                number=stage_number,
                directory=directory,
                metadata=metadata,
                metadata_digest=metadata_digest,
                metrics=metrics,
                epochs=epochs,
                updates=updates,
            )
        )
        previous_digest = metadata_digest
        previous_finished = finished

    declared_stage_count = stages[-1].metadata.get("pipeline_stages")
    if declared_stage_count is not None:
        declared_stage_count = _require_int(
            declared_stage_count, "final stage pipeline_stages"
        )
        if declared_stage_count != len(stages):
            raise ProvenanceValidationError(
                "final stage pipeline_stages does not match the supplied training directories"
            )

    return Publication(
        stages=tuple(stages),
        artifact_digest=artifact_digest,
        finished_block=finished_block,
    )


def publish(publication: Publication, wandb_client: Any) -> None:
    """Publish an already-validated lineage through an injected W&B client."""

    final_metadata = publication.stages[-1].metadata
    stage_metadata = {
        f"stage_{stage.number}": {
            "metadata_digest": stage.metadata_digest,
            "metadata": stage.metadata,
        }
        for stage in publication.stages
    }
    run = wandb_client.init(
        entity=ENTITY,
        project=PROJECT,
        name=HOTKEY,
        config={
            **final_metadata,
            "mt_hotkey": HOTKEY,
            "mt_track": TRACK,
            "mt_class": HARDWARE_CLASS,
            "mt_base_model": PINNED_BASE_MODEL,
            "mt_corpus_version": CORPUS_VERSION,
            "mt_training_stages": len(publication.stages),
            "mt_stage_metadata": stage_metadata,
        },
    )
    global_step = 0
    try:
        for stage in publication.stages:
            for metric in stage.metrics:
                global_step += 1
                payload = {
                    **metric,
                    "mt_stage": stage.number,
                    "mt_stage_epoch": metric["epoch"],
                    "mt_stage_step": metric["step"],
                }
                wandb_client.log(payload, step=global_step)
        run.summary["mt_artifact_digest"] = publication.artifact_digest
        run.summary["mt_finished_at"] = publication.finished_block
        run.summary["mt_training_records"] = publication.record_count
        run.summary["mt_training_stages"] = len(publication.stages)
    finally:
        wandb_client.finish()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir",
        action="append",
        dest="training_dirs",
        type=Path,
        required=True,
        metavar="PATH",
        help="training directory, repeat oldest to newest for a multi-stage run",
    )
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--finished-block", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, wandb_client: Any | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        publication = validate_publication(
            args.training_dirs,
            args.artifact_digest,
            args.finished_block,
        )
    except ProvenanceValidationError as exc:
        raise SystemExit(f"provenance validation failed: {exc}") from exc

    if wandb_client is None:
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise SystemExit("WANDB_API_KEY is required; no anonymous/fabricated fallback is used")
        try:
            wandb_client = importlib.import_module("wandb")
        except ImportError as exc:
            raise SystemExit("the wandb package is required to publish provenance") from exc

    publish(publication, wandb_client)
    print(
        f"published {publication.record_count} records across "
        f"{len(publication.stages)} stage(s) for {HOTKEY}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
