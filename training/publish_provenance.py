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
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from training import boundary_contrastive as boundary
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import boundary_contrastive as boundary


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
MAX_AUGMENTATION_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_BOUNDARY_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_CALIBRATION_METADATA_BYTES = 16 * 1024 * 1024
MAX_CALIBRATION_CORPUS_BYTES = 64 * 1024 * 1024
MAX_LINEAGE_ASSET_BYTES = 32 * 1024 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CORE_METRIC_FIELDS = frozenset({"step", "epoch", "elapsed_s"})
_TRAIN_METRIC_FIELDS = frozenset({"loss", "learning_rate"})
_VALIDATION_METRIC_FIELDS = frozenset({"validation_loss", "validation_perplexity"})
_TARGET_CONTROL_FIELDS = frozenset(
    {
        "entity_match",
        "gold_canonicalization",
        "entity_text_token_weight",
        "validation_loss",
    }
)
_TARGET_SETTING_FIELDS = frozenset(
    {"gold_canonicalization", "entity_text_token_weight"}
)
TARGET_CONTROLS_SCHEMA = "microtensor.extract-target-controls.v1"
ENTITY_TEXT_TOKEN_BINDING = (
    "assistant_token_offset_overlaps_canonical_json_entity_text_value_span"
)
WEIGHTED_TRAINING_LOSS = (
    "weighted_shifted_causal_lm_cross_entropy_normalized_by_valid_weight"
)
ENTITY_SUBSTITUTION_ALGORITHM = "literal_same_type_train_fold_v1"
MAX_ENTITY_SUBSTITUTION_EXAMPLES = 1_024
ENTITY_SUBSTITUTION_COMPOSITION = "append_after_source_disease_oversampling"
CALIBRATION_LINEAGE_SCHEMA = "microtensor.gguf-imatrix-lineage.v1"
IMATRIX_CORPUS_SCHEMA = "microtensor.imatrix-corpus.v1"
LLAMA_CPP_REVISION = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
ATTN_V_Q6_OVERRIDE = r"^blk\.[0-9]+\.attn_v\.weight$=Q6_K"
WEIGHT_SOUP_SCHEMA = "microtensor.extract-weight-soup.v1"
WEIGHT_SOUP_METADATA_FILENAME = "soup_metadata.json"
_WEIGHT_SOUP_INPUT_FIELDS = frozenset(
    {
        "kind",
        "soup_schema",
        "soup_metadata_digest",
        "output_manifest_digest",
        "index_digest",
        "tokenizer_digest",
    }
)

_ENTITY_SUBSTITUTION_BASE_FIELDS = frozenset(
    {
        "algorithm",
        "enabled",
        "seed",
        "requested_examples",
        "augmented_examples",
        "replacement_count",
    }
)
_ENTITY_SUBSTITUTION_ENABLED_FIELDS = _ENTITY_SUBSTITUTION_BASE_FIELDS | frozenset(
    {
        "source_training_rows",
        "eligible_source_rows",
        "ineligible_source_rows",
        "globally_ambiguous_surfaces",
        "no_compatible_donor_rows",
        "donor_entity_counts",
        "source_training_refs_digest",
        "heldout_refs_digest",
        "donor_pool_digest",
    }
)


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
class CalibrationLineage:
    """A validated deterministic GGUF conversion/calibration declaration."""

    manifest: dict[str, Any]
    manifest_digest: str


@dataclass(frozen=True)
class BoundaryContrastiveLineage:
    """A stage-local boundary manifest independently replayed from its corpus."""

    stage_number: int
    manifest: dict[str, Any]
    manifest_digest: str
    corpus_digest: str


@dataclass(frozen=True)
class WeightSoupLineage:
    """A soup checkpoint revalidated for one explicit training stage."""

    stage_number: int
    metadata: dict[str, Any]
    metadata_digest: str
    output_manifest_digest: str
    index_digest: str
    tokenizer_digest: str


@dataclass(frozen=True)
class Publication:
    """Everything needed to publish, after local validation has completed."""

    stages: tuple[TrainingStage, ...]
    artifact_digest: str
    finished_block: int
    calibration: CalibrationLineage
    weight_soups: tuple[WeightSoupLineage, ...]
    boundary_contrastive: tuple[BoundaryContrastiveLineage, ...]

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


def _read_regular_file_without_symlinks(
    path: Path, *, maximum: int, label: str
) -> tuple[bytes, tuple[int, int]]:
    """Read through directory descriptors while rejecting every symlink component."""

    raw = os.fspath(path)
    lexical = Path(raw)
    if not raw or any(part in {".", ".."} for part in lexical.parts):
        raise ProvenanceValidationError(
            f"{label} path must not contain empty, dot, or parent components: {path}"
        )
    absolute = Path(os.path.abspath(raw))
    if len(absolute.parts) < 2:
        raise ProvenanceValidationError(f"{label} is not a file path: {path}")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_descriptor = -1
    file_descriptor = -1
    try:
        directory_descriptor = os.open(absolute.anchor, directory_flags)
        for component in absolute.parts[1:-1]:
            next_descriptor = os.open(
                component, directory_flags, dir_fd=directory_descriptor
            )
            opened_directory = os.fstat(next_descriptor)
            if not stat.S_ISDIR(opened_directory.st_mode):
                os.close(next_descriptor)
                raise ProvenanceValidationError(
                    f"{label} path component is not a directory: {component}"
                )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            absolute.parts[-1], file_flags, dir_fd=directory_descriptor
        )
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ProvenanceValidationError(f"{label} is not a regular file: {path}")
        if opened.st_size > maximum:
            raise ProvenanceValidationError(
                f"{label} exceeds the {maximum}-byte limit: {path}"
            )
        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = -1
            payload = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ProvenanceValidationError(
            f"{label} has a symlink, unsafe component, or unreadable file: {path}"
        ) from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)

    if len(payload) > maximum:
        raise ProvenanceValidationError(
            f"{label} exceeds the {maximum}-byte limit: {path}"
        )
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise ProvenanceValidationError(
            f"{label} changed while it was being read: {path}"
        )
    return payload, (after.st_dev, after.st_ino)


def _snapshot_regular_file_identity(
    path: Path, *, maximum: int, label: str
) -> tuple[int, str, tuple[int, int]]:
    """Hash a stable regular file and retain its opened inode identity."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise ProvenanceValidationError(
            f"{label} is not a readable regular file: {path}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise ProvenanceValidationError(f"{label} is not a regular file: {path}")
    if before.st_size > maximum:
        raise ProvenanceValidationError(
            f"{label} exceeds the {maximum}-byte limit: {path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise ProvenanceValidationError(
                f"{label} changed before it could be hashed: {path}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise ProvenanceValidationError(
                        f"{label} exceeds the {maximum}-byte limit: {path}"
                    )
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ProvenanceValidationError(
            f"{label} is not a readable regular file: {path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or size != after.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ProvenanceValidationError(f"{label} changed while it was hashed: {path}")
    return (
        size,
        "sha256:" + digest.hexdigest(),
        (after.st_dev, after.st_ino),
    )


def _snapshot_regular_file(
    path: Path, *, maximum: int, label: str
) -> tuple[int, str]:
    """Hash a stable regular file without exposing its inode identity."""

    size, digest, _identity = _snapshot_regular_file_identity(
        path, maximum=maximum, label=label
    )
    return size, digest


def _require_exact_fields(
    value: object, fields: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceValidationError(f"{label} must be an object")
    if frozenset(value) != fields:
        raise ProvenanceValidationError(
            f"{label} must contain exactly {sorted(fields)!r}"
        )
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProvenanceValidationError(
            f"{label} must be a non-empty, already stripped string"
        )
    return value


def _resolve_claimed_file(base: Path, value: object, label: str) -> Path:
    relative = _require_string(value, f"{label}.path")
    path = Path(relative)
    if (
        path.is_absolute()
        or "\\" in relative
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProvenanceValidationError(
            f"{label}.path must be a normalized relative path without traversal"
        )
    try:
        root = base.resolve(strict=True)
        candidate = root / path
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ProvenanceValidationError(
            f"{label}.path does not resolve inside the manifest directory"
        ) from exc
    return candidate


def _validate_file_claim(
    value: object,
    *,
    base: Path,
    label: str,
    maximum: int = MAX_LINEAGE_ASSET_BYTES,
) -> tuple[Path, tuple[int, int]]:
    claim = _require_exact_fields(
        value, frozenset({"path", "bytes", "sha256"}), label
    )
    path = _resolve_claimed_file(base, claim["path"], label)
    claimed_size = _require_int(claim["bytes"], f"{label}.bytes")
    claimed_digest = _require_digest(claim["sha256"], f"{label}.sha256")
    actual_size, actual_digest, actual_identity = _snapshot_regular_file_identity(
        path, maximum=maximum, label=label
    )
    if claimed_size != actual_size:
        raise ProvenanceValidationError(f"{label}.bytes does not match the file")
    if claimed_digest != actual_digest:
        raise ProvenanceValidationError(f"{label}.sha256 does not match the file")
    return path, actual_identity


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

    kind = training_input.get("kind")
    tokenizer_digest = _require_digest(
        training_input.get("tokenizer_digest"),
        f"stage {stage_number} training_input.tokenizer_digest",
    )

    if kind == "deterministic_weight_soup":
        if frozenset(training_input) != _WEIGHT_SOUP_INPUT_FIELDS:
            raise ProvenanceValidationError(
                f"stage {stage_number} deterministic_weight_soup input must contain exactly "
                f"{sorted(_WEIGHT_SOUP_INPUT_FIELDS)!r}"
            )
        if parent_metadata_digest is not None:
            raise ProvenanceValidationError(
                f"stage {stage_number} deterministic_weight_soup must start the supplied lineage"
            )
        if training_input["soup_schema"] != WEIGHT_SOUP_SCHEMA:
            raise ProvenanceValidationError(
                f"stage {stage_number} training_input.soup_schema is not supported"
            )
        for field in (
            "soup_metadata_digest",
            "output_manifest_digest",
            "index_digest",
        ):
            _require_digest(
                training_input[field], f"stage {stage_number} training_input.{field}"
            )
        if tokenizer_digest != BASE_TOKENIZER_DIGEST:
            raise ProvenanceValidationError(
                f"stage {stage_number} soup tokenizer digest is not allowlisted"
            )
        return

    weights_digest = _require_digest(
        training_input.get("weights_digest"),
        f"stage {stage_number} training_input.weights_digest",
    )

    if parent_metadata_digest is None:
        if kind != "huggingface_snapshot":
            raise ProvenanceValidationError(
                "stage 1 training_input.kind must be a huggingface_snapshot or "
                "deterministic_weight_soup"
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

    if kind != "derived_model":
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


def _load_weight_soup_module() -> Any:
    module_name = "training.build_weight_soup" if __package__ else "build_weight_soup"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProvenanceValidationError(
            "deterministic weight-soup validation dependencies are unavailable"
        ) from exc
    error_type = getattr(module, "SoupValidationError", None)
    if (
        getattr(module, "SCHEMA", None) != WEIGHT_SOUP_SCHEMA
        or getattr(module, "METADATA_FILENAME", None) != WEIGHT_SOUP_METADATA_FILENAME
        or not callable(getattr(module, "validate_weight_soup_checkpoint", None))
        or not isinstance(error_type, type)
        or not issubclass(error_type, Exception)
    ):
        raise ProvenanceValidationError(
            "weight-soup validator does not match the publisher schema"
        )
    return module


def _validate_weight_soup_checkpoints(
    stages: Sequence[TrainingStage],
    checkpoints: Mapping[int, Path] | None,
) -> tuple[WeightSoupLineage, ...]:
    supplied: dict[int, Path] = {}
    if checkpoints is not None:
        if not isinstance(checkpoints, Mapping):
            raise ProvenanceValidationError("soup checkpoints must be a stage-to-path mapping")
        for raw_stage, raw_path in checkpoints.items():
            stage_number = _require_int(raw_stage, "soup checkpoint stage")
            if not isinstance(raw_path, Path):
                raise ProvenanceValidationError(
                    f"soup checkpoint for stage {stage_number} must be a Path"
                )
            supplied[stage_number] = raw_path

    expected = {
        stage.number
        for stage in stages
        if stage.metadata["training_input"]["kind"] == "deterministic_weight_soup"
    }
    if set(supplied) != expected:
        missing = sorted(expected - set(supplied))
        extra = sorted(set(supplied) - expected)
        raise ProvenanceValidationError(
            f"soup checkpoint mapping does not exactly match soup-input stages; "
            f"missing={missing}, extra={extra}"
        )
    if not expected:
        return ()

    module = _load_weight_soup_module()
    lineages: list[WeightSoupLineage] = []
    for stage in stages:
        if stage.number not in expected:
            continue
        checkpoint = supplied[stage.number]
        try:
            validated = module.validate_weight_soup_checkpoint(checkpoint)
        except (OSError, RuntimeError, module.SoupValidationError) as exc:
            raise ProvenanceValidationError(
                f"stage {stage.number} weight-soup checkpoint is invalid: {exc}"
            ) from exc

        training_input = stage.metadata["training_input"]
        returned = {
            "soup_metadata_digest": _require_digest(
                getattr(validated, "metadata_digest", None),
                f"stage {stage.number} validated soup metadata digest",
            ),
            "output_manifest_digest": _require_digest(
                getattr(validated, "output_manifest_digest", None),
                f"stage {stage.number} validated soup output manifest digest",
            ),
            "index_digest": _require_digest(
                getattr(validated, "index_digest", None),
                f"stage {stage.number} validated soup index digest",
            ),
            "tokenizer_digest": _require_digest(
                getattr(validated, "tokenizer_digest", None),
                f"stage {stage.number} validated soup tokenizer digest",
            ),
        }
        for field, actual in returned.items():
            if training_input[field] != actual:
                raise ProvenanceValidationError(
                    f"stage {stage.number} training_input.{field} does not match "
                    "the validated soup checkpoint"
                )

        metadata_payload = _read_regular_file(
            checkpoint / WEIGHT_SOUP_METADATA_FILENAME,
            maximum=MAX_METADATA_BYTES,
            label=f"stage {stage.number} weight-soup metadata",
        )
        metadata_digest = _digest_bytes(metadata_payload)
        metadata = _parse_json_object(
            metadata_payload, f"stage {stage.number} weight-soup metadata"
        )
        if (
            metadata_digest != returned["soup_metadata_digest"]
            or metadata.get("schema") != WEIGHT_SOUP_SCHEMA
        ):
            raise ProvenanceValidationError(
                f"stage {stage.number} weight-soup metadata changed after validation"
            )
        lineages.append(
            WeightSoupLineage(
                stage_number=stage.number,
                metadata=metadata,
                metadata_digest=metadata_digest,
                output_manifest_digest=returned["output_manifest_digest"],
                index_digest=returned["index_digest"],
                tokenizer_digest=returned["tokenizer_digest"],
            )
        )
    return tuple(lineages)


def _validate_target_controls(
    metadata: Mapping[str, Any], settings: Mapping[str, Any], stage_number: int
) -> bool:
    """Validate the opt-in v1 scorer-aligned target-control declaration.

    Under this schema, an entity-weighted token is any assistant token whose
    tokenizer offset overlaps the canonical JSON string-content span for an
    entity's ``text`` value (quotes excluded, JSON escapes included).  Training
    uses weighted shifted causal-LM cross entropy normalized by valid weight;
    validation remains ordinary unweighted causal-LM loss.

    Metadata created before these controls had neither the top-level object nor
    its two settings, and remains valid.  Once either setting exists, the
    complete declaration is mandatory so stripping only the semantic object
    cannot silently weaken provenance.
    """

    setting_fields = frozenset(settings) & _TARGET_SETTING_FIELDS
    if "target_controls" not in metadata:
        if setting_fields:
            raise ProvenanceValidationError(
                f"stage {stage_number} target_controls is required when target-control "
                "settings are present"
            )
        return False

    controls = metadata["target_controls"]
    if not isinstance(controls, dict):
        raise ProvenanceValidationError(f"stage {stage_number} target_controls must be an object")
    if frozenset(controls) != _TARGET_CONTROL_FIELDS:
        raise ProvenanceValidationError(
            f"stage {stage_number} target_controls must contain exactly "
            f"{sorted(_TARGET_CONTROL_FIELDS)!r}"
        )
    if setting_fields != _TARGET_SETTING_FIELDS:
        raise ProvenanceValidationError(
            f"stage {stage_number} settings must contain both target-control settings"
        )

    gold_canonicalization = settings["gold_canonicalization"]
    if not isinstance(gold_canonicalization, str) or gold_canonicalization not in {
        "none",
        "first",
        "sorted",
    }:
        raise ProvenanceValidationError(
            f"stage {stage_number} settings.gold_canonicalization must be none, first, or sorted"
        )
    if controls["gold_canonicalization"] != gold_canonicalization:
        raise ProvenanceValidationError(
            f"stage {stage_number} target_controls.gold_canonicalization must exactly match settings"
        )

    settings_weight = _require_number(
        settings["entity_text_token_weight"],
        f"stage {stage_number} settings.entity_text_token_weight",
        minimum=1.0,
    )
    controls_weight = _require_number(
        controls["entity_text_token_weight"],
        f"stage {stage_number} target_controls.entity_text_token_weight",
        minimum=1.0,
    )
    if controls["entity_text_token_weight"] != settings["entity_text_token_weight"]:
        raise ProvenanceValidationError(
            f"stage {stage_number} target_controls.entity_text_token_weight must exactly match settings"
        )
    if controls_weight != settings_weight:
        raise ProvenanceValidationError(
            f"stage {stage_number} target-control weights are not numerically identical"
        )
    if settings_weight > 1.0 and gold_canonicalization == "none":
        raise ProvenanceValidationError(
            f"stage {stage_number} entity token weighting requires canonicalized gold"
        )
    if controls["entity_match"] != "exact_text_and_type_set":
        raise ProvenanceValidationError(
            f"stage {stage_number} target_controls.entity_match has unknown semantics"
        )
    if controls["validation_loss"] != "ordinary_unweighted_causal_lm":
        raise ProvenanceValidationError(
            f"stage {stage_number} target_controls.validation_loss has unknown semantics"
        )
    return True


def _validate_entity_substitution(
    metadata: Mapping[str, Any],
    settings: Mapping[str, Any],
    stage_number: int,
    directory: Path,
) -> None:
    """Bind opt-in entity substitution settings to its exact generated manifest."""

    has_setting = "entity_substitution_examples" in settings
    has_declaration = "augmentation" in metadata
    if not has_setting and not has_declaration:
        return
    if has_setting != has_declaration:
        missing = "augmentation" if has_setting else "settings.entity_substitution_examples"
        raise ProvenanceValidationError(
            f"stage {stage_number} {missing} is required for entity substitution"
        )

    label = f"stage {stage_number} entity substitution"
    requested = _require_int(
        settings["entity_substitution_examples"],
        f"stage {stage_number} settings.entity_substitution_examples",
        minimum=0,
    )
    seed = _require_int(
        settings.get("seed"), f"stage {stage_number} settings.seed", minimum=0
    )
    if requested > MAX_ENTITY_SUBSTITUTION_EXAMPLES:
        raise ProvenanceValidationError(
            f"{label} exceeds {MAX_ENTITY_SUBSTITUTION_EXAMPLES} examples"
        )
    outer = _require_exact_fields(
        metadata["augmentation"],
        frozenset({"entity_substitution"}),
        f"stage {stage_number} augmentation",
    )
    enabled = requested > 0
    expected_composition = (
        "disabled_for_boundary_contrastive"
        if settings.get("boundary_contrastive") is True
        else ENTITY_SUBSTITUTION_COMPOSITION
    )
    fields = _ENTITY_SUBSTITUTION_BASE_FIELDS | frozenset({"composition"})
    if enabled:
        fields |= _ENTITY_SUBSTITUTION_ENABLED_FIELDS | frozenset(
            {"manifest_file", "manifest_digest"}
        )
    claim = _require_exact_fields(outer["entity_substitution"], fields, label)
    augmented = _require_int(
        claim["augmented_examples"], f"{label} augmented_examples", minimum=0
    )
    claimed_seed = _require_int(claim["seed"], f"{label} seed", minimum=0)
    claimed_requested = _require_int(
        claim["requested_examples"], f"{label} requested_examples", minimum=0
    )
    replacement_count = _require_int(
        claim["replacement_count"], f"{label} replacement_count", minimum=0
    )
    if (
        claim["algorithm"] != ENTITY_SUBSTITUTION_ALGORITHM
        or claim["enabled"] is not enabled
        or claim["composition"] != expected_composition
        or claimed_seed != seed
        or claimed_requested != requested
        or augmented > requested
        or replacement_count != augmented
    ):
        raise ProvenanceValidationError(f"{label} settings or counts are inconsistent")

    count_minimums = {
        "source_training_examples": 1,
        "disease_extra_examples": 0,
        "training_examples": 1,
        "skipped_training_examples": 0,
    }
    counts = {
        field: _require_int(
            metadata.get(field), f"stage {stage_number} {field}", minimum=minimum
        )
        for field, minimum in count_minimums.items()
    }
    if (
        counts["training_examples"] + counts["skipped_training_examples"]
        != counts["source_training_examples"] + counts["disease_extra_examples"] + augmented
    ):
        raise ProvenanceValidationError(
            f"stage {stage_number} training example counts do not bind augmentation"
        )
    if not enabled:
        if augmented:
            raise ProvenanceValidationError(f"{label} is disabled but generated examples")
        return

    for field in (
        "source_training_refs_digest",
        "heldout_refs_digest",
        "donor_pool_digest",
    ):
        _require_digest(claim[field], f"{label} {field}")
    source_rows = _require_int(
        claim["source_training_rows"], f"{label} source_training_rows"
    )
    if source_rows != counts["source_training_examples"]:
        raise ProvenanceValidationError(f"{label} source row count is inconsistent")
    if claim["manifest_file"] != "entity_substitution_manifest.json":
        raise ProvenanceValidationError(f"{label} manifest filename is not allowlisted")
    manifest_digest = _require_digest(claim["manifest_digest"], f"{label} manifest_digest")
    manifest_bytes = _read_regular_file(
        directory / claim["manifest_file"],
        maximum=MAX_AUGMENTATION_MANIFEST_BYTES,
        label=f"{label} manifest",
    )
    if _digest_bytes(manifest_bytes) != manifest_digest:
        raise ProvenanceValidationError(f"{label} manifest_digest does not match the file")
    manifest = _parse_json_object(manifest_bytes, f"{label} manifest")
    _require_exact_fields(
        manifest,
        _ENTITY_SUBSTITUTION_ENABLED_FIELDS | frozenset({"examples"}),
        f"{label} manifest",
    )
    summary = {key: value for key, value in manifest.items() if key != "examples"}
    expected = {
        key: value
        for key, value in claim.items()
        if key not in {"composition", "manifest_file", "manifest_digest"}
    }
    examples = manifest["examples"]
    if (
        summary != expected
        or not isinstance(examples, list)
        or len(examples) != augmented
    ):
        raise ProvenanceValidationError(
            f"{label} manifest does not match training metadata"
        )


def _canonical_boundary_json(value: object, label: str) -> bytes:
    try:
        return boundary.canonical_json_bytes(value)
    except ValueError as exc:
        raise ProvenanceValidationError(
            f"{label} is not canonical strict JSON"
        ) from exc


def _validate_boundary_lineages(
    stages: Sequence[TrainingStage],
    corpora: Mapping[int, Path] | None,
) -> tuple[BoundaryContrastiveLineage, ...]:
    """Independently replay every declared boundary experiment from exact corpus bytes."""

    setting_fields = frozenset(
        {
            "boundary_contrastive",
            "boundary_contrastive_examples",
            "boundary_contrastive_lambda",
            "boundary_contrastive_margin",
        }
    )
    supplied: dict[int, Path] = {}
    if corpora is not None:
        if not isinstance(corpora, Mapping):
            raise ProvenanceValidationError(
                "boundary corpora must be a stage-to-path mapping"
            )
        for raw_stage, raw_path in corpora.items():
            stage_number = _require_int(raw_stage, "boundary corpus stage")
            if not isinstance(raw_path, Path):
                raise ProvenanceValidationError(
                    f"boundary corpus for stage {stage_number} must be a Path"
                )
            supplied[stage_number] = raw_path

    expected: set[int] = set()
    for stage in stages:
        settings = stage.metadata.get("settings")
        if not isinstance(settings, dict):
            raise ProvenanceValidationError(
                f"stage {stage.number} settings must be an object"
            )
        present_settings = frozenset(settings) & setting_fields
        has_declaration = "boundary_contrastive" in stage.metadata
        if present_settings and present_settings != setting_fields:
            raise ProvenanceValidationError(
                f"stage {stage.number} boundary settings must be all present or all absent"
            )
        if bool(present_settings) != has_declaration:
            raise ProvenanceValidationError(
                f"stage {stage.number} boundary settings and declaration must appear together"
            )
        if has_declaration:
            expected.add(stage.number)

    if set(supplied) != expected:
        missing = sorted(expected - set(supplied))
        extra = sorted(set(supplied) - expected)
        raise ProvenanceValidationError(
            "boundary corpus mapping does not exactly match boundary stages; "
            f"missing={missing}, extra={extra}"
        )
    if not expected:
        return ()

    lineages: list[BoundaryContrastiveLineage] = []
    for stage in stages:
        if stage.number not in expected:
            continue
        label = f"stage {stage.number} boundary contrastive"
        metadata = stage.metadata
        settings = metadata["settings"]
        if settings["boundary_contrastive"] is not True:
            raise ProvenanceValidationError(f"{label} must be explicitly enabled")
        requested = _require_int(
            settings["boundary_contrastive_examples"],
            f"{label} requested examples",
            minimum=0,
        )
        if (
            requested > boundary.MAX_BOUNDARY_CONTRASTIVE_EXAMPLES
            or requested % 2
        ):
            raise ProvenanceValidationError(
                f"{label} requested examples must be even and in [0, 512]"
            )
        pairwise_lambda = _require_number(
            settings["boundary_contrastive_lambda"],
            f"{label} pairwise lambda",
        )
        margin = _require_number(
            settings["boundary_contrastive_margin"], f"{label} margin"
        )
        if (
            pairwise_lambda > boundary.MAX_BOUNDARY_CONTRASTIVE_LAMBDA
            or margin > boundary.MAX_BOUNDARY_CONTRASTIVE_MARGIN
        ):
            raise ProvenanceValidationError(f"{label} loss controls are out of range")

        expected_controls = {
            "seed": boundary.BOUNDARY_OUTER_SEED,
            "max_length": 512,
            "validation_examples": boundary.BOUNDARY_OUTER_EXAMPLES,
            "disease_row_weight": 1.0,
            "gold_canonicalization": "none",
            "entity_text_token_weight": 1.0,
            "entity_substitution_examples": 0,
        }
        actual_controls = {key: settings.get(key) for key in expected_controls}
        if _canonical_boundary_json(
            actual_controls, f"{label} training controls"
        ) != _canonical_boundary_json(
            expected_controls, f"{label} expected training controls"
        ):
            raise ProvenanceValidationError(
                f"{label} training controls are not the fixed v1 recipe"
            )
        training_input = metadata.get("training_input")
        if (
            not isinstance(training_input, dict)
            or training_input.get("tokenizer_digest")
            != boundary.BASE_TOKENIZER_DIGEST
        ):
            raise ProvenanceValidationError(
                f"{label} tokenizer digest is not allowlisted"
            )
        if metadata.get("corpus_file_digest") != boundary.BOUNDARY_CORPUS_FILE_DIGEST:
            raise ProvenanceValidationError(
                f"{label} corpus digest is not the pinned public corpus"
            )
        expected_counts = {
            "source_training_examples": boundary.BOUNDARY_INNER_TRAIN_EXAMPLES,
            "training_examples": boundary.BOUNDARY_INNER_TRAIN_EXAMPLES,
            "validation_examples": boundary.BOUNDARY_INNER_VALIDATION_EXAMPLES,
            "skipped_training_examples": 0,
            "skipped_validation_examples": 0,
            "disease_extra_examples": 0,
        }
        for field, expected_count in expected_counts.items():
            actual_count = _require_int(
                metadata.get(field), f"{label} {field}", minimum=0
            )
            if actual_count != expected_count:
                raise ProvenanceValidationError(
                    f"{label} {field} must be exactly {expected_count}"
                )

        corpus_payload, corpus_identity = _read_regular_file_without_symlinks(
            supplied[stage.number],
            maximum=boundary.BOUNDARY_CORPUS_BYTES,
            label=f"{label} corpus",
        )
        try:
            rows = boundary.parse_boundary_corpus(corpus_payload)
            expected_manifest, fold, corruption = boundary.build_boundary_manifest(
                rows,
                skipped_refs=boundary.BOUNDARY_SKIPPED_ENCODED_REFS,
                requested_examples=requested,
                pairwise_lambda=settings["boundary_contrastive_lambda"],
                margin=settings["boundary_contrastive_margin"],
            )
            row_by_ref = {row["ref"]: row for row in rows}
            expected_disease_source_examples = boundary.disease_row_count(
                [
                    row_by_ref[ref]
                    for ref in fold.manifest["inner_train_refs"]
                ]
            )
        except ValueError as exc:
            raise ProvenanceValidationError(f"{label} replay failed: {exc}") from exc
        actual_disease_source_examples = _require_int(
            metadata.get("disease_source_examples"),
            f"{label} disease_source_examples",
            minimum=0,
        )
        if actual_disease_source_examples != expected_disease_source_examples:
            raise ProvenanceValidationError(
                f"{label} disease_source_examples must be exactly "
                f"{expected_disease_source_examples}"
            )
        if (
            requested == boundary.MAX_BOUNDARY_CONTRASTIVE_EXAMPLES
            and corruption.manifest["direction_counts"]
            != {"expansion": 256, "contraction": 256}
        ):
            raise ProvenanceValidationError(
                f"{label} did not replay 512 balanced boundary pairs"
            )

        manifest_path = stage.directory / "boundary_contrastive_manifest.json"
        manifest_payload, manifest_identity = _read_regular_file_without_symlinks(
            manifest_path,
            maximum=MAX_BOUNDARY_MANIFEST_BYTES,
            label=f"{label} manifest",
        )
        if manifest_identity == corpus_identity:
            raise ProvenanceValidationError(
                f"{label} corpus and manifest must be distinct files"
            )
        manifest_digest = _digest_bytes(manifest_payload)
        manifest = _parse_json_object(manifest_payload, f"{label} manifest")
        if _canonical_boundary_json(
            manifest, f"{label} manifest"
        ) != _canonical_boundary_json(
            expected_manifest, f"{label} replayed manifest"
        ):
            raise ProvenanceValidationError(
                f"{label} manifest does not match independent corpus replay"
            )
        expected_summary = boundary.boundary_metadata_summary(
            expected_manifest, manifest_digest=manifest_digest
        )
        declaration = metadata["boundary_contrastive"]
        if _canonical_boundary_json(
            declaration, f"{label} metadata declaration"
        ) != _canonical_boundary_json(
            expected_summary, f"{label} expected metadata summary"
        ):
            raise ProvenanceValidationError(
                f"{label} metadata summary does not match its replayed manifest"
            )
        lineages.append(
            BoundaryContrastiveLineage(
                stage_number=stage.number,
                manifest=manifest,
                manifest_digest=manifest_digest,
                corpus_digest=boundary.digest_bytes(corpus_payload),
            )
        )
    return tuple(lineages)


def _validate_metadata_times_and_counts(
    metadata: Mapping[str, Any],
    stage_number: int,
    directory: Path,
) -> tuple[int, int, int, int, float]:
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ProvenanceValidationError(f"stage {stage_number} settings must be an object")
    _validate_target_controls(metadata, settings, stage_number)
    _validate_entity_substitution(metadata, settings, stage_number, directory)
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


def _validate_source_model(
    value: object, final_stage: TrainingStage, manifest_base: Path
) -> dict[str, tuple[int, str]]:
    """Hash the exact merged directory associated with the final training stage."""

    source = _require_exact_fields(
        value,
        frozenset({"directory", "training_metadata_sha256", "files"}),
        "calibration source_model",
    )
    merged = final_stage.directory / "merged"
    try:
        expected_directory = unicodedata.normalize(
            "NFC",
            merged.resolve(strict=True)
            .relative_to(manifest_base.resolve(strict=True))
            .as_posix(),
        )
    except (OSError, ValueError) as exc:
        raise ProvenanceValidationError(
            "calibration source model must be inside the manifest directory"
        ) from exc
    if (
        source["directory"] != expected_directory
        or _require_digest(
            source["training_metadata_sha256"],
            "calibration source_model.training_metadata_sha256",
        )
        != final_stage.metadata_digest
    ):
        raise ProvenanceValidationError(
            "calibration source model does not bind the final training stage"
        )
    try:
        details = merged.lstat()
        entries = sorted(merged.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ProvenanceValidationError(
            f"calibration source model directory is unavailable: {merged}"
        ) from exc
    if not stat.S_ISDIR(details.st_mode):
        raise ProvenanceValidationError(
            "calibration source model must be a non-symlink directory"
        )
    if any(not stat.S_ISREG(entry.lstat().st_mode) for entry in entries):
        raise ProvenanceValidationError(
            "calibration source model entries must all be regular files"
        )

    claims = source["files"]
    if not isinstance(claims, list) or not claims:
        raise ProvenanceValidationError(
            "calibration source_model.files must be a non-empty list"
        )
    snapshots: dict[str, tuple[int, str]] = {}
    names: list[str] = []
    for index, raw_claim in enumerate(claims, start=1):
        label = f"calibration source model file {index}"
        claim = _require_exact_fields(
            raw_claim, frozenset({"path", "bytes", "sha256"}), label
        )
        name = _require_string(claim["path"], f"{label}.path")
        if Path(name).name != name or "\\" in name:
            raise ProvenanceValidationError(f"{label}.path must be a basename")
        _validate_file_claim(claim, base=merged, label=label)
        names.append(name)
        snapshots[name] = (
            _require_int(claim["bytes"], f"{label}.bytes"),
            _require_digest(claim["sha256"], f"{label}.sha256"),
        )

    if names != sorted(names) or names != [entry.name for entry in entries]:
        raise ProvenanceValidationError(
            "calibration source_model.files must be a sorted exact directory inventory"
        )
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(snapshots) or not any(
        name.endswith(".safetensors") for name in snapshots
    ):
        raise ProvenanceValidationError("calibration source model inventory is incomplete")
    return snapshots


def _validate_imatrix_corpus_metadata(
    metadata: dict[str, Any],
    *,
    corpus_payload: bytes,
    stage_corpus_digest: str,
    source_files: Mapping[str, tuple[int, str]],
) -> None:
    """Validate sidecar links to the training corpus, tokenizer, and output bytes."""

    label = "calibration corpus metadata"
    _require_exact_fields(
        metadata,
        frozenset(
            {
                "schema",
                "source",
                "tokenizer",
                "selection",
                "rendering",
                "output",
                "records",
                "runtime",
            }
        ),
        label,
    )
    source = _require_exact_fields(
        metadata["source"],
        frozenset(
            {
                "corpus_bytes",
                "corpus_file_sha256",
                "corpus_version",
                "loader",
                "public_train_rows",
            }
        ),
        f"{label}.source",
    )
    if (
        metadata["schema"] != IMATRIX_CORPUS_SCHEMA
        or source["corpus_file_sha256"] != stage_corpus_digest
        or source["corpus_version"] != CORPUS_VERSION
        or source["loader"]
        not in {"train_extract.load_rows", "training.train_extract.load_rows"}
    ):
        raise ProvenanceValidationError(f"{label}.source is not allowlisted")
    _require_int(source["corpus_bytes"], f"{label}.source.corpus_bytes")
    _require_int(source["public_train_rows"], f"{label}.source.public_train_rows")

    tokenizer = _require_exact_fields(
        metadata["tokenizer"],
        frozenset(
            {
                "artifacts",
                "chat_template_sha256",
                "chat_template_source",
                "loader",
                "local_files_only",
                "runtime_class",
                "trust_remote_code",
                "model_type",
                "tokenizer_class",
            }
        ),
        f"{label}.tokenizer",
    )
    expected_tokenizer = {
        "loader": "transformers.AutoTokenizer.from_pretrained",
        "local_files_only": True,
        "trust_remote_code": False,
        "model_type": "qwen3",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    if any(tokenizer.get(key) != value for key, value in expected_tokenizer.items()):
        raise ProvenanceValidationError(f"{label}.tokenizer is not allowlisted")
    _require_digest(
        tokenizer["chat_template_sha256"], f"{label}.tokenizer.chat_template_sha256"
    )
    artifact_names = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if tokenizer["chat_template_source"] == "chat_template.jinja":
        artifact_names.add("chat_template.jinja")
    elif tokenizer["chat_template_source"] != "tokenizer_config.json:chat_template":
        raise ProvenanceValidationError(f"{label}.tokenizer template source is unknown")
    artifacts = tokenizer["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_names:
        raise ProvenanceValidationError(f"{label}.tokenizer artifacts are incomplete")
    for name, raw_snapshot in artifacts.items():
        snapshot = _require_exact_fields(
            raw_snapshot,
            frozenset({"bytes", "sha256"}),
            f"{label}.tokenizer.artifacts.{name}",
        )
        claimed = (
            _require_int(snapshot["bytes"], f"{label}.tokenizer.{name}.bytes"),
            _require_digest(snapshot["sha256"], f"{label}.tokenizer.{name}.sha256"),
        )
        if source_files.get(name) != claimed:
            raise ProvenanceValidationError(
                f"{label} tokenizer artifact {name} differs from the source model"
            )

    expected_rendering = {
        "add_generation_prompt": False,
        "canonical_gold": {
            "deduplicate": "exact_text_type_pair",
            "entity_order": "unicode_text_then_type",
            "json": "utf8_compact_sorted_keys",
            "substring_scope": "inputs.text",
        },
        "enable_thinking": False,
        "record_separator": "none; each Qwen rendering ends with <|im_end|>\\n",
        "tokenize": False,
    }
    if metadata["rendering"] != expected_rendering:
        raise ProvenanceValidationError(f"{label}.rendering is not allowlisted")

    selection = _require_exact_fields(
        metadata["selection"],
        frozenset(
            {
                "algorithm",
                "eligible_examples",
                "included_examples",
                "included_refs",
                "max_examples",
                "omitted_after_cap",
                "rejected_rows",
                "reserve_examples",
                "reserved_refs",
                "seed",
            }
        ),
        f"{label}.selection",
    )
    seed = _require_int(selection["seed"], f"{label}.selection.seed", minimum=0)
    reserve = _require_int(
        selection["reserve_examples"], f"{label}.selection.reserve_examples", minimum=0
    )
    included = _require_int(
        selection["included_examples"], f"{label}.selection.included_examples"
    )
    maximum = selection["max_examples"]
    if maximum is not None:
        _require_int(maximum, f"{label}.selection.max_examples")
    records = metadata["records"]
    if (
        selection["algorithm"] != "random.Random(seed).shuffle(rows)"
        or seed != 92
        or reserve not in {0, 384}
        or not isinstance(selection["included_refs"], list)
        or len(selection["included_refs"]) != included
        or not isinstance(selection["reserved_refs"], list)
        or len(selection["reserved_refs"]) != reserve
        or not isinstance(selection["rejected_rows"], list)
        or not isinstance(records, list)
        or len(records) != included
    ):
        raise ProvenanceValidationError(f"{label}.selection is inconsistent")

    output = _require_exact_fields(
        metadata["output"],
        frozenset({"bytes", "records", "sha256"}),
        f"{label}.output",
    )
    output_bytes = _require_int(output["bytes"], f"{label}.output.bytes")
    output_records = _require_int(output["records"], f"{label}.output.records")
    output_digest = _require_digest(output["sha256"], f"{label}.output.sha256")
    if (
        output_bytes != len(corpus_payload)
        or output_records != len(records)
        or output_digest != _digest_bytes(corpus_payload)
    ):
        raise ProvenanceValidationError(
            f"{label}.output does not match calibration corpus bytes"
        )


def _require_gguf(path: Path, label: str) -> None:
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        raise ProvenanceValidationError(f"{label} cannot be inspected") from exc
    if magic != b"GGUF":
        raise ProvenanceValidationError(f"{label} is not a GGUF file")


def _artifact_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProvenanceValidationError(
                f"calibration artifact tree contains a symlink: {path}"
            )
        if path.is_file():
            parts = path.relative_to(root).parts
            if parts in {("manifest.json",), ("artifact.enc",)} or any(
                part.startswith(".") for part in parts
            ):
                continue
            files.append(path)
    if not files:
        raise ProvenanceValidationError("calibration artifact tree has no files")
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        _size, file_digest = _snapshot_regular_file(
            path, maximum=MAX_LINEAGE_ASSET_BYTES, label="calibration artifact file"
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\x00")
    return "sha256:" + digest.hexdigest()


def _validate_calibration_lineage(
    path: Path,
    *,
    stages: Sequence[TrainingStage],
    artifact_digest: str,
    stage_corpus_digest: str,
) -> CalibrationLineage:
    """Bind source model, public calibration, imatrix, and final artifact bytes."""

    manifest_path = Path(path)
    manifest_bytes = _read_regular_file(
        manifest_path, maximum=MAX_METADATA_BYTES, label="calibration lineage manifest"
    )
    manifest = _parse_json_object(manifest_bytes, "calibration lineage manifest")
    _require_exact_fields(
        manifest,
        frozenset(
            {
                "schema",
                "llama_cpp_revision",
                "artifact_tree_digest",
                "source_model",
                "conversion",
                "calibration",
                "quantization",
            }
        ),
        "calibration lineage manifest",
    )
    declared_artifact = _require_digest(
        manifest["artifact_tree_digest"], "calibration lineage artifact_tree_digest"
    )
    if (
        manifest["schema"] != CALIBRATION_LINEAGE_SCHEMA
        or manifest["llama_cpp_revision"] != LLAMA_CPP_REVISION
        or declared_artifact != artifact_digest
    ):
        raise ProvenanceValidationError(
            "calibration lineage identity does not match the allowlisted publication"
        )

    base = manifest_path.parent
    source = manifest["source_model"]
    source_files = _validate_source_model(source, stages[-1], base)
    conversion = _require_exact_fields(
        manifest["conversion"],
        frozenset({"tool", "arguments", "outtype", "output"}),
        "calibration conversion",
    )
    converted_path, converted_identity = _validate_file_claim(
        conversion["output"], base=base, label="calibration F16 model"
    )
    expected_conversion_arguments = [
        source["directory"],
        "--outfile",
        conversion["output"]["path"],
        "--outtype",
        "f16",
    ]
    if (
        conversion["tool"] != "convert_hf_to_gguf.py"
        or conversion["outtype"] != "f16"
        or conversion["arguments"] != expected_conversion_arguments
    ):
        raise ProvenanceValidationError(
            "calibration conversion ordered arguments are not allowlisted"
        )
    _require_gguf(converted_path, "calibration F16 model")

    calibration = _require_exact_fields(
        manifest["calibration"],
        frozenset(
            {"tool", "arguments", "corpus", "metadata", "imatrix", "settings"}
        ),
        "calibration declaration",
    )
    if calibration["tool"] != "llama-imatrix":
        raise ProvenanceValidationError("calibration tool is not allowlisted")
    settings = _require_exact_fields(
        calibration["settings"],
        frozenset(
            {
                "offline",
                "ctx_size",
                "chunks",
                "no_ppl",
                "process_output",
                "parse_special",
                "output_format",
            }
        ),
        "calibration settings",
    )
    expected_settings = {
        "offline": True,
        "ctx_size": 512,
        "no_ppl": True,
        "process_output": False,
        "parse_special": True,
        "output_format": "gguf",
    }
    if (
        any(settings.get(key) != value for key, value in expected_settings.items())
        or settings["chunks"] != -1
    ):
        raise ProvenanceValidationError("calibration settings are not allowlisted")

    corpus_path, corpus_identity = _validate_file_claim(
        calibration["corpus"],
        base=base,
        label="calibration corpus",
        maximum=MAX_CALIBRATION_CORPUS_BYTES,
    )
    metadata_path, metadata_identity = _validate_file_claim(
        calibration["metadata"],
        base=base,
        label="calibration corpus metadata",
        maximum=MAX_CALIBRATION_METADATA_BYTES,
    )
    imatrix_path, imatrix_identity = _validate_file_claim(
        calibration["imatrix"], base=base, label="calibration imatrix"
    )
    expected_imatrix_arguments = [
        "--offline",
        "--model",
        conversion["output"]["path"],
        "--file",
        calibration["corpus"]["path"],
        "--output",
        calibration["imatrix"]["path"],
        "--ctx-size",
        "512",
        "--chunks",
        "-1",
        "--no-ppl",
        "--parse-special",
    ]
    if calibration["arguments"] != expected_imatrix_arguments:
        raise ProvenanceValidationError(
            "calibration imatrix ordered arguments are not allowlisted"
        )
    corpus_payload = _read_regular_file(
        corpus_path, maximum=MAX_CALIBRATION_CORPUS_BYTES, label="calibration corpus"
    )
    sidecar = _parse_json_object(
        _read_regular_file(
            metadata_path,
            maximum=MAX_CALIBRATION_METADATA_BYTES,
            label="calibration corpus metadata",
        ),
        "calibration corpus metadata",
    )
    _validate_imatrix_corpus_metadata(
        sidecar,
        corpus_payload=corpus_payload,
        stage_corpus_digest=stage_corpus_digest,
        source_files=source_files,
    )
    _require_gguf(imatrix_path, "calibration imatrix")

    quantization = _require_exact_fields(
        manifest["quantization"],
        frozenset({"tool", "arguments", "output"}),
        "calibration quantization",
    )
    output_path, output_identity = _validate_file_claim(
        quantization["output"], base=base, label="calibration quantized artifact"
    )
    base_arguments = [
        "--imatrix",
        calibration["imatrix"]["path"],
        conversion["output"]["path"],
        quantization["output"]["path"],
        "Q4_K_M",
    ]
    override_arguments = [
        *base_arguments[:2],
        "--tensor-type",
        ATTN_V_Q6_OVERRIDE,
        *base_arguments[2:],
    ]
    if (
        quantization["tool"] != "llama-quantize"
        or quantization["arguments"] not in (base_arguments, override_arguments)
    ):
        raise ProvenanceValidationError(
            "calibration quantization ordered arguments are not allowlisted"
        )
    _require_gguf(output_path, "calibration quantized artifact")
    if _artifact_tree_digest(output_path.parent) != declared_artifact:
        raise ProvenanceValidationError(
            "calibration artifact tree digest does not match quantized output"
        )
    paths = {
        corpus_path.resolve(),
        metadata_path.resolve(),
        imatrix_path.resolve(),
        converted_path.resolve(),
        output_path.resolve(),
    }
    identities = {
        corpus_identity,
        metadata_identity,
        imatrix_identity,
        converted_identity,
        output_identity,
    }
    if len(paths) != 5 or len(identities) != 5:
        raise ProvenanceValidationError(
            "calibration roles must reference distinct files and inodes"
        )

    return CalibrationLineage(
        manifest=manifest,
        manifest_digest=_digest_bytes(manifest_bytes),
    )


def validate_publication(
    training_dirs: Sequence[Path],
    artifact_digest: str,
    finished_block: int,
    *,
    calibration_manifest: Path | None = None,
    weight_soup_checkpoints: Mapping[int, Path] | None = None,
    boundary_corpora: Mapping[int, Path] | None = None,
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
        reserved_metadata_fields = sorted(
            field for field in metadata if field.startswith("mt_")
        )
        if reserved_metadata_fields:
            raise ProvenanceValidationError(
                f"stage {stage_number} uses publisher-reserved metadata fields: "
                f"{reserved_metadata_fields}"
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
            metadata, stage_number, directory
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

    if corpus_file_digest is None:
        raise ProvenanceValidationError("training lineage has no corpus digest")
    boundary_lineages = _validate_boundary_lineages(stages, boundary_corpora)
    weight_soups = _validate_weight_soup_checkpoints(stages, weight_soup_checkpoints)
    if calibration_manifest is None:
        raise ProvenanceValidationError(
            "calibration manifest is required for every publication"
        )
    calibration = _validate_calibration_lineage(
        Path(calibration_manifest),
        stages=stages,
        artifact_digest=artifact_digest,
        stage_corpus_digest=corpus_file_digest,
    )

    return Publication(
        stages=tuple(stages),
        artifact_digest=artifact_digest,
        finished_block=finished_block,
        calibration=calibration,
        weight_soups=weight_soups,
        boundary_contrastive=boundary_lineages,
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
    target_control_semantics: dict[str, Any] = {}
    if any("target_controls" in stage.metadata for stage in publication.stages):
        target_control_semantics["mt_target_controls_semantics"] = {
            "schema": TARGET_CONTROLS_SCHEMA,
            "canonicalization": {
                "none": "preserve_raw_gold_target",
                "first": "strict_exact_pair_deduplication_in_first_occurrence_order",
                "sorted": "strict_exact_pair_deduplication_in_lexicographic_order",
            },
            "malformed_gold": "reject_without_repair",
            "entity_text_token_binding": ENTITY_TEXT_TOKEN_BINDING,
            "entity_text_weighting_loss": WEIGHTED_TRAINING_LOSS,
            "weighting_active_when": "entity_text_token_weight_greater_than_one",
            "validation_loss": "ordinary_unweighted_causal_lm",
        }
    boundary_lineage: dict[str, Any] = {}
    if publication.boundary_contrastive:
        boundary_lineage["mt_boundary_contrastive_lineage"] = {
            f"stage_{lineage.stage_number}": {
                "schema": lineage.manifest["schema"],
                "manifest_digest": lineage.manifest_digest,
                "corpus_digest": lineage.corpus_digest,
                "manifest": lineage.manifest,
            }
            for lineage in publication.boundary_contrastive
        }
    soup_lineage: dict[str, Any] = {}
    if publication.weight_soups:
        soup_lineage["mt_weight_soup_lineage"] = {
            f"stage_{soup.stage_number}": {
                "schema": WEIGHT_SOUP_SCHEMA,
                "metadata_digest": soup.metadata_digest,
                "output_manifest_digest": soup.output_manifest_digest,
                "index_digest": soup.index_digest,
                "tokenizer_digest": soup.tokenizer_digest,
                "metadata": soup.metadata,
            }
            for soup in publication.weight_soups
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
            **target_control_semantics,
            **boundary_lineage,
            **soup_lineage,
            "mt_calibration_lineage": {
                "manifest_digest": publication.calibration.manifest_digest,
                "manifest": publication.calibration.manifest,
            },
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
        run.summary["mt_calibration_manifest_digest"] = (
            publication.calibration.manifest_digest
        )
    finally:
        wandb_client.finish()


def _index_weight_soup_checkpoints(
    entries: Sequence[Sequence[str]],
) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for entry in entries:
        if len(entry) != 2:
            raise ProvenanceValidationError(
                "each weight-soup checkpoint requires STAGE and PATH"
            )
        stage_text, path_text = entry
        try:
            stage_number = int(stage_text)
        except ValueError as exc:
            raise ProvenanceValidationError(
                f"weight-soup checkpoint stage is not an integer: {stage_text!r}"
            ) from exc
        if stage_number < 1 or str(stage_number) != stage_text:
            raise ProvenanceValidationError(
                "weight-soup checkpoint stages must be canonical positive integers"
            )
        if stage_number in checkpoints:
            raise ProvenanceValidationError(
                f"duplicate weight-soup checkpoint for stage {stage_number}"
            )
        if not path_text or path_text != path_text.strip():
            raise ProvenanceValidationError(
                f"weight-soup checkpoint path for stage {stage_number} is invalid"
            )
        checkpoints[stage_number] = Path(path_text)
    return checkpoints


def _index_boundary_corpora(
    entries: Sequence[Sequence[str]],
) -> dict[int, Path]:
    corpora: dict[int, Path] = {}
    for entry in entries:
        if len(entry) != 2:
            raise ProvenanceValidationError(
                "each boundary corpus requires STAGE and PATH"
            )
        stage_text, path_text = entry
        try:
            stage_number = int(stage_text)
        except ValueError as exc:
            raise ProvenanceValidationError(
                f"boundary corpus stage is not an integer: {stage_text!r}"
            ) from exc
        if stage_number < 1 or str(stage_number) != stage_text:
            raise ProvenanceValidationError(
                "boundary corpus stages must be canonical positive integers"
            )
        if stage_number in corpora:
            raise ProvenanceValidationError(
                f"duplicate boundary corpus for stage {stage_number}"
            )
        if not path_text or path_text != path_text.strip():
            raise ProvenanceValidationError(
                f"boundary corpus path for stage {stage_number} is invalid"
            )
        corpora[stage_number] = Path(path_text)
    return corpora


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
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument(
        "--boundary-corpus",
        action="append",
        nargs=2,
        default=[],
        metavar=("STAGE", "PATH"),
        help=(
            "exact public corpus for a 1-based boundary-contrastive training stage; "
            "repeat as needed"
        ),
    )
    parser.add_argument(
        "--weight-soup-checkpoint",
        action="append",
        nargs=2,
        default=[],
        metavar=("STAGE", "PATH"),
        help=(
            "checkpoint for a 1-based deterministic_weight_soup training stage; "
            "repeat as needed"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None, *, wandb_client: Any | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        weight_soup_checkpoints = _index_weight_soup_checkpoints(
            args.weight_soup_checkpoint
        )
        boundary_corpora = _index_boundary_corpora(args.boundary_corpus)
        publication = validate_publication(
            args.training_dirs,
            args.artifact_digest,
            args.finished_block,
            calibration_manifest=args.calibration_manifest,
            weight_soup_checkpoints=weight_soup_checkpoints,
            boundary_corpora=boundary_corpora,
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
