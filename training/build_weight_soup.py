#!/usr/bin/env python3
"""Build one deterministic Qwen3 weight soup from completed merged checkpoints.

Only model-weight deltas are averaged.  Configuration and tokenizer artifacts
are copied byte-for-byte from the exact allowlisted base snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCHEMA = "microtensor.extract-weight-soup.v1"
BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PINNED_BASE_MODEL = f"{BASE_MODEL}@{BASE_REVISION}"
MODEL_FILENAME = "model.safetensors"
INDEX_FILENAME = "model.safetensors.index.json"
METADATA_FILENAME = "soup_metadata.json"
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})
_ELEMENT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}

# Exact files from the local Hugging Face snapshot named in the arena allowlist.
PINNED_BASE_FILES: dict[str, str] = {
    "config.json": "sha256:660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": (
        "sha256:2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
    ),
    "merges.txt": "sha256:8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    MODEL_FILENAME: "sha256:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": (
        "sha256:d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
    ),
    "vocab.json": "sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
PINNED_COPY_FILES = tuple(name for name in PINNED_BASE_FILES if name != MODEL_FILENAME)
PINNED_TIED_ALIASES = (("lm_head.weight", "model.embed_tokens.weight"),)


class SoupValidationError(ValueError):
    """Raised before publication when any model-soup input is untrusted."""


@dataclass(frozen=True)
class BaseAllowlist:
    model: str
    revision: str
    files: Mapping[str, str]
    copy_files: tuple[str, ...]
    tied_aliases: tuple[tuple[str, str], ...]

    @property
    def identity(self) -> str:
        return f"{self.model}@{self.revision}"


PINNED_ALLOWLIST = BaseAllowlist(
    model=BASE_MODEL,
    revision=BASE_REVISION,
    files=PINNED_BASE_FILES,
    copy_files=PINNED_COPY_FILES,
    tied_aliases=PINNED_TIED_ALIASES,
)


@dataclass(frozen=True)
class SourceInput:
    model_dir: Path
    weight: str


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    size: int
    digest: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    @property
    def bytes(self) -> int:
        count = math.prod(self.shape)
        return count * _ELEMENT_BYTES[self.dtype]


@dataclass(frozen=True)
class ValidatedSource:
    supplied_path: str
    model_dir: Path
    model_file: FileSnapshot
    config_file: FileSnapshot
    tokenizer_file: FileSnapshot
    training_metadata_file: FileSnapshot
    training_metadata: dict[str, Any]
    normalized_decimal: str
    normalized_float: float
    normalized_float_hex: str
    has_tied_aliases: bool


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _manifest_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    try:
        details = absolute.lstat()
    except OSError as exc:
        raise SoupValidationError(f"{label} is not a readable directory: {path}") from exc
    if not stat.S_ISDIR(details.st_mode):
        raise SoupValidationError(f"{label} must be a non-symlink directory: {path}")
    return absolute


def _snapshot(path: Path, label: str, *, maximum: int | None = None) -> FileSnapshot:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SoupValidationError(f"{label} is not a readable regular file: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise SoupValidationError(f"{label} must be a non-symlink regular file: {path}")
    if maximum is not None and before.st_size > maximum:
        raise SoupValidationError(f"{label} exceeds the {maximum}-byte limit: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    digest = hashlib.sha256()
    count = 0
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise SoupValidationError(f"{label} changed before hashing: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(4 * 1024 * 1024):
                count += len(chunk)
                if maximum is not None and count > maximum:
                    raise SoupValidationError(f"{label} exceeds the {maximum}-byte limit: {path}")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise SoupValidationError(f"{label} could not be hashed: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        opened.st_dev != after.st_dev
        or opened.st_ino != after.st_ino
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
        or count != after.st_size
    ):
        raise SoupValidationError(f"{label} changed while hashing: {path}")
    return FileSnapshot(
        path=path,
        size=count,
        digest="sha256:" + digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
    )


def _assert_unchanged(snapshot: FileSnapshot, label: str) -> None:
    repeated = _snapshot(snapshot.path, label)
    if repeated != snapshot:
        raise SoupValidationError(f"{label} changed after validation: {snapshot.path}")


def _read_json(snapshot: FileSnapshot, label: str) -> dict[str, Any]:
    if snapshot.size > MAX_JSON_BYTES:
        raise SoupValidationError(f"{label} exceeds the {MAX_JSON_BYTES}-byte limit")
    try:
        parsed = json.loads(
            snapshot.path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SoupValidationError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SoupValidationError(f"{label} must contain a JSON object")
    return parsed


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def normalize_weights(sources: Sequence[SourceInput]) -> tuple[tuple[str, float, str], ...]:
    if len(sources) < 2:
        raise SoupValidationError("a weight soup requires at least two explicitly ordered sources")
    values: list[Decimal] = []
    for index, source in enumerate(sources, start=1):
        try:
            value = Decimal(source.weight)
        except (InvalidOperation, ValueError) as exc:
            raise SoupValidationError(f"source {index} weight is not a decimal number") from exc
        if not value.is_finite() or value < 0:
            raise SoupValidationError(f"source {index} weight must be finite and nonnegative")
        values.append(value)
    total = sum(values, Decimal(0))
    if total <= 0:
        raise SoupValidationError("at least one source weight must be positive")
    with localcontext() as context:
        context.prec = 80
        normalized = [value / total for value in values]
        last_positive = max(index for index, value in enumerate(normalized) if value > 0)
        normalized[last_positive] += Decimal(1) - sum(normalized, Decimal(0))

    floats = [float(value) for value in normalized]
    if not all(math.isfinite(value) and value >= 0 for value in floats):
        raise SoupValidationError("normalized weights are not finite nonnegative floats")
    last_positive = max(index for index, value in enumerate(floats) if value > 0)
    floats[last_positive] = 1.0 - math.fsum(
        value for index, value in enumerate(floats) if index != last_positive
    )
    if floats[last_positive] < 0 or math.fsum(floats) != 1.0:
        raise SoupValidationError("weights could not be normalized exactly for accumulation")
    return tuple(
        (_canonical_decimal(decimal), numeric, numeric.hex())
        for decimal, numeric in zip(normalized, floats, strict=True)
    )


def _normalize_config(config: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = dict(config)
    value.pop("transformers_version", None)
    legacy_dtype = value.pop("torch_dtype", None)
    current_dtype = value.pop("dtype", None)
    if legacy_dtype is not None and current_dtype is not None and legacy_dtype != current_dtype:
        raise SoupValidationError(f"{label} declares conflicting model dtypes")
    dtype = current_dtype if current_dtype is not None else legacy_dtype
    if dtype != "bfloat16":
        raise SoupValidationError(f"{label} model dtype must be exactly bfloat16")

    rope_scaling = value.pop("rope_scaling", None)
    rope_theta = value.pop("rope_theta", None)
    rope_parameters = value.pop("rope_parameters", None)
    if rope_scaling is not None:
        raise SoupValidationError(f"{label} uses unsupported rope scaling")
    if rope_parameters is None:
        rope_parameters = {"rope_theta": rope_theta, "rope_type": "default"}
    if not isinstance(rope_parameters, dict) or set(rope_parameters) != {
        "rope_theta",
        "rope_type",
    }:
        raise SoupValidationError(f"{label} has unsupported rope parameters")
    if rope_parameters.get("rope_type") != "default":
        raise SoupValidationError(f"{label} rope type must be default")

    layers = value.get("num_hidden_layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise SoupValidationError(f"{label} has an invalid layer count")
    layer_types = value.pop("layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * layers
    if layer_types != ["full_attention"] * layers:
        raise SoupValidationError(f"{label} layer types are not the pinned full-attention layout")

    value["dtype"] = dtype
    value["rope_parameters"] = rope_parameters
    value["layer_types"] = layer_types
    value.setdefault("pad_token_id", None)
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SoupValidationError(f"{label} must be sha256: followed by 64 lowercase hex")
    return value


def _validate_training_metadata(metadata: Mapping[str, Any], allowlist: BaseAllowlist) -> None:
    if metadata.get("base_model") != allowlist.identity:
        raise SoupValidationError("source training metadata is not bound to the allowlisted base")
    training_input = metadata.get("training_input")
    if not isinstance(training_input, dict):
        raise SoupValidationError("source training metadata has no training_input object")
    kind = training_input.get("kind")
    _require_digest(training_input.get("weights_digest"), "training input weights_digest")
    _require_digest(training_input.get("tokenizer_digest"), "training input tokenizer_digest")
    if kind == "huggingface_snapshot":
        if training_input.get("revision") != allowlist.revision:
            raise SoupValidationError("source training input has the wrong base revision")
        if training_input["weights_digest"] != allowlist.files[MODEL_FILENAME]:
            raise SoupValidationError("source training input has the wrong base weight identity")
        if training_input["tokenizer_digest"] != allowlist.files["tokenizer.json"]:
            raise SoupValidationError("source training input has the wrong base tokenizer identity")
    elif kind == "derived_model":
        _require_digest(
            training_input.get("parent_metadata_digest"),
            "training input parent_metadata_digest",
        )
    else:
        raise SoupValidationError("source training input kind must be a snapshot or derived model")
    finished = metadata.get("finished_at_unix")
    updates = metadata.get("updates")
    elapsed = metadata.get("elapsed_s")
    if isinstance(finished, bool) or not isinstance(finished, int) or finished <= 0:
        raise SoupValidationError("source training metadata is not marked finished")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise SoupValidationError("source training metadata has no completed updates")
    if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
        raise SoupValidationError("source training metadata has invalid elapsed_s")
    try:
        elapsed_number = float(elapsed)
    except OverflowError as exc:
        raise SoupValidationError("source training metadata has invalid elapsed_s") from exc
    if not math.isfinite(elapsed_number) or elapsed_number < 0:
        raise SoupValidationError("source training metadata has invalid elapsed_s")


def _inspect_safetensors(path: Path, label: str) -> tuple[dict[str, TensorSpec], dict[str, str]]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if metadata != {"format": "pt"}:
                raise SoupValidationError(f"{label} safetensors metadata must be exactly format=pt")
            specs = {
                name: TensorSpec(
                    tuple(int(size) for size in handle.get_slice(name).get_shape()),
                    str(handle.get_slice(name).get_dtype()),
                )
                for name in handle.keys()  # noqa: SIM118 - safe_open is not an iterable mapping
            }
    except SoupValidationError:
        raise
    except Exception as exc:
        raise SoupValidationError(f"{label} is not a readable safetensors model: {exc}") from exc
    if not specs:
        raise SoupValidationError(f"{label} contains no tensors")
    for name, spec in specs.items():
        if not name or spec.dtype not in _FLOAT_DTYPES:
            raise SoupValidationError(f"{label} tensor {name!r} has unsupported dtype {spec.dtype}")
        if not spec.shape or any(size <= 0 for size in spec.shape):
            raise SoupValidationError(f"{label} tensor {name!r} has an invalid shape")
    return specs, metadata


def _schema_records(specs: Mapping[str, TensorSpec]) -> list[dict[str, Any]]:
    return [
        {"dtype": specs[name].dtype, "name": name, "shape": list(specs[name].shape)}
        for name in sorted(specs)
    ]


def _chunk_rows(spec: TensorSpec, chunk_bytes: int) -> int:
    row_elements = math.prod(spec.shape[1:])
    row_bytes = max(1, row_elements * _ELEMENT_BYTES[spec.dtype])
    return max(1, chunk_bytes // row_bytes)


def _tensor_chunk(handle: Any, name: str, start: int, stop: int) -> torch.Tensor:
    return handle.get_slice(name)[start:stop]


def _validate_alias_values(
    path: Path,
    aliases: Sequence[tuple[str, str]],
    specs: Mapping[str, TensorSpec],
    chunk_bytes: int,
    label: str,
) -> None:
    if not aliases:
        return
    with safe_open(path, framework="pt", device="cpu") as handle:
        for alias, target in aliases:
            if specs[alias] != specs[target]:
                raise SoupValidationError(
                    f"{label} tied tensors {alias} and {target} differ in schema"
                )
            rows = _chunk_rows(specs[target], chunk_bytes)
            for start in range(0, specs[target].shape[0], rows):
                stop = min(specs[target].shape[0], start + rows)
                if not torch.equal(
                    _tensor_chunk(handle, alias, start, stop),
                    _tensor_chunk(handle, target, start, stop),
                ):
                    raise SoupValidationError(
                        f"{label} tied tensors {alias} and {target} differ in value"
                    )


def _write_new(path: Path, payload: bytes, mode: int = 0o644) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_tensor(
    name: str,
    spec: TensorSpec,
    base_handle: Any,
    source_handles: Sequence[Any],
    weights: Sequence[float],
    chunk_bytes: int,
) -> torch.Tensor:
    result = torch.empty(spec.shape, dtype=_tensor_chunk(base_handle, name, 0, 1).dtype)
    rows = _chunk_rows(spec, chunk_bytes)
    for start in range(0, spec.shape[0], rows):
        stop = min(spec.shape[0], start + rows)
        base = _tensor_chunk(base_handle, name, start, stop)
        if not bool(torch.isfinite(base).all()):
            raise SoupValidationError(f"base tensor {name} contains a non-finite value")
        base_float = base.float()
        accumulator = base_float.clone()
        for source, weight in zip(source_handles, weights, strict=True):
            candidate = _tensor_chunk(source, name, start, stop)
            if not bool(torch.isfinite(candidate).all()):
                raise SoupValidationError(f"source tensor {name} contains a non-finite value")
            if weight:
                accumulator.add_(candidate.float() - base_float, alpha=weight)
        if not bool(torch.isfinite(accumulator).all()):
            raise SoupValidationError(f"weighted tensor {name} is not finite")
        converted = accumulator.to(dtype=result.dtype)
        if not bool(torch.isfinite(converted).all()):
            raise SoupValidationError(f"weighted tensor {name} overflows its source dtype")
        result[start:stop].copy_(converted)
    return result.contiguous()


def _source_metadata_record(source: ValidatedSource, position: int) -> dict[str, Any]:
    metadata = source.training_metadata
    return {
        "position": position,
        "supplied_model_dir": source.supplied_path,
        "normalized_weight_decimal": source.normalized_decimal,
        "normalized_weight_float_hex": source.normalized_float_hex,
        "files": {
            "config.json": {
                "bytes": source.config_file.size,
                "sha256": source.config_file.digest,
            },
            MODEL_FILENAME: {
                "bytes": source.model_file.size,
                "sha256": source.model_file.digest,
            },
            "tokenizer.json": {
                "bytes": source.tokenizer_file.size,
                "sha256": source.tokenizer_file.digest,
            },
        },
        "parent_training_metadata": {
            "bytes": source.training_metadata_file.size,
            "sha256": source.training_metadata_file.digest,
            "training_input": metadata["training_input"],
        },
    }


def build_weight_soup(
    *,
    base_dir: Path,
    sources: Sequence[SourceInput],
    output_dir: Path,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    allowlist: BaseAllowlist = PINNED_ALLOWLIST,
) -> dict[str, Any]:
    """Validate all inputs, build in a sibling staging directory, then publish once."""

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise SoupValidationError("chunk_bytes must be a positive integer")
    normalized = normalize_weights(sources)
    base = _directory(base_dir, "allowlisted base")
    output = output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise SoupValidationError("output directory already exists; refusing to overwrite it")
    for candidate in (base, *(source.model_dir.absolute() for source in sources)):
        if output == candidate or output.is_relative_to(candidate):
            raise SoupValidationError(
                "output directory must not be inside an input model directory"
            )

    base_files: dict[str, FileSnapshot] = {}
    for name, expected in allowlist.files.items():
        snapshot = _snapshot(base / name, f"base {name}")
        if snapshot.digest != expected:
            raise SoupValidationError(f"base {name} does not match the exact allowlisted snapshot")
        base_files[name] = snapshot
    base_config = _read_json(base_files["config.json"], "base config")
    normalized_base_config = _normalize_config(base_config, "base config")
    base_specs_full, _metadata = _inspect_safetensors(base_files[MODEL_FILENAME].path, "base model")

    aliases = tuple(allowlist.tied_aliases)
    for alias, target in aliases:
        if alias not in base_specs_full or target not in base_specs_full:
            raise SoupValidationError("the allowlisted tied-tensor schema is incomplete")
    _validate_alias_values(
        base_files[MODEL_FILENAME].path,
        aliases,
        base_specs_full,
        chunk_bytes,
        "base model",
    )
    alias_names = {alias for alias, _target in aliases}
    base_specs = {name: spec for name, spec in base_specs_full.items() if name not in alias_names}
    schema_records = _schema_records(base_specs)
    tensor_schema_digest = _manifest_digest(schema_records)
    architecture_digest = _manifest_digest(normalized_base_config)

    validated: list[ValidatedSource] = []
    seen_directories: set[Path] = set()
    for position, (source, weights) in enumerate(zip(sources, normalized, strict=True), start=1):
        supplied = str(source.model_dir)
        model_dir = _directory(source.model_dir, f"source {position}")
        if model_dir == base or model_dir in seen_directories:
            raise SoupValidationError("base and source model directories must all be distinct")
        seen_directories.add(model_dir)
        model_file = _snapshot(model_dir / MODEL_FILENAME, f"source {position} model")
        config_file = _snapshot(
            model_dir / "config.json", f"source {position} config", maximum=MAX_JSON_BYTES
        )
        tokenizer_file = _snapshot(model_dir / "tokenizer.json", f"source {position} tokenizer")
        parent_file = _snapshot(
            model_dir.parent / "training_metadata.json",
            f"source {position} parent training metadata",
            maximum=MAX_JSON_BYTES,
        )
        config = _read_json(config_file, f"source {position} config")
        if _normalize_config(config, f"source {position} config") != normalized_base_config:
            raise SoupValidationError(f"source {position} architecture does not match the base")
        training_metadata = _read_json(parent_file, f"source {position} parent training metadata")
        _validate_training_metadata(training_metadata, allowlist)

        specs, _source_safetensors_metadata = _inspect_safetensors(
            model_file.path, f"source {position} model"
        )
        source_keys = set(specs)
        effective_keys = set(base_specs)
        full_keys = set(base_specs_full)
        if source_keys == effective_keys:
            has_aliases = False
        elif source_keys == full_keys:
            has_aliases = True
            _validate_alias_values(
                model_file.path, aliases, specs, chunk_bytes, f"source {position} model"
            )
        else:
            missing = sorted(effective_keys - source_keys)
            extra = sorted(source_keys - full_keys)
            raise SoupValidationError(
                f"source {position} tensor keys mismatch; missing={missing!r}, extra={extra!r}"
            )
        for name, expected in base_specs.items():
            actual = specs.get(name)
            if actual != expected:
                raise SoupValidationError(
                    f"source {position} tensor {name!r} schema mismatch: "
                    f"expected {expected}, got {actual}"
                )
        decimal_weight, float_weight, float_hex = weights
        validated.append(
            ValidatedSource(
                supplied_path=supplied,
                model_dir=model_dir,
                model_file=model_file,
                config_file=config_file,
                tokenizer_file=tokenizer_file,
                training_metadata_file=parent_file,
                training_metadata=training_metadata,
                normalized_decimal=decimal_weight,
                normalized_float=float_weight,
                normalized_float_hex=float_hex,
                has_tied_aliases=has_aliases,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    parent = _directory(output.parent, "output parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    published = False
    try:
        weight_map: dict[str, str] = {}
        output_records: list[dict[str, Any]] = []
        tensor_names = sorted(base_specs)
        width = max(5, len(str(len(tensor_names))))
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ExitStack() as stack:
                base_handle = stack.enter_context(
                    safe_open(base_files[MODEL_FILENAME].path, framework="pt", device="cpu")
                )
                source_handles = [
                    stack.enter_context(
                        safe_open(source.model_file.path, framework="pt", device="cpu")
                    )
                    for source in validated
                ]
                weights = [source.normalized_float for source in validated]
                for index, name in enumerate(tensor_names, start=1):
                    shard_name = (
                        f"model-{index:0{width}d}-of-{len(tensor_names):0{width}d}.safetensors"
                    )
                    shard = staging / shard_name
                    with torch.inference_mode():
                        tensor = _output_tensor(
                            name,
                            base_specs[name],
                            base_handle,
                            source_handles,
                            weights,
                            chunk_bytes,
                        )
                    save_file({name: tensor}, shard, metadata={"format": "pt"})
                    _sync_file(shard)
                    with safe_open(shard, framework="pt", device="cpu") as check:
                        if list(check.keys()) != [name] or check.metadata() != {"format": "pt"}:
                            raise SoupValidationError(
                                f"written shard failed verification: {shard_name}"
                            )
                        actual = TensorSpec(
                            tuple(int(size) for size in check.get_slice(name).get_shape()),
                            str(check.get_slice(name).get_dtype()),
                        )
                        if actual != base_specs[name]:
                            raise SoupValidationError(
                                f"written tensor {name!r} changed schema unexpectedly"
                            )
                    snapshot = _snapshot(shard, f"output shard {shard_name}")
                    output_records.append(
                        {"bytes": snapshot.size, "path": shard_name, "sha256": snapshot.digest}
                    )
                    weight_map[name] = shard_name
                    del tensor
        finally:
            torch.set_num_threads(old_threads)

        index_payload = _json_bytes(
            {
                "metadata": {"total_size": sum(spec.bytes for spec in base_specs.values())},
                "weight_map": weight_map,
            }
        )
        _write_new(staging / INDEX_FILENAME, index_payload)
        index_snapshot = _snapshot(staging / INDEX_FILENAME, "output index")
        output_records.append(
            {
                "bytes": index_snapshot.size,
                "path": INDEX_FILENAME,
                "sha256": index_snapshot.digest,
            }
        )

        for name in allowlist.copy_files:
            destination = staging / name
            _write_new(destination, base_files[name].path.read_bytes())
            copied = _snapshot(destination, f"copied base {name}")
            if copied.digest != base_files[name].digest or copied.size != base_files[name].size:
                raise SoupValidationError(f"copied base file changed unexpectedly: {name}")
            output_records.append({"bytes": copied.size, "path": name, "sha256": copied.digest})

        output_records.sort(key=lambda record: str(record["path"]))
        metadata: dict[str, Any] = {
            "schema": SCHEMA,
            "base": {
                "identity": allowlist.identity,
                "files": {
                    name: {"bytes": base_files[name].size, "sha256": base_files[name].digest}
                    for name in sorted(base_files)
                },
                "architecture_sha256": architecture_digest,
                "tensor_schema_sha256": tensor_schema_digest,
                "tied_aliases_omitted": [
                    {"alias": alias, "target": target} for alias, target in aliases
                ],
            },
            "algorithm": {
                "formula": "base + sum(normalized_weight_i * (source_i - base))",
                "accumulation_dtype": "float32",
                "chunk_bytes": chunk_bytes,
                "source_order_is_significant": True,
                "torch_num_threads": 1,
                "output_dtype": "exactly_match_each_base_tensor",
                "sharding": "one_tensor_per_safetensors_file",
            },
            "sources": [
                _source_metadata_record(source, position)
                for position, source in enumerate(validated, start=1)
            ],
            "runtime": {
                "python": platform.python_version(),
                "safetensors": safetensors.__version__,
                "torch": torch.__version__,
            },
            "output": {
                "files": output_records,
                "manifest_sha256": _manifest_digest(output_records),
                "model_tensor_bytes": sum(spec.bytes for spec in base_specs.values()),
                "tensor_count": len(base_specs),
            },
        }

        all_inputs = [*base_files.values()]
        for source in validated:
            all_inputs.extend(
                (
                    source.model_file,
                    source.config_file,
                    source.tokenizer_file,
                    source.training_metadata_file,
                )
            )
        for snapshot in all_inputs:
            _assert_unchanged(snapshot, "model-soup input")

        _write_new(staging / METADATA_FILENAME, _json_bytes(metadata))
        _sync_directory(staging)
        if output.exists() or output.is_symlink():
            raise SoupValidationError(
                "output appeared during construction; refusing to overwrite it"
            )
        os.rename(staging, output)
        published = True
        _sync_directory(parent)
        return metadata
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("MERGED_MODEL_DIR", "WEIGHT"),
        required=True,
        help="repeat in the exact accumulation order; weights are normalized by the tool",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_inputs = [SourceInput(Path(path), weight) for path, weight in args.source]
    try:
        metadata = build_weight_soup(
            base_dir=args.base,
            sources=source_inputs,
            output_dir=args.output,
            chunk_bytes=args.chunk_bytes,
        )
    except (OSError, RuntimeError, SoupValidationError) as exc:
        raise SystemExit(f"refusing weight soup: {exc}") from exc
    metadata_path = args.output.absolute() / METADATA_FILENAME
    metadata_snapshot = _snapshot(metadata_path, "written soup metadata")
    print(
        json.dumps(
            {
                "metadata_sha256": metadata_snapshot.digest,
                "output": str(args.output),
                "output_manifest_sha256": metadata["output"]["manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
