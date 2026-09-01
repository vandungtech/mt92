#!/usr/bin/env python3
"""Fail-closed validation and W&B publication for the code-track candidate.

Publication is deliberately the last operation.  Every local input is replayed
and validated twice: once when a :class:`Publication` is prepared and again
immediately before ``wandb.init``.  This module never serializes or prints credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import evaluate_code
    from training import evaluate_code_gguf as gguf
    from training import historical_code_candidate as historical
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import evaluate_code  # type: ignore[no-redef]
    import evaluate_code_gguf as gguf  # type: ignore[no-redef]
    import historical_code_candidate as historical  # type: ignore[no-redef]


ENTITY: Final[str] = "microtensor"
PROJECT: Final[str] = "training-runs"
HOTKEY: Final[str] = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"
TRACK: Final[str] = "code"
HARDWARE_CLASS: Final[str] = "mt-3g"
BASE_MODEL: Final[str] = candidate.QWEN3_BASE_MODEL
CORPUS_VERSION: Final[str] = historical.CORPUS_VERSION
TRAINING_SCHEMA: Final[str] = "microtensor.code.training.v5"
CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v1"
CALIBRATION_SCHEMA: Final[str] = "microtensor.calibration-execution-receipt.v1"
DETERMINISM_REPLAY_SCHEMA: Final[str] = "microtensor.code.gguf-determinism-replay.v1"
OBSOLETE_CALIBRATED_CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v2"
CALIBRATED_CONVERSION_SCHEMA: Final[str] = "microtensor.code.gguf-conversion.v3"
IMATRIX_CALIBRATION_SCHEMA: Final[str] = "microtensor.code.imatrix-calibration.v2"
RUNTIME_LIBRARY_SCHEMA: Final[str] = "microtensor.code.llama-cpp-runtime-libraries.v1"
LLAMA_CPP_ROOT: Final[Path] = Path("/tmp/llama.cpp")  # noqa: S108 - protocol-pinned root
LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT: Final[tuple[tuple[str, str, int, str], ...]] = (
    (
        "build/bin/libggml-base.so.0",
        "build/bin/libggml-base.so.0.22.0",
        939_528,
        "sha256:73106e6e34d4f6dcd9f4ffca57f132070c48e93c9dd409df2333eea1b7c4806f",
    ),
    (
        "build/bin/libggml-cpu.so.0",
        "build/bin/libggml-cpu.so.0.22.0",
        1_143_800,
        "sha256:196c9f2c112e51f17b79375e80b102c72bc872b3e4fc17295ab1564533812807",
    ),
    (
        "build/bin/libggml.so.0",
        "build/bin/libggml.so.0.22.0",
        56_376,
        "sha256:eaea0b8964d5acee7ce26bb4895137df772ad24387f45fbb51158495f596fa29",
    ),
    (
        "build/bin/libllama-common.so.0",
        "build/bin/libllama-common.so.0.3.0",
        5_909_928,
        "sha256:ce12dd60805687b1dfbd574033d5163089979a1b2b556cb8a6c65b85af7048f5",
    ),
    (
        "build/bin/libllama-quantize-impl.so",
        "build/bin/libllama-quantize-impl.so",
        89_792,
        "sha256:d79664774038f0f42eccee8b1d5772b2bbe0f7840181401c73e30f2986113cbb",
    ),
    (
        "build/bin/libllama.so.0",
        "build/bin/libllama.so.0.3.0",
        4_692_320,
        "sha256:8c809635a537f48c79bb058034ae9eb3c437693bc8b4fc13e0035c0be7bad8ed",
    ),
)
LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT: Final[tuple[tuple[str, int, str], ...]] = (
    (
        "build/bin/llama-imatrix",
        343_128,
        "sha256:3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35",
    ),
    (
        "build/bin/llama-quantize",
        17_928,
        "sha256:e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580",
    ),
)
LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT: Final[tuple[tuple[str, str], ...]] = (
    ("build/bin/libggml-base.so", "libggml-base.so.0"),
    ("build/bin/libggml-base.so.0", "libggml-base.so.0.22.0"),
    ("build/bin/libggml-cpu.so", "libggml-cpu.so.0"),
    ("build/bin/libggml-cpu.so.0", "libggml-cpu.so.0.22.0"),
    ("build/bin/libggml.so", "libggml.so.0"),
    ("build/bin/libggml.so.0", "libggml.so.0.22.0"),
    ("build/bin/libllama-common.so", "libllama-common.so.0"),
    ("build/bin/libllama-common.so.0", "libllama-common.so.0.3.0"),
    ("build/bin/libllama.so", "libllama.so.0"),
    ("build/bin/libllama.so.0", "libllama.so.0.3.0"),
)
LLAMA_CPP_REVISION: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
NO_CALIBRATION_CLAIM: Final[str] = (
    "none: no independently validated calibration execution receipt was supplied"
)
VALIDATED_CALIBRATION_CLAIM: Final[str] = (
    "exact supplied calibration execution receipt validated and byte-bound"
)
MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
MAX_METRICS_RECORDS: Final[int] = 1_000_000
WANDB_PAYLOAD_SCHEMA: Final[str] = "microtensor.code.wandb-provenance.v1"
WANDB_SDK_VERSION: Final[str] = "0.29.0"
WANDB_ERROR_REPORTING_VARIABLE: Final[str] = "WANDB_ERROR_REPORTING"
WANDB_REDACTED_HOST: Final[str] = "microtensor-provenance"
WANDB_REDACTED_PROGRAM: Final[str] = "microtensor-provenance"
WANDB_REDACTED_PROGRAM_ABSPATH: Final[str] = "/microtensor-provenance"
WANDB_RUN_ID_PREFIX: Final[str] = "mt92-"
WANDB_RUN_ID_HEX_LENGTH: Final[int] = 40
WANDB_PENDING_NAME_PREFIX: Final[str] = "pending-"
WANDB_READBACK_TIMEOUT_SECONDS: Final[int] = 30
WANDB_AUTOMATIC_CONFIG_FIELDS: Final[tuple[str, ...]] = ("_wandb", "wandb_version")
WANDB_AUTOMATIC_HISTORY_FIELDS: Final[tuple[str, ...]] = (
    "_runtime",
    "_step",
    "_timestamp",
)
WANDB_AUTOMATIC_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "_runtime",
    "_step",
    "_timestamp",
    "_wandb",
)
APPLICATION_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "mt_artifact_digest",
    "mt_finished_at",
    "mt_training_records",
    "mt_training_schema",
    "mt_conversion_receipt_digest",
)
_DIGEST: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CodeProvenanceError(ValueError):
    """Raised when provenance validation or guarded publication refuses."""


class WandbPublicationState(str, Enum):
    """Terminal knowledge about a deterministic W&B publication attempt."""

    PENDING_FAILED = "pending_failed"
    COMMITTED = "committed"
    OUTCOME_UNCERTAIN = "outcome_uncertain"


class WandbPublicationStateError(CodeProvenanceError):
    """A terminal, explicitly non-retryable W&B publication result."""

    retry_forbidden = True

    def __init__(self, message: str, *, state: WandbPublicationState, run_id: str) -> None:
        super().__init__(message)
        self.state = state
        self.run_id = run_id


class WandbPendingFailedError(WandbPublicationStateError):
    """The deterministic run ID was consumed but the hotkey marker was not committed."""

    def __init__(self, message: str, *, run_id: str) -> None:
        super().__init__(message, state=WandbPublicationState.PENDING_FAILED, run_id=run_id)


class WandbPostCommitError(WandbPublicationStateError):
    """The final-name commit succeeded, but a later verification or cleanup failed."""

    def __init__(self, message: str, *, run_id: str) -> None:
        super().__init__(message, state=WandbPublicationState.COMMITTED, run_id=run_id)


class WandbOutcomeUncertainError(WandbPublicationStateError):
    """The final-name commit may have happened and requires read-only reconciliation."""

    def __init__(self, message: str, *, run_id: str) -> None:
        super().__init__(message, state=WandbPublicationState.OUTCOME_UNCERTAIN, run_id=run_id)


class PayloadExportOutcomeUncertainError(CodeProvenanceError):
    """A payload export could not prove final state after its no-replace link."""

    retry_forbidden = True
    state = "outcome_uncertain"


class PayloadExportPostCommitError(CodeProvenanceError):
    """The payload is durably published but staging cleanup failed."""

    retry_forbidden = True
    state = "committed"


@dataclass(frozen=True)
class PublicationRequest:
    training_run: Path
    training_dataset: Path
    source_corpus: Path
    base: Path
    artifact: Path
    artifact_digest: str
    load_spec: Path
    conversion_receipt: Path
    finished_block: int
    hf_diagnostic: Path | None = None
    gguf_diagnostic: Path | None = None
    calibration_receipt: Path | None = None
    llama_cpp: Path | None = None
    calibration_current_dataset: Path | None = None
    calibration_current_source_corpus: Path | None = None


@dataclass(frozen=True)
class Publication:
    request: PublicationRequest
    training_metadata: dict[str, Any]
    training_lineage: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]
    artifact: dict[str, Any]
    load_manifest: dict[str, Any]
    conversion: dict[str, Any]
    conversion_receipt_identity: dict[str, Any]
    hf_diagnostic: dict[str, Any] | None
    gguf_diagnostic: dict[str, Any] | None
    calibration: dict[str, Any] | None
    finished_block: int


@dataclass(frozen=True)
class WandbPublicationOutcome:
    """A remotely verified final-name commitment."""

    state: WandbPublicationState
    run_id: str
    resolution: str
    retry_forbidden: bool = True


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CodeProvenanceError(f"{label} must be lowercase sha256:<64 hex>")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CodeProvenanceError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodeProvenanceError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodeProvenanceError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CodeProvenanceError(f"{label} must be a finite number >= {minimum}")
    return result


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected or any(not isinstance(key, str) for key in value):
        raise CodeProvenanceError(
            f"{label} fields changed: expected {sorted(expected)}, got {sorted(value)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CodeProvenanceError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CodeProvenanceError(f"{label} must be an array")
    return value


def _reject_constant(value: str) -> None:
    raise CodeProvenanceError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodeProvenanceError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _read_regular(path: Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> bytes:
    """Read one stable regular file without following a final-component symlink."""

    path = Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise CodeProvenanceError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise CodeProvenanceError(f"{label} must be a regular non-symlink file: {path}")
    if before.st_size > maximum:
        raise CodeProvenanceError(f"{label} exceeds the {maximum}-byte limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CodeProvenanceError(f"{label} changed type while opening")
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(opened, field) for field in stable_fields):
            raise CodeProvenanceError(f"{label} changed while being opened")
        if opened.st_size > maximum:
            raise CodeProvenanceError(f"{label} exceeds the {maximum}-byte limit")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CodeProvenanceError(f"{label} ended while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CodeProvenanceError(f"{label} grew while being read")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise CodeProvenanceError(f"{label} could not be read safely: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
        raise CodeProvenanceError(f"{label} changed while being read")
    return b"".join(chunks)


def _json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeProvenanceError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    return dict(_mapping(value, label)), raw


def _file_identity(path: Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    raw = _read_regular(path, label, maximum=maximum)
    return {"bytes": len(raw), "digest": _digest_bytes(raw)}


def _parse_metrics(
    path: Path, metadata: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    raw = _read_regular(path, "training metrics")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise CodeProvenanceError(f"training metrics line {number} is empty")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodeProvenanceError(f"training metrics line {number} is invalid: {exc}") from exc
        record = dict(_mapping(value, f"training metrics line {number}"))
        records.append(record)
        if len(records) > MAX_METRICS_RECORDS:
            raise CodeProvenanceError("training metrics record limit exceeded")

    updates = _positive_int(metadata.get("updates"), "training update count")
    if len(records) != updates + 1:
        raise CodeProvenanceError("v5 final metrics must contain every update and one export event")
    update_fields = frozenset(
        {
            "step",
            "epoch",
            "loss",
            "loss_mass",
            "supervised_tokens",
            "terminal_eos_tokens",
            "terminal_eos_loss_weight",
            "microbatches",
            "gradient_norm",
            "learning_rate",
            "elapsed_s",
        }
    )
    previous_elapsed = -1.0
    for index, record in enumerate(records[:-1], 1):
        _exact_keys(record, update_fields, f"training metric {index}")
        if record["step"] != index:
            raise CodeProvenanceError("training metric steps are not exact and contiguous")
        _positive_int(record["epoch"], f"training metric {index} epoch")
        for key in ("loss", "loss_mass", "gradient_norm", "learning_rate", "elapsed_s"):
            minimum = 0.0 if key != "learning_rate" else float.fromhex("0x0.0000000000001p-1022")
            _finite(record[key], f"training metric {index} {key}", minimum=minimum)
        for key in ("supervised_tokens", "terminal_eos_tokens", "microbatches"):
            _positive_int(record[key], f"training metric {index} {key}")
        _finite(record["terminal_eos_loss_weight"], "terminal EOS weight", minimum=1.0)
        elapsed = float(record["elapsed_s"])
        if elapsed < previous_elapsed:
            raise CodeProvenanceError("training metric elapsed time moved backwards")
        previous_elapsed = elapsed

    export = records[-1]
    expected_export = {"event": "export_selection", **dict(metadata["selection"])}
    if export != expected_export:
        raise CodeProvenanceError("terminal export metric does not match the v5 selection receipt")
    identity = {"bytes": len(raw), "digest": _digest_bytes(raw)}
    if identity["digest"] != metadata.get("metrics_digest"):
        raise CodeProvenanceError("training metrics digest changed")
    return tuple(records), identity


def _validate_load_manifest(payload: Any, artifact: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(_mapping(payload, "load manifest"))
    _exact_keys(
        manifest,
        frozenset(
            {"format", "quantization", "entrypoint", "max_input", "preprocessing", "base_model"}
        ),
        "load manifest",
    )
    quantization = manifest.get("quantization")
    if (
        manifest.get("format") != "gguf"
        or quantization not in gguf.SUPPORTED_QUANTIZATIONS
        or manifest.get("entrypoint") != artifact["entrypoint"]["path"]
        or manifest.get("preprocessing") != {"tokenizer": "tokenizer.json"}
        or manifest.get("base_model") != BASE_MODEL
    ):
        raise CodeProvenanceError("load manifest does not match the pinned code GGUF contract")
    maximum = _mapping(manifest.get("max_input"), "load manifest max_input")
    _exact_keys(maximum, frozenset({"tokens"}), "load manifest max_input")
    tokens = _positive_int(maximum.get("tokens"), "load manifest max_input tokens")
    if not gguf.MIN_CONTEXT_TOKENS <= tokens <= gguf.MAX_CONTEXT_TOKENS:
        raise CodeProvenanceError("load manifest max_input tokens are outside the audited range")
    return manifest


def _validate_generic_conversion(
    receipt: Mapping[str, Any],
    *,
    training_lineage: Mapping[str, Any],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
    calibration_digest: str | None,
) -> dict[str, Any]:
    _exact_keys(
        receipt,
        frozenset(
            {
                "schema",
                "status",
                "track",
                "hardware_class",
                "base_model",
                "llama_cpp_revision",
                "source",
                "conversion",
                "artifact",
                "load_manifest",
                "calibration_receipt_digest",
            }
        ),
        "conversion receipt",
    )
    required = {
        "schema": CONVERSION_SCHEMA,
        "status": "complete",
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise CodeProvenanceError("conversion receipt identity changed")
    source = _mapping(receipt["source"], "conversion source")
    _exact_keys(
        source, frozenset({"training_metadata_digest", "merged_tree_digest"}), "conversion source"
    )
    expected_source = {
        "training_metadata_digest": training_lineage["receipt"]["digest"],
        "merged_tree_digest": training_lineage["run"]["merged"]["digest"],
    }
    if dict(source) != expected_source:
        raise CodeProvenanceError("conversion receipt crosses training lineages")
    declared_artifact = _mapping(receipt["artifact"], "conversion artifact")
    expected_artifact = {
        "tree_digest": artifact["tree_digest"],
        "entrypoint_digest": artifact["entrypoint"]["digest"],
        "entrypoint_bytes": artifact["entrypoint"]["bytes"],
        "quantization": load_manifest["quantization"],
    }
    if dict(declared_artifact) != expected_artifact:
        raise CodeProvenanceError("conversion receipt does not bind the final artifact")
    if receipt["load_manifest"] != load_manifest:
        raise CodeProvenanceError("conversion receipt load manifest changed")
    if receipt["calibration_receipt_digest"] != calibration_digest:
        raise CodeProvenanceError("conversion receipt calibration binding changed")

    conversion = _mapping(receipt["conversion"], "conversion execution")
    _exact_keys(
        conversion,
        frozenset({"converter_digest", "quantizer_digest", "commands"}),
        "conversion execution",
    )
    _require_digest(conversion["converter_digest"], "converter digest")
    _require_digest(conversion["quantizer_digest"], "quantizer digest")
    commands = _sequence(conversion["commands"], "conversion commands")
    if len(commands) != 2:
        raise CodeProvenanceError("conversion receipt must contain convert_f16 then quantize")
    for index, (value, expected_name) in enumerate(
        zip(commands, ("convert_f16", "quantize"), strict=True), 1
    ):
        command = _mapping(value, f"conversion command {index}")
        _exact_keys(
            command,
            frozenset({"name", "argv", "returncode", "started_at_unix_ns", "finished_at_unix_ns"}),
            f"conversion command {index}",
        )
        if command["name"] != expected_name or command["returncode"] != 0:
            raise CodeProvenanceError("conversion command order or exit status changed")
        argv = _sequence(command["argv"], f"conversion command {index} argv")
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise CodeProvenanceError("conversion command argv must contain non-empty strings")
        started = _positive_int(command["started_at_unix_ns"], "conversion command start")
        finished = _positive_int(command["finished_at_unix_ns"], "conversion command finish")
        if finished < started:
            raise CodeProvenanceError("conversion command finished before it started")
    return dict(receipt)


def _calibration_snapshot_files(value: Any, label: str) -> list[dict[str, Any]]:
    tree = _mapping(value, label)
    files = _sequence(tree.get("files"), f"{label} files")
    result: list[dict[str, Any]] = []
    root = Path(str(tree.get("path", "")))
    for item in files:
        entry = _mapping(item, f"{label} file")
        try:
            relative = Path(str(entry["path"])).relative_to(root).as_posix()
        except (KeyError, ValueError) as exc:
            raise CodeProvenanceError(f"{label} file escapes its root") from exc
        result.append(
            {"path": relative, "bytes": entry.get("bytes"), "digest": entry.get("sha256")}
        )
    return result


def _validate_calibration(
    path: Path,
    *,
    training_lineage: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, raw = _json_file(path, "calibration execution receipt")
    expected_top = frozenset(
        {"schema", "llama_cpp", "execution", "tools", "inputs", "outputs", "post_run_integrity"}
    )
    _exact_keys(receipt, expected_top, "calibration execution receipt")
    if receipt.get("schema") != CALIBRATION_SCHEMA:
        raise CodeProvenanceError("calibration receipt schema changed")
    checkout = _mapping(receipt["llama_cpp"], "calibration llama.cpp")
    if (
        checkout.get("revision_before") != LLAMA_CPP_REVISION
        or checkout.get("revision_after") != LLAMA_CPP_REVISION
        or checkout.get("clean_before") is not True
        or checkout.get("clean_after") is not True
    ):
        raise CodeProvenanceError("calibration receipt does not bind the clean pinned llama.cpp")
    inputs = _mapping(receipt["inputs"], "calibration inputs")
    source = _mapping(inputs.get("source_model"), "calibration source model")
    if source.get("before") != source.get("after"):
        raise CodeProvenanceError("calibration source model changed during execution")
    if source.get("training_metadata_sha256") != training_lineage["receipt"]["digest"]:
        raise CodeProvenanceError("calibration receipt crosses training lineages")
    if (
        _calibration_snapshot_files(source.get("before"), "calibration source model")
        != training_lineage["run"]["merged"]["files"]
    ):
        raise CodeProvenanceError("calibration source inventory differs from the merged model")
    outputs = _mapping(receipt["outputs"], "calibration outputs")
    quantized = _mapping(outputs.get("quantized_artifact"), "calibration quantized artifact")
    if (
        quantized.get("sha256") != artifact["entrypoint"]["digest"]
        or quantized.get("bytes") != artifact["entrypoint"]["bytes"]
    ):
        raise CodeProvenanceError("calibration output does not match the GGUF entrypoint")
    commands = _sequence(
        _mapping(receipt["execution"], "calibration execution").get("commands"),
        "calibration commands",
    )
    if [
        (
            _mapping(item, "calibration command").get("name"),
            _mapping(item, "calibration command").get("returncode"),
        )
        for item in commands
    ] != [("convert_f16", 0), ("build_imatrix", 0), ("quantize_q4_k_m", 0)]:
        raise CodeProvenanceError("calibration command sequence or status changed")
    integrity = _mapping(receipt["post_run_integrity"], "calibration post-run integrity")
    required_true = {
        "confirmed",
        "source_model_unchanged",
        "inputs_unchanged",
        "tools_unchanged",
        "llama_cpp_revision_and_cleanliness_rechecked",
        "final_outputs_match_held_inodes",
        "final_outputs_rechecked_immediately_before_commit",
    }
    if any(integrity.get(key) is not True for key in required_true):
        raise CodeProvenanceError("calibration receipt lacks required post-run integrity")
    return receipt, {"bytes": len(raw), "digest": _digest_bytes(raw)}


def _calibrated_converter_module() -> Any:
    """Load calibrated-conversion replay code without creating an import cycle."""

    try:
        module = importlib.import_module("training.convert_code_gguf")
    except (ImportError, RuntimeError) as exc:
        raise CodeProvenanceError(f"calibrated converter validation is unavailable: {exc}") from exc
    expected = {
        "CALIBRATED_CONVERSION_SCHEMA": CALIBRATED_CONVERSION_SCHEMA,
        "CALIBRATION_SCHEMA": IMATRIX_CALIBRATION_SCHEMA,
        "LLAMA_CPP_REVISION": LLAMA_CPP_REVISION,
        "ENTRYPOINT": "model.gguf",
        "LOAD_SPEC_NAME": "load-spec.json",
        "RECEIPT_NAME": "conversion-receipt.json",
        "CALIBRATION_RECEIPT_NAME": "calibration-receipt.json",
        "ARTIFACT_NAME": "artifact",
        "F16_NAME": "model-f16.gguf",
        "CALIBRATION_CORPUS_NAME": "calibration.txt",
        "IMATRIX_NAME": "calibration.imatrix.gguf",
        "CALIBRATION_PROFILE": "code-public-imatrix128-v1",
        "CALIBRATION_SEED": 92,
        "CALIBRATION_CURRENT_ROWS": 78,
        "CALIBRATION_DIAGNOSTIC_ROWS": 16,
        "CALIBRATION_HISTORICAL_ROWS": 434,
        "CALIBRATION_TOTAL_ROWS": 512,
        "CALIBRATION_CHUNKS": 128,
        "CALIBRATION_CONTEXT_TOKENS": 512,
        "CALIBRATION_EOS_TOKEN": "<|im_end|>",
        "CALIBRATION_EOS_TOKEN_ID": 151645,
        "CALIBRATION_RENDER_SCHEMA": "prompt-completion-im-end-utf8-v1",
        "CALIBRATION_SELECTION_ALGORITHM": "sha256-seed-ref-ascending-v1",
        "CALIBRATION_MAX_BYTES": 16 * 1024 * 1024,
        "MAX_CAPTURED_LOG_BYTES": 1 * 1024 * 1024,
        "RUNTIME_LIBRARY_SCHEMA": RUNTIME_LIBRARY_SCHEMA,
        "LLAMA_CPP_ROOT": LLAMA_CPP_ROOT,
        "LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT": LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT,
        "LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT": LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT,
        "LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT": LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT,
    }
    changed = [name for name, value in expected.items() if getattr(module, name, None) != value]
    if changed:
        raise CodeProvenanceError(
            f"calibrated converter contract changed: {', '.join(sorted(changed))}"
        )
    if getattr(module, "provenance", None) is not sys.modules.get(__name__):
        raise CodeProvenanceError("calibrated converter imported a different provenance module")
    return module


def _json_exact(actual: Any, expected: Any, label: str) -> None:
    """Compare JSON structure without Python's ``1 == 1.0`` coercion."""

    try:
        actual_raw = json.dumps(
            actual,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_raw = json.dumps(
            expected,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodeProvenanceError(f"{label} is not canonical JSON: {exc}") from exc
    if actual_raw != expected_raw:
        raise CodeProvenanceError(f"{label} changed")


def _strict_calibrated_bundle(request: PublicationRequest, module: Any) -> Path:
    if request.calibration_receipt is None:
        raise CodeProvenanceError("conversion-v3 requires --calibration-receipt")
    root = request.conversion_receipt.parent
    if root.is_symlink() or not root.is_dir():
        raise CodeProvenanceError("calibrated bundle root must be a regular non-symlink directory")
    try:
        resolved_root = root.resolve(strict=True)
        expected_paths = {
            "artifact": resolved_root / module.ARTIFACT_NAME,
            "load specification": resolved_root / module.LOAD_SPEC_NAME,
            "calibration receipt": resolved_root / module.CALIBRATION_RECEIPT_NAME,
            "conversion receipt": resolved_root / module.RECEIPT_NAME,
        }
        supplied_paths = {
            "artifact": request.artifact,
            "load specification": request.load_spec,
            "calibration receipt": request.calibration_receipt,
            "conversion receipt": request.conversion_receipt,
        }
        for label, supplied in supplied_paths.items():
            if supplied.is_symlink() or supplied.resolve(strict=True) != expected_paths[label]:
                raise CodeProvenanceError(
                    f"conversion-v3 {label} is not the exact four-file bundle member"
                )
    except CodeProvenanceError:
        raise
    except OSError as exc:
        raise CodeProvenanceError(f"calibrated bundle path validation failed: {exc}") from exc

    files: set[str] = set()
    try:
        for path in resolved_root.rglob("*"):
            relative = path.relative_to(resolved_root).as_posix()
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode):
                raise CodeProvenanceError(f"calibrated bundle contains a symlink: {relative}")
            if stat.S_ISDIR(status.st_mode):
                continue
            if not stat.S_ISREG(status.st_mode):
                raise CodeProvenanceError(f"calibrated bundle contains a special file: {relative}")
            files.add(relative)
    except CodeProvenanceError:
        raise
    except OSError as exc:
        raise CodeProvenanceError(f"calibrated bundle inventory failed: {exc}") from exc
    expected_files = {
        f"{module.ARTIFACT_NAME}/{module.ENTRYPOINT}",
        module.LOAD_SPEC_NAME,
        module.CALIBRATION_RECEIPT_NAME,
        module.RECEIPT_NAME,
    }
    if files != expected_files:
        raise CodeProvenanceError(
            "conversion-v3 bundle must contain exactly model, load, calibration, and conversion"
        )
    return resolved_root


def _forbid_raw_calibration_fields(value: Any, label: str = "calibration receipt") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CodeProvenanceError(f"{label} contains a non-string field")
            if key.casefold() in {"prompt", "completion", "gold"}:
                raise CodeProvenanceError(f"{label} serializes raw calibration text")
            _forbid_raw_calibration_fields(item, label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _forbid_raw_calibration_fields(item, label)


def _validated_runtime_mode(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-7]{4}", value) is None:
        raise CodeProvenanceError(f"{label} mode is not four octal digits")
    if int(value, 8) & 0o022:
        raise CodeProvenanceError(f"{label} is group/world writable")
    return value


def _validate_runtime_libraries(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    closure = dict(_mapping(value, "llama.cpp runtime library closure"))
    _exact_keys(
        closure,
        frozenset(
            {
                "schema",
                "root",
                "directories",
                "build_bin_namespace",
                "symlinks",
                "executables",
                "libraries",
            }
        ),
        "llama.cpp runtime library closure",
    )
    if closure.get("schema") != RUNTIME_LIBRARY_SCHEMA:
        raise CodeProvenanceError("llama.cpp runtime library schema changed")
    if closure.get("root") != str(LLAMA_CPP_ROOT):
        raise CodeProvenanceError("llama.cpp runtime library root changed")
    directories = list(_sequence(closure.get("directories"), "runtime library directories"))
    if len(directories) != 3:
        raise CodeProvenanceError("runtime library directory closure changed")
    for value, expected_path in zip(directories, (".", "build", "build/bin"), strict=True):
        directory = _mapping(value, f"runtime directory {expected_path}")
        _exact_keys(directory, frozenset({"path", "mode"}), f"runtime directory {expected_path}")
        if directory.get("path") != expected_path:
            raise CodeProvenanceError("runtime library directory order or path changed")
        _validated_runtime_mode(directory.get("mode"), f"runtime directory {expected_path}")

    namespace_paths: set[str] = set()
    for relative, _expected_bytes, _expected_digest in LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT:
        namespace_paths.add(relative)
    for (
        loader_path,
        target_path,
        _expected_bytes,
        _expected_digest,
    ) in LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT:
        namespace_paths.update((loader_path, target_path))
    for relative, target in LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT:
        namespace_paths.add(relative)
        namespace_paths.add(f"build/bin/{target}")
    expected_namespace = sorted(namespace_paths)
    namespace = list(_sequence(closure.get("build_bin_namespace"), "runtime build/bin namespace"))
    if any(not isinstance(entry, str) for entry in namespace):
        raise CodeProvenanceError("runtime build/bin namespace contains a non-string path")
    if namespace != expected_namespace:
        raise CodeProvenanceError("runtime build/bin namespace changed")

    symlinks = list(_sequence(closure.get("symlinks"), "runtime symlink closure"))
    if len(symlinks) != len(LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT):
        raise CodeProvenanceError("runtime symlink closure changed")
    for value, (expected_path, expected_target) in zip(
        symlinks,
        LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT,
        strict=True,
    ):
        link = _mapping(value, f"runtime symlink {expected_path}")
        _exact_keys(
            link,
            frozenset({"path", "target"}),
            f"runtime symlink {expected_path}",
        )
        if link.get("path") != expected_path or link.get("target") != expected_target:
            raise CodeProvenanceError(f"runtime symlink edge changed: {expected_path}")

    executables = list(_sequence(closure.get("executables"), "runtime executable closure"))
    if len(executables) != len(LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT):
        raise CodeProvenanceError("runtime executable closure changed")
    for value, contract in zip(
        executables,
        LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT,
        strict=True,
    ):
        executable_path, expected_bytes, expected_digest = contract
        executable = _mapping(value, f"runtime executable {executable_path}")
        _exact_keys(
            executable,
            frozenset({"path", "bytes", "digest", "mode"}),
            f"runtime executable {executable_path}",
        )
        if (
            executable.get("path") != executable_path
            or _positive_int(executable.get("bytes"), f"runtime executable {executable_path} bytes")
            != expected_bytes
            or _require_digest(
                executable.get("digest"), f"runtime executable {executable_path} digest"
            )
            != expected_digest
        ):
            raise CodeProvenanceError(f"runtime executable identity changed: {executable_path}")
        mode = _validated_runtime_mode(
            executable.get("mode"), f"runtime executable {executable_path}"
        )
        if int(mode, 8) & 0o111 == 0:
            raise CodeProvenanceError(f"runtime executable is not executable: {executable_path}")

    libraries = list(_sequence(closure.get("libraries"), "runtime libraries"))
    if len(libraries) != len(LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT):
        raise CodeProvenanceError("runtime library closure does not contain exactly six libraries")
    for value, contract in zip(libraries, LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT, strict=True):
        loader_path, target_path, expected_bytes, expected_digest = contract
        library = _mapping(value, f"runtime library {loader_path}")
        _exact_keys(
            library,
            frozenset({"loader_path", "target_path", "bytes", "digest", "mode"}),
            f"runtime library {loader_path}",
        )
        if (
            library.get("loader_path") != loader_path
            or library.get("target_path") != target_path
            or _positive_int(library.get("bytes"), f"runtime library {loader_path} bytes")
            != expected_bytes
            or _require_digest(library.get("digest"), f"runtime library {loader_path} digest")
            != expected_digest
        ):
            raise CodeProvenanceError(f"runtime library identity changed: {loader_path}")
        _validated_runtime_mode(library.get("mode"), f"runtime library {loader_path}")
    if expected is not None:
        _json_exact(closure, dict(expected), "runtime library closure binding")
    return closure


def _rendered_calibration_identity(rows: Sequence[Any], module: Any) -> dict[str, Any]:
    if len(rows) != module.CALIBRATION_TOTAL_ROWS:
        raise CodeProvenanceError("calibration replay did not select exactly 78+434 rows")
    hasher = hashlib.sha256()
    total = 0
    for index, value in enumerate(rows, 1):
        row = _mapping(value, f"calibration replay row {index}")
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, str) or not prompt:
            raise CodeProvenanceError(f"calibration replay row {index} prompt changed")
        if not isinstance(completion, str) or not completion:
            raise CodeProvenanceError(f"calibration replay row {index} completion changed")
        raw = (prompt + completion + module.CALIBRATION_EOS_TOKEN + "\n").encode("utf-8")
        total += len(raw)
        if total > module.CALIBRATION_MAX_BYTES:
            raise CodeProvenanceError("rendered calibration corpus exceeds its reviewed ceiling")
        hasher.update(raw)
    if total < 1:
        raise CodeProvenanceError("rendered calibration corpus is empty")
    return {
        "name": module.CALIBRATION_CORPUS_NAME,
        "bytes": total,
        "digest": "sha256:" + hasher.hexdigest(),
    }


def _validate_calibrated_environment(
    value: Any,
    expected: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = dict(_mapping(value, "calibrated command environment"))
    fixed = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": ".microtensor-empty-pycache",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    _exact_keys(environment, frozenset({"PATH", *fixed}), "calibrated command environment")
    path = environment.get("PATH")
    if not isinstance(path, str) or not path:
        raise CodeProvenanceError("calibrated command PATH is empty")
    if any(environment.get(name) != expected_value for name, expected_value in fixed.items()):
        raise CodeProvenanceError("calibrated command isolation environment changed")
    if any(not isinstance(item, str) for item in environment.values()):
        raise CodeProvenanceError("calibrated command environment contains a non-string value")
    if expected is not None:
        _json_exact(environment, dict(expected), "calibrated command environment")
    return environment  # type: ignore[return-value]


def _validate_command_stream(value: Any, label: str, maximum: int) -> None:
    stream = _mapping(value, label)
    _exact_keys(
        stream,
        frozenset({"bytes", "captured_bytes", "captured_digest", "digest", "truncated"}),
        label,
    )
    total = _nonnegative_int(stream.get("bytes"), f"{label} bytes")
    captured = _nonnegative_int(stream.get("captured_bytes"), f"{label} captured bytes")
    if captured != min(total, maximum):
        raise CodeProvenanceError(f"{label} captured byte count changed")
    digest = _require_digest(stream.get("digest"), f"{label} digest")
    captured_digest = _require_digest(stream.get("captured_digest"), f"{label} captured digest")
    truncated = stream.get("truncated")
    if not isinstance(truncated, bool) or truncated is not (total > captured):
        raise CodeProvenanceError(f"{label} truncation declaration changed")
    if captured == total and captured_digest != digest:
        raise CodeProvenanceError(f"{label} complete and captured digests differ")
    if captured == 0 and captured_digest != _digest_bytes(b""):
        raise CodeProvenanceError(f"{label} empty captured digest changed")


def _validate_calibrated_command(
    value: Any,
    *,
    name: str,
    argv: Sequence[str],
    cwd_role: str,
    common_environment: Mapping[str, str] | None,
    minimum_start: int,
    maximum_log_bytes: int,
) -> tuple[dict[str, str], int]:
    command = _mapping(value, f"{name} command")
    _exact_keys(
        command,
        frozenset(
            {
                "name",
                "argv",
                "cwd_role",
                "environment",
                "returncode",
                "started_at_unix_ns",
                "finished_at_unix_ns",
                "stdout",
                "stderr",
            }
        ),
        f"{name} command",
    )
    exact_argv = _sequence(command.get("argv"), f"{name} command argv")
    if command.get("name") != name or list(exact_argv) != list(argv):
        raise CodeProvenanceError(f"{name} command argv changed")
    if any(not isinstance(item, str) or not item for item in exact_argv):
        raise CodeProvenanceError(f"{name} command argv contains a non-string or empty value")
    if command.get("cwd_role") != cwd_role:
        raise CodeProvenanceError(f"{name} command cwd role changed")
    returncode = command.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
        raise CodeProvenanceError(f"{name} command return code changed")
    started = _positive_int(command.get("started_at_unix_ns"), f"{name} command start")
    finished = _positive_int(command.get("finished_at_unix_ns"), f"{name} command finish")
    if started < minimum_start or finished < started:
        raise CodeProvenanceError(f"{name} command timing or order changed")
    environment = _validate_calibrated_environment(command.get("environment"), common_environment)
    _validate_command_stream(command.get("stdout"), f"{name} stdout", maximum_log_bytes)
    _validate_command_stream(command.get("stderr"), f"{name} stderr", maximum_log_bytes)
    return environment, finished


def _validate_calibrated_commands(
    value: Any,
    *,
    expected: Sequence[tuple[str, Sequence[str]]],
    cwd_role: str,
    common_environment: Mapping[str, str] | None,
    minimum_start: int,
    maximum_log_bytes: int,
) -> tuple[list[Any], dict[str, str], int]:
    commands = list(_sequence(value, f"{cwd_role} commands"))
    if len(commands) != len(expected):
        raise CodeProvenanceError(f"{cwd_role} must contain exactly three commands")
    environment = dict(common_environment) if common_environment is not None else None
    finished = minimum_start
    for command, (name, argv) in zip(commands, expected, strict=True):
        environment, finished = _validate_calibrated_command(
            command,
            name=name,
            argv=argv,
            cwd_role=cwd_role,
            common_environment=environment,
            minimum_start=finished,
            maximum_log_bytes=maximum_log_bytes,
        )
    if environment is None:
        raise CodeProvenanceError(f"{cwd_role} commands are empty")
    return commands, environment, finished


def _validate_determinism_replay(
    value: Any,
    *,
    expected_commands: Sequence[tuple[str, Sequence[str]]],
    common_environment: Mapping[str, str],
    minimum_start: int,
    maximum_log_bytes: int,
    f16_digest: str,
    imatrix_digest: str,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    replay = dict(_mapping(value, "determinism replay"))
    _exact_keys(
        replay,
        frozenset(
            {
                "schema",
                "commands",
                "f16_digest",
                "imatrix_digest",
                "entrypoint_digest",
                "entrypoint_bytes",
                "artifact_tree_digest",
                "matches_primary",
            }
        ),
        "determinism replay",
    )
    if (
        replay.get("schema") != DETERMINISM_REPLAY_SCHEMA
        or replay.get("matches_primary") is not True
    ):
        raise CodeProvenanceError("determinism replay identity or success claim changed")
    entrypoint = _mapping(artifact.get("entrypoint"), "artifact entrypoint")
    expected_identity = {
        "f16_digest": f16_digest,
        "imatrix_digest": imatrix_digest,
        "entrypoint_digest": entrypoint["digest"],
        "entrypoint_bytes": entrypoint["bytes"],
        "artifact_tree_digest": artifact["tree_digest"],
    }
    for field, expected_value in expected_identity.items():
        if field.endswith("digest"):
            _require_digest(replay.get(field), f"determinism replay {field}")
        elif field == "entrypoint_bytes":
            _positive_int(replay.get(field), "determinism replay entrypoint bytes")
        if replay.get(field) != expected_value:
            raise CodeProvenanceError(f"determinism replay {field} changed")
    _validate_calibrated_commands(
        replay.get("commands"),
        expected=expected_commands,
        cwd_role="determinism_replay",
        common_environment=common_environment,
        minimum_start=minimum_start,
        maximum_log_bytes=maximum_log_bytes,
    )
    return replay


def _validate_calibrated_v3(
    request: PublicationRequest,
    *,
    receipt: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
    calibration_identity: Mapping[str, Any],
    training_lineage: Mapping[str, Any],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _calibrated_converter_module()
    if (
        request.llama_cpp is None
        or request.calibration_current_dataset is None
        or request.calibration_current_source_corpus is None
    ):
        raise CodeProvenanceError(
            "conversion-v3 requires --llama-cpp, --calibration-current-dataset, "
            "and --calibration-current-source-corpus"
        )
    if load_manifest.get("quantization") != "Q4_K_M":
        raise CodeProvenanceError("conversion-v3 requires an exact Q4_K_M load manifest")
    try:
        llama_cpp_root = request.llama_cpp.resolve(strict=True)
    except OSError as exc:
        raise CodeProvenanceError(f"pinned llama.cpp root is unavailable: {exc}") from exc
    if llama_cpp_root != LLAMA_CPP_ROOT:
        raise CodeProvenanceError(f"llama.cpp checkout must resolve exactly to {LLAMA_CPP_ROOT}")

    bundle = _strict_calibrated_bundle(request, module)
    conversion_request = module.ConversionRequest(
        training_run=request.training_run,
        training_dataset=request.training_dataset,
        source_corpus=request.source_corpus,
        base=request.base,
        llama_cpp=llama_cpp_root,
        converter=llama_cpp_root / "convert_hf_to_gguf.py",
        quantizer=llama_cpp_root / "build" / "bin" / "llama-quantize",
        output_bundle=bundle,
        quantization="Q4_K_M",
        max_input_tokens=load_manifest["max_input"]["tokens"],
        calibration_profile=module.CALIBRATION_PROFILE,
        calibration_current_dataset=request.calibration_current_dataset,
        calibration_current_source_corpus=request.calibration_current_source_corpus,
        imatrix_tool=llama_cpp_root / "build" / "bin" / "llama-imatrix",
    )
    try:
        toolchain = dict(_mapping(module._toolchain_identity(conversion_request), "toolchain"))
        rows, material_value = module._load_calibration_material(conversion_request)
        material = dict(_mapping(material_value, "calibration replay material"))
        model_metadata = dict(
            _mapping(
                module._validate_calibrated_model_metadata(request.artifact / module.ENTRYPOINT),
                "calibrated model metadata",
            )
        )
    except CodeProvenanceError:
        raise
    except Exception as exc:
        raise CodeProvenanceError(f"conversion-v3 local replay failed: {exc}") from exc

    _exact_keys(
        material,
        frozenset({"profile", "source", "selection"}),
        "calibration replay material",
    )
    if material.get("profile") != module.CALIBRATION_PROFILE:
        raise CodeProvenanceError("calibration replay profile changed")
    source = dict(_mapping(material.get("source"), "calibration replay source"))
    selection = dict(_mapping(material.get("selection"), "calibration replay selection"))
    _exact_keys(
        selection,
        frozenset(
            {
                "algorithm",
                "seed",
                "current_rows",
                "current_refs_digest",
                "diagnostic_rows_excluded",
                "diagnostic_refs_digest",
                "historical_pool_rows",
                "historical_selected_rows",
                "historical_selected_refs_digest",
                "total_rows",
            }
        ),
        "calibration replay selection",
    )
    expected_counts = {
        "seed": module.CALIBRATION_SEED,
        "current_rows": module.CALIBRATION_CURRENT_ROWS,
        "diagnostic_rows_excluded": module.CALIBRATION_DIAGNOSTIC_ROWS,
        "historical_pool_rows": 8_000,
        "historical_selected_rows": module.CALIBRATION_HISTORICAL_ROWS,
        "total_rows": module.CALIBRATION_TOTAL_ROWS,
    }
    if selection.get("algorithm") != module.CALIBRATION_SELECTION_ALGORITHM:
        raise CodeProvenanceError("calibration selection algorithm changed")
    for field, expected_value in expected_counts.items():
        if _positive_int(selection.get(field), f"calibration selection {field}") != expected_value:
            raise CodeProvenanceError("calibration replay is not the exact 78+434 selection")
    for field in (
        "current_refs_digest",
        "diagnostic_refs_digest",
        "historical_selected_refs_digest",
    ):
        _require_digest(selection.get(field), f"calibration selection {field}")
    corpus_identity = _rendered_calibration_identity(rows, module)

    calibration = dict(calibration_receipt)
    _exact_keys(
        calibration,
        frozenset(
            {
                "schema",
                "status",
                "profile",
                "track",
                "hardware_class",
                "base_model",
                "llama_cpp_revision",
                "source",
                "selection",
                "rendering",
                "toolchain",
                "commands",
                "determinism_replay",
                "intermediate",
                "artifact",
                "load_manifest",
            }
        ),
        "imatrix calibration receipt",
    )
    expected_identity = {
        "schema": IMATRIX_CALIBRATION_SCHEMA,
        "status": "complete",
        "profile": module.CALIBRATION_PROFILE,
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
    }
    if any(calibration.get(key) != value for key, value in expected_identity.items()):
        raise CodeProvenanceError("imatrix calibration receipt identity changed")
    _json_exact(calibration.get("source"), source, "calibration source identity")
    _json_exact(calibration.get("selection"), selection, "calibration selection")

    rendering = _mapping(calibration.get("rendering"), "calibration rendering")
    _exact_keys(
        rendering,
        frozenset(
            {"schema", "encoding", "expression", "eos_token", "eos_token_id", "rows", "corpus"}
        ),
        "calibration rendering",
    )
    expected_rendering = {
        "schema": module.CALIBRATION_RENDER_SCHEMA,
        "encoding": "UTF-8",
        "expression": "prompt + completion + <|im_end|> + LF",
        "eos_token": module.CALIBRATION_EOS_TOKEN,
        "eos_token_id": module.CALIBRATION_EOS_TOKEN_ID,
        "rows": module.CALIBRATION_TOTAL_ROWS,
        "corpus": corpus_identity,
    }
    _json_exact(rendering, expected_rendering, "calibration rendering")

    _exact_keys(
        toolchain,
        frozenset({"root", "revision", "converter", "imatrix", "quantizer", "runtime_libraries"}),
        "replayed llama.cpp toolchain",
    )
    if toolchain.get("root") != str(llama_cpp_root):
        raise CodeProvenanceError("replayed llama.cpp root changed")
    if toolchain.get("revision") != LLAMA_CPP_REVISION:
        raise CodeProvenanceError("replayed llama.cpp revision changed")
    declared_tools = _mapping(calibration.get("toolchain"), "calibration toolchain")
    _exact_keys(
        declared_tools,
        frozenset({"converter_digest", "imatrix_digest", "quantizer_digest", "runtime_libraries"}),
        "calibration toolchain",
    )
    tool_digests: dict[str, str] = {}
    actual_runtime_libraries = _validate_runtime_libraries(toolchain.get("runtime_libraries"))
    _validate_runtime_libraries(
        declared_tools.get("runtime_libraries"),
        expected=actual_runtime_libraries,
    )
    for receipt_name, tool_name in (
        ("converter_digest", "converter"),
        ("imatrix_digest", "imatrix"),
        ("quantizer_digest", "quantizer"),
    ):
        actual = _require_digest(
            _mapping(toolchain.get(tool_name), f"{tool_name} identity").get("digest"),
            f"replayed {tool_name} digest",
        )
        if declared_tools.get(receipt_name) != actual:
            raise CodeProvenanceError(f"calibration {tool_name} digest changed")
        tool_digests[receipt_name] = actual

    try:
        merged_root = (request.training_run.resolve(strict=True) / "merged").resolve(strict=True)
    except OSError as exc:
        raise CodeProvenanceError(f"validated merged model path is unavailable: {exc}") from exc
    converter_path = str(_mapping(toolchain["converter"], "converter identity")["path"])
    imatrix_path = str(_mapping(toolchain["imatrix"], "imatrix identity")["path"])
    quantizer_path = str(_mapping(toolchain["quantizer"], "quantizer identity")["path"])
    command_argv = (
        (
            "convert_f16",
            (converter_path, str(merged_root), "--outfile", module.F16_NAME, "--outtype", "f16"),
        ),
        (
            "calibrate_imatrix",
            (
                imatrix_path,
                "--offline",
                "--model",
                module.F16_NAME,
                "--file",
                module.CALIBRATION_CORPUS_NAME,
                "--output",
                module.IMATRIX_NAME,
                "--output-format",
                "gguf",
                "--ctx-size",
                str(module.CALIBRATION_CONTEXT_TOKENS),
                "--chunks",
                str(module.CALIBRATION_CHUNKS),
                "--batch-size",
                "512",
                "--ubatch-size",
                "512",
                "--threads",
                "1",
                "--threads-batch",
                "1",
                "--device",
                "none",
                "--gpu-layers",
                "0",
                "--fit",
                "off",
                "--flash-attn",
                "off",
                "--no-ppl",
                "--parse-special",
                "--output-frequency",
                str(module.CALIBRATION_CHUNKS + 1),
                "--save-frequency",
                "0",
            ),
        ),
        (
            "quantize",
            (
                quantizer_path,
                "--imatrix",
                module.IMATRIX_NAME,
                module.F16_NAME,
                f"{module.ARTIFACT_NAME}/{module.ENTRYPOINT}",
                "Q4_K_M",
                "1",
            ),
        ),
    )
    primary_commands, command_environment, primary_finished = _validate_calibrated_commands(
        calibration.get("commands"),
        expected=command_argv,
        cwd_role="private_staging",
        common_environment=None,
        minimum_start=0,
        maximum_log_bytes=module.MAX_CAPTURED_LOG_BYTES,
    )

    intermediate = _mapping(calibration.get("intermediate"), "calibration intermediate")
    _exact_keys(intermediate, frozenset({"f16", "imatrix"}), "calibration intermediate")
    f16 = _mapping(intermediate.get("f16"), "calibration F16 identity")
    _exact_keys(f16, frozenset({"bytes", "digest", "file_type"}), "calibration F16 identity")
    _positive_int(f16.get("bytes"), "calibration F16 bytes")
    f16_digest = _require_digest(f16.get("digest"), "calibration F16 digest")
    if _positive_int(f16.get("file_type"), "calibration F16 file type") != 1:
        raise CodeProvenanceError("calibration F16 file type changed")
    imatrix = _mapping(intermediate.get("imatrix"), "importance matrix identity")
    _exact_keys(
        imatrix,
        frozenset(
            {
                "bytes",
                "digest",
                "version",
                "tensor_count",
                "entries_count",
                "datasets",
                "chunk_count",
                "chunk_size",
            }
        ),
        "importance matrix identity",
    )
    _positive_int(imatrix.get("bytes"), "importance matrix bytes")
    imatrix_digest = _require_digest(imatrix.get("digest"), "importance matrix digest")
    if _positive_int(imatrix.get("version"), "importance matrix GGUF version") not in {2, 3}:
        raise CodeProvenanceError("importance matrix GGUF version changed")
    _positive_int(imatrix.get("tensor_count"), "importance matrix tensor count")
    entries_count = _positive_int(imatrix.get("entries_count"), "importance matrix entry count")
    chunks = _positive_int(imatrix.get("chunk_count"), "importance matrix chunk count")
    chunk_size = _positive_int(imatrix.get("chunk_size"), "importance matrix chunk size")
    if (
        imatrix.get("datasets") != [module.CALIBRATION_CORPUS_NAME]
        or chunks != module.CALIBRATION_CHUNKS
        or chunk_size != module.CALIBRATION_CONTEXT_TOKENS
    ):
        raise CodeProvenanceError("importance matrix dataset or chunk metadata changed")

    declared_artifact = _mapping(calibration.get("artifact"), "calibration artifact")
    expected_calibration_artifact = {
        "tree_digest": artifact["tree_digest"],
        "entrypoint_digest": artifact["entrypoint"]["digest"],
        "entrypoint_bytes": artifact["entrypoint"]["bytes"],
        "quantization": "Q4_K_M",
        "calibration_metadata": model_metadata,
    }
    _json_exact(declared_artifact, expected_calibration_artifact, "calibration artifact")
    _exact_keys(
        model_metadata,
        frozenset(
            {
                "imatrix_file",
                "imatrix_dataset",
                "imatrix_entries_count",
                "imatrix_chunks_count",
            }
        ),
        "calibrated model metadata",
    )
    model_entries = _positive_int(
        model_metadata.get("imatrix_entries_count"), "model imatrix entry count"
    )
    model_chunks = _positive_int(
        model_metadata.get("imatrix_chunks_count"), "model imatrix chunk count"
    )
    if (
        model_metadata.get("imatrix_file") != module.IMATRIX_NAME
        or model_metadata.get("imatrix_dataset") != module.CALIBRATION_CORPUS_NAME
        or model_entries != entries_count
        or model_chunks != module.CALIBRATION_CHUNKS
    ):
        raise CodeProvenanceError("model and importance-matrix metadata bindings changed")
    _json_exact(calibration.get("load_manifest"), load_manifest, "calibration load manifest")

    calibration_replay = _validate_determinism_replay(
        calibration.get("determinism_replay"),
        expected_commands=command_argv,
        common_environment=command_environment,
        minimum_start=primary_finished,
        maximum_log_bytes=module.MAX_CAPTURED_LOG_BYTES,
        f16_digest=f16_digest,
        imatrix_digest=imatrix_digest,
        artifact=artifact,
    )

    conversion_receipt = dict(receipt)
    _exact_keys(
        conversion_receipt,
        frozenset(
            {
                "schema",
                "status",
                "track",
                "hardware_class",
                "base_model",
                "llama_cpp_revision",
                "source",
                "conversion",
                "artifact",
                "load_manifest",
                "calibration_receipt_digest",
            }
        ),
        "conversion-v3 receipt",
    )
    expected_conversion_identity = {
        "schema": CALIBRATED_CONVERSION_SCHEMA,
        "status": "complete",
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "base_model": BASE_MODEL,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
    }
    if any(
        conversion_receipt.get(key) != value for key, value in expected_conversion_identity.items()
    ):
        raise CodeProvenanceError("conversion-v3 receipt identity changed")
    expected_source = {
        "training_metadata_digest": training_lineage["receipt"]["digest"],
        "merged_tree_digest": training_lineage["run"]["merged"]["digest"],
    }
    _json_exact(conversion_receipt.get("source"), expected_source, "conversion-v3 source")
    expected_conversion_artifact = {
        "tree_digest": artifact["tree_digest"],
        "entrypoint_digest": artifact["entrypoint"]["digest"],
        "entrypoint_bytes": artifact["entrypoint"]["bytes"],
        "quantization": "Q4_K_M",
    }
    _json_exact(
        conversion_receipt.get("artifact"),
        expected_conversion_artifact,
        "conversion-v3 artifact",
    )
    _json_exact(
        conversion_receipt.get("load_manifest"),
        load_manifest,
        "conversion-v3 load manifest",
    )
    calibration_digest = str(calibration_identity["digest"])
    if conversion_receipt.get("calibration_receipt_digest") != calibration_digest:
        raise CodeProvenanceError("conversion-v3 calibration receipt binding changed")
    execution = _mapping(conversion_receipt.get("conversion"), "conversion-v3 execution")
    _exact_keys(
        execution,
        frozenset(
            {
                "converter_digest",
                "imatrix_digest",
                "quantizer_digest",
                "runtime_libraries",
                "commands",
                "determinism_replay",
            }
        ),
        "conversion-v3 execution",
    )
    for field, expected_digest in tool_digests.items():
        if execution.get(field) != expected_digest:
            raise CodeProvenanceError(f"conversion-v3 {field} changed")
    _validate_runtime_libraries(
        execution.get("runtime_libraries"),
        expected=actual_runtime_libraries,
    )
    conversion_commands, _, _ = _validate_calibrated_commands(
        execution.get("commands"),
        expected=command_argv,
        cwd_role="private_staging",
        common_environment=command_environment,
        minimum_start=0,
        maximum_log_bytes=module.MAX_CAPTURED_LOG_BYTES,
    )
    conversion_replay = _validate_determinism_replay(
        execution.get("determinism_replay"),
        expected_commands=command_argv,
        common_environment=command_environment,
        minimum_start=primary_finished,
        maximum_log_bytes=module.MAX_CAPTURED_LOG_BYTES,
        f16_digest=f16_digest,
        imatrix_digest=imatrix_digest,
        artifact=artifact,
    )
    _json_exact(conversion_commands, primary_commands, "cross-receipt primary commands")
    _json_exact(conversion_replay, calibration_replay, "cross-receipt determinism replay")
    _forbid_raw_calibration_fields(calibration)
    return conversion_receipt, {
        "identity": dict(calibration_identity),
        "schema": IMATRIX_CALIBRATION_SCHEMA,
        "claim": VALIDATED_CALIBRATION_CLAIM,
        "receipt": calibration,
    }


def _diagnostic_files(root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise CodeProvenanceError(f"{label} must be a regular non-symlink directory")
    summary, summary_raw = _json_file(root / "summary.json", f"{label} summary")
    results_path = root / "results.jsonl"
    results = _file_identity(results_path, f"{label} results")
    declared = _mapping(summary.get("results"), f"{label} results declaration")
    if (
        declared.get("file") != "results.jsonl"
        or declared.get("bytes") != results["bytes"]
        or declared.get("digest") != results["digest"]
    ):
        raise CodeProvenanceError(f"{label} result bytes changed")
    return summary, {
        "summary": {"bytes": len(summary_raw), "digest": _digest_bytes(summary_raw)},
        "results": results,
    }


def _validate_hf_diagnostic(path: Path, training_lineage: Mapping[str, Any]) -> dict[str, Any]:
    summary, identity = _diagnostic_files(path, "HF diagnostic")
    if (
        summary.get("schema") != evaluate_code.SCHEMA
        or summary.get("status") != "complete"
        or summary.get("track") != TRACK
        or summary.get("hardware_class") != HARDWARE_CLASS
        or summary.get("quality_claim") != evaluate_code.QUALITY_CLAIM
        or summary.get("runtime_claim") != evaluate_code.HF_RUNTIME_CLAIM
        or summary.get("lineage_claim") != evaluate_code.SEPARATE_LINEAGE_CLAIM
    ):
        raise CodeProvenanceError("HF diagnostic identity or non-pass@1 claim changed")
    if summary.get("model") != training_lineage["run"]:
        raise CodeProvenanceError("HF diagnostic crosses training lineages")
    return {**identity, "claim": evaluate_code.QUALITY_CLAIM}


def _validate_gguf_diagnostic(
    path: Path,
    *,
    training_lineage: Mapping[str, Any],
    artifact: Mapping[str, Any],
    load_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    summary, identity = _diagnostic_files(path, "GGUF diagnostic")
    if (
        summary.get("schema") != gguf.SCHEMA
        or summary.get("track") != TRACK
        or summary.get("hardware_class") != HARDWARE_CLASS
        or summary.get("base_model") != BASE_MODEL
        or summary.get("quality_claim") != gguf.QUALITY_CLAIM
        or summary.get("lineage_claim") != gguf.LINEAGE_CLAIM
        or summary.get("artifact") != artifact
        or summary.get("training_lineage") != training_lineage
    ):
        raise CodeProvenanceError("GGUF diagnostic identity, artifact, or lineage changed")
    configuration = _mapping(summary.get("configuration"), "GGUF diagnostic configuration")
    results = _mapping(summary.get("results"), "GGUF diagnostic results")
    if (
        configuration.get("load_manifest") != load_manifest
        or configuration.get("artifact_digest") != artifact["tree_digest"]
        or results.get("quality_score") is not None
        or results.get("execution_pass_at_1") is not None
    ):
        raise CodeProvenanceError("GGUF diagnostic makes an unsupported quality or load claim")
    expected_configuration_digest = candidate.digest_bytes(
        candidate.canonical_json_bytes(configuration)
    )
    if summary.get("configuration_digest") != expected_configuration_digest:
        raise CodeProvenanceError("GGUF diagnostic configuration digest changed")
    return {**identity, "claim": gguf.QUALITY_CLAIM}


def validate_publication(request: PublicationRequest) -> Publication:
    """Validate all required bytes and optional diagnostics without network access."""

    artifact_digest = _require_digest(request.artifact_digest, "artifact digest")
    finished_block = _positive_int(request.finished_block, "finished block")
    try:
        training_lineage, _modules = gguf.load_v5_training_lineage(
            request.training_run,
            request.training_dataset,
            request.source_corpus,
            request.base,
        )
        metadata, _raw_metadata = _json_file(
            request.training_run / "training_metadata.json", "v5 training metadata"
        )
        metadata_identity = {
            "bytes": len(_raw_metadata),
            "digest": _digest_bytes(_raw_metadata),
        }
        lineage_receipt = _mapping(training_lineage.get("receipt"), "training lineage receipt")
        lineage_run = _mapping(training_lineage.get("run"), "training lineage run")
        lineage_run_metadata = _mapping(
            lineage_run.get("training_metadata"), "deep-validated training metadata"
        )
        if dict(lineage_run_metadata) != metadata_identity or any(
            lineage_receipt.get(key) != value for key, value in metadata_identity.items()
        ):
            raise CodeProvenanceError("training metadata bytes differ from the validated lineage")
        if metadata.get("schema") != TRAINING_SCHEMA:
            raise CodeProvenanceError("only the completed historical8000 v5 receipt is publishable")
        metrics, metrics_identity = _parse_metrics(request.training_run / "metrics.jsonl", metadata)
        lineage_run_metrics = _mapping(
            lineage_run.get("metrics"), "deep-validated training metrics"
        )
        if dict(lineage_run_metrics) != metrics_identity:
            raise CodeProvenanceError("training metrics bytes differ from the validated lineage")
        load_payload, _raw_load = _json_file(request.load_spec, "load manifest")
        quantization = load_payload.get("quantization")
        if not isinstance(quantization, str):
            raise CodeProvenanceError("load manifest quantization must be a string")
        artifact = gguf.artifact_identity(
            request.artifact,
            entrypoint=str(load_payload.get("entrypoint", "")),
            expected_digest=artifact_digest,
            quantization=quantization,
        )
        load_manifest = _validate_load_manifest(load_payload, artifact)
    except CodeProvenanceError:
        raise
    except Exception as exc:
        raise CodeProvenanceError(f"training/artifact validation failed: {exc}") from exc

    calibration: dict[str, Any] | None = None
    conversion, conversion_raw = _json_file(request.conversion_receipt, "conversion receipt")
    conversion_schema = conversion.get("schema")
    calibrated_inputs = (
        request.llama_cpp,
        request.calibration_current_dataset,
        request.calibration_current_source_corpus,
    )
    if conversion_schema == CONVERSION_SCHEMA:
        if any(value is not None for value in calibrated_inputs):
            raise CodeProvenanceError("conversion-v3 inputs are forbidden for conversion-v1")
        if load_manifest["quantization"] == "Q4_K_M":
            raise CodeProvenanceError(
                "Q4_K_M publication requires calibrated conversion-v3 and imatrix-v2 receipts"
            )
        calibration_digest: str | None = None
        if request.calibration_receipt is not None:
            calibration_payload, calibration_identity = _validate_calibration(
                request.calibration_receipt,
                training_lineage=training_lineage,
                artifact=artifact,
            )
            calibration = {
                "identity": calibration_identity,
                "schema": calibration_payload["schema"],
                "claim": VALIDATED_CALIBRATION_CLAIM,
            }
            calibration_digest = str(calibration_identity["digest"])
        conversion = _validate_generic_conversion(
            conversion,
            training_lineage=training_lineage,
            artifact=artifact,
            load_manifest=load_manifest,
            calibration_digest=calibration_digest,
        )
    elif conversion_schema == CALIBRATED_CONVERSION_SCHEMA:
        if request.calibration_receipt is None:
            raise CodeProvenanceError("conversion-v3 requires --calibration-receipt")
        calibration_payload, calibration_raw = _json_file(
            request.calibration_receipt,
            "imatrix calibration receipt",
        )
        conversion, calibration = _validate_calibrated_v3(
            request,
            receipt=conversion,
            calibration_receipt=calibration_payload,
            calibration_identity={
                "bytes": len(calibration_raw),
                "digest": _digest_bytes(calibration_raw),
            },
            training_lineage=training_lineage,
            artifact=artifact,
            load_manifest=load_manifest,
        )
    elif conversion_schema == OBSOLETE_CALIBRATED_CONVERSION_SCHEMA:
        raise CodeProvenanceError(
            "conversion-v2 lacks the pinned llama.cpp runtime-library closure"
        )
    else:
        raise CodeProvenanceError("conversion receipt schema is not supported")
    hf_diagnostic = (
        _validate_hf_diagnostic(request.hf_diagnostic, training_lineage)
        if request.hf_diagnostic is not None
        else None
    )
    gguf_diagnostic = (
        _validate_gguf_diagnostic(
            request.gguf_diagnostic,
            training_lineage=training_lineage,
            artifact=artifact,
            load_manifest=load_manifest,
        )
        if request.gguf_diagnostic is not None
        else None
    )
    return Publication(
        request=request,
        training_metadata=metadata,
        training_lineage=training_lineage,
        metrics=metrics,
        artifact=artifact,
        load_manifest=load_manifest,
        conversion=conversion,
        conversion_receipt_identity={
            "bytes": len(conversion_raw),
            "digest": _digest_bytes(conversion_raw),
        },
        hf_diagnostic=hf_diagnostic,
        gguf_diagnostic=gguf_diagnostic,
        calibration=calibration,
        finished_block=finished_block,
    )


def _wandb_config(publication: Publication) -> dict[str, Any]:
    config: dict[str, Any] = {
        "mt_hotkey": HOTKEY,
        "mt_track": TRACK,
        "mt_class": HARDWARE_CLASS,
        "mt_base_model": BASE_MODEL,
        "mt_corpus_version": CORPUS_VERSION,
        "mt_artifact_digest": publication.artifact["tree_digest"],
        "mt_finished_at": publication.finished_block,
        "mt_training_schema": TRAINING_SCHEMA,
        "mt_training_lineage": publication.training_lineage,
        "mt_training_metadata": publication.training_metadata,
        "mt_conversion_receipt": {
            "identity": publication.conversion_receipt_identity,
            "receipt": publication.conversion,
        },
        "mt_artifact": publication.artifact,
        "mt_load_manifest": publication.load_manifest,
        "mt_quality_claim": historical.FINAL_ALL_PUBLIC_QUALITY_CLAIM,
        "mt_calibration_claim": (
            publication.calibration["claim"]
            if publication.calibration is not None
            else NO_CALIBRATION_CLAIM
        ),
    }
    if publication.hf_diagnostic is not None:
        config["mt_hf_diagnostic"] = publication.hf_diagnostic
    if publication.gguf_diagnostic is not None:
        config["mt_gguf_diagnostic"] = publication.gguf_diagnostic
    if publication.calibration is not None:
        config["mt_calibration_receipt"] = publication.calibration
    return config


def _wandb_summary(publication: Publication) -> dict[str, Any]:
    return {
        "mt_artifact_digest": publication.artifact["tree_digest"],
        "mt_finished_at": publication.finished_block,
        "mt_training_records": len(publication.metrics),
        "mt_training_schema": TRAINING_SCHEMA,
        "mt_conversion_receipt_digest": publication.conversion_receipt_identity["digest"],
    }


def _wandb_log_records(publication: Publication) -> list[dict[str, Any]]:
    return [
        {"step": global_step, "payload": dict(metric)}
        for global_step, metric in enumerate(publication.metrics, 1)
    ]


def _canonical_json_value_bytes(value: Any, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodeProvenanceError(f"{label} is not canonical JSON: {exc}") from exc


def _wandb_run_id_from_envelope(
    controls: Any,
    config: Any,
    logs: Any,
    summary: Any,
) -> str:
    material = {
        "schema": WANDB_PAYLOAD_SCHEMA,
        "destination": {"entity": ENTITY, "project": PROJECT, "name": HOTKEY},
        "controls": controls,
        "config": config,
        "logs": logs,
        "summary": summary,
    }
    digest = hashlib.sha256(
        _canonical_json_value_bytes(material, "W&B run identity material")
    ).hexdigest()
    return WANDB_RUN_ID_PREFIX + digest[:WANDB_RUN_ID_HEX_LENGTH]


def _wandb_run_id(publication: Publication) -> str:
    return _wandb_run_id_from_envelope(
        _wandb_controls(),
        _wandb_config(publication),
        _wandb_log_records(publication),
        _wandb_summary(publication),
    )


def _wandb_pending_name(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.startswith(WANDB_RUN_ID_PREFIX):
        raise CodeProvenanceError("pending W&B display name requires the deterministic run ID")
    return WANDB_PENDING_NAME_PREFIX + run_id


def _wandb_environment_policy() -> dict[str, Any]:
    return {
        "forbidden_ambient_prefix": "WANDB_",
        "allowed_ambient_variables": ["WANDB_API_KEY"],
        "required_nonempty_ambient_variables": ["WANDB_API_KEY"],
        "forced_process_variables": {WANDB_ERROR_REPORTING_VARIABLE: "false"},
        "credential_values_in_envelope": False,
    }


def _wandb_identity_defaults() -> dict[str, Any]:
    return {
        "entity": None,
        "fork_from": None,
        "project": None,
        "resume_from": None,
        "run_group": None,
        "run_id": None,
        "run_job_type": None,
        "run_name": None,
        "run_notes": None,
        "run_tags": None,
        "sweep_id": None,
    }


def _wandb_isolation_policy() -> dict[str, Any]:
    return {
        "root": "fresh-private-mode-0700",
        "workspace_settings": "absent-inside-private-root",
        "system_settings": "absent-inside-private-root",
        "credential_source": "WANDB_API_KEY-only",
        "error_reporting": "disabled-before-sdk-import-and-through-teardown",
        "preimported_real_sdk": "forbidden",
        "effective_identity_defaults": _wandb_identity_defaults(),
        "sdk_setup": "explicit-with-private-settings-before-init",
        "sdk_teardown": "after-returned-run-finish",
        "environment_cleanup": "restore-pre-import-WANDB-variables",
        "cleanup": "private-root-after-sdk-teardown",
    }


def _wandb_service_metadata_policy() -> dict[str, Any]:
    return {
        "authorization_scope": (
            "exact-application-envelope-plus-disclosed-bounded-service-metadata"
        ),
        "wire_transcript_authorized_exactly": False,
        "application_summary_fields": list(APPLICATION_SUMMARY_FIELDS),
        "reserved_config": {
            "_wandb": {
                "producer": "W&B SDK and service",
                "namespace": "reserved-and-not-application-config",
                "generated_content": [
                    "metric definitions and summary directives",
                    "run start-time and SDK bookkeeping",
                    "telemetry feature, import, environment, platform, Python, SDK and core fields",
                ],
            },
            "wandb_version": {
                "producer": "W&B SDK or service",
                "presence": "optional legacy reserved version field",
            },
        },
        "automatic_history_fields": list(WANDB_AUTOMATIC_HISTORY_FIELDS),
        "automatic_summary_paths": [
            "_runtime",
            "_step",
            "_timestamp",
            "_wandb.runtime",
        ],
        "run_record_service_fields": [
            "deterministic_run_id",
            "display_name",
            "entity",
            "project",
            "redacted_host",
            "redacted_program_path",
            "start_time",
        ],
        "telemetry_fields": [
            "feature_flags",
            "optional_environment_kind_flags",
            "optional_huggingface_version",
            "platform_and_architecture",
            "python_version",
            "recognized_import_presence",
            "sdk_and_core_versions",
        ],
        "nondeterministic_service_fields": [
            "config._wandb.start_time",
            "history._runtime",
            "history._timestamp",
            "run.start_time",
            "summary._runtime",
            "summary._timestamp",
        ],
        "allowed_sdk_transaction_records": [
            "exit",
            "header",
            "history",
            "metric",
            "run",
            "summary",
            "telemetry",
        ],
        "disabled_automatic_collectors": [
            "code",
            "console",
            "git",
            "machine_hardware_inventory",
            "requirements",
            "system_stats",
            "third_party_error_reporting",
        ],
    }


def _wandb_publication_lifecycle_policy() -> dict[str, Any]:
    return {
        "pending_display_name": "pending-<deterministic-run-id>",
        "pending_name_never_equals_hotkey": True,
        "application_write_order": [
            "metric_definition",
            "all_authorized_logs",
            "non-commit-summary-fields",
            "mt_artifact_digest",
        ],
        "commit_summary_field": "mt_artifact_digest",
        "success_finish_exit_code": 0,
        "failure_finish_exit_code": 1,
        "first_remote_readback": {
            "required_state": "finished",
            "required_display_name": "pending-<deterministic-run-id>",
            "application_config": "canonical-exact-after-only-reserved-config-removal",
            "application_summary": "canonical-exact-after-only-disclosed-automatic-removal",
            "history": "all-rows-canonical-exact-after-only-disclosed-automatic-removal",
        },
        "final_commit_marker": "rename-display-name-to-hotkey-after-first-readback",
        "update_exception_resolution": {
            "exact-final-readback": "committed",
            "exact-pending-readback": "pending_failed",
            "indeterminate-readback": "outcome_uncertain",
        },
        "update_acknowledgement": "marks-committed-before-second-readback",
        "second_remote_readback": "exact-final-refetch-after-update-ack",
        "terminal_states": {
            "committed": "never-replay; postcommit-errors-preserve-committed-state",
            "pending_failed": "final-marker-absent; deterministic-id-consumed; retry-forbidden",
            "outcome_uncertain": "never-retry-or-delete; read-only-reconcile-by-run-id",
        },
        "readback_timeout_seconds": WANDB_READBACK_TIMEOUT_SECONDS,
    }


def _wandb_controls() -> dict[str, Any]:
    return {
        "sdk": {"package": "wandb", "version": WANDB_SDK_VERSION},
        "environment": _wandb_environment_policy(),
        "isolation": _wandb_isolation_policy(),
        "service_metadata_policy": _wandb_service_metadata_policy(),
        "publication_lifecycle": _wandb_publication_lifecycle_policy(),
        "run_identity": {
            "algorithm": "sha256-canonical-envelope-controls-and-application-first-40-hex",
            "prefix": WANDB_RUN_ID_PREFIX,
            "resume": "never",
            "replay": "existing-initialized-run-id-refused",
            "pending_failure": "deterministic-id-consumed-and-retry-forbidden",
        },
        "settings": {
            "base_url": "https://api.wandb.ai",
            "config_paths": [],
            "console": "off",
            "disable_code": True,
            "disable_git": True,
            "disable_git_fork_point": True,
            "disable_job_creation": True,
            "host": WANDB_REDACTED_HOST,
            "label_disable": True,
            "launch": False,
            "mode": "online",
            "program": WANDB_REDACTED_PROGRAM,
            "program_abspath": WANDB_REDACTED_PROGRAM_ABSPATH,
            "program_relpath": WANDB_REDACTED_PROGRAM,
            "quiet": True,
            "reinit": "create_new",
            "resume": "never",
            "sagemaker_disable": True,
            "save_code": False,
            "silent": True,
            "sync_tensorboard": False,
            "x_disable_machine_info": True,
            "x_disable_meta": True,
            "x_disable_stats": True,
            "x_disable_viewer": True,
            "x_save_requirements": False,
            "x_server_side_derived_summary": False,
            "x_server_side_expand_glob_metrics": False,
        },
        "metric_definition": {
            "name": "*",
            "summary": "none",
            "overwrite": True,
        },
    }


def _validate_wandb_environment(controls: Mapping[str, Any]) -> None:
    policy = dict(_mapping(controls["environment"], "W&B environment policy"))
    if policy != _wandb_environment_policy():
        raise CodeProvenanceError("W&B environment policy changed")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None or not api_key.strip():
        raise CodeProvenanceError("publish requires a nonempty WANDB_API_KEY")
    allowed = frozenset(policy["allowed_ambient_variables"])
    forbidden = sorted(
        name
        for name in os.environ
        if name.startswith(policy["forbidden_ambient_prefix"]) and name not in allowed
    )
    if forbidden:
        raise CodeProvenanceError("ambient W&B variables are forbidden: " + ", ".join(forbidden))


def _require_pristine_wandb_client(wandb_client: Any) -> None:
    if getattr(wandb_client, "run", None) is not None:
        raise CodeProvenanceError("an existing W&B run makes publication ambiguous")
    if getattr(wandb_client, "__name__", None) != "wandb":
        return
    try:
        setup_module = importlib.import_module("wandb.sdk.wandb_setup")
    except ImportError as exc:
        raise CodeProvenanceError("W&B setup state is not inspectable") from exc
    if getattr(setup_module, "_singleton", None) is not None:
        raise CodeProvenanceError("W&B SDK was initialized before the isolated publication")


def _preflight_real_wandb_import() -> None:
    loaded_wandb = sorted(
        name for name in sys.modules if name == "wandb" or name.startswith("wandb.")
    )
    if loaded_wandb:
        raise CodeProvenanceError("the real W&B SDK was imported before isolated publication")
    loaded_sentry = sorted(
        name for name in sys.modules if name == "sentry_sdk" or name.startswith("sentry_sdk.")
    )
    if loaded_sentry:
        raise CodeProvenanceError("Sentry was imported before isolated W&B publication")
    try:
        installed = importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CodeProvenanceError("the pinned W&B SDK is not installed") from exc
    if installed != WANDB_SDK_VERSION:
        raise CodeProvenanceError(
            f"W&B distribution version changed: expected {WANDB_SDK_VERSION}, got {installed}"
        )


def _require_disabled_wandb_error_reporting(wandb_client: Any) -> None:
    if getattr(wandb_client, "__name__", None) != "wandb":
        return
    try:
        env_module = importlib.import_module("wandb.env")
        analytics_module = importlib.import_module("wandb.analytics")
    except ImportError as exc:
        raise CodeProvenanceError("W&B error-reporting state is not inspectable") from exc
    enabled = getattr(env_module, "error_reporting_enabled", None)
    get_sentry = getattr(analytics_module, "get_sentry", None)
    if not callable(enabled) or enabled() is not False or not callable(get_sentry):
        raise CodeProvenanceError("W&B third-party error reporting is not disabled")
    reporter = get_sentry()
    if getattr(reporter, "_enabled", None) is not False:
        raise CodeProvenanceError("W&B Sentry reporter is not disabled")


def _require_wandb_settings(
    settings: Any,
    expected_settings: Mapping[str, Any],
    label: str,
) -> None:
    missing = object()
    for key, expected in expected_settings.items():
        actual = getattr(settings, key, missing)
        if actual is missing or actual != expected:
            raise CodeProvenanceError(f"{label} {key!r} changed")


def _wandb_environment_snapshot() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name.startswith("WANDB_")}


def _restore_wandb_environment(snapshot: Mapping[str, str]) -> None:
    for name in tuple(os.environ):
        if name.startswith("WANDB_") and name not in snapshot:
            del os.environ[name]
    for name, value in snapshot.items():
        os.environ[name] = value


def publication_payload(publication: Publication) -> dict[str, Any]:
    """Build the canonical application envelope, not a W&B wire transcript."""

    run_id = _wandb_run_id(publication)
    return {
        "schema": WANDB_PAYLOAD_SCHEMA,
        "destination": {
            "entity": ENTITY,
            "project": PROJECT,
            "name": HOTKEY,
            "id": run_id,
            "pending_name": _wandb_pending_name(run_id),
        },
        "controls": _wandb_controls(),
        "config": _wandb_config(publication),
        "logs": _wandb_log_records(publication),
        "summary": _wandb_summary(publication),
    }


def canonical_payload_bytes(publication: Publication) -> bytes:
    """Return deterministic UTF-8 JSON terminated by exactly one newline."""

    return _canonical_json_value_bytes(publication_payload(publication), "W&B payload") + b"\n"


def _revalidated_payload_bytes(publication: Publication) -> bytes:
    revalidated = validate_publication(publication.request)
    current = canonical_payload_bytes(publication)
    expected = canonical_payload_bytes(revalidated)
    if current != expected:
        raise CodeProvenanceError("publication inputs changed after validation")
    return expected


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_payload(publication: Publication, output: Path) -> tuple[int, str]:
    """Publish a complete mode-0600 payload atomically without replacing a destination."""

    payload = _revalidated_payload_bytes(publication)
    path = Path(output)
    parent = path.parent
    link_created = False
    published_durably = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{path.name}.staging-",
            dir=parent,
        ) as temporary:
            staging_root = Path(temporary)
            root_stat = staging_root.lstat()
            if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
                raise CodeProvenanceError("W&B payload staging root is not mode-0700")
            staging = staging_root / "payload"
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = -1
            opened: os.stat_result | None = None
            try:
                descriptor = os.open(staging, flags, 0o600)
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise CodeProvenanceError("W&B payload staging output is not a regular file")
                os.fchmod(descriptor, 0o600)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written < 1:
                        raise CodeProvenanceError("W&B payload output write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                after = os.fstat(descriptor)
                if (
                    after.st_dev != opened.st_dev
                    or after.st_ino != opened.st_ino
                    or after.st_size != len(payload)
                    or stat.S_IMODE(after.st_mode) != 0o600
                ):
                    raise CodeProvenanceError("W&B payload staging identity changed while writing")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if opened is None:
                raise CodeProvenanceError("W&B payload staging descriptor was not opened")
            try:
                os.link(staging, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise CodeProvenanceError(f"W&B payload output already exists: {path}") from exc
            link_created = True
            try:
                published = path.lstat()
                if (
                    not stat.S_ISREG(published.st_mode)
                    or published.st_dev != opened.st_dev
                    or published.st_ino != opened.st_ino
                    or published.st_size != len(payload)
                    or stat.S_IMODE(published.st_mode) != 0o600
                ):
                    raise CodeProvenanceError("atomically published W&B payload identity changed")
                _fsync_directory(parent)
                published_durably = True
            except BaseException as publish_exc:
                raise PayloadExportOutcomeUncertainError(
                    "payload export failed after the no-replace link; destination was "
                    "intentionally preserved and requires read-only reconciliation before retry"
                ) from publish_exc
    except BaseException as exc:
        if published_durably:
            raise PayloadExportPostCommitError(
                "payload is durably committed, but private staging cleanup failed; "
                "retry is forbidden"
            ) from exc
        if link_created:
            if isinstance(exc, PayloadExportOutcomeUncertainError):
                raise
            raise PayloadExportOutcomeUncertainError(
                "payload export failed after the no-replace link; destination was "
                "intentionally preserved and requires read-only reconciliation before retry"
            ) from exc
        if isinstance(exc, CodeProvenanceError):
            raise
        if isinstance(exc, OSError):
            raise CodeProvenanceError(f"W&B payload could not be published safely: {path}") from exc
        raise
    return len(payload), _digest_bytes(payload)


def _strict_payload_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeProvenanceError(f"authorized W&B payload is not strict JSON: {exc}") from exc
    payload = dict(_mapping(value, "authorized W&B payload"))
    if _canonical_json_value_bytes(payload, "authorized W&B payload") + b"\n" != raw:
        raise CodeProvenanceError("authorized W&B payload is not canonical JSON")
    return payload


def authorize_payload(
    publication: Publication,
    payload_file: Path,
    authorized_payload_digest: str,
) -> dict[str, Any]:
    """Bind explicit authorization to the exact, freshly reconstructed payload bytes."""

    authorized_digest = _require_digest(authorized_payload_digest, "authorized W&B payload digest")
    expected = _revalidated_payload_bytes(publication)
    supplied = _read_regular(payload_file, "authorized W&B payload")
    if supplied != expected:
        raise CodeProvenanceError(
            "authorized W&B payload does not exactly match current validated inputs"
        )
    actual_digest = _digest_bytes(supplied)
    if actual_digest != authorized_digest:
        raise CodeProvenanceError(
            f"authorized W&B payload digest mismatch: expected {authorized_digest}, "
            f"got {actual_digest}"
        )
    return _strict_payload_object(supplied)


def _canonical_remote_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json_value_bytes(actual, label) != _canonical_json_value_bytes(expected, label):
        raise CodeProvenanceError(f"remote W&B {label} differs from the authorized payload")


def _remote_application_summary(value: Any) -> dict[str, Any]:
    summary = dict(_mapping(value, "remote W&B summary"))
    nested = summary.get("_wandb")
    if nested is not None:
        nested_mapping = _mapping(nested, "remote W&B automatic summary")
        if set(nested_mapping) - {"runtime"}:
            raise CodeProvenanceError("remote W&B summary added an undisclosed automatic field")
    for field in WANDB_AUTOMATIC_SUMMARY_FIELDS:
        summary.pop(field, None)
    return summary


def _remote_application_history(
    rows: Any,
    expected_logs: Sequence[tuple[int, dict[str, Any]]],
) -> None:
    try:
        remote_rows = list(rows)
    except (TypeError, ValueError) as exc:
        raise CodeProvenanceError("remote W&B history is not iterable") from exc
    if len(remote_rows) != len(expected_logs):
        raise CodeProvenanceError("remote W&B history row count changed")
    automatic = frozenset(WANDB_AUTOMATIC_HISTORY_FIELDS)
    for (expected_step, expected_payload), raw_row in zip(expected_logs, remote_rows, strict=True):
        row = dict(_mapping(raw_row, f"remote W&B history row {expected_step}"))
        unexpected = set(row) - set(expected_payload) - automatic
        if unexpected:
            raise CodeProvenanceError("remote W&B history added an undisclosed automatic field")
        if row.get("_step") != expected_step:
            raise CodeProvenanceError("remote W&B history step changed")
        application = {key: value for key, value in row.items() if key not in automatic}
        _canonical_remote_equal(application, expected_payload, f"history row {expected_step}")


def _remote_readback(
    client: Any,
    *,
    destination: Mapping[str, Any],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    logs: Sequence[tuple[int, dict[str, Any]]],
    expected_names: Sequence[str],
) -> Any:
    factory = getattr(client, "Api", None)
    if not callable(factory):
        raise CodeProvenanceError("W&B client does not provide the required remote API")
    try:
        api = factory(
            overrides={"base_url": _wandb_controls()["settings"]["base_url"]},
            timeout=WANDB_READBACK_TIMEOUT_SECONDS,
        )
        remote = api.run(f"{destination['entity']}/{destination['project']}/{destination['id']}")
        if (
            getattr(remote, "id", None) != destination["id"]
            or getattr(remote, "entity", None) != destination["entity"]
            or getattr(remote, "project", None) != destination["project"]
            or getattr(remote, "name", None) not in expected_names
        ):
            raise CodeProvenanceError("remote W&B run identity changed")
        if str(getattr(remote, "state", "")).lower() != "finished":
            raise CodeProvenanceError("remote W&B run is not successfully finished")
        remote_config = dict(_mapping(getattr(remote, "config", None), "remote W&B config"))
        _canonical_remote_equal(remote_config, dict(config), "application config")
        raw_config = dict(
            _mapping(getattr(remote, "rawconfig", remote_config), "remote W&B raw config")
        )
        unexpected_config = set(raw_config) - set(config) - set(WANDB_AUTOMATIC_CONFIG_FIELDS)
        if unexpected_config:
            raise CodeProvenanceError("remote W&B config added an undisclosed reserved field")
        raw_application_config = {
            key: value
            for key, value in raw_config.items()
            if key not in WANDB_AUTOMATIC_CONFIG_FIELDS
        }
        _canonical_remote_equal(raw_application_config, dict(config), "raw application config")
        _canonical_remote_equal(
            _remote_application_summary(getattr(remote, "summary", None)),
            dict(summary),
            "application summary",
        )
        scan_history = getattr(remote, "scan_history", None)
        if not callable(scan_history):
            raise CodeProvenanceError("remote W&B run does not support complete history scanning")
        _remote_application_history(
            scan_history(page_size=1_000, min_step=0, use_cache=False),
            logs,
        )
        return remote
    except CodeProvenanceError:
        raise
    except Exception as exc:
        raise CodeProvenanceError(f"remote W&B readback failed closed: {exc}") from exc


def _publish_authorized_payload(
    payload: Mapping[str, Any], wandb_client: Any
) -> WandbPublicationOutcome:
    _exact_keys(
        payload,
        frozenset({"schema", "destination", "controls", "config", "logs", "summary"}),
        "authorized W&B payload",
    )
    if payload["schema"] != WANDB_PAYLOAD_SCHEMA:
        raise CodeProvenanceError("W&B payload schema changed")

    destination = _mapping(payload["destination"], "W&B destination")
    _exact_keys(
        destination,
        frozenset({"entity", "project", "name", "id", "pending_name"}),
        "W&B destination",
    )
    if (
        destination["entity"] != ENTITY
        or destination["project"] != PROJECT
        or destination["name"] != HOTKEY
    ):
        raise CodeProvenanceError("W&B destination changed")
    expected_run_id = _wandb_run_id_from_envelope(
        payload["controls"],
        payload["config"],
        payload["logs"],
        payload["summary"],
    )
    if destination["id"] != expected_run_id:
        raise CodeProvenanceError("deterministic W&B run ID changed")
    if destination["pending_name"] != _wandb_pending_name(expected_run_id):
        raise CodeProvenanceError("pending W&B display name changed")

    controls = _mapping(payload["controls"], "W&B controls")
    _exact_keys(
        controls,
        frozenset(
            {
                "sdk",
                "environment",
                "isolation",
                "service_metadata_policy",
                "publication_lifecycle",
                "run_identity",
                "settings",
                "metric_definition",
            }
        ),
        "W&B controls",
    )
    expected_controls = _wandb_controls()
    if dict(controls) != expected_controls:
        raise CodeProvenanceError("W&B controls changed")
    _validate_wandb_environment(controls)

    config = _mapping(payload["config"], "W&B application config")
    summary = _mapping(payload["summary"], "W&B application summary")
    _exact_keys(summary, frozenset(APPLICATION_SUMMARY_FIELDS), "W&B application summary")
    logs = _sequence(payload["logs"], "W&B logs")
    checked_logs: list[tuple[int, dict[str, Any]]] = []
    for expected_step, item in enumerate(logs, 1):
        record = _mapping(item, f"W&B log {expected_step}")
        _exact_keys(record, frozenset({"step", "payload"}), f"W&B log {expected_step}")
        if record["step"] != expected_step:
            raise CodeProvenanceError("W&B log steps are not exact and contiguous")
        checked_logs.append(
            (expected_step, dict(_mapping(record["payload"], f"W&B log {expected_step} payload")))
        )

    sdk = _mapping(controls["sdk"], "W&B SDK")
    if dict(sdk) != {"package": "wandb", "version": WANDB_SDK_VERSION}:
        raise CodeProvenanceError("W&B SDK contract changed")
    base_settings = dict(_mapping(controls["settings"], "W&B settings"))

    if wandb_client is not None and getattr(wandb_client, "__name__", None) == "wandb":
        raise CodeProvenanceError("a pre-imported real W&B client is forbidden")
    if wandb_client is None:
        _preflight_real_wandb_import()

    environment_before_import = _wandb_environment_snapshot()
    forced_environment = _mapping(
        _mapping(controls["environment"], "W&B environment policy")["forced_process_variables"],
        "forced W&B environment",
    )
    forced_items: list[tuple[str, str]] = []
    for name, value in forced_environment.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise CodeProvenanceError("forced W&B environment must contain strings")
        forced_items.append((name, value))

    publication_state: WandbPublicationState | None = None
    outcome: WandbPublicationOutcome | None = None
    operation_error: BaseException | None = None

    try:
        for name, value in forced_items:
            os.environ[name] = value
        with tempfile.TemporaryDirectory(prefix="microtensor-wandb-") as temporary:
            root = Path(temporary)
            root_stat = root.lstat()
            if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
                raise CodeProvenanceError("private W&B root is not a mode-0700 directory")
            system_settings = root / "system-settings"
            credentials_file = root / "credentials.json"
            if system_settings.exists() or credentials_file.exists():
                raise CodeProvenanceError("private W&B root was not empty")

            settings_payload = {
                **base_settings,
                "credentials_file": str(credentials_file),
                "root_dir": str(root),
                "settings_system": str(system_settings),
            }
            current_directory = -1
            try:
                current_directory = os.open(
                    ".",
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
                )
                os.chdir(root)
                client = wandb_client
                if client is None:
                    client = importlib.import_module("wandb")
                if getattr(client, "__version__", None) != WANDB_SDK_VERSION:
                    raise CodeProvenanceError("W&B SDK version changed")
                _require_pristine_wandb_client(client)
                _require_disabled_wandb_error_reporting(client)
                factory = getattr(client, "Settings", None)
                if not callable(factory):
                    raise CodeProvenanceError(
                        "W&B client does not provide the required Settings contract"
                    )
                try:
                    settings = factory(**settings_payload)
                except Exception as exc:
                    raise CodeProvenanceError(f"W&B privacy settings were refused: {exc}") from exc
                _require_wandb_settings(
                    settings,
                    settings_payload,
                    "constructed W&B privacy setting",
                )
                setup_method = getattr(client, "setup", None)
                teardown_method = getattr(client, "teardown", None)
                if not callable(setup_method) or not callable(teardown_method):
                    raise CodeProvenanceError(
                        "W&B client does not provide the required setup/teardown contract"
                    )

                try:
                    try:
                        setup_state = setup_method(settings=settings)
                    except Exception as exc:
                        raise CodeProvenanceError(f"isolated W&B setup was refused: {exc}") from exc
                    effective_settings = getattr(setup_state, "settings", None)
                    if effective_settings is None:
                        raise CodeProvenanceError("effective global W&B settings are unavailable")
                    _require_wandb_settings(
                        effective_settings,
                        settings_payload,
                        "effective global W&B setting",
                    )
                    _require_wandb_settings(
                        effective_settings,
                        _wandb_identity_defaults(),
                        "effective global W&B identity setting",
                    )

                    run = None
                    finished_successfully = False
                    try:
                        publication_state = WandbPublicationState.PENDING_FAILED
                        run = client.init(
                            entity=destination["entity"],
                            project=destination["project"],
                            name=destination["pending_name"],
                            id=destination["id"],
                            config=dict(config),
                            settings=settings,
                        )
                        if getattr(run, "id", None) != destination["id"]:
                            raise CodeProvenanceError("W&B returned a different run ID")
                        metric_definition = _mapping(
                            controls["metric_definition"], "W&B metric definition"
                        )
                        run.define_metric(**dict(metric_definition))
                        for step, record_payload in checked_logs:
                            run.log(record_payload, step=step)
                        for key, value in summary.items():
                            if key != "mt_artifact_digest":
                                run.summary[key] = value
                        run.summary["mt_artifact_digest"] = summary["mt_artifact_digest"]
                        run.finish(exit_code=0)
                        finished_successfully = True
                    except BaseException as publish_exc:
                        if run is not None and not finished_successfully:
                            try:
                                run.finish(exit_code=1)
                            except Exception as finish_exc:
                                raise CodeProvenanceError(
                                    "W&B publication failed and failure finalization was refused"
                                ) from finish_exc
                        raise publish_exc

                    remote = _remote_readback(
                        client,
                        destination=destination,
                        config=config,
                        summary=summary,
                        logs=checked_logs,
                        expected_names=(str(destination["pending_name"]),),
                    )
                    remote.name = destination["name"]
                    update_remote = getattr(remote, "update", None)
                    if not callable(update_remote):
                        raise CodeProvenanceError("remote W&B run cannot commit its final name")
                    try:
                        update_remote()
                    except BaseException as update_exc:
                        publication_state = WandbPublicationState.OUTCOME_UNCERTAIN
                        try:
                            resolved_remote = _remote_readback(
                                client,
                                destination=destination,
                                config=config,
                                summary=summary,
                                logs=checked_logs,
                                expected_names=(
                                    str(destination["pending_name"]),
                                    str(destination["name"]),
                                ),
                            )
                        except BaseException as readback_exc:
                            raise WandbOutcomeUncertainError(
                                "W&B final-name update raised and exact remote state could not "
                                "be reconciled; never retry or delete this deterministic run ID",
                                run_id=expected_run_id,
                            ) from readback_exc
                        if resolved_remote.name == destination["name"]:
                            publication_state = WandbPublicationState.COMMITTED
                            outcome = WandbPublicationOutcome(
                                state=WandbPublicationState.COMMITTED,
                                run_id=expected_run_id,
                                resolution="exact_readback_after_update_exception",
                            )
                        else:
                            publication_state = WandbPublicationState.PENDING_FAILED
                            raise WandbPendingFailedError(
                                "W&B final-name update failed with the exact pending name still "
                                "present; deterministic run ID is consumed and retry is forbidden",
                                run_id=expected_run_id,
                            ) from update_exc
                    else:
                        publication_state = WandbPublicationState.COMMITTED
                        outcome = WandbPublicationOutcome(
                            state=WandbPublicationState.COMMITTED,
                            run_id=expected_run_id,
                            resolution="update_ack",
                        )
                        _remote_readback(
                            client,
                            destination=destination,
                            config=config,
                            summary=summary,
                            logs=checked_logs,
                            expected_names=(str(destination["name"]),),
                        )
                finally:
                    teardown_method()
            finally:
                if current_directory >= 0:
                    try:
                        os.fchdir(current_directory)
                    finally:
                        os.close(current_directory)
    except BaseException as exc:
        operation_error = exc
    try:
        _restore_wandb_environment(environment_before_import)
    except BaseException as restore_exc:
        if operation_error is None:
            operation_error = restore_exc

    if operation_error is not None:
        if isinstance(operation_error, WandbPublicationStateError):
            raise operation_error
        if publication_state is WandbPublicationState.COMMITTED:
            raise WandbPostCommitError(
                "W&B final name is committed, but postcommit verification or cleanup failed; "
                f"never retry this deterministic run ID: {operation_error}",
                run_id=expected_run_id,
            ) from operation_error
        if publication_state is WandbPublicationState.OUTCOME_UNCERTAIN:
            raise WandbOutcomeUncertainError(
                "W&B commit outcome is uncertain; never retry or delete and reconcile "
                f"read-only by deterministic run ID: {operation_error}",
                run_id=expected_run_id,
            ) from operation_error
        if publication_state is WandbPublicationState.PENDING_FAILED:
            raise WandbPendingFailedError(
                "W&B pending publication failed after consuming its deterministic run ID; "
                f"the final marker is absent and retry is forbidden: {operation_error}",
                run_id=expected_run_id,
            ) from operation_error
        raise operation_error

    if outcome is None:
        raise CodeProvenanceError("W&B publication ended without a terminal outcome")
    return outcome


def publish(
    publication: Publication,
    payload_file: Path,
    authorized_payload_digest: str,
    wandb_client: Any,
) -> WandbPublicationOutcome:
    """Publish only after exact file and digest authorization are revalidated."""

    payload = authorize_payload(publication, payload_file, authorized_payload_digest)
    return _publish_authorized_payload(payload, wandb_client)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "export", "publish"))
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--load-spec", type=Path, required=True)
    parser.add_argument("--conversion-receipt", type=Path, required=True)
    parser.add_argument("--finished-block", type=int, required=True)
    parser.add_argument("--hf-diagnostic", type=Path)
    parser.add_argument("--gguf-diagnostic", type=Path)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--llama-cpp", type=Path)
    parser.add_argument("--calibration-current-dataset", type=Path)
    parser.add_argument("--calibration-current-source-corpus", type=Path)
    parser.add_argument("--payload-file", type=Path)
    parser.add_argument("--authorized-payload-digest")
    return parser


def main(argv: Sequence[str] | None = None, *, wandb_client: Any | None = None) -> int:
    args = _parser().parse_args(argv)
    request = PublicationRequest(
        training_run=args.training_run,
        training_dataset=args.training_dataset,
        source_corpus=args.source_corpus,
        base=args.base,
        artifact=args.artifact,
        artifact_digest=args.artifact_digest,
        load_spec=args.load_spec,
        conversion_receipt=args.conversion_receipt,
        finished_block=args.finished_block,
        hf_diagnostic=args.hf_diagnostic,
        gguf_diagnostic=args.gguf_diagnostic,
        calibration_receipt=args.calibration_receipt,
        llama_cpp=args.llama_cpp,
        calibration_current_dataset=args.calibration_current_dataset,
        calibration_current_source_corpus=args.calibration_current_source_corpus,
    )
    publication_outcome: WandbPublicationOutcome | None = None
    try:
        publication = validate_publication(request)
        payload_bytes: int | None = None
        payload_digest: str | None = None
        if args.action == "export":
            if args.payload_file is None:
                raise CodeProvenanceError("export requires --payload-file")
            payload_bytes, payload_digest = export_payload(publication, args.payload_file)
        elif args.action == "publish":
            if args.payload_file is None or args.authorized_payload_digest is None:
                raise CodeProvenanceError(
                    "publish requires --payload-file and --authorized-payload-digest"
                )
            payload = authorize_payload(
                publication,
                args.payload_file,
                args.authorized_payload_digest,
            )
            _validate_wandb_environment(_mapping(payload["controls"], "W&B controls"))
            publication_outcome = _publish_authorized_payload(payload, wandb_client)
    except (CodeProvenanceError, ImportError) as exc:
        raise SystemExit(f"code provenance refused: {exc}") from exc
    result: dict[str, Any] = {
        "action": args.action,
        "artifact_digest": publication.artifact["tree_digest"],
        "finished_block": publication.finished_block,
        "metrics": len(publication.metrics),
        "calibration_claim": (
            publication.calibration["claim"]
            if publication.calibration is not None
            else NO_CALIBRATION_CLAIM
        ),
    }
    if payload_bytes is not None and payload_digest is not None:
        result["payload"] = {
            "path": str(args.payload_file),
            "bytes": payload_bytes,
            "digest": payload_digest,
        }
    if publication_outcome is not None:
        result["publication"] = {
            "state": publication_outcome.state.value,
            "run_id": publication_outcome.run_id,
            "resolution": publication_outcome.resolution,
            "retry_forbidden": publication_outcome.retry_forbidden,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
