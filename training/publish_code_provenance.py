#!/usr/bin/env python3
"""Fail-closed validation and W&B publication for the code-track candidate.

Publication is deliberately the last operation.  Every local input is replayed
and validated twice: once when a :class:`Publication` is prepared and again
immediately before ``wandb.init``.  This module never reads credentials.
"""

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
LLAMA_CPP_REVISION: Final[str] = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
NO_CALIBRATION_CLAIM: Final[str] = (
    "none: no independently validated calibration execution receipt was supplied"
)
VALIDATED_CALIBRATION_CLAIM: Final[str] = (
    "exact supplied calibration execution receipt validated and byte-bound"
)
MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
MAX_METRICS_RECORDS: Final[int] = 1_000_000
_DIGEST: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CodeProvenanceError(ValueError):
    """Raised before any remote operation when provenance is incomplete."""


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
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
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


def _parse_metrics(path: Path, metadata: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
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
    if _digest_bytes(raw) != metadata.get("metrics_digest"):
        raise CodeProvenanceError("training metrics digest changed")
    return tuple(records)


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
) -> tuple[dict[str, Any], str]:
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
    return receipt, _digest_bytes(raw)


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
        if metadata.get("schema") != TRAINING_SCHEMA:
            raise CodeProvenanceError("only the completed historical8000 v5 receipt is publishable")
        metrics = _parse_metrics(request.training_run / "metrics.jsonl", metadata)
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
    calibration_digest: str | None = None
    if request.calibration_receipt is not None:
        calibration_payload, calibration_digest = _validate_calibration(
            request.calibration_receipt,
            training_lineage=training_lineage,
            artifact=artifact,
        )
        calibration = {
            "identity": _file_identity(request.calibration_receipt, "calibration receipt"),
            "schema": calibration_payload["schema"],
            "claim": VALIDATED_CALIBRATION_CLAIM,
        }

    conversion, conversion_raw = _json_file(request.conversion_receipt, "conversion receipt")
    if conversion.get("schema") != CONVERSION_SCHEMA:
        raise CodeProvenanceError("conversion receipt schema is not supported")
    conversion = _validate_generic_conversion(
        conversion,
        training_lineage=training_lineage,
        artifact=artifact,
        load_manifest=load_manifest,
        calibration_digest=calibration_digest,
    )
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


def publish(publication: Publication, wandb_client: Any) -> None:
    """Revalidate, then publish through an injected W&B-compatible client."""

    revalidated = validate_publication(publication.request)
    if revalidated != publication:
        raise CodeProvenanceError("publication inputs changed after validation")
    run = wandb_client.init(
        entity=ENTITY,
        project=PROJECT,
        name=HOTKEY,
        config=_wandb_config(publication),
    )
    try:
        for global_step, metric in enumerate(publication.metrics, 1):
            wandb_client.log(dict(metric), step=global_step)
        run.summary["mt_artifact_digest"] = publication.artifact["tree_digest"]
        run.summary["mt_finished_at"] = publication.finished_block
        run.summary["mt_training_records"] = len(publication.metrics)
        run.summary["mt_training_schema"] = TRAINING_SCHEMA
        run.summary["mt_conversion_receipt_digest"] = publication.conversion_receipt_identity[
            "digest"
        ]
    finally:
        wandb_client.finish()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "publish"))
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
    )
    try:
        publication = validate_publication(request)
        if args.action == "publish":
            client = wandb_client
            if client is None:
                client = importlib.import_module("wandb")
            publish(publication, client)
    except (CodeProvenanceError, ImportError) as exc:
        raise SystemExit(f"code provenance refused: {exc}") from exc
    print(
        json.dumps(
            {
                "action": args.action,
                "artifact_digest": publication.artifact["tree_digest"],
                "finished_block": publication.finished_block,
                "metrics": len(publication.metrics),
                "calibration_claim": (
                    publication.calibration["claim"]
                    if publication.calibration is not None
                    else NO_CALIBRATION_CLAIM
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
