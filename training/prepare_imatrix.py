#!/usr/bin/env python3
"""Build a deterministic, public-only llama.cpp importance-matrix corpus.

The emitted text is a byte-for-byte concatenation of Qwen chat-template
renderings.  Each rendering contains the public task prompt and a canonical
gold assistant answer.  This script prepares calibration input only; it does
not run llama-imatrix or quantize a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 92
STAGE1_RESERVE = 384
SUPPORTED_RESERVES = (0, STAGE1_RESERVE)
METADATA_SCHEMA = "microtensor.imatrix-corpus.v1"
ENTITY_TYPES = frozenset(("Chemical", "Disease"))


class GoldValidationError(ValueError):
    """A stable, provenance-safe reason for rejecting a corpus row."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TrainingAPI:
    corpus_version: str
    loader_name: str
    load_rows: Callable[[Path], list[dict[str, Any]]]


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_training_api() -> TrainingAPI:
    """Load the existing recipe lazily so this module stays stdlib-importable."""

    module_name = "training.train_extract" if __package__ else "train_extract"
    module = importlib.import_module(module_name)
    version = getattr(module, "CORPUS_VERSION", None)
    loader = getattr(module, "load_rows", None)
    if not isinstance(version, str) or not callable(loader):
        raise RuntimeError("training.train_extract has no usable corpus API")
    return TrainingAPI(version, f"{module_name}.load_rows", loader)


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(details.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular, non-symlink file: {path}")
    return details


def inspect_tokenizer(path: Path) -> tuple[Path, dict[str, Any], str]:
    """Bind rendering to local Qwen tokenizer files and its exact template."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"tokenizer directory is unavailable: {path}") from exc
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError("tokenizer must be a local, non-symlink directory")

    required = ("tokenizer.json", "tokenizer_config.json", "config.json")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in required:
        candidate = resolved / name
        details = _regular_file(candidate, f"tokenizer {name}")
        artifacts[name] = {"bytes": details.st_size, "sha256": sha256_file(candidate)}

    config_path = resolved / "tokenizer_config.json"
    try:
        tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("tokenizer_config.json is not valid UTF-8 JSON") from exc
    if not isinstance(tokenizer_config, Mapping):
        raise ValueError("tokenizer_config.json must contain an object")
    if tokenizer_config.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError("tokenizer_class must be Qwen2Tokenizer")

    model_config_path = resolved / "config.json"
    try:
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("config.json is not valid UTF-8 JSON") from exc
    if not isinstance(model_config, Mapping) or model_config.get("model_type") != "qwen3":
        raise ValueError("tokenizer directory must identify a Qwen3 model")

    inline_template = tokenizer_config.get("chat_template")
    template_path = resolved / "chat_template.jinja"
    if isinstance(inline_template, str) and inline_template:
        template = inline_template
        template_source = "tokenizer_config.json:chat_template"
    elif template_path.exists():
        details = _regular_file(template_path, "tokenizer chat_template.jinja")
        try:
            template = template_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("chat_template.jinja is not valid UTF-8") from exc
        artifacts["chat_template.jinja"] = {
            "bytes": details.st_size,
            "sha256": sha256_file(template_path),
        }
        template_source = "chat_template.jinja"
    else:
        raise ValueError("local Qwen tokenizer has no chat template")
    if not all(marker in template for marker in ("<|im_start|>", "<|im_end|>", "enable_thinking")):
        raise ValueError("local tokenizer does not contain the expected Qwen chat template")

    identity = {
        "artifacts": artifacts,
        "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
        "chat_template_source": template_source,
        "model_type": "qwen3",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    return resolved, identity, template


def load_local_tokenizer(path: Path) -> Any:
    """Load only the supplied directory; never resolve a Hub identifier."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to render the local tokenizer") from exc
    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )


def _parse_gold(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GoldValidationError("malformed_gold_json") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"entities"}:
        raise GoldValidationError("malformed_gold_object")
    return raw


def canonical_gold(row: Mapping[str, Any]) -> str:
    """Return stable entity JSON, rejecting unsafe corrections or guesses."""

    prompt = row.get("prompt")
    inputs = row.get("inputs")
    if not isinstance(prompt, str) or not prompt:
        raise GoldValidationError("malformed_prompt")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("text"), str):
        raise GoldValidationError("malformed_source_text")
    source_text = inputs["text"]

    payload = _parse_gold(row.get("gold"))
    entities = payload.get("entities")
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        raise GoldValidationError("malformed_entities")

    pairs: set[tuple[str, str]] = set()
    for entity in entities:
        if not isinstance(entity, Mapping) or set(entity) != {"text", "type"}:
            raise GoldValidationError("malformed_entity")
        text_value = entity.get("text")
        entity_type = entity.get("type")
        if not isinstance(text_value, str) or not isinstance(entity_type, str):
            raise GoldValidationError("malformed_entity_values")
        if not text_value or text_value != text_value.strip():
            raise GoldValidationError("malformed_entity_text")
        if entity_type not in ENTITY_TYPES:
            raise GoldValidationError("unsupported_entity_type")
        if text_value not in source_text:
            raise GoldValidationError("gold_text_not_substring")
        pairs.add((text_value, entity_type))

    canonical = {
        "entities": [
            {"text": text_value, "type": entity_type}
            for text_value, entity_type in sorted(pairs)
        ]
    }
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_record(tokenizer: Any, row: Mapping[str, Any], gold: str) -> str:
    prompt = str(row["prompt"])
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gold},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer returned an empty or non-text chat rendering")
    if "\x00" in rendered or prompt not in rendered or gold not in rendered:
        raise ValueError("tokenizer rendering did not preserve the exact prompt and gold answer")
    if not rendered.endswith("<|im_end|>\n"):
        raise ValueError("Qwen chat rendering has no deterministic assistant boundary")
    return rendered


def _check_unique_refs(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        ref = row.get("ref")
        if not isinstance(ref, str) or not ref:
            raise ValueError("every public row must have a non-empty string ref")
        if ref in seen:
            raise ValueError(f"duplicate public row ref: {ref}")
        seen.add(ref)


def _refuse_input_alias(output: Path, inputs: Sequence[Path]) -> None:
    output_absolute = output.absolute()
    for source in inputs:
        try:
            if output_absolute == source.absolute() or (
                output.exists() and source.exists() and output.samefile(source)
            ):
                raise ValueError(f"output must not replace input: {source}")
        except OSError as exc:
            raise ValueError(f"could not validate output path: {output}") from exc


def atomic_write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    """Replace a non-symlink destination with a fully synced regular file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _regular_file(path, "existing output")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _regular_file(path, "written output")


def prepare_calibration(
    *,
    corpus: Path,
    tokenizer_path: Path,
    output: Path,
    reserve_examples: int,
    max_examples: int | None,
    seed: int = SEED,
    training_api: TrainingAPI | None = None,
    tokenizer_loader: Callable[[Path], Any] = load_local_tokenizer,
) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"seed must be {SEED} to match the training split")
    if reserve_examples not in SUPPORTED_RESERVES:
        raise ValueError(f"reserve examples must be one of {SUPPORTED_RESERVES}")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max examples must be positive when supplied")

    corpus_details = _regular_file(corpus, "public corpus")
    resolved_tokenizer, tokenizer_identity, expected_template = inspect_tokenizer(tokenizer_path)
    metadata_path = output.with_name(output.name + ".metadata.json")
    tokenizer_files = tuple(
        resolved_tokenizer / name for name in tokenizer_identity["artifacts"]
    )
    _refuse_input_alias(output, (corpus, *tokenizer_files))
    _refuse_input_alias(metadata_path, (corpus, output))

    api = training_api or load_training_api()
    rows = api.load_rows(corpus)
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("training corpus loader returned malformed rows")
    _check_unique_refs(rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    tokenizer = tokenizer_loader(resolved_tokenizer)
    runtime_template = getattr(tokenizer, "chat_template", None)
    if runtime_template != expected_template:
        raise ValueError("loaded tokenizer chat template does not match its local identity file")

    reserved = shuffled[:reserve_examples]
    candidates = shuffled[reserve_examples:]
    rejected: list[dict[str, Any]] = []
    canonical_by_ref: dict[str, str] = {}
    candidate_refs = {str(row["ref"]) for row in candidates}
    for shuffled_index, row in enumerate(shuffled):
        ref = str(row["ref"])
        try:
            canonical_by_ref[ref] = canonical_gold(row)
        except GoldValidationError as exc:
            rejected.append(
                {
                    "code": exc.code,
                    "ref": ref,
                    "selection": "candidate" if ref in candidate_refs else "reserve",
                    "shuffled_index": shuffled_index,
                }
            )

    eligible = [row for row in candidates if str(row["ref"]) in canonical_by_ref]
    included = eligible if max_examples is None else eligible[:max_examples]
    if not included:
        raise ValueError("selection produced no valid calibration rows")

    output_parts: list[bytes] = []
    record_manifest: list[dict[str, Any]] = []
    for row in included:
        ref = str(row["ref"])
        rendered = render_record(tokenizer, row, canonical_by_ref[ref])
        encoded = rendered.encode("utf-8")
        output_parts.append(encoded)
        record_manifest.append(
            {"bytes": len(encoded), "ref": ref, "sha256": sha256_bytes(encoded)}
        )
    output_payload = b"".join(output_parts)

    metadata: dict[str, Any] = {
        "schema": METADATA_SCHEMA,
        "source": {
            "corpus_bytes": corpus_details.st_size,
            "corpus_file_sha256": sha256_file(corpus),
            "corpus_version": api.corpus_version,
            "loader": api.loader_name,
            "public_train_rows": len(rows),
        },
        "tokenizer": {
            **tokenizer_identity,
            "loader": "transformers.AutoTokenizer.from_pretrained",
            "local_files_only": True,
            "runtime_class": type(tokenizer).__name__,
            "trust_remote_code": False,
        },
        "selection": {
            "algorithm": "random.Random(seed).shuffle(rows)",
            "eligible_examples": len(eligible),
            "included_examples": len(included),
            "included_refs": [str(row["ref"]) for row in included],
            "max_examples": max_examples,
            "omitted_after_cap": len(eligible) - len(included),
            "rejected_rows": rejected,
            "reserve_examples": reserve_examples,
            "reserved_refs": [str(row["ref"]) for row in reserved],
            "seed": seed,
        },
        "rendering": {
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
        },
        "output": {
            "bytes": len(output_payload),
            "records": len(record_manifest),
            "sha256": sha256_bytes(output_payload),
        },
        "records": record_manifest,
        "runtime": {"python": platform.python_version()},
    }
    serialized_metadata = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
    metadata_payload = (serialized_metadata + "\n").encode("utf-8")

    atomic_write(output, output_payload)
    atomic_write(metadata_path, metadata_payload)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument(
        "--reserve-examples",
        choices=SUPPORTED_RESERVES,
        default=STAGE1_RESERVE,
        type=int,
        help="384 excludes the exact stage-1 reserve; 0 uses all public rows",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="deterministically keep only the first N eligible rows after shuffle/reserve",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = prepare_calibration(
            corpus=args.corpus,
            tokenizer_path=args.tokenizer,
            output=args.output,
            reserve_examples=args.reserve_examples,
            max_examples=args.max_examples,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"refusing calibration corpus: {exc}") from exc
    print(json.dumps(metadata["output"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
