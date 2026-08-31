#!/usr/bin/env python3
"""Fail-closed preparation for the official historical 8,000-row code corpus.

This module is intentionally separate from :mod:`training.code_candidate` so
the BigCodeBench-94 bootstrap remains the default, byte-for-byte contract.  It
validates one exact public response and prepares ``completion = gold`` without
parsing, extracting, executing, or otherwise rewriting corpus content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]


CORPUS_PROFILE: Final[str] = "historical8000"
TRACK: Final[str] = candidate.TRACK
HARDWARE_CLASS: Final[str] = candidate.HARDWARE_CLASS
CORPUS_VERSION: Final[str] = (
    "sha256:7299bd7c25056246c944ae0d38c7d0b0817b87ff1022ce331aa2bca865bc2f06"
)
PUBLIC_CORPUS_URL: Final[str] = (
    "https://api.microtensor.cloud/v1/corpora/"
    "sha256%3A7299bd7c25056246c944ae0d38c7d0b0817b87ff1022ce331aa2bca865bc2f06/public"
)
PUBLIC_CORPUS_RESPONSE_BYTES: Final[int] = 19_023_989
PUBLIC_CORPUS_RAW_DIGEST: Final[str] = (
    "sha256:eb76adcaabdd11c9ce0005c22e50a8530397c32127515a4461b1340e77e2d4b5"
)
PUBLIC_CORPUS_CANONICAL_BYTES: Final[int] = 19_023_989
PUBLIC_CORPUS_CANONICAL_DIGEST: Final[str] = (
    "sha256:18fad3468cdd409b39a4786a982c098e1378445083e913e9a215669f0acbebdc"
)
EXPECTED_COUNTS: Final[dict[str, int]] = {"train": 8_000, "fixed": 38, "rotating": 73}
EXPECTED_MANIFEST: Final[dict[str, Any]] = {
    "track": candidate.TRACK,
    "counts": EXPECTED_COUNTS,
    "limits": {"max_tests": 12, "max_test_bytes": 65_536},
    "source": {
        "train": "ise-uiuc/Magicoder-OSS-Instruct-75K, python subset",
        "scored": "bzantium/livecodebench, LeetCode subset, cutoff 2024-11-01",
    },
    "digests": {
        "tasks": "sha256:e66c0b3f04ea6a78d5ec05da42d1c9d4c3604a86200b35ab9d7c82c88f2cf356",
        "tests": "sha256:2191818c1649b5ed5e07078c7fe88ec43471534f12fc226c39084317c8f0fc4c",
    },
    "license": "CC BY 4.0 (LiveCodeBench)",
    "description": (
        "LeetCode problems from LiveCodeBench published on or after 2024-11-01. "
        "Solutions are methods on `class Solution`; the entry point names the method."
    ),
    "entry_point_style": "class_method",
}
ROOT_KEYS: Final[frozenset[str]] = candidate.ROOT_KEYS
TASK_KEYS: Final[frozenset[str]] = frozenset(
    {"gold", "max_output_tokens", "partition", "prompt", "ref"}
)
REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^mgc-[0-9]+$")
DATASET_SCHEMA: Final[str] = "microtensor.code.prepared.v2"
SPLIT_ALGORITHM: Final[str] = candidate.SPLIT_ALGORITHM
PREPARED_ROW_KEYS: Final[frozenset[str]] = candidate.PREPARED_ROW_KEYS
PREPARED_MANIFEST_KEYS: Final[frozenset[str]] = candidate.PREPARED_MANIFEST_KEYS
TARGET_CONSTRUCTION: Final[str] = "gold_verbatim"
TRAINING_TARGET_CONSTRUCTION: Final[str] = (
    "raw prompt -> verbatim public gold text (may contain fenced code and explanatory prose)"
)
DEVELOPMENT_QUALITY_CLAIM: Final[str] = (
    "none: the public train projection exposes no scored tests; holdout loss and structural "
    "diagnostics are non-execution training diagnostics"
)
FINAL_ALL_PUBLIC_QUALITY_CLAIM: Final[str] = (
    "none: all 8000 public training examples were used; no holdout or execution pass@1 was measured"
)
FINAL_ALL_PUBLIC_HOLDOUT_CLAIM: Final[str] = (
    "all 8000 public training examples were training inputs; no holdout or execution pass@1 "
    "was measured"
)


class HistoricalCandidateError(candidate.CandidateError):
    """Historical corpus bytes or prepared data violate the reviewed contract."""


# These helpers are deliberately identical across both public-corpus profiles.
canonical_json_bytes = candidate.canonical_json_bytes
digest_bytes = candidate.digest_bytes
digest_file = candidate.digest_file
_refs_digest = candidate._refs_digest


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise HistoricalCandidateError(f"{label} contains a non-string key")
    found = frozenset(value)
    if found != expected:
        raise HistoricalCandidateError(
            f"{label} keys changed: expected {sorted(expected)}, got {sorted(found)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalCandidateError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise HistoricalCandidateError(f"{label} must be an array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoricalCandidateError(f"{label} must be a non-empty string")
    return value


_DEFAULT_CANONICAL_DIGEST: Final[str] = ""


def validate_public_corpus_payload(
    payload: Any,
    *,
    expected_canonical_digest: str | None = _DEFAULT_CANONICAL_DIGEST,
) -> candidate.CorpusValidation:
    """Validate the exact historical public response without parsing any task code."""

    if expected_canonical_digest == _DEFAULT_CANONICAL_DIGEST:
        expected_canonical_digest = PUBLIC_CORPUS_CANONICAL_DIGEST
    root = _mapping(payload, "historical public corpus")
    _exact_keys(root, ROOT_KEYS, "historical public corpus")
    if root.get("version") != CORPUS_VERSION:
        raise HistoricalCandidateError("historical public corpus version changed")
    if root.get("track") != candidate.TRACK:
        raise HistoricalCandidateError("historical public corpus track is not code")
    if root.get("counts") != EXPECTED_COUNTS:
        raise HistoricalCandidateError("historical public corpus counts changed")
    if root.get("reference_model") != "" or root.get("reference") != []:
        raise HistoricalCandidateError("historical public corpus unexpectedly exposes references")
    manifest = _mapping(root.get("manifest"), "historical manifest")
    _exact_keys(manifest, frozenset(EXPECTED_MANIFEST), "historical manifest")
    if dict(manifest) != EXPECTED_MANIFEST:
        raise HistoricalCandidateError("historical manifest changed")

    tasks = _sequence(root.get("tasks"), "historical tasks")
    if len(tasks) != EXPECTED_COUNTS["train"]:
        raise HistoricalCandidateError("historical task count differs from the pinned train count")
    refs: list[str] = []
    for index, raw_task in enumerate(tasks):
        task = _mapping(raw_task, f"historical tasks[{index}]")
        _exact_keys(task, TASK_KEYS, f"historical tasks[{index}]")
        ref = _nonempty_string(task.get("ref"), f"historical tasks[{index}].ref")
        if REF_PATTERN.fullmatch(ref) is None:
            raise HistoricalCandidateError(f"historical task ref {ref!r} is outside mgc")
        if task.get("partition") != "train":
            raise HistoricalCandidateError(f"historical task {ref!r} is not train")
        _nonempty_string(task.get("prompt"), f"historical task {ref!r} prompt")
        _nonempty_string(task.get("gold"), f"historical task {ref!r} gold")
        if task.get("max_output_tokens") != 1024:
            raise HistoricalCandidateError(f"historical task {ref!r} changed its output budget")
        refs.append(ref)
    if len(set(refs)) != len(refs):
        raise HistoricalCandidateError("historical public corpus repeats a task ref")

    canonical = candidate.canonical_json_bytes(root)
    if len(canonical) != PUBLIC_CORPUS_CANONICAL_BYTES:
        raise HistoricalCandidateError("historical canonical JSON byte count changed")
    canonical_digest = candidate.digest_bytes(canonical)
    if expected_canonical_digest is not None and canonical_digest != expected_canonical_digest:
        raise HistoricalCandidateError(
            "historical canonical content changed: "
            f"expected {expected_canonical_digest}, got {canonical_digest}"
        )
    return candidate.CorpusValidation(
        canonical_digest=canonical_digest,
        canonical_bytes=len(canonical),
        task_count=len(tasks),
        reference_count=0,
        refs_digest=candidate.digest_bytes(candidate.canonical_json_bytes(sorted(refs))),
    )


def load_public_corpus(path: Path) -> tuple[dict[str, Any], candidate.CorpusValidation]:
    if path.is_symlink():
        raise HistoricalCandidateError(
            f"historical public corpus must be a regular non-symlink file: {path}"
        )
    source = candidate.assert_tmpfs_path(path, must_exist=True)
    if not source.is_file():
        raise HistoricalCandidateError(
            f"historical public corpus must be a regular non-symlink file: {path}"
        )
    if source.stat().st_size != PUBLIC_CORPUS_RESPONSE_BYTES:
        raise HistoricalCandidateError("historical public corpus raw byte count changed")
    if digest_file(source) != PUBLIC_CORPUS_RAW_DIGEST:
        raise HistoricalCandidateError("historical public corpus raw digest changed")
    payload = candidate._strict_json(source.read_bytes(), str(source))
    validation = validate_public_corpus_payload(payload)
    return dict(payload), validation


def source_corpus_identity() -> dict[str, Any]:
    """Return the constant identity revalidated before historical training."""

    return {
        "corpus_version": CORPUS_VERSION,
        "raw_bytes": PUBLIC_CORPUS_RESPONSE_BYTES,
        "raw_digest": PUBLIC_CORPUS_RAW_DIGEST,
        "canonical_bytes": PUBLIC_CORPUS_CANONICAL_BYTES,
        "canonical_digest": PUBLIC_CORPUS_CANONICAL_DIGEST,
    }


def _prepared_record(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "completion": task["gold"],
        "max_output_tokens": task["max_output_tokens"],
        "prompt": task["prompt"],
        "ref": task["ref"],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dataset(
    corpus_path: Path,
    output: Path,
    *,
    holdout_examples: int,
    seed: int,
) -> dict[str, Any]:
    if isinstance(holdout_examples, bool) or not isinstance(holdout_examples, int):
        raise HistoricalCandidateError("holdout_examples must be an integer")
    if not 0 <= holdout_examples < EXPECTED_COUNTS["train"]:
        raise HistoricalCandidateError("holdout_examples must leave at least one training task")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise HistoricalCandidateError("seed must be an integer")
    out = candidate.assert_tmpfs_path(output)
    if out.exists():
        raise HistoricalCandidateError(f"historical dataset output already exists: {out}")
    payload, validation = load_public_corpus(corpus_path)
    tasks = list(payload["tasks"])
    ranked = sorted(
        tasks,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['ref']}".encode()).hexdigest(),
            row["ref"],
        ),
    )
    heldout_refs = {str(row["ref"]) for row in ranked[:holdout_examples]}
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: str(row["ref"])):
        record = _prepared_record(task)
        (holdout_rows if task["ref"] in heldout_refs else train_rows).append(record)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        train_path = staging / "train.jsonl"
        holdout_path = staging / "holdout.jsonl"
        _write_jsonl(train_path, train_rows)
        _write_jsonl(holdout_path, holdout_rows)
        manifest = {
            "schema": DATASET_SCHEMA,
            "track": candidate.TRACK,
            "hardware_class": candidate.HARDWARE_CLASS,
            "corpus_version": CORPUS_VERSION,
            "corpus_canonical_digest": validation.canonical_digest,
            "source_file_digest": candidate.digest_file(corpus_path),
            "split_algorithm": candidate.SPLIT_ALGORITHM,
            "seed": seed,
            "train_examples": len(train_rows),
            "holdout_examples": len(holdout_rows),
            "train_refs_digest": candidate._refs_digest([str(row["ref"]) for row in train_rows]),
            "holdout_refs_digest": candidate._refs_digest(
                [str(row["ref"]) for row in holdout_rows]
            ),
            "train_file_digest": candidate.digest_file(train_path),
            "holdout_file_digest": candidate.digest_file(holdout_path),
            "target_construction": TARGET_CONSTRUCTION,
            "quality_claim": (
                DEVELOPMENT_QUALITY_CLAIM if holdout_rows else FINAL_ALL_PUBLIC_QUALITY_CLAIM
            ),
        }
        candidate._write_json(staging / "manifest.json", manifest)
        os.replace(staging, out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_prepared_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = candidate._strict_json(line.encode("utf-8"), f"{label}:{number}")
            if not isinstance(row, dict) or frozenset(row) != PREPARED_ROW_KEYS:
                raise HistoricalCandidateError(f"{label}:{number} has unexpected fields")
            ref = _nonempty_string(row["ref"], f"{label}:{number} ref")
            if REF_PATTERN.fullmatch(ref) is None:
                raise HistoricalCandidateError(f"{label}:{number} has an invalid historical ref")
            _nonempty_string(row["prompt"], f"{label}:{number} prompt")
            _nonempty_string(row["completion"], f"{label}:{number} completion")
            if row["max_output_tokens"] != 1024:
                raise HistoricalCandidateError(f"{label}:{number} changed its token budget")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalCandidateError(f"{label} is unreadable: {exc}") from exc
    return rows


def _load_prepared_rows(path: Path, label: str) -> list[dict[str, Any]]:
    return load_prepared_rows(path, label)


def _load_prepared_dataset_files(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = candidate.assert_tmpfs_path(path, must_exist=True)
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    holdout_path = root / "holdout.jsonl"
    files = (manifest_path, train_path, holdout_path)
    if any(item.is_symlink() or not item.is_file() for item in files):
        raise HistoricalCandidateError("historical prepared files must be regular non-symlinks")
    try:
        manifest = candidate._strict_json(manifest_path.read_bytes(), str(manifest_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalCandidateError(
            f"historical prepared manifest is unreadable: {exc}"
        ) from exc
    manifest_mapping = _mapping(manifest, "historical prepared manifest")
    _exact_keys(manifest_mapping, PREPARED_MANIFEST_KEYS, "historical prepared manifest")
    required = {
        "schema": DATASET_SCHEMA,
        "track": candidate.TRACK,
        "hardware_class": candidate.HARDWARE_CLASS,
        "corpus_version": CORPUS_VERSION,
        "corpus_canonical_digest": PUBLIC_CORPUS_CANONICAL_DIGEST,
        "source_file_digest": PUBLIC_CORPUS_RAW_DIGEST,
        "split_algorithm": candidate.SPLIT_ALGORITHM,
        "target_construction": TARGET_CONSTRUCTION,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise HistoricalCandidateError(f"historical prepared manifest field {key!r} changed")
    seed = manifest.get("seed")
    train_examples = manifest.get("train_examples")
    holdout_examples = manifest.get("holdout_examples")
    for label, value in (
        ("seed", seed),
        ("train_examples", train_examples),
        ("holdout_examples", holdout_examples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HistoricalCandidateError(f"historical prepared {label} must be non-negative")
    if train_examples + holdout_examples != EXPECTED_COUNTS["train"]:
        raise HistoricalCandidateError("historical prepared split does not contain all 8000 rows")
    if train_examples < 1:
        raise HistoricalCandidateError("historical prepared split has no training rows")
    expected_claim = (
        DEVELOPMENT_QUALITY_CLAIM if holdout_examples else FINAL_ALL_PUBLIC_QUALITY_CLAIM
    )
    if manifest.get("quality_claim") != expected_claim:
        raise HistoricalCandidateError("historical prepared quality claim changed")
    for data_path, digest_key in (
        (train_path, "train_file_digest"),
        (holdout_path, "holdout_file_digest"),
    ):
        if candidate.digest_file(data_path) != manifest.get(digest_key):
            raise HistoricalCandidateError(f"historical {data_path.name} digest changed")
    rows = load_prepared_rows(train_path, "train.jsonl")
    holdout_rows = load_prepared_rows(holdout_path, "holdout.jsonl")
    for found, count_key, refs_key, label in (
        (rows, "train_examples", "train_refs_digest", "training"),
        (holdout_rows, "holdout_examples", "holdout_refs_digest", "holdout"),
    ):
        if len(found) != manifest.get(count_key):
            raise HistoricalCandidateError(f"historical {label} row count changed")
        refs = [str(row["ref"]) for row in found]
        if len(set(refs)) != len(refs) or candidate._refs_digest(refs) != manifest.get(refs_key):
            raise HistoricalCandidateError(f"historical {label} refs changed")
    train_refs = {str(row["ref"]) for row in rows}
    holdout_refs = {str(row["ref"]) for row in holdout_rows}
    if train_refs & holdout_refs:
        raise HistoricalCandidateError("historical train and holdout refs overlap")
    if len(train_refs | holdout_refs) != EXPECTED_COUNTS["train"]:
        raise HistoricalCandidateError("historical prepared refs are incomplete")
    return rows, dict(manifest)


def replay_prepared_dataset(dataset_root: Path, corpus_path: Path) -> dict[str, Any]:
    """Revalidate raw bytes and replay the exact deterministic preparation in memory."""

    train_rows, manifest = _load_prepared_dataset_files(dataset_root)
    holdout_rows = load_prepared_rows(dataset_root / "holdout.jsonl", "holdout.jsonl")
    payload, _validation = load_public_corpus(corpus_path)
    tasks = list(payload["tasks"])
    holdout_examples = int(manifest["holdout_examples"])
    seed = int(manifest["seed"])
    ranked = sorted(
        tasks,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['ref']}".encode()).hexdigest(),
            row["ref"],
        ),
    )
    heldout_refs = {str(row["ref"]) for row in ranked[:holdout_examples]}
    expected_train: list[dict[str, Any]] = []
    expected_holdout: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: str(row["ref"])):
        record = _prepared_record(task)
        (expected_holdout if task["ref"] in heldout_refs else expected_train).append(record)
    if train_rows != expected_train or holdout_rows != expected_holdout:
        raise HistoricalCandidateError(
            "historical prepared rows are not an exact replay of the pinned public corpus"
        )
    return source_corpus_identity()


def load_prepared_dataset(
    path: Path,
    corpus_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load prepared training rows only after exact pinned-source replay."""

    rows, manifest = _load_prepared_dataset_files(path)

    replay_prepared_dataset(path, corpus_path)
    return rows, manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate", help="validate the exact historical response")
    validate.add_argument("corpus", type=Path)
    prepare = actions.add_parser("prepare", help="prepare verbatim-gold SFT JSONL")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--holdout-examples", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=92)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.action == "validate":
            _payload, validation = load_public_corpus(args.corpus)
            result: Any = {
                "canonical_digest": validation.canonical_digest,
                "canonical_bytes": validation.canonical_bytes,
                "task_count": validation.task_count,
                "reference_count": validation.reference_count,
                "refs_digest": validation.refs_digest,
            }
        else:
            result = prepare_dataset(
                args.corpus,
                args.output,
                holdout_examples=args.holdout_examples,
                seed=args.seed,
            )
    except (candidate.CandidateError, OSError) as exc:
        raise SystemExit(f"historical code candidate refused: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
