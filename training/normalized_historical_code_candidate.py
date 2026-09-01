#!/usr/bin/env python3
"""Prepare one exact code-only projection of the historical public corpus.

The existing :mod:`training.historical_code_candidate` profile intentionally
preserves all 8,000 public ``gold`` strings verbatim.  This separate profile
does not change that contract.  It deterministically selects the longest
non-empty, statically parseable scorer-compatible fenced block from each gold
string and excludes rows without one.  It also excludes three exact PEP 701
targets that Python 3.12 accepts but the pinned Python 3.11 training runtime
rejects, making the projection identical on both runtimes.

Corpus text is treated only as data.  Candidate blocks are inspected with
``ast.parse(..., mode="exec")``; no corpus code is imported, bytecode-compiled,
evaluated, or executed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

try:
    from training import code_candidate as candidate
    from training import historical_code_candidate as historical
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import code_candidate as candidate  # type: ignore[no-redef]
    import historical_code_candidate as historical  # type: ignore[no-redef]


CORPUS_PROFILE: Final[str] = "historical7730-normalized-v1"
DATASET_SCHEMA: Final[str] = "microtensor.code.prepared.historical-normalized.v1"
NORMALIZATION_SCHEMA: Final[str] = "microtensor.code.historical-normalization.v1"
NORMALIZATION_ALGORITHM: Final[str] = (
    "longest_nonempty_statically_parseable_scorer_compatible_fenced_block_v1"
)
TARGET_CONSTRUCTION: Final[str] = NORMALIZATION_ALGORITHM
TRAINING_TARGET_CONSTRUCTION: Final[str] = (
    "raw prompt -> normalized non-empty statically parseable Python from the longest "
    "scorer-compatible fenced public-gold block"
)
FINAL_ALL_PUBLIC_QUALITY_CLAIM: Final[str] = (
    "none: all 7730 normalized public training examples were used; 270 source rows were "
    "deterministically excluded, including three exact Python-3.11-incompatible PEP 701 "
    "targets; no holdout or execution pass@1 was measured"
)
FINAL_ALL_PUBLIC_HOLDOUT_CLAIM: Final[str] = (
    "all 7730 normalized public examples were training inputs; 270 source rows were excluded "
    "by the pinned cross-runtime static normalization; no holdout or execution pass@1 was "
    "measured"
)

TRACK: Final[str] = historical.TRACK
HARDWARE_CLASS: Final[str] = historical.HARDWARE_CLASS
CORPUS_VERSION: Final[str] = historical.CORPUS_VERSION
PUBLIC_CORPUS_URL: Final[str] = historical.PUBLIC_CORPUS_URL
PUBLIC_CORPUS_RESPONSE_BYTES: Final[int] = historical.PUBLIC_CORPUS_RESPONSE_BYTES
PUBLIC_CORPUS_RAW_DIGEST: Final[str] = historical.PUBLIC_CORPUS_RAW_DIGEST
PUBLIC_CORPUS_CANONICAL_BYTES: Final[int] = historical.PUBLIC_CORPUS_CANONICAL_BYTES
PUBLIC_CORPUS_CANONICAL_DIGEST: Final[str] = historical.PUBLIC_CORPUS_CANONICAL_DIGEST
REF_PATTERN: Final[re.Pattern[str]] = historical.REF_PATTERN
PREPARED_ROW_KEYS: Final[frozenset[str]] = historical.PREPARED_ROW_KEYS

EXPECTED_SEED: Final[int] = 92
EXPECTED_SOURCE_EXAMPLES: Final[int] = 8_000
EXPECTED_TRAIN_EXAMPLES: Final[int] = 7_730
EXPECTED_HOLDOUT_EXAMPLES: Final[int] = 0
EXPECTED_EXCLUDED_EXAMPLES: Final[int] = 270
EXPECTED_TRAIN_FILE_BYTES: Final[int] = 15_681_824
EXPECTED_TRAIN_FILE_DIGEST: Final[str] = (
    "sha256:10fd0cc986802fc78e5ac39384fab1f109401a95fcec1af5e2ce9c3f0efa4e03"
)
EXPECTED_HOLDOUT_FILE_BYTES: Final[int] = 0
EXPECTED_HOLDOUT_FILE_DIGEST: Final[str] = (
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
EXPECTED_TRAIN_REFS_CANONICAL_BYTES: Final[int] = 90_610
EXPECTED_TRAIN_REFS_DIGEST: Final[str] = (
    "sha256:7ca162ac14570f270952ab4d5872b02fbdbc370ff84ea56b81793c48d89bdfc5"
)
EXPECTED_HOLDOUT_REFS_DIGEST: Final[str] = (
    "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES: Final[int] = 3_184
EXPECTED_EXCLUDED_REFS_DIGEST: Final[str] = (
    "sha256:03859ad7b36efe69a3a202ad203697490c50de810a2ff51e00d2abb32d96f35d"
)
EXPECTED_TARGET_PAIRS_CANONICAL_BYTES: Final[int] = 5_731_044
EXPECTED_TARGET_PAIRS_DIGEST: Final[str] = (
    "sha256:8518c0bffc7d7c2122e2ffa896dce82b9c13a40ae0f06c0392acc0384552e79c"
)
EXPECTED_RAW_COMPLETION_UTF8_BYTES: Final[int] = 5_270_659
EXCLUDED_REFS_FILE: Final[str] = "excluded-refs.json"
FINAL_ALL_PUBLIC_SPLIT_ALGORITHM: Final[str] = "ref_ascending_all_normalized_rows_v1"
SCORER_FENCE_PATTERN: Final[str] = r"```(?:python|py)?\s*\n(.*?)```"
SCORER_FENCE: Final[re.Pattern[str]] = re.compile(SCORER_FENCE_PATTERN, re.DOTALL)
AST_FEATURE_VERSION: Final[tuple[int, int]] = (3, 10)
PYTHON311_INCOMPATIBLE_TARGETS: Final[tuple[tuple[str, str], ...]] = (
    (
        "mgc-10018",
        "sha256:709b3adf209408ef80461dda77447e71a7c5cb03a8ec3a09cd6ac2dc29407476",
    ),
    (
        "mgc-22547",
        "sha256:8ea9945d886739e48ee950633e7d76e8633c5a79f7c184a6ecaf229d784d0607",
    ),
    (
        "mgc-31043",
        "sha256:acadafb7075c4ee1f332bd13efa86b34d9e4d13eca3b16498044f538f08c39bd",
    ),
)
PYTHON311_INCOMPATIBLE_TARGET_DIGESTS: Final[frozenset[str]] = frozenset(
    digest for _ref, digest in PYTHON311_INCOMPATIBLE_TARGETS
)
NORMALIZATION_CONTRACT: Final[dict[str, Any]] = {
    "schema": NORMALIZATION_SCHEMA,
    "algorithm": NORMALIZATION_ALGORITHM,
    "fence_pattern": SCORER_FENCE_PATTERN,
    "regex_flags": ["DOTALL"],
    "candidate_whitespace": "str.strip",
    "candidate_nonempty_required": True,
    "static_parser": "ast.parse",
    "static_parser_mode": "exec",
    "ast_feature_version": list(AST_FEATURE_VERSION),
    "python311_compatibility_exclusions": [
        {"ref": ref, "normalized_target_digest": digest}
        for ref, digest in PYTHON311_INCOMPATIBLE_TARGETS
    ],
    "compatibility_exclusion_order": "SHA-256 of stripped UTF-8 block before ast.parse",
    "acceptance_allowlist": False,
    "selection": "maximum Python character length; earliest source-order tie",
    "row_without_candidate": "exclude",
    "corpus_code_imported": False,
    "corpus_code_bytecode_compiled": False,
    "corpus_code_executed": False,
}
PREPARED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "track",
        "hardware_class",
        "corpus_profile",
        "corpus_version",
        "corpus_canonical_digest",
        "source_file_digest",
        "split_algorithm",
        "seed",
        "source_examples",
        "train_examples",
        "holdout_examples",
        "excluded_examples",
        "train_refs_canonical_bytes",
        "train_refs_digest",
        "holdout_refs_digest",
        "excluded_refs_file",
        "excluded_refs_canonical_bytes",
        "excluded_refs_digest",
        "train_file_bytes",
        "train_file_digest",
        "holdout_file_bytes",
        "holdout_file_digest",
        "target_pairs_canonical_bytes",
        "target_pairs_digest",
        "raw_completion_utf8_bytes",
        "target_construction",
        "normalization",
        "quality_claim",
    }
)
DATASET_FILES: Final[frozenset[str]] = frozenset(
    {"manifest.json", "train.jsonl", "holdout.jsonl", EXCLUDED_REFS_FILE}
)


class NormalizedHistoricalCandidateError(candidate.CandidateError):
    """The normalized historical projection violates its exact static contract."""


canonical_json_bytes = candidate.canonical_json_bytes
digest_bytes = candidate.digest_bytes
digest_file = candidate.digest_file
_refs_digest = candidate._refs_digest


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise NormalizedHistoricalCandidateError(f"{label} contains a non-string key")
    found = frozenset(value)
    if found != expected:
        raise NormalizedHistoricalCandidateError(
            f"{label} keys changed: expected {sorted(expected)}, got {sorted(found)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NormalizedHistoricalCandidateError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NormalizedHistoricalCandidateError(f"{label} must be a non-empty string")
    return value


def _parseable_python(source: str) -> bool:
    """Return static Python syntax status without importing or executing source."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(
                source,
                filename="<normalized-historical-static>",
                mode="exec",
                feature_version=AST_FEATURE_VERSION,
            )
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError, OverflowError):
        return False
    return True


def normalize_gold(gold: str) -> str | None:
    """Select the pinned scorer-compatible code block using AST inspection only."""

    if not isinstance(gold, str):
        raise NormalizedHistoricalCandidateError("historical gold must be a string")
    candidates: list[tuple[int, str]] = []
    for index, block in enumerate(SCORER_FENCE.findall(gold)):
        source = block.strip()
        if not source:
            continue
        # Python 3.12's PEP 701 parser accepts three exact public targets that
        # Python 3.11 (the pinned training runtime) rejects.  Hash exclusion is
        # deliberately performed before AST parsing so both runtimes project
        # the same rows without executing or importing corpus content.
        if digest_bytes(source.encode("utf-8")) in PYTHON311_INCOMPATIBLE_TARGET_DIGESTS:
            continue
        if _parseable_python(source):
            candidates.append((index, source))
    if not candidates:
        return None
    # Python's max keeps the first item on a key tie, giving the declared
    # earliest-source-order tie break without a second ordering rule.
    return max(candidates, key=lambda item: len(item[1]))[1]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _target_pairs_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    pairs = [{"completion": row["completion"], "ref": row["ref"]} for row in rows]
    return canonical_json_bytes(pairs)


def _excluded_refs_bytes(refs: Sequence[str]) -> bytes:
    return canonical_json_bytes(list(refs))


def _normalization_contract_copy() -> dict[str, Any]:
    """Return a detached strict-JSON copy so callers cannot mutate the contract."""

    copied = candidate._strict_json(
        canonical_json_bytes(NORMALIZATION_CONTRACT),
        "normalization contract",
    )
    if not isinstance(copied, dict):
        raise NormalizedHistoricalCandidateError("normalization contract copy is malformed")
    return copied


def _normalize_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    excluded_refs: list[str] = []
    seen_refs: set[str] = set()
    for index, raw_task in enumerate(tasks):
        task = _mapping(raw_task, f"normalization tasks[{index}]")
        ref = _nonempty_string(task.get("ref"), f"normalization tasks[{index}] ref")
        if REF_PATTERN.fullmatch(ref) is None:
            raise NormalizedHistoricalCandidateError(f"normalization ref {ref!r} is outside mgc")
        if ref in seen_refs:
            raise NormalizedHistoricalCandidateError(f"normalization repeats ref {ref!r}")
        seen_refs.add(ref)
        prompt = _nonempty_string(task.get("prompt"), f"normalization task {ref!r} prompt")
        gold = _nonempty_string(task.get("gold"), f"normalization task {ref!r} gold")
        if task.get("max_output_tokens") != 1024:
            raise NormalizedHistoricalCandidateError(
                f"normalization task {ref!r} changed its output budget"
            )
        completion = normalize_gold(gold)
        if completion is None:
            excluded_refs.append(ref)
            continue
        train_rows.append(
            {
                "completion": completion,
                "max_output_tokens": 1024,
                "prompt": prompt,
                "ref": ref,
            }
        )

    train_rows.sort(key=lambda row: str(row["ref"]))
    excluded_refs.sort()
    train_bytes = _jsonl_bytes(train_rows)
    refs_bytes = canonical_json_bytes([str(row["ref"]) for row in train_rows])
    excluded_bytes = _excluded_refs_bytes(excluded_refs)
    target_pairs = _target_pairs_bytes(train_rows)
    return {
        "train_rows": train_rows,
        "excluded_refs": excluded_refs,
        "train_bytes": train_bytes,
        "train_file_digest": digest_bytes(train_bytes),
        "train_refs_canonical_bytes": len(refs_bytes),
        "train_refs_digest": digest_bytes(refs_bytes),
        "excluded_bytes": excluded_bytes,
        "excluded_refs_digest": digest_bytes(excluded_bytes),
        "target_pairs_canonical_bytes": len(target_pairs),
        "target_pairs_digest": digest_bytes(target_pairs),
        "raw_completion_utf8_bytes": sum(
            len(str(row["completion"]).encode("utf-8")) for row in train_rows
        ),
    }


def _validate_pinned_normalization(result: Mapping[str, Any]) -> None:
    train_rows = result.get("train_rows")
    excluded_refs = result.get("excluded_refs")
    train_bytes = result.get("train_bytes")
    excluded_bytes = result.get("excluded_bytes")
    if not isinstance(train_rows, list) or not isinstance(excluded_refs, list):
        raise NormalizedHistoricalCandidateError("normalization result rows are malformed")
    if not isinstance(train_bytes, bytes) or not isinstance(excluded_bytes, bytes):
        raise NormalizedHistoricalCandidateError("normalization result bytes are malformed")
    observed = {
        "source examples": len(train_rows) + len(excluded_refs),
        "training examples": len(train_rows),
        "excluded examples": len(excluded_refs),
        "train file bytes": len(train_bytes),
        "train refs canonical bytes": result.get("train_refs_canonical_bytes"),
        "excluded refs canonical bytes": len(excluded_bytes),
        "target pairs canonical bytes": result.get("target_pairs_canonical_bytes"),
        "raw completion UTF-8 bytes": result.get("raw_completion_utf8_bytes"),
    }
    expected = {
        "source examples": EXPECTED_SOURCE_EXAMPLES,
        "training examples": EXPECTED_TRAIN_EXAMPLES,
        "excluded examples": EXPECTED_EXCLUDED_EXAMPLES,
        "train file bytes": EXPECTED_TRAIN_FILE_BYTES,
        "train refs canonical bytes": EXPECTED_TRAIN_REFS_CANONICAL_BYTES,
        "excluded refs canonical bytes": EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
        "target pairs canonical bytes": EXPECTED_TARGET_PAIRS_CANONICAL_BYTES,
        "raw completion UTF-8 bytes": EXPECTED_RAW_COMPLETION_UTF8_BYTES,
    }
    for label, wanted in expected.items():
        if observed[label] != wanted:
            raise NormalizedHistoricalCandidateError(
                f"normalized historical {label} changed: expected {wanted}, got {observed[label]}"
            )
    digests = {
        "train file digest": (result.get("train_file_digest"), EXPECTED_TRAIN_FILE_DIGEST),
        "train refs digest": (result.get("train_refs_digest"), EXPECTED_TRAIN_REFS_DIGEST),
        "excluded refs digest": (
            result.get("excluded_refs_digest"),
            EXPECTED_EXCLUDED_REFS_DIGEST,
        ),
        "target pairs digest": (
            result.get("target_pairs_digest"),
            EXPECTED_TARGET_PAIRS_DIGEST,
        ),
    }
    for label, (found, wanted) in digests.items():
        if found != wanted:
            raise NormalizedHistoricalCandidateError(
                f"normalized historical {label} changed: expected {wanted}, got {found}"
            )


def load_public_corpus(path: Path) -> tuple[dict[str, Any], candidate.CorpusValidation]:
    """Load only the exact historical source accepted by the existing profile."""

    try:
        return historical.load_public_corpus(path)
    except historical.HistoricalCandidateError as exc:
        raise NormalizedHistoricalCandidateError(str(exc)) from exc


def source_corpus_identity() -> dict[str, Any]:
    """Return exact source and normalization identities for training provenance."""

    return {
        **historical.source_corpus_identity(),
        "profile": CORPUS_PROFILE,
        "source_examples": EXPECTED_SOURCE_EXAMPLES,
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "excluded_examples": EXPECTED_EXCLUDED_EXAMPLES,
        "excluded_refs_digest": EXPECTED_EXCLUDED_REFS_DIGEST,
        "target_pairs_digest": EXPECTED_TARGET_PAIRS_DIGEST,
        "normalization": _normalization_contract_copy(),
    }


def _validate_final_arguments(*, holdout_examples: int, seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != EXPECTED_SEED:
        raise NormalizedHistoricalCandidateError(
            f"normalized historical seed must equal {EXPECTED_SEED}"
        )
    if (
        isinstance(holdout_examples, bool)
        or not isinstance(holdout_examples, int)
        or holdout_examples != EXPECTED_HOLDOUT_EXAMPLES
    ):
        raise NormalizedHistoricalCandidateError(
            "normalized historical profile is final-only and requires zero holdout examples"
        )


def _manifest(
    *,
    source_file_digest: str,
    corpus_canonical_digest: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    train_bytes = result["train_bytes"]
    excluded_bytes = result["excluded_bytes"]
    if not isinstance(train_bytes, bytes) or not isinstance(excluded_bytes, bytes):
        raise NormalizedHistoricalCandidateError("normalization byte identities are malformed")
    return {
        "schema": DATASET_SCHEMA,
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "corpus_profile": CORPUS_PROFILE,
        "corpus_version": CORPUS_VERSION,
        "corpus_canonical_digest": corpus_canonical_digest,
        "source_file_digest": source_file_digest,
        "split_algorithm": FINAL_ALL_PUBLIC_SPLIT_ALGORITHM,
        "seed": EXPECTED_SEED,
        "source_examples": EXPECTED_SOURCE_EXAMPLES,
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "holdout_examples": EXPECTED_HOLDOUT_EXAMPLES,
        "excluded_examples": EXPECTED_EXCLUDED_EXAMPLES,
        "train_refs_canonical_bytes": result["train_refs_canonical_bytes"],
        "train_refs_digest": result["train_refs_digest"],
        "holdout_refs_digest": EXPECTED_HOLDOUT_REFS_DIGEST,
        "excluded_refs_file": EXCLUDED_REFS_FILE,
        "excluded_refs_canonical_bytes": len(excluded_bytes),
        "excluded_refs_digest": result["excluded_refs_digest"],
        "train_file_bytes": len(train_bytes),
        "train_file_digest": result["train_file_digest"],
        "holdout_file_bytes": EXPECTED_HOLDOUT_FILE_BYTES,
        "holdout_file_digest": EXPECTED_HOLDOUT_FILE_DIGEST,
        "target_pairs_canonical_bytes": result["target_pairs_canonical_bytes"],
        "target_pairs_digest": result["target_pairs_digest"],
        "raw_completion_utf8_bytes": result["raw_completion_utf8_bytes"],
        "target_construction": TARGET_CONSTRUCTION,
        "normalization": _normalization_contract_copy(),
        "quality_claim": FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }


def prepare_dataset(
    corpus_path: Path,
    output: Path,
    *,
    holdout_examples: int = EXPECTED_HOLDOUT_EXAMPLES,
    seed: int = EXPECTED_SEED,
) -> dict[str, Any]:
    """Create the exact 7,730/0 normalized dataset below volatile tmpfs."""

    _validate_final_arguments(holdout_examples=holdout_examples, seed=seed)
    if output.is_symlink():
        raise NormalizedHistoricalCandidateError(
            f"normalized dataset output must not be a symlink: {output}"
        )
    out = candidate.assert_tmpfs_path(output)
    if out.exists() or out.is_symlink():
        raise NormalizedHistoricalCandidateError(
            f"normalized historical dataset output already exists: {out}"
        )
    payload, validation = load_public_corpus(corpus_path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise NormalizedHistoricalCandidateError("historical source tasks must be an array")
    result = _normalize_tasks(tasks)
    _validate_pinned_normalization(result)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        train_bytes = result["train_bytes"]
        excluded_bytes = result["excluded_bytes"]
        if not isinstance(train_bytes, bytes) or not isinstance(excluded_bytes, bytes):
            raise NormalizedHistoricalCandidateError("normalization bytes are malformed")
        (staging / "train.jsonl").write_bytes(train_bytes)
        (staging / "holdout.jsonl").write_bytes(b"")
        (staging / EXCLUDED_REFS_FILE).write_bytes(excluded_bytes)
        manifest = _manifest(
            source_file_digest=digest_file(corpus_path),
            corpus_canonical_digest=validation.canonical_digest,
            result=result,
        )
        candidate._write_json(staging / "manifest.json", manifest)
        os.replace(staging, out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_prepared_rows(path: Path, label: str) -> list[dict[str, Any]]:
    """Load normalized rows and re-run AST syntax inspection without execution."""

    if path.is_symlink() or not path.is_file():
        raise NormalizedHistoricalCandidateError(f"{label} must be a regular non-symlink file")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                raise NormalizedHistoricalCandidateError(f"{label}:{number} is blank")
            row = candidate._strict_json(line.encode("utf-8"), f"{label}:{number}")
            if not isinstance(row, dict) or frozenset(row) != PREPARED_ROW_KEYS:
                raise NormalizedHistoricalCandidateError(f"{label}:{number} has unexpected fields")
            ref = _nonempty_string(row["ref"], f"{label}:{number} ref")
            if REF_PATTERN.fullmatch(ref) is None:
                raise NormalizedHistoricalCandidateError(
                    f"{label}:{number} has an invalid historical ref"
                )
            _nonempty_string(row["prompt"], f"{label}:{number} prompt")
            completion = _nonempty_string(row["completion"], f"{label}:{number} completion")
            if not _parseable_python(completion):
                raise NormalizedHistoricalCandidateError(
                    f"{label}:{number} completion is not statically parseable Python"
                )
            if row["max_output_tokens"] != 1024:
                raise NormalizedHistoricalCandidateError(
                    f"{label}:{number} changed its token budget"
                )
            rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NormalizedHistoricalCandidateError(f"{label} is unreadable: {exc}") from exc
    return rows


def _load_prepared_rows(path: Path, label: str) -> list[dict[str, Any]]:
    return load_prepared_rows(path, label)


def _load_excluded_refs(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise NormalizedHistoricalCandidateError("excluded refs must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) != EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES:
        raise NormalizedHistoricalCandidateError("excluded refs byte count changed")
    if digest_bytes(raw) != EXPECTED_EXCLUDED_REFS_DIGEST:
        raise NormalizedHistoricalCandidateError("excluded refs digest changed")
    payload = candidate._strict_json(raw, str(path))
    if not isinstance(payload, list) or any(not isinstance(ref, str) for ref in payload):
        raise NormalizedHistoricalCandidateError("excluded refs must be a string array")
    refs = list(payload)
    if refs != sorted(refs) or len(set(refs)) != len(refs):
        raise NormalizedHistoricalCandidateError("excluded refs are not sorted and unique")
    if any(REF_PATTERN.fullmatch(ref) is None for ref in refs):
        raise NormalizedHistoricalCandidateError("excluded refs contain a ref outside mgc")
    if canonical_json_bytes(refs) != raw:
        raise NormalizedHistoricalCandidateError("excluded refs are not canonical JSON")
    if len(refs) != EXPECTED_EXCLUDED_EXAMPLES:
        raise NormalizedHistoricalCandidateError("excluded refs count changed")
    return refs


def _load_prepared_dataset_files(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if path.is_symlink():
        raise NormalizedHistoricalCandidateError(
            f"normalized prepared root must not be a symlink: {path}"
        )
    root = candidate.assert_tmpfs_path(path, must_exist=True)
    if not root.is_dir():
        raise NormalizedHistoricalCandidateError("normalized prepared root is not a directory")
    try:
        names = frozenset(item.name for item in root.iterdir())
    except OSError as exc:
        raise NormalizedHistoricalCandidateError(
            f"normalized prepared root is unreadable: {exc}"
        ) from exc
    if names != DATASET_FILES:
        raise NormalizedHistoricalCandidateError(
            f"normalized prepared file set changed: expected {sorted(DATASET_FILES)}, "
            f"got {sorted(names)}"
        )
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    holdout_path = root / "holdout.jsonl"
    excluded_path = root / EXCLUDED_REFS_FILE
    files = (manifest_path, train_path, holdout_path, excluded_path)
    if any(item.is_symlink() or not item.is_file() for item in files):
        raise NormalizedHistoricalCandidateError(
            "normalized prepared files must be regular non-symlinks"
        )
    try:
        manifest = candidate._strict_json(manifest_path.read_bytes(), str(manifest_path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NormalizedHistoricalCandidateError(
            f"normalized prepared manifest is unreadable: {exc}"
        ) from exc
    manifest_mapping = _mapping(manifest, "normalized prepared manifest")
    _exact_keys(manifest_mapping, PREPARED_MANIFEST_KEYS, "normalized prepared manifest")
    required = {
        "schema": DATASET_SCHEMA,
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "corpus_profile": CORPUS_PROFILE,
        "corpus_version": CORPUS_VERSION,
        "corpus_canonical_digest": PUBLIC_CORPUS_CANONICAL_DIGEST,
        "source_file_digest": PUBLIC_CORPUS_RAW_DIGEST,
        "split_algorithm": FINAL_ALL_PUBLIC_SPLIT_ALGORITHM,
        "seed": EXPECTED_SEED,
        "source_examples": EXPECTED_SOURCE_EXAMPLES,
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "holdout_examples": EXPECTED_HOLDOUT_EXAMPLES,
        "excluded_examples": EXPECTED_EXCLUDED_EXAMPLES,
        "train_refs_canonical_bytes": EXPECTED_TRAIN_REFS_CANONICAL_BYTES,
        "train_refs_digest": EXPECTED_TRAIN_REFS_DIGEST,
        "holdout_refs_digest": EXPECTED_HOLDOUT_REFS_DIGEST,
        "excluded_refs_file": EXCLUDED_REFS_FILE,
        "excluded_refs_canonical_bytes": EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
        "excluded_refs_digest": EXPECTED_EXCLUDED_REFS_DIGEST,
        "train_file_bytes": EXPECTED_TRAIN_FILE_BYTES,
        "train_file_digest": EXPECTED_TRAIN_FILE_DIGEST,
        "holdout_file_bytes": EXPECTED_HOLDOUT_FILE_BYTES,
        "holdout_file_digest": EXPECTED_HOLDOUT_FILE_DIGEST,
        "target_pairs_canonical_bytes": EXPECTED_TARGET_PAIRS_CANONICAL_BYTES,
        "target_pairs_digest": EXPECTED_TARGET_PAIRS_DIGEST,
        "raw_completion_utf8_bytes": EXPECTED_RAW_COMPLETION_UTF8_BYTES,
        "target_construction": TARGET_CONSTRUCTION,
        "normalization": _normalization_contract_copy(),
        "quality_claim": FINAL_ALL_PUBLIC_QUALITY_CLAIM,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise NormalizedHistoricalCandidateError(
                f"normalized prepared manifest field {key!r} changed"
            )
    identities = (
        (train_path, EXPECTED_TRAIN_FILE_BYTES, EXPECTED_TRAIN_FILE_DIGEST, "train file"),
        (
            holdout_path,
            EXPECTED_HOLDOUT_FILE_BYTES,
            EXPECTED_HOLDOUT_FILE_DIGEST,
            "holdout file",
        ),
        (
            excluded_path,
            EXPECTED_EXCLUDED_REFS_CANONICAL_BYTES,
            EXPECTED_EXCLUDED_REFS_DIGEST,
            "excluded refs",
        ),
    )
    for data_path, expected_bytes, expected_digest, label in identities:
        if data_path.stat().st_size != expected_bytes:
            raise NormalizedHistoricalCandidateError(f"normalized {label} byte count changed")
        if digest_file(data_path) != expected_digest:
            raise NormalizedHistoricalCandidateError(f"normalized {label} digest changed")

    rows = load_prepared_rows(train_path, "train.jsonl")
    holdout_rows = load_prepared_rows(holdout_path, "holdout.jsonl")
    excluded_refs = _load_excluded_refs(excluded_path)
    if len(rows) != EXPECTED_TRAIN_EXAMPLES or holdout_rows:
        raise NormalizedHistoricalCandidateError("normalized prepared split is not 7730/0")
    refs = [str(row["ref"]) for row in rows]
    if refs != sorted(refs) or len(set(refs)) != len(refs):
        raise NormalizedHistoricalCandidateError(
            "normalized training refs are not sorted and unique"
        )
    if len(canonical_json_bytes(refs)) != EXPECTED_TRAIN_REFS_CANONICAL_BYTES:
        raise NormalizedHistoricalCandidateError("normalized training refs byte count changed")
    if _refs_digest(refs) != EXPECTED_TRAIN_REFS_DIGEST:
        raise NormalizedHistoricalCandidateError("normalized training refs digest changed")
    if set(refs) & set(excluded_refs):
        raise NormalizedHistoricalCandidateError("normalized retained and excluded refs overlap")
    if len(refs) + len(excluded_refs) != EXPECTED_SOURCE_EXAMPLES:
        raise NormalizedHistoricalCandidateError(
            "normalized retained and excluded refs are incomplete"
        )
    target_pairs = _target_pairs_bytes(rows)
    if len(target_pairs) != EXPECTED_TARGET_PAIRS_CANONICAL_BYTES:
        raise NormalizedHistoricalCandidateError("normalized target pairs byte count changed")
    if digest_bytes(target_pairs) != EXPECTED_TARGET_PAIRS_DIGEST:
        raise NormalizedHistoricalCandidateError("normalized target pairs digest changed")
    completion_bytes = sum(len(str(row["completion"]).encode("utf-8")) for row in rows)
    if completion_bytes != EXPECTED_RAW_COMPLETION_UTF8_BYTES:
        raise NormalizedHistoricalCandidateError("normalized completion byte count changed")
    return rows, dict(manifest), excluded_refs


def replay_prepared_dataset(dataset_root: Path, corpus_path: Path) -> dict[str, Any]:
    """Recompute normalization from exact source and compare every prepared row."""

    rows, _manifest_payload, excluded_refs = _load_prepared_dataset_files(dataset_root)
    payload, _validation = load_public_corpus(corpus_path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise NormalizedHistoricalCandidateError("historical source tasks must be an array")
    expected = _normalize_tasks(tasks)
    _validate_pinned_normalization(expected)
    if rows != expected["train_rows"] or excluded_refs != expected["excluded_refs"]:
        raise NormalizedHistoricalCandidateError(
            "normalized prepared rows or exclusions are not an exact replay of the pinned source"
        )
    return source_corpus_identity()


def load_prepared_dataset(
    path: Path,
    corpus_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load training rows only after exact source-normalization replay."""

    rows, manifest, _excluded_refs = _load_prepared_dataset_files(path)
    replay_prepared_dataset(path, corpus_path)
    return rows, manifest


def validate_source_projection(corpus_path: Path) -> dict[str, Any]:
    """Validate exact source bytes and all pinned normalization identities in memory."""

    payload, validation = load_public_corpus(corpus_path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise NormalizedHistoricalCandidateError("historical source tasks must be an array")
    result = _normalize_tasks(tasks)
    _validate_pinned_normalization(result)
    return {
        "corpus_canonical_bytes": validation.canonical_bytes,
        "corpus_canonical_digest": validation.canonical_digest,
        "source_examples": EXPECTED_SOURCE_EXAMPLES,
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "holdout_examples": EXPECTED_HOLDOUT_EXAMPLES,
        "excluded_examples": EXPECTED_EXCLUDED_EXAMPLES,
        "train_file_bytes": len(result["train_bytes"]),
        "train_file_digest": result["train_file_digest"],
        "train_refs_digest": result["train_refs_digest"],
        "excluded_refs_digest": result["excluded_refs_digest"],
        "target_pairs_digest": result["target_pairs_digest"],
        "raw_completion_utf8_bytes": result["raw_completion_utf8_bytes"],
        "normalization": _normalization_contract_copy(),
        "safety": {
            "corpus_code_imported": False,
            "corpus_code_bytecode_compiled": False,
            "corpus_code_executed": False,
            "static_ast_parse_only": True,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate", help="validate the exact normalized projection")
    validate.add_argument("corpus", type=Path)
    prepare = actions.add_parser("prepare", help="prepare the final 7730/0 normalized JSONL")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--holdout-examples", type=int, default=EXPECTED_HOLDOUT_EXAMPLES)
    prepare.add_argument("--seed", type=int, default=EXPECTED_SEED)
    replay = actions.add_parser("replay", help="replay a prepared dataset from exact source")
    replay.add_argument("--corpus", type=Path, required=True)
    replay.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.action == "validate":
            result: Any = validate_source_projection(args.corpus)
        elif args.action == "prepare":
            result = prepare_dataset(
                args.corpus,
                args.output,
                holdout_examples=args.holdout_examples,
                seed=args.seed,
            )
        else:
            result = replay_prepared_dataset(args.dataset, args.corpus)
    except (candidate.CandidateError, OSError) as exc:
        raise SystemExit(f"normalized historical code candidate refused: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
