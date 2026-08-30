#!/usr/bin/env python3
"""Dependency-free row selection shared by the local evaluation scripts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from training import boundary_contrastive as boundary
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import boundary_contrastive as boundary


SELECTION_SCHEMA = "microtensor.evaluation_selection.v1"
LEGACY_SELECTION = "legacy-shuffle-v1"
BOUNDARY_INNER_SELECTION = "boundary-inner-v1"
SELECTION_MODES = (LEGACY_SELECTION, BOUNDARY_INNER_SELECTION)
LEGACY_ALGORITHM = "python_random_mt19937_shuffle_prefix_v1"
CORPUS_VERSION = boundary.CORPUS_VERSION
BOUNDARY_TOKENIZER_SKIPPED_REFS = (
    "bc5cdr-train-03995",
    "bc5cdr-train-04805",
)


@dataclass(frozen=True)
class EvaluationSelection:
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _load_legacy_rows(payload: bytes) -> list[dict[str, Any]]:
    try:
        corpus = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("corpus is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("corpus is not valid JSON") from exc
    if not isinstance(corpus, dict) or str(corpus.get("version")) != CORPUS_VERSION:
        version = corpus.get("version") if isinstance(corpus, dict) else None
        raise ValueError(f"expected corpus {CORPUS_VERSION}, got {version}")
    tasks = corpus.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("corpus tasks must be a list")
    rows = [
        row
        for row in tasks
        if isinstance(row, dict) and row.get("partition") == "train"
    ]
    if len(rows) != boundary.BOUNDARY_CORPUS_TRAIN_EXAMPLES:
        raise ValueError(f"expected 4816 public train rows, found {len(rows)}")
    return rows


def _boundary_inner_selection(payload: bytes) -> EvaluationSelection:
    if tuple(boundary.BOUNDARY_SKIPPED_ENCODED_REFS) != BOUNDARY_TOKENIZER_SKIPPED_REFS:
        raise ValueError("boundary tokenizer skip allowlist does not match the evaluator")

    rows = boundary.parse_boundary_corpus(payload)
    outer_rows, remaining_rows = boundary.boundary_outer_partition(rows)
    outer_refs = [str(row["ref"]) for row in outer_rows]
    skipped_allowlist = set(BOUNDARY_TOKENIZER_SKIPPED_REFS)
    skipped_refs = [
        str(row["ref"])
        for row in remaining_rows
        if row.get("ref") in skipped_allowlist
    ]
    if (
        len(skipped_allowlist) != 2
        or tuple(sorted(skipped_refs)) != tuple(sorted(skipped_allowlist))
    ):
        raise ValueError(
            "boundary inner selection requires the exact two tokenizer-skipped refs "
            "outside the outer reserve"
        )

    encoded_rows = tuple(
        row for row in remaining_rows if row.get("ref") not in skipped_allowlist
    )
    encoded_refs = [str(row["ref"]) for row in encoded_rows]
    if len(encoded_rows) != boundary.BOUNDARY_EXPECTED_ENCODED_EXAMPLES:
        raise ValueError("boundary inner selection did not reproduce 4,430 encoded rows")
    fold = boundary.split_boundary_encoded_refs(encoded_refs, outer_refs=outer_refs)
    selected_rows = tuple(encoded_rows[index] for index in fold.validation_indices)
    selected_refs = [str(row["ref"]) for row in selected_rows]
    if (
        len(fold.train_indices) != boundary.BOUNDARY_INNER_TRAIN_EXAMPLES
        or len(selected_rows) != boundary.BOUNDARY_INNER_VALIDATION_EXAMPLES
        or boundary.refs_digest(selected_refs)
        != fold.manifest["inner_validation_refs_digest"]
    ):
        raise ValueError("boundary inner selection has unexpected fold identity")

    manifest = {
        "schema": SELECTION_SCHEMA,
        "mode": BOUNDARY_INNER_SELECTION,
        "corpus_version": CORPUS_VERSION,
        "corpus_file_digest_algorithm": boundary.BOUNDARY_FILE_DIGEST_ALGORITHM,
        "corpus_file_digest": boundary.digest_bytes(payload),
        "corpus_file_bytes": len(payload),
        "refs_digest_algorithm": boundary.BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
        "tokenizer_digest": boundary.BASE_TOKENIZER_DIGEST,
        "max_length": 512,
        "outer_algorithm": boundary.BOUNDARY_OUTER_SPLIT_ALGORITHM,
        "inner_algorithm": boundary.BOUNDARY_INNER_SPLIT_ALGORITHM,
        "seed": boundary.BOUNDARY_OUTER_SEED,
        "outer_seed": boundary.BOUNDARY_OUTER_SEED,
        "inner_seed": boundary.BOUNDARY_OUTER_SEED,
        "corpus_train_examples": len(rows),
        "outer_examples": len(outer_rows),
        "post_outer_examples": len(remaining_rows),
        "skipped_examples": len(skipped_refs),
        "encoded_examples": len(encoded_rows),
        "inner_train_examples": len(fold.train_indices),
        "inner_validation_examples": len(fold.validation_indices),
        "selected_examples": len(selected_rows),
        "outer_refs_digest": fold.manifest["outer_refs_digest"],
        "skipped_refs_digest": boundary.refs_digest(skipped_refs),
        "selected_refs_digest": fold.manifest["inner_validation_refs_digest"],
        "selected_ordered_refs_digest": boundary.canonical_json_digest(selected_refs),
    }
    return EvaluationSelection(selected_rows, manifest)


def select_evaluation_rows(
    corpus_path: Path,
    *,
    selection: str | None,
    seed: int | None,
    examples: int | None,
    legacy_examples: int,
) -> EvaluationSelection:
    """Select evaluation rows, preserving each evaluator's legacy defaults."""

    mode = selection or LEGACY_SELECTION
    if mode not in SELECTION_MODES:
        raise ValueError(f"unsupported evaluation selection mode: {mode}")
    if mode == BOUNDARY_INNER_SELECTION:
        if seed is not None and seed != boundary.BOUNDARY_OUTER_SEED:
            raise ValueError(
                f"{BOUNDARY_INNER_SELECTION} requires --seed "
                f"{boundary.BOUNDARY_OUTER_SEED}"
            )
        if examples is not None and examples != boundary.BOUNDARY_INNER_VALIDATION_EXAMPLES:
            raise ValueError(
                f"{BOUNDARY_INNER_SELECTION} requires --examples "
                f"{boundary.BOUNDARY_INNER_VALIDATION_EXAMPLES}"
            )
        return _boundary_inner_selection(corpus_path.read_bytes())

    resolved_seed = 92 if seed is None else seed
    resolved_examples = legacy_examples if examples is None else examples
    rows = _load_legacy_rows(corpus_path.read_bytes())
    random.Random(resolved_seed).shuffle(rows)
    selected_rows = tuple(rows[:resolved_examples])
    return EvaluationSelection(
        selected_rows,
        {
            "schema": SELECTION_SCHEMA,
            "mode": LEGACY_SELECTION,
            "algorithm": LEGACY_ALGORITHM,
            "corpus_version": CORPUS_VERSION,
            "seed": resolved_seed,
            "requested_examples": resolved_examples,
            "selected_examples": len(selected_rows),
        },
    )
