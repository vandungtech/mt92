#!/usr/bin/env python3
"""Deterministic, dependency-free boundary-contrastive lineage primitives.

Both the trainer and provenance publisher import this module.  Keeping corpus
partitioning and corruption generation here lets the publisher replay a
declared experiment from the public corpus without importing torch,
transformers, or trusting the trainer's emitted manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CORPUS_VERSION = "sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8"
BASE_TOKENIZER_DIGEST = (
    "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)
BOUNDARY_SCHEMA = "microtensor.boundary_contrastive.v1"
BOUNDARY_OUTER_SPLIT_ALGORITHM = "python_random_mt19937_seed92_prefix384_v1"
BOUNDARY_INNER_SPLIT_ALGORITHM = "sha256_rank_encoded_ref_seed92_v1"
BOUNDARY_CORRUPTION_ALGORITHM = "single_entity_text_boundary_codepoint_v1"
BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM = "sha256_canonical_json_utf8_v1"
BOUNDARY_TEXT_DIGEST_ALGORITHM = "sha256_utf8_text_v1"
BOUNDARY_FILE_DIGEST_ALGORITHM = "sha256_file_bytes_v1"
BOUNDARY_CORPUS_FILE_DIGEST = (
    "sha256:fb5f1332493b1abe759da91cec1bd2cdd932c7076c5fae8163b5f33cfbea05a2"
)
BOUNDARY_CORPUS_BYTES = 4_266_196
BOUNDARY_OUTER_SEED = 92
BOUNDARY_OUTER_EXAMPLES = 384
BOUNDARY_CORPUS_TRAIN_EXAMPLES = 4_816
BOUNDARY_INNER_VALIDATION_EXAMPLES = 384
BOUNDARY_EXPECTED_ENCODED_EXAMPLES = 4_430
BOUNDARY_INNER_TRAIN_EXAMPLES = 4_046
BOUNDARY_SKIPPED_ENCODED_REFS = (
    "bc5cdr-train-03995",
    "bc5cdr-train-04805",
)
MAX_BOUNDARY_CONTRASTIVE_EXAMPLES = 512
MAX_BOUNDARY_CONTRASTIVE_LAMBDA = 1.0
MAX_BOUNDARY_CONTRASTIVE_MARGIN = 20.0
ENTITY_TYPES = ("Chemical", "Disease")


@dataclass(frozen=True)
class BoundaryFoldSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BoundaryCorruption:
    source_ref: str
    negative_row: dict[str, Any]
    record: dict[str, Any]


@dataclass(frozen=True)
class BoundaryCorruptionResult:
    pairs: tuple[BoundaryCorruption, ...]
    manifest: dict[str, Any]


class _BoundaryIneligible(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def strict_json_loads(raw: str, ref: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key {key!r} for {ref}")
            payload[key] = value
        return payload

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard numeric constant {value} for {ref}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid gold JSON for {ref}") from exc


def _strict_corpus_json(payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        parsed = strict_json_loads(decoded, "boundary corpus")
    except UnicodeDecodeError as exc:
        raise ValueError("boundary corpus is not UTF-8") from exc
    if not isinstance(parsed, dict):
        raise ValueError("boundary corpus must contain a JSON object")
    return parsed


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical strict JSON") from exc
    return encoded.encode("utf-8")


def stable_hash(*parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def refs_digest(refs: Sequence[str]) -> str:
    return canonical_json_digest(sorted(refs))


def text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_boundary_corpus(payload: bytes) -> tuple[dict[str, Any], ...]:
    """Parse only the exact pinned public corpus bytes used by this recipe."""

    if len(payload) != BOUNDARY_CORPUS_BYTES:
        raise ValueError(
            f"boundary corpus must contain exactly {BOUNDARY_CORPUS_BYTES} bytes"
        )
    if digest_bytes(payload) != BOUNDARY_CORPUS_FILE_DIGEST:
        raise ValueError("boundary corpus bytes do not match the pinned public corpus")
    corpus = _strict_corpus_json(payload)
    if corpus.get("version") != CORPUS_VERSION or corpus.get("track") != "extract":
        raise ValueError("boundary corpus identity is not allowlisted")
    tasks = corpus.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("boundary corpus tasks must be a list")
    rows = tuple(row for row in tasks if isinstance(row, dict) and row.get("partition") == "train")
    if len(rows) != BOUNDARY_CORPUS_TRAIN_EXAMPLES:
        raise ValueError(
            "boundary corpus must contain exactly 4,816 public train rows"
        )
    if any(not isinstance(row, dict) for row in tasks):
        raise ValueError("boundary corpus tasks must contain only objects")
    return rows


def boundary_outer_partition(
    rows: Sequence[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Seal the historical seed-92 reserve without inspecting row contents."""

    if len(rows) != BOUNDARY_CORPUS_TRAIN_EXAMPLES:
        raise ValueError("boundary outer split requires exactly 4,816 rows")
    refs: list[str] = []
    for row in rows:
        ref = row.get("ref")
        if not isinstance(ref, str) or not ref or ref != ref.strip():
            raise ValueError("boundary split requires non-empty, already stripped refs")
        if row.get("partition") != "train":
            raise ValueError(f"boundary split accepts only train-fold rows, got {ref}")
        refs.append(ref)
    if len(set(refs)) != len(refs):
        raise ValueError("boundary split requires unique refs")
    shuffled = list(rows)
    random.Random(BOUNDARY_OUTER_SEED).shuffle(shuffled)
    return (
        tuple(shuffled[:BOUNDARY_OUTER_EXAMPLES]),
        tuple(shuffled[BOUNDARY_OUTER_EXAMPLES:]),
    )


def split_boundary_encoded_refs(
    encoded_refs: Sequence[str],
    *,
    outer_refs: Sequence[str],
) -> BoundaryFoldSplit:
    """Hash-rank encoded, non-reserve refs into a stable inner split."""

    refs = list(encoded_refs)
    reserved = list(outer_refs)
    if (
        len(refs) != BOUNDARY_EXPECTED_ENCODED_EXAMPLES
        or len(reserved) != BOUNDARY_OUTER_EXAMPLES
    ):
        raise ValueError(
            "boundary inner split requires exactly 4,430 encoded and 384 outer refs"
        )
    if any(not isinstance(ref, str) or not ref or ref != ref.strip() for ref in refs):
        raise ValueError("encoded refs must be non-empty, already stripped strings")
    if any(
        not isinstance(ref, str) or not ref or ref != ref.strip()
        for ref in reserved
    ):
        raise ValueError("outer refs must be non-empty, already stripped strings")
    if len(set(refs)) != len(refs) or len(set(reserved)) != len(reserved):
        raise ValueError("boundary split refs must be unique")
    outer_overlap = set(refs) & set(reserved)
    if outer_overlap:
        raise ValueError(
            f"encoded pool overlaps outer reserve: {sorted(outer_overlap)[0]}"
        )

    ranked = sorted(
        refs,
        key=lambda ref: (
            stable_hash(
                BOUNDARY_INNER_SPLIT_ALGORITHM,
                BOUNDARY_OUTER_SEED,
                ref,
            ),
            ref,
        ),
    )
    validation_refs = set(ranked[:BOUNDARY_INNER_VALIDATION_EXAMPLES])
    validation_indices = tuple(
        index for index, ref in enumerate(refs) if ref in validation_refs
    )
    train_indices = tuple(
        index for index, ref in enumerate(refs) if ref not in validation_refs
    )
    train_refs = [refs[index] for index in train_indices]
    inner_validation_refs = [refs[index] for index in validation_indices]
    train_set = set(train_refs)
    validation_set = set(inner_validation_refs)
    outer_set = set(reserved)
    overlaps = {
        "outer_inner_train": len(outer_set & train_set),
        "outer_inner_validation": len(outer_set & validation_set),
        "inner_train_inner_validation": len(train_set & validation_set),
    }
    if any(overlaps.values()) or len(train_indices) + len(validation_indices) != len(
        refs
    ):
        raise ValueError("boundary split is not an exact disjoint partition")
    return BoundaryFoldSplit(
        train_indices,
        validation_indices,
        {
            "refs_digest_algorithm": BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
            "outer_algorithm": BOUNDARY_OUTER_SPLIT_ALGORITHM,
            "outer_seed": BOUNDARY_OUTER_SEED,
            "outer_examples": len(reserved),
            "inner_algorithm": BOUNDARY_INNER_SPLIT_ALGORITHM,
            "inner_seed": BOUNDARY_OUTER_SEED,
            "encoded_examples": len(refs),
            "inner_train_examples": len(train_refs),
            "inner_validation_examples": len(inner_validation_refs),
            "outer_refs": sorted(reserved),
            "inner_train_refs": sorted(train_refs),
            "inner_validation_refs": sorted(inner_validation_refs),
            "outer_refs_digest": refs_digest(reserved),
            "encoded_refs_digest": refs_digest(refs),
            "inner_train_refs_digest": refs_digest(train_refs),
            "inner_validation_refs_digest": refs_digest(inner_validation_refs),
            "overlap_counts": overlaps,
        },
    )


def _literal_occurrences(text: str, needle: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return tuple(spans)
        spans.append((start, start + len(needle)))
        cursor = start + 1


def _boundary_change_is_exact(
    source: str, corrupted: str, direction: str, side: str
) -> bool:
    if direction == "expansion" and len(corrupted) == len(source) + 1:
        return (side == "left" and corrupted[1:] == source) or (
            side == "right" and corrupted[:-1] == source
        )
    if direction == "contraction" and len(corrupted) + 1 == len(source):
        return (side == "left" and source[1:] == corrupted) or (
            side == "right" and source[:-1] == corrupted
        )
    return False


def _boundary_row_candidates(
    row: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    ref = row.get("ref")
    if not isinstance(ref, str) or not ref:
        raise ValueError("boundary corruption requires a non-empty ref")
    if row.get("partition") != "train":
        raise ValueError(f"boundary corruption accepts only train-fold rows, got {ref}")
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"invalid inputs for {ref}")
    input_text = inputs.get("text")
    if not isinstance(input_text, str) or not input_text:
        raise ValueError(f"invalid input text for {ref}")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.endswith(input_text):
        raise ValueError(f"prompt is not bound to its input text for {ref}")
    if not prompt[: len(prompt) - len(input_text)].endswith("\n\nText: "):
        raise ValueError(f"prompt has an unexpected input boundary for {ref}")
    raw_gold = row.get("gold")
    if not isinstance(raw_gold, str):
        raise ValueError(f"boundary corruption requires raw gold text for {ref}")
    payload = strict_json_loads(raw_gold, ref)
    if not isinstance(payload, Mapping) or set(payload) != {"entities"}:
        raise ValueError(f"gold must contain only entities for {ref}")
    raw_entities = payload["entities"]
    if not isinstance(raw_entities, list):
        raise ValueError(f"gold entities must be a list for {ref}")

    entities: list[dict[str, str]] = []
    pairs: set[tuple[str, str]] = set()
    texts: set[str] = set()
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, Mapping) or set(raw_entity) != {"text", "type"}:
            raise ValueError(f"malformed gold entity {index} for {ref}")
        entity_text = raw_entity["text"]
        entity_type = raw_entity["type"]
        if (
            not isinstance(entity_text, str)
            or not isinstance(entity_type, str)
            or not entity_text
            or entity_text != entity_text.strip()
            or entity_type not in ENTITY_TYPES
            or any(ord(character) < 32 for character in entity_text)
        ):
            raise ValueError(f"invalid gold entity {index} for {ref}")
        pair = (entity_text, entity_type)
        if pair in pairs:
            raise _BoundaryIneligible("duplicate_entity")
        if entity_text in texts:
            raise _BoundaryIneligible("ambiguous_entity_text")
        pairs.add(pair)
        texts.add(entity_text)
        entities.append({"text": entity_text, "type": entity_type})
    if not entities:
        raise _BoundaryIneligible("no_entities")
    if any(entity["text"] not in input_text for entity in entities):
        raise _BoundaryIneligible("gold_surface_absent")

    candidates: dict[str, list[dict[str, Any]]] = {
        "expansion": [],
        "contraction": [],
    }
    seen_negative_targets: set[str] = set()
    for entity_index, entity in enumerate(entities):
        source_text = entity["text"]
        encoded_source = json.dumps(source_text, ensure_ascii=False)
        literal_spans = _literal_occurrences(raw_gold, encoded_source)
        if len(literal_spans) != 1:
            continue
        replacements: list[tuple[str, str, str]] = []
        if len(source_text) > 1:
            replacements.extend(
                (
                    ("contraction", "left", source_text[1:]),
                    ("contraction", "right", source_text[:-1]),
                )
            )
        for start, end in _literal_occurrences(input_text, source_text):
            if start > 0:
                replacements.append(
                    ("expansion", "left", input_text[start - 1 : end])
                )
            if end < len(input_text):
                replacements.append(
                    ("expansion", "right", input_text[start : end + 1])
                )
        for direction, side, corrupted_text in replacements:
            if (
                not corrupted_text
                or corrupted_text != corrupted_text.strip()
                or any(ord(character) < 32 for character in corrupted_text)
                or corrupted_text not in input_text
                or corrupted_text in texts
                or not _boundary_change_is_exact(
                    source_text, corrupted_text, direction, side
                )
            ):
                continue
            literal_start, literal_end = literal_spans[0]
            encoded_corrupted = json.dumps(corrupted_text, ensure_ascii=False)
            negative_gold = (
                raw_gold[:literal_start]
                + encoded_corrupted
                + raw_gold[literal_end:]
            )
            if negative_gold in seen_negative_targets:
                continue
            expected_entities = [dict(value) for value in entities]
            expected_entities[entity_index]["text"] = corrupted_text
            if strict_json_loads(negative_gold, ref) != {"entities": expected_entities}:
                raise ValueError(
                    f"generated boundary corruption changed extra fields for {ref}"
                )
            seen_negative_targets.add(negative_gold)
            candidates[direction].append(
                {
                    "direction": direction,
                    "side": side,
                    "entity_index": entity_index,
                    "source_text": source_text,
                    "corrupted_text": corrupted_text,
                    "negative_gold": negative_gold,
                }
            )
    if not candidates["expansion"] and not candidates["contraction"]:
        raise _BoundaryIneligible("no_valid_boundary")
    return candidates


def generate_boundary_corruptions(
    rows: Sequence[dict[str, Any]],
    *,
    heldout_refs: set[str],
    seed: int,
    max_examples: int,
) -> BoundaryCorruptionResult:
    """Create a balanced, deterministic set of raw-positive boundary pairs."""

    if isinstance(max_examples, bool) or not isinstance(max_examples, int):
        raise ValueError("boundary contrastive examples must be an integer")
    if (
        not 0 <= max_examples <= MAX_BOUNDARY_CONTRASTIVE_EXAMPLES
        or max_examples % 2
    ):
        raise ValueError(
            "boundary contrastive examples must be an even integer in "
            f"[0, {MAX_BOUNDARY_CONTRASTIVE_EXAMPLES}]"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("boundary corruption seed must be an integer")
    if any(not isinstance(ref, str) or not ref for ref in heldout_refs):
        raise ValueError("held-out refs must be non-empty strings")
    ordered_rows = sorted(rows, key=lambda row: str(row.get("ref", "")))
    refs = [row.get("ref") for row in ordered_rows]
    if any(not isinstance(ref, str) or not ref for ref in refs) or len(
        set(refs)
    ) != len(refs):
        raise ValueError("boundary corruption requires unique non-empty source refs")
    source_refs = set(refs)
    overlap = source_refs & heldout_refs
    if overlap:
        raise ValueError(
            f"boundary source and held-out refs overlap: {sorted(overlap)[0]}"
        )

    eligible: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    ineligible: dict[str, int] = {}
    for row in ordered_rows:
        try:
            candidates = _boundary_row_candidates(row)
        except _BoundaryIneligible as exc:
            ineligible[exc.reason] = ineligible.get(exc.reason, 0) + 1
            continue
        selected: dict[str, dict[str, Any]] = {}
        for direction, values in candidates.items():
            if values:
                selected[direction] = min(
                    values,
                    key=lambda value: (
                        stable_hash(
                            BOUNDARY_CORRUPTION_ALGORITHM,
                            seed,
                            row["ref"],
                            direction,
                            value["side"],
                            value["entity_index"],
                            value["corrupted_text"],
                        ),
                        value["side"],
                        value["entity_index"],
                        value["corrupted_text"],
                    ),
                )
        eligible.append((row, selected))

    expansion_only = [entry for entry in eligible if set(entry[1]) == {"expansion"}]
    contraction_only = [
        entry for entry in eligible if set(entry[1]) == {"contraction"}
    ]
    both = [
        entry
        for entry in eligible
        if set(entry[1]) == {"expansion", "contraction"}
    ]

    def rank(
        entries: Sequence[tuple[dict[str, Any], dict[str, dict[str, Any]]]],
        label: str,
    ) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
        return sorted(
            entries,
            key=lambda entry: (
                stable_hash(
                    BOUNDARY_CORRUPTION_ALGORITHM,
                    seed,
                    "source",
                    label,
                    entry[0]["ref"],
                ),
                entry[0]["ref"],
            ),
        )

    expansion_only = rank(expansion_only, "expansion")
    contraction_only = rank(contraction_only, "contraction")
    both = rank(both, "balanced")
    per_direction = max_examples // 2
    while per_direction:
        expansion_need = per_direction - min(per_direction, len(expansion_only))
        contraction_need = per_direction - min(per_direction, len(contraction_only))
        if expansion_need + contraction_need <= len(both):
            break
        per_direction -= 1
    expansion_take = min(per_direction, len(expansion_only))
    contraction_take = min(per_direction, len(contraction_only))
    expansion_need = per_direction - expansion_take
    contraction_need = per_direction - contraction_take
    selected_expansions = [
        *expansion_only[:expansion_take],
        *both[:expansion_need],
    ]
    selected_contractions = [
        *contraction_only[:contraction_take],
        *both[expansion_need : expansion_need + contraction_need],
    ]
    selected_entries = [
        (row, values["expansion"]) for row, values in selected_expansions
    ] + [
        (row, values["contraction"]) for row, values in selected_contractions
    ]
    selected_entries.sort(
        key=lambda entry: (
            stable_hash(
                BOUNDARY_CORRUPTION_ALGORITHM,
                seed,
                "pair",
                entry[0]["ref"],
                entry[1]["direction"],
            ),
            entry[0]["ref"],
        )
    )

    pairs: list[BoundaryCorruption] = []
    records: list[dict[str, Any]] = []
    for row, candidate in selected_entries:
        source_ref = row["ref"]
        positive_gold = row["gold"]
        negative_gold = candidate["negative_gold"]
        negative_id = stable_hash(
            BOUNDARY_CORRUPTION_ALGORITHM,
            seed,
            source_ref,
            candidate["direction"],
            candidate["side"],
            candidate["entity_index"],
            candidate["corrupted_text"],
        )[:16]
        negative_ref = f"{source_ref}::boundary-negative::{negative_id}"
        if negative_ref in source_refs or negative_ref in heldout_refs:
            raise ValueError(f"generated boundary ref collides: {negative_ref}")
        negative_row = dict(row)
        negative_row["ref"] = negative_ref
        negative_row["gold"] = negative_gold
        record = {
            "source_ref": source_ref,
            "negative_ref": negative_ref,
            "direction": candidate["direction"],
            "side": candidate["side"],
            "entity_index": candidate["entity_index"],
            "source_text": candidate["source_text"],
            "corrupted_text": candidate["corrupted_text"],
            "positive_gold_digest": text_digest(positive_gold),
            "negative_gold_digest": text_digest(negative_gold),
        }
        records.append(record)
        pairs.append(BoundaryCorruption(source_ref, negative_row, record))

    direction_counts = {
        direction: sum(record["direction"] == direction for record in records)
        for direction in ("expansion", "contraction")
    }
    if direction_counts["expansion"] != direction_counts["contraction"]:
        raise ValueError("boundary pair selection is not direction-balanced")
    manifest = {
        "algorithm": BOUNDARY_CORRUPTION_ALGORITHM,
        "refs_and_records_digest_algorithm": BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
        "gold_text_digest_algorithm": BOUNDARY_TEXT_DIGEST_ALGORITHM,
        "enabled": True,
        "seed": seed,
        "requested_examples": max_examples,
        "source_training_rows": len(rows),
        "eligible_source_rows": len(eligible),
        "ineligible_source_rows": dict(sorted(ineligible.items())),
        "generated_examples": len(pairs),
        "direction_counts": direction_counts,
        "source_refs_digest": refs_digest(refs),
        "heldout_refs_digest": refs_digest(sorted(heldout_refs)),
        "pair_source_refs_digest": refs_digest(
            [pair.source_ref for pair in pairs]
        ),
        "positive_targets_digest": canonical_json_digest(
            [
                {
                    "ref": record["source_ref"],
                    "digest": record["positive_gold_digest"],
                }
                for record in records
            ]
        ),
        "negative_targets_digest": canonical_json_digest(
            [
                {
                    "ref": record["negative_ref"],
                    "digest": record["negative_gold_digest"],
                }
                for record in records
            ]
        ),
        "pairs_digest": canonical_json_digest(records),
        "examples": records,
    }
    return BoundaryCorruptionResult(tuple(pairs), manifest)


def disease_row_count(rows: Sequence[dict[str, Any]]) -> int:
    """Count Disease-bearing rows with strict raw-gold parsing."""

    count = 0
    for row in rows:
        ref = row.get("ref")
        raw_gold = row.get("gold")
        if not isinstance(ref, str) or not ref or not isinstance(raw_gold, str):
            raise ValueError("boundary disease count requires ref and raw gold")
        payload = strict_json_loads(raw_gold, ref)
        if not isinstance(payload, Mapping) or set(payload) != {"entities"}:
            raise ValueError(f"gold must contain only entities for {ref}")
        entities = payload["entities"]
        if not isinstance(entities, list) or any(
            not isinstance(entity, Mapping)
            or set(entity) != {"text", "type"}
            or not isinstance(entity["text"], str)
            or not isinstance(entity["type"], str)
            for entity in entities
        ):
            raise ValueError(f"malformed gold entities for {ref}")
        count += int(any(entity["type"] == "Disease" for entity in entities))
    return count


def build_boundary_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    skipped_refs: Sequence[str],
    requested_examples: int,
    pairwise_lambda: float,
    margin: float,
) -> tuple[dict[str, Any], BoundaryFoldSplit, BoundaryCorruptionResult]:
    """Replay the complete v1 split and corruption manifest from corpus rows."""

    if isinstance(pairwise_lambda, bool) or not isinstance(
        pairwise_lambda, int | float
    ):
        raise ValueError("boundary pairwise lambda must be a number")
    if isinstance(margin, bool) or not isinstance(margin, int | float):
        raise ValueError("boundary margin must be a number")
    pairwise_value = float(pairwise_lambda)
    margin_value = float(margin)
    if not 0.0 <= pairwise_value <= MAX_BOUNDARY_CONTRASTIVE_LAMBDA:
        raise ValueError("boundary pairwise lambda is outside its allowed range")
    if not 0.0 <= margin_value <= MAX_BOUNDARY_CONTRASTIVE_MARGIN:
        raise ValueError("boundary margin is outside its allowed range")

    outer_rows, remaining_rows = boundary_outer_partition(rows)
    skipped = tuple(sorted(skipped_refs))
    if skipped != tuple(sorted(BOUNDARY_SKIPPED_ENCODED_REFS)):
        raise ValueError("boundary skipped refs do not match the pinned tokenizer allowlist")
    remaining_by_ref = {row["ref"]: row for row in remaining_rows}
    if len(remaining_by_ref) != len(remaining_rows) or any(
        ref not in remaining_by_ref for ref in skipped
    ):
        raise ValueError("boundary skipped refs are not exact non-reserve corpus rows")
    skipped_set = set(skipped)
    encoded_refs = [
        row["ref"] for row in remaining_rows if row["ref"] not in skipped_set
    ]
    outer_refs = [row["ref"] for row in outer_rows]
    fold = split_boundary_encoded_refs(encoded_refs, outer_refs=outer_refs)
    source_rows = [
        remaining_by_ref[encoded_refs[index]] for index in fold.train_indices
    ]
    heldout_refs = (
        set(outer_refs)
        | set(fold.manifest["inner_validation_refs"])
        | skipped_set
    )
    corruption = generate_boundary_corruptions(
        source_rows,
        heldout_refs=heldout_refs,
        seed=BOUNDARY_OUTER_SEED,
        max_examples=requested_examples,
    )
    if len(corruption.pairs) != requested_examples:
        raise ValueError(
            "public inner train did not yield every requested balanced boundary pair"
        )
    manifest = {
        "schema": BOUNDARY_SCHEMA,
        "embedded_digest_algorithms": {
            "refs_and_records": BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
            "gold_text": BOUNDARY_TEXT_DIGEST_ALGORITHM,
        },
        "settings": {
            "enabled": True,
            "requested_examples": requested_examples,
            "pairwise_lambda": pairwise_lambda,
            "margin": margin,
            "positive_target": "raw_gold",
            "score": "mean_assistant_log_probability",
            "pair_penalty_reduction": (
                "sum_over_positive_count_unpaired_positive_zero"
            ),
            "objective": (
                "positive_ce_plus_lambda_times_reduced_pair_penalty"
            ),
        },
        "split": fold.manifest,
        "corruption": corruption.manifest,
        "skipped_encoded_refs": list(skipped),
        "skipped_refs_digest_algorithm": BOUNDARY_CANONICAL_JSON_DIGEST_ALGORITHM,
        "skipped_encoded_refs_digest": refs_digest(skipped),
    }
    return manifest, fold, corruption


def boundary_metadata_summary(
    manifest: Mapping[str, Any],
    *,
    manifest_digest: str,
) -> dict[str, Any]:
    """Derive the exact compact training-metadata claim for a full manifest."""

    split = manifest["split"]
    corruption = manifest["corruption"]
    settings = manifest["settings"]
    return {
        "schema": manifest["schema"],
        "embedded_digest_algorithms": manifest["embedded_digest_algorithms"],
        **settings,
        "manifest_file": "boundary_contrastive_manifest.json",
        "manifest_digest_algorithm": BOUNDARY_FILE_DIGEST_ALGORITHM,
        "manifest_digest": manifest_digest,
        "split": {
            key: value
            for key, value in split.items()
            if key
            not in {"outer_refs", "inner_train_refs", "inner_validation_refs"}
        },
        "corruption": {
            key: value for key, value in corruption.items() if key != "examples"
        },
        "skipped_encoded_examples": len(manifest["skipped_encoded_refs"]),
        "skipped_encoded_refs_digest": manifest["skipped_encoded_refs_digest"],
    }
