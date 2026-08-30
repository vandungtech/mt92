#!/usr/bin/env python3
"""Fine-tune the allowlisted Qwen3-0.6B model on Microtensor's public extract split.

The script deliberately writes a local, append-only metrics trail.  Publishing
that trail to W&B happens only after the final GGUF digest exists; see
``publish_provenance.py``.  This keeps the provenance record truthful and makes
an interrupted training run resumable without inventing metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


try:
    from training import build_weight_soup as weight_soup
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    import build_weight_soup as weight_soup

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CORPUS_VERSION = "sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8"
HOTKEY = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"
BASE_WEIGHTS_DIGEST = "sha256:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
BASE_TOKENIZER_DIGEST = "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
ENTITY_SUBSTITUTION_ALGORITHM = "literal_same_type_train_fold_v1"
ENTITY_TYPES = ("Chemical", "Disease")
MAX_ENTITY_SUBSTITUTION_EXAMPLES = 1_024


@dataclass(frozen=True)
class Settings:
    training_method: str
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    disease_row_weight: float
    validation_examples: int
    gold_canonicalization: str = "none"
    entity_text_token_weight: float = 1.0
    entity_substitution_examples: int = 0


@dataclass(frozen=True)
class CanonicalTarget:
    """A strict scorer-equivalent target and its entity-value character spans."""

    content: str
    entity_text_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AugmentationSurface:
    """A typed gold surface bound to exact, non-overlapping input spans."""

    text: str
    entity_type: str
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BoundAugmentationRow:
    row: dict[str, Any]
    ref: str
    input_text: str
    prompt_prefix: str
    entities: tuple[dict[str, str], ...]
    surfaces: tuple[AugmentationSurface, ...]


@dataclass(frozen=True)
class EntitySubstitutionResult:
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


class _AugmentationIneligible(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_loads(raw: str, ref: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key {key!r} for {ref}")
            payload[key] = value
        return payload

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid gold JSON for {ref}") from exc


def _literal_occurrences(text: str, needle: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return tuple(spans)
        spans.append((start, start + len(needle)))
        cursor = start + 1


def _bind_augmentation_row(row: dict[str, Any]) -> BoundAugmentationRow:
    ref = row.get("ref")
    if not isinstance(ref, str) or not ref or ref != ref.strip():
        raise ValueError("augmentation rows require a non-empty, already stripped ref")
    if row.get("partition") != "train":
        raise ValueError(f"entity substitution accepts only train-fold rows, got {ref}")
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"invalid inputs for {ref}")
    input_text = inputs.get("text")
    if not isinstance(input_text, str) or not input_text:
        raise ValueError(f"invalid input text for {ref}")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.endswith(input_text):
        raise ValueError(f"prompt is not bound to its input text for {ref}")
    prompt_prefix = prompt[: len(prompt) - len(input_text)]
    if not prompt_prefix.endswith("\n\nText: "):
        raise ValueError(f"prompt has an unexpected input boundary for {ref}")

    raw_gold = row.get("gold")
    payload = _strict_json_loads(raw_gold, ref) if isinstance(raw_gold, str) else raw_gold
    if not isinstance(payload, Mapping) or set(payload) != {"entities"}:
        raise ValueError(f"gold must contain only entities for {ref}")
    raw_entities = payload["entities"]
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        raise ValueError(f"gold entities must be a list for {ref}")

    entities: list[dict[str, str]] = []
    surface_types: dict[str, str] = {}
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
        prior_type = surface_types.setdefault(entity_text, entity_type)
        if prior_type != entity_type:
            raise _AugmentationIneligible("ambiguous_entity_type")
        entities.append({"text": entity_text, "type": entity_type})

    if not entities:
        raise _AugmentationIneligible("no_entities")
    surfaces: list[AugmentationSurface] = []
    occupied: list[tuple[int, int, str]] = []
    for entity_text, entity_type in surface_types.items():
        spans = _literal_occurrences(input_text, entity_text)
        if not spans:
            raise _AugmentationIneligible("gold_surface_absent")
        for start, end in spans:
            if (start > 0 and input_text[start - 1].isalnum()) or (
                end < len(input_text) and input_text[end].isalnum()
            ):
                raise _AugmentationIneligible("alphanumeric_boundary_collision")
        for previous, current in zip(spans, spans[1:], strict=False):
            if current[0] < previous[1]:
                raise _AugmentationIneligible("self_overlapping_surface")
        occupied.extend((start, end, entity_text) for start, end in spans)
        surfaces.append(AugmentationSurface(entity_text, entity_type, spans))
    occupied.sort()
    for previous, current in zip(occupied, occupied[1:], strict=False):
        if current[0] < previous[1] and current[2] != previous[2]:
            raise _AugmentationIneligible("overlapping_entity_surfaces")
    surfaces.sort(key=lambda surface: (surface.entity_type, surface.text))
    return BoundAugmentationRow(
        row, ref, input_text, prompt_prefix, tuple(entities), tuple(surfaces)
    )


def _stable_hash(*parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def augment_train_fold_entity_substitutions(
    rows: list[dict[str, Any]],
    *,
    heldout_refs: set[str],
    seed: int,
    max_examples: int,
) -> EntitySubstitutionResult:
    """Create bounded synthetic copies using donors from this exact train fold."""

    if isinstance(max_examples, bool) or not isinstance(max_examples, int):
        raise ValueError("entity substitution examples must be an integer")
    if not 0 <= max_examples <= MAX_ENTITY_SUBSTITUTION_EXAMPLES:
        raise ValueError(
            f"entity substitution examples must be in [0, {MAX_ENTITY_SUBSTITUTION_EXAMPLES}]"
        )
    if max_examples == 0:
        return EntitySubstitutionResult(
            (),
            {
                "algorithm": ENTITY_SUBSTITUTION_ALGORITHM,
                "enabled": False,
                "seed": seed,
                "requested_examples": 0,
                "augmented_examples": 0,
                "replacement_count": 0,
                "examples": [],
            },
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("entity substitution seed must be an integer")
    if any(not isinstance(ref, str) or not ref for ref in heldout_refs):
        raise ValueError("held-out refs must be non-empty strings")

    ordered_rows = sorted(rows, key=lambda row: str(row.get("ref", "")))
    refs = [row.get("ref") for row in ordered_rows]
    if any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != len(refs):
        raise ValueError("entity substitution requires unique non-empty train-fold refs")
    source_refs = set(refs)
    overlap = source_refs & heldout_refs
    if overlap:
        raise ValueError(f"train and held-out refs overlap: {sorted(overlap)[0]}")

    bound_rows: list[BoundAugmentationRow] = []
    ineligible: dict[str, int] = {}
    for row in ordered_rows:
        try:
            bound_rows.append(_bind_augmentation_row(row))
        except _AugmentationIneligible as exc:
            ineligible[exc.reason] = ineligible.get(exc.reason, 0) + 1

    observed_types: dict[str, set[str]] = {}
    for source in bound_rows:
        for surface in source.surfaces:
            observed_types.setdefault(surface.text, set()).add(surface.entity_type)
    ambiguous_surfaces = {
        text for text, entity_types in observed_types.items() if len(entity_types) != 1
    }
    donor_by_type: dict[str, list[tuple[str, str]]] = {
        entity_type: [] for entity_type in ENTITY_TYPES
    }
    for source in bound_rows:
        for surface in source.surfaces:
            if surface.text not in ambiguous_surfaces:
                donor_by_type[surface.entity_type].append((surface.text, source.ref))
    for entity_type in ENTITY_TYPES:
        donor_by_type[entity_type] = sorted(set(donor_by_type[entity_type]))
    donor_records = [
        {"type": entity_type, "text": text, "source_ref": donor_ref}
        for entity_type in ENTITY_TYPES
        for text, donor_ref in donor_by_type[entity_type]
    ]

    candidates = sorted(
        bound_rows,
        key=lambda source: (_stable_hash(seed, "source", source.ref), source.ref),
    )
    augmented_rows: list[dict[str, Any]] = []
    augmented_refs: set[str] = set()
    records: list[dict[str, Any]] = []
    no_compatible_donor = 0
    for source in candidates:
        if len(augmented_rows) >= max_examples:
            break
        candidate_surfaces = sorted(
            (surface for surface in source.surfaces if surface.text not in ambiguous_surfaces),
            key=lambda surface: (
                _stable_hash(seed, "surface", source.ref, surface.entity_type, surface.text),
                surface.entity_type,
                surface.text,
            ),
        )
        selected: tuple[AugmentationSurface, str, str] | None = None
        source_texts = {surface.text.casefold() for surface in source.surfaces}
        for surface in candidate_surfaces:
            ranked_donors = sorted(
                donor_by_type[surface.entity_type],
                key=lambda donor: (
                    _stable_hash(
                        seed,
                        "donor",
                        source.ref,
                        surface.entity_type,
                        surface.text,
                        donor[0],
                        donor[1],
                    ),
                    donor[0],
                    donor[1],
                ),
            )
            for donor_text, donor_ref in ranked_donors:
                donor_folded = donor_text.casefold()
                if (
                    donor_ref == source.ref
                    or donor_text == surface.text
                    or donor_folded in source.input_text.casefold()
                    or any(ord(character) < 32 for character in donor_text)
                    or any(
                        existing in donor_folded or donor_folded in existing
                        for existing in source_texts
                    )
                ):
                    continue
                selected = (surface, donor_text, donor_ref)
                break
            if selected is not None:
                break
        if selected is None:
            no_compatible_donor += 1
            continue

        surface, donor_text, donor_ref = selected
        transformed_text = source.input_text
        for start, end in reversed(surface.spans):
            if transformed_text[start:end] != surface.text:
                raise ValueError(f"source span changed while augmenting {source.ref}")
            transformed_text = transformed_text[:start] + donor_text + transformed_text[end:]
        transformed_entities = [
            {
                "text": (
                    donor_text
                    if entity["text"] == surface.text and entity["type"] == surface.entity_type
                    else entity["text"]
                ),
                "type": entity["type"],
            }
            for entity in source.entities
        ]
        transformed_gold = json.dumps({"entities": transformed_entities}, ensure_ascii=False)
        transformation_id = _stable_hash(
            seed, source.ref, surface.entity_type, surface.text, donor_text, donor_ref
        )[:16]
        augmented_ref = f"{source.ref}::entity-substitution::{transformation_id}"
        if (
            augmented_ref in source_refs
            or augmented_ref in heldout_refs
            or augmented_ref in augmented_refs
        ):
            raise ValueError(
                f"generated augmentation ref collides with an existing ref: {augmented_ref}"
            )
        transformed_inputs = dict(source.row["inputs"])
        transformed_inputs["text"] = transformed_text
        augmented_row = dict(source.row)
        augmented_row.update(
            {
                "ref": augmented_ref,
                "inputs": transformed_inputs,
                "prompt": source.prompt_prefix + transformed_text,
                "gold": transformed_gold,
            }
        )
        try:
            _bind_augmentation_row(augmented_row)
        except _AugmentationIneligible as exc:
            raise ValueError(
                f"generated augmentation for {source.ref} violates {exc.reason}"
            ) from exc
        augmented_rows.append(augmented_row)
        augmented_refs.add(augmented_ref)
        records.append(
            {
                "source_ref": source.ref,
                "augmented_ref": augmented_ref,
                "donor_ref": donor_ref,
                "type": surface.entity_type,
                "source_text": surface.text,
                "donor_text": donor_text,
                "occurrence_count": len(surface.spans),
            }
        )

    manifest = {
        "algorithm": ENTITY_SUBSTITUTION_ALGORITHM,
        "enabled": True,
        "seed": seed,
        "requested_examples": max_examples,
        "source_training_rows": len(rows),
        "eligible_source_rows": len(bound_rows),
        "ineligible_source_rows": dict(sorted(ineligible.items())),
        "globally_ambiguous_surfaces": len(ambiguous_surfaces),
        "no_compatible_donor_rows": no_compatible_donor,
        "donor_entity_counts": {
            entity_type: len(donor_by_type[entity_type]) for entity_type in ENTITY_TYPES
        },
        "source_training_refs_digest": "sha256:" + _stable_hash(sorted(refs)),
        "heldout_refs_digest": "sha256:" + _stable_hash(sorted(heldout_refs)),
        "donor_pool_digest": "sha256:" + _stable_hash(donor_records),
        "augmented_examples": len(augmented_rows),
        "replacement_count": len(records),
        "examples": records,
    }
    return EntitySubstitutionResult(tuple(augmented_rows), manifest)



def canonicalize_gold(raw: Any, order: str = "first") -> CanonicalTarget:
    """Serialize a strict set of ``(text, type)`` pairs deterministically.

    The scorer strips surrounding whitespace from otherwise valid values.  A
    training target must never silently repair such a value, so this function
    rejects it instead.  Likewise, any malformed member rejects the complete
    target rather than being partially rescued.
    """

    if order not in {"first", "sorted"}:
        raise ValueError("gold canonicalization order must be first or sorted")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("gold is not valid JSON") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"entities"}:
        raise ValueError("gold must be an object containing only entities")
    entities = raw["entities"]
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        raise ValueError("gold entities must be a list")

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for entity in entities:
        if not isinstance(entity, Mapping) or set(entity) != {"text", "type"}:
            raise ValueError("every gold entity must contain exactly text and type")
        text_value = entity["text"]
        entity_type = entity["type"]
        if not isinstance(text_value, str) or not isinstance(entity_type, str):
            raise ValueError("gold entity text and type must be strings")
        if (
            not text_value
            or not entity_type
            or text_value != text_value.strip()
            or entity_type != entity_type.strip()
        ):
            raise ValueError("gold entity text and type must be non-empty and already stripped")
        pair = (text_value, entity_type)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    if order == "sorted":
        pairs.sort()

    content = '{"entities":['
    spans: list[tuple[int, int]] = []
    for index, (text_value, entity_type) in enumerate(pairs):
        if index:
            content += ","
        content += '{"text":'
        encoded_text = json.dumps(text_value, ensure_ascii=False)
        span_start = len(content) + 1
        content += encoded_text
        spans.append((span_start, len(content) - 1))
        content += ',"type":' + json.dumps(entity_type, ensure_ascii=False) + "}"
    content += "]}"
    return CanonicalTarget(content, tuple(spans))


def bind_entity_token_weights(
    offset_mapping: Sequence[Sequence[int]],
    labels: Sequence[int],
    entity_spans: Sequence[tuple[int, int]],
    entity_weight: float,
) -> list[float]:
    """Bind rendered entity spans to target tokens, rejecting ambiguous gaps."""

    if not math.isfinite(entity_weight) or entity_weight <= 1.0:
        raise ValueError("entity token weight must be finite and greater than 1.0")
    if len(offset_mapping) != len(labels):
        raise ValueError("token offsets and labels do not align")
    offsets: list[tuple[int, int]] = []
    for raw_offset in offset_mapping:
        if not isinstance(raw_offset, Sequence) or len(raw_offset) != 2:
            raise ValueError("tokenizer returned a malformed offset")
        start, end = raw_offset
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError("tokenizer returned a malformed offset")
        offsets.append((start, end))

    weights = [0.0 if int(label) == -100 else 1.0 for label in labels]
    for span_start, span_end in entity_spans:
        if span_start < 0 or span_end <= span_start:
            raise ValueError("entity text has an invalid rendered span")
        selected: list[int] = []
        covered: list[tuple[int, int]] = []
        for index, (token_start, token_end) in enumerate(offsets):
            if token_start < span_end and token_end > span_start:
                if int(labels[index]) == -100:
                    raise ValueError("an entity token overlaps the masked prompt")
                selected.append(index)
                covered.append((max(token_start, span_start), min(token_end, span_end)))
        if not selected:
            raise ValueError("an entity span could not be bound to any token")
        cursor = span_start
        for covered_start, covered_end in sorted(covered):
            if covered_start > cursor:
                raise ValueError("token offsets leave a gap inside an entity span")
            cursor = max(cursor, covered_end)
        if cursor < span_end:
            raise ValueError("token offsets do not cover a complete entity span")
        for index in selected:
            weights[index] = entity_weight
    return weights


def weighted_causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor
) -> torch.Tensor:
    """Return weighted next-token cross entropy with the causal shift applied."""

    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected [batch, sequence, vocab] logits and aligned target matrices")
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("logits, labels, and loss weights do not align")
    if logits.shape[1] < 2:
        raise ValueError("causal language-model loss requires at least two tokens")
    if not bool(torch.isfinite(loss_weights).all()) or bool((loss_weights < 0).any()):
        raise ValueError("loss weights must be finite and non-negative")

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].to(device=shift_logits.device, dtype=torch.float32)
    valid = shift_labels.ne(-100)
    effective_weights = shift_weights * valid.to(dtype=torch.float32)
    denominator = effective_weights.sum()
    if not bool(denominator > 0):
        raise ValueError("weighted loss has no supervised target tokens")
    token_losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    return (token_losses * effective_weights).sum() / denominator


class EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        *,
        gold_canonicalization: str = "none",
        entity_text_token_weight: float = 1.0,
    ) -> None:
        if gold_canonicalization not in {"none", "first", "sorted"}:
            raise ValueError("gold canonicalization must be none, first, or sorted")
        if not math.isfinite(entity_text_token_weight) or entity_text_token_weight < 1.0:
            raise ValueError("entity text token weight must be finite and at least 1.0")
        if entity_text_token_weight > 1.0 and gold_canonicalization == "none":
            raise ValueError("entity token weighting requires canonicalized gold")
        self.items: list[dict[str, torch.Tensor]] = []
        self.skipped = 0
        for row in rows:
            user = {"role": "user", "content": str(row["prompt"])}
            canonical = (
                canonicalize_gold(row["gold"], gold_canonicalization)
                if gold_canonicalization != "none"
                else None
            )
            assistant_content = canonical.content if canonical is not None else str(row["gold"])
            assistant = {"role": "assistant", "content": assistant_content}
            prefix = tokenizer.apply_chat_template(
                [user], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            complete = tokenizer.apply_chat_template(
                [user, assistant],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            if entity_text_token_weight > 1.0:
                if not complete.startswith(prefix) or not complete[len(prefix) :].startswith(
                    assistant_content
                ):
                    raise ValueError(
                        f"could not locate rendered assistant content for {row.get('ref', '<unknown>')}"
                    )
                encoded = tokenizer(
                    complete,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                input_ids = encoded.input_ids
                offset_mapping = encoded.offset_mapping
            else:
                input_ids = tokenizer(complete, add_special_tokens=False).input_ids
                offset_mapping = None
            if len(input_ids) > max_length:
                self.skipped += 1
                continue
            labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
            if len(labels) != len(input_ids) or all(value == -100 for value in labels):
                raise ValueError(f"could not locate assistant tokens for {row.get('ref', '<unknown>')}")
            item = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            if offset_mapping is not None:
                assert canonical is not None
                assistant_start = len(prefix)
                spans = tuple(
                    (assistant_start + start, assistant_start + end)
                    for start, end in canonical.entity_text_spans
                )
                weights = bind_entity_token_weights(
                    offset_mapping,
                    labels,
                    spans,
                    entity_text_token_weight,
                )
                item["loss_weights"] = torch.tensor(weights, dtype=torch.float32)
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in items)
        weighted = ["loss_weights" in item for item in items]
        if any(weighted) and not all(weighted):
            raise ValueError("cannot collate a mixture of weighted and unweighted examples")
        ids = torch.full((len(items), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        attention = torch.zeros((len(items), width), dtype=torch.long)
        loss_weights = torch.zeros((len(items), width), dtype=torch.float32) if all(weighted) else None
        for index, item in enumerate(items):
            length = len(item["input_ids"])
            if len(item["labels"]) != length or (
                loss_weights is not None and len(item["loss_weights"]) != length
            ):
                raise ValueError("encoded example fields do not align")
            ids[index, :length] = item["input_ids"]
            labels[index, :length] = item["labels"]
            attention[index, :length] = 1
            if loss_weights is not None:
                loss_weights[index, :length] = item["loss_weights"]
        batch = {"input_ids": ids, "attention_mask": attention, "labels": labels}
        if loss_weights is not None:
            batch["loss_weights"] = loss_weights
        return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--training-method", choices=("lora", "full"), default="lora")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--disease-row-weight", type=float, default=1.0)
    parser.add_argument("--validation-examples", type=int, default=384)
    parser.add_argument(
        "--gold-canonicalization",
        choices=("none", "first", "sorted"),
        default="none",
    )
    parser.add_argument("--entity-text-token-weight", type=float, default=1.0)
    parser.add_argument("--entity-substitution-examples", type=int, default=0)
    return parser.parse_args(argv)


def row_has_disease(row: dict[str, Any]) -> bool:
    raw = row.get("gold")
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid gold JSON for {row.get('ref', '<unknown>')}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        raise ValueError(f"invalid gold entities for {row.get('ref', '<unknown>')}")
    return any(
        isinstance(entity, dict) and str(entity.get("type", "")).casefold() == "disease"
        for entity in payload["entities"]
    )


def oversample_disease_rows(
    rows: list[dict[str, Any]], weight: float, seed: int
) -> tuple[list[dict[str, Any]], int, int]:
    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError("disease row weight must be finite and at least 1.0")
    disease_rows = [row for row in rows if row_has_disease(row)]
    extra_count = round((weight - 1.0) * len(disease_rows))
    if extra_count == 0:
        return list(rows), len(disease_rows), 0
    repeats, remainder = divmod(extra_count, len(disease_rows))
    extras = disease_rows * repeats
    if remainder:
        extras.extend(random.Random(seed ^ 0xD15EA5E).sample(disease_rows, remainder))
    return [*rows, *extras], len(disease_rows), len(extras)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def verify_training_input(path: Path) -> dict[str, str]:
    soup_metadata = path / weight_soup.METADATA_FILENAME
    soup_index = path / weight_soup.INDEX_FILENAME
    if (
        soup_metadata.exists()
        or soup_metadata.is_symlink()
        or soup_index.exists()
        or soup_index.is_symlink()
    ):
        validated = weight_soup.validate_weight_soup_checkpoint(path)
        return {
            "kind": "deterministic_weight_soup",
            "soup_schema": weight_soup.SCHEMA,
            "soup_metadata_digest": validated.metadata_digest,
            "output_manifest_digest": validated.output_manifest_digest,
            "index_digest": validated.index_digest,
            "tokenizer_digest": validated.tokenizer_digest,
        }

    weights = path / "model.safetensors"
    tokenizer = path / "tokenizer.json"
    if not weights.is_file() or not tokenizer.is_file():
        raise ValueError("training input must contain model.safetensors and tokenizer.json")
    identity = {
        "weights_digest": sha256(weights),
        "tokenizer_digest": sha256(tokenizer),
    }
    marker = path / ".cache" / "huggingface" / "trees" / f"{BASE_REVISION}.json"
    if marker.is_file():
        if (
            identity["weights_digest"] != BASE_WEIGHTS_DIGEST
            or identity["tokenizer_digest"] != BASE_TOKENIZER_DIGEST
        ):
            raise ValueError("base snapshot files do not match the allowlisted revision")
        return {"kind": "huggingface_snapshot", "revision": BASE_REVISION, **identity}

    parent = path.parent / "training_metadata.json"
    if not parent.is_file():
        raise ValueError("derived training input has no parent training metadata")
    try:
        parent_metadata = json.loads(parent.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"parent training metadata is unreadable: {exc}") from exc
    if parent_metadata.get("base_model") != f"{BASE_MODEL}@{BASE_REVISION}":
        raise ValueError("derived training input is not bound to the allowlisted base")
    return {
        "kind": "derived_model",
        "parent_metadata_digest": sha256(parent),
        **identity,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("version")) != CORPUS_VERSION:
        raise ValueError(f"expected corpus {CORPUS_VERSION}, got {payload.get('version')}")
    rows = [row for row in payload.get("tasks", []) if row.get("partition") == "train"]
    if len(rows) != 4_816:
        raise ValueError(f"expected 4816 public train rows, found {len(rows)}")
    return rows


def append_metric(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        if "loss_weights" in batch:
            raise ValueError("validation loss must remain ordinary unweighted causal-LM loss")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        total += float(loss) * int(batch["input_ids"].shape[0])
        count += int(batch["input_ids"].shape[0])
    model.train()
    return total / max(1, count)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this training recipe")

    settings = Settings(
        training_method=args.training_method,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        disease_row_weight=args.disease_row_weight,
        validation_examples=args.validation_examples,
        gold_canonicalization=args.gold_canonicalization,
        entity_text_token_weight=args.entity_text_token_weight,
        entity_substitution_examples=args.entity_substitution_examples,
    )
    if not 0.0 <= settings.lora_dropout < 1.0:
        raise SystemExit("--lora-dropout must be in [0, 1)")
    if (
        not math.isfinite(settings.entity_text_token_weight)
        or settings.entity_text_token_weight < 1.0
    ):
        raise SystemExit("--entity-text-token-weight must be finite and at least 1.0")
    if settings.entity_text_token_weight > 1.0 and settings.gold_canonicalization == "none":
        raise SystemExit("entity token weighting requires --gold-canonicalization first or sorted")
    if not 0 <= settings.entity_substitution_examples <= MAX_ENTITY_SUBSTITUTION_EXAMPLES:
        raise SystemExit(
            "--entity-substitution-examples must be in "
            f"[0, {MAX_ENTITY_SUBSTITUTION_EXAMPLES}]"
        )
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.out / "adapter"
    merged_dir = args.out / "merged"
    metrics_path = args.out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    training_input = verify_training_input(args.base)
    is_weight_soup = training_input["kind"] == "deterministic_weight_soup"
    rows = load_rows(args.corpus)
    random.Random(settings.seed).shuffle(rows)
    validation_rows = rows[: settings.validation_examples]
    source_train_rows = rows[settings.validation_examples :]
    base_train_rows, disease_source_examples, disease_extra_examples = oversample_disease_rows(
        source_train_rows, settings.disease_row_weight, settings.seed
    )
    augmentation = augment_train_fold_entity_substitutions(
        source_train_rows,
        heldout_refs={row["ref"] for row in validation_rows},
        seed=settings.seed,
        max_examples=settings.entity_substitution_examples,
    )
    train_rows = [*base_train_rows, *augmentation.rows]
    augmentation_metadata = {
        key: value for key, value in augmentation.manifest.items() if key != "examples"
    }
    if augmentation.manifest["enabled"]:
        augmentation_manifest_path = args.out / "entity_substitution_manifest.json"
        augmentation_manifest_path.write_text(
            json.dumps(augmentation.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        augmentation_metadata.update(
            {
                "manifest_file": augmentation_manifest_path.name,
                "manifest_digest": sha256(augmentation_manifest_path),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = EncodedDataset(
        train_rows,
        tokenizer,
        settings.max_length,
        gold_canonicalization=settings.gold_canonicalization,
        entity_text_token_weight=settings.entity_text_token_weight,
    )
    validation_data = EncodedDataset(
        validation_rows,
        tokenizer,
        settings.max_length,
        gold_canonicalization=settings.gold_canonicalization,
    )
    collator = Collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    if is_weight_soup and verify_training_input(args.base) != training_input:
        raise ValueError("weight-soup input identity changed before model load")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if is_weight_soup:
        parameters = tuple(model.parameters())
        buffers = tuple(model.buffers())
        if any(tensor.is_meta for tensor in (*parameters, *buffers)):
            raise ValueError("weight-soup model was not fully materialized by Transformers")
        with torch.no_grad():
            for tensor in (*parameters, *buffers):
                tensor.data = tensor.detach().clone()
        if verify_training_input(args.base) != training_input:
            raise ValueError("weight-soup input identity changed during model load")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if settings.training_method == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.lora_rank,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    device = torch.device("cuda:0")
    model.to(device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        fused=True,
    )
    updates_per_epoch = math.ceil(len(train_loader) / settings.gradient_accumulation)
    total_updates = updates_per_epoch * settings.epochs
    warmup = max(1, int(total_updates * settings.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_updates)

    metadata = {
        "hotkey": HOTKEY,
        "track": "extract",
        "hardware_class": "mt-3g",
        "base_model": f"{BASE_MODEL}@{BASE_REVISION}",
        "training_input": training_input,
        "corpus_version": CORPUS_VERSION,
        "corpus_file_digest": sha256(args.corpus),
        "settings": asdict(settings),
        "target_controls": {
            "entity_match": "exact_text_and_type_set",
            "gold_canonicalization": settings.gold_canonicalization,
            "entity_text_token_weight": settings.entity_text_token_weight,
            "validation_loss": "ordinary_unweighted_causal_lm",
        },
        "augmentation": {
            "entity_substitution": {
                **augmentation_metadata,
                "composition": "append_after_source_disease_oversampling",
            }
        },
        "training_examples": len(train_data),
        "source_training_examples": len(source_train_rows),
        "disease_source_examples": disease_source_examples,
        "disease_extra_examples": disease_extra_examples,
        "validation_examples": len(validation_data),
        "skipped_training_examples": train_data.skipped,
        "skipped_validation_examples": validation_data.skipped,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "started_at_unix": int(time.time()),
    }
    (args.out / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"trainable params: {trainable_count:,} || all params: {parameter_count:,} || "
        f"trainable%: {100.0 * trainable_count / parameter_count:.4f}",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    update = 0
    accumulated_loss = 0.0
    started = time.monotonic()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            loss_weights = batch.pop("loss_weights", None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if loss_weights is None:
                    raw_loss = model(**batch).loss
                else:
                    labels = batch.pop("labels")
                    output = model(**batch)
                    raw_loss = weighted_causal_lm_loss(output.logits, labels, loss_weights)
                loss = raw_loss / settings.gradient_accumulation
            loss.backward()
            accumulated_loss += float(loss.detach()) * settings.gradient_accumulation

            should_update = (
                micro_step % settings.gradient_accumulation == 0 or micro_step == len(train_loader)
            )
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            metric = {
                "step": update,
                "epoch": epoch,
                "loss": accumulated_loss / settings.gradient_accumulation,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            append_metric(metrics_path, metric)
            accumulated_loss = 0.0
            if update == 1 or update % 10 == 0:
                print(json.dumps(metric, sort_keys=True), flush=True)

        validation_loss = evaluate(model, validation_loader, device)
        metric = {
            "step": update,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(20.0, validation_loss)),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        append_metric(metrics_path, metric)
        print(json.dumps(metric, sort_keys=True), flush=True)
        if settings.training_method == "lora":
            model.save_pretrained(adapter_dir, safe_serialization=True)
            tokenizer.save_pretrained(adapter_dir)

    model.config.use_cache = True
    merged = model.merge_and_unload() if settings.training_method == "lora" else model
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merged_dir)
    metadata["finished_at_unix"] = int(time.time())
    metadata["elapsed_s"] = round(time.monotonic() - started, 3)
    metadata["updates"] = update
    (args.out / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"merged model: {merged_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
