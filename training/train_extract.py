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


BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CORPUS_VERSION = "sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8"
HOTKEY = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"
BASE_WEIGHTS_DIGEST = "sha256:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
BASE_TOKENIZER_DIGEST = "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


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


@dataclass(frozen=True)
class CanonicalTarget:
    """A strict scorer-equivalent target and its entity-value character spans."""

    content: str
    entity_text_spans: tuple[tuple[int, int], ...]


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
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.out / "adapter"
    merged_dir = args.out / "merged"
    metrics_path = args.out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    training_input = verify_training_input(args.base)
    rows = load_rows(args.corpus)
    random.Random(settings.seed).shuffle(rows)
    validation_rows = rows[: settings.validation_examples]
    source_train_rows = rows[settings.validation_examples :]
    train_rows, disease_source_examples, disease_extra_examples = oversample_disease_rows(
        source_train_rows, settings.disease_row_weight, settings.seed
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

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
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
