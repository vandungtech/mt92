#!/usr/bin/env python3
"""Evaluate the deterministic held-out split with Microtensor's exact entity metric."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_extract import CORPUS_VERSION, load_rows

Entity = tuple[str, str]


def parse_entities(value: Any) -> set[Entity] | None:
    """Mirror the strict schema used by microtensor.scoring.extraction."""
    if isinstance(value, str):
        try:
            value = json.loads(value.strip())
        except ValueError:
            return None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 1
        and isinstance(value[0], Mapping)
        and "entities" in value[0]
    ):
        value = value[0]
    items = value.get("entities") if isinstance(value, Mapping) else value
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return None
    entities: set[Entity] = set()
    for item in items:
        if not isinstance(item, Mapping):
            return None
        text_value = item.get("text")
        kind = item.get("type")
        if not isinstance(text_value, str) or not isinstance(kind, str):
            return None
        pair = (text_value.strip(), kind.strip())
        if not pair[0] or not pair[1]:
            return None
        entities.add(pair)
    return entities


def counts(prediction: set[Entity] | None, gold: set[Entity]) -> tuple[int, int, int]:
    found = prediction or set()
    return len(found & gold), len(found - gold), len(gold - found)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--examples", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    rows = load_rows(args.corpus)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.examples]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    true_positive = false_positive = false_negative = malformed = 0
    generated = 0
    with args.output.open("w", encoding="utf-8") as sink:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(row["prompt"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to("cuda:0")
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output[:, encoded.input_ids.shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, generated_text in zip(batch, texts, strict=True):
                prediction = parse_entities(generated_text)
                gold = parse_entities(str(row["gold"]))
                if gold is None:
                    raise ValueError(f"malformed corpus gold for {row['ref']}")
                tp, fp, fn = counts(prediction, gold)
                true_positive += tp
                false_positive += fp
                false_negative += fn
                malformed += int(prediction is None)
                generated += 1
                sink.write(
                    json.dumps(
                        {
                            "ref": row["ref"],
                            "output": generated_text,
                            "prediction": sorted(prediction) if prediction is not None else None,
                            "gold": sorted(gold),
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(f"generated {generated}/{len(rows)}", flush=True)

    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    summary = {
        "corpus_version": CORPUS_VERSION,
        "examples": generated,
        "entity_micro_f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "malformed_outputs": malformed,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
