#!/usr/bin/env python3
"""Score a quantized GGUF with the exact pinned Microtensor engine and metric."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.core.tracks import Decoding
from microtensor.harness.contract import Request
from microtensor.harness.engines.gguf import GgufEngine
from microtensor.scoring.extraction import gold_entities, micro_f1, parse_entities

CORPUS_VERSION = "sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8"


def load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("version")) != CORPUS_VERSION:
        raise ValueError(f"expected corpus {CORPUS_VERSION}, got {payload.get('version')}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("corpus tasks must be a list")
    rows = [row for row in tasks if isinstance(row, dict) and row.get("partition") == "train"]
    if len(rows) != 4_816:
        raise ValueError(f"expected 4816 public train rows, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--quantization", default="Q4_K_M")
    args = parser.parse_args()

    rows = load_rows(args.corpus)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.examples]
    engine = GgufEngine()
    engine.load(
        args.model,
        LoadManifest(
            format=ArtifactFormat.GGUF,
            quantization=args.quantization,
            entrypoint=args.model.name,
            max_input={"tokens": 512},
            preprocessing={"tokenizer": "tokenizer.json"},
            base_model=(
                "Qwen/Qwen3-0.6B@"
                "c1899de289a04d12100db370d81485cdf75e47ca"
            ),
        ),
    )

    predictions = []
    golds = []
    malformed = 0
    latency_ms = []
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as sink:
            for index, row in enumerate(rows, start=1):
                response = engine.generate(
                    Request(
                        task_ref=str(row["ref"]),
                        prompt=str(row["prompt"]),
                        inputs=dict(row.get("inputs") or {}),
                        max_output_tokens=int(row.get("max_output_tokens", 512)),
                        decoding=Decoding.GREEDY,
                        chat=True,
                        seed=0,
                        nonce="local-heldout",
                    )
                )
                prediction = parse_entities(response.output) if response.ok else None
                gold = gold_entities(row["gold"])
                predictions.append(prediction)
                golds.append(gold)
                malformed += int(prediction is None)
                latency_ms.append(response.total_ms)
                sink.write(
                    json.dumps(
                        {
                            "ref": row["ref"],
                            "ok": response.ok,
                            "output": response.output,
                            "error": response.error,
                            "latency_ms": response.total_ms,
                            "prediction": sorted(prediction) if prediction is not None else None,
                            "gold": sorted(gold),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                if index == 1 or index % 8 == 0:
                    print(f"generated {index}/{len(rows)}", flush=True)
    finally:
        engine.unload()

    ordered_latency = sorted(latency_ms)
    p95_index = max(0, min(len(ordered_latency) - 1, int(len(ordered_latency) * 0.95)))
    summary = {
        "examples": len(rows),
        "entity_micro_f1": micro_f1(predictions, golds),
        "malformed_outputs": malformed,
        "mean_latency_ms": sum(latency_ms) / max(1, len(latency_ms)),
        "p95_latency_ms": ordered_latency[p95_index] if ordered_latency else 0.0,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
