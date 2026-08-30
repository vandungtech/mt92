#!/usr/bin/env python3
"""Publish an existing local training trail after the final artifact digest exists."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb


ENTITY = "microtensor"
PROJECT = "training-runs"
HOTKEY = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--finished-block", type=int, required=True)
    args = parser.parse_args()
    if not args.artifact_digest.startswith("sha256:"):
        raise SystemExit("--artifact-digest must be the full sha256:... artifact digest")
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is required; no anonymous/fabricated fallback is used")

    metadata = json.loads((args.training_dir / "training_metadata.json").read_text())
    metrics = [
        json.loads(line)
        for line in (args.training_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=HOTKEY,
        config={
            **metadata,
            "mt_track": "extract",
            "mt_class": "mt-3g",
            "mt_base_model": metadata["base_model"],
            "mt_corpus_version": metadata["corpus_version"],
        },
    )
    for metric in metrics:
        step = int(metric.get("step", 0))
        wandb.log(metric, step=step)
    run.summary["mt_artifact_digest"] = args.artifact_digest
    run.summary["mt_finished_at"] = args.finished_block
    run.summary["mt_training_records"] = len(metrics)
    wandb.finish()
    print(f"published {len(metrics)} records for {HOTKEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
