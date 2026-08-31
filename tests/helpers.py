from __future__ import annotations

import hashlib
import json
from pathlib import Path


def base_env(root: Path, *, dry_run: bool = True) -> dict[str, str]:
    return {
        "MT_NETUID": "92",
        "MT_NETWORK": "finney",
        "MT_WALLET_NAME": "you-cold",
        "MT_WALLET_HOTKEY": "you-hot1",
        "MT_WALLET_PATH": str(root / "wallets"),
        "MT_HOME": str(root / "upstream"),
        "MMC_EXPECTED_UID": "32",
        "MMC_STATE_DIR": str(root / "state"),
        "MMC_ARTIFACT_DIR": str(root / "artifact"),
        "MMC_SELFCHECK_PATH": str(root / "upstream" / "selfcheck.json"),
        "MMC_TRACK": "extract",
        "MMC_HARDWARE_CLASS": "mt-3g",
        "MMC_ENTRYPOINT": "model.gguf",
        "MMC_ARTIFACT_FORMAT": "gguf",
        "MMC_QUANTIZATION": "Q4_K_M",
        "MMC_MAX_INPUT_TOKENS": "512",
        "MMC_TOKENIZER": "tokenizer.json",
        "MMC_BASE_MODEL": "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
        "MMC_SOURCE_TEMPLATE": "s3:public/uid-{uid}/round-{round}",
        "MMC_COORDINATOR_URL": "https://coordinator.example",
        "MMC_DRY_RUN": "true" if dry_run else "false",
        "MMC_TRANSACTION_AUTHORIZATION": ("netuid92-uid32-you-hot1-commitment-fee0-deposit0-v1"),
        "MMC_VERIFY_INTERVAL_SECONDS": "900",
    }


def coordinator_payload(
    *,
    index: int = 7,
    start: int = 100,
    close: int = 200,
    end: int = 300,
    phase: str = "submissions",
    anchored: bool = True,
) -> dict[str, object]:
    config = {
        "competitions": [["extract", "mt-3g"]],
        "arenas": {
            "extract/mt-3g": {
                "allowed_base_models": ["Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"],
                "ceilings": {
                    "max_size_bytes": 1_610_612_736,
                    "max_rss_bytes": 3_221_225_472,
                    "max_p95_ms": 15_000,
                },
            }
        },
        "version": 1,
    }
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "round": index,
        "start_block": start,
        "close_block": close,
        "end_block": end,
        "phase": phase,
        "anchored": anchored,
        "config": config,
        "config_hash": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


V030_TEST_BASE_MODEL = "Fixture/CodeModel@abcdef0"


def v030_env(root: Path, *, dry_run: bool = True, activation_block: int = 100) -> dict[str, str]:
    env = base_env(root, dry_run=dry_run)
    env.update(
        {
            "MMC_V030_ACTIVATION_BLOCK": str(activation_block),
            "MMC_TRACK": "code",
            "MMC_HARDWARE_CLASS": "mt-3g",
            "MMC_ENTRYPOINT": "candidate.gguf",
            "MMC_ARTIFACT_FORMAT": "gguf",
            "MMC_QUANTIZATION": "TEST_Q8",
            "MMC_MAX_INPUT_TOKENS": "1024",
            "MMC_TOKENIZER": "tokenizer.json",
            "MMC_BASE_MODEL": V030_TEST_BASE_MODEL,
        }
    )
    return env


def v030_coordinator_payload(
    *,
    index: int = 7,
    start: int = 100,
    close: int = 7_300,
    end: int = 14_500,
    phase: str = "submissions",
    anchored: bool = True,
    base_model: str = V030_TEST_BASE_MODEL,
    block_hash: str | None = None,
) -> dict[str, object]:
    config = {
        "version": 1,
        "mechanism_version": "0.3.0",
        "corpus_version": "2026.1",
        "also_accept_rounds": [],
        "genesis_block": -17_805_488,
        "round_blocks": 21_600,
        "submission_closes_before_blocks": 7_200,
        "tasks_per_round": 200,
        "replication": 3,
        "competitions": [["code", "mt-3g"]],
        "tracks": {"code": {"metric": "execution_pass_rate", "emission_share": 1.0}},
        "class_weights": {"mt-3g": 1.0},
        "classes": {
            "mt-3g": {
                "max_size_bytes": 1_610_612_736,
                "max_rss_bytes": 3_221_225_472,
                "max_p95_ms": 45_000,
            }
        },
        "arenas": {
            "code/mt-3g": {
                "allowed_base_models": [base_model],
                "cpu_seconds_per_artifact": 12_000,
                "tasks_per_round": 60,
                "ceilings": {
                    "max_size_bytes": 1_610_612_736,
                    "max_rss_bytes": 3_221_225_472,
                    "max_p95_ms": 45_000,
                },
                "environment_digest": "env:a9b6b17587d8aaea",
            }
        },
        "role_baselines": {"front": "", "router": "", "specialist": ""},
    }
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if block_hash is None:
        block_hash = "" if phase == "submissions" else "0x" + "c" * 64
    return {
        "corpus_digest": "sha256:" + "b" * 64,
        "round": index,
        "settled": False,
        "phase": phase,
        "start_block": start,
        "seed_block": close,
        "close_block": close,
        "end_block": end,
        "block_hash": block_hash,
        "anchored": anchored,
        "config": config,
        "config_hash": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
