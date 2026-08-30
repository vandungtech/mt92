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
        "MMC_VERIFY_INTERVAL_SECONDS": "900",
    }


def coordinator_payload(
    *, index: int = 7, start: int = 100, close: int = 200, end: int = 300,
    phase: str = "submissions", anchored: bool = True,
) -> dict[str, object]:
    config = {
        "competitions": [["extract", "mt-3g"]],
        "arenas": {
            "extract/mt-3g": {
                "allowed_base_models": [
                    "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
                ],
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
