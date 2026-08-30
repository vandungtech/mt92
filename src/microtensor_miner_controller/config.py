from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import ConfigError

UPSTREAM_COMMIT = "d0e002f887d038bf3ea4af65b499137a755620d7"
UPSTREAM_RELEASE = "0.1.14"
EXPECTED_ENTRYPOINT = "model.gguf"
EXPECTED_FORMAT = "gguf"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_MAX_INPUT_TOKENS = 512
EXPECTED_TOKENIZER = "tokenizer.json"
EXPECTED_BASE_MODEL = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"


def _text(env: Mapping[str, str], key: str, default: str = "", *, required: bool = False) -> str:
    value = env.get(key, default).strip()
    if required and not value:
        raise ConfigError(f"{key} is required")
    return value


def _integer(env: Mapping[str, str], key: str, default: int, *, minimum: int | None = None) -> int:
    raw = env.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, not {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}")
    return value


def _boolean(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be true or false, not {raw!r}")


def _path(env: Mapping[str, str], key: str, default: str, *, required: bool = False) -> Path:
    raw = _text(env, key, default, required=required)
    return Path(raw).expanduser().absolute()


def _validate_template(template: str) -> None:
    allowed = {"round", "uid", "hotkey"}
    fields: set[str] = set()
    try:
        for _, name, format_spec, conversion in string.Formatter().parse(template):
            if name is None:
                continue
            if name not in allowed or format_spec or conversion:
                raise ConfigError(
                    "MMC_SOURCE_TEMPLATE permits only plain {round}, {uid}, and {hotkey} fields"
                )
            fields.add(name)
    except ValueError as exc:
        raise ConfigError(f"MMC_SOURCE_TEMPLATE is malformed: {exc}") from exc
    if "round" not in fields:
        raise ConfigError("MMC_SOURCE_TEMPLATE must contain {round} to isolate each round")


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    netuid: int
    network: str
    endpoint: str
    wallet_name: str
    wallet_hotkey: str
    wallet_path: Path
    expected_uid: int
    upstream_home: Path
    state_dir: Path
    artifact_dir: Path
    selfcheck_path: Path
    selfcheck_binding_path: Path
    track: str
    hardware_class: str
    source_template: str
    entrypoint: str
    artifact_format: str
    quantization: str
    max_input_tokens: int
    tokenizer: str
    base_model: str
    coordinator_url: str
    allow_chain_schedule_fallback: bool
    require_anchored_coordinator: bool
    dry_run: bool
    allow_unverified_upstream: bool
    poll_seconds: int
    retry_seconds: int
    deadline_margin_blocks: int
    verify_attempts: int
    verify_interval_seconds: int
    health_max_age_seconds: int
    coordinator_timeout_seconds: int
    log_level: str
    round_blocks: int
    genesis_block: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ControllerConfig":
        source = dict(os.environ if env is None else env)
        upstream_home = _path(source, "MT_HOME", "/var/lib/microtensor-miner/upstream")
        state_dir = _path(source, "MMC_STATE_DIR", "/var/lib/microtensor-miner/controller")
        artifact_dir = _path(source, "MMC_ARTIFACT_DIR", "", required=True)
        selfcheck_default = str(upstream_home / "selfcheck.json")
        binding_default = str(upstream_home / "selfcheck.binding.json")
        source_template = _text(source, "MMC_SOURCE_TEMPLATE", required=True)
        _validate_template(source_template)

        config = cls(
            netuid=_integer(source, "MT_NETUID", 92, minimum=0),
            network=_text(source, "MT_NETWORK", "finney"),
            endpoint=_text(source, "MT_ENDPOINT"),
            wallet_name=_text(source, "MT_WALLET_NAME", "you-cold", required=True),
            wallet_hotkey=_text(source, "MT_WALLET_HOTKEY", "you-hot1", required=True),
            wallet_path=_path(source, "MT_WALLET_PATH", "", required=True),
            expected_uid=_integer(source, "MMC_EXPECTED_UID", 32, minimum=0),
            upstream_home=upstream_home,
            state_dir=state_dir,
            artifact_dir=artifact_dir,
            selfcheck_path=_path(source, "MMC_SELFCHECK_PATH", selfcheck_default),
            selfcheck_binding_path=_path(
                source, "MMC_SELFCHECK_BINDING_PATH", binding_default
            ),
            track=_text(source, "MMC_TRACK", "extract", required=True),
            hardware_class=_text(source, "MMC_HARDWARE_CLASS", "mt-3g", required=True),
            source_template=source_template,
            entrypoint=_text(source, "MMC_ENTRYPOINT", EXPECTED_ENTRYPOINT, required=True),
            artifact_format=_text(source, "MMC_ARTIFACT_FORMAT", EXPECTED_FORMAT, required=True),
            quantization=_text(source, "MMC_QUANTIZATION", EXPECTED_QUANTIZATION),
            max_input_tokens=_integer(
                source, "MMC_MAX_INPUT_TOKENS", EXPECTED_MAX_INPUT_TOKENS, minimum=1
            ),
            tokenizer=_text(source, "MMC_TOKENIZER", EXPECTED_TOKENIZER, required=True),
            base_model=_text(
                source,
                "MMC_BASE_MODEL",
                EXPECTED_BASE_MODEL,
                required=True,
            ),
            coordinator_url=_text(
                source, "MMC_COORDINATOR_URL", "https://coordinator.microtensor.cloud", required=True
            ).rstrip("/"),
            allow_chain_schedule_fallback=_boolean(
                source, "MMC_ALLOW_CHAIN_SCHEDULE_FALLBACK", False
            ),
            require_anchored_coordinator=_boolean(
                source, "MMC_REQUIRE_ANCHORED_COORDINATOR", True
            ),
            dry_run=_boolean(source, "MMC_DRY_RUN", True),
            allow_unverified_upstream=_boolean(source, "MMC_ALLOW_UNVERIFIED_UPSTREAM", False),
            poll_seconds=_integer(source, "MMC_POLL_SECONDS", 30, minimum=1),
            retry_seconds=_integer(source, "MMC_RETRY_SECONDS", 30, minimum=1),
            deadline_margin_blocks=_integer(
                source, "MMC_DEADLINE_MARGIN_BLOCKS", 40, minimum=1
            ),
            verify_attempts=_integer(source, "MMC_VERIFY_ATTEMPTS", 5, minimum=1),
            verify_interval_seconds=_integer(
                source, "MMC_VERIFY_INTERVAL_SECONDS", 900, minimum=30
            ),
            health_max_age_seconds=_integer(
                source, "MMC_HEALTH_MAX_AGE_SECONDS", 180, minimum=10
            ),
            coordinator_timeout_seconds=_integer(
                source, "MMC_COORDINATOR_TIMEOUT_SECONDS", 10, minimum=1
            ),
            log_level=_text(source, "MMC_LOG_LEVEL", "INFO").upper(),
            round_blocks=_integer(source, "MT_ROUND_BLOCKS", 21600, minimum=1),
            genesis_block=_integer(source, "MT_GENESIS_BLOCK", -17805488),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if self.netuid != 92:
            raise ConfigError("this deployment is pinned to Microtensor netuid 92")
        if self.network != "finney" and not self.endpoint:
            raise ConfigError("MT_NETWORK must be finney unless MT_ENDPOINT is explicitly set")
        if self.wallet_name != "you-cold":
            raise ConfigError("MT_WALLET_NAME must be you-cold for this deployment")
        if self.wallet_hotkey != "you-hot1":
            raise ConfigError("MT_WALLET_HOTKEY must be you-hot1 for this deployment")
        if self.expected_uid != 32:
            raise ConfigError("MMC_EXPECTED_UID must remain 32 for this registered miner")
        if (self.track, self.hardware_class) != ("extract", "mt-3g"):
            raise ConfigError("the pinned upstream currently opens only extract/mt-3g")
        expected_load = (
            EXPECTED_ENTRYPOINT,
            EXPECTED_FORMAT,
            EXPECTED_QUANTIZATION,
            EXPECTED_MAX_INPUT_TOKENS,
            EXPECTED_TOKENIZER,
            EXPECTED_BASE_MODEL,
        )
        observed_load = (
            self.entrypoint,
            self.artifact_format,
            self.quantization,
            self.max_input_tokens,
            self.tokenizer,
            self.base_model,
        )
        if observed_load != expected_load:
            raise ConfigError(
                "this deployment accepts only model.gguf / gguf / Q4_K_M / 512 tokens / "
                "tokenizer.json / the pinned Qwen3-0.6B revision"
            )
        if not self.coordinator_url.startswith("https://"):
            raise ConfigError("MMC_COORDINATOR_URL must use HTTPS")
        if not self.source_template.startswith(("s3:", "r2:")):
            raise ConfigError(
                "supervised live uploads support only per-round s3: or r2: sources; "
                "HF revisions are immutable and https/ipfs have no upstream uploader"
            )
        if self.allow_unverified_upstream and not self.dry_run:
            raise ConfigError("MMC_ALLOW_UNVERIFIED_UPSTREAM is permitted only with MMC_DRY_RUN=true")
        if not self.dry_run:
            raise ConfigError(
                "live S3/R2 activation is refused: verification with miner credentials does not "
                "prove validators can fetch the artifact without those credentials"
            )
        if not self.require_anchored_coordinator:
            raise ConfigError("MMC_REQUIRE_ANCHORED_COORDINATOR must remain true")
        if self.allow_chain_schedule_fallback and not self.dry_run:
            raise ConfigError("chain-schedule fallback is diagnostic-only and requires dry-run")

    def source_for(self, round_index: int, hotkey: str) -> str:
        if round_index < 0:
            raise ConfigError("round index must not be negative")
        rendered = self.source_template.format(
            round=round_index,
            uid=self.expected_uid,
            hotkey=hotkey,
        )
        if (
            "|" in rendered
            or any(c.isspace() for c in rendered)
            or any(c in rendered for c in "?#@\\")
        ):
            raise ConfigError("rendered source contains a forbidden delimiter or credential marker")
        _, _, locator = rendered.partition(":")
        bucket, separator, prefix = locator.partition("/")
        if not separator or not bucket or not prefix or "//" in locator:
            raise ConfigError("rendered object-store source must be bucket/non-empty-prefix")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ConfigError("rendered object-store bucket name is invalid")
        if any(part in {"", ".", ".."} for part in prefix.split("/")):
            raise ConfigError("rendered object-store source contains an unsafe path component")
        return rendered
