from __future__ import annotations

import os
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError
from .upstream_gate import DEFAULT_MAX_AGE_SECONDS, MIN_MAX_AGE_SECONDS

UPSTREAM_COMMIT = "d0e002f887d038bf3ea4af65b499137a755620d7"
UPSTREAM_RELEASE = "0.1.14"
SIGNED_V030_RELEASE = "0.3.2"
SIGNED_V030_MECHANISM_VERSION = "0.3.0"
SIGNED_V030_WHEEL_SHA256 = "3629a48b248365070bf5bf190c1584498019c4e1e5ced7c3d175472ff0749e71"
SIGNED_V030_INSTALLED_TREE_SCHEMA = "microtensor.signed-wheel-tree.v1"
SIGNED_V030_INSTALLED_TREE_FILES = 137
SIGNED_V030_INSTALLED_TREE_BYTES = 901_899
SIGNED_V030_INSTALLED_TREE_SHA256 = (
    "f93d75ef1bc4d2fc9ffbb5e7cd63b37a00bc1a57e6cdee158219a5d6983b8c92"
)
SIGNED_V030_RELEASE_SIGNING_KEY = (
    "0x3d8ea239db66637d762ffedf71ad6c0c487c7bc73d5a50d9dd86a0fbc22bdb16"
)
SIGNED_V030_PROVENANCE_REQUIRED = False
SIGNED_V030_CONFIG_VERSION = 1
SIGNED_V030_CORPUS_VERSION = "2026.1"
SIGNED_V030_TRACK = "code"
SIGNED_V030_HARDWARE_CLASS = "mt-3g"
SIGNED_V030_METRIC = "execution_pass_rate"
SIGNED_V030_EMISSION_SHARE = 1.0
SIGNED_V030_COORDINATOR_URL = "https://coordinator.microtensor.cloud"
BITTENSOR_VERSION = "10.5.0"
BITTENSOR_WALLET_VERSION = "4.1.1"
SUBSTRATE_INTERFACE_VERSION = "2.2.1"
FINNEY_GENESIS_HASH = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
FINNEY_RUNTIME_SPEC_VERSION = 452
FINNEY_TRANSACTION_VERSION = 1
FINNEY_RUNTIME_CODE_HASH = "0x40a8c3c99a47d6739b086236308535fab26d5fd4cc5c88eb83f6a3c8b928f7cc"
TRANSACTION_AUTHORIZATION = "netuid92-uid32-you-hot1-commitment-fee0-deposit0-v1"
AUTHORIZED_GITHUB_REPOSITORY = "vandungtech/mt92"

# Upstream 53e4df6 added an opt-in coordinator override, MT_RESULT_WORKER, that collapses
# reconciliation to a single named worker's report. It is unpublished, unanchored, and not
# covered by the signed v0.3.2 runtime or any on-chain commitment. A miner that carried it
# in its environment could contribute to validators computing different weights for the same
# round, so this controller refuses to load while it is set. See
# docs/upstream-audits/53e4df648a89fad6586e1ac69916b20e747fd972.md.
FORBIDDEN_ENVIRONMENT_VARIABLES: tuple[str, ...] = ("MT_RESULT_WORKER",)
AUTHORIZED_HOTKEY_SS58 = "5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r"
EXPECTED_ENTRYPOINT = "model.gguf"
EXPECTED_FORMAT = "gguf"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_MAX_INPUT_TOKENS = 512
EXPECTED_TOKENIZER = "tokenizer.json"
EXPECTED_BASE_MODEL = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
_GITHUB_RELEASE_SOURCE = re.compile(
    r"^github\.com/"
    r"([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)/"
    r"([A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?)/"
    r"releases/download/"
    r"([A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?)$"
)


def _reject_forbidden(env: Mapping[str, str]) -> None:
    """Refuse to build a configuration while a forbidden variable is set.

    Presence with an empty or whitespace-only value is tolerated so an operator can
    keep an explicit ``MT_RESULT_WORKER=`` placeholder that proves the override is
    off. Any non-empty value is a hard refusal.
    """

    for name in FORBIDDEN_ENVIRONMENT_VARIABLES:
        value = env.get(name)
        if value is not None and value.strip():
            raise ConfigError(
                f"{name} must never be set for this miner; it is an unpublished, unanchored "
                "upstream override that can make validators compute different weights"
            )


def _text(env: Mapping[str, str], key: str, default: str = "", *, required: bool = False) -> str:
    value = env.get(key, default).strip()
    if required and not value:
        raise ConfigError(f"{key} is required")
    return value


def _integer(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = env.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, not {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be at most {maximum}")
    return value


def _optional_integer(
    env: Mapping[str, str], key: str, *, minimum: int | None = None
) -> int | None:
    raw = env.get(key, "").strip()
    if not raw:
        return None
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


def _optional_path(env: Mapping[str, str], key: str) -> Path | None:
    raw = _text(env, key)
    return Path(raw).expanduser().absolute() if raw else None


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
    upstream_observer_status_path: Path
    upstream_observer_max_age_seconds: int
    artifact_dir: Path
    artifact_competition_binding_path: Path
    selfcheck_path: Path
    selfcheck_binding_path: Path
    v030_activation_block: int | None
    track: str
    hardware_class: str
    source_template: str
    github_token_file: Path | None
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
    transaction_authorization: str
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
    def from_env(cls, env: Mapping[str, str] | None = None) -> ControllerConfig:
        source = dict(os.environ if env is None else env)
        _reject_forbidden(source)
        v030_activation_block = _optional_integer(source, "MMC_V030_ACTIVATION_BLOCK", minimum=0)
        signed_v030 = v030_activation_block is not None
        upstream_home = _path(source, "MT_HOME", "/var/lib/microtensor-miner/upstream")
        state_dir = _path(source, "MMC_STATE_DIR", "/var/lib/microtensor-miner/controller")
        observer_status_default = str(state_dir.parent / "upstream-observer" / "status.json")
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
            upstream_observer_status_path=_path(
                source,
                "MMC_UPSTREAM_OBSERVER_STATUS_PATH",
                observer_status_default,
            ),
            upstream_observer_max_age_seconds=_integer(
                source,
                "MMC_UPSTREAM_OBSERVER_MAX_AGE_SECONDS",
                DEFAULT_MAX_AGE_SECONDS,
                minimum=MIN_MAX_AGE_SECONDS,
                maximum=DEFAULT_MAX_AGE_SECONDS,
            ),
            artifact_dir=artifact_dir,
            artifact_competition_binding_path=_path(
                source,
                "MMC_ARTIFACT_COMPETITION_BINDING_PATH",
                "",
                required=True,
            ),
            selfcheck_path=_path(source, "MMC_SELFCHECK_PATH", selfcheck_default),
            selfcheck_binding_path=_path(source, "MMC_SELFCHECK_BINDING_PATH", binding_default),
            v030_activation_block=v030_activation_block,
            track=_text(
                source, "MMC_TRACK", SIGNED_V030_TRACK if signed_v030 else "extract", required=True
            ),
            hardware_class=_text(
                source,
                "MMC_HARDWARE_CLASS",
                SIGNED_V030_HARDWARE_CLASS if signed_v030 else "mt-3g",
                required=True,
            ),
            source_template=source_template,
            github_token_file=_optional_path(source, "MMC_GITHUB_TOKEN_FILE"),
            entrypoint=_text(source, "MMC_ENTRYPOINT", "" if signed_v030 else EXPECTED_ENTRYPOINT),
            artifact_format=_text(
                source, "MMC_ARTIFACT_FORMAT", "" if signed_v030 else EXPECTED_FORMAT
            ),
            quantization=_text(
                source, "MMC_QUANTIZATION", "" if signed_v030 else EXPECTED_QUANTIZATION
            ),
            max_input_tokens=_integer(
                source,
                "MMC_MAX_INPUT_TOKENS",
                0 if signed_v030 else EXPECTED_MAX_INPUT_TOKENS,
                minimum=0 if signed_v030 else 1,
            ),
            tokenizer=_text(source, "MMC_TOKENIZER", "" if signed_v030 else EXPECTED_TOKENIZER),
            base_model=_text(
                source,
                "MMC_BASE_MODEL",
                "" if signed_v030 else EXPECTED_BASE_MODEL,
            ),
            coordinator_url=_text(
                source,
                "MMC_COORDINATOR_URL",
                "https://coordinator.microtensor.cloud",
                required=True,
            ).rstrip("/"),
            allow_chain_schedule_fallback=_boolean(
                source, "MMC_ALLOW_CHAIN_SCHEDULE_FALLBACK", False
            ),
            require_anchored_coordinator=_boolean(source, "MMC_REQUIRE_ANCHORED_COORDINATOR", True),
            dry_run=_boolean(source, "MMC_DRY_RUN", True),
            transaction_authorization=_text(source, "MMC_TRANSACTION_AUTHORIZATION"),
            allow_unverified_upstream=_boolean(source, "MMC_ALLOW_UNVERIFIED_UPSTREAM", False),
            poll_seconds=_integer(source, "MMC_POLL_SECONDS", 30, minimum=1),
            retry_seconds=_integer(source, "MMC_RETRY_SECONDS", 30, minimum=1),
            deadline_margin_blocks=_integer(source, "MMC_DEADLINE_MARGIN_BLOCKS", 40, minimum=1),
            verify_attempts=_integer(source, "MMC_VERIFY_ATTEMPTS", 5, minimum=1),
            verify_interval_seconds=_integer(
                source, "MMC_VERIFY_INTERVAL_SECONDS", 900, minimum=30
            ),
            health_max_age_seconds=_integer(source, "MMC_HEALTH_MAX_AGE_SECONDS", 180, minimum=10),
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
        if not self.dry_run and self.network != "finney":
            raise ConfigError("live mode requires MT_NETWORK=finney")
        if not self.dry_run and self.endpoint:
            raise ConfigError("live mode refuses custom MT_ENDPOINT values")
        if self.network != "finney" and not self.endpoint:
            raise ConfigError("non-finney diagnostics require an explicit MT_ENDPOINT")
        if self.wallet_name != "you-cold":
            raise ConfigError("MT_WALLET_NAME must be you-cold for this deployment")
        if self.wallet_hotkey != "you-hot1":
            raise ConfigError("MT_WALLET_HOTKEY must be you-hot1 for this deployment")
        if self.expected_uid != 32:
            raise ConfigError("MMC_EXPECTED_UID must remain 32 for this registered miner")
        observed_load = (
            self.entrypoint,
            self.artifact_format,
            self.quantization,
            self.max_input_tokens,
            self.tokenizer,
            self.base_model,
        )
        if self.uses_signed_v030:
            if (self.track, self.hardware_class) != (
                SIGNED_V030_TRACK,
                SIGNED_V030_HARDWARE_CLASS,
            ):
                raise ConfigError(
                    "MMC_V030_ACTIVATION_BLOCK enables only the signed code/mt-3g profile"
                )
            if (
                not all(
                    (
                        self.entrypoint,
                        self.artifact_format,
                        self.quantization,
                        self.tokenizer,
                        self.base_model,
                    )
                )
                or self.max_input_tokens < 1
            ):
                raise ConfigError(
                    "signed v0.3 activation requires an explicit final model/load spec; "
                    "no candidate is selected by default"
                )
            if self.artifact_format != "gguf" or not self.entrypoint.endswith(".gguf"):
                raise ConfigError("signed v0.3 activation currently supports only a GGUF load spec")
            if self.allow_unverified_upstream:
                raise ConfigError("signed v0.3 activation never permits an unverified upstream")
            if self.allow_chain_schedule_fallback:
                raise ConfigError("signed v0.3 activation never permits chain-schedule fallback")
            if not self.dry_run and self.coordinator_url != SIGNED_V030_COORDINATOR_URL:
                raise ConfigError(
                    "live signed v0.3 activation requires the official coordinator URL "
                    f"{SIGNED_V030_COORDINATOR_URL}"
                )
        else:
            if (self.track, self.hardware_class) != ("extract", "mt-3g"):
                raise ConfigError("code/mt-3g requires an explicit MMC_V030_ACTIVATION_BLOCK")
            expected_load = (
                EXPECTED_ENTRYPOINT,
                EXPECTED_FORMAT,
                EXPECTED_QUANTIZATION,
                EXPECTED_MAX_INPUT_TOKENS,
                EXPECTED_TOKENIZER,
                EXPECTED_BASE_MODEL,
            )
            if observed_load != expected_load:
                raise ConfigError(
                    "this deployment accepts only model.gguf / gguf / Q4_K_M / 512 tokens / "
                    "tokenizer.json / the pinned Qwen3-0.6B revision"
                )
        if not self.coordinator_url.startswith("https://"):
            raise ConfigError("MMC_COORDINATOR_URL must use HTTPS")
        scheme = self.source_template.partition(":")[0]
        if scheme not in {"s3", "r2", "https"}:
            raise ConfigError(
                "supervised uploads support s3/r2 diagnostics or an immutable GitHub https release"
            )
        self.source_for(0, "5ConfigProbe")
        if self.transaction_authorization not in {"", TRANSACTION_AUTHORIZATION}:
            raise ConfigError("MMC_TRANSACTION_AUTHORIZATION is not an approved policy")
        if not self.dry_run and self.transaction_authorization != TRANSACTION_AUTHORIZATION:
            raise ConfigError(
                "live mode requires the exact zero-cost MMC_TRANSACTION_AUTHORIZATION"
            )
        if scheme == "https" and self.github_token_file is None and not self.dry_run:
            raise ConfigError("MMC_GITHUB_TOKEN_FILE is required for live GitHub releases")
        if scheme != "https" and self.github_token_file is not None:
            raise ConfigError("MMC_GITHUB_TOKEN_FILE is valid only with a GitHub https source")
        if self.allow_unverified_upstream and not self.dry_run:
            raise ConfigError(
                "MMC_ALLOW_UNVERIFIED_UPSTREAM is permitted only with MMC_DRY_RUN=true"
            )
        if not self.dry_run and scheme != "https":
            raise ConfigError(
                "live S3/R2 activation is refused: verification with miner credentials does not "
                "prove validators can fetch the artifact without those credentials"
            )
        if not self.require_anchored_coordinator:
            raise ConfigError("MMC_REQUIRE_ANCHORED_COORDINATOR must remain true")
        if self.allow_chain_schedule_fallback and not self.dry_run:
            raise ConfigError("chain-schedule fallback is diagnostic-only and requires dry-run")

    @property
    def uses_signed_v030(self) -> bool:
        return self.v030_activation_block is not None

    @property
    def upstream_release(self) -> str:
        return SIGNED_V030_RELEASE if self.uses_signed_v030 else UPSTREAM_RELEASE

    @property
    def provenance_required(self) -> bool:
        return SIGNED_V030_PROVENANCE_REQUIRED if self.uses_signed_v030 else True

    @staticmethod
    def github_release_coordinates(source: str) -> tuple[str, str, str] | None:
        if not source.startswith("https:"):
            return None
        match = _GITHUB_RELEASE_SOURCE.fullmatch(source.removeprefix("https:"))
        if match is None:
            return None
        owner, repo, tag = match.groups()
        return owner, repo, tag

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
        scheme, separator, locator = rendered.partition(":")
        if not separator:
            raise ConfigError("rendered source has no scheme")
        if scheme == "https":
            coordinates = self.github_release_coordinates(rendered)
            if coordinates is None:
                raise ConfigError(
                    "https sources must be github.com/OWNER/REPO/releases/download/TAG"
                )
            owner, repo, tag = coordinates
            if f"{owner}/{repo}" != AUTHORIZED_GITHUB_REPOSITORY:
                raise ConfigError(
                    "GitHub release source is not the authorized vandungtech/mt92 repository"
                )
            if (
                "--" in owner
                or repo.casefold().endswith(".git")
                or ".." in tag
                or tag.casefold().endswith(".lock")
            ):
                raise ConfigError("rendered GitHub release source is invalid")
            return rendered
        if scheme not in {"s3", "r2"}:
            raise ConfigError("rendered source scheme is not supervised")
        bucket, separator, prefix = locator.partition("/")
        if not separator or not bucket or not prefix or "//" in locator:
            raise ConfigError("rendered object-store source must be bucket/non-empty-prefix")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ConfigError("rendered object-store bucket name is invalid")
        if any(part in {"", ".", ".."} for part in prefix.split("/")):
            raise ConfigError("rendered object-store source contains an unsafe path component")
        return rendered
