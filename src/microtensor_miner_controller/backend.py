from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
import logging
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

from .binding import validate_binding
from .config import (
    AUTHORIZED_GITHUB_REPOSITORY,
    AUTHORIZED_HOTKEY_SS58,
    BITTENSOR_VERSION,
    BITTENSOR_WALLET_VERSION,
    FINNEY_GENESIS_HASH,
    FINNEY_RUNTIME_CODE_HASH,
    FINNEY_RUNTIME_SPEC_VERSION,
    FINNEY_TRANSACTION_VERSION,
    SIGNED_V030_MECHANISM_VERSION,
    SIGNED_V030_RELEASE,
    SIGNED_V030_RELEASE_SIGNING_KEY,
    SIGNED_V030_WHEEL_SHA256,
    SUBSTRATE_INTERFACE_VERSION,
    TRANSACTION_AUTHORIZATION,
    UPSTREAM_COMMIT,
    UPSTREAM_RELEASE,
    ControllerConfig,
)
from .errors import AuthorizationRefused, PreflightError, VerificationError
from .models import (
    PackagedArtifact,
    PreflightSnapshot,
    PublishReceipt,
    RoundWindow,
)

log = logging.getLogger(__name__)

_PINNED_MODEL = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{7,40}$")
_AUTHORIZED_CALL_MODULE = "Commitments"
_AUTHORIZED_CALL_FUNCTION = "set_commitment"
_AUTHORIZED_PERIOD_BLOCKS = 128
_AUTHORIZED_MAX_FEE_RAO = 0
_AUTHORIZED_MAX_DEPOSIT_RAO = 0
_AUTHORIZED_SIGNATURE_VERSION = 1
_RUNTIME_CODE_STORAGE_KEY = "0x3a636f6465"
_HASH = re.compile(r"^0x[0-9a-f]{64}$")


def _strict_json_object(raw: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for key, value in pairs:
            if key in found:
                raise ValueError(f"duplicate JSON key {key!r}")
            found[key] = value
        return found

    def nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


def _publishable_files(manifest: Any) -> tuple[str, ...]:
    from microtensor.miner.package import publishable_files

    return tuple(publishable_files(manifest))


class Backend(Protocol):
    def preflight(self) -> PreflightSnapshot: ...

    def block(self) -> int: ...

    def chain_round(self, block: int) -> RoundWindow: ...

    def verify_round_anchor(self, window: RoundWindow) -> None: ...

    def validate_round(self, window: RoundWindow) -> None: ...

    def assert_registered(self) -> None: ...

    def load_local(self) -> PackagedArtifact | None: ...

    def package(self, round_index: int, source: str) -> PackagedArtifact: ...

    def upload(self, packaged: PackagedArtifact) -> None: ...

    def validate_commitment(self, packaged: PackagedArtifact) -> str: ...

    def verify_source(self, packaged: PackagedArtifact, *, full: bool) -> None: ...

    def verify_provenance(self, packaged: PackagedArtifact, block: int) -> None: ...

    def publish(self, packaged: PackagedArtifact) -> PublishReceipt: ...

    def verify_on_chain(self, packaged: PackagedArtifact) -> str: ...

    def close(self) -> None: ...


class MicrotensorBackend:
    """Thin adapter over the exact pinned upstream APIs.

    Imports are intentionally lazy: status/health and unit tests do not need Bittensor,
    W&B, boto3, or the upstream package installed.
    """

    def __init__(self, config: ControllerConfig, *, sleep: Any = time.sleep) -> None:
        self.config = config
        self._sleep = sleep
        self._wallet: Any = None
        self._client: Any = None
        self._hotkey = ""
        self._upstream_commit = ""

    @property
    def hotkey(self) -> str:
        if not self._hotkey:
            raise PreflightError("wallet has not passed preflight")
        return self._hotkey

    def preflight(self) -> PreflightSnapshot:
        self._verify_transaction_dependencies()
        commit, version = self._verify_upstream()
        self._validate_artifact_config()

        try:
            from microtensor.chain.client import SubtensorClient
            from microtensor.chain.config import ChainConfig
            from microtensor.chain.wallet import hotkey_address, load_wallet
            from microtensor.core.tracks import is_competable
        except ImportError as exc:
            raise PreflightError(f"pinned microtensor runtime is unavailable: {exc}") from exc

        if not is_competable(self.config.track, self.config.hardware_class):
            raise PreflightError(
                f"{self.config.track}/{self.config.hardware_class} is not live in the "
                "pinned upstream"
            )

        chain = ChainConfig(
            netuid=self.config.netuid,
            network=self.config.network,
            endpoint=self.config.endpoint,
            wallet_name=self.config.wallet_name,
            wallet_hotkey=self.config.wallet_hotkey,
            wallet_path=str(self.config.wallet_path),
        )
        try:
            self._wallet = load_wallet(chain)
            self._hotkey = hotkey_address(self._wallet)
            self._client = SubtensorClient(chain, self._wallet)
            snapshot = self._client.snapshot(refresh=True)
            actual_uid = snapshot.uid_of(self._hotkey)
            head = self._client.block()
        except Exception as exc:
            self.close()
            raise PreflightError(f"wallet/metagraph preflight failed: {exc}") from exc
        if actual_uid != self.config.expected_uid:
            self.close()
            raise PreflightError(
                f"hotkey {self._hotkey} maps to UID {actual_uid}, expected "
                f"{self.config.expected_uid}"
            )

        # Constructing the exact upstream config also validates the rendered source and
        # competition before any file is rewritten or extrinsic is signed.
        self._miner_config(self.config.source_for(0, self._hotkey))
        return PreflightSnapshot(
            hotkey=self._hotkey,
            uid=actual_uid,
            chain_head=head,
            upstream_version=version,
            upstream_commit=commit,
        )

    @staticmethod
    def _verify_transaction_dependencies() -> None:
        expected = {
            "bittensor": BITTENSOR_VERSION,
            "bittensor-wallet": BITTENSOR_WALLET_VERSION,
            "async-substrate-interface": SUBSTRATE_INTERFACE_VERSION,
        }
        for distribution, required in expected.items():
            try:
                observed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise PreflightError(
                    f"required transaction dependency is missing: {distribution}"
                ) from exc
            if observed != required:
                raise PreflightError(
                    f"{distribution} version is {observed}, expected exact audited pin {required}"
                )

    def _verify_upstream(self) -> tuple[str, str]:
        try:
            distribution = importlib.metadata.distribution("microtensor-subnet")
            version = distribution.version
        except importlib.metadata.PackageNotFoundError as exc:
            raise PreflightError("microtensor-subnet is not installed") from exc

        if self.config.uses_signed_v030:
            return self._verify_signed_v030(distribution, version)
        return self._verify_legacy_upstream(distribution, version)

    def _verify_legacy_upstream(self, distribution: Any, version: str) -> tuple[str, str]:
        if version != UPSTREAM_RELEASE:
            raise PreflightError(
                f"microtensor-subnet version is {version}, expected {UPSTREAM_RELEASE} from the pin"
            )

        direct_raw = distribution.read_text("direct_url.json")
        commit = ""
        if direct_raw:
            try:
                direct = json.loads(direct_raw)
                vcs = direct.get("vcs_info") or {}
                commit = str(vcs.get("commit_id", ""))
            except (json.JSONDecodeError, AttributeError, TypeError):
                commit = ""
        if commit != UPSTREAM_COMMIT:
            if not self.config.allow_unverified_upstream:
                found = commit or "no PEP 610 commit metadata"
                raise PreflightError(
                    f"upstream commit is {found}; expected {UPSTREAM_COMMIT}. "
                    "Install this project's pinned direct dependency."
                )
            log.warning("upstream commit metadata is unverified; dry-run is enforced")
            commit = "unverified"
        self._upstream_commit = commit
        return commit, version

    def _verify_signed_v030(self, distribution: Any, version: str) -> tuple[str, str]:
        if version != SIGNED_V030_RELEASE:
            raise PreflightError(
                f"microtensor-subnet version is {version}, expected signed release "
                f"{SIGNED_V030_RELEASE}"
            )
        direct_raw = distribution.read_text("direct_url.json")
        if not direct_raw:
            raise PreflightError("signed v0.3 install has no PEP 610 archive identity")
        try:
            direct = _strict_json_object(direct_raw)
            archive = direct["archive_info"]
            hashes = archive["hashes"]
            digest = hashes["sha256"]
            legacy_hash = archive["hash"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PreflightError(
                "signed v0.3 install has malformed PEP 610 archive identity"
            ) from exc
        if not isinstance(digest, str) or not isinstance(legacy_hash, str):
            raise PreflightError("signed v0.3 install has malformed PEP 610 archive hashes")
        if digest != SIGNED_V030_WHEEL_SHA256 or legacy_hash != f"sha256={digest}":
            raise PreflightError(
                f"signed v0.3 wheel hash is {digest or 'missing'}, expected "
                f"{SIGNED_V030_WHEEL_SHA256}"
            )

        try:
            constants = importlib.import_module("microtensor.core.constants")
            release = constants.RELEASE_VERSION
            mechanism = constants.MECHANISM_VERSION
            signing_key = constants.RELEASE_SIGNING_KEY
        except (AttributeError, ImportError) as exc:
            raise PreflightError("signed v0.3 runtime identity constants are unavailable") from exc
        if release != SIGNED_V030_RELEASE:
            raise PreflightError(
                f"runtime release identity is {release!r}, expected {SIGNED_V030_RELEASE!r}"
            )
        if mechanism != SIGNED_V030_MECHANISM_VERSION:
            raise PreflightError(
                f"runtime mechanism identity is {mechanism!r}, expected "
                f"{SIGNED_V030_MECHANISM_VERSION!r}"
            )
        if signing_key != SIGNED_V030_RELEASE_SIGNING_KEY:
            raise PreflightError("runtime release signing key differs from the audited v0.3 key")

        identity = f"sha256:{digest}"
        self._upstream_commit = identity
        return identity, version

    def _validate_artifact_config(self) -> None:
        config = self.config
        if config.source_template.startswith("https:") and not config.dry_run:
            self._read_github_token()

        if not config.wallet_path.is_dir():
            raise PreflightError(f"wallet path is not a directory: {config.wallet_path}")
        self._validate_wallet_permissions()
        if not config.artifact_dir.is_dir():
            raise PreflightError(f"artifact directory is missing: {config.artifact_dir}")
        entrypoint = config.artifact_dir / config.entrypoint
        if not entrypoint.is_file():
            raise PreflightError(f"artifact entrypoint is missing: {entrypoint}")
        if entrypoint.suffix.lower() != ".gguf":
            raise PreflightError("GGUF load spec requires a .gguf entrypoint")
        if (config.artifact_dir / "artifact.enc").exists():
            raise PreflightError(
                "artifact.enc exists; this controller categorically refuses sealing"
            )
        if not config.selfcheck_path.is_file():
            raise PreflightError(
                f"selfcheck is missing: {config.selfcheck_path}; run the pinned upstream "
                "selfcheck first"
            )
        self._declared()
        validate_binding(config)
        if not _PINNED_MODEL.fullmatch(config.base_model):
            raise PreflightError("MMC_BASE_MODEL must be <org>/<repo>@<7-40 hex commit>")

        existing = config.artifact_dir / "manifest.json"
        if existing.is_file():
            try:
                raw = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PreflightError(f"existing manifest is unreadable: {exc}") from exc
            if raw.get("sealed"):
                raise PreflightError(
                    "existing manifest is sealed; remove it only after manual review"
                )

    def _read_github_token(self) -> str:
        path = self.config.github_token_file
        if path is None:
            raise PreflightError("GitHub token file is required for live HTTPS upload")

        descriptor = -1
        try:
            before = path.lstat()
            self._validate_github_token_metadata(before)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            self._validate_github_token_metadata(opened)
            if before.st_dev != opened.st_dev or before.st_ino != opened.st_ino:
                raise PreflightError("GitHub token file changed while it was opened")
            fingerprint = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_uid,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            with os.fdopen(descriptor, "rb") as token_stream:
                descriptor = -1
                raw = token_stream.read(4097)
                after = os.fstat(token_stream.fileno())
            self._validate_github_token_metadata(after)
            if fingerprint != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise PreflightError("GitHub token file changed while it was read")
        except PreflightError:
            raise
        except OSError:
            raise PreflightError("GitHub token file is unavailable or unsafe") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        if len(raw) > 4096:
            raise PreflightError("GitHub token file exceeds the size limit")
        try:
            token = raw.decode("ascii")
        except UnicodeDecodeError:
            raise PreflightError("GitHub token file must contain ASCII") from None
        if token.endswith("\n"):
            token = token[:-1]
        if "\n" in token or "\r" in token or not token.isascii():
            raise PreflightError("GitHub token file must contain exactly one token")
        try:
            from .github_release import GitHubTransport, ReleasePublishError

            GitHubTransport(token)
        except (ImportError, ReleasePublishError):
            raise PreflightError("GitHub token file contains an invalid token") from None
        return token

    @staticmethod
    def _validate_github_token_metadata(metadata: os.stat_result) -> None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PreflightError("GitHub token file must be a regular non-symlink")
        if metadata.st_uid != os.geteuid():
            raise PreflightError("GitHub token file must be owned by the effective user")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise PreflightError("GitHub token file mode must be exactly 0600")
        if metadata.st_nlink != 1:
            raise PreflightError("GitHub token file must have exactly one hard link")

    def _validate_wallet_permissions(self) -> None:
        hotkey_file = (
            self.config.wallet_path
            / self.config.wallet_name
            / "hotkeys"
            / self.config.wallet_hotkey
        )
        try:
            metadata = hotkey_file.lstat()
        except OSError as exc:
            raise PreflightError(
                f"configured hotkey file is unavailable: {hotkey_file}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PreflightError(f"hotkey file must be a regular non-symlink: {hotkey_file}")
        if metadata.st_uid != os.geteuid():
            raise PreflightError(
                f"hotkey file must be owned by effective UID {os.geteuid()}: {hotkey_file}"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PreflightError(
                f"hotkey file must not grant group/world permissions: {hotkey_file}"
            )

    def _declared(self) -> Any:
        try:
            from microtensor.core.protocol import DeclaredEnvelope
        except ImportError as exc:
            raise PreflightError(f"upstream protocol is unavailable: {exc}") from exc
        try:
            raw = json.loads(self.config.selfcheck_path.read_text(encoding="utf-8"))
            values = {
                key: int(raw[key]) for key in ("size_bytes", "peak_rss_bytes", "p95_latency_ms")
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PreflightError(f"selfcheck is invalid: {exc}") from exc
        if any(value <= 0 for value in values.values()):
            raise PreflightError("selfcheck values must all be positive")
        return DeclaredEnvelope(**values)

    def _load_spec(self) -> Any:
        from microtensor.core.protocol import ArtifactFormat, LoadManifest

        return LoadManifest(
            format=ArtifactFormat(self.config.artifact_format),
            quantization=self.config.quantization,
            entrypoint=self.config.entrypoint,
            max_input={"tokens": self.config.max_input_tokens},
            preprocessing={"tokenizer": self.config.tokenizer},
            base_model=self.config.base_model,
        )

    def _chain_config(self) -> Any:
        from microtensor.chain.config import ChainConfig

        return ChainConfig(
            netuid=self.config.netuid,
            network=self.config.network,
            endpoint=self.config.endpoint,
            wallet_name=self.config.wallet_name,
            wallet_hotkey=self.config.wallet_hotkey,
            wallet_path=str(self.config.wallet_path),
        )

    def _miner_config(self, source: str) -> Any:
        from microtensor.miner.config import MinerConfig

        return MinerConfig(
            chain=self._chain_config(),
            home=self.config.upstream_home,
            artifact_dir=self.config.artifact_dir,
            track=self.config.track,
            hardware_class=self.config.hardware_class,
            source=source,
            entrypoint=self.config.entrypoint,
            artifact_format=self.config.artifact_format,
            quantization=self.config.quantization,
            max_input_tokens=self.config.max_input_tokens,
            tokenizer=self.config.tokenizer,
            base_model=self.config.base_model,
            round_blocks=self.config.round_blocks,
            genesis_block=self.config.genesis_block,
            allow_unsandboxed=False,
        )

    def block(self) -> int:
        if self._client is None:
            raise PreflightError("chain client has not passed preflight")
        return int(self._client.block())

    def chain_round(self, block: int) -> RoundWindow:
        from microtensor.chain.rounds import round_for_block

        found = round_for_block(
            block,
            length=self.config.round_blocks,
            genesis=self.config.genesis_block,
        )
        # Upstream Round.end_block is inclusive. The controller consistently stores an
        # exclusive end so coordinator and fallback freshness checks use one convention.
        return RoundWindow(
            index=found.index,
            start_block=found.start_block,
            close_block=found.close_block,
            end_block=found.end_block + 1,
            phase="submissions" if block < found.close_block else "evaluation",
            source="chain",
        )

    def verify_round_anchor(self, window: RoundWindow) -> None:
        if window.source != "coordinator":
            return
        if self._client is None:
            raise PreflightError("chain client has not passed preflight")
        try:
            from microtensor.chain.anchor import read_anchor
            from microtensor.core.constants import COORDINATOR_HOTKEY

            anchor = read_anchor(self._client, COORDINATOR_HOTKEY)
        except Exception as exc:
            raise VerificationError(
                f"could not independently read the coordinator anchor from chain: {exc}"
            ) from exc
        if anchor is None:
            raise VerificationError("the coordinator has no independently readable chain anchor")
        if anchor.round_index != window.index:
            raise VerificationError(
                f"chain anchor is for round {anchor.round_index}, not served round {window.index}"
            )
        if anchor.config_hash != window.config_hash:
            raise VerificationError(
                "served coordinator config hash does not match the independent chain anchor"
            )
        self._verify_v030_seed_hash(window)

    def _verify_v030_seed_hash(self, window: RoundWindow) -> None:
        if not self.config.uses_signed_v030:
            return
        if window.seed_block != window.close_block:
            raise VerificationError("served round seed block differs from its close block")
        if window.phase == "submissions":
            if window.block_hash:
                raise VerificationError("submission phase disclosed a future seed block hash")
            return
        if window.phase != "evaluation" or _HASH.fullmatch(window.block_hash) is None:
            raise VerificationError("evaluation phase has no canonical seed block hash")
        if self._client is None:
            raise PreflightError("chain client has not passed preflight")
        try:
            observed = str(self._client.block_hash(window.seed_block)).lower()
        except Exception as exc:
            raise VerificationError(
                f"could not independently read seed block {window.seed_block} from chain: {exc}"
            ) from exc
        if _HASH.fullmatch(observed) is None or observed != window.block_hash:
            raise VerificationError(
                "served evaluation seed hash does not match the independent chain block hash"
            )

    def validate_round(self, window: RoundWindow) -> None:
        declared = self._declared()
        checks = (
            ("size", int(declared.size_bytes), window.max_size_bytes),
            ("peak RSS", int(declared.peak_rss_bytes), window.max_rss_bytes),
            ("p95 latency", int(declared.p95_latency_ms), window.max_p95_ms),
        )
        for label, value, ceiling in checks:
            if ceiling <= 0:
                continue
            if value > ceiling:
                raise VerificationError(
                    f"selfcheck {label} {value} exceeds anchored round ceiling {ceiling}"
                )
        binding = validate_binding(self.config)
        if window.max_size_bytes > 0 and binding["artifact_total_bytes"] > window.max_size_bytes:
            raise VerificationError(
                "actual publishable artifact bytes exceed the anchored round size ceiling"
            )

    def assert_registered(self) -> None:
        if self._client is None:
            raise PreflightError("chain client has not passed preflight")
        try:
            uid = self._client.snapshot(refresh=True).uid_of(self.hotkey)
        except Exception as exc:
            raise VerificationError(f"registration refresh failed: {exc}") from exc
        if uid != self.config.expected_uid:
            raise VerificationError(
                f"hotkey now maps to UID {uid}, expected {self.config.expected_uid}"
            )

    @staticmethod
    def _wrapped(manifest: Any) -> PackagedArtifact:
        return PackagedArtifact(
            round_index=int(manifest.round_index),
            source=str(manifest.source),
            hotkey=str(manifest.hotkey),
            manifest_digest=str(manifest.digest()),
            artifact_digest=str(manifest.artifact_digest),
            file_count=len(manifest.files),
            total_bytes=int(manifest.total_bytes),
            sealed=manifest.sealed is not None,
            native=manifest,
        )

    def load_local(self) -> PackagedArtifact | None:
        from microtensor.miner.package import PackageError, load_packaged

        source = self.config.source_for(0, self.hotkey)
        try:
            # load_packaged only needs artifact_dir; its source is replaced below by the
            # manifest's own source for exact verification after a restart.
            manifest = load_packaged(self._miner_config(source))
        except PackageError:
            return None
        packaged = self._wrapped(manifest)
        if packaged.sealed:
            raise VerificationError("local manifest is sealed; refusing to manage it")
        return packaged

    def package(self, round_index: int, source: str) -> PackagedArtifact:
        from microtensor.miner.package import package

        if self._wallet is None:
            raise PreflightError("wallet has not passed preflight")
        manifest = package(
            self._miner_config(source),
            round_index,
            self._load_spec(),
            self._declared(),
            self._wallet,
            seal=False,
        )
        packaged = self._wrapped(manifest)
        if packaged.sealed:
            raise VerificationError("upstream unexpectedly produced a sealed manifest")
        if packaged.round_index != round_index or packaged.source != source:
            raise VerificationError("packaged manifest does not match the requested round/source")
        if packaged.hotkey != self.hotkey or not getattr(manifest, "signature", ""):
            raise VerificationError("packaged manifest is not signed by the configured hotkey")
        return packaged

    def _native(self, packaged: PackagedArtifact) -> Any:
        if packaged.native is not None:
            return packaged.native
        from microtensor.miner.package import load_packaged

        manifest = load_packaged(self._miner_config(packaged.source))
        if manifest.digest() != packaged.manifest_digest:
            raise VerificationError("local manifest changed since it was recorded")
        return manifest

    def validate_commitment(self, packaged: PackagedArtifact) -> str:
        from microtensor.core.constants import MAX_COMMITMENT_BYTES
        from microtensor.miner.publish import commitment_for

        native = self._native(packaged)
        if (
            packaged.sealed
            or getattr(native, "sealed", None) is not None
            or packaged.hotkey != self.hotkey
            or getattr(native, "hotkey", "") != self.hotkey
            or not getattr(native, "signature", "")
        ):
            raise VerificationError(
                "commitment requires an unsealed manifest signed by the configured hotkey"
            )
        try:
            payload = commitment_for(
                self._miner_config(packaged.source),
                native,
                packaged.round_index,
            ).encode()
        except Exception as exc:
            raise VerificationError(f"commitment cannot be encoded for chain: {exc}") from exc
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_COMMITMENT_BYTES:
            raise VerificationError("encoded commitment exceeds the 128-byte chain limit")
        return payload

    @staticmethod
    def _authorization_integer(value: Any, label: str) -> int:
        try:
            raw = value.rao
        except AttributeError:
            try:
                raw = value.value
            except AttributeError:
                raw = value
            except Exception as exc:
                raise AuthorizationRefused(f"{label} could not be proven") from exc
        except Exception as exc:
            raise AuthorizationRefused(f"{label} could not be proven") from exc
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise AuthorizationRefused(f"{label} was not an exact integer")
        converted = int(raw)
        if converted < 0:
            raise AuthorizationRefused(f"{label} was negative")
        return converted

    @classmethod
    def _authorized_chain_head(cls, substrate: Any) -> str:
        try:
            genesis = str(substrate.get_block_hash(0)).lower()
            head = str(substrate.get_chain_head()).lower()
            runtime = substrate.get_block_runtime_info(head)
            code_response = substrate.rpc_request(
                "state_getStorageHash", [_RUNTIME_CODE_STORAGE_KEY, head]
            )
        except Exception as exc:
            raise AuthorizationRefused(
                f"Finney runtime identity could not be proven before signing: {exc}"
            ) from exc
        if genesis != FINNEY_GENESIS_HASH:
            raise AuthorizationRefused(
                f"chain genesis is {genesis}, not the authorized Finney genesis"
            )
        if not _HASH.fullmatch(head):
            raise AuthorizationRefused("chain head response was not a canonical hash")
        if not isinstance(runtime, Mapping):
            raise AuthorizationRefused("runtime version response was incomplete")
        if "specVersion" not in runtime or "transactionVersion" not in runtime:
            raise AuthorizationRefused("runtime version response omitted pinned fields")
        spec = cls._authorization_integer(runtime["specVersion"], "runtime spec version")
        transaction = cls._authorization_integer(
            runtime["transactionVersion"], "runtime transaction version"
        )
        if spec != FINNEY_RUNTIME_SPEC_VERSION:
            raise AuthorizationRefused(
                f"runtime spec is {spec}, expected pinned {FINNEY_RUNTIME_SPEC_VERSION}"
            )
        if transaction != FINNEY_TRANSACTION_VERSION:
            raise AuthorizationRefused(
                "runtime transaction version is "
                f"{transaction}, expected pinned {FINNEY_TRANSACTION_VERSION}"
            )
        if not isinstance(code_response, Mapping) or not isinstance(
            code_response.get("result"), str
        ):
            raise AuthorizationRefused("runtime code hash response was incomplete")
        code_hash = str(code_response["result"]).lower()
        if code_hash != FINNEY_RUNTIME_CODE_HASH:
            raise AuthorizationRefused(
                f"runtime code hash is {code_hash}, not the audited Finney runtime"
            )
        return head

    def _assert_live_transaction_policy(self) -> None:
        config = self.config
        if config.dry_run:
            raise AuthorizationRefused("chain submission is forbidden while MMC_DRY_RUN=true")
        if config.transaction_authorization != TRANSACTION_AUTHORIZATION:
            raise AuthorizationRefused("the exact zero-cost transaction authorization is absent")
        if config.netuid != 92 or config.expected_uid != 32:
            raise AuthorizationRefused("authorized netuid 92 / UID 32 identity changed")
        if (config.wallet_name, config.wallet_hotkey) != ("you-cold", "you-hot1"):
            raise AuthorizationRefused("authorized you-cold / you-hot1 wallet selection changed")
        if self.hotkey != AUTHORIZED_HOTKEY_SS58:
            raise AuthorizationRefused("loaded hotkey is not the authorized UID-32 you-hot1 key")
        if config.network != "finney" or config.endpoint:
            raise AuthorizationRefused("live commitment requires finney with no custom endpoint")
        try:
            self._verify_transaction_dependencies()
            source = config.source_for(0, self.hotkey)
            coordinates = config.github_release_coordinates(source)
        except Exception as exc:
            raise AuthorizationRefused(
                f"live dependency or artifact-source policy could not be proven: {exc}"
            ) from exc
        if (
            coordinates is None
            or f"{coordinates[0]}/{coordinates[1]}" != AUTHORIZED_GITHUB_REPOSITORY
        ):
            raise AuthorizationRefused(
                "artifact source is not the authorized vandungtech/mt92 repo"
            )

    def _assert_authorized_registration(self) -> None:
        self._assert_live_transaction_policy()
        if self._client is None:
            raise AuthorizationRefused("chain client has not passed preflight")
        client_netuid = self._authorization_integer(
            getattr(self._client, "netuid", None), "chain client netuid"
        )
        if self.config.netuid != 92 or client_netuid != 92:
            raise AuthorizationRefused("authorized netuid changed from 92")
        try:
            self.assert_registered()
        except VerificationError as exc:
            raise AuthorizationRefused(
                f"you-hot1 registration at UID 32 could not be proven: {exc}"
            ) from exc

    @classmethod
    def _inspect_signed_extrinsic(
        cls,
        extrinsic: Any,
        signer: Any,
        expected_call: Mapping[str, Any],
        nonce: int,
        era: Mapping[str, int],
    ) -> str:
        value = getattr(extrinsic, "value", None)
        expected_keys = {
            "account_id",
            "address",
            "asset_id",
            "call",
            "call_args",
            "call_function",
            "call_module",
            "era",
            "mode",
            "nonce",
            "signature",
            "signature_version",
            "tip",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise AuthorizationRefused("signed extrinsic fields differ from the audited shape")
        try:
            expected_account = "0x" + signer.public_key.hex()
        except Exception as exc:
            raise AuthorizationRefused("signing account identity could not be proven") from exc
        if value["account_id"] != expected_account or value["address"] != expected_account:
            raise AuthorizationRefused("signed extrinsic uses a different account")
        if (
            value["call_module"] != expected_call["call_module"]
            or value["call_function"] != expected_call["call_function"]
            or value["call_args"] != expected_call["call_args"]
            or value["call"] != expected_call
        ):
            raise AuthorizationRefused("signed extrinsic contains a different call")
        if cls._authorization_integer(value["nonce"], "signed nonce") != nonce:
            raise AuthorizationRefused("signed extrinsic nonce differs from the fee estimate")
        signed_era = value["era"]
        if not isinstance(signed_era, Mapping) or dict(signed_era) != dict(era):
            raise AuthorizationRefused("signed extrinsic era differs from the fee estimate")
        if cls._authorization_integer(value["tip"], "signed tip") != 0:
            raise AuthorizationRefused("signed extrinsic contains a nonzero tip")
        asset = value["asset_id"]
        if (
            not isinstance(asset, Mapping)
            or set(asset) != {"tip", "asset_id"}
            or cls._authorization_integer(asset["tip"], "signed asset tip") != 0
            or asset["asset_id"] is not None
        ):
            raise AuthorizationRefused("signed extrinsic contains an unauthorized fee asset")
        if value["mode"] != "Disabled":
            raise AuthorizationRefused("signed extrinsic enables an unauthorized mode")
        if (
            cls._authorization_integer(value["signature_version"], "signed signature version")
            != _AUTHORIZED_SIGNATURE_VERSION
        ):
            raise AuthorizationRefused("signed extrinsic uses an unexpected signature version")
        signature = value["signature"]
        if not isinstance(signature, str) or not re.fullmatch(r"0x[0-9a-f]{128}", signature):
            raise AuthorizationRefused("signed extrinsic signature could not be proven")
        data = getattr(extrinsic, "data", None)
        if data is None or not str(data).startswith("0x") or len(str(data)) <= 2:
            raise AuthorizationRefused("signed extrinsic has no encoded data")
        try:
            raw_hash = str(extrinsic.extrinsic_hash.hex()).lower()
        except Exception as exc:
            raise AuthorizationRefused("signed extrinsic hash could not be proven") from exc
        extrinsic_hash = raw_hash if raw_hash.startswith("0x") else f"0x{raw_hash}"
        if not _HASH.fullmatch(extrinsic_hash):
            raise AuthorizationRefused("signed extrinsic hash was not canonical")
        return extrinsic_hash

    def _submit_authorized_commitment(self, payload: str) -> None:
        """Submit exactly one direct zero-cost commitment or stop permanently.

        This deliberately does not call the SDK's higher-level metadata helper: the
        exact composed call is inspected, estimated, and then passed unchanged to the
        direct signer. There is no internal retry; the controller reconciles chain state
        before a later attempt if an inclusion response is ambiguous.
        """
        self._assert_live_transaction_policy()
        if self._client is None or self._wallet is None:
            raise AuthorizationRefused("wallet and chain client have not passed preflight")
        self._assert_authorized_registration()

        try:
            encoded = payload.encode("ascii")
        except UnicodeEncodeError:
            raise AuthorizationRefused("commitment payload is not ASCII") from None
        if not 1 <= len(encoded) <= 128:
            raise AuthorizationRefused("commitment payload is outside the Raw1-Raw128 allowlist")

        subtensor = self._client.subtensor
        substrate = getattr(subtensor, "substrate", None)
        if substrate is None:
            raise AuthorizationRefused("direct Substrate transaction interface is unavailable")
        signer = getattr(self._wallet, "hotkey", None)
        if (
            signer is None
            or str(getattr(signer, "ss58_address", "")) != self.hotkey
            or self.hotkey != AUTHORIZED_HOTKEY_SS58
        ):
            raise AuthorizationRefused("the direct signer is not the registered you-hot1 hotkey")

        try:
            head_before = self._authorized_chain_head(substrate)
            info = {"fields": [[{f"Raw{len(encoded)}": encoded}]]}
            call_args: dict[str, Any] = {"netuid": 92, "info": info}
            expected_call: dict[str, Any] = {
                "call_module": _AUTHORIZED_CALL_MODULE,
                "call_function": _AUTHORIZED_CALL_FUNCTION,
                "call_args": call_args,
            }
            call = substrate.compose_call(
                call_module=_AUTHORIZED_CALL_MODULE,
                call_function=_AUTHORIZED_CALL_FUNCTION,
                call_params=call_args,
                block_hash=head_before,
            )
        except AuthorizationRefused:
            raise
        except Exception as exc:
            raise AuthorizationRefused(
                f"authorized commitment call could not be composed: {exc}"
            ) from exc

        if getattr(call, "value", None) != expected_call:
            raise AuthorizationRefused(
                "composed transaction differs from Commitments.set_commitment(netuid=92)"
            )

        deposits: dict[str, int] = {}
        try:
            for name in ("InitialDeposit", "FieldDeposit"):
                deposits[name] = self._authorization_integer(
                    substrate.get_constant("Commitments", name, block_hash=head_before),
                    f"Commitments.{name}",
                )
        except AuthorizationRefused:
            raise
        except Exception as exc:
            raise AuthorizationRefused(
                f"commitment deposit could not be proven before signing: {exc}"
            ) from exc
        required_deposit = deposits["InitialDeposit"] + deposits["FieldDeposit"]
        if required_deposit != _AUTHORIZED_MAX_DEPOSIT_RAO:
            raise AuthorizationRefused(
                f"required commitment deposit is {required_deposit} rao, authorized maximum is 0"
            )

        try:
            nonce = self._authorization_integer(
                substrate.get_account_next_index(signer.ss58_address),
                "hotkey account nonce",
            )
            era: dict[str, int] = {"period": _AUTHORIZED_PERIOD_BLOCKS}
            payment = substrate.get_payment_info(
                call=call,
                keypair=signer,
                era=era,
                nonce=nonce,
                tip=0,
                tip_asset_id=None,
            )
        except AuthorizationRefused:
            raise
        except Exception as exc:
            raise AuthorizationRefused(
                f"transaction fee could not be estimated before signing: {exc}"
            ) from exc
        if not isinstance(payment, Mapping) or "partial_fee" not in payment:
            raise AuthorizationRefused("transaction fee estimate response was incomplete")
        estimated_fee = self._authorization_integer(
            payment["partial_fee"], "estimated transaction fee"
        )
        if estimated_fee != _AUTHORIZED_MAX_FEE_RAO:
            raise AuthorizationRefused(
                f"estimated transaction fee is {estimated_fee} rao, authorized maximum is 0"
            )
        if not isinstance(era, dict) or set(era) != {"period", "current"}:
            raise AuthorizationRefused("fee estimate did not use an explicit mortal era")
        if self._authorization_integer(era["period"], "mortal era period") != 128:
            raise AuthorizationRefused("fee estimate changed the authorized 128-block era")
        self._authorization_integer(era["current"], "mortal era current block")
        if getattr(call, "value", None) != expected_call:
            raise AuthorizationRefused("fee estimation mutated the authorized commitment call")

        # Refresh all mutable authorization inputs immediately before the only
        # private-key operation, including the nonce used by the fee estimate.
        self._assert_authorized_registration()
        self._authorized_chain_head(substrate)
        try:
            nonce_before_signing = self._authorization_integer(
                substrate.get_account_next_index(signer.ss58_address),
                "pre-sign hotkey account nonce",
            )
        except AuthorizationRefused:
            raise
        except Exception as exc:
            raise AuthorizationRefused(
                f"hotkey nonce could not be refreshed before signing: {exc}"
            ) from exc
        if nonce_before_signing != nonce:
            raise AuthorizationRefused("hotkey nonce changed after the fee estimate")

        try:
            refreshed_era = dict(era)
            refreshed_payment = substrate.get_payment_info(
                call=call,
                keypair=signer,
                era=refreshed_era,
                nonce=nonce,
                tip=0,
                tip_asset_id=None,
            )
        except Exception as exc:
            raise AuthorizationRefused(
                f"transaction fee could not be refreshed before signing: {exc}"
            ) from exc
        if not isinstance(refreshed_payment, Mapping) or "partial_fee" not in refreshed_payment:
            raise AuthorizationRefused("refreshed transaction fee response was incomplete")
        refreshed_fee = self._authorization_integer(
            refreshed_payment["partial_fee"], "refreshed transaction fee"
        )
        if refreshed_fee != _AUTHORIZED_MAX_FEE_RAO:
            raise AuthorizationRefused(
                f"refreshed transaction fee is {refreshed_fee} rao, authorized maximum is 0"
            )
        if refreshed_era != dict(era):
            raise AuthorizationRefused("refreshed fee estimate changed the signed mortal era")

        signed_era = dict(era)
        try:
            extrinsic = substrate.create_signed_extrinsic(
                call=call,
                keypair=signer,
                era=signed_era,
                nonce=nonce,
                tip=0,
                tip_asset_id=None,
            )
        except Exception as exc:
            raise AuthorizationRefused(
                f"signed commitment could not be created; no submission attempted: {exc}"
            ) from exc
        extrinsic_hash = self._inspect_signed_extrinsic(
            extrinsic, signer, expected_call, nonce, era
        )

        # Signing does not authorize submission under stale state. Recheck policy,
        # registration, chain identity, and nonce immediately before broadcasting.
        self._assert_authorized_registration()
        self._authorized_chain_head(substrate)
        try:
            nonce_before_submit = self._authorization_integer(
                substrate.get_account_next_index(signer.ss58_address),
                "pre-submit hotkey account nonce",
            )
        except AuthorizationRefused:
            raise
        except Exception as exc:
            raise AuthorizationRefused(
                f"hotkey nonce could not be refreshed before submission: {exc}"
            ) from exc
        if nonce_before_submit != nonce:
            raise AuthorizationRefused("hotkey nonce changed after signing")

        try:
            final_era = dict(era)
            final_payment = substrate.get_payment_info(
                call=call,
                keypair=signer,
                era=final_era,
                nonce=nonce,
                tip=0,
                tip_asset_id=None,
            )
        except Exception as exc:
            raise AuthorizationRefused(
                f"transaction fee could not be proven immediately before submit: {exc}"
            ) from exc
        if not isinstance(final_payment, Mapping) or "partial_fee" not in final_payment:
            raise AuthorizationRefused("final transaction fee response was incomplete")
        final_fee = self._authorization_integer(
            final_payment["partial_fee"], "final transaction fee"
        )
        if final_fee != _AUTHORIZED_MAX_FEE_RAO:
            raise AuthorizationRefused(
                f"final transaction fee is {final_fee} rao, authorized maximum is 0"
            )
        if final_era != dict(era) or getattr(call, "value", None) != expected_call:
            raise AuthorizationRefused("final fee estimate changed the signed transaction envelope")
        self._inspect_signed_extrinsic(extrinsic, signer, expected_call, nonce, era)

        try:
            receipt = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
        except Exception as exc:
            raise AuthorizationRefused(
                "commitment submission outcome is ambiguous; operator reconciliation "
                "is required before any retry"
            ) from exc

        try:
            receipt_hash = str(getattr(receipt, "extrinsic_hash", "")).lower()
            block_hash = str(getattr(receipt, "block_hash", "")).lower()
            if getattr(receipt, "finalized", None) is not True:
                raise AuthorizationRefused("commitment receipt did not prove finalization")
            if receipt_hash != extrinsic_hash or not _HASH.fullmatch(receipt_hash):
                raise AuthorizationRefused("receipt does not identify the signed extrinsic")
            if not _HASH.fullmatch(block_hash):
                raise AuthorizationRefused("commitment receipt omitted a canonical block hash")
            if getattr(receipt, "is_success", None) is not True:
                raise AuthorizationRefused("finalized commitment did not prove success")
            charged_fee = self._authorization_integer(
                getattr(receipt, "total_fee_amount", None), "receipt total fee"
            )
        except AuthorizationRefused:
            raise
        except Exception as exc:
            raise AuthorizationRefused(
                "finalized commitment receipt could not be proven; operator reconciliation "
                "is required before any retry"
            ) from exc
        if charged_fee != _AUTHORIZED_MAX_FEE_RAO:
            raise AuthorizationRefused(
                f"commitment receipt reports a nonzero fee ({charged_fee} rao)"
            )

    def upload(self, packaged: PackagedArtifact) -> None:
        if self.config.dry_run:
            raise VerificationError("upload is forbidden while MMC_DRY_RUN=true")

        scheme = packaged.source.partition(":")[0]
        if scheme != "https":
            raise VerificationError(
                "live upload requires an immutable GitHub HTTPS release; S3/R2 are refused"
            )
        self._upload_github(packaged)

    def _upload_github(self, packaged: PackagedArtifact) -> None:
        try:
            from .github_release import GitHubReleasePublisher

            coordinates = self.config.github_release_coordinates(packaged.source)
            if coordinates is None:
                raise VerificationError("GitHub release coordinates are invalid")
            owner, repo, tag = coordinates
            token = self._read_github_token()
            names = _publishable_files(self._native(packaged))
            if not names or len(names) != len(set(names)):
                raise VerificationError("publishable artifact file set is invalid")
            assets = {name: self.config.artifact_dir / name for name in names}
            result = GitHubReleasePublisher(
                owner=owner,
                repo=repo,
                tag=tag,
                token=token,
            ).publish(assets)
            published_names = tuple(asset.name for asset in result.assets)
            if (
                result.source != packaged.source
                or len(published_names) != len(names)
                or set(published_names) != set(names)
            ):
                raise VerificationError("GitHub release result does not match the package")
        except Exception:
            raise VerificationError("GitHub immutable release upload failed") from None

    def verify_source(self, packaged: PackagedArtifact, *, full: bool) -> None:
        from microtensor.chain.wallet import verify_payload
        from microtensor.miner.publish import commitment_for
        from microtensor.registry.cache import ArtifactCache
        from microtensor.registry.fetch import fetch_manifest, materialise

        local = self._native(packaged)
        commitment = commitment_for(
            self._miner_config(packaged.source), local, packaged.round_index
        )
        verify_root = self.config.state_dir / "verify"
        verify_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with tempfile.TemporaryDirectory(prefix="round-verify-", dir=verify_root) as temporary:
                root = Path(temporary)
                remote = fetch_manifest(
                    commitment,
                    workdir=root / "manifests",
                    attempts=self.config.verify_attempts,
                )
                if remote.digest() != packaged.manifest_digest:
                    raise VerificationError(
                        "remote manifest digest differs from the packaged manifest"
                    )
                if remote.hotkey != self.hotkey or not remote.signature:
                    raise VerificationError(
                        "remote manifest is unsigned or names a different hotkey"
                    )
                if not verify_payload(remote.hotkey, remote.body(), remote.signature):
                    raise VerificationError("remote manifest signature is invalid")
                if remote.sealed is not None:
                    raise VerificationError("remote manifest is sealed")
                if full:
                    cap = max(remote.total_bytes * 2, remote.total_bytes + 1)
                    cache = ArtifactCache(root / "cache", cap_bytes=cap)
                    materialise(
                        remote,
                        cache,
                        workdir=root / "work",
                        attempts=self.config.verify_attempts,
                    )
        except VerificationError:
            raise
        except Exception as exc:
            kind = "full artifact" if full else "manifest"
            raise VerificationError(f"remote {kind} verification failed: {exc}") from exc

    def verify_provenance(self, packaged: PackagedArtifact, block: int) -> None:
        try:
            from microtensor.provenance.record import best_verdict
            from microtensor.provenance.wandb_store import WandbStore

            runs = WandbStore().candidates(self.hotkey)
            verdict = best_verdict(
                runs,
                hotkey=self.hotkey,
                artifact_digest=packaged.artifact_digest,
                track=self.config.track,
                hardware_class=self.config.hardware_class,
                commit_block=block,
                allowed_base_models=frozenset({self.config.base_model}),
            )
        except Exception as exc:
            raise VerificationError(f"provenance verification failed: {exc}") from exc
        if not verdict.admissible:
            raise VerificationError(f"provenance rejected: {verdict.reason}")

    def publish(self, packaged: PackagedArtifact) -> PublishReceipt:
        self._assert_live_transaction_policy()
        if packaged.hotkey != self.hotkey:
            raise AuthorizationRefused("packaged artifact is signed by a different hotkey")
        try:
            expected_source = self.config.source_for(packaged.round_index, self.hotkey)
            coordinates = self.config.github_release_coordinates(packaged.source)
        except Exception as exc:
            raise AuthorizationRefused(
                f"packaged artifact source could not be proven: {exc}"
            ) from exc
        if packaged.source != expected_source:
            raise AuthorizationRefused(
                "packaged artifact source differs from the authorized round release"
            )
        if (
            coordinates is None
            or f"{coordinates[0]}/{coordinates[1]}" != AUTHORIZED_GITHUB_REPOSITORY
        ):
            raise AuthorizationRefused("packaged artifact is not in vandungtech/mt92")
        payload = self.validate_commitment(packaged)
        self._submit_authorized_commitment(payload)
        return PublishReceipt(round_index=packaged.round_index, payload=payload)

    def verify_on_chain(self, packaged: PackagedArtifact) -> str:
        if self._client is None:
            raise PreflightError("chain client has not passed preflight")
        expected = self.validate_commitment(packaged)
        observed = ""
        for attempt in range(1, self.config.verify_attempts + 1):
            found = self._client.commitments([self.hotkey])
            observed = str(found.get(self.hotkey, ""))
            if observed == expected:
                return expected
            if attempt < self.config.verify_attempts:
                self._sleep(12)
        raise VerificationError(
            "on-chain readback did not exactly match the signed round/source/digest commitment"
        )

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
