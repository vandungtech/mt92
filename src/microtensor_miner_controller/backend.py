from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from .binding import validate_binding
from .config import ControllerConfig, UPSTREAM_COMMIT, UPSTREAM_RELEASE
from .errors import PreflightError, VerificationError
from .models import (
    PackagedArtifact,
    PreflightSnapshot,
    PublishReceipt,
    RoundWindow,
)

log = logging.getLogger(__name__)

_PINNED_MODEL = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{7,40}$")


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
                f"{self.config.track}/{self.config.hardware_class} is not live in the pinned upstream"
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
                f"hotkey {self._hotkey} maps to UID {actual_uid}, expected {self.config.expected_uid}"
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

    def _verify_upstream(self) -> tuple[str, str]:
        try:
            distribution = importlib.metadata.distribution("microtensor-subnet")
            version = distribution.version
        except importlib.metadata.PackageNotFoundError as exc:
            raise PreflightError("microtensor-subnet is not installed") from exc
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

    def _validate_artifact_config(self) -> None:
        config = self.config
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
            raise PreflightError("artifact.enc exists; this controller categorically refuses sealing")
        if not config.selfcheck_path.is_file():
            raise PreflightError(
                f"selfcheck is missing: {config.selfcheck_path}; run the pinned upstream selfcheck first"
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
                raise PreflightError("existing manifest is sealed; remove it only after manual review")
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
            raise PreflightError(f"configured hotkey file is unavailable: {hotkey_file}: {exc}") from exc
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
                key: int(raw[key])
                for key in ("size_bytes", "peak_rss_bytes", "p95_latency_ms")
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

        try:
            payload = commitment_for(
                self._miner_config(packaged.source),
                self._native(packaged),
                packaged.round_index,
            ).encode()
        except Exception as exc:
            raise VerificationError(f"commitment cannot be encoded for chain: {exc}") from exc
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_COMMITMENT_BYTES:
            raise VerificationError("encoded commitment exceeds the 128-byte chain limit")
        return payload

    def upload(self, packaged: PackagedArtifact) -> None:
        from microtensor.miner.package import publishable_files
        from microtensor.miner.upload import plan_upload, upload

        manifest = self._native(packaged)
        scheme, _, locator = packaged.source.partition(":")
        plan = plan_upload(
            self.config.artifact_dir,
            scheme,
            locator,
            publishable_files(manifest),
        )
        upload(plan, self.config.artifact_dir)

    def verify_source(self, packaged: PackagedArtifact, *, full: bool) -> None:
        from microtensor.chain.wallet import verify_payload
        from microtensor.miner.publish import commitment_for
        from microtensor.registry.cache import ArtifactCache
        from microtensor.registry.fetch import fetch_manifest, materialise

        local = self._native(packaged)
        commitment = commitment_for(self._miner_config(packaged.source), local, packaged.round_index)
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
                    raise VerificationError("remote manifest digest differs from the packaged manifest")
                if remote.hotkey != self.hotkey or not remote.signature:
                    raise VerificationError("remote manifest is unsigned or names a different hotkey")
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
        from microtensor.miner.publish import publish

        if self._client is None:
            raise PreflightError("chain client has not passed preflight")
        result = publish(
            self._miner_config(packaged.source),
            self._client,
            packaged.round_index,
            self._native(packaged),
        )
        return PublishReceipt(round_index=result.round_index, payload=result.payload)

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
            try:
                client.close()
            except Exception:
                pass
