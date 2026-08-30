from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from .backend import Backend
from .config import ControllerConfig
from .coordinator import fetch_current_round, resolve_round
from .errors import ControllerError, RoundRefused, VerificationError
from .models import PackagedArtifact, PreflightSnapshot, RoundWindow, VerificationProofs
from .redaction import redact_text
from .state import StateStore, utc_timestamp

log = logging.getLogger(__name__)


class Controller:
    def __init__(
        self,
        config: ControllerConfig,
        backend: Backend,
        state: StateStore,
        *,
        stop_event: threading.Event | None = None,
        clock: Callable[[], float] = time.time,
        round_fetcher: Callable[[str, int], dict[str, Any]] = fetch_current_round,
    ) -> None:
        self.config = config
        self.backend = backend
        self.state = state
        self.stop_event = stop_event or threading.Event()
        self.clock = clock
        self.round_fetcher = round_fetcher
        self.preflight_snapshot: PreflightSnapshot | None = None

    def preflight(self) -> PreflightSnapshot:
        self.state.write(
            "preflight",
            ok=False,
            message="validating pinned runtime, artifact, wallet, UID, and live competition",
            details=self._static_details(),
        )
        snapshot = self.backend.preflight()
        self.preflight_snapshot = snapshot
        self.state.write(
            "ready",
            ok=False,
            message="preflight passed; no round has been fully verified yet",
            details={**self._static_details(), "preflight": snapshot.to_dict()},
        )
        log.info(
            "preflight passed for UID %d on netuid %d (%s/%s)",
            snapshot.uid,
            self.config.netuid,
            self.config.track,
            self.config.hardware_class,
        )
        return snapshot

    def preflight_only(self) -> int:
        with self.state.lock():
            try:
                self.preflight()
                return 0
            except ControllerError as exc:
                self._failure("preflight_refused", exc)
                return 2
            except Exception as exc:
                self._failure("preflight_error", exc)
                return 1
            finally:
                self.backend.close()

    def run(self, *, once: bool = False) -> int:
        with self.state.lock():
            try:
                snapshot = self.preflight()
                while not self.stop_event.is_set():
                    try:
                        outcome = self.cycle(snapshot)
                    except RoundRefused as exc:
                        self._failure("round_refused", exc)
                        outcome = "refused"
                    except ControllerError as exc:
                        self._failure("error", exc)
                        outcome = "error"
                    except Exception as exc:
                        self._failure("error", exc)
                        outcome = "error"

                    if once:
                        return 0 if outcome in {"verified", "dry_run"} else 2
                    delay = (
                        self.config.poll_seconds
                        if outcome in {"verified", "waiting", "dry_run"}
                        else self.config.retry_seconds
                    )
                    self.stop_event.wait(delay)
                self.state.write(
                    "stopped",
                    ok=False,
                    message="controller stopped cleanly",
                    details=self._static_details(),
                )
                return 0
            except ControllerError as exc:
                self._failure("preflight_refused", exc)
                return 2
            except Exception as exc:
                self._failure("fatal", exc)
                return 1
            finally:
                self.backend.close()

    def cycle(self, snapshot: PreflightSnapshot) -> str:
        # Registration can change while this long-running process remains alive. Keep
        # the read-only monitor honest instead of relying only on startup preflight.
        self.backend.assert_registered()
        head = self.backend.block()
        window = self._resolve_round(head)
        source = self.config.source_for(window.index, snapshot.hotkey)
        common = self._round_details(snapshot, window, head, source)

        previous = self.state.read_status()

        if self.config.dry_run:
            if not window.accepts_at(head, self.config.deadline_margin_blocks):
                self.state.write(
                    "waiting",
                    ok=False,
                    message=f"round {window.index} is outside the safe submission window",
                    details=common,
                )
                return "waiting"
            self.state.write(
                "dry_run",
                ok=False,
                message=(
                    f"dry-run plan is valid for round {window.index}; no package, upload, "
                    "signature, or chain write was attempted"
                ),
                details={**common, "planned_actions": self._planned_actions()},
            )
            log.info("dry-run planned round %d; no external write attempted", window.index)
            return "dry_run"

        local = self.backend.load_local()
        matching = self._local_matches(local, window.index, source, snapshot.hotkey)
        proofs_valid = (
            matching
            and self._proofs_valid_for(
                previous,
                window.index,
                source=source,
                hotkey=snapshot.hotkey,
                packaged=local,
            )
        )

        if not window.accepts_at(head, self.config.deadline_margin_blocks):
            if matching:
                assert local is not None
                if not proofs_valid or self._verification_due(previous):
                    return self._reverify(snapshot, window, head, source, local)
            message = (
                f"round {window.index} is outside the safe submission window; "
                + ("matching commitment remains verified" if proofs_valid else "no verified commitment")
            )
            self.state.write(
                "waiting",
                ok=proofs_valid,
                message=message,
                details=common,
            )
            return "waiting"

        if matching:
            assert local is not None
            if (
                proofs_valid
                and previous.get("ok") is True
                and not self._verification_due(previous)
            ):
                self.state.heartbeat(now=self.clock())
                return "verified"
            if proofs_valid:
                return self._reverify(snapshot, window, head, source, local)

        return self._submit(snapshot, window, head, source, local)

    def _submit(
        self,
        snapshot: PreflightSnapshot,
        window: RoundWindow,
        head: int,
        source: str,
        local: PackagedArtifact | None,
    ) -> str:
        self.backend.assert_registered()
        common = self._round_details(snapshot, window, head, source)
        reusable = self._local_matches(local, window.index, source, snapshot.hotkey)

        if reusable:
            assert local is not None
            packaged = local
            self.state.write(
                "recovering",
                ok=False,
                message="reusing a current signed manifest after restart/interruption",
                details={**common, "artifact": packaged.public_dict()},
            )
        else:
            self._assert_deadline(window, head, "package")
            self.state.write(
                "packaging",
                ok=False,
                message=f"signing an unsealed manifest for round {window.index}",
                details=common,
            )
            packaged = self.backend.package(window.index, source)
            if packaged.sealed:
                raise VerificationError("backend returned a sealed artifact")

        artifact_details = {**common, "artifact": packaged.public_dict()}
        self.state.write(
            "validating_commitment",
            ok=False,
            message="validating exact chain encoding before any external upload",
            details=artifact_details,
        )
        encoded = self.backend.validate_commitment(packaged)
        artifact_details["commitment_fingerprint"] = self._fingerprint(encoded)

        provenance_block = self.backend.block()
        self.state.write(
            "verifying_provenance",
            ok=False,
            message="checking public training candidates against the artifact digest",
            details={**artifact_details, "chain_head": provenance_block},
        )
        self.backend.verify_provenance(packaged, provenance_block)

        remote_ready = False
        if reusable:
            self.state.write(
                "verifying_source",
                ok=False,
                message="checking whether the complete exact remote artifact already exists",
                details=artifact_details,
            )
            try:
                self.backend.verify_source(packaged, full=True)
                remote_ready = True
            except VerificationError:
                log.warning("exact remote recovery probe failed; a controlled re-upload is required")

        if not remote_ready:
            upload_head, refreshed = self._refresh_same_round(window)
            self._assert_deadline(refreshed, upload_head, "upload")
            self.state.write(
                "uploading",
                ok=False,
                message="uploading the round-specific artifact tree and manifest",
                details={**artifact_details, "chain_head": upload_head},
            )
            self.backend.upload(packaged)
            self.state.write(
                "verifying_source",
                ok=False,
                message="fetching and hashing the complete remote artifact",
                details=artifact_details,
            )
            self.backend.verify_source(packaged, full=True)

        # A crash can happen after the extrinsic lands but before status.json is replaced.
        # Reconcile first so restart never blindly duplicates an already exact commitment.
        try:
            payload = self.backend.verify_on_chain(packaged)
        except VerificationError:
            refreshed_head, refreshed = self._refresh_same_round(window)
            self._assert_deadline(refreshed, refreshed_head, "publish")
            self.state.write(
                "publishing",
                ok=False,
                message="submitting the signed commitment on chain",
                details={**artifact_details, "chain_head": refreshed_head},
            )
            receipt = self.backend.publish(packaged)
            if receipt.round_index != packaged.round_index:
                raise VerificationError("publish receipt names a different round")
            self.state.write(
                "verifying_on_chain",
                ok=False,
                message="reading the exact commitment back from chain",
                details=artifact_details,
            )
            payload = self.backend.verify_on_chain(packaged)

        return self._verified(
            window,
            packaged,
            payload,
            common,
            source_full=True,
            message="source, provenance, and exact on-chain commitment verified",
        )
    def _reverify(
        self,
        snapshot: PreflightSnapshot,
        window: RoundWindow,
        head: int,
        source: str,
        packaged: PackagedArtifact,
    ) -> str:
        self.backend.assert_registered()
        common = self._round_details(snapshot, window, head, source)
        artifact_details = {**common, "artifact": packaged.public_dict()}
        self.state.write(
            "reverifying",
            ok=False,
            message="refreshing remote manifest, provenance, and on-chain proof",
            details=artifact_details,
        )
        self.backend.verify_source(packaged, full=True)
        self.backend.verify_provenance(packaged, window.close_block)
        payload = self.backend.verify_on_chain(packaged)
        return self._verified(
            window,
            packaged,
            payload,
            common,
            source_full=True,
            message="periodic source, provenance, and on-chain verification passed",
        )

    def _verified(
        self,
        window: RoundWindow,
        packaged: PackagedArtifact,
        payload: str,
        common: Mapping[str, Any],
        *,
        source_full: bool,
        message: str,
    ) -> str:
        proofs = VerificationProofs(
            source=True,
            provenance=True,
            on_chain=True,
            source_full=source_full,
        )
        if not proofs.complete:
            raise VerificationError("internal proof set is incomplete")
        self.state.mark_verified(
            round_index=window.index,
            message=message,
            details={
                **dict(common),
                "artifact": packaged.public_dict(),
                "proofs": proofs.to_dict(),
                "commitment_fingerprint": self._fingerprint(payload),
            },
            now=self.clock(),
        )
        log.info("round %d fully verified", window.index)
        return "verified"

    def _resolve_round(self, head: int) -> RoundWindow:
        chain_round = self.backend.chain_round(head)
        previous = self.state.read_status()
        window = resolve_round(
            base_url=self.config.coordinator_url,
            timeout=self.config.coordinator_timeout_seconds,
            chain_head=head,
            chain_round=chain_round,
            track=self.config.track,
            hardware_class=self.config.hardware_class,
            require_anchored=self.config.require_anchored_coordinator,
            allow_chain_fallback=self.config.allow_chain_schedule_fallback,
            previous_status=previous,
            fetcher=self.round_fetcher,
            base_model=self.config.base_model,
        )
        self.backend.verify_round_anchor(window)
        self.backend.validate_round(window)
        if window.source == "coordinator":
            # Persist the first accepted shape. validate_served_round refuses rollback or
            # mutation of this tuple on every subsequent fetch.
            previous["last_coordinator_round"] = window.index
            previous["last_coordinator_window"] = {
                "start_block": window.start_block,
                "close_block": window.close_block,
                "end_block": window.end_block,
                "config_hash": window.config_hash,
            }
            self.state.write(
                str(previous.get("phase", "round_resolved")),
                ok=bool(previous.get("ok", False)),
                message=str(previous.get("message", "trusted coordinator round resolved")),
                details=previous,
                preserve=previous,
            )
        return window

    def _refresh_same_round(self, expected: RoundWindow) -> tuple[int, RoundWindow]:
        head = self.backend.block()
        observed = self._resolve_round(head)
        if (
            observed.index,
            observed.start_block,
            observed.close_block,
            observed.end_block,
            observed.config_hash,
        ) != (
            expected.index,
            expected.start_block,
            expected.close_block,
            expected.end_block,
            expected.config_hash,
        ):
            raise RoundRefused("trusted round changed while preparing the submission")
        return head, observed

    def _assert_deadline(self, window: RoundWindow, block: int, action: str) -> None:
        if not window.accepts_at(block, self.config.deadline_margin_blocks):
            raise RoundRefused(
                f"refusing to {action} at block {block}; safe cutoff is "
                f"{window.close_block - self.config.deadline_margin_blocks}"
            )

    def _proofs_valid_for(
        self,
        state: Mapping[str, Any],
        round_index: int,
        *,
        source: str,
        hotkey: str,
        packaged: PackagedArtifact | None,
    ) -> bool:
        proofs = state.get("proofs")
        artifact = state.get("artifact")
        if packaged is None or not isinstance(artifact, dict):
            return False
        expected_artifact = packaged.public_dict()
        return (
            state.get("ok") is True
            and state.get("round") == round_index
            and state.get("last_success_round") == round_index
            and state.get("source") == source
            and artifact == expected_artifact
            and artifact.get("source") == source
            and artifact.get("hotkey") == hotkey
            and isinstance(proofs, dict)
            and all(
                proofs.get(key) is True
                for key in ("source", "source_full", "provenance", "on_chain")
            )
        )
    def _verification_due(self, state: Mapping[str, Any]) -> bool:
        try:
            elapsed = self.clock() - float(state["last_verified_at_epoch"])
        except (KeyError, TypeError, ValueError):
            return True
        return elapsed >= self.config.verify_interval_seconds

    @staticmethod
    def _local_matches(
        local: PackagedArtifact | None,
        round_index: int,
        source: str,
        hotkey: str,
    ) -> bool:
        return bool(
            local is not None
            and not local.sealed
            and local.round_index == round_index
            and local.source == source
            and local.hotkey == hotkey
        )

    def _round_details(
        self,
        snapshot: PreflightSnapshot,
        window: RoundWindow,
        head: int,
        source: str,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            **self._static_details(),
            "preflight": snapshot.to_dict(),
            "round": window.index,
            "chain_head": head,
            "round_window": window.to_dict(),
            "source": source,
        }
        if window.source == "coordinator":
            details["last_coordinator_round"] = window.index
            details["last_coordinator_window"] = {
                "start_block": window.start_block,
                "close_block": window.close_block,
                "end_block": window.end_block,
                "config_hash": window.config_hash,
            }
        return details

    def _static_details(self) -> dict[str, Any]:
        return {
            "dry_run": self.config.dry_run,
            "netuid": self.config.netuid,
            "network": self.config.network,
            "expected_uid": self.config.expected_uid,
            "wallet_name": self.config.wallet_name,
            "wallet_hotkey": self.config.wallet_hotkey,
            "competition": f"{self.config.track}/{self.config.hardware_class}",
        }

    @staticmethod
    def _planned_actions() -> list[str]:
        return [
            "package unsealed round-specific manifest",
            "upload artifact and manifest",
            "download and hash full remote artifact",
            "verify W&B provenance",
            "publish commitment",
            "read exact commitment back from chain",
        ]

    @staticmethod
    def _fingerprint(payload: str) -> str:
        import hashlib

        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def _failure(self, phase: str, exc: BaseException) -> None:
        previous = self.state.read_status()
        failures = int(previous.get("consecutive_failures", 0) or 0) + 1
        raw_message = str(exc) or exc.__class__.__name__
        message = redact_text(raw_message, self.state.secrets)
        self.state.write(
            phase,
            ok=False,
            message=message,
            details={
                **self._static_details(),
                "consecutive_failures": failures,
                "proofs": VerificationProofs(False, False, False, False).to_dict(),
            },
        )
        log.error("%s: %s", phase, message)
        print(f"{utc_timestamp()} ERROR {phase}: {message}", file=sys.stderr, flush=True)
