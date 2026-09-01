from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from .backend import Backend
from .config import ControllerConfig
from .coordinator import fetch_current_round, resolve_round
from .errors import AuthorizationRefused, ControllerError, RoundRefused, VerificationError
from .leaderboard import RankMonitor
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
        reason = self._authorization_latch_reason()
        if reason is not None:
            raise AuthorizationRefused(f"transaction authorization remains latched: {reason}")
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
            except AuthorizationRefused as exc:
                self._authorization_failure(exc)
                return 3
            except ControllerError as exc:
                self._failure("preflight_refused", exc)
                return 2
            except Exception as exc:
                self._failure("preflight_error", exc)
                return 1
            finally:
                self.backend.close()

    def run(self, *, once: bool = False) -> int:
        rank_monitor: RankMonitor | None = None
        with self.state.lock():
            try:
                snapshot = self.preflight()
                if not once:
                    rank_monitor = self._start_rank_monitor(snapshot.hotkey)
                while not self.stop_event.is_set():
                    try:
                        outcome = self.cycle(snapshot)
                    except AuthorizationRefused as exc:
                        self._authorization_failure(exc)
                        return 3
                    except RoundRefused as exc:
                        if self.config.uses_signed_v030:
                            self._waiting_for_trusted_round(exc)
                            outcome = "waiting"
                        else:
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
            except AuthorizationRefused as exc:
                self._authorization_failure(exc)
                return 3
            except ControllerError as exc:
                self._failure("preflight_refused", exc)
                return 2
            except Exception as exc:
                self._failure("fatal", exc)
                return 1
            finally:
                if rank_monitor is not None:
                    try:
                        rank_monitor.stop()
                    except Exception as exc:
                        log.warning("rank observer did not stop cleanly: %s", exc)
                self.backend.close()

    def _start_rank_monitor(self, hotkey: str) -> RankMonitor | None:
        try:
            monitor = RankMonitor(
                self.state,
                track=self.config.track,
                hardware_class=self.config.hardware_class,
                hotkey=hotkey,
            )
            monitor.start()
            return monitor
        except Exception as exc:
            # Public standing is observational only: it never gates or authorises a cycle.
            message = redact_text(str(exc) or exc.__class__.__name__, self.state.secrets)
            log.warning("rank observer unavailable; submission supervision continues: %s", message)
            return None

    def cycle(self, snapshot: PreflightSnapshot) -> str:
        reason = self._authorization_latch_reason()
        if reason is not None:
            raise AuthorizationRefused(f"transaction authorization remains latched: {reason}")

        # Registration can change while this long-running process remains alive. Keep
        # the read-only monitor honest instead of relying only on startup preflight.
        self.backend.assert_registered()
        head = self.backend.block()
        activation = self.config.v030_activation_block
        if activation is not None and head < activation:
            self.state.write(
                "waiting",
                ok=False,
                message=(
                    f"signed v0.3 activation is blocked until chain height {activation}; "
                    f"current head is {head}"
                ),
                details={
                    **self._static_details(),
                    "preflight": snapshot.to_dict(),
                    "chain_head": head,
                    "proofs": VerificationProofs(False, False, False, False).to_dict(),
                },
            )
            return "waiting"
        window = self._resolve_round(head)
        source = self.config.source_for(window.index, snapshot.hotkey)
        common = self._round_details(snapshot, window, head, source)

        previous = self.state.read_status()

        pending = self.state.read_submission_pending()
        if pending:
            local = self.backend.load_local()
            return self._reconcile_pending(
                snapshot,
                window,
                head,
                source,
                local,
                pending,
            )

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
        proofs_valid = matching and self._proofs_valid_for(
            previous,
            window.index,
            source=source,
            hotkey=snapshot.hotkey,
            packaged=local,
        )

        if not window.accepts_at(head, self.config.deadline_margin_blocks):
            if matching:
                if local is None:
                    raise VerificationError("matching local artifact unexpectedly disappeared")
                if not proofs_valid or self._verification_due(previous):
                    return self._reverify(snapshot, window, head, source, local)
            message = f"round {window.index} is outside the safe submission window; " + (
                "matching commitment remains verified" if proofs_valid else "no verified commitment"
            )
            self.state.write(
                "waiting",
                ok=proofs_valid,
                message=message,
                details=common,
            )
            return "waiting"

        if matching:
            if local is None:
                raise VerificationError("matching local artifact unexpectedly disappeared")
            if proofs_valid and previous.get("ok") is True and not self._verification_due(previous):
                self.state.heartbeat(now=self.clock())
                return "verified"
            if proofs_valid:
                return self._reverify(snapshot, window, head, source, local)

        return self._submit(snapshot, window, head, source, local)

    def _reconcile_pending(
        self,
        snapshot: PreflightSnapshot,
        window: RoundWindow,
        head: int,
        source: str,
        local: PackagedArtifact | None,
        pending: Mapping[str, Any],
    ) -> str:
        if pending.get("marker_invalid") is True or pending.get("submission_pending") is not True:
            raise AuthorizationRefused(
                "submission pending marker is invalid; operator review is required"
            )

        pending_round = pending.get("round")
        pending_source = pending.get("source")
        pending_hotkey = pending.get("hotkey")
        pending_fingerprint = pending.get("commitment_fingerprint")
        provenance_block = pending.get("provenance_block")
        if (
            type(pending_round) is not int
            or not isinstance(pending_source, str)
            or not pending_source
            or not isinstance(pending_hotkey, str)
            or not pending_hotkey
            or not isinstance(pending_fingerprint, str)
            or not pending_fingerprint.startswith("sha256:")
            or type(provenance_block) is not int
            or provenance_block < 0
            or provenance_block > head
        ):
            raise AuthorizationRefused(
                "submission pending marker fields are invalid; operator review is required"
            )
        if (
            pending_round != window.index
            or pending_source != source
            or pending_hotkey != snapshot.hotkey
        ):
            raise AuthorizationRefused(
                "submission pending marker does not match the current round, source, and hotkey"
            )
        if not self._local_matches(local, window.index, source, snapshot.hotkey):
            raise AuthorizationRefused(
                "submission pending marker has no matching local signed manifest"
            )
        if local is None:
            raise AuthorizationRefused(
                "submission pending marker has no matching local signed manifest"
            )

        try:
            encoded = self.backend.validate_commitment(local)
        except Exception as exc:
            raise AuthorizationRefused(
                "pending commitment cannot be reconstructed exactly; operator review is required"
            ) from exc
        fingerprint = self._fingerprint(encoded)
        if fingerprint != pending_fingerprint:
            raise AuthorizationRefused(
                "pending marker fingerprint differs from the local signed commitment"
            )

        common = self._round_details(snapshot, window, head, source)
        artifact_details = {
            **common,
            "artifact": local.public_dict(),
            "commitment_fingerprint": fingerprint,
            "submission_pending": True,
        }
        self.state.write(
            "reconciling_submission",
            ok=False,
            message="performing read-only reconciliation of an uncertain submission",
            details=artifact_details,
        )
        try:
            self.backend.assert_registered()
            payload = self.backend.verify_on_chain(local)
        except Exception as exc:
            raise AuthorizationRefused(
                "pending submission is not proven exact on chain; operator review is required"
            ) from exc
        if self._fingerprint(payload) != fingerprint:
            raise AuthorizationRefused(
                "on-chain readback differs from the pending commitment fingerprint"
            )

        try:
            self.backend.verify_source(local, full=True)
            self.backend.verify_provenance(local, provenance_block)
        except Exception as exc:
            raise AuthorizationRefused(
                "pending submission proof bundle could not be reverified; "
                "operator review is required"
            ) from exc
        try:
            self.state.clear_submission_pending(fingerprint)
        except Exception as exc:
            raise AuthorizationRefused(
                "exact on-chain submission was verified but its pending marker could not be cleared"
            ) from exc

        return self._verified(
            window,
            local,
            payload,
            common,
            source_full=True,
            message=(
                "pending submission reconciled read-only; source, provenance policy, "
                "and exact on-chain commitment verified"
            ),
        )

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
            if local is None:
                raise VerificationError("reusable local artifact unexpectedly disappeared")
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
            message="checking the signed upstream provenance policy",
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
                log.warning(
                    "exact remote recovery probe failed; a controlled re-upload is required"
                )

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
            self.backend.assert_registered()
            commitment_fingerprint = self._fingerprint(encoded)
            try:
                self.state.mark_submission_pending(
                    round_index=packaged.round_index,
                    source=packaged.source,
                    hotkey=packaged.hotkey,
                    commitment_fingerprint=commitment_fingerprint,
                    details={"provenance_block": provenance_block},
                    now=self.clock(),
                )
            except Exception as exc:
                raise AuthorizationRefused(
                    "commitment was not submitted because its durable pending marker "
                    "could not be recorded"
                ) from exc

            try:
                broadcast_head, broadcast_window = self._refresh_same_round(window)
                self._assert_deadline(broadcast_window, broadcast_head, "publish")
                self.backend.assert_registered()
            except Exception:
                try:
                    self.state.clear_submission_pending(commitment_fingerprint)
                except Exception as clear_exc:
                    raise AuthorizationRefused(
                        "commitment was not submitted, but its pending marker could not be cleared"
                    ) from clear_exc
                raise

            try:
                receipt = self.backend.publish(packaged)
                if receipt.round_index != packaged.round_index:
                    raise AuthorizationRefused(
                        "publish receipt names a different round; submission outcome is ambiguous"
                    )
                self.state.write(
                    "verifying_on_chain",
                    ok=False,
                    message="reading the exact commitment back from chain",
                    details=artifact_details,
                )
                payload = self.backend.verify_on_chain(packaged)
                if self._fingerprint(payload) != commitment_fingerprint:
                    raise AuthorizationRefused(
                        "post-submission chain readback differs from the pending commitment"
                    )
            except AuthorizationRefused:
                raise
            except Exception as exc:
                raise AuthorizationRefused(
                    "submission outcome is ambiguous after publish began; "
                    "operator review is required"
                ) from exc

            try:
                self.state.clear_submission_pending(commitment_fingerprint)
            except Exception as exc:
                raise AuthorizationRefused(
                    "exact on-chain submission was verified but its pending marker "
                    "could not be cleared"
                ) from exc

        return self._verified(
            window,
            packaged,
            payload,
            common,
            source_full=True,
            message="source, provenance policy, and exact on-chain commitment verified",
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
            message="refreshing remote manifest, provenance policy, and on-chain proof",
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
            message="periodic source, provenance policy, and on-chain verification passed",
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
            strict_v030=self.config.uses_signed_v030,
        )
        try:
            self.backend.verify_round_anchor(window)
            self.backend.validate_round(window)
        except VerificationError as exc:
            if not self.config.uses_signed_v030:
                raise
            raise RoundRefused(f"trusted signed-v0.3 round is unavailable: {exc}") from exc
        if window.source == "coordinator":
            # Persist the first accepted shape. validate_served_round refuses rollback or
            # mutation of this tuple on every subsequent fetch.
            previous["last_coordinator_round"] = window.index
            previous["last_coordinator_window"] = self._coordinator_window_details(window)
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
        activation = self.config.v030_activation_block
        if activation is not None and head < activation:
            raise RoundRefused(
                f"signed v0.3 activation height {activation} is above chain head {head}"
            )
        observed = self._resolve_round(head)
        identity = (
            "index",
            "start_block",
            "seed_block",
            "close_block",
            "end_block",
            "phase",
            "block_hash",
            "config_hash",
            "mechanism_version",
            "corpus_version",
            "corpus_digest",
        )
        if tuple(getattr(observed, key) for key in identity) != tuple(
            getattr(expected, key) for key in identity
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
            details["last_coordinator_window"] = self._coordinator_window_details(window)
        return details

    @staticmethod
    def _coordinator_window_details(window: RoundWindow) -> dict[str, Any]:
        legacy = {
            "start_block": window.start_block,
            "close_block": window.close_block,
            "end_block": window.end_block,
            "config_hash": window.config_hash,
        }
        if not window.mechanism_version:
            return legacy
        return {
            "round": window.index,
            "start_block": window.start_block,
            "seed_block": window.seed_block,
            "close_block": window.close_block,
            "end_block": window.end_block,
            "phase": window.phase,
            "block_hash": window.block_hash,
            "config_hash": window.config_hash,
            "mechanism_version": window.mechanism_version,
            "corpus_version": window.corpus_version,
            "corpus_digest": window.corpus_digest,
        }

    def _static_details(self) -> dict[str, Any]:
        return {
            "dry_run": self.config.dry_run,
            "netuid": self.config.netuid,
            "network": self.config.network,
            "expected_uid": self.config.expected_uid,
            "wallet_name": self.config.wallet_name,
            "wallet_hotkey": self.config.wallet_hotkey,
            "competition": f"{self.config.track}/{self.config.hardware_class}",
            "protocol_profile": (
                "signed-v0.3" if self.config.uses_signed_v030 else "legacy-v0.1.14"
            ),
            "upstream_release": self.config.upstream_release,
            "provenance_required": self.config.provenance_required,
            "v030_activation_block": self.config.v030_activation_block,
            "source": None,
            "source_template": self.config.source_template,
            "transaction_authorization": {
                "call": "Commitments.set_commitment",
                "netuid": 92,
                "uid": 32,
                "wallet_hotkey": "you-hot1",
                "authorized_max_fee_rao": 0,
                "authorized_max_deposit_rao": 0,
                "fee_and_deposit_check": "performed immediately before signing",
                "tip_rao": 0,
                "mev_protection": False,
            },
        }

    @staticmethod
    def _planned_actions() -> list[str]:
        return [
            "package unsealed round-specific manifest",
            "upload artifact and manifest",
            "download and hash full remote artifact",
            "satisfy the signed upstream provenance policy",
            "estimate 0-rao fee and confirm 0-rao commitment deposits",
            "publish one direct Commitments.set_commitment with the hotkey",
            "read exact commitment back from chain",
        ]

    @staticmethod
    def _fingerprint(payload: str) -> str:
        import hashlib

        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def _authorization_latch_reason(self) -> str | None:
        try:
            marker = self.state.read_authorization_refusal()
        except Exception:
            return "authorization refusal marker could not be checked; operator review is required"
        if marker:
            reason = marker.get("authorization_reason")
            if isinstance(reason, str) and reason:
                return reason
            return "authorization refusal marker is invalid; operator review is required"

        previous = self.state.read_status()
        if previous.get("authorization_latched") is True:
            reason = previous.get("authorization_reason")
            if isinstance(reason, str) and reason:
                return reason
            return "legacy authorization latch requires operator review"
        return None

    def _waiting_for_trusted_round(self, exc: BaseException) -> None:
        raw_message = str(exc) or exc.__class__.__name__
        message = redact_text(raw_message, self.state.secrets)
        self.state.write(
            "waiting",
            ok=False,
            message=message,
            details={
                **self._static_details(),
                "proofs": VerificationProofs(False, False, False, False).to_dict(),
            },
        )
        log.warning("waiting for a trusted signed-v0.3 round: %s", message)

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

    def _authorization_failure(self, exc: BaseException) -> None:
        failures = 1
        try:
            previous = self.state.read_status()
            failures = int(previous.get("consecutive_failures", 0) or 0) + 1
        except Exception:
            failures = 1

        try:
            raw_message = str(exc) or exc.__class__.__name__
            message = redact_text(raw_message, self.state.secrets)
        except Exception:
            message = "transaction authorization was refused; operator review is required"

        try:
            details = {
                **self._static_details(),
                "authorization_latched": True,
                "authorization_reason": message,
                "consecutive_failures": failures,
                "proofs": VerificationProofs(False, False, False, False).to_dict(),
            }
        except Exception:
            details = {
                "authorization_latched": True,
                "authorization_reason": message,
                "consecutive_failures": failures,
            }

        persistence_failures: list[str] = []
        try:
            self.state.latch_authorization_refusal(message, details=details)
        except Exception as persist_exc:
            persistence_failures.append(f"dedicated latch: {persist_exc.__class__.__name__}")
        try:
            self.state.write(
                "authorization_refused",
                ok=False,
                message=message,
                details=details,
            )
        except Exception as persist_exc:
            persistence_failures.append(f"status: {persist_exc.__class__.__name__}")

        with contextlib.suppress(Exception):
            log.critical("authorization_refused: %s", message)
            if persistence_failures:
                log.critical(
                    "authorization refusal persistence failed (%s); process will still exit 3",
                    ", ".join(persistence_failures),
                )
        with contextlib.suppress(Exception):
            print(
                f"{utc_timestamp()} CRITICAL authorization_refused: {message}",
                file=sys.stderr,
                flush=True,
            )
