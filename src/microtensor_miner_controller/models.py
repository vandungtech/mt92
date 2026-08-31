from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RoundWindow:
    index: int
    start_block: int
    close_block: int
    end_block: int
    phase: str
    source: str
    config_hash: str = ""
    seed_block: int = 0
    block_hash: str = ""
    max_size_bytes: int = 0
    max_rss_bytes: int = 0
    max_p95_ms: int = 0
    mechanism_version: str = ""
    corpus_version: str = ""
    corpus_digest: str = ""
    metric: str = ""
    emission_share: float = 0.0
    cpu_seconds_per_artifact: int = 0
    tasks_per_round: int = 0
    environment_digest: str = ""

    @property
    def accepts_submissions(self) -> bool:
        return self.phase == "submissions"

    def accepts_at(self, block: int, margin_blocks: int = 0) -> bool:
        return (
            self.accepts_submissions
            and self.start_block <= block
            and block < self.close_block - margin_blocks
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreflightSnapshot:
    hotkey: str
    uid: int
    chain_head: int
    upstream_version: str
    upstream_commit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PackagedArtifact:
    round_index: int
    source: str
    hotkey: str
    manifest_digest: str
    artifact_digest: str
    file_count: int
    total_bytes: int
    sealed: bool = False
    native: Any = field(default=None, repr=False, compare=False)

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("native", None)
        return payload


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    round_index: int
    payload: str


@dataclass(frozen=True, slots=True)
class VerificationProofs:
    source: bool
    provenance: bool
    on_chain: bool
    source_full: bool = False

    @property
    def complete(self) -> bool:
        return self.source and self.provenance and self.on_chain and self.source_full

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)
