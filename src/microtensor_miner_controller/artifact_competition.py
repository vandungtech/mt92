from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .binding import artifact_digest
from .config import ControllerConfig
from .errors import ArtifactCompetitionBindingError, PreflightError
from .protected_file import ProtectedFileError, read_root_service_file

ARTIFACT_COMPETITION_BINDING_SCHEMA_VERSION = 1
MAX_ARTIFACT_COMPETITION_BINDING_BYTES = 4096
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPETITION_NAME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_digest",
        "track",
        "hardware_class",
    }
)


def _read_private_binding(path: Path) -> bytes:
    try:
        payload = read_root_service_file(
            path,
            label="artifact competition binding",
            maximum_bytes=MAX_ARTIFACT_COMPETITION_BINDING_BYTES,
        )
    except ProtectedFileError as exc:
        raise ArtifactCompetitionBindingError(str(exc)) from None
    if not payload:
        raise ArtifactCompetitionBindingError("artifact competition binding must not be empty")
    return payload


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


def _parse_binding(path: Path) -> dict[str, Any]:
    encoded = _read_private_binding(path)
    try:
        raw = encoded.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ArtifactCompetitionBindingError(
            "artifact competition binding is not strict UTF-8"
        ) from None
    try:
        observed = _strict_json_object(raw)
    except (json.JSONDecodeError, ValueError):
        raise ArtifactCompetitionBindingError(
            "artifact competition binding is not strict JSON"
        ) from None

    if set(observed) != _FIELDS:
        raise ArtifactCompetitionBindingError(
            "artifact competition binding has missing or extra fields"
        )
    if (
        type(observed["schema_version"]) is not int
        or observed["schema_version"] != ARTIFACT_COMPETITION_BINDING_SCHEMA_VERSION
    ):
        raise ArtifactCompetitionBindingError(
            "artifact competition binding has an unsupported schema version"
        )
    digest = observed["artifact_digest"]
    track = observed["track"]
    hardware_class = observed["hardware_class"]
    if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
        raise ArtifactCompetitionBindingError(
            "artifact competition binding has a non-canonical artifact digest"
        )
    if type(track) is not str or _COMPETITION_NAME.fullmatch(track) is None:
        raise ArtifactCompetitionBindingError(
            "artifact competition binding has an invalid track"
        )
    if (
        type(hardware_class) is not str
        or _COMPETITION_NAME.fullmatch(hardware_class) is None
    ):
        raise ArtifactCompetitionBindingError(
            "artifact competition binding has an invalid hardware class"
        )
    return observed


def validate_artifact_competition_binding(
    config: ControllerConfig,
) -> dict[str, Any]:
    observed = _parse_binding(config.artifact_competition_binding_path)
    bound_competition = (observed["track"], observed["hardware_class"])
    configured_competition = (config.track, config.hardware_class)
    if bound_competition != configured_competition:
        raise ArtifactCompetitionBindingError(
            "artifact competition binding targets "
            f"{bound_competition[0]}/{bound_competition[1]}, but the controller targets "
            f"{configured_competition[0]}/{configured_competition[1]}"
        )
    try:
        current_digest, _file_count, _total_bytes = artifact_digest(config.artifact_dir)
    except PreflightError as exc:
        raise ArtifactCompetitionBindingError(
            "the current artifact tree digest cannot be established"
        ) from exc
    except OSError:
        raise ArtifactCompetitionBindingError(
            "the current artifact tree digest cannot be established"
        ) from None
    if observed["artifact_digest"] != current_digest:
        raise ArtifactCompetitionBindingError(
            "artifact competition binding digest does not match the current artifact tree"
        )
    return observed
