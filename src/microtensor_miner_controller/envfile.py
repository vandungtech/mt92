from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

from .errors import ConfigError
from .protected_file import ProtectedFileError, read_root_service_file

_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_VALUE_CHARS = frozenset("\r\n\x00`$")


def load_env_file(
    path: Path,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Load a deliberately small dotenv subset after strict ownership/mode checks.

    The file is data, never shell code: each non-comment line is one unquoted
    ``UPPER_CASE_NAME=value`` assignment. Existing process variables win, which lets
    an operator inject a one-off override without modifying the protected file.
    """

    destination = os.environ if environ is None else environ
    resolved = Path(path).expanduser().absolute()
    try:
        encoded = read_root_service_file(
            resolved,
            label="environment file",
            maximum_bytes=64 * 1024,
        )
    except ProtectedFileError as exc:
        raise ConfigError(str(exc)) from None
    try:
        raw = encoded.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ConfigError("environment file is not valid UTF-8") from None

    loaded: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _NAME.fullmatch(name):
            raise ConfigError(f"invalid environment assignment on line {line_number}")
        if name in seen:
            raise ConfigError(f"duplicate environment variable {name} on line {line_number}")
        seen.add(name)
        if value != value.strip():
            raise ConfigError(f"environment value for {name} must not have edge whitespace")
        if value.startswith(('"', "'")) or value.endswith(('"', "'")):
            raise ConfigError(f"environment value for {name} must be unquoted")
        if any(character in value for character in _FORBIDDEN_VALUE_CHARS):
            raise ConfigError(f"environment value for {name} contains a forbidden character")
        if name not in destination:
            destination[name] = value
        loaded.append(name)
    return tuple(loaded)
