from __future__ import annotations

import os
import re
import stat
from collections.abc import MutableMapping
from pathlib import Path

from .errors import ConfigError

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
        metadata = resolved.lstat()
    except OSError as exc:
        raise ConfigError(f"environment file is unavailable: {resolved}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ConfigError(f"environment file must be a regular non-symlink: {resolved}")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ConfigError(f"environment file must have mode 0600: {resolved}")
    if metadata.st_uid != os.geteuid():
        raise ConfigError(
            f"environment file must be owned by effective UID {os.geteuid()}: {resolved}"
        )

    try:
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"environment file is unreadable UTF-8: {resolved}: {exc}") from exc
    if len(raw.encode("utf-8")) > 64 * 1024:
        raise ConfigError("environment file exceeds 64 KiB")

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
