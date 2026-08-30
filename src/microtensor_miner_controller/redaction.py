from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT = re.compile(
    r"(?i)\b(token|secret|password|passwd|api[_-]?key|credential|signature)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_URL_CREDS = re.compile(r"(https?://)([^/@\s:]+):([^/@\s]+)@", re.IGNORECASE)
_SENSITIVE_QUERY = frozenset(
    {"token", "access_token", "api_key", "apikey", "key", "secret", "signature", "credential"}
)


def secret_values(env: Mapping[str, str]) -> tuple[str, ...]:
    found: list[str] = []
    for key, value in env.items():
        upper = key.upper()
        if value and len(value) >= 4 and any(
            marker in upper
            for marker in ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "CREDENTIAL")
        ):
            found.append(value)
    return tuple(sorted(set(found), key=len, reverse=True))


def redact_text(value: object, secrets: Sequence[str] = ()) -> str:
    text = str(value)
    for secret in sorted((s for s in secrets if len(s) >= 4), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    text = _BEARER.sub(r"\1" + REDACTED, text)
    text = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    text = _URL_CREDS.sub(r"\1" + REDACTED + "@", text)
    return text


def safe_source(source: str, secrets: Sequence[str] = ()) -> str:
    """Keep a source useful to an operator while dropping URL credentials/query secrets."""

    clean = str(source)
    for secret in sorted((s for s in secrets if len(s) >= 4), key=len, reverse=True):
        clean = clean.replace(secret, REDACTED)
    clean = _BEARER.sub(r"\1" + REDACTED, clean)
    clean = _URL_CREDS.sub(r"\1" + REDACTED + "@", clean)
    scheme, separator, locator = clean.partition(":")
    if not separator:
        return clean
    if scheme not in {"http", "https"}:
        return clean.split("?", 1)[0]

    # Microtensor's source is normally https:host/path rather than https://host/path.
    parsed = urlsplit(clean if locator.startswith("//") else f"{scheme}://{locator.lstrip('/')}")

    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    query = urlencode([
        (key, REDACTED if key.lower() in _SENSITIVE_QUERY else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    rendered = urlunsplit((scheme, netloc, parsed.path, query, ""))
    return f"{scheme}:" + rendered.removeprefix(f"{scheme}://")


def sanitize(value: Any, secrets: Sequence[str] = (), *, key: str = "") -> Any:
    upper = key.upper()
    if any(marker in upper for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")):
        return REDACTED if value else value
    if isinstance(value, Mapping):
        return {str(k): sanitize(v, secrets, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, secrets) for item in value]
    if isinstance(value, str):
        if key.lower() == "source":
            return safe_source(value, secrets)
        return redact_text(value, secrets)
    return value


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Sequence[str] = ()) -> None:
        super().__init__()
        self.secrets = tuple(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.getMessage(), self.secrets)
        record.args = ()
        if record.exc_info:
            # The formatted exception may contain a URL/token. Suppress the raw traceback at
            # normal levels; DEBUG users can reproduce locally without persisting credentials.
            record.exc_info = None
            record.exc_text = None
        return True
