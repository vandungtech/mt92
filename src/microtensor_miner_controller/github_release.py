from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import mimetypes
import os
import re
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.parse import quote

API_VERSION = "2026-03-10"
API_HOST = "api.github.com"
UPLOAD_HOST = "uploads.github.com"
USER_AGENT = "microtensor-miner-controller/0.1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ASSET_BYTES = 2 * 1024 * 1024 * 1024
HASH_CHUNK_BYTES = 4 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
PAGE_SIZE = 100
MAX_PAGES = 100

_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_REPO = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?\Z")
_TAG = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?\Z")
_FILENAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?\Z")


class ReleasePublishError(RuntimeError):
    """A fail-closed GitHub release publication error."""


@dataclass(frozen=True, slots=True)
class ApiResponse:
    """Small, secret-safe response envelope used by transports."""

    status: int
    data: Any = field(repr=False)


class ReleaseTransport(Protocol):
    """Injectable transport; implementations must not retain whole asset bodies."""

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> ApiResponse: ...

    def upload_asset(
        self,
        *,
        owner: str,
        repo: str,
        release_id: int,
        name: str,
        content_type: str,
        size: int,
        stream: BinaryIO,
    ) -> ApiResponse: ...


class GitHubTransport:
    """Minimal GitHub REST transport with chunked reads from local asset streams."""

    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: int = 60,
        upload_chunk_bytes: int = UPLOAD_CHUNK_BYTES,
    ) -> None:
        _validate_token(token)
        if timeout_seconds < 1 or upload_chunk_bytes < 1:
            raise ReleasePublishError("GitHub transport limits are invalid")
        self.__token = token
        self.timeout_seconds = timeout_seconds
        self.upload_chunk_bytes = upload_chunk_bytes

    def __repr__(self) -> str:
        return (
            "GitHubTransport(token=[REDACTED], "
            f"timeout_seconds={self.timeout_seconds}, "
            f"upload_chunk_bytes={self.upload_chunk_bytes})"
        )

    def _headers(
        self,
        *,
        content_type: str | None = None,
        size: int | None = None,
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.__token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        if size is not None:
            headers["Content-Length"] = str(size)
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> ApiResponse:
        if method not in {"GET", "POST", "PUT", "PATCH"} or not _safe_api_path(path):
            raise ReleasePublishError("GitHub API request is invalid")
        try:
            body = None
            content_type = None
            if payload is not None:
                body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                content_type = "application/json"
            connection = http.client.HTTPSConnection(API_HOST, timeout=self.timeout_seconds)
            try:
                connection.request(
                    method,
                    path,
                    body=body,
                    headers=self._headers(
                        content_type=content_type,
                        size=len(body) if body else None,
                    ),
                )
                return _read_json_response(connection.getresponse())
            finally:
                with contextlib.suppress(Exception):
                    connection.close()
        except ReleasePublishError:
            raise
        except Exception:
            raise ReleasePublishError("GitHub API request failed") from None

    def upload_asset(
        self,
        *,
        owner: str,
        repo: str,
        release_id: int,
        name: str,
        content_type: str,
        size: int,
        stream: BinaryIO,
    ) -> ApiResponse:
        if release_id < 1 or size < 0:
            raise ReleasePublishError("GitHub asset upload is invalid")
        target = (
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/releases/"
            f"{release_id}/assets?name={quote(name, safe='')}"
        )
        connection = http.client.HTTPSConnection(UPLOAD_HOST, timeout=self.timeout_seconds)
        try:
            connection.putrequest("POST", target)
            for key, value in self._headers(content_type=content_type, size=size).items():
                connection.putheader(key, value)
            connection.endheaders()
            remaining = size
            while remaining:
                chunk = stream.read(min(self.upload_chunk_bytes, remaining))
                if not isinstance(chunk, bytes) or not chunk or len(chunk) > remaining:
                    raise ReleasePublishError("GitHub asset stream ended unexpectedly")
                connection.send(chunk)
                remaining -= len(chunk)
            return _read_json_response(connection.getresponse())
        except ReleasePublishError:
            raise
        except Exception:
            raise ReleasePublishError("GitHub asset upload failed") from None
        finally:
            with contextlib.suppress(Exception):
                connection.close()


@dataclass(frozen=True, slots=True)
class PublishedAsset:
    name: str
    size: int
    digest: str
    download_url: str


@dataclass(frozen=True, slots=True)
class PublishedRelease:
    owner: str
    repo: str
    tag: str
    release_id: int
    source: str
    html_url: str
    assets: tuple[PublishedAsset, ...]
    already_published: bool


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedAsset:
    name: str
    stream: BinaryIO = field(compare=False)
    size: int
    digest: str
    content_type: str
    fingerprint: tuple[int, int, int, int]


class GitHubReleasePublisher:
    """Publish an exact asset set, then prove that the release is immutable."""

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        tag: str,
        token: str,
        transport: ReleaseTransport | None = None,
    ) -> None:
        _validate_token(token)
        _validate_coordinates(owner, repo, tag)
        self.owner = owner
        self.repo = repo
        self.tag = tag
        self._transport = transport if transport is not None else GitHubTransport(token)
        self._repo_path = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"

    def __repr__(self) -> str:
        return "GitHubReleasePublisher(token=[REDACTED])"

    def publish(
        self,
        assets: Mapping[str, str | os.PathLike[str]],
    ) -> PublishedRelease:
        with ExitStack() as stack:
            prepared = self._prepare_assets(assets, stack)
            self._verify_public_repository()
            self._enable_and_verify_immutability()

            existing = self._find_release()
            if existing is not None and existing.get("draft") is False:
                release_id = self._validate_published_release(existing)
                listed = self._list_assets(release_id)
                self._verify_asset_set(listed, prepared, require_complete=True)
                return self._result(release_id, prepared, already_published=True)

            release = self._create_draft() if existing is None else existing
            release_id = self._validate_draft_release(release)

            listed = self._list_assets(release_id)
            present = self._verify_asset_set(listed, prepared, require_complete=False)
            for asset in prepared:
                if asset.name in present:
                    continue
                self._upload(release_id, asset)

            listed = self._list_assets(release_id)
            self._verify_asset_set(listed, prepared, require_complete=True)
            self._publish_draft(release_id)

            final = self._request("GET", f"{self._repo_path}/releases/{release_id}")
            self._expect(final, {200}, "release verification")
            final_id = self._validate_published_release(_object(final.data))
            if final_id != release_id:
                raise ReleasePublishError("GitHub release identity mismatch")
            listed = self._list_assets(release_id)
            self._verify_asset_set(listed, prepared, require_complete=True)
            return self._result(release_id, prepared, already_published=False)

    def _prepare_assets(
        self,
        assets: Mapping[str, str | os.PathLike[str]],
        stack: ExitStack,
    ) -> tuple[_PreparedAsset, ...]:
        if not isinstance(assets, Mapping) or not assets or len(assets) > 100:
            raise ReleasePublishError("GitHub release asset set is invalid")
        names = list(assets)
        if any(not isinstance(name, str) or not _FILENAME.fullmatch(name) for name in names):
            raise ReleasePublishError("GitHub release asset filename is invalid")
        if len({name.casefold() for name in names}) != len(names):
            raise ReleasePublishError("GitHub release asset filenames are ambiguous")

        prepared: list[_PreparedAsset] = []
        try:
            for name in sorted(names):
                raw_path = assets[name]
                if not isinstance(raw_path, (str, os.PathLike)):
                    raise ReleasePublishError("GitHub release asset path is invalid")
                path = Path(raw_path)
                if path.is_symlink():
                    raise ReleasePublishError("GitHub release assets cannot be symbolic links")
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags)
                stream = stack.enter_context(os.fdopen(descriptor, "rb"))
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
                    raise ReleasePublishError(
                        "GitHub release asset must be a non-empty regular file"
                    )
                if before.st_size > MAX_ASSET_BYTES:
                    raise ReleasePublishError("GitHub release asset exceeds the size limit")
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = stream.read(HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                after = os.fstat(stream.fileno())
                fingerprint = _fingerprint(before)
                if total != before.st_size or _fingerprint(after) != fingerprint:
                    raise ReleasePublishError("GitHub release asset changed while hashing")
                stream.seek(0)
                content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                prepared.append(
                    _PreparedAsset(
                        name=name,
                        stream=stream,
                        size=before.st_size,
                        digest=f"sha256:{digest.hexdigest()}",
                        content_type=content_type,
                        fingerprint=fingerprint,
                    )
                )
        except ReleasePublishError:
            raise
        except Exception:
            raise ReleasePublishError("GitHub release asset preparation failed") from None
        return tuple(prepared)

    def _verify_public_repository(self) -> None:
        response = self._request("GET", self._repo_path)
        self._expect(response, {200}, "repository verification")
        repository = _object(response.data)
        if repository.get("private") is not False:
            raise ReleasePublishError("GitHub repository is not proven public")
        full_name = repository.get("full_name")
        expected = f"{self.owner}/{self.repo}"
        if not isinstance(full_name, str) or full_name.casefold() != expected.casefold():
            raise ReleasePublishError("GitHub repository identity mismatch")

    def _enable_and_verify_immutability(self) -> None:
        endpoint = f"{self._repo_path}/immutable-releases"
        enabled = self._request("PUT", endpoint)
        self._expect(enabled, {204}, "immutable-release enablement")
        verified = self._request("GET", endpoint)
        self._expect(verified, {200}, "immutable-release verification")
        if _object(verified.data).get("enabled") is not True:
            raise ReleasePublishError("GitHub immutable releases are not proven enabled")

    def _find_release(self) -> Mapping[str, Any] | None:
        direct = self._request(
            "GET",
            f"{self._repo_path}/releases/tags/{quote(self.tag, safe='')}",
        )
        self._expect(direct, {200, 404}, "release lookup")
        if direct.status == 200:
            return _object(direct.data)

        matches: list[Mapping[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            response = self._request(
                "GET",
                f"{self._repo_path}/releases?per_page={PAGE_SIZE}&page={page}",
            )
            self._expect(response, {200}, "draft release lookup")
            releases = _array(response.data)
            for item in releases:
                release = _object(item)
                if release.get("tag_name") == self.tag:
                    matches.append(release)
            if len(releases) < PAGE_SIZE:
                break
        else:
            raise ReleasePublishError("GitHub release lookup exceeded the safe page limit")
        if len(matches) > 1:
            raise ReleasePublishError("GitHub release tag is ambiguous")
        return matches[0] if matches else None

    def _create_draft(self) -> Mapping[str, Any]:
        response = self._request(
            "POST",
            f"{self._repo_path}/releases",
            payload={
                "tag_name": self.tag,
                "name": self.tag,
                "draft": True,
                "prerelease": False,
                "generate_release_notes": False,
            },
        )
        self._expect(response, {201}, "draft release creation")
        return _object(response.data)

    def _validate_draft_release(self, release: Mapping[str, Any]) -> int:
        release_id = _positive_id(release.get("id"))
        if (
            release.get("tag_name") != self.tag
            or release.get("draft") is not True
            or release.get("prerelease") is not False
            or release.get("immutable") is not False
        ):
            raise ReleasePublishError("GitHub draft release metadata mismatch")
        return release_id

    def _validate_published_release(self, release: Mapping[str, Any]) -> int:
        release_id = _positive_id(release.get("id"))
        if (
            release.get("tag_name") != self.tag
            or release.get("draft") is not False
            or release.get("prerelease") is not False
            or release.get("immutable") is not True
        ):
            raise ReleasePublishError("GitHub published release is not proven immutable")
        return release_id

    def _list_assets(self, release_id: int) -> list[Mapping[str, Any]]:
        assets: list[Mapping[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            response = self._request(
                "GET",
                f"{self._repo_path}/releases/{release_id}/assets"
                f"?per_page={PAGE_SIZE}&page={page}",
            )
            self._expect(response, {200}, "release asset lookup")
            batch = _array(response.data)
            assets.extend(_object(item) for item in batch)
            if len(batch) < PAGE_SIZE:
                break
        else:
            raise ReleasePublishError("GitHub asset lookup exceeded the safe page limit")
        return assets

    def _verify_asset_set(
        self,
        raw_assets: list[Mapping[str, Any]],
        expected_assets: tuple[_PreparedAsset, ...],
        *,
        require_complete: bool,
    ) -> set[str]:
        expected = {asset.name: asset for asset in expected_assets}
        present: set[str] = set()
        for raw in raw_assets:
            name = raw.get("name")
            if not isinstance(name, str) or name in present or name not in expected:
                raise ReleasePublishError("GitHub release contains an unexpected asset")
            self._verify_asset(raw, expected[name])
            present.add(name)
        if require_complete and present != set(expected):
            raise ReleasePublishError("GitHub release asset set is incomplete")
        return present

    @staticmethod
    def _verify_asset(raw: Mapping[str, Any], expected: _PreparedAsset) -> None:
        size = raw.get("size")
        if (
            raw.get("name") != expected.name
            or isinstance(size, bool)
            or size != expected.size
            or raw.get("state") != "uploaded"
        ):
            raise ReleasePublishError("GitHub release asset metadata mismatch")
        digest = raw.get("digest")
        if digest is not None and digest != expected.digest:
            raise ReleasePublishError("GitHub release asset digest mismatch")

    def _upload(self, release_id: int, asset: _PreparedAsset) -> None:
        try:
            if _fingerprint(os.fstat(asset.stream.fileno())) != asset.fingerprint:
                raise ReleasePublishError("GitHub release asset changed before upload")
            asset.stream.seek(0)
            response = self._transport.upload_asset(
                owner=self.owner,
                repo=self.repo,
                release_id=release_id,
                name=asset.name,
                content_type=asset.content_type,
                size=asset.size,
                stream=asset.stream,
            )
            if not isinstance(response, ApiResponse):
                raise ReleasePublishError("GitHub transport returned an invalid response")
        except Exception:
            raise ReleasePublishError("GitHub transport failed during asset upload") from None
        self._expect(response, {201}, "release asset upload")
        self._verify_asset(_object(response.data), asset)
        if (
            asset.stream.tell() != asset.size
            or _fingerprint(os.fstat(asset.stream.fileno())) != asset.fingerprint
        ):
            raise ReleasePublishError("GitHub release asset changed during upload")

    def _publish_draft(self, release_id: int) -> None:
        response = self._request(
            "PATCH",
            f"{self._repo_path}/releases/{release_id}",
            payload={"draft": False, "prerelease": False},
        )
        self._expect(response, {200}, "draft release publication")
        release = _object(response.data)
        if (
            _positive_id(release.get("id")) != release_id
            or release.get("tag_name") != self.tag
            or release.get("draft") is not False
            or release.get("prerelease") is not False
        ):
            raise ReleasePublishError("GitHub publication response mismatch")

    def _result(
        self,
        release_id: int,
        assets: tuple[_PreparedAsset, ...],
        *,
        already_published: bool,
    ) -> PublishedRelease:
        prefix = f"https://github.com/{self.owner}/{self.repo}/releases/download/{self.tag}"
        published_assets = tuple(
            PublishedAsset(
                name=asset.name,
                size=asset.size,
                digest=asset.digest,
                download_url=f"{prefix}/{asset.name}",
            )
            for asset in assets
        )
        return PublishedRelease(
            owner=self.owner,
            repo=self.repo,
            tag=self.tag,
            release_id=release_id,
            source=f"https:github.com/{self.owner}/{self.repo}/releases/download/{self.tag}",
            html_url=f"https://github.com/{self.owner}/{self.repo}/releases/tag/{self.tag}",
            assets=published_assets,
            already_published=already_published,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> ApiResponse:
        try:
            response = self._transport.request(method, path, payload=payload)
            if not isinstance(response, ApiResponse):
                raise ReleasePublishError("GitHub transport returned an invalid response")
            return response
        except Exception:
            raise ReleasePublishError("GitHub transport request failed") from None

    @staticmethod
    def _expect(response: ApiResponse, statuses: set[int], operation: str) -> None:
        if response.status not in statuses:
            raise ReleasePublishError(f"GitHub {operation} failed")


def _validate_token(token: str) -> None:
    if (
        not isinstance(token, str)
        or len(token) < 8
        or len(token) > 4096
        or token.strip() != token
        or any(ord(character) < 33 or ord(character) == 127 for character in token)
    ):
        raise ReleasePublishError("GitHub token is invalid")


def _validate_coordinates(owner: str, repo: str, tag: str) -> None:
    if not isinstance(owner, str) or not _OWNER.fullmatch(owner) or "--" in owner:
        raise ReleasePublishError("GitHub owner is invalid")
    if (
        not isinstance(repo, str)
        or not _REPO.fullmatch(repo)
        or repo in {".", ".."}
        or repo.casefold().endswith(".git")
    ):
        raise ReleasePublishError("GitHub repository name is invalid")
    if (
        not isinstance(tag, str)
        or not _TAG.fullmatch(tag)
        or ".." in tag
        or tag.casefold().endswith(".lock")
    ):
        raise ReleasePublishError("GitHub release tag is invalid")


def _safe_api_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and path.startswith("/")
        and not path.startswith("//")
        and "\r" not in path
        and "\n" not in path
        and "#" not in path
    )


def _read_json_response(response: http.client.HTTPResponse) -> ApiResponse:
    try:
        raw = response.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ReleasePublishError("GitHub API response exceeded the size limit")
        data: Any = None if not raw else json.loads(raw.decode("utf-8"))
        return ApiResponse(status=int(response.status), data=data)
    except ReleasePublishError:
        raise
    except Exception:
        raise ReleasePublishError("GitHub API returned an invalid response") from None


def _object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleasePublishError("GitHub API response object is invalid")
    return value


def _array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ReleasePublishError("GitHub API response array is invalid")
    return value


def _positive_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReleasePublishError("GitHub release identity is invalid")
    return value


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
