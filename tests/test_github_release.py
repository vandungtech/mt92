from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from microtensor_miner_controller.github_release import (
    API_VERSION,
    ApiResponse,
    GitHubReleasePublisher,
    GitHubTransport,
    ReleasePublishError,
)

OWNER = "microtensor-miner"
REPO = "artifacts"
TAG = "r1238"
TOKEN = "github_pat_test_secret_never_echo"  # noqa: S105
BASE = f"/repos/{OWNER}/{REPO}"


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _asset(name: str, data: bytes) -> dict[str, Any]:
    return {
        "id": 100,
        "name": name,
        "size": len(data),
        "state": "uploaded",
        "digest": _digest(data),
    }


def _release(
    release_id: int,
    *,
    draft: bool,
    immutable: bool,
    assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": release_id,
        "tag_name": TAG,
        "draft": draft,
        "prerelease": False,
        "immutable": immutable,
        "assets": list(assets or []),
    }


class FakeTransport:
    """Stateful GitHub double. Asset streams are consumed in seven-byte chunks."""

    def __init__(self) -> None:
        self.repo_private = False
        self.immutable_put_status = 204
        self.immutable_get_status = 200
        self.immutable_enabled = True
        self.publish_immutable = True
        self.upload_digest_mismatch = False
        self.raise_error: Exception | None = None
        self.releases: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str, str]] = []
        self.uploads: list[str] = []
        self.upload_read_sizes: list[int] = []
        self.patch_count = 0
        self.next_release_id = 41

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> ApiResponse:
        self.calls.append(("request", method, path))
        if self.raise_error is not None:
            raise self.raise_error
        if method == "GET" and path == BASE:
            return ApiResponse(
                200,
                {"private": self.repo_private, "full_name": f"{OWNER}/{REPO}"},
            )
        if path == f"{BASE}/immutable-releases" and method == "PUT":
            return ApiResponse(self.immutable_put_status, None)
        if path == f"{BASE}/immutable-releases" and method == "GET":
            return ApiResponse(
                self.immutable_get_status,
                {"enabled": self.immutable_enabled},
            )
        if method == "GET" and path == f"{BASE}/releases/tags/{TAG}":
            published = [
                item
                for item in self.releases
                if item["tag_name"] == TAG and not item["draft"]
            ]
            return ApiResponse(200, dict(published[0])) if published else ApiResponse(404, None)
        if method == "GET" and path.startswith(f"{BASE}/releases?per_page="):
            return ApiResponse(200, [dict(item) for item in self.releases])
        if method == "POST" and path == f"{BASE}/releases":
            if payload is None:
                raise AssertionError("create-release payload is missing")
            release = _release(
                self.next_release_id,
                draft=bool(payload["draft"]),
                immutable=False,
            )
            release["tag_name"] = payload["tag_name"]
            release["prerelease"] = payload["prerelease"]
            self.releases.append(release)
            self.next_release_id += 1
            return ApiResponse(201, dict(release))

        asset_match = re.fullmatch(
            re.escape(BASE) + r"/releases/(\d+)/assets\?per_page=100&page=\d+",
            path,
        )
        if method == "GET" and asset_match:
            release = self._by_id(int(asset_match.group(1)))
            return ApiResponse(200, [dict(item) for item in release["assets"]])

        release_match = re.fullmatch(re.escape(BASE) + r"/releases/(\d+)", path)
        if release_match:
            release = self._by_id(int(release_match.group(1)))
            if method == "PATCH":
                if payload != {"draft": False, "prerelease": False}:
                    raise AssertionError("publication payload is incorrect")
                release["draft"] = False
                release["prerelease"] = False
                release["immutable"] = self.publish_immutable
                self.patch_count += 1
                return ApiResponse(200, dict(release))
            if method == "GET":
                return ApiResponse(200, dict(release))
        return ApiResponse(599, None)

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
        del content_type
        self.calls.append(("upload", "POST", name))
        if self.raise_error is not None:
            raise self.raise_error
        if owner != OWNER:
            raise AssertionError("upload owner is incorrect")
        if repo != REPO:
            raise AssertionError("upload repository is incorrect")
        body = bytearray()
        while True:
            chunk = stream.read(7)
            if not chunk:
                break
            self.upload_read_sizes.append(len(chunk))
            body.extend(chunk)
        if len(body) != size:
            raise AssertionError("upload stream length is incorrect")
        digest = _digest(bytes(body))
        if self.upload_digest_mismatch:
            digest = "sha256:" + "0" * 64
        asset = {
            "id": 100 + len(self.uploads),
            "name": name,
            "size": size,
            "state": "uploaded",
            "digest": digest,
        }
        self._by_id(release_id)["assets"].append(asset)
        self.uploads.append(name)
        return ApiResponse(201, dict(asset))

    def _by_id(self, release_id: int) -> dict[str, Any]:
        return next(item for item in self.releases if item["id"] == release_id)


class GitHubReleasePublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.model = b"model-contents-" * 257
        self.manifest = b'{"schema_version":1}\n'
        model_path = root / "model.gguf"
        manifest_path = root / "manifest.json"
        model_path.write_bytes(self.model)
        manifest_path.write_bytes(self.manifest)
        self.assets = {
            "model.gguf": model_path,
            "manifest.json": manifest_path,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publisher(self, fake: FakeTransport) -> GitHubReleasePublisher:
        return GitHubReleasePublisher(
            owner=OWNER,
            repo=REPO,
            tag=TAG,
            token=TOKEN,
            transport=fake,
        )

    def test_creates_uploads_and_proves_immutable_release(self) -> None:
        fake = FakeTransport()

        result = self.publisher(fake).publish(self.assets)

        self.assertFalse(result.already_published)
        self.assertEqual(result.release_id, 41)
        self.assertEqual(
            result.source,
            "https:github.com/microtensor-miner/artifacts/releases/download/r1238",
        )
        self.assertEqual({item.name for item in result.assets}, set(self.assets))
        self.assertEqual(sorted(fake.uploads), ["manifest.json", "model.gguf"])
        self.assertEqual(fake.patch_count, 1)
        self.assertTrue(fake.releases[0]["immutable"])
        self.assertTrue(fake.upload_read_sizes)
        self.assertLessEqual(max(fake.upload_read_sizes), 7)
        self.assertEqual(
            fake.calls[:3],
            [
                ("request", "GET", BASE),
                ("request", "PUT", f"{BASE}/immutable-releases"),
                ("request", "GET", f"{BASE}/immutable-releases"),
            ],
        )

    def test_private_repository_is_refused_before_enablement(self) -> None:
        fake = FakeTransport()
        fake.repo_private = True

        with self.assertRaisesRegex(ReleasePublishError, "not proven public"):
            self.publisher(fake).publish(self.assets)

        self.assertEqual(fake.calls, [("request", "GET", BASE)])
        self.assertFalse(fake.releases)

    def test_immutable_enablement_or_verification_failure_stops_before_release(self) -> None:
        cases = (
            (500, 200, True),
            (204, 404, True),
            (204, 200, False),
        )
        for put_status, get_status, enabled in cases:
            with self.subTest(put_status=put_status, get_status=get_status, enabled=enabled):
                fake = FakeTransport()
                fake.immutable_put_status = put_status
                fake.immutable_get_status = get_status
                fake.immutable_enabled = enabled
                with self.assertRaises(ReleasePublishError):
                    self.publisher(fake).publish(self.assets)
                self.assertFalse(fake.releases)
                self.assertFalse(fake.uploads)
                self.assertEqual(fake.patch_count, 0)

    def test_existing_asset_mismatch_is_never_overwritten(self) -> None:
        fake = FakeTransport()
        wrong = _asset("model.gguf", self.model)
        wrong["size"] += 1
        fake.releases.append(_release(7, draft=True, immutable=False, assets=[wrong]))

        with self.assertRaisesRegex(ReleasePublishError, "metadata mismatch"):
            self.publisher(fake).publish(self.assets)

        self.assertFalse(fake.uploads)
        self.assertEqual(fake.patch_count, 0)
        self.assertEqual(fake.releases[0]["assets"], [wrong])

    def test_uploaded_digest_mismatch_stops_before_publication(self) -> None:
        fake = FakeTransport()
        fake.upload_digest_mismatch = True

        with self.assertRaisesRegex(ReleasePublishError, "digest mismatch"):
            self.publisher(fake).publish(self.assets)

        self.assertEqual(len(fake.uploads), 1)
        self.assertEqual(fake.patch_count, 0)

    def test_existing_published_nonimmutable_release_is_refused(self) -> None:
        fake = FakeTransport()
        fake.releases.append(
            _release(
                9,
                draft=False,
                immutable=False,
                assets=[
                    _asset("model.gguf", self.model),
                    _asset("manifest.json", self.manifest),
                ],
            )
        )

        with self.assertRaisesRegex(ReleasePublishError, "not proven immutable"):
            self.publisher(fake).publish(self.assets)

        self.assertFalse(fake.uploads)
        self.assertEqual(fake.patch_count, 0)

    def test_already_published_immutable_release_is_idempotent(self) -> None:
        fake = FakeTransport()
        fake.releases.append(
            _release(
                11,
                draft=False,
                immutable=True,
                assets=[
                    _asset("model.gguf", self.model),
                    _asset("manifest.json", self.manifest),
                ],
            )
        )

        result = self.publisher(fake).publish(self.assets)

        self.assertTrue(result.already_published)
        self.assertEqual(result.release_id, 11)
        self.assertFalse(fake.uploads)
        self.assertEqual(fake.patch_count, 0)
        self.assertFalse(any(call[1] == "POST" for call in fake.calls))

    def test_recovers_draft_without_reuploading_matching_asset(self) -> None:
        fake = FakeTransport()
        fake.releases.append(
            _release(
                13,
                draft=True,
                immutable=False,
                assets=[_asset("model.gguf", self.model)],
            )
        )

        result = self.publisher(fake).publish(self.assets)

        self.assertFalse(result.already_published)
        self.assertEqual(result.release_id, 13)
        self.assertEqual(fake.uploads, ["manifest.json"])
        self.assertEqual(fake.patch_count, 1)

    def test_published_release_requires_exact_asset_set(self) -> None:
        fake = FakeTransport()
        fake.releases.append(
            _release(
                17,
                draft=False,
                immutable=True,
                assets=[_asset("model.gguf", self.model)],
            )
        )

        with self.assertRaisesRegex(ReleasePublishError, "incomplete"):
            self.publisher(fake).publish(self.assets)

        self.assertFalse(fake.uploads)
        self.assertEqual(fake.patch_count, 0)

    def test_final_published_release_must_report_immutable(self) -> None:
        fake = FakeTransport()
        fake.publish_immutable = False

        with self.assertRaisesRegex(ReleasePublishError, "not proven immutable"):
            self.publisher(fake).publish(self.assets)

        self.assertEqual(fake.patch_count, 1)

    def test_token_is_absent_from_repr_and_transport_errors(self) -> None:
        fake = FakeTransport()
        fake.raise_error = ReleasePublishError(TOKEN)
        publisher = self.publisher(fake)

        with self.assertRaises(ReleasePublishError) as raised:
            publisher.publish(self.assets)

        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn(TOKEN, repr(raised.exception))
        self.assertNotIn(TOKEN, repr(publisher))
        self.assertNotIn(TOKEN, repr(GitHubTransport(TOKEN)))
        self.assertNotIn(TOKEN, repr(ApiResponse(500, {"echo": TOKEN})))

    def test_invalid_coordinates_and_filenames_fail_before_transport(self) -> None:
        fake = FakeTransport()
        with self.assertRaises(ReleasePublishError):
            GitHubReleasePublisher(
                owner="../owner",
                repo=REPO,
                tag=TAG,
                token=TOKEN,
                transport=fake,
            )
        with self.assertRaises(ReleasePublishError):
            self.publisher(fake).publish({"../model.gguf": next(iter(self.assets.values()))})
        self.assertFalse(fake.calls)

    def test_uses_current_immutable_release_api_version(self) -> None:
        transport = GitHubTransport(TOKEN)
        headers = transport._headers()
        self.assertEqual(API_VERSION, "2026-03-10")
        self.assertEqual(headers["X-GitHub-Api-Version"], "2026-03-10")


if __name__ == "__main__":
    unittest.main()
