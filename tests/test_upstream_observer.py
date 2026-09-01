from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from microtensor_miner_controller.state import StateStore
from microtensor_miner_controller.upstream_observer import (
    AUDITED_UPSTREAM_HEAD,
    CANDIDATE_REF,
    GitResult,
    ObservationError,
    UpstreamObserver,
    _changed_files,
    _miner_impact,
    _static_constant,
)

LEGACY_V032_HEAD = "3cc29eb7a3e432b3697eb63e89ccb33e4dc27119"
D77_MANIFEST_BLOB = "816671b4005fe81c657c3b5b77f88ba87c4d0ede"


def _fixture_canonical_hash(body: dict[str, object]) -> str:
    """Inert reimplementation of canonical_hash for this ASCII-only fixture."""

    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _fixture_digest_matches(committed: str, computed: str) -> bool:
    digest = computed.removeprefix("sha256:")
    return digest[: len(committed)] == committed.strip().lower()


def _d77_accepts_digest_branch(
    manifest_body: dict[str, object],
    *,
    commitment_digest: str,
    commitment_source: str,
) -> bool:
    """Inert model of d77's new-digest/legacy-digest match branches."""

    content_body = dict(manifest_body)
    manifest_source = str(content_body.pop("source"))
    if _fixture_digest_matches(commitment_digest, _fixture_canonical_hash(content_body)):
        return True
    if _fixture_digest_matches(commitment_digest, _fixture_canonical_hash(manifest_body)):
        return commitment_source == manifest_source
    return False


def _v032_accepts_digest_branch(
    manifest_body: dict[str, object],
    *,
    commitment_digest: str,
    commitment_source: str,
) -> bool:
    """Inert model of the signed v0.3.2 digest/source match branches."""

    return commitment_source == manifest_body["source"] and _fixture_digest_matches(
        commitment_digest,
        _fixture_canonical_hash(manifest_body),
    )


def _constants(release: str = "0.3.2", provenance: str = "False") -> str:
    return (
        f'RELEASE_VERSION: Final[str] = "{release}"\n'
        'MECHANISM_VERSION: Final[str] = "0.3.0"\n'
        f"PROVENANCE_REQUIRED: Final[bool] = {provenance}\n"
    )


class FakeGit:
    def __init__(
        self,
        advertised: str,
        *,
        changed: str = "",
        cached: str | None = AUDITED_UPSTREAM_HEAD,
        cached_is_ancestor: bool = True,
        constants_source: str | None = None,
    ) -> None:
        self.advertised = advertised
        self.cached = cached
        self.candidate: str | None = None
        self.initial_cached = cached
        self.cached_is_ancestor = cached_is_ancestor
        self.changed = changed
        self.constants_source = constants_source
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, repo: Path, arguments: tuple[str, ...], accepted: frozenset[int]
    ) -> GitResult:
        del repo
        self.calls.append(arguments)
        if arguments == ("remote", "get-url", "origin"):
            return GitResult(0, "https://github.com/microtensor-io/microtensor-subnet.git\n")
        if arguments == ("rev-parse", "HEAD"):
            return GitResult(0, AUDITED_UPSTREAM_HEAD + "\n")
        if arguments == (
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main",
        ):
            if self.cached is None:
                self.assert_allowed(accepted, 128)
                return GitResult(128, "")
            return GitResult(0, self.cached + "\n")
        if arguments == ("rev-parse", "--verify", CANDIDATE_REF):
            if self.candidate is None:
                raise AssertionError("candidate ref was not fetched")
            return GitResult(0, self.candidate + "\n")
        if arguments[0] == "ls-remote":
            return GitResult(0, f"{self.advertised}\trefs/heads/main\n")
        if arguments[0] == "fetch":
            self.candidate = self.advertised
            return GitResult(0, "")
        if arguments[0] == "update-ref":
            if arguments != (
                "update-ref",
                "refs/remotes/origin/main",
                self.advertised,
                self.cached or ("0" * 40),
            ):
                raise AssertionError(f"unexpected update-ref call: {arguments}")
            self.cached = self.advertised
            return GitResult(0, "")
        if arguments[0] == "show":
            if self.constants_source is not None:
                return GitResult(0, self.constants_source)
            return GitResult(
                0,
                _constants("0.3.3", "True")
                if self.advertised != AUDITED_UPSTREAM_HEAD
                else _constants(),
            )
        if arguments[0] == "merge-base":
            is_cached_check = (
                self.initial_cached is not None
                and self.initial_cached != AUDITED_UPSTREAM_HEAD
                and arguments[2] == self.initial_cached
            )
            result = 0 if not is_cached_check or self.cached_is_ancestor else 1
            self.assert_allowed(accepted, result)
            return GitResult(result, "")
        if arguments[0] == "diff":
            return GitResult(0, self.changed)
        if arguments[0] == "rev-list":
            return GitResult(0, "3\n")
        raise AssertionError(f"unexpected git call: {arguments}")

    @staticmethod
    def assert_allowed(accepted: frozenset[int], value: int) -> None:
        if value not in accepted:
            raise AssertionError(f"return code {value} was not accepted")


class UpstreamObserverTests(unittest.TestCase):
    def test_audited_head_is_exact_compatibility_reviewed_commit(self) -> None:
        self.assertEqual(
            AUDITED_UPSTREAM_HEAD,
            "d77adc945de763f8b3b2d71fef8193090ede7001",
        )
        self.assertNotEqual(AUDITED_UPSTREAM_HEAD, LEGACY_V032_HEAD)

    def test_d77_statically_accepts_legacy_digest_only_with_equal_source(self) -> None:
        # This is an inert fixture/reimplementation of the two digest branches in
        # d77's manifest blob. It deliberately does not import, compile, or execute
        # any code from the unsigned upstream commit.
        self.assertEqual(D77_MANIFEST_BLOB, "816671b4005fe81c657c3b5b77f88ba87c4d0ede")
        body: dict[str, object] = {
            "version": "0.3.0",
            "hotkey": "5Hotkey",
            "round_index": 1239,
            "track": "code",
            "hardware_class": "mt-3g",
            "source": "https:github.com/example/artifacts/r1239",
            "artifact_digest": "sha256:" + ("a" * 64),
            "files": [
                {
                    "path": "model.gguf",
                    "digest": "sha256:" + ("b" * 64),
                    "size_bytes": 1024,
                }
            ],
            "load": {"format": "gguf", "entrypoint": "model.gguf"},
            "declared": {
                "size_bytes": 1024,
                "peak_rss_bytes": 2048,
                "p95_latency_ms": 100,
            },
            "system": None,
            "sealed": None,
        }
        legacy_digest = _fixture_canonical_hash(body).removeprefix("sha256:")[:32]
        content_body = dict(body)
        content_body.pop("source")
        content_digest = _fixture_canonical_hash(content_body).removeprefix("sha256:")[:32]
        source = str(body["source"])

        self.assertNotEqual(legacy_digest, content_digest)
        self.assertTrue(
            _d77_accepts_digest_branch(
                body,
                commitment_digest=legacy_digest,
                commitment_source=source,
            )
        )
        self.assertFalse(
            _d77_accepts_digest_branch(
                body,
                commitment_digest=legacy_digest,
                commitment_source="https:github.com/example/artifacts/moved",
            )
        )
        self.assertFalse(
            _v032_accepts_digest_branch(
                body,
                commitment_digest=content_digest,
                commitment_source=source,
            )
        )

    def test_current_audited_head_is_healthy_and_does_not_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            state = StateStore(root / "state")
            git = FakeGit(AUDITED_UPSTREAM_HEAD)
            payload = UpstreamObserver(repo, state, runner=git).poll_once()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["phase"], "current")
        self.assertTrue(payload["observation_succeeded"])
        self.assertIn("origin_observed_at", payload)
        self.assertFalse(payload["review_required"])
        self.assertFalse(any(call[0] == "fetch" for call in git.calls))

    def test_new_origin_is_fetched_and_marks_miner_review_required(self) -> None:
        advertised = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            state = StateStore(root / "state")
            git = FakeGit(
                advertised,
                changed="microtensor/core/constants.py\x00README.md\x00",
            )
            payload = UpstreamObserver(repo, state, runner=git).poll_once()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["phase"], "review_required")
        self.assertEqual(payload["origin_head"], advertised)
        self.assertTrue(payload["fetched"])
        self.assertTrue(payload["miner_impact_review_required"])
        self.assertEqual(payload["commits_since_audit"], 3)
        self.assertTrue(any(call[0] == "fetch" for call in git.calls))
        diff_call = next(call for call in git.calls if call[0] == "diff")
        self.assertIn("-z", diff_call)
        self.assertIn("--no-ext-diff", diff_call)
        self.assertIn("--no-textconv", diff_call)

    def test_impact_classifier_covers_submission_and_evaluation_contracts(self) -> None:
        self.assertTrue(_miner_impact(["microtensor/miner/package.py"]))
        self.assertTrue(_miner_impact(["neurons/miner.py"]))
        self.assertTrue(_miner_impact(["deploy/miner.service"]))
        self.assertTrue(_miner_impact(["scripts/release.py"]))
        self.assertTrue(_miner_impact(["docs/mechanism.md"]))
        self.assertTrue(_miner_impact([".github/workflows/release.yml"]))
        self.assertTrue(_miner_impact(["pyproject.toml"]))
        self.assertFalse(_miner_impact(["README.md"]))

    def test_unexpected_origin_fails_closed_without_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            state = StateStore(root / "state")

            def wrong_origin(
                repo_path: Path, arguments: tuple[str, ...], accepted: frozenset[int]
            ) -> GitResult:
                del repo_path, accepted
                if arguments == ("remote", "get-url", "origin"):
                    return GitResult(
                        0,
                        "https://credential-secret@example.invalid/subnet.git\n",
                    )
                raise AssertionError("observer continued after origin mismatch")

            payload = UpstreamObserver(repo, state, runner=wrong_origin).poll_once()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["phase"], "check_error")
        self.assertIn("does not match", payload["message"])
        self.assertNotIn("credential-secret", payload["message"])

    def test_failure_after_success_clears_stale_observation_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            state = StateStore(root / "state")
            first = UpstreamObserver(
                repo,
                state,
                runner=FakeGit(AUDITED_UPSTREAM_HEAD),
            ).poll_once()
            self.assertIn("origin_head", first)

            def fail(
                repo_path: Path, arguments: tuple[str, ...], accepted: frozenset[int]
            ) -> GitResult:
                del repo_path, arguments, accepted
                raise ObservationError("simulated observation failure")

            failed = UpstreamObserver(repo, state, runner=fail).poll_once()

        self.assertFalse(failed["observation_succeeded"])
        self.assertIn("attempted_at", failed)
        for stale_key in (
            "origin_head",
            "origin_observed_at",
            "review_required",
            "miner_impact_review_required",
            "changed_files",
        ):
            self.assertNotIn(stale_key, failed)

    def test_missing_cached_origin_ref_fetches_exact_main(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            git = FakeGit(AUDITED_UPSTREAM_HEAD, cached=None)
            payload = UpstreamObserver(
                repo,
                StateStore(root / "state"),
                runner=git,
            ).poll_once()

        self.assertTrue(payload["ok"])
        self.assertIsNone(payload["cached_origin_head_before"])
        fetch_call = next(call for call in git.calls if call[0] == "fetch")
        self.assertIn("--no-write-fetch-head", fetch_call)
        self.assertIn("--no-recurse-submodules", fetch_call)
        self.assertEqual(
            fetch_call[-1],
            f"+refs/heads/main:{CANDIDATE_REF}",
        )

    def test_history_rewrite_latch_survives_failure_after_ref_update(self) -> None:
        class FailAfterUpdate(FakeGit):
            def __call__(
                self,
                repo: Path,
                arguments: tuple[str, ...],
                accepted: frozenset[int],
            ) -> GitResult:
                result = super().__call__(repo, arguments, accepted)
                if arguments[0] == "update-ref":
                    raise ObservationError("simulated termination after update-ref")
                return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            state = StateStore(root / "state")
            git = FailAfterUpdate(
                AUDITED_UPSTREAM_HEAD,
                cached="b" * 40,
                cached_is_ancestor=False,
            )
            failed = UpstreamObserver(repo, state, runner=git).poll_once()
            self.assertFalse(failed["ok"])
            self.assertTrue(failed["history_rewrite_latched"])

            recovered = UpstreamObserver(
                repo,
                state,
                runner=FakeGit(AUDITED_UPSTREAM_HEAD),
            ).poll_once()
            self.assertFalse(recovered["ok"])
            self.assertEqual(recovered["phase"], "review_required")
            self.assertTrue(recovered["history_rewrite_latched"])

    def test_cached_history_rewrite_latches_until_status_is_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "subnet"
            repo.mkdir()
            state = StateStore(root / "state")
            rewritten = UpstreamObserver(
                repo,
                state,
                runner=FakeGit(
                    AUDITED_UPSTREAM_HEAD,
                    cached="b" * 40,
                    cached_is_ancestor=False,
                ),
            ).poll_once()
            self.assertTrue(rewritten["history_rewrite_detected"])
            self.assertTrue(rewritten["history_rewrite_latched"])
            self.assertTrue(rewritten["review_required"])

            still_latched = UpstreamObserver(
                repo,
                state,
                runner=FakeGit(AUDITED_UPSTREAM_HEAD),
            ).poll_once()
            self.assertTrue(still_latched["history_rewrite_latched"])
            self.assertTrue(still_latched["review_required"])

            state.status_path.unlink()
            cleared = UpstreamObserver(
                repo,
                state,
                runner=FakeGit(AUDITED_UPSTREAM_HEAD),
            ).poll_once()
            self.assertFalse(cleared["history_rewrite_latched"])
            self.assertFalse(cleared["review_required"])

    def test_constants_are_exact_top_level_literals(self) -> None:
        self.assertEqual(
            _static_constant('RELEASE_VERSION: Final[str] = "0.3.2"\n', "RELEASE_VERSION", str),
            "0.3.2",
        )
        with self.assertRaises(ObservationError):
            _static_constant("RELEASE_VERSION = build_version()\n", "RELEASE_VERSION", str)
        with self.assertRaises(ObservationError):
            _static_constant(
                'if True:\n    RELEASE_VERSION = "0.3.2"\n',
                "RELEASE_VERSION",
                str,
            )
        with self.assertRaises(ObservationError):
            _static_constant(
                'RELEASE_VERSION = "0.3.2"\nRELEASE_VERSION = "0.3.3"\n',
                "RELEASE_VERSION",
                str,
            )

    def test_nul_paths_are_strictly_validated(self) -> None:
        paths, truncated = _changed_files("docs/miner guide.md\x00microtensor/core/x.py\x00")
        self.assertEqual(paths, ("docs/miner guide.md", "microtensor/core/x.py"))
        self.assertFalse(truncated)
        for unsafe in (
            "../secret\x00",
            "/absolute\x00",
            "microtensor//module.py\x00",
            "microtensor/../module.py\x00",
            "microtensor\\module.py\x00",
            "microtensor/line\nfeed.py\x00",
            "not-nul-delimited\n",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ObservationError):
                _changed_files(unsafe)


if __name__ == "__main__":
    unittest.main()
