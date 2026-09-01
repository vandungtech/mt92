from __future__ import annotations

import copy
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import run_code_gguf_diagnostic as launcher


def _identity(raw: bytes) -> dict[str, object]:
    return {"bytes": len(raw), "sha256": launcher._digest_bytes(raw)}


@contextmanager
def _exact_parent_environment() -> object:
    pid_by_descriptor: dict[int, int] = {}

    def synthetic_pidfd_open(pid: int, _flags: int) -> int:
        try:
            descriptor = os.open(
                f"/proc/{pid}",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except FileNotFoundError as exc:
            raise ProcessLookupError(exc.errno, str(exc)) from exc
        pid_by_descriptor[descriptor] = pid
        return descriptor

    def synthetic_pidfd_signal(
        descriptor: int,
        requested_signal: int,
        _siginfo: object,
        _flags: int,
    ) -> None:
        os.kill(pid_by_descriptor[descriptor], requested_signal)

    with (
        mock.patch.dict(os.environ, launcher.EXACT_ENVIRONMENT, clear=True),
        mock.patch.object(os, "pidfd_open", side_effect=synthetic_pidfd_open, create=True),
        mock.patch.object(
            signal,
            "pidfd_send_signal",
            side_effect=synthetic_pidfd_signal,
            create=True,
        ),
    ):
        yield


def _public_git(commit: str) -> dict[str, object]:
    return {
        "origin": launcher.REPOSITORY,
        "normalized_origin": launcher.REPOSITORY,
        "commit": commit,
        "commit_object_verified_locally": True,
        "current_head": commit,
        "commit_ancestor_of_current_head_verified_locally": True,
        "advertised_remote_ref": launcher._ADVERTISED_REMOTE_REF,
        "advertised_remote_ref_head": commit,
        "commit_ancestor_of_advertised_remote_ref_verified_locally": True,
        "commit_blobs_match_worktree_bytes": True,
        "blobs": {},
        "raw_github_readback_performed_by_launcher": False,
    }


def _boundary(task_id: int = 31337) -> dict[str, object]:
    return {
        "schema": "microtensor.code.launcher-child-boundary.v1",
        "status": "ready",
        "parent_environment": {
            "exact": True,
            "keys": sorted(launcher.EXACT_ENVIRONMENT),
            "canonical_json_bytes": launcher.ENVIRONMENT_CANONICAL_BYTES,
            "canonical_json_sha256": launcher.ENVIRONMENT_CANONICAL_DIGEST,
        },
        "task_ids": [task_id],
        "task_count": 1,
        "sigchld_disposition": "SIG_DFL",
        "sigalrm_disposition": "SIG_DFL",
        "realtime_timer_clear": True,
        "blocked_signal_mask_empty": True,
        "child_subreaper_state": 0,
        "proc_children_empty": True,
        "waitpid_echild_verified": True,
    }


def _addendum_payload() -> dict[str, object]:
    source_identity = {"bytes": 10, "sha256": "sha256:" + "1" * 64}
    validator_identity = {"bytes": 20, "sha256": "sha256:" + "2" * 64}
    return {
        "schema": launcher.ADDENDUM_SCHEMA,
        "status": launcher.ADDENDUM_STATUS,
        "public_source": {
            "repository": launcher.REPOSITORY,
            "commit": "3" * 40,
            "raw_readback_verified": True,
            "files": {
                launcher.LAUNCHER_RELATIVE: source_identity,
                launcher.VALIDATOR_RELATIVE: validator_identity,
            },
        },
        "experiment_spec": {
            "path": launcher.SPEC_RELATIVE,
            "bytes": launcher.SPEC_BYTES,
            "sha256": launcher.SPEC_DIGEST,
        },
        "interpreter": {
            "path": str(launcher.INTERPRETER_PATH),
            "resolved_path": str(launcher.INTERPRETER_RESOLVED),
            "bytes": launcher.INTERPRETER_BYTES,
            "sha256": launcher.INTERPRETER_DIGEST,
        },
        "execution": {
            "cwd": str(launcher.SOURCE_ROOT),
            "shell": False,
            "stdin": "/dev/null",
            "umask": launcher.UMASK_TEXT,
            "close_fds": True,
            "new_session": True,
            "parent_death_signal": "SIGKILL",
            "timeout_seconds": launcher.TIMEOUT_SECONDS,
            "environment_exact": copy.deepcopy(launcher.EXACT_ENVIRONMENT),
            "environment_canonical_json_bytes": launcher.ENVIRONMENT_CANONICAL_BYTES,
            "environment_canonical_json_sha256": launcher.ENVIRONMENT_CANONICAL_DIGEST,
            "parent_environment_exact": True,
            "report_root": str(launcher.REPORT_ROOT),
            "post_exec_inspection": copy.deepcopy(launcher.POST_EXEC_INSPECTION),
            "containment": copy.deepcopy(launcher.CONTAINMENT_CONTRACT),
            "repeat_policy": copy.deepcopy(launcher.REPEAT_POLICY),
        },
        "preflight": {
            "validator_path": launcher.VALIDATOR_RELATIVE,
            "validator_mode": "in_process_static_only",
            "model_engine_construction_permitted": False,
            "required_checks": list(launcher.PREFLIGHT_CHECKS),
            "source_commit": launcher.SOURCE_COMMIT,
            "artifact_tree_sha256": launcher.ARTIFACT_TREE_DIGEST,
            "artifact_entrypoint_bytes": launcher.ARTIFACT_ENTRYPOINT_BYTES,
            "artifact_entrypoint_sha256": launcher.ARTIFACT_ENTRYPOINT_DIGEST,
        },
        "invocations": list(launcher.expected_invocations()),
    }


def _write_addendum(root: Path, payload: dict[str, object]) -> Path:
    path = root / launcher.ADDENDUM_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(launcher._pretty_json_bytes(payload))
    return path


def _synthetic_contract(root: Path) -> launcher.LaunchContract:
    report_root = root / "base" / "receipts" / "run"
    outputs = tuple(root / "outputs" / repeat for repeat in launcher.REPEATS)
    invocations: list[launcher.Invocation] = []
    for repeat, output in zip(launcher.REPEATS, outputs, strict=True):
        argv = (sys.executable, "-c", "raise SystemExit(0)", "--out", str(output))
        raw = launcher._canonical_json_bytes(list(argv))
        invocations.append(
            launcher.Invocation(
                repeat=repeat,
                argv=argv,
                output_root=output,
                argv_canonical_json_bytes=len(raw),
                argv_canonical_json_sha256=launcher._digest_bytes(raw),
            )
        )
    raw = b'{"synthetic":true}\n'
    addendum = root / "synthetic-addendum.json"
    addendum.write_bytes(raw)
    return launcher.LaunchContract(
        path=addendum,
        raw=raw,
        digest=launcher._digest_bytes(raw),
        public_commit="4" * 40,
        public_git=_public_git("4" * 40),
        repository_root=root,
        experiment_spec=root / "spec.json",
        validator_path=root / "validator.py",
        interpreter_path=Path(sys.executable),
        interpreter_resolved=Path(sys.executable).resolve(),
        environment=dict(launcher.EXACT_ENVIRONMENT),
        report_root=report_root,
        invocations=(invocations[0], invocations[1], invocations[2]),
    )


def _outcome(
    invocation: launcher.Invocation | None = None,
    *,
    returncode: int = 0,
    timed_out: bool = False,
) -> launcher.ProcessOutcome:
    argv_digest = (
        invocation.argv_canonical_json_sha256 if invocation is not None else "sha256:" + "0" * 64
    )
    boundary = _boundary(os.getpid())
    containment = {
        **launcher.CONTAINMENT_CONTRACT,
        "pre_fork_boundary": boundary,
        "prior_subreaper_state": 0,
        "subreaper_enabled": True,
        "direct_pid": 4242,
        "direct_sigkill_attempted": timed_out,
        "direct_terminal_status_observed": True,
        "direct_returncode": returncode,
        "process_group_sigkill_attempted": True,
        "process_group_absent": True,
        "descendants_observed": False,
        "observed_descendant_pids": [],
        "pidfd_signaled_descendant_pids": [],
        "terminal_waitpid_echild_verified": True,
        "subreaper_restored": True,
    }
    return launcher.ProcessOutcome(
        pid=4242,
        returncode=returncode,
        timed_out=timed_out,
        inspection={
            **launcher.POST_EXEC_INSPECTION,
            "pid": 4242,
            "ptrace_stop_signal": 5,
            "cmdline": {"matched": True, "canonical_json_sha256": argv_digest},
            "environ": {
                "matched": True,
                "keys": sorted(launcher.EXACT_ENVIRONMENT),
                "canonical_json_bytes": launcher.ENVIRONMENT_CANONICAL_BYTES,
                "canonical_json_sha256": launcher.ENVIRONMENT_CANONICAL_DIGEST,
            },
            "cwd": {"matched": True, "path": str(launcher.SOURCE_ROOT)},
            "exe": {"matched": True, "path": str(Path(sys.executable).resolve())},
            "fd0": {"matched_dev_null": True, "proc_link": "/dev/null"},
            "open_fds": [0, 1, 2],
            "umask": launcher.UMASK_TEXT,
            "process_group": 4242,
        },
        containment=containment,
        started_at_utc="2026-09-01T00:00:00.000000Z",
        started_at_unix_ns=1,
        finished_at_utc="2026-09-01T00:00:01.000000Z",
        finished_at_unix_ns=1_000_000_001,
    )


def _preflight(_: launcher.LaunchContract) -> launcher.Preflight:
    digest = "sha256:" + "a" * 64
    return launcher.Preflight(
        report={
            "source": {
                "root": str(launcher.SOURCE_ROOT),
                "commit": launcher.SOURCE_COMMIT,
                "status_empty": True,
            },
            "interpreter": {
                "path": sys.executable,
                "resolved_path": str(Path(sys.executable).resolve()),
                "bytes": 1,
                "sha256": digest,
            },
            "experiment_spec": {
                "bytes": launcher.SPEC_BYTES,
                "sha256": launcher.SPEC_DIGEST,
            },
            "artifact": {
                "tree_sha256": launcher.ARTIFACT_TREE_DIGEST,
                "entrypoint_bytes": launcher.ARTIFACT_ENTRYPOINT_BYTES,
                "entrypoint_sha256": launcher.ARTIFACT_ENTRYPOINT_DIGEST,
            },
            "configuration_sha256": digest,
            "evaluation_dataset_sha256": digest,
            "training_lineage_sha256": digest,
            "runtime_sha256": digest,
            "conversion_replays_sha256": digest,
            "process_escape_scan": {
                "schema": "microtensor.code.python-process-escape-scan.v1",
                "status": "passed",
                "files": [
                    {
                        "label": "synthetic",
                        "path": "/synthetic.py",
                        "bytes": 1,
                        "sha256": digest,
                    }
                ],
                "forbidden_modules": sorted(launcher._FORBIDDEN_PROCESS_MODULES),
                "forbidden_os_primitives": sorted(launcher._FORBIDDEN_OS_PRIMITIVES),
                "python_process_or_session_primitives_found": False,
                "native_extensions_or_transitive_imports_attested": False,
                "scope": launcher.CONTAINMENT_SCOPE,
            },
            "checks": list(launcher.PREFLIGHT_CHECKS),
            "model_engine_constructed": False,
        },
        validator=SimpleNamespace(name="synthetic-static-validator"),
    )


def _validation(
    _validator: object,
    _contract: launcher.LaunchContract,
    repeat: str,
) -> tuple[dict[str, object], str]:
    completed = launcher.REPEATS.index(repeat) + 1
    complete = completed == len(launcher.REPEATS)
    report: dict[str, object] = {
        "schema": "microtensor.code.gguf-diagnostic-validation.v1",
        "status": "validated" if complete else "partially_validated",
        "through": repeat,
        "aggregate": {
            "validated_repeat_hard_gates_passed": True,
            "all_declared_local_gates_passed": complete,
        },
        "claim": {"remaining_local_repeats": list(launcher.REPEATS[completed:])},
    }
    return report, launcher._digest_bytes(launcher._canonical_json_bytes(report))


def _stalled_pre_exec_child(
    _invocation: launcher.Invocation,
    _contract: launcher.LaunchContract,
    error_read: int,
    _error_write: int,
    _parent_pid: int,
    _library: object,
) -> None:
    os.close(error_read)
    while True:
        signal.pause()


def _inspection_refusal(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise launcher.LaunchRefused("synthetic post-exec inspection failure")


def _slow_inspection(*_args: object, **_kwargs: object) -> dict[str, object]:
    time.sleep(10)
    return {}


class ExactContractTests(unittest.TestCase):
    def test_environment_and_all_three_argv_bindings_are_exact(self) -> None:
        environment_raw = launcher._canonical_json_bytes(launcher.EXACT_ENVIRONMENT)
        self.assertEqual(len(launcher.EXACT_ENVIRONMENT), 16)
        self.assertEqual(len(environment_raw), 429)
        self.assertEqual(
            launcher._digest_bytes(environment_raw),
            "sha256:9103f1e3ef395e681510f8044ab3f4861352748c8ff8efb928266a0b81a7ce94",
        )
        records = launcher.expected_invocations()
        self.assertEqual([record["repeat"] for record in records], ["r1", "r2", "r3"])
        self.assertEqual([len(record["argv"]) for record in records], [29, 29, 29])
        self.assertEqual(
            [record["argv_canonical_json_bytes"] for record in records],
            [1090, 1090, 1090],
        )
        self.assertEqual(
            [record["argv_canonical_json_sha256"] for record in records],
            [
                "sha256:a5d63356b3c097b13d114a76c9a73b8ea74ff5b6fcccb7401d9f346160849737",
                "sha256:5821cff6ac1f8bb361ce8fc35641ff30e8a669fed788f3a0d763f7c5a476563c",
                "sha256:d0a4c6cbb0a2632d02bf06f7813c5f64d76260bd4b423abc94ecc7ddf2862a98",
            ],
        )
        self.assertEqual(
            [record["output_root"] for record in records],
            [str(path) for path in launcher.OUTPUT_ROOTS],
        )

    def test_strict_json_rejects_duplicates_nonfinite_and_bad_utf8(self) -> None:
        cases = (
            (b'{"a":1,"a":2}', "repeats JSON key"),
            (b'{"a":NaN}', "non-finite"),
            (b'{"a":1e999}', "non-finite"),
            (b'{"a":"\\ud800"}', "lone surrogate"),
            (b'{"a":' + b"9" * 5000 + b"}", "strict UTF-8 JSON"),
            (b'"\xff"', "strict UTF-8"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(launcher.LaunchRefused, message):
                launcher._strict_json(raw, "synthetic")

    def test_addendum_accepts_only_canonical_exact_contract_and_public_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _addendum_payload()
            path = _write_addendum(root, payload)
            identities = {
                str(root / launcher.LAUNCHER_RELATIVE): launcher.FileIdentity(
                    10, "sha256:" + "1" * 64
                ),
                str(root / launcher.VALIDATOR_RELATIVE): launcher.FileIdentity(
                    20, "sha256:" + "2" * 64
                ),
                str(root / launcher.SPEC_RELATIVE): launcher.FileIdentity(
                    launcher.SPEC_BYTES, launcher.SPEC_DIGEST
                ),
            }

            def fake_identity(candidate: Path, _label: str, **_kwargs: object) -> object:
                return identities[str(candidate)]

            with (
                mock.patch.object(launcher, "_repository_root", return_value=root),
                mock.patch.object(launcher, "_file_identity", side_effect=fake_identity),
                mock.patch.object(
                    launcher,
                    "_validate_public_git_binding",
                    return_value=_public_git("3" * 40),
                ),
            ):
                contract = launcher._load_contract(path)
                self.assertEqual(contract.environment, launcher.EXACT_ENVIRONMENT)
                self.assertFalse(contract.public_git["raw_github_readback_performed_by_launcher"])

                mutations = (
                    lambda item: item["execution"]["environment_exact"].update(
                        {"INHERITED_SECRET": "forbidden"}
                    ),
                    lambda item: item["invocations"][0]["argv"].append("--undeclared"),
                    lambda item: item["public_source"].update({"raw_readback_verified": False}),
                    lambda item: item["execution"].update({"timeout_seconds": 901}),
                )
                for mutate in mutations:
                    changed = copy.deepcopy(payload)
                    mutate(changed)
                    path.write_bytes(launcher._pretty_json_bytes(changed))
                    with self.assertRaises(launcher.LaunchRefused):
                        launcher._load_contract(path)

    def test_noncanonical_pretty_addendum_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _addendum_payload()
            path = _write_addendum(root, payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with (
                mock.patch.object(launcher, "_repository_root", return_value=root),
                self.assertRaisesRegex(launcher.LaunchRefused, "canonical sorted JSON"),
            ):
                launcher._load_contract(path)

    def test_public_git_binding_checks_origin_ancestry_and_exact_commit_blobs(self) -> None:
        commit = "3" * 40
        head = "4" * 40
        remote_head = "5" * 40
        files = {
            launcher.LAUNCHER_RELATIVE: b"launcher\n",
            launcher.VALIDATOR_RELATIVE: b"validator\n",
        }
        declared = {
            relative: launcher.FileIdentity(len(raw), launcher._digest_bytes(raw))
            for relative, raw in files.items()
        }

        def git_output(
            _root: Path,
            arguments: object,
            _label: str,
            **_kwargs: object,
        ) -> bytes:
            values = tuple(arguments)
            if values == ("remote", "get-url", "origin"):
                return (launcher.REPOSITORY + ".git\n").encode()
            if values == ("cat-file", "-t", commit):
                return b"commit\n"
            if values == ("rev-parse", "--verify", "HEAD^{commit}"):
                return f"{head}\n".encode()
            if values == (
                "rev-parse",
                "--verify",
                f"{launcher._ADVERTISED_REMOTE_REF}^{{commit}}",
            ):
                return f"{remote_head}\n".encode()
            if values[:2] == ("merge-base", "--is-ancestor"):
                return b""
            revision = str(values[-1])
            relative = revision.split(":", 1)[1]
            if values[:2] == ("cat-file", "-s"):
                return f"{len(files[relative])}\n".encode()
            if values[0] == "show":
                return files[relative]
            raise AssertionError(values)

        with mock.patch.object(launcher, "_git_output", side_effect=git_output):
            evidence = launcher._validate_public_git_binding(Path("/synthetic"), commit, declared)
        self.assertTrue(evidence["commit_ancestor_of_current_head_verified_locally"])
        self.assertTrue(evidence["commit_ancestor_of_advertised_remote_ref_verified_locally"])
        self.assertFalse(evidence["raw_github_readback_performed_by_launcher"])

        broken_files = dict(files)
        broken_files[launcher.LAUNCHER_RELATIVE] = b"changed!!\n"
        files = broken_files
        with (
            mock.patch.object(launcher, "_git_output", side_effect=git_output),
            self.assertRaisesRegex(launcher.LaunchRefused, "byte count changed"),
        ):
            launcher._validate_public_git_binding(Path("/synthetic"), commit, declared)

        def nonancestor(
            root: Path,
            arguments: object,
            label: str,
            **kwargs: object,
        ) -> bytes:
            if tuple(arguments) == ("merge-base", "--is-ancestor", commit, head):
                raise launcher.LaunchRefused("public source HEAD ancestry Git inspection refused")
            return git_output(root, arguments, label, **kwargs)

        files = {
            launcher.LAUNCHER_RELATIVE: b"launcher\n",
            launcher.VALIDATOR_RELATIVE: b"validator\n",
        }
        with (
            mock.patch.object(launcher, "_git_output", side_effect=nonancestor),
            self.assertRaisesRegex(launcher.LaunchRefused, "HEAD ancestry"),
        ):
            launcher._validate_public_git_binding(Path("/synthetic"), commit, declared)


class OnceOnlyStateMachineTests(unittest.TestCase):
    def _patches(self, root: Path, contract: launcher.LaunchContract) -> object:
        return (
            mock.patch.object(launcher, "REPORT_BASE", root / "base"),
            mock.patch.object(launcher, "_load_contract", return_value=contract),
            mock.patch.object(
                launcher,
                "_git_source_identity",
                return_value={
                    "root": str(launcher.SOURCE_ROOT),
                    "commit": launcher.SOURCE_COMMIT,
                    "status_empty": True,
                },
            ),
            mock.patch.object(
                launcher,
                "_signed_interpreter_identity",
                return_value={
                    "path": str(contract.interpreter_path),
                    "resolved_path": str(contract.interpreter_resolved),
                    "bytes": 1,
                    "sha256": "sha256:" + "a" * 64,
                },
            ),
            mock.patch.object(
                launcher,
                "_load_validator",
                return_value=SimpleNamespace(name="post-static-validator"),
            ),
        )

    def test_success_is_validated_and_r2_requires_revalidation_of_r1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)
            calls: list[str] = []

            def process(invocation: launcher.Invocation, _contract: object) -> object:
                invocation.output_root.mkdir()
                return _outcome(invocation)

            def validation(validator: object, active: object, repeat: str) -> object:
                calls.append(repeat)
                return _validation(validator, active, repeat)

            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                first = launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=validation,
                )
                second = launcher._launch_repeat(
                    contract,
                    "r2",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=validation,
                )

            self.assertEqual(first["status"], "partially_validated")
            self.assertEqual(second["status"], "partially_validated")
            self.assertEqual(calls, ["r1", "r1", "r2"])
            self.assertEqual(
                json.loads((contract.report_root / "r2-launch-receipt.json").read_text())["status"],
                "partially_validated",
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in contract.report_root.iterdir())
            )

    def test_unexpected_report_entry_is_refused_before_preflight_or_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            contract = _synthetic_contract(root)
            with (
                _exact_parent_environment(),
                mock.patch.object(launcher, "REPORT_BASE", root / "base"),
                mock.patch.object(launcher, "_load_contract", return_value=contract),
            ):
                launcher._ensure_report_root(contract.report_root, root / "base")
                (contract.report_root / "undeclared.txt").write_text("unexpected", encoding="utf-8")
                preflight = mock.Mock(side_effect=AssertionError("must fail before preflight"))
                with self.assertRaisesRegex(launcher.LaunchRefused, "inventory changed"):
                    launcher._launch_repeat(contract, "r1", preflight_runner=preflight)
            preflight.assert_not_called()

    def test_timeout_consumes_repeat_writes_rejection_and_forbids_retry_or_r2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)
            runner = mock.Mock(return_value=_outcome(returncode=-9, timed_out=True))
            with (
                _exact_parent_environment(),
                mock.patch.object(launcher, "REPORT_BASE", root / "base"),
                mock.patch.object(launcher, "_load_contract", return_value=contract),
            ):
                with self.assertRaisesRegex(launcher.LaunchRefused, "900s timeout"):
                    launcher._launch_repeat(
                        contract,
                        "r1",
                        process_runner=runner,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )
                receipt = json.loads(
                    (contract.report_root / "r1-launch-receipt.json").read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["status"], "rejected")
                self.assertEqual(receipt["outcome"], "permanently_rejected")
                self.assertTrue(receipt["process"]["timed_out"])
                with self.assertRaisesRegex(launcher.LaunchRefused, "inventory changed"):
                    launcher._launch_repeat(
                        contract,
                        "r1",
                        process_runner=runner,
                        preflight_runner=_preflight,
                    )
                with self.assertRaisesRegex(launcher.LaunchRefused, "prior diagnostic root"):
                    launcher._launch_repeat(
                        contract,
                        "r2",
                        process_runner=runner,
                        preflight_runner=_preflight,
                    )
            self.assertEqual(runner.call_count, 1)

    def test_nonzero_exit_is_permanent_and_never_invokes_static_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)
            validator = mock.Mock(side_effect=AssertionError("must not validate failed output"))
            with (
                _exact_parent_environment(),
                mock.patch.object(launcher, "REPORT_BASE", root / "base"),
                mock.patch.object(launcher, "_load_contract", return_value=contract),
                self.assertRaisesRegex(launcher.LaunchRefused, "exited 9"),
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=lambda *_: _outcome(returncode=9),
                    preflight_runner=_preflight,
                    validation_runner=validator,
                )
            validator.assert_not_called()

    def test_r4_is_refused_before_any_namespace_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _synthetic_contract(root)
            with self.assertRaisesRegex(launcher.LaunchRefused, "no r4"):
                launcher._launch_repeat(contract, "r4")
            self.assertFalse(contract.report_root.exists())

    def test_parent_environment_refusal_precedes_namespace_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            with (
                mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True),
                self.assertRaisesRegex(launcher.LaunchRefused, "exactly the declared 16-key"),
            ):
                launcher._launch_repeat(contract, "r1")
            self.assertFalse(contract.report_root.exists())

    @unittest.skipUnless(
        sys.platform == "linux" and hasattr(signal, "pthread_sigmask"), "Linux only"
    )
    def test_blocked_signal_mask_is_refused_before_namespace_mutation_or_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            try:
                with (
                    _exact_parent_environment(),
                    mock.patch.object(launcher.os, "fork") as fork,
                    self.assertRaisesRegex(launcher.LaunchRefused, "blocked-signal mask"),
                ):
                    launcher._launch_repeat(contract, "r1")
                fork.assert_not_called()
                self.assertFalse(contract.report_root.exists())
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/status").is_file(), "Linux only"
    )
    def test_existing_subreaper_is_refused_before_namespace_mutation_or_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            library = launcher._libc()
            previous_state = launcher._prctl_child_subreaper(library)
            launcher._set_child_subreaper(library, 1)
            try:
                with (
                    _exact_parent_environment(),
                    mock.patch.object(launcher.os, "fork") as fork,
                    self.assertRaisesRegex(launcher.LaunchRefused, "child-subreaper state"),
                ):
                    launcher._launch_repeat(contract, "r1")
                fork.assert_not_called()
                self.assertFalse(contract.report_root.exists())
            finally:
                launcher._set_child_subreaper(library, previous_state)

    def test_fresh_previous_validation_digest_must_match_prior_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)

            def process(invocation: launcher.Invocation, _contract: object) -> object:
                invocation.output_root.mkdir()
                return _outcome(invocation)

            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=_validation,
                )

                def altered_validation(
                    validator: object,
                    active: launcher.LaunchContract,
                    repeat: str,
                ) -> tuple[dict[str, object], str]:
                    report, _ = _validation(validator, active, repeat)
                    return report, "sha256:" + "f" * 64

                with self.assertRaisesRegex(launcher.LaunchRefused, "digest differs"):
                    launcher._launch_repeat(
                        contract,
                        "r2",
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=altered_validation,
                    )
            self.assertFalse((contract.report_root / "r2-attempt.json").exists())

    def test_fresh_preflight_digest_must_match_every_prior_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)

            def create_output(invocation: launcher.Invocation, _contract: object) -> object:
                invocation.output_root.mkdir()
                return _outcome(invocation)

            process = mock.Mock(side_effect=create_output)
            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=_validation,
                )
                receipt_path = contract.report_root / "r1-launch-receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["preflight"]["configuration_sha256"] = "sha256:" + "f" * 64
                receipt_path.write_bytes(launcher._canonical_json_bytes(receipt) + b"\n")
                process.reset_mock()

                with self.assertRaisesRegex(launcher.LaunchRefused, "preflight digest differs"):
                    launcher._launch_repeat(
                        contract,
                        "r2",
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )
            process.assert_not_called()
            self.assertFalse((contract.report_root / "r2-attempt.json").exists())

    def test_contract_change_during_preflight_is_refused_before_marker_or_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)
            changed_raw = b'{"changed":true}\n'
            changed = replace(
                contract,
                raw=changed_raw,
                digest=launcher._digest_bytes(changed_raw),
            )
            process = mock.Mock(side_effect=AssertionError("process must not run"))
            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                mock.patch.object(launcher, "_load_contract", return_value=changed),
                patches[2],
                patches[3],
                patches[4],
                self.assertRaisesRegex(launcher.LaunchRefused, "contract changed"),
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=_validation,
                )
            process.assert_not_called()
            self.assertFalse((contract.report_root / "r1-attempt.json").exists())

    def test_r3_recomputes_every_prior_validation_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)

            def create_output(invocation: launcher.Invocation, _contract: object) -> object:
                invocation.output_root.mkdir()
                return _outcome(invocation)

            process = mock.Mock(side_effect=create_output)
            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                for repeat in ("r1", "r2"):
                    launcher._launch_repeat(
                        contract,
                        repeat,
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )

                forged = "sha256:" + "e" * 64
                r1_path = contract.report_root / "r1-launch-receipt.json"
                r1_receipt = json.loads(r1_path.read_text(encoding="utf-8"))
                r1_receipt["validation"]["report_sha256"] = forged
                r1_path.write_bytes(launcher._canonical_json_bytes(r1_receipt) + b"\n")
                r2_path = contract.report_root / "r2-launch-receipt.json"
                r2_receipt = json.loads(r2_path.read_text(encoding="utf-8"))
                r2_receipt["previous_validation"]["report_sha256"] = forged
                r2_path.write_bytes(launcher._canonical_json_bytes(r2_receipt) + b"\n")
                process.reset_mock()

                with self.assertRaisesRegex(
                    launcher.LaunchRefused,
                    "fresh r1 validation digest differs",
                ):
                    launcher._launch_repeat(
                        contract,
                        "r3",
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )
            process.assert_not_called()
            self.assertFalse((contract.report_root / "r3-attempt.json").exists())

    def test_prior_receipt_attempt_identity_is_bound_to_marker_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)

            def process(invocation: launcher.Invocation, _contract: object) -> object:
                invocation.output_root.mkdir()
                return _outcome(invocation)

            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=_validation,
                )
                receipt_path = contract.report_root / "r1-launch-receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["attempt"]["sha256"] = "sha256:" + "0" * 64
                receipt_path.write_bytes(launcher._canonical_json_bytes(receipt) + b"\n")
                with self.assertRaisesRegex(launcher.LaunchRefused, "attempt marker"):
                    launcher._launch_repeat(
                        contract,
                        "r2",
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )
            self.assertFalse((contract.report_root / "r2-attempt.json").exists())

    def test_r2_previous_validation_digest_is_chained_to_r1_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)

            def process(invocation: launcher.Invocation, _contract: object) -> object:
                invocation.output_root.mkdir()
                return _outcome(invocation)

            patches = self._patches(root, contract)
            with (
                _exact_parent_environment(),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
            ):
                for repeat in ("r1", "r2"):
                    launcher._launch_repeat(
                        contract,
                        repeat,
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )
                receipt_path = contract.report_root / "r2-launch-receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["previous_validation"]["report_sha256"] = "sha256:" + "0" * 64
                receipt_path.write_bytes(launcher._canonical_json_bytes(receipt) + b"\n")
                with self.assertRaisesRegex(
                    launcher.LaunchRefused,
                    "previous validation evidence changed",
                ):
                    launcher._launch_repeat(
                        contract,
                        "r3",
                        process_runner=process,
                        preflight_runner=_preflight,
                        validation_runner=_validation,
                    )
            self.assertFalse((contract.report_root / "r3-attempt.json").exists())


class DirectExecBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/status").is_file(), "Linux only"
    )
    def test_synthetic_direct_exec_has_exact_held_boundary_and_no_inherited_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _synthetic_contract(root)
            argv = (sys.executable, "-c", "raise SystemExit(0)")
            raw = launcher._canonical_json_bytes(list(argv))
            invocation = launcher.Invocation(
                repeat="r1",
                argv=argv,
                output_root=root / "unused",
                argv_canonical_json_bytes=len(raw),
                argv_canonical_json_sha256=launcher._digest_bytes(raw),
            )
            contract = launcher.LaunchContract(
                **{
                    **contract.__dict__,
                    "interpreter_path": Path(sys.executable),
                    "interpreter_resolved": Path(sys.executable).resolve(),
                    "invocations": (invocation, contract.invocations[1], contract.invocations[2]),
                }
            )

            with _exact_parent_environment():
                outcome = launcher._run_traced_process(invocation, contract, timeout_seconds=5)

            self.assertEqual(outcome.returncode, 0)
            self.assertFalse(outcome.timed_out)
            self.assertEqual(outcome.inspection["open_fds"], [0, 1, 2])
            self.assertEqual(outcome.inspection["umask"], "0077")
            self.assertEqual(
                outcome.inspection["environ"]["keys"], sorted(launcher.EXACT_ENVIRONMENT)
            )
            self.assertFalse(outcome.inspection["subsequent_runtime_attested"])
            self.assertIn(
                "does not attest later evaluator or model state",
                outcome.inspection["evidence_scope"],
            )

    def test_proc_reader_is_bounded_and_never_uses_path_read_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic-proc-file"
            path.write_bytes(b"12345")
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded")):
                self.assertEqual(launcher._read_proc_file(path, "synthetic", maximum=5), b"12345")
                with self.assertRaisesRegex(launcher.LaunchRefused, "byte ceiling"):
                    launcher._read_proc_file(path, "synthetic", maximum=4)

    def test_extra_environment_is_refused_before_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            contract.environment["INHERITED_SECRET"] = "synthetic-value"  # noqa: S105
            with (
                mock.patch.object(launcher.os, "fork") as fork,
                self.assertRaisesRegex(launcher.LaunchRefused, "environment changed"),
            ):
                launcher._run_traced_process(contract.invocations[0], contract, timeout_seconds=1)
            fork.assert_not_called()

    def test_same_key_environment_value_mutation_is_refused_before_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            contract.environment["WANDB_MODE"] = "online"
            with (
                mock.patch.object(launcher.os, "fork") as fork,
                self.assertRaisesRegex(launcher.LaunchRefused, "environment changed"),
            ):
                launcher._run_traced_process(contract.invocations[0], contract, timeout_seconds=1)
            fork.assert_not_called()

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/status").is_file(), "Linux only"
    )
    def test_pre_exec_timeout_is_bounded_and_direct_pid_is_terminally_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            started = time.monotonic()
            with (
                _exact_parent_environment(),
                self.assertRaises(launcher.LaunchProcessRefused) as raised,
            ):
                launcher._run_traced_process(
                    contract.invocations[0],
                    contract,
                    timeout_seconds=1,
                    _child_target=_stalled_pre_exec_child,
                )
            self.assertLess(time.monotonic() - started, 3)
            process = raised.exception.process
            self.assertTrue(process["started"])
            self.assertTrue(process["timed_out"])
            self.assertEqual(process["stage"], "pre_exec_timeout_cleanup")
            self.assertEqual(process["returncode"], -signal.SIGKILL)
            self.assertTrue(raised.exception.containment["terminal_waitpid_echild_verified"])
            with self.assertRaises(ProcessLookupError):
                os.kill(process["pid"], 0)

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/status").is_file(), "Linux only"
    )
    def test_held_child_inspection_is_interrupted_by_the_shared_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _synthetic_contract(Path(temporary))
            started = time.monotonic()
            with (
                _exact_parent_environment(),
                self.assertRaises(launcher.LaunchProcessRefused) as raised,
            ):
                launcher._run_traced_process(
                    contract.invocations[0],
                    contract,
                    timeout_seconds=1,
                    _inspection_runner=_slow_inspection,
                )
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(
                raised.exception.process["stage"],
                "exec_stop_inspection_timeout_cleanup",
            )
            self.assertTrue(raised.exception.process["timed_out"])
            self.assertTrue(raised.exception.containment["terminal_waitpid_echild_verified"])

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/status").is_file(), "Linux only"
    )
    def test_escaped_descendant_is_pidfd_killed_reaped_and_permanently_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _synthetic_contract(root)
            sentinel = root / "escaped-grandchild-wrote.txt"
            source = (
                "import os, sys, time\n"
                "child = os.fork()\n"
                "if child == 0:\n"
                "    os.setsid()\n"
                "    grandchild = os.fork()\n"
                "    if grandchild > 0:\n"
                "        os._exit(0)\n"
                "    time.sleep(1.2)\n"
                "    with open(sys.argv[1], 'wb') as stream:\n"
                "        stream.write(b'escaped')\n"
                "    os._exit(0)\n"
                "os.waitpid(child, 0)\n"
                "raise SystemExit(0)\n"
            )
            argv = (sys.executable, "-c", source, str(sentinel))
            raw = launcher._canonical_json_bytes(list(argv))
            invocation = launcher.Invocation(
                repeat="r1",
                argv=argv,
                output_root=root / "unused",
                argv_canonical_json_bytes=len(raw),
                argv_canonical_json_sha256=launcher._digest_bytes(raw),
            )
            contract = launcher.LaunchContract(
                **{
                    **contract.__dict__,
                    "invocations": (invocation, contract.invocations[1], contract.invocations[2]),
                }
            )
            with (
                _exact_parent_environment(),
                self.assertRaisesRegex(
                    launcher.LaunchProcessRefused,
                    "created an observed descendant",
                ) as raised,
            ):
                launcher._run_traced_process(invocation, contract, timeout_seconds=5)
            containment = raised.exception.containment
            self.assertTrue(containment["descendants_observed"])
            self.assertTrue(containment["observed_descendant_pids"])
            self.assertTrue(containment["pidfd_signaled_descendant_pids"])
            self.assertTrue(containment["process_group_absent"])
            self.assertTrue(containment["terminal_waitpid_echild_verified"])
            for pid in containment["observed_descendant_pids"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            self.assertFalse(sentinel.exists())
            time.sleep(1.5)
            self.assertFalse(sentinel.exists())

    def test_inspection_failure_receipt_keeps_started_pid_and_cleanup_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)

            def process(
                invocation: launcher.Invocation,
                active: launcher.LaunchContract,
            ) -> launcher.ProcessOutcome:
                return launcher._run_traced_process(
                    invocation,
                    active,
                    timeout_seconds=5,
                    _inspection_runner=_inspection_refusal,
                )

            with (
                _exact_parent_environment(),
                mock.patch.object(launcher, "REPORT_BASE", root / "base"),
                mock.patch.object(launcher, "_load_contract", return_value=contract),
                self.assertRaisesRegex(launcher.LaunchRefused, "inspection failure"),
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=_validation,
                )
            receipt = json.loads(
                (contract.report_root / "r1-launch-receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(receipt["process"]["started"])
            self.assertGreater(receipt["process"]["pid"], 0)
            self.assertEqual(receipt["process"]["stage"], "exec_stop_inspection_failed")
            self.assertIsNone(receipt["inspection"])
            self.assertTrue(receipt["containment"]["process"]["terminal_waitpid_echild_verified"])
            self.assertFalse(receipt["public_source"]["launcher_raw_github_readback_performed"])

    def test_cleanup_failure_rejects_before_validation_and_is_truthful_in_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir(mode=0o700)
            (root / "outputs").mkdir()
            contract = _synthetic_contract(root)
            validator = mock.Mock(side_effect=AssertionError("must not validate"))

            def process(
                invocation: launcher.Invocation,
                active: launcher.LaunchContract,
            ) -> launcher.ProcessOutcome:
                return launcher._run_traced_process(invocation, active, timeout_seconds=5)

            with (
                _exact_parent_environment(),
                mock.patch.object(launcher, "REPORT_BASE", root / "base"),
                mock.patch.object(launcher, "_load_contract", return_value=contract),
                mock.patch.object(
                    launcher,
                    "_drain_adopted_children",
                    side_effect=launcher.LaunchRefused("synthetic cleanup proof failure"),
                ),
                self.assertRaisesRegex(launcher.LaunchRefused, "cleanup proof failure"),
            ):
                launcher._launch_repeat(
                    contract,
                    "r1",
                    process_runner=process,
                    preflight_runner=_preflight,
                    validation_runner=validator,
                )
            validator.assert_not_called()
            receipt = json.loads(
                (contract.report_root / "r1-launch-receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "rejected")
            self.assertIn("bounded containment cleanup failed", receipt["error"])
            self.assertFalse(receipt["containment"]["process"]["terminal_waitpid_echild_verified"])
            self.assertIsNone(receipt["containment"]["post_process_boundary"])

    def test_static_escape_scan_rejects_process_creation_primitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "runtime.py"
            raw = b"import os\nos.fork()\n"
            source.write_bytes(raw)
            record = {
                "path": str(source),
                "bytes": len(raw),
                "digest": launcher._digest_bytes(raw),
            }
            context = SimpleNamespace(
                runtime=SimpleNamespace(
                    identity={
                        "microtensor": {"signed_source_files": {"runtime": record}},
                        "tool_sources": {},
                        "llama_cpp": {"module": record},
                    }
                )
            )
            with self.assertRaisesRegex(launcher.LaunchRefused, "process/session primitives"):
                launcher._static_process_escape_scan(context)


class AtomicRecordTests(unittest.TestCase):
    def test_atomic_publication_never_replaces_existing_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            path = root / "receipt.json"
            first = launcher._atomic_publish_noreplace(path, {"value": 1})
            before = path.read_bytes()
            with self.assertRaisesRegex(launcher.LaunchRefused, "already exists"):
                launcher._atomic_publish_noreplace(path, {"value": 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(first.as_dict(), _identity(before))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
