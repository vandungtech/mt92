from __future__ import annotations

import base64
import errno
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import code_conversion_runtime as runtime
from training import exec_attested_binary as exec_wrapper
from training import run_pinned_llama_converter as converter_wrapper


class IdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        # Attest the interpreter actually in use, not its resolved base. `.resolve()`
        # follows the venv symlink out to the bare uv CPython build, whose sys.path has
        # no venv site-packages, so a distribution probe there finds nothing installed.
        # attest_file records the symlink chain, so the unresolved path attests fine.
        self.python = Path(sys.executable)

    def test_file_identity_records_symlink_and_rejects_mutable_target(self) -> None:
        target = self.root / "tool"
        target.write_bytes(b"fixture\n")
        target.chmod(0o500)
        alias = self.root / "tool-link"
        alias.symlink_to(target.name)

        identity = runtime.attest_file(alias, require_executable=True)

        self.assertEqual(identity["requested_path"], str(alias))
        self.assertEqual(identity["resolved_path"], str(target))
        self.assertEqual(identity["symlinks"][0]["target"], target.name)
        self.assertEqual(identity["file"]["mode"], "0500")
        self.assertRegex(identity["file"]["sha256"], r"\Asha256:[0-9a-f]{64}\Z")

        target.chmod(0o722)
        with self.assertRaisesRegex(runtime.RuntimeRefused, "world writable"):
            runtime.attest_file(alias, require_executable=True)

        target.chmod(0o4500)
        with self.assertRaisesRegex(runtime.RuntimeRefused, "setuid/setgid"):
            runtime.attest_file(alias, require_executable=True)

    def test_python_and_recursive_elf_runtime_are_attested(self) -> None:
        closure = runtime.attest_elf_closure(self.python)

        self.assertTrue(closure["is_elf"])
        self.assertGreaterEqual(len(closure["objects"]), 2)
        self.assertTrue(any(edge["kind"] == "interpreter" for edge in closure["edges"]))
        self.assertTrue(any(edge["kind"] == "needed" for edge in closure["edges"]))
        for edge in closure["edges"]:
            self.assertIn("mode", edge["alias"]["file"])
            self.assertIn("sha256", edge["alias"]["file"])

        attestation = runtime.attest_runtime(
            self.python,
            distributions=("iniconfig",),
            modules=("json",),
            executables=(Path("/bin/true"),),
        )
        self.assertEqual(attestation["schema"], runtime.RUNTIME_SCHEMA)
        self.assertEqual(attestation["interpreter"]["python"]["implementation"], "CPython")
        self.assertEqual(attestation["distributions"][0]["requested_name"], "iniconfig")
        self.assertEqual(attestation["modules"][0]["name"], "json")
        self.assertEqual(len(attestation["executables"]), 1)
        self.assertTrue(attestation["executables"][0]["elf"]["is_elf"])
        self.assertTrue(attestation["modules"][0]["trees"][0]["entries"])
        self.assertRegex(attestation["sha256"], r"\Asha256:[0-9a-f]{64}\Z")

    def test_fixed_environment_has_no_ambient_search_or_secret_channels(self) -> None:
        environment = dict(runtime.FIXED_ENVIRONMENT)

        forbidden = {
            "HOME",
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "AWS_ACCESS_KEY_ID",
            "WANDB_API_KEY",
            "HF_TOKEN",
        }
        self.assertTrue(forbidden.isdisjoint(environment))
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["WANDB_MODE"], "disabled")
        with self.assertRaises(TypeError):
            runtime.FIXED_ENVIRONMENT["PATH"] = "/bin"  # type: ignore[index]

    def test_requested_source_tree_and_parent_race_guard(self) -> None:
        source = self.root / "source"
        source.mkdir(mode=0o700)
        (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        attestation = runtime.attest_runtime(self.python, trees=(source,))

        self.assertEqual(attestation["trees"][0]["root"], str(source))
        self.assertTrue(attestation["trees"][0]["entries"])
        runtime._verify_preexec_parent(101, 101)
        with self.assertRaises(OSError) as raised:
            runtime._verify_preexec_parent(101, 202)
        self.assertEqual(raised.exception.errno, errno.ESRCH)


class WrapperTests(unittest.TestCase):
    def test_pinned_converter_is_prepared_but_never_executed(self) -> None:
        prior_path = list(sys.path)
        observed_argv: list[str] = []

        def observe(_path: str, *, run_name: str) -> None:
            self.assertEqual(run_name, "__main__")
            observed_argv.extend(sys.argv)

        try:
            with (
                mock.patch.dict(os.environ, converter_wrapper.EXACT_ENVIRONMENT, clear=True),
                mock.patch.object(
                    converter_wrapper.runpy, "run_path", side_effect=observe
                ) as run_path,
            ):
                converter_wrapper.main(["fixture-model", "--outtype", "f16"])
            run_path.assert_called_once_with(
                str(converter_wrapper.CONVERTER), run_name="__main__"
            )
        finally:
            sys.path[:] = prior_path
        self.assertEqual(
            observed_argv,
            [str(converter_wrapper.CONVERTER), "fixture-model", "--outtype", "f16"],
        )
        self.assertEqual(sys.path, prior_path)

    def test_pinned_converter_rejects_remote_before_runpy(self) -> None:
        with (
            mock.patch.dict(os.environ, converter_wrapper.EXACT_ENVIRONMENT, clear=True),
            mock.patch.object(converter_wrapper.runpy, "run_path") as run_path,
            self.assertRaisesRegex(RuntimeError, "remote conversion is forbidden"),
        ):
            converter_wrapper.main(["fixture", "--remote"])
        run_path.assert_not_called()


class ContainedCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.python = Path(sys.executable).resolve()

    def _script(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o400)
        return path

    def _request(
        self,
        script: Path,
        *arguments: str,
        files: tuple[Path, ...] = (),
        timeout: float = 5.0,
        grace: float = 0.1,
        capture: int = 256,
    ) -> runtime.CommandRequest:
        return runtime.CommandRequest(
            interpreter=self.python,
            argv=(str(self.python), str(script), *arguments),
            cwd=self.root,
            record_path=self.root / f"{script.stem}.record.json",
            files=(script, *files),
            timeout_seconds=timeout,
            term_grace_seconds=grace,
            cleanup_seconds=1.5,
            max_log_bytes=capture,
        )

    def _exec_request(
        self,
        target: Path,
        *arguments: str,
        files: tuple[Path, ...] = (),
        timeout: float = 5.0,
        grace: float = 0.1,
    ) -> runtime.CommandRequest:
        wrapper = Path(exec_wrapper.__file__).resolve()
        return runtime.CommandRequest(
            interpreter=self.python,
            argv=(str(self.python), str(wrapper), str(target), *arguments),
            cwd=self.root,
            record_path=self.root / f"exec-{target.name}-{len(arguments)}.record.json",
            executables=(target,),
            files=(wrapper, *files),
            exec_target=target,
            timeout_seconds=timeout,
            term_grace_seconds=grace,
            cleanup_seconds=1.5,
            max_log_bytes=512,
        )

    def test_success_uses_exact_environment_and_commits_bounded_logs(self) -> None:
        script = self._script(
            "success.py",
            """
import json
import os
import sys

print(json.dumps(sorted(os.environ)))
sys.stdout.write("x" * 4096)
sys.stderr.write("fixture-stderr")
""".lstrip(),
        )
        request = self._request(script, capture=512)

        record = runtime.run_contained_command(request)

        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["process"]["direct"]["returncode"], 0)
        self.assertFalse(record["process"]["timed_out"])
        self.assertFalse(record["runtime_mutated"])
        self.assertTrue(record["containment"]["terminal_waitpid_echild_verified"])
        self.assertFalse(record["containment"]["descendants_observed"])
        self.assertEqual(record["containment"]["pdeathsig"]["signal"], "SIGKILL")
        self.assertTrue(
            record["containment"]["pdeathsig"][
                "configured_and_verified_in_single_thread_preexec"
            ]
        )
        self.assertEqual(
            record["containment"]["immediate_preexec_boundary"],
            record["containment"]["pre_fork_boundary"],
        )
        self.assertGreater(record["stdout"]["bytes"], record["stdout"]["captured_bytes"])
        self.assertTrue(record["stdout"]["truncated"])
        self.assertLessEqual(record["stdout"]["captured_bytes"], 512)
        captured = base64.b64decode(record["stdout"]["captured_base64"])
        environment_keys = set(json.loads(captured.splitlines()[0]))
        self.assertEqual(environment_keys, set(runtime.FIXED_ENVIRONMENT))
        self.assertNotIn("PATH", environment_keys)
        on_disk = json.loads(request.record_path.read_text(encoding="ascii"))
        self.assertEqual(on_disk, record)
        self.assertEqual(stat.S_IMODE(request.record_path.stat().st_mode), 0o600)

    def test_timeout_uses_term_or_kill_and_commits_refusal(self) -> None:
        script = self._script("timeout.py", "import time\ntime.sleep(30)\n")
        request = self._request(script, timeout=0.15, grace=0.05)

        with self.assertRaises(runtime.CommandExecutionError) as raised:
            runtime.run_contained_command(request)

        record = raised.exception.record
        self.assertEqual(record["status"], "refused")
        self.assertTrue(record["process"]["timed_out"])
        self.assertTrue(record["containment"]["term_signals"])
        self.assertTrue(record["containment"]["terminal_waitpid_echild_verified"])
        self.assertEqual(json.loads(request.record_path.read_text(encoding="ascii")), record)

    def test_detached_descendant_is_killed_reaped_and_refused(self) -> None:
        script = self._script(
            "descendant.py",
            """
import subprocess
import sys

subprocess.Popen(
    [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
    close_fds=True,
    start_new_session=True,
)
print("spawned")
""".lstrip(),
        )
        request = self._request(script)

        with self.assertRaises(runtime.CommandExecutionError) as raised:
            runtime.run_contained_command(request)

        containment = raised.exception.record["containment"]
        self.assertTrue(containment["descendants_observed"])
        self.assertTrue(containment["observed_descendant_pids"])
        self.assertTrue(containment["reaped_descendants"])
        self.assertTrue(containment["terminal_waitpid_echild_verified"])

    def test_runtime_mutation_is_detected_after_child_is_reaped(self) -> None:
        victim = self.root / "attested-tool"
        victim.write_bytes(b"before")
        victim.chmod(0o400)
        script = self._script(
            "mutate.py",
            """
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_bytes(b"after")
""".lstrip(),
        )
        request = self._request(script, str(victim), files=(victim,))

        with self.assertRaises(runtime.CommandExecutionError) as raised:
            runtime.run_contained_command(request)

        record = raised.exception.record
        self.assertTrue(record["runtime_mutated"])
        self.assertIn("runtime identity mutated", record["process"]["failure"])
        self.assertTrue(record["containment"]["terminal_waitpid_echild_verified"])

    def test_exec_wrapper_success_preserves_pid_boundary_without_descendants(self) -> None:
        target = Path("/bin/true")
        request = self._exec_request(target)

        record = runtime.run_contained_command(request)

        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["execution_kind"], "exec-wrapper")
        self.assertEqual(record["exec_target"], str(target))
        self.assertEqual(record["process"]["direct"]["returncode"], 0)
        self.assertFalse(record["containment"]["descendants_observed"])
        self.assertTrue(record["containment"]["terminal_waitpid_echild_verified"])
        self.assertTrue(
            record["containment"]["pdeathsig"][
                "configured_and_verified_in_single_thread_preexec"
            ]
        )

    def test_exec_wrapper_timeout_is_bounded_and_recorded(self) -> None:
        request = self._exec_request(Path("/bin/sleep"), "30", timeout=0.15, grace=0.05)

        with self.assertRaises(runtime.CommandExecutionError) as raised:
            runtime.run_contained_command(request)

        record = raised.exception.record
        self.assertTrue(record["process"]["timed_out"])
        self.assertTrue(record["containment"]["term_signals"])
        self.assertTrue(record["containment"]["terminal_waitpid_echild_verified"])

    def test_exec_wrapper_detects_post_exec_runtime_mutation(self) -> None:
        victim = self.root / "exec-attested-input"
        victim.write_bytes(b"before")
        victim.chmod(0o400)
        script = self._script(
            "exec-mutate.py",
            "import pathlib, sys\npathlib.Path(sys.argv[1]).write_bytes(b'after')\n",
        )
        request = self._exec_request(
            self.python, str(script), str(victim), files=(script, victim)
        )

        with self.assertRaises(runtime.CommandExecutionError) as raised:
            runtime.run_contained_command(request)

        record = raised.exception.record
        self.assertTrue(record["runtime_mutated"])
        self.assertIn("runtime identity mutated", record["process"]["failure"])
        self.assertTrue(record["containment"]["terminal_waitpid_echild_verified"])

    def test_relative_interpreter_and_unattested_inline_code_are_rejected(self) -> None:
        script = self._script("unused.py", "print('unused')\n")
        relative = runtime.CommandRequest(
            interpreter=Path("python3"),
            argv=("python3", str(script)),
            cwd=self.root,
            record_path=self.root / "relative.json",
            files=(script,),
        )
        with self.assertRaisesRegex(runtime.RuntimeRefused, "must be absolute"):
            runtime.run_contained_command(relative)

        inline = runtime.CommandRequest(
            interpreter=self.python,
            argv=(str(self.python), "-c", "print('hidden')"),
            cwd=self.root,
            record_path=self.root / "inline.json",
        )
        with self.assertRaisesRegex(runtime.RuntimeRefused, "must be absolute"):
            runtime.run_contained_command(inline)


if __name__ == "__main__":
    unittest.main()
