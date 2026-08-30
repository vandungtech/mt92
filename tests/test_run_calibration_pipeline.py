from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from training import run_calibration_pipeline as pipeline


def stream(payload: bytes = b"") -> pipeline.StreamDigest:
    return pipeline.StreamDigest(
        bytes=len(payload),
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        captured_bytes=len(payload),
        capture_limit_bytes=1024,
        truncated=False,
    )


class PipelineFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.chmod(0o755)

        self.source = self.root / "stage" / "merged"
        self.source.mkdir(parents=True)
        self.source.chmod(0o755)
        self.source.parent.chmod(0o755)
        source_files = {
            "config.json": b'{"model_type":"qwen3"}\n',
            "model.safetensors": b"tiny-weights",
            "tokenizer.json": b'{}\n',
            "tokenizer_config.json": b'{}\n',
        }
        for name, payload in source_files.items():
            path = self.source / name
            path.write_bytes(payload)
            path.chmod(0o644)
        self.training_metadata = self.source.parent / "training_metadata.json"
        self.training_metadata.write_bytes(b'{"finished":true}\n')
        self.training_metadata.chmod(0o644)

        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.corpus = self.evidence / "calibration.txt"
        self.evidence.chmod(0o755)
        self.corpus.write_bytes(b"public calibration text\n")
        self.sidecar = self.evidence / "calibration.metadata.json"
        self.corpus.chmod(0o644)
        self.sidecar.write_bytes(b'{"schema":"test"}\n')

        self.sidecar.chmod(0o644)
        self.llama = self.root / "llama.cpp"
        (self.llama / "build" / "bin").mkdir(parents=True)
        (self.llama / ".git").mkdir()
        self.llama.chmod(0o700)
        (self.llama / ".git").chmod(0o755)
        self.git_config = self.llama / ".git" / "config"
        self.git_config.write_bytes(b"[core]\n\tbare = false\n")
        self.git_config.chmod(0o644)
        (self.llama / "build").chmod(0o755)
        (self.llama / "build" / "bin").chmod(0o755)
        self.converter = self.llama / "convert_hf_to_gguf.py"
        self.imatrix_tool = self.llama / "build" / "bin" / "llama-imatrix"
        self.quantizer = self.llama / "build" / "bin" / "llama-quantize"
        self.converter.write_bytes(b"# pinned converter\n")
        self.imatrix_tool.write_bytes(b"pinned-imatrix-binary")
        self.converter.chmod(0o644)
        self.quantizer.write_bytes(b"pinned-quantizer-binary")
        self.imatrix_tool.chmod(0o755)
        self.quantizer.chmod(0o755)

        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.python = self.tools / "python3"
        self.git = self.tools / "git"
        self.tools.chmod(0o755)
        self.python.write_bytes(b"fake-python")
        self.git.write_bytes(b"fake-git")
        self.python.chmod(0o755)
        self.git.chmod(0o755)

        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.outputs.chmod(0o755)
        self.converted = self.outputs / "model-f16.gguf"
        self.imatrix = self.outputs / "calibration.imatrix.gguf"
        self.quantized = self.outputs / "model-q4.gguf"
        self.receipt = self.root / "calibration-execution-receipt.json"
        self.request = pipeline.PipelineRequest(
            source_model_dir=self.source,
            training_metadata=self.training_metadata,
            calibration_corpus=self.corpus,
            corpus_metadata=self.sidecar,
            converted_model=self.converted,
            imatrix=self.imatrix,
            quantized_artifact=self.quantized,
            receipt=self.receipt,
            llama_cpp_dir=self.llama,
            python_executable=self.python,
            capture_limit_bytes=1024,
        )
        self.calls: list[tuple[str, tuple[str, ...], dict[str, str], int]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def clean_git(_git: Path, _repository: Path, arguments: tuple[str, ...]) -> str:
        if arguments[0] == "rev-parse":
            return pipeline.LLAMA_CPP_REVISION
        if arguments[0] == "status":
            return ""
        raise AssertionError(arguments)

    def successful_runner(
        self,
        name: str,
        argv: tuple[str, ...],
        environment: dict[str, str],
        capture_limit: int,
    ) -> pipeline.CommandRecord:
        self.calls.append((name, argv, dict(environment), capture_limit))
        if name == "convert_f16":
            output = Path(argv[argv.index("--outfile") + 1])
            output.write_bytes(b"GGUF-f16-output")
        elif name == "build_imatrix":
            output = Path(argv[argv.index("--output") + 1])
            output.write_bytes(b"GGUF-imatrix-output")
        elif name == "quantize_q4_k_m":
            output = Path(argv[-2])
            output.write_bytes(b"GGUF-q4-output")
        else:
            raise AssertionError(name)
        return pipeline.CommandRecord(
            name=name,
            argv=tuple(argv),
            started_at_utc="2026-08-30T00:00:00.000000Z",
            started_at_unix_ns=1,
            finished_at_utc="2026-08-30T00:00:01.000000Z",
            finished_at_unix_ns=2,
            returncode=0,
            stdout=stream(name.encode()),
            stderr=stream(),
        )

    def execute_pipeline(
        self,
        *,
        request: pipeline.PipelineRequest | None = None,
        runner: object | None = None,
        git_output: object | None = None,
    ) -> pipeline.PipelineResult:
        with (
            mock.patch.object(pipeline.shutil, "which", return_value=str(self.git)),
            mock.patch.object(
                pipeline,
                "_git_output",
                side_effect=git_output or self.clean_git,
            ),
            mock.patch.object(
                pipeline,
                "_run_command",
                side_effect=runner or self.successful_runner,
            ),
        ):
            return pipeline.run_calibration_pipeline(request or self.request)

    def assert_no_pipeline_temporaries(self) -> None:
        names = [path.name for path in self.root.rglob("*.partial")]
        self.assertEqual(names, [])


class RunCalibrationPipelineTests(PipelineFixture):
    def test_success_records_exact_execution_and_commits_receipt_last(self) -> None:
        result = self.execute_pipeline()

        self.assertEqual(result.receipt, self.receipt)
        self.assertTrue(result.post_run_integrity_confirmed)
        self.assertTrue(result.durability_confirmed)
        self.assertEqual(self.converted.read_bytes(), b"GGUF-f16-output")
        self.assertEqual(self.imatrix.read_bytes(), b"GGUF-imatrix-output")
        self.assertEqual(self.quantized.read_bytes(), b"GGUF-q4-output")
        self.assertEqual(
            result.receipt_sha256,
            "sha256:" + hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
        )
        self.assert_no_pipeline_temporaries()

        receipt = json.loads(self.receipt.read_bytes())
        self.assertEqual(receipt["schema"], pipeline.SCHEMA)
        self.assertEqual(
            receipt["llama_cpp"]["revision_before"],
            pipeline.LLAMA_CPP_REVISION,
        )
        self.assertEqual(
            receipt["llama_cpp"]["revision_after"],
            pipeline.LLAMA_CPP_REVISION,
        )
        self.assertTrue(receipt["llama_cpp"]["clean_before"])
        self.assertTrue(receipt["llama_cpp"]["clean_after"])
        self.assertFalse(receipt["execution"]["shell"])
        self.assertEqual(
            receipt["execution"]["environment_overrides"],
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "disabled",
            },
        )

        commands = receipt["execution"]["commands"]
        self.assertEqual(
            [command["name"] for command in commands],
            ["convert_f16", "build_imatrix", "quantize_q4_k_m"],
        )
        conversion = commands[0]["argv"]
        self.assertEqual(conversion[:3], [str(self.python), str(self.converter), str(self.source)])
        self.assertEqual(conversion[-2:], ["--outtype", "f16"])
        self.assertNotEqual(conversion[conversion.index("--outfile") + 1], str(self.converted))

        imatrix = commands[1]["argv"]
        self.assertEqual(imatrix[0], str(self.imatrix_tool))
        self.assertEqual(
            [item for item in imatrix if item in {"--offline", "--no-ppl", "--parse-special"}],
            ["--offline", "--no-ppl", "--parse-special"],
        )
        self.assertEqual(imatrix[imatrix.index("--ctx-size") + 1], "512")
        self.assertEqual(imatrix[imatrix.index("--chunks") + 1], "-1")
        self.assertEqual(imatrix[imatrix.index("--file") + 1], str(self.corpus))
        self.assertNotEqual(imatrix[imatrix.index("--output") + 1], str(self.imatrix))

        quantization = commands[2]["argv"]
        self.assertEqual(quantization[0], str(self.quantizer))
        self.assertEqual(quantization[1], "--imatrix")
        self.assertEqual(quantization[-1], "Q4_K_M")
        self.assertNotEqual(quantization[-2], str(self.quantized))
        for command in commands:
            self.assertEqual(command["returncode"], 0)
            self.assertRegex(command["stdout"]["sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(command["stderr"]["sha256"], r"^sha256:[0-9a-f]{64}$")

        source = receipt["inputs"]["source_model"]
        self.assertEqual(source["before"], source["after"])
        self.assertEqual(source["before"]["tree_algorithm"], pipeline.TREE_ALGORITHM)
        self.assertRegex(source["before"]["tree_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            source["training_metadata_sha256"],
            receipt["inputs"]["training_metadata"]["before"]["sha256"],
        )
        self.assertEqual(
            receipt["inputs"]["calibration_corpus"]["before"],
            receipt["inputs"]["calibration_corpus"]["after"],
        )
        self.assertEqual(
            receipt["inputs"]["corpus_metadata"]["before"],
            receipt["inputs"]["corpus_metadata"]["after"],
        )
        for tool in receipt["tools"].values():
            self.assertEqual(tool["before"], tool["after"])
            self.assertIn("bytes", tool["before"])
            self.assertIn("sha256", tool["before"])
        for name, path in {
            "converted_model": self.converted,
            "imatrix": self.imatrix,
            "quantized_artifact": self.quantized,
        }.items():
            self.assertEqual(receipt["outputs"][name]["path"], str(path))
            self.assertEqual(receipt["outputs"][name]["mode"], "0o0644")
        self.assertEqual(
            receipt["post_run_integrity"]["receipt_commit"],
            "no_replace_hard_link_from_held_inode",
        )
        self.assertTrue(receipt["post_run_integrity"]["confirmed"])
        self.assertTrue(
            receipt["post_run_integrity"][
                "final_outputs_rechecked_immediately_before_commit"
            ]
        )

    def test_nonzero_step_fails_closed_without_final_outputs_or_receipt(self) -> None:
        def failing_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            record = self.successful_runner(name, argv, environment, capture_limit)
            if name == "build_imatrix":
                return pipeline.CommandRecord(
                    **{**record.__dict__, "returncode": 17}
                )
            return record

        with self.assertRaisesRegex(pipeline.PipelineError, "return code 17"):
            self.execute_pipeline(runner=failing_runner)
        self.assertFalse(self.converted.exists())
        self.assertFalse(self.imatrix.exists())
        self.assertFalse(self.quantized.exists())
        self.assertFalse(self.receipt.exists())
        self.assert_no_pipeline_temporaries()

    def test_keyboard_interrupt_rolls_back_and_never_commits_receipt(self) -> None:
        def interrupted_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            if name == "build_imatrix":
                raise KeyboardInterrupt
            return self.successful_runner(name, argv, environment, capture_limit)

        with self.assertRaises(KeyboardInterrupt):
            self.execute_pipeline(runner=interrupted_runner)
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
        self.assert_no_pipeline_temporaries()

    def test_interrupt_after_output_link_before_tracking_removes_link(self) -> None:
        real_link = pipeline._link_descriptor_no_replace

        def link_then_interrupt(descriptor: int, destination: Path) -> tuple[int, int]:
            identity = real_link(descriptor, destination)
            if destination == self.converted:
                raise KeyboardInterrupt
            return identity

        with mock.patch.object(
            pipeline, "_link_descriptor_no_replace", side_effect=link_then_interrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.execute_pipeline()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
        self.assert_no_pipeline_temporaries()

    def test_interrupt_after_receipt_link_before_tracking_removes_everything(self) -> None:
        real_link = pipeline._link_descriptor_no_replace

        def link_then_interrupt(descriptor: int, destination: Path) -> tuple[int, int]:
            identity = real_link(descriptor, destination)
            if destination == self.receipt:
                raise KeyboardInterrupt
            return identity

        with mock.patch.object(
            pipeline, "_link_descriptor_no_replace", side_effect=link_then_interrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.execute_pipeline()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
        self.assert_no_pipeline_temporaries()

    def test_rollback_failure_is_observable_and_directory_is_fsynced(self) -> None:
        real_link = pipeline._link_descriptor_no_replace
        real_unlink = pipeline._unlink_if_identity
        real_fsync = pipeline._fsync_directory
        fsynced: list[Path] = []

        def link_then_interrupt(descriptor: int, destination: Path) -> tuple[int, int]:
            identity = real_link(descriptor, destination)
            if destination == self.converted:
                raise KeyboardInterrupt
            return identity

        def refuse_final(path: Path, identity: tuple[int, int]) -> bool:
            if path == self.converted:
                return False
            return real_unlink(path, identity)

        def recording_fsync(path: Path) -> None:
            fsynced.append(path)
            real_fsync(path)

        with (
            mock.patch.object(
                pipeline,
                "_link_descriptor_no_replace",
                side_effect=link_then_interrupt,
            ),
            mock.patch.object(
                pipeline, "_unlink_if_identity", side_effect=refuse_final
            ),
            mock.patch.object(
                pipeline, "_fsync_directory", side_effect=recording_fsync
            ),
            self.assertRaisesRegex(
                pipeline.PipelineError, "rollback was incomplete.*model-f16.gguf"
            ),
        ):
            self.execute_pipeline()
        self.assertIn(self.outputs, fsynced)
        self.assertTrue(self.converted.exists())
        self.converted.unlink()
        self.assertFalse(self.receipt.exists())
        self.assert_no_pipeline_temporaries()

    def test_rollback_directory_fsync_failure_is_observable(self) -> None:
        real_link = pipeline._link_descriptor_no_replace

        def link_then_interrupt(descriptor: int, destination: Path) -> tuple[int, int]:
            identity = real_link(descriptor, destination)
            if destination == self.converted:
                raise KeyboardInterrupt
            return identity

        with (
            mock.patch.object(
                pipeline,
                "_link_descriptor_no_replace",
                side_effect=link_then_interrupt,
            ),
            mock.patch.object(
                pipeline,
                "_fsync_directory",
                side_effect=OSError("synthetic fsync failure"),
            ),
            self.assertRaisesRegex(
                pipeline.PipelineError, "rollback was incomplete.*fsync"
            ),
        ):
            self.execute_pipeline()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())

    def test_final_outputs_are_reverified_immediately_before_receipt_commit(self) -> None:
        real_mkstemp = pipeline.tempfile.mkstemp
        calls = 0

        def mutate_on_receipt_temp(*args: object, **kwargs: object) -> tuple[int, str]:
            nonlocal calls
            created = real_mkstemp(*args, **kwargs)
            calls += 1
            if calls == 4:
                self.quantized.write_bytes(b"GGUF-mutated-after-first-verification")
            return created

        with mock.patch.object(
            pipeline.tempfile, "mkstemp", side_effect=mutate_on_receipt_temp
        ), self.assertRaisesRegex(pipeline.PipelineError, "bytes or identity"):
            self.execute_pipeline()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
        self.assert_no_pipeline_temporaries()

    def test_input_mutation_after_commands_fails_closed(self) -> None:
        def mutating_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            record = self.successful_runner(name, argv, environment, capture_limit)
            if name == "quantize_q4_k_m":
                self.corpus.write_bytes(b"mutated after execution\n")
            return record

        with self.assertRaisesRegex(pipeline.PipelineError, "calibration_corpus changed"):
            self.execute_pipeline(runner=mutating_runner)
        self.assertFalse(self.receipt.exists())
        self.assertFalse(self.converted.exists())
        self.assertFalse(self.imatrix.exists())
        self.assertFalse(self.quantized.exists())
        self.assert_no_pipeline_temporaries()

    def test_source_tree_mutation_after_commands_fails_closed(self) -> None:
        def mutating_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            record = self.successful_runner(name, argv, environment, capture_limit)
            if name == "quantize_q4_k_m":
                (self.source / "model.safetensors").write_bytes(b"changed")
            return record

        with self.assertRaisesRegex(pipeline.PipelineError, "source model tree changed"):
            self.execute_pipeline(runner=mutating_runner)
        self.assertFalse(self.receipt.exists())
        self.assert_no_pipeline_temporaries()

    def test_tool_mutation_after_commands_fails_closed(self) -> None:
        def mutating_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            record = self.successful_runner(name, argv, environment, capture_limit)
            if name == "quantize_q4_k_m":
                self.quantizer.write_bytes(b"changed-tool")
                self.quantizer.chmod(0o755)
            return record

        with self.assertRaisesRegex(pipeline.PipelineError, "tool llama-quantize changed"):
            self.execute_pipeline(runner=mutating_runner)
        self.assertFalse(self.receipt.exists())
        self.assert_no_pipeline_temporaries()
    def test_final_git_check_precedes_every_final_identity_snapshot(self) -> None:
        status_calls = 0

        def mutating_git(
            _git: Path, _repository: Path, arguments: tuple[str, ...]
        ) -> str:
            nonlocal status_calls
            if arguments[0] == "rev-parse":
                return pipeline.LLAMA_CPP_REVISION
            status_calls += 1
            if status_calls == 2:
                (self.source / "model.safetensors").write_bytes(b"changed-by-git")
                self.quantizer.write_bytes(b"changed-tool-by-git")
            return ""

        with self.assertRaisesRegex(
            pipeline.PipelineError, "source model tree changed"
        ):
            self.execute_pipeline(git_output=mutating_git)
        self.assertEqual(status_calls, 2)
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
        self.assert_no_pipeline_temporaries()


    def test_preexisting_destination_is_never_overwritten_and_runner_is_not_called(self) -> None:
        self.converted.write_bytes(b"keep-me")
        runner = mock.Mock()
        with self.assertRaisesRegex(pipeline.PipelineError, "already exists"):
            self.execute_pipeline(runner=runner)
        self.assertEqual(self.converted.read_bytes(), b"keep-me")
        self.assertFalse(self.receipt.exists())
        runner.assert_not_called()

    def test_destination_race_does_not_overwrite_and_rolls_back_prior_install(self) -> None:
        def racing_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            record = self.successful_runner(name, argv, environment, capture_limit)
            if name == "quantize_q4_k_m":
                self.imatrix.write_bytes(b"foreign-winner")
            return record

        with self.assertRaisesRegex(pipeline.PipelineError, "refusing to overwrite"):
            self.execute_pipeline(runner=racing_runner)
        self.assertFalse(self.converted.exists())
        self.assertEqual(self.imatrix.read_bytes(), b"foreign-winner")
        self.assertFalse(self.quantized.exists())
        self.assertFalse(self.receipt.exists())
        self.assert_no_pipeline_temporaries()

    def test_dirty_or_wrong_llama_checkout_is_rejected_before_execution(self) -> None:
        def dirty(_git: Path, _repository: Path, arguments: tuple[str, ...]) -> str:
            if arguments[0] == "rev-parse":
                return pipeline.LLAMA_CPP_REVISION
            return " M convert_hf_to_gguf.py"

        runner = mock.Mock()
        with self.assertRaisesRegex(pipeline.PipelineError, "clean worktree"):
            self.execute_pipeline(runner=runner, git_output=dirty)
        runner.assert_not_called()
        self.assertFalse(self.receipt.exists())

        def wrong(_git: Path, _repository: Path, arguments: tuple[str, ...]) -> str:
            return "0" * 40 if arguments[0] == "rev-parse" else ""

        with self.assertRaisesRegex(pipeline.PipelineError, "revision must be exactly"):
            self.execute_pipeline(runner=runner, git_output=wrong)
        runner.assert_not_called()

    def test_symlinked_input_and_output_ancestor_are_rejected(self) -> None:
        source_alias = self.root / "source-alias"
        source_alias.symlink_to(self.source, target_is_directory=True)
        request = pipeline.PipelineRequest(
            **{**self.request.__dict__, "source_model_dir": source_alias}
        )
        with self.assertRaisesRegex(pipeline.PipelineError, "symlink component"):
            self.execute_pipeline(request=request)

        output_alias = self.root / "output-alias"
        output_alias.symlink_to(self.outputs, target_is_directory=True)
        request = pipeline.PipelineRequest(
            **{
                **self.request.__dict__,
                "converted_model": output_alias / "model-f16.gguf",
            }
        )
        with self.assertRaisesRegex(pipeline.PipelineError, "symlink component"):
            self.execute_pipeline(request=request)

    def test_source_tree_symlink_is_rejected(self) -> None:
        (self.source / "unsafe-link").symlink_to(self.corpus)
        runner = mock.Mock()
        with self.assertRaisesRegex(pipeline.PipelineError, "contains a symlink"):
            self.execute_pipeline(runner=runner)
        runner.assert_not_called()

    def test_evidence_input_cannot_hardlink_alias_a_source_file(self) -> None:
        self.corpus.unlink()
        os.link(self.source / "config.json", self.corpus)
        runner = mock.Mock()
        git_output = mock.Mock()
        with self.assertRaisesRegex(pipeline.PipelineError, "pairwise-distinct inodes"):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_writable_evidence_parent_is_rejected_before_execution(self) -> None:
        self.evidence.chmod(0o775)
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError,
            "calibration corpus directory component is group- or world-writable",
        ):
            self.execute_pipeline(runner=runner)
        runner.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_writable_tool_parent_is_rejected_before_execution(self) -> None:
        self.tools.chmod(0o775)
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError,
            "Python executable directory component is group- or world-writable",
        ):
            self.execute_pipeline(runner=runner)
        runner.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_llama_checkout_requires_private_effective_uid_boundary(self) -> None:
        self.llama.chmod(0o755)
        git_output = mock.Mock()
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError, "exclusive private trust boundary"
        ):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()

    def test_private_source_ancestor_allows_writable_directories(self) -> None:
        self.source.parent.chmod(0o700)
        self.source.chmod(0o775)
        result = self.execute_pipeline()
        self.assertTrue(result.post_run_integrity_confirmed)
        self.assertTrue(self.receipt.exists())

    def test_writable_source_leaf_is_rejected_below_private_boundary(self) -> None:
        self.source.parent.chmod(0o700)
        (self.source / "config.json").chmod(0o666)
        git_output = mock.Mock()
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError, "source model file config.json must not be"
        ):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()

    def test_private_llama_ancestor_allows_writable_directories(self) -> None:
        (self.llama / "build").chmod(0o775)
        (self.llama / "build" / "bin").chmod(0o775)
        result = self.execute_pipeline()
        self.assertTrue(result.post_run_integrity_confirmed)
        self.assertTrue(self.receipt.exists())

    def test_writable_llama_leaf_with_public_hardlink_is_rejected(self) -> None:
        public_alias = self.root / "public-converter"
        os.link(self.converter, public_alias)
        self.converter.chmod(0o666)
        git_output = mock.Mock()
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError, "tool convert_hf_to_gguf.py must not be"
        ):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()

    def test_writable_git_leaf_is_rejected_before_git_is_executed(self) -> None:
        self.git.chmod(0o777)
        git_output = mock.Mock()
        runner = mock.Mock()
        with self.assertRaisesRegex(pipeline.PipelineError, "tool git must not be"):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()

    def test_writable_git_metadata_is_rejected_before_git_execution(self) -> None:
        self.git_config.chmod(0o666)
        git_output = mock.Mock()
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError, "Git metadata file must not be"
        ):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()

    def test_git_config_external_helper_is_rejected_before_git_execution(self) -> None:
        self.git_config.write_bytes(
            b"[core]\n\tbare = false\n[credential]\n\thelper = !attacker\n"
        )
        git_output = mock.Mock()
        runner = mock.Mock()
        with self.assertRaisesRegex(
            pipeline.PipelineError, "may not select external helpers"
        ):
            self.execute_pipeline(runner=runner, git_output=git_output)
        git_output.assert_not_called()
        runner.assert_not_called()

    def test_partial_clone_promisor_cannot_reach_path_remote_helper(self) -> None:
        marker = self.root / "remote-helper-executed"
        helper_directory = self.root / "remote-helper-bin"
        helper_directory.mkdir()
        helper_directory.chmod(0o755)
        helper = helper_directory / "git-remote-probe"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )
        helper.chmod(0o755)
        configs = (
            (
                b"[core]\n\tbare = false\n\trepositoryformatversion = 1\n"
                b"[extensions]\n\tpartialClone = origin\n"
                b'[remote "origin"]\n\turl = probe::payload\n'
            ),
            (
                b"[core]\n\tbare = false\n"
                b'[remote "origin"]\n\turl = probe::payload\n'
                b"\tpromisor = true\n"
            ),
        )
        helper_path = (
            str(helper_directory)
            + os.pathsep
            + os.environ.get("PATH", "")
        )
        with mock.patch.dict(os.environ, {"PATH": helper_path}):
            for config in configs:
                with self.subTest(config=config):
                    self.git_config.write_bytes(config)
                    git_output = mock.Mock()
                    runner = mock.Mock()
                    with self.assertRaisesRegex(
                        pipeline.PipelineError, "partial-clone/promisor"
                    ):
                        self.execute_pipeline(
                            runner=runner, git_output=git_output
                        )
                    git_output.assert_not_called()
                    runner.assert_not_called()
                    self.assertFalse(marker.exists())


    def test_all_temporaries_are_mode_0600_when_linked(self) -> None:
        real_link = pipeline._link_descriptor_no_replace
        link_modes: list[int] = []

        def record_mode(descriptor: int, destination: Path) -> tuple[int, int]:
            link_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
            return real_link(descriptor, destination)

        with mock.patch.object(
            pipeline, "_link_descriptor_no_replace", side_effect=record_mode
        ):
            self.execute_pipeline()
        self.assertEqual(link_modes, [0o600, 0o600, 0o600, 0o600])
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_interrupt_before_third_temporary_creation_cleans_first_two(self) -> None:
        real_make = pipeline._make_temporary_output

        def interrupt_before_third(
            final: Path,
            registry: list[pipeline._TemporaryOutput],
            *,
            hold_open: bool = False,
        ) -> pipeline._TemporaryOutput:
            if final == self.quantized:
                raise KeyboardInterrupt
            return real_make(final, registry, hold_open=hold_open)

        with mock.patch.object(
            pipeline, "_make_temporary_output", side_effect=interrupt_before_third
        ), self.assertRaises(KeyboardInterrupt):
            self.execute_pipeline()
        self.assert_no_pipeline_temporaries()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())

    def test_interrupt_during_temporary_fchmod_cleans_new_file(self) -> None:
        real_fchmod = pipeline.os.fchmod
        interrupted = False

        def interrupt_once(descriptor: int, mode: int) -> None:
            nonlocal interrupted
            if mode == 0o600 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            real_fchmod(descriptor, mode)

        with (
            mock.patch.object(pipeline.os, "fchmod", side_effect=interrupt_once),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.execute_pipeline()
        self.assert_no_pipeline_temporaries()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())

    def test_interrupt_during_temporary_fstat_cleans_new_file(self) -> None:
        real_fstat = pipeline.os.fstat
        interrupted = False

        def interrupt_once(descriptor: int) -> os.stat_result:
            nonlocal interrupted
            try:
                target = os.readlink(f"/proc/self/fd/{descriptor}")
            except OSError:
                target = ""
            if target.endswith(".partial") and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return real_fstat(descriptor)

        with (
            mock.patch.object(pipeline.os, "fstat", side_effect=interrupt_once),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.execute_pipeline()
        self.assert_no_pipeline_temporaries()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
    def test_sigint_after_mkstemp_is_deferred_until_registration(self) -> None:
        if os.name != "posix" or not hasattr(pipeline.signal, "pthread_sigmask"):
            self.skipTest("requires POSIX pthread signal masks")
        real_mkstemp = pipeline.tempfile.mkstemp
        signaled = False

        def create_then_signal(*args: object, **kwargs: object) -> tuple[int, str]:
            nonlocal signaled
            created = real_mkstemp(*args, **kwargs)
            if not signaled:
                signaled = True
                os.kill(os.getpid(), pipeline.signal.SIGINT)
            return created

        with mock.patch.object(
            pipeline.tempfile, "mkstemp", side_effect=create_then_signal
        ), self.assertRaises(KeyboardInterrupt):
            self.execute_pipeline()
        self.assertTrue(signaled)
        self.assert_no_pipeline_temporaries()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())


    def test_interrupt_after_receipt_temp_registration_cleans_everything(self) -> None:
        real_make = pipeline._make_temporary_output

        def make_then_interrupt(
            final: Path,
            registry: list[pipeline._TemporaryOutput],
            *,
            hold_open: bool = False,
        ) -> pipeline._TemporaryOutput:
            temporary = real_make(final, registry, hold_open=hold_open)
            if final == self.receipt:
                raise KeyboardInterrupt
            return temporary

        with mock.patch.object(
            pipeline, "_make_temporary_output", side_effect=make_then_interrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.execute_pipeline()
        self.assert_no_pipeline_temporaries()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())

    def test_temporary_unlink_failure_is_observable(self) -> None:
        real_unlink = pipeline._unlink_if_identity

        def refuse_partial(path: Path, identity: tuple[int, int]) -> bool:
            if path.name.endswith(".partial"):
                return False
            return real_unlink(path, identity)

        with mock.patch.object(
            pipeline, "_unlink_if_identity", side_effect=refuse_partial
        ), self.assertRaisesRegex(
            pipeline.PipelineError, "temporary cleanup was incomplete"
        ):
            self.execute_pipeline()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())
        for path in self.root.rglob("*.partial"):
            path.unlink()

    def test_temporary_directory_fsync_failure_is_observable(self) -> None:
        real_fsync = pipeline._fsync_directory

        def fail_output_directory(path: Path) -> None:
            if path == self.outputs:
                raise OSError("synthetic temporary-directory fsync failure")
            real_fsync(path)

        with mock.patch.object(
            pipeline, "_fsync_directory", side_effect=fail_output_directory
        ), self.assertRaisesRegex(
            pipeline.PipelineError,
            "temporary cleanup was incomplete.*fsync temporary directory",
        ):
            self.execute_pipeline()
        for path in (self.converted, self.imatrix, self.quantized, self.receipt):
            self.assertFalse(path.exists())

    def test_output_that_does_not_begin_with_gguf_fails_closed(self) -> None:
        def bad_runner(
            name: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
            capture_limit: int,
        ) -> pipeline.CommandRecord:
            record = self.successful_runner(name, argv, environment, capture_limit)
            if name == "convert_f16":
                Path(argv[argv.index("--outfile") + 1]).write_bytes(b"not-gguf")
            return record

        with self.assertRaisesRegex(pipeline.PipelineError, "not a GGUF"):
            self.execute_pipeline(runner=bad_runner)
        self.assertFalse(self.receipt.exists())
        self.assert_no_pipeline_temporaries()


class CommandCaptureTests(unittest.TestCase):
    def test_command_uses_argv_without_shell_and_bounds_captured_bytes(self) -> None:
        stdout_payload = b"0123456789"
        stderr_payload = b"error"

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(stdout_payload)
                self.stderr = io.BytesIO(stderr_payload)
                self.pid = 12345
                self.returncode: int | None = None

            def wait(self, timeout: int | None = None) -> int:
                del timeout
                self.returncode = 0
                return 0

            def poll(self) -> int | None:
                return self.returncode

        fake = FakeProcess()
        with (
            mock.patch.object(
                pipeline.subprocess, "Popen", return_value=fake
            ) as popen,
            mock.patch.object(
                pipeline.os, "killpg", side_effect=ProcessLookupError
            ),
        ):
            record = pipeline._run_command(
                "tiny", ("/safe/tool", "--flag", "value"), {"OFFLINE": "1"}, 4
            )
        positional, keywords = popen.call_args
        self.assertEqual(positional[0], ["/safe/tool", "--flag", "value"])
        self.assertIs(keywords["shell"], False)
        self.assertEqual(keywords["stdin"], pipeline.subprocess.DEVNULL)
        self.assertEqual(keywords["env"]["OFFLINE"], "1")
        self.assertEqual(record.stdout.bytes, len(stdout_payload))
        self.assertEqual(record.stdout.captured_bytes, 4)
        self.assertTrue(record.stdout.truncated)
        self.assertEqual(
            record.stdout.sha256,
            "sha256:" + hashlib.sha256(stdout_payload).hexdigest(),
        )
        self.assertEqual(record.stderr.bytes, len(stderr_payload))
        self.assertEqual(record.stderr.captured_bytes, 4)
        self.assertTrue(record.stderr.truncated)

    def test_git_inspection_clears_inherited_redirection_and_pins_options(self) -> None:
        payload = (pipeline.LLAMA_CPP_REVISION + "\n").encode()

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(payload)
                self.stderr = io.BytesIO()
                self.pid = 19876
                self.returncode: int | None = None

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                self.returncode = 0
                return 0

            def poll(self) -> int | None:
                return self.returncode

        fake = FakeProcess()
        repository = Path("/private/llama.cpp")
        with (
            mock.patch.dict(
                os.environ,
                {
                    "GIT_ALLOW_PROTOCOL": "file:ssh",
                    "GIT_CONFIG_COUNT": "9",
                    "GIT_DIR": "/attacker/repository",
                    "GIT_NO_LAZY_FETCH": "0",
                    "GIT_PROTOCOL_FROM_USER": "1",
                },
            ),
            mock.patch.object(
                pipeline.subprocess, "Popen", return_value=fake
            ) as popen,
            mock.patch.object(
                pipeline.os, "killpg", side_effect=ProcessLookupError
            ),
        ):
            output = pipeline._git_output(
                Path("/safe/git"), repository, ("rev-parse", "--verify", "HEAD")
            )

        self.assertEqual(output, pipeline.LLAMA_CPP_REVISION)
        positional, keywords = popen.call_args
        argv = positional[0]
        self.assertEqual(
            argv[-5:], ["-C", str(repository), "rev-parse", "--verify", "HEAD"]
        )
        for guard in (
            "core.askPass=",
            "core.fsmonitor=false",
            "core.hooksPath=/dev/null",
            "credential.helper=",
            "credential.interactive=never",
            "protocol.allow=never",
            "protocol.ext.allow=never",
        ):
            self.assertIn(guard, argv)
        self.assertNotIn("GIT_CONFIG_COUNT", keywords["env"])
        self.assertNotIn("GIT_DIR", keywords["env"])
        self.assertEqual(keywords["env"]["GIT_ALLOW_PROTOCOL"], "")
        self.assertEqual(keywords["env"]["GIT_NO_LAZY_FETCH"], "1")
        self.assertEqual(keywords["env"]["GIT_PROTOCOL_FROM_USER"], "0")
        self.assertEqual(
            {name: keywords["env"][name] for name in pipeline._GIT_ENVIRONMENT},
            pipeline._GIT_ENVIRONMENT,
        )
        self.assertIs(keywords["shell"], False)


    def test_capture_allocation_failure_cleans_spawned_group_and_pipes(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.pid = 20876

            def wait(self, timeout: float | None = None) -> int:
                if timeout is None:
                    raise AssertionError("leader wait should not run")
                return -pipeline.signal.SIGTERM

            def poll(self) -> int | None:
                return None

        fake = FakeProcess()
        group_alive = True

        def signal_group(_pid: int, signum: int) -> None:
            nonlocal group_alive
            if signum == 0:
                if group_alive:
                    return
                raise ProcessLookupError
            if signum == pipeline.signal.SIGTERM:
                group_alive = False

        with (
            mock.patch.object(pipeline.subprocess, "Popen", return_value=fake),
            mock.patch.object(
                pipeline._CaptureState,
                "create",
                side_effect=RuntimeError("synthetic capture allocation failure"),
            ),
            mock.patch.object(
                pipeline.os, "killpg", side_effect=signal_group
            ) as killpg,
            self.assertRaisesRegex(RuntimeError, "capture allocation failure"),
        ):
            pipeline._execute_argv(("/safe/tool",), {}, 4)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGTERM), killpg.mock_calls)
        self.assertTrue(fake.stdout.closed)
        self.assertTrue(fake.stderr.closed)

    def test_sigint_after_spawn_is_deferred_until_process_is_tracked(self) -> None:
        if os.name != "posix" or not hasattr(pipeline.signal, "pthread_sigmask"):
            self.skipTest("requires POSIX pthread signal masks")

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.pid = 21876

            def wait(self, timeout: float | None = None) -> int:
                if timeout is None:
                    raise AssertionError("leader wait should not run")
                return -pipeline.signal.SIGTERM

            def poll(self) -> int | None:
                return None

        fake = FakeProcess()
        group_alive = True

        def spawn_then_signal(*_args: object, **_kwargs: object) -> FakeProcess:
            os.kill(os.getpid(), pipeline.signal.SIGINT)
            return fake

        def signal_group(_pid: int, signum: int) -> None:
            nonlocal group_alive
            if signum == 0:
                if group_alive:
                    return
                raise ProcessLookupError
            if signum == pipeline.signal.SIGTERM:
                group_alive = False

        with (
            mock.patch.object(
                pipeline.subprocess, "Popen", side_effect=spawn_then_signal
            ),
            mock.patch.object(
                pipeline.os, "killpg", side_effect=signal_group
            ) as killpg,
            self.assertRaises(KeyboardInterrupt),
        ):
            pipeline._execute_argv(("/safe/tool",), {}, 4)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGTERM), killpg.mock_calls)
        self.assertTrue(fake.stdout.closed)
        self.assertTrue(fake.stderr.closed)

    def test_lingering_descendant_streams_are_bounded_and_process_group_is_stopped(
        self,
    ) -> None:
        class BlockingStream:
            def __init__(self) -> None:
                self.closed = threading.Event()

            def read(self, _size: int) -> bytes:
                self.closed.wait(1.0)
                return b""

            def close(self) -> None:
                self.closed.set()

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = BlockingStream()
                self.stderr = BlockingStream()
                self.pid = 23456
                self.returncode = 0

            def wait(self, timeout: int | None = None) -> int:
                del timeout
                return 0

            def poll(self) -> int:
                return 0

        fake = FakeProcess()
        group_alive = True

        def signal_group(_pid: int, signum: int) -> None:
            nonlocal group_alive
            if signum == 0:
                if group_alive:
                    return
                raise ProcessLookupError
            if signum == pipeline.signal.SIGKILL:
                group_alive = False
                fake.stdout.closed.set()
                fake.stderr.closed.set()

        started = time.monotonic()
        with (
            mock.patch.object(pipeline.subprocess, "Popen", return_value=fake),
            mock.patch.object(pipeline, "_CAPTURE_JOIN_TIMEOUT_SECONDS", 0.01),
            mock.patch.object(
                pipeline, "_CAPTURE_KILL_JOIN_TIMEOUT_SECONDS", 0.01
            ),
            mock.patch.object(pipeline, "_PROCESS_GROUP_GRACE_SECONDS", 0.01),
            mock.patch.object(pipeline, "_PROCESS_GROUP_POLL_SECONDS", 0.001),
            mock.patch.object(
                pipeline.os, "killpg", side_effect=signal_group
            ) as killpg,
            self.assertRaisesRegex(
                pipeline.PipelineError, "left descendant processes"
            ),
        ):
            pipeline._execute_argv(("/safe/tool",), {}, 4)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGTERM), killpg.mock_calls)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGKILL), killpg.mock_calls)


    def test_lingering_descendant_that_closed_streams_is_stopped(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO(b"done")
                self.stderr = io.BytesIO()
                self.pid = 34567

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

            def poll(self) -> int:
                return 0

        fake = FakeProcess()
        group_alive = True

        def signal_group(_pid: int, signum: int) -> None:
            nonlocal group_alive
            if signum == 0:
                if group_alive:
                    return
                raise ProcessLookupError
            if signum == pipeline.signal.SIGKILL:
                group_alive = False

        with (
            mock.patch.object(pipeline.subprocess, "Popen", return_value=fake),
            mock.patch.object(pipeline, "_PROCESS_GROUP_GRACE_SECONDS", 0.01),
            mock.patch.object(pipeline, "_PROCESS_GROUP_POLL_SECONDS", 0.001),
            mock.patch.object(
                pipeline.os, "killpg", side_effect=signal_group
            ) as killpg,
            self.assertRaisesRegex(
                pipeline.PipelineError, "left descendant processes"
            ),
        ):
            pipeline._execute_argv(("/safe/tool",), {}, 4)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGTERM), killpg.mock_calls)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGKILL), killpg.mock_calls)
        self.assertTrue(fake.stdout.closed)
        self.assertTrue(fake.stderr.closed)

    def test_interrupted_leader_exit_after_term_still_kills_descendants(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.pid = 45678

            def wait(self, timeout: float | None = None) -> int:
                if timeout is None:
                    raise KeyboardInterrupt
                return -pipeline.signal.SIGTERM

            def poll(self) -> int | None:
                return None

        fake = FakeProcess()
        group_alive = True

        def signal_group(_pid: int, signum: int) -> None:
            nonlocal group_alive
            if signum == 0:
                if group_alive:
                    return
                raise ProcessLookupError
            if signum == pipeline.signal.SIGKILL:
                group_alive = False

        with (
            mock.patch.object(pipeline.subprocess, "Popen", return_value=fake),
            mock.patch.object(pipeline, "_PROCESS_GROUP_GRACE_SECONDS", 0.01),
            mock.patch.object(pipeline, "_PROCESS_GROUP_POLL_SECONDS", 0.001),
            mock.patch.object(
                pipeline.os, "killpg", side_effect=signal_group
            ) as killpg,
            self.assertRaises(KeyboardInterrupt),
        ):
            pipeline._execute_argv(("/safe/tool",), {}, 4)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGTERM), killpg.mock_calls)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGKILL), killpg.mock_calls)
        self.assertTrue(fake.stdout.closed)
        self.assertTrue(fake.stderr.closed)

    def test_second_capture_thread_start_failure_cleans_group_and_fds(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.pid = 56789

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

            def poll(self) -> int | None:
                return None

        fake = FakeProcess()
        group_alive = True
        starts = 0
        real_start = threading.Thread.start

        def fail_second_start(thread: threading.Thread) -> None:
            nonlocal starts
            starts += 1
            if starts == 2:
                raise RuntimeError("synthetic second start failure")
            real_start(thread)

        def signal_group(_pid: int, signum: int) -> None:
            nonlocal group_alive
            if signum == 0:
                if group_alive:
                    return
                raise ProcessLookupError
            if signum == pipeline.signal.SIGKILL:
                group_alive = False

        with (
            mock.patch.object(pipeline.subprocess, "Popen", return_value=fake),
            mock.patch.object(
                pipeline.threading.Thread, "start", new=fail_second_start
            ),
            mock.patch.object(pipeline, "_PROCESS_GROUP_GRACE_SECONDS", 0.01),
            mock.patch.object(pipeline, "_PROCESS_GROUP_POLL_SECONDS", 0.001),
            mock.patch.object(
                pipeline.os, "killpg", side_effect=signal_group
            ) as killpg,
            self.assertRaisesRegex(RuntimeError, "second start failure"),
        ):
            pipeline._execute_argv(("/safe/tool",), {}, 4)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGTERM), killpg.mock_calls)
        self.assertIn(mock.call(fake.pid, pipeline.signal.SIGKILL), killpg.mock_calls)
        self.assertTrue(fake.stdout.closed)
        self.assertTrue(fake.stderr.closed)

if __name__ == "__main__":
    unittest.main()
