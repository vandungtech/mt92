from __future__ import annotations

import ast
import copy
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import convert_code_gguf as converter
from training import evaluate_code_gguf as evaluator
from training import publish_code_provenance as provenance


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _gguf(file_type: int) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    return b"".join(
        (
            b"GGUF",
            struct.pack("<I", 3),
            struct.pack("<Q", 0),
            struct.pack("<Q", 2),
            string("general.architecture"),
            struct.pack("<I", 8),
            string("qwen3"),
            string("general.file_type"),
            struct.pack("<I", 4),
            struct.pack("<I", file_type),
        )
    )


class ConversionFixture(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.dataset = self.root / "dataset"
        self.base = self.root / "base"
        self.checkout = self.root / "llama.cpp"
        for directory in (self.run / "merged", self.dataset, self.base, self.checkout):
            directory.mkdir(parents=True)
        (self.run / "merged" / "config.json").write_text("{}\n", encoding="utf-8")
        self.source = self.root / "source.json"
        self.source.write_text("{}\n", encoding="utf-8")
        self.converter = self.checkout / "convert_hf_to_gguf.py"
        self.quantizer = self.checkout / "build" / "bin" / "llama-quantize"
        self.quantizer.parent.mkdir(parents=True)
        self.converter.write_text("pinned converter fixture\n", encoding="utf-8")
        self.quantizer.write_text("pinned quantizer fixture\n", encoding="utf-8")
        self.converter.chmod(0o700)
        self.quantizer.chmod(0o700)
        self.output = self.root / "q8-bundle"
        self.request = converter.ConversionRequest(
            training_run=self.run,
            training_dataset=self.dataset,
            source_corpus=self.source,
            base=self.base,
            llama_cpp=self.checkout,
            converter=self.converter,
            quantizer=self.quantizer,
            output_bundle=self.output,
            quantization="Q8_0",
            max_input_tokens=1024,
        )
        self.lineage = {
            "schema": evaluator.TRAINING_SCHEMA_V5,
            "receipt": {"digest": _digest("a")},
            "run": {
                "merged": {
                    "digest": _digest("b"),
                    "files": [
                        {
                            "path": "config.json",
                            "bytes": 3,
                            "digest": _digest("c"),
                        }
                    ],
                }
            },
        }
        self.tools = {
            "root": str(self.checkout.resolve()),
            "revision": converter.LLAMA_CPP_REVISION,
            "converter": {
                "path": str(self.converter.resolve()),
                "digest": _digest("d"),
            },
            "quantizer": {
                "path": str(self.quantizer.resolve()),
                "digest": _digest("e"),
            },
        }
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def completed_process(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        exact = [str(item) for item in argv]
        self.calls.append((exact, dict(kwargs)))
        if exact[0] == str(self.converter.resolve()):
            Path(exact[exact.index("--outfile") + 1]).write_bytes(b"temporary f16")
        elif exact[0] == str(self.quantizer.resolve()):
            file_type = evaluator.SUPPORTED_QUANTIZATIONS[exact[3]]
            Path(exact[2]).write_bytes(_gguf(file_type))
        else:
            self.fail(f"unexpected executable: {exact[0]}")
        return subprocess.CompletedProcess(exact, 0, stdout=b"ok", stderr=b"")

    def patches(self):
        return (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(converter, "_toolchain_identity", return_value=self.tools),
            mock.patch.object(converter.subprocess, "run", side_effect=self.completed_process),
        )

    def run_conversion(self) -> dict[str, object]:
        lineage, tools, process = self.patches()
        with lineage, tools, process:
            return converter.convert(self.request)


class SuccessfulConversionTests(ConversionFixture):
    def test_atomic_bundle_matches_publication_schema_and_exact_commands(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"WANDB_API_KEY": "must-not-reach-child", "BT_WALLET_PATH": "/secret"},
        ):
            result = self.run_conversion()

        self.assertEqual(result["output_bundle"], str(self.output.resolve()))
        self.assertEqual(
            converter._bundle_file_set(self.output),
            frozenset(
                {
                    "artifact/model.gguf",
                    converter.LOAD_SPEC_NAME,
                    converter.RECEIPT_NAME,
                }
            ),
        )
        self.assertFalse((self.output / converter.F16_NAME).exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])

        self.assertEqual(len(self.calls), 2)
        convert_argv, convert_options = self.calls[0]
        quantize_argv, quantize_options = self.calls[1]
        self.assertEqual(convert_argv[0], str(self.converter.resolve()))
        self.assertEqual(convert_argv[1], str((self.run / "merged").resolve()))
        self.assertEqual(convert_argv[2], "--outfile")
        self.assertEqual(convert_argv[-2:], ["--outtype", "f16"])
        self.assertEqual(quantize_argv[0], str(self.quantizer.resolve()))
        self.assertEqual(quantize_argv[1], convert_argv[3])
        self.assertEqual(quantize_argv[3], "Q8_0")
        for options in (convert_options, quantize_options):
            self.assertIs(options["shell"], False)
            self.assertIs(options["check"], False)
            environment = options["env"]
            self.assertIsInstance(environment, dict)
            self.assertNotIn("WANDB_API_KEY", environment)
            self.assertNotIn("BT_WALLET_PATH", environment)
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

        load = json.loads((self.output / converter.LOAD_SPEC_NAME).read_bytes())
        receipt = json.loads((self.output / converter.RECEIPT_NAME).read_bytes())
        self.assertEqual(load["max_input"], {"tokens": 1024})
        self.assertEqual(load["preprocessing"], {"tokenizer": "tokenizer.json"})
        self.assertEqual(receipt["schema"], provenance.CONVERSION_SCHEMA)
        self.assertIsNone(receipt["calibration_receipt_digest"])
        self.assertEqual(
            receipt["source"],
            {
                "training_metadata_digest": _digest("a"),
                "merged_tree_digest": _digest("b"),
            },
        )
        self.assertEqual(
            [command["name"] for command in receipt["conversion"]["commands"]],
            ["convert_f16", "quantize"],
        )
        self.assertEqual(
            [command["argv"] for command in receipt["conversion"]["commands"]],
            [convert_argv, quantize_argv],
        )
        self.assertTrue(
            all(command["returncode"] == 0 for command in receipt["conversion"]["commands"])
        )

        identity = evaluator.artifact_identity(
            self.output / converter.ARTIFACT_NAME,
            entrypoint=converter.ENTRYPOINT,
            expected_digest=receipt["artifact"]["tree_digest"],
            quantization="Q8_0",
        )
        provenance._validate_generic_conversion(
            receipt,
            training_lineage=self.lineage,
            artifact=identity,
            load_manifest=load,
            calibration_digest=None,
        )

    def test_q4_k_m_header_and_receipt_are_consistent(self) -> None:
        self.request = converter.replace(self.request, quantization="Q4_K_M")
        self.run_conversion()
        receipt = json.loads((self.output / converter.RECEIPT_NAME).read_bytes())
        self.assertEqual(receipt["artifact"]["quantization"], "Q4_K_M")
        identity = evaluator.artifact_identity(
            self.output / converter.ARTIFACT_NAME,
            entrypoint=converter.ENTRYPOINT,
            expected_digest=receipt["artifact"]["tree_digest"],
            quantization="Q4_K_M",
        )
        self.assertEqual(identity["entrypoint"]["gguf"]["file_type"], 15)

    def test_q5_k_m_load_manifest_receipt_and_header_are_consistent(self) -> None:
        self.request = converter.replace(self.request, quantization="Q5_K_M")
        self.run_conversion()
        load = json.loads((self.output / converter.LOAD_SPEC_NAME).read_bytes())
        receipt = json.loads((self.output / converter.RECEIPT_NAME).read_bytes())
        self.assertEqual(load["quantization"], "Q5_K_M")
        self.assertEqual(receipt["artifact"]["quantization"], "Q5_K_M")
        self.assertEqual(receipt["load_manifest"], load)
        self.assertEqual(self.calls[1][0][3], "Q5_K_M")
        identity = evaluator.artifact_identity(
            self.output / converter.ARTIFACT_NAME,
            entrypoint=converter.ENTRYPOINT,
            expected_digest=receipt["artifact"]["tree_digest"],
            quantization="Q5_K_M",
        )
        self.assertEqual(identity["entrypoint"]["gguf"]["file_type"], 17)
        provenance._validate_generic_conversion(
            receipt,
            training_lineage=self.lineage,
            artifact=identity,
            load_manifest=load,
            calibration_digest=None,
        )


class FailureTests(ConversionFixture):
    def test_quantizer_failure_publishes_nothing_and_cleans_only_staging(self) -> None:
        unrelated = self.root / "keep-me"
        unrelated.write_text("owned by caller\n", encoding="utf-8")

        def fail_quantizer(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            exact = [str(item) for item in argv]
            self.calls.append((exact, dict(kwargs)))
            if exact[0] == str(self.converter.resolve()):
                Path(exact[3]).write_bytes(b"temporary f16")
                return subprocess.CompletedProcess(exact, 0, stdout=b"", stderr=b"")
            return subprocess.CompletedProcess(exact, 9, stdout=b"", stderr=b"refused")

        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(converter, "_toolchain_identity", return_value=self.tools),
            mock.patch.object(converter.subprocess, "run", side_effect=fail_quantizer),
            self.assertRaisesRegex(converter.ConversionRefused, "return code 9"),
        ):
            converter.convert(self.request)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "owned by caller\n")

    def test_lineage_change_after_commands_refuses_publication(self) -> None:
        changed = copy.deepcopy(self.lineage)
        changed["run"]["merged"]["digest"] = _digest("f")
        with (
            mock.patch.object(converter, "_load_lineage", side_effect=(self.lineage, changed)),
            mock.patch.object(converter, "_toolchain_identity", return_value=self.tools),
            mock.patch.object(converter.subprocess, "run", side_effect=self.completed_process),
            self.assertRaisesRegex(converter.ConversionRefused, "training lineage changed"),
        ):
            converter.convert(self.request)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])

    def test_existing_output_is_refused_before_any_child_process(self) -> None:
        self.output.mkdir()
        with (
            mock.patch.object(converter, "_load_lineage") as lineage,
            mock.patch.object(converter.subprocess, "run") as process,
            self.assertRaisesRegex(converter.ConversionRefused, "already exists"),
        ):
            converter.convert(self.request)
        lineage.assert_not_called()
        process.assert_not_called()

    def test_unsupported_contract_values_are_refused(self) -> None:
        for request, message in (
            (
                converter.replace(self.request, quantization="Q6_K"),
                "Q8_0, Q5_K_M, or Q4_K_M",
            ),
            (converter.replace(self.request, max_input_tokens=511), "must be in"),
            (converter.replace(self.request, max_input_tokens=True), "must be an integer"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(converter.ConversionRefused, message),
            ):
                converter.convert(request)


class StaticSafetyTests(unittest.TestCase):
    def test_wrapper_contains_no_shell_true_or_dynamic_code_execution(self) -> None:
        source = Path(converter.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"compile", "eval", "exec"}
        }
        self.assertEqual(forbidden, set())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        self.assertIsInstance(keyword.value, ast.Constant)
                        self.assertIs(keyword.value.value, False)

    def test_toolchain_requires_exact_paths_revision_and_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            checkout = root / "llama.cpp"
            converter_path = checkout / "convert_hf_to_gguf.py"
            quantizer_path = checkout / "build" / "bin" / "llama-quantize"
            git = root / "git"
            quantizer_path.parent.mkdir(parents=True)
            for path in (converter_path, quantizer_path, git):
                path.write_text("fixture\n", encoding="utf-8")
                path.chmod(0o700)
            request = converter.ConversionRequest(
                training_run=root,
                training_dataset=root,
                source_corpus=root / "source",
                base=root,
                llama_cpp=checkout,
                converter=converter_path,
                quantizer=quantizer_path,
                output_bundle=root / "out",
                quantization="Q8_0",
                max_input_tokens=1024,
            )

            def git_result(argv: tuple[str, ...], *, cwd: Path) -> bytes:
                self.assertEqual(cwd, checkout.resolve())
                if argv[-1] == "--show-toplevel":
                    return f"{checkout.resolve()}\n".encode()
                if argv[-1] == "HEAD":
                    return f"{converter.LLAMA_CPP_REVISION}\n".encode()
                return b""

            with (
                mock.patch.object(converter.shutil, "which", return_value=str(git)),
                mock.patch.object(converter, "_read_only_command", side_effect=git_result),
            ):
                identity = converter._toolchain_identity(request)
            self.assertEqual(identity["revision"], converter.LLAMA_CPP_REVISION)
            self.assertEqual(identity["converter"]["path"], str(converter_path.resolve()))
            self.assertEqual(identity["quantizer"]["path"], str(quantizer_path.resolve()))
