from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import convert_code_gguf as converter
from training import evaluate_code_gguf as evaluator
from training import publish_code_provenance as provenance


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _runtime_closure(
    mode: str = "0755",
    *,
    root: Path = converter.LLAMA_CPP_ROOT,
) -> dict[str, object]:
    namespace_names = {
        Path(relative).name
        for relative, _size, _digest_value in converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
    }
    for loader, target, _size, _digest_value in converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT:
        namespace_names.update((Path(loader).name, Path(target).name))
    for relative, target in converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT:
        namespace_names.update((Path(relative).name, target))
    return {
        "schema": converter.RUNTIME_LIBRARY_SCHEMA,
        "root": str(root),
        "directories": [{"path": path, "mode": mode} for path in (".", "build", "build/bin")],
        "build_bin_namespace": [f"build/bin/{name}" for name in sorted(namespace_names)],
        "symlinks": [
            {"path": relative, "target": target}
            for relative, target in converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT
        ],
        "executables": [
            {
                "path": relative,
                "bytes": size,
                "digest": digest,
                "mode": mode,
            }
            for relative, size, digest in converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT
        ],
        "libraries": [
            {
                "loader_path": loader,
                "target_path": target,
                "bytes": size,
                "digest": digest,
                "mode": mode,
            }
            for loader, target, size, digest in converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT
        ],
    }


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


def _contract_gguf(
    metadata: list[tuple[str, int, object]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    *,
    version: int = 3,
) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def metadata_value(kind: int, value: object) -> bytes:
        if kind == 4:
            return struct.pack("<I", int(value))
        if kind == 8:
            return string(str(value))
        if kind == 9:
            items = list(value)
            return b"".join(
                (
                    struct.pack("<I", 8),
                    struct.pack("<Q", len(items)),
                    *(string(str(item)) for item in items),
                )
            )
        raise AssertionError(f"unsupported synthetic metadata type: {kind}")

    header = b"".join(
        (
            b"GGUF",
            struct.pack("<I", version),
            struct.pack("<Q", len(tensors)),
            struct.pack("<Q", len(metadata)),
            *(
                string(key) + struct.pack("<I", kind) + metadata_value(kind, value)
                for key, kind, value in metadata
            ),
        )
    )
    descriptors: list[bytes] = []
    offset = 0
    data: list[tuple[int, bytes]] = []
    for name, dimensions, tensor_type, raw in tensors:
        descriptors.append(
            b"".join(
                (
                    string(name),
                    struct.pack("<I", len(dimensions)),
                    *(struct.pack("<Q", dimension) for dimension in dimensions),
                    struct.pack("<I", tensor_type),
                    struct.pack("<Q", offset),
                )
            )
        )
        data.append((offset, raw))
        offset = (offset + len(raw) + 31) // 32 * 32
    prefix = header + b"".join(descriptors)
    prefix += b"\0" * ((-len(prefix)) % 32)
    payload = bytearray(prefix)
    data_start = len(prefix)
    for tensor_offset, raw in data:
        required = data_start + tensor_offset
        if len(payload) < required:
            payload.extend(b"\0" * (required - len(payload)))
        payload.extend(raw)
    return bytes(payload)


def _f16_gguf() -> bytes:
    return _contract_gguf(
        [("general.architecture", 8, "qwen3"), ("general.file_type", 4, 1)],
        [("token_embd.weight", (1,), 0, struct.pack("<f", 1.0))],
    )


def _imatrix_gguf(
    *,
    dataset: str = "calibration.txt",
    paired: bool = True,
    chunks: int = 128,
    version: int = 3,
    sums_dimensions: tuple[int, ...] = (2,),
    counts_dimensions: tuple[int, ...] = (1,),
    count_value: float = 65536.0,
) -> bytes:
    sums_values = math.prod(sums_dimensions)
    counts_values = math.prod(counts_dimensions)
    tensors = [
        (
            "blk.0.attn_q.weight.in_sum2",
            sums_dimensions,
            0,
            struct.pack(
                f"<{sums_values}f",
                *(float(index + 1) for index in range(sums_values)),
            ),
        ),
    ]
    if paired:
        tensors.append(
            (
                "blk.0.attn_q.weight.counts",
                counts_dimensions,
                0,
                struct.pack(
                    f"<{counts_values}f",
                    *([count_value] * counts_values),
                ),
            )
        )
    return _contract_gguf(
        [
            ("general.type", 8, "imatrix"),
            ("imatrix.datasets", 9, [dataset]),
            ("imatrix.chunk_count", 4, chunks),
            ("imatrix.chunk_size", 4, 512),
        ],
        tensors,
        version=version,
    )


def _calibrated_model_gguf(
    *,
    entries: int = 1,
    imatrix_file: str = "calibration.imatrix.gguf",
    replay_variant: bool = False,
) -> bytes:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "qwen3"),
        ("general.file_type", 4, 15),
        ("quantize.imatrix.file", 8, imatrix_file),
        ("quantize.imatrix.dataset", 8, "calibration.txt"),
        ("quantize.imatrix.entries_count", 4, entries),
        ("quantize.imatrix.chunks_count", 4, 128),
    ]
    if replay_variant:
        metadata.append(("synthetic.replay_variant", 8, "different bytes"))
    return _contract_gguf(
        metadata,
        [("token_embd.weight", (1,), 0, struct.pack("<f", 1.0))],
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
            "runtime_libraries": _runtime_closure(),
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
            self.assertNotIn("PYTHONPATH", environment)
            self.assertEqual(environment["PYTHONHASHSEED"], "0")
            self.assertEqual(environment["PYTHONPYCACHEPREFIX"], ".microtensor-empty-pycache")
            self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
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

    def test_generic_runtime_closure_mutation_at_prepublish_refuses_bundle(self) -> None:
        changed_tools = copy.deepcopy(self.tools)
        changed_tools["runtime_libraries"]["libraries"][0]["digest"] = _digest("9")
        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                side_effect=(self.tools, self.tools, changed_tools),
            ) as toolchain,
            mock.patch.object(
                converter.subprocess,
                "run",
                side_effect=self.completed_process,
            ),
            self.assertRaisesRegex(
                converter.ConversionRefused,
                "immediately before publication changed",
            ),
        ):
            converter.convert(self.request)
        self.assertEqual(toolchain.call_count, 3)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])


class PythonImportSurfaceLifecycleTests(ConversionFixture):
    def _inject_extension(self) -> Path:
        package = self.checkout / "conversion"
        package.mkdir(exist_ok=True)
        extension = package / "injected.so"
        extension.write_bytes(b"ignored extension fixture")
        return extension

    def _guarded_toolchain(self, _request: converter.ConversionRequest) -> dict[str, object]:
        converter._assert_llama_python_import_surface(self.checkout)
        return self.tools

    def _assert_unpublished_cleanup(self) -> None:
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])

    def test_preflight_extension_is_refused_before_any_tool_pass(self) -> None:
        self._inject_extension()
        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                side_effect=self._guarded_toolchain,
            ) as toolchain,
            mock.patch.object(converter.subprocess, "run") as process,
            self.assertRaisesRegex(converter.ConversionRefused, "extension-module candidate"),
        ):
            converter.convert(self.request)
        self.assertEqual(toolchain.call_count, 1)
        process.assert_not_called()
        self._assert_unpublished_cleanup()

    def test_post_tool_pass_extension_is_refused_by_final_recheck(self) -> None:
        def injecting_process(
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            completed = self.completed_process(argv, **kwargs)
            if str(argv[0]) == str(self.quantizer.resolve()):
                self._inject_extension()
            return completed

        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                side_effect=self._guarded_toolchain,
            ) as toolchain,
            mock.patch.object(
                converter.subprocess,
                "run",
                side_effect=injecting_process,
            ) as process,
            self.assertRaisesRegex(converter.ConversionRefused, "extension-module candidate"),
        ):
            converter.convert(self.request)
        self.assertEqual(toolchain.call_count, 2)
        self.assertEqual(process.call_count, 2)
        self._assert_unpublished_cleanup()

    def test_prepublish_extension_is_refused_by_third_recheck(self) -> None:
        calls = 0

        def inject_after_final(
            request: converter.ConversionRequest,
        ) -> dict[str, object]:
            nonlocal calls
            converter._assert_llama_python_import_surface(request.llama_cpp)
            calls += 1
            if calls == 2:
                self._inject_extension()
            return self.tools

        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                side_effect=inject_after_final,
            ) as toolchain,
            mock.patch.object(
                converter.subprocess,
                "run",
                side_effect=self.completed_process,
            ),
            self.assertRaisesRegex(converter.ConversionRefused, "extension-module candidate"),
        ):
            converter.convert(self.request)
        self.assertEqual(toolchain.call_count, 3)
        self.assertEqual(calls, 2)
        self._assert_unpublished_cleanup()


class CalibratedConversionTests(ConversionFixture):
    def setUp(self) -> None:
        super().setUp()
        self.imatrix_tool = self.checkout / "build" / "bin" / "llama-imatrix"
        self.imatrix_tool.write_text("pinned imatrix fixture\n", encoding="utf-8")
        self.imatrix_tool.chmod(0o700)
        self.current_dataset = self.root / "current-dataset"
        self.current_dataset.mkdir()
        self.current_source = self.root / "current-source.json"
        self.current_source.write_text("{}\n", encoding="utf-8")
        self.request = converter.replace(
            self.request,
            quantization="Q4_K_M",
            calibration_profile=converter.CALIBRATION_PROFILE,
            calibration_current_dataset=self.current_dataset,
            calibration_current_source_corpus=self.current_source,
            imatrix_tool=self.imatrix_tool,
        )
        self.calibrated_tools = copy.deepcopy(self.tools)
        self.calibrated_tools["imatrix"] = {
            "path": str(self.imatrix_tool.resolve()),
            "digest": _digest("f"),
        }
        self.rows = [
            {
                "ref": f"synthetic-{index:03d}",
                "prompt": f"PROMPT-SENTINEL-{index}-λ\n",
                "completion": f"COMPLETION-SENTINEL-{index}\n",
                "max_output_tokens": 1024,
            }
            for index in range(converter.CALIBRATION_TOTAL_ROWS)
        ]
        self.material = {
            "profile": converter.CALIBRATION_PROFILE,
            "source": {
                "current": {"corpus": {"digest": _digest("1")}},
                "historical": {"corpus": {"digest": _digest("2")}},
            },
            "selection": {
                "current_rows": converter.CALIBRATION_CURRENT_ROWS,
                "historical_selected_rows": converter.CALIBRATION_HISTORICAL_ROWS,
                "total_rows": converter.CALIBRATION_TOTAL_ROWS,
            },
        }
        self.calibrated_calls: list[tuple[list[str], Path, Path, str]] = []
        self.rendered_corpora: list[bytes] = []
        self.quantized_models: list[bytes] = []
        self.imatrix_bytes = _imatrix_gguf()
        self.model_bytes = _calibrated_model_gguf()
        self.replay_model_bytes = self.model_bytes

    def bounded_command(
        self,
        name: str,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        log_root: Path,
        cwd_role: str = "private_staging",
    ) -> dict[str, object]:
        exact = [str(item) for item in argv]
        self.calibrated_calls.append((exact, cwd, log_root, cwd_role))
        if exact[0] == str(self.converter.resolve()):
            (cwd / exact[exact.index("--outfile") + 1]).write_bytes(_f16_gguf())
        elif exact[0] == str(self.imatrix_tool.resolve()):
            self.rendered_corpora.append((cwd / converter.CALIBRATION_CORPUS_NAME).read_bytes())
            (cwd / exact[exact.index("--output") + 1]).write_bytes(self.imatrix_bytes)
        elif exact[0] == str(self.quantizer.resolve()):
            output_bytes = (
                self.model_bytes if not self.quantized_models else self.replay_model_bytes
            )
            (cwd / exact[4]).write_bytes(output_bytes)
            self.quantized_models.append((cwd / exact[4]).read_bytes())
        else:
            self.fail(f"unexpected calibrated executable: {exact[0]}")
        empty_digest = "sha256:" + hashlib.sha256(b"").hexdigest()
        for stream_name in ("stdout", "stderr"):
            path = log_root / f"{name}.{stream_name}"
            path.write_bytes(b"")
            path.chmod(0o600)
        stream = {
            "bytes": 0,
            "captured_bytes": 0,
            "captured_digest": empty_digest,
            "digest": empty_digest,
            "truncated": False,
        }
        return {
            "name": name,
            "argv": exact,
            "cwd_role": cwd_role,
            "environment": converter._small_child_environment(single_thread=True),
            "returncode": 0,
            "started_at_unix_ns": len(self.calibrated_calls),
            "finished_at_unix_ns": len(self.calibrated_calls),
            "stdout": dict(stream),
            "stderr": dict(stream),
        }

    def run_calibrated(self) -> dict[str, object]:
        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                return_value=self.calibrated_tools,
            ),
            mock.patch.object(
                converter,
                "_load_calibration_material",
                return_value=(self.rows, self.material),
            ),
            mock.patch.object(
                converter,
                "_bounded_conversion_command",
                side_effect=self.bounded_command,
            ),
        ):
            return converter.convert(self.request)

    def test_calibrated_q4_exact_relative_argv_receipts_and_cleanup(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "must-not-reach-child",
                "BT_WALLET_PATH": "/secret",
                "HOME": "/also-secret",
            },
        ):
            result = self.run_calibrated()

        self.assertEqual(result["calibration_profile"], converter.CALIBRATION_PROFILE)
        self.assertEqual(
            converter._bundle_file_set(self.output),
            frozenset(
                {
                    "artifact/model.gguf",
                    converter.LOAD_SPEC_NAME,
                    converter.CALIBRATION_RECEIPT_NAME,
                    converter.RECEIPT_NAME,
                }
            ),
        )
        self.assertEqual(len(self.calibrated_calls), 6)
        convert_argv, convert_cwd, _logs, convert_role = self.calibrated_calls[0]
        imatrix_argv, imatrix_cwd, _logs, imatrix_role = self.calibrated_calls[1]
        quantize_argv, quantize_cwd, _logs, quantize_role = self.calibrated_calls[2]
        self.assertEqual(convert_cwd, imatrix_cwd)
        self.assertEqual(imatrix_cwd, quantize_cwd)
        self.assertEqual(
            (convert_role, imatrix_role, quantize_role),
            ("private_staging", "private_staging", "private_staging"),
        )
        replay_calls = self.calibrated_calls[3:]
        self.assertEqual(
            [call[0] for call in replay_calls], [call[0] for call in self.calibrated_calls[:3]]
        )
        self.assertTrue(all(call[1] == replay_calls[0][1] for call in replay_calls))
        self.assertNotEqual(replay_calls[0][1], convert_cwd)
        self.assertTrue(all(call[3] == "determinism_replay" for call in replay_calls))
        self.assertEqual(convert_argv[3], converter.F16_NAME)
        self.assertEqual(
            imatrix_argv,
            [
                str(self.imatrix_tool.resolve()),
                "--offline",
                "--model",
                converter.F16_NAME,
                "--file",
                converter.CALIBRATION_CORPUS_NAME,
                "--output",
                converter.IMATRIX_NAME,
                "--output-format",
                "gguf",
                "--ctx-size",
                "512",
                "--chunks",
                "128",
                "--batch-size",
                "512",
                "--ubatch-size",
                "512",
                "--threads",
                "1",
                "--threads-batch",
                "1",
                "--device",
                "none",
                "--gpu-layers",
                "0",
                "--fit",
                "off",
                "--flash-attn",
                "off",
                "--no-ppl",
                "--parse-special",
                "--output-frequency",
                "129",
                "--save-frequency",
                "0",
            ],
        )
        self.assertEqual(
            quantize_argv,
            [
                str(self.quantizer.resolve()),
                "--imatrix",
                converter.IMATRIX_NAME,
                converter.F16_NAME,
                "artifact/model.gguf",
                "Q4_K_M",
                "1",
            ],
        )
        expected_corpus = b"".join(
            (row["prompt"] + row["completion"] + converter.CALIBRATION_EOS_TOKEN + "\n").encode(
                "utf-8"
            )
            for row in self.rows
        )
        self.assertEqual(self.rendered_corpora, [expected_corpus, expected_corpus])
        self.assertEqual(self.rendered_corpora[0].count(b"<|im_end|>\n"), 512)
        self.assertEqual(len(self.quantized_models), 2)
        self.assertEqual(self.quantized_models[0], self.quantized_models[1])
        self.assertEqual(
            hashlib.sha256(self.quantized_models[0]).hexdigest(),
            hashlib.sha256(self.quantized_models[1]).hexdigest(),
        )

        calibration = json.loads((self.output / converter.CALIBRATION_RECEIPT_NAME).read_bytes())
        conversion = json.loads((self.output / converter.RECEIPT_NAME).read_bytes())
        self.assertEqual(calibration["schema"], converter.CALIBRATION_SCHEMA)
        self.assertEqual(conversion["schema"], converter.CALIBRATED_CONVERSION_SCHEMA)
        receipt_commands = [
            *calibration["commands"],
            *calibration["determinism_replay"]["commands"],
        ]
        expected_environment = converter._small_child_environment(single_thread=True)
        for command in receipt_commands:
            self.assertEqual(command["environment"], expected_environment)
            self.assertNotIn("PYTHONPATH", command["environment"])
            self.assertEqual(command["environment"]["PYTHONHASHSEED"], "0")
            self.assertEqual(
                command["environment"]["PYTHONPYCACHEPREFIX"],
                ".microtensor-empty-pycache",
            )
        self.assertEqual(
            calibration["toolchain"]["runtime_libraries"],
            _runtime_closure(),
        )
        self.assertEqual(
            conversion["conversion"]["runtime_libraries"],
            _runtime_closure(),
        )
        self.assertEqual(
            conversion["calibration_receipt_digest"],
            "sha256:"
            + hashlib.sha256(
                (self.output / converter.CALIBRATION_RECEIPT_NAME).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            [command["name"] for command in conversion["conversion"]["commands"]],
            ["convert_f16", "calibrate_imatrix", "quantize"],
        )
        replay = conversion["conversion"]["determinism_replay"]
        self.assertIs(replay["matches_primary"], True)
        self.assertEqual(
            [command["argv"] for command in replay["commands"]],
            [command["argv"] for command in conversion["conversion"]["commands"]],
        )
        self.assertEqual(
            replay["entrypoint_digest"],
            conversion["artifact"]["entrypoint_digest"],
        )
        serialized = json.dumps((calibration, conversion))
        self.assertNotIn("PROMPT-SENTINEL", serialized)
        self.assertNotIn("COMPLETION-SENTINEL", serialized)
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])

    def test_malformed_imatrix_cleans_only_owned_staging(self) -> None:
        unrelated = self.root / "keep-me"
        unrelated.write_text("caller-owned\n", encoding="utf-8")
        self.imatrix_bytes = _imatrix_gguf(paired=False)
        with self.assertRaisesRegex(converter.ConversionRefused, "tensor pairs"):
            self.run_calibrated()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "caller-owned\n")

    def test_byte_different_replay_model_refuses_publication(self) -> None:
        self.replay_model_bytes = _calibrated_model_gguf(replay_variant=True)
        with self.assertRaisesRegex(converter.ConversionRefused, "determinism replay artifact"):
            self.run_calibrated()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])

    def test_model_imatrix_entry_count_mismatch_refuses_publication(self) -> None:
        self.model_bytes = _calibrated_model_gguf(entries=2)
        self.replay_model_bytes = self.model_bytes
        with self.assertRaisesRegex(converter.ConversionRefused, "entry count differs"):
            self.run_calibrated()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])

    def test_calibrated_receipts_must_bind_identical_runtime_closure(self) -> None:
        self.run_calibrated()
        calibration = json.loads((self.output / converter.CALIBRATION_RECEIPT_NAME).read_bytes())
        conversion = json.loads((self.output / converter.RECEIPT_NAME).read_bytes())
        changed_conversion = copy.deepcopy(conversion)
        changed_conversion["conversion"]["runtime_libraries"]["directories"][0]["mode"] = "0700"
        command_argv = tuple(
            (command["name"], command["argv"]) for command in conversion["conversion"]["commands"]
        )
        with self.assertRaisesRegex(
            converter.ConversionRefused,
            "bind different runtime libraries",
        ):
            converter._validate_calibrated_receipts(
                calibration_receipt=calibration,
                conversion_receipt=changed_conversion,
                calibration_digest=conversion["calibration_receipt_digest"],
                expected_calibration=calibration,
                expected_conversion=changed_conversion,
                command_argv=command_argv,
                replay_command_argv=command_argv,
            )

    def test_runtime_closure_mutation_at_prepublish_refuses_bundle(self) -> None:
        changed_tools = copy.deepcopy(self.calibrated_tools)
        changed_tools["runtime_libraries"]["libraries"][0]["digest"] = _digest("9")
        with (
            mock.patch.object(converter, "_load_lineage", return_value=self.lineage),
            mock.patch.object(
                converter,
                "_toolchain_identity",
                side_effect=(
                    self.calibrated_tools,
                    self.calibrated_tools,
                    changed_tools,
                ),
            ),
            mock.patch.object(
                converter,
                "_load_calibration_material",
                return_value=(self.rows, self.material),
            ),
            mock.patch.object(
                converter,
                "_bounded_conversion_command",
                side_effect=self.bounded_command,
            ),
            self.assertRaisesRegex(
                converter.ConversionRefused,
                "immediately before publication changed",
            ),
        ):
            converter.convert(self.request)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f"{converter._STAGING_MARKER}*")), [])


class CalibrationSelectionTests(ConversionFixture):
    def setUp(self) -> None:
        super().setUp()
        self.current_dataset = self.root / "current-dataset"
        self.current_dataset.mkdir()
        self.current_source = self.root / "current-source.json"
        self.current_source.write_text("{}\n", encoding="utf-8")
        self.current_rows = [
            {
                "ref": f"bigcodebench-{index}",
                "prompt": f"current prompt {index} λ\n",
                "completion": f"def task_func_{index}():\n    return {index}\n",
                "max_output_tokens": 1024,
            }
            for index in reversed(range(converter.CALIBRATION_CURRENT_ROWS))
        ]
        self.holdout_rows = [
            {
                "ref": f"bigcodebench-{1000 + index}",
                "prompt": f"diagnostic prompt {index}\n",
                "completion": f"def task_func_{index}():\n    return None\n",
                "max_output_tokens": 1024,
            }
            for index in range(converter.CALIBRATION_DIAGNOSTIC_ROWS)
        ]
        self.historical_rows = [
            {
                "ref": f"mgc-{index}",
                "prompt": f"historical prompt {index}\n",
                "completion": f"class Solution{index}:\n    pass\n",
                "max_output_tokens": 1024,
            }
            for index in reversed(range(8000))
        ]
        self.current_manifest = {
            "seed": 92,
            "train_examples": 78,
            "holdout_examples": 16,
            "source_file_digest": _digest("3"),
        }
        self.historical_manifest = {
            "seed": 92,
            "train_examples": 8000,
            "holdout_examples": 0,
        }
        self.request = converter.replace(
            self.request,
            quantization="Q4_K_M",
            calibration_profile=converter.CALIBRATION_PROFILE,
            calibration_current_dataset=self.current_dataset,
            calibration_current_source_corpus=self.current_source,
            imatrix_tool=self.quantizer,
        )

    def material_patches(self):
        validation = SimpleNamespace(
            canonical_bytes=10,
            canonical_digest=_digest("4"),
            task_count=94,
            refs_digest=_digest("5"),
        )
        return (
            mock.patch.object(
                converter.candidate,
                "assert_tmpfs_path",
                side_effect=lambda path, *, must_exist: Path(path).resolve(strict=must_exist),
            ),
            mock.patch.object(
                converter.candidate,
                "load_public_corpus",
                return_value=({}, validation),
            ),
            mock.patch.object(
                converter.candidate,
                "load_prepared_dataset",
                return_value=(self.current_rows, self.current_manifest),
            ),
            mock.patch.object(
                converter.candidate,
                "_load_prepared_rows",
                return_value=self.holdout_rows,
            ),
            mock.patch.object(
                converter.gguf,
                "_replay_current94",
                return_value=(self.current_rows, self.holdout_rows),
            ),
            mock.patch.object(
                converter.historical_candidate,
                "load_prepared_dataset",
                return_value=(self.historical_rows, self.historical_manifest),
            ),
            mock.patch.object(converter, "_strict_current_jsonl_matches"),
            mock.patch.object(
                converter,
                "_content_identity",
                return_value={"bytes": 3, "digest": _digest("3")},
            ),
            mock.patch.object(
                converter,
                "_dataset_identity",
                side_effect=lambda _root, manifest, label: {
                    "label": label,
                    "manifest": dict(manifest),
                },
            ),
        )

    def load_material(self):
        with ExitStack() as stack:
            for patch in self.material_patches():
                stack.enter_context(patch)
            return converter._load_calibration_material(self.request)

    def test_exact_78_plus_seed_ranked_434_selection(self) -> None:
        rows, snapshot = self.load_material()
        ranked = sorted(
            self.historical_rows,
            key=lambda row: (
                hashlib.sha256(f"92:{row['ref']}".encode()).hexdigest(),
                row["ref"],
            ),
        )[:434]
        expected = sorted(self.current_rows, key=lambda row: row["ref"]) + ranked
        self.assertEqual([row["ref"] for row in rows], [row["ref"] for row in expected])
        self.assertEqual(snapshot["selection"]["current_rows"], 78)
        self.assertEqual(snapshot["selection"]["diagnostic_rows_excluded"], 16)
        self.assertEqual(snapshot["selection"]["historical_selected_rows"], 434)
        self.assertEqual(snapshot["selection"]["total_rows"], 512)
        selected_refs = [row["ref"] for row in ranked]
        expected_digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    selected_refs,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        self.assertEqual(
            snapshot["selection"]["historical_selected_refs_digest"],
            expected_digest,
        )
        self.assertTrue(
            set(snapshot["source"]["current"]["prepared_dataset"]["manifest"])
            >= {"train_examples", "holdout_examples"}
        )

    def test_reserved_qwen_control_token_is_refused(self) -> None:
        self.current_rows[0]["completion"] += "<|im_end|>"
        with self.assertRaisesRegex(converter.ConversionRefused, "reserved Qwen control token"):
            self.load_material()


class CalibratedMetadataTests(unittest.TestCase):
    def test_imatrix_accepts_canonical_dense_and_expert_dimensions(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            imatrix = Path(temporary) / "imatrix.gguf"
            for sums_dimensions, counts_dimensions in (
                ((2,), (1,)),
                ((2, 2), (1, 2)),
            ):
                with self.subTest(
                    sums_dimensions=sums_dimensions,
                    counts_dimensions=counts_dimensions,
                ):
                    imatrix.write_bytes(
                        _imatrix_gguf(
                            sums_dimensions=sums_dimensions,
                            counts_dimensions=counts_dimensions,
                        )
                    )
                    self.assertEqual(
                        converter._validate_imatrix_gguf(imatrix)["entries_count"],
                        1,
                    )

    def test_imatrix_rejects_noncanonical_pair_dimensions(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            imatrix = Path(temporary) / "imatrix.gguf"
            for label, sums_dimensions, counts_dimensions in (
                ("rank-one sums with rank-two counts", (2,), (1, 2)),
                ("rank-two sums with rank-one counts", (2, 2), (1,)),
                ("rank-one multiple counts", (4,), (2,)),
                ("nondivisible mismatched expert axes", (2, 2), (1, 3)),
                ("reversed expert count axes", (2, 3), (3, 1)),
                ("rank-three tensors", (2, 2, 2), (1, 2, 2)),
                ("padded singleton axes", (2, 1), (1, 1)),
            ):
                with self.subTest(label=label):
                    imatrix.write_bytes(
                        _imatrix_gguf(
                            sums_dimensions=sums_dimensions,
                            counts_dimensions=counts_dimensions,
                        )
                    )
                    with self.assertRaisesRegex(
                        converter.ConversionRefused,
                        "tensor pair dimensions changed",
                    ):
                        converter._validate_imatrix_gguf(imatrix)

    def test_imatrix_rejects_version_and_count_metadata_inconsistency(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            imatrix = Path(temporary) / "imatrix.gguf"
            for raw, message in (
                (_imatrix_gguf(version=2), "version changed"),
                (_imatrix_gguf(count_value=65535.0), "maximum count"),
            ):
                with self.subTest(message=message):
                    imatrix.write_bytes(raw)
                    with self.assertRaisesRegex(converter.ConversionRefused, message):
                        converter._validate_imatrix_gguf(imatrix)

    def test_imatrix_rejects_invalid_count_values(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            imatrix = Path(temporary) / "imatrix.gguf"
            for count_value in (0.0, 65536.5, math.inf, math.nan):
                with self.subTest(count_value=count_value):
                    imatrix.write_bytes(_imatrix_gguf(count_value=count_value))
                    with self.assertRaisesRegex(
                        converter.ConversionRefused,
                        "invalid count value",
                    ):
                        converter._validate_imatrix_gguf(imatrix)

    def test_imatrix_and_final_model_metadata_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            imatrix = root / "imatrix.gguf"
            model = root / "model.gguf"
            imatrix.write_bytes(_imatrix_gguf())
            model.write_bytes(_calibrated_model_gguf())
            self.assertEqual(converter._validate_imatrix_gguf(imatrix)["entries_count"], 1)
            self.assertEqual(
                converter._validate_calibrated_model_metadata(model)["imatrix_entries_count"],
                1,
            )

            for raw, message in (
                (_imatrix_gguf(dataset="/random/private/path"), "imatrix.datasets"),
                (_imatrix_gguf(paired=False), "tensor pairs"),
                (_imatrix_gguf(chunks=127), "imatrix.chunk_count"),
                (_imatrix_gguf()[:-2], "beyond end of file"),
            ):
                with self.subTest(message=message):
                    imatrix.write_bytes(raw)
                    with self.assertRaisesRegex(converter.ConversionRefused, message):
                        converter._validate_imatrix_gguf(imatrix)
            model.write_bytes(_calibrated_model_gguf(entries=0))
            with self.assertRaisesRegex(converter.ConversionRefused, "not positive"):
                converter._validate_calibrated_model_metadata(model)
            model.write_bytes(_calibrated_model_gguf(imatrix_file="/random/imatrix.gguf"))
            with self.assertRaisesRegex(converter.ConversionRefused, "quantize.imatrix.file"):
                converter._validate_calibrated_model_metadata(model)


class UncalibratedQuantizationTests(ConversionFixture):
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


class CalibratedArgumentTests(ConversionFixture):
    def test_calibration_arguments_are_all_or_nothing_and_q4_only(self) -> None:
        partial = converter.replace(
            self.request,
            quantization="Q4_K_M",
            calibration_profile=converter.CALIBRATION_PROFILE,
        )
        full_q8 = converter.replace(
            self.request,
            calibration_profile=converter.CALIBRATION_PROFILE,
            calibration_current_dataset=self.dataset,
            calibration_current_source_corpus=self.source,
            imatrix_tool=self.quantizer,
        )
        for request, message in (
            (partial, "all-or-nothing"),
            (full_q8, "only for Q4_K_M"),
        ):
            with (
                self.subTest(message=message),
                mock.patch.object(converter, "_load_lineage") as lineage,
                self.assertRaisesRegex(converter.ConversionRefused, message),
            ):
                converter.convert(request)
            lineage.assert_not_called()


class PrivateCommandTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cwd = self.root / "private"
        self.cwd.mkdir(mode=0o700)
        self.argv = ("/synthetic/no-execution",)

    def test_preexisting_pycache_symlink_is_refused_before_launch(self) -> None:
        target = self.root / "cache-target"
        target.mkdir()
        (self.cwd / converter._CHILD_PYCACHE_PREFIX).symlink_to(
            target,
            target_is_directory=True,
        )
        with (
            mock.patch.object(converter.subprocess, "run") as process,
            self.assertRaisesRegex(converter.ConversionRefused, "pycache prefix must be absent"),
        ):
            converter._conversion_command("fixture", self.argv, cwd=self.cwd)
        process.assert_not_called()

    def test_pycache_created_by_child_is_refused_after_exit(self) -> None:
        def create_cache(
            argv: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess:
            (self.cwd / converter._CHILD_PYCACHE_PREFIX).mkdir()
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        with (
            mock.patch.object(converter.subprocess, "run", side_effect=create_cache),
            self.assertRaisesRegex(converter.ConversionRefused, "after exit.*pycache"),
        ):
            converter._conversion_command("fixture", self.argv, cwd=self.cwd)

    def test_top_level_backend_candidate_is_refused_before_launch(self) -> None:
        (self.cwd / "libggml-cpu.so").write_bytes(b"injected")
        with (
            mock.patch.object(converter.subprocess, "run") as process,
            self.assertRaisesRegex(converter.ConversionRefused, "shared-library candidate"),
        ):
            converter._conversion_command("fixture", self.argv, cwd=self.cwd)
        process.assert_not_called()

    def test_nested_hwcaps_candidate_is_refused_before_launch(self) -> None:
        hwcaps = self.cwd / "glibc-hwcaps" / "x86-64-v3"
        hwcaps.mkdir(parents=True)
        (hwcaps / "libggml-cpu.so.0").write_bytes(b"injected")
        with (
            mock.patch.object(converter.subprocess, "run") as process,
            self.assertRaisesRegex(converter.ConversionRefused, "shared-library candidate"),
        ):
            converter._conversion_command("fixture", self.argv, cwd=self.cwd)
        process.assert_not_called()

    def test_backend_created_by_child_is_refused_after_exit(self) -> None:
        def inject_backend(
            argv: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess:
            (self.cwd / "libggml-rpc.so").write_bytes(b"injected")
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        with (
            mock.patch.object(converter.subprocess, "run", side_effect=inject_backend),
            self.assertRaisesRegex(converter.ConversionRefused, "shared-library candidate"),
        ):
            converter._conversion_command("fixture", self.argv, cwd=self.cwd)


class PythonImportSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "llama.cpp"
        (self.root / "conversion" / "__pycache__").mkdir(parents=True)
        (self.root / "gguf-py" / "gguf" / "__pycache__").mkdir(parents=True)
        (self.root / "conversion" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "gguf-py" / "gguf" / "__init__.py").write_text(
            "",
            encoding="utf-8",
        )
        (self.root / "conversion" / "__pycache__" / "base.cpython-311.pyc").write_bytes(
            b"inert checkout bytecode"
        )
        (self.root / "gguf-py" / "gguf" / "__pycache__" / "lazy.cpython-311.pyc").write_bytes(
            b"inert checkout bytecode"
        )

    def test_clean_source_tree_allows_inert_pycache_and_native_build_trees(self) -> None:
        for path in (
            self.root / "build" / "bin" / "libggml-cpu.so",
            self.root / "build-cuda" / "bin" / "libggml-cuda.so",
        ):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"separately attested native build fixture")
        converter._assert_llama_python_import_surface(self.root)

    def test_extension_candidates_across_import_paths_are_refused(self) -> None:
        for relative in (
            "shadow.so",
            "shadow/__init__.so",
            "conversion/shadow.abi3.so",
            "gguf-py/gguf/shadow.cpython-311-x86_64-linux-gnu.so",
        ):
            candidate_path = self.root / relative
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_bytes(b"ignored extension fixture")
            with (
                self.subTest(relative=relative),
                self.assertRaisesRegex(
                    converter.ConversionRefused,
                    "extension-module candidate",
                ),
            ):
                converter._assert_llama_python_import_surface(self.root)
            candidate_path.unlink()

    def test_legacy_sourceless_bytecode_is_refused(self) -> None:
        (self.root / "conversion" / "shadow.pyc").write_bytes(b"legacy bytecode")
        with self.assertRaisesRegex(converter.ConversionRefused, "legacy sourceless bytecode"):
            converter._assert_llama_python_import_surface(self.root)

    def test_nested_symlink_is_refused_without_traversal(self) -> None:
        outside = self.root.parent / "outside"
        outside.mkdir()
        (outside / "shadow.so").write_bytes(b"outside")
        (self.root / "conversion" / "alias").symlink_to(
            outside,
            target_is_directory=True,
        )
        with self.assertRaisesRegex(converter.ConversionRefused, "contains a symlink"):
            converter._assert_llama_python_import_surface(self.root)

    def test_special_entry_is_refused(self) -> None:
        fifo = self.root / "conversion" / "injected"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(converter.ConversionRefused, "special filesystem entry"):
            converter._assert_llama_python_import_surface(self.root)

    def test_mutation_between_snapshots_is_refused(self) -> None:
        original_fwalk = os.fwalk
        calls = 0

        def mutate_after_first(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            yield from original_fwalk(*args, **kwargs)
            if calls == 1:
                (self.root / "conversion" / "late.py").write_text(
                    "changed between snapshots\n",
                    encoding="utf-8",
                )

        with (
            mock.patch.object(
                converter.os,
                "fwalk",
                side_effect=mutate_after_first,
            ),
            self.assertRaisesRegex(converter.ConversionRefused, "changed during inspection"),
        ):
            converter._assert_llama_python_import_surface(self.root)
        self.assertEqual(calls, 2)


class RuntimeLibraryClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "llama.cpp"
        self.bin = self.root / "build" / "bin"
        self.bin.mkdir(parents=True)
        for directory in (self.root, self.root / "build", self.bin):
            directory.chmod(0o700)
        self.raw = b"pinned shared library fixture\n"
        self.executable_raw = b"pinned executable fixture\n"
        self.target = self.bin / "libfixture.so.1.0"
        self.loader = self.bin / "libfixture.so.1"
        self.unversioned = self.bin / "libfixture.so"
        self.executable = self.bin / "fixture-tool"
        self.target.write_bytes(self.raw)
        self.target.chmod(0o700)
        self.loader.symlink_to(self.target.name)
        self.unversioned.symlink_to(self.loader.name)
        self.executable.write_bytes(self.executable_raw)
        self.executable.chmod(0o700)
        self.contract = (
            (
                "build/bin/libfixture.so.1",
                "build/bin/libfixture.so.1.0",
                len(self.raw),
                "sha256:" + hashlib.sha256(self.raw).hexdigest(),
            ),
        )
        self.symlink_contract = (
            ("build/bin/libfixture.so", "libfixture.so.1"),
            ("build/bin/libfixture.so.1", "libfixture.so.1.0"),
        )
        self.executable_contract = (
            (
                "build/bin/fixture-tool",
                len(self.executable_raw),
                "sha256:" + hashlib.sha256(self.executable_raw).hexdigest(),
            ),
        )

    def closure(self) -> dict[str, object]:
        with (
            mock.patch.object(
                converter,
                "LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT",
                self.contract,
            ),
            mock.patch.object(
                converter,
                "LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT",
                self.symlink_contract,
            ),
            mock.patch.object(
                converter,
                "LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT",
                self.executable_contract,
            ),
        ):
            return converter._runtime_library_closure(self.root)

    def test_exact_symlink_target_hash_and_modes_are_attested(self) -> None:
        closure = self.closure()
        self.assertEqual(closure["schema"], converter.RUNTIME_LIBRARY_SCHEMA)
        self.assertEqual(closure["root"], str(self.root))
        self.assertEqual(
            closure["build_bin_namespace"],
            [
                "build/bin/fixture-tool",
                "build/bin/libfixture.so",
                "build/bin/libfixture.so.1",
                "build/bin/libfixture.so.1.0",
            ],
        )
        self.assertEqual(
            closure["symlinks"],
            [{"path": path, "target": target} for path, target in self.symlink_contract],
        )
        self.assertEqual(
            closure["executables"],
            [
                {
                    "path": self.executable_contract[0][0],
                    "bytes": len(self.executable_raw),
                    "digest": self.executable_contract[0][2],
                    "mode": "0700",
                }
            ],
        )
        self.assertEqual(
            closure["directories"],
            [
                {"path": ".", "mode": "0700"},
                {"path": "build", "mode": "0700"},
                {"path": "build/bin", "mode": "0700"},
            ],
        )
        self.assertEqual(
            closure["libraries"],
            [
                {
                    "loader_path": self.contract[0][0],
                    "target_path": self.contract[0][1],
                    "bytes": len(self.raw),
                    "digest": self.contract[0][3],
                    "mode": "0700",
                }
            ],
        )

    def test_missing_target_is_refused(self) -> None:
        self.target.unlink()
        with self.assertRaisesRegex(converter.ConversionRefused, "build/bin namespace changed"):
            self.closure()

    def test_repointed_loader_is_refused(self) -> None:
        self.loader.unlink()
        self.loader.symlink_to("different.so")
        with self.assertRaisesRegex(converter.ConversionRefused, "escaped or was repointed"):
            self.closure()

    def test_repointed_unversioned_loader_is_refused(self) -> None:
        self.unversioned.unlink()
        self.unversioned.symlink_to("../outside.so")
        with self.assertRaisesRegex(converter.ConversionRefused, "escaped or was repointed"):
            self.closure()

    def test_extra_backend_changes_exact_namespace(self) -> None:
        (self.bin / "libggml-cuda.so").write_bytes(b"injected")
        with self.assertRaisesRegex(converter.ConversionRefused, "build/bin namespace changed"):
            self.closure()

    def test_nested_hwcaps_directory_changes_exact_namespace(self) -> None:
        hwcaps = self.bin / "glibc-hwcaps" / "x86-64-v3"
        hwcaps.mkdir(parents=True)
        (hwcaps / "libggml-cpu.so.0").write_bytes(b"injected")
        with self.assertRaisesRegex(converter.ConversionRefused, "build/bin namespace changed"):
            self.closure()

    def test_escaping_symlink_chain_is_refused(self) -> None:
        outside = self.root.parent / "outside.so"
        outside.write_bytes(self.raw)
        outside.chmod(0o700)
        self.target.unlink()
        self.target.symlink_to(outside)
        with self.assertRaisesRegex(converter.ConversionRefused, "final target must be regular"):
            self.closure()

    def test_hash_change_is_refused(self) -> None:
        self.target.write_bytes(b"X" * len(self.raw))
        self.target.chmod(0o700)
        with self.assertRaisesRegex(converter.ConversionRefused, "target bytes changed"):
            self.closure()

    def test_executable_hash_change_is_refused(self) -> None:
        self.executable.write_bytes(b"X" * len(self.executable_raw))
        with self.assertRaisesRegex(
            converter.ConversionRefused, "runtime executable bytes changed"
        ):
            self.closure()

    def test_late_executable_mutation_is_caught_by_final_recheck(self) -> None:
        original = converter._attest_runtime_regular_file
        mutated = False

        def mutate_after_first_attestation(
            *args: object,
            **kwargs: object,
        ) -> dict[str, object]:
            nonlocal mutated
            identity = original(*args, **kwargs)
            if kwargs.get("executable") is True and not mutated:
                mutated = True
                self.executable.write_bytes(b"X" * len(self.executable_raw))
            return identity

        with (
            mock.patch.object(
                converter,
                "_attest_runtime_regular_file",
                side_effect=mutate_after_first_attestation,
            ),
            self.assertRaisesRegex(
                converter.ConversionRefused,
                "executable final recheck bytes changed",
            ),
        ):
            self.closure()

    def test_group_writable_target_is_refused(self) -> None:
        self.target.chmod(0o770)
        with self.assertRaisesRegex(converter.ConversionRefused, "target must not be"):
            self.closure()

    def test_group_writable_directory_is_refused(self) -> None:
        self.bin.chmod(0o770)
        with self.assertRaisesRegex(converter.ConversionRefused, "build/bin must not be"):
            self.closure()


class RuntimeLibraryReceiptTests(unittest.TestCase):
    def test_production_contract_cardinality_and_names_are_exact(self) -> None:
        self.assertEqual(
            [entry[1] for entry in converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT],
            [
                "build/bin/libggml-base.so.0.22.0",
                "build/bin/libggml-cpu.so.0.22.0",
                "build/bin/libggml.so.0.22.0",
                "build/bin/libllama-common.so.0.3.0",
                "build/bin/libllama-quantize-impl.so",
                "build/bin/libllama.so.0.3.0",
            ],
        )
        self.assertEqual(
            converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT,
            (
                ("build/bin/libggml-base.so", "libggml-base.so.0"),
                ("build/bin/libggml-base.so.0", "libggml-base.so.0.22.0"),
                ("build/bin/libggml-cpu.so", "libggml-cpu.so.0"),
                ("build/bin/libggml-cpu.so.0", "libggml-cpu.so.0.22.0"),
                ("build/bin/libggml.so", "libggml.so.0"),
                ("build/bin/libggml.so.0", "libggml.so.0.22.0"),
                ("build/bin/libllama-common.so", "libllama-common.so.0"),
                ("build/bin/libllama-common.so.0", "libllama-common.so.0.3.0"),
                ("build/bin/libllama.so", "libllama.so.0"),
                ("build/bin/libllama.so.0", "libllama.so.0.3.0"),
            ),
        )
        self.assertEqual(
            [entry[0] for entry in converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT],
            [
                "build/bin/llama-imatrix",
                "build/bin/llama-quantize",
            ],
        )
        self.assertEqual(
            _runtime_closure()["build_bin_namespace"],
            [
                "build/bin/libggml-base.so",
                "build/bin/libggml-base.so.0",
                "build/bin/libggml-base.so.0.22.0",
                "build/bin/libggml-cpu.so",
                "build/bin/libggml-cpu.so.0",
                "build/bin/libggml-cpu.so.0.22.0",
                "build/bin/libggml.so",
                "build/bin/libggml.so.0",
                "build/bin/libggml.so.0.22.0",
                "build/bin/libllama-common.so",
                "build/bin/libllama-common.so.0",
                "build/bin/libllama-common.so.0.3.0",
                "build/bin/libllama-quantize-impl.so",
                "build/bin/libllama.so",
                "build/bin/libllama.so.0",
                "build/bin/libllama.so.0.3.0",
                "build/bin/llama-imatrix",
                "build/bin/llama-quantize",
            ],
        )
        self.assertEqual(len(converter.LLAMA_CPP_RUNTIME_LIBRARY_CONTRACT), 6)
        self.assertEqual(len(converter.LLAMA_CPP_RUNTIME_SYMLINK_CONTRACT), 10)
        self.assertEqual(len(converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT), 2)

    def test_exact_expanded_runtime_receipt_validates(self) -> None:
        closure = _runtime_closure()
        self.assertEqual(
            converter._validate_runtime_library_receipt(closure, "fixture"),
            closure,
        )

    def test_float_and_bool_byte_counts_are_refused(self) -> None:
        for section, message in (
            ("libraries", "library bytes must be an integer"),
            ("executables", "executable bytes must be an integer"),
        ):
            baseline = _runtime_closure()
            entries = baseline[section]
            self.assertIsInstance(entries, list)
            expected = entries[0]["bytes"]
            for value in (float(expected), True):
                changed = copy.deepcopy(baseline)
                changed[section][0]["bytes"] = value
                with (
                    self.subTest(section=section, value=value),
                    self.assertRaisesRegex(converter.ConversionRefused, message),
                ):
                    converter._validate_runtime_library_receipt(changed, "fixture")

    def test_root_namespace_and_symlink_edges_are_exact(self) -> None:
        for mutate, message in (
            (
                lambda value: value.update({"root": "/different/llama.cpp"}),
                "root changed",
            ),
            (
                lambda value: value["build_bin_namespace"].append("build/bin/libggml-cuda.so"),
                "namespace changed",
            ),
            (
                lambda value: value["symlinks"][0].update({"target": "other.so"}),
                "symlink edge changed",
            ),
        ):
            closure = copy.deepcopy(_runtime_closure())
            mutate(closure)
            with self.assertRaisesRegex(converter.ConversionRefused, message):
                converter._validate_runtime_library_receipt(closure, "fixture")

    def test_tool_identity_must_match_closure_executable(self) -> None:
        closure = _runtime_closure()
        for relative, size, digest in converter.LLAMA_CPP_BUILD_BIN_EXECUTABLE_CONTRACT:
            identity = {
                "path": str(converter.LLAMA_CPP_ROOT / relative),
                "bytes": size,
                "digest": digest,
                "mode": "0o755",
            }
            converter._bind_runtime_executable_identity(
                identity,
                closure,
                relative,
                "fixture tool",
            )
            changed = dict(identity)
            changed["digest"] = _digest("9")
            with self.assertRaisesRegex(
                converter.ConversionRefused,
                "differs from runtime closure",
            ):
                converter._bind_runtime_executable_identity(
                    changed,
                    closure,
                    relative,
                    "fixture tool",
                )


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

    def test_alternate_llama_cpp_root_is_refused_before_tool_inspection(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            checkout = root / "llama.cpp"
            checkout.mkdir()
            request = converter.ConversionRequest(
                training_run=root,
                training_dataset=root,
                source_corpus=root / "source",
                base=root,
                llama_cpp=checkout,
                converter=checkout / "convert_hf_to_gguf.py",
                quantizer=checkout / "build/bin/llama-quantize",
                output_bundle=root / "out",
                quantization="Q8_0",
                max_input_tokens=1024,
            )
            with self.assertRaisesRegex(
                converter.ConversionRefused,
                "must resolve exactly",
            ):
                converter._toolchain_identity(request)

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

            runtime_identity = _runtime_closure(root=checkout.resolve())
            quantizer_identity = evaluator.file_identity(quantizer_path, "fixture quantizer")
            runtime_executables = runtime_identity["executables"]
            self.assertIsInstance(runtime_executables, list)
            quantizer_closure = next(
                entry
                for entry in runtime_executables
                if entry["path"] == "build/bin/llama-quantize"
            )
            quantizer_closure.update(
                {
                    "bytes": quantizer_identity["bytes"],
                    "digest": quantizer_identity["digest"],
                    "mode": "0700",
                }
            )
            with (
                mock.patch.object(converter, "LLAMA_CPP_ROOT", checkout.resolve()),
                mock.patch.object(
                    converter,
                    "_assert_llama_python_import_surface",
                ) as import_surface,
                mock.patch.object(converter.shutil, "which", return_value=str(git)),
                mock.patch.object(converter, "_read_only_command", side_effect=git_result),
                mock.patch.object(
                    converter,
                    "_runtime_library_closure",
                    return_value=runtime_identity,
                ) as runtime_closure,
            ):
                identity = converter._toolchain_identity(request)
            self.assertEqual(identity["revision"], converter.LLAMA_CPP_REVISION)
            self.assertEqual(identity["converter"]["path"], str(converter_path.resolve()))
            self.assertEqual(identity["quantizer"]["path"], str(quantizer_path.resolve()))
            self.assertEqual(identity["runtime_libraries"], runtime_identity)
            import_surface.assert_called_once_with(checkout.resolve())
            runtime_closure.assert_called_once_with(checkout.resolve())
