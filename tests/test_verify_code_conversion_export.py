from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from training import verify_code_conversion_export as verifier

ROOT = Path(__file__).resolve().parents[1]
SPEC_TEMPLATE = ROOT / "deploy/conversion-worker/worker-spec.current94-v8.json"
MANIFEST_TEMPLATE = ROOT / "deploy/conversion-worker/input-manifest.current94-v8.json"
EMPTY_DIGEST = "sha256:" + hashlib.sha256(b"").hexdigest()


def _digest(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"bytes": len(raw), "digest": _digest(raw)}


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def _write_json(path: Path, value: Any) -> None:
    _write(
        path,
        (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _synthetic_gguf(
    *,
    architecture: str = "qwen2",
    file_type: int = 15,
    version: int = 3,
    entries: int = 7,
) -> bytes:
    metadata: list[tuple[str, int, str | int]] = [
        ("general.alignment", 4, 32),
        ("general.architecture", 8, architecture),
        ("general.file_type", 4, file_type),
        ("quantize.imatrix.chunks_count", 4, 128),
        ("quantize.imatrix.dataset", 8, "calibration.txt"),
        ("quantize.imatrix.entries_count", 4, entries),
        ("quantize.imatrix.file", 8, "calibration.imatrix.gguf"),
    ]
    raw = bytearray(struct.pack("<4sIQQ", b"GGUF", version, 1, len(metadata)))
    for key, kind, value in metadata:
        raw.extend(_gguf_string(key))
        raw.extend(struct.pack("<I", kind))
        if kind == 8:
            raw.extend(_gguf_string(str(value)))
        else:
            raw.extend(struct.pack("<I", int(value)))
    raw.extend(_gguf_string("weight"))
    raw.extend(struct.pack("<I", 1))
    raw.extend(struct.pack("<Q", 1))
    raw.extend(struct.pack("<I", 0))
    raw.extend(struct.pack("<Q", 0))
    raw.extend(b"\0" * (-len(raw) % 32))
    raw.extend(b"\0")
    return bytes(raw)


def _synthetic_static_elf(*, dynamic: bool = False) -> bytes:
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        2,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        1,
        0,
        0,
        0,
    )
    program = struct.pack(
        "<IIQQQQQQ",
        2 if dynamic else 1,
        5,
        0,
        0,
        0,
        120,
        120,
        4096,
    )
    return header + program


def _stream() -> dict[str, Any]:
    return {
        "bytes": 0,
        "captured_bytes": 0,
        "captured_digest": EMPTY_DIGEST,
        "digest": EMPTY_DIGEST,
        "truncated": False,
    }


def _command(expected: dict[str, Any], environment: dict[str, str], role: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": expected["name"],
        "argv": expected["argv"],
        "cwd_role": "private_staging" if role == "primary" else "determinism_replay",
        "environment": environment,
        "returncode": 0,
        "started_at_unix_ns": 1,
        "finished_at_unix_ns": 2,
        "stdout": _stream(),
        "stderr": _stream(),
    }
    if expected["name"] == "convert_f16":
        result["launch"] = {
            "method": "proc-self-fd",
            "executed_object": {
                "portable": {
                    "path": "/opt/python/bin/python3.11",
                    "bytes": 21_333_768,
                    "digest": (
                        "sha256:"
                        "96d1b01675f2492922ec6f6ed8445791d2d3231ccae727cda521db30494b751e"
                    ),
                    "mode": "0o755",
                },
                "worker_observation": {
                    "device": 1,
                    "inode": 2,
                    "mtime_ns": 3,
                    "ctime_ns": 4,
                },
            },
        }
    return result


def _runtime_receipt(spec: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(spec["runtime_library_contract"])
    runtime["directories"] = [
        {"mode": "0755", "path": "."},
        {"mode": "0755", "path": "build"},
        {"mode": "0755", "path": "build/bin"},
    ]
    return runtime


def _conversion_source() -> dict[str, Any]:
    return {
        "training_schema": "microtensor.code.training.v4",
        "dataset_schema": "microtensor.code.prepared.v1",
        "corpus_profile": "bigcodebench94",
        "training_metadata_digest": (
            "sha256:1e983beff4f32f574a57352b61c2e4f29d9a4922d59d71b1b722902255a3ef10"
        ),
        "training_metrics_digest": (
            "sha256:1c2947a3bed290d01880698b144331ef4f148368634514ef7de396d90d67169e"
        ),
        "merged_tree_digest": (
            "sha256:5b05fe2ec5c145c5f88c28acfb5ab37a6c724816188a7022284d8581b0d356ee"
        ),
        "source_corpus": {
            "bytes": 152_605,
            "digest": (
                "sha256:1c37a0e212936bfac8c86f955ad61fd378f58603413b45ece88382d528ace9d5"
            ),
            "canonical_digest": (
                "sha256:f126ea986aeeb45eecb3a63e850bbe2f6572c01d24142eed639b2dfbddcea4cd"
            ),
            "task_count": 94,
            "refs_digest": _digest("current refs"),
        },
        "prepared_dataset": {
            "manifest_digest": (
                "sha256:7c51718bf4728284d8fd131c16cc2f9845c6b74d4c9de71d012d4f28e71a51a2"
            ),
            "train_digest": (
                "sha256:927670027ab9a456187ebfd9779f7057e626f7eb16fc99f24e16e45d1a8e7769"
            ),
            "holdout_digest": (
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "train_examples": 94,
            "holdout_examples": 0,
        },
        "base_snapshot": {
            "base_model": verifier.BASE_MODEL,
            "required_bytes": 3_098_955_668,
            "files": {
                "model.safetensors": {
                    "bytes": 3_087_467_144,
                    "sha256": (
                        "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"
                    ),
                }
            },
        },
    }


def _completed_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_TEMPLATE.read_text(encoding="utf-8"))
    manifest["status"] = "sealed_and_independently_reviewed"
    manifest["sealed"] = True
    manifest["unresolved"] = []
    manifest["independent_review"] = {
        "status": "accepted",
        "reviewer": "fixture-reviewer",
        "review_digest": _digest("manifest review"),
    }
    manifest["snapshot"] = {
        "technology": "fixture-immutable-snapshot",
        "immutable_for_run": True,
        "host_writers_excluded": True,
    }
    _write_json(path, manifest)
    return manifest


def _completed_spec(
    manifest_path: Path,
    key_path: Path,
    verifier_identity: dict[str, Any],
) -> dict[str, Any]:
    spec = json.loads(SPEC_TEMPLATE.read_text(encoding="utf-8"))
    spec["status"] = "ready_and_independently_reviewed"
    spec["runnable"] = True
    spec["unresolved"] = []
    spec["independent_review"] = {
        "status": "accepted",
        "reviewer": "fixture-reviewer",
        "review_digest": _digest("spec review"),
    }
    image_digest = _digest("fixture image")
    spec["image"]["digest"] = image_digest
    spec["image"]["reference"] = f"microtensor-converter@{image_digest}"
    spec["image"]["sbom"] = {
        "name": "fixture.spdx.json",
        "bytes": 12,
        "digest": _digest("fixture sbom"),
    }
    spec["image"]["source_closure"]["worktree_state"] = "clean_immutable_snapshot"
    spec["image"]["source_closure"]["tree_digest"] = _digest("fixture source tree")
    spec["input_manifest_identity"] = _identity(manifest_path)
    spec["oci_config_identity"] = {
        "bytes": 10,
        "digest": _digest("oci config"),
    }
    spec["security"]["profiles"]["seccomp"] = {
        "name": "fixture-seccomp.json",
        "bytes": 11,
        "digest": _digest("seccomp"),
    }
    spec["security"]["profiles"]["lsm"] = {
        "name": "fixture-apparmor.profile",
        "bytes": 13,
        "digest": _digest("lsm"),
    }
    key_identity = _identity(key_path)
    spec["receipt_signature"] = {
        "scheme": "fixture-detached-v1",
        "key_id": "fixture-worker-key",
        "trusted_public_key": {"name": key_path.name, **key_identity},
        "verifier": verifier_identity,
    }
    return spec


def _runner_evidence(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": verifier.RUNNER_PREFLIGHT_SCHEMA,
        "status": "accepted",
        "independent_review": {
            "status": "accepted",
            "reviewer": "fixture-runner-reviewer",
            "review_digest": _digest("runner review"),
        },
        "runtime": {
            "name": "fixture-rootless-oci",
            "version": "1.0.0",
            "rootless_or_userns_remapped": True,
            "host_effective_uid": 1000,
            "cgroup_v2_delegated_writable": True,
            "network_namespace_creation": True,
        },
        "platform": copy.deepcopy(spec["platform"]),
        "resources": copy.deepcopy(spec["resources"]),
        "cgroup": copy.deepcopy(spec["cgroup"]),
        "security": copy.deepcopy(spec["security"]),
        "mounts": copy.deepcopy(spec["mounts"]),
        "image": copy.deepcopy(spec["image"]),
        "oci_config": copy.deepcopy(spec["oci_config_identity"]),
        "command": copy.deepcopy(spec["command"]),
        "expected_child_environment": copy.deepcopy(
            spec["expected_child_environment"]
        ),
    }


def _build_receipts(export: Path, spec: dict[str, Any], model_identity: dict[str, Any]) -> None:
    load_spec = verifier._expected_load_spec()
    tree_digest = verifier._official_tree_digest(model_identity["digest"])
    environment = spec["expected_child_environment"]
    primary = [
        _command(expected, environment, "primary")
        for expected in spec["expected_internal_commands"]
    ]
    replay_commands = [
        _command(expected, environment, "replay")
        for expected in spec["expected_internal_commands"]
    ]
    converter_python = copy.deepcopy(primary[0]["launch"]["executed_object"])
    replay = {
        "schema": verifier.REPLAY_SCHEMA,
        "matches_primary": True,
        "entrypoint_bytes": model_identity["bytes"],
        "entrypoint_digest": model_identity["digest"],
        "artifact_tree_digest": tree_digest,
        "f16_digest": _digest("f16"),
        "imatrix_digest": _digest("imatrix"),
        "commands": replay_commands,
    }
    runtime = _runtime_receipt(spec)
    artifact = {
        "entrypoint_bytes": model_identity["bytes"],
        "entrypoint_digest": model_identity["digest"],
        "quantization": "Q4_K_M",
        "tree_digest": tree_digest,
    }
    calibration_artifact = {
        **artifact,
        "calibration_metadata": {
            "imatrix_chunks_count": 128,
            "imatrix_dataset": "calibration.txt",
            "imatrix_entries_count": 7,
            "imatrix_file": "calibration.imatrix.gguf",
        },
    }
    calibration = {
        "schema": verifier.CALIBRATION_SCHEMA,
        "status": "complete",
        "profile": verifier.CALIBRATION_PROFILE,
        "track": "code",
        "hardware_class": "mt-3g",
        "base_model": verifier.BASE_MODEL,
        "llama_cpp_revision": verifier.LLAMA_CPP_REVISION,
        "source": {"fixture": "public-input-identities-are-worker-bound"},
        "selection": {
            "algorithm": "sha256-seed-ref-ascending-v1",
            "seed": 92,
            "current_rows": 78,
            "diagnostic_rows_excluded": 16,
            "auxiliary_pool_rows": 7_730,
            "auxiliary_selected_rows": 434,
            "total_rows": 512,
            "current_refs_digest": _digest("current"),
            "diagnostic_refs_digest": _digest("diagnostic"),
            "auxiliary_selected_refs_digest": _digest("auxiliary"),
        },
        "rendering": {
            "schema": "prompt-completion-im-end-utf8-v1",
            "encoding": "UTF-8",
            "expression": "prompt + completion + <|im_end|> + LF",
            "eos_token": "<|im_end|>",
            "eos_token_id": 151_645,
            "rows": 512,
            "corpus": {
                "name": "calibration.txt",
                "bytes": 1,
                "digest": _digest("calibration corpus"),
            },
        },
        "toolchain": {
            "converter_digest": (
                "sha256:e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f"
            ),
            "converter_python": copy.deepcopy(converter_python),
            "imatrix_digest": (
                "sha256:3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35"
            ),
            "quantizer_digest": (
                "sha256:e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580"
            ),
            "runtime_libraries": runtime,
        },
        "commands": primary,
        "determinism_replay": replay,
        "intermediate": {
            "f16": {"file_type": 1, "bytes": 1, "digest": _digest("f16")},
            "imatrix": {
                "entries_count": 7,
                "chunk_count": 128,
                "chunk_size": 512,
                "datasets": ["calibration.txt"],
                "bytes": 1,
                "digest": _digest("imatrix"),
            },
        },
        "artifact": calibration_artifact,
        "load_manifest": load_spec,
    }
    calibration_path = export / "bundle/calibration-receipt.json"
    _write_json(calibration_path, calibration)
    conversion = {
        "schema": verifier.CONVERSION_SCHEMA,
        "status": "complete",
        "track": "code",
        "hardware_class": "mt-3g",
        "base_model": verifier.BASE_MODEL,
        "llama_cpp_revision": verifier.LLAMA_CPP_REVISION,
        "source": _conversion_source(),
        "conversion": {
            "converter_digest": (
                "sha256:e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f"
            ),
            "converter_python": copy.deepcopy(converter_python),
            "imatrix_digest": (
                "sha256:3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35"
            ),
            "quantizer_digest": (
                "sha256:e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580"
            ),
            "runtime_libraries": runtime,
            "commands": primary,
            "determinism_replay": replay,
        },
        "artifact": artifact,
        "load_manifest": load_spec,
        "calibration_receipt_digest": _identity(calibration_path)["digest"],
    }
    _write_json(export / "bundle/conversion-receipt.json", conversion)


def _build_fixture(root: Path) -> dict[str, Any]:
    trusted = root / "trusted"
    export = root / "export"
    trusted.mkdir(mode=0o700)
    export.mkdir(mode=0o700)
    (export / "bundle/artifact").mkdir(parents=True, mode=0o700)
    (export / "bundle").chmod(0o700)
    (export / "bundle/artifact").chmod(0o700)

    manifest_path = trusted / "input-manifest.json"
    manifest = _completed_manifest(manifest_path)
    key_path = trusted / "worker-receipt.pub"
    _write(key_path, b"fixture public key\n")
    verifier_path = trusted / "fixture-offline-verifier"
    _write(verifier_path, _synthetic_static_elf())
    verifier_path.chmod(0o500)
    verifier_identity = {
        "name": verifier_path.name,
        **_identity(verifier_path),
        "format": "static-elf-linux-amd64",
        "closure": "single-file-no-pt-interp-no-dynamic",
    }
    spec = _completed_spec(manifest_path, key_path, verifier_identity)
    evidence_path = trusted / "runner-preflight.json"
    _write_json(evidence_path, _runner_evidence(spec))
    spec["runner_preflight_evidence"] = {
        "name": evidence_path.name,
        **_identity(evidence_path),
    }

    model_path = export / "bundle/artifact/model.gguf"
    _write(model_path, _synthetic_gguf())
    model_identity = _identity(model_path)
    _write_json(export / "bundle/load-spec.json", verifier._expected_load_spec())
    _build_receipts(export, spec, model_identity)

    output_identities = [
        {"path": relative, **_identity(export / relative)}
        for relative in sorted(verifier.BUNDLE_FILES)
    ]
    spec["expected_output"]["file_identities"] = output_identities
    spec_path = trusted / "worker-spec.json"
    _write_json(spec_path, spec)

    spec_identity = _identity(spec_path)
    manifest_identity = _identity(manifest_path)
    input_evidence = {
        "aggregate_digest": manifest["aggregate_digest"],
        "files_verified": manifest["file_count"],
        "verified": True,
    }
    worker_receipt = {
        "schema": verifier.WORKER_RECEIPT_SCHEMA,
        "status": "complete",
        "worker_spec": {**spec_identity, "schema": verifier.SPEC_SCHEMA},
        "input_manifest": {**manifest_identity, "schema": verifier.INPUT_MANIFEST_SCHEMA},
        "image": {
            "digest": spec["image"]["digest"],
            "reference": spec["image"]["reference"],
            "sbom": spec["image"]["sbom"],
            "source_closure": spec["image"]["source_closure"],
        },
        "oci_config": spec["oci_config_identity"],
        "runner_preflight_evidence": spec["runner_preflight_evidence"],
        "security_profiles": spec["security"]["profiles"],
        "signature": {
            "detached": True,
            "key_id": spec["receipt_signature"]["key_id"],
            "message_file": "worker-receipt.json",
            "scheme": spec["receipt_signature"]["scheme"],
            "signature_file": "worker-receipt.sig",
        },
        "execution": {
            "capabilities": [],
            "command": spec["command"],
            "container_removed_before_receipt": True,
            "container_stopped_before_export": True,
            "exit_code": 0,
            "export_started_after_container_stop": True,
            "mounts": spec["mounts"],
            "network_mode": "none",
            "no_new_privileges": True,
            "oom_killed": False,
            "platform": spec["platform"],
            "private_namespaces": [
                "cgroup",
                "ipc",
                "mount",
                "network",
                "pid",
                "user",
                "uts",
            ],
            "resources": spec["resources"],
            "root_filesystem_read_only": True,
            "runtime_mode": "rootless_or_userns_remapped",
            "timed_out": False,
            "cgroup": {
                "limits": spec["cgroup"],
                "empty_before_export": True,
                "remaining_processes": 0,
            },
        },
        "inputs": {
            "aggregate_digest": manifest["aggregate_digest"],
            "snapshot_immutable": True,
            "mounts_read_only": True,
            "preflight": input_evidence,
            "postflight": input_evidence,
        },
        "output": {
            "exact_file_set": True,
            "private_intermediates_absent": True,
            "files": output_identities,
        },
    }
    receipt_path = export / "worker-receipt.json"
    _write_json(receipt_path, worker_receipt)
    _write(export / "worker-receipt.sig", b"fixture signature\n")
    return {
        "export": export,
        "spec": spec_path,
        "manifest": manifest_path,
        "key": key_path,
        "signature_verifier": verifier_path,
        "runner_preflight_evidence": evidence_path,
        "verifier_identity": verifier_identity,
        "worker_receipt": receipt_path,
    }


def _signature_ok(
    verifier_fd: int,
    message_fd: int,
    signature_fd: int,
    trusted_public_key_fd: int,
    *,
    scheme: str,
    key_id: str,
) -> bool:
    return (
        os.pread(verifier_fd, 4, 0) == b"\x7fELF"
        and os.pread(message_fd, 1, 0) == b"{"
        and os.pread(signature_fd, 1024, 0) == b"fixture signature\n"
        and os.pread(trusted_public_key_fd, 1024, 0) == b"fixture public key\n"
        and scheme == "fixture-detached-v1"
        and key_id == "fixture-worker-key"
    )


def _verify(fixture: dict[str, Any], signature_hook: Any = _signature_ok) -> dict[str, Any]:
    # The production verifier intentionally rejects EUID 0.  The repository test
    # environment can be root, so fixture calls substitute the fixture owner's UID
    # while dedicated tests below exercise the real root-refusal policy.
    with mock.patch.object(
        verifier, "_verification_euid", return_value=os.geteuid()
    ):
        return verifier.verify_export(
            fixture["export"],
            fixture["spec"],
            fixture["manifest"],
            fixture["key"],
            fixture["signature_verifier"],
            fixture["runner_preflight_evidence"],
            signature_verifier=signature_hook,
        )


def _preflight(fixture: dict[str, Any]) -> dict[str, Any]:
    with mock.patch.object(
        verifier, "_verification_euid", return_value=os.geteuid()
    ):
        return verifier.preflight_only(
            fixture["spec"],
            fixture["manifest"],
            fixture["key"],
            fixture["signature_verifier"],
            fixture["runner_preflight_evidence"],
        )


class ConversionExportVerifierTests(unittest.TestCase):
    def test_templates_are_explicitly_non_runnable_and_fail_closed(self) -> None:
        spec = json.loads(SPEC_TEMPLATE.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(spec["status"], "incomplete_non_runnable")
        self.assertIs(spec["runnable"], False)
        self.assertEqual(manifest["status"], "incomplete_non_runnable")
        self.assertIs(manifest["sealed"], False)
        unresolved_text = json.dumps([spec["unresolved"], manifest["unresolved"]])
        for category in ("image", "sbom", "seccomp", "lsm", "signature", "output", "review"):
            self.assertIn(category, unresolved_text.lower())
        with self.assertRaises(verifier.VerificationRefused):
            verifier._validate_worker_spec(spec)
        with self.assertRaises(verifier.VerificationRefused):
            verifier._validate_input_manifest(manifest)

    def test_complete_synthetic_export_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            result = _verify(fixture)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["artifact"]["architecture"], "qwen2")
        self.assertEqual(result["artifact"]["file_type"], 15)

    def test_root_ancestry_fstat_failure_closes_uncommitted_fd(self) -> None:
        real_open = os.open
        real_fstat = os.fstat
        real_close = os.close
        opened: list[int] = []

        def tracking_open(*args: Any, **kwargs: Any) -> int:
            descriptor = real_open(*args, **kwargs)
            opened.append(descriptor)
            return descriptor

        with (
            mock.patch.object(verifier.os, "open", side_effect=tracking_open),
            mock.patch.object(
                verifier.os,
                "fstat",
                side_effect=OSError("fixture root fstat failure"),
            ),
            self.assertRaisesRegex(OSError, "fixture root fstat failure"),
        ):
            verifier._open_secure_ancestry(Path("/"), "fixture ancestry")

        self.assertEqual(len(opened), 1)
        leaked: list[int] = []
        for descriptor in opened:
            try:
                real_fstat(descriptor)
            except OSError:
                continue
            leaked.append(descriptor)
            real_close(descriptor)
        self.assertEqual(leaked, [])

    def test_component_ancestry_fstat_failure_closes_every_open_fd(self) -> None:
        real_open = os.open
        real_fstat = os.fstat
        real_close = os.close
        opened: list[int] = []
        fstat_calls = 0

        def tracking_open(*args: Any, **kwargs: Any) -> int:
            descriptor = real_open(*args, **kwargs)
            opened.append(descriptor)
            return descriptor

        def fail_second_fstat(descriptor: int) -> os.stat_result:
            nonlocal fstat_calls
            fstat_calls += 1
            if fstat_calls == 2:
                raise OSError("fixture component fstat failure")
            return real_fstat(descriptor)

        with (
            mock.patch.object(verifier.os, "open", side_effect=tracking_open),
            mock.patch.object(
                verifier.os, "fstat", side_effect=fail_second_fstat
            ),
            self.assertRaisesRegex(OSError, "fixture component fstat failure"),
        ):
            verifier._open_secure_ancestry(
                ROOT / "fd-leak-fixture", "fixture ancestry"
            )

        self.assertEqual(len(opened), 2)
        leaked: list[int] = []
        for descriptor in opened:
            try:
                real_fstat(descriptor)
            except OSError:
                continue
            leaked.append(descriptor)
            real_close(descriptor)
        self.assertEqual(leaked, [])

    def test_preflight_only_validates_without_export_or_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            with (
                mock.patch.object(
                    verifier,
                    "_inspect_export",
                    side_effect=AssertionError("preflight touched export"),
                ),
                mock.patch.object(
                    verifier.CommandSignatureVerifier,
                    "__call__",
                    side_effect=AssertionError("preflight invoked signature verifier"),
                ),
            ):
                result = _preflight(fixture)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(
            result["input_aggregate_digest"],
            verifier.PINNED_INPUT_AGGREGATE_DIGEST,
        )

    def test_template_placeholders_refuse_before_final_copy_modes(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationRefused, "unresolved"):
            verifier.preflight_only(
                SPEC_TEMPLATE,
                MANIFEST_TEMPLATE,
                Path("/does/not/exist/key"),
                Path("/does/not/exist/verifier"),
                Path("/does/not/exist/evidence"),
            )

    def test_finalized_private_modes_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            fixture["spec"].chmod(0o640)
            with self.assertRaisesRegex(verifier.VerificationRefused, "0600"):
                _preflight(fixture)

    def test_runner_preflight_evidence_must_cross_bind_exact_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            evidence = json.loads(
                fixture["runner_preflight_evidence"].read_text(encoding="utf-8")
            )
            evidence["command"][0] = "/bin/true"
            _write_json(fixture["runner_preflight_evidence"], evidence)
            spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
            spec["runner_preflight_evidence"] = {
                "name": fixture["runner_preflight_evidence"].name,
                **_identity(fixture["runner_preflight_evidence"]),
            }
            _write_json(fixture["spec"], spec)
            with self.assertRaisesRegex(
                verifier.VerificationRefused, "runner preflight field"
            ):
                _preflight(fixture)

    def test_signature_verifier_must_be_a_static_elf_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            _write(fixture["signature_verifier"], _synthetic_static_elf(dynamic=True))
            fixture["signature_verifier"].chmod(0o500)
            spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
            spec["receipt_signature"]["verifier"] = {
                "name": fixture["signature_verifier"].name,
                **_identity(fixture["signature_verifier"]),
                "format": "static-elf-linux-amd64",
                "closure": "single-file-no-pt-interp-no-dynamic",
            }
            _write_json(fixture["spec"], spec)
            with self.assertRaisesRegex(
                verifier.VerificationRefused, "dynamic/interpreter"
            ):
                _preflight(fixture)

    def test_root_execution_and_wrong_ownership_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            with (
                mock.patch.object(verifier.os, "geteuid", return_value=0),
                self.assertRaisesRegex(verifier.VerificationRefused, "non-root"),
            ):
                verifier.preflight_only(
                    fixture["spec"],
                    fixture["manifest"],
                    fixture["key"],
                    fixture["signature_verifier"],
                    fixture["runner_preflight_evidence"],
                )
            wrong_uid = os.geteuid() + 1
            with (
                mock.patch.object(
                    verifier, "_verification_euid", return_value=wrong_uid
                ),
                self.assertRaisesRegex(
                    verifier.VerificationRefused, "owned by verifier EUID"
                ),
            ):
                verifier.preflight_only(
                    fixture["spec"],
                    fixture["manifest"],
                    fixture["key"],
                    fixture["signature_verifier"],
                    fixture["runner_preflight_evidence"],
                )
            with self.assertRaisesRegex(
                verifier.VerificationRefused, "owned by verifier EUID"
            ):
                verifier._inspect_export(
                    fixture["export"],
                    expected_uid=wrong_uid,
                    hash_files=False,
                )

    def test_export_parent_must_be_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _build_fixture(root)
            root.chmod(0o755)
            with self.assertRaisesRegex(
                verifier.VerificationRefused, "private 0700|parent mode"
            ):
                _verify(fixture)

    def test_exact_internal_argv_and_child_environment_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            complete = json.loads(fixture["spec"].read_text(encoding="utf-8"))
        mutations = {
            "bin_true": lambda value: value["expected_internal_commands"][0][
                "argv"
            ].__setitem__(0, "/bin/true"),
            "ld_preload": lambda value: value["expected_child_environment"].__setitem__(
                "LD_PRELOAD", "/opt/fixture/evil.so"
            ),
            "module_launcher": lambda value: value["command"].__setitem__(
                3, "-m"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(complete)
                mutate(candidate)
                with self.assertRaises(verifier.VerificationRefused):
                    verifier._validate_worker_spec(candidate)

    def test_paths_must_be_canonical_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            complete_spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
            complete_manifest = json.loads(
                fixture["manifest"].read_text(encoding="utf-8")
            )
        noncanonical = copy.deepcopy(complete_spec)
        noncanonical["image"]["source_closure"]["files"][0]["path"] = "a//b"
        with self.assertRaises(verifier.VerificationRefused):
            verifier._validate_worker_spec(noncanonical)
        duplicated = copy.deepcopy(complete_spec)
        duplicated["image"]["source_closure"]["files"][1]["path"] = duplicated[
            "image"
        ]["source_closure"]["files"][0]["path"]
        with self.assertRaises(verifier.VerificationRefused):
            verifier._validate_worker_spec(duplicated)
        manifest_path = copy.deepcopy(complete_manifest)
        manifest_path["inputs"][0]["files"][0]["path"] = "a//b"
        with self.assertRaises(verifier.VerificationRefused):
            verifier._validate_input_manifest(manifest_path)

    def test_input_aggregate_is_pinned_not_merely_self_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
        manifest["inputs"][0]["files"][0]["digest"] = _digest("mutated input")
        manifest["aggregate_digest"] = verifier._digest_bytes(
            verifier._canonical_bytes(manifest["inputs"])
        )
        with self.assertRaisesRegex(verifier.VerificationRefused, "pinned aggregate"):
            verifier._validate_input_manifest(manifest)

    def test_q4_m541_contract_is_exact(self) -> None:
        spec = json.loads(SPEC_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(verifier.MAX_INPUT_TOKENS, 541)
        token_index = spec["command"].index("--max-input-tokens") + 1
        self.assertEqual(spec["command"][token_index], "541")
        self.assertIn("q4-m541", spec["execution_protocol"]["export_root"])
        self.assertEqual(verifier._expected_load_spec()["max_input"]["tokens"], 541)

    def test_isolated_absolute_script_resolves_reviewed_sibling_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            training = Path(temporary) / "training"
            training.mkdir(mode=0o700)
            sibling = training / "fixture_sibling.py"
            launcher = training / "fixture_launcher.py"
            _write(sibling, b'VALUE = "sibling-ok"\n')
            _write(
                launcher,
                (
                    b"import sys\n"
                    b"try:\n"
                    b"    from training import fixture_sibling\n"
                    b"except ModuleNotFoundError:\n"
                    b"    import fixture_sibling\n"
                    b"print(fixture_sibling.VALUE, *sys.argv[1:])\n"
                ),
            )
            bootstrap = (
                "import runpy,sys,types;"
                "_p=types.ModuleType('training');"
                f"_p.__path__=[{str(training)!r}];"
                "_p.__package__='training';"
                "sys.modules['training']=_p;"
                f"runpy.run_path({str(launcher)!r},run_name='__main__')"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", bootstrap, "--marker", "ok"],
                cwd="/",
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=10,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, b"sibling-ok --marker ok\n")

    def test_converter_python_observation_is_cross_bound_everywhere(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            conversion = json.loads(
                (fixture["export"] / "bundle/conversion-receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            calibration_path = fixture["export"] / "bundle/calibration-receipt.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            load_spec = json.loads(
                (fixture["export"] / "bundle/load-spec.json").read_text(
                    encoding="utf-8"
                )
            )
            spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
            model = verifier._static_gguf_identity(
                fixture["export"] / "bundle/artifact/model.gguf"
            )
            tree_digest = verifier._official_tree_digest(model["digest"])
            calibration_digest = _identity(calibration_path)["digest"]
            verifier._validate_conversion_receipts(
                conversion,
                calibration,
                load_spec,
                spec,
                model,
                tree_digest,
                calibration_digest,
            )
            mutations = {
                "conversion": lambda conv, _cal: conv["conversion"][
                    "converter_python"
                ]["worker_observation"].__setitem__("inode", 99),
                "primary": lambda conv, _cal: conv["conversion"]["commands"][0][
                    "launch"
                ]["executed_object"]["worker_observation"].__setitem__("inode", 99),
                "replay": lambda conv, _cal: conv["conversion"][
                    "determinism_replay"
                ]["commands"][0]["launch"]["executed_object"][
                    "worker_observation"
                ].__setitem__("inode", 99),
                "calibration": lambda _conv, cal: cal["toolchain"][
                    "converter_python"
                ]["worker_observation"].__setitem__("inode", 99),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    changed_conversion = copy.deepcopy(conversion)
                    changed_calibration = copy.deepcopy(calibration)
                    mutate(changed_conversion, changed_calibration)
                    with self.assertRaises(verifier.VerificationRefused):
                        verifier._validate_conversion_receipts(
                            changed_conversion,
                            changed_calibration,
                            load_spec,
                            spec,
                            model,
                            tree_digest,
                            calibration_digest,
                        )

    def test_signature_inputs_are_descriptor_bound_during_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            receipt = fixture["worker_receipt"]
            backup = receipt.with_name("held-original")

            def replace_message_path(
                _verifier_fd: int,
                message_fd: int,
                _signature_fd: int,
                _key_fd: int,
                *,
                scheme: str,
                key_id: str,
            ) -> bool:
                self.assertEqual(os.pread(message_fd, 1, 0), b"{")
                os.replace(receipt, backup)
                _write(receipt, b'{"replacement":true}\n')
                self.assertEqual(os.pread(message_fd, 1, 0), b"{")
                return bool(scheme and key_id)

            with self.assertRaises(verifier.VerificationRefused):
                _verify(fixture, replace_message_path)
            receipt.unlink()
            os.replace(backup, receipt)

    def test_transient_model_swap_restore_cannot_change_held_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            model = fixture["export"] / "bundle/artifact/model.gguf"
            backup = model.with_name("held-original.gguf")
            original_identity = _identity(model)
            replacement = _synthetic_gguf(entries=8)
            replacement_digest = _digest(replacement)
            real_dup = os.dup
            real_read_exact = verifier._read_exact
            state = {"swapped": False, "restored": False}

            def swap_before_held_dup(descriptor: int) -> int:
                if not state["swapped"]:
                    os.replace(model, backup)
                    _write(model, replacement)
                    state["swapped"] = True
                return real_dup(descriptor)

            def restore_before_final_recheck(
                handle: Any,
                size: int,
                file_size: int,
                label: str,
            ) -> bytes:
                raw = real_read_exact(handle, size, file_size, label)
                if (
                    state["swapped"]
                    and not state["restored"]
                    and label == "tensor 'weight' offset"
                ):
                    self.assertEqual(_identity(model)["digest"], replacement_digest)
                    model.unlink()
                    os.replace(backup, model)
                    state["restored"] = True
                return raw

            try:
                with (
                    mock.patch.object(
                        verifier.os, "dup", side_effect=swap_before_held_dup
                    ),
                    mock.patch.object(
                        verifier,
                        "_read_exact",
                        side_effect=restore_before_final_recheck,
                    ),
                    self.assertRaisesRegex(
                        verifier.VerificationRefused,
                        "changed during held parsing",
                    ),
                ):
                    _verify(fixture)
            finally:
                if model.exists() and backup.exists():
                    model.unlink()
                    os.replace(backup, model)
                elif backup.exists():
                    os.replace(backup, model)

            self.assertTrue(state["swapped"])
            self.assertTrue(state["restored"])
            self.assertEqual(_identity(model), original_identity)

    def test_command_signature_adapter_uses_proc_fd_for_every_input(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")
        with mock.patch.object(subprocess, "run", return_value=completed) as run:
            accepted = verifier.CommandSignatureVerifier()(
                3,
                4,
                5,
                6,
                scheme="fixture",
                key_id="key",
            )
        self.assertIs(accepted, True)
        arguments, options = run.call_args
        self.assertEqual(options["executable"], "/proc/self/fd/3")
        self.assertEqual(options["pass_fds"], (3, 4, 5, 6))
        self.assertEqual(arguments[0][0], "/proc/self/fd/3")
        for descriptor in (4, 5, 6):
            self.assertIn(f"/proc/self/fd/{descriptor}", arguments[0])

    def test_each_security_and_output_placeholder_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            complete = json.loads(fixture["spec"].read_text(encoding="utf-8"))
        mutations = [
            lambda value: value["image"].__setitem__("digest", "UNRESOLVED:image"),
            lambda value: value["image"]["sbom"].__setitem__("digest", "UNRESOLVED:sbom"),
            lambda value: value["security"]["profiles"]["seccomp"].__setitem__(
                "digest", "UNRESOLVED:seccomp"
            ),
            lambda value: value["security"]["profiles"]["lsm"].__setitem__(
                "digest", "UNRESOLVED:lsm"
            ),
            lambda value: value["receipt_signature"]["trusted_public_key"].__setitem__(
                "digest", "UNRESOLVED:key"
            ),
            lambda value: value["receipt_signature"]["verifier"].__setitem__(
                "digest", "UNRESOLVED:verifier"
            ),
            lambda value: value["receipt_signature"]["verifier"].__setitem__(
                "closure", "UNRESOLVED:verifier_closure"
            ),
            lambda value: value["runner_preflight_evidence"].__setitem__(
                "digest", "UNRESOLVED:runner_evidence"
            ),
            lambda value: value["expected_output"]["file_identities"][0].__setitem__(
                "digest", "UNRESOLVED:output"
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                candidate = copy.deepcopy(complete)
                mutate(candidate)
                with self.assertRaises(verifier.VerificationRefused):
                    verifier._validate_worker_spec(candidate)

    def test_duplicate_json_keys_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            _write(path, b'{"value":1,"value":2}\n')
            with self.assertRaises(verifier.VerificationRefused):
                verifier._strict_json(path, "duplicate fixture", maximum_bytes=1024)

    def test_unexpected_file_symlink_and_hardlink_are_refused(self) -> None:
        for attack in ("extra", "symlink", "hardlink"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                fixture = _build_fixture(Path(temporary))
                model = fixture["export"] / "bundle/artifact/model.gguf"
                if attack == "extra":
                    _write(fixture["export"] / "unexpected", b"x")
                elif attack == "symlink":
                    external = Path(temporary) / "external.gguf"
                    _write(external, model.read_bytes())
                    model.unlink()
                    model.symlink_to(external)
                else:
                    os.link(model, Path(temporary) / "model-hardlink")
                with self.assertRaises(verifier.VerificationRefused):
                    _verify(fixture)

    def test_signature_failure_and_signature_time_mutation_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))

            def reject_signature(
                _verifier_fd: int,
                _message_fd: int,
                _signature_fd: int,
                _key_fd: int,
                *,
                scheme: str,
                key_id: str,
            ) -> bool:
                return bool(scheme and key_id) and False

            with self.assertRaises(verifier.VerificationRefused):
                _verify(fixture, reject_signature)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))

            def mutate_during_signature(
                _verifier_fd: int,
                _message_fd: int,
                _signature_fd: int,
                _key_fd: int,
                *,
                scheme: str,
                key_id: str,
            ) -> bool:
                model = fixture["export"] / "bundle/artifact/model.gguf"
                model.write_bytes(model.read_bytes() + b"x")
                return bool(scheme and key_id)

            with self.assertRaises(verifier.VerificationRefused):
                _verify(fixture, mutate_during_signature)

    def test_signed_receipt_cross_binding_mutations_are_refused(self) -> None:
        mutations = {
            "image": lambda value: value["image"].__setitem__("digest", _digest("wrong image")),
            "oci_config": lambda value: value["oci_config"].__setitem__(
                "digest", _digest("wrong config")
            ),
            "profile": lambda value: value["security_profiles"]["seccomp"].__setitem__(
                "digest", _digest("wrong profile")
            ),
            "runner_evidence": lambda value: value[
                "runner_preflight_evidence"
            ].__setitem__("digest", _digest("wrong runner evidence")),
            "cgroup": lambda value: value["execution"]["cgroup"].__setitem__(
                "remaining_processes", 1
            ),
            "input": lambda value: value["inputs"].__setitem__(
                "aggregate_digest", _digest("wrong inputs")
            ),
            "output": lambda value: value["output"]["files"][0].__setitem__(
                "digest", _digest("wrong output")
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = _build_fixture(Path(temporary))
                receipt = json.loads(fixture["worker_receipt"].read_text(encoding="utf-8"))
                mutate(receipt)
                _write_json(fixture["worker_receipt"], receipt)
                with self.assertRaises(verifier.VerificationRefused):
                    _verify(fixture, lambda *_args, **_kwargs: True)

    def test_exact_load_spec_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _build_fixture(Path(temporary))
            load_path = fixture["export"] / "bundle/load-spec.json"
            load = json.loads(load_path.read_text(encoding="utf-8"))
            load["max_input"]["tokens"] = 513
            _write_json(load_path, load)
            with self.assertRaises(verifier.VerificationRefused):
                _verify(fixture, lambda *_args, **_kwargs: True)

    def test_static_gguf_checks_version_architecture_file_type_and_metadata(self) -> None:
        cases = {
            "version": {"version": 2},
            "architecture": {"architecture": "qwen3"},
            "file_type": {"file_type": 14},
            "entries": {"entries": 0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.gguf"
            _write(valid, _synthetic_gguf())
            identity = verifier._static_gguf_identity(valid)
            self.assertEqual(identity["version"], 3)
            self.assertEqual(identity["architecture"], "qwen2")
            for label, arguments in cases.items():
                with self.subTest(label=label):
                    path = root / f"{label}.gguf"
                    _write(path, _synthetic_gguf(**arguments))
                    with self.assertRaises(verifier.VerificationRefused):
                        verifier._static_gguf_identity(path)

    def test_model_size_ceiling_is_checked_before_content_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oversized.gguf"
            with path.open("wb") as handle:
                handle.seek(verifier.MAX_MODEL_BYTES)
                handle.write(b"x")
            path.chmod(0o600)
            with self.assertRaises(verifier.VerificationRefused):
                verifier._static_gguf_identity(path)


if __name__ == "__main__":
    unittest.main()
