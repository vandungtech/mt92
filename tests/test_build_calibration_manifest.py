from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import tempfile
import unicodedata
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training import build_calibration_manifest as builder


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class BuildCalibrationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.llama_cpp = self.root / "llama.cpp"
        self.llama_cpp.mkdir()

        self.stage = self.root / "stage"
        self.source = self.stage / "merged"
        self.source.mkdir(parents=True)
        (self.stage / "training_metadata.json").write_bytes(b'{"finished":true}\n')
        source_payloads = {
            "chat_template.jinja": b"template\n",
            "config.json": b'{"model_type":"qwen3"}\n',
            "generation_config.json": b"{}\n",
            "model.safetensors": b"tiny-model-weights",
            "tokenizer.json": b'{"version":"1"}\n',
            "tokenizer_config.json": b"{}\n",
        }
        for name, payload in source_payloads.items():
            (self.source / name).write_bytes(payload)

        evidence = self.root / "evidence"
        evidence.mkdir()
        (evidence / "model-f16.gguf").write_bytes(b"GGUF-f16")
        (evidence / "calibration.txt").write_bytes(b"rendered-record\n")
        (evidence / "calibration.metadata.json").write_bytes(b"{}\n")
        (evidence / "model.imatrix.gguf").write_bytes(b"GGUF-imatrix")

        self.artifact_dir = self.root / "candidate"
        self.artifact_dir.mkdir()
        (self.artifact_dir / "model.gguf").write_bytes(b"GGUF-q4")
        self.soup = self.root / "soup"
        self.soup.mkdir()
        self.output = self.root / "calibration-lineage.json"
        self.request = builder.ManifestRequest(
            output=self.output,
            training_dirs=("stage",),
            source_model_dir="stage/merged",
            converted_model="evidence/model-f16.gguf",
            corpus="evidence/calibration.txt",
            corpus_metadata="evidence/calibration.metadata.json",
            imatrix="evidence/model.imatrix.gguf",
            quantized_artifact="candidate/model.gguf",
            quantization_profile=builder.STANDARD_Q4_PROFILE,
            finished_block=123,
            llama_cpp_dir=self.llama_cpp,
            weight_soup_checkpoints=(("1", "soup"),),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def clean_git(_repository: Path, arguments: tuple[str, ...]) -> str:
        if arguments[0] == "rev-parse":
            return builder.provenance.LLAMA_CPP_REVISION
        if arguments[0] == "status":
            return ""
        raise AssertionError(arguments)

    def validating_publication(
        self,
        training_dirs: tuple[Path, ...],
        artifact_digest: str,
        finished_block: int,
        *,
        calibration_manifest: Path,
        weight_soup_checkpoints: dict[int, Path],
    ) -> SimpleNamespace:
        temporary_manifest = Path(calibration_manifest)
        self.assertTrue(temporary_manifest.is_file())
        self.assertEqual(temporary_manifest.parent, self.root)
        self.assertFalse(self.output.exists())
        self.assertEqual(training_dirs, (self.stage,))
        self.assertEqual(finished_block, 123)
        self.assertEqual(weight_soup_checkpoints, {1: self.soup})
        payload = temporary_manifest.read_bytes()
        return SimpleNamespace(
            artifact_digest=artifact_digest,
            calibration=SimpleNamespace(manifest_digest=digest(payload)),
        )

    def build(
        self,
        request: builder.ManifestRequest | None = None,
        *,
        validator: object | None = None,
        git_side_effect: object | None = None,
    ) -> builder.BuiltCalibrationManifest:
        with (
            mock.patch.object(
                builder,
                "_git_output",
                side_effect=git_side_effect or self.clean_git,
            ),
            mock.patch.object(
                builder.provenance,
                "validate_publication",
                side_effect=validator or self.validating_publication,
            ) as validate,
        ):
            result = builder.build_calibration_manifest(request or self.request)
        self.validation_mock = validate
        return result

    def assert_no_temporary_manifest(self, output: Path | None = None) -> None:
        destination = output or self.output
        self.assertEqual(
            list(destination.parent.glob(f".{destination.name}.*.tmp")),
            [],
        )

    def test_builds_exact_standard_manifest_then_installs_atomically(self) -> None:
        nested = self.artifact_dir / "nested"
        nested.mkdir()
        (nested / "note.txt").write_bytes(b"public-sidecar")
        (self.artifact_dir / ".ignored").write_bytes(b"ignored")

        built = self.build()
        payload = self.output.read_bytes()
        manifest = json.loads(payload)
        self.assertEqual(payload, builder._json_bytes(manifest))
        self.assertEqual(built.manifest, manifest)
        self.assertEqual(built.manifest_digest, digest(payload))
        self.assertEqual(
            built.artifact_tree_digest,
            builder.provenance._artifact_tree_digest(self.artifact_dir),
        )
        self.assertTrue(built.committed)
        self.assertTrue(built.installed_integrity_confirmed)
        self.assertTrue(built.temporary_cleanup_complete)
        self.assertTrue(built.durability_confirmed)
        self.assertEqual(built.post_commit_warnings, ())
        self.assertEqual(
            built.attestation_semantics,
            builder.ATTESTATION_SEMANTICS,
        )
        installed = self.output.stat()
        self.assertEqual(
            (built.installed_device, built.installed_inode),
            (installed.st_dev, installed.st_ino),
        )
        self.assertEqual(
            manifest["conversion"]["arguments"],
            [
                "stage/merged",
                "--outfile",
                "evidence/model-f16.gguf",
                "--outtype",
                "f16",
            ],
        )
        self.assertEqual(
            manifest["calibration"]["arguments"],
            [
                "--offline",
                "--model",
                "evidence/model-f16.gguf",
                "--file",
                "evidence/calibration.txt",
                "--output",
                "evidence/model.imatrix.gguf",
                "--ctx-size",
                "512",
                "--chunks",
                "-1",
                "--no-ppl",
                "--parse-special",
            ],
        )
        self.assertEqual(
            manifest["calibration"]["settings"],
            {
                "offline": True,
                "ctx_size": 512,
                "chunks": -1,
                "no_ppl": True,
                "process_output": False,
                "parse_special": True,
                "output_format": "gguf",
            },
        )
        self.assertEqual(
            manifest["quantization"]["arguments"],
            [
                "--imatrix",
                "evidence/model.imatrix.gguf",
                "evidence/model-f16.gguf",
                "candidate/model.gguf",
                "Q4_K_M",
            ],
        )
        names = [entry["path"] for entry in manifest["source_model"]["files"]]
        self.assertEqual(names, sorted(path.name for path in self.source.iterdir()))
        self.assertEqual(
            manifest["source_model"]["training_metadata_sha256"],
            digest((self.stage / "training_metadata.json").read_bytes()),
        )
        self.assertEqual(os.stat(self.output).st_mode & 0o777, 0o644)
        self.assert_no_temporary_manifest()
        self.assertEqual(self.validation_mock.call_count, 1)

    def test_audited_override_has_only_the_exact_tensor_type_arguments(self) -> None:
        output = self.root / "override.json"
        request = dataclasses.replace(
            self.request,
            output=output,
            quantization_profile=builder.ATTN_V_Q6_PROFILE,
        )

        built = self.build(request)
        self.assertEqual(
            built.manifest["quantization"]["arguments"],
            [
                "--imatrix",
                "evidence/model.imatrix.gguf",
                "--tensor-type",
                builder.provenance.ATTN_V_Q6_OVERRIDE,
                "evidence/model-f16.gguf",
                "candidate/model.gguf",
                "Q4_K_M",
            ],
        )

    def test_rejects_wrong_revision_and_dirty_checkout(self) -> None:
        cases = (
            ("revision must be exactly", lambda _repo, _args: "0" * 40),
            (
                "clean worktree",
                lambda _repo, args: (
                    builder.provenance.LLAMA_CPP_REVISION
                    if args[0] == "rev-parse"
                    else "?? unexpected-file"
                ),
            ),
        )
        for index, (pattern, git_result) in enumerate(cases):
            output = self.root / f"bad-git-{index}.json"
            request = dataclasses.replace(self.request, output=output)
            with (
                self.subTest(pattern=pattern),
                mock.patch.object(builder, "_git_output", side_effect=git_result),
                self.assertRaisesRegex(builder.ManifestBuildError, pattern),
            ):
                builder.build_calibration_manifest(request)
            self.assertFalse(output.exists())

    def test_rejects_non_normalized_or_traversing_path_strings(self) -> None:
        decomposed = unicodedata.normalize("NFD", "é") + "/model.gguf"
        cases = (
            "/absolute/model.gguf",
            "../evidence/model-f16.gguf",
            "./evidence/model-f16.gguf",
            "evidence//model-f16.gguf",
            "evidence\\model-f16.gguf",
            " evidence/model-f16.gguf",
            decomposed,
        )
        with mock.patch.object(
            builder, "_git_output", side_effect=self.clean_git
        ):
            for index, value in enumerate(cases):
                output = self.root / f"bad-path-{index}.json"
                request = dataclasses.replace(
                    self.request,
                    output=output,
                    converted_model=value,
                )
                with (
                    self.subTest(value=value),
                    self.assertRaises(builder.ManifestBuildError),
                ):
                    builder.build_calibration_manifest(request)
                self.assertFalse(output.exists())

    def test_rejects_unallowlisted_profile_and_non_gguf_evidence(self) -> None:
        profile_output = self.root / "bad-profile.json"
        profile_request = dataclasses.replace(
            self.request,
            output=profile_output,
            quantization_profile="q5_k_m",
        )
        with self.assertRaisesRegex(builder.ManifestBuildError, "profile must be"):
            builder.build_calibration_manifest(profile_request)
        self.assertFalse(profile_output.exists())

        (self.root / self.request.converted_model).write_bytes(b"not-gguf")
        gguf_output = self.root / "bad-gguf.json"
        gguf_request = dataclasses.replace(self.request, output=gguf_output)
        with (
            mock.patch.object(builder, "_git_output", side_effect=self.clean_git),
            self.assertRaisesRegex(builder.ManifestBuildError, "not a GGUF"),
        ):
            builder.build_calibration_manifest(gguf_request)
        self.assertFalse(gguf_output.exists())

    def test_rejects_symlinks_anywhere_in_the_artifact_tree(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_bytes(b"outside")
        (self.artifact_dir / "linked.txt").symlink_to(outside)
        output = self.root / "artifact-symlink.json"
        request = dataclasses.replace(self.request, output=output)
        with (
            mock.patch.object(builder, "_git_output", side_effect=self.clean_git),
            self.assertRaisesRegex(builder.ManifestBuildError, "symlink"),
        ):
            builder.build_calibration_manifest(request)
        self.assertFalse(output.exists())

    def test_rejects_invalid_soup_stage_and_path_without_normalizing_them(self) -> None:
        cases = (
            (("01", "soup"),),
            (("1", "soup/"),),
            ((1, "soup"),),
        )
        with mock.patch.object(
            builder, "_git_output", side_effect=self.clean_git
        ):
            for index, entries in enumerate(cases):
                output = self.root / f"bad-soup-{index}.json"
                request = dataclasses.replace(
                    self.request,
                    output=output,
                    weight_soup_checkpoints=entries,  # type: ignore[arg-type]
                )
                with self.subTest(entries=entries), self.assertRaises(
                    builder.ManifestBuildError
                ):
                    builder.build_calibration_manifest(request)
                self.assertFalse(output.exists())

    def test_rejects_duplicate_roles_symlinks_and_incomplete_inventory(self) -> None:
        duplicate_request = dataclasses.replace(
            self.request,
            imatrix=self.request.converted_model,
        )
        with (
            mock.patch.object(builder, "_git_output", side_effect=self.clean_git),
            self.assertRaisesRegex(builder.ManifestBuildError, "five distinct"),
        ):
            builder.build_calibration_manifest(duplicate_request)

        linked = self.root / "evidence" / "linked.gguf"
        linked.symlink_to(self.root / self.request.converted_model)
        symlink_request = dataclasses.replace(
            self.request,
            output=self.root / "symlink.json",
            converted_model="evidence/linked.gguf",
        )
        with (
            mock.patch.object(builder, "_git_output", side_effect=self.clean_git),
            self.assertRaisesRegex(builder.ManifestBuildError, "symlink component"),
        ):
            builder.build_calibration_manifest(symlink_request)

        (self.source / "config.json").unlink()
        incomplete_request = dataclasses.replace(
            self.request,
            output=self.root / "incomplete.json",
        )
        with (
            mock.patch.object(builder, "_git_output", side_effect=self.clean_git),
            self.assertRaisesRegex(builder.ManifestBuildError, "inventory is incomplete"),
        ):
            builder.build_calibration_manifest(incomplete_request)

    def test_validation_failure_and_binding_mismatch_leave_no_output(self) -> None:
        def reject(*_args: object, **_kwargs: object) -> object:
            self.assertFalse(self.output.exists())
            raise builder.provenance.ProvenanceValidationError("sidecar mismatch")

        with self.assertRaisesRegex(builder.ManifestBuildError, "sidecar mismatch"):
            self.build(validator=reject)
        self.assertFalse(self.output.exists())
        self.assert_no_temporary_manifest()

        def wrong_binding(
            _training_dirs: object,
            artifact_digest: str,
            _finished_block: int,
            *,
            calibration_manifest: Path,
            weight_soup_checkpoints: object,
        ) -> SimpleNamespace:
            del calibration_manifest, weight_soup_checkpoints
            return SimpleNamespace(
                artifact_digest=artifact_digest,
                calibration=SimpleNamespace(manifest_digest="sha256:" + "0" * 64),
            )

        mismatch_output = self.root / "mismatch.json"
        mismatch_request = dataclasses.replace(self.request, output=mismatch_output)
        with self.assertRaisesRegex(builder.ManifestBuildError, "did not bind"):
            self.build(mismatch_request, validator=wrong_binding)
        self.assertFalse(mismatch_output.exists())
        self.assert_no_temporary_manifest(mismatch_output)

    def test_existing_or_racing_output_is_never_overwritten(self) -> None:
        self.output.write_bytes(b"sentinel")
        with self.assertRaisesRegex(builder.ManifestBuildError, "already exists"):
            builder.build_calibration_manifest(self.request)
        self.assertEqual(self.output.read_bytes(), b"sentinel")

        race_output = self.root / "race.json"
        race_request = dataclasses.replace(self.request, output=race_output)

        def race(
            _training_dirs: object,
            artifact_digest: str,
            _finished_block: int,
            *,
            calibration_manifest: Path,
            weight_soup_checkpoints: object,
        ) -> SimpleNamespace:
            del weight_soup_checkpoints
            payload = Path(calibration_manifest).read_bytes()
            race_output.write_bytes(b"racing-writer")
            return SimpleNamespace(
                artifact_digest=artifact_digest,
                calibration=SimpleNamespace(manifest_digest=digest(payload)),
            )

        with self.assertRaisesRegex(builder.ManifestBuildError, "appeared"):
            self.build(race_request, validator=race)
        self.assertEqual(race_output.read_bytes(), b"racing-writer")
        self.assert_no_temporary_manifest(race_output)

    def test_checkout_is_rechecked_after_publication_validation(self) -> None:
        statuses = iter(("", " M convert_hf_to_gguf.py"))

        def changing_git(_repository: Path, arguments: tuple[str, ...]) -> str:
            if arguments[0] == "rev-parse":
                return builder.provenance.LLAMA_CPP_REVISION
            return next(statuses)

        with self.assertRaisesRegex(builder.ManifestBuildError, "clean worktree"):
            self.build(git_side_effect=changing_git)
        self.assertFalse(self.output.exists())
        self.assert_no_temporary_manifest()

    def test_rejects_symlink_ancestors_for_training_evidence_and_soup(self) -> None:
        view = self.root / "view"
        view.symlink_to(self.root, target_is_directory=True)
        cases = (
            ("training", {"training_dirs": ("view/stage",)}),
            (
                "evidence",
                {"converted_model": "view/evidence/model-f16.gguf"},
            ),
            (
                "soup",
                {"weight_soup_checkpoints": (("1", "view/soup"),)},
            ),
        )
        with mock.patch.object(
            builder, "_git_output", side_effect=self.clean_git
        ):
            for index, (label, changes) in enumerate(cases):
                output = self.root / f"ancestor-symlink-{index}.json"
                request = dataclasses.replace(
                    self.request,
                    output=output,
                    **changes,
                )
                with (
                    self.subTest(label=label),
                    self.assertRaisesRegex(
                        builder.ManifestBuildError,
                        "symlink component",
                    ),
                ):
                    builder.build_calibration_manifest(request)
                self.assertFalse(output.exists())

    def test_rejects_hardlink_aliases_across_evidence_roles(self) -> None:
        alias = self.root / "evidence" / "imatrix-hardlink.gguf"
        os.link(self.root / self.request.converted_model, alias)
        request = dataclasses.replace(
            self.request,
            output=self.root / "hardlink-alias.json",
            imatrix="evidence/imatrix-hardlink.gguf",
        )
        with (
            mock.patch.object(builder, "_git_output", side_effect=self.clean_git),
            self.assertRaisesRegex(builder.ManifestBuildError, "five distinct"),
        ):
            builder.build_calibration_manifest(request)
        self.assertFalse(request.output.exists())

    def test_replaced_temporary_path_cannot_install_unvalidated_bytes(self) -> None:
        replaced: list[Path] = []

        def replace_after_validation(
            _training_dirs: object,
            artifact_digest: str,
            _finished_block: int,
            *,
            calibration_manifest: Path,
            weight_soup_checkpoints: object,
        ) -> SimpleNamespace:
            del weight_soup_checkpoints
            path = Path(calibration_manifest)
            payload = path.read_bytes()
            path.unlink()
            path.write_bytes(b"unvalidated replacement")
            replaced.append(path)
            return SimpleNamespace(
                artifact_digest=artifact_digest,
                calibration=SimpleNamespace(manifest_digest=digest(payload)),
            )

        with self.assertRaisesRegex(
            builder.ManifestBuildError,
            "no longer names the created inode",
        ):
            self.build(validator=replace_after_validation)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0].read_bytes(), b"unvalidated replacement")
        replaced[0].unlink()
        self.assert_no_temporary_manifest()

    def test_same_inode_mutation_at_link_boundary_is_rejected_precommit(self) -> None:
        temporary: list[Path] = []

        def remember_temporary(
            _training_dirs: object,
            artifact_digest: str,
            _finished_block: int,
            *,
            calibration_manifest: Path,
            weight_soup_checkpoints: object,
        ) -> SimpleNamespace:
            del weight_soup_checkpoints
            path = Path(calibration_manifest)
            payload = path.read_bytes()
            temporary.append(path)
            return SimpleNamespace(
                artifact_digest=artifact_digest,
                calibration=SimpleNamespace(manifest_digest=digest(payload)),
            )

        original_linker = builder._link_bound_descriptor

        def mutate_then_link(
            descriptor: int,
            destination: Path,
            identity: tuple[int, int],
            expected_size: int,
            expected_fingerprint: tuple[int, int, int, int, int, int],
            expected_digest: str,
        ) -> object:
            path = temporary[0]
            payload = path.read_bytes()
            path.write_bytes(b"[" + payload[1:])
            path.chmod(0o600)
            return original_linker(
                descriptor,
                destination,
                identity,
                expected_size,
                expected_fingerprint,
                expected_digest,
            )

        with (
            mock.patch.object(
                builder,
                "_link_bound_descriptor",
                side_effect=mutate_then_link,
            ),
            self.assertRaisesRegex(
                builder.ManifestBuildError,
                "changed after final verification",
            ),
        ):
            self.build(validator=remember_temporary)

        self.assertFalse(self.output.exists())
        self.assertEqual(len(temporary), 1)
        self.assertFalse(temporary[0].exists())
        self.assert_no_temporary_manifest()

    def test_post_link_mutation_is_reported_as_committed_but_unverified(self) -> None:
        original_link = os.link

        def link_then_mutate(*args: object, **kwargs: object) -> None:
            original_link(*args, **kwargs)  # type: ignore[arg-type]
            descriptor = int(str(args[0]))
            self.assertEqual(os.pwrite(descriptor, b"[", 0), 1)
            os.fsync(descriptor)

        with mock.patch.object(
            builder.os,
            "link",
            side_effect=link_then_mutate,
        ):
            built = self.build()

        self.assertTrue(built.committed)
        self.assertFalse(built.installed_integrity_confirmed)
        self.assertTrue(self.output.exists())
        self.assertNotEqual(
            digest(self.output.read_bytes()),
            built.manifest_digest,
        )
        self.assertTrue(
            any(
                warning.startswith(
                    "installed manifest integrity verification failed after commit:"
                )
                for warning in built.post_commit_warnings
            )
        )
        self.assert_no_temporary_manifest()

    def test_cli_distinguishes_committed_but_unverified_output(self) -> None:
        built = self.build()
        unverified = dataclasses.replace(
            built,
            installed_integrity_confirmed=False,
            post_commit_warnings=("simulated post-commit integrity failure",),
        )
        arguments = [
            "--output",
            str(self.output),
            "--training-dir",
            "stage",
            "--source-model-dir",
            "stage/merged",
            "--converted-model",
            "evidence/model-f16.gguf",
            "--corpus",
            "evidence/calibration.txt",
            "--corpus-metadata",
            "evidence/calibration.metadata.json",
            "--imatrix",
            "evidence/model.imatrix.gguf",
            "--quantized-artifact",
            "candidate/model.gguf",
            "--finished-block",
            "123",
            "--llama-cpp-dir",
            str(self.llama_cpp),
        ]
        output = io.StringIO()
        with (
            mock.patch.object(
                builder,
                "build_calibration_manifest",
                return_value=unverified,
            ),
            redirect_stdout(output),
        ):
            status = builder.main(arguments)

        self.assertEqual(status, builder.POST_COMMIT_INTEGRITY_EXIT_STATUS)
        reported = json.loads(output.getvalue())
        self.assertTrue(reported["committed"])
        self.assertFalse(reported["installed_integrity_confirmed"])
        self.assertEqual(
            reported["post_commit_warnings"],
            ["simulated post-commit integrity failure"],
        )

    def test_post_commit_cleanup_and_fsync_failures_return_reported_success(
        self,
    ) -> None:
        with (
            mock.patch.object(
                builder,
                "_unlink_bound_path",
                return_value=(False, "simulated cleanup failure"),
            ),
            mock.patch.object(
                builder,
                "_fsync_directory",
                side_effect=OSError("simulated fsync failure"),
            ),
        ):
            built = self.build()

        self.assertTrue(built.committed)
        self.assertTrue(built.installed_integrity_confirmed)
        self.assertFalse(built.temporary_cleanup_complete)
        self.assertFalse(built.durability_confirmed)
        self.assertEqual(
            built.post_commit_warnings,
            (
                "simulated cleanup failure",
                "manifest directory fsync failed after commit: "
                "simulated fsync failure",
            ),
        )
        self.assertEqual(
            self.output.read_bytes(),
            builder._json_bytes(built.manifest),
        )
        temporaries = list(
            self.output.parent.glob(f".{self.output.name}.*.tmp")
        )
        self.assertEqual(len(temporaries), 1)
        self.assertEqual(
            (temporaries[0].stat().st_dev, temporaries[0].stat().st_ino),
            (built.installed_device, built.installed_inode),
        )
        temporaries[0].unlink()
        self.assert_no_temporary_manifest()

    def test_real_builder_to_publisher_fixture(self) -> None:
        from tests.test_publish_provenance import PublishProvenanceTests

        fixture = PublishProvenanceTests(
            "test_valid_single_stage_keeps_cli_and_config_compatibility"
        )
        fixture.setUp()
        try:
            stage, _metadata, _records, metadata_bytes = fixture.write_stage(
                "builder-stage"
            )
            _existing, _manifest, assets = fixture.write_calibration_manifest(
                stage,
                metadata_bytes,
            )
            llama_cpp = fixture.root / "llama.cpp"
            llama_cpp.mkdir()
            output = fixture.root / "built-calibration-lineage.json"

            def relative(path: Path) -> str:
                return path.relative_to(fixture.root).as_posix()

            request = builder.ManifestRequest(
                output=output,
                training_dirs=(relative(stage),),
                source_model_dir=relative(stage / "merged"),
                converted_model=relative(assets["converted"]),
                corpus=relative(assets["corpus"]),
                corpus_metadata=relative(assets["metadata"]),
                imatrix=relative(assets["imatrix"]),
                quantized_artifact=relative(assets["artifact"]),
                quantization_profile=builder.STANDARD_Q4_PROFILE,
                finished_block=8_955_436,
                llama_cpp_dir=llama_cpp,
            )
            with mock.patch.object(
                builder,
                "_git_output",
                side_effect=self.clean_git,
            ):
                built = builder.build_calibration_manifest(request)

            publication = builder.provenance.validate_publication(
                (stage,),
                built.artifact_tree_digest,
                request.finished_block,
                calibration_manifest=output,
                weight_soup_checkpoints={},
            )
            self.assertEqual(
                publication.calibration.manifest_digest,
                built.manifest_digest,
            )
            self.assertEqual(
                output.read_bytes(),
                builder._json_bytes(built.manifest),
            )
            self.assertTrue(built.committed)
            self.assertTrue(built.installed_integrity_confirmed)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
