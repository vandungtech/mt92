#!/usr/bin/env python3
"""Fail-closed local bootstrap for a separate Microtensor code candidate.

This module intentionally has no training-framework dependency.  It validates
the immutable public corpus, prepares prompt/full-solution JSONL files, checks
the exact recommended base snapshot, and reports resource requirements.  The
actual trainer and evaluator import these checks; neither is allowed to fetch
anything implicitly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

AUDITED_UNSIGNED_UPSTREAM_COMMIT: Final[str] = "2a147a0c4a2c5810e72c4f414ad7bee9b7c8bdf6"
MECHANISM_VERSION: Final[str] = "0.2.0"
TRACK: Final[str] = "code"
HARDWARE_CLASS: Final[str] = "mt-3g"
CORPUS_VERSION: Final[str] = (
    "sha256:b38efa530c7fa11d7515c54d35a99b2b74235c71c597193537c3220bc9aacc66"
)
PUBLIC_CORPUS_URL: Final[str] = f"https://api.microtensor.cloud/v1/corpora/{CORPUS_VERSION}/public"
PUBLIC_CORPUS_CANONICAL_DIGEST: Final[str] = (
    "sha256:f126ea986aeeb45eecb3a63e850bbe2f6572c01d24142eed639b2dfbddcea4cd"
)
PUBLIC_CORPUS_RESPONSE_BYTES: Final[int] = 152_605
MAX_PUBLIC_CORPUS_BYTES: Final[int] = 1 << 20
EVALUATION_ENVIRONMENT_DIGEST: Final[str] = "env:a9b6b17587d8aaea"

EXPECTED_COUNTS: Final[dict[str, int]] = {"train": 94, "fixed": 126, "rotating": 412}
EXPECTED_TASKS_DIGEST: Final[str] = (
    "sha256:01cd0195b50f0dc663628aa9e3da2886a01a4dd36e59dfb1bafda37067077228"
)
EXPECTED_TESTS_DIGEST: Final[str] = (
    "sha256:56d29e02a50a083335b540077da05aae1b41edc576354c2fc19b0d07e4952d81"
)
EXPECTED_ENTRY_POINT: Final[str] = "task_func"
EXPECTED_SOURCE: Final[str] = "BigCodeBench"
EXPECTED_SOURCE_MANIFEST: Final[str] = "bigcode/bigcodebench at v0.1.4"
EXPECTED_LICENSE: Final[str] = "Apache-2.0"

ALLOWED_BASE_MODELS: Final[tuple[str, ...]] = (
    "Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
)
RECOMMENDED_BASE_MODEL: Final[str] = ALLOWED_BASE_MODELS[0]

# Exact file identities from the Hugging Face revision API with blobs=true.
# Non-LFS files use Git's SHA-1 blob identity; the LFS weight uses SHA-256.
RECOMMENDED_BASE_FILES: Final[dict[str, dict[str, int | str]]] = {
    "config.json": {"size": 659, "git_blob": "e2aa18293b0dd66539341467fa454878a5adb2be"},
    "generation_config.json": {
        "size": 243,
        "git_blob": "c28f9c697cbc09f047434efec57556505af31111",
    },
    "merges.txt": {
        "size": 1_671_839,
        "git_blob": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    },
    "model.safetensors": {
        "size": 988_097_824,
        "sha256": "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44",
    },
    "tokenizer.json": {
        "size": 7_031_645,
        "git_blob": "443909a61d429dff23010e5bddd28ff530edda00",
    },
    "tokenizer_config.json": {
        "size": 7_305,
        "git_blob": "acee076f49bf3c0298e15de0909d1da7b392f0c3",
    },
    "vocab.json": {
        "size": 2_776_833,
        "git_blob": "4783fe10ac3adce15ac8f358ef5462739852c569",
    },
}
RECOMMENDED_BASE_REQUIRED_BYTES: Final[int] = 999_586_348
RECOMMENDED_BASE_REPOSITORY_BYTES: Final[int] = 999_604_233
MIN_TRAINING_TMPFS_FREE_BYTES: Final[int] = 12 * 1024**3
MAX_SELECTED_ARTIFACT_BYTES: Final[int] = 1_610_612_736
TMPFS_MOUNT: Final[Path] = Path("/dev/shm")  # noqa: S108 - required volatile mount
DEFAULT_TMPFS_ROOT: Final[Path] = TMPFS_MOUNT / "microtensor-code"

ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {"counts", "manifest", "reference", "reference_model", "tasks", "track", "version"}
)
TASK_KEYS: Final[frozenset[str]] = frozenset(
    {"gold", "inputs", "max_output_tokens", "partition", "prompt", "ref"}
)
INPUT_KEYS: Final[frozenset[str]] = frozenset({"code_prompt", "entry_point", "libraries", "source"})
REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^bigcodebench-[0-9]+$")
DATASET_SCHEMA: Final[str] = "microtensor.code.prepared.v1"
SPLIT_ALGORITHM: Final[str] = "sha256_seed_ref_ascending_v1"
PREPARED_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {"completion", "max_output_tokens", "prompt", "ref"}
)
PREPARED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "track",
        "hardware_class",
        "corpus_version",
        "corpus_canonical_digest",
        "source_file_digest",
        "split_algorithm",
        "seed",
        "train_examples",
        "holdout_examples",
        "train_refs_digest",
        "holdout_refs_digest",
        "train_file_digest",
        "holdout_file_digest",
        "target_construction",
        "quality_claim",
    }
)


class CandidateError(ValueError):
    """A candidate input violates the pinned local contract."""


@dataclass(frozen=True)
class CorpusValidation:
    canonical_digest: str
    canonical_bytes: int
    task_count: int
    reference_count: int
    refs_digest: str


@dataclass(frozen=True)
class ResourcePlan:
    base_repository_download_bytes: int
    corpus_download_bytes: int
    minimum_tmpfs_free_bytes: int
    maximum_selected_artifact_bytes: int
    persistent_free_bytes: int
    tmpfs_free_bytes: int
    base_cached: bool
    corpus_cached: bool


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _strict_json(raw: bytes, source: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for key, value in pairs:
            if key in found:
                raise CandidateError(f"{source} repeats JSON key {key!r}")
            found[key] = value
        return found

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{source} is not strict UTF-8 JSON: {exc}") from exc


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CandidateError(f"{field} must be an array")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateError(f"{field} must be a non-empty string")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    keys = frozenset(str(key) for key in value)
    if keys != expected:
        raise CandidateError(
            f"{field} keys changed: expected {sorted(expected)}, got {sorted(keys)}"
        )


def _validate_manifest(raw: Any) -> frozenset[str]:
    manifest = _mapping(raw, "manifest")
    if manifest.get("track") != TRACK:
        raise CandidateError("manifest.track is not code")
    if manifest.get("counts") != EXPECTED_COUNTS:
        raise CandidateError("manifest counts differ from the pinned corpus")
    if manifest.get("metric") != "execution_pass_rate":
        raise CandidateError("manifest metric is not execution_pass_rate")
    if manifest.get("source") != EXPECTED_SOURCE_MANIFEST:
        raise CandidateError("manifest source changed")
    if manifest.get("license") != EXPECTED_LICENSE:
        raise CandidateError("manifest license changed")
    if manifest.get("ground_truth") != "module":
        raise CandidateError("manifest ground truth is not module tests")
    digests = _mapping(manifest.get("digests"), "manifest.digests")
    if digests.get("tasks") != EXPECTED_TASKS_DIGEST:
        raise CandidateError("manifest task digest changed")
    if digests.get("tests") != EXPECTED_TESTS_DIGEST:
        raise CandidateError("manifest hidden-test digest changed")
    convention = _mapping(manifest.get("binding_convention"), "binding_convention")
    if convention.get("entry_point") != EXPECTED_ENTRY_POINT:
        raise CandidateError("manifest entry point changed")
    if convention.get("runner") != "unittest":
        raise CandidateError("manifest test runner changed")
    libraries = _sequence(manifest.get("required_libraries"), "required_libraries")
    if any(not isinstance(item, str) or not item for item in libraries):
        raise CandidateError("manifest required_libraries contains a non-string")
    if len(set(libraries)) != len(libraries):
        raise CandidateError("manifest required_libraries contains duplicates")
    return frozenset(libraries)


def validate_public_corpus_payload(
    payload: Any,
    *,
    expected_canonical_digest: str | None = PUBLIC_CORPUS_CANONICAL_DIGEST,
) -> CorpusValidation:
    """Validate the exact public code release without executing any corpus code."""

    root = _mapping(payload, "public corpus")
    _exact_keys(root, ROOT_KEYS, "public corpus")
    if root.get("version") != CORPUS_VERSION:
        raise CandidateError("public corpus version changed")
    if root.get("track") != TRACK:
        raise CandidateError("public corpus track is not code")
    if root.get("counts") != EXPECTED_COUNTS:
        raise CandidateError("public corpus counts changed")
    if root.get("reference_model") != "" or root.get("reference") != []:
        raise CandidateError(
            "this release was pinned with no public reference completions; review any change"
        )
    allowed_libraries = _validate_manifest(root.get("manifest"))

    tasks = _sequence(root.get("tasks"), "tasks")
    if len(tasks) != EXPECTED_COUNTS["train"]:
        raise CandidateError("public task count does not equal the declared train count")
    refs: list[str] = []
    for index, raw_task in enumerate(tasks):
        task = _mapping(raw_task, f"tasks[{index}]")
        _exact_keys(task, TASK_KEYS, f"tasks[{index}]")
        ref = _nonempty_string(task.get("ref"), f"tasks[{index}].ref")
        if REF_PATTERN.fullmatch(ref) is None:
            raise CandidateError(f"task ref {ref!r} is outside the pinned source namespace")
        if task.get("partition") != "train":
            raise CandidateError(f"task {ref!r} is not in the public train partition")
        prompt = _nonempty_string(task.get("prompt"), f"task {ref!r} prompt")
        gold = _nonempty_string(task.get("gold"), f"task {ref!r} gold")
        if task.get("max_output_tokens") != 1024:
            raise CandidateError(f"task {ref!r} changed its output-token budget")

        inputs = _mapping(task.get("inputs"), f"task {ref!r} inputs")
        _exact_keys(inputs, INPUT_KEYS, f"task {ref!r} inputs")
        code_prompt = _nonempty_string(inputs.get("code_prompt"), f"task {ref!r} code_prompt")
        if code_prompt not in prompt:
            raise CandidateError(f"task {ref!r} prompt is not bound to its code prompt")
        if inputs.get("entry_point") != EXPECTED_ENTRY_POINT:
            raise CandidateError(f"task {ref!r} changed its entry point")
        if inputs.get("source") != EXPECTED_SOURCE:
            raise CandidateError(f"task {ref!r} changed its source")
        libraries = _sequence(inputs.get("libraries"), f"task {ref!r} libraries")
        if any(not isinstance(item, str) or not item for item in libraries):
            raise CandidateError(f"task {ref!r} libraries contains a non-string")
        if list(libraries) != sorted(set(libraries)):
            raise CandidateError(f"task {ref!r} libraries are not sorted and unique")
        if any(not isinstance(item, str) or item not in allowed_libraries for item in libraries):
            raise CandidateError(f"task {ref!r} names a library outside the manifest")

        # BigCodeBench exposes the completion suffix in gold.  The scorer needs
        # a complete solution module, so training targets code_prompt + gold.
        try:
            tree = ast.parse(code_prompt + gold, filename=ref)
        except SyntaxError as exc:
            raise CandidateError(f"task {ref!r} does not form valid complete Python") from exc
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == EXPECTED_ENTRY_POINT
            for node in tree.body
        ):
            raise CandidateError(f"task {ref!r} has no top-level task_func")
        refs.append(ref)

    if len(set(refs)) != len(refs):
        raise CandidateError("public corpus repeats a task ref")
    canonical = canonical_json_bytes(root)
    canonical_digest = digest_bytes(canonical)
    if expected_canonical_digest is not None and canonical_digest != expected_canonical_digest:
        raise CandidateError(
            "public response content changed without a reviewed local contract: "
            f"expected {expected_canonical_digest}, got {canonical_digest}"
        )
    return CorpusValidation(
        canonical_digest=canonical_digest,
        canonical_bytes=len(canonical),
        task_count=len(tasks),
        reference_count=0,
        refs_digest=digest_bytes(canonical_json_bytes(sorted(refs))),
    )


def load_public_corpus(path: Path) -> tuple[dict[str, Any], CorpusValidation]:
    if path.is_symlink() or not path.is_file():
        raise CandidateError(f"public corpus must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size > MAX_PUBLIC_CORPUS_BYTES:
        raise CandidateError(f"public corpus exceeds the {MAX_PUBLIC_CORPUS_BYTES}-byte cap")
    payload = _strict_json(path.read_bytes(), str(path))
    validation = validate_public_corpus_payload(payload)
    return dict(payload), validation


def assert_tmpfs_path(path: Path, *, must_exist: bool = False) -> Path:
    root = TMPFS_MOUNT.resolve(strict=True)
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise CandidateError(f"volatile path must be absolute and below {root}: {path}")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise CandidateError(f"volatile path cannot be resolved: {path}: {exc}") from exc
    if resolved == root or root not in resolved.parents:
        raise CandidateError(f"path must stay below volatile tmpfs {root}: {path}")
    return resolved


def _git_blob_digest(path: Path) -> str:
    size = path.stat().st_size
    hasher = hashlib.sha1()  # noqa: S324 - Git object identity, not security
    hasher.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_recommended_base_snapshot(path: Path) -> dict[str, Any]:
    root = assert_tmpfs_path(path, must_exist=True)
    if not root.is_dir():
        raise CandidateError(f"base snapshot is not a directory: {root}")
    files: dict[str, dict[str, int | str]] = {}
    for name, expected in RECOMMENDED_BASE_FILES.items():
        candidate = root / name
        if not candidate.is_file():
            raise CandidateError(f"base snapshot is missing {name}")
        resolved = candidate.resolve(strict=True)
        shm = TMPFS_MOUNT.resolve(strict=True)
        if resolved != shm and shm not in resolved.parents:
            raise CandidateError(f"base file {name} resolves outside /dev/shm")
        actual_size = candidate.stat().st_size
        if actual_size != expected["size"]:
            raise CandidateError(
                f"base file {name} has {actual_size} bytes, expected {expected['size']}"
            )
        if "sha256" in expected:
            identity = digest_file(candidate).removeprefix("sha256:")
            wanted = str(expected["sha256"])
            algorithm = "sha256"
        else:
            identity = _git_blob_digest(candidate)
            wanted = str(expected["git_blob"])
            algorithm = "git_blob_sha1"
        if identity != wanted:
            raise CandidateError(f"base file {name} does not match the pinned revision")
        files[name] = {"bytes": actual_size, algorithm: identity}
    if sum(int(value["bytes"]) for value in files.values()) != RECOMMENDED_BASE_REQUIRED_BYTES:
        raise CandidateError("base snapshot byte accounting changed")
    return {
        "base_model": RECOMMENDED_BASE_MODEL,
        "required_bytes": RECOMMENDED_BASE_REQUIRED_BYTES,
        "files": files,
    }


def _refs_digest(refs: Sequence[str]) -> str:
    return digest_bytes(canonical_json_bytes(list(refs)))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_dataset(
    corpus_path: Path,
    output: Path,
    *,
    holdout_examples: int,
    seed: int,
) -> dict[str, Any]:
    if isinstance(holdout_examples, bool) or not isinstance(holdout_examples, int):
        raise CandidateError("holdout_examples must be an integer")
    if not 0 <= holdout_examples < EXPECTED_COUNTS["train"]:
        raise CandidateError("holdout_examples must leave at least one training task")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CandidateError("seed must be an integer")
    out = assert_tmpfs_path(output)
    if out.exists():
        raise CandidateError(f"dataset output already exists: {out}")
    payload, validation = load_public_corpus(corpus_path)
    tasks = list(payload["tasks"])
    ranked = sorted(
        tasks,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['ref']}".encode()).hexdigest(),
            row["ref"],
        ),
    )
    heldout_refs = {str(row["ref"]) for row in ranked[:holdout_examples]}
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: str(row["ref"])):
        completion = str(task["inputs"]["code_prompt"]) + str(task["gold"])
        record = {
            "completion": completion,
            "max_output_tokens": task["max_output_tokens"],
            "prompt": task["prompt"],
            "ref": task["ref"],
        }
        (holdout_rows if task["ref"] in heldout_refs else train_rows).append(record)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        train_path = staging / "train.jsonl"
        holdout_path = staging / "holdout.jsonl"
        _write_jsonl(train_path, train_rows)
        _write_jsonl(holdout_path, holdout_rows)
        manifest = {
            "schema": DATASET_SCHEMA,
            "track": TRACK,
            "hardware_class": HARDWARE_CLASS,
            "corpus_version": CORPUS_VERSION,
            "corpus_canonical_digest": validation.canonical_digest,
            "source_file_digest": digest_file(corpus_path),
            "split_algorithm": SPLIT_ALGORITHM,
            "seed": seed,
            "train_examples": len(train_rows),
            "holdout_examples": len(holdout_rows),
            "train_refs_digest": _refs_digest([str(row["ref"]) for row in train_rows]),
            "holdout_refs_digest": _refs_digest([str(row["ref"]) for row in holdout_rows]),
            "train_file_digest": digest_file(train_path),
            "holdout_file_digest": digest_file(holdout_path),
            "target_construction": "inputs.code_prompt + gold",
            "quality_claim": (
                "none: the public release exposes no tests; holdout supports only "
                "structural diagnostics"
            ),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_prepared_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or frozenset(row) != PREPARED_ROW_KEYS:
                raise CandidateError(f"{label}:{number} has unexpected fields")
            ref = _nonempty_string(row["ref"], f"{label}:{number} ref")
            if REF_PATTERN.fullmatch(ref) is None:
                raise CandidateError(f"{label}:{number} has an invalid task ref")
            _nonempty_string(row["prompt"], f"{label}:{number} prompt")
            completion = _nonempty_string(row["completion"], f"{label}:{number} completion")
            if row["max_output_tokens"] != 1024:
                raise CandidateError(f"{label}:{number} changed its token budget")
            try:
                tree = ast.parse(completion, filename=ref)
            except SyntaxError as exc:
                raise CandidateError(f"{label}:{number} completion is not valid Python") from exc
            if not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == EXPECTED_ENTRY_POINT
                for node in tree.body
            ):
                raise CandidateError(f"{label}:{number} has no top-level task_func")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{label} is unreadable: {exc}") from exc
    return rows


def load_prepared_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = assert_tmpfs_path(path, must_exist=True)
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    holdout_path = root / "holdout.jsonl"
    prepared_files = (manifest_path, train_path, holdout_path)
    if any(candidate.is_symlink() or not candidate.is_file() for candidate in prepared_files):
        raise CandidateError("prepared dataset files must not be symlinks")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"prepared dataset manifest is unreadable: {exc}") from exc
    _exact_keys(
        _mapping(manifest, "prepared manifest"), PREPARED_MANIFEST_KEYS, "prepared manifest"
    )
    required = {
        "schema": DATASET_SCHEMA,
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "corpus_version": CORPUS_VERSION,
        "corpus_canonical_digest": PUBLIC_CORPUS_CANONICAL_DIGEST,
        "split_algorithm": SPLIT_ALGORITHM,
        "target_construction": "inputs.code_prompt + gold",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CandidateError(f"prepared dataset manifest field {key!r} changed")
    for data_path, digest_key in (
        (train_path, "train_file_digest"),
        (holdout_path, "holdout_file_digest"),
    ):
        if digest_file(data_path) != manifest.get(digest_key):
            raise CandidateError(f"prepared dataset {data_path.name} digest changed")
    rows = _load_prepared_rows(train_path, "train.jsonl")
    holdout_rows = _load_prepared_rows(holdout_path, "holdout.jsonl")
    for found, count_key, refs_key, label in (
        (rows, "train_examples", "train_refs_digest", "training"),
        (holdout_rows, "holdout_examples", "holdout_refs_digest", "holdout"),
    ):
        if len(found) != manifest.get(count_key):
            raise CandidateError(f"prepared {label} row count differs from its manifest")
        refs = [str(row["ref"]) for row in found]
        if len(set(refs)) != len(refs) or _refs_digest(refs) != manifest.get(refs_key):
            raise CandidateError(f"prepared {label} refs differ from their manifest")
    train_refs = {str(row["ref"]) for row in rows}
    holdout_refs = {str(row["ref"]) for row in holdout_rows}
    if train_refs & holdout_refs:
        raise CandidateError("prepared train and holdout refs overlap")
    if len(train_refs | holdout_refs) != EXPECTED_COUNTS["train"]:
        raise CandidateError("prepared dataset does not contain all 94 public task refs")
    return rows, manifest


def resource_plan(
    *,
    persistent_path: Path,
    tmpfs_root: Path = DEFAULT_TMPFS_ROOT,
    base_path: Path | None = None,
    corpus_path: Path | None = None,
) -> ResourcePlan:
    tmpfs = assert_tmpfs_path(tmpfs_root)
    tmpfs_probe = tmpfs if tmpfs.exists() else tmpfs.parent
    persistent_probe = persistent_path.resolve(strict=True)
    return ResourcePlan(
        base_repository_download_bytes=RECOMMENDED_BASE_REPOSITORY_BYTES,
        corpus_download_bytes=PUBLIC_CORPUS_RESPONSE_BYTES,
        minimum_tmpfs_free_bytes=MIN_TRAINING_TMPFS_FREE_BYTES,
        maximum_selected_artifact_bytes=MAX_SELECTED_ARTIFACT_BYTES,
        persistent_free_bytes=shutil.disk_usage(persistent_probe).free,
        tmpfs_free_bytes=shutil.disk_usage(tmpfs_probe).free,
        base_cached=bool(base_path and base_path.is_dir()),
        corpus_cached=bool(corpus_path and corpus_path.is_file()),
    )


def fetch_public_corpus(output: Path) -> CorpusValidation:
    destination = assert_tmpfs_path(output)
    if destination.exists():
        raise CandidateError(f"refusing to overwrite existing corpus: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        PUBLIC_CORPUS_URL,
        headers={"Accept": "application/json", "User-Agent": "microtensor-code-bootstrap/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_PUBLIC_CORPUS_BYTES:
                raise CandidateError("server declared an oversized public corpus")
            raw = response.read(MAX_PUBLIC_CORPUS_BYTES + 1)
    except CandidateError:
        raise
    except Exception as exc:
        raise CandidateError(f"public corpus fetch failed: {exc}") from exc
    if len(raw) > MAX_PUBLIC_CORPUS_BYTES:
        raise CandidateError("public corpus exceeded the download cap")
    payload = _strict_json(raw, PUBLIC_CORPUS_URL)
    validation = validate_public_corpus_payload(payload)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    fetch = actions.add_parser("fetch", help="fetch and validate the 152,605-byte corpus")
    fetch.add_argument("--output", type=Path, default=DEFAULT_TMPFS_ROOT / "public.json")

    validate = actions.add_parser("validate", help="validate an existing public response")
    validate.add_argument("corpus", type=Path)

    prepare = actions.add_parser("prepare", help="prepare offline SFT JSONL in tmpfs")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--holdout-examples", type=int, default=16)
    prepare.add_argument("--seed", type=int, default=92)

    plan = actions.add_parser("plan", help="print byte-exact download and disk requirements")
    plan.add_argument("--persistent-path", type=Path, default=Path.cwd())
    plan.add_argument("--tmpfs-root", type=Path, default=DEFAULT_TMPFS_ROOT)
    plan.add_argument("--base", type=Path, default=None)
    plan.add_argument("--corpus", type=Path, default=None)

    base = actions.add_parser("verify-base", help="hash the exact offline 0.5B base snapshot")
    base.add_argument("path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "fetch":
            result: Any = asdict(fetch_public_corpus(args.output))
        elif args.action == "validate":
            _payload, validation = load_public_corpus(args.corpus)
            result = asdict(validation)
        elif args.action == "prepare":
            result = prepare_dataset(
                args.corpus,
                args.output,
                holdout_examples=args.holdout_examples,
                seed=args.seed,
            )
        elif args.action == "verify-base":
            result = verify_recommended_base_snapshot(args.path)
        else:
            result = asdict(
                resource_plan(
                    persistent_path=args.persistent_path,
                    tmpfs_root=args.tmpfs_root,
                    base_path=args.base,
                    corpus_path=args.corpus,
                )
            )
    except (CandidateError, OSError) as exc:
        raise SystemExit(f"code candidate bootstrap refused: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
