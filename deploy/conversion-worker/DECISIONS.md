# Conversion handoff: recorded operator decisions

Recorded 2026-09-02. The operator delegated these to the agent with the instruction
"for all questions, just do your recommendations", stating the goal as live mining as
soon as possible. Each decision below is the agent's recommendation, adopted. They are
written down because several of them reduce assurance relative to what
`docs/CONVERSION_WORKER.md` asks for, and that reduction must be legible rather than
implicit.

## 1. Independence: SELF-ATTESTED, not independent

`docs/CONVERSION_WORKER.md` asks for named independent reviewers of the finalized worker
spec and sealed manifest, a second operator to compare the generated OCI config to the
JSON contract, and a second operator to review the verification receipt. The deployment
is operated by one person, so none of those roles is filled by a second party.

`_validate_review` in `training/verify_code_conversion_export.py` accepts any non-empty
`reviewer` string with a well-formed `review_digest`. Putting an invented name there
would make the verifier pass while asserting something false. Instead, the review fields
record the operator as the reviewer and this file as the review record, and the chain is
described as self-attested wherever it is reported.

**What is lost:** the signature and the review fields attest "the operator's own process
checked this", not "an independent party checked this". Every mechanical binding the
verifier performs — image identity, cgroup evidence, input digests before and after,
profile identities, GGUF structure, the load spec, the size ceiling — is still genuinely
enforced. What is not enforced is a second pair of eyes.

## 2. One conversion run, identities pinned from it

The two receipts cannot be pre-pinned. `conversion-receipt.json` and
`calibration-receipt.json` embed `started_at_unix_ns` / `finished_at_unix_ns` for every
command, including the determinism replay, and they embed SHA-256 digests of the raw
unfiltered stdout/stderr of `convert_hf_to_gguf.py`, `llama-imatrix` and
`llama-quantize`, which print timing. Two runs cannot produce identical receipt bytes.

`_validate_worker_receipt` requires `spec.expected_output.file_identities` to equal the
identities of the bytes actually exported. A characterize-then-reproduce scheme therefore
cannot succeed for those two files, no matter how the run is authorized.

The decision is to perform **one** authorized conversion run under the full security
contract, pin all four identities from that exact export, and verify that export. The
`model.gguf` itself is bit-reproducible — the pipeline's own determinism replay re-runs
all three stages and compares artifacts — so the artifact's reproducibility is
established independently of the receipts.

**What is lost:** the guarantee that the outputs were predicted before the run. This is
the specific thing `docs/CONVERSION_WORKER.md` warns against under "There is no bootstrap
exception for unknown output hashes". It is accepted knowingly, and it is a consequence
of decision 1: with an independent second party, the reference artifact could have come
from them instead.

## 3. Signature scheme: Ed25519, verifier built in-house

`receipt_signature.scheme` is `ed25519`. `key_id` is the lowercase hex SHA-256 of the raw
32-byte public key. The trusted public key file is 64 lowercase hex characters with at
most one trailing newline; the detached signature file is 128 lowercase hex characters
with at most one trailing newline.

The offline verifier is `deploy/conversion-worker/offline-receipt-verifier/main.go`,
built with `CGO_ENABLED=0` so the result is a static ELF64 with no `PT_INTERP` and no
`PT_DYNAMIC`, as `_validate_static_verifier_elf` requires. Ed25519 comes from the Go
standard library; there is no third-party cryptographic dependency.

## 4. Signing key lives on the training host

The worker-receipt signing private key is generated and held on the **training** host,
which runs no container and holds no miner hotkey. Only the public key travels to the
runner. This is materially better than the single-machine arrangement, where the signing
key would have sat on the same kernel as the hotkey and the GitHub write token.

## 5. LSM: userns-remapped rootful Podman

Rootless Podman refuses a named AppArmor profile — it accepts only `unconfined` — while
`security.profiles.lsm` requires a named identity and
`_validate_runner_preflight_evidence` re-checks `evidence.security == spec.security`.
`security.runtime_mode` is `rootless_or_userns_remapped`, so running userns-remapped
**rootful** Podman is permitted by the contract and is the option that keeps a real,
pinned, enforced LSM profile. The profile is installed by root at provisioning time
(`apparmor_parser` needs `CAP_MAC_ADMIN`) and pinned by exact bytes and SHA-256
afterwards.

## 6. Source closure corrected

`image.source_closure.files` was missing `training/train_code.py` (54,031 bytes) and
`training/evaluate_code.py` (51,409 bytes). Both are in the transitive import closure of
`convert_code_gguf.py`, so an image built to the previous list would have failed at
runtime with `ImportError`. They are now listed.
