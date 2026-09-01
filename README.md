# Microtensor miner controller

This repository supervises one registered Microtensor hotkey safely across round
boundaries. It targets netuid 92 on Finney, wallet `you-cold/you-hot1`, expected UID
32, and `code/mt-3g` under the signed Microtensor v0.3.0 mechanism. The install pin is
the official v0.3.0 release wheel; preflight independently requires its exact SHA-256
`742a25240a4a95f272c2894f5176a4f084b4b4eb5fb2af24dc8f7b464c2d0133`, signed
runtime constants, and activation block `8966795` before accepting a round.

The controller remains fail-closed unless the coordinator publishes a coherent,
chain-anchored v0.3 round. A release version string by itself is never treated as
identity proof.

It does **not** train a model or turn an empty wallet into a miner. A valid artifact,
self-check, public training provenance, and upload destination must exist first. The
controller defaults to dry-run and reports unhealthy until it proves a real submission.

## Why this controller exists

The upstream `mt miner run` process remembers publications only in memory and does not
repackage a manifest when the round changes. A manifest is signed for one specific
round, so simply recommitting it cannot keep a miner eligible. This controller performs
the complete transition for each newly opened round:

1. validate the exact signed upstream wheel and runtime identity;
2. unlock the configured hotkey and prove it currently maps to UID 32;
3. accept only a coherent, anchored coordinator round;
4. sign a fresh, explicitly unsealed manifest for that round;
5. upload the exact manifest assets to a unique immutable public GitHub Release;
6. download and hash the complete remote artifact;
7. verify W&B provenance against the artifact digest;
8. publish only while safely before the close block; and
9. read the exact source/round/digest commitment back from chain.

Only step 9 followed by all earlier proofs writes `"ok": true` to health state.

## Safety defaults

- `MMC_DRY_RUN=true` prevents packaging, signing, uploading, and chain writes.
- Coordinator failure, rollback, changed bounds, stale bounds, unanchored config,
  inconsistent phase, config-hash mismatch, or a closed competition all fail closed.
- Chain-schedule fallback is disabled. It is used only if
  `MMC_ALLOW_CHAIN_SCHEDULE_FALLBACK=true` is deliberately set.
- Sealed artifacts are categorically rejected. There is no reveal-key failure mode.
- Each source must contain `{round}` so a new manifest cannot overwrite another
  round's still-relevant namespace.
- Live upload accepts only
  `https:github.com/OWNER/REPO/releases/download/TAG`, with `{round}` in the tag.
  Arbitrary HTTPS, HF, IPFS, S3, and R2 live destinations are refused.
- The publisher enables and rechecks GitHub release immutability, creates or recovers a
  draft, streams each exact manifest asset, and never deletes or overwrites a release asset.
- A direct backend upload is refused in dry-run, closing accidental write bypasses.
- The GitHub token is read only from `MMC_GITHUB_TOKEN_FILE`; that file must be owned by
  the service user, be a regular non-symlink, and have mode exactly 0600.
- Logs and JSON state redact secret-looking assignments, bearer tokens, URL userinfo,
  query credentials, and secret environment values.

## Artifact prerequisites

The controller is not paired with a candidate until all validation gates pass. A
deployment must prepare these runtime inputs, which remain Git-ignored:

- `runtime/artifact/model.gguf` (the tokenizer is embedded in this GGUF);
- a valid upstream self-check JSON containing positive `size_bytes`,
  `peak_rss_bytes`, and `p95_latency_ms`;
- a base model locator pinned as `<org>/<repo>@<7-40 hex commit>`;
- a W&B run named by the hotkey SS58 address in
  `microtensor/training-runs`, with the required competition/base-model fields and
  the artifact digest;
- an initialized public GitHub repository with a compact owner/repository name; and
- a fine-grained GitHub token with repository Contents write and Administration write,
  stored in a separate service-user-owned mode-0600 regular file.

The controller does not fabricate any of these. Validators measure the artifact, so a
dummy model or invented provenance would only create an invalid on-chain submission.

## Installation

Python 3.10 or newer and Git are required. The dependency declaration installs the
official signed v0.3.0 wheel. Runtime preflight verifies its PEP 610 archive identity
and the exact digest above.

```bash
cd /opt/microtensor-miner
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install .
```

For local ONNX/GGUF self-check tooling, install `.[selfcheck]`. Training dependencies
belong to the separate training pipeline, not this always-on signer.

The runtime verifies PEP 610 `direct_url.json` metadata and refuses the signed v0.3
profile if the installed wheel cannot be proven. The legacy escape hatch
`MMC_ALLOW_UNVERIFIED_UPSTREAM=true` is accepted only in dry-run and is categorically
rejected when `MMC_V030_ACTIVATION_BLOCK` is set.

## Configuration

Copy `.env.example` outside the repository:

```bash
sudo install -d -m 0750 /etc/microtensor-miner
sudo install -m 0600 .env.example /etc/microtensor-miner/miner.env
sudoedit /etc/microtensor-miner/miner.env
```

Replace every placeholder value. Do not quote or add shell commands: the controller
parses a deliberately small `UPPER_CASE_NAME=value` data format and never sources it as
shell code. Keep it owned by the service user or root and mode 0600.

The service user must be the wallet owner or have read access to only the required
wallet tree. Do not copy wallet material into this Git repository or an image.

Use the exact Microtensor HTTPS locator form below; it intentionally has no `//` after
`https:`:

```dotenv
MMC_SOURCE_TEMPLATE=https:github.com/vandungtech/mt92/releases/download/r{round}
MMC_GITHUB_TOKEN_FILE=/etc/microtensor-miner/github.token
MMC_TRANSACTION_AUTHORIZATION=netuid92-uid32-you-hot1-commitment-fee0-deposit0-v1
MMC_V030_ACTIVATION_BLOCK=8966795
MMC_TRACK=code
MMC_HARDWARE_CLASS=mt-3g
MMC_ENTRYPOINT=model.gguf
MMC_ARTIFACT_FORMAT=gguf
MMC_QUANTIZATION=Q4_K_M
MMC_MAX_INPUT_TOKENS=512
MMC_TOKENIZER=tokenizer.json
MMC_BASE_MODEL=Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
```

The repository must already be public and initialized with at least one commit. Keep the
combined owner/repository name roughly 30 characters or fewer for current four-digit
rounds. The exact allowance depends on the round and digest; the controller validates
the complete 128-byte chain encoding before it performs an external upload.

Create the token as a separate file, never as an environment value or command argument:

```bash
sudo install -o SERVICE_USER -g SERVICE_GROUP -m 0600 /dev/null \
  /etc/microtensor-miner/github.token
sudoedit /etc/microtensor-miner/github.token
```

The file must contain only the token, with at most one final newline. The fine-grained
token needs Contents write to manage releases and Administration write to enable
immutable releases. Preflight rejects a symlink, non-regular or multiply linked file,
wrong owner, any mode other than 0600, malformed content, or a file that changes while read.

After publishing, the normal source verifier downloads and hashes the complete artifact
through the public HTTPS locator. GitHub credentials are not used for that fetch, so a
successful verification exercises the same anonymous route available to validators.

## Dry-run and activation

First leave `MMC_DRY_RUN=true`:

```bash
MMC_ENV_FILE=/etc/microtensor-miner/miner.env scripts/preflight.sh
MMC_ENV_FILE=/etc/microtensor-miner/miner.env \
  .venv/bin/microtensor-miner-controller run --once
```

Dry-run performs read-only upstream, wallet, metagraph, chain, and coordinator checks.
It writes local status/health files but does not rewrite the artifact, upload, sign, or
submit an extrinsic. A valid plan exits zero while health remains `ok: false` by design.

Live mode is supported only for the immutable GitHub source. Before setting
`MMC_DRY_RUN=false`, confirm all of the following:

- the compact repository exists, is initialized, and is publicly readable;
- the protected token file passes preflight and has the permissions described above;
- the exact artifact digest has admissible public W&B provenance; and
- the coordinator reports a coherent, anchored submissions phase with enough blocks left.

The authorization identifier does not relax any check and is not permission to spend
TAO. Immediately before signing, the controller refreshes the UID 32 registration,
verifies the audited Finney genesis and runtime identity, composes exactly
`Commitments.set_commitment(netuid=92)`, and checks `InitialDeposit=0` and
`FieldDeposit=0`. It estimates the exact nonce, mortal era, zero-tip envelope that it
would sign and requires `partial_fee=0`. The signed envelope is then decoded and checked
again before direct submission with `you-hot1`; MEV Shield is never invoked.

A nonzero or unprovable cost, changed registration/runtime, different call or signer, or
anomalous receipt writes a durable authorization-refusal latch and exits 3. A separate
submission-pending marker is flushed before broadcast and is cleared only after exact
on-chain readback. After a crash or ambiguous response the controller reconciles that
marker read-only and never blindly resubmits. Supervisor treats exit 3 as an expected
stop, so it will not retry until the operator has reviewed the condition.

S3 and R2 remain configuration diagnostics only and are categorically refused for live
upload. An interrupted draft is recoverable: matching assets are reused, missing assets
are uploaded, and a published release is accepted idempotently only when it is immutable
and its complete asset set matches. Any extra or mismatched asset fails without deletion
or overwrite.

After changing the environment, rerun preflight and exercise one live one-shot before
starting continuous submission:

```bash
MMC_ENV_FILE=/etc/microtensor-miner/miner.env scripts/preflight.sh
MMC_ENV_FILE=/etc/microtensor-miner/miner.env \
  .venv/bin/microtensor-miner-controller run --once
```

Only after the one-shot reaches verified health should continuous submission be enabled.

## Supervisord

`deploy/supervisord.conf` is a template. It assumes:

- repository and virtualenv at `/opt/microtensor-miner`;
- environment at `/etc/microtensor-miner/miner.env`;
- service user `microtensor`; and
- writable state below `/var/lib/microtensor-miner`.

Change `user=` to the wallet-owning OS account before enabling it. Supervisor retries
unexpected exits three times, forwards SIGTERM to the process group, and sends logs to
stdout. The controller itself handles transient coordinator/chain failures without ever
turning an unverified state green.

No inbound network port is required. The process uses outbound HTTPS/WSS for Finney,
the coordinator, W&B, the GitHub API, and public release downloads.

This workspace deployment also includes `deploy/supervisor-host.conf`, which points at
the actual `/workspace/microtensor-miner` paths and the protected runtime environment.
Installing that file under `/etc/supervisor/conf.d/` makes the controller part of the
host supervisor instead of depending on an interactive shell. It runs as root only
because this host's registered hotkey file is root-owned and mode 0600; do not copy or
relax the wallet permissions to run it under another account.

## Status and health

State is atomically replaced with mode 0600:

- `$MMC_STATE_DIR/status.json`: phase, sanitized plan/artifact details, trusted round,
  and proof timestamps;
- `$MMC_STATE_DIR/health.json`: compact supervisor-facing health;
- `$MMC_STATE_DIR/rank.json`: best-effort public leaderboard standing for the configured
  hotkey, refreshed independently every five minutes; and
- `$MMC_STATE_DIR/authorization-refusal.json`: durable fail-stop reason requiring
  operator review;
- `$MMC_STATE_DIR/submission-pending.json`: pre-broadcast recovery marker that prevents
  automatic duplicate submission; and
- `$MMC_STATE_DIR/controller.lock`: prevents duplicate signers.

```bash
.venv/bin/microtensor-miner-controller status
.venv/bin/microtensor-miner-controller health
scripts/healthcheck.sh
```

`health` exits zero only for a fully verified current-round state that has received a
recent controller heartbeat. Missing, dry-run, waiting without a commitment, refused,
error, or stale state exits 2.

A normal verified state includes all four proof flags:

```json
{
  "ok": true,
  "phase": "verified",
  "proofs": {
    "source": true,
    "source_full": true,
    "provenance": true,
    "on_chain": true
  }
}
```

The commitment itself is not copied to logs/state because a source may be sensitive;
only its SHA-256 fingerprint is retained.

## Rank-one observer

Continuous `run` mode starts a daemon observer for the official public
`code/mt-3g` leaderboard:

`https://api.microtensor.cloud/v1/arenas/code/mt-3g/leaderboard`

It matches the exact SS58 hotkey returned by wallet preflight and atomically replaces
`$MMC_STATE_DIR/rank.json` with mode 0600. The document records the server-supplied
rank, quality, expected cost in milliseconds, frontier membership, exclusive
hypervolume share, current rank-one leader, and whether the `rank == 1` goal is met.
It also records the leaderboard round and observation time so an old public board is
visible rather than mistaken for a current chain round.

```bash
jq . /var/lib/microtensor-miner/controller/rank.json
```

`reachability: true` means the HTTPS request and response schema succeeded. A reachable
board that does not contain this hotkey writes `found: false` and `rank: null`. A
timeout, HTTP error, malformed payload, wrong competition, or duplicate hotkey writes
`reachability: false`, clears the current rank fields, and leaves `goal_achieved: false`.

This observer is intentionally not an admission proof. It runs on a separate daemon
thread, never reads or changes `status.json` or `health.json`, and its output is never
consulted by packaging, upload, provenance, publication, or chain-verification code.
Public API failure therefore cannot block or authorise a submission. One-shot and
preflight commands do not start the periodic observer.

The host deployment also runs a credential-free observer for the target
`code/mt-3g` board, independently of any legacy or waiting submission profile. Its
latest snapshot is written to `runtime/code-rank/rank.json`; this observer can only
perform public HTTPS reads and local state writes.

## Round and restart behavior

The accepted coordinator window is persisted. The controller refuses a later response
that rolls back the round or mutates the same round's start, close, end, or config hash.
An old response whose bounds do not contain the live chain head is stale and refused.

On restart, a current local manifest is reused. The controller re-verifies source and
provenance, checks chain before publishing, and only sends an extrinsic when exact
readback is absent. This closes the crash window between a successful extrinsic and the
atomic status update.

When the next coordinator round opens, the old manifest is stale by definition. The
controller signs, uploads, verifies, and publishes a fresh per-round manifest
automatically. It will not start a write if fewer than
`MMC_DEADLINE_MARGIN_BLOCKS` remain, and it refreshes the trusted round immediately
before publishing.

## Tests

Core tests use fake backends and local fixtures:

```bash
PYTHONPATH=tests .venv/bin/python -m pytest -q
```

They do not import Bittensor, contact the coordinator, read wallets, install packages,
or submit anything.
