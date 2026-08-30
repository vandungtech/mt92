# Microtensor miner controller

This repository supervises one registered Microtensor hotkey safely across round
boundaries. It is pinned to upstream commit
`73815395e03daa0d6a25ad93030bdb5124cb61cf` (release `0.1.14`), netuid 92 on
Finney, wallet `you-cold/you-hot1`, expected UID 32, and the currently live
`extract/mt-3g` competition.

It does **not** train a model or turn an empty wallet into a miner. A valid artifact,
self-check, public training provenance, and upload destination must exist first. The
controller defaults to dry-run and reports unhealthy until it proves a real submission.

## Why this controller exists

The upstream `mt miner run` process remembers publications only in memory and does not
repackage a manifest when the round changes. A manifest is signed for one specific
round, so simply recommitting it cannot keep a miner eligible. This controller performs
the complete transition for each newly opened round:

1. validate the exact installed upstream commit;
2. unlock the configured hotkey and prove it currently maps to UID 32;
3. accept only a coherent, anchored coordinator round;
4. sign a fresh, explicitly unsealed manifest for that round;
5. upload to a unique per-round S3/R2 prefix;
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
- Live autonomous upload permits only `s3:` and `r2:`. Upstream accepts `ipfs:` but
  cannot upload or fetch it; `https:` has no uploader; an HF commit SHA is immutable
  and cannot receive each round's new manifest.
- Logs and JSON state redact secret-looking assignments, bearer tokens, URL userinfo,
  query credentials, and secret environment values.

## Artifact prerequisites

This checked-in deployment is already paired with the locally built artifact. A fresh
deployment must prepare these runtime inputs, which remain Git-ignored:

- `runtime/artifact/model.gguf` (the tokenizer is embedded in this GGUF);
- a valid upstream self-check JSON containing positive `size_bytes`,
  `peak_rss_bytes`, and `p95_latency_ms`;
- a base model locator pinned as `<org>/<repo>@<7-40 hex commit>`;
- a W&B run named by the hotkey SS58 address in
  `microtensor/training-runs`, with the required competition/base-model fields and
  the artifact digest; and
- a validator-fetchable S3/R2 namespace plus uploader credentials.

The controller does not fabricate any of these. Validators measure the artifact, so a
dummy model or invented provenance would only create an invalid on-chain submission.

## Installation

Python 3.10 or newer and Git are required. The dependency declaration installs the
upstream project directly at the reviewed commit.

```bash
cd /opt/microtensor-miner
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install .
```

For local ONNX/GGUF self-check tooling, install `.[selfcheck]`. Training dependencies
belong to the separate training pipeline, not this always-on signer.

The runtime verifies PEP 610 `direct_url.json` metadata and refuses live operation if
the installed upstream commit cannot be proven. The escape hatch
`MMC_ALLOW_UNVERIFIED_UPSTREAM=true` is accepted only in dry-run.

## Configuration

Copy `.env.example` outside the repository:

```bash
sudo install -d -m 0750 /etc/microtensor-miner
sudo install -m 0600 .env.example /etc/microtensor-miner/miner.env
sudoedit /etc/microtensor-miner/miner.env
```

Replace every `REPLACE_...` value. Do not quote or add shell commands: the controller
parses a deliberately small `UPPER_CASE_NAME=value` data format and never sources it as
shell code. Keep it owned by the service user or root and mode 0600.

The service user must be the wallet owner or have read access to only the required
wallet tree. Do not copy wallet material into this Git repository or an image.

The upload namespace should be public to validators. The controller verifies remote
bytes with the configured runtime credentials; that check does not prove an unrelated
validator possesses private bucket credentials.

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

This audited snapshot deliberately rejects `MMC_DRY_RUN=false`. The upstream S3/R2
probe uses the miner's object-store credentials, which cannot prove that an unrelated
validator can fetch the same bytes anonymously. Activation therefore also requires a
validator-anonymous source verifier, public W&B provenance, a real immutable public
destination, and an open coordinator round. After implementing and auditing that
verifier and removing the explicit fail-closed gate in `config.py`, exercise a live
one-shot before starting continuous submission:

```bash
MMC_ENV_FILE=/etc/microtensor-miner/miner.env \
  .venv/bin/microtensor-miner-controller run --once
```

Only then should `scripts/run-controller.sh` be used for continuous submission.

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
the coordinator, W&B, and the artifact store.

## Status and health

State is atomically replaced with mode 0600:

- `$MMC_STATE_DIR/status.json`: phase, sanitized plan/artifact details, trusted round,
  and proof timestamps;
- `$MMC_STATE_DIR/health.json`: compact supervisor-facing health; and
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

Core tests use only the standard library and fake backends:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

They do not import Bittensor, contact the coordinator, read wallets, install packages,
or submit anything.
