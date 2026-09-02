# Microtensor miner launch TODO

Status snapshot: 2026-09-02 11:05 UTC

Target: Finney `netuid 92`, wallet `you-cold/you-hot1`, SS58
`5HgeNAYMw7piRNCNgGuRyaDnJUsoazZpxEbT7G7RukHSNw3r`, registration confirmed at UID `32`
(256 neurons, block 8,979,170), competition `code/mt-3g`.

This checklist distinguishes a process that is merely running from a miner that is
actually eligible to submit. Do not describe the launch as complete until every item
under **Launch acceptance** passes.

## Architecture

The deployment is now split across two hosts:

- **Miner host** — `dung-dev2` (address in the operator's records, not published
  here). Bare metal, Ubuntu
  22.04.2, Xeon E5-2667 v4, 64 GiB (`MemTotal` 67,303,641,088), 219 GB free on `/` plus
  two 880 GB NVMe. Runs the miner service and will host the rootless OCI conversion
  runner. Chosen because it can create user namespaces and therefore can regenerate
  `selfcheck.json`.
- **Training host** — the original GPU VPS. Retains the RTX 5090 (32 GiB, driver
  590.48.01, CUDA 13) and the training tree. Training only; its miner programs are
  stopped. It **cannot** host the miner long term: `unshare --user` and `unshare --net`
  both return EPERM, cgroup v2 is mounted read-only, `CAP_SYS_ADMIN` is absent, and `/`
  has under 2 GB free.

## Current state

- Sections 1 and 2 are **COMPLETE** on the miner host.
- `microtensor-upstream` and `microtensor-code-rank` run as the dedicated non-root
  `microtensor` account (uid 996). `microtensor-miner` is installed with
  `autostart=false` and is deliberately stopped, per section 2's last item.
- The upstream audit gate is **green**: `ok=true`, `phase=current`,
  `review_required=false`, `miner_impact_review_required=false`,
  `local_checkout_at_origin=true`, `commits_since_audit=0`, origin head
  `53e4df648a89fad6586e1ac69916b20e747fd972`, release `0.3.2`,
  `provenance_required=false`.
- The installed runtime tree matches the pinned signed v0.3.2 wheel exactly
  (137 files, 901,899 bytes, `sha256:f93d75ef…`). Keep `PYTHONDONTWRITEBYTECODE=1` on
  every invocation: bytecode caches under `microtensor/`, `neurons/` or the
  `microtensor_subnet-0.3.2.dist-info/` tree break this digest and hard-fail preflight.
- Only `you-hot1` and public metadata are in the service wallet; the copied hotkey was
  verified to derive the authorized SS58. No private coldkey exists on either host.
- Preflight and one `run --once` cycle both reach the intended fail-closed refusal:
  `artifact competition binding targets extract/mt-3g, but the controller targets
  code/mt-3g`. Exit 2, no authorization latch, no pending marker.
- The repository work is committed and pushed to `origin/main` at
  `65a6687637f8a649c2610436090fa9ae6105d485`.
- The 35 pinned conversion inputs (6,392,237,563 bytes, aggregate
  `sha256:323e2b10…`) are being copied off the training host's volatile tmpfs to
  `/nvme0n1-disk/microtensor/inputs-snapshot` on the miner host, with a full
  per-file digest re-verification on arrival.
- The current artifact is still bound to `extract/mt-3g` while the controller targets
  `code/mt-3g`. This mismatch is intentional and fail-closed. It is **not** a launchable
  code submission.
- `MT_RESULT_WORKER` is absent, and `ControllerConfig.from_env` now refuses to load if it
  is ever set. `tests/test_result_worker_guard.py` scans the tracked and untracked-not-
  ignored tree to keep it out.
- No model/artifact upload, signature, GitHub release, W&B upload, or chain transaction
  is pending or authorized by this checklist.
- The former runner candidate `administrator@93.127.134.170` is **retired**: it is
  707,244,032 bytes below the memory gate and the operator declined to resize it.

## Non-negotiable safety rules

- Keep `MMC_DRY_RUN=true` until the exact final `code/mt-3g` artifact and its binding
  have passed independent verification.
- Keep the installed runtime on the exact signed v0.3.2 wheel. Advancing the inert
  upstream observer checkout does not authorize installing unsigned `main`.
- Never set `MT_RESULT_WORKER`.
- Copy only `you-hot1` and required public wallet metadata into the service wallet.
  Never copy the coldkey or any other private hotkey.
- Never paste passwords, API keys, wallet JSON, seed phrases, or private keys into this
  repository, logs, shell history, or chat.
- Stop before any transaction unless registration is still UID 32 on netuid 92, the
  estimated fee is exactly `0 TAO`, the required deposit is exactly `0 TAO`, and the
  call is only the authorized Microtensor artifact commitment.
- No staking, transfer, registration, re-registration, cloud purchase, or other paid
  action is authorized.

## 1. Finish and review the repository migration — COMPLETE

- [ ] Finish the root-controlled service-file policy for:
  - `/etc/microtensor-miner/miner.env`
  - `/etc/microtensor-miner/github.token`
  - `/etc/microtensor-miner/artifact-competition.binding.json`
- [ ] Require exact `root:microtensor` ownership, mode `0640`, one hard link, bounded
  descriptor reads, `O_NOFOLLOW`, and before/after identity checks.
- [ ] Keep the service hotkey owned by `microtensor:microtensor`, mode `0600`, regular,
  non-symlink, and single-link.
- [ ] Finish the continuous preflight retry regression: safe upstream/binding refusals
  stay in-process; authorization refusal still latches and exits `3`; `--once` exits
  `2` on refusal without retrying.
- [ ] Complete the audit note for upstream head `53e4df...`, update both observer pin
  constants, and add a regression that prevents `MT_RESULT_WORKER` from entering miner
  configuration or deployment files.
- [ ] Update all Supervisor templates to run each program as `microtensor`, with explicit
  `HOME`, `USER`, `LOGNAME`, `XDG_CACHE_HOME`, `TMPDIR`, `PYTHONSAFEPATH`, and
  `PYTHONDONTWRITEBYTECODE` values.
- [ ] Update `.env.example`, launcher/preflight/health scripts, and `README.md` to use
  `/etc/microtensor-miner` and `/var/lib/microtensor-miner`.
- [ ] Review tool-created `*.orig` files and remove only the new redundant patch backups.
  Preserve the pre-existing `tests/test_state.py.orig` unless separately reviewed.
- [ ] Preserve the user's unrelated untracked `.github/` and conversion wrapper files;
  do not stage or modify them.

Required verification:

```bash
cd /workspace/microtensor-miner
env PYTHONPATH=tests PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m pytest -q
.venv/bin/ruff check src tests
git diff --check
git status --short
```

- [ ] Review the complete diff.
- [ ] Commit only the intended migration/audit files.
- [ ] Push the reviewed commit to `https://github.com/vandungtech/mt92.git`.

## 2. Provision the dedicated service account — COMPLETE (on the miner host)

Run these interactively as an administrator only after the repository tests pass:

```bash
sudo groupadd --system microtensor
sudo useradd --system \
  --gid microtensor \
  --home-dir /var/lib/microtensor-miner \
  --create-home \
  --shell /usr/sbin/nologin \
  microtensor

sudo install -d -o root -g microtensor -m 0750 /etc/microtensor-miner
sudo install -d -o microtensor -g microtensor -m 0700 \
  /var/lib/microtensor-miner \
  /var/lib/microtensor-miner/wallets \
  /var/lib/microtensor-miner/wallets/you-cold \
  /var/lib/microtensor-miner/wallets/you-cold/hotkeys \
  /var/lib/microtensor-miner/controller \
  /var/lib/microtensor-miner/upstream \
  /var/lib/microtensor-miner/upstream-observer \
  /var/lib/microtensor-miner/upstream-checkout \
  /var/lib/microtensor-miner/code-rank \
  /var/lib/microtensor-miner/artifact \
  /var/lib/microtensor-miner/tmp \
  /var/cache/microtensor-miner
sudo install -d -o microtensor -g microtensor -m 0750 /var/log/microtensor-miner
```

- [ ] Confirm `id microtensor` shows no `root`, `sudo`, `adm`, `lxd`, Docker, or other
  privileged supplemental group.
- [ ] Install only the exact `you-hot1` private hotkey into the new wallet path as
  `microtensor:microtensor 0600`. Copy public metadata only if the wallet library proves
  it is required. Do not copy the coldkey or other hotkeys.
- [ ] Verify the copied hotkey derives the exact authorized SS58 address before any
  network or signing operation.
- [ ] Install `miner.env`, `github.token`, and the artifact binding as
  `root:microtensor 0640`. Do not place token text on a command line.
- [ ] Set these paths in `miner.env` and keep W&B blank for signed v0.3.2:

```dotenv
MT_WALLET_PATH=/var/lib/microtensor-miner/wallets
MT_HOME=/var/lib/microtensor-miner/upstream
MMC_STATE_DIR=/var/lib/microtensor-miner/controller
MMC_ARTIFACT_DIR=/var/lib/microtensor-miner/artifact
MMC_ARTIFACT_COMPETITION_BINDING_PATH=/etc/microtensor-miner/artifact-competition.binding.json
MMC_SELFCHECK_PATH=/var/lib/microtensor-miner/upstream/selfcheck.json
MMC_SELFCHECK_BINDING_PATH=/var/lib/microtensor-miner/upstream/selfcheck.binding.json
MMC_UPSTREAM_OBSERVER_STATUS_PATH=/var/lib/microtensor-miner/upstream-observer/status.json
MMC_GITHUB_TOKEN_FILE=/etc/microtensor-miner/github.token
MMC_DRY_RUN=true
WANDB_API_KEY=
```

- [ ] Create the inert observer checkout as `microtensor`, detached at the accepted
  audited head. Do not import or execute checkout code.
- [ ] Back up the existing Supervisor configuration files to explicit dated filenames.
- [ ] Install the reviewed non-root Supervisor definitions, then run
  `supervisorctl reread` and `supervisorctl update`.
- [ ] Keep the miner program disabled/stopped until Sections 3 and 4 are complete.
- [ ] Keep the original root wallet and workspace runtime unchanged for rollback.

## 3. Provision the rootless OCI conversion runner — IN PROGRESS (on the miner host)

Runner: `administrator@93.127.134.170` (Linux/amd64).

The administrator must perform this interactively because passwordless `sudo` is not
available. Do not send the sudo password to an agent.

- [ ] Resize physical RAM to at least 36 GiB. Current RAM is 707,244,032 bytes below the
  strict 32 GiB minimum used by the worker specification.
- [ ] Provide `/dev/shm` of at least 17,179,869,184 bytes. Current `/dev/shm` is
  353,624,064 bytes short.
- [ ] Install and pin a rootless OCI stack: Podman, `crun`, `uidmap`,
  `fuse-overlayfs`, a reviewed rootless network helper, `jq`, and `rg`.
- [ ] Create separate unprivileged worker and verifier accounts. Neither may belong to
  `sudo`, `adm`, `lxd`, Docker, or another root-equivalent group.
- [ ] Allocate non-overlapping subordinate UID/GID ranges.
- [ ] Delegate CPU, memory, and pids controllers so the job can enforce exactly four
  CPUs and the specified memory limits. CPU delegation is currently missing.
- [ ] Install reviewed pinned seccomp and AppArmor profiles.
- [ ] Re-run a read-only runner preflight and save its exact JSON receipt.
- [ ] Do not transfer model input, build an image, or launch a workload until the final
  worker specification and input manifest contain no `UNRESOLVED:` values.

No machine resize, package purchase, or cloud charge is authorized by the existing
zero-TAO transaction authorization. If the host provider would charge money, stop and
ask first.

## 4. Produce and bind a real `code/mt-3g` artifact

- [ ] Obtain explicit authorization for the separately scoped characterization run.
  The existing transaction/publication authorizations do not authorize executing the
  model or converter merely to discover missing output identities.
- [ ] Characterize the pinned converter/reference output in the isolated runner.
- [ ] Resolve every worker-spec and input-manifest identity, digest, size, runtime,
  command, resource limit, and expected output field before launch.
- [ ] Transfer inputs only after both local and runner-side digests match.
- [ ] Run the conversion as the dedicated rootless worker, never as `administrator` or
  root.
- [ ] Verify returned artifacts and receipts as the separate verifier account.
- [ ] Copy the verified final GGUF and self-check data into the local non-root runtime.
- [ ] Confirm the GGUF load specification matches `code/mt-3g` and the signed v0.3.2
  evaluator requirements.
- [ ] Compute the exact artifact-tree digest.
- [ ] As root, create a new `root:microtensor 0640` competition binding containing that
  exact digest and `track=code`, `hardware_class=mt-3g`.
- [ ] Never relabel or reuse the current `extract/mt-3g` binding as a code artifact.

## 5. Read-only launch rehearsal

- [ ] Confirm the upstream observer reports the exact accepted head, fresh status,
  `review_required=false`, and `MT_RESULT_WORKER` absent.
- [ ] Confirm `you-hot1` is still registered at UID 32 on netuid 92.
- [ ] Confirm the protected configuration files are readable but not writable by
  `microtensor`.
- [ ] Confirm no other private hotkey exists in the service wallet.
- [ ] Run preflight and one dry-run cycle as the service account:

```bash
sudo -u microtensor env \
  HOME=/var/lib/microtensor-miner \
  XDG_CACHE_HOME=/var/cache/microtensor-miner \
  TMPDIR=/var/lib/microtensor-miner/tmp \
  MMC_ENV_FILE=/etc/microtensor-miner/miner.env \
  /workspace/microtensor-miner/scripts/preflight.sh

sudo -u microtensor env \
  HOME=/var/lib/microtensor-miner \
  XDG_CACHE_HOME=/var/cache/microtensor-miner \
  TMPDIR=/var/lib/microtensor-miner/tmp \
  /workspace/microtensor-miner/.venv/bin/microtensor-miner-controller \
  --env-file /etc/microtensor-miner/miner.env run --once
```

- [ ] Require phase `dry_run`, no pending marker, no authorization latch, and no package,
  upload, signature, release, or chain write.
- [ ] Start the continuous Supervisor program only after the one-shot dry run passes.
- [ ] Verify all three child processes run as UID `microtensor`, not UID 0.

## 6. Enable live round submission

Do this only after the dry-run rehearsal and a final operator review:

- [ ] Confirm the public GitHub repository is reachable and immutable releases are
  enabled.
- [ ] Confirm a current fine-grained GitHub token is installed only at
  `/etc/microtensor-miner/github.token`; never paste it into `miner.env`.
- [ ] Reconfirm exact artifact binding, registration, source length, anchored coordinator
  round, deadline margin, zero fee, and zero deposit.
- [ ] Set `MMC_DRY_RUN=false` with `sudoedit`; change no other value.
- [ ] Run one live `--once` cycle first.
- [ ] If any estimated cost is nonzero, any different call appears, registration changes,
  or verification is ambiguous, require exit `3`, preserve the latch/pending marker, and
  stop for operator review.
- [ ] Require exact anonymous source verification and exact on-chain readback before
  accepting phase `verified` and `ok=true`.
- [ ] Only then enable continuous Supervisor autostart.

## Launch acceptance

Launch is complete only when all of the following are true:

- [ ] `microtensor-miner`, `microtensor-upstream`, and `microtensor-code-rank` run as the
  dedicated non-root account.
- [ ] The signed runtime identity and upstream audit gate pass.
- [ ] The artifact is genuinely `code/mt-3g` and its exact digest matches the protected
  binding.
- [ ] The current round has a verified immutable public source and exact zero-cost
  on-chain commitment readback.
- [ ] Controller health is fresh with `phase=verified`, `ok=true`, and all source,
  full-source, provenance-policy, and on-chain proofs true.
- [ ] There is no authorization-refusal or unresolved submission-pending marker.

Useful status checks after migration:

```bash
supervisorctl status microtensor-miner microtensor-upstream microtensor-code-rank
sudo -u microtensor /workspace/microtensor-miner/.venv/bin/microtensor-miner-controller \
  --env-file /etc/microtensor-miner/miner.env status
sudo -u microtensor jq . /var/lib/microtensor-miner/code-rank/rank.json
```

Public leaderboard:

`https://api.microtensor.cloud/v1/arenas/code/mt-3g/leaderboard`

Reaching rank 1 is a later optimization goal, not evidence by itself that launch safety
or submission verification succeeded.

## Inputs still needed from the operator

1. Interactive administrator completion of the runner prerequisites in Section 3,
   including any provider-side resize. Ask before accepting a nonzero charge.
2. Explicit authorization for the isolated characterization run in Section 4.
3. A rotated/current GitHub token installed locally at the protected path if the current
   token has been revoked or is absent. Do not send the token in chat.

No W&B key is currently needed: the exact signed v0.3.2 runtime has
`PROVENANCE_REQUIRED=False`. Keep `WANDB_API_KEY=` blank unless a later signed release is
separately audited and enables provenance again.
