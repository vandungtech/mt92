# Qwen2.5 conversion worker handoff

## Your one required action

Provide a **separate Linux/amd64 rootless OCI job runner** with at least **4 CPU,
32 GiB RAM, 16 GiB tmpfs, and 20 GiB persistent workspace**. Do not provide
wallets, API tokens, SSH material, cloud credentials, or any other secret to
this repository or to the conversion container. Also provide a dedicated
**non-root verification account** that exclusively owns the finalized trust
directory and received export; the verifier refuses effective UID 0.

That external runner is mandatory. <code>training/convert_code_gguf.py</code> is
conversion logic, **not a containment boundary**, and it must not be run on this
host. The current host has no usable OCI runtime, has a read-only cgroup v2
mount, lacks <code>CAP_SYS_ADMIN</code>, refuses unprivileged mount/user
namespace creation, and has insufficient persistent free space for the reviewed
job. Those constraints make local conversion a hard **NO-GO**.

The checked-in worker spec and input manifest are intentionally
<code>incomplete_non_runnable</code>. No conversion launch is authorized until
every <code>UNRESOLVED:</code> value has been replaced with independently
established evidence, both documents have been independently reviewed, and the
verifier accepts their completed forms. Never replace a placeholder with a
guessed digest, key, profile, attestation, size, or output identity.

## Files in this handoff

- <code>deploy/conversion-worker/input-manifest.current94-v8.json</code> pins 35
  public input files totaling exactly 6,392,237,563 bytes.
- <code>deploy/conversion-worker/worker-spec.current94-v8.json</code> pins the
  platform, resource, mount, command, environment, llama.cpp runtime closure,
  and export protocol. It exposes all evidence that is still missing.
- <code>training/verify_code_conversion_export.py</code> is a standard-library,
  inert export verifier. It does bounded byte/JSON/GGUF parsing and never
  imports or executes the model, converter, corpus, generated code, or export.
- <code>tests/test_verify_code_conversion_export.py</code> uses synthetic
  fixtures only.

The hashes under <code>image.source_closure.files</code> are a capture of the
currently reviewed work in progress. The worktree is not a sealed build source,
so the final clean source snapshot and its tree digest remain unresolved.
Recompute those entries if the immutable build source differs.

## Required external worker

The operator must provision all of the following before asking for launch:

1. A Linux/amd64 OCI runtime operating rootless or with explicitly reviewed
   user-namespace remapping. The runtime must create private user, mount, PID,
   network, IPC, UTS, and cgroup namespaces.
2. Four CPU of quota, <code>memory.max=34359738368</code>,
   <code>memory.swap.max=0</code>, <code>pids.max=128</code>, a
   17,179,869,184-byte <code>/dev/shm</code> tmpfs, and at least
   21,474,836,480 bytes of persistent host workspace.
3. An immutable image referenced by <code>name@sha256:...</code>, its raw OCI
   config identity, a complete SBOM identity, and a clean source-closure
   identity.
4. A reviewed seccomp profile and a reviewed LSM profile (AppArmor, SELinux, or
   another host-enforced LSM), each supplied by exact filename, byte count, and
   SHA-256 digest.
5. No network egress, a read-only root filesystem,
   <code>no-new-privileges</code>, and an empty Linux capability set (equivalent
   to <code>--cap-drop=ALL</code>). Do not mount a Docker/Podman socket, the host
   home directory, <code>/run</code>, credential stores, wallet directories,
   host devices, or persistent export storage into the container.
6. Read-only, <code>rbind,nodev,noexec,nosuid</code> input mounts exactly as
   declared in the spec. The 20 GiB persistent workspace stays outside the
   container. The only writable in-container storage is the declared ephemeral
   tmpfs.
7. A worker receipt signing key held by the external runner control plane, not
   by this repository or container. Return only the public key. Also provide an
   exact, independently reviewed, self-contained static Linux/amd64 ELF
   verifier executable and its complete single-file closure identity. It must
   have no <code>PT_INTERP</code> or <code>PT_DYNAMIC</code>. Do not send the
   private key.
8. A concrete
   <code>microtensor.code.oci-runner-preflight.v1</code> evidence document,
   independently reviewed and byte-pinned in
   <code>runner_preflight_evidence</code>. It must cross-bind the runtime
   version/capabilities and the exact platform, resources, cgroup, security,
   mounts, image, OCI config, command, and child environment in the final spec.

Image construction and input delivery may happen before the job, but the job
must start with its image and inputs already present and run with networking
disabled. Package registries, metadata services, and remote model hubs must be
unreachable during execution.

## Known and unresolved evidence

The following values are known and pinned:

| Item | Pinned value |
| --- | --- |
| Base | Qwen/Qwen2.5-Coder-1.5B-Instruct at revision 2e1fd397ee46e1388853d2af2c993145b0f1098a |
| Input aggregate | 35 files, 6,392,237,563 bytes; SHA-256 323e2b10ab7aeabf0e6a09a6c4b2297a45b145b6a8dfebe7cd6075c8c8db42cb |
| llama.cpp source revision | c589f0ed10c643678c4707dd160c21ac7633ebc0 |
| Converter | 12,798 bytes; SHA-256 e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f |
| llama-imatrix | 343,128 bytes; SHA-256 3661d870d8645bb1c770328dcf2e4bf7f4bf076e70a6c8beabc1b60085499a35 |
| llama-quantize | 17,928 bytes; SHA-256 e7d4504b4db541f9a17ae920a8b505bc07159055400319ee056f4309bd800580 |
| Output contract | GGUF v3, qwen2, Q4_K_M (general.file_type=15), q4-m541 / maximum input 541 tokens, code-public-imatrix128-v1 |
| Model ceiling | At most 1,610,612,736 bytes (1.5 GiB) |

These must be supplied or calculated, never inferred:

- immutable image reference/digest, raw OCI config identity, complete SBOM
  identity, clean source snapshot/tree identity;
- immutable input-snapshot technology and evidence that host writers are
  excluded for the whole job, followed by the finalized manifest identity;
- seccomp and LSM profile identities;
- receipt signature scheme, key ID, trusted public-key identity, and pinned
  static-ELF offline-verifier identity/closure;
- concrete independently reviewed external-runner preflight evidence and its
  exact filename, size, and digest;
- exact identities of the four expected bundle files, obtained from an
  independently produced and reviewed deterministic reference artifact;
- named independent reviewers and review-record digests for the finalized
  manifest and worker spec.

There is no bootstrap exception for unknown output hashes. If no independent
reference artifact exists, stop and request a separately authorized
characterization procedure. Do not weaken the output pin or use this
non-runnable spec to manufacture the evidence it requires.

## Finalization

Work on copies of both JSON templates. Keep the immutable originals and review
records alongside the external runner, outside the hostile export.

1. Materialize every input into one immutable snapshot. Verify every file
   against the manifest, make the snapshot read-only, and exclude all other host
   writers. Replace the snapshot placeholders, set <code>sealed</code> true,
   set <code>status</code> to
   <code>sealed_and_independently_reviewed</code>, clear
   <code>unresolved</code>, and record an accepted independent review.
2. Calculate the exact finalized manifest identity:

~~~sh
mt_manifest=deploy/conversion-worker/input-manifest.current94-v8.json
stat -c '%s' "$mt_manifest"
sha256sum "$mt_manifest"
jq -cS '.inputs' "$mt_manifest" | sha256sum
~~~

   The last digest must remain
   323e2b10ab7aeabf0e6a09a6c4b2297a45b145b6a8dfebe7cd6075c8c8db42cb.
   Put the first two values in the worker spec's
   <code>input_manifest_identity</code>.
3. Resolve the image, OCI config, SBOM, clean source closure, seccomp profile,
   LSM profile, public receipt key, static offline verifier, and concrete runner
   preflight evidence identities. Each file identity is its exact regular-file
   byte count and SHA-256 digest. Keep the private signing key in the runner
   control plane.
4. Insert independently reviewed identities for all four expected bundle files.
   Set the worker spec status to
   <code>ready_and_independently_reviewed</code>, set
   <code>runnable</code> true, clear <code>unresolved</code>, and record the
   accepted independent review.
5. Have a second operator compare the generated OCI config to the JSON contract.
   Pin that raw config by bytes and digest. If the runtime cannot represent the
   contract exactly, refuse the launch.

The repository templates may retain normal source-control modes because their
placeholders make them NO-GO before any finalized-mode check. Finalized
materials must instead be installed into one private absolute directory owned
by the dedicated non-root verifier account. Run this provisioning block as an
administrator, then leave the administrator session; do not run the verifier
itself with <code>sudo</code> or effective UID 0:

~~~sh
set -eu
mt_verify_user=microtensor-verify
mt_verify_group=microtensor-verify
mt_private=/absolute/private/conversion-trust
test "$(id -u "$mt_verify_user")" -ne 0
install -d -o "$mt_verify_user" -g "$mt_verify_group" -m 0700 "$mt_private"
install -o "$mt_verify_user" -g "$mt_verify_group" -m 0600 /source/worker-spec.current94-v8.final.json "$mt_private/worker-spec.current94-v8.final.json"
install -o "$mt_verify_user" -g "$mt_verify_group" -m 0600 /source/input-manifest.current94-v8.final.json "$mt_private/input-manifest.current94-v8.final.json"
install -o "$mt_verify_user" -g "$mt_verify_group" -m 0600 /source/worker-receipt.pub "$mt_private/worker-receipt.pub"
install -o "$mt_verify_user" -g "$mt_verify_group" -m 0600 /source/runner-preflight.current94-v8.json "$mt_private/runner-preflight.current94-v8.json"
install -o "$mt_verify_user" -g "$mt_verify_group" -m 0500 /source/offline-receipt-verifier "$mt_private/offline-receipt-verifier"
~~~

Do not put the signing/private key there or on this host. Use explicit absolute
source paths. Every component of each absolute path must be a real directory,
not a symlink. In particular, do not put whitespace between a line-continuation
backslash and its newline.

The exact top-level argv is the <code>command</code> array in the worker spec.
Execute it directly with <code>cwd=/opt/microtensor-miner</code>,
<code>stdin=/dev/null</code>, umask <code>0077</code>, the exact declared
environment, and no shell.

The launcher is the exact reviewed <code>-c</code> bootstrap in the command
array under pinned <code>/opt/python/bin/python3.11 -I -B</code>. Isolated mode
does not put an absolute script's directory on <code>sys.path</code>, so the
bootstrap constructs an in-memory <code>training</code> package whose only
package path is <code>/opt/microtensor-miner/training</code>, then uses
<code>runpy.run_path</code> on the absolute reviewed
<code>convert_code_gguf.py</code>. This makes its reviewed sibling-module
imports resolvable without installing or trusting an ambient package. Do not
change the bootstrap, use <code>-m</code>, substitute <code>/bin/true</code>,
or add a shell wrapper.

## Execution and return protocol

The external control plane must:

1. Verify the sealed manifest, image/config/SBOM, profiles, key/verifier, source
   closure, and every input. Record preflight evidence.
2. Create one container with the exact OCI contract and start the exact argv
   once. Do not retry a failed tool start without a new authorization and review.
3. Wait for exit code zero. Refuse timeout, OOM kill, changed inputs, unexpected
   processes, or any remaining PID in the job cgroup. Hash all inputs again.
4. With the container stopped and cgroup empty, copy only the four bundle files
   from ephemeral storage into fresh private host storage. Remove the container
   before creating or signing the worker receipt.
5. Statically verify the copied bundle. Create <code>worker-receipt.json</code>
   cross-binding the exact spec, manifest, runner preflight evidence,
   image/SBOM, profiles, pre/post input checks, cgroup evidence, command, and
   four outputs. Sign its exact bytes outside the container.
6. Return exactly:

~~~text
export/
├── bundle/
│   ├── artifact/
│   │   └── model.gguf
│   ├── calibration-receipt.json
│   ├── conversion-receipt.json
│   └── load-spec.json
├── worker-receipt.json
└── worker-receipt.sig
~~~

Do not return the F16 model, imatrix, rendered calibration corpus, logs, caches,
root filesystem, inputs, private key, OCI state, or any other file. Transfer the
completed spec, sealed manifest, runner preflight evidence, trusted public key,
and offline verifier separately; they must remain outside
<code>export/</code>.

## Verification on this host

The template check should currently prove **NO-GO**:

~~~sh
cd /workspace/microtensor-miner
jq -e '.status == "incomplete_non_runnable" and .runnable == false' \
  deploy/conversion-worker/worker-spec.current94-v8.json
jq -e '.status == "incomplete_non_runnable" and .sealed == false' \
  deploy/conversion-worker/input-manifest.current94-v8.json
rg -n '"UNRESOLVED:' \
  deploy/conversion-worker/worker-spec.current94-v8.json \
  deploy/conversion-worker/input-manifest.current94-v8.json
~~~

Before launch, no placeholder may remain and both documents must be reviewed:

~~~sh
cd /workspace/microtensor-miner
! rg -n '"UNRESOLVED:' \
  deploy/conversion-worker/worker-spec.current94-v8.json \
  deploy/conversion-worker/input-manifest.current94-v8.json
jq -e '.status == "ready_and_independently_reviewed" and .runnable == true
       and .unresolved == [] and .independent_review.status == "accepted"' \
  deploy/conversion-worker/worker-spec.current94-v8.json
jq -e '.status == "sealed_and_independently_reviewed" and .sealed == true
       and .unresolved == [] and .independent_review.status == "accepted"' \
  deploy/conversion-worker/input-manifest.current94-v8.json
~~~

Then run the verifier's no-export/no-signature preflight against the private
final copies. This fully parses and validates the spec, manifest, public key,
static verifier closure, and external-runner evidence identities. It does not
open an export or invoke the signature verifier. Run it while logged in as the
dedicated owner, and prove the process is non-root first:

~~~sh
test "$(id -u)" -ne 0
/usr/bin/python3 -I -B /workspace/microtensor-miner/training/verify_code_conversion_export.py --preflight-only --worker-spec /absolute/private/conversion-trust/worker-spec.current94-v8.final.json --input-manifest /absolute/private/conversion-trust/input-manifest.current94-v8.final.json --trusted-public-key /absolute/private/conversion-trust/worker-receipt.pub --signature-verifier /absolute/private/conversion-trust/offline-receipt-verifier --runner-preflight-evidence /absolute/private/conversion-trust/runner-preflight.current94-v8.json
~~~

No external launch is authorized unless this prints an accepted preflight
receipt and a second operator reviews that receipt and its exact inputs.

After receiving the export, use only the separately pinned public key and
offline verifier. Receive the export directly as the same dedicated non-root
account into a newly created private parent. Do not first receive it as root and
then change ownership:

~~~sh
cd /workspace/microtensor-miner
umask 077
test "$(id -u)" -ne 0
/usr/bin/python3 -I -B /workspace/microtensor-miner/training/verify_code_conversion_export.py --export-root /absolute/private/path/export --worker-spec /absolute/private/conversion-trust/worker-spec.current94-v8.final.json --input-manifest /absolute/private/conversion-trust/input-manifest.current94-v8.final.json --trusted-public-key /absolute/private/conversion-trust/worker-receipt.pub --signature-verifier /absolute/private/conversion-trust/offline-receipt-verifier --runner-preflight-evidence /absolute/private/conversion-trust/runner-preflight.current94-v8.json
~~~

The export parent, export root, and both subdirectories must be mode
<code>0700</code>; all six returned files must be mode <code>0600</code>, be
regular non-links, have exactly one hard link, and be owned by the verifier
EUID. The private trust parent and every trust file must have that same owner.
Every absolute-path component is opened and held with no-follow directory
descriptors, and all six export files stay open on one identity from signature
verification through JSON/GGUF parsing and final recheck.

Treat a returned tree as hostile. Do **not** run <code>chmod</code>,
<code>chown</code>, <code>find -exec</code>, or another path-following
normalization command over it. Require the transfer process, running as the
dedicated account with umask <code>0077</code>, to create the exact modes above.
If any type, mode, owner, link count, or entry is wrong, discard that received
copy and request a fresh correctly created export; do not repair it in place.

The verifier refuses placeholders, null identities, duplicate JSON keys,
non-finite JSON, extra/missing export entries, links, special files, writable
paths, oversized files, a changed signature hook, failed/mis-bound receipts,
nonempty cgroups, changed inputs/image/config/profiles, a non-exact
Qwen2/Q4_K_M/541 load spec, a model over 1.5 GiB, and malformed or
misidentified GGUF v3 metadata. It never loads model tensors or imports export
content.

## Machine-checkable launch checklist

Every command must exit zero on the external runner before launch:

~~~sh
set -eu
mt_spec=/trusted/worker-spec.current94-v8.final.json
mt_inputs=/trusted/input-manifest.current94-v8.final.json
mt_runner_evidence=/trusted/runner-preflight.current94-v8.json
mt_public_key=/trusted/worker-receipt.pub
mt_signature_verifier=/trusted/offline-receipt-verifier
mt_uid=$(id -u)

test "$mt_uid" -ne 0
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
test "$(getconf _NPROCESSORS_ONLN)" -ge 4
test "$(awk '/MemTotal:/ {print $2 * 1024}' /proc/meminfo)" -ge 34359738368
test "$(df -PB1 /trusted | awk 'NR == 2 {print $4}')" -ge 21474836480
test "$(stat -c %a /trusted)" = 700
test "$(stat -c %a "$mt_spec")" = 600
test "$(stat -c %a "$mt_inputs")" = 600
test "$(stat -c %a "$mt_runner_evidence")" = 600
test "$(stat -c %a "$mt_public_key")" = 600
test "$(stat -c %a "$mt_signature_verifier")" = 500
test "$(stat -c %u /trusted)" = "$mt_uid"
test "$(stat -c %u "$mt_spec")" = "$mt_uid"
test "$(stat -c %u "$mt_inputs")" = "$mt_uid"
test "$(stat -c %u "$mt_runner_evidence")" = "$mt_uid"
test "$(stat -c %u "$mt_public_key")" = "$mt_uid"
test "$(stat -c %u "$mt_signature_verifier")" = "$mt_uid"

! rg -q '"UNRESOLVED:' "$mt_spec" "$mt_inputs" "$mt_runner_evidence"
test "$(jq -r '.schema' "$mt_runner_evidence")" = microtensor.code.oci-runner-preflight.v1
test "$(jq -r '.status' "$mt_runner_evidence")" = accepted
jq -e '.schema == "microtensor.code.oci-worker-spec.current94-v8.v1"
       and .status == "ready_and_independently_reviewed"
       and .runnable == true and .unresolved == []
       and .platform == {"architecture":"amd64","os":"linux",
         "rootless_or_userns_remapped":true}
       and .security.network_mode == "none"
       and .security.root_filesystem_read_only == true
       and .security.no_new_privileges == true
       and .security.capabilities == []
       and .cgroup == {"cpu_quota_cores":4,
         "memory_max_bytes":34359738368,
         "memory_swap_max_bytes":0,"pids_max":128}' "$mt_spec"
jq -e '.schema == "microtensor.code.oci-input-manifest.current94-v8.v1"
       and .status == "sealed_and_independently_reviewed"
       and .sealed == true and .unresolved == []
       and .file_count == 35 and .total_input_bytes == 6392237563
       and .aggregate_digest ==
         "sha256:323e2b10ab7aeabf0e6a09a6c4b2297a45b145b6a8dfebe7cd6075c8c8db42cb"' \
  "$mt_inputs"
test "$(jq -cS '.inputs' "$mt_inputs" | sha256sum | cut -d' ' -f1)" = \
  323e2b10ab7aeabf0e6a09a6c4b2297a45b145b6a8dfebe7cd6075c8c8db42cb
~~~

The OCI runtime must additionally emit evidence, later signed into the worker
receipt, that it used the exact image/config/profile identities,
rootless/userns-remapped mode, private namespaces, disabled networking,
read-only rootfs, empty capabilities, declared cgroup values, exact mounts, and
an empty cgroup before export. A successful process exit without that receipt is
not an acceptable result.
