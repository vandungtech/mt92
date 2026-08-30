# Artifact record

The active deployable weights are intentionally ignored by Git and live at
`runtime/artifact/model.gguf`. This record binds those exact local bytes to
their reviewed training, calibration, evaluation, and self-check evidence
without committing a 396 MB binary. It does not describe any soup,
augmentation, or other candidate still under evaluation.

## Identity

- Track/class: `extract/mt-3g`
- Format/quantization: GGUF v3 / `Q4_K_M`
- Entrypoint size: `396704736` bytes
- Entrypoint SHA-256:
  `sha256:fe0f34195627765155ecd98a309052c5efb5a4b3e977bce7cfbbe6ba564162e0`
- Microtensor artifact digest:
  `sha256:316e9d2b06184e0f9c0f385e1f4850a96879b755f41bb7d4966637142ff144ee`
- Artifact tree: one file, `model.gguf`, totaling `396704736` bytes
- Base:
  `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
- Public corpus version:
  `sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8`
- Public corpus file SHA-256:
  `sha256:fb5f1332493b1abe759da91cec1bd2cdd932c7076c5fae8163b5f33cfbea05a2`
- Converter, imatrix engine, and quantizer: llama.cpp
  `c589f0ed10c643678c4707dd160c21ac7633ebc0`

GGUF embeds the tokenizer and chat template, so the active publishable tree
contains only `model.gguf`. The bound load specification fixes
`max_input.tokens=512` and a `tokenizer.json` preprocessing reference. Its
canonical SHA-256 is
`sha256:12f805e5e63a0b48954cef98c6579b61b88ef03a27d658df632df687c0ec6e1e`.

## Training lineage

Stage one is
`runtime/training/experiments/lora-r64-e2-seed92`; its metadata SHA-256 is
`sha256:178040ccd12dc855d0584d2f0505c4086fe7da6e51c08000b8295736f2aab9f7`.
It trained a rank-64, alpha-128 LoRA for two epochs with seed 92, learning rate
`2e-4`, batch size 8, gradient accumulation 4, maximum length 512, warmup
ratio 0.05, and weight decay 0.01. It used 4,430 public examples and held out
384 deterministic examples.

Before subsequent selection, that stage scored `0.9362637362637363` entity
micro-F1 on the 384-example reserve: 639 true positives, 42 false positives,
45 false negatives, and zero malformed outputs. The summary evidence SHA-256
is `sha256:e55919d79a76ae28500964152e1b72f344fb78afcef727e15b3ad3dc59c4a10c`;
the row-level JSONL SHA-256 is
`sha256:1cbb039f05cf79dd0bb33312370bb9e6953e8668e3b10be3a1ed547778d8da8b`.

Stage two is
`runtime/training/experiments/lora-r64-e2-full-e1`; its metadata SHA-256 is
`sha256:965c04d58bb9251a1eba8c543b3a1c7bcc80c2491e3e3bb0de9b868f455d0ecc`
and it names the stage-one metadata digest as its parent. It ran one epoch over
all 4,814 usable public examples with seed 92, learning rate `5e-5`, rank 64,
alpha 128, batch size 8, gradient accumulation 4, maximum length 512, warmup
ratio 0.05, and weight decay 0.01.

## Public-only imatrix calibration

The active bytes were quantized from stage two with a final all-public
importance matrix. The deterministic calibration corpus used seed 92, reserve
size 0, maximum 512 examples, and selected 512 of 4,815 eligible public rows;
one public row was rejected because its gold text was not a source substring.
Thinking and generation prompts were disabled, and gold entities were rendered
as compact, sorted-key JSON after exact text/type deduplication.

The calibration evidence is bound as follows:

- rendered text:
  `runtime/training/imatrix/final-seed92-r0-n512.txt`,
  310901 bytes,
  `sha256:0588fa83ddf7768846d5e491c7878b2c7b66fe4ba0c9afd12ca4fd118b82bf9d`
- metadata sidecar:
  `runtime/training/imatrix/final-seed92-r0-n512.txt.metadata.json`,
  `sha256:7be2a3835d05781667e80f4d4aff55ccad55e548c6c75a17655a5fb815fb105b`
- final importance matrix:
  `runtime/training/imatrix/final-seed92-r0-n512.imatrix.gguf`,
  1177088 bytes,
  `sha256:ba06cfd26ab829208c3068d02fa9165084ea58cab6a4dee31a989b3cc8d52f41`

The imatrix pass used the pinned CUDA build in offline mode, context size 512,
`--no-ppl`, `--parse-special`, no `--process-output`, and all 142
available chunks. Quantization used that exact matrix and `Q4_K_M`.

On the untouched stage-one reserve, the ordinary Q4 control scored
`0.921496698459281` (628 TP, 51 FP, 56 FN, one malformed), while the
same-class calibrated Q4 control scored `0.9306062819576333` (637 TP, 48 FP,
47 FN, zero malformed). This supports the calibration method, but repeated
comparisons against the reserve make it selection evidence rather than a fresh
unbiased estimate.

The uncalibrated final-stage Q4 scored `0.9612289685442574` on the same 384
public rows. The active calibrated bytes scored `0.9670329670329672`: 660 TP,
21 FP, 24 FN, zero malformed, `3475.478379628233` ms mean latency, and
`6134.182253852487` ms p95.
The active row-level evidence SHA-256 is
`sha256:6821a35a7d46f98af41fc8b65b4239e2bfecc4134ac3183be038b87387bc98af`;
its summary SHA-256 is
`sha256:76d915e19524980f9691b0d94e526acf65163f281ac5f7b4c2bd288c6c8d41f8`.
Stage two trained on these 384 rows, so this final score is training-slice and
quantization evidence, not a hidden-performance estimate.

## Official self-check

The exact active Q4 artifact passed the pinned upstream `mt miner selfcheck`.
The persisted result
(`sha256:e31be3e835ebedea7a295d70741cd848ed84c343d99d750fa8640cf14439279f`)
records:

- proposed size: `436375209` bytes
- proposed peak RSS: `899547545` bytes
- proposed p95 latency: `4099` ms

The same run reported TTFT p50/p95 of `3718/3727` ms, throughput of
`18.3` tokens/s, and an estimated 200-task cost of `3550` CPU-seconds.
Those console diagnostics are not serialized in the three-field self-check
JSON. The private binding file has SHA-256
`sha256:dc9cf8b8b0d8b1910ac03eebb1e87c96f2d6efd3d2364d29da5fce937f041bf9`
and binds the persisted self-check, exact artifact tree, and load specification.

## Publication and submission status

Training finished by Finney block `8955436`. The complete local pending
record is at ignored `runtime/pending-provenance.json`.

**This artifact is unpublished and unsubmitted.** No W&B run was published,
no immutable public artifact URL was created, no coordinator submission was
made, and no chain commitment was written. At build time the required W&B API
key and authenticated immutable-host credentials were absent, and no open
coordinator submission window was observed. Treating the local service as a
successful submission would therefore be inaccurate.
