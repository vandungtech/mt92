# Artifact record

The active deployable weights are intentionally ignored by Git and live at
`runtime/artifact/model.gguf`. This record binds those exact local bytes to
reviewed training, calibration, evaluation, and official self-check evidence
without committing a 396 MB binary.

## Identity

- Track/class: `extract/mt-3g`
- Format/quantization: GGUF v3 / `Q4_K_M`
- Entrypoint: `model.gguf`, `396704736` bytes
- Entrypoint SHA-256:
  `sha256:903b76dfae36ecb650808e282800333e96da8a606f96355cc735455bf8651ddd`
- Microtensor artifact-tree digest:
  `sha256:3c9a5b064e2641570989cb605f605aaa8db7b5cfbaa73f36bbb0d2ff25fb5d66`
- Base:
  `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
- Public corpus version:
  `sha256:492ea6e7b791f03be0989b07eee0dc9ba722d35d2f274743c6dc33420c383ff8`
- Public corpus file SHA-256:
  `sha256:fb5f1332493b1abe759da91cec1bd2cdd932c7076c5fae8163b5f33cfbea05a2`
- Converter, imatrix engine, and quantizer: llama.cpp
  `c589f0ed10c643678c4707dd160c21ac7633ebc0`

The active file is a hard link to
`runtime/candidates/soup-r64-equal3-full-e1-q4/model.gguf`. GGUF embeds the
tokenizer and chat template, so the publishable tree contains only
`model.gguf`. The load specification's canonical SHA-256 is
`sha256:12f805e5e63a0b48954cef98c6579b61b88ef03a27d658df632df687c0ec6e1e`.

## Training lineage

The continuation starts from the deterministic equal-weight soup at
`runtime/training/soups/r64-seed92-equal3`. Its metadata SHA-256 is
`sha256:4237e19e8997edb4a9895461f758974d9a05270d38328983acb3d0eb72cff1c4`;
its output-manifest digest is
`sha256:4dd85b2ff353591eb26abdd26df073813ec550d5ceecba48fbf02244a25e4db7`,
and its tensor-index digest is
`sha256:a9291111e64caafb4bb8b4703e0b90ab00999dc9474276da6b4d94efac5edda5`.

The soup combines three rank-64 parents in order at equal normalized weight:

- `lora-r64-e2-seed92`, metadata
  `sha256:178040ccd12dc855d0584d2f0505c4086fe7da6e51c08000b8295736f2aab9f7`
- `lora-r64-e2-drop0-seed92`, metadata
  `sha256:0892e2e8afe42fc366e30ce1a979d54a8e39187ca469e686e7c2f77fcd7476de`
- `lora-r64-e2-disease125-seed92`, metadata
  `sha256:7b0ec1993e498d5077e1ff0aaa5513b7a1278661717e4943a36bb99a355651d0`

The final continuation is
`runtime/training/experiments/soup-r64-equal3-full-e1`; its metadata SHA-256 is
`sha256:c6d65337a035932aceae77941ee8371e48a8dd4625ac3a637b6bb2a5a4ea5eb2`.
It trained one epoch with seed 92, learning rate `5e-5`, rank 64, alpha 128,
LoRA dropout 0.05, batch size 8, gradient accumulation 4, maximum length 512,
warmup ratio 0.05, and weight decay 0.01. It used all 4,814 encodable public
examples; two overlength examples were skipped. Training finished at Unix time
`1788079473` and had completed by Finney block `8956890`.

## Conversion and public-only calibration

The merged model converted to F16 at
`runtime/training/experiments/soup-r64-equal3-full-e1/model-f16.gguf`,
`1198182080` bytes, SHA-256
`sha256:2562a6afb8e75683e06c3054724fd0a0e4a528ed83a8af6951a0a7f4be564275`.

The deterministic public calibration corpus used seed 92, reserve size 0, and
512 examples. Thinking and generation prompts were disabled, and gold entities
were rendered as compact sorted-key JSON after exact text/type deduplication.

- rendered text: `runtime/training/imatrix/final-seed92-r0-n512.txt`,
  `310901` bytes,
  `sha256:0588fa83ddf7768846d5e491c7878b2c7b66fe4ba0c9afd12ca4fd118b82bf9d`
- metadata sidecar:
  `runtime/training/imatrix/final-seed92-r0-n512.txt.metadata.json`,
  `98185` bytes,
  `sha256:7be2a3835d05781667e80f4d4aff55ccad55e548c6c75a17655a5fb815fb105b`
- final imatrix:
  `runtime/training/experiments/soup-r64-equal3-full-e1/model-imatrix.gguf`,
  `1177088` bytes,
  `sha256:9dc4a84b75775f4c357a82d7534ad388eacbbc187a761022beed00637a526d93`

The pass used the pinned CUDA build in offline mode, context size 512, all
available chunks, `--no-ppl`, `--parse-special`, and no
`--process-output`. Quantization used that exact matrix and `Q4_K_M`.

The no-clobber manifest
`calibration-lineage.soup-r64-equal3-full-e1.json` has SHA-256
`sha256:ebdafaa7d84c73dfc04c70f7a062c4f253c026365671905281f64ba4b1cb1fd7`.
It binds the exact source inventory, conversion, corpus, sidecar, imatrix,
quantized output, artifact tree, clean pinned llama.cpp checkout, and soup
checkpoint. This is a byte-bound caller attestation of the canonical recipe,
not a retroactive historical execution receipt.

## Evaluation evidence

The unquantized continuation scored `0.9743589743589743` entity micro-F1 on
the fixed 384-row public slice: 665 TP, 16 FP, 19 FN, and zero malformed. Its
row-level and summary SHA-256 values are
`sha256:e153fda8efecc47e8b324d399a40861c3ee8329c08565577ab17e43c8eafa8a2`
and
`sha256:363ee2803471d6ccea46c193610f90e07052ac610e5f778e8c2a99576348a03e`.

The exact active Q4 scored `0.9765051395007343`: 665 TP, 13 FP, 19 FN, zero
malformed, `3472.1451612131204` ms mean latency, and
`6138.441361486912` ms p95. Its row-level and summary SHA-256 values are
`sha256:41fea10b49db1c0bda30ac83adeae2aa333e60a6167fa18d321cf64f6feda2c9`
and
`sha256:0c9aa6498337c6350a99bd074d745b6c9003c3415c55d37ffba0dacb8486e8a5`.

The continuation trained on the encodable members of this public slice, so
these are training-slice and quantization-selection measurements, not a hidden
or official leaderboard estimate.

## Official self-check and rollback

The exact active artifact passed pinned upstream release `v0.1.14` at commit
`d0e002f887d038bf3ea4af65b499137a755620d7`. The persisted self-check SHA-256
is `sha256:f6ed5aa907d0a6d899c9665aace5664e700c934284f7e5b4ea7f610eb179bc79`
and records:

- proposed size: `436375209` bytes
- proposed peak RSS: `899759308` bytes
- proposed p95 latency: `4086` ms

The run reported TTFT p50/p95 of `3712/3715` ms, throughput of `18.8`
tokens/s, and an estimated 200-task cost of `3465` CPU-seconds. Those console
diagnostics are not serialized in the self-check JSON. The private binding
SHA-256 is
`sha256:496eb8a8c6c571e7aa66044fcee65d23138742545f020bf4d8d941042c9f990b`.

The superseded active artifact and its self-check remain privately at
`runtime/rollbacks/active-fe0f3419-before-soup-903b76df/`.

## Publication and submission status

The local pending record is at ignored `runtime/pending-provenance.json`.

**This artifact is unpublished and unsubmitted.** No public W&B run, immutable
public URL, coordinator submission, or on-chain commitment exists for these
bytes. The required credentials are absent, and the coordinator is serving a
stale settled round. Treating the dry-run controller, local quality result, or
local manifest as an official submission or rank would be inaccurate.
