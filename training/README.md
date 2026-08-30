# Training the extract artifact

This directory records the reproducible model-building half of the miner. It
uses only Microtensor's public `extract` train split and the exact allowlisted
Qwen3-0.6B revision. Runtime outputs live below ignored `runtime/`; wallet
material and credentials never belong in this tree.

## Environment

Create a separate training environment and install `requirements.txt`.
Download the arena's named corpus version, Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`, and llama.cpp at
`c589f0ed10c643678c4707dd160c21ac7633ebc0`.

## Train and evaluate

```bash
python training/train_extract.py \
  --corpus /path/to/public.json \
  --base /path/to/Qwen3-0.6B \
  --out runtime/training/qwen3-0.6b-extract

python training/evaluate_extract.py \
  --corpus /path/to/public.json \
  --model runtime/training/qwen3-0.6b-extract/merged \
  --output runtime/training/qwen3-0.6b-extract/heldout.jsonl
```

The default recipe reserves the same deterministic 384 examples before constructing the
training loader. Controlled public-only experiments may select `--training-method full`,
set `--lora-dropout`, or use `--disease-row-weight` at or above 1.0. Disease weighting
deterministically appends only extra copies of disease-containing public training rows
after the reserve is split, and records both source and added row counts in metadata.

Every run writes a merged model, complete settings and input identities, and an
append-only metric trail; LoRA runs also write an adapter. The reserve is never optimized
by that run. After multiple hyperparameters are compared against the same reserve, its
scores are selection evidence rather than an unbiased hidden-performance estimate.

### Boundary-contrastive experiment

Boundary contrastive training is an explicit, isolated experiment; all default invocations retain
the ordinary causal-LM loader and loss. Enable it with raw gold and the fixed 512-token recipe:

```bash
python training/train_extract.py \
  --corpus /path/to/public.json \
  --base /path/to/Qwen3-0.6B \
  --out runtime/training/experiments/boundary-contrastive-seed92 \
  --boundary-contrastive
```

This mode requires `--seed 92` and first seals the historical 384-row seed-92 outer reserve.
`load_rows` necessarily JSON-decodes the corpus container; after that, this path inspects only each
outer row's ref and partition. It never semantically parses an outer prompt or gold target, tokenizes
one, trains on one, or uses one to choose corruptions. Of the 4,432 remaining rows, the two existing
over-length rows are rejected before a separately named SHA-256 rank split creates a 384-row inner
validation fold and 4,046-row inner train fold. Training and pair generation use only inner-train;
the outer reserve remains untouched for later assessment.

Up to 512 inner-train rows receive one deterministic negative whose entity text is expanded or
contracted by exactly one boundary code point. Expansion and contraction counts are equal, the
positive is the corpus's raw gold string, and each negative retains its type and every other JSON
field. The objective is ordinary positive causal CE plus
`lambda * sum(pair_softplus) / positive_count`, where each pair term is
`softplus(mean_logp_negative - mean_logp_positive + margin)`. Log probabilities are length-normalized
over assistant tokens only; an unpaired positive contributes zero to the auxiliary numerator and
still counts in the denominator, keeping its coefficient stable across batch composition. Lambda
`0.1` (allowed range `[0, 1]`) and margin `0` (allowed range `[0, 20]`) apply only after
`--boundary-contrastive` is supplied; non-finite controls or loss intermediates fail closed. These
conservative bounds keep the auxiliary computation away from floating-point overflow.

`boundary_contrastive_manifest.json` records the exact outer, inner-train, and inner-validation refs,
all ref/target/pair digests, zero-overlap counts, skipped refs, corruption choices, and full objective
settings. The training metadata embeds its digest and summary. Provenance publication must validate
that manifest before presenting such a run as admitted evidence. Embedded ref/record digests use
canonical UTF-8 JSON (`sha256_canonical_json_utf8_v1`), gold digests use the exact UTF-8 string
(`sha256_utf8_text_v1`), and `manifest_digest` covers the exact pretty-printed manifest file bytes
(`sha256_file_bytes_v1`).

Convert with the pinned checkout:

```bash
LLAMA_CPP_DIR=/path/to/llama.cpp \
TRAIN_PYTHON=/path/to/training-python \
  training/build_gguf.sh \
  runtime/training/qwen3-0.6b-extract/merged \
  runtime/artifact/model.gguf Q4_K_M
```

Then sample the model through Microtensor's exact pinned CPU engine:

```bash
PYTHONPATH=/path/to/microtensor-subnet \
  /path/to/validator-python training/evaluate_gguf.py \
  --corpus /path/to/public.json \
  --model runtime/artifact/model.gguf \
  --output runtime/training/qwen3-0.6b-extract/heldout-gguf.jsonl
```

Run `mt miner selfcheck` against the final artifact tree. GGUF embeds the
tokenizer and chat template; accepted GGUF manifests name `tokenizer.json`
in preprocessing but do not package a redundant external tokenizer file.

## Calibration-aware Q4

Greedy extraction can be sensitive to quantization. Build importance-matrix
input only from the public corpus. When comparing against the stage-one
reserve, exclude those exact rows before applying any example cap:

```bash
python training/prepare_imatrix.py \
  --corpus /path/to/public.json \
  --tokenizer /path/to/merged-model \
  --output runtime/training/imatrix/stage1.txt \
  --reserve-examples 384 \
  --max-examples 512
```

The adjacent metadata sidecar binds the exact corpus, tokenizer, template,
selection order, rejected rows, rendered records, and output digest. For a
final all-public calibration, use `--reserve-examples 0`.

```bash
"$TRAIN_PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_MODEL_DIR" \
  --outfile runtime/training/imatrix/model-f16.gguf \
  --outtype f16

"$LLAMA_CPP_DIR/build/bin/llama-imatrix" \
  --offline \
  --model runtime/training/imatrix/model-f16.gguf \
  --file runtime/training/imatrix/stage1.txt \
  --output runtime/training/imatrix/stage1.imatrix.gguf \
  --ctx-size 512 \
  --chunks 128 \
  --no-ppl --process-output --parse-special

"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  --imatrix runtime/training/imatrix/stage1.imatrix.gguf \
  runtime/training/imatrix/model-f16.gguf runtime/artifact/model.gguf \
  Q4_K_M
```

Changing calibration data or quantizer settings creates new artifact bytes and
must be recorded in provenance and re-run through the exact validator engine.

## Deterministic weight soups

`build_weight_soup.py` combines completed merged Qwen3 checkpoints without
mixing their tokenizer or configuration files. It validates the exact pinned
base, source architectures and tensor schemas, and each parent
`training_metadata.json`. Finite nonnegative weights are normalized before the
ordered float32 CPU calculation `base + sum(weight * (source - base))`.

```bash
python training/build_weight_soup.py \
  --base /path/to/Qwen3-0.6B \
  --source runtime/training/experiment-a/merged 1 \
  --source runtime/training/experiment-b/merged 1 \
  --output runtime/training/soups/a-b-equal
```

The destination must not exist. One tensor is written per safetensors shard in
a sibling staging directory before an atomic rename. Config and tokenizer
files are copied byte-for-byte from the base. `soup_metadata.json` binds the
source model/config/tokenizer and parent-metadata hashes, normalized weights,
algorithm settings, schemas, and every output hash.

A soup checkpoint can also be used as a continuation base:

```bash
python training/train_extract.py \
  --corpus runtime/upstream/public-corpus.json \
  --base runtime/training/soups/a-b-equal \
  --out runtime/training/experiments/a-b-continuation
```

Before loading model weights, the trainer revalidates the pinned base and
tokenizer, deterministic algorithm, source parent-metadata bindings, exact
declared output hashes and manifest digest, and model index. It opens every
shard to derive and pin tensor keys, shapes, dtypes, byte count, and finiteness.
Missing or undeclared weight shards and partial soup markers are rejected.

For soup bases, the complete check runs again immediately before and after
Transformers loads the model. Loaded parameters and buffers must be non-meta
and are cloned away from loader/file backing before the post-load check. The
new run's `training_metadata.json` records `training_input.kind` as
`deterministic_weight_soup` together with the soup metadata, output manifest,
index, and tokenizer digests.

This constructs a candidate; it does not establish quality. Evaluate the HF
model, independently convert/calibrate its GGUF, run the exact validator
self-check, and publish the complete training lineage plus soup metadata. The
tied `lm_head.weight` is omitted only after equality with the embedding is
verified. Output bytes depend on the pinned PyTorch and safetensors versions
recorded in metadata.

## Provenance

The publisher intentionally has no anonymous fallback. The live mechanism
requires a resolvable run in `microtensor/training-runs`; without a real
`WANDB_API_KEY`, a miner must wait instead of claiming admission.

After the exact artifact digest is known, export `WANDB_API_KEY` through a
secure method and pass every training directory oldest to newest. The
publisher validates the complete local lineage before importing W&B:

```bash
python training/publish_provenance.py \
  --training-dir runtime/training/experiments/lora-r64-e2-seed92 \
  --training-dir runtime/training/experiments/lora-r64-e2-full-e1 \
  --artifact-digest sha256:... \
  --finished-block FINNEY_BLOCK \
  --calibration-manifest runtime/training/calibration-lineage.json
```

When stage 1 was trained from a deterministic weight soup, also bind that
stage to its checkpoint explicitly:

```bash
  --weight-soup-checkpoint 1 runtime/training/soups/a-b-equal
```

Repeat the stage-numbered option for every supplied stage whose
`training_input.kind` is `deterministic_weight_soup`; missing, duplicate, and
extra mappings are rejected. The publisher revalidates each complete soup
checkpoint before W&B is imported and includes its exact `soup_metadata.json`
and validated digests in the public run config. The retained metadata preserves
the source-file and parent-metadata digest claims, but validating it later does
not reopen the original source checkpoint directories or recompute the soup.

Only then package for an open coordinator round, upload the complete
round-specific tree, and commit it on chain.
