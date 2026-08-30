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
  --finished-block FINNEY_BLOCK
```

Only then package for an open coordinator round, upload the complete
round-specific tree, and commit it on chain.
