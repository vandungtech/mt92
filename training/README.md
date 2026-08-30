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

The script reserves the same deterministic 384 examples for an honest
pre-quantization benchmark. It writes a LoRA adapter, merged model, metadata,
and an append-only metric trail.

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

## Provenance

The publisher intentionally has no anonymous fallback. The live mechanism
requires a resolvable run in `microtensor/training-runs`; without a real
`WANDB_API_KEY`, a miner must wait instead of claiming admission.

After the exact artifact digest is known:

```bash
WANDB_API_KEY=... python training/publish_provenance.py \
  --training-dir runtime/training/qwen3-0.6b-extract \
  --artifact-digest sha256:... \
  --finished-block FINNEY_BLOCK
```

Only then package for an open coordinator round, upload the complete
round-specific tree, and commit it on chain.
