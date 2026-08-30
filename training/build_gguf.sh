#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 MERGED_MODEL_DIR OUTPUT_GGUF [QUANTIZATION]" >&2
  exit 2
fi

merged_dir=$1
output=$2
quantization=${3:-Q4_K_M}
llama_cpp_dir=${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to the pinned llama.cpp checkout}
python_bin=${TRAIN_PYTHON:-python3}
expected_commit=c589f0ed10c643678c4707dd160c21ac7633ebc0
actual_commit=$(git -C "$llama_cpp_dir" rev-parse HEAD)

if [ "$actual_commit" != "$expected_commit" ]; then
  echo "refusing converter mismatch: expected llama.cpp $expected_commit, got $actual_commit" >&2
  exit 2
fi
if [ ! -d "$merged_dir" ]; then
  echo "merged model directory is missing: $merged_dir" >&2
  exit 2
fi
quantizer=$llama_cpp_dir/build/bin/llama-quantize
if [ ! -x "$quantizer" ]; then
  echo "llama-quantize is not built: $quantizer" >&2
  exit 2
fi

output_dir=$(dirname "$output")
mkdir -p "$output_dir"
temporary=$output_dir/.model-f16.partial.gguf
trap 'rm -f -- "$temporary"' EXIT HUP INT TERM

"$python_bin" "$llama_cpp_dir/convert_hf_to_gguf.py" \
  "$merged_dir" --outfile "$temporary" --outtype f16
"$quantizer" "$temporary" "$output" "$quantization"
chmod 0644 "$output"
