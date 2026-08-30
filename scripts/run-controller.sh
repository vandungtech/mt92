#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
env_file=${MMC_ENV_FILE:-$repo_dir/runtime/miner.env}
controller_bin=${MMC_CONTROLLER_BIN:-$repo_dir/.venv/bin/microtensor-miner-controller}

if [ ! -r "$env_file" ]; then
  echo "refusing to start: environment file is not readable" >&2
  exit 2
fi
if [ ! -x "$controller_bin" ]; then
  echo "refusing to start: controller executable is unavailable" >&2
  exit 2
fi

exec "$controller_bin" --env-file "$env_file" run
