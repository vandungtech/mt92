#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
env_file=${MMC_ENV_FILE:-$repo_dir/runtime/miner.env}
controller_bin=${MMC_CONTROLLER_BIN:-$repo_dir/.venv/bin/microtensor-miner-controller}

if [ ! -r "$env_file" ] || [ ! -x "$controller_bin" ]; then
  echo '{"ok":false,"phase":"missing","message":"healthcheck prerequisites unavailable"}'
  exit 2
fi

exec "$controller_bin" --env-file "$env_file" health --compact
