#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash scripts/run_pre_attention_suite.sh configs/local/pre_attention_benchmark
#
# La directory passata deve contenere le stesse sottocartelle delle config ufficiali:
# real/, encoding_sweep/, feature_extractor_sweep/, head_sweep/, robustness/.

CONFIG_ROOT="${1:-configs/pre_attention_benchmark}"

run_group() {
  local group="$1"
  local dir="${CONFIG_ROOT}/${group}"
  if [[ -d "${dir}" ]]; then
    echo "=== Suite group: ${group} ==="
    python scripts/run_sweep.py --config-dir "${dir}" --continue-on-error
  else
    echo "skip: ${dir} non esiste"
  fi
}

# Prima i check reali brevi, poi gli sweep veri.
run_group real
run_group encoding_sweep
run_group feature_extractor_sweep
run_group head_sweep
run_group robustness
