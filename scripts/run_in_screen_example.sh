#!/usr/bin/env bash
set -euo pipefail
# Example for SMILIES servers. Run inside a GNU screen session, ideally with:
# screen -S preattn_cifar_quick -L -Logfile logs/preattn_cifar_quick.log
#
# Singularity example:
# singularity exec --nv \
#   --bind /home/users/$USER:/workspace \
#   containers/pre_attention_benchmark.sif \
#   bash -lc "cd /workspace/pre_attention_benchmark_framework && python -m pre_attention_benchmark.training.train --config configs/pre_attention_benchmark/real/cifar10_dvs_quick_check.yaml"
export PYTHONPATH=${PYTHONPATH:-}:$PWD/src
python -m pre_attention_benchmark.training.train --config "${1:-configs/pre_attention_benchmark/real/cifar10_dvs_quick_check.yaml}"
