#!/usr/bin/env bash
set -euo pipefail
# Example for SMILIES servers. Run inside a GNU screen session, ideally with:
# screen -S m1_cifar_smoke -L -Logfile logs/m1_cifar_smoke.log
#
# Singularity example:
# singularity exec --nv \
#   --bind /home/users/$USER:/workspace \
#   containers/milestone1.sif \
#   bash -lc "cd /workspace/milestone1_framework && python -m m1_benchmark.training.train --config configs/milestone1/real/cifar10_dvs_smoke.yaml"
export PYTHONPATH=${PYTHONPATH:-}:$PWD/src
python -m m1_benchmark.training.train --config "${1:-configs/milestone1/smoke/synthetic_smoke.yaml}"
