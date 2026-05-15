#!/usr/bin/env bash
set -euo pipefail
# Example for SMILIES servers. Run inside a GNU screen session.
# screen -S m1_smoke
# singularity exec --bind "$PWD:/workspace" milestone1.sif \
#   python -m m1_benchmark.training.train --config /workspace/configs/milestone1/smoke/synthetic_smoke.yaml
export PYTHONPATH=${PYTHONPATH:-}:$PWD/src
python -m m1_benchmark.training.train --config configs/milestone1/smoke/synthetic_smoke.yaml
