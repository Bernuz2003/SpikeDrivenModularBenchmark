#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from m1_benchmark.training.train import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-dir', required=True)
    args = ap.parse_args()
    cfgs = sorted(Path(args.config_dir).glob('*.yaml'))
    if not cfgs:
        raise SystemExit(f'No YAML configs found in {args.config_dir}')
    for cfg in cfgs:
        print(f'\n=== Running {cfg} ===')
        run(cfg)

if __name__ == '__main__':
    main()
