#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from m1_benchmark.config import load_config, validate_milestone1_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args = ap.parse_args()
    cfg = load_config(args.config)
    validate_milestone1_config(cfg)
    print('OK: config satisfies static Milestone-1 gates')

if __name__ == '__main__':
    main()
