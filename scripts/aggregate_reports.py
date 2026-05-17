#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pre_attention_benchmark.reporting.report import aggregate_reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default='runs/pre_attention_benchmark')
    ap.add_argument('--out-dir', default='reports/pre_attention_benchmark')
    args = ap.parse_args()
    # Aggrega solo gli artifact gia prodotti: non ricalcola metriche e non tocca checkpoint.
    aggregate_reports(args.runs_dir, args.out_dir)

if __name__ == '__main__':
    main()
