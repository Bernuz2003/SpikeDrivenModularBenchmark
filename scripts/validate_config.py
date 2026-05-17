#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pre_attention_benchmark.config import load_config, validate_pre_attention_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args = ap.parse_args()
    cfg = load_config(args.config)
    # Validazione statica: non scarica dataset e non istanzia il modello.
    validate_pre_attention_config(cfg)
    print('OK: config satisfies static pre-attention benchmark gates')

if __name__ == '__main__':
    main()
