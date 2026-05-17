#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pre_attention_benchmark.training.train import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-dir', required=True)
    ap.add_argument('--continue-on-error', action='store_true')
    args = ap.parse_args()
    cfgs = sorted(Path(args.config_dir).glob('*.yaml'))
    if not cfgs:
        raise SystemExit(f'No YAML configs found in {args.config_dir}')
    for cfg in cfgs:
        # Sequenziale di proposito: sui server e piu facile associare log e artifact
        # a una singola config, senza processi concorrenti che si pestano i piedi.
        print(f'\n=== Running {cfg} ===')
        try:
            run(cfg)
        except Exception:
            if not args.continue_on_error:
                raise
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
