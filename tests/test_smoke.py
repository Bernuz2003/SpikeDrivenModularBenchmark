from pathlib import Path
import sys
import shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from m1_benchmark.training.train import run


def test_smoke_run(tmp_path):
    cfg_path = ROOT / 'configs/milestone1/smoke/synthetic_smoke.yaml'
    # The YAML writes to runs/milestone1/synthetic_smoke; clean before test.
    out = ROOT / 'runs/milestone1/synthetic_smoke'
    if out.exists():
        shutil.rmtree(out)
    summary = run(cfg_path)
    assert summary['attention'] == 'identity'
    assert summary['accuracy_top1'] >= 0.0
    assert (out / 'metrics_layerwise.csv').exists()
    assert (out / 'metrics_summary.json').exists()
    assert (out / 'profile.json').exists()
    assert (out / 'report.md').exists()
