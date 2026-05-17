from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from m1_benchmark.training.train import run


def test_smoke_run(tmp_path):
    cfg_path = ROOT / 'configs/milestone1/smoke/synthetic_smoke.yaml'
    out = tmp_path / 'synthetic_smoke'
    cfg_text = cfg_path.read_text(encoding='utf-8').replace('output_dir: runs/milestone1/synthetic_smoke', f'output_dir: {out}')
    tmp_cfg = tmp_path / 'synthetic_smoke.yaml'
    tmp_cfg.write_text(cfg_text, encoding='utf-8')
    summary = run(tmp_cfg)
    assert summary['attention'] == 'identity'
    assert summary['accuracy_top1'] >= 0.0
    assert (out / 'metrics_layerwise.csv').exists()
    assert (out / 'metrics_summary.json').exists()
    assert (out / 'profile.json').exists()
    assert (out / 'report.md').exists()
    assert (out / 'config.json').exists()
    assert (out / 'dataset_split.json').exists()
    assert (out / 'input_spike_profile.json').exists()
    header = (out / 'metrics_layerwise.csv').read_text(encoding='utf-8').splitlines()[0]
    for col in ['spike_density_timestep', 'sops_proxy', 'state_mem_bits', 'hf_ratio']:
        assert col in header
