from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from m1_benchmark.config import load_config, validate_milestone1_config, ConfigError


def test_attention_not_identity_rejected():
    cfg = load_config(ROOT / 'configs/milestone1/invalid/attention_not_identity.yaml')
    with pytest.raises(ConfigError):
        validate_milestone1_config(cfg)


def test_sew_residual_rejected():
    cfg = load_config(ROOT / 'configs/milestone1/invalid/sew_residual.yaml')
    with pytest.raises(ConfigError):
        validate_milestone1_config(cfg)
