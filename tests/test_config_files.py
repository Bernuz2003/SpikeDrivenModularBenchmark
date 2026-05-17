from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pre_attention_benchmark.config import ConfigError, load_config, validate_pre_attention_config


def test_all_public_configs_validate_statically():
    # Non istanziamo i dataset reali qui: questo test controlla i gate statici
    # sulle YAML ufficiali, restando eseguibile anche offline.
    cfgs = [
        p
        for p in (ROOT / 'configs/pre_attention_benchmark').rglob('*.yaml')
        if '/invalid/' not in str(p)
    ]
    assert cfgs
    for path in cfgs:
        validate_pre_attention_config(load_config(path))


@pytest.mark.parametrize('path', sorted((ROOT / 'configs/pre_attention_benchmark/invalid').glob('*.yaml')))
def test_invalid_configs_are_rejected(path: Path):
    with pytest.raises(ConfigError):
        validate_pre_attention_config(load_config(path))
