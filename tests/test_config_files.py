from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pre_attention_benchmark.config import (
    ConfigError,
    expand_config_placeholders,
    has_unresolved_placeholders,
    load_config,
    validate_pre_attention_config,
)


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


def test_path_placeholders_expand_from_variables():
    cfg = {
        'dataset': {'root': '${PREATTN_DATA_ROOT}'},
        'logging': {'output_dir': '${PREATTN_RUNS_ROOT}/run_a'},
    }
    out = expand_config_placeholders(
        cfg,
        {
            'PREATTN_DATA_ROOT': '/tmp/data',
            'PREATTN_RUNS_ROOT': '/tmp/runs',
        },
    )
    assert out['dataset']['root'] == '/tmp/data'
    assert out['logging']['output_dir'] == '/tmp/runs/run_a'
    assert not has_unresolved_placeholders(out)
