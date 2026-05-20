from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pre_attention_benchmark.models.feature_extractors import FEATURE_EXTRACTOR_NAMES, build_feature_extractor
from pre_attention_benchmark.training.utils import tensor_is_binary


@pytest.mark.parametrize('name', sorted(FEATURE_EXTRACTOR_NAMES))
def test_main_feature_extractor_families_emit_binary_tokens(name: str):
    cfg = {
        'name': name,
        'channels': [4, 8, 16],
        'residual': 'ms',
        'output_format': 'tokens',
    }
    fe = build_feature_extractor(cfg, in_channels=2, surrogate_alpha=4.0)
    fe.eval()
    x = torch.randint(0, 2, (2, 3, 2, 32, 32)).float()

    with torch.no_grad():
        y = fe(x)

    assert y.shape == (2, 3, 16, 16)
    assert tensor_is_binary(y)
    meta = fe.describe()
    assert meta['uses_ms_residual'] is True
    assert meta['num_stages'] == 3
    assert meta['token_count_N'] == 16
    assert meta['embedding_dim_D'] == 16


def test_strided_conv_stem_uses_learned_downsampling_without_maxpool():
    fe = build_feature_extractor(
        {'name': 'strided_conv_stem_ms', 'channels': [4, 8, 16], 'residual': 'ms'},
        in_channels=2,
    )
    assert not any(module.__class__.__name__ == 'TDMaxPool' for module in fe.modules())


def test_polarity_separable_stem_requires_two_polarity_channels():
    with pytest.raises(ValueError, match='exactly 2 channels'):
        build_feature_extractor(
            {'name': 'polarity_separable_stem_ms', 'channels': [4, 8, 16], 'residual': 'ms'},
            in_channels=1,
        )
