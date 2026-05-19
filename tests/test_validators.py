from pathlib import Path
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pre_attention_benchmark.config import load_config, validate_pre_attention_config, ConfigError
from pre_attention_benchmark.datasets import CachedEncodedDataset, EncodedEventDataset, build_encoder
from pre_attention_benchmark.metrics.collector import MetricsCollector
from pre_attention_benchmark.models.heads import build_head, describe_head
from pre_attention_benchmark.models.validators import validate_hidden_outputs


def test_attention_not_identity_rejected():
    cfg = load_config(ROOT / 'configs/pre_attention_benchmark/invalid/attention_not_identity.yaml')
    with pytest.raises(ConfigError):
        validate_pre_attention_config(cfg)


def test_sew_residual_rejected():
    cfg = load_config(ROOT / 'configs/pre_attention_benchmark/invalid/sew_residual.yaml')
    with pytest.raises(ConfigError):
        validate_pre_attention_config(cfg)


def test_encoder_outputs_binary_and_metadata():
    # Il test usa eventi minimi costruiti in memoria: controlla l'encoder senza
    # dipendere dal download dei dataset reali.
    enc = build_encoder({'name': 'fixed_event_count_binary', 'T': 4, 'binarize': True}, {'height': 8, 'width': 8})
    events = {
        't': torch.arange(10).numpy(),
        'x': torch.arange(10).numpy() % 8,
        'y': torch.arange(10).numpy() % 8,
        'p': torch.arange(10).numpy() % 2,
        'label': 0,
        'height': 8,
        'width': 8,
    }
    spikes = enc(events)
    assert spikes.shape == (4, 2, 8, 8)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})
    assert enc.describe()['controls_event_count'] is True


def test_pixel_threshold_filters_local_event_counts():
    enc = build_encoder(
        {'name': 'fixed_time_binary', 'T': 1, 'binarize': True, 'pixel_threshold': 2},
        {'height': 4, 'width': 4},
    )
    events = {
        't': torch.arange(4).numpy(),
        'x': torch.tensor([1, 1, 1, 2]).numpy(),
        'y': torch.tensor([1, 1, 1, 2]).numpy(),
        'p': torch.tensor([1, 1, 1, 1]).numpy(),
        'label': 0,
        'height': 4,
        'width': 4,
    }
    spikes = enc(events)
    assert spikes[0, 1, 1, 1] == 1
    assert spikes[0, 1, 2, 2] == 0
    assert enc.describe()['pixel_threshold'] == 2


def test_cached_encoded_dataset_materializes_uint8_once():
    class TinyRaw(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                't': torch.arange(3).numpy(),
                'x': torch.tensor([idx, idx, idx]).numpy(),
                'y': torch.tensor([0, 0, 0]).numpy(),
                'p': torch.tensor([1, 1, 1]).numpy(),
                'label': idx,
                'height': 4,
                'width': 4,
            }

    enc = build_encoder({'name': 'fixed_time_binary', 'T': 1, 'binarize': True}, {'height': 4, 'width': 4})
    encoded = EncodedEventDataset(TinyRaw(), enc, split_name='train')
    cached = CachedEncodedDataset.materialize(encoded, cache_dtype='uint8')
    spikes, label = cached[1]
    assert cached.spikes.dtype == torch.uint8
    assert spikes.dtype == torch.float32
    assert label.item() == 1
    assert cached.cache_encoded is True


def test_new_heads_forward_and_metadata():
    token_x = torch.zeros(2, 3, 4, 5)
    token_x[:, :, 0] = 1
    avg_head = build_head({'name': 'spatio_temporal_avg_readout'}, 5, 7)
    avg_logits = avg_head(token_x)
    assert avg_logits.shape == (2, 7)
    assert describe_head(avg_head, 5, 7)['readout_signal'] == 'firing_rate'

    acc_head = build_head({'name': 'class_neuron_accumulator', 'threshold': 0.1}, 5, 7)
    acc_logits = acc_head(token_x)
    assert acc_logits.shape == (2, 7)
    assert describe_head(acc_head, 5, 7)['uses_surrogate_class_threshold'] is True

    map_x = torch.ones(2, 3, 5, 4, 4)
    sv_head = build_head(
        {'name': 'spikevision_spatial_pooling', 'threshold': 3.0},
        5,
        7,
        feature_shape=(1, 3, 5, 4, 4),
    )
    sv_logits = sv_head(map_x)
    assert sv_logits.shape == (2, 7)
    assert describe_head(sv_head, 5, 7)['requires_feature_map'] is True


def test_spikevision_head_requires_feature_map_config():
    cfg = load_config(ROOT / 'configs/pre_attention_benchmark/head_sweep/spikevision_spatial_pooling.yaml')
    validate_pre_attention_config(cfg)
    cfg['model']['feature_extractor']['output_format'] = 'tokens'
    with pytest.raises(ConfigError, match='spikevision_spatial_pooling requires'):
        validate_pre_attention_config(cfg)


def test_hidden_boundary_validator_catches_non_binary_output():
    class BadBoundary(torch.nn.Module):
        emits_hidden_spikes = True

        def forward(self, x):
            return x + 0.5

    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bad = BadBoundary()

        def forward(self, x, validate_hidden_binary=False):
            return self.bad(x).flatten(1)[:, :2]

    sample = torch.zeros(1, 2, 2, 4, 4)
    with pytest.raises(RuntimeError, match='Hidden spike communication violated'):
        validate_hidden_outputs(BadModel(), sample, torch.device('cpu'))


def test_temporal_density_supports_tokens():
    # Caso token [B,T,N,D], diverso dalle mappe [B,T,C,H,W] viste dai conv.
    collector = MetricsCollector(torch.nn.Identity(), run_id='test')
    x = torch.zeros(2, 3, 4, 5)
    x[:, 1] = 1
    mean, std, burst, per_timestep = collector._temporal_density(x, True)
    assert per_timestep == [0.0, 1.0, 0.0]
    assert mean > 0
    assert std > 0
    assert burst > 0
