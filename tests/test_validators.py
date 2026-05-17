from pathlib import Path
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pre_attention_benchmark.config import load_config, validate_pre_attention_config, ConfigError
from pre_attention_benchmark.datasets import build_encoder
from pre_attention_benchmark.metrics.collector import MetricsCollector
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
