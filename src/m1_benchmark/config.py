from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping
import copy
import json
import subprocess
import yaml


class ConfigError(ValueError):
    """Raised when a configuration violates Milestone-1 constraints."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    cfg['_config_path'] = str(path)
    return apply_defaults(cfg)


def apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('experiment', {})
    cfg['experiment'].setdefault('name', 'unnamed_run')
    cfg['experiment'].setdefault('milestone', 'M1')
    cfg['experiment'].setdefault('seed', 42)

    cfg.setdefault('dataset', {})
    cfg['dataset'].setdefault('name', 'synthetic_dvs')
    cfg['dataset'].setdefault('height', 32)
    cfg['dataset'].setdefault('width', 32)
    cfg['dataset'].setdefault('num_classes', 10)
    cfg['dataset'].setdefault('num_train', 256)
    cfg['dataset'].setdefault('num_val', 64)

    cfg.setdefault('encoder', {})
    cfg['encoder'].setdefault('name', 'fixed_time_binary')
    cfg['encoder'].setdefault('T', 8)
    cfg['encoder'].setdefault('polarity_channels', True)
    cfg['encoder'].setdefault('binarize', True)

    cfg.setdefault('model', {})
    cfg['model'].setdefault('feature_extractor', {})
    cfg['model']['feature_extractor'].setdefault('name', 'conv_bn_lif_maxpool')
    cfg['model']['feature_extractor'].setdefault('channels', [16, 32])
    cfg['model']['feature_extractor'].setdefault('residual', 'none')
    cfg['model']['feature_extractor'].setdefault('output_format', 'tokens')

    cfg['model'].setdefault('attention', {})
    cfg['model']['attention'].setdefault('name', 'identity')

    cfg['model'].setdefault('head', {})
    cfg['model']['head'].setdefault('name', 'last_timestep_spike_readout')
    cfg['model']['head'].setdefault('terminal_readout', True)

    cfg.setdefault('training', {})
    cfg['training'].setdefault('epochs', 2)
    cfg['training'].setdefault('batch_size', 16)
    cfg['training'].setdefault('optimizer', 'adamw')
    cfg['training'].setdefault('lr', 1e-3)
    cfg['training'].setdefault('weight_decay', 1e-4)
    cfg['training'].setdefault('num_workers', 0)
    cfg['training'].setdefault('torch_num_threads', 1)
    cfg['training'].setdefault('device', 'auto')
    cfg['training'].setdefault('surrogate_alpha', 4.0)
    cfg['training'].setdefault('profile_batches', 1)
    cfg['training'].setdefault('resume_from', None)
    cfg['training'].setdefault('log_interval_batches', 25)

    cfg.setdefault('evaluation', {})
    cfg['evaluation'].setdefault('robustness', {})
    cfg['evaluation']['robustness'].setdefault('enabled', False)
    cfg['evaluation']['robustness'].setdefault('event_drop_rates', [0.1, 0.2, 0.3])
    cfg['evaluation']['robustness'].setdefault('temporal_jitter', [0.05])
    cfg['evaluation']['robustness'].setdefault('polarity_drop', [0.2])
    cfg['evaluation']['robustness'].setdefault('timestep_shuffle', True)
    cfg['evaluation']['robustness'].setdefault('early_accuracy', True)

    cfg.setdefault('logging', {})
    cfg['logging'].setdefault('save_layer_metrics', True)
    cfg['logging'].setdefault('save_frequency_metrics', True)
    cfg['logging'].setdefault('save_temporal_metrics', True)
    cfg['logging'].setdefault('output_dir', f"runs/milestone1/{cfg['experiment']['name']}")
    return cfg


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return None


def get_git_metadata() -> dict[str, Any]:
    """Return lightweight Git provenance for reproducibility artifacts."""
    meta: dict[str, Any] = {
        'commit': None,
        'dirty': None,
        'remote': None,
        'branch': None,
    }
    try:
        meta['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        meta['branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--short'], stderr=subprocess.DEVNULL, text=True)
        meta['dirty'] = bool(status.strip())
        remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], stderr=subprocess.DEVNULL, text=True).strip()
        meta['remote'] = remote or None
    except Exception:
        pass
    return meta


def save_config(cfg: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with out_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)


def save_config_json(cfg: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def validate_milestone1_config(cfg: dict[str, Any]) -> None:
    """Fail-fast quality gates that are known statically from YAML."""
    milestone = str(cfg.get('experiment', {}).get('milestone', 'M1')).upper()
    if milestone != 'M1':
        raise ConfigError(f"Milestone 1 runner received milestone={milestone!r}; expected 'M1'.")

    attention = cfg.get('model', {}).get('attention', {}).get('name', 'identity')
    if attention != 'identity':
        raise ConfigError(
            f"Milestone 1 requires attention.name='identity'. Got {attention!r}."
        )

    fe = cfg.get('model', {}).get('feature_extractor', {})
    residual = fe.get('residual', 'none')
    if residual not in ('none', None, 'ms'):
        raise ConfigError(
            f"Milestone 1 forbids non-MS residuals. Got residual={residual!r}."
        )

    if str(residual).lower() == 'sew':
        raise ConfigError("SEW residual is forbidden because it can propagate multi-bit spike sums.")

    enc = cfg.get('encoder', {})
    if enc.get('binarize', True) is not True:
        raise ConfigError("Milestone 1 requires encoder.binarize=true for hidden spike-driven input.")

    enc_name = enc.get('name', '')
    if 'count' in enc_name and enc.get('binarize', True) is not True:
        raise ConfigError("Count-valued encodings must be binarized before entering the hidden pipeline.")

    output_format = fe.get('output_format', 'tokens')
    if output_format not in ('tokens', 'feature_map'):
        raise ConfigError(f"feature_extractor.output_format must be 'tokens' or 'feature_map', got {output_format!r}.")

    head = cfg.get('model', {}).get('head', {})
    if head.get('terminal_readout', True) is not True:
        raise ConfigError("Milestone 1 requires head.terminal_readout=true; logits are allowed only at the terminal boundary.")


def deep_update(base: MutableMapping[str, Any], patch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in patch.items():
        if isinstance(v, MutableMapping) and isinstance(base.get(k), MutableMapping):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
