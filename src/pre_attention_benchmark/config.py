from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping
import copy
import json
import os
import re
import subprocess
import yaml

_PLACEHOLDER_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
_LOCAL_PATHS_FILE = Path('configs/local/paths.yaml')


class ConfigError(ValueError):
    """Raised when a configuration violates pre-attention benchmark constraints."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    cfg['_config_path'] = str(path)
    return expand_config_placeholders(apply_defaults(cfg))


def load_path_variables(path_file: str | Path | None = None) -> dict[str, str]:
    variables: dict[str, str] = {}
    local_file = Path(path_file or os.environ.get('PREATTN_PATHS_FILE', _LOCAL_PATHS_FILE))
    if local_file.exists():
        with local_file.open('r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        for key, value in raw.items():
            variables[str(key)] = str(value)
    # Le variabili d'ambiente vincono sul file locale: comodo negli script batch.
    variables.update({k: v for k, v in os.environ.items() if k.startswith('PREATTN_')})
    return variables


def expand_config_placeholders(value: Any, variables: dict[str, str] | None = None) -> Any:
    variables = load_path_variables() if variables is None else variables
    if isinstance(value, dict):
        return {k: expand_config_placeholders(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config_placeholders(v, variables) for v in value]
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            return variables.get(name, match.group(0))

        return os.path.expanduser(_PLACEHOLDER_RE.sub(repl, value))
    return value


def has_unresolved_placeholders(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_unresolved_placeholders(v) for v in value.values())
    if isinstance(value, list):
        return any(has_unresolved_placeholders(v) for v in value)
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.search(value))


def apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('experiment', {})
    cfg['experiment'].setdefault('name', 'unnamed_run')
    cfg['experiment'].setdefault('benchmark', 'pre_attention')
    cfg['experiment'].setdefault('seed', 42)

    cfg.setdefault('dataset', {})
    cfg['dataset'].setdefault('name', 'cifar10_dvs')
    cfg['dataset'].setdefault('root', 'data/tonic')
    cfg['dataset'].setdefault('height', 128)
    cfg['dataset'].setdefault('width', 128)
    cfg['dataset'].setdefault('num_classes', 10)
    cfg['dataset'].setdefault('num_train', 256)
    cfg['dataset'].setdefault('num_val', 64)
    cfg['dataset'].setdefault('cache_encoded', True)
    cfg['dataset'].setdefault('cache_dtype', 'uint8')

    cfg.setdefault('encoder', {})
    cfg['encoder'].setdefault('name', 'fixed_time_binary')
    cfg['encoder'].setdefault('T', 8)
    cfg['encoder'].setdefault('polarity_channels', True)
    cfg['encoder'].setdefault('binarize', True)
    cfg['encoder'].setdefault('pixel_threshold', 0)

    cfg.setdefault('model', {})
    cfg['model'].setdefault('feature_extractor', {})
    cfg['model']['feature_extractor'].setdefault('name', 'conv_bn_lif_maxpool')
    cfg['model']['feature_extractor'].setdefault('channels', [16, 32])
    cfg['model']['feature_extractor'].setdefault('residual', 'none')
    cfg['model']['feature_extractor'].setdefault('output_format', 'tokens')

    cfg['model'].setdefault('attention', {})
    cfg['model']['attention'].setdefault('name', 'identity')

    cfg['model'].setdefault('head', {})
    cfg['model']['head'].setdefault('name', 'spatio_temporal_avg_readout')
    cfg['model']['head'].setdefault('terminal_readout', True)

    cfg.setdefault('training', {})
    cfg['training'].setdefault('epochs', 2)
    cfg['training'].setdefault('batch_size', 16)
    cfg['training'].setdefault('optimizer', 'adamw')
    cfg['training'].setdefault('lr', 1e-3)
    cfg['training'].setdefault('weight_decay', 1e-4)
    cfg['training'].setdefault('scheduler', 'cosine')
    cfg['training'].setdefault('min_lr', 1e-5)
    cfg['training'].setdefault('num_workers', 0)
    cfg['training'].setdefault('pin_memory', True)
    cfg['training'].setdefault('torch_num_threads', 1)
    cfg['training'].setdefault('device', 'auto')
    cfg['training'].setdefault('surrogate_alpha', 4.0)
    cfg['training'].setdefault('profile_batches', 1)
    cfg['training'].setdefault('resume_from', None)
    cfg['training'].setdefault('log_interval_batches', 50)

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
    cfg['logging'].setdefault('output_dir', f"runs/pre_attention_benchmark/{cfg['experiment']['name']}")
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


def validate_pre_attention_config(cfg: dict[str, Any]) -> None:
    """Fail-fast quality gates that are known statically from YAML."""
    # Il nome del benchmark e un piccolo guard-rail: evita di lanciare config
    # pensate per fasi future dove l'attention non sara piu Identity.
    benchmark = str(cfg.get('experiment', {}).get('benchmark', 'pre_attention')).lower()
    if benchmark != 'pre_attention':
        raise ConfigError(f"pre-attention benchmark runner received benchmark={benchmark!r}; expected 'pre_attention'.")

    attention = cfg.get('model', {}).get('attention', {}).get('name', 'identity')
    if attention != 'identity':
        raise ConfigError(
            f"pre-attention benchmark requires attention.name='identity'. Got {attention!r}."
        )

    fe = cfg.get('model', {}).get('feature_extractor', {})
    residual = fe.get('residual', 'none')
    if residual not in ('none', None, 'ms'):
        raise ConfigError(
            f"pre-attention benchmark forbids non-MS residuals. Got residual={residual!r}."
        )

    if str(residual).lower() == 'sew':
        raise ConfigError("SEW residual is forbidden because it can propagate multi-bit spike sums.")

    enc = cfg.get('encoder', {})
    allowed_encoders = {'fixed_time_binary', 'fixed_event_count_binary'}
    enc_name = enc.get('name', '')
    if enc_name not in allowed_encoders:
        raise ConfigError(f"encoder.name must be one of {sorted(allowed_encoders)}, got {enc_name!r}.")
    if enc.get('binarize', True) is not True:
        raise ConfigError("pre-attention benchmark requires encoder.binarize=true for hidden spike-driven input.")

    if 'count' in enc_name and enc.get('binarize', True) is not True:
        raise ConfigError("Count-valued encodings must be binarized before entering the hidden pipeline.")

    try:
        pixel_threshold = int(enc.get('pixel_threshold', 0))
    except (TypeError, ValueError) as e:
        raise ConfigError("encoder.pixel_threshold must be an integer >= 0.") from e
    if pixel_threshold < 0:
        raise ConfigError("encoder.pixel_threshold must be >= 0.")

    cache_dtype = cfg.get('dataset', {}).get('cache_dtype', 'uint8')
    if cache_dtype not in ('uint8', 'bool'):
        raise ConfigError("dataset.cache_dtype must be 'uint8' or 'bool'.")

    output_format = fe.get('output_format', 'tokens')
    if output_format not in ('tokens', 'feature_map', 'maps'):
        raise ConfigError(f"feature_extractor.output_format must be 'tokens' or 'feature_map', got {output_format!r}.")

    head = cfg.get('model', {}).get('head', {})
    allowed_heads = {'spatio_temporal_avg_readout', 'spikevision_spatial_pooling', 'class_neuron_accumulator'}
    head_name = head.get('name', '')
    if head_name not in allowed_heads:
        raise ConfigError(f"head.name must be one of {sorted(allowed_heads)}, got {head_name!r}.")
    if head.get('terminal_readout', True) is not True:
        raise ConfigError("pre-attention benchmark requires head.terminal_readout=true; logits are allowed only at the terminal boundary.")
    if head_name == 'spikevision_spatial_pooling' and output_format not in ('feature_map', 'maps'):
        raise ConfigError("spikevision_spatial_pooling requires feature_extractor.output_format='feature_map' or 'maps'.")


def deep_update(base: MutableMapping[str, Any], patch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in patch.items():
        if isinstance(v, MutableMapping) and isinstance(base.get(k), MutableMapping):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
