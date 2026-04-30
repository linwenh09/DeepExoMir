"""Lightweight YAML config loader for DeepExoMir.

The training/inference pipelines use plain dict-of-dicts configs read from
the YAML files under ``configs/``.  This module is a single import shim so
that downstream users can do::

    from deepexomir.config import load_config
    cfg = load_config("configs/model_config_v19.yaml")
    cfg["model"]["d_model"]

without having to know whether the project uses YAML, JSON, OmegaConf, or
hydra under the hood.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config file as a plain dict.

    Parameters
    ----------
    path : str or Path
        Path to a YAML file (e.g. ``configs/model_config_v19.yaml`` or
        ``configs/model_config_v19_noStructure.yaml``).

    Returns
    -------
    dict
        Nested dict matching the YAML document structure.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    yaml.YAMLError
        If the file cannot be parsed as YAML.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Top-level YAML must be a mapping; got {type(cfg).__name__}")
    return dict(cfg)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins).

    Useful when combining ``model_config_v19.yaml`` with a user-supplied
    inference-time tweak (e.g. batch size, device).
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = merge_configs(dict(out[k]), dict(v))
        else:
            out[k] = v
    return out


__all__ = ["load_config", "merge_configs"]
