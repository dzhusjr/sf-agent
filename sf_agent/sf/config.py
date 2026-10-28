"""YAML configs with `base:` inheritance and dotted command-line overrides:
    python train.py -c configs/lstm.yaml --set env.backend=dummy train.total_timesteps=20000
"""
from __future__ import annotations

import copy
import os

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = _merge(base, cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg
