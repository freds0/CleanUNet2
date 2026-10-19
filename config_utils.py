"""
YAML configuration helpers.

- `base: <path>` (or a list of paths) inside a YAML file inherits from other files; keys of the
  child are deep-merged on top of the base (paths are relative to the child file).
- `model.architecture` may be a dict or a path to a YAML file holding the CleanUNet2 arguments.
- Command-line overrides use dotted keys: `trainer.max_steps=200 data.batch_size=8`; keys under
  `model.architecture.` are applied to the resolved architecture (e.g. `model.architecture.fusion.shift=1`).
"""

import copy
import os
from typing import Iterable

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (dicts are merged, other values replaced)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    """Load a YAML file and resolve its ``base`` inheritance (paths relative to the file)."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    bases = cfg.pop("base", None)
    if bases:
        merged = {}
        for b in [bases] if isinstance(bases, str) else bases:
            merged = deep_merge(merged, load_yaml(os.path.join(os.path.dirname(path), b)))
        cfg = deep_merge(merged, cfg)
    return cfg


def resolve_architecture(arch, config_dir: str = ".") -> dict:
    """Return the CleanUNet2 constructor kwargs from a dict or a YAML path."""
    if arch is None:
        return {}
    if isinstance(arch, str):
        path = arch if os.path.isabs(arch) or os.path.exists(arch) else os.path.join(config_dir, arch)
        return load_yaml(path)
    return copy.deepcopy(arch)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply `a.b.c=value` overrides (values are parsed as YAML scalars/lists/dicts)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def load_config(path: str, overrides: Iterable[str] = ()) -> dict:
    """Load a training config. Overrides of `model.architecture.<key>` are applied after the architecture
    file is resolved, so they also work when `model.architecture` is a path."""
    prefix = "model.architecture."
    overrides = list(overrides or [])
    arch_overrides = [o[len(prefix):] for o in overrides if o.startswith(prefix)]
    cfg = apply_overrides(load_yaml(path), [o for o in overrides if not o.startswith(prefix)])
    model = cfg.setdefault("model", {})
    model["architecture"] = apply_overrides(
        resolve_architecture(model.get("architecture"), os.path.dirname(path)), arch_overrides)
    return cfg
