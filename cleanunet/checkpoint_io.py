"""
Checkpoint loading helpers.

Supported sources:
  - PyTorch Lightning checkpoints of this repository (keys "model.clean_unet.*", "model.clean_spec_net.*", ...).
  - Older Lightning checkpoints of a single CleanUNet / CleanSpecNet (keys "model.*").
  - Official CleanUMamba checkpoints (.pkl with "model_state_dict" + "network_config", fp16 weights).
    Their encoder/decoder/tsfm_conv/tsfm_Mamba_layers/norm_f keys match CleanUNet with a Mamba bottleneck.
"""

from typing import Iterable, Optional

import torch
import torch.nn as nn


def _load(path: str) -> dict:
    """Load a checkpoint without executing arbitrary pickled code."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:  # Lightning checkpoints may contain extra (non-tensor) objects
        print(f"[WARNING] weights_only load failed for '{path}' ({type(err).__name__}); "
              "retrying with weights_only=False. Only do this for checkpoints you trust.")
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(path: str, strip_prefixes: Iterable[str] = ()) -> dict:
    """Return the model state dict stored in a checkpoint, with optional prefixes removed."""
    ckpt = _load(path)
    for key in ("state_dict", "model_state_dict", "generator"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    out = {}
    for k, v in ckpt.items():
        for p in strip_prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def load_submodule_weights(module: nn.Module, path: str, submodule_name: str, prefix: Optional[str] = None,
                           strict: bool = True):
    """Load weights of `module` from a checkpoint that may hold it under several prefixes.

    Tried in order: an explicit `prefix`, "model.<submodule_name>.", "<submodule_name>.", "model.".
    """
    state_dict = extract_state_dict(path)
    candidates = [prefix] if prefix else [f"model.{submodule_name}.", f"{submodule_name}.", "model."]
    own_keys = set(module.state_dict().keys())
    for p in candidates:
        sub = {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p)}
        if sub and len(own_keys & set(sub)) > 0:
            sub = {k: v.float() if torch.is_floating_point(v) else v for k, v in sub.items()}
            missing, unexpected = module.load_state_dict(sub, strict=strict)
            print(f"[INFO] Loaded {submodule_name} weights from '{path}' (prefix '{p}').")
            return missing, unexpected
    raise ValueError(f"No compatible keys for '{submodule_name}' in '{path}' (tried prefixes {candidates}).")


def load_cleanumamba_into_cleanunet(cleanunet: nn.Module, path: str, check_config: bool = True):
    """Initialize a CleanUNet (Mamba bottleneck) from an official CleanUMamba checkpoint."""
    ckpt = _load(path)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"'{path}' does not look like a CleanUMamba checkpoint (no 'model_state_dict').")
    if cleanunet.bottleneck_type != "mamba":
        raise ValueError("CleanUMamba weights require bottleneck type 'mamba'")

    if check_config and "network_config" in ckpt:
        cfg = ckpt["network_config"]
        expected = {
            "channels_H": cleanunet.channels_H, "max_H": cleanunet.max_H,
            "encoder_n_layers": cleanunet.encoder_n_layers, "kernel_size": cleanunet.kernel_size,
            "stride": cleanunet.stride, "tsfm_d_model": cleanunet.tsfm_d_model,
            "tsfm_n_layers": cleanunet.tsfm_n_layers,
        }
        diff = {k: (cfg.get(k), v) for k, v in expected.items() if k in cfg and cfg[k] != v}
        if diff:
            raise ValueError(f"CleanUMamba network_config mismatch (checkpoint, model): {diff}")

    state_dict = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = cleanunet.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in CleanUMamba checkpoint: {unexpected[:10]}")
    # Only extra (non-CleanUMamba) modules such as residual projections may be missing
    if missing:
        print(f"[WARNING] {len(missing)} CleanUNet parameters not found in the CleanUMamba checkpoint: {missing[:5]}")
    n = sum(v.numel() for v in state_dict.values())
    print(f"[INFO] Loaded CleanUMamba weights ({n / 1e6:.2f}M parameters) from '{path}'.")
    return missing, unexpected
