"""
Sequence backbones shared by CleanUNet (bottleneck) and CleanSpecNet.

The Mamba stack mirrors the layout of ``mamba_ssm.models.mixer_seq_simple``
(pre-norm residual blocks with ``.norm`` / ``.mixer`` attributes and a final
``norm_f``), which is also the layout used by CleanUMamba. Keeping these names
makes CleanUMamba checkpoints and its pruning utilities directly usable.

``create_block`` from mamba_ssm is intentionally not used: its signature changed
between mamba-ssm 1.x and 2.x (``d_intermediate`` became mandatory).
"""

import math
from functools import partial

import torch
import torch.nn as nn

from .mamba_torch import TorchMamba, TorchInferenceParams

try:  # Optional CUDA kernels
    from mamba_ssm import Mamba as CudaMamba
    from mamba_ssm import Mamba2 as CudaMamba2
    from mamba_ssm.utils.generation import InferenceParams as CudaInferenceParams
    HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover - depends on the environment
    CudaMamba = CudaMamba2 = CudaInferenceParams = None
    HAS_MAMBA_SSM = False


SEQUENCE_TYPES = ("transformer", "mamba", "mamba2", "lstm")


def resolve_backend(backend: str = "auto") -> str:
    """Return 'cuda' or 'torch' for the requested Mamba backend."""
    if backend == "auto":
        return "cuda" if HAS_MAMBA_SSM and torch.cuda.is_available() else "torch"
    if backend == "cuda" and not HAS_MAMBA_SSM:
        raise ImportError("backend='cuda' requested but mamba_ssm is not installed")
    if backend not in ("cuda", "torch"):
        raise ValueError(f"Unknown Mamba backend '{backend}' (use auto | cuda | torch)")
    return backend


def make_inference_params(backend: str):
    """Create an (empty) inference-state container for streaming."""
    if backend == "cuda":
        return CudaInferenceParams(max_seqlen=1, max_batch_size=1, seqlen_offset=1)
    return TorchInferenceParams(max_seqlen=1, max_batch_size=1, seqlen_offset=1)


def build_mixer(seq_type: str, d_model: int, layer_idx: int, d_state: int = 64, d_conv: int = 4,
                expand: float = 4, headdim: int = 64, backend: str = "auto", use_fast_path: bool = True):
    """Instantiate a single (unidirectional) Mamba / Mamba2 mixer."""
    backend = resolve_backend(backend)
    if seq_type == "mamba":
        if backend == "cuda":
            return CudaMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                             use_fast_path=use_fast_path, layer_idx=layer_idx)
        return TorchMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=layer_idx)
    if seq_type == "mamba2":
        if backend != "cuda":
            raise ValueError("Mamba2 requires the mamba_ssm CUDA kernels (backend='cuda')")
        if (d_model * expand) % headdim != 0:
            raise ValueError(f"d_model*expand ({d_model * expand}) must be divisible by headdim ({headdim})")
        return CudaMamba2(d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim,
                          layer_idx=layer_idx)
    raise ValueError(f"'{seq_type}' is not a Mamba mixer type")


class BiMambaMixer(nn.Module):
    """Non-causal mixer: a forward and a time-reversed Mamba whose outputs are summed."""

    def __init__(self, mixer_fn):
        super().__init__()
        self.fwd = mixer_fn()
        self.bwd = mixer_fn()

    def forward(self, x, inference_params=None):
        """Sum of a forward Mamba and a Mamba run on the time-reversed sequence (offline only)."""
        if inference_params is not None:
            raise RuntimeError("Bidirectional Mamba cannot run in streaming mode")
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)


class MambaBlock(nn.Module):
    """Pre-norm residual block, same layout and parameter names as mamba_ssm's Block."""

    def __init__(self, d_model: int, mixer: nn.Module, norm_epsilon: float = 1e-5, residual_in_fp32: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.mixer = mixer
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, hidden_states, residual=None, inference_params=None):
        """Pre-norm residual block (mamba_ssm layout). Returns (hidden_states, residual)."""
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def build_mamba_layers(d_model: int, n_layers: int, seq_type: str = "mamba", d_state: int = 64,
                       d_conv: int = 4, expand: float = 4, headdim: int = 64, bidirectional: bool = False,
                       backend: str = "auto", use_fast_path: bool = True, norm_epsilon: float = 1e-5,
                       residual_in_fp32: bool = True):
    """Return (ModuleList of MambaBlocks, final LayerNorm)."""
    layers = nn.ModuleList()
    for i in range(n_layers):
        mixer_fn = partial(build_mixer, seq_type, d_model, i, d_state=d_state, d_conv=d_conv, expand=expand,
                           headdim=headdim, backend=backend, use_fast_path=use_fast_path)
        mixer = BiMambaMixer(mixer_fn) if bidirectional else mixer_fn()
        layers.append(MambaBlock(d_model, mixer, norm_epsilon=norm_epsilon, residual_in_fp32=residual_in_fp32))
    norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)
    init_mamba_weights(layers, n_layers)
    return layers, norm_f


def run_mamba_layers(layers: nn.ModuleList, norm_f: nn.LayerNorm, x: torch.Tensor, inference_params=None):
    """Apply a Mamba stack to x of shape (B, T, D)."""
    hidden_states, residual = x, None
    for layer in layers:
        hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
    residual = hidden_states + residual if residual is not None else hidden_states
    return norm_f(residual.to(dtype=norm_f.weight.dtype))


def init_mamba_weights(module: nn.Module, n_layer: int, rescale_prenorm_residual: bool = True):
    """GPT-2 style init used by mamba_ssm / CleanUMamba: zero Linear biases (unless flagged)
    and rescale the residual output projections by 1/sqrt(n_layer)."""
    for m in module.modules():
        if isinstance(m, nn.Linear) and m.bias is not None and not getattr(m.bias, "_no_reinit", False):
            nn.init.zeros_(m.bias)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_layer)


class MambaStack(nn.Module):
    """Standalone (B, T, D) -> (B, T, D) Mamba stack (used by CleanSpecNet)."""

    def __init__(self, d_model: int, n_layers: int, **kwargs):
        super().__init__()
        self.layers, self.norm_f = build_mamba_layers(d_model, n_layers, **kwargs)

    def forward(self, x, inference_params=None):
        """(B, T, d_model) -> (B, T, d_model) through the blocks and the final norm."""
        return run_mamba_layers(self.layers, self.norm_f, x, inference_params)


class LSTMStack(nn.Module):
    """Unidirectional (causal) or bidirectional LSTM with an output projection back to d_model."""

    def __init__(self, d_model: int, n_layers: int, bidirectional: bool = False, **unused):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(2 * d_model, d_model) if bidirectional else nn.Identity()

    def forward(self, x, inference_params=None):
        """(B, T, d_model) -> (B, T, d_model); a bidirectional LSTM is projected back to d_model."""
        return self.proj(self.lstm(x)[0])
