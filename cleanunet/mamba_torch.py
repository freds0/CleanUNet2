"""
Pure-PyTorch reference implementation of the Mamba (v1) mixer.

It uses exactly the same parameter names and shapes as ``mamba_ssm.Mamba``
(in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj), so state dicts are
interchangeable between both backends. The selective scan is computed with a
sequential loop: it is slow, but it runs on CPU, does not need compiled
kernels and serves as a numerical reference in tests.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TorchInferenceParams:
    """Minimal stand-in for ``mamba_ssm.utils.generation.InferenceParams``."""
    max_seqlen: int = 1
    max_batch_size: int = 1
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)


class TorchMamba(nn.Module):
    """Mamba v1 mixer written with plain PyTorch ops."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: int = None,
        device=None,
        dtype=None,
        **unused,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

        # Same initialization as mamba_ssm.Mamba
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    # ------------------------------------------------------------------
    def _compute_dtype(self):
        """float32, or float64 when the module itself is in double precision."""
        return torch.float64 if self.in_proj.weight.dtype == torch.float64 else torch.float32

    def _ssm_params(self, x):
        """x: (B, L, d_inner) -> dt (B, L, d_inner), B (B, L, N), C (B, L, N)."""
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        return dt, B, C

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D) -> (B, L, D)."""
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, hidden_states.shape[0])
            if inference_params.seqlen_offset > 0:
                outs = [self.step(hidden_states[:, t:t + 1], conv_state, ssm_state)
                        for t in range(hidden_states.shape[1])]
                return torch.cat(outs, dim=1)

        dtype = hidden_states.dtype
        cd = self._compute_dtype()
        hidden_states = hidden_states.to(cd)
        L = hidden_states.shape[1]
        xz = F.linear(hidden_states, self.in_proj.weight.to(cd),
                      None if self.in_proj.bias is None else self.in_proj.bias.to(cd))
        x, z = xz.chunk(2, dim=-1)
        x = F.conv1d(x.transpose(1, 2), self.conv1d.weight.to(cd),
                     None if self.conv1d.bias is None else self.conv1d.bias.to(cd),
                     padding=self.d_conv - 1, groups=self.d_inner)[..., :L]
        x = F.silu(x).transpose(1, 2)                         # (B, L, d_inner)

        dt, B, C = self._ssm_params(x)
        A = -torch.exp(self.A_log.to(cd))                    # (d_inner, N)
        h = x.new_zeros(x.shape[0], self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A)              # (B, d_inner, N)
            dBx = dt[:, t, :, None] * B[:, t, None, :] * x[:, t, :, None]
            h = dA * h + dBx
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D.to(cd)
        y = y * F.silu(z)
        out = self.out_proj(y.to(self.out_proj.weight.dtype))
        return out.to(dtype)

    @torch.no_grad()
    def step(self, hidden_states, conv_state, ssm_state):
        """Single-token update. hidden_states: (B, 1, D); states are updated in place."""
        dtype = hidden_states.dtype
        cd = self._compute_dtype()
        xz = self.in_proj(hidden_states.squeeze(1).to(self.in_proj.weight.dtype)).to(cd)
        x, z = xz.chunk(2, dim=-1)                             # (B, d_inner)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        x = (conv_state * self.conv1d.weight.to(cd).squeeze(1)).sum(-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias.to(cd)
        x = F.silu(x)
        dt, B, C = self._ssm_params(x.to(self.x_proj.weight.dtype))
        dt, B, C = dt.to(cd), B.to(cd), C.to(cd)
        A = -torch.exp(self.A_log.to(cd))
        dA = torch.exp(dt[..., None] * A)
        ssm_state.copy_(ssm_state * dA + dt[..., None] * B[:, None, :] * x[..., None])
        y = (ssm_state * C[:, None, :]).sum(-1) + x * self.D.to(cd)
        y = y * F.silu(z)
        return self.out_proj(y.to(self.out_proj.weight.dtype)).unsqueeze(1).to(dtype)

    def allocate_inference_cache(self, batch_size, max_seqlen=1, dtype=None, **kwargs):
        """Zero convolution and SSM states for single-token (streaming) steps."""
        device = self.out_proj.weight.device
        cd = self._compute_dtype()
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=cd)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=cd)
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size):
        assert self.layer_idx is not None, "layer_idx is required for stateful inference"
        cache = inference_params.key_value_memory_dict
        if self.layer_idx not in cache:
            cache[self.layer_idx] = self.allocate_inference_cache(batch_size)
        return cache[self.layer_idx]
