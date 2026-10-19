"""
Conditioning / fusion of CleanSpecNet information into CleanUNet (see IMPROVEMENT_PLAN.md).

A frame-rate condition c (B, D, T_frames) -- CleanSpecNet hidden features, its denoised spectrogram,
the implied mask, or control signals (noisy / oracle spectrogram) -- goes through a small causal trunk
and per-point heads that modulate CleanUNet features at named injection points:

    enc_s{2..}   output of an encoder block            skip_s{2..}  skip connection
    bott_pre     after tsfm_conv1 (before the sequence model)
    bott_post    after the sequence model (before tsfm_conv2)
    dec_s{2..}   decoder block input (after adding the skip)

Every operator starts as the identity (last layer zero-initialized or a zero gate), so a model built
from pretrained branches initially reproduces its waveform branch exactly.

Time alignment: frame j covers samples [j*hop, (j+1)*hop). A position n at stride s starts at sample
n*s and uses frame floor(n*s/hop) (+ `shift`). With shift 0 this adds no lookahead to CleanUNet.
When the frames are faster than the positions (hop < stride, e.g. hop 128 at the bottleneck, N6b), the
stride/hop frames inside each position's span are averaged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

OPERATORS = ("film", "film_tanh", "adaln_zero", "add", "gate", "concat_mlp", "xattn")
# number of per-channel parameters each operator needs from its head
_N_PARAMS = {"film": 2, "film_tanh": 2, "adaln_zero": 3, "add": 1, "gate": 1}


def _zero_(layer: nn.Module):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _small_(layer: nn.Module, std: float = 1e-3):
    nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_last(layer: nn.Module, mode: str):
    """Initialize the layer that makes an operator the identity: 'last' = zeros, 'small' = N(0, 1e-3)."""
    if mode == "last":
        _zero_(layer)
    elif mode == "small":
        _small_(layer)
    else:
        raise ValueError(f"zero_init must be 'last' or 'small', got '{mode}'")


class ChannelNorm(nn.Module):
    """LayerNorm over channels at every time step of a (B, C, T) tensor (causal)."""

    def __init__(self, channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=affine)

    def forward(self, x):
        """LayerNorm over the channel axis of (B, C, T)."""
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    """Conv1d with left padding only, so output frame t depends on frames <= t."""
    def forward(self, x):
        """Left-pad by kernel_size - 1 and convolve."""
        k = self.kernel_size[0]
        return super().forward(F.pad(x, (k - 1, 0)) if k > 1 else x)


class CondTrunk(nn.Module):
    """Frame-rate condition encoder: [LayerNorm] -> CausalConv(k) -> GELU -> (Conv1x1 -> GELU) * (layers-1).

    layers=0 gives a normalized identity (the heads then act as plain linear maps of the condition).
    """

    def __init__(self, in_dim: int, channels: int = 256, kernel: int = 3, layers: int = 2, in_norm: bool = True):
        super().__init__()
        self.in_norm = ChannelNorm(in_dim) if in_norm else nn.Identity()
        mods: List[nn.Module] = []
        dim = in_dim
        for i in range(layers):
            mods += [CausalConv1d(dim, channels, kernel if i == 0 else 1), nn.GELU()]
            dim = channels
        self.net = nn.Sequential(*mods)
        self.out_dim = dim

    def forward(self, c):
        """Shared trunk: normalized condition (B, D, T) -> features for the per-point heads."""
        return self.net(self.in_norm(c))


def frames_to_positions(p: torch.Tensor, repeat: int, length: int, mode: str = "repeat",
                        shift: int = 0, convT: nn.Module = None, pool: int = 1) -> torch.Tensor:
    """Resample frame-rate parameters (B, C, T_f) to `length` positions of stride hop/repeat
    (or hop*pool when the frames are faster than the positions)."""
    if shift:
        # position j uses frame j + shift (clamped to the sequence); shift > 0 adds shift * hop of lookahead
        T = p.shape[-1]
        idx = (torch.arange(T, device=p.device) + shift).clamp(0, T - 1)
        p = p.index_select(-1, idx)
    if pool > 1:   # frames j*pool .. j*pool+pool-1 all lie inside position j: average them
        p = F.avg_pool1d(p, pool, stride=pool, ceil_mode=True)
    if repeat > 1:
        if mode == "repeat":
            p = p.repeat_interleave(repeat, dim=-1)
        elif mode == "linear":
            # position k of frame j: (1 - a_k) * p[j-1] + a_k * p[j], a_k = (k + 1) / repeat  (no lookahead)
            prev = torch.cat([p[..., :1], p[..., :-1]], dim=-1)
            a = (torch.arange(repeat, device=p.device, dtype=p.dtype) + 1) / repeat
            p = (prev.unsqueeze(-1) * (1 - a) + p.unsqueeze(-1) * a).flatten(-2)
        elif mode == "convT":
            p = convT(p)
        else:
            raise ValueError(f"Unknown resample mode '{mode}' (repeat | linear | convT)")
    if p.shape[-1] < length:
        p = F.pad(p, (0, length - p.shape[-1]), mode="replicate")
    return p[..., :length]


class FusionPoint(nn.Module):
    """One injection point: head(s) on the trunk features + an operator on the CleanUNet features."""

    def __init__(self, op: str, channels: int, repeat: int, trunk_dim: int, resample: str = "repeat",
                 shift: int = 0, zero_init: str = "last", hidden: int = 256, n_heads: int = 4, window: int = 8,
                 pool: int = 1):
        super().__init__()
        if op not in OPERATORS:
            raise ValueError(f"Unknown fusion operator '{op}' (use one of {OPERATORS})")
        self.op, self.C, self.repeat, self.resample, self.shift = op, channels, repeat, resample, shift
        self.pool = pool

        if op in _N_PARAMS:
            out = _N_PARAMS[op] * channels
            self.head = nn.Conv1d(trunk_dim, out, 1)
            init_last(self.head, zero_init)
            if op == "adaln_zero":
                self.norm = ChannelNorm(channels, affine=False)
        elif op == "concat_mlp":
            out = min(channels, hidden)
            self.head = nn.Conv1d(trunk_dim, out, 1)
            h = min(channels, hidden)
            self.mlp = nn.Sequential(nn.Conv1d(channels + out, h, 1), nn.GELU(), nn.Conv1d(h, channels, 1))
            init_last(self.mlp[-1], zero_init)
        else:  # xattn: causal windowed cross-attention to the frames, gated by tanh(a), a = 0
            if repeat != 1 or pool != 1:
                raise ValueError("xattn is only supported at frame rate (bott_pre / bott_post / stride == hop)")
            out = hidden
            self.head = nn.Conv1d(trunk_dim, 2 * hidden, 1)          # keys and values
            self.q_norm = ChannelNorm(channels)
            self.q = nn.Conv1d(channels, hidden, 1)
            self.o = nn.Conv1d(hidden, channels, 1)
            self.gate = nn.Parameter(torch.zeros(1))
            self.n_heads, self.window = n_heads, window
        self.convT = None
        if resample == "convT" and repeat > 1:
            # depthwise transposed conv initialized to frame repetition
            self.convT = nn.ConvTranspose1d(out, out, repeat, stride=repeat, groups=out)
            nn.init.ones_(self.convT.weight)
            nn.init.zeros_(self.convT.bias)

    def _params(self, z, length):
        return frames_to_positions(self.head(z), self.repeat, length, self.resample, self.shift, self.convT,
                                   self.pool)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """h: CleanUNet features (B, C, T); z: trunk features (B, D, T_frames)."""
        T = h.shape[-1]
        if self.op == "xattn":
            return h + torch.tanh(self.gate).to(h.dtype) * self._xattn(h, z)
        p = self._params(z, T).to(h.dtype)
        if self.op == "film":
            g, b = p.chunk(2, dim=1)
            return h * (1 + g) + b
        if self.op == "film_tanh":
            g, b = p.chunk(2, dim=1)
            return h * (1 + torch.tanh(g)) + b
        if self.op == "adaln_zero":
            a, g, b = p.chunk(3, dim=1)
            return h + a * ((1 + g) * self.norm(h) + b)
        if self.op == "add":
            return h + p
        if self.op == "gate":
            return h * (1 + torch.tanh(p))
        return h + self.mlp(torch.cat([h, p], dim=1))            # concat_mlp

    def _xattn(self, h, z):
        B, _, T = h.shape
        kv = self.head(z)                                        # (B, 2*hid, T_f)
        kv = frames_to_positions(kv, 1, T, "repeat", self.shift)  # align frames with tokens
        k, v = kv.chunk(2, dim=1)
        W, nh = self.window, self.n_heads
        # causal windows: token t attends to frames t-W+1 .. t
        k = F.pad(k, (W - 1, 0), mode="replicate").unfold(-1, W, 1)    # (B, hid, T, W)
        v = F.pad(v, (W - 1, 0), mode="replicate").unfold(-1, W, 1)
        q = self.q(self.q_norm(h))                                       # (B, hid, T)
        d = q.shape[1] // nh
        q = q.view(B, nh, d, T)
        k = k.reshape(B, nh, d, T, W)
        v = v.reshape(B, nh, d, T, W)
        att = torch.einsum("bhdt,bhdtw->bhtw", q.float(), k.float()) / math.sqrt(d)
        att = att.softmax(dim=-1)
        out = torch.einsum("bhtw,bhdtw->bhdt", att, v.float()).reshape(B, nh * d, T)
        return self.o(out.to(h.dtype))


class FusionHub(nn.Module):
    """Builds the trunk and the fusion points; `hooks(c)` returns {point_name: callable(h) -> h'}.

    Args:
        points_spec: {point_name: (channels, stride)} offered by CleanUNet.fusion_points().
        cond_dim: channels of the frame-rate condition.
        hop: STFT hop (samples per frame).
        points: {point_name: {op: ..., [hidden, n_heads, window]}}.
    """

    def __init__(self, points_spec: Dict[str, Tuple[int, int]], cond_dim: int, hop: int, points: dict,
                 trunk: dict = None, resample: str = "repeat", shift: int = 0, zero_init: str = "last"):
        super().__init__()
        self.trunk = CondTrunk(cond_dim, **(trunk or {}))
        self.points = nn.ModuleDict()
        for name, opts in (points or {}).items():
            if name not in points_spec:
                raise ValueError(f"Unknown fusion point '{name}'. Available: {sorted(points_spec)}")
            opts = dict(opts or {})
            channels, stride = points_spec[name]
            if hop % stride == 0:
                repeat, pool = hop // stride, 1
            elif stride % hop == 0:
                repeat, pool = 1, stride // hop
            else:
                raise ValueError(f"Fusion point '{name}' (stride {stride}) is incompatible with hop {hop}")
            self.points[name] = FusionPoint(opts.pop("op", "film_tanh"), channels, repeat,
                                            self.trunk.out_dim, resample=resample, shift=shift,
                                            zero_init=zero_init, pool=pool, **opts)

    def hooks(self, cond: torch.Tensor) -> dict:
        """Run the trunk once and return {point name: hook(features) -> modulated features}."""
        z = self.trunk(cond)
        return {name: (lambda h, m=m: m(h, z)) for name, m in self.points.items()}


class InputEstimateFusion(nn.Module):
    """S5: a time-domain estimate (iSTFT of the CleanSpecNet magnitude with the noisy phase) enters the
    first encoder level through its own (zero-initialized) strided convolution, added to enc_s2."""

    def __init__(self, channels: int, kernel: int, stride: int, zero_init: str = "last"):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel, stride)
        self.act = nn.GELU()
        self.out = nn.Conv1d(channels, channels, 1)
        init_last(self.out, zero_init)

    def forward(self, h: torch.Tensor, est: torch.Tensor, cleanunet) -> torch.Tensor:
        """Add a projection of the spectral waveform estimate to the first encoder features."""
        from .cleanunet import padding
        e = padding(est.to(h.dtype), cleanunet.encoder_n_layers, cleanunet.kernel_size, cleanunet.stride)
        e = self.out(self.act(self.conv(e)))
        return h + e[..., :h.shape[-1]]


class BranchExchange(nn.Module):
    """O7: bidirectional exchange between the CleanUNet bottleneck Mamba stack and the CleanSpecNet
    Mamba backbone, run in lockstep (frame j <-> token j; requires hop == CleanUNet total stride).

    After wave layer i, wave and spectral residual streams receive zero-initialized projections of each
    other: h_wave += Z(LN(stream_spec)), h_spec += Z(LN(stream_wave)).
    """

    def __init__(self, d_wave: int, d_spec: int, n_wave: int, n_spec: int, zero_init: str = "last"):
        super().__init__()
        # spectral layer after which exchange i happens (spread evenly over the deeper stack)
        self.spec_after = [max(0, round((i + 1) * n_spec / n_wave) - 1) for i in range(n_wave)]
        self.s2w = nn.ModuleList()
        self.w2s = nn.ModuleList()
        for _ in range(n_wave):
            self.s2w.append(nn.Sequential(nn.LayerNorm(d_spec), nn.Linear(d_spec, d_wave)))
            self.w2s.append(nn.Sequential(nn.LayerNorm(d_wave), nn.Linear(d_wave, d_spec)))
            init_last(self.s2w[-1][1], zero_init)
            init_last(self.w2s[-1][1], zero_init)

    @staticmethod
    def _match(x, T):
        if x.shape[1] < T:
            x = F.pad(x.transpose(1, 2), (0, T - x.shape[1]), mode="replicate").transpose(1, 2)
        return x[:, :T]

    def run(self, wave_layers, wave_norm, spec_layers, spec_norm, xw, xs):
        """xw: (B, T_w, d_wave) bottleneck tokens; xs: (B, T_s, d_spec) projected spectral features.
        Returns (wave output (B, T_w, d_wave), spectral hidden (B, T_s, d_spec))."""
        hw, rw = xw, None
        hs, rs = xs, None
        k = 0
        for i, layer in enumerate(wave_layers):
            while k <= self.spec_after[i]:
                hs, rs = spec_layers[k](hs, rs)
                k += 1
            hw, rw = layer(hw, rw)
            sw = hw + rw
            ss = hs + rs
            hw = hw + self._match(self.s2w[i](ss.to(hw.dtype)), sw.shape[1]).to(hw.dtype)
            hs = hs + self._match(self.w2s[i](sw.to(hs.dtype)), ss.shape[1]).to(hs.dtype)
        while k < len(spec_layers):
            hs, rs = spec_layers[k](hs, rs)
            k += 1
        out_w = wave_norm((hw + rw).to(dtype=wave_norm.weight.dtype))
        out_s = spec_norm((hs + rs).to(dtype=spec_norm.weight.dtype))
        return out_w, out_s
