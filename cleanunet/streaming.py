"""
Constant-memory streaming inference for CleanUNet with a Mamba bottleneck.

Ported from CleanUMamba (feed / flush / _denoise_frame), with these changes:
  - works as a wrapper around a CleanUNet instance (no state stored in the model),
  - no in-place modification of the input, supports encoder BatchNorm and residual projections,
  - works with both Mamba backends (mamba_ssm CUDA kernels or the pure PyTorch fallback).

The model processes frames of `frame_length = valid_length(1)` samples and advances by
`total_stride` samples per step (E8: 766 / 256 samples = 48 / 16 ms at 16 kHz; E6: 190 / 64).
Each step produces exactly one bottleneck token, so the Mamba layers run in single-token mode.

Limitations: only the waveform branch (CleanUNet) is supported; the hybrid CleanUNet2 needs a
causal frame-wise STFT front-end, which is not implemented yet.
"""

import time

import torch

from .sequence import make_inference_params


class StreamingCleanUNet:
    """Constant-memory streaming wrapper around a Mamba CleanUNet.

    Audio is fed in chunks of any size; every hop of ``total_stride`` samples produces one bottleneck token, so the
    Mamba layers run in single-token mode with cached states. The encoder and decoder keep the small buffers their
    convolutions need. The output matches offline inference except for the input normalization, which here is a
    running estimate (see ``__init__``).
    """
    def __init__(self, net, normalize_input: bool = True, prior_std: float = None, prior_frames: float = 0.0):
        """With normalize_input, every frame is divided by the running mean of the per-frame std. `prior_std` /
        `prior_frames` start that mean from `prior_frames` pseudo-frames at `prior_std`: without them the first
        frames of an utterance, often noise only, give a scale ~3x too small (IMPROVEMENT_PLAN.md, 14.7.4).
        prior_frames = inf keeps the scale fixed at prior_std."""
        if net.bottleneck_type not in ("mamba", "mamba2"):
            raise ValueError("Streaming is only supported for the Mamba bottleneck")
        if net.kernel_size != 2 * net.stride:
            raise ValueError("Streaming overlap-add assumes kernel_size == 2 * stride")
        if prior_frames and prior_std is None:
            raise ValueError("prior_frames needs prior_std")
        self.net = net
        self.normalize_input = normalize_input
        self.prior_std = prior_std
        self.prior_frames = float(prior_frames)
        self.total_stride = net.total_stride
        self.frame_length = net.valid_length(1)
        self.reset()

    def reset(self):
        """Clear all buffers and states before a new utterance."""
        p = next(self.net.parameters())
        self.pending = torch.zeros(self.net.channels_input, 0, device=p.device, dtype=p.dtype)
        self.state = {}
        self.inference_params = make_inference_params(self.net.mamba_backend)
        self.frames = 0
        self.input_std = self.prior_std if self.prior_frames else 0.0
        self.total_time = 0.0

    @property
    def time_per_frame(self) -> float:
        """Mean processing time per frame so far, in seconds."""
        return self.total_time / self.frames if self.frames else 0.0

    @torch.no_grad()
    def feed(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: (1, n) new samples -> (1, m) denoised samples (m is a multiple of total_stride)."""
        if chunk.dim() != 2 or chunk.shape[0] != 1:
            raise ValueError(f"Expected a (1, n) mono chunk, got {tuple(chunk.shape)}")
        begin = time.time()
        self.pending = torch.cat([self.pending, chunk.to(self.pending)], dim=1)
        outs = []
        while self.pending.shape[1] >= self.frame_length:
            self.frames += 1
            frame = self.pending[:, :self.frame_length]
            if self.normalize_input:
                # running average of the per-frame standard deviation (after the prior pseudo-frames)
                std = frame.std(dim=1, keepdim=True) + 1e-3
                n = self.frames + self.prior_frames
                self.input_std = std / n + (1 - 1 / n) * self.input_std
                frame = frame / self.input_std
            out = self._denoise_frame(frame)[:, :self.total_stride]
            if self.normalize_input:
                out = out * self.input_std
            outs.append(out)
            self.pending = self.pending[:, self.total_stride:]
        self.total_time += time.time() - begin
        if outs:
            return torch.cat(outs, dim=1)
        return self.pending.new_zeros(1, 0)

    @torch.no_grad()
    def flush(self) -> torch.Tensor:
        """Pad the remaining samples with zeros and return the last denoised samples."""
        pending_length = self.pending.shape[1]
        out = self.feed(self.pending.new_zeros(1, self.frame_length))
        return out[:, :pending_length]

    def _denoise_frame(self, frame: torch.Tensor) -> torch.Tensor:
        net, K, S = self.net, self.net.kernel_size, self.net.stride
        x = frame.unsqueeze(0)                                    # (1, 1, frame_length)

        skip_connections = []
        total_stride = self.total_stride
        for i, encode in enumerate(net.encoder):
            total_stride //= S
            length = x.shape[2]
            prev = self.state.get(f"enc{i}")
            if prev is not None:
                # only recompute the outputs that depend on new samples
                offset = length - K - S * ((length - K) // S - prev.shape[-1])
                x = x[..., offset:]
            x = encode(x)
            if prev is not None:
                x = torch.cat([prev, x], dim=-1)
            self.state[f"enc{i}"] = x[..., total_stride:]
            skip_connections.append(x)
        if net.residual_projection:
            skip_connections = [proj(s) for proj, s in zip(net.residual_projection_layers, skip_connections)]
        skip_connections = skip_connections[::-1]

        x = net.tsfm_conv1(x)
        x = net.run_bottleneck(x.permute(0, 2, 1), inference_params=self.inference_params).permute(0, 2, 1)
        x = net.tsfm_conv2(x)

        for i, block in enumerate(net.decoder):
            x = x + skip_connections[i][..., :x.shape[-1]]
            conv_t = block[2]
            x = conv_t(block[1](block[0](x)))
            # overlap-add with the tail of the previous frame (bias counted once)
            prev = self.state.get(f"dec{i}")
            self.state[f"dec{i}"] = x[..., -S:] - conv_t.bias.view(-1, 1)
            x = x[..., :-S].clone()
            if prev is not None:
                x[..., :S] += prev
            for post in block[3:]:                                # optional BatchNorm / ReLU
                x = post(x)
        return x[0]


@torch.no_grad()
def stream_denoise(net, audio: torch.Tensor, chunk_size: int = 4096, normalize_input: bool = True,
                   prior_std: float = None, prior_frames: float = 0.0) -> torch.Tensor:
    """Denoise a whole (1, L) signal chunk by chunk; returns (1, L)."""
    streamer = StreamingCleanUNet(net, normalize_input=normalize_input, prior_std=prior_std,
                                  prior_frames=prior_frames)
    outs = [streamer.feed(audio[:, i:i + chunk_size]) for i in range(0, audio.shape[1], chunk_size)]
    outs.append(streamer.flush())
    return torch.cat(outs, dim=1)[:, :audio.shape[1]]
