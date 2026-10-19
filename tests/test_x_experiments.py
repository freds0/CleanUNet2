"""Tests for the follow-ups of the analysis of section 14.7 of IMPROVEMENT_PLAN.md: the prior of the streaming input
scale, the band-limited PESQ and the metric discriminator restricted to the band above 100 Hz."""

import os

import numpy as np
import pytest
import soundfile as sf
import torch

from conftest import DATA_DIR, requires_data
from test_v_experiments import wave_arch
from test_w_experiments import _module_with_loss

SR = 16000


def _small_net():
    from cleanunet.cleanunet2 import CleanUNet2
    torch.manual_seed(0)
    return CleanUNet2(**wave_arch()).clean_unet.eval().double()


def _signal(n=5000):
    torch.manual_seed(1)
    x = torch.randn(1, n, dtype=torch.float64) * 0.1
    x[:, :1500] *= 0.05                           # a quiet start, as in most utterances
    return x


def test_streaming_without_prior_is_unchanged():
    from cleanunet.streaming import stream_denoise
    net, x = _small_net(), _signal()
    with torch.no_grad():
        assert torch.equal(stream_denoise(net, x), stream_denoise(net, x, prior_std=0.3, prior_frames=0.0))


def test_streaming_prior_starts_the_running_scale_and_inf_fixes_it():
    from cleanunet.streaming import StreamingCleanUNet, stream_denoise
    net, x = _small_net(), _signal()
    st = StreamingCleanUNet(net, prior_std=0.2, prior_frames=10)
    with torch.no_grad():
        st.feed(x[:, :st.frame_length + 3 * st.total_stride])
    frames = [x[:, i * st.total_stride:i * st.total_stride + st.frame_length] for i in range(st.frames)]
    stds = [float(f.std() + 1e-3) for f in frames]
    assert st.frames == 4
    assert float(st.input_std) == pytest.approx((10 * 0.2 + sum(stds)) / (10 + 4), rel=1e-12)
    with torch.no_grad():
        fixed = stream_denoise(net, x, prior_std=0.2, prior_frames=float("inf"))
        scaled = stream_denoise(net, x / 0.2, normalize_input=False) * 0.2
    assert (fixed - scaled).abs().max() < 1e-12
    with pytest.raises(ValueError):
        StreamingCleanUNet(net, prior_frames=5)


def test_highpass_removes_the_low_band_only():
    from cleanunet.metric_disc import fft_highpass
    from evaluation.metrics import highpass
    t = np.arange(SR) / SR
    low, high = np.sin(2 * np.pi * 30 * t), np.sin(2 * np.pi * 1000 * t)
    mid = slice(2000, -2000)                      # away from the edges of the zero-phase filter
    assert np.abs(highpass(low)[mid]).max() < 0.01
    assert np.abs(highpass(high)[mid] - high[mid]).max() < 1e-3
    # the differentiable FFT version of the discriminator has the same response
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR)
    y = fft_highpass(torch.from_numpy(x).view(1, -1), 100.0, SR).view(-1).numpy()
    assert np.abs(y - highpass(x))[mid].max() < 0.02 * np.abs(x).max()


@requires_data
def test_band_limited_pesq_ignores_hum_below_100_hz():
    from cleanunet.metric_disc import batch_pesq
    from evaluation.metrics import evaluate_pair
    clean, _ = sf.read(os.path.join(DATA_DIR, "test/clean/p232_001.flac"), dtype="float64")
    hum = 0.05 * np.sin(2 * np.pi * 40 * np.arange(len(clean)) / SR)
    m = evaluate_pair(clean, clean + hum, full=False, band_limited=True)
    assert m["pesq_wb_hp"] > m["pesq_wb"] + 0.3 and m["pesq_wb_hp"] > 4.4
    L = min(len(clean), 32768)
    ref = torch.from_numpy(np.stack([clean[:L]] * 2)).float()
    est = torch.from_numpy(np.stack([(clean + hum)[:L]] * 2)).float()
    plain, band = batch_pesq(ref, est), batch_pesq(ref, est, highpass_hz=100.0)
    assert torch.all(band > plain + 0.05)


def test_discriminator_with_highpass_ignores_the_band_below_it():
    from cleanunet.metric_disc import MetricDiscriminator
    torch.manual_seed(0)
    ref, est = torch.randn(2, 1, SR) * 0.1, torch.randn(2, 1, SR) * 0.1
    hum = 0.3 * torch.sin(2 * np.pi * 20 * torch.arange(SR) / SR)       # 20 whole periods: no FFT leakage
    for hp, same in ((100.0, True), (0.0, False)):
        torch.manual_seed(0)
        d = MetricDiscriminator(channels=4, highpass_hz=hp).eval()
        with torch.no_grad():
            a, b = d(ref, est), d(ref, est + hum)
        assert torch.allclose(a, b, atol=1e-3) == same       # without the high-pass: ~0.06


def test_module_passes_the_highpass_to_the_target_and_the_discriminator():
    module = _module_with_loss(wave_arch(), {"mode": "waveform",
                                             "metric_gan": {"weight": 0.3, "highpass_hz": 100,
                                                            "disc": {"channels": 4}}})
    assert module.metric_disc.highpass_hz == 100.0
    assert module.pesq_fn.keywords["highpass_hz"] == 100.0
    plain = _module_with_loss(wave_arch(), {"mode": "waveform", "metric_gan": {"weight": 0.3}})
    assert plain.metric_disc.highpass_hz == 0.0 and plain.pesq_fn.keywords["highpass_hz"] == 0.0
