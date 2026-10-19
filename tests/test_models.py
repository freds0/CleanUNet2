"""Shape, alignment, causality and initialization tests for the CleanUNet2 components."""

import pytest
import torch

from cleanunet.cleanspecnet import CleanSpecNet
from cleanunet.cleanunet import CleanUNet
from cleanunet.cleanunet2 import CleanUNet2
from cleanunet.spectral import Spectrogram
from conftest import requires_cuda, requires_mamba_ssm

SMALL_UNET = dict(channels_H=16, max_H=64, encoder_n_layers=4, kernel_size=4, stride=2)


def small_cleanunet2(bottleneck, specnet_backbone, **kw):
    return CleanUNet2(
        cleanunet_params={**SMALL_UNET, "bottleneck": bottleneck},
        cleanspecnet_params={"conv_hidden_dim": 16, "hidden_dim": 32,
                             "backbone": specnet_backbone, "causal": kw.pop("causal", True)},
        spectrogram={"n_fft": 256, "hop_length": 64, "win_length": 256},
        **kw)


# ---------------------------------------------------------------- spectrogram
@pytest.mark.parametrize("alignment", ["center", "causal"])
@pytest.mark.parametrize("L", [8192, 8000, 12345])
def test_spectrogram_frames_cover_signal(alignment, L):
    spec = Spectrogram(alignment=alignment)
    s = spec(torch.randn(2, 1, L))
    assert s.shape == (2, 513, -(-L // 256))          # ceil(L / hop) frames, no stretching needed
    assert s.shape[-1] * 256 >= L


def test_spectrogram_causal_alignment_uses_no_future_samples():
    spec = Spectrogram(alignment="causal")
    x = torch.randn(1, 1, 4096)
    y = x.clone()
    y[..., 2048:] += 1.0                               # perturb samples >= 2048
    a, b = spec(x), spec(y)
    # frame j covers [(j+1)*hop - n_fft, (j+1)*hop): frames j < 8 end at or before sample 2048
    assert torch.allclose(a[..., :8], b[..., :8])
    assert not torch.allclose(a[..., 8], b[..., 8])


def test_upsampled_condition_is_cropped_not_stretched():
    model = small_cleanunet2({"type": "transformer", "n_layers": 1, "d_model": 32, "n_head": 4, "d_inner": 64},
                             {"type": "transformer", "n_layers": 1, "n_head": 4})
    spec = model.spectrogram(torch.randn(1, 1, 1000))
    up = model.spec_upsampler(spec)
    assert up.shape[-1] == spec.shape[-1] * 64 >= 1000


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("bottleneck", [
    {"type": "transformer", "n_layers": 2, "d_model": 32, "n_head": 4, "d_inner": 64},
    {"type": "mamba", "n_layers": 2, "d_model": 32, "d_state": 8, "expand": 2, "backend": "torch"},
    {"type": "lstm", "n_layers": 2, "d_model": 32},
])
@pytest.mark.parametrize("specnet", [
    {"type": "transformer", "n_layers": 1, "n_head": 4},
    {"type": "mamba", "n_layers": 1, "d_state": 8, "expand": 2, "backend": "torch"},
    {"type": "lstm", "n_layers": 1},
])
def test_cleanunet2_forward_backward(bottleneck, specnet):
    model = small_cleanunet2(bottleneck, specnet, conditioning_type="film", conditioning_zero_init=True)
    x = torch.randn(2, 1, 3000) * 0.1
    out = model(x)
    assert out["wav"].shape == (2, 1, 3000)
    assert out["spec"].shape == (2, 129, 47)
    (out["wav"].pow(2).mean() + out["spec"].mean()).backward()


def test_bidirectional_and_fusion():
    model = small_cleanunet2({"type": "mamba", "n_layers": 1, "d_model": 32, "d_state": 8, "expand": 2,
                              "backend": "torch"},
                             {"type": "mamba", "n_layers": 1, "d_state": 8, "expand": 2, "backend": "torch",
                              "bidirectional": True},
                             causal=False, bottleneck_fusion=True)
    assert model.fusion.repeat == 4                    # hop 64 / total stride 2^4
    assert model(torch.randn(1, 1, 2000))["wav"].shape == (1, 1, 2000)
    with pytest.raises(ValueError):
        small_cleanunet2({"type": "transformer", "n_layers": 1, "d_model": 32, "n_head": 4, "d_inner": 64},
                         {"type": "mamba", "n_layers": 1, "bidirectional": True, "backend": "torch"}, causal=True)


def test_spec_only_and_waveform_only():
    model = small_cleanunet2({"type": "transformer", "n_layers": 1, "d_model": 32, "n_head": 4, "d_inner": 64},
                             {"type": "transformer", "n_layers": 1, "n_head": 4})
    out = model(torch.randn(1, 1, 2000), spec_only=True)
    assert out["wav"] is None and out["spec"].shape[-1] == 32
    wave_only = CleanUNet2(use_spec_branch=False, cleanunet_params=SMALL_UNET)
    assert not hasattr(wave_only, "clean_spec_net")
    assert wave_only(torch.randn(1, 1, 2000))["spec"] is None


@pytest.mark.parametrize("method", ["addition", "concatenation", "film"])
def test_zero_init_conditioner_equals_waveform_branch(method):
    torch.manual_seed(0)
    model = small_cleanunet2({"type": "transformer", "n_layers": 1, "d_model": 32, "n_head": 4, "d_inner": 64},
                             {"type": "transformer", "n_layers": 1, "n_head": 4},
                             conditioning_type=method, conditioning_zero_init=True, bottleneck_fusion=True).eval()
    x = torch.randn(2, 1, 3000) * 0.1
    with torch.no_grad():
        hybrid = model(x)["wav"]
        scale = model.input_scale(x)
        wave_only = model.clean_unet(x / scale) * scale
    assert torch.allclose(hybrid, wave_only, atol=1e-5)


# ---------------------------------------------------------------- causality
def _causality_check(net, L=6000, t0=4000):
    x = torch.randn(1, 1, L)
    y = x.clone()
    y[..., t0:] += torch.randn(1, 1, L - t0)
    with torch.no_grad():
        a, b = net(x), net(y)
    lookahead = net.valid_length(1) if hasattr(net, "valid_length") else 0
    assert torch.allclose(a[..., :t0 - lookahead], b[..., :t0 - lookahead], atol=1e-5)
    assert not torch.allclose(a[..., t0:], b[..., t0:])


@pytest.mark.parametrize("bottleneck", [
    {"type": "transformer", "n_layers": 2, "d_model": 32, "n_head": 4, "d_inner": 64, "causal": True},
    {"type": "mamba", "n_layers": 2, "d_model": 32, "d_state": 8, "expand": 2, "backend": "torch"},
    {"type": "lstm", "n_layers": 1, "d_model": 32},
])
def test_cleanunet_is_causal_up_to_lookahead(bottleneck):
    net = CleanUNet(**SMALL_UNET, bottleneck=bottleneck, normalize_input=False).eval()
    _causality_check(net)


@pytest.mark.parametrize("backbone", [{"type": "transformer", "n_layers": 1, "n_head": 4},
                                      {"type": "mamba", "n_layers": 1, "d_state": 8, "backend": "torch"}])
def test_causal_cleanspecnet(backbone):
    net = CleanSpecNet(input_channels=33, conv_hidden_dim=8, hidden_dim=16, causal=True, backbone=backbone).eval()
    x = torch.randn(1, 33, 50)
    y = x.clone()
    y[..., 30:] += 1.0
    with torch.no_grad():
        assert torch.allclose(net(x)[..., :30], net(y)[..., :30], atol=1e-5)


# ---------------------------------------------------------------- legacy compatibility
def test_legacy_transformer_keys_are_unchanged():
    """Default CleanUNet / CleanSpecNet keep the parameter names of the original repository."""
    keys = set(CleanUNet().state_dict())
    assert "tsfm_encoder.layer_stack.0.slf_attn.w_qs.weight" in keys and "encoder.0.0.weight" in keys
    assert sum(p.numel() for p in CleanUNet().parameters()) == 46_071_937  # verified against the original implementation (git HEAD)
    spec_keys = set(CleanSpecNet().state_dict())
    assert "attention_layers.0.self_attn.in_proj_weight" in spec_keys
    assert sum(p.numel() for p in CleanSpecNet().parameters()) == 12_487_365


@requires_cuda
@requires_mamba_ssm
def test_cuda_and_torch_mamba_backends_match():
    torch.manual_seed(0)
    kw = dict(**SMALL_UNET, normalize_input=False)
    cuda_net = CleanUNet(bottleneck={"type": "mamba", "n_layers": 2, "d_model": 64, "backend": "cuda"}, **kw)
    torch_net = CleanUNet(bottleneck={"type": "mamba", "n_layers": 2, "d_model": 64, "backend": "torch"}, **kw)
    torch_net.load_state_dict(cuda_net.state_dict())
    x = torch.randn(2, 1, 4000)
    with torch.no_grad():
        a = cuda_net.cuda().eval()(x.cuda()).cpu()
        b = torch_net.eval()(x)
    assert (a - b).abs().max() < 1e-3 * max(1.0, a.abs().max().item())
