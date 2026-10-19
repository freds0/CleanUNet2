"""Tests for the conditioning / fusion variants of IMPROVEMENT_PLAN.md."""

import pytest
import torch

from cleanunet.cleanunet2 import CleanUNet2
from cleanunet.fusion import frames_to_positions
from cleanunet.spectral import Spectrogram

MAMBA = {"type": "mamba", "n_layers": 2, "d_model": 32, "d_state": 8, "expand": 2, "backend": "torch"}
SPEC_MAMBA = {"type": "mamba", "n_layers": 3, "d_state": 8, "expand": 2, "backend": "torch"}


def fusion_model(fusion, unet=None, spectrogram=None, conditioning_type="none", **kw):
    """Small model with the E8 alignment rule (hop == total stride) unless overridden."""
    unet = unet or dict(channels_H=8, max_H=32, encoder_n_layers=4, kernel_size=4, stride=2, bottleneck=MAMBA)
    torch.manual_seed(0)
    return CleanUNet2(cleanunet_params=unet,
                      cleanspecnet_params={"conv_hidden_dim": 16, "hidden_dim": 32, "num_conv_layers": 2,
                                           "backbone": SPEC_MAMBA, "causal": True},
                      spectrogram=spectrogram or {"n_fft": 64, "hop_length": 16, "win_length": 64},
                      conditioning_type=conditioning_type, conditioning_zero_init=True, fusion=fusion, **kw)


VARIANTS = {
    "P1_enc_s2": {"points": {"enc_s2": {"op": "film_tanh"}}},
    "P2_enc_fine": {"points": {f"enc_s{s}": {"op": "film_tanh"} for s in (2, 4, 8)}},
    "P3_bott_pre": {"points": {"bott_pre": {"op": "film_tanh"}}},
    "P4_bott_post": {"points": {"bott_post": {"op": "film_tanh"}}},
    "P5_dec": {"points": {f"dec_s{s}": {"op": "film_tanh"} for s in (16, 8)}},
    "P7_skip_gates": {"points": {f"skip_s{s}": {"op": "gate"} for s in (2, 4, 8, 16)}},
    "P8_postfilter": {"points": {"out": {}}},
    "S3_mask": {"signal": "mask", "points": {"bott_pre": {"op": "film"}}},
    "S4_concat": {"signal": ["spec_hidden", "mask"], "points": {"bott_pre": {"op": "film"}}},
    "S5_estimate": {"input_estimate": True},
    "K2_noisy": {"signal": "noisy_spec", "points": {"bott_pre": {"op": "film"}}},
    "O3_adaln": {"points": {"bott_pre": {"op": "adaln_zero"}}},
    "O4_add": {"points": {"dec_s4": {"op": "add"}}},
    "O5_concat_mlp": {"points": {"enc_s4": {"op": "concat_mlp"}}},
    "O6_xattn": {"points": {"bott_post": {"op": "xattn", "hidden": 16, "window": 4}}},
    "O7_exchange": {"exchange": True},
    "G1_linear": {"trunk": {"layers": 0}, "points": {"bott_pre": {"op": "film"}}},
    "R2_linear": {"resample": "linear", "points": {"enc_s2": {"op": "film_tanh"}}},
    "R3_convT": {"resample": "convT", "points": {"enc_s4": {"op": "film_tanh"}}},
    "L+1_shift": {"shift": 1, "points": {"bott_pre": {"op": "film_tanh"}}},
    "T4_detach": {"detach": True, "points": {"bott_pre": {"op": "film_tanh"}}},
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_fusion_starts_as_identity_and_trains(name):
    model = fusion_model(VARIANTS[name]).eval()
    x = torch.randn(2, 1, 1000) * 0.1
    with torch.no_grad():
        off = model(x, fusion_enabled=False)["wav"]
        on = model(x)["wav"]
    assert on.shape == x.shape
    assert torch.allclose(on, off, atol=1e-5), (on - off).abs().max()

    # every fusion module receives a gradient at the first step
    model.train()
    out = model(x)
    loss = out["wav"].pow(2).mean() + (out["spec"].abs().mean() if out["spec"] is not None else 0)
    loss.backward()
    fusion_params = [(n, p) for n, p in model.named_parameters()
                     if n.split(".")[0] in ("fusion_hub", "postfilter", "input_estimate", "exchange")]
    assert fusion_params
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in fusion_params)


def test_offline_combination_starts_as_identity_and_trains_every_module():
    """Phase C: bottleneck point + time-domain input estimate (S5) + output fusion (P8) in one model."""
    model = fusion_model({"input_estimate": True, "points": {"bott_pre": {"op": "film_tanh"}, "out": {}}}).eval()
    x = torch.randn(2, 1, 1000) * 0.1
    with torch.no_grad():
        off = model(x, fusion_enabled=False)["wav"]
        on = model(x)["wav"]
    assert torch.allclose(on, off, atol=1e-5), (on - off).abs().max()

    model.train()
    out = model(x)
    (out["wav"].pow(2).mean() + out["spec"].abs().mean()).backward()
    for module in ("fusion_hub", "postfilter", "input_estimate"):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(module + ".")]
        assert grads, module
        assert any(g is not None and g.abs().sum() > 0 for g in grads), module


@pytest.mark.parametrize("name", ["P1_enc_s2", "P3_bott_pre", "P7_skip_gates", "R2_linear", "O6_xattn",
                                  "O7_exchange"])
def test_fusion_adds_no_lookahead_with_e8_geometry(name):
    """With n_fft 1024 / hop 256 (center) and an 8-level U-Net, fusing frame j into position j keeps the
    CleanUNet lookahead (766 samples for E8)."""
    unet = dict(channels_H=4, max_H=8, encoder_n_layers=8, kernel_size=4, stride=2,
                bottleneck={**MAMBA, "d_model": 16}, normalize_input=False)
    model = fusion_model(VARIANTS[name], unet=unet,
                         spectrogram={"n_fft": 1024, "hop_length": 256, "win_length": 1024},
                         normalize_input=False).eval()   # global std normalization sees the future
    # make the fusion non-trivial
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.split(".")[0] in ("fusion_hub", "exchange"):
                p.add_(torch.randn_like(p) * 0.1)
    L, t0 = 12000, 8000
    x = torch.randn(1, 1, L)
    y = x.clone()
    y[..., t0:] += torch.randn(1, 1, L - t0)
    with torch.no_grad():
        a, b = model(x)["wav"], model(y)["wav"]
    lookahead = model.clean_unet.valid_length(1)
    assert lookahead == 766
    assert torch.allclose(a[..., :t0 - lookahead], b[..., :t0 - lookahead], atol=1e-5)
    assert not torch.allclose(a[..., t0:], b[..., t0:])


def test_oracle_signal_needs_clean_and_condition_override():
    model = fusion_model({"signal": "oracle_spec", "points": {"bott_pre": {"op": "film"}}}).eval()
    x, c = torch.randn(1, 1, 800) * 0.1, torch.randn(1, 1, 800) * 0.1
    assert model.needs_clean
    with pytest.raises(ValueError):
        model(x)
    with torch.no_grad():
        model.fusion_hub.points["bott_pre"].head.weight.normal_()
        a = model(x, clean=c)["wav"]
        b = model(x, cond_noisy=torch.randn(1, 1, 800), cond_clean=torch.randn(1, 1, 800))["wav"]
    assert not torch.allclose(a, b)


def test_original_conditioning_still_works_with_fusion():
    model = fusion_model({"points": {"bott_pre": {"op": "film_tanh"}}}, conditioning_type="film").eval()
    x = torch.randn(1, 1, 900) * 0.1
    with torch.no_grad():
        assert torch.allclose(model(x)["wav"], model(x, fusion_enabled=False)["wav"], atol=1e-5)


def test_no_upsampler_when_conditioning_is_none():
    model = fusion_model({"points": {"bott_pre": {}}})
    assert not hasattr(model, "spec_upsampler") and not hasattr(model, "conditioner")


@pytest.mark.parametrize("mode", ["repeat", "linear"])
def test_frames_to_positions_alignment(mode):
    p = torch.arange(4.0).view(1, 1, 4)
    out = frames_to_positions(p, 4, 16, mode)
    if mode == "repeat":
        assert out.flatten().tolist() == [v for v in range(4) for _ in range(4)]
    else:   # never uses frame j+1 for positions of frame j
        assert out[0, 0, 3].item() == 0 and out[0, 0, 7].item() == 1 and out[0, 0, 4].item() == 0.25
    shifted = frames_to_positions(p, 1, 4, shift=1)
    assert shifted.flatten().tolist() == [1, 2, 3, 3]


@pytest.mark.parametrize("alignment", ["center", "causal"])
def test_istft_inverts_stft(alignment):
    s = Spectrogram(alignment=alignment)
    for L in (16000, 32768, 4001):
        x = torch.randn(2, 1, L)
        y = s.istft(s.complex_stft(x), L)
        # with causal alignment the last samples are only covered by a window tail and are faded out
        n = L - 64 if alignment == "causal" else L
        assert torch.allclose(y[..., :n], x[..., :n], atol=1e-4)


def test_fusion_model_loads_pretrained_branches(tmp_path):
    src = fusion_model(None)
    dst = fusion_model({"points": {"bott_pre": {"op": "film_tanh"}, "dec_s2": {"op": "film_tanh"}}})
    path = tmp_path / "branches.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in src.state_dict().items()}}, path)
    dst.load_cleanunet_weights(str(path))
    dst.load_cleanspecnet_weights(str(path))
    x = torch.randn(1, 1, 1000) * 0.1
    with torch.no_grad():
        assert torch.allclose(dst.eval()(x)["wav"], src.eval()(x)["wav"], atol=1e-5)
