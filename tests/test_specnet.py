"""Tests for the CleanSpecNet track (IMPROVEMENT_PLAN.md, section 6)."""

import math

import pytest
import torch

from cleanunet.cleanspecnet import CleanSpecNet, cumulative_norm
from cleanunet.cleanunet2 import CleanUNet2
from cleanunet.fusion import frames_to_positions
from cleanunet.subband import SubbandSpecNet
from cleanunet.tf_mamba import TFMambaSpecNet
from losses import anti_wrapping, phase_losses

MAMBA = {"type": "mamba", "n_layers": 1, "d_state": 8, "expand": 2, "backend": "torch"}
F_BINS = 33


def conv1d_net(**kw):
    """A small 1-D CleanSpecNet with a Mamba backbone."""
    return CleanSpecNet(input_channels=F_BINS, conv_hidden_dim=8, hidden_dim=16, num_conv_layers=2,
                        backbone=MAMBA, **kw)


def tf_net(**kw):
    """A small dual-path time-frequency Mamba spectral branch."""
    return TFMambaSpecNet(input_channels=F_BINS, channels=8, n_blocks=2, d_state=8, hidden_dim=16,
                          hidden_subbands=4, backend="torch", **kw)


def subband_net(**kw):
    """A small sub-band spectral branch with a 1-D full-band model."""
    return SubbandSpecNet(input_channels=F_BINS, neighbors=3, d_model=8, n_layers=1, d_state=8, backend="torch",
                          fullband={"conv_hidden_dim": 8, "hidden_dim": 16, "num_conv_layers": 2,
                                    "backbone": MAMBA}, **kw)


BUILDERS = {
    "conv1d": lambda **kw: conv1d_net(**kw),
    "conv1d_linear": lambda **kw: conv1d_net(frontend="linear", **kw),
    "conv1d_cumnorm": lambda **kw: conv1d_net(input_norm="cumulative", **kw),
    "tf_mamba": lambda **kw: tf_net(**kw),
    "tf_mamba_cumnorm": lambda **kw: tf_net(input_norm="cumulative", **kw),
    "subband": lambda **kw: subband_net(**kw),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
@pytest.mark.parametrize("groups,phase", [(1, False), (3, True)])
def test_shapes_and_time_causality(name, groups, phase):
    torch.manual_seed(0)
    net = BUILDERS[name](feature_groups=groups, predict_phase=phase, output_mode="direct").eval()
    T, t0 = 40, 25
    x = torch.rand(2, groups * F_BINS, T)
    y = x.clone()
    y[..., t0:] += torch.rand_like(y[..., t0:])
    with torch.no_grad():
        a, b = net.run(x), net.run(y)
    assert a["spec"].shape == (2, F_BINS, T)
    assert a["hidden"].shape == (2, T, net.hidden_dim)
    assert (a["phase"] is not None) == phase
    if phase:
        assert a["phase"].shape == (2, F_BINS, T) and a["phase"].abs().max() <= math.pi + 1e-5
    for key in ("spec", "hidden", "phase"):
        if a[key] is None:
            continue
        past = (slice(None), slice(0, t0)) if key == "hidden" else (slice(None), slice(None), slice(0, t0))
        assert torch.allclose(a[key][past], b[key][past], atol=1e-5), key
    assert not torch.allclose(a["spec"][..., t0:], b["spec"][..., t0:])


@pytest.mark.parametrize("name", ["conv1d", "conv1d_linear", "tf_mamba", "subband"])
def test_residual_output_starts_as_identity(name):
    net = BUILDERS[name](output_mode="residual").eval()
    x = torch.rand(1, F_BINS, 20)
    with torch.no_grad():
        assert torch.allclose(net(x), x, atol=1e-6)


def test_subband_starts_as_its_fullband_model():
    net = subband_net(output_mode="mask").eval()
    x = torch.rand(1, F_BINS, 20)
    with torch.no_grad():
        assert torch.allclose(net.run(x)["spec"], net.fullband.run(x)["spec"])


@pytest.mark.parametrize("n_freq", [33, 64, 257])
def test_tf_mamba_handles_any_number_of_bins(n_freq):
    net = TFMambaSpecNet(input_channels=n_freq, channels=4, n_blocks=1, d_state=4, hidden_dim=8,
                         hidden_subbands=2, backend="torch", output_mode="mask").eval()
    with torch.no_grad():
        assert net(torch.rand(1, n_freq, 12)).shape == (1, n_freq, 12)


def test_cumulative_norm_is_causal_and_scale_invariant():
    x = torch.rand(1, 5, 30) + 0.1
    y = x.clone()
    y[..., 20:] *= 5
    assert torch.allclose(cumulative_norm(x)[..., :20], cumulative_norm(y)[..., :20])
    assert torch.allclose(cumulative_norm(3 * x), cumulative_norm(x), atol=1e-3)


def test_phase_losses_ignore_wrapping():
    phase = torch.rand(1, 6, 10) * 6 - 3
    assert phase_losses(phase + 2 * math.pi, phase).item() < 1e-5
    assert phase_losses(phase + 0.5, phase).item() > 0.4
    assert torch.allclose(anti_wrapping(torch.tensor([2 * math.pi - 0.1])), torch.tensor([0.1]), atol=1e-6)


def hybrid(spec_params, spectrogram=None, **kw):
    """A small hybrid CleanUNet2 (Mamba bottleneck) with the given spectral-branch parameters."""
    torch.manual_seed(0)
    return CleanUNet2(
        cleanunet_params=dict(channels_H=8, max_H=16, encoder_n_layers=4, kernel_size=4, stride=2,
                              bottleneck={"type": "mamba", "n_layers": 1, "d_model": 16, "d_state": 4,
                                          "expand": 2, "backend": "torch"}),
        cleanspecnet_params=spec_params,
        spectrogram=spectrogram or {"n_fft": 64, "hop_length": 16, "win_length": 64},
        conditioning_type="none", **kw)


SPEC_VARIANTS = {
    "conv1d_mag_phase": {"conv_hidden_dim": 8, "hidden_dim": 16, "num_conv_layers": 2, "backbone": MAMBA,
                         "input_features": "mag_phase", "predict_phase": True, "output_mode": "mask"},
    "tf_mamba": {"architecture": "tf_mamba", "channels": 4, "n_blocks": 1, "d_state": 4, "hidden_dim": 16,
                 "hidden_subbands": 2, "backend": "torch", "predict_phase": True},
    "subband": {"architecture": "subband", "neighbors": 2, "d_model": 8, "n_layers": 1, "d_state": 4,
                "backend": "torch", "fullband": {"conv_hidden_dim": 8, "hidden_dim": 16, "num_conv_layers": 1,
                                                 "backbone": MAMBA}},
}


@pytest.mark.parametrize("name", sorted(SPEC_VARIANTS))
def test_hybrid_with_new_spectral_branches(name):
    model = hybrid(dict(SPEC_VARIANTS[name]), fusion={"points": {"bott_pre": {"op": "film_tanh"}}}).eval()
    x = torch.randn(2, 1, 900) * 0.1
    with torch.no_grad():
        out = model(x)
        off = model(x, fusion_enabled=False)
        res = model.spec_branch(x / model.input_scale(x))
    assert out["wav"].shape == x.shape and torch.allclose(out["wav"], off["wav"], atol=1e-5)
    assert out["spec"].shape == res["spec"].shape == res["noisy_spec"].shape
    assert (out["phase"] is not None) == bool(SPEC_VARIANTS[name].get("predict_phase"))
    wave = model.spec_to_wave(out["spec"], x / out["scale"], x.shape[-1], phase=out["phase"])
    assert wave.shape == x.shape and torch.isfinite(wave).all()


def test_spectral_loss_terms_train():
    from lightning_modules.cleanunet_module import CleanUNetLightningModule
    arch = dict(cleanunet_params=dict(channels_H=8, max_H=16, encoder_n_layers=3,
                                      bottleneck={"type": "mamba", "n_layers": 1, "d_model": 16, "d_state": 4,
                                                  "expand": 2, "backend": "torch"}),
                cleanspecnet_params=dict(SPEC_VARIANTS["conv1d_mag_phase"]),
                spectrogram={"n_fft": 64, "hop_length": 16, "win_length": 64, "compression": "pow"},
                conditioning_type="none", fusion={"points": {"bott_pre": {"op": "film_tanh"}}})
    terms = {"l1": 1.0, "mse": 0.5, "wave": 0.2, "phase": 0.3, "complex": 0.1, "consistency": 0.1}
    for mode in ("spec", "hybrid"):
        module = CleanUNetLightningModule({"model": {"architecture": arch},
                                           "loss": {"mode": mode, "spec_terms": terms},
                                           "optim": {"schedule": "constant"}})
        batch = (torch.randn(2, 1, 2048) * 0.1, torch.randn(2, 1, 2048) * 0.1, ["a", "b"])
        logs = {}
        noisy, clean, _ = batch
        loss, out, logs = module._compute_losses(noisy, clean)
        loss.backward()
        assert torch.isfinite(loss)
        assert {f"spec/{k}" for k in terms} <= set(logs)
        head = module.model.clean_spec_net.phase_layer.weight
        assert head.grad is not None and head.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        CleanUNetLightningModule({"model": {"architecture": arch}, "loss": {"mode": "spec", "spec_terms": {"x": 1}}})


def test_fusion_pools_frames_when_hop_is_smaller_than_the_stride():
    p = torch.arange(8.0).view(1, 1, 8)
    assert frames_to_positions(p, 1, 4, pool=2).flatten().tolist() == [0.5, 2.5, 4.5, 6.5]
    # hop 8 < bottleneck stride 16: two frames per token, identity at initialization
    model = hybrid({"conv_hidden_dim": 8, "hidden_dim": 16, "num_conv_layers": 1, "backbone": MAMBA},
                   spectrogram={"n_fft": 32, "hop_length": 8, "win_length": 32},
                   fusion={"points": {"bott_pre": {"op": "film_tanh"}, "enc_s2": {"op": "film_tanh"}}}).eval()
    assert model.fusion_hub.points["bott_pre"].pool == 2 and model.fusion_hub.points["enc_s2"].repeat == 4
    x = torch.randn(1, 1, 700) * 0.1
    with torch.no_grad():
        for m in model.fusion_hub.points.values():
            m.head.weight.normal_()
        a = model(x)["wav"]
    assert a.shape == x.shape and torch.isfinite(a).all()


def test_concatenated_training_examples(tmp_path):
    import soundfile as sf
    from dataset import PairDataset
    (tmp_path / "clean").mkdir()
    (tmp_path / "noisy").mkdir()
    lines = []
    for i in range(3):
        c = torch.randn(4000).numpy() * 0.1
        n = c + torch.randn(4000).numpy() * 0.05
        sf.write(tmp_path / "clean" / f"{i}.flac", c, 16000)
        sf.write(tmp_path / "noisy" / f"{i}.flac", n, 16000)
        lines.append(f"clean/{i}.flac|noisy/{i}.flac")
    (tmp_path / "list.csv").write_text("\n".join(lines))
    ds = PairDataset(str(tmp_path), str(tmp_path / "list.csv"), segment_size=11000, concat_utterances=3)
    noisy, clean, _ = ds[0]
    assert noisy.shape == clean.shape == (1, 11000)
    assert clean[0, -100:].abs().sum() > 0        # long enough to need the other utterances (no padding)
    joined = ds._join(torch.ones(500), torch.zeros(500))
    assert len(joined) == 1000 - ds.crossfade and joined[0] == 1 and joined[-1] == 0


def test_unreadable_audio_names_the_file(tmp_path):
    import pickle
    from dataset import load_audio
    bad = tmp_path / "broken.flac"
    bad.write_bytes(b"not a flac file" * 10)
    with pytest.raises(RuntimeError, match="broken.flac") as info:
        load_audio(str(bad), 16000, retries=2)
    # a DataLoader worker sends the exception type and message back; both must survive
    again = pickle.loads(pickle.dumps(type(info.value)(str(info.value))))
    assert "broken.flac" in str(again)


def test_safe_atan2_has_a_finite_gradient_at_the_origin():
    from cleanunet.cleanspecnet import safe_atan2
    y = torch.tensor([0.0, 0.3, -1.2], requires_grad=True)
    x = torch.tensor([0.0, -0.5, 2.0], requires_grad=True)
    out = safe_atan2(y, x)
    assert torch.equal(out, torch.atan2(y, x))
    out.sum().backward()
    assert torch.isfinite(y.grad).all() and torch.isfinite(x.grad).all()
    y2, x2 = y.detach()[1:].requires_grad_(), x.detach()[1:].requires_grad_()
    torch.atan2(y2, x2).sum().backward()          # away from the origin: same gradient as atan2
    assert torch.allclose(y.grad[1:], y2.grad, rtol=1e-3) and torch.allclose(x.grad[1:], x2.grad, rtol=1e-3)
