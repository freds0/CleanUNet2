"""Tests for the experiments of section 13 of IMPROVEMENT_PLAN.md: condition dropout (V2), distillation (V6) and
the training-free blend of evaluate.py (V1)."""

import argparse

import pytest
import torch

from lightning_modules.cleanunet_module import CleanUNetLightningModule

MAMBA = {"type": "mamba", "n_layers": 1, "d_model": 16, "d_state": 4, "expand": 2, "backend": "torch"}
UNET = dict(channels_H=8, max_H=16, encoder_n_layers=4, kernel_size=4, stride=2, bottleneck=MAMBA)
SPEC = {"conv_hidden_dim": 8, "hidden_dim": 16, "num_conv_layers": 2, "causal": True,
        "backbone": {"type": "mamba", "n_layers": 1, "d_state": 4, "backend": "torch"}}
STFT = {"n_fft": 64, "hop_length": 16, "win_length": 64}
C3 = {"input_estimate": True, "points": {"bott_pre": {"op": "film_tanh"}, "out": {}}}


def hybrid_arch():
    """Architecture of a small hybrid with the C3 fusion (no waveform conditioning)."""
    return dict(cleanunet_params=UNET, cleanspecnet_params=SPEC, spectrogram=STFT, conditioning_type="none",
                fusion=C3)


def wave_arch():
    """Architecture of a small waveform-only model."""
    return dict(cleanunet_params=UNET, use_spec_branch=False)


def batch():
    """A fixed random (noisy, clean, ids) batch of two 2048-sample examples."""
    torch.manual_seed(0)
    return torch.randn(2, 1, 2048) * 0.1, torch.randn(2, 1, 2048) * 0.1, ["a", "b"]


def make_module(arch, **cfg):
    """A Lightning module for ``arch`` with a constant learning rate and extra config sections."""
    torch.manual_seed(0)
    return CleanUNetLightningModule({"model": {"architecture": arch}, "loss": {"mode": cfg.pop("mode", "hybrid")},
                                     "optim": {"schedule": "constant"}, **cfg})


def fusion_grads(module):
    """Gradients of the fusion parameters (fusion hub, post-filter, input estimate)."""
    return [p.grad for n, p in module.model.named_parameters()
            if n.split(".")[0] in ("fusion_hub", "postfilter", "input_estimate")]


def test_condition_dropout_skips_every_fusion_path_but_trains_the_branches():
    module = make_module(hybrid_arch(), fusion_dropout=0.999999)
    module.train()
    loss = module.training_step(batch(), 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(g is None for g in fusion_grads(module))
    spec = [p.grad for n, p in module.model.named_parameters() if n.startswith("clean_spec_net.")]
    assert any(g is not None and g.abs().sum() > 0 for g in spec)          # the spectral loss still applies
    unet = [p.grad for n, p in module.model.named_parameters() if n.startswith("clean_unet.")]
    assert any(g is not None and g.abs().sum() > 0 for g in unet)

    module = make_module(hybrid_arch(), fusion_dropout=0.0)
    module.train()
    module.training_step(batch(), 0).backward()
    assert any(g is not None and g.abs().sum() > 0 for g in fusion_grads(module))


def test_condition_dropout_rejects_p_one():
    with pytest.raises(ValueError):
        make_module(hybrid_arch(), fusion_dropout=1.0)


def save_lightning(tmp_path, arch, seed):
    """Save a randomly initialized model as a minimal Lightning checkpoint and return its path."""
    from cleanunet.cleanunet2 import CleanUNet2
    torch.manual_seed(seed)
    model = CleanUNet2(**arch)
    path = str(tmp_path / f"teacher{seed}.ckpt")
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                "hyper_parameters": {"model": {"architecture": arch}}}, path)
    return model, path


def test_distillation_adds_the_teacher_term_and_keeps_the_teacher_frozen(tmp_path):
    teacher, path = save_lightning(tmp_path, hybrid_arch(), seed=1)
    plain = make_module(wave_arch(), mode="waveform")
    kd = make_module(wave_arch(), mode="waveform", distill={"teacher_checkpoint": path, "weight": 0.5})
    assert "teacher" not in "".join(kd.state_dict())                       # the student checkpoint stays small
    plain.train()
    kd.train()
    b = batch()
    loss_plain, loss_kd = plain.training_step(b, 0), kd.training_step(b, 0)
    with torch.no_grad():
        target = teacher.eval()(b[0])["wav"]
        student = kd.model(b[0])["wav"]
        expected = 0.5 * kd.criterion(target, student)
    assert torch.allclose(loss_kd - loss_plain, expected, atol=1e-5)
    loss_kd.backward()
    assert all(p.grad is None for p in kd.teacher[0].parameters())


def test_distillation_needs_a_teacher():
    with pytest.raises(ValueError):
        make_module(wave_arch(), mode="waveform", distill={"weight": 1.0})


def test_blend_weight_zero_returns_the_model_output(tmp_path):
    from evaluate import blend, enhance
    from cleanunet.cleanunet2 import CleanUNet2
    spec_model, _ = save_lightning(tmp_path, hybrid_arch(), seed=2)
    torch.manual_seed(3)
    wave = CleanUNet2(**wave_arch()).eval()
    x = torch.randn(1, 1, 2048) * 0.1
    with torch.no_grad():
        y = wave(x)["wav"]
        assert torch.allclose(blend(y, x, spec_model.eval(), 0.0), y, atol=1e-4)
        mixed = blend(y, x, spec_model, 0.5)
    assert mixed.shape == y.shape and not torch.allclose(mixed, y, atol=1e-4)
    args = argparse.Namespace(device="cpu", streaming=False, condition="normal", bf16=False,
                              blend_model=spec_model, blend_w=0.5)
    assert torch.allclose(enhance(wave, x.view(-1), args), mixed.view(-1), atol=1e-6)
