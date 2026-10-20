"""Tests for queue Z of IMPROVEMENT_PLAN.md (section 14.11): unidirectional Mamba weights loaded into a bidirectional
bottleneck (Z6), the blend training mode of the spectral branch (Z5) and the band-limited scores of the output
ensembles (Z1)."""

import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from conftest import DATA_DIR, ROOT, requires_data
from test_v_experiments import MAMBA, SPEC, STFT, UNET, save_lightning, wave_arch
from test_w_experiments import _module_with_loss, _Pairs


def bidir_wave_arch():
    """The small waveform-only model of test_v_experiments with a bidirectional Mamba bottleneck."""
    return dict(cleanunet_params={**UNET, "bottleneck": {**MAMBA, "bidirectional": True}}, use_spec_branch=False)


def blend_arch():
    """Small model with a non-causal spectral branch and no conditioning path (the layout of loss.mode blend)."""
    spec = {**SPEC, "causal": False, "backbone": {**SPEC["backbone"], "bidirectional": True}}
    return dict(cleanunet_params=UNET, cleanspecnet_params=spec, spectrogram=STFT, conditioning_type="none")


def test_unidirectional_weights_load_into_a_bidirectional_bottleneck(tmp_path):
    from cleanunet.cleanunet2 import CleanUNet2
    uni, path = save_lightning(tmp_path, wave_arch(), seed=1)
    bi = CleanUNet2(**bidir_wave_arch())
    bi.load_cleanunet_weights(path)
    layers = bi.clean_unet.tsfm_Mamba_layers
    for i, layer in enumerate(layers):
        fwd, bwd = layer.mixer.fwd, layer.mixer.bwd
        src = uni.clean_unet.tsfm_Mamba_layers[i].mixer
        assert torch.equal(fwd.in_proj.weight, src.in_proj.weight) and torch.equal(bwd.in_proj.weight, src.in_proj.weight)
        assert torch.equal(fwd.out_proj.weight, src.out_proj.weight) and bwd.out_proj.weight.abs().sum() == 0
    x = torch.randn(1, 1, 2048) * 0.1
    uni.eval()
    bi.eval()
    with torch.no_grad():
        assert torch.allclose(bi(x)["wav"], uni(x)["wav"], atol=1e-6)        # it starts as the causal model

        # once the backward copy is trained, later bottleneck tokens change the earlier outputs
        from cleanunet.sequence import run_mamba_layers
        for layer in layers:
            layer.mixer.bwd.out_proj.weight.normal_(0, 0.1)
        h = torch.randn(1, 64, 16)
        h2 = h.clone()
        h2[:, 32:] = torch.randn(1, 32, 16)
        run_bi = lambda t: run_mamba_layers(layers, bi.clean_unet.norm_f, t)[:, :32]  # noqa: E731
        run_uni = lambda t: run_mamba_layers(uni.clean_unet.tsfm_Mamba_layers, uni.clean_unet.norm_f, t)[:, :32]  # noqa: E731
        assert (run_bi(h) - run_bi(h2)).abs().max() > 1e-4
        assert torch.equal(run_uni(h), run_uni(h2))


def test_bidirectional_checkpoints_load_unchanged(tmp_path):
    from cleanunet.cleanunet2 import CleanUNet2
    bi, path = save_lightning(tmp_path, bidir_wave_arch(), seed=2)
    other = CleanUNet2(**bidir_wave_arch())
    other.load_cleanunet_weights(path)
    for (n, a), b in zip(bi.state_dict().items(), other.state_dict().values()):
        assert torch.equal(a, b), n


def test_blend_mode_trains_the_spectral_branch_through_the_blend():
    from cleanunet.blend import blend
    module = _module_with_loss(blend_arch(), {"mode": "blend", "blend_w": 0.25}, model={"architecture": blend_arch(),
                                                                                      "train_cleanunet": False})
    noisy, clean = torch.randn(2, 1, 2048) * 0.1, torch.randn(2, 1, 2048) * 0.1
    module.eval()
    with torch.no_grad():
        total, out, logs = module._compute_losses(noisy, clean)
        y = module.model(noisy)["wav"]                     # waveform branch alone (no conditioning path)
        assert torch.allclose(out["wav"], blend(y, noisy, module.model, 0.25), atol=1e-6)
    assert "spec_loss" in logs and "waveform_loss" in logs

    module.train()
    total, out, logs = module._compute_losses(noisy, clean)
    total.backward()
    unet = [p.grad for n, p in module.model.named_parameters() if n.startswith("clean_unet.")]
    spec = [p.grad for n, p in module.model.named_parameters() if n.startswith("clean_spec_net.")]
    assert all(g is None for g in unet)
    assert any(g is not None and g.abs().sum() > 0 for g in spec)


def test_blend_mode_needs_a_frozen_waveform_branch_and_a_spectral_branch():
    with pytest.raises(ValueError):
        _module_with_loss(blend_arch(), {"mode": "blend"})
    with pytest.raises(ValueError):
        _module_with_loss(wave_arch(), {"mode": "blend"}, model={"architecture": wave_arch(), "train_cleanunet": False})


def test_blend_mode_runs_with_the_metric_discriminator():
    import pytorch_lightning as pl
    module = _module_with_loss(blend_arch(), {"mode": "blend", "blend_w": 0.25,
                                             "metric_gan": {"weight": 0.5, "warmup_steps": 1, "pesq_workers": 1,
                                                            "highpass_hz": 100,
                                                            "disc": {"channels": 4, "n_fft": 64, "hop_length": 16}}},
                               model={"architecture": blend_arch(), "train_cleanunet": False},
                               trainer={"gradient_clip_val": 10.0})
    module.pesq_fn = lambda ref, est, pool=None: torch.tensor([0.6, 0.5])
    unet0 = [p.detach().clone() for p in module.model.clean_unet.parameters()]
    spec0 = [p.detach().clone() for p in module.model.clean_spec_net.parameters()]
    trainer = pl.Trainer(accelerator="cpu", max_steps=6, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                         num_sanity_val_steps=0)
    trainer.fit(module, torch.utils.data.DataLoader(_Pairs(), batch_size=2))
    assert int(module.mgan_step) == 3
    assert all(torch.equal(a, b) for a, b in zip(unet0, module.model.clean_unet.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(spec0, module.model.clean_spec_net.parameters()))


@requires_data
def test_ensemble_outputs_reports_band_limited_pesq(tmp_path):
    pairs = ["test/clean/p232_001.flac|test/noisy/p232_001.flac", "test/clean/p232_002.flac|test/noisy/p232_002.flac"]
    filelist = tmp_path / "list.csv"
    filelist.write_text("\n".join(pairs) + "\n")
    a = tmp_path / "a"
    a.mkdir()
    for line in pairs:
        clean_rel = line.split("|")[0]
        sf.write(a / os.path.basename(clean_rel), 0.9 * sf.read(os.path.join(DATA_DIR, clean_rel))[0], 16000)
    out = tmp_path / "ens"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts/ensemble_outputs.py"), "--inputs", str(a),
                    "--filelist", str(filelist), "--out_dir", str(out), "--fast", "--band_limited", "--workers", "1"],
                   check=True, cwd=ROOT, capture_output=True)
    import json
    s = json.load(open(out / "summary.json"))["summary"]
    assert s["n_files"] == 2 and np.isfinite(s["pesq_wb_hp"]) and s["pesq_wb_hp"] > 4.0
