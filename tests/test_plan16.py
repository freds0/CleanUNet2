"""Tests of the code added for the architecture plan (IMPROVEMENT_PLAN.md, 16.4): compressed complex loss, the
per-epoch last checkpoint, the waveform-branch capacity options and the spectral phase in the blend."""
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.demos.boring_classes import BoringModel

from cleanunet.blend import blend_with_spec
from cleanunet.cleanunet import CleanUNet
from cleanunet.cleanunet2 import CleanUNet2
from losses import CompressedComplexLoss
from train import LastEveryEpochCheckpoint

LSTM = {"type": "lstm", "n_layers": 1, "d_model": 64}
SMALL = dict(channels_H=8, max_H=32, encoder_n_layers=4, bottleneck=LSTM)


def test_compressed_complex_loss_sees_phase():
    torch.manual_seed(0)
    loss = CompressedComplexLoss()
    x = torch.randn(2, 8000)
    assert float(loss(x, x)) == pytest.approx(0.0, abs=1e-8)
    # a pure delay leaves the magnitudes unchanged but must be penalized
    shifted = torch.roll(x, 3, dims=-1)
    assert float(loss(shifted, x)) > 0.05


def test_compressed_complex_loss_gradient_is_finite_at_silence():
    loss = CompressedComplexLoss()
    est = torch.zeros(1, 4000, requires_grad=True)
    loss(est, torch.randn(1, 4000)).backward()
    assert torch.isfinite(est.grad).all()


@pytest.mark.parametrize("extra", [{"gated_skips": True},
                                   {"level_dconv": {"levels": [1, 2]}}])
def test_capacity_options_start_near_the_base_model(extra):
    """Gated skips start as the identity; DConv branches start with a LayerScale of 1e-4."""
    torch.manual_seed(0)
    base = CleanUNet(**SMALL).eval()
    new = CleanUNet(**SMALL, **extra).eval()
    missing, unexpected = new.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing
    x = torch.randn(2, 1, 4096) * 0.1
    with torch.no_grad():
        a, b = base(x), new(x)
    assert torch.allclose(a, b, atol=1e-3 * float(a.abs().max()) + 1e-6)


def test_capacity_options_train():
    torch.manual_seed(0)
    m = CleanUNet(**SMALL, level_dconv={"levels": [1, 3], "depth": 2}, gated_skips=True)
    x = torch.randn(2, 1, 4096) * 0.1
    m(x).pow(2).mean().backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if "dconv" in n or "skip_gates" in n)


def _tf_model():
    return CleanUNet2(
        use_spec_branch=True, conditioning_type="none",
        cleanunet_params=SMALL,
        spectrogram={"n_fft": 128, "hop_length": 32, "win_length": 128, "compression": "pow"},
        cleanspecnet_params={"architecture": "tf_mamba", "input_features": "mag_phase", "causal": False,
                             "channels": 8, "full_res_channels": 4, "n_blocks": 1, "freq_downsample": 1,
                             "predict_phase": True, "backend": "torch"}).eval()


def test_blend_with_spectral_phase():
    torch.manual_seed(0)
    spec_model = _tf_model()
    x = torch.randn(1, 1, 4000) * 0.1
    y = x * 0.5
    with torch.no_grad():
        wave_phase = blend_with_spec(y, x, spec_model, 0.5)
        spec_phase = blend_with_spec(y, x, spec_model, 0.5, phase_source="spec")
    assert wave_phase.shape == spec_phase.shape == (1, 1, 4000)
    assert torch.isfinite(spec_phase).all()
    assert not torch.allclose(wave_phase, spec_phase)
    with pytest.raises(ValueError):
        blend_with_spec(y, x, spec_model, 0.5, phase_source="noisy")


class _Pairs(torch.utils.data.Dataset):
    def __len__(self):
        """Eight examples."""
        return 8

    def __getitem__(self, i):
        """A reproducible (noisy, clean, id) example of 2048 samples."""
        g = torch.Generator().manual_seed(i)
        clean = torch.randn(1, 2048, generator=g) * 0.1
        return clean + torch.randn(1, 2048, generator=g) * 0.05, clean, str(i)


def test_metric_gan_trains_the_spectral_branch_through_its_resynthesis():
    import pytorch_lightning as pl
    from lightning_modules.cleanunet_module import CleanUNetLightningModule
    arch = {"use_spec_branch": True, "conditioning_type": "none", "cleanunet_params": SMALL,
            "spectrogram": {"n_fft": 128, "hop_length": 32, "win_length": 128, "compression": "pow"},
            "cleanspecnet_params": {"architecture": "tf_mamba", "input_features": "mag_phase", "causal": False,
                                    "channels": 8, "full_res_channels": 4, "n_blocks": 1, "freq_downsample": 1,
                                    "predict_phase": True, "backend": "torch"}}
    module = CleanUNetLightningModule({
        "model": {"architecture": arch, "train_cleanunet": False},
        "loss": {"mode": "spec", "spec_terms": {"mse": 0.9, "phase": 0.3},
                 "metric_gan": {"weight": 0.5, "warmup_steps": 1, "pesq_workers": 1,
                                "disc": {"channels": 4, "n_fft": 64, "hop_length": 16}}},
        "optim": {"lr": 1e-3}, "trainer": {"gradient_clip_val": 10.0}})
    module.pesq_fn = lambda ref, est, pool=None: torch.full((ref.shape[0],), 0.6)
    spec0 = [p.detach().clone() for p in module.model.clean_spec_net.parameters()]
    wave0 = [p.detach().clone() for p in module.model.clean_unet.parameters()]
    disc0 = [p.detach().clone() for p in module.metric_disc.parameters()]
    trainer = pl.Trainer(accelerator="cpu", max_steps=6, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                         num_sanity_val_steps=0)
    trainer.fit(module, torch.utils.data.DataLoader(_Pairs(), batch_size=2))
    assert int(module.mgan_step) == 3
    assert any(not torch.equal(a, b) for a, b in zip(spec0, module.model.clean_spec_net.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(disc0, module.metric_disc.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(wave0, module.model.clean_unet.parameters()))


class _FallingMetricModel(BoringModel):
    """Logs a validation metric that only gets worse, so that no top-k checkpoint is saved after the first epoch."""

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        self.log("val/pesq", -float(self.current_epoch))
        return out


@pytest.mark.parametrize("val_check_interval", [1.0, 0.5])   # validation at epoch end (2026-11-05 bug) and mid-epoch
def test_last_checkpoint_follows_every_validation(tmp_path, val_check_interval):
    """last.ckpt is the end of training even when the monitored metric stopped improving (IMPROVEMENT_PLAN.md, 16.8)."""
    cb = LastEveryEpochCheckpoint(dirpath=str(tmp_path), monitor="val/pesq", mode="max", save_top_k=1,
                                  save_last=True)
    trainer = pl.Trainer(max_epochs=3, limit_train_batches=4, limit_val_batches=2, val_check_interval=val_check_interval,
                         callbacks=[cb], logger=False, enable_progress_bar=False, enable_model_summary=False,
                         accelerator="cpu", num_sanity_val_steps=0)
    trainer.fit(_FallingMetricModel())
    last = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    assert last["global_step"] == trainer.global_step == 12
    best = torch.load(cb.best_model_path, map_location="cpu", weights_only=False)
    assert best["epoch"] == 0                                  # the top-k file is still the best one
