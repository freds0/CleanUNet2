"""Data, augmentation, checkpoint round-trip and Lightning module tests."""

import os

import torch

from augment import Augmenter
from cleanunet.cleanunet2 import CleanUNet2
from config_utils import apply_overrides, load_config
from conftest import DATA_DIR, ROOT, requires_data


def test_augmenter_keeps_mixture_consistent():
    torch.manual_seed(0)
    clean = torch.randn(4, 1, 8000)
    noise = torch.randn(4, 1, 8000) * 0.3
    aug = Augmenter(remix_p=1.0, bandmask_p=1.0, shift_p=1.0, shift_max=100, gain_p=1.0, snr_p=1.0)
    noisy2, clean2 = aug(clean + noise, clean)
    assert noisy2.shape == clean2.shape == clean.shape
    assert not torch.allclose(clean2, clean)
    identity = Augmenter()
    n, c = identity(clean + noise, clean)
    assert torch.equal(n, clean + noise) and torch.equal(c, clean)


def test_bandmask_removes_energy_in_band():
    aug = Augmenter(bandmask_p=1.0)
    t = torch.arange(16000) / 16000
    x = torch.sin(2 * torch.pi * 1000 * t).view(1, 1, -1)
    aug.band_edges_hz = [800.0, 1200.0, 1300.0]       # every possible band contains 1 kHz
    aug.bandmask_bands, aug.bandmask_maxwidth = 3, 1.0
    out = aug._bandmask(torch.cat([x, x], dim=1))
    assert out[..., 1000:-1000].pow(2).mean() < 0.05 * x.pow(2).mean()


def test_config_inheritance_and_overrides():
    cfg = load_config(os.path.join(ROOT, "configs/train_stageA_cleanunet_mamba.yaml"), ["data.batch_size=4"])
    assert cfg["data"]["batch_size"] == 4
    assert cfg["loss"]["mode"] == "waveform"
    assert cfg["model"]["architecture"]["use_spec_branch"] is False
    assert cfg["model"]["architecture"]["cleanunet_params"]["bottleneck"]["type"] == "mamba"
    cfg = load_config(os.path.join(ROOT, "configs/model/../train.yaml"))
    assert cfg["model"]["architecture"]["cleanspecnet_params"]["backbone"]["type"] == "mamba"
    assert apply_overrides({}, ["a.b=[1, 2]"]) == {"a": {"b": [1, 2]}}


def test_component_checkpoint_roundtrip(tmp_path):
    """Stage C loads stage-A / stage-B Lightning checkpoints by submodule prefix."""
    kw = dict(cleanunet_params=dict(channels_H=8, max_H=16, encoder_n_layers=3,
                                    bottleneck={"type": "transformer", "n_layers": 1, "d_model": 16,
                                                "n_head": 2, "d_inner": 32}),
              cleanspecnet_params={"conv_hidden_dim": 8, "hidden_dim": 16, "num_attention_layers": 1, "num_heads": 2},
              spectrogram={"n_fft": 64, "hop_length": 16, "win_length": 64})
    src, dst = CleanUNet2(**kw), CleanUNet2(**kw)
    path = tmp_path / "stage.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in src.state_dict().items()}}, path)
    dst.load_cleanunet_weights(str(path))
    dst.load_cleanspecnet_weights(str(path))
    for k, v in src.state_dict().items():
        if k.startswith(("clean_unet.", "clean_spec_net.")):     # glue layers are not part of the components
            assert torch.equal(v, dst.state_dict()[k]), k


@requires_data
def test_dataset_same_gain_and_crops():
    from dataset import PairDataset
    ds = PairDataset(DATA_DIR, os.path.join(ROOT, "filelists/train.csv"), segment_size=32768, max_files=5)
    noisy, clean, file_id = ds[0]
    assert noisy.shape == clean.shape == (1, 32768)
    full = PairDataset(DATA_DIR, os.path.join(ROOT, "filelists/val.csv"), max_files=2)
    noisy, clean, _ = full[0]
    assert abs(noisy.abs().max().item() - 1.0) < 1e-5    # scaled by the noisy peak ...
    assert clean.abs().max() < noisy.abs().max() * 2      # ... and the clean target by the same factor


def test_lightning_module_steps_and_optimizer():
    from lightning_modules.cleanunet_module import CleanUNetLightningModule
    arch = dict(cleanunet_params=dict(channels_H=8, max_H=16, encoder_n_layers=3,
                                      bottleneck={"type": "mamba", "n_layers": 1, "d_model": 16, "d_state": 4,
                                                  "expand": 2, "backend": "torch"}),
                cleanspecnet_params={"conv_hidden_dim": 8, "hidden_dim": 16,
                                     "backbone": {"type": "mamba", "n_layers": 1, "d_state": 4, "backend": "torch"}},
                spectrogram={"n_fft": 64, "hop_length": 16, "win_length": 64},
                conditioning_type="film", conditioning_zero_init=True)
    for mode in ("hybrid", "waveform", "spec"):
        module = CleanUNetLightningModule({"model": {"architecture": arch},
                                           "loss": {"mode": mode, "weight_phase": 0.1},
                                           "optim": {"schedule": "constant", "weight_decay": 0.01}})
        batch = (torch.randn(2, 1, 4096) * 0.1, torch.randn(2, 1, 4096) * 0.1, ["a", "b"])
        loss = module.training_step(batch, 0)
        loss.backward()
        assert torch.isfinite(loss)
    opt = module.configure_optimizers()
    no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    a_log = [p for n, p in module.model.named_parameters() if n.endswith("A_log")]
    assert any(any(p is q for q in g["params"]) for g in no_decay for p in a_log)
