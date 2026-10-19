"""Tests for the external-noise augmentation (noise_bank.py, PairDataset ext_noise)."""

import os

import numpy as np
import pytest
import soundfile as sf

from conftest import DATA_DIR, requires_data
from dataset import PairDataset
from noise_bank import NoiseBank, active_speech_power, mix_at_snr

FS = 16000


def fake_musan(root, seconds=2.0):
    """A MUSAN-like tree: two noise files (one too short), three talkers, one music file."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * FS)) / FS
    files = {"noise/free-sound/n1.wav": 0.1 * rng.standard_normal(len(t)),
             "noise/free-sound/short.wav": 0.1 * rng.standard_normal(FS // 2),
             "music/fma/m1.wav": 0.2 * np.sin(2 * np.pi * 220 * t)}
    for k, f0 in enumerate((120, 180, 240)):
        files[f"speech/librivox/s{k}.wav"] = 0.3 * np.sin(2 * np.pi * f0 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    for rel, x in files.items():
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        sf.write(os.path.join(root, rel), x.astype(np.float32), FS)
    return str(root)


def test_noise_bank_draws_every_kind_at_the_requested_length(tmp_path):
    root = fake_musan(tmp_path)
    bank = NoiseBank(root, weights={"noise": 1, "babble": 1, "music": 1}, babble_talkers=(2, 3))
    assert [p for p, _ in bank.files["noise"]] == ["noise/free-sound/n1.wav"]     # the 0.5 s file is left out
    assert os.path.exists(os.path.join(root, f"index_{FS}.json"))
    kinds = set()
    for _ in range(60):
        x, kind = bank.sample(50000)        # longer than any file: excerpts are joined
        kinds.add(kind)
        assert x.shape == (50000,) and x.dtype == np.float32 and np.abs(x).max() > 0
    assert kinds == {"noise", "babble", "music"}
    # the cached index is read back
    assert NoiseBank(root, weights={"music": 1}).files == bank.files


def test_noise_bank_rejects_a_missing_source(tmp_path):
    root = fake_musan(tmp_path)
    os.rename(os.path.join(root, "music"), os.path.join(root, "music_off"))
    with pytest.raises(ValueError):
        NoiseBank(root, weights={"music": 1})
    NoiseBank(root, weights={"noise": 1, "music": 0})      # a kind with weight 0 needs no files


def test_mix_at_snr_sets_the_p56_snr():
    rng = np.random.default_rng(1)
    t = np.arange(3 * FS) / FS
    clean = 0.1 * np.sin(2 * np.pi * 300 * t) * (t % 1.0 < 0.6)
    noise = rng.standard_normal(len(t))
    for snr in (-5.0, 0.0, 12.5):
        noisy = mix_at_snr(clean, noise, snr, FS)
        measured = 10 * np.log10(active_speech_power(clean, FS) / np.mean((noisy - clean) ** 2))
        assert abs(measured - snr) < 1e-6
    assert np.array_equal(mix_at_snr(np.zeros(100), noise[:100], 0.0, FS), np.zeros(100))


@requires_data
def test_pair_dataset_replaces_the_noise_of_training_crops(tmp_path):
    root = fake_musan(tmp_path / "musan")
    filelist = tmp_path / "pairs.csv"
    with open("filelists/val.csv") as src, open(filelist, "w") as dst:
        dst.writelines([next(src) for _ in range(4)])
    base = PairDataset(DATA_DIR, str(filelist), segment_size=16000)
    ext = PairDataset(DATA_DIR, str(filelist), segment_size=16000,
                      ext_noise={"dir": root, "p": 1.0, "snr_db": [5, 5], "weights": {"noise": 1}})
    for i in range(4):
        noisy, clean, _ = ext[i]
        assert noisy.shape == clean.shape == (1, 16000)
        assert abs(float(noisy.abs().max()) - 1.0) < 1e-4          # rescaled so that the mixture peaks at 1
        c, n = clean.flatten().numpy().astype(np.float64), (noisy - clean).flatten().numpy().astype(np.float64)
        # P.56 thresholds are absolute levels, so measuring after the final rescaling moves the SNR a little
        if active_speech_power(c, FS) > 1e-6:
            assert abs(10 * np.log10(active_speech_power(c, FS) / np.mean(n ** 2)) - 5.0) < 0.5
    with pytest.raises(ValueError):
        PairDataset(DATA_DIR, str(filelist), ext_noise={"dir": root, "p": 0.5})    # no segment_size
    assert base.noise_bank is None
