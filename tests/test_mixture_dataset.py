"""Tests for mixture_dataset.py (the multi-corpus training set of the all-data model) on tiny synthetic lists."""

import csv
import os

import numpy as np
import pytest
import soundfile as sf

from mixture_dataset import MixtureDataset

SEG = 4096


def tone(n, sr, f=440.0, amp=0.5):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def write_list(path, rows):
    with open(path, "w", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(rows)


@pytest.fixture
def lists(tmp_path):
    """Lists of every kind: pairs whose noisy file equals the clean one, speech at 48 kHz, noise, a stereo RIR."""
    d = tmp_path / "audio"
    d.mkdir()
    clean = tone(20000, 16000)
    sf.write(d / "c.wav", clean, 16000)
    sf.write(d / "n.wav", clean, 16000)
    sf.write(d / "s48.wav", tone(3 * 48000, 48000, f=300.0), 48000)
    sf.write(d / "noise.wav", np.random.default_rng(0).normal(0, 0.1, 1000).astype(np.float32), 16000)
    rir = np.zeros((800, 2), dtype=np.float32)
    rir[100, 0], rir[300, 0], rir[50, 1] = 1.0, 0.3, 1.0                 # channel 0: direct path at 100
    sf.write(d / "rir.wav", rir, 16000)
    out = tmp_path / "lists"
    out.mkdir()
    write_list(out / "pairs_same.tsv", [(d / "c.wav", d / "n.wav", 16000, 20000)])
    write_list(out / "pairs_missing.tsv", [(d / "x.wav", d / "y.wav", 16000, 20000)])
    write_list(out / "speech_s48.tsv", [(d / "s48.wav", 48000, 3 * 48000)])
    write_list(out / "noise_n.tsv", [(d / "noise.wav", 16000, 1000)])
    write_list(out / "rir_r.tsv", [(d / "rir.wav", 16000, 800)])
    return str(out)


def test_pairs_give_the_same_crop_of_both_files(lists):
    ds = MixtureDataset(lists, pairs={"same": 1.0}, segment_size=SEG, epoch_size=7)
    assert len(ds) == 7
    noisy, clean, name = ds[0]
    assert noisy.shape == clean.shape == (1, SEG) and name == "same"
    assert np.allclose(noisy.numpy(), clean.numpy())
    assert abs(float(noisy.abs().max()) - 1.0) < 1e-5                   # scaled so that the noisy crop peaks at 1


def test_mixtures_resample_reverberate_and_add_noise(lists):
    mix = dict(weight=1.0, speech={"s48": 1.0}, noise={"n": 1.0}, snr_db=[0, 0], rir_p=0.0)
    ds = MixtureDataset(lists, mix=mix, segment_size=SEG, epoch_size=4)
    noisy, clean, name = ds[0]
    assert name == "mix" and noisy.shape == (1, SEG)
    spec = np.abs(np.fft.rfft(clean.numpy()[0]))
    assert abs(np.argmax(spec) * 16000 / SEG - 300.0) < 8               # 48 kHz speech read at 16 kHz
    residual = (noisy - clean).numpy()[0]
    assert 0.5 < np.mean(residual ** 2) / np.mean(clean.numpy() ** 2) < 3    # SNR 0 dB (P.56 vs plain power)

    ds = MixtureDataset(lists, mix=dict(mix, rir={"r": 1.0}, rir_p=1.0), segment_size=SEG, epoch_size=4)
    x = np.zeros(SEG, dtype=np.float32)
    x[10] = 1.0
    y = ds._reverb(x)
    assert y[10] == pytest.approx(1.0) and y[210] == pytest.approx(0.3, abs=1e-4)  # direct path of channel 0
    assert ds[0][0].shape == (1, SEG)


def test_babble_and_unreadable_files(lists, capsys):
    mix = dict(weight=1.0, speech={"s48": 1.0}, noise={"babble": 1.0}, babble_speech="s48", babble_talkers=[2, 2])
    ds = MixtureDataset(lists, pairs={"missing": 1.0}, mix=mix, segment_size=SEG, epoch_size=4)
    for _ in range(5):
        noisy, clean, name = ds[0]
        assert name == "mix" and np.isfinite(noisy.numpy()).all()       # the missing pair is replaced by a mixture
    with pytest.raises(ValueError, match="unknown mixture options"):
        MixtureDataset(lists, pairs={"same": 1.0}, typo=1)


def test_fill_joins_files_instead_of_zero_padding(lists):
    seg = 50000                                                        # longer than the 20,000-sample pair
    ds = MixtureDataset(lists, pairs={"same": 1.0}, segment_size=seg, epoch_size=4, fill=True, crossfade=160)
    noisy, clean, _ = ds[0]
    assert noisy.shape == (1, seg) and np.allclose(noisy.numpy(), clean.numpy())
    assert np.abs(clean.numpy()[0, -2000:]).max() > 0.1                # no zero padding at the end
    mix = dict(weight=1.0, speech={"s48": 1.0}, noise={"n": 1.0}, snr_db=[30, 30])
    ds = MixtureDataset(lists, mix=mix, segment_size=4 * 16000, epoch_size=4, fill=True)
    noisy, clean, _ = ds[0]                                            # 3 s of speech fill a 4 s crop
    tail = clean.numpy()[0, -8000:]
    assert np.sqrt(np.mean(tail ** 2)) > 0.3 * np.sqrt(np.mean(clean.numpy() ** 2))
    no_fill = MixtureDataset(lists, pairs={"same": 1.0}, segment_size=seg, epoch_size=4)
    assert np.abs(no_fill[0][1].numpy()[0, -2000:]).max() == 0.0       # without fill: zero padding
