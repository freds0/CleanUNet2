"""Tests for the unseen-noise validation set (scripts/make_val_unseen.py) and the phase-U helpers."""

import os
import sys

import numpy as np

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_val_unseen import snr_db  # noqa: E402
from noise_bank import active_speech_power  # noqa: E402
from run_ablation import spearman  # noqa: E402


def bursts(fs=16000, cycles=3, amplitude=0.1):
    """1 s of a 440 Hz tone followed by 1 s of silence, `cycles` times."""
    t = np.arange(fs) / fs
    tone = amplitude * np.sin(2 * np.pi * 440 * t)
    return np.concatenate([np.concatenate([tone, np.zeros(fs)]) for _ in range(cycles)])


def test_active_speech_power_ignores_pauses():
    x, fs = bursts(), 16000
    tone_power = 0.1 ** 2 / 2
    active_db = 10 * np.log10(active_speech_power(x, fs) / tone_power)
    mean_db = 10 * np.log10(np.mean(x ** 2) / tone_power)
    # a pause only counts while the 30 ms envelope decays to the threshold (~0.1 s) and for the 200 ms hangover
    # after it: about -1.1 dB, against -3 dB for the mean power
    assert -1.5 < active_db < -0.5, active_db
    assert mean_db < -2.9


def test_snr_uses_the_active_speech_level():
    x, fs = bursts(), 16000
    noise = np.random.default_rng(0).standard_normal(len(x))
    gain = np.sqrt(active_speech_power(x, fs) / (np.mean(noise ** 2) * 10 ** (7.5 / 10)))
    assert abs(snr_db(x, gain * noise, fs, p56=True) - 7.5) < 1e-6
    assert snr_db(x, gain * noise, fs, p56=False) < 7.5 - 1.5


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(spearman([1, 2], [2, 1]))
