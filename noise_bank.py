"""
External noise for training-data augmentation (step 3 of the unseen-noise study, results/abl/U_*_val_unseen.md).

VoiceBank-DEMAND has ten training noise types, and a spectral branch trained on them alone does not generalize to
new noise. `NoiseBank` draws noise excerpts from another corpus laid out like MUSAN (openslr.org/17, CC BY 4.0):
recordings from `noise/`, babble made of several talkers from `speech/`, and `music/`. DEMAND must stay out of it:
the test set and filelists/val_unseen.csv are made of DEMAND environments.

`mix_at_snr` sets the SNR against the active speech level of ITU-T P.56, the measure VoiceBank-DEMAND used for its
mixtures (scripts/make_val_unseen.py --check_log reproduces the nominal SNRs of its log files with it).
"""

import json
import os
import random

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def active_speech_power(x, fs, nbits=16):
    """Mean power of x over its active samples, ITU-T P.56 method B (envelope thresholds 2^-15 .. 2^-1).

    A sample is active for a threshold when the envelope (two cascaded one-pole smoothers, 30 ms, of |x|) reached
    the threshold within the last 200 ms (hangover). The active level is where the mean power over the active
    samples lies 15.9 dB above the threshold, interpolated in dB between the two thresholds around it."""
    x = np.asarray(x, dtype=np.float64)
    g = np.exp(-1.0 / (fs * 0.03))
    env = lfilter([1 - g], [1, -g], lfilter([1 - g], [1, -g], np.abs(x)))
    hang = int(np.ceil(fs * 0.2))
    thresholds = 2.0 ** np.arange(-15, nbits - 16)
    idx = np.arange(len(x))
    active = np.empty(len(thresholds))
    for j, c in enumerate(thresholds):
        last = np.maximum.accumulate(np.where(env >= c, idx, -hang - 1))
        active[j] = np.count_nonzero(idx - last <= hang)
    energy = float(np.sum(x ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        level = 10 * np.log10(energy / active)        # dB, mean power over the active samples
    delta = level - 20 * np.log10(thresholds)
    for j in range(1, len(thresholds)):
        if active[j] > 0 and delta[j] <= 15.9:
            if not np.isfinite(level[j - 1]) or delta[j - 1] <= 15.9:
                break
            t = (delta[j - 1] - 15.9) / (delta[j - 1] - delta[j])
            return 10 ** ((level[j - 1] + t * (level[j] - level[j - 1])) / 10)
    return energy / max(len(x), 1)     # no crossing (e.g. a very quiet signal): plain mean power


def mix_at_snr(clean, noise, snr_db, fs):
    """clean + noise, the noise scaled to `snr_db` against the P.56 active level of `clean` (1-D numpy arrays).
    A silent clean signal or noise gives the clean signal back."""
    speech = active_speech_power(clean, fs)
    noise_power = float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))
    if speech <= 0 or noise_power <= 0:
        return np.array(clean, copy=True)
    return clean + noise * np.sqrt(speech / (noise_power * 10 ** (snr_db / 10)))


class NoiseBank:
    """Random noise excerpts from <root>/{noise,speech,music}/**/*.wav (the MUSAN layout).

    Args:
        root: corpus root.
        weights: probability of each kind of excerpt: `noise`, `babble` (equal-level sum of several excerpts of
            `speech/`) and `music`; a kind with weight 0 needs no files.
        babble_talkers: (min, max) number of talkers of a babble excerpt.
        random_sign: multiply every talker of a babble excerpt and every noise / music excerpt by a random sign.
            The recordings of a corpus share a polarity, and speech waveforms are asymmetric: without it a model
            trained with MUSAN babble learns to keep the speech of VoiceBank's polarity and to remove the other one,
            and fails on a polarity-inverted input (IMPROVEMENT_PLAN.md, 14.3).
        sample_rate: files at another rate are left out (MUSAN is 16 kHz).
        min_seconds: shorter files are left out.
    The file index (paths and lengths) is cached in <root>/index_<sample_rate>.json.
    """

    SOURCES = {"noise": "noise", "babble": "speech", "music": "music"}

    def __init__(self, root, weights=None, babble_talkers=(3, 7), sample_rate=16000, min_seconds=1.0,
                 random_sign=False):
        self.root = root
        self.random_sign = bool(random_sign)
        self.sample_rate = sample_rate
        self.weights = {k: float(v) for k, v in (weights or {"noise": 0.6, "babble": 0.25, "music": 0.15}).items()}
        unknown = set(self.weights) - set(self.SOURCES)
        if unknown:
            raise ValueError(f"unknown noise kinds {sorted(unknown)}; expected {sorted(self.SOURCES)}")
        self.kinds = [k for k, w in self.weights.items() if w > 0]
        if not self.kinds:
            raise ValueError("NoiseBank needs at least one kind with a positive weight")
        self.babble_talkers = tuple(babble_talkers)
        min_frames = int(min_seconds * sample_rate)
        self.files = {sub: [(p, n) for p, n in entries if n >= min_frames] for sub, entries in self._index().items()}
        for kind in self.kinds:
            if not self.files.get(self.SOURCES[kind]):
                raise ValueError(f"no usable files in {os.path.join(root, self.SOURCES[kind])}")

    def _index(self):
        cache = os.path.join(self.root, f"index_{self.sample_rate}.json")
        if os.path.exists(cache):
            with open(cache) as fh:
                return json.load(fh)
        files = {}
        for sub in sorted(set(self.SOURCES.values())):
            entries = []
            for dirpath, _, names in sorted(os.walk(os.path.join(self.root, sub))):
                for name in sorted(names):
                    if name.endswith(".wav"):
                        path = os.path.join(dirpath, name)
                        info = sf.info(path)
                        if info.samplerate == self.sample_rate:
                            entries.append((os.path.relpath(path, self.root), info.frames))
            files[sub] = entries
        tmp = f"{cache}.{os.getpid()}.tmp"      # several workers or jobs may build the index at the same time
        with open(tmp, "w") as fh:
            json.dump(files, fh)
        os.replace(tmp, cache)
        return files

    def _excerpt(self, sub, length):
        """`length` samples from random files of <root>/<sub>, one after the other when a file is shorter."""
        out = np.zeros(length, dtype=np.float32)
        pos = 0
        while pos < length:
            path, frames = random.choice(self.files[sub])
            take = min(frames, length - pos)
            start = random.randint(0, frames - take)
            x, _ = sf.read(os.path.join(self.root, path), start=start, frames=take, dtype="float32", always_2d=True)
            if len(x) == 0:
                continue
            out[pos:pos + len(x)] = x.mean(axis=1)
            pos += len(x)
        return out

    def sample(self, length):
        """(excerpt of `length` samples, kind)."""
        kind = random.choices(self.kinds, weights=[self.weights[k] for k in self.kinds])[0]
        sign = (lambda: random.choice((-1.0, 1.0))) if self.random_sign else (lambda: 1.0)  # noqa: E731
        if kind == "babble":
            talkers = [self._excerpt("speech", length) for _ in range(random.randint(*self.babble_talkers))]
            noise = sum(sign() * t / (np.sqrt(np.mean(t ** 2)) + 1e-8) for t in talkers)
        else:
            noise = sign() * self._excerpt(self.SOURCES[kind], length)
        return noise.astype(np.float32), kind
