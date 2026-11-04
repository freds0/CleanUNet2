"""
Multi-corpus training set of the all-data model (IMPROVEMENT_PLAN.md, 15).

Every example is a random training crop of `segment_size` samples at `sample_rate`, drawn from one component:
  - a corpus of pre-mixed (clean, noisy) pairs, `pairs_<name>.tsv` (the same crop of both files), or
  - an on-the-fly mixture: clean speech from `speech_<name>.tsv`, optionally convolved with a room impulse response
    (`rir_<name>.tsv`, aligned on its direct path; the target keeps the reverberation, so the task stays denoising),
    plus noise from `noise_<name>.tsv` or babble (several talkers of a speech list, each with a random sign) at an SNR
    drawn uniformly from `snr_db` against the P.56 active speech level (noise_bank.mix_at_snr).
The lists are written by scripts/datasets/make_training_lists.py (absolute path, sample rate, frames; pairs have the
clean path first). Files at other rates are read at their own rate (only the samples of the crop) and resampled.
Each example is scaled so that its noisy signal peaks at 1, the clean signal by the same gain.

With `fill`, a file shorter than the crop is followed by other files of the same list (pairs: each scaled by its
noisy peak; speech: each scaled to the RMS of the first) joined with a linear crossfade of `crossfade` samples applied
identically to clean and noisy, so long crops (IMPROVEMENT_PLAN.md, 15) contain speech instead of zero padding.
Noise is always made of consecutive excerpts.

The dataset has no order: `epoch_size` fixes the number of examples per epoch and the index is ignored. A file that
cannot be read (or gives a silent or non-finite crop) is replaced by another draw. With `worker_stderr`, the
DataLoader workers write their standard error (libmpg123 warns about many FMA files) to that file instead.
"""

import csv
import math
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torchaudio
from scipy.signal import fftconvolve

from noise_bank import mix_at_snr


def read_list(path):
    """Rows of a TSV list: tuples whose last two fields (sample rate, frames) are ints."""
    with open(path, newline="") as fh:
        return [tuple(r[:-2]) + (int(r[-2]), int(r[-1])) for r in csv.reader(fh, delimiter="\t") if r]


class MixtureDataset(torch.utils.data.Dataset):
    """Yields (noisy (1, L), clean (1, L), component name).

    Args:
        lists_dir: directory of the TSV lists.
        pairs: {name: weight} of the pre-mixed pair corpora (pairs_<name>.tsv).
        mix: on-the-fly mixtures, {weight, speech: {name: w}, noise: {name: w}, snr_db: [low, high], rir_p,
            rir: {name: w}, babble_speech: name of a speech list, babble_talkers: [min, max], max_rir_seconds}.
            The noise name `babble` draws babble instead of a noise list.
        segment_size: crop length in samples at `sample_rate`.
        epoch_size: examples per epoch.
        fill / crossfade: fill crops longer than a file with further files (see above), crossfade in samples.
        worker_stderr: file that receives the standard error of the DataLoader workers (appended).
    """

    def __init__(self, lists_dir, pairs=None, mix=None, segment_size=32768, sample_rate=16000, epoch_size=160000,
                 worker_stderr=None, fill=False, crossfade=160, **unused):
        super().__init__()
        if unused:
            raise ValueError(f"unknown mixture options: {sorted(unused)}")
        self.segment_size = int(segment_size)
        self.sample_rate = int(sample_rate)
        self.epoch_size = int(epoch_size)
        self.fill = bool(fill)
        self.crossfade = int(crossfade)
        load = lambda kind, name: read_list(os.path.join(lists_dir, f"{kind}_{name}.tsv"))  # noqa: E731
        self.components, weights = [], []
        for name, w in (pairs or {}).items():
            if w > 0:
                self.components.append(("pairs", name))
                weights.append(float(w))
        self.pairs = {name: load("pairs", name) for kind, name in self.components}
        mix = dict(mix or {})
        if float(mix.get("weight", 0)) > 0:
            self.components.append(("mix", "mix"))
            weights.append(float(mix["weight"]))
            self.speech_w = {k: float(v) for k, v in mix["speech"].items() if v > 0}
            self.noise_w = {k: float(v) for k, v in mix["noise"].items() if v > 0}
            self.rir_w = {k: float(v) for k, v in (mix.get("rir") or {}).items() if v > 0}
            self.babble_speech = mix.get("babble_speech", "librispeech")
            needed = set(self.speech_w) | ({self.babble_speech} if "babble" in self.noise_w else set())
            self.speech = {k: load("speech", k) for k in needed}
            self.noise = {k: load("noise", k) for k in self.noise_w if k != "babble"}
            self.rir = {k: load("rir", k) for k in self.rir_w}
            self.snr_db = tuple(float(v) for v in mix.get("snr_db", (-5.0, 20.0)))
            self.rir_p = float(mix.get("rir_p", 0.0)) if self.rir_w else 0.0
            self.babble_talkers = tuple(int(v) for v in mix.get("babble_talkers", (3, 7)))
            self.max_rir = int(float(mix.get("max_rir_seconds", 2.0)) * self.sample_rate)
        if not self.components:
            raise ValueError("MixtureDataset needs at least one component with a positive weight")
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self._resamplers = {}
        self._failures = 0
        self.worker_stderr = worker_stderr
        self._redirected = False

    def __len__(self):
        return self.epoch_size

    # ---- reading ---------------------------------------------------------------------------------------------------
    def _resample(self, x, sr):
        if sr == self.sample_rate:
            return x
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
        return self._resamplers[sr](torch.from_numpy(x)).numpy()

    def _read(self, path, start, frames, sr, channel=None):
        """`frames` samples from `start`, mono (mean of the channels, or one channel), resampled."""
        x, _ = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
        x = x[:, channel] if channel is not None else x.mean(axis=1)
        return self._resample(np.ascontiguousarray(x), sr)

    def _fit(self, x, length):
        return x[:length] if len(x) >= length else np.pad(x, (0, length - len(x)))

    def _excerpt(self, entry, length):
        """A random excerpt of at most `length` output samples of a list entry (path, sr, frames)."""
        path, sr, frames = entry
        n = min(frames, math.ceil(length * sr / self.sample_rate))
        start = random.randint(0, frames - n)
        return self._read(path, start, n, sr)[:length]

    def _crop(self, entry, length):
        """A random excerpt of `length` output samples of a list entry; zero-padded if the file is shorter."""
        return self._fit(self._excerpt(entry, length), length)

    def _join(self, a, b):
        """Concatenate with a linear crossfade of `self.crossfade` samples."""
        n = min(self.crossfade, len(a), len(b))
        if n == 0:
            return np.concatenate([a, b])
        ramp = np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
        return np.concatenate([a[:-n], a[-n:] * (1 - ramp) + b[:n] * ramp, b[n:]])

    def _filled(self, draw, length):
        """Join the pieces returned by draw(samples still needed) until `length` samples (at most 64 pieces)."""
        pieces = draw(length)
        for _ in range(63):
            if not self.fill or len(pieces[0]) >= length:
                break
            nxt = draw(length - len(pieces[0]) + self.crossfade)
            pieces = tuple(self._join(a, b) for a, b in zip(pieces, nxt))
        return tuple(self._fit(x, length) for x in pieces)

    def _loop(self, rows, length):
        """`length` samples made of random excerpts of random files of a list, one after the other."""
        out, pos = np.zeros(length, dtype=np.float32), 0
        while pos < length:
            path, sr, frames = random.choice(rows)
            need = min(length - pos, math.ceil(frames * self.sample_rate / sr))
            x = self._crop((path, sr, frames), need)
            out[pos:pos + need] = x
            pos += need
        return out

    # ---- components ------------------------------------------------------------------------------------------------
    def _pair(self, name):
        def draw(length):                      # the same excerpt of both files, scaled by its noisy peak
            clean_path, noisy_path, sr, frames = random.choice(self.pairs[name])
            n = min(frames, math.ceil(length * sr / self.sample_rate))
            start = random.randint(0, frames - n)
            clean = self._read(clean_path, start, n, sr)[:length]
            noisy = self._read(noisy_path, start, n, sr)[:length]
            m = min(len(clean), len(noisy))
            gain = 1.0 / (float(np.abs(noisy[:m]).max()) + 1e-8)
            return clean[:m] * gain, noisy[:m] * gain
        return self._filled(draw, self.segment_size)

    def _speech(self):
        rows = self.speech[random.choices(list(self.speech_w), weights=list(self.speech_w.values()))[0]]
        level = []

        def draw(length):                      # later pieces get the RMS of the first one
            x = self._excerpt(random.choice(rows), length)
            rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            if not level:
                level.append(rms)
            elif rms > 1e-6:
                x = x * (level[0] / rms)
            return (x,)
        for _ in range(5):                     # avoid crops that are (almost) all silence
            level.clear()
            x, = self._filled(draw, self.segment_size)
            if np.sqrt(np.mean(x ** 2)) > 1e-3:
                break
        return x

    def _noise(self):
        kind = random.choices(list(self.noise_w), weights=list(self.noise_w.values()))[0]
        if kind != "babble":
            return self._loop(self.noise[kind], self.segment_size)
        rows = self.speech[self.babble_speech]
        talkers = [self._loop(rows, self.segment_size) for _ in range(random.randint(*self.babble_talkers))]
        return sum(random.choice((-1.0, 1.0)) * t / (np.sqrt(np.mean(t ** 2)) + 1e-8) for t in talkers)

    def _reverb(self, speech):
        rows = self.rir[random.choices(list(self.rir_w), weights=list(self.rir_w.values()))[0]]
        path, sr, frames = random.choice(rows)
        rir = self._read(path, 0, min(frames, math.ceil(self.max_rir * sr / self.sample_rate)), sr, channel=0)
        rir = rir[int(np.argmax(np.abs(rir))):]          # direct path at lag 0: the target stays aligned
        if not np.any(rir):
            return speech
        return fftconvolve(speech, rir / np.abs(rir[0]))[:len(speech)].astype(np.float32)

    def _mix(self):
        clean = self._speech()
        if self.rir_p > 0 and random.random() < self.rir_p:
            clean = self._reverb(clean)
        noisy = mix_at_snr(clean, self._noise(), random.uniform(*self.snr_db), self.sample_rate)
        return clean, noisy.astype(np.float32)

    def _redirect_stderr(self):
        """In a DataLoader worker, send file descriptor 2 (C libraries included) to `worker_stderr`."""
        self._redirected = True
        if self.worker_stderr and torch.utils.data.get_worker_info() is not None:
            sys.stderr.flush()
            fd = os.open(self.worker_stderr, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(fd, 2)
            os.close(fd)

    def __getitem__(self, index):
        if not self._redirected:
            self._redirect_stderr()
        for attempt in range(10):
            kind, name = random.choices(self.components, weights=self.weights)[0]
            try:
                clean, noisy = self._pair(name) if kind == "pairs" else self._mix()
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop a long run
                self._failures += 1
                if self._failures <= 20:
                    print(f"[MixtureDataset] {name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue
            peak = float(np.abs(noisy).max())
            if np.isfinite(peak) and peak > 1e-4 and np.all(np.isfinite(clean)):
                gain = 1.0 / peak
                return (torch.from_numpy(noisy * gain).float().unsqueeze(0),
                        torch.from_numpy(clean * gain).float().unsqueeze(0), name)
        raise RuntimeError("MixtureDataset: 10 draws in a row failed")
