#!/usr/bin/env python
"""Clean 16 kHz utterances of the VCTK speakers that VoiceBank-DEMAND does not use, processed like its clean files, for
an evaluation set with many unseen speakers (IMPROVEMENT_PLAN.md, 14.9). The test set and val_unseen have only two
speakers each (14.8.2).

Reads the extracted VCTK-Corpus-0.92 (Edinburgh DataShare 10283/3443, CC BY 4.0). The corpus is downloaded data, so
run this script with ``python -I``. VoiceBank-DEMAND clean files are peak-normalized (~0.49 at 16 kHz); --calibrate
checks which microphone and which level reproduce them, on the test speakers p232 and p257.

Usage:
    python -I scripts/prepare_vctk_clean.py --calibrate
    python -I scripts/prepare_vctk_clean.py            # <data_dir>/<name>/clean/*.flac, filelists/<name>_clean.csv
    python scripts/make_val_unseen.py --filelist filelists/<name>_clean.csv --name <name> --seed 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load16(path):
    """Read a 48 kHz VCTK file, downmix to mono and resample to 16 kHz."""
    x, fs = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    if fs != 48000:
        sys.exit(f"[ERROR] {path}: {fs} Hz, expected 48000")
    return resample_poly(x, 1, 3)


def vb_speakers(data_dir):
    """Speaker IDs used by VoiceBank-DEMAND (train and test)."""
    return {os.path.basename(f)[:4] for split in ("train", "test")
            for f in glob.glob(os.path.join(data_dir, split, "clean", "*.flac"))}


def align(ref, x):
    """Lag of x inside ref (ref[lag + n] ~ x[n]), correlation and least-squares gain on the overlap."""
    c = fftconvolve(ref, x[::-1], mode="full")
    lag = int(np.argmax(np.abs(c))) - (len(x) - 1)
    a, b = max(lag, 0), max(-lag, 0)
    n = min(len(ref) - a, len(x) - b)
    r, y = ref[a:a + n], x[b:b + n]
    return lag, float(np.corrcoef(r, y)[0, 1]), float(np.dot(r, y) / np.dot(y, y))


def calibrate(args):
    """Compare VoiceBank-DEMAND test files with mic1 / mic2 of VCTK: correlation, gain, offset and length."""
    files = sorted(glob.glob(os.path.join(args.data_dir, "test", "clean", "*.flac")))[::10]
    rows = {m: [] for m in ("mic1", "mic2")}
    for f in files:
        fid = os.path.splitext(os.path.basename(f))[0]
        ref, _ = sf.read(f, dtype="float64")
        for mic in rows:
            p = os.path.join(args.vctk_dir, "wav48_silence_trimmed", fid[:4], f"{fid}_{mic}.flac")
            if not os.path.exists(p):
                continue
            x = load16(p)
            lag, corr, gain = align(ref, x)
            rows[mic].append((corr, gain, np.abs(ref).max(), np.abs(x).max() * gain, lag / 16000,
                              (len(ref) - len(x)) / 16000))
    for mic, r in rows.items():
        if not r:
            continue
        r = np.array(r)
        print(f"{mic}: {len(r)} files; corr median {np.median(r[:, 0]):.4f} (min {r[:, 0].min():.4f}); "
              f"gain median {np.median(r[:, 1]):.3f} (5-95% {np.percentile(r[:, 1], 5):.3f}-"
              f"{np.percentile(r[:, 1], 95):.3f}); VB peak {np.median(r[:, 2]):.3f}; "
              f"lag median {np.median(r[:, 4]):+.3f} s (5-95% {np.percentile(r[:, 4], 5):+.3f} to "
              f"{np.percentile(r[:, 4], 95):+.3f}); VB minus VCTK length median {np.median(r[:, 5]):+.3f} s "
              f"(5-95% {np.percentile(r[:, 5], 5):+.3f} to {np.percentile(r[:, 5], 95):+.3f})")


def make(args):
    """Write --per_speaker clean utterances of every unused speaker and the clean filelist."""
    rng = np.random.default_rng(args.seed)
    root = os.path.join(args.vctk_dir, "wav48_silence_trimmed")
    used = vb_speakers(args.data_dir)
    speakers = sorted(s for s in os.listdir(root) if s not in used
                      and glob.glob(os.path.join(root, s, f"*_{args.mic}.flac")))
    out = os.path.join(args.data_dir, args.name, "clean")
    os.makedirs(out, exist_ok=True)
    pairs, n_short = [], 0
    for spk in speakers:
        cand = sorted(glob.glob(os.path.join(root, spk, f"*_{args.mic}.flac")))
        rng.shuffle(cand)
        k = 0
        for p in cand:
            if k == args.per_speaker:
                break
            x = load16(p)
            if not args.min_dur <= len(x) / 16000 <= args.max_dur:
                n_short += 1
                continue
            x = x * (args.peak / np.abs(x).max())
            fid = os.path.basename(p)[:-len(f"_{args.mic}.flac")]
            sf.write(os.path.join(out, f"{fid}.flac"), x, 16000, subtype="PCM_16")
            pairs.append(f"{args.name}/clean/{fid}.flac|{args.name}/clean/{fid}.flac\n")
            k += 1
    with open(os.path.join(REPO, "filelists", f"{args.name}_clean.csv"), "w") as fh:
        fh.writelines(pairs)
    info = os.path.join(args.vctk_dir, "speaker-info.txt")
    if os.path.exists(info):
        keep = [line for line in open(info, errors="replace") if line.split()[:1] and line.split()[0] in speakers]
        with open(os.path.join(args.data_dir, args.name, "speaker-info.txt"), "w") as fh:
            fh.writelines(keep)
    print(f"[INFO] {len(pairs)} files from {len(speakers)} speakers ({args.mic}, peak {args.peak}) in {out}; "
          f"{n_short} candidates skipped for duration")


def main():
    """Parse the arguments and run ``calibrate`` or ``make``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vctk_dir", default="/home/fred/DATASETS/VCTK_dl_20261016/extracted")
    ap.add_argument("--data_dir", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    ap.add_argument("--name", default="vctk_unseen")
    ap.add_argument("--mic", default="mic1")
    ap.add_argument("--per_speaker", type=int, default=10)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=12.0)
    ap.add_argument("--peak", type=float, default=0.493, help="peak of each file (VoiceBank-DEMAND clean: ~0.49)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    calibrate(args) if args.calibrate else make(args)


if __name__ == "__main__":
    main()
