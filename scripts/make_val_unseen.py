#!/usr/bin/env python
"""Create a validation set whose noise types appear neither in the training set nor in the test set.

The VoiceBank-DEMAND test set mixes unseen speakers with five DEMAND environments that the training set does not
contain (bus, cafe, living, office, psquare) at 2.5 / 7.5 / 12.5 / 17.5 dB. The validation speakers of
filelists/val.csv (p226, p287) come from the training split and share its ten noise types and its 0-15 dB SNRs, so
they over-rate models that rely on a component which does not generalize to new noise (results/abl/phase6_*.md).

This script mixes the clean utterances of the validation speakers with the five DEMAND environments that
VoiceBank-DEMAND leaves out (DWASHING, NFIELD, NPARK, NRIVER, OHALLWAY; channel 1, 16 kHz) at the SNRs of the test
set, with the speech level measured as in ITU-T P.56 (active speech level). Every (noise, SNR) condition gets the same
number of files, and each file takes a random excerpt of its noise recording. Outputs:
    <data_dir>/<name>/noisy/<id>.flac          16-bit FLAC, like the rest of the dataset
    <data_dir>/<name>/clean/<id>.flac          only with --rescale, for the mixtures that had to be scaled down
    <data_dir>/<name>/log_<name>.txt           "<id> <noise> <SNR>", like the VoiceBank-DEMAND log files
    filelists/<name>.csv                       "clean|noisy" relative to data_dir (clean = the original files)

Usage:
    python scripts/make_val_unseen.py --demand_dir /home/fred/DATASETS/DEMAND
    # the same noise types at -5 / 0 dB (stress test of the robustness; IMPROVEMENT_PLAN.md, section 14)
    python scripts/make_val_unseen.py --name val_unseen_low --snrs -5 0 --seed 1 --rescale
    # check which speech level reproduces the nominal SNRs of the official log file
    python scripts/make_val_unseen.py --check_log logfiles/log_testset.txt --check_filelist filelists/test.csv
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noise_bank import active_speech_power  # noqa: E402

DEMAND_UNUSED = {"DWASHING": "washing", "NFIELD": "field", "NPARK": "park", "NRIVER": "river",
                 "OHALLWAY": "hallway"}
TEST_SNRS = (2.5, 7.5, 12.5, 17.5)


def snr_db(clean, noise, fs, p56):
    """SNR of (clean, noise) in dB, with the P.56 active speech level or the plain mean power."""
    speech = active_speech_power(clean, fs) if p56 else float(np.mean(clean ** 2))
    return 10 * np.log10(speech / max(float(np.mean(noise ** 2)), 1e-20))


def read_log(path):
    """{id: (noise label, SNR)} from a VoiceBank-DEMAND log file."""
    with open(path) as fh:
        return {f[0]: (f[1], float(f[2])) for f in (line.split() for line in fh) if len(f) == 3}


def check(args):
    """Measure the SNR of official mixtures with the P.56 speech level and with the plain mean power, and compare
    both to the nominal SNR of the log file."""
    log = read_log(args.check_log)
    pairs = [line.strip().split("|") for line in open(args.check_filelist) if line.strip()]
    rows = []
    for clean_rel, noisy_rel in pairs:
        fid = os.path.splitext(os.path.basename(clean_rel))[0]
        if fid not in log:
            continue
        clean, fs = sf.read(os.path.join(args.data_dir, clean_rel))
        noisy, _ = sf.read(os.path.join(args.data_dir, noisy_rel))
        n = min(len(clean), len(noisy))
        clean, noise = clean[:n], noisy[:n] - clean[:n]
        rows.append((log[fid][1], snr_db(clean, noise, fs, True), snr_db(clean, noise, fs, False)))
    rows = np.array(rows)
    print(f"{len(rows)} files with a nominal SNR in {args.check_log}")
    print("nominal  n    P.56 level: mean (sd)    mean power: mean (sd)")
    for nominal in sorted(set(rows[:, 0])):
        sel = rows[rows[:, 0] == nominal]
        print(f"{nominal:7.1f} {len(sel):4d}    {sel[:, 1].mean():6.2f} ({sel[:, 1].std():.2f})"
              f"          {sel[:, 2].mean():6.2f} ({sel[:, 2].std():.2f})")


def make(args):
    """Mix the clean utterances of --filelist with the unused DEMAND environments and write the new split."""
    rng = np.random.default_rng(args.seed)
    noises = {}
    for env, label in DEMAND_UNUSED.items():
        x, fs = sf.read(os.path.join(args.demand_dir, env, "ch01.wav"))
        if fs != args.sample_rate:
            sys.exit(f"[ERROR] {env}/ch01.wav is at {fs} Hz, expected {args.sample_rate} Hz (use the *_16k archives)")
        noises[label] = x.astype(np.float64)
    pairs = [line.strip().split("|") for line in open(args.filelist) if line.strip()]
    conditions = [(label, snr) for label in noises for snr in args.snrs]
    order = np.concatenate([rng.permutation(len(conditions)) for _ in range(-(-len(pairs) // len(conditions)))])
    out_dir = os.path.join(args.data_dir, args.name, "noisy")
    os.makedirs(out_dir, exist_ok=True)
    out_pairs, log_lines, peaks, rescaled = [], [], [], 0
    for (clean_rel, _), k in zip(pairs, order):
        label, snr = conditions[k]
        fid = os.path.splitext(os.path.basename(clean_rel))[0]
        clean, fs = sf.read(os.path.join(args.data_dir, clean_rel))
        noise, speech = noises[label], active_speech_power(clean, fs)
        # the clean reference stays the original file, so a mixture that would clip takes another noise excerpt
        # instead of being rescaled; with --rescale, a mixture whose every excerpt clips (low SNRs) is scaled down
        # together with a copy of its clean reference (<name>/clean/<id>.flac)
        best = None
        for _ in range(20):
            start = int(rng.integers(0, len(noise) - len(clean)))
            excerpt = noise[start:start + len(clean)]
            noisy = clean + np.sqrt(speech / (np.mean(excerpt ** 2) * 10 ** (snr / 10))) * excerpt
            if np.max(np.abs(noisy)) < 1.0:
                break
            if best is None or np.max(np.abs(noisy)) < np.max(np.abs(best)):
                best = noisy
        else:
            if not args.rescale:
                sys.exit(f"[ERROR] {fid}: every excerpt of '{label}' clips at {snr:g} dB (see --rescale)")
            gain = 0.99 / np.max(np.abs(best))
            noisy = best * gain
            clean_rel = f"{args.name}/clean/{fid}.flac"
            os.makedirs(os.path.join(args.data_dir, args.name, "clean"), exist_ok=True)
            sf.write(os.path.join(args.data_dir, clean_rel), clean * gain, fs, subtype="PCM_16")
            rescaled += 1
        peaks.append(np.max(np.abs(noisy)))
        noisy_rel = f"{args.name}/noisy/{fid}.flac"
        sf.write(os.path.join(args.data_dir, noisy_rel), noisy, fs, subtype="PCM_16")
        out_pairs.append(f"{clean_rel}|{noisy_rel}\n")
        log_lines.append(f"{fid} {label} {snr:g}\n")
    with open(os.path.join(args.data_dir, args.name, f"log_{args.name}.txt"), "w") as fh:
        fh.writelines(log_lines)
    with open(os.path.join(args.out_dir, f"{args.name}.csv"), "w") as fh:
        fh.writelines(out_pairs)
    counts = {c: 0 for c in conditions}
    for line in log_lines:
        _, label, snr = line.split()
        counts[(label, float(snr))] += 1
    print(f"[INFO] {len(out_pairs)} mixtures in {out_dir}; max peak {max(peaks):.3f}; "
          f"files per condition {min(counts.values())}-{max(counts.values())}; rescaled with their reference: "
          f"{rescaled}")
    print(f"[INFO] filelist {os.path.join(args.out_dir, args.name + '.csv')}")


def main():
    """Build a split, or only compare SNR measures with --check_log."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    ap.add_argument("--demand_dir", default="/home/fred/DATASETS/DEMAND")
    ap.add_argument("--filelist", default="filelists/val.csv", help="clean utterances to mix (clean|noisy pairs)")
    ap.add_argument("--name", default="val_unseen")
    ap.add_argument("--out_dir", default="filelists")
    ap.add_argument("--snrs", type=float, nargs="+", default=list(TEST_SNRS))
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescale", action="store_true",
                    help="scale a mixture that clips with every noise excerpt down, together with a copy of its clean "
                         "reference (needed below ~0 dB)")
    ap.add_argument("--check_log", help="VoiceBank-DEMAND log file: only compare the SNR measures with it")
    ap.add_argument("--check_filelist", default="filelists/test.csv")
    args = ap.parse_args()
    if args.check_log:
        check(args)
    else:
        make(args)


if __name__ == "__main__":
    main()
