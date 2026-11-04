#!/usr/bin/env python
"""
Multi-domain development set (IMPROVEMENT_PLAN.md, 16.4, item 0.1): an equal number of files from three domains
that no training list uses, so that the all-data models are not selected on VoiceBank alone.

  vb          filelists/val_unseen.csv (VoiceBank speakers p226 / p287, unseen DEMAND noises)
  ears_wham   EARS-WHAM v2 16 kHz valid (speakers p100-p101, WHAM! cv noise)
  libri2mix   Libri2Mix min dev, mix_single (LibriSpeech dev-clean readers, WHAM! cv noise)

Writes filelists/dev_multi.csv (all domains, interleaved) and filelists/dev_multi_<domain>.csv.

Usage: python scripts/make_dev_multi.py [--per_domain 300] [--root /home/fred/DATASETS_CleanUNet]
"""
import argparse
import glob
import os
import random


def ears_wham(root):
    """(clean, noisy) pairs of EARS-WHAM v2 valid; noisy files carry the SNR in their name."""
    base = os.path.join(root, "ears_wham_sources/EARS-WHAM_v2_16k/valid")
    pairs = []
    for noisy in sorted(glob.glob(os.path.join(base, "noisy/*/*.wav"))):
        spk, name = noisy.split(os.sep)[-2:]
        clean = os.path.join(base, "clean", spk, name.split("_")[0] + ".wav")
        if os.path.exists(clean):
            pairs.append((clean, noisy))
    return pairs


def libri2mix(root):
    """(s1, mix_single) pairs of Libri2Mix min dev."""
    base = os.path.join(root, "librimix/Libri2Mix/wav16k/min/dev")
    return [(os.path.join(base, "s1", os.path.basename(m)), m)
            for m in sorted(glob.glob(os.path.join(base, "mix_single/*.wav")))]


def filelist(path):
    with open(path) as fh:
        return [tuple(line.strip().split("|")) for line in fh if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_domain", type=int, default=300)
    p.add_argument("--root", default="/home/fred/DATASETS_CleanUNet")
    p.add_argument("--out_dir", default="filelists")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    domains = {"vb": filelist(os.path.join(args.out_dir, "val_unseen.csv")),
               "ears_wham": ears_wham(args.root), "libri2mix": libri2mix(args.root)}
    rng = random.Random(args.seed)
    picked = {}
    for name, pairs in domains.items():
        if len(pairs) < args.per_domain:
            raise SystemExit(f"{name}: only {len(pairs)} pairs")
        picked[name] = sorted(rng.sample(pairs, args.per_domain))
        with open(os.path.join(args.out_dir, f"dev_multi_{name}.csv"), "w") as fh:
            fh.writelines(f"{c}|{n}\n" for c, n in picked[name])
        print(f"{name}: {args.per_domain} of {len(pairs)}")
    with open(os.path.join(args.out_dir, "dev_multi.csv"), "w") as fh:
        for rows in zip(*picked.values()):
            fh.writelines(f"{c}|{n}\n" for c, n in rows)


if __name__ == "__main__":
    main()
