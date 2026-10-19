#!/usr/bin/env python
"""Create train/val/test filelists ("clean|noisy" relative paths) for VoiceBank-DEMAND.

Expected layout (as produced by download_voicebank_demand_16k.py):
    <data_dir>/train/{clean,noisy}/<id>.<ext>
    <data_dir>/test/{clean,noisy}/<id>.<ext>

The validation set is made of whole speakers held out from the training split
(default: p226 and p287), so model selection never touches the test set.
"""
import argparse
import os
import sys


def list_pairs(data_dir, split, ext):
    clean_dir = os.path.join(data_dir, split, "clean")
    noisy_dir = os.path.join(data_dir, split, "noisy")
    names = sorted(f for f in os.listdir(clean_dir) if f.endswith(ext))
    missing = [n for n in names if not os.path.exists(os.path.join(noisy_dir, n))]
    if missing:
        sys.exit(f"[ERROR] {len(missing)} clean files of split '{split}' have no noisy counterpart, e.g. {missing[:3]}")
    return [(f"{split}/clean/{n}", f"{split}/noisy/{n}") for n in names]


def write(path, pairs):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"{c}|{n}\n" for c, n in pairs)
    print(f"[INFO] {path}: {len(pairs)} pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    parser.add_argument("--out_dir", default="filelists")
    parser.add_argument("--ext", default=".flac")
    parser.add_argument("--val_speakers", nargs="*", default=["p226", "p287"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = list_pairs(args.data_dir, "train", args.ext)
    test = list_pairs(args.data_dir, "test", args.ext)
    speaker = lambda pair: os.path.basename(pair[0]).split("_")[0]  # noqa: E731
    val = [p for p in train if speaker(p) in args.val_speakers]
    train = [p for p in train if speaker(p) not in args.val_speakers]
    if args.val_speakers and not val:
        sys.exit(f"[ERROR] None of the validation speakers {args.val_speakers} found in the training split")

    write(os.path.join(args.out_dir, "train.csv"), train)
    write(os.path.join(args.out_dir, "val.csv"), val)
    write(os.path.join(args.out_dir, "test.csv"), test)


if __name__ == "__main__":
    main()
