#!/usr/bin/env python
"""Weighted average of the enhanced audio of several systems (training-free output ensemble, IMPROVEMENT_PLAN.md,
section 14), scored like evaluate.py.

Each input directory holds the <file>.flac written by `evaluate.py --save_audio` for the same file list. The output
directory gets metrics.csv and summary.json in the format of evaluate.py (read by scripts/compare_runs.py).

Usage:
    python scripts/ensemble_outputs.py --inputs results/abl/W_audio/RX_C3_s1/val_unseen \
        results/abl/W_audio/blend_s1/val_unseen --weights 0.5 0.5 --filelist filelists/val_unseen.csv \
        --out_dir results/abl/W_ens_s1/val_unseen --fast
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import get_dataset_filelist  # noqa: E402
from evaluate import default_snr_edges, input_snr, print_summary, snr_summary, write_results  # noqa: E402
from evaluation.metrics import evaluate_many, summarize  # noqa: E402


def average(paths, weights):
    """Weighted mean of equally long signals (the first signal's length if they differ by a few samples)."""
    sigs = [sf.read(p, dtype="float64")[0] for p in paths]
    L = min(len(s) for s in sigs)
    return sum(w * s[:L] for w, s in zip(weights, sigs))


def main():
    """Average enhanced waveforms of several systems (optionally weighted) and score the average."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="directories with the enhanced <file>.flac")
    ap.add_argument("--weights", nargs="+", type=float, help="one weight per input (normalized; default: equal)")
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--data_dir", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fast", action="store_true", help="only PESQ-WB, STOI and SI-SDR")
    ap.add_argument("--robust_pesq", action="store_true")
    ap.add_argument("--save_audio", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample_rate", type=int, default=16000)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.inputs)
    if len(weights) != len(args.inputs):
        ap.error("--weights needs one value per input")
    weights = [w / sum(weights) for w in weights]

    pairs = get_dataset_filelist(args.filelist)
    os.makedirs(args.out_dir, exist_ok=True)
    jobs, names, snrs = [], [], []
    for clean_rel, noisy_rel in pairs:
        name = os.path.splitext(os.path.basename(clean_rel))[0]
        clean, sr = sf.read(os.path.join(args.data_dir, clean_rel), dtype="float64")
        noisy, _ = sf.read(os.path.join(args.data_dir, noisy_rel), dtype="float64")
        if sr != args.sample_rate:
            sys.exit(f"[ERROR] {clean_rel}: {sr} Hz, expected {args.sample_rate} Hz")
        missing = [d for d in args.inputs if not os.path.exists(os.path.join(d, f"{name}.flac"))]
        if missing:
            sys.exit(f"[ERROR] {name}.flac missing in {missing}")
        est = average([os.path.join(d, f"{name}.flac") for d in args.inputs], weights)
        n = min(len(clean), len(est))
        snrs.append(input_snr(clean[:len(noisy)], noisy[:len(clean)]))
        names.append(name)
        jobs.append((clean[:n], est[:n]))
        if args.save_audio:
            sf.write(os.path.join(args.out_dir, f"{name}.flac"), est, args.sample_rate)

    results = evaluate_many(jobs, sample_rate=args.sample_rate, full=not args.fast, num_workers=args.workers,
                            robust=args.robust_pesq)
    summary = summarize(results)
    summary["by_snr"] = snr_summary(results, snrs, [float(e) for e in default_snr_edges(args.filelist).split(",")])
    print_summary(summary)
    write_results(args.out_dir, names, snrs, results, summary, {**vars(args), "weights": weights})
    print(f"[INFO] Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
