#!/usr/bin/env python
"""Robustness report of a set of systems (IMPROVEMENT_PLAN.md, section 14): beyond the mean PESQ, how stable a
system is across seeds, how often it fails on a file, which noise type it handles worst, and how it does at low SNR.

Input: a JSON spec
    {"ref": "<system>",
     "splits": {"<split>": {"labels": "<VoiceBank-DEMAND-style log: id noise snr>", "robust": "<subdir>" | null,
                            "inverted": "<subdir>" | null}},
     "systems": {"<system>": ["results/abl/<run>", ...], ...},          # one run per seed
     "streaming": {"<label>": {"streaming": "<dir>", "offline": "<dir>"}, ...}}   # optional, summary.json dirs
Every run directory holds <split>/metrics.csv (evaluate.py / ensemble_outputs.py); `robust` names a sibling
directory with the pesq_wb_r column (evaluate.py --robust_pesq); `inverted` one with the results on the
polarity-inverted input (evaluate.py --invert).

Per split and system: PESQ-WB mean +- standard deviation over the seeds, its delta against the reference system,
robust PESQ, STOI, SI-SDR (and CSIG / CBAK / COVL when present), the share of files whose seed-averaged PESQ is
more than 0.1 below the reference's, the 1st percentile of that per-file delta, and the worst noise type (delta).
"""
import argparse
import csv
import json
import os
import statistics

import numpy as np


def read_csv(path):
    with open(path) as fh:
        return {r["file"]: r for r in csv.DictReader(fh) if not r.get("error")}


def per_file(runs, split, metric):
    """{file: mean over the runs} for the files every run scored, or {} when a run lacks the metric."""
    tables = []
    for r in runs:
        path = os.path.join(r, split, "metrics.csv")
        if not os.path.exists(path):
            return {}
        t = read_csv(path)
        vals = {f: float(row[metric]) for f, row in t.items() if row.get(metric, "") != ""}
        if not vals:
            return {}
        tables.append(vals)
    files = set.intersection(*(set(t) for t in tables)) if tables else set()
    return {f: float(np.mean([t[f] for t in tables])) for f in files}


def run_means(runs, split, metric):
    out = []
    for r in runs:
        path = os.path.join(r, split, "metrics.csv")
        if not os.path.exists(path):
            continue
        vals = [float(row[metric]) for row in read_csv(path).values() if row.get(metric, "") != ""]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def read_labels(path):
    labels = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 2:
                    labels[f[0]] = f[1]
    return labels


def cell(values, fmt="{:.4f}"):
    if not values:
        return "-"
    if len(values) == 1:
        return fmt.format(values[0])
    return fmt.format(statistics.mean(values)) + " ± " + fmt.format(statistics.stdev(values))


def split_table(spec, split, info):
    systems, ref_name = spec["systems"], spec["ref"]
    ref = per_file(systems[ref_name], split, "pesq_wb")
    ref_mean = float(np.mean(list(ref.values()))) if ref else None
    labels = read_labels(info.get("labels"))
    robust, inverted = info.get("robust"), info.get("inverted")
    composite = any(run_means(systems[ref_name], split, k) for k in ("csig",))
    head = ["system", "seeds", "PESQ-WB", "Δ vs " + ref_name, "robust PESQ", "STOI", "SI-SDR",
            "files Δ<-0.1", "1% worst Δ", "worst noise (Δ)"] + (["CSIG", "CBAK", "COVL"] if composite else []) + \
           (["PESQ, inverted input", "loss when inverted"] if inverted else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, runs in systems.items():
        means = run_means(runs, split, "pesq_wb")
        if not means:
            continue
        pf = per_file(runs, split, "pesq_wb")
        common = sorted(set(pf) & set(ref))
        d = np.array([pf[f] - ref[f] for f in common]) if common else np.array([])
        delta = f"{statistics.mean(means) - ref_mean:+.4f}" if ref_mean is not None else "-"
        tail = f"{(d < -0.1).mean():.1%}" if d.size else "-"
        p1 = f"{np.quantile(d, 0.01):+.3f}" if d.size else "-"
        worst = "-"
        if labels and d.size:
            by = {}
            for f, v in zip(common, d):
                if f in labels:
                    by.setdefault(labels[f], []).append(v)
            if by:
                k = min(by, key=lambda n: np.mean(by[n]))
                worst = f"{k} ({np.mean(by[k]):+.3f})"
        rob = run_means(runs, robust, "pesq_wb_r") if robust else []
        row = [name, str(len(means)), cell(means), delta, cell(rob), cell(run_means(runs, split, "stoi")),
               cell(run_means(runs, split, "si_sdr"), "{:.2f}"), tail, p1, worst]
        if composite:
            row += [cell(run_means(runs, split, k)) for k in ("csig", "cbak", "covl")]
        if inverted:
            inv = run_means(runs, inverted, "pesq_wb")
            loss = f"{statistics.mean(means) - statistics.mean(inv):+.3f}" if inv else "-"
            row += [cell(inv), loss]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.spec) as fh:
        spec = json.load(fh)
    parts = ["# Robustness report", "",
             "PESQ-WB: mean ± standard deviation over the seeds (a single value: one model, e.g. a soup). "
             "Per-file columns use the seed-averaged PESQ against the reference system's. Robust PESQ: mean over "
             "neutral leading-silence pads (evaluation/metrics.py)."]
    for split, info in spec["splits"].items():
        parts += ["", f"## {split}", ""] + split_table(spec, split, info)
    if spec.get("streaming"):
        parts += ["", "## Streaming (test set, frame-by-frame with the running input normalization)", "",
                  "| system | PESQ-WB offline | PESQ-WB streaming | STOI streaming | SI-SDR streaming |",
                  "|---|---|---|---|---|"]
        for label, entry in spec["streaming"].items():
            stream_dir, offline_dir = entry["streaming"], entry["offline"]
            vals = []
            for d, k in ((offline_dir, "pesq_wb"), (stream_dir, "pesq_wb"), (stream_dir, "stoi"),
                         (stream_dir, "si_sdr")):
                path = os.path.join(d, "summary.json")
                vals.append(json.load(open(path))["summary"].get(k) if os.path.exists(path) else None)
            parts.append(f"| {label} | " + " | ".join("-" if v is None else f"{v:.4f}" for v in vals) + " |")
    with open(args.out, "w") as fh:
        fh.write("\n".join(parts) + "\n")
    print(f"[INFO] report written to {args.out}")


if __name__ == "__main__":
    main()
