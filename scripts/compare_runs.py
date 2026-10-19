#!/usr/bin/env python
"""Compare ablation runs against a reference (IMPROVEMENT_PLAN.md, section 3.4).

Each run directory holds `val/metrics.csv` (written by evaluate.py) and optionally
`val_shuffle/` (condition taken from another file) for the dependency measure.
Several reference directories (e.g. the K0 seeds) are averaged per file.

Reported per run: mean metric, delta vs the reference, paired bootstrap 95% CI of the per-file
delta, Wilcoxon p-value (Holm-adjusted over the compared runs), dependency
(metric with the true condition minus metric with another file's condition), and the delta per
input-SNR bin.

Usage:
    python scripts/compare_runs.py --ref results/abl/K0_s1 results/abl/K0_s2 results/abl/K0_s3 \
        --runs results/abl/P* [--metric pesq_wb] [--split val] [--out results/abl/phase1.md]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def read_metrics(path, metric):
    """{file: value} of one metric from a metrics.csv, skipping failed rows."""
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("error") or r.get(metric, "") == "":
                continue
            rows[r["file"]] = (float(r[metric]), float(r["snr_in"]) if r.get("snr_in") else np.nan)
    return rows


def holm(pvals):
    """Holm-Bonferroni adjustment of a list of p-values."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    """Compare each run with the reference per file (bootstrap CI, Wilcoxon test with Holm correction) and print a table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True, help="reference run dir(s); averaged per file")
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (globs allowed)")
    ap.add_argument("--metric", default="pesq_wb")
    ap.add_argument("--split", default="val", help="sub-directory with metrics.csv")
    ap.add_argument("--edges", default="2.5,7.5,12.5", help="input-SNR bin edges (dB)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", help="also write the table (markdown) to this file")
    args = ap.parse_args()

    from scipy.stats import wilcoxon

    refs = [read_metrics(os.path.join(d, args.split, "metrics.csv"), args.metric) for d in args.ref]
    files = sorted(set.intersection(*[set(r) for r in refs]))
    ref = np.array([np.mean([r[f][0] for r in refs]) for f in files])
    snr = np.array([refs[0][f][1] for f in files])
    ref_runs = [np.mean([r[f][0] for f in files]) for r in refs]
    edges = [float(e) for e in args.edges.split(",")]
    bins = np.digitize(snr, edges)
    labels = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]

    run_dirs = sorted({d for pattern in args.runs for d in glob.glob(pattern)} - set(args.ref))
    rng = np.random.default_rng(0)
    rows, pvals = [], []
    for d in run_dirs:
        path = os.path.join(d, args.split, "metrics.csv")
        if not os.path.exists(path):
            continue
        m = read_metrics(path, args.metric)
        common = [i for i, f in enumerate(files) if f in m]
        if not common:
            continue
        x = np.array([m[files[i]][0] for i in common])
        delta = x - ref[common]
        boot = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(args.n_boot)]
        p = wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
        dep = np.nan
        shuf = os.path.join(d, f"{args.split}_shuffle", "summary.json")
        if os.path.exists(shuf):
            with open(shuf) as fh:
                dep = x.mean() - json.load(fh)["summary"][args.metric]
        by_snr = [delta[bins[common] == b].mean() if np.any(bins[common] == b) else np.nan
                  for b in range(len(labels))]
        name = os.path.basename(d.rstrip("/"))
        if name in ("val", "test"):     # a split directory given directly (--split ""): keep the run name
            name = f"{os.path.basename(os.path.dirname(d.rstrip('/')))}/{name}"
        rows.append(dict(run=name, n=len(common), mean=x.mean(), delta=delta.mean(),
                         lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=p, dep=dep, by_snr=by_snr))
        pvals.append(p)
    for r, padj in zip(rows, holm(np.array(pvals)) if pvals else []):
        r["p_holm"] = padj

    sd = np.std(ref_runs, ddof=1) if len(ref_runs) > 1 else float("nan")
    lines = [f"Reference: {len(args.ref)} run(s), {args.metric} = {np.mean(ref_runs):.4f} "
             f"(per-run: {', '.join(f'{v:.4f}' for v in ref_runs)}; sigma = {sd:.4f}); {len(files)} files", "",
             "| run | n | mean | delta | 95% CI | p (Holm) | dependency | " + " | ".join(f"delta {l} dB" for l in labels) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(labels)]
    for r in sorted(rows, key=lambda r: -r["delta"]):
        lines.append(f"| {r['run']} | {r['n']} | {r['mean']:.4f} | {r['delta']:+.4f} | "
                     f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p_holm']:.2g} | "
                     + ("-" if np.isnan(r["dep"]) else f"{r['dep']:+.4f}") + " | "
                     + " | ".join("-" if np.isnan(v) else f"{v:+.3f}" for v in r["by_snr"]) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
