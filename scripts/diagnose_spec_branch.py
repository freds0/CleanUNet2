#!/usr/bin/env python
"""Stand-alone evaluation of spectral branches (IMPROVEMENT_PLAN.md: D0, D1, D2 and protocol PN).

For every CleanSpecNet checkpoint (any architecture of the track N) and every file:
  D1  PESQ / STOI / SI-SDR of the estimated magnitude resynthesized with the noisy phase
      ("+noisy_phase") and, for models with a phase head, with the predicted phase ("+pred_phase");
      the oracle (clean magnitude + noisy phase) gives the ceiling of magnitude-only processing.
  D2a error of the power-law compressed magnitude, | |est|^0.3 - |clean|^0.3 |, per frequency band and
      per input-SNR bin. Each model is analysed with its own STFT; the noisy input and the waveform
      branch are analysed with every STFT that appears (suffix @n_fft/hop when several do).
  drift  error after 4 s / error in the first 2 s (files longer than 4 s), and the same ratio relative
      to the noisy input: values well above 1 mean the model degrades on sequences longer than the
      2 s training crops.
With --wave_ckpt (waveform-only model, e.g. stage A):
  complementarity  fraction of time-frequency bins where the CleanSpecNet magnitude is closer to the
      clean one than |STFT(waveform-branch output)| (unweighted and weighted by the clean power);
  D2b per-bin oracle "best of the two" magnitudes with the waveform-branch phase (late-fusion ceiling),
      D2c training-free blends |Y|^(1-w) |S_hat|^w, for the last --spec_ckpt.

Usage:
    python scripts/diagnose_spec_branch.py \
        --wave_ckpt logs/stageA_cleanunet_mamba/checkpoints/best-epoch016-pesq2.755.ckpt \
        --spec_ckpt B=logs/stageB_cleanspecnet_mamba/checkpoints/last.ckpt \
        --spec_ckpt Bplus=logs/abl/Bplus/last.ckpt \
        --filelist filelists/val.csv --out results/abl/D1D2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleanunet.cleanunet2 import CleanUNet2  # noqa: E402
from dataset import get_dataset_filelist, load_audio  # noqa: E402
from evaluation.metrics import evaluate_many, summarize  # noqa: E402

BANDS_HZ = (0, 1000, 2000, 4000, 8001)
P = 0.3            # common error domain: |magnitude|^0.3 of the window-normalized STFT
SR = 16000


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CleanUNet2(**ckpt["hyper_parameters"]["model"]["architecture"])
    model.load_cleanunet2_weights(ckpt_path)
    return model.to(device).eval()


def framing(S):
    return (S.n_fft, S.hop_length, S.win_length, S.alignment)


class Acc:
    """Sum / count of an error map per (frequency band, SNR bin)."""

    def __init__(self, n_snr):
        self.s = np.zeros((len(BANDS_HZ) - 1, n_snr))
        self.c = np.zeros((len(BANDS_HZ) - 1, n_snr))

    def add(self, err, band_idx, snr_bin):
        e = err.cpu().numpy()[0]
        for b in range(len(BANDS_HZ) - 1):
            sel = e[band_idx == b]
            self.s[b, snr_bin] += sel.sum()
            self.c[b, snr_bin] += sel.size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave_ckpt", default=None, help="Lightning checkpoint of the waveform-only model (D2)")
    ap.add_argument("--spec_ckpt", action="append", required=True, help="NAME=checkpoint with a CleanSpecNet")
    ap.add_argument("--data_dir", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    ap.add_argument("--filelist", default="filelists/val.csv")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--edges", default="2.5,7.5,12.5")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    wave = load_model(args.wave_ckpt, args.device) if args.wave_ckpt else None
    specs = {}
    for item in args.spec_ckpt:
        name, path = item.split("=", 1)
        specs[name] = load_model(path, args.device)
    best_name = list(specs)[-1]
    framings = {}
    for m in specs.values():
        framings.setdefault(framing(m.spectrogram), m.spectrogram)
    multi = len(framings) > 1
    tag = {f: (f"@{f[0]}/{f[1]}" if multi else "") for f in framings}
    edges = [float(e) for e in args.edges.split(",")]
    n_snr = len(edges) + 1

    err, systems = {}, {}
    drift = {}          # name -> list of (late/early ratio of the system, of the noisy input)
    comp = {}           # name -> [sum(better), count, sum(better * w), sum(w)]

    def acc(name):
        return err.setdefault(name, Acc(n_snr))

    def put(name, c_np, wav):
        systems.setdefault(name, []).append((c_np, wav))

    pairs = get_dataset_filelist(args.filelist)[:args.max_files]
    with torch.no_grad():
        for clean_rel, noisy_rel in tqdm(pairs, desc="Diagnosing"):
            clean = load_audio(os.path.join(args.data_dir, clean_rel), SR)
            noisy = load_audio(os.path.join(args.data_dir, noisy_rel), SR)
            c_np, n_np = clean.numpy(), noisy.numpy()
            snr = 10 * np.log10((c_np ** 2).sum() / max(((n_np - c_np) ** 2).sum(), 1e-12))
            snr_bin = int(np.digitize(snr, edges))
            x = noisy.to(args.device).view(1, 1, -1)
            c = clean.to(args.device).view(1, 1, -1)
            L = x.shape[-1]
            scale = x.std() + 1e-3
            xn, cn = x / scale, c / scale
            y = wave(x)["wav"] if wave is not None else None

            per_framing = {}
            for f, S in framings.items():
                X = S.complex_stft(xn)
                mag_c = S.complex_stft(cn).abs() * S.scale            # window-normalized linear magnitudes
                ref = mag_c.pow(P)
                band_idx = np.digitize(np.linspace(0, SR / 2, S.n_freq), BANDS_HZ) - 1
                e_noisy = ((X.abs() * S.scale).pow(P) - ref).abs()
                acc(f"noisy{tag[f]}").add(e_noisy, band_idx, snr_bin)
                d = {"X": X, "mag_c": mag_c, "ref": ref, "band_idx": band_idx, "e_noisy": e_noisy}
                if y is not None:
                    Y = S.complex_stft(y / scale)
                    d["mag_y"] = Y.abs() * S.scale
                    d["phase_y"] = Y / Y.abs().clamp(min=1e-8)
                    d["e_wave"] = (d["mag_y"].pow(P) - ref).abs()
                    acc(f"wave_branch{tag[f]}").add(d["e_wave"], band_idx, snr_bin)
                per_framing[f] = d
                if f == next(iter(framings)):
                    put("oracle_mag+noisy_phase", c_np,
                        (S.istft(mag_c / S.scale * X / X.abs().clamp(min=1e-8), L) * scale).view(-1).cpu().numpy())
            if y is not None:
                put("wave_branch", c_np, y.view(-1).float().cpu().numpy())

            for name, m in specs.items():
                S = m.spectrogram
                d = per_framing[framing(S)]
                X, ref, band_idx = d["X"], d["ref"], d["band_idx"]
                T = X.shape[-1]
                res = m.spec_branch(xn)
                mag_s = S.decompress(res["spec"][..., :T].float())
                e_s = (mag_s.pow(P) - ref).abs()
                acc(f"cleanspecnet_{name}").add(e_s, band_idx, snr_bin)

                def synth(mag, unit):
                    return (S.istft(mag / S.scale * unit, L) * scale).view(-1).float().cpu().numpy()

                put(f"cleanspecnet_{name}+noisy_phase", c_np, synth(mag_s, X / X.abs().clamp(min=1e-8)))
                if res.get("phase") is not None:
                    unit = torch.polar(torch.ones_like(res["phase"][..., :T]).float(), res["phase"][..., :T].float())
                    put(f"cleanspecnet_{name}+pred_phase", c_np, synth(mag_s, unit))

                # drift: error after 4 s relative to the first 2 s
                fps = SR / S.hop_length
                if T > 4 * fps:
                    early, late = slice(0, int(2 * fps)), slice(int(4 * fps), T)
                    r_sys = (e_s[..., late].mean() / e_s[..., early].mean().clamp(min=1e-8)).item()
                    e_n = d["e_noisy"]
                    r_noisy = (e_n[..., late].mean() / e_n[..., early].mean().clamp(min=1e-8)).item()
                    drift.setdefault(name, []).append((r_sys, r_noisy))

                if y is None:
                    continue
                better = (e_s < d["e_wave"]).float()
                w = d["mag_c"].pow(2)
                cm = comp.setdefault(name, [0.0, 0.0, 0.0, 0.0])
                cm[0] += better.sum().item()
                cm[1] += better.numel()
                cm[2] += (better * w).sum().item()
                cm[3] += w.sum().item()
                if name != best_name:
                    continue
                # (b) per-bin oracle choice and (c) training-free blends, phase of the waveform branch
                pick_spec = e_s < d["e_wave"]
                best = torch.where(pick_spec, mag_s, d["mag_y"])
                acc(f"oracle_best_of_two_{name}").add((best.pow(P) - ref).abs(), band_idx, snr_bin)
                put(f"oracle_best_of_two_{name}", c_np, synth(best, d["phase_y"]))
                for wgt in (0.25, 0.5):
                    mag = (d["mag_y"].clamp(min=1e-8).log() * (1 - wgt) + mag_s.clamp(min=1e-8).log() * wgt).exp()
                    put(f"blend_w{wgt}_{name}", c_np, synth(mag, d["phase_y"]))

    labels_b = [f"{a / 1000:g}-{b / 1000:g}k" for a, b in zip(BANDS_HZ[:-1], BANDS_HZ[1:])]
    labels_s = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]
    report = {"log_mag_l1": {}, "metrics": {}, "drift": {}, "complementarity": {}}
    print("\nD2a: mean | |est|^0.3 - |clean|^0.3 | by band (all SNRs) and by input SNR (all bands)")
    print("system".ljust(40) + "".join(l.ljust(10) for l in labels_b + labels_s) + "all")
    for name, a in err.items():
        by_band = a.s.sum(1) / np.maximum(a.c.sum(1), 1)
        by_snr = a.s.sum(0) / np.maximum(a.c.sum(0), 1)
        total = a.s.sum() / max(a.c.sum(), 1)
        report["log_mag_l1"][name] = {"by_band": dict(zip(labels_b, by_band.tolist())),
                                      "by_snr": dict(zip(labels_s, by_snr.tolist())), "all": float(total)}
        print(name.ljust(40) + "".join(f"{v:<10.4f}" for v in list(by_band) + list(by_snr)) + f"{total:.4f}")

    if drift:
        print("\nDrift (error after 4 s / error in the first 2 s; files > 4 s)")
        for name, rs in drift.items():
            r_sys, r_noisy = np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs])
            report["drift"][name] = {"ratio": float(r_sys), "ratio_noisy": float(r_noisy),
                                     "relative": float(r_sys / max(r_noisy, 1e-8)), "n_files": len(rs)}
            print(f"  {name:32s} ratio {r_sys:.3f}  noisy {r_noisy:.3f}  relative {r_sys / max(r_noisy, 1e-8):.3f}"
                  f"  (n={len(rs)})")
    if comp:
        print("\nComplementarity with the waveform branch (fraction of bins where CleanSpecNet is closer to clean)")
        for name, (b, n, bw, w) in comp.items():
            report["complementarity"][name] = {"fraction": b / max(n, 1), "fraction_power_weighted": bw / max(w, 1e-12)}
            print(f"  {name:32s} {b / max(n, 1):.3f} (power-weighted {bw / max(w, 1e-12):.3f})")

    print("\nD1 / D2b / D2c: metrics of the resynthesized signals")
    for name, jobs in systems.items():
        summ = summarize(evaluate_many(jobs, sample_rate=SR, full=False, num_workers=args.workers))
        report["metrics"][name] = summ
        print(f"{name:40s} PESQ {summ.get('pesq_wb', float('nan')):.3f}  STOI {summ.get('stoi', float('nan')):.3f}"
              f"  SI-SDR {summ.get('si_sdr', float('nan')):.2f}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "diagnostics.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[INFO] Written {args.out}/diagnostics.json")


if __name__ == "__main__":
    main()
