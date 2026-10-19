#!/usr/bin/env python
"""
Evaluate a model (or the unprocessed noisy input) on whole utterances.

Metrics: PESQ-WB/NB, STOI, SI-SDR, CSIG, CBAK, COVL, SSNR (+ WSS, LLR), per file (CSV) and averaged (JSON).

Examples:
    # Noisy baseline
    python evaluate.py --noisy --filelist filelists/test.csv
    # Lightning checkpoint trained with a given config
    python evaluate.py --config configs/train_stageA_cleanunet_mamba.yaml --checkpoint logs/.../best.ckpt
    # Official CleanUMamba checkpoint, zero-shot, offline and streaming
    python evaluate.py --architecture configs/model/cleanunet_mamba_e8.yaml \
        --cleanumamba ../cleanumamba/checkpoints/models/CleanUMamba-3N-E8-full.pkl [--streaming]
    # Fusion ablations: architecture read from the Lightning checkpoint; condition taken from another file
    python evaluate.py --checkpoint logs/abl/P3/pc2/last.ckpt --filelist filelists/val.csv --condition shuffle
    # Training-free blend with a CleanSpecNet, test-time augmentation and the robust PESQ (IMPROVEMENT_PLAN.md, 14)
    python evaluate.py --checkpoint logs/abl/K0extfull_s1/last.ckpt --blend_spec logs/abl/N_R_ext50_final/last.ckpt \
        --blend_w 0.25 --tta flip --robust_pesq
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cleanunet.blend import blend  # noqa: F401  (also imported from here by the tests)
from config_utils import load_config, resolve_architecture
from dataset import get_dataset_filelist, load_audio
from evaluation.metrics import METRIC_NAMES, evaluate_many, summarize

# Test-time augmentation (--tta): input shifts in samples (a quarter of the E8 total stride of 256 apart, so every
# shift meets the strided encoder at another phase) and / or the sign flip; the outputs are realigned and averaged.
TTA_SHIFTS = (0, 64, 128, 192)


def build_model(args):
    from cleanunet.cleanunet2 import CleanUNet2
    if args.config:
        arch = load_config(args.config)["model"]["architecture"]
    elif args.architecture:
        arch = resolve_architecture(args.architecture)
    else:   # Lightning checkpoints store the resolved training config
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        arch = ckpt["hyper_parameters"]["model"]["architecture"]
    model = CleanUNet2(**arch)
    if args.cleanumamba:
        model.load_cleanumamba_weights(args.cleanumamba)
    if args.checkpoint:
        model.load_cleanunet2_weights(args.checkpoint)
    return model


def _fit_length(x: torch.Tensor, L: int) -> torch.Tensor:
    """Tile or crop a 1-D signal to L samples."""
    reps = -(-L // x.shape[-1])
    return x.repeat(reps)[:L]


def input_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    return float(10 * np.log10((clean ** 2).sum() / max((noise ** 2).sum(), 1e-12) + 1e-12))


def _tta_plan(mode: str):
    shifts = TTA_SHIFTS if mode in ("shift", "flipshift") else (0,)
    signs = (1.0, -1.0) if mode in ("flip", "flipshift") else (1.0,)
    return [(s, g) for s in shifts for g in signs]


@torch.no_grad()
def enhance(model, noisy: torch.Tensor, args, clean: torch.Tensor = None, cond=None) -> torch.Tensor:
    """noisy: (L,) -> enhanced (L,) on CPU. `cond` = (noisy, clean) of another file for --condition shuffle.
    With args.invert the model gets the polarity-inverted input and its output is inverted back."""
    polarity = -1.0 if getattr(args, "invert", False) else 1.0
    x = polarity * noisy.to(args.device).view(1, 1, -1)
    tta = getattr(args, "tta", "none")
    if args.streaming:
        from cleanunet.streaming import stream_denoise
        if model.use_spec_branch:
            raise ValueError("--streaming only supports waveform-only models (use_spec_branch: false)")
        if tta != "none":
            raise ValueError("--tta is not supported with --streaming")
        # CleanUNet2 normalizes outside CleanUNet; streaming uses its own running normalization
        y = stream_denoise(model.clean_unet, x.view(1, -1), chunk_size=args.chunk_size,
                           prior_std=getattr(args, "stream_prior_std", None),
                           prior_frames=getattr(args, "stream_prior_frames", 0.0))
        return (polarity * y).view(-1).float().cpu()
    kw = {}
    if getattr(model, "needs_clean", False) and clean is not None:
        kw["clean"] = polarity * clean.to(args.device).view(1, 1, -1)
    if args.condition == "off":
        kw["fusion_enabled"] = False
    elif args.condition == "shuffle":
        if tta != "none":
            raise ValueError("--tta is not supported with --condition shuffle")
        L = x.shape[-1]
        kw["cond_noisy"] = polarity * _fit_length(cond[0], L).to(args.device).view(1, 1, -1)
        kw["cond_clean"] = polarity * _fit_length(cond[1], L).to(args.device).view(1, 1, -1)
    plan = _tta_plan(tta)
    if len(plan) > 1 and "clean" in kw:
        raise ValueError("--tta is not supported with oracle conditions")
    y = 0.0
    for shift, sign in plan:
        xs = F.pad(x, (shift, 0)) if shift else x
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and x.is_cuda):
            out = model(sign * xs, **kw)["wav"]
        y = y + sign * out.float()[..., shift:]
    y = y / len(plan)
    if getattr(args, "blend_model", None) is not None:
        y = blend(y, x, args.blend_model, args.blend_w)
    return (polarity * y).view(-1).float().cpu()


def snr_summary(results, snrs, edges):
    """Summary of the files of each input-SNR bin."""
    bins = np.digitize(snrs, edges)
    labels = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]
    out = {}
    for b, label in enumerate(labels):
        sel = [r for r, k in zip(results, bins) if k == b]
        if sel:
            out[label] = summarize(sel)
    return out


def default_snr_edges(filelist: str) -> str:
    return "5,10,15" if "test" in os.path.basename(filelist) else "2.5,7.5,12.5"


def write_results(out_dir, names, snrs, results, summary, args_dict):
    """metrics.csv (per file) and summary.json, the format read by scripts/compare_runs.py."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "snr_in"] + list(METRIC_NAMES) + ["error"])
        for n, snr, r in zip(names, snrs, results):
            w.writerow([n, f"{snr:.2f}"] + [r.get(k, "") for k in METRIC_NAMES] + [r.get("error", "")])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"args": args_dict, "summary": summary}, fh, indent=2)


def print_summary(summary):
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()
                      if k != "by_snr"}, indent=2))
    print("PESQ-WB by input SNR: " + ", ".join(f"{k}: {v.get('pesq_wb', float('nan')):.3f} (n={v['n_files']})"
                                               for k, v in summary["by_snr"].items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="training YAML (its model.architecture is used)")
    p.add_argument("--architecture", help="model architecture YAML (alternative to --config)")
    p.add_argument("--checkpoint", help="Lightning / CleanUNet2 checkpoint")
    p.add_argument("--cleanumamba", help="official CleanUMamba .pkl checkpoint for the waveform branch")
    p.add_argument("--noisy", action="store_true", help="evaluate the unprocessed noisy input")
    p.add_argument("--data_dir", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    p.add_argument("--filelist", default="filelists/test.csv")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--out_dir", default=None, help="where to write metrics (and enhanced audio with --save_audio)")
    p.add_argument("--save_audio", action="store_true")
    p.add_argument("--streaming", action="store_true", help="frame-by-frame streaming inference (Mamba only)")
    p.add_argument("--chunk_size", type=int, default=4096)
    p.add_argument("--stream_prior_std", type=float, default=None,
                   help="streaming: start the running input scale from this value (IMPROVEMENT_PLAN.md, 14.7.4)")
    p.add_argument("--stream_prior_frames", type=float, default=0.0,
                   help="streaming: weight of --stream_prior_std in frames (62 = ~1 s with E8; inf = fixed scale)")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast for offline inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--fast", action="store_true", help="only PESQ-WB, STOI and SI-SDR (no composite measures)")
    p.add_argument("--condition", choices=["normal", "shuffle", "off"], default="normal",
                   help="spectral condition from the input (normal), from another file (shuffle) or "
                        "disabled (off); measures how much a hybrid model depends on its condition")
    p.add_argument("--blend_spec", help="Lightning checkpoint of a CleanSpecNet: blend its magnitude with the "
                                        "model output, without training (--blend_w)")
    p.add_argument("--blend_w", type=float, default=0.5, help="weight of the CleanSpecNet magnitude (log domain)")
    p.add_argument("--tta", choices=["none", "flip", "shift", "flipshift"], default="none",
                   help="test-time augmentation: sign flip and / or input shifts (TTA_SHIFTS), outputs averaged")
    p.add_argument("--invert", action="store_true",
                   help="feed the polarity-inverted input and invert the output back (polarity invariance)")
    p.add_argument("--robust_pesq", action="store_true",
                   help="also compute pesq_wb_r, the mean PESQ-WB over neutral leading-silence pads")
    p.add_argument("--band_limited", action="store_true",
                   help="also compute pesq_wb_hp, PESQ-WB with reference and estimate high-passed at 100 Hz")
    p.add_argument("--snr_edges", default=None,
                   help="comma-separated input-SNR bin edges in dB for the per-SNR summary "
                        "(default: 5,10,15 for the test list, 2.5,7.5,12.5 otherwise)")
    args = p.parse_args()
    if args.snr_edges is None:
        args.snr_edges = default_snr_edges(args.filelist)

    pairs = get_dataset_filelist(args.filelist)[:args.max_files]
    model = None
    if not args.noisy:
        if not (args.config or args.architecture or args.checkpoint):
            p.error("--config, --architecture or --checkpoint is required unless --noisy is given")
        model = build_model(args).to(args.device).eval()
        print(f"[INFO] Model parameters: {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M")
    args.blend_model = None
    if args.blend_spec:
        if model is None:
            p.error("--blend_spec needs a model")
        from cleanunet.cleanunet2 import CleanUNet2
        ck = torch.load(args.blend_spec, map_location="cpu", weights_only=False)
        args.blend_model = CleanUNet2(**ck["hyper_parameters"]["model"]["architecture"])
        args.blend_model.load_cleanunet2_weights(args.blend_spec)
        args.blend_model = args.blend_model.to(args.device).eval()
        print(f"[INFO] Blend with {args.blend_spec}, w = {args.blend_w}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    jobs, names, snrs, proc_time, audio_time = [], [], [], 0.0, 0.0
    load = lambda rel: load_audio(os.path.join(args.data_dir, rel), args.sample_rate)  # noqa: E731
    for idx, (clean_rel, noisy_rel) in enumerate(tqdm(pairs, desc="Enhancing")):
        clean = load(clean_rel)
        noisy = load(noisy_rel)
        snrs.append(input_snr(clean.numpy(), noisy.numpy()))
        if model is None:
            enhanced = noisy
        else:
            cond = None
            if args.condition == "shuffle":   # a fixed derangement: file i uses file i + N/2
                c_rel, n_rel = pairs[(idx + max(1, len(pairs) // 2)) % len(pairs)]
                cond = (load(n_rel), load(c_rel))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.time()
            enhanced = enhance(model, noisy, args, clean=clean, cond=cond)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            proc_time += time.time() - t
            audio_time += len(noisy) / args.sample_rate
        name = os.path.splitext(os.path.basename(clean_rel))[0]
        names.append(name)
        jobs.append((clean.numpy(), enhanced.numpy()))
        if args.out_dir and args.save_audio:
            sf.write(os.path.join(args.out_dir, f"{name}.flac"), enhanced.numpy(), args.sample_rate)

    results = evaluate_many(jobs, sample_rate=args.sample_rate, full=not args.fast, num_workers=args.workers,
                            robust=args.robust_pesq, band_limited=args.band_limited)
    summary = summarize(results)
    if model is not None:
        summary["rtf"] = proc_time / max(audio_time, 1e-9)
        summary["params_M"] = sum(q.numel() for q in model.parameters()) / 1e6
        if args.blend_model is not None:
            summary["params_M"] += sum(q.numel() for q in args.blend_model.clean_spec_net.parameters()) / 1e6
    summary["by_snr"] = snr_summary(results, snrs, [float(e) for e in args.snr_edges.split(",")])
    print_summary(summary)
    if args.out_dir:
        write_results(args.out_dir, names, snrs, results, summary,
                      {k: v for k, v in vars(args).items() if k != "blend_model"})
        print(f"[INFO] Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
