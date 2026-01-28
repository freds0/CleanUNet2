#!/usr/bin/env python3
"""
CleanUNet2 – Inference Script (Config-aware Version)
----------------------------------------------------
Loads inference settings from a YAML file containing:

  inference:
  model:
  audio:
  output:
  runtime:

This version is fully compatible with CleanUNetLightningModule,
which requires: forward(waveform, spectrogram).
"""

import os
import json
from pathlib import Path
from glob import glob
import argparse
import yaml

import torch
import torchaudio
from tqdm import tqdm
from argparse import Namespace

# Lightning Module Wrapper
try:
    from lightning_modules.cleanunet_module import CleanUNetLightningModule
except:
    from cleanunet.cleanunet2 import CleanUNet2 as CleanUNetLightningModule


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    return ckpt


def mono_and_resample(wav, orig_sr, target_sr, device):
    """Convert stereo to mono and resample if needed."""
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)(wav)
    return wav


def normalize_audio(x):
    """Peak-normalize audio."""
    peak = x.abs().max()
    if peak > 1e-9:
        return x / peak
    return x


def undo_normalize(x, peak):
    """Invert peak normalization."""
    return x * peak


def sliding_windows(audio, segment_size, hop_size):
    """Yield sliding windows [1, segment_size]."""
    T = audio.shape[-1]

    if T <= segment_size:
        seg = torch.zeros((1, segment_size), device=audio.device)
        seg[0, :T] = audio
        yield 0, T, seg
        return

    start = 0
    while start < T:
        end = start + segment_size
        if end <= T:
            yield start, end, audio[:, start:end]
        else:
            seg = torch.zeros((1, segment_size), device=audio.device)
            L = T - start
            seg[0, :L] = audio[:, start:start + L]
            yield start, T, seg
            break
        start += hop_size


def overlap_add(out_buf, seg_out, start, end, window):
    seg_len = end - start
    out_buf[:, start:end] += seg_out[:, :seg_len] * window[:seg_len].unsqueeze(0)
    return out_buf


# -------------------------------------------------------
# Main Inference Logic
# -------------------------------------------------------

def run_inference(cfg):
    inf = cfg["inference"]
    model_cfg = cfg["model"]
    audio_cfg = cfg["audio"]
    output_cfg = cfg["output"]
    runtime_cfg = cfg["runtime"]

    verbose = runtime_cfg.get("verbose", True)

    # -----------------------------
    # Device setup
    # -----------------------------
    device = torch.device(
        "cpu" if inf.get("force_cpu", False) else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if verbose:
        print(f"[INFO] Using device: {device}")

    # -----------------------------
    # Instantiate model
    # -----------------------------
    if verbose:
        print("[INFO] Instantiating CleanUNetLightningModule...")

    model = CleanUNetLightningModule(Namespace(**model_cfg))
    model.to(device)
    model.eval()

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    ckpt_path = inf["checkpoint_path"]
    if verbose:
        print(f"[INFO] Loading checkpoint: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    # -----------------------------
    # Collect files
    # -----------------------------
    input_dir = inf["input_dir"]
    pattern = inf.get("input_pattern", "*.wav")

    files = sorted(glob(os.path.join(input_dir, pattern)))
    if len(files) == 0:
        print("[WARN] No input WAV files found.")
        return

    out_dir = inf.get("output_dir", "denoised_results")
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[INFO] Found {len(files)} files.")

    # -----------------------------
    # Audio params
    # -----------------------------
    target_sr = audio_cfg["target_sample_rate"]
    normalize_flag = audio_cfg.get("normalize", True)

    segment_size = 16384
    hop_size = segment_size // 2

    # STFT params (matches training)
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    stft_window = torch.hann_window(n_fft).to(device)

    window_ola = torch.hann_window(segment_size).to(device)

    # Metrics report
    save_metrics = runtime_cfg.get("save_metrics_report", False)
    metrics_output = {}

    # AMP context
    use_amp = inf.get("use_amp", True)
    amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16

    # -----------------------------
    # PROCESS FILES
    # -----------------------------
    for wav_path in tqdm(files, desc="Inference"):
        try:
            wav, orig_sr = torchaudio.load(wav_path)
            wav = wav.to(device)

            wav = mono_and_resample(wav, orig_sr, target_sr, device)
            raw_peak = wav.abs().max()

            if normalize_flag:
                wav = normalize_audio(wav)

            T = wav.shape[-1]
            out_buf = torch.zeros((1, T), device=device)
            weight_buf = torch.zeros((1, T), device=device)

            for start, end, seg in sliding_windows(wav, segment_size, hop_size):

                # Compute STFT magnitude spectrogram
                spec = torch.stft(
                    seg.squeeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=stft_window,
                    return_complex=True
                ).abs()
                spec = spec.unsqueeze(0)  # [1, F, frames]

                wav_in = seg.unsqueeze(0)  # [1,1,T]

                # Mixed precision inference
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        enhanced, _ = model(wav_in, spec)

                if enhanced.dim() == 2:
                    enhanced = enhanced.unsqueeze(1)

                enhanced = enhanced.squeeze(0)  # [1,T]

                seg_len = end - start
                w = window_ola[:seg_len]
                out_buf = overlap_add(out_buf, enhanced[:, :seg_len], start, end, w)
                weight_buf[:, start:end] += w.unsqueeze(0)

            mask = weight_buf > 1e-8
            out_buf[mask] /= weight_buf[mask]

            enhanced = out_buf[:, :T]

            if normalize_flag and output_cfg.get("undo_normalize", True):
                enhanced = undo_normalize(enhanced, raw_peak)

            enhanced = enhanced.squeeze(0).cpu()

            if orig_sr != target_sr:
                enhanced = torchaudio.transforms.Resample(target_sr, orig_sr)(enhanced.unsqueeze(0)).squeeze(0)

            # Save file
            out_path = os.path.join(out_dir, Path(wav_path).stem + "." + output_cfg.get("format", "wav"))

            if not inf.get("overwrite", False) and os.path.exists(out_path):
                print(f"[WARN] File exists, skipping: {out_path}")
                continue

            torchaudio.save(out_path, enhanced.unsqueeze(0), orig_sr)

            # Save metrics?
            metrics_output[Path(wav_path).name] = {
                "length": int(T),
                "peak_before": float(raw_peak),
                "peak_after": float(enhanced.abs().max()),
            }

        except Exception as e:
            print(f"[ERROR] Failed processing {wav_path}: {e}")

    # Metrics JSON
    if save_metrics:
        json_path = runtime_cfg.get("metrics_report_path", "inference_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_output, f, indent=2)
        print(f"[INFO] Metrics report saved to: {json_path}")

    print(f"[INFO] Inference complete. Output stored in: {out_dir}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CleanUNet2 inference script")
    p.add_argument("--config", required=True, help="YAML configuration path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.backends.cudnn.benchmark = True
    run_inference(cfg)
