#!/usr/bin/env python3
"""
CleanUNet2 with X-Vectors - Inference Script (Stage-2 Model)
-------------------------------------------------------------
Inference using Stage-2 trained model (no X-Vector extractor needed).

The Stage-2 model learned to replicate Stage-1's latent vectors without
using the X-Vector extractor, enabling fast inference while maintaining
the benefits of speaker information learned during training.

Usage:
    python inference_xvector.py --config configs/inference_xvector.yaml
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

# Import Stage-2 Lightning Module
from lightning_modules.cleanunet_xvector_stage2_module import CleanUNet2Stage2Module


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def load_checkpoint(path, device):
    """Load checkpoint from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
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
    """Overlap-add reconstruction."""
    seg_len = end - start
    out_buf[:, start:end] += seg_out[:, :seg_len] * window[:seg_len].unsqueeze(0)
    return out_buf


# -------------------------------------------------------
# Main Inference Logic
# -------------------------------------------------------

def run_inference(cfg):
    """Run inference on audio files using Stage-2 model."""
    inf = cfg["inference"]
    audio_cfg = cfg["audio"]
    output_cfg = cfg["output"]
    runtime_cfg = cfg["runtime"]

    verbose = runtime_cfg.get("verbose", True)

    # Device setup
    device = torch.device(
        "cpu" if inf.get("force_cpu", False) else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if verbose:
        print("=" * 80)
        print("CleanUNet2 with X-Vectors - Inference (Stage-2 Model)")
        print("=" * 80)
        print(f"[INFO] Using device: {device}")

    # Load checkpoint
    ckpt_path = inf["checkpoint_path"]
    if verbose:
        print(f"[INFO] Loading Stage-2 checkpoint: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, device)

    # Instantiate model from checkpoint
    if verbose:
        print("[INFO] Instantiating CleanUNet2Stage2Module...")

    model = CleanUNet2Stage2Module.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        strict=False
    )
    model.to(device)
    model.eval()

    if verbose:
        print("[INFO] Model loaded successfully!")
        print("[INFO] Stage-2 model does NOT use X-Vector extractor during inference.")

    # Collect files
    input_dir = inf["input_dir"]
    pattern = inf.get("input_pattern", "*.wav")

    files = sorted(glob(os.path.join(input_dir, pattern)))
    if len(files) == 0:
        print("[WARN] No input WAV files found.")
        return

    out_dir = inf.get("output_dir", "denoised_xvector_results")
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[INFO] Found {len(files)} files to process.")
        print(f"[INFO] Output directory: {out_dir}")

    # Audio parameters
    target_sr = audio_cfg["target_sample_rate"]
    normalize_flag = audio_cfg.get("normalize", True)

    segment_size = audio_cfg.get("segment_size", 16384)
    hop_size = segment_size // 2

    # STFT parameters (must match training)
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    stft_window = torch.hann_window(n_fft).to(device)

    window_ola = torch.hann_window(segment_size).to(device)

    # Metrics
    save_metrics = runtime_cfg.get("save_metrics_report", False)
    metrics_output = {}

    # AMP context
    use_amp = inf.get("use_amp", True)
    amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16

    # Process files
    if verbose:
        print("\n[INFO] Starting inference...")

    for wav_path in tqdm(files, desc="Processing"):
        try:
            # Load audio
            wav, orig_sr = torchaudio.load(wav_path)
            wav = wav.to(device)

            # Preprocess
            wav = mono_and_resample(wav, orig_sr, target_sr, device)
            raw_peak = wav.abs().max()

            if normalize_flag:
                wav = normalize_audio(wav)

            T = wav.shape[-1]
            out_buf = torch.zeros((1, T), device=device)
            weight_buf = torch.zeros((1, T), device=device)

            # Process in sliding windows
            for start, end, seg in sliding_windows(wav, segment_size, hop_size):
                # Compute STFT spectrogram
                spec = torch.stft(
                    seg.squeeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=stft_window,
                    return_complex=True
                ).abs()
                spec = spec.unsqueeze(0)  # [1, F, frames]

                wav_in = seg.unsqueeze(0)  # [1, 1, T]

                # Inference (no X-Vectors needed!)
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp
                    ):
                        enhanced, _ = model(wav_in, spec)

                if enhanced.dim() == 2:
                    enhanced = enhanced.unsqueeze(1)

                enhanced = enhanced.squeeze(0)  # [1, T]

                # Overlap-add
                seg_len = end - start
                w = window_ola[:seg_len]
                out_buf = overlap_add(out_buf, enhanced[:, :seg_len], start, end, w)
                weight_buf[:, start:end] += w.unsqueeze(0)

            # Normalize overlap-add weights
            mask = weight_buf > 1e-8
            out_buf[mask] /= weight_buf[mask]

            enhanced = out_buf[:, :T]

            # Undo normalization if needed
            if normalize_flag and output_cfg.get("undo_normalize", True):
                enhanced = undo_normalize(enhanced, raw_peak)

            enhanced = enhanced.squeeze(0).cpu()

            # Resample back to original sample rate if needed
            if orig_sr != target_sr:
                enhanced = torchaudio.transforms.Resample(
                    target_sr, orig_sr
                )(enhanced.unsqueeze(0)).squeeze(0)

            # Save enhanced audio
            out_filename = Path(wav_path).stem + "_enhanced." + output_cfg.get("format", "wav")
            out_path = os.path.join(out_dir, out_filename)

            if not inf.get("overwrite", False) and os.path.exists(out_path):
                print(f"[WARN] File exists, skipping: {out_path}")
                continue

            torchaudio.save(out_path, enhanced.unsqueeze(0), orig_sr)

            # Collect metrics
            metrics_output[Path(wav_path).name] = {
                "length_samples": int(T),
                "peak_before": float(raw_peak),
                "peak_after": float(enhanced.abs().max()),
            }

        except Exception as e:
            print(f"[ERROR] Failed processing {wav_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save metrics report
    if save_metrics:
        json_path = runtime_cfg.get("metrics_report_path", "inference_xvector_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_output, f, indent=2)
        print(f"\n[INFO] Metrics report saved to: {json_path}")

    print(f"\n[INFO] Inference complete!")
    print(f"[INFO] Enhanced audio saved to: {out_dir}")
    print(f"[INFO] Processed {len(files)} files.")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CleanUNet2 with X-Vectors - Inference Script (Stage-2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference configuration YAML file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Enable cuDNN benchmark for faster inference
    torch.backends.cudnn.benchmark = True

    # Run inference
    run_inference(cfg)
