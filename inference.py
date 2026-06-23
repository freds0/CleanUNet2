#!/usr/bin/env python3
"""
CleanUNet2 – Stage-2 SSL Inference Script
-----------------------------------------
Runs a trained Stage-2 SSL-embeddings checkpoint over a set of noisy files.

It reads the SAME per-stage training config used to produce the checkpoint
(e.g. configs/config_hubert_plusplus_stage2.json), so the model architecture
(CleanUNet + SSL integration/latent-predictor dims) matches the weights. The
SSL extractor is NOT loaded — Stage-2 runs the latent_predictor only.

The checkpoint and output directory are passed on the CLI:

    python inference.py \
        --config configs/config_hubert_plusplus_stage2.json \
        --checkpoint experiments/hubert_plusplus/checkpoints/stage2/<ckpt>.ckpt \
        --output-dir results_hubert++

Input files default to the noisy column of the config's data.val_list_path
(resolved against data.data_dir); pass --input-dir to denoise a raw folder of
*.wav instead.
"""

import os
from pathlib import Path
from glob import glob
import argparse
import yaml

import torch
import torchaudio
from tqdm import tqdm

from cleanunet.cleanunet2_with_ssl_embeddings import CleanUNet2WithSSLEmbeddings
from cleanunet.ssl_extractor_factory import ssl_args_from_config
from spec_dataset import get_dataset_filelist


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    return ckpt


def build_stage2_model(cfg, checkpoint_path, device, verbose=True):
    """Build the Stage-2 SSL model and load a Lightning checkpoint into it.

    Mirrors how CleanUNet2SSLEmbeddingsStage2Module builds the model, so the
    architecture matches the trained weights. The SSL extractor is never built
    in Stage 2, so no backbone is downloaded here.
    """
    model_config = cfg.get("model", {})
    model_args = {
        "stage": "stage2",
        "conditioning_type": model_config.get("conditioning_type", "addition"),
        "cleanunet_params": model_config.get("cleanunet_params", {}),
        "cleanspecnet_params": model_config.get("cleanspecnet_params", {}),
        **ssl_args_from_config(model_config),
    }
    model = CleanUNet2WithSSLEmbeddings(**model_args).to(device).eval()

    if verbose:
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path, device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Strip the LightningModule 'model.' prefix to match the bare nn.Module.
    stripped = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if verbose:
        print(f"[INFO] Loaded weights (missing={len(missing)}, unexpected={len(unexpected)}).")
    return model


def collect_input_files(cfg, input_dir, pattern):
    """Return the list of noisy files to denoise.

    --input-dir overrides; otherwise use the noisy column of the config's
    data.val_list_path, resolved against data.data_dir.
    """
    if input_dir:
        return sorted(glob(os.path.join(input_dir, pattern)))

    data_cfg = cfg.get("data", {})
    list_path = data_cfg.get("val_list_path")
    data_dir = data_cfg.get("data_dir", ".")
    if not list_path:
        raise ValueError(
            "No --input-dir given and config has no data.val_list_path to fall back on."
        )
    pairs = get_dataset_filelist(list_path)  # [(clean_rel, noisy_rel), ...]
    return [os.path.join(data_dir, noisy_rel) for _clean_rel, noisy_rel in pairs]


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

def run_inference(cfg, checkpoint_path, output_dir, input_dir=None,
                  device_str="cuda", input_pattern="*.wav", use_amp=True, verbose=True):
    output_cfg = cfg.get("output", {})

    # -----------------------------
    # Device setup
    # -----------------------------
    device = torch.device(
        device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if device_str == "cuda" and device.type == "cpu":
        print("[WARN] CUDA not available; falling back to CPU.")
    if verbose:
        print(f"[INFO] Using device: {device}")

    # -----------------------------
    # Instantiate model + load checkpoint
    # -----------------------------
    if verbose:
        print("[INFO] Building Stage-2 SSL model...")
    model = build_stage2_model(cfg, checkpoint_path, device, verbose=verbose)

    # -----------------------------
    # Collect files
    # -----------------------------
    files = collect_input_files(cfg, input_dir, input_pattern)
    if len(files) == 0:
        print("[WARN] No input WAV files found.")
        return

    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[INFO] Found {len(files)} files.")

    # -----------------------------
    # Audio params
    # -----------------------------
    target_sr = cfg.get("data", {}).get("sampling_rate", 16000)
    normalize_flag = True

    segment_size = 16384
    hop_size = segment_size // 2

    # STFT params (matches training)
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    stft_window = torch.hann_window(n_fft).to(device)

    window_ola = torch.hann_window(segment_size).to(device)

    # AMP context
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
            torchaudio.save(out_path, enhanced.unsqueeze(0), orig_sr)

        except Exception as e:
            print(f"[ERROR] Failed processing {wav_path}: {e}")

    print(f"[INFO] Inference complete. Output stored in: {out_dir}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CleanUNet2 Stage-2 SSL inference script")
    p.add_argument("--config", required=True,
                   help="Per-stage training config (JSON/YAML) used to build the model.")
    p.add_argument("--checkpoint", required=True, help="Trained Stage-2 checkpoint (.ckpt).")
    p.add_argument("--output-dir", required=True, help="Directory to write denoised audio.")
    p.add_argument("--input-dir", default=None,
                   help="Folder of noisy *.wav to denoise. Default: noisy files from the "
                        "config's data.val_list_path (resolved against data.data_dir).")
    p.add_argument("--input-pattern", default="*.wav", help="Glob pattern for --input-dir.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device (default: cuda).")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed-precision inference.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.backends.cudnn.benchmark = True
    run_inference(
        cfg,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        device_str=args.device,
        input_pattern=args.input_pattern,
        use_amp=not args.no_amp,
    )
