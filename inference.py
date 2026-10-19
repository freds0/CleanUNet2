#!/usr/bin/env python3
"""
CleanUNet2 - inference script.

Denoises every audio file in a folder. The model processes whole utterances (its cost grows linearly
with the length for the Mamba variants); very long files are processed in overlapping chunks.

Settings come from a YAML file (see configs/inference.yaml):

  inference: input_dir, input_pattern, output_dir, checkpoint_path, cleanumamba_checkpoint,
             force_cpu, overwrite, use_amp, max_chunk_seconds, streaming
  model:     architecture (YAML path or dict, same format as the training configs)
  audio:     target_sample_rate
  output:    format
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
import yaml
from tqdm import tqdm

from cleanunet.cleanunet2 import CleanUNet2
from config_utils import resolve_architecture


def build_model(cfg: dict, config_dir: str, device: torch.device) -> CleanUNet2:
    """Build CleanUNet2 from the inference config and load the CleanUMamba or CleanUNet2 weights."""
    inf = cfg["inference"]
    arch = resolve_architecture(cfg["model"]["architecture"], config_dir)
    model = CleanUNet2(**arch)
    if inf.get("cleanumamba_checkpoint"):
        model.load_cleanumamba_weights(inf["cleanumamba_checkpoint"])
    if inf.get("checkpoint_path"):
        model.load_cleanunet2_weights(inf["checkpoint_path"])
    return model.to(device).eval()


@torch.no_grad()
def denoise(model: CleanUNet2, wav: torch.Tensor, sample_rate: int, max_chunk_seconds: float,
            use_amp: bool, streaming: bool) -> torch.Tensor:
    """wav: (L,) on the model device -> (L,)."""
    if streaming:
        from cleanunet.streaming import stream_denoise
        return stream_denoise(model.clean_unet, wav.view(1, -1)).view(-1)

    def run(x):
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=use_amp and x.device.type == "cuda"):
            return model(x.view(1, 1, -1))["wav"].float().view(-1)

    L = wav.shape[-1]
    chunk = int(max_chunk_seconds * sample_rate)
    if L <= chunk:
        return run(wav)
    # Overlap-add with linear cross-fades for very long files
    hop, fade = chunk // 2, chunk // 2
    out = torch.zeros(L, device=wav.device)
    weight = torch.zeros(L, device=wav.device)
    for start in range(0, L, hop):
        seg = wav[start:start + chunk]
        y = run(seg)
        w = torch.ones_like(y)
        n = min(fade, len(y))
        if start > 0:
            w[:n] = torch.linspace(0, 1, n, device=wav.device)
        if start + chunk < L:
            w[-n:] = torch.minimum(w[-n:], torch.linspace(1, 0, n, device=wav.device))
        out[start:start + len(y)] += y * w
        weight[start:start + len(y)] += w
        if start + chunk >= L:
            break
    return out / weight.clamp(min=1e-8)


def run_inference(cfg: dict, config_dir: str = "."):
    """Denoise every audio file of ``inference.input_dir`` (offline or streaming) and write the outputs."""
    inf, audio_cfg, out_cfg = cfg["inference"], cfg.get("audio", {}), cfg.get("output", {})
    device = torch.device("cpu" if inf.get("force_cpu", False) or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] Using device: {device}")
    model = build_model(cfg, config_dir, device)
    if inf.get("streaming") and model.use_spec_branch:
        raise ValueError("streaming=true only supports waveform-only models (use_spec_branch: false)")

    files = sorted(glob(os.path.join(inf["input_dir"], inf.get("input_pattern", "*.wav")), recursive=True))
    if not files:
        print("[WARN] No input files found.")
        return
    out_dir = inf.get("output_dir", "denoised_results")
    os.makedirs(out_dir, exist_ok=True)
    target_sr = int(audio_cfg.get("target_sample_rate", 16000))
    fmt = out_cfg.get("format", "wav")
    report = {}

    for path in tqdm(files, desc="Denoising"):
        out_path = os.path.join(out_dir, f"{Path(path).stem}.{fmt}")
        if os.path.exists(out_path) and not inf.get("overwrite", False):
            continue
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1)).to(device)
        if sr != target_sr:
            wav = AF.resample(wav, sr, target_sr)
        enhanced = denoise(model, wav, target_sr, float(inf.get("max_chunk_seconds", 60)),
                           bool(inf.get("use_amp", False)), bool(inf.get("streaming", False)))
        if sr != target_sr:
            enhanced = AF.resample(enhanced, target_sr, sr)
        sf.write(out_path, enhanced.cpu().numpy(), sr)
        report[Path(path).name] = {"seconds": len(data) / sr, "peak_in": float(data.max()),
                                   "peak_out": float(enhanced.abs().max())}

    if cfg.get("runtime", {}).get("save_report", False):
        with open(cfg["runtime"].get("report_path", os.path.join(out_dir, "report.json")), "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"[INFO] Inference complete. Output stored in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanUNet2 inference script")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    args = parser.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    run_inference(config, os.path.dirname(os.path.abspath(args.config)))
