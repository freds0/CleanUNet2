# generate_train_latents.py
"""
Generate per-file Stage-1 fused latents for the TRAIN split, from a trained Stage-1
checkpoint, for use as distillation targets in Stage-2 training.

Unlike generate_latents.py (which dumps per-batch val latents keyed by batch index), this
writes ONE file per training sample, keyed by MD5 of the clean-audio path:

    <output-dir>/<md5(clean_path)>.pt   # CPU tensor, shape (C, T) = fused_latent

Alignment requirement: the latent depends on the exact audio segment. This script crops
deterministically (per file path), and Stage-2 training must do the same
(`data.deterministic_crop: true` + `data.return_audio_paths: true`) so each sample matches
its cached target. Run with the SAME config used for Stage-1 (same SSL backbone/variant).

Usage:
    python generate_train_latents.py \
        --config configs/config_wavlm_plusplus.json \
        --checkpoint experiments/wavlm_plusplus/checkpoints/stage1/cleanunet-stage1-last.ckpt
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from spec_dataset import MelDataset, custom_collate_fn, latent_cache_key
from lightning_modules.cleanunet_ssl_embeddings_stage1_module import (
    CleanUNet2SSLEmbeddingsStage1Module,
)
from train import apply_stage_overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("generate_train_latents")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-file Stage-1 train latents (distillation targets) from a trained checkpoint."
    )
    p.add_argument("--config", required=True, help="Stage-1 config (JSON or YAML).")
    p.add_argument("--checkpoint", required=True, help="Trained Stage-1 checkpoint (.ckpt).")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write <md5>.pt files (default: config 'train_latents_dir' or 'train_latents_stage1').",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device (default: cuda).")
    p.add_argument("--force", action="store_true", help="Re-extract even if a latent file already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
    # Collapse the stage-1 override block (same path train.py --stage 1 uses).
    config = apply_stage_overrides(config, stage=1)

    out_dir = Path(args.output_dir or config.get("train_latents_dir", "train_latents_stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        logger.warning("CUDA not available; falling back to CPU.")

    # Build the Stage-1 module (backbone selected by model.ssl.type) and load weights.
    module = CleanUNet2SSLEmbeddingsStage1Module(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    module.load_state_dict(state_dict, strict=False)
    model = module.model.to(device).eval()

    # Train dataset with the SAME deterministic crop Stage-2 training will use, real
    # paired noisy audio (no augmentation), and clean paths for keying.
    data_cfg = config.get("data", {})
    dataset = MelDataset(
        data_dir=data_cfg.get("data_dir", "."),
        data_files=data_cfg.get("train_list_path"),
        segment_size=data_cfg.get("segment_size", 32000),
        sampling_rate=data_cfg.get("sampling_rate", 16000),
        return_audio_paths=True,
        deterministic_crop=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=custom_collate_fn,
    )

    logger.info("Generating train latents for %d samples -> %s", len(dataset), out_dir)
    extracted, skipped = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Train latents"):
            noisy_wav, noisy_spec, clean_wav, clean_spec, clean_paths = batch

            pending = [p for p in clean_paths
                       if args.force or not (out_dir / f"{latent_cache_key(p)}.pt").exists()]
            if not pending:
                skipped += len(clean_paths)
                continue

            noisy_wav = noisy_wav.to(device)
            noisy_spec = noisy_spec.to(device)
            clean_wav = clean_wav.to(device)

            _, _, latents = model(
                noisy_wav, noisy_spec,
                clean_audio=clean_wav,
                clean_audio_paths=clean_paths,
                return_latents=True,
            )
            fused = latents["fused_latent"].cpu()  # (B, C, T)

            for i, path in enumerate(clean_paths):
                cache_file = out_dir / f"{latent_cache_key(path)}.pt"
                if cache_file.exists() and not args.force:
                    skipped += 1
                    continue
                torch.save(fused[i].contiguous(), cache_file)
                extracted += 1

    logger.info("Done. Extracted: %d, skipped(existing): %d. Cache dir: %s",
                extracted, skipped, out_dir)
    logger.info("Set the Stage-2 config: train_latents_dir + data.deterministic_crop: true "
                "+ data.return_audio_paths: true")


if __name__ == "__main__":
    main()
