# generate_latents.py
"""
Generate Stage-1 latents from an ALREADY-TRAINED Stage-1 checkpoint, without retraining.

The Stage-1 latents consumed by Stage-2 are written by the Stage-1 module's
`validation_step` (one `val_batch_*.pt` per validation batch). This script loads a
trained Stage-1 checkpoint and runs a single validation pass over the val split, so the
latents reflect the trained model instead of an epoch-1 snapshot. No optimizer/training
step runs.

Usage:
    python generate_latents.py \
        --config configs/config_wavlm_plusplus.json \
        --checkpoint experiments/wavlm_plusplus/checkpoints/stage1/cleanunet-stage1-last.ckpt

    # Custom output dir / device:
    python generate_latents.py --config ... --checkpoint ... \
        --latents-dir stored_latents_stage1 --device cuda
"""

import argparse
import logging

import yaml
from pytorch_lightning import Trainer

from lightning_modules.data_module import CleanUNetDataModule
from lightning_modules.cleanunet_ssl_embeddings_stage1_module import (
    CleanUNet2SSLEmbeddingsStage1Module,
)
from train import apply_stage_overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("generate_latents")


def build_val_data_module(data_cfg: dict) -> CleanUNetDataModule:
    """Build the data module for the validation pass (no augmentation)."""
    return CleanUNetDataModule(
        data_dir=data_cfg.get("data_dir", "."),
        train_list_path=data_cfg.get("train_list_path"),
        val_list_path=data_cfg.get("val_list_path"),
        val_split=data_cfg.get("val_split"),
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 4),
        persistent_workers=data_cfg.get("persistent_workers", False),
        segment_size=data_cfg.get("segment_size"),
        sampling_rate=data_cfg.get("sampling_rate", data_cfg.get("sample_rate", 16000)),
        augmentations=None,  # latents must come from the clean val pass, not augmented audio
        use_preextracted_embeddings=data_cfg.get("use_preextracted_embeddings", False),
        quick_test=data_cfg.get("quick_test", False),
        quick_test_samples=data_cfg.get("quick_test_samples", 30),
        datasets=data_cfg.get("datasets"),
        noise_dir=data_cfg.get("noise_dir"),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Stage-1 latents from a trained Stage-1 checkpoint (no retraining)."
    )
    p.add_argument("--config", required=True, help="Path to the Stage-1 config (JSON or YAML).")
    p.add_argument("--checkpoint", required=True, help="Trained Stage-1 checkpoint (.ckpt).")
    p.add_argument(
        "--latents-dir",
        default=None,
        help="Override output dir for latents (default: config 'latents_dir' or 'stored_latents_stage1').",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Accelerator to use (default: auto).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load config and collapse the stage-1 override block (same path as train.py --stage 1).
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
    config = apply_stage_overrides(config, stage=1)

    if args.latents_dir is not None:
        config["latents_dir"] = args.latents_dir

    # Build the validation data module and the Stage-1 module.
    data_module = build_val_data_module(config.get("data", {}))
    model = CleanUNet2SSLEmbeddingsStage1Module(config)

    # Minimal trainer: no logging, no checkpointing, just a single validation pass.
    trainer_cfg = config.get("trainer", {})
    trainer = Trainer(
        accelerator=args.device,
        devices=trainer_cfg.get("devices", 1) if args.device != "cpu" else 1,
        precision=trainer_cfg.get("precision", "32-true"),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", True),
        num_sanity_val_steps=0,
    )

    logger.info("Running a single validation pass to generate latents from: %s", args.checkpoint)
    # ckpt_path loads the trained Stage-1 weights into `model` before validating.
    trainer.validate(model, datamodule=data_module, ckpt_path=args.checkpoint)

    logger.info("Done. Latents written to: %s", model.latents_dir)
    logger.info("Point the Stage-2 config 'latents_dir' at that directory to train Stage 2.")


if __name__ == "__main__":
    main()
