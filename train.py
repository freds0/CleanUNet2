# train.py
"""
Training entrypoint for CleanUNet2 with SSL embeddings (multi-backbone: wav2vec2 / hubert / wavlm / w2v-bert / whisper).
Supports TensorBoard and WandB logging, and quick test mode for sanity checks.

Usage:
    # Standard training
    python train.py --config configs/train.yaml --stage 1

    # Quick test (1 epoch, 30 samples)
    python train.py --config configs/train.yaml --stage 1 --quick-test
"""

import yaml
import argparse
from argparse import Namespace
import logging
import copy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# SSL-embeddings stage modules are imported lazily in train() per stage.
from lightning_modules.data_module import CleanUNetDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

def _safe_instantiate_callbacks(callbacks_config: dict):
    """
    Instantiate callbacks from a config dict in a safe, explicit manner.
    We do not evaluate arbitrary code or call constructors by string automatically.
    Supported: ModelCheckpoint, EarlyStopping.
    """
    callbacks = []
    for name, cb_cfg in (callbacks_config or {}).items():
        if not isinstance(cb_cfg, dict):
            logger.warning("Callback config for '%s' is not a dict; skipping.", name)
            continue

        cfg = copy.deepcopy(cb_cfg)  # don't mutate original
        target = cfg.pop("_target_", None)
        if target is None:
            logger.warning("Callback '%s' has no '_target_' field; skipping.", name)
            continue

        # Choose known callbacks explicitly (avoid dynamic eval)
        if "ModelCheckpoint" in target:
            logger.info("Instantiating ModelCheckpoint for callback '%s'.", name)
            callbacks.append(ModelCheckpoint(**cfg))
        elif "EarlyStopping" in target:
            logger.info("Instantiating EarlyStopping for callback '%s'.", name)
            callbacks.append(EarlyStopping(**cfg))
        else:
            logger.warning("Callback '%s' with target '%s' is not supported and will be ignored.", name, target)
    return callbacks

def _create_single_logger(cfg: dict):
    """
    Helper to instantiate a single logger based on its _target_.
    """
    if not cfg:
        return None
    
    # Deepcopy to avoid modifying the original config dict
    config_copy = copy.deepcopy(cfg)
    target = config_copy.pop("_target_", None)

    if not target:
        # If no target, we can't instantiate
        return None

    if "TensorBoardLogger" in target:
        logger.info("Instantiating TensorBoardLogger.")
        return TensorBoardLogger(**config_copy)
    
    elif "WandbLogger" in target:
        logger.info("Instantiating WandbLogger.")
        # Ensure 'save_dir' exists or let Wandb handle it
        return WandbLogger(**config_copy)
    
    else:
        logger.warning(f"Logger target '{target}' not supported. Skipping.")
        return None

def _safe_instantiate_logger(logger_config: dict):
    """
    Instantiate logger(s) from config safely.
    Supports 'choice' logic: 'tensorboard', 'wandb', or 'both'.
    """
    if not logger_config:
        logger.info("No logger configuration provided; proceeding without logger.")
        return None

    # Check for 'choice' key to determine strategy
    choice = logger_config.get("choice", None)

    # Strategy 1: 'choice' logic (new structure)
    if choice:
        choice = choice.lower()
        loggers_list = []

        if choice in ["tensorboard", "both"]:
            tb_conf = logger_config.get("tensorboard")
            if tb_conf:
                l = _create_single_logger(tb_conf)
                if l: loggers_list.append(l)
        
        if choice in ["wandb", "both"]:
            wb_conf = logger_config.get("wandb")
            if wb_conf:
                l = _create_single_logger(wb_conf)
                if l: loggers_list.append(l)

        if not loggers_list:
            logger.warning(f"Logger choice was '{choice}' but no valid configuration found.")
            return None
        
        # If only one logger, return it directly; otherwise return list
        return loggers_list[0] if len(loggers_list) == 1 else loggers_list

    # Strategy 2: Direct instantiation (legacy structure with _target_ at root)
    if "_target_" in logger_config:
        return _create_single_logger(logger_config)

    return None

def train(config: dict, quick_test: bool = False, quick_test_samples: int = 30, quick_test_epochs: int = 1, stage: int = None):
    """
    Main training function with SSL embeddings support (backbone selected by model.ssl.type).

    Args:
        config: configuration dictionary (loaded from YAML).
        quick_test: If True, run quick sanity check (1 epoch, small subset).
        quick_test_samples: Number of samples for quick test.
        quick_test_epochs: Number of epochs for quick test.
    """
    # Validate minimal config structure
    if "data" not in config:
        raise KeyError("Missing 'data' section in config.")
    if "trainer" not in config:
        raise KeyError("Missing 'trainer' section in config.")

    # Handle quick test mode
    if quick_test:
        logger.info(f"🚀 QUICK TEST MODE: {quick_test_samples} samples, {quick_test_epochs} epoch(s)")
        config = copy.deepcopy(config)
        config["trainer"]["max_epochs"] = quick_test_epochs
        config["trainer"]["log_every_n_steps"] = 1  # Log every batch for quick test
        config["data"]["quick_test"] = True
        config["data"]["quick_test_samples"] = quick_test_samples

    data_cfg = config.get("data", {})

    # Load augmentation config (from separate file or inline)
    augmentations = None
    aug_config_path = data_cfg.get("augmentation_config")
    if aug_config_path:
        with open(aug_config_path, "r") as f:
            aug_cfg = yaml.safe_load(f)
        if aug_cfg.get("enabled", False):
            augmentations = aug_cfg.get("techniques")
            logger.info(f"Augmentation loaded from {aug_config_path}: {len(augmentations)} techniques")
    else:
        aug_cfg = data_cfg.get("augmentation", {})
        if aug_cfg.get("enabled", False):
            augmentations = aug_cfg.get("techniques")

    data_module_kwargs = {
        "data_dir": data_cfg.get("data_dir", "."),
        "train_list_path": data_cfg.get("train_list_path"),
        "val_list_path": data_cfg.get("val_list_path"),
        "val_split": data_cfg.get("val_split"),
        "batch_size": data_cfg.get("batch_size", 8),
        "num_workers": data_cfg.get("num_workers", 4),
        "persistent_workers": data_cfg.get("persistent_workers", False),
        "segment_size": data_cfg.get("segment_size"),
        "sampling_rate": data_cfg.get("sampling_rate", data_cfg.get("sample_rate", 16000)),
        "augmentations": augmentations,
        "use_preextracted_embeddings": data_cfg.get("use_preextracted_embeddings", True),
        "quick_test": data_cfg.get("quick_test", False),
        "quick_test_samples": data_cfg.get("quick_test_samples", 30),
        # Multi-dataset support
        "datasets": data_cfg.get("datasets"),
        "noise_dir": data_cfg.get("noise_dir"),
        # Stage-2 latent-distillation support
        "return_audio_paths": data_cfg.get("return_audio_paths", False),
        "deterministic_crop": data_cfg.get("deterministic_crop", False),
    }

    # Instantiate DataModule and LightningModule
    logger.info("Instantiating data module (SSL embeddings pipeline).")
    data_module = CleanUNetDataModule(**data_module_kwargs)

    # Select the SSL-embeddings Lightning module based on the training stage.
    current_stage = stage if stage is not None else config.get("pipeline", {}).get("stage", 1)
    if current_stage == 2:
        logger.info("Instantiating CleanUNet2SSLEmbeddingsStage2Module (replicate latents, no extractor).")
        from lightning_modules.cleanunet_ssl_embeddings_stage2_module import CleanUNet2SSLEmbeddingsStage2Module
        model = CleanUNet2SSLEmbeddingsStage2Module(config)
    else:
        logger.info("Instantiating CleanUNet2SSLEmbeddingsStage1Module (SSL embeddings, learnable layer softmax).")
        from lightning_modules.cleanunet_ssl_embeddings_stage1_module import CleanUNet2SSLEmbeddingsStage1Module
        model = CleanUNet2SSLEmbeddingsStage1Module(config)

    # Instantiate callbacks safely
    callbacks = _safe_instantiate_callbacks(config.get("callbacks", {}))

    # Instantiate logger safely (supports TB, WandB, or Both)
    lightning_logger = _safe_instantiate_logger(config.get("logger", {}))

    # Create the Trainer
    logger.info("Creating PyTorch Lightning Trainer.")
    trainer_kwargs = copy.deepcopy(config.get("trainer", {}))

    trainer = Trainer(
        logger=lightning_logger,
        callbacks=callbacks,
        **trainer_kwargs
    )

    # Resume from checkpoint if configured. Prefer pipeline.resume_from_checkpoint
    # (where the configs place it and where the startup log reads it from); fall
    # back to a top-level key for backward compatibility.
    ckpt_path = (
        config.get("pipeline", {}).get("resume_from_checkpoint")
        or config.get("resume_from_checkpoint")
    )
    if ckpt_path:
        logger.info("Resuming training from checkpoint: %s", ckpt_path)

    # Start training
    logger.info("Starting training run (SSL embeddings, multi-backbone).")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    if quick_test:
        logger.info("✅ Quick test completed successfully!")
    else:
        logger.info("Training finished.")

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins on leaves)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_stage_overrides(config: dict, stage: int) -> dict:
    """
    Collapse a single-file-per-variant config into a flat config for `stage`.

    The config carries shared sections (model/loss/data/...) plus optional
    `stage1` and `stage2` sub-sections holding per-stage overrides. Both are
    popped so they never leak into the trainer, and the selected one is
    deep-merged onto the shared base.
    """
    config = copy.deepcopy(config)
    stage1_over = config.pop('stage1', {})
    stage2_over = config.pop('stage2', {})
    overrides = stage1_over if stage == 1 else stage2_over
    return _deep_merge(config, overrides)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CleanUNet2 with SSL embeddings (multi-backbone; +/++ layer fusion).")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON or YAML).")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Training stage (1 or 2).")
    parser.add_argument("--quick-test", action="store_true", help="Run quick sanity check (1 epoch, 30 samples).")
    parser.add_argument("--quick-test-samples", type=int, default=30, help="Number of samples for quick test.")
    parser.add_argument("--quick-test-epochs", type=int, default=1, help="Number of epochs for quick test.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config. yaml.safe_load parses both YAML and JSON (JSON is a subset),
    # so the per-variant config_<model>_{plus,plusplus}.json files load directly.
    try:
        with open(args.config, "r") as fh:
            config_dict = yaml.safe_load(fh)
        config = config_dict
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config (YAML/JSON): {e}")
        exit(1)

    # Determine stage (explicit pipeline.stage overrides the CLI flag), then collapse
    # the stage1/stage2 override sub-sections onto the shared base for that stage.
    stage = config.get('pipeline', {}).get('stage', args.stage)
    config = apply_stage_overrides(config, stage)

    # Set precision hint for tensor cores if available (optional)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("medium")
            logger.info("Set float32 matmul precision to 'medium'.")
        except Exception as e:
            logger.warning("Could not set float32 matmul precision: %s", str(e))

    # Log pipeline info
    logger.info("=" * 70)
    logger.info("CleanUNet2 Training Pipeline (SSL embeddings, multi-backbone)")
    logger.info("=" * 70)

    # Get stage and checkpoint info
    pipeline_cfg = config.get('pipeline', {})
    stage = pipeline_cfg.get('stage', args.stage)
    resume_ckpt = pipeline_cfg.get('resume_from_checkpoint')
    stage1_ckpt = pipeline_cfg.get('stage1_checkpoint')

    logger.info(f"🎯 Training Stage: {stage}")
    logger.info(f"🧪 Quick Test Mode: {args.quick_test}")

    # Checkpoint loading information
    if stage == 1:
        logger.info("=" * 70)
        logger.info("STAGE 1: Training WITH pre-extracted SSL embeddings")
        logger.info("=" * 70)
        if resume_ckpt:
            logger.info(f"📂 Will RESUME from existing checkpoint: {resume_ckpt}")
        else:
            logger.info("🆕 Starting fresh (no resume checkpoint)")
        logger.info(f"📊 Will save checkpoints to: experiments/checkpoints/stage1/")
        logger.info(f"💾 Latest checkpoint will be: cleanunet-stage1-last.ckpt")
    elif stage == 2:
        logger.info("=" * 70)
        logger.info("STAGE 2: Training WITHOUT pre-extracted embeddings")
        logger.info("=" * 70)
        if stage1_ckpt:
            logger.info(f"✅ Will LOAD Stage 1 checkpoint: {stage1_ckpt}")
            logger.info(f"   Stage 1 weights will initialize the model")
            logger.info(f"   Model will then learn to replicate embeddings internally")
        else:
            logger.warning("⚠️  No Stage 1 checkpoint specified!")
            logger.warning("⚠️  Starting with random initialization")
        if resume_ckpt:
            logger.info(f"📂 Will RESUME from existing Stage 2 checkpoint: {resume_ckpt}")
        logger.info(f"📊 Will save checkpoints to: experiments/checkpoints/stage2/")
        logger.info(f"💾 Latest checkpoint will be: cleanunet-stage2-last.ckpt")

    logger.info("=" * 70)

    # Run training
    train(config, quick_test=args.quick_test, quick_test_samples=args.quick_test_samples,
          quick_test_epochs=args.quick_test_epochs, stage=stage)
