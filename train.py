# train.py
"""
Training entrypoint for CleanUNet2 with Wav2Vec2 embeddings (Wav2Vec2 ONLY).
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

from lightning_modules.cleanunet_module import CleanUNetLightningModule
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

def train(config: dict, quick_test: bool = False, quick_test_samples: int = 30, quick_test_epochs: int = 1):
    """
    Main training function with Wav2Vec2 embeddings support.

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

    # Merge model+data config into hyperparameters for the LightningModule
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    hparams_dict = {**model_cfg, **data_cfg}
    hparams = Namespace(**hparams_dict)

    # Instantiate DataModule and LightningModule
    logger.info("Instantiating data module (Wav2Vec2 embeddings pipeline).")
    data_module = CleanUNetDataModule(**data_cfg)

    logger.info("Instantiating model (CleanUNetLightningModule with Wav2Vec2).")
    model = CleanUNetLightningModule(hparams)

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

    # Resume from checkpoint if configured
    ckpt_path = config.get("resume_from_checkpoint", None)
    if ckpt_path:
        logger.info("Resuming training from checkpoint: %s", ckpt_path)

    # Start training
    logger.info("Starting training run (Wav2Vec2 ONLY - XVector removed).")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    if quick_test:
        logger.info("✅ Quick test completed successfully!")
    else:
        logger.info("Training finished.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CleanUNet2 with Wav2Vec2 embeddings (Wav2Vec2 ONLY).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Training stage (1 or 2).")
    parser.add_argument("--quick-test", action="store_true", help="Run quick sanity check (1 epoch, 30 samples).")
    parser.add_argument("--quick-test-samples", type=int, default=30, help="Number of samples for quick test.")
    parser.add_argument("--quick-test-epochs", type=int, default=1, help="Number of epochs for quick test.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load and validate config using new unified config system
    try:
        with open(args.config, "r") as fh:
            config_dict = yaml.safe_load(fh)

        # For now, we'll keep the dict-based approach for backward compatibility
        # In the future, could use: from configs import load_train_config
        config = config_dict
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config YAML: {e}")
        exit(1)

    # Set precision hint for tensor cores if available (optional)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("medium")
            logger.info("Set float32 matmul precision to 'medium'.")
        except Exception as e:
            logger.warning("Could not set float32 matmul precision: %s", str(e))

    # Log pipeline info
    logger.info("=" * 70)
    logger.info("CleanUNet2 Training Pipeline (Wav2Vec2 ONLY - XVector removed)")
    logger.info(f"Stage: {config.get('pipeline', {}).get('stage', args.stage)}")
    logger.info(f"Quick Test Mode: {args.quick_test}")
    logger.info("=" * 70)

    # Run training
    train(config, quick_test=args.quick_test, quick_test_samples=args.quick_test_samples,
          quick_test_epochs=args.quick_test_epochs)
