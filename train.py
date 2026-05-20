# train.py
"""
Training entrypoint for CleanUNet2 with safer callback/logger instantiation.
Supports TensorBoard and WandB.

Usage:
    python train.py --config configs/train.yaml
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

# Make sure these imports point to the correct modules in your repo
from lightning_modules.cleanunet_module import CleanUNetLightningModule
from lightning_modules.data_module import CleanUNetDataModule

# Configure a simple logger for console output (INFO level)
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

def train(config: dict, quick_test: bool = False, quick_test_samples: int = 30,
          quick_test_epochs: int = 1, stage: int = None):
    """
    Main training function.

    Args:
        config: configuration dictionary (loaded from YAML).
        quick_test: Run quick test mode (1 epoch, N samples)
        quick_test_samples: Number of samples for quick test
        quick_test_epochs: Number of epochs for quick test
        stage: Override training stage from config (1 or 2)
    """
    # Validate minimal config structure
    if "data" not in config:
        raise KeyError("Missing 'data' section in config.")
    if "trainer" not in config:
        raise KeyError("Missing 'trainer' section in config.")

    # Log run information
    pipeline_cfg = config.get("pipeline", {})
    current_stage = stage if stage else pipeline_cfg.get("stage", 1)
    logger.info(f"[INFO] Stage: {current_stage}")
    logger.info(f"[INFO] Model: WavLM (microsoft/wavlm-large, 768-D)")

    if quick_test:
        logger.info(f"[INFO] Quick Test Mode: True ({quick_test_epochs} epoch, {quick_test_samples} samples)")

    # Merge model+data config into hyperparameters for the LightningModule (non-destructive)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    hparams_dict = {**model_cfg, **data_cfg}
    hparams = Namespace(**hparams_dict)

    # Instantiate DataModule with quick test support
    logger.info("Instantiating data module.")
    data_cfg_copy = data_cfg.copy()
    if quick_test:
        data_cfg_copy["quick_test"] = True
        data_cfg_copy["quick_test_samples"] = quick_test_samples
        # Override batch size for quick test
        data_cfg_copy["batch_size"] = min(data_cfg_copy.get("batch_size", 32), quick_test_samples)
    data_module = CleanUNetDataModule(**data_cfg_copy)

    logger.info("Instantiating model (CleanUNetLightningModule).")
    model = CleanUNetLightningModule(hparams)

    # Instantiate callbacks safely
    callbacks = _safe_instantiate_callbacks(config.get("callbacks", {}))

    # Instantiate logger safely (supports TB, WandB, or Both)
    lightning_logger = _safe_instantiate_logger(config.get("logger", {}))

    # Create the Trainer
    logger.info("Creating PyTorch Lightning Trainer.")
    trainer_kwargs = copy.deepcopy(config.get("trainer", {}))

    # Override max_epochs for quick test
    if quick_test:
        trainer_kwargs["max_epochs"] = quick_test_epochs

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
    logger.info("Starting training run.")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    logger.info("Training finished.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CleanUNet2 (WavLM-only) using a YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test: 1 epoch on 30 samples (<5 min).")
    parser.add_argument("--quick-test-samples", type=int, default=30, help="Number of samples for quick test (default: 30).")
    parser.add_argument("--quick-test-epochs", type=int, default=1, help="Number of epochs for quick test (default: 1).")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Training stage: 1 or 2 (overrides config).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config file (YAML)
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    # Set precision hint for tensor cores if available (optional)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("medium")
            logger.info("Set float32 matmul precision to 'medium'.")
        except Exception as e:
            logger.warning("Could not set float32 matmul precision: %s", str(e))

    # Run training
    train(
        config,
        quick_test=args.quick_test,
        quick_test_samples=args.quick_test_samples,
        quick_test_epochs=args.quick_test_epochs,
        stage=args.stage
    )
