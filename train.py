# train.py
"""
Training entrypoint for CleanUNet2 with safer callback/logger instantiation

Usage:
    python train.py --config configs/config.yaml
"""

import yaml
import argparse
from argparse import Namespace
import logging
import copy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

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

def _safe_instantiate_logger(logger_config: dict):
    """
    Instantiate logger from config in a safe way. Currently supports TensorBoardLogger.
    """
    if not logger_config:
        logger.info("No logger configuration provided; proceeding without logger.")
        return None

    cfg = copy.deepcopy(logger_config)
    target = cfg.pop("_target_", None)
    if target is None:
        logger.warning("Logger config missing '_target_' field. Skipping logger creation.")
        return None

    if "TensorBoardLogger" in target:
        logger.info("Instantiating TensorBoardLogger.")
        return TensorBoardLogger(**cfg)
    else:
        raise ValueError(f"Logger '{target}' not supported. Add support in _safe_instantiate_logger.")

def train(config: dict):
    """
    Main training function.

    Args:
        config: configuration dictionary (loaded from YAML).
    """
    # Validate minimal config structure
    if "data" not in config:
        raise KeyError("Missing 'data' section in config.")
    if "trainer" not in config:
        raise KeyError("Missing 'trainer' section in config.")

    # Merge model+data config into hyperparameters for the LightningModule (non-destructive)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    hparams_dict = {**model_cfg, **data_cfg}
    hparams = Namespace(**hparams_dict)

    # Instantiate DataModule and LightningModule
    logger.info("Instantiating data module.")
    data_module = CleanUNetDataModule(**data_cfg)

    logger.info("Instantiating model (CleanUNetLightningModule).")
    model = CleanUNetLightningModule(hparams)

    # Instantiate callbacks safely
    callbacks = _safe_instantiate_callbacks(config.get("callbacks", {}))

    # Instantiate logger safely
    lightning_logger = _safe_instantiate_logger(config.get("logger", {}))

    # Create the Trainer
    logger.info("Creating PyTorch Lightning Trainer.")
    trainer_kwargs = copy.deepcopy(config.get("trainer", {}))
    trainer = Trainer(logger=lightning_logger, callbacks=callbacks, **trainer_kwargs)

    # Resume from checkpoint if configured
    ckpt_path = config.get("resume_from_checkpoint", None)
    if ckpt_path:
        logger.info("Resuming training from checkpoint: %s", ckpt_path)

    # Start training
    logger.info("Starting training run.")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    logger.info("Training finished.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CleanUNet2 using a YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config file (YAML)
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    # Set precision hint for tensor cores if available (optional)
    # torch.set_float32_matmul_precision exists in newer PyTorch versions
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("medium")
            logger.info("Set float32 matmul precision to 'medium'.")
        except Exception as e:
            logger.warning("Could not set float32 matmul precision: %s", str(e))

    # Run training
    train(config)

