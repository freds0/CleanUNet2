# train_xvector.py
"""
Training entrypoint for CleanUNet2 with X-Vector integration (Two-stage training).

Stage-1: Train with X-Vectors extracted from clean audio
Stage-2: Train to replicate Stage-1 latents without X-Vector extractor

Usage:
    # Stage-1 training
    python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1

    # Stage-2 training (requires Stage-1 checkpoint)
    python train_xvector.py --config configs/train_xvector_vanilla_stage2.yaml --stage stage2
"""

import yaml
import argparse
import logging
import copy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Import Lightning modules for both stages
from lightning_modules.cleanunet_xvector_stage1_module import CleanUNet2Stage1Module
from lightning_modules.cleanunet_xvector_stage2_module import CleanUNet2Stage2Module
from lightning_modules.data_module import CleanUNetDataModule

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_xvector")


def _safe_instantiate_callbacks(callbacks_config: dict):
    """
    Instantiate callbacks from a config dict safely.
    Supported: ModelCheckpoint, EarlyStopping.
    """
    callbacks = []
    for name, cb_cfg in (callbacks_config or {}).items():
        if not isinstance(cb_cfg, dict):
            logger.warning("Callback config for '%s' is not a dict; skipping.", name)
            continue

        cfg = copy.deepcopy(cb_cfg)
        target = cfg.pop("_target_", None)
        if target is None:
            logger.warning("Callback '%s' has no '_target_' field; skipping.", name)
            continue

        if "ModelCheckpoint" in target:
            logger.info("Instantiating ModelCheckpoint for callback '%s'.", name)
            callbacks.append(ModelCheckpoint(**cfg))
        elif "EarlyStopping" in target:
            logger.info("Instantiating EarlyStopping for callback '%s'.", name)
            callbacks.append(EarlyStopping(**cfg))
        else:
            logger.warning("Callback '%s' with target '%s' is not supported.", name, target)
    return callbacks


def _create_single_logger(cfg: dict):
    """Helper to instantiate a single logger based on its _target_."""
    if not cfg:
        return None

    config_copy = copy.deepcopy(cfg)
    target = config_copy.pop("_target_", None)

    if not target:
        return None

    if "TensorBoardLogger" in target:
        logger.info("Instantiating TensorBoardLogger.")
        return TensorBoardLogger(**config_copy)

    elif "WandbLogger" in target:
        logger.info("Instantiating WandbLogger.")
        return WandbLogger(**config_copy)

    else:
        logger.warning(f"Logger target '{target}' not supported.")
        return None


def _safe_instantiate_logger(logger_config: dict):
    """
    Instantiate logger(s) from config safely.
    Supports 'choice' logic: 'tensorboard', 'wandb', or 'both'.
    """
    if not logger_config:
        logger.info("No logger configuration provided; proceeding without logger.")
        return None

    choice = logger_config.get("choice", None)

    if choice:
        choice = choice.lower()
        loggers_list = []

        if choice in ["tensorboard", "both"]:
            tb_conf = logger_config.get("tensorboard")
            if tb_conf:
                l = _create_single_logger(tb_conf)
                if l:
                    loggers_list.append(l)

        if choice in ["wandb", "both"]:
            wb_conf = logger_config.get("wandb")
            if wb_conf:
                l = _create_single_logger(wb_conf)
                if l:
                    loggers_list.append(l)

        if not loggers_list:
            logger.warning(f"Logger choice was '{choice}' but no valid configuration found.")
            return None

        return loggers_list[0] if len(loggers_list) == 1 else loggers_list

    if "_target_" in logger_config:
        return _create_single_logger(logger_config)

    return None


def train(config: dict, stage: str):
    """
    Main training function.

    Args:
        config: configuration dictionary (loaded from YAML).
        stage: training stage ('stage1' or 'stage2').
    """
    # Validate config structure
    if "data" not in config:
        raise KeyError("Missing 'data' section in config.")
    if "trainer" not in config:
        raise KeyError("Missing 'trainer' section in config.")

    # Validate stage
    if stage not in ['stage1', 'stage2']:
        raise ValueError(f"Invalid stage: {stage}. Must be 'stage1' or 'stage2'.")

    logger.info("=" * 80)
    logger.info(f"Starting Two-Stage Training - {stage.upper()}")
    logger.info("=" * 80)

    # Get config sections
    data_cfg = config.get("data", {})

    # Instantiate DataModule
    logger.info("Instantiating data module.")
    data_module = CleanUNetDataModule(**data_cfg)

    # Instantiate Lightning Module based on stage
    if stage == 'stage1':
        logger.info("Instantiating CleanUNet2Stage1Module (with X-Vectors).")
        model = CleanUNet2Stage1Module(config)
    else:  # stage2
        logger.info("Instantiating CleanUNet2Stage2Module (without X-Vectors).")

        # Validate Stage-1 checkpoint
        stage1_ckpt = config.get('stage1_checkpoint')
        if not stage1_ckpt:
            raise ValueError("Stage-2 requires 'stage1_checkpoint' in config.")

        model = CleanUNet2Stage2Module(config)

    # Instantiate callbacks safely
    callbacks = _safe_instantiate_callbacks(config.get("callbacks", {}))

    # Instantiate logger safely
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
    logger.info("Starting training run.")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    logger.info("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CleanUNet2 with X-Vectors (Two-stage training)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=['stage1', 'stage2'],
        required=True,
        help="Training stage: 'stage1' (with X-Vectors) or 'stage2' (without X-Vectors)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config file (YAML)
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    # Set precision hint for tensor cores if available
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("medium")
            logger.info("Set float32 matmul precision to 'medium'.")
        except Exception as e:
            logger.warning("Could not set float32 matmul precision: %s", str(e))

    # Run training
    train(config, args.stage)
