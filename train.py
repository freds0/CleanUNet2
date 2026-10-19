# train.py
"""
Training entrypoint for CleanUNet2 (PyTorch Lightning). Supports TensorBoard and WandB.

Usage:
    python train.py --config configs/train_stageA_cleanunet_mamba.yaml
    python train.py --config configs/train.yaml trainer.max_steps=1000 data.batch_size=8   # dotted overrides
"""

import argparse
import copy
import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config_utils import load_config
from lightning_modules.cleanunet_module import CleanUNetLightningModule
from lightning_modules.data_module import CleanUNetDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

_CALLBACKS = {"ModelCheckpoint": ModelCheckpoint, "EarlyStopping": EarlyStopping,
              "LearningRateMonitor": LearningRateMonitor}
_LOGGERS = {"TensorBoardLogger": TensorBoardLogger, "WandbLogger": WandbLogger}


def _instantiate(cfg: dict, registry: dict, kind: str):
    """Instantiate a known class from a config dict with a `_target_` key (no dynamic imports)."""
    cfg = copy.deepcopy(cfg)
    target = cfg.pop("_target_", "")
    for name, cls in registry.items():
        if target.endswith(name):
            return cls(**cfg)
    logger.warning("%s target '%s' is not supported and will be ignored.", kind, target)
    return None


def build_callbacks(callbacks_config: dict):
    """Instantiate the callbacks of the ``callbacks`` config section (ModelCheckpoint, ...)."""
    callbacks = [_instantiate(c, _CALLBACKS, "Callback") for c in (callbacks_config or {}).values()
                 if isinstance(c, dict)]
    return [c for c in callbacks if c is not None]


def build_logger(logger_config: dict):
    """Supports `choice: tensorboard | wandb | both | none`."""
    if not logger_config:
        return None
    choice = str(logger_config.get("choice", "tensorboard")).lower()
    if choice == "none":
        return False
    names = ["tensorboard", "wandb"] if choice == "both" else [choice]
    loggers = [_instantiate(logger_config[n], _LOGGERS, "Logger") for n in names if n in logger_config]
    loggers = [l for l in loggers if l is not None]
    if not loggers:
        return None
    return loggers[0] if len(loggers) == 1 else loggers


def train(config: dict):
    """Train with a resolved config: seed, data module, Lightning module, logger, callbacks and Trainer.

    Resumes from ``resume_from_checkpoint`` when that file exists (a missing file starts a new run).
    """
    for section in ("data", "trainer", "model"):
        if section not in config:
            raise KeyError(f"Missing '{section}' section in config.")
    pl.seed_everything(int(config.get("seed", 1234)), workers=True)

    data_module = CleanUNetDataModule(**config["data"])
    model = CleanUNetLightningModule(config)

    lightning_logger = build_logger(config.get("logger"))
    callbacks = build_callbacks(config.get("callbacks"))
    if not lightning_logger:  # LearningRateMonitor needs a logger
        callbacks = [c for c in callbacks if not isinstance(c, LearningRateMonitor)]
    trainer_cfg = copy.deepcopy(config["trainer"])
    if not model.automatic_optimization:
        # manual optimization (e.g. the metric discriminator): the module clips its own gradients
        trainer_cfg.pop("gradient_clip_val", None)
        trainer_cfg.pop("gradient_clip_algorithm", None)
    trainer = Trainer(logger=lightning_logger, callbacks=callbacks, **trainer_cfg)

    ckpt_path = config.get("resume_from_checkpoint")
    if ckpt_path and not os.path.exists(ckpt_path):
        logger.info("Checkpoint '%s' not found; starting a new run.", ckpt_path)
        ckpt_path = None
    elif ckpt_path:
        logger.info("Resuming training from checkpoint: %s", ckpt_path)

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    logger.info("Training finished. Best checkpoint: %s",
                getattr(trainer.checkpoint_callback, "best_model_path", None))
    return trainer


def parse_args():
    """``--config`` plus dotted ``key=value`` overrides."""
    parser = argparse.ArgumentParser(description="Train CleanUNet2 using a YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. trainer.max_steps=1000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    torch.set_float32_matmul_precision("high")
    train(cfg)
