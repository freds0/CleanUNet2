"""
Configuration module for CleanUNet2 (Wav2Vec2 ONLY).

This module provides utilities for loading, validating, and accessing configurations.

Example usage:
    from configs import load_train_config, load_inference_config

    # Load training config
    train_cfg = load_train_config('configs/train.yaml')
    print(train_cfg.pipeline.stage)

    # Load inference config
    infer_cfg = load_inference_config('configs/inference.yaml')
    print(infer_cfg.checkpoint_path)
"""

from .config import (
    TrainConfig,
    InferenceConfig,
    PipelineConfig,
    ModelConfig,
    DataConfig,
    Wav2Vec2Config,
    LossConfig,
)

import yaml
from pathlib import Path


def load_train_config(config_path: str) -> TrainConfig:
    """Load and validate training configuration from YAML.

    Args:
        config_path: Path to train.yaml configuration file.

    Returns:
        TrainConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    try:
        return TrainConfig(**cfg_dict)
    except TypeError as e:
        raise ValueError(f"Invalid training config: {e}")


def load_inference_config(config_path: str) -> InferenceConfig:
    """Load and validate inference configuration from YAML.

    Args:
        config_path: Path to inference.yaml configuration file.

    Returns:
        InferenceConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    try:
        return InferenceConfig(**cfg_dict)
    except TypeError as e:
        raise ValueError(f"Invalid inference config: {e}")


__all__ = [
    "TrainConfig",
    "InferenceConfig",
    "PipelineConfig",
    "ModelConfig",
    "DataConfig",
    "Wav2Vec2Config",
    "LossConfig",
    "load_train_config",
    "load_inference_config",
]
