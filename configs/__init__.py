"""Configuration loading and validation utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
from .config import TrainConfig, InferenceConfig, create_train_config_from_dict, validate_config


def load_train_config(config_path: Union[str, Path]) -> TrainConfig:
    """
    Load and validate training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        TrainConfig object with validated parameters

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {str(e)}")

    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    validate_config(config_dict)
    return create_train_config_from_dict(config_dict)


def load_inference_config(config_path: Union[str, Path]) -> InferenceConfig:
    """
    Load and validate inference configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        InferenceConfig object with validated parameters

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {str(e)}")

    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    try:
        return InferenceConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Error creating InferenceConfig: {str(e)}")


def load_raw_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load raw configuration dictionary from YAML file (no validation).

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Raw configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {str(e)}")

    return config_dict if config_dict is not None else {}


__all__ = [
    "load_train_config",
    "load_inference_config",
    "load_raw_config",
    "TrainConfig",
    "InferenceConfig",
]
