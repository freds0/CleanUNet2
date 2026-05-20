#!/usr/bin/env python3
"""
Configuration validation script for CleanUNet2 (WavLM-only).

Validates YAML config files before training or inference.

Usage:
    python scripts/validate_config.py --config configs/train.yaml
    python scripts/validate_config.py --config configs/inference.yaml --type inference
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_train_config, load_inference_config, load_raw_config


def validate_train_config(config_path: str) -> bool:
    """Validate training configuration."""
    print(f"\n[INFO] Validating training config: {config_path}")

    try:
        config = load_train_config(config_path)
        print("✅ Configuration loaded successfully!")

        # Print summary
        print(f"\n[SUMMARY]")
        print(f"  Stage: {config.pipeline.stage}")
        print(f"  Data dir: {config.data.data_dir}")
        print(f"  Model: WavLM ({config.model.wavlm.model_name})")
        print(f"  Embeddings: {config.model.wavlm.embedding_dim}-D")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Max epochs: {config.trainer.max_epochs}")
        print(f"  Learning rate: {config.optimizer.lr}")
        print(f"  Augmentation: {config.data.augmentations.enabled if config.data.augmentations else False}")

        # Validate Stage 2 requirements
        if config.pipeline.stage == 2:
            if not config.pipeline.stage1_checkpoint:
                print("❌ ERROR: Stage 2 requires stage1_checkpoint to be set!")
                return False
            print(f"  Stage 1 checkpoint: {config.pipeline.stage1_checkpoint}")

        print("\n✅ Configuration is valid!")
        return True

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return False
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False


def validate_inference_config(config_path: str) -> bool:
    """Validate inference configuration."""
    print(f"\n[INFO] Validating inference config: {config_path}")

    try:
        config = load_inference_config(config_path)
        print("✅ Configuration loaded successfully!")

        # Print summary
        print(f"\n[SUMMARY]")
        print(f"  Checkpoint: {config.checkpoint_path}")
        print(f"  Input dir: {config.input_output.input_dir}")
        print(f"  Output dir: {config.input_output.output_dir}")
        print(f"  Model: WavLM ({config.wavlm_config.model_name})")
        print(f"  Embeddings: {config.wavlm_config.embedding_dim}-D")
        print(f"  Device: {config.device}")
        print(f"  Batch size: {config.batch_size}")

        print("\n✅ Configuration is valid!")
        return True

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return False
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False


def validate_yaml_syntax(config_path: str) -> bool:
    """Validate YAML syntax only."""
    print(f"\n[INFO] Validating YAML syntax: {config_path}")

    try:
        config = load_raw_config(config_path)
        print("✅ YAML syntax is valid!")

        # Print config sections
        print(f"\n[SECTIONS FOUND]")
        for section in sorted(config.keys()):
            print(f"  - {section}")

        return True

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return False
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate CleanUNet2 (WavLM) configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["train", "inference", "yaml"],
        default="train",
        help="Configuration type to validate (default: train)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CleanUNet2 (WavLM-only) Configuration Validator")
    print("=" * 70)

    if args.type == "train":
        success = validate_train_config(args.config)
    elif args.type == "inference":
        success = validate_inference_config(args.config)
    else:  # yaml
        success = validate_yaml_syntax(args.config)

    print("=" * 70)

    if success:
        print("\n✅ Validation PASSED! Configuration is ready to use.")
        return 0
    else:
        print("\n❌ Validation FAILED! Please fix the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
