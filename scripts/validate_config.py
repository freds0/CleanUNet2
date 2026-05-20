#!/usr/bin/env python3
"""
Configuration validation script for CleanUNet2 (Wav2Vec2 ONLY).

Validates that the YAML configuration can be loaded, parsed, and satisfies
all type constraints and validation rules.

Usage:
    python scripts/validate_config.py --config configs/train.yaml
    python scripts/validate_config.py --config configs/inference.yaml
"""

import sys
import yaml
import argparse
from pathlib import Path


def validate_train_config(config_path: str) -> bool:
    """Validate training configuration."""
    print(f"Validating training config: {config_path}")
    print("=" * 70)

    try:
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        # Check required sections
        required_sections = ['pipeline', 'trainer', 'model', 'data', 'losses']
        for section in required_sections:
            if section not in cfg_dict:
                print(f"❌ Missing required section: {section}")
                return False
            print(f"✅ Section '{section}' found")

        # Validate pipeline.stage
        pipeline = cfg_dict.get('pipeline', {})
        stage = pipeline.get('stage')
        if stage not in [1, 2]:
            print(f"❌ Invalid pipeline.stage: {stage} (must be 1 or 2)")
            return False
        print(f"✅ Pipeline stage: {stage}")

        # Stage 2 validation
        if stage == 2:
            if 'stage1_checkpoint' not in cfg_dict or not cfg_dict['stage1_checkpoint']:
                print("❌ Stage 2 requires 'stage1_checkpoint' to be set")
                return False
            print(f"✅ Stage 2 checkpoint configured")

        # Validate model configuration
        model = cfg_dict.get('model', {})
        if 'embeddings' not in model:
            print("❌ Model missing 'embeddings' configuration (Wav2Vec2 required)")
            return False
        embeddings = model['embeddings']
        if embeddings.get('input_dim') != 1024:
            print(f"⚠️  Wav2Vec2 input_dim is {embeddings.get('input_dim')}, expected 1024")
        print("✅ Model configuration valid (Wav2Vec2)")

        # Validate data configuration
        data = cfg_dict.get('data', {})
        if 'data_dir' not in data:
            print("⚠️  data.data_dir not set (required for actual training)")
        else:
            print(f"✅ data_dir: {data['data_dir']}")

        print("\n" + "=" * 70)
        print("✅ Training config validation PASSED!")
        return True

    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"❌ YAML parse error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def validate_inference_config(config_path: str) -> bool:
    """Validate inference configuration."""
    print(f"Validating inference config: {config_path}")
    print("=" * 70)

    try:
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        # Check required fields
        if 'checkpoint_path' not in cfg_dict:
            print("⚠️  checkpoint_path not set (required for actual inference)")
        else:
            print(f"✅ checkpoint_path: {cfg_dict['checkpoint_path']}")

        # Check input/output
        if 'input_output' in cfg_dict:
            io_cfg = cfg_dict['input_output']
            print(f"✅ Input: {io_cfg.get('input_dir', 'N/A')}")
            print(f"✅ Output: {io_cfg.get('output_dir', 'N/A')}")

        # Check embeddings
        if 'embeddings' in cfg_dict:
            emb = cfg_dict['embeddings']
            print(f"✅ Embeddings: {emb.get('model_name')} (layer {emb.get('layer')})")

        print("\n" + "=" * 70)
        print("✅ Inference config validation PASSED!")
        return True

    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"❌ YAML parse error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate CleanUNet2 configurations")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--type", type=str, choices=['train', 'inference'], default='train',
                        help="Configuration type (default: train)")
    args = parser.parse_args()

    if args.type == 'train':
        success = validate_train_config(args.config)
    else:
        success = validate_inference_config(args.config)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
