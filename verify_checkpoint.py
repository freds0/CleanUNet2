#!/usr/bin/env python3
"""
Checkpoint Verification Tool for CleanUNet2-GAN

This script helps verify that a Vanilla checkpoint is compatible with the GAN training.
It checks the checkpoint structure, lists available keys, and validates architecture compatibility.

Usage:
    python verify_checkpoint.py --checkpoint path/to/checkpoint.ckpt
    python verify_checkpoint.py --checkpoint path/to/checkpoint.ckpt --conditioning addition
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict


def verify_checkpoint(checkpoint_path: str, conditioning_type: str = "addition"):
    """
    Verify a CleanUNet2 checkpoint for compatibility with GAN training.

    Args:
        checkpoint_path: Path to the checkpoint file
        conditioning_type: Expected conditioning method (addition, concatenation, film)
    """
    print("=" * 80)
    print("CleanUNet2 Checkpoint Verification Tool")
    print("=" * 80)
    print(f"\nCheckpoint Path: {checkpoint_path}")
    print(f"Expected Conditioning Type: {conditioning_type}\n")

    # Check if file exists
    if not Path(checkpoint_path).exists():
        print(f"❌ ERROR: Checkpoint file not found at {checkpoint_path}")
        return False

    # Load checkpoint
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully\n")
    except Exception as e:
        print(f"❌ ERROR: Failed to load checkpoint: {e}")
        return False

    # Check checkpoint structure
    print("-" * 80)
    print("Checkpoint Structure:")
    print("-" * 80)

    checkpoint_keys = list(checkpoint.keys())
    print(f"Top-level keys: {checkpoint_keys}\n")

    # Extract state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Found 'state_dict' key (PyTorch Lightning format)")
    elif isinstance(checkpoint, dict) and any(k.startswith('model.') for k in checkpoint.keys()):
        state_dict = checkpoint
        print("✅ Direct state_dict format detected")
    else:
        print("❌ ERROR: Could not find valid state_dict in checkpoint")
        print(f"   Available keys: {checkpoint_keys}")
        return False

    # Analyze model keys
    print("\n" + "-" * 80)
    print("Model Components Analysis:")
    print("-" * 80)

    # Group keys by component
    components = defaultdict(list)
    for key in state_dict.keys():
        # Remove 'model.' prefix for analysis
        clean_key = key.replace('model.', '')

        if clean_key.startswith('clean_unet.'):
            components['CleanUNet'].append(key)
        elif clean_key.startswith('clean_spec_net.'):
            components['CleanSpecNet'].append(key)
        elif clean_key.startswith('conditioner.'):
            components['Conditioner'].append(key)
        elif clean_key.startswith('spec_upsampler.'):
            components['SpecUpsampler'].append(key)
        else:
            components['Other'].append(key)

    # Print component statistics
    print(f"\n{'Component':<20} {'Parameters':<15} {'Status'}")
    print("-" * 60)

    for component, keys in sorted(components.items()):
        status = "✅ Found" if len(keys) > 0 else "❌ Missing"
        print(f"{component:<20} {len(keys):<15} {status}")

    # Show sample keys for each component
    print("\n" + "-" * 80)
    print("Sample Keys (first 3 per component):")
    print("-" * 80)

    for component, keys in sorted(components.items()):
        if len(keys) > 0:
            print(f"\n{component}:")
            for key in keys[:3]:
                clean_key = key.replace('model.', '')
                print(f"  - {clean_key}")
            if len(keys) > 3:
                print(f"  ... and {len(keys) - 3} more")

    # Check conditioning compatibility
    print("\n" + "-" * 80)
    print("Conditioning Compatibility Check:")
    print("-" * 80)

    conditioner_keys = components['Conditioner']

    if conditioning_type == "addition":
        # Addition doesn't require extra parameters
        print(f"✅ Conditioning type '{conditioning_type}' requires no extra parameters")
        compatible = True
    elif conditioning_type == "concatenation":
        # Check for concat_proj
        has_concat = any('concat_proj' in k for k in conditioner_keys)
        if has_concat:
            print(f"✅ Found 'concat_proj' layers for '{conditioning_type}' conditioning")
            compatible = True
        else:
            print(f"⚠️  WARNING: No 'concat_proj' layers found for '{conditioning_type}' conditioning")
            print("   This might cause issues if the checkpoint used a different conditioning type")
            compatible = False
    elif conditioning_type == "film":
        # Check for film_gen
        has_film = any('film_gen' in k for k in conditioner_keys)
        if has_film:
            print(f"✅ Found 'film_gen' layer for '{conditioning_type}' conditioning")
            compatible = True
        else:
            print(f"⚠️  WARNING: No 'film_gen' layer found for '{conditioning_type}' conditioning")
            print("   This might cause issues if the checkpoint used a different conditioning type")
            compatible = False
    else:
        print(f"❌ ERROR: Unknown conditioning type '{conditioning_type}'")
        compatible = False

    # Check for discriminator keys (should not exist in Vanilla checkpoints)
    print("\n" + "-" * 80)
    print("Discriminator Check:")
    print("-" * 80)

    has_discriminator = any('discriminator' in k for k in state_dict.keys())
    if has_discriminator:
        print("⚠️  WARNING: Found discriminator keys in checkpoint")
        print("   This appears to be a GAN checkpoint, not a Vanilla checkpoint")
    else:
        print("✅ No discriminator keys found (expected for Vanilla checkpoint)")

    # Total parameters
    print("\n" + "-" * 80)
    print("Summary:")
    print("-" * 80)

    total_params = sum(p.numel() for p in state_dict.values() if torch.is_tensor(p))
    total_keys = len(state_dict)

    print(f"Total Keys: {total_keys}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Checkpoint Size: {Path(checkpoint_path).stat().st_size / (1024**2):.2f} MB")

    # Final verdict
    print("\n" + "=" * 80)
    if compatible and len(components['CleanUNet']) > 0 and len(components['CleanSpecNet']) > 0:
        print("✅ VERDICT: Checkpoint is COMPATIBLE with CleanUNet2-GAN training")
        print("\nYou can use it in your config like this:")
        print(f"""
model:
  cleanunet2_checkpoint: "{checkpoint_path}"
  conditioning_type: "{conditioning_type}"
  train_cleanunet: true
  train_cleanspecnet: true
""")
    else:
        print("❌ VERDICT: Checkpoint may have COMPATIBILITY ISSUES")
        print("\nPlease check the warnings above before proceeding.")

    print("=" * 80)

    return compatible


def main():
    parser = argparse.ArgumentParser(
        description="Verify CleanUNet2 checkpoint compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_checkpoint.py --checkpoint ../CleanUNet2-Vanilla/logs/checkpoints/best-model.ckpt
  python verify_checkpoint.py --checkpoint path/to/checkpoint.ckpt --conditioning film
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the checkpoint file to verify'
    )

    parser.add_argument(
        '--conditioning',
        type=str,
        default='addition',
        choices=['addition', 'concatenation', 'film'],
        help='Expected conditioning type (default: addition)'
    )

    args = parser.parse_args()

    verify_checkpoint(args.checkpoint, args.conditioning)


if __name__ == "__main__":
    main()
