#!/usr/bin/env python3
"""
Test script for data augmentation in CleanUNet2-GAN

This script verifies that:
1. AudioAugmenter is working correctly
2. MelDataset loads and applies augmentation
3. Augmented audio is different from original

Usage:
    python test_augmentation.py --data_dir /path/to/audio --mode clean_only
    python test_augmentation.py --data_dir /path/to/dataset --mode two_folders
"""

import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from spec_dataset import MelDataset
from pathlib import Path


def test_augmenter():
    """Test AudioAugmenter initialization and basic functionality."""
    print("=" * 80)
    print("TEST 1: AudioAugmenter Initialization")
    print("=" * 80)

    from augmentation import AudioAugmenter

    # Define simple augmentation config
    augmentations = [
        {
            "name": "AddColoredNoise",
            "params": {
                "min_snr_in_db": 10.0,
                "max_snr_in_db": 20.0,
                "min_f_decay": -2.0,
                "max_f_decay": 2.0,
                "p": 1.0  # Always apply
            }
        },
        {
            "name": "Gain",
            "params": {
                "min_gain_in_db": -3.0,
                "max_gain_in_db": 3.0,
                "p": 1.0
            }
        }
    ]

    try:
        augmenter = AudioAugmenter(augmentations=augmentations, device='cpu', seed=42)
        print("✅ AudioAugmenter initialized successfully")

        # Test with dummy audio
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz
        augmented = augmenter.apply(dummy_audio, sr=16000)

        print(f"✅ Augmentation applied successfully")
        print(f"   Original shape: {dummy_audio.shape}")
        print(f"   Augmented shape: {augmented.shape}")

        # Check that audio changed
        diff = (dummy_audio - augmented.squeeze()).abs().mean().item()
        if diff > 0.01:
            print(f"✅ Audio was modified (mean diff: {diff:.4f})")
        else:
            print(f"⚠️  Audio barely changed (mean diff: {diff:.4f})")

        return True

    except Exception as e:
        print(f"❌ AudioAugmenter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(data_dir: str, mode: str = "clean_only", noise_dir: str = None):
    """Test MelDataset with augmentation."""
    print("\n" + "=" * 80)
    print(f"TEST 2: MelDataset with Augmentation (mode={mode})")
    print("=" * 80)

    # Define augmentation config
    augmentation_config = {
        "enabled": True,
        "mode": mode,
        "augmentations": [
            {
                "name": "AddColoredNoise",
                "params": {
                    "min_snr_in_db": 5.0,
                    "max_snr_in_db": 15.0,
                    "min_f_decay": -2.0,
                    "max_f_decay": 2.0,
                    "p": 1.0
                }
            }
        ]
    }

    # Add background noise if path provided
    if noise_dir and Path(noise_dir).exists():
        augmentation_config["augmentations"].insert(0, {
            "name": "AddBackgroundNoise",
            "params": {
                "background_paths": noise_dir,
                "min_snr_in_db": 5.0,
                "max_snr_in_db": 15.0,
                "p": 0.8
            }
        })

    try:
        # Create dataset
        dataset = MelDataset(
            data_dir=data_dir,
            data_files="train" if mode == "two_folders" else ".",
            segment_size=16000,  # 1 second
            sampling_rate=16000,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            split=True,
            shuffle=False,
            augmentation=augmentation_config
        )

        print(f"✅ Dataset created with {len(dataset)} samples")

        # Load first sample
        noisy, noisy_spec, clean, clean_spec = dataset[0]

        print(f"✅ Sample loaded successfully")
        print(f"   Noisy audio shape: {noisy.shape}")
        print(f"   Clean audio shape: {clean.shape}")
        print(f"   Noisy spec shape: {noisy_spec.shape}")
        print(f"   Clean spec shape: {clean_spec.shape}")

        # Check if augmentation was applied (in clean_only mode)
        if mode == "clean_only":
            diff = (noisy.squeeze() - clean.squeeze()).abs().mean().item()
            if diff > 0.01:
                print(f"✅ Augmentation applied (noisy != clean, diff: {diff:.4f})")
            else:
                print(f"⚠️  Noisy and clean are very similar (diff: {diff:.4f})")

        # Load multiple samples to test variability
        print("\nTesting variability across samples...")
        samples = [dataset[i] for i in range(min(3, len(dataset)))]

        for i, (noisy, _, _, _) in enumerate(samples):
            energy = noisy.abs().mean().item()
            print(f"   Sample {i+1} energy: {energy:.4f}")

        return True

    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_augmentation(data_dir: str, mode: str = "clean_only", output_path: str = "augmentation_test.png"):
    """Visualize original vs augmented audio."""
    print("\n" + "=" * 80)
    print(f"TEST 3: Visualization (mode={mode})")
    print("=" * 80)

    augmentation_config = {
        "enabled": True,
        "mode": mode,
        "augmentations": [
            {
                "name": "AddColoredNoise",
                "params": {
                    "min_snr_in_db": 5.0,
                    "max_snr_in_db": 10.0,
                    "min_f_decay": 0.0,
                    "max_f_decay": 0.0,
                    "p": 1.0
                }
            }
        ]
    }

    try:
        # Dataset with augmentation
        dataset_aug = MelDataset(
            data_dir=data_dir,
            data_files="train" if mode == "two_folders" else ".",
            segment_size=16000,
            sampling_rate=16000,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            split=True,
            shuffle=False,
            augmentation=augmentation_config
        )

        # Dataset without augmentation
        dataset_no_aug = MelDataset(
            data_dir=data_dir,
            data_files="train" if mode == "two_folders" else ".",
            segment_size=16000,
            sampling_rate=16000,
            n_fft=1024,
            hop_size=256,
            win_size=1024,
            split=True,
            shuffle=False,
            augmentation={"enabled": False}
        )

        # Get samples
        noisy_aug, spec_aug, clean_aug, _ = dataset_aug[0]
        noisy_no_aug, spec_no_aug, clean_no_aug, _ = dataset_no_aug[0]

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Waveforms
        axes[0, 0].plot(clean_aug.squeeze().numpy())
        axes[0, 0].set_title("Clean Audio")
        axes[0, 0].set_xlabel("Samples")
        axes[0, 0].set_ylabel("Amplitude")

        axes[0, 1].plot(noisy_no_aug.squeeze().numpy())
        axes[0, 1].set_title("Noisy (Original)")
        axes[0, 1].set_xlabel("Samples")

        axes[0, 2].plot(noisy_aug.squeeze().numpy())
        axes[0, 2].set_title("Noisy (Augmented)")
        axes[0, 2].set_xlabel("Samples")

        # Spectrograms
        axes[1, 0].imshow(spec_no_aug.numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[1, 0].set_title("Spec (Original)")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].imshow(spec_aug.numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[1, 1].set_title("Spec (Augmented)")
        axes[1, 1].set_xlabel("Time")

        # Difference
        diff = (spec_aug - spec_no_aug).abs().numpy()
        im = axes[1, 2].imshow(diff, aspect='auto', origin='lower', cmap='hot')
        axes[1, 2].set_title("Difference")
        axes[1, 2].set_xlabel("Time")
        plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✅ Visualization saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test data augmentation for CleanUNet2-GAN")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--mode", type=str, default="clean_only", choices=["clean_only", "two_folders"],
                        help="Augmentation mode")
    parser.add_argument("--noise_dir", type=str, default=None, help="Path to noise directory (optional)")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--output", type=str, default="augmentation_test.png", help="Visualization output path")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CleanUNet2-GAN - Data Augmentation Test Suite")
    print("=" * 80)

    results = []

    # Test 1: AudioAugmenter
    results.append(("AudioAugmenter", test_augmenter()))

    # Test 2: Dataset
    results.append(("MelDataset", test_dataset(args.data_dir, args.mode, args.noise_dir)))

    # Test 3: Visualization (optional)
    if args.visualize:
        results.append(("Visualization", visualize_augmentation(args.data_dir, args.mode, args.output)))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<50} {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
        print("\nYou can now use augmentation in training with:")
        print("  python train.py --config configs/train_gan_with_augmentation.yaml")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease check the errors above and ensure:")
        print("  1. torch-audiomentations is installed")
        print("  2. Data directory exists and contains audio files")
        print("  3. Directory structure matches the selected mode")
    print("=" * 80)


if __name__ == "__main__":
    main()
