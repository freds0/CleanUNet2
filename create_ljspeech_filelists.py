#!/usr/bin/env python3
"""
Create train/validation filelists for LJSpeech dataset.

LJSpeech contains only clean audio, so both columns in the filelist
will point to the same file. Data augmentation will create the noisy version.

Usage:
    python create_ljspeech_filelists.py --data_dir /path/to/LJSpeech-1.1
"""

import os
import argparse
import random
from pathlib import Path


def create_ljspeech_filelists(data_dir, output_dir="filelists", train_ratio=0.9, seed=42):
    """
    Create train and validation filelists for LJSpeech.

    Args:
        data_dir (str): Path to LJSpeech-1.1 directory
        output_dir (str): Directory to save filelists
        train_ratio (float): Ratio of training data (default: 0.9 = 90% train, 10% val)
        seed (int): Random seed for reproducibility
    """
    print("=" * 80)
    print("LJSpeech Filelist Generator")
    print("=" * 80)

    # Validate input directory
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    wav_dir = data_dir / "wavs"
    if not wav_dir.exists():
        raise ValueError(f"wavs directory not found: {wav_dir}")

    # Get all wav files
    wav_files = sorted([f.name for f in wav_dir.glob("*.wav")])

    if len(wav_files) == 0:
        raise ValueError(f"No .wav files found in {wav_dir}")

    print(f"\n[INFO] Found {len(wav_files)} audio files in {wav_dir}")

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(wav_files)

    # Split into train/val
    split_idx = int(train_ratio * len(wav_files))
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]

    print(f"[INFO] Split: {len(train_files)} train, {len(val_files)} validation")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write train filelist
    train_path = output_dir / "ljspeech_train.txt"
    with open(train_path, "w") as f:
        for wav_file in train_files:
            # Both columns point to same file (clean audio)
            # Augmentation will create noisy version
            path = f"wavs/{wav_file}"
            f.write(f"{path},{path}\n")

    print(f"[INFO] Created train filelist: {train_path}")

    # Write validation filelist
    val_path = output_dir / "ljspeech_val.txt"
    with open(val_path, "w") as f:
        for wav_file in val_files:
            path = f"wavs/{wav_file}"
            f.write(f"{path},{path}\n")

    print(f"[INFO] Created validation filelist: {val_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files: {len(wav_files)}")
    print(f"Training files: {len(train_files)} ({len(train_files)/len(wav_files)*100:.1f}%)")
    print(f"Validation files: {len(val_files)} ({len(val_files)/len(wav_files)*100:.1f}%)")
    print(f"\nFilelists saved to: {output_dir.absolute()}")
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Update config file:")
    print("   configs/train_xvector_ljspeech_augmented.yaml")
    print("")
    print("2. Set data_dir:")
    print(f"   data_dir: \"{data_dir.absolute()}\"")
    print("")
    print("3. Verify filelist paths:")
    print(f"   train_list_path: \"{train_path}\"")
    print(f"   val_list_path: \"{val_path}\"")
    print("")
    print("4. Start training:")
    print("   python train_xvector.py --config configs/train_xvector_ljspeech_augmented.yaml --stage stage1")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation filelists for LJSpeech dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to LJSpeech-1.1 directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="filelists",
        help="Directory to save filelists (default: filelists)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of training data (default: 0.9)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    try:
        create_ljspeech_filelists(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
