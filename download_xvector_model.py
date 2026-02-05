#!/usr/bin/env python3
"""
Script to manually download the X-Vector model from HuggingFace.

This is useful when:
1. Your training machine doesn't have internet access
2. HuggingFace Hub is temporarily unavailable
3. You want to cache the model for faster setup

Usage:
    python download_xvector_model.py [--savedir PATH]

The model will be saved to the specified directory (default: pretrained_models/spkrec-xvect-voxceleb)
You can then copy this directory to your training machine and reference it in the config.
"""

import argparse
import sys
import os

def download_model(savedir="pretrained_models/spkrec-xvect-voxceleb"):
    """
    Download X-Vector model from HuggingFace.

    Args:
        savedir (str): Directory to save the model
    """
    print("=" * 80)
    print("X-Vector Model Download Script")
    print("=" * 80)
    print(f"Model: speechbrain/spkrec-xvect-voxceleb")
    print(f"Save directory: {savedir}")
    print("")

    # Check if model already exists
    if os.path.exists(savedir):
        hyperparams_file = os.path.join(savedir, "hyperparams.yaml")
        if os.path.exists(hyperparams_file):
            print(f"[INFO] Model already exists at: {savedir}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                print("[INFO] Using existing model. Exiting.")
                return

    print("\n[INFO] Installing required dependencies...")
    try:
        import speechbrain
        print(f"[OK] SpeechBrain version: {speechbrain.__version__}")
    except ImportError:
        print("[ERROR] SpeechBrain not installed!")
        print("\nPlease install it with:")
        print("  pip install speechbrain")
        sys.exit(1)

    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
    except ImportError:
        print("[ERROR] PyTorch not installed!")
        sys.exit(1)

    print("\n[INFO] Starting download...")
    print("[INFO] This may take a few minutes (model is ~500MB)")
    print("")

    try:
        # Import compatibility fixes
        import ruamel.yaml
        if hasattr(ruamel.yaml, 'Loader') and not hasattr(ruamel.yaml.Loader, 'max_depth'):
            ruamel.yaml.Loader.max_depth = None
        if hasattr(ruamel.yaml, 'SafeLoader') and not hasattr(ruamel.yaml.SafeLoader, 'max_depth'):
            ruamel.yaml.SafeLoader.max_depth = None
    except ImportError:
        pass

    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        from speechbrain.inference import EncoderClassifier

    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"}
        )

        print("\n" + "=" * 80)
        print("[SUCCESS] Model downloaded successfully!")
        print("=" * 80)
        print(f"Location: {os.path.abspath(savedir)}")
        print("")
        print("Files downloaded:")
        for root, dirs, files in os.walk(savedir):
            level = root.replace(savedir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"{subindent}{file} ({file_size:.2f} MB)")

        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("")
        print("1. If downloading on a different machine, copy the entire directory:")
        print(f"   {os.path.abspath(savedir)}")
        print("")
        print("2. Update your training config to use the local model:")
        print("")
        print("   # In configs/train_xvector_vanilla_stage1.yaml")
        print("   model:")
        print(f"     xvector_local_path: '{os.path.abspath(savedir)}'")
        print("")
        print("3. Start training:")
        print("   python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1")
        print("")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] Download failed!")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        print("")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Try running with VPN if HuggingFace is blocked")
        print("3. Check HuggingFace Hub status: https://status.huggingface.co")
        print("4. Try again later if the service is temporarily unavailable")
        print("")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download X-Vector model from HuggingFace for CleanUNet2 training"
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="pretrained_models/spkrec-xvect-voxceleb",
        help="Directory to save the model (default: pretrained_models/spkrec-xvect-voxceleb)"
    )

    args = parser.parse_args()

    download_model(args.savedir)


if __name__ == "__main__":
    main()
