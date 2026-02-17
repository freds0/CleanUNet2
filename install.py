#!/usr/bin/env python
"""
Quick installation script for CleanUNet2-wav2vec2
Cross-platform (Windows/Linux/macOS)

Usage:
    python install.py            # Default: CUDA 11.8
    python install.py cuda118    # CUDA 11.8
    python install.py cuda121    # CUDA 12.1
    python install.py cpu        # CPU only
"""

import sys
import subprocess
import platform


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print("-" * 60)

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        sys.exit(1)

    return result


def check_python_version():
    """Check if Python version is adequate."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"

    print(f"Detected Python version: {version_str}")

    if major < 3 or (major == 3 and minor < 9):
        print(f"\n❌ ERROR: Python 3.9 or higher is required!")
        print(f"Current version: {version_str}")
        sys.exit(1)

    return version_str


def test_imports():
    """Test if all packages are installed correctly."""
    print("\n" + "=" * 80)
    print("Testing imports...")
    print("=" * 80)

    try:
        import torch
        import torchaudio
        import pytorch_lightning
        import transformers
        import speechbrain

        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ TorchAudio: {torchaudio.__version__}")
        print(f"✓ PyTorch Lightning: {pytorch_lightning.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ SpeechBrain: OK")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU only)")

        return True

    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        return False


def main():
    print("=" * 80)
    print("CleanUNet2-wav2vec2 Installation Script")
    print("=" * 80)
    print()

    # Check Python version
    python_version = check_python_version()

    # Determine CUDA version
    cuda_version = sys.argv[1] if len(sys.argv) > 1 else "cuda118"

    valid_options = ["cuda118", "cuda121", "cpu"]
    if cuda_version not in valid_options:
        print(f"\n❌ ERROR: Invalid option '{cuda_version}'")
        print(f"Valid options: {', '.join(valid_options)}")
        sys.exit(1)

    print(f"CUDA/CPU option: {cuda_version}")
    print(f"Platform: {platform.system()}")
    print()

    # Step 1: Install PyTorch
    print("\nStep 1/3: Installing PyTorch...")
    print("=" * 80)

    torch_cmd = {
        "cuda118": "pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
        "cuda121": "pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121",
        "cpu": "pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu"
    }

    run_command(torch_cmd[cuda_version], f"Installing PyTorch ({cuda_version})")

    # Step 2: Install other dependencies
    print("\n\nStep 2/3: Installing other dependencies...")
    print("=" * 80)

    run_command("pip install -r requirements.txt", "Installing requirements.txt")

    # Step 3: Test installation
    print("\n\nStep 3/3: Verifying installation...")
    print("=" * 80)

    if not test_imports():
        print("\n❌ Installation verification failed!")
        print("Some packages may not have been installed correctly.")
        sys.exit(1)

    # Success!
    print("\n" + "=" * 80)
    print("✅ Installation Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print()
    print("For Wav2Vec2 (recommended):")
    print("  1. Extract embeddings:")
    print("     python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml")
    print()
    print("  2. Train Stage-1:")
    print("     python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1")
    print()
    print("For X-Vectors:")
    print("  1. Train Stage-1:")
    print("     python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1")
    print()
    print("Documentation:")
    print("  - Wav2Vec2 Guide: WAV2VEC2_GUIDE.md")
    print("  - Quick Start: README_WAV2VEC2.md")
    print("  - Installation: INSTALLATION.md")
    print()
    print("Test the system:")
    print("  python test_wav2vec2.py")
    print()
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Installation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
