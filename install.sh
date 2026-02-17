#!/bin/bash
# Quick installation script for CleanUNet2-wav2vec2
# Usage: bash install.sh [cuda118|cuda121|cpu]

set -e  # Exit on error

echo "============================================================================"
echo "CleanUNet2-wav2vec2 Installation Script"
echo "============================================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Detected Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo "ERROR: Python 3.9 or higher is required!"
    exit 1
fi

# Determine CUDA version
CUDA_VERSION=${1:-cuda118}  # Default to CUDA 11.8

echo "CUDA/CPU option: $CUDA_VERSION"
echo ""

# Install PyTorch based on CUDA version
echo "Step 1/3: Installing PyTorch..."
echo "----------------------------------------"

case $CUDA_VERSION in
    cuda118)
        echo "Installing PyTorch with CUDA 11.8 support..."
        pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    cuda121)
        echo "Installing PyTorch with CUDA 12.1 support..."
        pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
    cpu)
        echo "Installing PyTorch (CPU only)..."
        pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "ERROR: Invalid CUDA option. Use: cuda118, cuda121, or cpu"
        exit 1
        ;;
esac

echo ""
echo "Step 2/3: Installing other dependencies..."
echo "----------------------------------------"
pip install -r requirements.txt

echo ""
echo "Step 3/3: Verifying installation..."
echo "----------------------------------------"

# Test imports
python -c "
import torch
import torchaudio
import pytorch_lightning
import transformers
import speechbrain

print('✓ PyTorch:', torch.__version__)
print('✓ TorchAudio:', torchaudio.__version__)
print('✓ PyTorch Lightning:', pytorch_lightning.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ SpeechBrain: OK')

if torch.cuda.is_available():
    print('✓ CUDA available:', torch.cuda.get_device_name(0))
else:
    print('⚠ CUDA not available (CPU only)')
"

echo ""
echo "============================================================================"
echo "Installation Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "For Wav2Vec2 (recommended):"
echo "  1. Extract embeddings:"
echo "     python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml"
echo ""
echo "  2. Train Stage-1:"
echo "     python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1"
echo ""
echo "For X-Vectors:"
echo "  1. Train Stage-1:"
echo "     python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1"
echo ""
echo "Documentation:"
echo "  - Wav2Vec2 Guide: WAV2VEC2_GUIDE.md"
echo "  - Quick Start: README_WAV2VEC2.md"
echo "  - Installation: INSTALLATION.md"
echo ""
echo "Test the system:"
echo "  python test_wav2vec2.py"
echo ""
echo "============================================================================"
