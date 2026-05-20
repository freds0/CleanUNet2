# 🎵 CleanUNet2: Hybrid Speech Denoising with Wav2Vec2 Embeddings

**Version**: 2.0 (Wav2Vec2 ONLY - XVector removed)  
**Status**: ✅ Production Ready  
**License**: MIT

---

## 📢 Overview

**CleanUNet2** is a deep-learning architecture for **speech enhancement** that combines:

- **🎤 Spectrogram Refinement**: CleanSpecNet (frequency domain transformer)
- **🌊 Waveform Denoising**: CleanUNet (multi-scale encoder-decoder)
- **🧠 Wav2Vec2 Embeddings**: 1024-D self-supervised representations (facebook/wav2vec2-xls-r-2b)
- **🔗 Hybrid Conditioning**: FiLM-based fusion of spectrogram and waveform features

This is a **refactored, production-ready version** with:
- ✅ Wav2Vec2-exclusive pipeline (XVector completely removed)
- ✅ Unified configuration system
- ✅ Professional directory structure
- ✅ Type-safe parameters
- ✅ Comprehensive documentation

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd CleanUNet2-wav2vec-src

# Install dependencies (Wav2Vec2 ONLY)
pip install -r requirements.txt

# Verify installation
python scripts/validate_config.py --config configs/train.yaml
```

### 2. Quick Test (1 epoch, 30 samples)

```bash
# Run sanity check
python scripts/train.py --config configs/train.yaml --quick-test

# Expected output:
# [INFO] Stage: 1
# [INFO] Quick Test Mode: True
# ... training loop ...
# ✅ Quick test completed successfully!
```

### 3. Train Stage 1

```bash
# Stage 1 training (baseline, no augmentation)
python scripts/train.py --config configs/train.yaml --stage 1

# Stage 1 with augmentation (enable in configs/train.yaml)
# Edit: augmentation.enabled: true
python scripts/train.py --config configs/train.yaml --stage 1
```

### 4. Train Stage 2

```bash
# Edit configs/train.yaml:
# pipeline.stage: 2
# stage1_checkpoint: "path/to/stage1/best-checkpoint.ckpt"

python scripts/train.py --config configs/train.yaml --stage 2
```

### 5. Run Inference

```bash
# Edit configs/inference.yaml:
# checkpoint_path: "path/to/best-checkpoint.ckpt"
# input_output.input_dir: "path/to/noisy/audio/"
# input_output.output_dir: "denoised_output/"

python scripts/inference.py --config configs/inference.yaml
```

---

## 📁 Directory Structure (New)

```
CleanUNet2-wav2vec-src/
│
├── src/cleanunet/                     # Core model code
│   ├── models/                        # Model architectures
│   │   ├── cleanunet.py              # Waveform UNet
│   │   ├── cleanspecnet.py           # Spectrogram network
│   │   ├── cleanunet2.py             # Hybrid model
│   │   └── conditioner.py            # Spec↔waveform fusion
│   ├── embeddings/                    # Wav2Vec2 extraction
│   │   ├── wav2vec2_extractor.py
│   │   └── wav2vec2_cache.py
│   ├── loss.py                        # Multi-resolution STFT, phase losses
│   ├── metrics.py                     # PESQ, STOI, SI-SDR
│   └── util.py                        # Utilities
│
├── data/                              # Data loading & augmentation
│   ├── dataset.py                     # MelDataset for audio pairs
│   └── augmentation.py                # On-the-fly augmentation
│
├── training/                          # PyTorch Lightning training
│   ├── lightning_module.py            # CleanUNetLightningModule
│   └── data_module.py                 # DataModule for dataset handling
│
├── scripts/                           # Executable scripts
│   ├── train.py                       # Main training entry point
│   ├── inference.py                   # Inference pipeline
│   ├── extract_embeddings.py          # Wav2Vec2 pre-extraction
│   ├── validate_config.py             # Config validation
│   └── ...                            # Other utilities
│
├── configs/                           # Unified configuration
│   ├── config.py                      # Type-safe schema
│   ├── train.yaml                     # Training (Stage 1 & 2)
│   └── inference.yaml                 # Inference config
│
├── tests/                             # Test suite
│   ├── test_models.py                 # Model architecture tests
│   ├── test_embeddings.py             # Wav2Vec2 tests
│   ├── test_dataset.py                # Dataset loading tests
│   └── conftest.py                    # Pytest fixtures
│
└── filelists/                         # Dataset pointers
    ├── train.csv                      # Training file pairs
    └── test.csv                       # Test file pairs
```

---

## 🎯 Key Features

### 1. **Wav2Vec2-Exclusive Pipeline**
- ✅ 1024-D embeddings from facebook/wav2vec2-xls-r-2b (layer 24)
- ✅ Pre-extraction with caching OR on-the-fly extraction
- ✅ No XVector dependencies (removed SpeechBrain)
- ✅ Lighter footprint (~100MB lighter than before)

### 2. **Unified Configuration**
- ✅ Single `train.yaml` for both Stage 1 & Stage 2
- ✅ Type-safe validation via Pydantic dataclasses
- ✅ Stage switching: just change `pipeline.stage: 1|2`
- ✅ Optional data augmentation (Stage 1 specific)
- ✅ See `CONFIGURATION_GUIDE.md` for details

### 3. **Quick Test Mode**
- ✅ Run 1 epoch on 30 samples for rapid iteration
- ✅ Command: `python scripts/train.py --config configs/train.yaml --quick-test`
- ✅ Completes in < 5 minutes
- ✅ Validates end-to-end pipeline

### 4. **Professional Repository**
- ✅ Clear separation: models, data, training, scripts, tests
- ✅ Organized imports and module structure
- ✅ Comprehensive documentation (README, SETUP, CONFIGURATION_GUIDE, MIGRATION)
- ✅ Type hints and docstrings throughout

### 5. **Multi-Resolution Losses**
- ✅ Waveform L1 loss
- ✅ Multi-resolution STFT loss (3 resolutions)
- ✅ Phase consistency loss
- ✅ Spectrogram magnitude loss

### 6. **Flexible Training**
- ✅ TensorBoard logging (built-in)
- ✅ Weights & Biases integration (optional)
- ✅ Checkpointing (best + periodic)
- ✅ Early stopping
- ✅ Gradient clipping
- ✅ Mixed precision training (16-mixed)

---

## 📊 Model Architecture

### Stage 1: Wav2Vec2-Conditioned Denoising

```
Input: (Noisy Audio, Noisy Spectrogram, Wav2Vec2 Embeddings)
         ↓
CleanSpecNet: Refine spectrogram using self-attention
         ↓
SpecUpsampler: Expand spectrogram to waveform length
         ↓
Conditioner (FiLM): Fuse spectrogram features with waveform
         ↓
CleanUNet: Multi-scale encoder-decoder with transformer bottleneck
         ↓
Output: Enhanced Waveform
```

### Stage 2: Latent Replication (Optional)

Train CleanUNet to replicate Stage 1 outputs without embeddings:
```
Input: (Noisy Audio, Noisy Spectrogram)  [No embeddings needed!]
         ↓
(same architecture as Stage 1)
         ↓
Output: Enhanced Waveform (mimics Stage 1)
```

---

## 📋 Configuration

### Essential Parameters

Edit `configs/train.yaml`:

```yaml
pipeline:
  stage: 1                              # 1 or 2

data:
  data_dir: "/path/to/VoiceBank-DEMAND/"
  train_list_path: "filelists/train.csv"
  val_list_path: "filelists/test.csv"
```

### Optional Parameters

```yaml
# Data augmentation (Stage 1 only)
augmentation:
  enabled: true                         # Enable augmentation
  # techniques: [..., see CONFIGURATION_GUIDE.md]

# Hyperparameters
trainer:
  max_epochs: 300
data:
  batch_size: 32
optimizer:
  lr: 5.0e-05
```

For comprehensive guide, see **`CONFIGURATION_GUIDE.md`**.

---

## 📊 Dataset Format

### Expected CSV Format

**`filelists/train.csv`**:
```
clean/p226_001.wav|noisy/p226_001.wav
clean/p226_002.wav|noisy/p226_002.wav
clean/p226_003.wav|noisy/p226_003.wav
...
```

### Supported Datasets

- **VoiceBank-DEMAND** (16 kHz) - primary
- **LJSpeech** (22 kHz, auto-resampled)
- **Custom datasets** (must provide filelists)

---

## 🧪 Testing

### Quick Test

```bash
# Sanity check: 1 epoch, 30 samples, < 5 min
python scripts/train.py --config configs/train.yaml --quick-test

# Custom quick test
python scripts/train.py \
  --config configs/train.yaml \
  --quick-test \
  --quick-test-samples 50 \
  --quick-test-epochs 1
```

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_embeddings.py -v

# With coverage
pytest tests/ --cov=src/
```

### Validation

```bash
# Validate training config
python scripts/validate_config.py --config configs/train.yaml

# Validate inference config
python scripts/validate_config.py --config configs/inference.yaml --type inference
```

---

## 📚 Documentation

- **`README.md`** (this file) - Overview & quick start
- **`SETUP.md`** (NEW) - Detailed installation & environment setup
- **`CONFIGURATION_GUIDE.md`** - All config parameters explained
- **`MIGRATION.md`** (NEW) - Migration from old structure, file organization
- **`STEP5_REORGANIZATION_PLAN.md`** - Reorganization details (for reference)

---

## 🔧 Common Tasks

### Extract Wav2Vec2 Embeddings (Pre-extraction)

```bash
# Pre-extract embeddings for faster training
python scripts/extract_embeddings.py --config configs/train.yaml

# Specify cache directory
python scripts/extract_embeddings.py \
  --config configs/train.yaml \
  --cache-dir cached_embeddings/wav2vec2
```

### Create Filelists

```bash
# Create filelists from two directories (clean + noisy)
python scripts/create_filelists.py \
  --clean-dir /path/to/clean/ \
  --noisy-dir /path/to/noisy/ \
  --output filelists/custom_train.csv
```

### Validate Dataset

```bash
# Check dataset integrity
python scripts/validate_data.py --filelist filelists/train.csv
```

---

## 📈 Training Results

### Metrics During Training

- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio

Logged to:
- TensorBoard: `experiments/*/tensorboard/`
- Weights & Biases: `cleanunet2_wav2vec2` project

### Checkpoints Saved

```
experiments/
└── stage1_wav2vec2_baseline/
    └── checkpoints/
        ├── best-epoch-XX-val_loss-Y.YYY.ckpt
        ├── epoch-0000.ckpt
        ├── epoch-0010.ckpt
        └── last.ckpt
```

---

## 🚨 Troubleshooting

### "CUDA out of memory"
Reduce `batch_size` or `segment_size` in `configs/train.yaml`:
```yaml
data:
  batch_size: 16          # Was 32
  segment_size: 16000     # Was 32000
```

### "Embeddings cache not found"
Pre-extract embeddings:
```bash
python scripts/extract_embeddings.py --config configs/train.yaml
```

### "Stage 2 requires stage1_checkpoint"
Set in `configs/train.yaml`:
```yaml
stage1_checkpoint: "experiments/stage1_baseline/checkpoints/best-epoch-XX-val_loss-Y.YYY.ckpt"
```

### "Module not found" (import errors)
Ensure you're in the project root:
```bash
cd CleanUNet2-wav2vec-src/
python scripts/train.py --config configs/train.yaml --quick-test
```

---

## 🤝 Citation

If you use this work, please cite:

```bibtex
@software{cleanunet2_wav2vec2,
  title={CleanUNet2: Hybrid Speech Denoising with Wav2Vec2 Embeddings},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## ✅ Checklist for New Users

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Validate config: `python scripts/validate_config.py --config configs/train.yaml`
- [ ] Run quick test: `python scripts/train.py --config configs/train.yaml --quick-test`
- [ ] Edit `configs/train.yaml` with your dataset path
- [ ] Extract embeddings: `python scripts/extract_embeddings.py --config configs/train.yaml`
- [ ] Start training: `python scripts/train.py --config configs/train.yaml --stage 1`

---

**Status**: ✅ Production Ready | **Version**: 2.0 (Wav2Vec2 ONLY) | **Updated**: 2026-05-19

For questions or issues, refer to the documentation or open an issue on GitHub.
