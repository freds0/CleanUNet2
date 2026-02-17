# Installation Guide - CleanUNet2-wav2vec2

## Quick Installation

### Option 1: Full Installation (Recommended)

Install all dependencies including Wav2Vec2 support:

```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation

If you only want to use X-Vectors (without Wav2Vec2):

```bash
# Install base dependencies
pip install torch>=2.1.0 torchaudio>=2.1.0 pytorch-lightning>=2.2.0
pip install librosa soundfile tqdm pesq pystoi torchmetrics[audio]
pip install numpy scipy PyYAML json5 tensorboard wandb
pip install matplotlib einops

# Install X-Vector dependencies
pip install speechbrain>=0.5.16 ruamel.yaml huggingface_hub
pip install torch-audiomentations>=0.11.0
```

### Option 3: Wav2Vec2 Only

If you only want Wav2Vec2 (without X-Vectors):

```bash
# Install base dependencies
pip install torch>=2.1.0 torchaudio>=2.1.0 pytorch-lightning>=2.2.0
pip install librosa soundfile tqdm pesq pystoi torchmetrics[audio]
pip install numpy scipy PyYAML json5 tensorboard wandb
pip install matplotlib einops

# Install Wav2Vec2 dependencies
pip install transformers>=4.30.0 huggingface_hub>=0.20.0
pip install torch-audiomentations>=0.11.0
```

## Version Compatibility

### Tested Configurations

**Configuration 1 (Recommended):**
- Python: 3.10 or 3.11
- PyTorch: 2.1.0 - 2.2.0
- PyTorch Lightning: 2.2.0 - 2.3.0
- Transformers: 4.30.0 - 4.40.0
- CUDA: 11.8 or 12.1

**Configuration 2 (Older systems):**
- Python: 3.9
- PyTorch: 2.0.0
- PyTorch Lightning: 2.0.0
- Transformers: 4.28.0
- CUDA: 11.7

### Known Issues

1. **transformers < 4.30.0**: May have issues with wav2vec2-xls-r-300m loading
2. **pytorch-lightning < 2.0**: Not compatible with current code
3. **Python < 3.9**: Not supported

## GPU Requirements

### Wav2Vec2 Extraction (Pre-training)

- **GPU Memory**: 6GB+ recommended
- **Speed**: ~10 files/second on RTX 3090
- **CPU fallback**: Available but 10x slower

### Training

- **Stage-1 (with embeddings)**:
  - Batch size 64: 12GB+ GPU memory
  - Batch size 32: 8GB+ GPU memory
  - Batch size 16: 6GB+ GPU memory

- **Stage-2 (without embeddings)**:
  - Same as Stage-1

### Pre-extracted Embeddings (Disk)

- 10k files: ~40 MB
- 100k files: ~400 MB

Very efficient storage!

## Installation Steps

### Step 1: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n cleanunet2 python=3.10
conda activate cleanunet2

# Or using venv
python -m venv cleanunet2_env
source cleanunet2_env/bin/activate  # Linux/Mac
# cleanunet2_env\Scripts\activate  # Windows
```

### Step 2: Install PyTorch

**With CUDA 11.8:**
```bash
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**With CUDA 12.1:**
```bash
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only:**
```bash
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python test_wav2vec2.py
```

This will test:
- Wav2Vec2 extractor
- Embedding cache
- Model integration

Expected output:
```
================================================================================
WAV2VEC2 EMBEDDING SYSTEM TEST SUITE
================================================================================

TEST 1: Wav2Vec2 Extractor
✅ TEST 1 PASSED!

TEST 2: Wav2Vec2 Cache
✅ TEST 2 PASSED!

TEST 3: Model Integration
✅ TEST 3 PASSED!

🎉 ALL TESTS PASSED!
```

## Troubleshooting

### Error: "No module named 'transformers'"

**Solution:**
```bash
pip install transformers>=4.30.0
```

### Error: "CUDA out of memory" during extraction

**Solution 1:** Use CPU for extraction (slower but works)
```bash
python extract_wav2vec2_embeddings.py --config your_config.yaml --device cpu
```

**Solution 2:** Reduce batch size in extraction script

### Error: "Failed to download wav2vec2 model"

**Solution:** Download manually on a machine with internet:
```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

model = Wav2Vec2Model.from_pretrained(
    'facebook/wav2vec2-xls-r-300m',
    cache_dir='pretrained_models/wav2vec2'
)
processor = Wav2Vec2Processor.from_pretrained(
    'facebook/wav2vec2-xls-r-300m',
    cache_dir='pretrained_models/wav2vec2'
)
```

Then copy `pretrained_models/wav2vec2/` to the target machine.

### Error: "speechbrain compatibility issues"

**Solution:** The code includes compatibility patches. If issues persist:
```bash
pip install speechbrain==0.5.16 ruamel.yaml==0.17.32
```

## Optional Dependencies

### For Development

```bash
pip install pytest black flake8 mypy
```

### For Visualization

```bash
pip install matplotlib seaborn plotly
```

### For Advanced Metrics

```bash
pip install jiwer  # For WER calculation
pip install mir_eval  # For audio metrics
```

## Updating

To update to the latest versions:

```bash
pip install --upgrade -r requirements.txt
```

To update specific packages:

```bash
pip install --upgrade transformers
pip install --upgrade pytorch-lightning
```

## Uninstallation

To completely remove the environment:

```bash
# If using conda
conda deactivate
conda env remove -n cleanunet2

# If using venv
deactivate
rm -rf cleanunet2_env/
```

## Platform-Specific Notes

### Linux

No special considerations. All features work out of the box.

### macOS

- GPU training not supported (no CUDA)
- Use CPU for extraction: `--device cpu`
- MPS (Metal Performance Shaders) not yet supported

### Windows

- Use forward slashes in paths or raw strings
- Some augmentations may be slower
- Recommended: Use WSL2 for better compatibility

## Next Steps

After installation:

1. **For Wav2Vec2**: See [WAV2VEC2_GUIDE.md](WAV2VEC2_GUIDE.md)
2. **For X-Vectors**: See [XVECTOR_CACHE_GUIDE.md](XVECTOR_CACHE_GUIDE.md)
3. **Quick Start**: See [README_WAV2VEC2.md](README_WAV2VEC2.md)

## Support

For installation issues:
1. Check this guide first
2. Run `test_wav2vec2.py` to diagnose issues
3. Check compatibility with your Python/PyTorch versions
4. Ensure CUDA toolkit matches PyTorch CUDA version

## License

Same as the main CleanUNet2 project.
