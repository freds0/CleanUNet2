# Requirements Summary - CleanUNet2-wav2vec2

## What Was Updated

The [requirements.txt](requirements.txt) file has been updated to include all dependencies for the new Wav2Vec2 embedding system.

## New Dependencies

### Wav2Vec2 Support (NEW!)

```
transformers>=4.30.0          # For Wav2Vec2 models (facebook/wav2vec2-xls-r-300m)
huggingface_hub>=0.20.0       # For downloading pretrained models
sentencepiece>=0.1.99         # Required by some transformers tokenizers
```

### Data Augmentation

```
torch-audiomentations>=0.11.0  # On-the-fly audio augmentation
```

## Complete Dependency List

### Core Deep Learning
- `torch>=2.1.0` - PyTorch framework
- `torchaudio>=2.1.0` - Audio processing for PyTorch
- `pytorch-lightning>=2.2.0` - High-level training framework

### Audio Processing
- `librosa>=0.10.1` - Audio analysis library
- `soundfile>=0.12.1` - Audio file I/O
- `tqdm>=4.65.0` - Progress bars

### Metrics
- `pesq>=0.0.4` - PESQ metric
- `pystoi>=0.3.3` - STOI metric
- `torchmetrics[audio]` - TorchMetrics with audio support

### Numeric/Utils
- `numpy>=1.23.0` - Numerical operations
- `scipy>=1.10.0` - Scientific computing
- `PyYAML>=6.0` - YAML configuration files
- `json5>=0.9.14` - JSON5 parsing

### Logging
- `tensorboard>=2.14.0` - TensorBoard logging
- `wandb` - Weights & Biases logging

### Optional
- `matplotlib>=3.6.0` - Plotting
- `einops>=0.7.0` - Tensor operations

### X-Vector Support
- `speechbrain>=0.5.16` - X-Vector extraction
- `ruamel.yaml` - YAML support for SpeechBrain

### Wav2Vec2 Support (NEW!)
- `transformers>=4.30.0` - Wav2Vec2 models
- `huggingface_hub>=0.20.0` - Model downloading
- `sentencepiece>=0.1.99` - Tokenizer support

### Augmentation
- `torch-audiomentations>=0.11.0` - Audio augmentation

## Installation

### Quick Install (All Features)

```bash
pip install -r requirements.txt
```

### Install with Specific CUDA Version

**CUDA 11.8:**
```bash
bash install.sh cuda118
# or
python install.py cuda118
```

**CUDA 12.1:**
```bash
bash install.sh cuda121
# or
python install.py cuda121
```

**CPU Only:**
```bash
bash install.sh cpu
# or
python install.py cpu
```

## Version Requirements

### Minimum Versions
- Python: 3.9+
- PyTorch: 2.1.0+
- PyTorch Lightning: 2.2.0+
- Transformers: 4.30.0+

### Recommended Versions
- Python: 3.10 or 3.11
- PyTorch: 2.1.0 - 2.2.0
- PyTorch Lightning: 2.2.0 - 2.3.0
- Transformers: 4.30.0 - 4.40.0

## Compatibility Notes

### Wav2Vec2 (transformers)
- ✅ Works with transformers >= 4.30.0
- ⚠️ May have issues with transformers < 4.30.0
- ✅ Compatible with PyTorch 2.0+

### SpeechBrain
- ✅ Includes compatibility patches for latest versions
- ✅ Works with huggingface_hub >= 0.20.0
- ⚠️ May need ruamel.yaml for some configurations

### TorchMetrics
- ✅ Use `torchmetrics[audio]` for PESQ/STOI support
- ✅ Compatible with PyTorch 2.0+
- ⚠️ Some audio metrics require specific sample rates

### torch-audiomentations
- ✅ Version 0.11.0+ recommended
- ✅ Supports on-the-fly augmentation
- ⚠️ Some augmentations slower on Windows

## GPU Requirements

### For Wav2Vec2 Embedding Extraction
- **Recommended**: 6GB+ GPU memory
- **Minimum**: 4GB GPU memory (with smaller batch)
- **CPU**: Supported but ~10x slower

### For Training
- **Batch size 64**: 12GB+ GPU memory
- **Batch size 32**: 8GB+ GPU memory
- **Batch size 16**: 6GB+ GPU memory

### For Inference
- **With pre-extracted embeddings**: 4GB+ GPU memory
- **Without embeddings**: 4GB+ GPU memory

## Disk Space

### Models
- Wav2Vec2-xls-r-300m: ~1.2 GB
- X-Vector model: ~200 MB

### Pre-extracted Embeddings
- 10k files: ~40 MB
- 100k files: ~400 MB
- 1M files: ~4 GB

Very efficient storage!

## Testing Installation

After installation, test the system:

```bash
python test_wav2vec2.py
```

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

### Common Issues

1. **CUDA version mismatch**
   - Ensure PyTorch CUDA version matches system CUDA
   - Use `nvidia-smi` to check system CUDA version

2. **transformers import error**
   - Update: `pip install --upgrade transformers`
   - Check version: `pip show transformers`

3. **speechbrain compatibility**
   - Code includes patches for compatibility
   - If issues persist: `pip install speechbrain==0.5.16`

4. **Out of memory during extraction**
   - Use CPU: `--device cpu`
   - Or reduce batch size in extraction script

## Updates

To update dependencies:

```bash
# Update all
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade transformers
pip install --upgrade pytorch-lightning
```

## Documentation

- **Complete Guide**: [WAV2VEC2_GUIDE.md](WAV2VEC2_GUIDE.md)
- **Installation**: [INSTALLATION.md](INSTALLATION.md)
- **Quick Start**: [README_WAV2VEC2.md](README_WAV2VEC2.md)
- **Implementation**: [WAV2VEC2_IMPLEMENTATION_SUMMARY.md](WAV2VEC2_IMPLEMENTATION_SUMMARY.md)

## Support

For dependency issues:
1. Check this summary first
2. See [INSTALLATION.md](INSTALLATION.md) for detailed instructions
3. Run `test_wav2vec2.py` to diagnose
4. Check version compatibility

## Changelog

### v2.0 (Current)
- ✨ Added Wav2Vec2 support (transformers, huggingface_hub)
- ✨ Added data augmentation (torch-audiomentations)
- 📝 Updated documentation with installation guides
- 🔧 Added installation scripts (install.sh, install.py)
- ✅ Added test suite (test_wav2vec2.py)

### v1.0 (Previous)
- ✅ X-Vector support (speechbrain)
- ✅ Basic training pipeline
- ✅ Two-stage training

## License

Same as the main CleanUNet2 project.
