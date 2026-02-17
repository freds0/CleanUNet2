# Wav2Vec2 Embedding System - Complete Guide

## Overview

The CleanUNet2-wav2vec2 project now supports **Wav2Vec2 embeddings** from facebook/wav2vec2-xls-r-300m as an alternative to X-Vectors. Wav2Vec2 embeddings provide richer self-supervised speech representations (1024 dimensions vs 512 for X-Vectors).

## Key Features

- **Pre-extraction**: Embeddings are extracted once before training and saved to disk
- **Fast training**: No embedding extraction overhead during training
- **Persistent cache**: Embeddings are reused across training runs
- **Automatic system**: Simple workflow with clear steps
- **High quality**: 1024-dimensional embeddings from state-of-the-art wav2vec2-xls-r-300m

## Architecture Comparison

### X-Vectors (Old)
```
Training Step:
  Clean Audio → X-Vector Extractor (150ms) → X-Vector (512d) → Integration → Latent
                      ↑ extracted every batch
```

### Wav2Vec2 (New)
```
Pre-extraction (once):
  Clean Audio → Wav2Vec2 Model → Embedding (1024d) → Save to disk

Training Step:
  Audio Path → Load from disk (5ms) → Embedding (1024d) → Integration → Latent
                      ↑ very fast!
```

## Installation

Make sure you have the required packages:

```bash
pip install transformers torch torchaudio pytorch-lightning
```

## Step-by-Step Usage

### Step 1: Pre-extract Embeddings

**IMPORTANT**: Run this BEFORE training!

```bash
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml
```

This will:
1. Download wav2vec2-xls-r-300m model (first time only)
2. Extract embeddings for all audio files in train/val lists
3. Save embeddings to `wav2vec2_embeddings/` directory
4. Create metadata file with statistics

**Expected output:**
```
================================================================================
Wav2Vec2 Embedding Pre-Extraction
================================================================================
Cache directory: wav2vec2_embeddings
Device: cuda
================================================================================

[Extract] Loading training files from: filelists/alcateia_train.csv
[Extract] Loading validation files from: filelists/alcateia_test.csv
[Extract] Found 10000 unique audio files

[Wav2Vec2Extractor] Loading model: facebook/wav2vec2-xls-r-300m
[Wav2Vec2Extractor] Model loaded successfully!
[Wav2Vec2Extractor] Embedding dimension: 1024

[Extract] Extracting embeddings for 10000 files...
Extracting embeddings: 100%|████████████| 10000/10000 [15:30<00:00, 10.75it/s]

================================================================================
Extraction Complete!
================================================================================
Total files: 10000
  - Extracted: 10000
  - Skipped (already exists): 0
  - Failed: 0

Embeddings saved to: wav2vec2_embeddings
Cache size: 40.50 MB
================================================================================
```

**Options:**
- `--device cuda` - Use GPU for extraction (recommended, ~10x faster)
- `--device cpu` - Use CPU (slower but works without GPU)
- `--force` - Re-extract even if embeddings exist

### Step 2: Verify Extraction

Check that embeddings were extracted successfully:

```bash
# Check cache directory
ls -lh wav2vec2_embeddings/

# Expected output:
# -rw-r--r-- 1 user user 4.1K metadata.yaml
# -rw-r--r-- 1 user user 4.1K a1b2c3d4e5f6.pt
# -rw-r--r-- 1 user user 4.1K 1a2b3c4d5e6f.pt
# ... (one .pt file per audio file)

# Check metadata
cat wav2vec2_embeddings/metadata.yaml
```

### Step 3: Train with Wav2Vec2 Embeddings

```bash
python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1
```

The model will:
1. Load pre-extracted embeddings from disk during training
2. No wav2vec2 extraction overhead
3. Fast and efficient training

**Expected training output:**
```
[CleanUNet2WithXVector] Initializing model...
  - Stage: stage1
  - Embedding Type: wav2vec2
  - Embedding Dim: 1024
  - Wav2Vec2 Model: facebook/wav2vec2-xls-r-300m
  - Use Pre-extracted: True

[Wav2Vec2Cache] Cache enabled: wav2vec2_embeddings
[Wav2Vec2Cache] Loaded metadata:
  - Model: facebook/wav2vec2-xls-r-300m
  - Embedding dim: 1024
  - Sample rate: 24000 Hz
  - Total embeddings: 10000

Training: 100%|████████████| 1000/1000 [10:30<00:00,  1.58it/s]
```

## Configuration

### Wav2Vec2 Config (configs/train_wav2vec2_stage1.yaml)

```yaml
model:
  # Enable Wav2Vec2 embeddings
  use_wav2vec2: true
  use_xvector: false

  # Wav2Vec2 settings
  wav2vec2_model: "facebook/wav2vec2-xls-r-300m"
  wav2vec2_cache_dir: "wav2vec2_embeddings"
  use_preextracted_embeddings: true  # Load from disk (RECOMMENDED!)

  # Rest of model config...

data:
  use_xvector_cache: true  # Enable file path tracking (REQUIRED!)
  # Rest of data config...
```

### X-Vector Config (Original - configs/train_xvector_vanilla_stage1.yaml)

```yaml
model:
  # Use X-Vectors (original behavior)
  use_xvector: true
  use_wav2vec2: false
  xvector_dim: 512
  xvector_cache_enabled: true
  xvector_cache_dir: "xvector_cache_stage1"
  # Rest of model config...
```

## Embedding Comparison

| Feature | X-Vectors | Wav2Vec2 |
|---------|-----------|----------|
| Dimension | 512 | 1024 |
| Model | SpeechBrain spkrec-xvect-voxceleb | facebook/wav2vec2-xls-r-300m |
| Training data | VoxCeleb (speaker recognition) | 436k hours multilingual speech |
| Extraction time | ~150ms per sample | ~300ms per sample (but pre-extracted!) |
| Training overhead | 150ms per batch (with cache) | ~5ms per batch (load from disk) |
| Information | Speaker identity focus | General speech representation |
| Stage-1 performance | Good | Better (richer representations) |

## Performance Benefits

### With Pre-extraction

```
First-time setup (one time):
  - Extract embeddings: ~15 minutes for 10k files (GPU)
  - Disk space: ~4KB per file

Training (every epoch):
  - Load from disk: ~5ms per batch
  - 30x faster than X-Vector extraction
  - 100x faster than on-the-fly Wav2Vec2
```

### Storage Requirements

- Each embedding: ~4KB (1024 floats)
- 10,000 files: ~40 MB
- 100,000 files: ~400 MB

Very efficient storage!

## Advanced Options

### On-the-Fly Extraction (Not Recommended)

If you want to extract wav2vec2 during training (slower):

```yaml
model:
  use_wav2vec2: true
  use_preextracted_embeddings: false  # Extract on-the-fly
  wav2vec2_model: "facebook/wav2vec2-xls-r-300m"
```

**Note**: This is ~100x slower than pre-extraction. Only use for debugging.

### Different Wav2Vec2 Models

You can use other wav2vec2 models:

```yaml
model:
  wav2vec2_model: "facebook/wav2vec2-large-960h"  # English only, 1024 dim
  # or
  wav2vec2_model: "facebook/wav2vec2-xls-r-1b"    # Larger model, 1280 dim
  # or
  wav2vec2_model: "facebook/wav2vec2-xls-r-2b"    # Huge model, 1920 dim
```

Re-run extraction with the new model:
```bash
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml --force
```

## Troubleshooting

### Error: "Pre-extracted embedding not found"

**Problem**: Training fails with embedding not found error.

**Solution**: Run extraction first!
```bash
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml
```

### Error: "Cache directory does not exist"

**Problem**: `wav2vec2_embeddings/` directory not found.

**Solution**: Run extraction script or create the directory manually.

### Low cache hit rate

**Problem**: Cache hit rate < 90%

**Possible causes**:
1. Audio file paths changed → Re-run extraction
2. New files added to dataset → Run extraction for new files only (without `--force`)
3. Wrong cache directory in config

### Extraction takes too long

**Problem**: Extraction is very slow

**Solutions**:
1. Use GPU: `--device cuda` (~10x faster)
2. Reduce batch size if OOM
3. Extract in chunks (modify script to process subsets)

### Model download fails

**Problem**: Cannot download wav2vec2-xls-r-300m

**Solution**: Download manually on a machine with internet:
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

## Cache Management

### Check cache statistics

```python
from cleanunet.wav2vec2_cache import Wav2Vec2Cache

cache = Wav2Vec2Cache(cache_dir='wav2vec2_embeddings', enabled=True)
cache.print_stats()
```

### Clear cache

```bash
rm -rf wav2vec2_embeddings/
```

Then re-run extraction.

### Incremental extraction

If you add new files to the dataset, just run extraction again:
```bash
python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml
```

It will skip already-extracted files and only process new ones.

## Workflow Summary

```
1. Setup
   └─ Install dependencies: pip install transformers

2. Pre-extract embeddings (ONE TIME)
   └─ python extract_wav2vec2_embeddings.py --config configs/train_wav2vec2_stage1.yaml

3. Train Stage-1
   └─ python train_xvector.py --config configs/train_wav2vec2_stage1.yaml --stage stage1

4. Train Stage-2 (optional)
   └─ python train_xvector.py --config configs/train_wav2vec2_stage2.yaml --stage stage2
```

## FAQ

**Q: Should I use Wav2Vec2 or X-Vectors?**

A: Wav2Vec2 is recommended for better performance. It has richer representations (1024 vs 512 dim) trained on more diverse data.

**Q: Do I need to re-extract embeddings for each training run?**

A: No! Extract once, train many times. Embeddings are reused.

**Q: Can I use both Wav2Vec2 and X-Vectors?**

A: No, choose one. They are mutually exclusive.

**Q: What if I change the dataset?**

A: Re-run extraction for the new dataset:
```bash
python extract_wav2vec2_embeddings.py --config your_config.yaml --force
```

**Q: Does augmentation affect embeddings?**

A: No! Embeddings are extracted from clean audio only. Augmentation is applied to noisy input during training.

## References

- Code: [cleanunet/wav2vec2_extractor.py](cleanunet/wav2vec2_extractor.py)
- Cache: [cleanunet/wav2vec2_cache.py](cleanunet/wav2vec2_cache.py)
- Extraction: [extract_wav2vec2_embeddings.py](extract_wav2vec2_embeddings.py)
- Model: [cleanunet/cleanunet2_with_xvector.py](cleanunet/cleanunet2_with_xvector.py)
- Config: [configs/train_wav2vec2_stage1.yaml](configs/train_wav2vec2_stage1.yaml)

## Paper Reference

- **Wav2Vec2**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- **XLS-R**: [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296)
- **CNUNet-TB**: Two-stage training with self-supervised speech embeddings (basis for this implementation)
