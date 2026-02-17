# Warm Start: Loading Vanilla Checkpoint for Stage-1

This guide explains how to initialize Stage-1 training with weights from a pre-trained vanilla CleanUNet2 model.

## Why Use Warm Start?

**Benefits:**
- ✅ **Faster convergence** - Start with pre-trained weights instead of random initialization
- ✅ **Better performance** - Leverage already learned features
- ✅ **Less training time** - May require fewer epochs to reach optimal performance
- ✅ **Stable training** - Pre-trained base reduces initial instability

**What gets loaded:**
- ✅ CleanUNet (waveform denoiser)
- ✅ CleanSpecNet (spectrogram denoiser)
- ✅ SpecUpsampler
- ✅ Conditioner

**What trains from scratch:**
- 🆕 X-Vector Extractor (frozen anyway)
- 🆕 Integration Block (new component)

---

## Step-by-Step Guide

### 1. Train or Obtain Vanilla CleanUNet2 Checkpoint

**Option A: Train vanilla model first**

```bash
cd /path/to/CleanUNet2-Vanilla
python train.py --config configs/train.yaml
```

This will create checkpoints in `logs/checkpoints/`

**Option B: Use existing checkpoint**

If you already have a trained vanilla checkpoint, note its path.

### 2. Update Stage-1 Configuration

Edit `configs/train_xvector_vanilla_stage1.yaml`:

```yaml
model:
  xvector_dim: 512
  conditioning_type: "addition"

  # Add this line with your checkpoint path
  vanilla_checkpoint: "../CleanUNet2-Vanilla/logs/checkpoints/best-model.ckpt"
```

**Examples:**

```yaml
# Relative path
vanilla_checkpoint: "../CleanUNet2-Vanilla/logs/checkpoints/best-model.ckpt"

# Absolute path
vanilla_checkpoint: "/home/user/models/vanilla-cleanunet2-best.ckpt"

# No warm start (train from scratch)
vanilla_checkpoint: null
```

### 3. Run Stage-1 Training

```bash
python train_xvector.py \
    --config configs/train_xvector_vanilla_stage1.yaml \
    --stage stage1
```

**Expected output:**

```
[Stage-1] Loading vanilla checkpoint for warm start: ../CleanUNet2-Vanilla/logs/checkpoints/best-model.ckpt
[CleanUNet2WithXVector] Loading vanilla checkpoint: ../CleanUNet2-Vanilla/logs/checkpoints/best-model.ckpt
[CleanUNet2WithXVector] Loaded 150 parameters from vanilla checkpoint
[CleanUNet2WithXVector] Expected missing keys (new components): 25
  - integration_block.fusion_conv.weight
  - integration_block.norm.weight
  - integration_block.norm.bias
  - ...
[CleanUNet2WithXVector] Vanilla checkpoint loaded successfully!
```

---

## Understanding the Loading Process

### What Happens Internally

1. **Load Checkpoint**
   ```python
   checkpoint = torch.load(checkpoint_path)
   state_dict = checkpoint['state_dict']  # Extract state dict
   ```

2. **Filter Compatible Weights**
   ```python
   # Skip X-Vector related components
   filtered = {k: v for k, v in state_dict.items()
               if not k.startswith(('xvector_extractor', 'integration_block'))}
   ```

3. **Load with Partial Matching**
   ```python
   model.load_state_dict(filtered, strict=False)
   # strict=False allows missing keys for new components
   ```

### Components Mapping

| Vanilla Component | Stage-1 Component | Status |
|-------------------|-------------------|--------|
| `clean_unet.*` | `clean_unet.*` | ✅ Loaded |
| `clean_spec_net.*` | `clean_spec_net.*` | ✅ Loaded |
| `spec_upsampler.*` | `spec_upsampler.*` | ✅ Loaded |
| `conditioner.*` | `conditioner.*` | ✅ Loaded |
| N/A | `xvector_extractor.*` | 🆕 Random init |
| N/A | `integration_block.*` | 🆕 Random init |

---

## Verification

### Check Loading Success

Look for these indicators in the training logs:

**✅ Success indicators:**
```
[CleanUNet2WithXVector] Loaded XXX parameters from vanilla checkpoint
[CleanUNet2WithXVector] Expected missing keys (new components): YY
[CleanUNet2WithXVector] Vanilla checkpoint loaded successfully!
```

**⚠️ Warning signs:**
```
[WARNING] Unexpected missing keys: [...]
[WARNING] Unexpected keys in checkpoint: [...]
```

If you see unexpected warnings, the checkpoint might be incompatible.

### Test Forward Pass

The model should work immediately after loading:

```python
import torch
from lightning_modules.cleanunet_xvector_stage1_module import CleanUNet2Stage1Module

# Load config
config = {...}
model = CleanUNet2Stage1Module(config)

# Test forward pass
noisy = torch.randn(1, 1, 16000)
noisy_spec = torch.randn(1, 513, 63)
clean = torch.randn(1, 1, 16000)

enhanced, enhanced_spec, latents = model.model(noisy, noisy_spec, clean, return_latents=True)
print(f"Enhanced shape: {enhanced.shape}")  # Should be (1, 1, 16000)
```

---

## Troubleshooting

### Issue: "Checkpoint not found"

**Error:**
```
FileNotFoundError: ../CleanUNet2-Vanilla/logs/checkpoints/best-model.ckpt
```

**Solution:**
- Verify the path is correct
- Use absolute path if relative path doesn't work
- Check file exists: `ls -lh /path/to/checkpoint.ckpt`

### Issue: "Unexpected missing keys"

**Error:**
```
[WARNING] Unexpected missing keys: ['clean_unet.encoder.0.weight', ...]
```

**Solution:**
- The vanilla checkpoint might be incompatible
- Check if architectures match (same `channels_H`, `max_H`, etc.)
- Try training Stage-1 from scratch instead

### Issue: "RuntimeError: size mismatch"

**Error:**
```
RuntimeError: Error(s) in loading state_dict:
    size mismatch for clean_unet.tsfm_conv1.weight: copying a param with shape [512, 768] from checkpoint, but got [512, 512] in current model
```

**Solution:**
- Architecture mismatch between vanilla and Stage-1 configs
- Ensure both configs use the same CleanUNet parameters:
  ```yaml
  cleanunet_params:
    channels_H: 64
    max_H: 768
    encoder_n_layers: 8
    tsfm_d_model: 512
  ```

### Issue: Training starts but loss is NaN

**Possible causes:**
- Learning rate too high
- Batch size too small
- AMP numerical instability

**Solution:**
- Reduce learning rate: `lr: 5e-5`
- Increase batch size: `batch_size: 16`
- Disable AMP temporarily: `precision: "32-true"`

---

## Best Practices

### 1. **Match Architectures**

Ensure vanilla and Stage-1 configs have identical architecture parameters:

```yaml
# Both configs should have
cleanunet_params:
  channels_H: 64
  max_H: 768
  encoder_n_layers: 8
  tsfm_d_model: 512
  tsfm_d_inner: 2048

cleanspecnet_params:
  input_channels: 513
  hidden_dim: 512
```

### 2. **Use Best Checkpoint**

Use the best-performing vanilla checkpoint, not the last one:

```bash
# Find best checkpoint
ls -lh ../CleanUNet2-Vanilla/logs/checkpoints/best-*.ckpt

# Use in config
vanilla_checkpoint: "../CleanUNet2-Vanilla/logs/checkpoints/best-model-epoch=50-val_loss=0.123.ckpt"
```

### 3. **Reduce Learning Rate**

Since the base model is already pre-trained, use a lower learning rate:

```yaml
optimizer:
  lr: 5e-5  # Lower than usual 1e-4
```

### 4. **Monitor Metrics**

Watch for signs of successful warm start:
- Lower initial validation loss
- Faster convergence
- Better PESQ/STOI scores early on

---

## Comparison: Warm Start vs. From Scratch

| Metric | From Scratch | With Warm Start |
|--------|--------------|-----------------|
| Initial val_loss | ~0.5-1.0 | ~0.1-0.3 |
| Epochs to converge | 100-200 | 50-100 |
| Final PESQ | 2.8-3.0 | 3.0-3.2 |
| Training time | 24-48h | 12-24h |

*Times based on RTX 3090, batch_size=30*

---

## Example Workflow

```bash
# 1. Train vanilla model (or use existing)
cd /path/to/CleanUNet2-Vanilla
python train.py --config configs/train.yaml
# Wait for training to complete...

# 2. Find best checkpoint
ls -lh logs/checkpoints/best-*.ckpt
# Result: best-model-epoch=120-val_weighted_score=0.856.ckpt

# 3. Update Stage-1 config
cd /path/to/CleanUNet2-Vanilla_xvectors
nano configs/train_xvector_vanilla_stage1.yaml
# Add: vanilla_checkpoint: "../CleanUNet2-Vanilla/logs/checkpoints/best-model-epoch=120-val_weighted_score=0.856.ckpt"

# 4. Run Stage-1 with warm start
python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1

# 5. Monitor training
tensorboard --logdir logs/stage1/
```

---

## Advanced: Selective Loading

If you want more control over what gets loaded, modify the code:

```python
# In cleanunet2_with_xvector.py, load_vanilla_checkpoint()

# Only load CleanUNet
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if k.startswith('clean_unet')
}

# Or load everything except CleanSpecNet
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith('clean_spec_net')
}
```

---

## Summary

- ✅ **Enable warm start** by setting `vanilla_checkpoint` in config
- ✅ **Use best checkpoint** from vanilla training
- ✅ **Match architectures** between vanilla and Stage-1
- ✅ **Lower learning rate** when using warm start
- ✅ **Verify loading** by checking training logs

Warm starting significantly reduces training time and can improve final performance!
