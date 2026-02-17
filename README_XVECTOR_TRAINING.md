# CleanUNet2 with X-Vectors: Two-Stage Training

This document describes the two-stage training approach for CleanUNet2 using X-Vector speaker embeddings, based on the CNUNet-TB paper strategy.

## Overview

### What is Two-Stage Training?

The two-stage training strategy enables the model to benefit from speaker information during training while maintaining fast inference speed:

- **Stage-1**: Train with X-Vectors extracted from clean audio
  - X-Vectors (512-dim speaker embeddings) are injected into the latent space
  - The model learns to denoise using speaker information as guidance
  - Fused latent vectors are saved for Stage-2

- **Stage-2**: Train to replicate Stage-1 latents WITHOUT X-Vectors
  - No X-Vector extractor is used (enabling fast inference)
  - The model learns to predict the fused latents from Stage-1
  - Loss combines reconstruction + latent replication (γ = 0.05)

### Key Benefits

1. **Better Performance**: Speaker information helps the model denoise more effectively
2. **Fast Inference**: Stage-2 model doesn't use X-Vector extractor at inference time
3. **No Speaker Identity Required**: The model learns patterns from speaker embeddings during training, but doesn't need speaker identity at inference

---

## Architecture

### Stage-1 Architecture

```
Noisy Audio ──┬──> CleanUNet Encoder ──> Latent Features ──┐
              │                                              │
              │                                              ├─> Integration Block ──> Fused Latent ──> CleanUNet Decoder ──> Enhanced Audio
              │                                              │
              └──> Clean Audio ──> X-Vector Extractor ──────┘
                                   (Frozen, SpeechBrain)

Noisy Spec ───> CleanSpecNet ──> Denoised Spec ──> SpecUpsampler ──> (Conditioning)
```

**Components:**
- **X-Vector Extractor**: Pretrained SpeechBrain model (spkrec-xvect-voxceleb)
- **Integration Block**: Fuses X-Vectors with latent features via concatenation + Conv1D
- **Latent Storage**: Fused latents are saved to disk during validation

### Stage-2 Architecture

```
Noisy Audio ──┬──> CleanUNet Encoder ──> Latent Features ──> Latent Predictor ──> Predicted Latent ──> CleanUNet Decoder ──> Enhanced Audio
              │
Noisy Spec ───┴──> CleanSpecNet ──> Denoised Spec ──> SpecUpsampler ──> (Conditioning)

Loss = Loss_Reconstruction + γ * Loss_Latent_Replication
       (γ = 0.05 from paper)
```

**Components:**
- **Latent Predictor**: 2-layer Conv1D that learns to replicate Stage-1 fused latents
- **No X-Vector Extractor**: Not used in Stage-2 (fast inference!)
- **Latent Replication Loss**: MSE between predicted latent and stored Stage-1 latent

---

## Installation

### 1. Install Dependencies

```bash
cd /home/fred/Projetos/AKCIT/INTERSPEECH2026/CleanUNet2-Vanilla_xvectors
pip install -r requirements.txt
```

Key dependencies added for X-Vector support:
- `speechbrain>=0.5.16` - For X-Vector extraction
- `huggingface_hub>=0.20.0` - For downloading pretrained models
- `ruamel.yaml` - Additional YAML support

### 2. Download X-Vector Model

The X-Vector extractor will automatically download the pretrained model from HuggingFace on first use:
- Model: `speechbrain/spkrec-xvect-voxceleb`
- Output: 512-dimensional speaker embeddings
- Location: `~/.cache/huggingface/hub/`

---

## Training

### Stage-1: Training with X-Vectors

**Configuration**: `configs/train_xvector_vanilla_stage1.yaml`

```bash
python train_xvector.py \
    --config configs/train_xvector_vanilla_stage1.yaml \
    --stage stage1
```

**What happens:**
1. X-Vectors are extracted from clean audio using frozen SpeechBrain model
2. X-Vectors are fused with latent features via Integration Block
3. Model is trained with reconstruction losses (waveform + spec + phase)
4. During validation, fused latents are saved to `stored_latents_stage1/`

**Important:** Stage-1 validation saves latent files like:
```
stored_latents_stage1/
├── val_batch_000000.pt
├── val_batch_000001.pt
├── val_batch_000002.pt
└── ...
```

These files contain:
- `fused_latent`: Stage-1 fused latent vectors (target for Stage-2)
- `xvector_emb`: X-Vector embeddings
- `latent`: Original latent features
- `noisy_wav`: Noisy audio
- `clean_wav`: Clean audio

### Stage-2: Training WITHOUT X-Vectors

**Configuration**: `configs/train_xvector_vanilla_stage2.yaml`

**IMPORTANT**: Before running Stage-2:
1. Complete Stage-1 training
2. Set `stage1_checkpoint` in the config to your best Stage-1 checkpoint
3. Ensure `stored_latents_stage1/` directory exists with saved latents

```bash
python train_xvector.py \
    --config configs/train_xvector_vanilla_stage2.yaml \
    --stage stage2
```

**What happens:**
1. Stage-1 weights are loaded (except X-Vector extractor)
2. Latent Predictor is initialized randomly
3. Model predicts fused latents using only noisy audio
4. Loss = Reconstruction Loss + γ * Latent Replication Loss
   - γ = 0.05 (from CNUNet-TB paper)
   - Latent loss computed only during validation (matches saved latents)

---

## Configuration

### Stage-1 Config (`train_xvector_vanilla_stage1.yaml`)

Key parameters:
```yaml
latents_dir: "stored_latents_stage1"  # Where to save fused latents

model:
  xvector_dim: 512  # SpeechBrain X-Vector dimension
  conditioning_type: "addition"  # Spec+waveform fusion method

losses:
  weight_waveform: 10.0  # Waveform reconstruction weight
  weight_spec: 1.0       # Spectrogram loss weight
  weight_phase: 1.0      # Phase loss weight
```

### Stage-2 Config (`train_xvector_vanilla_stage2.yaml`)

Key parameters:
```yaml
stage1_checkpoint: "logs/stage1_checkpoints/stage1-best.ckpt"  # REQUIRED!
latents_dir: "stored_latents_stage1"  # Where to load Stage-1 latents

losses:
  gamma_latent: 0.05  # Latent replication loss weight (γ from paper)
  # Other weights same as Stage-1
```

---

## Inference

After Stage-2 training completes, use the Stage-2 model for inference:

```bash
python inference_xvector.py --config configs/inference_xvector.yaml
```

**Configuration**: `configs/inference_xvector.yaml`

```yaml
inference:
  input_dir: "/path/to/noisy/audio/"
  output_dir: "denoised_xvector_results"
  checkpoint_path: "logs/stage2_checkpoints/stage2-best.ckpt"
  use_amp: true  # Use mixed precision for faster inference

audio:
  target_sample_rate: 16000
  normalize: true
  segment_size: 16384  # Adjust based on GPU memory
```

**Key Point**: Inference uses the Stage-2 model, which does NOT require X-Vector extraction. This makes inference fast while still benefiting from speaker information learned during training.

---

## File Structure

```
CleanUNet2-Vanilla_xvectors/
├── configs/
│   ├── train_xvector_vanilla_stage1.yaml  # Stage-1 training config
│   ├── train_xvector_vanilla_stage2.yaml  # Stage-2 training config
│   └── inference_xvector.yaml             # Inference config
│
├── cleanunet/
│   ├── cleanunet2_with_xvector.py         # Main model class
│   ├── integration_block.py               # X-Vector fusion module
│   └── xvector_extractor.py               # SpeechBrain wrapper
│
├── lightning_modules/
│   ├── cleanunet_xvector_stage1_module.py  # Stage-1 Lightning module
│   ├── cleanunet_xvector_stage2_module.py  # Stage-2 Lightning module
│   └── data_module.py                      # Data loading
│
├── train_xvector.py                        # Training script
├── inference_xvector.py                    # Inference script
├── requirements.txt                        # Dependencies
└── README_XVECTOR_TRAINING.md             # This file
```

---

## Loss Functions

### Stage-1 Loss

```python
Loss = w_waveform * Loss_waveform + w_spec * Loss_spec + w_phase * Loss_phase
```

Where:
- `Loss_waveform`: Multi-resolution STFT loss on waveform
- `Loss_spec`: L1 loss on log-magnitude spectrogram
- `Loss_phase`: Anti-wrapping phase loss

### Stage-2 Loss

```python
Loss = Loss_reconstruction + γ * Loss_latent

Loss_reconstruction = w_waveform * Loss_waveform + w_spec * Loss_spec + w_phase * Loss_phase
Loss_latent = MSE(predicted_latent, stored_stage1_latent)
```

Where:
- `γ = 0.05` (from CNUNet-TB paper, Equation 5)
- `Loss_latent` computed only during validation (matches saved batches)

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

**Stage-1 Metrics:**
- `train/loss` - Total training loss
- `val/loss` - Validation loss
- `val/pesq` - Perceptual Evaluation of Speech Quality
- `val/stoi` - Short-Time Objective Intelligibility
- `val/si_sdr` - Scale-Invariant Signal-to-Distortion Ratio
- `val/weighted_score` - Combined metric (used for checkpoint selection)

**Stage-2 Metrics:**
- All Stage-1 metrics, plus:
- `train/loss_latent` - Latent replication loss (training)
- `val/loss_latent` - Latent replication loss (validation)
- `val/loss_recon` - Reconstruction loss

### WandB

Set `logger.choice: "wandb"` in config and update project name:

```yaml
logger:
  choice: "wandb"
  wandb:
    project: "CleanUNet2_XVector"
    name: "stage1_with_xvectors"  # or "stage2_without_xvectors"
```

---

## Checkpoints

### Stage-1 Checkpoints

Location: `logs/stage1_checkpoints/`

- `stage1-best-{epoch}-{score}.ckpt` - Best model based on weighted_score
- `stage1-epoch-{epoch}.ckpt` - Periodic checkpoints
- `last.ckpt` - Most recent checkpoint

### Stage-2 Checkpoints

Location: `logs/stage2_checkpoints/`

- `stage2-best-{epoch}-{score}.ckpt` - Best model (use for inference!)
- `stage2-epoch-{epoch}.ckpt` - Periodic checkpoints
- `last.ckpt` - Most recent checkpoint

---

## Troubleshooting

### Issue: "Latents directory not found"

**Solution**: Run Stage-1 training first to generate latent files.

```bash
python train_xvector.py --config configs/train_xvector_vanilla_stage1.yaml --stage stage1
```

### Issue: "No latent files found in stored_latents_stage1"

**Solution**: Stage-1 must complete at least one validation epoch to save latents. Check:
1. Stage-1 training reached validation
2. `stored_latents_stage1/` directory exists
3. Directory contains `val_batch_*.pt` files

### Issue: "Stage-2 requires 'stage1_checkpoint' in config"

**Solution**: Set the checkpoint path in `train_xvector_vanilla_stage2.yaml`:

```yaml
stage1_checkpoint: "logs/stage1_checkpoints/stage1-best.ckpt"
```

### Issue: Out of Memory (OOM) during training

**Solutions**:
- Reduce `batch_size` in config
- Reduce `segment_size` in data config
- Use `precision: "16-mixed"` in trainer config
- Reduce model size (decrease `channels_H` or `encoder_n_layers`)

### Issue: X-Vector model download fails

**Solution**: Manually download the model:

```python
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)
```

---

## Expected Results

Based on the CNUNet-TB paper:

- **Stage-1 (with X-Vectors)**: Better denoising performance due to speaker information
- **Stage-2 (without X-Vectors)**: Similar performance to Stage-1, but faster inference
- **Improvement over Vanilla**: ~5-10% improvement in PESQ/STOI metrics

**Inference Speed:**
- Stage-1: Slower (requires X-Vector extraction)
- Stage-2: Fast (no X-Vector extractor needed)

---

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{cnunet_tb,
  title={Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings},
  author={...},
  booktitle={INTERSPEECH},
  year={2024}
}
```

---

## References

1. CNUNet-TB Paper: Two-stage training with self-supervised speech embeddings
2. SpeechBrain X-Vectors: https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
3. CleanUNet2: Original speech denoising architecture

---

## Notes

- The X-Vector extractor is **frozen** during Stage-1 training (no gradient updates)
- Latent files can be large (~500MB per 1000 validation batches)
- Stage-2 can take longer to converge than Stage-1
- The latent replication loss (γ) can be tuned (paper uses 0.05)
- For best results, train Stage-1 to convergence before starting Stage-2

---

For questions or issues, please refer to the original CleanUNet2 documentation or the CNUNet-TB paper.
