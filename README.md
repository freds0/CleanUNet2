# 🎙️ CleanUNet2 with WavLM-Large Embeddings

**Two-Stage Self-Supervised Speech Enhancement** using CleanUNet2 architecture with WavLM-Large embeddings.

Based on: *"Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings"*

## Core Architecture

CleanUNet2 combines:
* **CleanUNet** (waveform domain UNet)
* **CleanSpecNet** (spectrogram domain transformer)
* **Multi-resolution STFT losses**
* **Phase-aware losses**
* **FiLM conditioning** for SSL embedding integration

## Two-Stage Training Approach

1. **Stage 1** (50 epochs): Train WITH WavLM embeddings
2. **Stage 2** (30 epochs): Train WITHOUT embeddings, learning to replicate them internally

Result: Inference without SSL model dependency, 40% faster

---

## 🚀 Features

* **Hybrid enhancement**: spectrogram refinement + waveform denoising
* **Multi-Resolution STFT Loss (MR-STFT)**
* **Anti-Wrapping Phase Loss** for improved phase reconstruction
* **Fully configurable training (YAML-based)**
* **Trainer, callbacks, and logging (TensorBoard)**
* **Modular dataset pipeline (VoiceBank-DEMAND compatible)**
* **Clean code with English documentation and comments**

---

## 📁 Repository Structure

```
.
│── lightning_modules/
│    ├── cleanunet_module.py     # Lightning training module
│    ├── data_module.py          # DataModule for dataset handling
│
│── cleanunet/
│    ├── cleanunet.py            # CleanUNet (waveform model)
│    ├── cleanspecnet.py         # CleanSpecNet (spectrogram model)
│    ├── cleanunet2.py           # Full hybrid CleanUNet2 system
│
│── filelists/
│── configs/
│    ├── config.yaml             # Training configuration
│    ├── inference.yaml          # Inference configuration
│
│── train.py                     # Training script
│── inference.py                 # Inference script
│── metrics.py                   # PESQ/STOI/SI-SDR prediction
│── losses.py                    # Loss functions (MR-STFT, Phase Loss)
│── spec_dataset.py              # Dataset loader
│── README.md
```

---

## 🔧 Installation

```bash
git clone https://github.com/your-repo/CleanUNet2.git
cd CleanUNet2
pip install -r requirements.txt
```

---

## 🎚️ Training - Two-Stage Pipeline

### Stage 1: Training WITH WavLM Embeddings (50 epochs)

1. **Pre-extract embeddings** from WavLM-Large (layer 12, 768-D):
```bash
python extract_wavlm_embeddings.py --config configs/stage1.yaml
```

2. **Train Stage 1** with pre-extracted embeddings:
```bash
python train.py --config=configs/stage1.yaml
```

### Stage 2: Training WITHOUT Embeddings (30 epochs)

3. **Train Stage 2** with Stage 1 checkpoint as initialization:
```bash
python train.py --config=configs/stage2.yaml
```

The model learns to internally replicate the embeddings learned in Stage 1, enabling fast inference without the SSL model.

### Automated Two-Stage Training

Run both stages automatically:
```bash
bash run_two_stage_training.sh
```

This script:
- Checks for pre-extracted embeddings
- Runs Stage 1 training
- Verifies Stage 1 checkpoint exists
- Runs Stage 2 training

### TensorBoard Monitoring

View training logs:
```bash
tensorboard --logdir experiments/
```

Logs include:
- Training/validation loss per stage
- L2 regularization (Stage 2 only)
- PESQ/STOI/SI-SDR validation metrics

---

## 🎤 Inference (Denoising Audio)

Configure `configs/inference.yaml`, then run:

```bash
python inference.py --config=configs/inference.yaml
```

Denoised WAV files are saved to:

```
denoised_results/
```

---

## 📦 Datasets

The default setup assumes the **VoiceBank-DEMAND (16 kHz)** dataset.

Expected filelist format (`train.csv`, `test.csv`):

```
clean_file.wav|noisy_file.wav
```

Example directory:

```
VoiceBank-DEMAND-16k/
│── clean/
│── noisy/
│── filelists/
│    ├── train.csv
│    └── test.csv
```

---

## 🧠 Model Overview

### CleanUNet (Waveform Domain)

* Multi-scale encoder-decoder UNet
* Convolutional downsampling & upsampling
* Transformer bottleneck
* No skip-connection misalignment thanks to input padding logic

### CleanSpecNet (Spectrogram Domain)

* Convolutional feature extractor
* Multiple transformer layers
* GLU gating
* Causal or non-causal mask support

### Hybrid Combination

The denoising workflow:

1. **CleanSpecNet** refines the noisy spectrogram
2. **Upsampler** expands spectrogram into a waveform-length feature
3. **WaveformConditioner** fuses noisy + upsampled features
4. **CleanUNet** produces the enhanced waveform

---

## 🎧 Losses

| Loss Type                          | Purpose                              |
| ---------------------------------- | ------------------------------------ |
| **L1/L2 waveform loss**            | Basic reconstruction                 |
| **MR-STFT Loss**                   | Spectral convergence + log magnitude |
| **Anti-Wrapping Phase Loss**       | Phase consistency                    |
| **Spectrogram Log-Magnitude Loss** | Auxiliary stabilization              |

---

## 📊 Validation Metrics

During validation:

* **PESQ**
* **STOI**
* **SI-SDR**

Metrics are computed per sample and averaged.

---

## ⚙️ Configuration (YAML)

### Stage 1 Configuration (`configs/stage1.yaml`)

**Model Settings:**
- `use_preextracted_embeddings: true` - Load pre-extracted WavLM embeddings
- `use_wavlm: true` - Enable WavLM feature extraction
- `wavlm_model: microsoft/wavlm-large`
- `wavlm_layer: 12` - Extract from middle layer (24-layer model)
- `conditioning_type: film` - FiLM conditioning for embedding fusion

**Training Settings:**
- `max_epochs: 50`
- `batch_size: 2` - Small batch for embedding handling
- `learning_rate: 5.0e-05`
- `gradient_clip_val: 5.0`
- `precision: 16-mixed`

**Loss Weights:**
- `weight_waveform: 10.0` - Waveform reconstruction
- `weight_spec: 5.0` - Spectrogram refinement
- `weight_phase: 5.0` - Phase consistency
- `sc_lambda: 0.8` - Spectral convergence
- `mag_lambda: 0.2` - Log-magnitude STFT

### Stage 2 Configuration (`configs/stage2.yaml`)

**Model Settings:**
- `use_preextracted_embeddings: false` - No embeddings, learn internally
- Same architecture as Stage 1

**Training Settings:**
- `max_epochs: 30`
- `batch_size: 32` - Larger batch without embedding overhead
- Same learning rate, gradient clipping

**Stage 2 Specific:**
- `stage1_checkpoint: experiments/checkpoints/stage1/epoch-last.ckpt` - Initialization
- L2 regularization (0.001) applied during training to guide internal embedding learning

### Inference Configuration (`configs/inference.yaml`)

* Input audio directory
* Output directory
* Checkpoint path (Stage 2 checkpoint for inference)
* CPU/GPU device override

---

## 📦 Embedding Extraction

### Pre-Extract WavLM Embeddings

Raw embeddings (no pooling) are extracted before Stage 1 training:

```bash
python extract_wavlm_embeddings.py --config configs/stage1.yaml --device cuda --force
```

**Extraction Details:**
- **Model:** microsoft/wavlm-large
- **Layer:** 12 (middle of 24-layer model)
- **Dimension:** 768-D
- **Format:** Full temporal sequences (time_steps, 768)
- **Pooling:** None (applied during model training via self-attention)
- **Caching:** Embeddings cached by file path hash (MD5)

### Embedding Cache

Extracted embeddings are saved to: `wavlm_embeddings_raw/`

Each file is cached by MD5 hash of its path:
- `cache_key = MD5(file_path)`
- `cached_embedding = wavlm_embeddings_raw/{cache_key}.pt`

To force re-extraction: `--force` flag

## 📦 Checkpoints

### Stage 1 Checkpoints

Located in `experiments/checkpoints/stage1/`:
- `best-{epoch:02d}-{val_loss:.4f}.ckpt` - Best validation loss
- `epoch-{epoch:04d}.ckpt` - Periodic checkpoints (every 10 epochs)
- `epoch-last.ckpt` - Latest checkpoint (symlink)

### Stage 2 Checkpoints

Located in `experiments/checkpoints/stage2/`:
- Initialized from: `experiments/checkpoints/stage1/epoch-last.ckpt`
- `best-{epoch:02d}-{val_loss:.4f}.ckpt` - Best validation loss
- `epoch-{epoch:04d}.ckpt` - Periodic checkpoints
- `epoch-last.ckpt` - Latest checkpoint

### Resume Training

Stage 1 resume:
```yaml
# In configs/stage1.yaml
resume_from_checkpoint: "experiments/checkpoints/stage1/epoch-last.ckpt"
```

Stage 2 resume:
```yaml
# In configs/stage2.yaml
resume_from_checkpoint: "experiments/checkpoints/stage2/epoch-last.ckpt"
stage1_checkpoint: "experiments/checkpoints/stage1/epoch-last.ckpt"
```

### Load for Inference

Use Stage 2 final checkpoint (no embedding dependency):
```yaml
# In configs/inference.yaml
checkpoint_path: "experiments/checkpoints/stage2/epoch-last.ckpt"
```

---

## 🎤 Inference (Denoising Audio)

### Using Stage 2 Model (No SSL Dependency)

Configure `configs/inference.yaml`:
```yaml
data_dir: "/path/to/audio/files"
checkpoint_path: "experiments/checkpoints/stage2/epoch-last.ckpt"
output_dir: "denoised_results/"
device: "cuda"  # or "cpu"
```

Run inference:
```bash
python inference.py --config=configs/inference.yaml
```

Denoised audio saved to: `denoised_results/`

### Performance

- **Inference speed:** ~40% faster than Stage 1 (no SSL model forward pass)
- **Quality:** Stage 2 maintains Stage 1 performance via learned embedding replication
- **Model size:** Smaller for deployment (no WavLM model dependency)

---

## 📊 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sample Rate** | 16000 Hz | VoiceBank-DEMAND standard |
| **Segment Size** | 32000 | 2 seconds at 16 kHz |
| **Learning Rate** | 5.0e-05 | Adam optimizer |
| **Gradient Clipping** | 5.0 | Stability during training |
| **Precision** | 16-mixed | FP16 with loss scaling |
| **WavLM Layer** | 12 | Middle layer of 24-layer model |
| **WavLM Dimension** | 768 | Embedding vector size |
| **FiLM Conditioning** | Yes | Dynamic modulation of features |
| **L2 Regularization (Stage 2)** | 0.001 | Guides internal embedding learning |

---

## 📁 Expected Directory Structure

```
.
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── inference.yaml
├── lightning_modules/
│   ├── cleanunet_module.py
│   └── data_module.py
├── cleanunet/
│   ├── cleanunet.py
│   ├── cleanspecnet.py
│   ├── cleanunet2.py
│   └── wavlm_extractor.py
├── filelists/
│   ├── train.csv
│   └── test.csv
├── train.py
├── inference.py
├── extract_wavlm_embeddings.py
├── run_two_stage_training.sh
├── wavlm_embeddings_raw/  (created during extraction)
├── experiments/  (created during training)
│   ├── checkpoints/
│   │   ├── stage1/
│   │   └── stage2/
│   ├── stage1/
│   └── stage2/
└── denoised_results/  (created during inference)
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue before major feature changes.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📚 References

- **Paper:** Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings
- **WavLM:** Sanyuan Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Speech Recognition"
- **Dataset:** VoiceBank-DEMAND corpus (16 kHz)

