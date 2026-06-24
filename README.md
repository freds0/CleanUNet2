# 🎙️ CleanUNet2 — Baseline

**Single-stage speech enhancement** with the CleanUNet2 hybrid architecture —
**no SSL embeddings, no speaker embeddings**. This is the plain waveform +
spectrogram denoiser used as the baseline for the SSL/speaker-embedding experiments.

Based on: *"Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture
Using Self-Supervised Speech Embeddings"* (architecture only; this baseline drops the
self-supervised branch).

## Core Architecture

CleanUNet2 combines:
* **CleanUNet** — waveform-domain encoder-decoder U-Net with a Transformer bottleneck
* **CleanSpecNet** — spectrogram-domain denoiser (conv + self-attention)
* **SpecUpsampler** — expands the refined spectrogram into a waveform-length feature
* **Conditioner** — fuses noisy waveform + upsampled feature (`addition`, `concatenation`, or `film`)

Denoising workflow:
1. **CleanSpecNet** refines the noisy spectrogram.
2. **SpecUpsampler** expands it to a waveform-length feature.
3. **Conditioner** fuses the noisy waveform with that feature.
4. **CleanUNet** produces the enhanced waveform.

Training objective: multi-resolution STFT loss + anti-wrapping phase loss +
log-magnitude spectrogram loss, weighted (waveform / spec / phase).

---

## 📁 Repository Structure

```
.
├── configs/
│   ├── config.yaml             # Training configuration (single stage)
│   └── inference.yaml          # Inference configuration
├── lightning_modules/
│   ├── cleanunet_module.py     # Lightning training module (CleanUNetLightningModule)
│   └── data_module.py          # DataModule (paired / clean+noise / multi-dataset)
├── cleanunet/
│   ├── cleanunet.py            # CleanUNet (waveform model)
│   ├── cleanspecnet.py         # CleanSpecNet (spectrogram model)
│   ├── cleanunet2.py           # Full hybrid CleanUNet2 system
│   ├── util.py, logger.py      # Helpers
├── filelists/                  # train.csv / test.csv (clean|noisy)
├── train.py                    # Training script
├── inference.py                # Inference script (sliding-window OLA)
├── losses.py                   # MR-STFT + phase + waveform losses
├── metrics.py                  # PESQ / STOI / SI-SDR
├── spec_dataset.py             # Dataset loader
├── augmentation.py             # On-the-fly noise augmentation
└── requirements.txt
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## 🎚️ Training

Single stage — no embedding pre-extraction, no Stage-2:

```bash
python train.py --config configs/config.yaml
```

Quick smoke test (1 epoch, 30 samples):

```bash
python train.py --config configs/config.yaml --quick-test
```

Monitor:

```bash
tensorboard --logdir experiments/
```

Logged: train/val losses, plus PESQ / STOI / SI-SDR validation metrics.

---

## 🎤 Inference (Denoising Audio)

Edit `configs/inference.yaml` (set `checkpoint_path`, `input_dir`, `output_dir`;
keep `model.conditioning_type` equal to the training value), then run:

```bash
python inference.py --config configs/inference.yaml
```

Denoised WAV files are written to the configured `output_dir`.

---

## 📦 Datasets

Default setup assumes **VoiceBank-DEMAND (16 kHz)**.

Filelist format (`train.csv`, `test.csv`), one pair per line:

```
clean_file.wav|noisy_file.wav
```

---

## ⚙️ Configuration (`configs/config.yaml`)

The training module (`CleanUNetLightningModule`) reads model/loss/optimizer keys at
the config **root**; `trainer` / `data` / `callbacks` / `logger` are nested and read
by `train.py`.

| Key | Default | Notes |
|-----|---------|-------|
| `conditioning_type` | `addition` | `addition` \| `concatenation` \| `film` |
| `weight_waveform` / `weight_spec` / `weight_phase` | `10 / 5 / 5` | Loss weights |
| `lr` | `5.0e-05` | AdamW |
| `data.segment_size` | `32000` | 2 s @ 16 kHz |
| `data.batch_size` | `16` | |
| `trainer.precision` | `16-mixed` | |
| `trainer.gradient_clip_val` | `5.0` | |

`CleanUNet2` uses its default sub-module architecture (CleanUNet: `channels_H=64`,
`max_H=768`, 8 encoder layers, 5-layer Transformer; CleanSpecNet: 513 bins, 5 conv +
5 attention layers).

---

## 📚 References

- **Paper:** Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings
- **Dataset:** VoiceBank-DEMAND corpus (16 kHz)
