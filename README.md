# CleanUNet2 — Speech Enhancement with Speaker Embeddings

CleanUNet2 is a deep-learning architecture for **speech enhancement**, combining:

* **CleanUNet (waveform UNet)**
* **CleanSpecNet (frequency-domain transformer network)**
* **Speaker embeddings** for speaker-informed denoising
* **Two-stage training**: Stage 1 uses speaker embeddings, Stage 2 replicates without them

---

## Supported Speaker Embedding Models

| Model | Dim | Backend | Dependency |
|-------|-----|---------|------------|
| `xvector` | 512 | SpeechBrain | `speechbrain` |
| `ecapa` | 192 | SpeechBrain | `speechbrain` |
| `clova` | 512 | Local checkpoint | — |
| `redimnet` | 192 | torch.hub | — |
| `titanet` | 192 | NeMo | `nemo_toolkit[asr]` |
| `speakernet` | 256 | NeMo | `nemo_toolkit[asr]` |

The model is selected via the `speaker_model` field in the YAML config. The embedding dimension is handled automatically.

---

## Repository Structure

```
.
├── cleanunet/
│   ├── cleanunet.py                        # CleanUNet (waveform model)
│   ├── cleanspecnet.py                     # CleanSpecNet (spectrogram model)
│   ├── cleanunet2.py                       # Full hybrid CleanUNet2 system
│   ├── cleanunet2_with_speaker_embeddings.py  # Two-stage model with speaker embeddings
│   ├── speaker_extractor.py               # Multi-model speaker embedding extractor
│   ├── integration_block.py               # Embedding-latent fusion block
│   ├── xvector_cache.py                   # Embedding cache system
│
├── lightning_modules/
│   ├── cleanunet_speaker_embeddings_stage1_module.py
│   ├── cleanunet_speaker_embeddings_stage2_module.py
│   ├── data_module.py
│
├── configs/
│   ├── stage1_xvector.yaml    # Stage 1 configs (one per model)
│   ├── stage1_ecapa.yaml
│   ├── stage1_clova.yaml
│   ├── stage1_redimnet.yaml
│   ├── stage1_titanet.yaml
│   ├── stage1_speakernet.yaml
│   ├── stage2_xvector.yaml    # Stage 2 configs (one per model)
│   ├── stage2_ecapa.yaml
│   ├── stage2_clova.yaml
│   ├── stage2_redimnet.yaml
│   ├── stage2_titanet.yaml
│   ├── stage2_speakernet.yaml
│   ├── test_*.yaml            # Quick test configs
│   └── inference.yaml
│
├── filelists/
│   ├── train.csv
│   └── test.csv
│
├── train.py                   # Training entrypoint
├── inference.py               # Inference script
├── losses.py                  # Loss functions (MR-STFT, Phase Loss)
├── spec_dataset.py            # Dataset loader
└── metrics.py                 # PESQ/STOI/SI-SDR
```

---

## Installation

### Base environment (xvector, ecapa, clova, redimnet)

```bash
pip install torch torchaudio pytorch-lightning speechbrain
pip install torchmetrics pesq pystoi
```

### For TitaNet / SpeakerNet (NeMo models)

```bash
conda create -n nemo python=3.10
conda activate nemo
pip install 'nemo_toolkit[asr]'
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install speechbrain torchmetrics pesq pystoi
```

---

## Training

Two-stage training approach:

1. **Stage 1**: Speaker embeddings are extracted from clean audio and fused into the latent space
2. **Stage 2**: The model learns to replicate Stage 1 latents without the embedding extractor (fast inference)

### Stage 1

```bash
# X-Vector (512-dim)
python train.py --config configs/stage1_xvector.yaml --stage stage1

# ECAPA-TDNN (192-dim)
python train.py --config configs/stage1_ecapa.yaml --stage stage1

# Clova ResNet (512-dim)
python train.py --config configs/stage1_clova.yaml --stage stage1

# ReDimNet (192-dim)
python train.py --config configs/stage1_redimnet.yaml --stage stage1

# TitaNet-Large (192-dim) — requires nemo env
python train.py --config configs/stage1_titanet.yaml --stage stage1

# SpeakerNet (256-dim) — requires nemo env
python train.py --config configs/stage1_speakernet.yaml --stage stage1
```

### Stage 2

After Stage 1 finishes, update the `stage1_checkpoint` path in the corresponding Stage 2 config, then:

```bash
python train.py --config configs/stage2_xvector.yaml --stage stage2
python train.py --config configs/stage2_ecapa.yaml --stage stage2
python train.py --config configs/stage2_clova.yaml --stage stage2
python train.py --config configs/stage2_redimnet.yaml --stage stage2
python train.py --config configs/stage2_titanet.yaml --stage stage2
python train.py --config configs/stage2_speakernet.yaml --stage stage2
```

---

## Config Structure

The key model field that selects the speaker embedding model:

```yaml
model:
  speaker_model: titanet  # xvector | ecapa | clova | redimnet | titanet | speakernet

  # Optional: local path for clova checkpoint or redimnet variant override
  # speaker_model_local_path: /path/to/checkpoints_speaker_encoder_clova
  # speaker_model_local_path: "b6:ft_lm:vox2"  # for redimnet variants
```

Stage 2 configs must use the same `speaker_model` as the corresponding Stage 1 (to match the IntegrationBlock dimensions).

---

## Inference

```bash
python inference.py --config configs/inference.yaml
```

---

## Datasets

The default setup assumes **VoiceBank-DEMAND (16 kHz)**.

Filelist format (`train.csv`, `test.csv`):

```
train/clean/p226_001.wav|train/noisy/p226_001.wav
train/clean/p226_002.wav|train/noisy/p226_002.wav
```

---

## Architecture

### Two-Stage Training

```
Stage 1:
  noisy_wav → CleanSpecNet → SpecUpsampler → Conditioner → CleanUNet.encode()
                                                                ↓
  clean_wav → SpeakerExtractor → IntegrationBlock(latent, embedding)
                                                                ↓
                                                   CleanUNet.decode() → enhanced_wav

Stage 2:
  noisy_wav → CleanSpecNet → SpecUpsampler → Conditioner → CleanUNet.encode()
                                                                ↓
                                                   LatentPredictor(latent)
                                                                ↓
                                                   CleanUNet.decode() → enhanced_wav
```

Stage 2 does NOT use the speaker embedding extractor at inference time — it learns to predict the fused latent directly.

---

## Losses

| Loss | Purpose |
|------|---------|
| L1 waveform loss | Basic reconstruction |
| MR-STFT Loss | Spectral convergence + log magnitude |
| Anti-Wrapping Phase Loss | Phase consistency |
| Spectrogram Log-Magnitude Loss | Auxiliary stabilization |

---

## Validation Metrics

* **PESQ** (Perceptual Evaluation of Speech Quality)
* **STOI** (Short-Time Objective Intelligibility)
* **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)

---

## License

MIT License
