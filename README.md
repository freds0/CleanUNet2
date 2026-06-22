# 🎵 CleanUNet2-SSL: Speech Denoising with Self-Supervised Embeddings

**Version**: 3.0 (consolidated multi-backbone) 
**Status**: ✅ Active 
**License**: MIT

---

## 📢 Overview

**CleanUNet2-SSL** is a speech-enhancement architecture that conditions a hybrid
waveform/spectrogram denoiser on **self-supervised (SSL) speech embeddings**. It combines:

- **🎤 Spectrogram refinement** — CleanSpecNet (frequency-domain transformer)
- **🌊 Waveform denoising** — CleanUNet (multi-scale encoder-decoder)
- **🧠 SSL embeddings** — representations from one of five interchangeable backbones
- **🔗 Hybrid conditioning** — FiLM-based fusion of spectrogram and waveform features

This repo **consolidates all SSL backbones behind one CleanUNet2 base**. The SSL family
is selected purely by config (`model.ssl.type`); the base architecture, both Lightning
modules, and the training entry point are family-agnostic.

---

## 🧩 SSL backbones and the `+` / `++` variants

A single base, two layer-fusion variants. The only difference between `+` and `++` is
which hidden states feed the learnable softmax fusion (`model.ssl.selected_layers`):

| Variant | `selected_layers` | Meaning |
|---|---|---|
| `+`  (plus)     | `[first, mid, last]` | learnable softmax over **3 selected** hidden states |
| `++` (plusplus) | `"all"`              | learnable softmax over **all** hidden states |

Supported families (`model.ssl.type` → backbone):

| `type` | Backbone | Feature path |
|---|---|---|
| `wav2vec2` | `facebook/wav2vec2-xls-r-2b` | raw waveform |
| `hubert`   | `facebook/hubert-large-ll60k` | raw waveform |
| `wavlm`    | `microsoft/wavlm-large` | raw waveform |
| `w2v-bert` | `facebook/w2v-bert-2.0` | log-mel + attention mask |
| `whisper`  | `openai/whisper-large-v3` | mel features |

Dispatch lives in `cleanunet/ssl_extractor_factory.py`. Each family has its own extractor
module (`cleanunet/<family>_extractor.py`) sharing an identical constructor signature, so
the rest of the pipeline never branches on backbone type.

---

## 🚀 Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Quick test (1 epoch, 30 samples)

```bash
python train.py --config configs/config_wavlm_plusplus.json --stage 1 --quick-test
```

### 3. Train

Two-stage pipeline. **Stage 1** trains the SSL layer-softmax + the denoiser (SSL embeddings
injected into the latent space). **Stage 2** trains a latent predictor to reproduce the
Stage-1 latents *without* the SSL extractor at inference time.

```bash
# Stage 1, then Stage 2 (any family/variant):
python train.py --config configs/config_wavlm_plusplus.json --stage 1
python train.py --config configs/config_wavlm_plusplus.json --stage 2

# e.g. whisper '+':
python train.py --config configs/config_whisper_plus.json --stage 1
```

`--stage` selects which `stage1` / `stage2` override block is deep-merged onto the shared
config (checkpoint paths, logger run names, save dirs). The Stage-2 override sets
`pipeline.stage1_checkpoint`, which must point at a trained Stage-1 checkpoint.

> Configs default to **on-the-fly** SSL extraction (`model.ssl.use_preextracted: false`),
> which is required for the learnable layer-softmax to receive gradients.

#### Generating Stage-1 latents for Stage 2

Stage 2 trains by replicating the **fused latents** that Stage 1 writes (one
`val_batch_*.pt` per validation batch, in `latents_dir`, default `stored_latents_stage1/`).
Those files are produced inside the Stage-1 **validation loop**, so by default they reflect
whatever the model looked like at each validation pass — including the barely-trained
epoch-1 snapshot, accumulated across epochs.

To get a single, clean set of latents from a **fully-trained** Stage-1 checkpoint without
retraining, run one validation pass with `generate_latents.py`:

```bash
python generate_latents.py \
  --config configs/config_wavlm_plusplus.json \
  --checkpoint experiments/wavlm_plusplus/checkpoints/stage1/cleanunet-stage1-last.ckpt
# optional: --latents-dir stored_latents_stage1   --device auto|cpu|cuda
```

This loads the trained weights and reuses the Stage-1 `validation_step` (no optimizer
step), writing `val_batch_000000.pt …` over a single pass. Then train Stage 2 normally;
it reads the same `latents_dir` (default `stored_latents_stage1/`).

### 4. Inference

```bash
# Edit configs/inference.yaml: checkpoint_path, input_dir, output_dir
python inference.py --config configs/inference.yaml
```

Inference runs the **denoiser only** — no SSL extractor is loaded — so any trained
checkpoint (Stage 1 or Stage 2) works.

---

## 📁 Directory structure

```
CleanUNet2-SSL_Embeddings/
│
├── train.py                          # Training entry point (JSON/YAML config, --stage)
├── generate_latents.py               # Stage-1 latents from a trained ckpt (no retraining)
├── inference.py                      # Sliding-window denoising inference
├── losses.py                         # Multi-resolution STFT, phase, magnitude losses
├── metrics.py                        # PESQ, STOI, SI-SDR
├── spec_dataset.py                   # Audio-pair dataset (+ spectrograms)
├── augmentation.py                   # On-the-fly audio augmentation
├── extract_wav2vec2_embeddings.py    # (legacy) wav2vec2-only pre-extraction
│
├── cleanunet/
│   ├── cleanunet.py                  # Waveform UNet
│   ├── cleanspecnet.py               # Spectrogram network
│   ├── cleanunet2.py                 # Hybrid base model
│   ├── cleanunet2_with_ssl_embeddings.py   # Family-agnostic SSL-conditioned model
│   ├── ssl_extractor_factory.py      # ssl_type -> extractor dispatch + config mapping
│   ├── {wav2vec2,hubert,wavlm,w2vbert,whisper}_extractor.py
│   ├── ssl_cache.py                  # On-disk embedding cache (SSLEmbeddingCache)
│   ├── integration_block.py          # SSL ↔ latent fusion block
│   ├── util.py / logger.py
│
├── lightning_modules/
│   ├── cleanunet_ssl_embeddings_stage1_module.py
│   ├── cleanunet_ssl_embeddings_stage2_module.py
│   ├── cleanunet_module.py           # Vanilla denoiser module (inference)
│   └── data_module.py
│
├── configs/
│   ├── config_<family>_plus.json     # '+'  variant (3 selected layers)
│   ├── config_<family>_plusplus.json # '++' variant (all layers)
│   ├── config.py                     # Optional strict dataclass schema/validator
│   └── inference.yaml
│
├── scripts/                          # Pre-extraction / launch helpers
└── filelists/                        # train.csv / test.csv (clean|noisy pairs)
```

`<family>` ∈ `{wav2vec2, hubert, wavlm, w2vbert, whisper}` → 10 configs total.

---

## 📋 Configuration

Each config is a single JSON file with shared sections plus `stage1` / `stage2` override
blocks. The SSL-specific block:

```jsonc
"model": {
  "cleanunet": { ... },              // shared denoiser base (identical across families)
  "ssl": {
    "type": "wavlm",                 // wav2vec2 | hubert | wavlm | w2v-bert | whisper
    "model_name": "microsoft/wavlm-large",
    "embedding_dim": 1024,
    "use_weighted_layers": true,     // learnable softmax over layers
    "selected_layers": "all",        // [i,j,k] -> '+'  |  "all" -> '++'
    "layer_strategy": "all_layers",  // "selected" ('+') | "all_layers" ('++')
    "use_preextracted": false        // on-the-fly extraction (default)
  }
}
```

Common knobs:

```jsonc
"data":    { "data_dir": "/path/to/VoiceBank-DEMAND-16k/", "batch_size": 16, "segment_size": 32000 },
"trainer": { "max_epochs": 300, "precision": "16-mixed", "gradient_clip_val": 5.0 },
"optimizer": { "type": "adamw", "learning_rate": 5e-05 }
```

`train.py` parses configs with `yaml.safe_load` (JSON ⊂ YAML), so both `.json` and `.yaml`
configs work.

---

## 📊 Logging

Both **TensorBoard and Wandb** are enabled simultaneously via `logger.choice: "both"`.
Each config carries `logger.tensorboard` and `logger.wandb` blocks (with per-stage run
names). Set `choice` to `"tensorboard"` or `"wandb"` to use only one.

- TensorBoard: `./experiments/<family>_<variant>/<stage>/`
- Wandb project: `CleanUNet2-SSL`

Tracked metrics: **PESQ**, **STOI**, **SI-SDR**.

---

## 📊 Dataset format

`filelists/train.csv` / `filelists/test.csv`, one `clean|noisy` pair per line
(`,` separator also accepted):

```
clean/p226_001.wav|noisy/p226_001.wav
clean/p226_002.wav|noisy/p226_002.wav
```

Paths are relative to `data.data_dir` (absolute paths starting with `/` are used as-is).
Primary dataset: **VoiceBank-DEMAND** (16 kHz).

---

## 🧠 Architecture

### Stage 1 — SSL-conditioned denoising

```
Input: (Noisy waveform, Noisy spectrogram, SSL embeddings)
  → CleanSpecNet        : refine spectrogram (self-attention)
  → SpecUpsampler       : expand spectrogram to waveform length
  → Conditioner (FiLM)  : fuse spectrogram + SSL features into the waveform path
  → CleanUNet           : multi-scale encoder-decoder w/ transformer bottleneck
Output: Enhanced waveform   (Stage-1 latents are saved for Stage 2)
```

### Stage 2 — latent replication (no SSL at inference)

```
Input: (Noisy waveform, Noisy spectrogram)     # no SSL embeddings
  → latent predictor trained to reproduce Stage-1 latents
Output: Enhanced waveform
```

---

## 🚨 Troubleshooting

- **CUDA out of memory** — lower `data.batch_size` or `data.segment_size`.
- **`float`/`Half` dtype mismatch in an extractor** — the extractors derive device/dtype
  from `next(self.model.parameters())`; ensure you're on a version that includes that fix
  (needed under `precision: "16-mixed"`).
- **Stage 2 can't find a checkpoint** — set `pipeline.stage1_checkpoint` in the config's
  `stage2` block to a real Stage-1 checkpoint.
- **`ssl_type` unknown** — must be one of `wav2vec2 | hubert | wavlm | w2v-bert | whisper`
  (see `cleanunet/ssl_extractor_factory.py`).

---

## 📄 License

MIT — see LICENSE.
