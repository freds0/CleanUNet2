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
- **🪜 Hierarchical multi-scale fusion** — the default SSL→denoiser fusion (see below)

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

## 🔀 Fusion strategy (`model.fusion.type`)

How the SSL features modulate the CleanUNet is selected by `model.fusion.type`. The
**default is `hierarchical_multiscale`**; a `legacy_pooling` option is kept for
backward compatibility.

| `type` | Block | Idea |
|---|---|---|
| `hierarchical_multiscale` *(default)* | `HierarchicalMultiScaleBlock` | layer-wise hierarchy + multi-scale dilated convs + FiLM injection |
| `legacy_pooling` | `SequenceIntegrationBlock` | attention-pool the SSL sequence → broadcast → concat + 1×1 conv |

### Hierarchical multi-scale (default)

The block consumes the **stacked selected SSL layers** `[B, N, T, D]`
(`extractor.extract_selected_layers`) and splits them into two groups:

- **Acoustic** (lower layers) → `MultiScaleDilatedConv` (dilations `[1,2,4,8]`) →
  **GLOBAL FiLM** modulating the **first two CleanUNet encoder layers**.
- **Semantic** (upper layers) → `MultiScaleDilatedConv` → **PER-FRAME FiLM**
  modulating the **Transformer bottleneck** (resampled to the bottleneck rate).

FiLM is applied **inside** `CleanUNet.encode` (`encoder_film` / `bottleneck_film`).

**Model-agnostic across SSL backbones.** The acoustic/semantic split is **relative to
`N`** (the number of selected layers), so the same block works for every family and for
both `+` (few layers) and `++` (all layers) — no hard-coded layer indices. Defaults:
acoustic = lower half, semantic = upper half. Override per config with explicit
inclusive index ranges into the selected stack:

```jsonc
"model": {
  "fusion": {
    "type": "hierarchical_multiscale",
    "acoustic_layers": null,   // e.g. [0, 11]  (null -> lower half, auto)
    "semantic_layers": null    // e.g. [12, 24] (null -> upper half, auto)
  }
}
```

> ⚠️ `hierarchical_multiscale` requires **on-the-fly** extraction (it needs the live
> per-layer SSL stack): keep `model.ssl.use_preextracted: false` and
> `data.use_preextracted_embeddings: false`. The shipped configs are already set this way.
> Pairing it with `++` (`selected_layers: "all"`) gives the richest hierarchy.

---

## 🚀 Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Quick test (1 epoch, 30 samples)

```bash
python train.py --config configs/config_wavlm_plusplus_stage1.json --quick-test
```

### 3. Train

Two-stage pipeline with **one config file per stage** (`config_<family>_<variant>_stage1.json`
and `_stage2.json`), each self-contained with its own checkpoint/log folders. The stage is
read from `pipeline.stage` in the config, so `--stage` is optional.

**Stage 1** trains the SSL layer-softmax + the denoiser using SSL embeddings extracted from
the **clean** reference audio (an *oracle* setup). **Stage 2** (this fork) trains the
**same** full SSL-conditioned denoiser, but extracts the SSL embeddings from the **noisy**
input audio — the audio actually available at inference. There is no distillation and no
latent predictor; both stages train from scratch (or an optional vanilla warm start).

```bash
# Stage 1 — SSL embeddings from the CLEAN reference (oracle)
python train.py --config configs/config_wavlm_plusplus_stage1.json

# Stage 2 — SSL embeddings from the NOISY input (model.ssl_audio_source="noisy")
python train.py --config configs/config_wavlm_plusplus_stage2.json
```

The only difference between the two stages is `model.ssl_audio_source` (`"clean"` for
Stage 1, `"noisy"` for Stage 2). Stage 2 no longer needs `generate_latents.py`,
`train_latents_dir`, or a Stage-1 checkpoint; each config has its own
`./experiments/<variant>/stage{1,2}/...` checkpoint and log folders.

> Both stage configs default to **on-the-fly** SSL extraction
> (`model.ssl.use_preextracted: false`), required for the learnable layer-softmax to
> receive gradients.

> **Inference caveat.** Because Stage 2 keeps the SSL extractor (now fed the noisy input),
> a Stage-2 checkpoint must run that extractor at inference. The shipped `inference.py`
> still runs the denoiser only — update it to feed the noisy SSL embeddings before using a
> Stage-2 checkpoint.

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
├── generate_latents.py               # Per-file Stage-1 TRAIN latents (Stage-2 distillation targets)
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
│   ├── config_<family>_plus_stage1.json      # '+'  variant, Stage 1
│   ├── config_<family>_plus_stage2.json      # '+'  variant, Stage 2
│   ├── config_<family>_plusplus_stage1.json  # '++' variant, Stage 1
│   ├── config_<family>_plusplus_stage2.json  # '++' variant, Stage 2
│   ├── config.py                     # Optional strict dataclass schema/validator
│   └── inference.yaml
│
├── scripts/                          # Pre-extraction / launch helpers
└── filelists/                        # train.csv / test.csv (clean|noisy pairs)
```

`<family>` ∈ `{wav2vec2, hubert, wavlm, w2vbert, whisper}` × `{plus, plusplus}` × `{stage1, stage2}`
→ 20 configs total (one per family · variant · stage).

---

## 📋 Configuration

There is one self-contained JSON config per family · variant · **stage**. Stage-1 and
Stage-2 configs share the model/loss/data sections; the Stage-2 config only differs by
`pipeline.stage: 2` and `model.ssl_audio_source: "noisy"`. Each config points at its own
`./experiments/<variant>/...` checkpoint and log folders. The SSL-specific block:

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
    "use_preextracted": false        // on-the-fly extraction (required by hierarchical)
  },
  "fusion": {
    "type": "hierarchical_multiscale", // default | "legacy_pooling"
    "acoustic_layers": null,           // [lo,hi] into selected stack | null -> lower half
    "semantic_layers": null            // [lo,hi] into selected stack | null -> upper half
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
  → Conditioner (FiLM)  : fuse spectrogram into the waveform path
  → Hierarchical fusion : split SSL layers (acoustic/semantic), multi-scale dilated convs
  → CleanUNet           : encoder-decoder w/ transformer bottleneck;
                          acoustic→GLOBAL FiLM on early encoder, semantic→PER-FRAME FiLM on bottleneck
Output: Enhanced waveform   (Stage-1 latents are saved for Stage 2)
```

### Stage 2 — SSL embeddings from the noisy input

```
Input: (Noisy waveform, Noisy spectrogram, SSL embeddings OF the noisy input)
  → same hierarchical SSL→denoiser fusion as Stage 1 (acoustic/semantic FiLM)
  → CleanUNet encoder-decoder
Output: Enhanced waveform
```

Identical to Stage 1 except the SSL extractor is fed the **noisy** input instead of the
clean reference (`model.ssl_audio_source: "noisy"`).

---

## 🚨 Troubleshooting

- **CUDA out of memory** — lower `data.batch_size` or `data.segment_size`.
- **`float`/`Half` dtype mismatch in an extractor** — the extractors derive device/dtype
  from `next(self.model.parameters())`; ensure you're on a version that includes that fix
  (needed under `precision: "16-mixed"`).
- **Stage 2 loads the SSL backbone / is as slow as Stage 1** — expected: this fork's
  Stage 2 keeps the SSL extractor (fed the noisy input). It no longer uses
  `generate_latents.py`, `train_latents_dir`, or a Stage-1 checkpoint.
- **`ssl_type` unknown** — must be one of `wav2vec2 | hubert | wavlm | w2v-bert | whisper`
  (see `cleanunet/ssl_extractor_factory.py`).

---

## 📄 License

MIT — see LICENSE.
