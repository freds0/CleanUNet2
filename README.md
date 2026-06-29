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

## 🧬 Stage-2 latent predictor (`model.latent_predictor.type`)

In **Stage 2** the SSL extractor is gone; a **latent predictor** maps the plain CleanUNet
bottleneck back to the Stage-1 *fused* latent it was distilled from. Its architecture is
selectable via `model.latent_predictor.type` (lives in `cleanunet/latent_predictors.py`).
These are **Stage-2-only** — they change neither Stage-1 nor the cached distillation
targets, so the same Stage-1 checkpoint and latent cache stay valid across all variants.

| `type` | Class | Idea | Params* |
|---|---|---|---|
| `baseline` *(default)* | `BaselinePredictor` | original 2-layer 1×1 conv, per-frame (no temporal context) | 1.18M |
| `tcn` | `TCNPredictor` | temporal receptive field: dilated residual blocks (`[1,2,4,8]`), mirroring the Stage-1 generator | 14.8M |
| `residual` | `ResidualMLPPredictor` | predicts only the delta over the latent; zero-init → starts as identity | 1.18M |
| `norm` | `NormPredictor` | per-frame MLP + `GroupNorm` for training stability (AMP) | 1.18M |
| `conformer` | `ConformerPredictor` | self-attention (global/semantic context) + depthwise conv (local/acoustic) | 5.33M |
| `film` | `FiLMPredictor` | predicts FiLM `(γ, β)` and applies `(1+γ)·latent + β`; zero-init → identity | 1.77M |
| `unet` | `UNetTemporalPredictor` | temporal U-Net (down/process/up + additive skips) — long receptive field at lower param cost than an equally wide dilated stack | 18.9M |

<sub>*Param counts at the default bottleneck `latent_dim = 768`.</sub>

```jsonc
"model": {
  "latent_predictor": {
    "type": "tcn",      // baseline | tcn | residual | norm | conformer | film | unet
    "params": null       // optional dict forwarded to the predictor constructor
  }
}
```

If the block is omitted, the predictor defaults to `baseline` (back-compat with older
configs/checkpoints). One predictor is active per config; the shipped
`configs/config_wavlm_stage2_latent_<type>.json` set one variant each (WavLM embeddings),
sharing the same Stage-1 checkpoint and distillation latents.

> ℹ️ By default only the **bottleneck** latent is distilled. With `hierarchical_multiscale`
> fusion, Stage-1 also FiLM-modulates the early encoder layers (skip connections) — Stage-2
> can optionally reproduce those too via **skip-FiLM distillation** (`model.distill_skips`,
> see [Stage-2 skip-FiLM distillation](#-stage-2-skip-film-distillation-modeldistill_skips)),
> which lifts the ceiling that bottleneck-only distillation leaves in place.

### Per-architecture details

**Shared contract.** Every predictor is an `nn.Module` mapping the bottleneck
`latent (B, C, T)` → `predicted_latent (B, C, T)` (same shape), where `C = latent_dim`
(768 by default) and `T` is the bottleneck time axis. The model's forward is always
`predicted_latent = self.latent_predictor(latent)`, so swapping architectures never
touches the rest of the pipeline. The training objective is unchanged: an MSE between
`predicted_latent` and the cached Stage-1 `fused_latent`, plus the waveform/spec/phase
reconstruction losses on the decoded audio.

**Why architecture matters here.** The target `fused_latent` was produced in Stage-1 by
FiLM modulation driven by **multi-scale dilated convs** over the SSL layers and a
**per-frame** semantic FiLM resampled along time. So the target carries *temporal* and
*global* structure. A predictor that only sees one frame at a time (the `baseline`)
cannot, in principle, recover that structure — which is what motivates the variants.

#### `baseline` — `BaselinePredictor`

- **High level.** The original Stage-2 head: a position-wise (per-frame) 2-layer MLP. It
  re-maps each time step independently, with **zero temporal context**.
- **Low level.** `Conv1d(C, C, k=1) → PReLU → Conv1d(C, C, k=1)`. Kernel size 1 ⇒
  receptive field of exactly one frame. ~1.18M params (`2·C²`).
- **Trade-off.** Cheapest and matches legacy checkpoints, but structurally blind to the
  contextual modulation in the target — the weakest fit for `hierarchical_multiscale`.

#### `tcn` — `TCNPredictor`

- **High level.** Gives the predictor a **temporal receptive field** that mirrors how the
  Stage-1 target was generated (multi-scale dilated convs), so it can infer the
  context-dependent modulation rather than guess it per frame.
- **Low level.** Four `_DilatedResidualBlock`s with dilations `[1, 2, 4, 8]`. Each block:
  `Conv1d(C, C, k=3, dilation=d) → PReLU → Conv1d(C, C, k=3, dilation=d)` wrapped in a
  residual (`x + block(x)`); padding keeps `T` fixed. A final `Conv1d(C, C, k=1)`
  projects the output. The stacked dilations reach an effective receptive field of ~60
  frames. ~14.8M params (the largest variant — most of the cost is the `k=3` convs).
- **Trade-off.** Highest capacity / receptive field, highest param & compute cost.
  Best-justified choice when fusion is `hierarchical_multiscale`.

#### `residual` — `ResidualMLPPredictor`

- **High level.** Same per-frame MLP as `baseline`, but it predicts only the **delta**
  over the input latent and is **zero-initialised** so it starts as the identity. Keeps
  the decoded audio valid from epoch 0 (Stage-1 weights are warm-started) and makes the
  optimisation easier (learn a correction, not the whole latent).
- **Low level.** `fc1 = Conv1d(C, hidden, k=1)`, `PReLU`, `fc2 = Conv1d(hidden, C, k=1)`
  with `hidden = C`; `fc2.weight`/`fc2.bias` zero-init. Forward:
  `x + fc2(PReLU(fc1(x)))`. At init `fc2(...) = 0` ⇒ output `= x` exactly. ~1.18M params.
- **Trade-off.** Cheap and stable, but still per-frame (no temporal context) — it
  improves *trainability*, not receptive field.

#### `norm` — `NormPredictor`

- **High level.** The per-frame MLP plus a normalization layer for **training stability**,
  which matters under mixed precision (`16-mixed`).
- **Low level.** `Conv1d(C, C, k=1) → GroupNorm(num_groups=8, C) → PReLU →
  Conv1d(C, C, k=1)`. GroupNorm normalises across channel groups per frame (no batch/time
  coupling), so it is batch-size and length agnostic. ~1.18M params (+ tiny GN affine).
- **Trade-off.** Cheap stabiliser; like `baseline`/`residual` it has no temporal context.

#### `conformer` — `ConformerPredictor`

- **High level.** A Conformer-style block that gives the predictor **both** global context
  (self-attention) **and** local context (depthwise conv) — mirroring the acoustic-local
  vs. semantic-global split that the Stage-1 fusion encodes.
- **Low level.** Operates on `(B, T, C)` internally (transposes in/out). Three residual
  sub-modules, each pre-normed with `LayerNorm(C)`:
  1. **Self-attention** — `MultiheadAttention(C, num_heads=8, batch_first=True)`,
     `h = h + attn(LN(h))`. Provides utterance-level (semantic) context.
  2. **Convolution** — depthwise `Conv1d(C, C, k=7, groups=C) → GroupNorm(8, C) → PReLU →
     Conv1d(C, C, k=1)`, `h = h + conv(LN(h))`. Provides local (acoustic) context.
  3. **Feed-forward** — `Linear(C, 2C) → PReLU → Linear(2C, C)`, `h = h + ff(LN(h))`.
  ~5.33M params.
- **Trade-off.** Strong modelling capacity (global + local) at moderate cost; attention is
  `O(T²)` so it is the most sensitive to long bottleneck sequences.

#### `film` — `FiLMPredictor`

- **High level.** Instead of regressing the latent directly, it predicts a **FiLM
  modulation** `(γ, β)` and applies it to the input — baking in the *same multiplicative
  structure* the Stage-1 target actually has. Zero-initialised heads ⇒ starts as identity.
- **Low level.** Shared trunk `Conv1d(C, hidden, k=1) → PReLU` (`hidden = C`); two heads
  `to_gamma`/`to_beta = Conv1d(hidden, C, k=1)`, both zero-init. Forward:
  `(1 + γ(x))·x + β(x)`. At init `γ = β = 0` ⇒ output `= x`. ~1.77M params.
- **Trade-off.** Encodes the right inductive bias cheaply, but the `(γ, β)` are predicted
  per-frame (k=1) — it isolates the FiLM idea, not temporal context.

#### `unet` — `UNetTemporalPredictor`

- **High level.** A temporal **U-Net**: it processes the latent at reduced time resolution
  (`T → T/2 → T/4`) and reconstructs it, reaching a **long receptive field** for far fewer
  params than an equally wide dilated stack. A small conv at `T/4` already spans a long
  stretch of the signal, capturing the multi-scale structure of the target.
- **Low level.** `depth=2`. Encoder: per level `_ConvBlock` (`Conv1d(C,C,k=3)+PReLU`) then
  a strided `Conv1d(C,C,k=4,stride=2)` (halves `T`). A `_ConvBlock` bottleneck. Decoder:
  per level `ConvTranspose1d(C,C,k=4,stride=2)` (doubles `T`), an **additive** skip from
  the matching encoder level, then a `_ConvBlock`; a final `Conv1d(C,C,k=1)` projects. A
  `_match_len` helper crops/pads to absorb the ±1-frame mismatch on odd `T`. ~18.9M params.
- **Trade-off.** Long receptive field at lower param/compute cost than `tcn`; the
  down/upsampling can blur fine temporal detail if `depth` is too large.

---

## 🎯 Stage-2 distillation objective (`loss.*`)

On top of the reconstruction losses, Stage-2 adds distillation terms pulling the predicted
latents toward the cached Stage-1 targets:

`total = loss_recon + γ_latent · loss_latent + γ_skip · loss_skip`

| Knob | Default | Meaning |
|---|---|---|
| `loss.gamma_latent` | `0.05` | weight of the bottleneck latent loss (`loss_latent`) |
| `loss.cosine_weight` | `0.0` | adds `cosine_weight · (1 − cos)` to `loss_latent` — penalises per-frame **direction**, not just magnitude (the decoder is direction-sensitive). `0` ⇒ MSE only |
| `loss.gamma_skip` | `0.0` | weight of the skip-FiLM loss (`loss_skip`); requires `model.distill_skips: true` |

`loss_latent` is `MSE(predicted_latent, fused_latent)` (+ the optional cosine term), keyed
per clean-audio path against the deterministic-crop targets. The two new knobs default to
`0`, so **older configs are unaffected**. The shipped `..._unet_gamma{0.05,0.2,0.5}.json`
sweep `gamma_latent` with the cosine term on (`unet` predictor).

## 🪜 Stage-2 skip-FiLM distillation (`model.distill_skips`)

With `hierarchical_multiscale` fusion, Stage-1 modulates the **first two encoder skips**
with a GLOBAL FiLM `(γ, β)` derived from the SSL acoustic layers. Standard Stage-2 only
distills the bottleneck, so the decoder receives **unmodulated** skips — an inherent
quality ceiling. Skip-FiLM distillation removes it by reproducing that modulation:

- A **`SkipFiLMHead`** per modulated layer (encoder layers 0, 1) pools the unmodulated skip
  over time and regresses its `(γ, β)`; the head is **zero-initialised** so it starts as
  the identity (warm-started decoder undisturbed at epoch 0). The predicted FiLM is applied
  to the skips in place (encode reverses the skip list, so layer *i* sits at position
  `N-1-i`) before decoding.
- `loss_skip = Σ_layer [ MSE(γ̂, γ) + MSE(β̂, β) ]` against the cached Stage-1 targets,
  weighted by `loss.gamma_skip`.

```jsonc
"model": { "distill_skips": true },
"loss":  { "gamma_skip": 1.0 }       // tune: γ,β live on a different scale than the latent MSE
```

> ⚠️ **Cache must be regenerated.** Skip targets only exist in caches written by the updated
> `generate_latents.py`, which stores a dict `{"fused_latent": (C,T), "encoder_film":
> {layer: (γ, β)}}` for hierarchical fusion (legacy_pooling still writes a bare `(C,T)`
> tensor). Old bare-tensor caches have no `encoder_film`, so `loss_skip` would silently stay
> 0 — point the skip config at a **fresh** `train_latents_dir`/`val_latents_dir` and
> regenerate. Configs without `distill_skips` keep reading the old caches unchanged (the
> Stage-2 loader accepts both formats).

The shipped `config_wavlm_stage2_latent_unet_skipfilm.json` enables this end-to-end:
`unet` predictor + cosine + `gamma_latent=0.2` + `distill_skips` + `gamma_skip`, with its
own `experiments/wavlm_skipfilm/` cache.

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

**Stage 1** trains the SSL layer-softmax + the denoiser (SSL embeddings injected into the
latent space). **Stage 2** trains a latent predictor to reproduce the Stage-1 latents
*without* the SSL extractor at inference time — a **distillation** step that requires the
Stage-1 train latents as targets.

```bash
# 1) Stage 1
python train.py --config configs/config_wavlm_plusplus_stage1.json

# 2) Generate the Stage-1 TRAIN latents (distillation targets) — REQUIRED before Stage 2
python generate_latents.py \
  --config configs/config_wavlm_plusplus_stage1.json \
  --checkpoint experiments/wavlm_plusplus/checkpoints/stage1/cleanunet-stage1-last.ckpt \
  --output-dir experiments/wavlm_plusplus/train_latents

# 3) Stage 2 (reads train_latents_dir from the config)
python train.py --config configs/config_wavlm_plusplus_stage2.json
```

The Stage-2 config carries `pipeline.stage1_checkpoint`, `train_latents_dir`, and
`data.deterministic_crop`/`data.return_audio_paths` (needed to align each sample with its
cached latent target). **Stage 2 will not start unless the train latents exist** — it
raises a clear error pointing you at `generate_latents.py`.

> Configs default to **on-the-fly** SSL extraction (`model.ssl.use_preextracted: false`),
> which is required for the learnable layer-softmax to receive gradients.

#### Why the train latents (and not the val ones)?

The Stage-2 latent-replication loss is a distillation objective: the predicted latent must
match the Stage-1 **fused** latent for the *same* training sample. `generate_latents.py`
loads a fully-trained Stage-1 checkpoint and writes one target per train file, keyed by the
clean-audio path (`<md5>.pt`), using a **deterministic per-file crop** so the cached target
matches the segment Stage-2 training sees. The checkpoint is loaded with `strict=False`, so
any extra weights it carries that the config doesn't build (e.g. an unused self-attention
pooling head) are ignored — they don't affect the fused latent.

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
│   ├── config_wavlm_stage2_latent_unet_gamma<g>.json  # unet + cosine, γ_latent sweep (0.05/0.2/0.5)
│   ├── config_wavlm_stage2_latent_unet_skipfilm.json  # unet + cosine + skip-FiLM distillation
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
Stage-2 configs share the model/loss/data sections; the Stage-2 config additionally sets
`pipeline.stage1_checkpoint`, `train_latents_dir`, and `data.deterministic_crop` /
`data.return_audio_paths`. Each config points at its own `./experiments/<variant>/...`
checkpoint and log folders. The SSL-specific block:

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

### Stage 2 — latent replication (no SSL at inference)

```
Input: (Noisy waveform, Noisy spectrogram)     # no SSL embeddings
  → latent predictor trained to reproduce Stage-1 latents
    (architecture selectable via model.latent_predictor.type — see above)
Output: Enhanced waveform
```

---

## 🚨 Troubleshooting

- **CUDA out of memory** — lower `data.batch_size` or `data.segment_size`.
- **`float`/`Half` dtype mismatch in an extractor** — the extractors derive device/dtype
  from `next(self.model.parameters())`; ensure you're on a version that includes that fix
  (needed under `precision: "16-mixed"`).
- **Stage 2 can't find a checkpoint** — set `pipeline.stage1_checkpoint` in the
  `_stage2.json` config to a real Stage-1 checkpoint.
- **Stage 2 aborts: "No train latents found"** — Stage 2 trains only if the distillation
  targets exist. Run `generate_latents.py` (see step 2 above) and make sure its
  `--output-dir` matches `train_latents_dir` in the `_stage2.json` config.
- **`ssl_type` unknown** — must be one of `wav2vec2 | hubert | wavlm | w2v-bert | whisper`
  (see `cleanunet/ssl_extractor_factory.py`).

---

## 📄 License

MIT — see LICENSE.
