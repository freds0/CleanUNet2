# 🎙️ CleanUNet2 + WavLM — Configurable SSL Fusion (WavLM++_dev)

**Two-Stage Self-Supervised Speech Enhancement** built on the CleanUNet2 hybrid
architecture, with **microsoft/wavlm-large** self-supervised embeddings injected into
the denoiser — and **three interchangeable fusion strategies** selectable from the
config (`model.fusion_type`).

Based on: *"Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture
Using Self-Supervised Speech Embeddings."*

---

## Core architecture

CleanUNet2 combines two branches plus a conditioner:

* **CleanUNet** — waveform-domain encoder/decoder U-Net with a Transformer bottleneck.
* **CleanSpecNet** — spectrogram-domain denoiser (conv + self-attention).
* **SpecUpsampler + Conditioner** — fuse the refined spectrogram into the waveform path
  (`conditioning_type`: `addition` | `concatenation` | `film`).

On top of this, **WavLM embeddings are fused into the CleanUNet bottleneck latent**.

| Tensor | Shape | Notes |
|---|---|---|
| WavLM hidden states | `[B, 25, T_wavlm, 1024]` | 24 transformer layers + embedding output, 1024-D, ~50 Hz |
| WavLM sequence (weighted) | `[B, T_wavlm, 1024]` | learnable **softmax over all 25 layers** (SUPERB-style) |
| CleanUNet bottleneck latent | `[B, 768, T_unet]` | `C_unet = 768`, `T_unet ≈ L/256` |
| Encoder layer 1 / layer 2 | `[B, 64, ·] / [B, 128, ·]` | hierarchical acoustic-injection targets |

WavLM is **frozen**; only the per-layer softmax weights, the fusion block, and the
CleanUNet/CleanSpecNet weights are trained.

---

## Two-stage training

1. **Stage 1** — train *with* WavLM. The chosen fusion block injects the SSL information
   into the bottleneck; the resulting fused latent is saved as a distillation target.
2. **Stage 2** — drop the WavLM extractor. A `latent_predictor`
   (`Conv1d → PReLU → Conv1d`) learns to reproduce the Stage-1 bottleneck latent from the
   noisy audio alone → **fast inference with no WavLM dependency**.

---

## 🔀 Fusion options (the three architectures)

Select with `model.fusion_type` in the config. All three are implemented in
[`cleanunet/integration_block.py`](cleanunet/integration_block.py) and routed in
[`cleanunet/cleanunet2_with_ssl_embeddings.py`](cleanunet/cleanunet2_with_ssl_embeddings.py).

| `fusion_type` | Block | Temporal granularity | Extra loss | Pre-extract cache | Configs |
|---|---|---|---|---|---|
| `cross_attention_film` | `FiLMCrossAttentionBlock` | **per-frame** | — | **yes (enabled)** | `stage{1,2}_cross_attention_film.yaml` |
| `cvae_bottleneck` | `VariationalLatentBlock` | global (utterance) | **KL** | **yes (enabled)** | `stage{1,2}_cvae_bottleneck.yaml` |
| `hierarchical_multiscale` | `HierarchicalMultiScaleBlock` | early-global + bottleneck-per-frame | — | **no** (needs live all-layers) | `stage{1,2}_hierarchical_multiscale.yaml` |
| `legacy_pooling` | `SequenceIntegrationBlock` | global (utterance) | — | optional | (original behaviour) |

> The shipped configs set `use_preextracted_embeddings: true` for Options 1 & 2 (reuse the
> WavLM cache → faster epochs), so **extraction must run before Stage-1** for them.
> Option 3 stays on-the-fly (`false`) because it needs the live per-layer extractor.

---

### Option 1 — `cross_attention_film` (Dynamic Modulation Pipeline)

**Idea.** Every CleanUNet bottleneck frame *queries* the full WavLM sequence, then the
attended context modulates the latent through FiLM. No temporal pooling → the WavLM
temporal structure is preserved and the time alignment is learned by the attention.

**Flow** (`FiLMCrossAttentionBlock`):
```
latent [B, 768, T_unet] ──transpose──▶ Q [B, T_unet, 768]
WavLM  [B, T_wavlm, 1024] ───────────▶ K, V  (MultiheadAttention kdim=vdim=1024)
attended = MHA(Q, K, V)              ▶ [B, T_unet, 768]  (+ residual + LayerNorm)
γ = Conv1d_1x1(attended) ; β = Conv1d_1x1(attended)   ▶ [B, 768, T_unet]
fused = (1 + γ) * latent + β                          ▶ [B, 768, T_unet]
```
**Trainable:** cross-attention (Q/K/V projections) + the two FiLM 1×1 convs.
**Use when:** you want the strongest, most expressive fusion that keeps frame-level detail.
**Cost:** an extra attention over `T_wavlm` keys per step.

---

### Option 2 — `cvae_bottleneck` (Robust Variational Blueprint)

**Idea.** Treat the WavLM summary as a **conditional VAE** latent filter. WavLM features
parameterize a Gaussian `N(μ, σ²)`; a sample `z` modulates the bottleneck. A KL term
regularizes the conditioning space, which tends to make Stage-2 distillation more stable.

**Flow** (`VariationalLatentBlock`):
```
WavLM [B, T_wavlm, 1024] ──mean over time──▶ pooled [B, 1024]
μ = fc_mu(pooled) ; logvar = fc_logvar(pooled)        ▶ [B, 768]
Stage-1: z = μ + ε·exp(0.5·logvar),  ε ~ N(0, I)      ▶ [B, 768]   (reparameterization)
Stage-2: z = μ                                        ▶ deterministic
z broadcast over time → concat with latent → Conv1d_1x1 → LayerNorm → PReLU ▶ [B, 768, T_unet]
```
**Extra loss** (in [`losses.py`](losses.py) → `KLDivergenceLoss`):
`KL = -0.5 · Σ(1 + logvar − μ² − e^logvar)`, weighted by `losses.kl_weight` (default `0.001`).
The Stage-1 module adds `kl_weight · KL` only when `fusion_type == cvae_bottleneck`.
**Use when:** you want a regularized, smooth conditioning latent and a stable Stage-2 target.
**Cost:** global (utterance-level) conditioning — coarser than Option 1.

---

### Option 3 — `hierarchical_multiscale` (Multi-Scale Structural Network)

**Idea.** Different WavLM depths carry different information — lower layers are more
*acoustic*, higher layers more *semantic/phonetic*. Split the stack, process each group
with multi-scale dilated convolutions, and inject **hierarchically**: acoustic features
into the early encoder, semantic features into the bottleneck.

**Layer groups** (configurable via `acoustic_layers` / `semantic_layers`):
* **Acoustic** = mean of WavLM hidden states `1–8`
* **Semantic** = mean of WavLM hidden states `17–24`

**Multi-scale processor** (`MultiScaleDilatedConv`): 4 parallel `Conv1d(kernel=3,
dilation∈{1,2,4,8})`, concatenated and projected back (1×1) → `[B, 256, T_wavlm]`.

**Hierarchical injection** (applied **inside** `CleanUNet.encode` via `encoder_film` /
`bottleneck_film`):
```
Acoustic → MultiScaleDilatedConv → time-pool → GLOBAL FiLM (γ, β)
          ▶ encoder layer 1 [B,64] and layer 2 [B,128]   (time-broadcast)
Semantic → MultiScaleDilatedConv → PER-FRAME FiLM (γ, β [B,768,T_wavlm])
          ▶ bottleneck (linearly resampled T_wavlm → T_unet)
```
**Requires on-the-fly extraction** (`use_preextracted_embeddings: false`) because it needs
the **live** per-layer stack (`extract_all_layers`).
**Use when:** you want depth-aware conditioning that reaches both shallow and deep features.
**Note (Stage-2):** distillation replicates only the **bottleneck** latent; the early
acoustic injections are Stage-1 teacher-only.

---

## 🎚️ Training

The dataset path in the shipped configs is
`/raid/user_fredoliveira/DATASETS/VoiceBank-DEMAND-16k`, **100 epochs** per stage.

### Train all options sequentially (recommended)

```bash
bash train_all_fusion_options.sh                    # all three options
bash train_all_fusion_options.sh cvae_bottleneck    # a single option
DEVICE=cuda PYTHON=python bash train_all_fusion_options.sh
```
The script (1) pre-extracts the shared WavLM cache once, then (2) for each option runs
**Stage-1 → Stage-2**.

### Manual, per option

```bash
# build the shared all-layer WavLM cache (REQUIRED for Options 1 & 2; optional for Option 3)
python extract_wavlm_embeddings.py --config configs/stage1_cross_attention_film.yaml --device cuda

# Stage 1 (with WavLM), then Stage 2 (distillation)
python train.py --config configs/stage1_cross_attention_film.yaml --stage 1
python train.py --config configs/stage2_cross_attention_film.yaml --stage 2
```
Swap `cross_attention_film` for `cvae_bottleneck` or `hierarchical_multiscale`.

### Monitoring

```bash
tensorboard --logdir experiments/
```
Logged: total / waveform / spec / phase losses, `train/loss_kl` (CVAE only),
`val/loss_latent` (Stage-2), and PESQ / STOI / SI-SDR.

---

## ⚙️ Configuration reference

Per-option config pairs live in `configs/` (`stage1_<opt>.yaml`, `stage2_<opt>.yaml`),
each writing to an isolated `experiments/<opt>/...` tree. Key fields:

**`model`**
- `fusion_type` — `cross_attention_film` | `cvae_bottleneck` | `hierarchical_multiscale` | `legacy_pooling`
- `acoustic_layers: [1, 8]`, `semantic_layers: [17, 24]` — WavLM layer ranges (Option 3 only)
- `wavlm_model: microsoft/wavlm-large`, `wavlm_use_weighted_layers: true` (softmax over all layers)
- `use_preextracted_embeddings` — `true` for `cross_attention_film` / `cvae_bottleneck`
  (reuse the pre-extracted cache → run extraction first); `false` for
  `hierarchical_multiscale` (must extract on-the-fly via the live per-layer extractor)
- `cleanunet_params` / `cleanspecnet_params` — backbone architecture
- `conditioning_type: film` — spectrogram↔waveform conditioner (independent of the SSL fusion)

**`losses`**
- `weight_waveform: 10`, `weight_spec: 5`, `weight_phase: 5`
- `kl_weight: 0.001` — KL weight, **used only** by `cvae_bottleneck`
- `gamma_latent` (Stage-2) — weight of the latent-distillation term
- `sc_lambda`, `mag_lambda`, `stft_config` — MR-STFT settings

**`data`** — `data_dir`, `train_list_path`, `val_list_path`, `batch_size`, `segment_size: 32000`, `sampling_rate: 16000`
**`pipeline`** (Stage-2) — `stage1_checkpoint` (auto-pointed at the matching option's Stage-1 last checkpoint)

> ⚠️ The Stage-2 `fusion_type` **must match** Stage-1 so the loaded `fusion_block` weights line up.

---

## 📦 Embedding extraction

```bash
python extract_wavlm_embeddings.py --config configs/stage1_cross_attention_film.yaml --device cuda [--force]
```
Saves the **all-layer** WavLM stack per clean file to `model.wavlm_cache_dir`
(`wavlm_embeddings_raw/`), shape `(25, time, 1024)`, plus a `metadata.yaml`. The cache is
read only when `use_preextracted_embeddings: true` (Options 1/2). `hierarchical_multiscale`
always extracts on-the-fly.

---

## 🎤 Inference (denoising)

```bash
python inference.py --config configs/inference.yaml
```
Use a **Stage-2** checkpoint (no WavLM at inference). Set `model.conditioning_type` to the
value used at training time.

---

## 🧠 Branch details

**CleanUNet (waveform):** multi-scale conv encoder/decoder, Transformer bottleneck, padding
logic that keeps skip connections aligned. `encode()` accepts optional `encoder_film` /
`bottleneck_film` for hierarchical injection.
**CleanSpecNet (spectrogram):** conv feature extractor + GLU + self-attention layers.

## 🎧 Losses

| Loss | Purpose |
|---|---|
| L1/L2 waveform | Reconstruction |
| MR-STFT | Spectral convergence + log-magnitude |
| Anti-wrapping phase | Phase consistency |
| Log-magnitude spectrogram | Auxiliary stabilization |
| **KL divergence** | CVAE posterior regularization (`cvae_bottleneck` only) |
| **Latent distillation** | Stage-2 replication of the Stage-1 bottleneck latent |

## 📊 Validation metrics

PESQ, STOI, SI-SDR (per sample, averaged), plus a combined `weighted_score`.

---

## 📁 Directory layout

```
.
├── configs/
│   ├── stage1.yaml / stage2.yaml                  # templates (default fusion)
│   ├── stage1_cross_attention_film.yaml / stage2_…   # Option 1
│   ├── stage1_cvae_bottleneck.yaml / stage2_…        # Option 2
│   ├── stage1_hierarchical_multiscale.yaml / stage2_…# Option 3
│   └── inference.yaml
├── cleanunet/
│   ├── cleanunet.py            # CleanUNet (+ FiLM-injectable encode)
│   ├── cleanspecnet.py
│   ├── cleanunet2.py
│   ├── integration_block.py    # the 3 fusion blocks + MultiScaleDilatedConv
│   ├── cleanunet2_with_ssl_embeddings.py  # fusion routing
│   └── wavlm_extractor.py
├── lightning_modules/
│   ├── cleanunet_ssl_embeddings_stage1_module.py
│   └── cleanunet_ssl_embeddings_stage2_module.py
├── losses.py                   # MR-STFT, phase, KLDivergenceLoss
├── train.py / inference.py / extract_wavlm_embeddings.py
├── train_all_fusion_options.sh # extract → (stage1 → stage2) per option
├── filelists/{train,test}.csv
└── experiments/<fusion_type>/  # checkpoints, logs, stored_latents (per option)
```

---

## 📚 References

- **Paper:** Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings
- **WavLM:** Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full-Stack Speech Processing"
- **Dataset:** VoiceBank-DEMAND corpus (16 kHz)
