# CleanUNet2 (+ Mamba): Speech Denoising on Waveform and Spectrogram

CleanUNet2 combines a **waveform U-Net (CleanUNet)** with a **spectrogram denoiser (CleanSpecNet)**.
This branch adds **Mamba state-space models** to both branches, brings in the improvements of
[CleanUMamba](https://github.com/lab-emi/cleanumamba) (ISCAS 2025) and
[CleanUNet_MambaSSM](https://github.com/Matheuspp/CleanUNet_MambaSSM), and contains the tools of a large ablation
study on VoiceBank-DEMAND (16 kHz). The study is logged in `IMPROVEMENT_PLAN.md`.

The best system is a **training-free blend** of two separately trained networks. A waveform CleanUNet with a Mamba
bottleneck (**S30**) is trained with a metric discriminator and external noise. A bidirectional Mamba CleanSpecNet
(**W5**) estimates a magnitude, and its magnitude is mixed with the waveform estimate. With a bidirectional bottleneck
trained for 60 epochs, the blend reaches **PESQ-WB 3.446** on the VoiceBank-DEMAND test set (offline). The
streaming-capable S30 alone reaches **3.335**.

---

## Contents

- [Features](#features)
- [Recommended systems](#recommended-systems)
- [Architecture of the recommended system](#architecture-of-the-recommended-system)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation and inference](#evaluation-and-inference)
- [Experiment orchestration and research notes](#experiment-orchestration-and-research-notes)
- [Tests](#tests)
- [License and attribution](#license-and-attribution)

## Features

- **Configurable sequence backbones:** Transformer (original), **Mamba**, Mamba2 or LSTM in the CleanUNet
  bottleneck. CleanSpecNet can use Transformer, Mamba (causal or bidirectional), Mamba2 or LSTM.
  Two more spectral branches are available: dual-path time-frequency Mamba and sub-band.
- **CleanUMamba-compatible bottleneck:** the Mamba variant uses CleanUMamba's layout and parameter names, so the
  official DNS-trained checkpoints (`CleanUMamba-3N-E8/E6`) load directly into the waveform branch. This is verified
  to be numerically identical to the original implementation.
- **Two Mamba backends:** the `mamba_ssm` CUDA kernels, or a pure-PyTorch implementation with the same parameters
  (for CPU inference and tests).
- **Constant-memory streaming** of the waveform branch (48 ms lookahead for E8, 12 ms for E6). The input scale is a
  running estimate, which can start from a prior so the first second is not under-scaled.
- **Hybrid pipeline fixes:**
  - one STFT inside the model, used for training, inference and the loss;
  - frame-aligned upsampling without time stretching;
  - a single standard-deviation normalization;
  - zero-initialized conditioning;
  - a generic fusion hub: named injection points in CleanUNet, a spectral post-filter, an input estimate and a
    Mamba exchange.
- **Training-free combinations:**
  - magnitude blend of a waveform estimate with a CleanSpecNet (`cleanunet/blend.py`);
  - weight soups of seeds (`scripts/make_soup.py`);
  - output ensembles (`scripts/ensemble_outputs.py`).
- **Training recipe:**
  - AdamW (no decay on biases, norms, `A_log` or `D`), linear warm-up and cosine decay, gradient clipping;
  - bf16 with float32 losses, staged training;
  - Demucs-style GPU augmentation (Remix, BandMask, Shift, Gain);
  - external noise from MUSAN mixed at P.56 SNRs, and random polarity of the training pair;
  - a **metric discriminator** (MetricGAN-style PESQ predictor), optionally restricted to frequencies above 100 Hz;
  - knowledge distillation from frozen teachers.
- **Evaluation:**
  - full metrics (PESQ-WB/NB, STOI, SI-SDR, CSIG, CBAK, COVL, SSNR, WSS, LLR);
  - band-limited PESQ (> 100 Hz) and a robust PESQ;
  - DNSMOS P.835 / P.808;
  - test-time augmentation, polarity-inversion checks and streaming evaluation;
  - validation sets with **unseen noise** (`val_unseen`, `val_unseen_low`) and **80 unseen speakers**
    (`vctk_unseen`).

## Recommended systems

The numbers below are for the batch-16 recipe. "Soup" is the weight average of 3 seeds, which gives a single model.
The test set has 824 files from 2 speakers; `vctk_unseen` has 800 files from 80 unseen VCTK speakers with unseen
DEMAND noise.

| System | Params | Streaming | Test PESQ-WB | PESQ-NB | STOI | SI-SDR (dB) | vctk_unseen PESQ-WB |
|---|---|---|---|---|---|---|---|
| Control: K0ext30p soup (no metric loss) | 41.4M | yes | 3.165 | 3.758 | 0.954 | 18.74 | 3.153 |
| **S30 soup** (simplest strong model) | 41.4M | yes | 3.335 | 3.825 | 0.954 | 18.47 | 3.325 |
| S30hp soup (metric loss only above 100 Hz) | 41.4M | yes | 3.280 | 3.825 | 0.954 | 18.68 | 3.256 |
| S30hp soup, metric weight 1.0 | 41.4M | yes | 3.316 | 3.848 | 0.954 | 18.70 | 3.294 |
| Bidirectional S30 soup, 60 epochs (one network) | 52.4M | no | 3.428 | 3.885 | 0.957 | 19.14 | 3.397 |
| Bidirectional S30hp soup, metric weight 1.0 | 52.4M | no | 3.378 | 3.897 | 0.957 | 19.21 | 3.333 |
| S30 soup + W5 blend, w = 0.25 | 60.9M | no | 3.375 | 3.862 | 0.956 | 18.85 | 3.365 |
| S30hp soup + W5 blend, w = 0.25 | 60.9M | no | 3.326 | 3.861 | 0.956 | 19.00 | 3.300 |
| **Bidirectional S30 soup, 60 epochs + W5, w = 0.15** (best PESQ-WB) | 71.9M | no | **3.446** | 3.902 | **0.958** | 19.24 | **3.414** |
| **Bidirectional S30hp soup, weight 1.0 + W5, w = 0.25** (for use) | 71.9M | no | 3.388 | **3.913** | **0.958** | **19.32** | 3.352 |

Notes:
- **Streaming.** Streaming S30 with the scale prior (`--stream_prior_std 0.0604 --stream_prior_frames 62`) gives
  3.323 on the test set, against 3.335 offline. The S30hp soup with metric weight 1.0 has not been evaluated in
  streaming yet.
- **Bidirectional bottleneck and longer training.** The bidirectional bottleneck gains PESQ-WB +0.064 and SI-SDR
  +0.63 dB over S30 on the test set (30 epochs each). Training it for 60 epochs adds PESQ-WB +0.029, partly below
  100 Hz. With 30 epochs, the bidirectional S30 soup scores 3.399 alone and 3.421 with W5.
- **Where the metric loss gains.** Part of the PESQ-WB gain of the metric discriminator sits in the 50-100 Hz band,
  which PESQ-WB still weighs. That part costs SI-SDR, and DNSMOS reads it as background noise. In the speech band,
  the gain is PESQ-NB +0.055 over the control, positive for all 80 speakers of `vctk_unseen`. S30hp keeps that gain
  without the 50-100 Hz effect.
- **Which variant.** Use the 60-epoch bidirectional S30 + W5 for the PESQ-WB benchmark number, and the bidirectional
  S30hp with metric weight 1.0 + W5 when the speech band (PESQ-NB), SI-SDR and the low band matter. For streaming,
  use S30, or the S30hp with metric weight 1.0 for PESQ-NB (`IMPROVEMENT_PLAN.md`, sections 14.8, 14.9 and 14.11).

## Architecture of the recommended system

```
 noisy x ── / std(x) ──┬───────────────────────────┬
                       │                           │
             S30: CleanUNet (waveform)      W5: CleanSpecNet (spectrogram)
             8-level U-Net + 3 Mamba layers STFT 1024/256 -> linear front end
             41.4M, causal, 48 ms lookahead -> 5 bidirectional Mamba layers -> mask
                       │                           │
              waveform y -> STFT -> |Y|, phase   magnitude |S|
                       └─────────────┬─────────────┘
                 |M| = |Y|^(1-w) * |S|^w  (w = 0.25), phase of Y
                                     │
                             iSTFT * std(x)
```

- **S30 (waveform branch).**
  - Encoder: 8 blocks of Conv1d(k=4, stride 2) -> ReLU -> 1x1 Conv -> GLU, with channels 64 to 768. The bottleneck
    runs one token every 256 samples (16 ms).
  - Bottleneck: 1x1 Conv -> 3 unidirectional Mamba layers (d_model 512, d_state 64, expand 4) -> 1x1 Conv.
  - Bidirectional variant (offline; the best system, trained for 60 epochs): each Mamba layer also runs on the
    time-reversed sequence, and the two outputs are summed (`BiMambaMixer` in `cleanunet/sequence.py`). This adds
    11.0M parameters (52.4M in total).
  - Decoder: the mirror image, with transposed convolutions and additive skip connections.
  - Training: from the DNS-pretrained CleanUMamba-3N-E8 checkpoint, a VoiceBank-DEMAND fine-tune (stage A), then
    30 epochs with L1 + multi-resolution STFT, a metric discriminator (weight 0.3), MUSAN noise (p 0.5,
    -5 to 20 dB) and random polarity (p 0.5).
- **W5 (spectral branch).**
  - Input: log1p magnitude of a Hann STFT (1024 / 256), 513 bins.
  - Network: per-frame linear projection (513 -> 512) with a depthwise temporal convolution, 5 bidirectional Mamba
    layers, and a sigmoid mask on the compressed input magnitude. It predicts no phase.
  - Training: alone, 100 epochs, spectral L1 loss, with the same MUSAN noise.
- **Blend (`cleanunet/blend.py`).** For every time-frequency bin it takes the weighted geometric mean of the two
  magnitudes, a weighted mean in dB. The phase comes from the waveform branch. Nothing is trained; `w` is chosen
  on `val_unseen`.

## Repository structure

```
cleanunet/
  cleanunet.py        CleanUNet; bottleneck = transformer | mamba | mamba2 | lstm (+ CleanUMamba options)
  cleanspecnet.py     CleanSpecNet; backbone = transformer | mamba | mamba2 | lstm; causal or bidirectional
  tf_mamba.py         dual-path time-frequency Mamba spectral branch
  subband.py          full-band + sub-band spectral branch
  cleanunet2.py       hybrid model: STFT front end, upsampler, conditioner, bottleneck fusion, fusion hub
  fusion.py           fusion hub: frame-rate condition -> heads at named CleanUNet injection points
  postfilter.py       learned spectral post-filter on the CleanUNet output
  blend.py            magnitude blend with a CleanSpecNet (training-free, or differentiable for loss.mode blend)
  metric_disc.py      metric discriminator (PESQ predictor) and its losses
  sequence.py         Mamba blocks and stacks (mamba_ssm layout), BiMamba, LSTM
  mamba_torch.py      pure-PyTorch Mamba (same parameter names as mamba_ssm.Mamba)
  spectral.py         the single STFT used for inputs, losses and resynthesis
  streaming.py        frame-by-frame streaming inference (Mamba bottleneck)
  checkpoint_io.py    loaders for Lightning / CleanUNet2 / CleanUMamba checkpoints; uni- to bidirectional Mamba
  util.py, logger.py  utilities kept from the original CleanUNet code
lightning_modules/    Lightning module (losses, metric GAN, distillation, optimizer, validation) and data module
configs/
  base.yaml           shared training settings
  train*.yaml         training stages (inherit from base.yaml)
  ablation/           settings of the ablation runs (k0.yaml = waveform-only control / S30 base)
  model/              architecture presets; model/fusion/ = fusion variants
  inference.yaml
evaluation/           metrics.py (PESQ, STOI, SI-SDR, composite, band-limited / robust PESQ), composite.py,
                      dnsmos.py (DNSMOS P.835 / P.808)
scripts/
  make_filelists.py       train / val / test file lists
  make_val_unseen.py      validation sets with unseen DEMAND noise (val_unseen, val_unseen_low, vctk_unseen)
  prepare_vctk_clean.py   clean utterances of the VCTK speakers that VoiceBank-DEMAND does not use
  make_soup.py            weight soup of checkpoints with the same architecture
  ensemble_outputs.py     average the enhanced waveforms of several systems
  diagnose_spec_branch.py stand-alone CleanSpecNet metrics and blend diagnostics
  compare_runs.py         paired per-file comparison of runs (bootstrap CI, Wilcoxon + Holm)
  robustness_report.py    tails, worst noise and polarity checks of a set of systems
  run_ablation.py         restartable experiment queue of the ablation study
  show_metrics.py         print TensorBoard scalars
dataset.py  noise_bank.py  augment.py  losses.py  config_utils.py
train.py  evaluate.py  inference.py
tests/                pytest suite
augmentation.py, metrics.py   file-level tools of the original repository (not used by training)
```

## Installation

```bash
pip install -r requirements.txt
# Mamba CUDA kernels (optional but recommended): install the prebuilt wheels that match your
# torch / CUDA / Python versions with --no-deps (otherwise pip may replace torch). See requirements.txt.
```

The tested environment is the conda env `cleanunet-mamba`: Python 3.12, torch 2.8.0+cu128,
pytorch-lightning 2.6, mamba-ssm 2.3.2.post1, causal-conv1d 1.6.2.post1, RTX 4090.

Stage A starts from the official CleanUMamba checkpoint, expected at
`../cleanumamba/checkpoints/models/CleanUMamba-3N-E8-full.pkl`.

## Data

**VoiceBank-DEMAND (16 kHz)**, with the layout `<data_dir>/{train,test}/{clean,noisy}/<id>.flac`:

```bash
python scripts/make_filelists.py --data_dir /home/fred/DATASETS/VoiceBank-DEMAND-16k
```

This writes `filelists/train.csv` (10,802 pairs) and `filelists/val.csv` (770 pairs from the held-out speakers
p226 and p287). It also writes `filelists/test.csv` (824 pairs), which is never used for model selection.

**Evaluation sets with unseen conditions.** Select models on `val_unseen`: on these data it ranks systems like the
test set (Spearman 0.98 over 52 systems), while the old `val` does not (0.66).

| Set | Speakers | Noise | Built with |
|---|---|---|---|
| `val_unseen` | p226, p287 (770 files) | DEMAND environments VoiceBank-DEMAND never uses (washing, field, park, river, hallway) at the test SNRs | `scripts/make_val_unseen.py` |
| `val_unseen_low` | same | same environments at -5 / 0 dB | `scripts/make_val_unseen.py --name val_unseen_low --snrs -5 0 --seed 1 --rescale` |
| `vctk_unseen` | 80 VCTK speakers outside VoiceBank-DEMAND (800 files) | same as `val_unseen` | `python -I scripts/prepare_vctk_clean.py`, then `scripts/make_val_unseen.py --filelist filelists/vctk_unseen_clean.csv --name vctk_unseen --seed 2` |

**External sources:**
- DEMAND (Zenodo 1227121, CC BY 4.0), for the unseen-noise sets;
- MUSAN (CC BY 4.0), for the external training noise (`data.ext_noise`);
- VCTK 0.92 (Edinburgh DataShare 10283/3443, CC BY 4.0), for `vctk_unseen`.

Scripts that read downloaded data are meant to run with `python -I`.

## Training

All configs inherit from `configs/base.yaml`. Any key can be overridden on the command line with dotted keys.
Training resumes from `resume_from_checkpoint` when that file exists.

### Recipe of the recommended system

```bash
# 1. Stage A: waveform branch from the DNS-pretrained CleanUMamba-3N-E8 (20 epochs, best val PESQ)
python train.py --config configs/train_stageA_cleanunet_mamba.yaml trainer.max_epochs=20

# 2. S30: 30 epochs from stage A with MUSAN noise, random polarity and the metric discriminator (seeds 1, 2, 3)
python train.py --config configs/ablation/k0.yaml seed=1 trainer.max_epochs=30 \
    'data.ext_noise={dir: /path/to/musan, p: 0.5, snr_db: [-5, 20]}' data.num_workers=16 \
    augment_polarity=0.5 'loss.metric_gan={weight: 0.3}' \
    model.init.cleanunet_checkpoint=<stage A checkpoint> callbacks.best_checkpoint.dirpath=logs/S30_s1
#    S30hp: 'loss.metric_gan={weight: 0.3, highpass_hz: 100}'
#    S30hp with metric weight 1.0: 'loss.metric_gan={weight: 1.0, highpass_hz: 100}'
#    Bidirectional bottleneck (offline): model.architecture.cleanunet_params.bottleneck.bidirectional=true
#    (the stage A weights load into both directions, with the backward output projections zeroed); the best
#    system trains it with trainer.max_epochs=60

# 3. Weight soup of the seeds
python scripts/make_soup.py --ckpts logs/S30_s{1,2,3}/last.ckpt --out logs/S30_soup/last.ckpt

# 4. W5: bidirectional Mamba CleanSpecNet trained alone (100 epochs, spectral L1)
python train.py --config configs/ablation/stageB_plus.yaml trainer.max_epochs=100 \
    model.architecture.cleanspecnet_params.output_mode=mask model.architecture.cleanspecnet_params.frontend=linear \
    model.architecture.cleanspecnet_params.causal=false model.architecture.cleanspecnet_params.backbone.bidirectional=true \
    'data.ext_noise={dir: /path/to/musan, p: 0.5, snr_db: [-5, 20]}' data.num_workers=16 \
    callbacks.best_checkpoint.dirpath=logs/W5

# 5. Blend (no training): pick w on val_unseen, then evaluate (w = 0.15 for the bidirectional S30 soups)
python evaluate.py --checkpoint logs/S30_soup/last.ckpt --blend_spec logs/W5/last.ckpt --blend_w 0.25 \
    --filelist filelists/val_unseen.csv --out_dir results/S30_W5/val_unseen
```

`configs/ablation/k0.yaml` points `model.init.cleanunet_checkpoint` to the stage A checkpoint of the study; override
it with your own.

Keep the batch size of the configs (16). A rerun of this recipe with the largest batch that fits a 24 GB GPU (stage A
and S30 at 128, W5 at 1024, square-root learning-rate scaling, same epochs) lost 0.063 test PESQ-WB, because S30 and
W5 get 8-64x fewer optimizer steps; only stage A was unaffected (`IMPROVEMENT_PLAN.md`, section 14.10).

### Other recipes

```bash
# One-run hybrid CleanUNet2-Mamba, waveform branch initialized from CleanUMamba-E8 (DNS)
python train.py --config configs/train.yaml

# Staged hybrid recipe
python train.py --config configs/train_stageA_cleanunet_mamba.yaml      # A: waveform branch (from CleanUMamba)
python train.py --config configs/train_stageB_cleanspecnet_mamba.yaml   # B: CleanSpecNet alone (spectral L1)
python train.py --config configs/train_stageC1_glue.yaml                # C1: branches frozen, train the glue
python train.py --config configs/train_stageC2_joint.yaml               # C2: joint fine-tuning

# Corrected baseline (E0) and training from scratch
python train.py --config configs/train_baseline_transformer.yaml
python train.py --config configs/train_stageA_cleanunet_mamba.yaml model.init={} optim.lr=1e-4
```

On the unseen-noise sets the learned fusions of the hybrid did not beat the training-free blend
(`IMPROVEMENT_PLAN.md`, section 12.9), which is why the recommended system keeps the two branches separate.

### Main options

- `loss.mode`:
  - `waveform`: L1 + multi-resolution STFT;
  - `spec`: L1 on the log1p magnitude;
  - `hybrid`: both, with an optional anti-wrapping phase loss (`loss.weight_phase`);
  - `blend`: the output is the blend of a frozen waveform branch with the spectral branch (`loss.blend_w`), so the
    waveform losses train the spectral branch for its role in the blend (`configs/ablation/z5_blend.yaml`). It did
    not beat the separately trained W5 (`IMPROVEMENT_PLAN.md`, section 14.11).
- `loss.metric_gan`:
  - `{weight, d_lr, warmup_steps, pesq_workers, highpass_hz}`: the metric discriminator;
  - training then uses manual optimization, and PESQ targets are computed in a process pool.
- `loss.cmag`: compressed-magnitude MSE.
- `data.ext_noise`: `{dir, p, snr_db}`, external noise that replaces the dataset noise with probability `p`.
- `augment_polarity`: probability of multiplying a training pair by -1. Without it, models trained with MUSAN
  babble fail on polarity-inverted input.
- `distill`: `{weight, teacher_checkpoint}` or `{weight, teachers: [...]}`, distillation from frozen teachers.
  A teacher can be a blend.
- `fusion_dropout`, `fusion_curriculum`: training aids for the learned fusions.

Checkpoints are selected by validation PESQ-WB (`val/pesq`) when validation is enabled. Logs go to TensorBoard
(`logs/`) or W&B (`logger.choice`).

## Evaluation and inference

```bash
# Noisy baseline / a trained checkpoint / the official CleanUMamba checkpoint (zero-shot, offline or streaming)
python evaluate.py --noisy --filelist filelists/test.csv
python evaluate.py --checkpoint logs/S30_soup/last.ckpt --filelist filelists/test.csv --out_dir results/S30/test
python evaluate.py --architecture configs/model/cleanunet_mamba_e8.yaml \
    --cleanumamba ../cleanumamba/checkpoints/models/CleanUMamba-3N-E8-full.pkl [--streaming]

# Streaming with the input-scale prior (medians of the val split)
python evaluate.py --checkpoint logs/S30_soup/last.ckpt --streaming --stream_prior_std 0.0604 --stream_prior_frames 62

# DNSMOS (ONNX models downloaded from the DNS-Challenge repository; run isolated)
python -I evaluation/dnsmos.py --models /path/to/dnsmos_models --audio_dir <enhanced dir> \
    --names filelists/test.csv --out results/dnsmos/test.csv

# Denoise a folder with one model (offline, or streaming without the scale prior)
python inference.py --config configs/inference.yaml
```

`evaluate.py` writes per-file metrics (`metrics.csv`) and averages (`summary.json`, with PESQ-WB per input-SNR bin
and the real-time factor). Useful options:

| Option | Effect |
|---|---|
| `--fast` | only PESQ-WB, STOI and SI-SDR |
| `--band_limited` | adds `pesq_wb_hp`: PESQ-WB with reference and estimate high-passed at 100 Hz |
| `--robust_pesq` | adds `pesq_wb_r`, a PESQ that is less sensitive to the instability of single files |
| `--blend_spec CKPT --blend_w W` | training-free magnitude blend with a CleanSpecNet |
| `--streaming [--stream_prior_std S --stream_prior_frames N]` | frame-by-frame inference of a waveform-only Mamba model |
| `--tta flip/shift/flipshift`, `--invert` | test-time augmentation; polarity-inverted input (robustness check) |
| `--condition shuffle/off` | for hybrids: spectral condition from another file, or none (dependency check) |
| `--save_audio` | write the enhanced files (FLAC) |

**Composite metrics.** CSIG/CBAK/COVL use the Python implementation of CMGAN / CleanUMamba. On the noisy
VoiceBank-DEMAND test set it gives PESQ-WB 1.97, STOI 0.921 and SSNR 1.68, matching the literature. Its
CSIG/CBAK/COVL are 3.49/2.55/2.74, though, against the MATLAB values 3.35/2.44/2.63 reported in papers. Compare
composite metrics only within the same implementation. They are linear in PESQ, so they are not independent
evidence of a PESQ gain.

## Experiment orchestration and research notes

- `scripts/run_ablation.py` runs the phases of the ablation study as a restartable queue. Steps with existing
  results are skipped, interrupted trainings resume, and each step waits for free GPU memory. Its docstring lists
  the phases.
- `IMPROVEMENT_PLAN.md` is the research log: protocol, every phase with its results and analyses, and the current
  recommendations (section 0).
- `IMPROVEMENTS_CLAUDE.md` holds the initial architecture proposals.
- Run outputs go to `logs/` (checkpoints, TensorBoard) and `results/` (metrics); both are ignored by git.

## Tests

```bash
python -m pytest -q tests/
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/     # CPU only (e.g. while the GPU is training)
```

The suite covers:
- shapes, alignment and causality;
- legacy checkpoint compatibility, numerical equivalence with the original CleanUMamba code, and CUDA vs PyTorch
  Mamba parity;
- streaming vs offline equivalence and checkpoint round-trips;
- augmentation, external noise, the data pipeline and the Lightning module;
- fusion variants, the metric discriminator, distillation, the blend and its training mode, soups, the scale prior
  and the band-limited metrics;
- unidirectional checkpoints loaded into a bidirectional bottleneck.

Tests that need the CleanUMamba checkpoint, the dataset or CUDA are skipped when those are missing. With the GPU
hidden (`CUDA_VISIBLE_DEVICES=`), `test_no_missing_or_unexpected_keys_for_e8` fails: the official CleanUMamba
checkpoint stores CUDA tensors.

## License and attribution

MIT. Ported code keeps its attribution: CleanUMamba (MIT), CleanUNet (NVIDIA, MIT), CMGAN composite metrics (MIT)
and CleanUNet_MambaSSM (MIT). `mamba_ssm` is Apache-2.0. The DNSMOS models belong to the Microsoft DNS-Challenge
repository and are not included. The datasets keep their own licenses: VoiceBank-DEMAND, VCTK, DEMAND and MUSAN
(CC BY 4.0).
