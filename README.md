# CleanUNet2 (+ Mamba) — Hybrid Speech Denoising on Waveform and Spectrogram

CleanUNet2 combines a **waveform U-Net (CleanUNet)** with a **spectrogram denoiser (CleanSpecNet)**.
This branch adds **Mamba state-space models** to both branches and brings in the improvements of
[CleanUMamba](https://github.com/lab-emi/cleanumamba) (ISCAS 2025) and
[CleanUNet_MambaSSM](https://github.com/Matheuspp/CleanUNet_MambaSSM):

* **Configurable sequence backbones**: Transformer (original), **Mamba**, Mamba2 or LSTM in the CleanUNet
  bottleneck; Transformer, Mamba (causal or bidirectional), Mamba2 or LSTM in CleanSpecNet.
* **CleanUMamba-compatible bottleneck**: the Mamba variant uses the same layout and parameter names as
  CleanUMamba, so the official DNS-trained checkpoints (`CleanUMamba-3N-E8/E6`) load directly into the
  waveform branch (verified to be numerically identical to the original implementation).
* **Two Mamba backends**: `mamba_ssm` CUDA kernels, or a pure-PyTorch implementation with the same
  parameters (CPU inference, tests).
* **Constant-memory streaming** for the waveform branch (ported from CleanUMamba; 48 ms lookahead for E8,
  12 ms for E6), exact w.r.t. offline inference.
* **Fixed hybrid pipeline**: one STFT inside the model (training = inference = loss), frame-aligned
  upsampling without time stretching, a single std normalization, zero-initialized conditioning,
  optional *bottleneck fusion* of CleanSpecNet features into the CleanUNet bottleneck.
* **Training recipe**: AdamW (no decay on biases/norms/`A_log`/`D`), linear warmup + cosine decay, gradient
  clipping, bf16 with float32 losses, Demucs-style augmentation (Remix, BandMask, Shift, Gain), staged training.
* **Proper evaluation**: validation on held-out speakers with whole utterances; full test metrics
  (PESQ-WB/NB, STOI, SI-SDR, CSIG, CBAK, COVL, SSNR).

---

## Repository structure

```
cleanunet/
  cleanunet.py        CleanUNet; bottleneck = transformer | mamba | mamba2 | lstm (+ CleanUMamba options)
  cleanspecnet.py     CleanSpecNet; backbone = transformer | mamba | mamba2 | lstm; causal or not
  cleanunet2.py       hybrid model: STFT front-end, upsampler, conditioner, bottleneck fusion
  sequence.py         Mamba blocks/stacks (mamba_ssm layout), BiMamba, LSTM, initialization
  mamba_torch.py      pure-PyTorch Mamba (same parameter names as mamba_ssm.Mamba)
  spectral.py         the single STFT used for training, inference and the spectral loss
  streaming.py        frame-by-frame streaming inference (Mamba bottleneck)
  checkpoint_io.py    loaders for Lightning / CleanUNet2 / CleanUMamba checkpoints
lightning_modules/    Lightning module (losses, optimizer, validation metrics) and data module
configs/
  base.yaml           shared training settings
  train*.yaml         experiments / training stages (inherit from base.yaml)
  model/*.yaml        architecture presets
  inference.yaml
evaluation/           metrics (PESQ, STOI, SI-SDR, composite measures)
scripts/make_filelists.py
dataset.py  augment.py  losses.py  config_utils.py
train.py  evaluate.py  inference.py
tests/                pytest suite
```

## Installation

```bash
pip install -r requirements.txt
# Mamba CUDA kernels (optional but recommended): install the prebuilt wheels that match your
# torch / CUDA / Python versions with --no-deps (otherwise pip may replace torch). See requirements.txt.
```

The tested environment is the conda env `cleanunet-mamba`: Python 3.12, torch 2.8.0+cu128,
pytorch-lightning 2.6, mamba-ssm 2.3.2.post1, causal-conv1d 1.6.2.post1, RTX 4090.

## Data (VoiceBank-DEMAND 16 kHz)

```bash
# layout: <data_dir>/{train,test}/{clean,noisy}/<id>.flac
python scripts/make_filelists.py --data_dir /home/fred/DATASETS/VoiceBank-DEMAND-16k
```

This writes `filelists/train.csv` (10,802 pairs), `filelists/val.csv` (770 pairs: held-out speakers p226 and
p287) and `filelists/test.csv` (824 pairs). The test set is never used for model selection.

## Model presets (`configs/model/`)

| Preset | Description | Params |
|---|---|---|
| `cleanunet2_transformer.yaml` | E0: original CleanUNet2 architecture (Transformer + Transformer) with the pipeline fixes | 58.6M |
| `cleanunet_mamba_e8.yaml` | waveform-only CleanUNet with a 3-layer Mamba bottleneck (= CleanUMamba-3N-E8), 48 ms lookahead | 41.4M |
| `cleanunet_mamba_e6.yaml` | same with 6 levels (= CleanUMamba-3N-E6), 12 ms lookahead | 27.2M |
| `cleanunet2_mamba_e8.yaml` | **CleanUNet2-Mamba**: Mamba bottleneck + causal Mamba CleanSpecNet + zero-init FiLM | 52.6M |
| `cleanunet2_mamba_e8_fusion.yaml` | + bottleneck fusion (STFT hop 256 = U-Net stride 2^8) | 53.1M |
| `cleanunet2_mamba_e8_bidir.yaml` | offline variant: non-causal, bidirectional Mamba CleanSpecNet | |
| `cleanunet_mha_e8.yaml`, `cleanunet_lstm_e8.yaml`, `cleanunet_mamba2_e8.yaml` | bottleneck ablations | |
| `cleanunet_mamba_light.yaml` | CleanUNet_MambaSSM-style small model (H=32, k=5, BatchNorm, 1 Mamba layer) | |

## Training

All configs inherit from `configs/base.yaml`; any key can be overridden on the command line with dotted keys.

```bash
# One-run recipe: hybrid CleanUNet2-Mamba, waveform branch initialized from CleanUMamba-E8 (DNS)
python train.py --config configs/train.yaml

# Staged recipe
python train.py --config configs/train_stageA_cleanunet_mamba.yaml      # A: waveform branch (from CleanUMamba)
python train.py --config configs/train_stageB_cleanspecnet_mamba.yaml   # B: CleanSpecNet alone (spectral L1)
python train.py --config configs/train_stageC1_glue.yaml                # C1: branches frozen, train the glue
python train.py --config configs/train_stageC2_joint.yaml               # C2: joint fine-tuning

# Corrected baseline (E0) and overrides
python train.py --config configs/train_baseline_transformer.yaml
python train.py --config configs/train_stageA_cleanunet_mamba.yaml model.init={} optim.lr=1e-4   # A1: from scratch
```

Checkpoints are selected by validation PESQ-WB (`val/pesq`). Logs go to TensorBoard (`logs/`) or W&B
(`logger.choice`).

Loss modes (`loss.mode`): `waveform` (L1 + multi-resolution STFT), `spec` (L1 on the log1p magnitude),
`hybrid` (both; optional anti-wrapping phase loss with `loss.weight_phase`).

## Evaluation and inference

```bash
# Noisy baseline / a trained checkpoint / the official CleanUMamba checkpoint (zero-shot, offline or streaming)
python evaluate.py --noisy
python evaluate.py --config configs/train.yaml --checkpoint logs/cleanunet2_mamba/checkpoints/last.ckpt --out_dir results/run
python evaluate.py --architecture configs/model/cleanunet_mamba_e8.yaml \
    --cleanumamba ../cleanumamba/checkpoints/models/CleanUMamba-3N-E8-full.pkl [--streaming]

# Denoise a folder
python inference.py --config configs/inference.yaml
```

`evaluate.py` writes per-file metrics (`metrics.csv`) and averages (`summary.json`), plus the real-time factor.

Note on composite metrics: CSIG/CBAK/COVL use the Python implementation of CMGAN/CleanUMamba. On the noisy
VoiceBank-DEMAND test set it gives PESQ-WB 1.97, STOI 0.921 and SSNR 1.68 (matching the literature), but
CSIG/CBAK/COVL of 3.49/2.55/2.74 instead of the MATLAB values 3.35/2.44/2.63 reported in papers. Compare
composite metrics only within the same implementation.

## Tests

```bash
python -m pytest -q tests/
```

The suite covers shapes and alignment, causality, legacy checkpoint compatibility, CUDA vs PyTorch Mamba
parity, numerical equivalence with the original CleanUMamba code, streaming vs offline equivalence,
checkpoint round-trips, augmentation, the data pipeline and the Lightning module.

## Results

See `RESULTS.md`.

## License and attribution

MIT. Ported code keeps its attribution: CleanUMamba (MIT), CleanUNet (NVIDIA, MIT), CMGAN composite
metrics (MIT), CleanUNet_MambaSSM (MIT); `mamba_ssm` is Apache-2.0.
