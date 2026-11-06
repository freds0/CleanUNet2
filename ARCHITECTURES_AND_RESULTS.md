# Architectures and experimental results

State on 2026-11-06, 11:30. This document covers the architectures evaluated in this repository and the results of the
experiments. The full log, with every queue and decision, is in `IMPROVEMENT_PLAN.md` (the sections are cited below).
Raw results are in `results/all/` and `results/screen/` (`summary.json` and per-file `metrics.csv`).

## 1. Summary

- **Best system across domains, and on the VoiceBank test set:** TF2, the time-frequency branch v2 (`tf_l` size,
  2.48M parameters) trained on all the data for 300,000 steps (TF1) and then with the metric discriminator. Test
  PESQ-WB 3.478, SI-SDR 18.65 dB (5.8).
  - Against Z4 + W5 (3.446), it is ahead on `val_unseen` and test, 0.079 behind on `vctk_unseen`, and ahead on all
    three VoiceBank sets in PESQ-WB above 100 Hz and in PESQ-NB. Z4 + W5 keeps SI-SDR (+0.59 dB on test).
  - Off VoiceBank it beats every other system, except its own blend with M2c on three of the four sets (by 0.010
    to 0.028). TF2: DNS no_reverb 3.484, with_reverb 3.359, EARS-WHAM 2.694, Libri2Mix 2.711.
  - The waveform branch M2c adds almost nothing to it (+0.002 on the dev set).
- **Best system trained on VoiceBank only:** Z4 + W5. Test PESQ-WB 3.446, SI-SDR 19.24 dB. It fails off VoiceBank:
  1.189 PESQ-WB and −3.31 dB on DNS with_reverb.
- **Earlier all-data systems:** M2c + `tf_l` (screening budget, test 3.216) and M2c + MS (test 3.048). Both are
  superseded by TF1 and TF2.
- **Waveform-branch capacity:** none of the four variants passed its criterion. Dilated convolutions (dconv) are the
  most consistent.
- **High-band phase defect:** the from-scratch all-data models apply a fixed phase rotation above 5 kHz. A
  compressed complex loss removes it (M2c).

## 2. How systems are evaluated

### 2.1 Evaluation sets

All are at 16 kHz and scored with `evaluate.py --band_limited`.

| Set | Files | Content |
|---|---|---|
| `val_unseen` | 770 | VoiceBank speakers p226 / p287, DEMAND noises not used in training |
| `test` | 824 | VoiceBank-DEMAND official test set |
| `vctk_unseen` | 800 | 80 unseen VCTK speakers |
| `dns2020_no_reverb` | 150 | DNS Challenge 2020 synthetic |
| `dns2020_with_reverb` | 150 | The same, with a reverberant reference |
| `ears_wham_test` | 886 | EARS-WHAM v2 16 kHz test |
| `libri2mix_test` | 3,000 | Libri2Mix mix_single test |

### 2.2 Multi-domain dev set

`filelists/dev_multi.csv` has 900 files, 300 from each domain below. No training list uses them
(`scripts/make_dev_multi.py`). Selection uses the mean of the per-domain means, which is what `val/pesq` logs during
training.

| Domain | Source |
|---|---|
| VB | `val_unseen` |
| EARS-WHAM | EARS-WHAM v2 valid (speakers p100–p101, WHAM! cv noise) |
| Libri2Mix | Libri2Mix min dev (LibriSpeech dev-clean readers, WHAM! cv noise) |

The plan also listed URGENT 2026 validation and additive-noise rows. They were not included, so the dev set has three
domains, not the four or five of the plan (16.4, item 0.1).

### 2.3 Metrics

PESQ-WB is the main metric. Also reported: PESQ-NB, STOI, SI-SDR (dB), and the composite CSIG, CBAK and COVL. Higher
is better for all of them. Confidence intervals in this document are 95% paired bootstrap intervals over files.

## 3. Architectures

All offline models see the whole utterance. Parameter counts are in millions (M).

### 3.1 Waveform branch: CleanUNet

`cleanunet/cleanunet.py`. A U-Net on the raw waveform, with eight levels (E8) and 64–768 channels:

- **Encoder level:** Conv1d (kernel 4, stride 2) → ReLU → 1×1 conv → GLU.
- **Decoder level:** mirrors the encoder with ConvTranspose1d. The skip connections are additive.
- **Bottleneck:** the only sequence model, at 62.5 frames/s. It is the part that differs between the versions below.

| Version | Bottleneck | Params | Config |
|---|---|---|---|
| Causal Mamba (stage A, S30, CleanUMamba-3N-E8) | 3 unidirectional Mamba layers, d_model 512, d_state 64, expand 4 | 41.4M | `configs/model/cleanunet_mamba_e8.yaml` |
| Transformer (T1, T2) | 5 self-attention layers, 8 heads, d_model 512, d_inner 2048, no causal mask | 46.1M | `configs/model/cleanunet_tsfm_e8.yaml` |
| Bidirectional Mamba (Z4, M1, M2, M2c) | 3 bidirectional Mamba layers, same sizes | 52.4M | `configs/model/cleanunet_mamba_bidir_e8.yaml` |

### 3.2 Capacity variants of the waveform branch

Each adds one change to the bidirectional Mamba version (IMPROVEMENT_PLAN.md, 16.4, phase 2). The new modules start
close to the identity, and old checkpoints still load.

| Variant | Change | Params | Speed | Peak VRAM |
|---|---|---|---|---|
| base | none (M1's architecture) | 52.4M | 9.2 steps/s | 5.9 GB |
| dconv | Residual dilated convolutions at levels 3–6, encoder and decoder: 2 blocks per level of dilated Conv1d (kernel 3) → GroupNorm → GELU → 1×1 → GroupNorm → GLU → LayerScale 1e-4, channels compressed ×4 | 60.0M | 3.9 steps/s | 16.0 GB |
| levmamba | One bidirectional Mamba layer (d_model 256) at the decoder levels of 250 and 125 frames/s, with LayerScale | 55.3M | 7.4 steps/s | 6.8 GB |
| deepbott | 6 Mamba layers in the bottleneck instead of 3 | 74.5M | 7.1 steps/s | 6.6 GB |
| gated | Gated skips: `skip × 2σ(conv(concat))`, zero-initialized | 57.8M | 7.8 steps/s | 7.5 GB |

dconv at levels 1–6, as first planned, ran at 2.8 steps/s in 24 GB, so it was restricted to levels 3–6.

### 3.3 Spectral branch v1: frame-vector CleanSpecNet

`cleanunet/cleanspecnet.py`. Used by W5, S and MS.

- **Input:** STFT 1024 / 256 (64 ms window, 16 ms hop), log1p magnitude.
- **Structure:** one dense layer projects each frame from 513 to 512. Then a depthwise temporal convolution, the
  backbone over time, and a dense output layer with a sigmoid mask.
- **Limits:** only the dense projections model the frequency axis, and there is no phase output.

| Version | Backbone | Params | Config |
|---|---|---|---|
| W5 (VoiceBank), MS (all data) | 5 bidirectional Mamba layers, expand 2 | 19.5M | `configs/model/cleanunet2_mamba_spec.yaml` |
| S (all data) | 5 self-attention layers | 11.6M | `configs/model/cleanunet2_tsfm_spec.yaml` |

### 3.4 Spectral branch v2: dual-path time-frequency Mamba

`cleanunet/tf_mamba.py`, in the style of MP-SENet and SEMamba. `configs/model/tfmamba_v2_spec{,_s,_l}.yaml`.

- **Input:** STFT 512 / 128 (8 ms hop): power-0.3 compressed magnitude, plus cos and sin of the noisy phase.
- **Encoder:** 2-D convolutions with one ×2 frequency down-sampling (257 → 129 bins).
- **Core:** blocks of bidirectional Mamba over time (one sequence per bin) followed by bidirectional Mamba over
  frequency (one sequence per frame). d_state 16, d_conv 4, expand 2.
- **Decoders:** a bounded magnitude mask, and a phase decoder (pseudo real and imaginary parts, then atan2).

| Size | Channels | Full-resolution channels | Blocks | Params | Speed | Peak VRAM |
|---|---|---|---|---|---|---|
| `tf_s` | 48 | 24 | 4 | 0.73M | 7.4 steps/s | 6.4 GB |
| `tf_m` | 64 | 32 | 4 | 1.09M | 5.6 steps/s | 7.9 GB |
| `tf_l` | 96 | 32 | 6 | 2.48M | 2.6 steps/s | 15.1 GB |

Their checkpoints also hold an unused, frozen 52.4M waveform branch, so `summary.json` reports `params_M` of 53.2–54.9
for them. The counts above are the branch alone. Although `tf_l` is 8× smaller than MS, it runs 2-D sequence models
over every bin and frame, so it is slower: RTF 0.0085 against about 0.0012–0.0033 for the whole M2 + MS system.

### 3.5 Blend

`cleanunet/blend.py`, a fusion with no training:

`|Z| = |Y|^(1−w) · |S|^w`

Y is the waveform output and S the spectral output, in the STFT of the spectral branch. The phase comes from the
waveform branch by default, or from the spectral branch with `--blend_phase spec`.

### 3.6 Systems

| System | Waveform | Spectral | w | Params | Trained on |
|---|---|---|---|---|---|
| T2 + S | T2 (Transformer) | S | 0.5 | 57.6M | all data |
| M2 + MS | M2 (Mamba) | MS | 0.65 | 71.9M | all data |
| M2c + MS | M2c | MS | 0.65 | 71.9M | all data |
| M2c_vbft + MS | M2c, then VoiceBank fine-tune | MS | 0.35 | 71.9M | all data, then VoiceBank |
| M2c + `tf_l` | M2c | `tf_l` | 0.65 | 54.9M | all data (`tf_l`: screening budget) |
| TF1 | — | TF1 | 1.0 | 2.48M | all data |
| TF2 | — | TF2 | 1.0 | 2.48M | all data |
| M2c + TF1, M2c + TF2 | M2c | TF1, TF2 | 0.8 | 54.9M | all data |
| Z4 + W5 | Z4 (Mamba) | W5 | 0.15 | 71.9M | VoiceBank only |
| Z4 hp1.0 + W5 | Z4 hp1.0 | W5 | 0.25 | 71.9M | VoiceBank only |

## 4. Training recipes

### 4.1 Training data

- **VoiceBank-only track:** VoiceBank-DEMAND with 28 speakers, plus external MUSAN noise (p 0.5).
- **All-data track** (`configs/all_data/data.yaml`, details in 15). Each example comes from one component:

| Component | Hours | Weight |
|---|---|---|
| DNS 2020 synthetic pairs | 495.9 | 0.20 |
| VoiceBank-DEMAND 28 speakers | 8.8 | 0.05 |
| EARS-WHAM v2 train | 86.7 | 0.10 |
| Libri2Mix mix_single train | 199.7 | 0.05 |
| URGENT 2026, additive noise only | 83.5 | 0.10 |
| On-the-fly mixtures, SNR −5..20 dB, RIR p 0.2 | — | 0.50 |

Speakers and noises of the evaluation sets are left out of training.

### 4.2 All-data stages

Batch 16 of 15.36 s crops, bf16, polarity inversion p 0.5.

| Stage | Init | Losses | LR | Steps | Audio |
|---|---|---|---|---|---|
| T1, M1 | scratch | L1 + MR-STFT | 2e-4 | 133,300 | ~9,100 h |
| T2, M2 | T1, M1 | + metric discriminator 0.3 | 2e-5 | 5,332 | ~364 h |
| S, MS | scratch | L1 on log1p magnitude | 3e-4 | 39,990 | ~2,730 h |
| M2c | M2 | + compressed complex loss 1.0 (STFT 400 / 100, power 0.3) | 5e-5 | 5,332 | ~364 h |
| M2c_vbft | M2c | same, on VoiceBank + MUSAN, 2.048 s crops | 2e-5 | 15 epochs | — |

Two stages of the time-frequency branch, with batch 8 of 2.048 s crops (results in 5.8):

| Stage | Init | Losses | LR | Steps | Audio |
|---|---|---|---|---|---|
| TF1 | `tf_l` | those of the time-frequency screens (4.3) | 2e-4 | 300,000 | ~1,365 h |
| TF2 | TF1 | + metric discriminator 0.3 on the resynthesis | 2e-5 | 40,000 | ~182 h |

### 4.3 Screening protocol

Short runs on the all-data mixture, 10 epochs, validated on the dev set (`configs/screen/`).

| Runs | Batch | Steps | Audio | LR | Losses |
|---|---|---|---|---|---|
| base, dconv, levmamba, deepbott, gated | 16 × 4.096 s | 50,000 | ~910 h | 2e-4 | L1 + MR-STFT + compressed complex 1.0 |
| `tf_s`, `tf_m`, `tf_l` | 8 × 2.048 s | 100,000 | ~455 h | 3e-4 | magnitude MSE 0.9, anti-wrapping phase 0.3, complex 0.1, consistency 0.1, waveform 0.2 |

None of the screens has a metric discriminator. The time-frequency screens use shorter crops because 16 × 4.096 s took
22 GB at 1 step/s for `tf_s`, and the larger sizes did not fit.

### 4.4 VoiceBank-only lineage

Stage A is the DNS-pretrained CleanUMamba. S30 adds external noise, the metric discriminator (0.3), polarity inversion
and 30 epochs. Z4 adds the bidirectional bottleneck and 60 epochs, and soups (weight averages) of seeds are used.

## 5. Results

### 5.1 Full evaluation: PESQ-WB

Bold marks the best per set.

| System | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| noisy | 2.166 | 1.971 | 1.971 | 1.582 | 1.822 | 1.235 | 1.159 |
| T1 | 2.837 | 2.792 | 2.669 | 2.634 | 2.313 | 1.871 | 1.796 |
| M1 | 2.901 | 2.872 | 2.945 | 2.710 | 2.415 | 2.083 | 1.959 |
| T2 | 2.846 | 2.865 | 2.666 | 2.691 | 2.366 | 1.913 | 1.834 |
| M2 | 2.936 | 2.944 | 3.014 | 2.775 | 2.453 | 2.120 | 1.998 |
| M2c | 2.909 | 2.938 | 2.997 | 2.795 | 2.488 | 2.145 | 2.015 |
| M2c_vbft | 3.127 | 3.149 | 3.195 | 2.785 | 2.379 | 2.096 | 1.879 |
| T2 + S (w 0.5) | 2.942 | 2.976 | 2.839 | 2.931 | 2.739 | 2.138 | 2.064 |
| M2 + MS (w 0.65) | 3.041 | 3.060 | 3.017 | 3.062 | 2.903 | 2.418 | 2.335 |
| M2c + MS (w 0.65) | 3.019 | 3.048 | 2.997 | **3.068** | **2.918** | **2.426** | **2.343** |
| M2c_vbft + MS (w 0.35) | 3.160 | 3.188 | 3.212 | 3.031 | 2.723 | 2.326 | 2.149 |
| Z4 + W5 (w 0.15) | **3.419** | **3.446** | **3.414** | 2.777 | 1.189 | 2.178 | 1.902 |
| Z4 hp1.0 + W5 (w 0.25) | 3.376 | 3.403 | 3.356 | 2.921 | 1.225 | 2.200 | 1.958 |

M2c + `tf_l` and `tf_l` alone are in 5.7, and TF1 and TF2 in 5.8. TF2 supersedes the bold of every column except
`vctk_unseen`.

### 5.2 Full evaluation: SI-SDR (dB)

| System | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| noisy | 8.39 | 8.45 | 7.91 | 9.23 | 9.19 | 5.36 | 3.45 |
| M1 | 16.07 | 16.94 | 15.17 | 14.83 | 13.29 | 12.43 | 10.60 |
| M2 | 16.49 | 17.13 | 15.46 | 14.85 | 13.31 | 12.37 | 10.59 |
| M2c | 17.31 | 17.91 | 16.79 | 16.80 | 14.14 | 13.74 | 11.68 |
| M2c_vbft | 18.47 | 18.60 | 17.68 | 16.58 | 12.26 | 13.15 | 11.25 |
| M2 + MS | 16.93 | 18.00 | 15.75 | 15.65 | 14.80 | 13.20 | 11.31 |
| M2c + MS | 17.59 | 18.60 | 17.11 | **17.88** | **15.76** | **14.71** | **12.52** |
| M2c_vbft + MS | 18.43 | 18.90 | 17.84 | 17.41 | 13.62 | 13.99 | 11.95 |
| Z4 + W5 | 19.11 | 19.24 | 17.84 | 14.55 | −3.31 | 11.30 | 9.31 |
| Z4 hp1.0 + W5 | **19.17** | **19.31** | **17.89** | 15.57 | −2.54 | 10.94 | 9.41 |

### 5.3 Full evaluation: other metrics on the VoiceBank test set

| System | PESQ-NB | STOI | CSIG | CBAK | COVL |
|---|---|---|---|---|---|
| noisy | 2.880 | 0.921 | 3.488 | 2.554 | 2.743 |
| M2 | 3.563 | 0.948 | 4.396 | 3.544 | 3.733 |
| M2c | 3.584 | 0.948 | 4.404 | 3.581 | 3.733 |
| M2 + MS | 3.672 | 0.952 | 4.536 | 3.651 | 3.864 |
| M2c + MS | 3.669 | 0.952 | 4.529 | 3.672 | 3.853 |
| M2c_vbft + MS | 3.753 | 0.954 | 4.637 | 3.750 | 3.990 |
| Z4 + W5 | 3.902 | 0.958 | 4.754 | 3.878 | 4.208 |
| Z4 hp1.0 + W5 | 3.926 | 0.958 | 4.747 | 3.866 | 4.177 |

### 5.4 Dev set: every screen and reference

Each cell is PESQ-WB / SI-SDR (dB) / STOI. Means are over the three domains.

| System | VB | EARS-WHAM | Libri2Mix | Mean PESQ | Mean SI-SDR |
|---|---|---|---|---|---|
| base | 2.796 / 17.43 / 0.913 | 1.957 / 12.78 / 0.863 | 1.742 / 11.06 / 0.877 | 2.165 | 13.76 |
| dconv | 2.807 / 18.19 / 0.915 | 1.968 / 13.04 / 0.866 | 1.765 / 11.25 / 0.880 | 2.180 | 14.16 |
| levmamba | 2.812 / 17.77 / 0.914 | 1.963 / 12.87 / 0.863 | 1.753 / 11.09 / 0.878 | 2.176 | 13.91 |
| deepbott | 2.831 / 17.53 / 0.914 | 1.957 / 12.82 / 0.863 | 1.747 / 11.04 / 0.876 | 2.178 | 13.80 |
| gated | 2.783 / 17.55 / 0.914 | 1.956 / 12.79 / 0.862 | 1.735 / 11.06 / 0.876 | 2.158 | 13.80 |
| `tf_s` | 2.891 / 18.55 / 0.922 | 1.946 / 11.63 / 0.854 | 1.997 / 11.54 / 0.886 | 2.278 | 13.91 |
| `tf_m` | 3.017 / 18.98 / 0.923 | 2.230 / 13.80 / 0.889 | 2.204 / 12.60 / 0.907 | 2.484 | 15.13 |
| `tf_l` | 3.167 / 19.30 / 0.926 | 2.359 / 14.91 / 0.899 | 2.379 / 13.51 / 0.919 | 2.635 | 15.91 |
| M2 | 2.965 / 16.48 / 0.919 | 2.117 / 12.54 / 0.879 | 1.948 / 10.60 / 0.892 | 2.343 | 13.21 |
| M2c | 2.930 / 17.31 / 0.918 | 2.142 / 13.48 / 0.878 | 1.969 / 11.70 / 0.892 | 2.347 | 14.16 |
| M2 + MS (w 0.65) | 3.055 / 17.00 / 0.924 | 2.346 / 13.26 / 0.897 | 2.299 / 11.40 / 0.913 | 2.567 | 13.88 |
| M2c + MS (w 0.65) | 3.027 / 17.67 / 0.924 | 2.356 / 14.28 / 0.897 | 2.306 / 12.56 / 0.913 | 2.563 | 14.83 |
| M2c + `tf_l` (w 0.65, M2c phase) | 3.144 / 18.87 / 0.927 | 2.392 / 14.31 / 0.898 | 2.350 / 12.65 / 0.916 | 2.629 | 15.27 |
| M2c + `tf_l` (w 1.0, M2c phase) | 3.142 / 18.96 / 0.925 | 2.340 / 14.27 / 0.896 | 2.339 / 12.68 / 0.916 | 2.607 | 15.30 |
| M2c + `tf_l` (w 0.65, `tf_l` phase) | 3.183 / **19.28** / 0.928 | 2.449 / 15.07 / 0.903 | 2.440 / 13.58 / 0.921 | 2.691 | 15.98 |
| TF1 | 3.318 / 18.68 / 0.930 | 2.536 / 15.79 / 0.914 | 2.548 / 14.16 / 0.931 | 2.801 | 16.21 |
| M2c + TF1 (w 0.8, TF1 phase) | 3.311 / 18.93 / **0.931** | 2.579 / **15.84** / **0.915** | 2.579 / **14.21** / **0.932** | 2.823 | **16.33** |
| **TF2** | **3.442** / 18.32 / 0.930 | 2.629 / 15.66 / 0.914 | 2.671 / 14.08 / **0.932** | 2.914 | 16.02 |
| M2c + TF2 (w 0.8, TF2 phase) | 3.416 / 18.67 / **0.931** | **2.656** / 15.74 / **0.915** | **2.676** / 14.14 / **0.932** | **2.916** | 16.18 |

The VoiceBank-only systems (Z4 + W5) were not evaluated on the dev set.

**PESQ-WB validation curves** (one value per epoch, mean over the domains):

| Run | Curve |
|---|---|
| base | 1.621, 1.839, 1.924, 1.939, 2.012, 2.078, 2.075, 2.129, 2.145, 2.159 |
| dconv | 1.764, 1.929, 1.971, 1.999, 2.031, 2.113, 2.122, 2.149, 2.187, 2.175 |
| levmamba | 1.663, 1.885, 1.950, 1.992, 2.058, 2.095, 2.122, 2.145, 2.160, 2.169 |
| deepbott | 1.683, 1.906, 1.914, 1.991, 2.024, 2.063, 2.113, 2.159, 2.162, 2.171 |
| gated | 1.676, 1.859, 1.915, 1.978, 2.047, 2.050, 2.133, 2.145, 2.151, 2.151 |
| `tf_s` | 1.834, 2.146, 2.065, 2.229, 2.277, 2.224, 2.291, 2.237, 2.277, 2.280 |
| `tf_m` | 1.967, 2.213, 2.301, 2.335, 2.407, 2.432, 2.475, 2.494, 2.482, 2.478 |
| `tf_l` | 2.142, 2.350, 2.437, 2.483, 2.570, 2.561, 2.646, 2.618, 2.636, 2.626 |
| TF1 (30 epochs, from `tf_l`) | 2.578, 2.627, 2.602, 2.632, 2.674, 2.651, 2.717, 2.710, 2.729, 2.723, 2.723, 2.717, 2.677, 2.773, 2.759, 2.750, 2.756, 2.744, 2.740, 2.772, 2.787, 2.793, 2.790, 2.786, 2.792, 2.802, 2.799, 2.798, 2.802, 2.795 |
| TF2 (4 epochs, from TF1) | 2.869, 2.903, 2.915, 2.914 |

### 5.5 Waveform capacity variants against base

Difference in PESQ-WB [95% CI] / SI-SDR (dB).

| Variant | VB | EARS-WHAM | Libri2Mix |
|---|---|---|---|
| dconv | +0.011 [−0.005, +0.027] / +0.76 | +0.011 [+0.002, +0.021] / +0.27 | +0.023 [+0.015, +0.033] / +0.19 |
| levmamba | +0.016 [+0.003, +0.029] / +0.33 | +0.006 [−0.001, +0.014] / +0.09 | +0.011 [+0.002, +0.019] / +0.02 |
| deepbott | +0.035 [+0.021, +0.049] / +0.10 | +0.001 [−0.007, +0.008] / +0.05 | +0.005 [−0.003, +0.012] / −0.02 |
| gated | −0.014 [−0.026, −0.001] / +0.12 | 0.000 [−0.007, +0.007] / +0.01 | −0.007 [−0.015, 0.000] / 0.00 |

**Criterion** (16.4, phase 2): PESQ-WB ≥ +0.03, SI-SDR ≥ −0.3 dB and STOI ≥ −0.003, on at least 3 of the 4 domains.
No variant passes. Only deepbott reaches +0.03, and only on VB.

### 5.6 Time-frequency systems against M2 + MS

Difference in PESQ-WB [95% CI] / SI-SDR (dB).

| System | VB | EARS-WHAM | Libri2Mix | Mean |
|---|---|---|---|---|
| `tf_s` | −0.164 [−0.185, −0.140] / +1.55 | −0.400 [−0.423, −0.374] / −1.63 | −0.302 [−0.325, −0.281] / +0.14 | −0.289 / +0.02 |
| `tf_m` | −0.038 [−0.057, −0.019] / +1.98 | −0.116 [−0.130, −0.101] / +0.54 | −0.095 [−0.113, −0.078] / +1.21 | −0.083 / +1.24 |
| `tf_l` | +0.112 [+0.090, +0.135] / +2.30 | +0.013 [−0.003, +0.029] / +1.66 | +0.080 [+0.062, +0.097] / +2.11 | **+0.068** / +2.02 |
| M2c + `tf_l`, w 0.65, M2c phase | +0.089 [+0.073, +0.106] / +1.87 | +0.045 [+0.034, +0.058] / +1.05 | +0.051 [+0.041, +0.063] / +1.26 | +0.062 / +1.39 |
| M2c + `tf_l`, w 1.0, M2c phase | +0.087 [+0.066, +0.110] / +1.97 | −0.006 [−0.021, +0.008] / +1.01 | +0.040 [+0.025, +0.055] / +1.29 | +0.040 / +1.42 |
| M2c + `tf_l`, w 0.65, `tf_l` phase | +0.129 [+0.111, +0.146] / +2.28 | +0.103 [+0.091, +0.116] / +1.81 | +0.140 [+0.128, +0.153] / +2.19 | **+0.124** / +2.09 |

**Criterion** (16.4, phase 1): dev PESQ-WB ≥ +0.05 over M2 + MS, with SI-SDR not lower. `tf_l` passes on the mean. On
EARS-WHAM alone, its PESQ gain is not distinguishable from zero. `tf_s` and `tf_m` do not pass.

### 5.7 M2c + `tf_l` on the evaluation sets

The weight was picked on the dev set, with the phase of `tf_l`, by the mean of the per-domain means:

| w | 0.25 | 0.35 | 0.5 | 0.65 | 0.8 |
|---|---|---|---|---|---|
| Mean PESQ-WB | 2.596 | 2.636 | 2.676 | **2.691** | 2.681 |
| Mean SI-SDR (dB) | 15.59 | 15.73 | 15.89 | 15.98 | 16.00 |

PESQ-WB:

| System | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| `tf_l` alone | 3.161 | 3.136 | 3.104 | 3.270 | 2.980 | 2.411 | 2.417 |
| **M2c + `tf_l`** | 3.181 | 3.216 | 3.180 | **3.325** | **3.002** | **2.507** | **2.490** |
| M2 + MS | 3.041 | 3.060 | 3.017 | 3.062 | 2.903 | 2.418 | 2.335 |
| Z4 + W5 | **3.419** | **3.446** | **3.414** | 2.777 | 1.189 | 2.178 | 1.902 |
| M2c + `tf_l` − M2 + MS | +0.140 | +0.156 | +0.163 | +0.263 | +0.099 | +0.089 | +0.155 |
| M2c + `tf_l` − Z4 + W5 | −0.238 | −0.230 | −0.234 | +0.548 | +1.813 | +0.329 | +0.588 |

SI-SDR (dB):

| System | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| `tf_l` alone | 19.26 | 18.13 | 18.35 | 19.13 | 15.74 | 15.52 | 13.48 |
| M2c + `tf_l` | 19.22 | 18.76 | 18.32 | 19.18 | 15.87 | 15.63 | 13.54 |
| M2 + MS | 16.93 | 18.00 | 15.75 | 15.65 | 14.80 | 13.20 | 11.31 |
| Z4 + W5 | 19.11 | 19.24 | 17.84 | 14.55 | −3.31 | 11.30 | 9.31 |

Other metrics of M2c + `tf_l`:

| Metric | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| PESQ-NB | 3.844 | 3.837 | 3.888 | 3.695 | 3.489 | 3.060 | 3.069 |
| STOI | 0.927 | 0.958 | 0.848 | 0.975 | 0.933 | 0.916 | 0.926 |
| CSIG | 4.542 | 4.658 | 4.511 | 4.732 | 4.585 | 4.111 | 4.206 |
| CBAK | 3.775 | 3.754 | 3.680 | 4.054 | 3.866 | 3.309 | 3.294 |
| COVL | 3.950 | 4.021 | 3.920 | 4.144 | 3.876 | 3.365 | 3.398 |

- `tf_l` alone beats M2 + MS on PESQ-WB on 6 of 7 sets. EARS-WHAM is the exception (2.411 against 2.418).
- On the VoiceBank test set, M2c + `tf_l` beats the VoiceBank fine-tune M2c_vbft + MS (3.216 against 3.188) without
  losing the other domains. On `vctk_unseen` it does not (3.180 against 3.212).
- Against Z4 + W5, the VoiceBank deficit is PESQ-WB, PESQ-NB (3.837 against 3.902 on test) and test SI-SDR (−0.48 dB).
  SI-SDR is higher on `val_unseen` (+0.11 dB) and `vctk_unseen` (+0.48 dB).
- RTF is about 0.010, against 0.0012–0.0026 for M2 + MS.

### 5.8 Full time-frequency runs: TF1 and TF2

TF1 is `tf_l` trained for 300,000 steps, and TF2 is TF1 with the metric discriminator (4.2). The blend weight with
M2c was picked on the dev set (phase of the branch): w 0.8, the edge of the grid. Details in `IMPROVEMENT_PLAN.md`,
16.8.

Dev set, mean over the three domains:

| System | Mean PESQ-WB | Mean SI-SDR (dB) |
|---|---|---|
| M2c + `tf_l` (w 0.65) | 2.691 | 15.98 |
| TF1 | 2.801 | 16.21 |
| M2c + TF1 (w 0.8) | 2.823 | 16.33 |
| TF2 | 2.914 | 16.02 |
| M2c + TF2 (w 0.8) | 2.916 | 16.18 |

PESQ-WB:

| System | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| M2c + `tf_l` | 3.181 | 3.216 | 3.180 | 3.325 | 3.002 | 2.507 | 2.490 |
| TF1 | 3.304 | 3.336 | 3.184 | 3.397 | 3.247 | 2.589 | 2.579 |
| M2c + TF1 | 3.300 | 3.357 | 3.229 | 3.428 | 3.245 | 2.636 | 2.620 |
| **TF2** | **3.426** | **3.478** | 3.335 | 3.484 | **3.359** | 2.694 | 2.711 |
| M2c + TF2 | 3.404 | **3.478** | 3.351 | **3.494** | 3.346 | **2.722** | **2.724** |
| Z4 + W5 | 3.419 | 3.446 | **3.414** | 2.777 | 1.189 | 2.178 | 1.902 |

SI-SDR (dB):

| System | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| TF1 | 18.69 | 19.02 | 18.60 | **20.01** | 16.80 | 16.32 | 14.16 |
| M2c + TF1 | 18.92 | **19.29** | **18.65** | 20.00 | 16.77 | **16.36** | **14.20** |
| TF2 | 18.33 | 18.65 | 18.40 | 19.94 | **17.04** | 16.14 | 14.07 |
| M2c + TF2 | 18.65 | 19.03 | 18.45 | 19.95 | 16.96 | 16.22 | 14.12 |
| Z4 + W5 | **19.11** | 19.24 | 17.84 | 14.55 | −3.31 | 11.30 | 9.31 |

Other metrics of TF2:

| Metric | val_unseen | test | vctk_unseen | DNS no_reverb | DNS with_reverb | EARS-WHAM | Libri2Mix |
|---|---|---|---|---|---|---|---|
| PESQ-NB | 3.933 | 3.957 | 3.969 | 3.802 | 3.721 | 3.229 | 3.270 |
| PESQ-WB above 100 Hz | 3.558 | 3.571 | 3.572 | 3.510 | 3.388 | 2.714 | 2.769 |
| STOI | 0.929 | 0.961 | 0.851 | 0.979 | 0.943 | 0.924 | 0.936 |
| CSIG | 4.670 | 4.764 | 4.586 | 4.789 | 4.710 | 4.223 | 4.352 |
| CBAK | 3.842 | 3.855 | 3.756 | 4.166 | 4.102 | 3.381 | 3.409 |
| COVL | 4.171 | 4.242 | 4.047 | 4.275 | 4.168 | 3.516 | 3.584 |

For Z4 + W5 on `val_unseen` / test / `vctk_unseen`: PESQ-WB above 100 Hz 3.505 / 3.510 / 3.523, PESQ-NB 3.873 /
3.902 / 3.888, CSIG 4.696 / 4.754 / 4.683, COVL 4.171 / 4.208 / 4.153.

- **Full run:** TF1 − `tf_l` is +0.166 on the dev set and +0.072 to +0.267 PESQ-WB on the evaluation sets.
- **Metric discriminator:** TF2 − TF1 is +0.09 to +0.15 PESQ-WB on every set, and −0.07 to −0.37 dB of SI-SDR except
  on DNS with_reverb (+0.24 dB).
- **Blend:** with TF2, M2c changes PESQ-WB by −0.022 to +0.028 and adds 0.06–0.38 dB of SI-SDR.
- **Size:** TF2 alone uses 2.48M parameters. `evaluate.py` reports 54.9M because the checkpoint also holds the frozen
  M2c branch. Its RTF (0.0084 in `evaluate.py`) was not measured for the branch alone.

## 6. Findings

### 6.1 Mamba against Transformer

Mamba wins at every stage, on every set and metric. M1 − T1 is +0.06 to +0.28 PESQ-WB. M2 + MS − T2 + S is +0.08 on
test and +0.28 on EARS-WHAM (15.2.1).

### 6.2 VoiceBank specialization

- Off VoiceBank, M2 + MS beats both VoiceBank-only systems on all four sets. On DNS with_reverb the VoiceBank systems
  fall below the noisy input: they never learned to keep reverberation.
- On the VoiceBank test set, M2 + MS is 0.386 behind Z4 + W5.
- **Fine-tune control (M2c_vbft + MS):** closes 0.128 of that gap, about a third (test 3.188). It costs 0.100 on
  EARS-WHAM and 0.194 on Libri2Mix against M2c + MS.
- The rest may come from Z4's DNS pretraining and longer VoiceBank training. Item 0.5(b) of the plan, which would test
  this, was not run.

Blend weight of M2c_vbft + MS on `val_unseen`:

| w | 0.15 | 0.35 | 0.5 | 0.65 | 0.8 |
|---|---|---|---|---|---|
| PESQ-WB | 3.153 | **3.160** | 3.147 | 3.122 | 3.085 |
| SI-SDR (dB) | 18.51 | 18.43 | 18.29 | 18.06 | 17.75 |

### 6.3 Error budget

Each system is rescored with the clean phase or the clean magnitude (16.2). PESQ-WB / SI-SDR:

| System, set | As is | Perfect phase | Perfect magnitude |
|---|---|---|---|
| M2 + MS, VB test | 3.060 / 18.00 | 3.442 / 23.59 | 4.154 / 19.72 |
| M2 + MS, DNS no_reverb | 3.062 / 15.65 | 3.506 / 22.27 | 4.202 / 17.67 |
| M2 + MS, EARS-WHAM | 2.411 / 13.20 | 2.809 / 19.08 | 3.933 / 15.30 |
| Z4 + W5, VB test | 3.447 / 19.24 | 3.787 / 24.71 | 4.230 / 20.81 |

Magnitude errors cost about three times as much PESQ as phase errors. The gap of M2 + MS to Z4 + W5 on VoiceBank is
mostly magnitude, and it is uniform across input-SNR bins (−0.35 to −0.41).

### 6.4 High-band phase defect and its fix

Mean phase difference between output and clean reference on the VB test set, and its coherence:

| Band | 5–6.5 kHz | 6.5–7 kHz | 7–7.5 kHz | 7.5–8 kHz |
|---|---|---|---|---|
| Rotation, M2 | −12.8° | −74.1° | −108.5° | −138.7° |
| Rotation, M2c | +0.8° | −3.1° | −4.8° | −4.0° |
| Coherence, M2 | 0.962 | 0.975 | 0.971 | 0.974 |
| Coherence, M2c | 0.995 | 0.992 | 0.991 | 0.975 |

- **Cause:** training from scratch with losses whose only phase-sensitive term is the waveform L1, while 5–8 kHz holds
  about 1% of the energy. The data show no such rotation, and the DNS-pretrained lineage does not have it.
- **Fix:** the compressed complex loss (M2c).
- **Effect of M2c against M2:** SI-SDR rises on all seven sets (+0.78 to +1.95 dB): +0.78 dB on test and +1.37 dB
  on EARS-WHAM. PESQ-WB changes by −0.006 on test, +0.025 on EARS-WHAM, −0.027 on `val_unseen` and −0.017 on
  `vctk_unseen`.
- **Status against the gate** (coherence ≥ 0.99, rotation < 10°, SI-SDR ≥ +0.4 / +1.2 dB, PESQ ≥ −0.01): rotation and
  SI-SDR pass. Coherence passes except at 7.5–8 kHz. PESQ passes on test and EARS-WHAM, not on `val_unseen` or
  `vctk_unseen`.

### 6.5 Blend

The weight is near its ceiling for MS: picking the best weight per set adds at most +0.045, and +0.011 on average
(16.2). With `tf_l`, the phase source matters more than the weight: `tf_l` phase instead of M2c phase adds +0.039 to
+0.090 PESQ-WB at w 0.65. With the full branches the dev grid picks w 0.8, its edge, and with TF2 the blend adds
only +0.002 on the dev set (5.8).

### 6.6 VoiceBank-only history

Test PESQ-WB of the main milestones (sections 12 and 14):

| System | Test PESQ-WB | Params | Note |
|---|---|---|---|
| Original hybrid CleanUNet2-Mamba | 3.096 | — | Ties the waveform-only model (3.095): FiLM conditioning stayed near zero |
| K0full, waveform control | 3.085 | 41.4M | |
| K0extfull, + external noise | 3.164 | 41.4M | External noise: +0.079, no inference cost |
| RX_C3, learned fusion | 3.212 | 53.8M | Depends on polarity; a training-free blend ties it |
| S30 soup | 3.335 | 41.4M | + metric discriminator, polarity inversion, 30 epochs; streaming-compatible (3.323 streaming with the scale prior) |
| S30 soup + W5 | 3.375 | 60.9M | Training-free blend |
| Bidirectional S30 soup + W5 | 3.421 | 71.9M | Bidirectional bottleneck, 30 epochs |
| **Z4 soup + W5** | **3.446** | 71.9M | 60 epochs; Z4 soup alone 3.428 (52.4M) |
| Z4 hp1.0 soup + W5 | 3.403 | 71.9M | System for use: PESQ-NB 3.926, PESQ-WB above 100 Hz 3.554 |

**Also rejected or inconclusive:** output ensembles (+0.005 over soups), metric weight 0.5, W5 fine-tuned through the
blend, and larger batches (3.312 against 3.375). Learned fusion variants ended in ties.

## 7. Limitations

- **Unequal budgets on the dev set.** `tf_l` had ~455 h of audio and no metric discriminator. M2 had ~9,500 h and MS
  ~2,730 h. The time-frequency screens also saw half the audio of the waveform screens, with shorter crops and a
  higher learning rate, so branch comparisons are not like for like.
- **Screens are short.** The curves of the waveform variants still rise at epoch 10, so small differences could change
  in a full run. `tf_l` flattens after epoch 7 with this learning-rate schedule.
- **Two sets of files.** Sections 5.1–5.3 and 5.7 use the evaluation sets, and 5.4–5.6 the dev set. They cannot be
  compared directly.
- **Soup comparison.** Z4 is a soup of seeds, while the all-data models are single runs.
- **Objective metrics only.** The listening test (14.8, X5) has no answers. DNSMOS did not confirm the discriminator's
  gain on VoiceBank (14.8).
- **RTF** comes from `evaluate.py` runs on a shared GPU, not from a dedicated benchmark.
- **`last.ckpt` is not the end of training.** `train.LastEveryEpochCheckpoint` never writes (`IMPROVEMENT_PLAN.md`,
  16.8), so `last.ckpt` is the latest of the two best validation checkpoints: epoch 29 of 30 for TF1, epoch 9 of 10
  for `tf_l`. The evaluations used those, so they are selected on the dev set, one epoch from the end at most. The
  callback is fixed for later runs.
- **The blend weight sits at the edge of the grid** (w 0.8) for TF1 and TF2.

## 8. Incidents in the last queues

**Full time-frequency runs** (`logs/plan16/run_tf_full.sh`, 2026-11-03 12:04 to 2026-11-06 01:54). No run or
evaluation failed.

- The machine stopped on 2026-11-03 between 20:40 and 20:50, during the validation of TF1 epoch 8, with no clean
  shutdown. It came back on 2026-11-04 at 16:07. The cause is unknown (`kern.log` is not readable from this account).
- TF1 resumed from `last.ckpt` (step 70,000) on 2026-11-04 at 19:02, losing about 10,000 steps, and ran to the end
  without a restart. TF2 ran in one attempt.

**Plan queue.** The queue of the plan (`logs/plan16/run_plan16.sh`) ran from 2026-11-01 12:23 to 2026-11-03 03:41. No evaluation file
failed.

- Three runs stalled, and the watchdog killed and resumed them: M2c at 13:45, base at 15:49 and dconv at 18:10.
  - The stack dumps of base and dconv show the main process waiting for a data-loader batch. Why the loader stalled is
    not known.
  - The first dump of M2c is inside PESQ, in the metric-discriminator pool, which section 15 already notes relies on
    the watchdog.
- The watchdog killed only the main process. In two cases orphaned loader workers kept the GPU memory, which stopped
  the queue from 18:10 to 22:51 until they were killed by hand. The next queue (`logs/plan16/run_tf_full.sh`) starts
  each training in its own process group and kills the whole group.

## 9. Next steps

The full time-frequency runs are done (5.8).

**Proposed, not started:**

- Phase 3: blend gate and target-style conditioning, on top of TF2.
- Phase 4: final model. dconv is the only waveform candidate worth a longer test, given its cost.
- A larger time-frequency branch: each size so far improved every domain. A metric loss that keeps SI-SDR.
- Use TF2 alone as the all-data system, unless the SI-SDR of the blend matters.
