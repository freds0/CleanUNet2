# Architecture improvement proposals: CleanUNet2-Mamba

*Author: Claude (Opus 5.5) · 2026-10-12 · branch `feature/mamba`*

This document lists architecture changes to evaluate experimentally on CleanUNet2-Mamba. Each proposal gives:
- the motivation, backed by measurements from this repository;
- the hypothesis to test;
- the estimated cost;
- the design of the experiment;
- the success criterion.

Markers used in the text:

| Marker | Meaning |
|---|---|
| ✔ | Measured in this repository (scripts and numbers in section 1) |
| ◇ | Estimate, to be confirmed |
| ⚑ | Already exists as a configuration option; only the experiment is missing |

---

## 0. Summary: the priorities

| # | Proposal | Main reason | Implementation | GPU (screening) |
|---|---|---|---|---|
| 1 | **A0** Exact, fast upsampler | The upsampler takes ~83% of a hybrid training step ✔. The exact reformulation is 4.3× faster ✔ | ~30 lines, weights unchanged | none |
| 2 | **A5** Controls (LR, seeds, shuffled condition) | Without them, gains cannot be attributed to the spectral branch | ~20 lines | ~2 h |
| 3 | **C1** Causal normalization | The whole streaming loss (−0.16 PESQ ✔) comes from the causal normalization | ~40 lines | 0 h (inference only) to 1 h |
| 4 | **A1** Frequency-aware upsampler | The current upsampler cannot weigh frequency bands ✔ (analysis) and reduces everything to 1 channel | ~80 lines | ~1 h per variant ◇ |
| 5 | **A2/A3** Bottleneck and multi-scale fusion | STFT hop (256) = total U-Net stride (2⁸): fusion is natural and cheap | ⚑ / ~60 lines | ~1 h per variant ◇ |
| 6 | **D2** SI-SDR term in the loss | Fine-tuning raised PESQ (+0.27) but lowered SI-SDR (−0.37 dB, p≈1e-14) ✔ | ~15 lines | ~1 h |
| 7 | **B1** Residual or mask CleanSpecNet | Today the spectral branch regresses the magnitude directly, with no input-to-output path | ~20 lines | ~30 min |
| 8 | **C2** BiMamba bottleneck (offline mode) | Measures the cost of causality; can start from the DNS weights with no initial loss | ~40 lines | ~1 h |

---

## 1. Measured evidence

### 1.1 Test results (VoiceBank-DEMAND, 824 files) ✔

All these results use the **waveform-only model**: CleanUNet with a Mamba bottleneck, identical to CleanUMamba-E8, 41.4M parameters.

| System | PESQ-WB | STOI | SI-SDR | CSIG | CBAK | COVL |
|---|---|---|---|---|---|---|
| Noisy | 1.97 | 0.921 | 8.4 | 3.49 | 2.55 | 2.74 |
| CleanUMamba-E8 (DNS), zero-shot | 2.82 | 0.942 | 18.9 | 4.18 | 3.59 | 3.55 |
| Stage A (E8 fine-tuned for 20 epochs) | **3.09** | **0.953** | 18.6 | **4.57** | **3.67** | **3.90** |
| Stage A, streaming (16 ms hop) | 2.93 | 0.945 | 17.4 | 4.46 | 3.57 | 3.76 |

CSIG, CBAK and COVL use the Python implementation of CMGAN/CleanUMamba. On noisy audio it gives ~0.1 more than the published values (computed in MATLAB), so compare these three metrics only between results of this repository.

### 1.2 Hybrid vs waveform-only (validation, 770 files, preliminary) ✔

PESQ-WB per epoch. The hybrid uses the same waveform branch initialized from DNS, plus the CleanSpecNet-Mamba and zero-initialized FiLM.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Waveform-only (stage A) | 2.666 | 2.644 | 2.662 | 2.673 | 2.674 | 2.671 | 2.677 | 2.726 | 2.699 | 2.701 | 2.722 |
| Hybrid | 2.638 | 2.684 | 2.680 | 2.672 | 2.712 | 2.670 | 2.714 | 2.738 | 2.703 | 2.742 | 2.732 |
| Difference | −0.03 | +0.04 | +0.02 | 0.00 | +0.04 | 0.00 | +0.04 | +0.01 | 0.00 | +0.04 | +0.01 |

The mean difference is **+0.015**: small, noisy and confounded. The waveform branch trains with an effective LR of 3e-5 in the hybrid (1e-4 × 0.3) and of 5e-5 in stage A (see A5).

### 1.3 Where the cost is ✔

**FLOPs**, counted with `torch.utils.flop_counter` on a 2 s forward pass, batch 1:

| Model | Total | CleanUNet | CleanSpecNet | SpecUpsampler |
|---|---|---|---|---|
| Mamba hybrid | 24.4 GMAC | 21.3 (87%) | 1.39 (6%) | 1.72 (7%) |
| Transformer hybrid (E0) | 24.7 GMAC | 22.0 (89%) | 0.92 (4%) | 1.72 (7%) |
| Mamba waveform-only | 21.3 GMAC | 21.3 | — | — |

The elementwise operations of the Mamba scan are not included in this count.

**Time per step** (forward + backward, bf16, batch 2, with the GPU shared with another training job):

| Variant | Time |
|---|---|
| Full hybrid | 145 ms |
| Hybrid with the upsampler replaced by a trivial operation | 24 ms |
| CleanUNet only | 18 ms |
| CleanSpecNet only | 6 ms |
| Upsampler only, in bf16 | 121 ms (forward alone: 108 ms) |
| Upsampler only, in fp32 | 56 ms |

**Upsampler memory:** a 1.8 GB peak with batch 2, i.e. ~14 GB with batch 16 ◇. This explains the ~17.7 GB of GPU memory used by the hybrid trainings.

**Conclusion:** the upsampler is 7% of the arithmetic but ~83% of the time. The bottleneck is an implementation issue: the `ConvTranspose2d` with stride equal to the kernel along time falls into a slow cuDNN kernel in bf16, with huge activations of shape (B, 32, 513, 16T). The rest of the hybrid costs only ~1.3× the waveform-only model.

### 1.4 Design limitation of the SpecUpsampler (code analysis) ✔

Both `ConvTranspose2d` layers use kernel 3 with stride 1 along frequency, i.e. the same weights for every band. They are followed by a mean over the 513 bands. Consequences:
- The conditioning signal of each sample is essentially the mean of one nonlinear function applied to local neighbourhoods of the spectrum.
- The module **cannot give different weights to different bands**: energy at 200 Hz and at 6 kHz counts the same.
- All spectral information reaches CleanUNet as **a single channel**.

### 1.5 Performance per input SNR (test) ✔

**PESQ-WB:**

| Input SNR | < 5 dB (n=218) | 5–10 (n=209) | 10–15 (n=206) | ≥ 15 (n=191) |
|---|---|---|---|---|
| Noisy | 1.46 | 1.74 | 2.15 | 2.61 |
| Zero-shot | 2.43 | 2.76 | 2.96 | 3.19 |
| Stage A | 2.68 | 3.02 | 3.23 | 3.50 |
| Streaming | 2.66 | 2.92 | 3.03 | 3.16 |

**SI-SDR (dB):**

| Input SNR | < 5 dB | 5–10 | 10–15 | ≥ 15 |
|---|---|---|---|---|
| Zero-shot | 16.3 | 18.6 | 19.9 | 21.2 |
| Stage A | 16.0 | 17.8 | 19.4 | 21.4 |
| Streaming | 16.1 | 17.5 | 18.0 | 18.0 |

What these numbers show:
- **There is more room for improvement at low SNR:** PESQ 2.68 below 5 dB, against 3.50 above 15 dB. After stage A, 58 files still have PESQ < 2.0.
- **Fine-tuning traded SI-SDR for PESQ** precisely in the low and middle SNR ranges. The loss is L1 + MR-STFT, with no SI-SDR term.
- **The streaming loss grows with the SNR:** −0.02, −0.10, −0.20 and −0.34 PESQ in the four ranges. SI-SDR saturates at ~18 dB. This is the pattern of a **time-varying gain**, as expected from the cumulative normalization.

### 1.6 Origin of the streaming loss ✔

With the normalization disabled, streaming reproduces the offline mode exactly:
- in float64, the difference is 2.5e-14;
- in fp32/CUDA on real speech, the agreement is above 40 dB.

So the −0.16 PESQ loss (p≈2.5e-50) comes **entirely from the causal normalization**. The streamer divides the input by the running mean of the per-frame standard deviations, while the model was trained with the global standard deviation of the utterance.

### 1.7 Statistical power ✔

- Standard deviation of the per-file paired PESQ differences: ~0.33, which gives a standard error of 0.012 over the 824 test files. Differences of ~0.03 stand out from the file-to-file noise.
- The **variance between seeds has not been measured yet** (proposal A5b). Until then, differences < 0.03 between single runs are not conclusive.

---

## 2. Proposals

Format of each proposal: **Motivation** · **Change** · **Hypothesis** · **Cost** · **Experiment** · **Success** · **Risks**.

Unless stated otherwise, comparisons use:
- 20 training epochs, starting the waveform branch from the DNS weights;
- checkpoint selection by validation PESQ;
- one seed for screening and three for confirmation.

### Group A: link between the branches (fusion)

#### A0. Exact, fast upsampler (engineering, prerequisite)
- **Motivation:** 1.3.
- **Change:** with stride equal to the kernel along time (16), each `ConvTranspose2d` is exactly equivalent to a `Conv2d` with kernel (3, 1), with the weights flipped along frequency and rearranged, followed by a channel-to-time rearrangement (pixel shuffle). The parameters stay the same, so checkpoints remain compatible.
- **Verified with a prototype:**
  - maximum difference from the original: 1.0e-5 ✔;
  - forward + backward in bf16 with batch 4: 238 → 55 ms ✔.

  The core of the prototype (valid when `stride[1] == kernel[1]` and `padding[1] == 0`):
  ```python
  def convT_as_conv(x, convT):                      # x: (B, Cin, F, T) -> (B, Cout, F, T*kt)
      W = convT.weight                              # (Cin, Cout, kf, kt)
      Cin, Cout, kf, kt = W.shape
      Wc = W.flip(2).permute(1, 3, 0, 2).reshape(Cout * kt, Cin, kf, 1)
      y = F.conv2d(x, Wc, padding=(kf - 1 - convT.padding[0], 0))      # (B, Cout*kt, F, T)
      B, _, Fq, T = y.shape
      y = y.view(B, Cout, kt, Fq, T).permute(0, 1, 3, 4, 2).reshape(B, Cout, Fq, T * kt)
      return y + convT.bias.view(1, -1, 1, 1)
  ```
- **Expected effect:** hybrid step from ~145 to ~52 ms (≈2.8×), estimated from the measurements of section 1.3 ◇.
- **Experiment:** none for metrics. A regression test (numerical equivalence) and timing one epoch are enough.
- **Success:** hybrid epoch ≤ 3.5 min (today ~8 min).

#### A1. Frequency-aware, lightweight, multi-channel upsampler
- **Motivation:** 1.3 and 1.4.
- **Change:** a new module (`upsampler.type: freq_proj`) with three layers:
  1. `Conv1d(513 → C, k=1)`: separate weights for each band;
  2. `ConvTranspose1d(C, C, 16, stride 16)` + GELU;
  3. `ConvTranspose1d(C, K, 16, stride 16)`, producing (B, K, 256·T), cropped to length L.

  Two ways to inject the result into the waveform branch:
  - **(i) FiLM**, with the `Conditioner` accepting `cond_channels = K` (it already does);
  - **(ii) concatenation at the first encoder layer:** `channels_input = 1 + K`. The weights of `encoder.0.0` are extended with zeros for the K new channels, so the model starts identical to the pretrained branch.
- **Hypothesis H1:** a richer conditioning gives hybrid ≥ stage A + 0.03 PESQ on validation, with the gain concentrated at SNR < 10 dB.
- **Cost** ◇: ~0.03 GMAC per 2 s (against 1.7 for the original) and ~50× smaller activations.
- **Experiment:** K ∈ {1, 8} × form ∈ {FiLM, concatenation} = 4 screening runs, compared with the current hybrid (R3) and with stage A at the same LR (A5a).
- **Risks:** it no longer reproduces the CleanUNet2 paper exactly. The original upsampler stays available as `upsampler.type: original`.

#### A2. Bottleneck fusion ⚑ (with variants)
- **Motivation:** with hop 256 = 2⁸, STFT frame *j* and bottleneck token *j* advance in lockstep (see `../PLANO_CleanUNet2_Mamba.md`, section 2.1). The option already exists as `bottleneck_fusion: true` (zero-initialized FiLM).
- **Variants:**
  - (a) FiLM (already implemented);
  - (b) concatenation + `Linear(2·d → d)`, initialized as the identity on the waveform-branch half;
  - (c) **alignment shift** ∈ {−1, 0, +1} frame. With `alignment: center`, the centre of frame *j* is at 256j+128, while the receptive field of token *j* ([256j, 256j+766)) is centred at 256j+383, about one frame ahead;
  - (d) **bottleneck-only fusion**, without the waveform conditioning (`use_waveform_conditioning: false`). This would be the cheapest hybrid: ~24 ms per step, against 18 ms for the waveform-only model (section 1.3).
- **Hypothesis H2:** bottleneck fusion matches or beats the waveform conditioning, at a fraction of the cost.
- **Cost:** +0.5M parameters; negligible compute.

#### A3. Multi-scale fusion
- **Change:** besides the bottleneck, apply zero-initialized FiLM at the inputs of the next two decoder levels (stride 128 and 64), repeating the CleanSpecNet features 2× and 4×. Requires per-level hooks in `CleanUNet.forward` (`decoder_hooks`).
- **Hypothesis H3:** it mostly improves the low-SNR ranges (1.5).
- **Experiment:** A2(a) against A2(a)+A3.

#### A4. Spectral post-filter (late fusion)
- **Change:**
  1. Compute `Y = STFT(ŷ)` of the CleanUNet output.
  2. Build the inputs `[compress|Y|, Ŝ (CleanSpecNet output)]`.
  3. Pass them through a small network (conv + 2 Mamba layers over time, d=128) that produces a bounded mask `M = 1 + α·tanh(Δ)`, with Δ starting at zero (identity).
  4. The final output is `ŷ' = iSTFT(M ⊙ Y)`.
- **Hypothesis H4:** it reduces residual noise (CBAK and SSNR go up), especially stationary high-frequency noise.
- **Cost:** low compute. In streaming it adds the latency of the STFT window.
- **Risks:** iSTFT artefacts; the phase is not corrected.

#### A5. Controls (needed to interpret group A)
- **(a) Matched LR:** repeat stage A with the waveform-branch LR equal to the hybrid's (3e-5), or the hybrid with 5e-5.
- **(b) Variance between seeds:** 3 seeds of stage A (~30 min each) to estimate σ between runs.
- **(c) Shuffled condition:** the waveform branch receives the spectrogram of *another* utterance of the batch (a permutation). If PESQ stays the same as with the normal hybrid, the conditioning carries no useful information.
- **Implementation:** a `shuffle_condition` flag in CleanUNet2, used only for training and evaluating the control.

### Group B: spectral branch (CleanSpecNet)

#### B1. Residual or mask output
- **Motivation:** the `output_layer` regresses the compressed magnitude directly, with no input-to-output path.
- **Change** (`output_mode`):
  - `residual`: `Ŝ = relu(X + Δ)`, with the last layer initialized at zero (starts as the identity);
  - `mask`: `Ŝ = compress(M ⊙ |X|)`, with `M = β·σ(m)` and β ≈ 1.5;
  - `direct`: the current behaviour.
- **Hypothesis H5:** faster convergence and lower spectral L1, which improves the conditioning.
- **Experiment:** stage B (spectral branch only, 10 epochs) comparing the validation L1; the best mode moves on to the hybrid.

#### B2. Time-frequency backbone (dual-path Mamba)
- **Motivation:** today the 513 bands become channels of a `Conv1d`, and the frequency structure is lost in the first layer.
- **Change** (`backbone.type: tf_mamba`):
  1. `Conv2d` front end with 1→32 channels, reducing frequency to ~129 bands (two strides of 2).
  2. N = 4 alternating blocks: Mamba along time for each band (causal) and BiMamba along frequency for each frame (frequency need not be causal).
  3. `ConvTranspose2d` decoder back to 513 bands.
- **Hypothesis H6:** better spectral estimates, especially at low SNR. Architectures from the literature that model time and frequency explicitly (such as the MP-SENet/SEMamba family) usually do well on VoiceBank. This is a hypothesis, not a number to reproduce.
- **Cost** ◇: ~16k tokens per example per block, with d=32–64; time must be measured before scaling up.
- **Risks:** memory and time. Only after A0/A1.

#### B3. Input representation ⚑ (partial)
- `log1p` compression against power 0.3 (⚑ `spectrogram.compression: pow`).
- Compressed complex input (real and imaginary parts, 2×F channels), so the spectral branch sees the phase.
- Cheap: 2 to 3 stage-B runs.

#### B4. Separate causality and an offline bidirectional version ⚑ (partial)
- Separate the `conv_causal` and `backbone_causal` flags. This allows reproducing the original CleanUNet2 exactly (causal attention with non-causal convolutions).
- Measure the **cost of causality** in the spectral branch with `cleanunet2_mamba_e8_bidir.yaml` ⚑.

#### B5. Spectral branch with phase estimation (magnitude and phase decoders)
- Produces a full spectral estimate, which makes it possible to combine the two estimates (waveform and spectrum) with learned time-frequency weights.
- High cost; postponed until after B2.

### Group C: waveform branch (CleanUNet-Mamba)

#### C1. Causal normalization (closing the streaming gap)
- **Motivation:** 1.5 and 1.6.
- **Step 1, no training** (only in `StreamingCleanUNet`): evaluate the stage-A model in streaming with:
  - (i) the running mean of the per-frame std (the current one);
  - (ii) RMS with an exponential moving average, τ ∈ {0.5, 1, 2} s;
  - (iii) warm-up then freeze: estimate over the first 0.5 s, then fix;
  - (iv) the true cumulative std over all samples seen so far.
- **Step 2, with training:** train with the same causal normalization (`normalization.type: cumulative | ema`), or without normalization and with ±12 dB gain augmentation.
- **Success:** streaming gap ≤ 0.05 PESQ, with no SNR dependence.

#### C2. Bidirectional bottleneck (offline mode)
- **Change:** a `BiMambaMixer` in the bottleneck (the class already exists in `sequence.py`).
  - The forward branch gets the DNS weights.
  - The backward branch starts as a copy, with `out_proj` zeroed. So the model starts identical to the causal one.
  - The loader must remap `mixer.*` → `mixer.fwd.*`.
- **Hypothesis H7:** an offline gain of at least +0.05 PESQ, which quantifies the cost of causality in the waveform branch.

#### C3. Mamba at multiple scales
- **Change:** one Mamba layer in each of the two deepest decoder levels (stride 128 and 64), projecting 768 → 256 channels, with `out_proj` zeroed.
- **Cost** ◇: +3 to 5M parameters. Streaming stays possible (one token per level every 128 or 64 samples), but needs per-level state.
- **Hypothesis:** modelling long context at a finer time resolution as well helps with transients.

#### C4. Gated skip connections, or residual projection ⚑ (partial)
- **Gate:** `x + g ⊙ skip`, with `g = σ(conv1x1([x, skip]))` and an initial bias of +4 (g ≈ 1, which preserves the pretrained weights).
- **Residual projection:** `residual_projection` ⚑ already exists, but it is initialized randomly, which would break the DNS initialization. It must be initialized as the identity.

#### C5. Mamba scale ⚑
- Variations: `n_layers` ∈ {3, 5}, `d_state` ∈ {16, 64, 128}, `expand` ∈ {2, 4}, and Mamba2 ⚑.
- Waveform branch only, from scratch, 30 epochs (~40 min each). This complements the waveform-only MHA × Mamba runs of queue 2.

#### C6. Latency: E6, E7 and E8 ⚑ (partial)
- E6: 190 samples = 12 ms of lookahead; a DNS checkpoint exists (`CleanUMamba-3N-E6-high`).
- E7: 382 samples = 24 ms.
- E8: 766 samples = 48 ms.
- Goal: the PESQ × latency curve, also in streaming.

#### C7. Cheap ablations ⚑
- `glu_activation` (SiLU/GELU) and `encoder_norm: batch` (from CleanUNet_MambaSSM).
- Only when training from scratch, because the pretrained weights assume a sigmoid GLU and no normalization.

### Group D: training objectives tied to the architecture

#### D1. Consistency loss between the branches
- `L1(compress|STFT(ŷ)|, sg(Ŝ))`: the waveform output is made to agree with the spectral estimate. The symmetric version is also worth testing.
- Weights ∈ {0.1, 0.5}.

#### D2. SI-SDR term
- **Motivation:** 1.5 and 1.7. PESQ went up +0.27 while SI-SDR went down −0.37 dB.
- **Change:** add `λ·(−SI-SDR)` to the loss, with λ ∈ {0.01, 0.05, 0.1} at the appropriate scale.
- **Output of the experiment:** the PESQ × SI-SDR frontier.

#### D3. Metric discriminator (MetricGAN style)
- A network that learns to predict PESQ and is used as a discriminator.
- Risk of artefacts and instability. Low priority.

### Group E: efficiency and real-time use

#### E1. Distillation: offline hybrid → causal waveform-only model
- **Teacher:** the best offline hybrid (B4 or bidirectional C2).
- **Student:** the causal waveform-only model, E8 or E6.
- **Losses:** L1 + MR-STFT against the clean audio, plus L1 against the teacher output. Optionally, feature KD on the skips, as in CleanUMamba (`cleanumamba/src/util/util.py:259-290`).
- **Hypothesis:** the student closes at least half of the gap between the two.

#### E2. Structured pruning (port from CleanUMamba)
- The module names are already compatible (`tsfm_Mamba_layers`, `norm_f`).
- Target sizes: 5M, 2M and 1M parameters, with KD.

#### E3. Streaming of the hybrid
- `Spectrogram(alignment="causal")` ⚑, plus the Mamba `step` in CleanSpecNet and a frame-by-frame STFT buffer.
- Evaluate `n_fft` 512 (32 ms window) to contain the latency.

---

## 3. Experimental protocol

1. **Data:** training with 10,802 files, validation with 770 (p226/p287) and test with 824. Selection by validation PESQ-WB; the test set is used only once, at the end.
2. **Screening:** one seed.
   - 20 epochs when the waveform branch starts from the DNS weights; 30 epochs when the model trains from scratch.
   - A variant moves on to confirmation if it beats the paired reference (same initialization, LR and number of epochs) by ≥ 0.03 validation PESQ, or by ≥ 2σ between seeds once A5b is measured.
3. **Confirmation:** 3 seeds. Report mean ± standard deviation.
4. **Statistical tests:** per-file paired Wilcoxon test on the test set (PESQ, STOI and SI-SDR), with Holm correction.
5. **Mandatory report for each variant:**
   - parameters, GMAC per second of audio, time per step and peak memory;
   - RTF and algorithmic latency;
   - results per SNR range (the four ranges of section 1.5);
   - streaming metrics, when the variant is causal.
6. **Efficiency comparisons:** compare at an equal budget of epochs and also of GPU time.
7. **Existing tools:**
   - `evaluate.py` (per-file metrics and `summary.json`);
   - `scripts/show_metrics.py` (training curves);
   - `torch.utils.flop_counter` (FLOPs).

## 4. Roadmap and GPU budget (RTX 4090)

Reference times per epoch:
- waveform-only: ~1.3 min ✔;
- current hybrid: ~8 min ✔;
- hybrid with A0: ~3 min ◇;
- hybrid with A1 or bottleneck-only fusion: ~1.7–2 min ◇.

| Phase | Content | GPU ◇ | Depends on |
|---|---|---|---|
| 0 | A0; C1 step 1 (inference only); A5 (a, b, c); finish the current queue | ~3 h | — |
| 1 | Fusion: A1 (4 variants), A2 (b, c, d), A3 | ~6–8 h | A0, A5 |
| 2 | Spectral branch: B1, B3, B4; then B2 | ~6–10 h | Phase 1 |
| 3 | Waveform branch: C1 step 2, C2, C5, C6, C4/C7 | ~8–12 h | A5b |
| 4 | Objectives: D1, D2 (λ sweep); D3 optional | ~3–5 h | Phases 1–3 |
| 5 | Deployment: E1, E3, E2 | ~8–12 h | Phases 1–4 |
| — | 3-seed confirmation of the finalists | +~50% | — |

## 5. Where to change the code

| Proposal | Files | New configuration | Test to add |
|---|---|---|---|
| A0 | `cleanunet/cleanunet2.py` (`SpecUpsampler.forward`) | — | equivalence with `ConvTranspose2d` |
| A1 | `cleanunet2.py` (new class); `cleanunet.py` (`channels_input > 1`); `checkpoint_io.py` (extend `encoder.0.0` with zeros) | `upsampler: {type: freq_proj, channels: 64, out_channels: K}`; `conditioning_type: encoder_concat` | zero initialization ⇒ output equal to the waveform branch |
| A2 | `cleanunet2.py` (`BottleneckFusion`) | `bottleneck_fusion: {type: film\|concat, shift: 0}`; `use_waveform_conditioning` | same |
| A3 | `cleanunet.py` (`decoder_hooks`); `cleanunet2.py` | `multiscale_fusion: [256, 128, 64]` | same |
| A4 | new `cleanunet/postfilter.py` | `postfilter: {...}` | identity at initialization |
| A5c | `cleanunet2.py` | `shuffle_condition: true` | — |
| B1 | `cleanspecnet.py` | `output_mode: direct\|residual\|mask` | identity at initialization (residual) |
| B2 | new `cleanunet/tf_mamba.py` | `backbone: {type: tf_mamba, ...}` | shapes; causality along time |
| B3 | `spectral.py` | `compression: pow`; `representation: complex` | — |
| B4 | `cleanspecnet.py` | `conv_causal`, `backbone_causal` | causality |
| C1 | `streaming.py`; `cleanunet2.py` | `normalization: {type: global\|cumulative\|ema\|warmup, tau}` | streaming = offline with the same normalization |
| C2 | `sequence.py`; `checkpoint_io.py` | `bottleneck: {bidirectional: true}` | identity at initialization against the causal model |
| C3 | `cleanunet.py` | `decoder_mamba_levels: [...]` | causality; streaming |
| C4 | `cleanunet.py` | `skip_gate: true` | approximate identity at initialization |
| D1, D2 | `lightning_modules/cleanunet_module.py` | `loss.weight_consistency`, `loss.weight_sisdr` | — |
| E1 | `lightning_modules/` (frozen teacher) | `teacher: {architecture, checkpoint}` | — |
| E3 | `streaming.py` | — | streaming = offline (hybrid) |

Common principle: **every new module starts as the identity**, with the last layer zeroed or a gate close to 1. Each variant then starts exactly from the pretrained reference model, and the comparison measures the effect of the module rather than a worse initialization.

## 6. Decision points and risks

**Decisions after the current queue:**

| Queue result | Decision |
|---|---|
| Hybrid with DNS weights (R3) − stage A at the same LR < 0.03, and the A5c control equal to R3 | The current conditioning does not help. Prioritize A1/A2 and consider removing the waveform conditioning (A2d) |
| Mamba hybrid from scratch (R5) ≥ Transformer (E0), at equal or lower cost | Mamba becomes the default backbone in the next phases |
| C1, step 1, already closes the streaming gap | No need to retrain with causal normalization |

**Risks:**
- **VoiceBank is small** (9.4 h): larger variants (B2, C3) may overfit. Mitigation: start from the DNS weights, use augmentation and early stopping on validation.
- **Generalization to other data:** if the DNS test set (no reverb) is downloaded, evaluate the finalists on it without retraining.
- **Composite metrics:** the Python implementation gives values different from the published ones (section 1.1). Compare only within this repository.
