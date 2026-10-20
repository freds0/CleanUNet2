# Ablation plan: CleanSpecNet → CleanUNet conditioning and fusion

*Author: Claude (Opus 5.5) · 2026-10-13, updated 2026-10-20 · branch `feature/mamba` · details Groups A (fusion) and B (CleanSpecNet) of `IMPROVEMENTS_CLAUDE.md`*

| Marker | Meaning |
|---|---|
| ✔ | Measured in this repository |
| ◇ | Estimate, to be confirmed |
| ⚑ | Already exists in the code or the configuration |

---

## 0. Summary

**Status on 2026-10-20, 19:35.** Queue Z (14.11) finished; conclusions at the end of 14.11. With a bidirectional Mamba bottleneck in the waveform branch (Z6), the S30 soup + W5 (w 0.15) scores 3.421 on the test set (+0.046 over 3.375). Trained for 60 epochs instead of 30 (Z4), the bidirectional soup gains PESQ-WB +0.029 more on the test set: alone it scores 3.428 (one 52.4M network, offline, above every earlier blend), and with W5 (w 0.15) **3.446**, the best so far. The S30hp final with metric weight 1.0 (Z3) gains PESQ-NB +0.023 and PESQ-WB +0.036 over S30hp on the test set, with SI-SDR and STOI unchanged. The two together (Z7: bidirectional bottleneck, weight 1.0 above 100 Hz) gain PESQ-NB +0.049 over Z3 on the test set; with W5 (w 0.25) they give the highest PESQ-NB (3.913), PESQ-WB above 100 Hz (3.531) and SI-SDR (19.32 dB) measured so far, with a full-band PESQ-WB of 3.388. Rejected: output ensembles (Z1), the compressed-magnitude loss and metric weight 0.5 (Z2), W5 fine-tuned through the blend (Z5), and 60 epochs on the bidirectional weight-1.0 recipe (Z4, PESQ-NB +0.013 against the +0.015 of the rule).

**Status on 2026-10-19, 19:00.** Queue BB (14.10) finished: the S30 + W5 system retrained with the largest batch that fits the GPU scores 3.312 on the test set, against 3.375 with batch 16, so the batch-16 recipe stays. Queue Y (14.9) finished earlier the same day: training-free checks and a new set with 80 unseen speakers (vctk_unseen), which confirms the test results. Queue X (14.8) finished on 2026-10-16 at 10:53 with no failures, and the complementary analysis (14.8.2) revised the case for S30hp. Queue X covered streaming with a scale prior, DNSMOS, a listening page and the metric discriminator restricted to the band above 100 Hz (S30hp). Queue W (section 14: best system and simplest robust model) finished at 04:43: results in 14.4–14.6 and analysis in 14.7.

- **New best system (queue Z, 14.11):** the same blend with the soup of the **bidirectional S30 trained for 60 epochs** (Z4, w = 0.15) gives **PESQ 3.446** on the test set, 3.419 on val_unseen and 3.414 on vctk_unseen. It has 71.9M parameters and is offline. The soup alone gives 3.428 (52.4M); with 30 epochs (Z6), the blend gives 3.421.
- **New system for use (queue Z, 14.11):** the **bidirectional S30hp with metric weight 1.0** soup (Z7) + W5 (w = 0.25): PESQ-NB 3.913, PESQ-WB above 100 Hz 3.531 and SI-SDR 19.32 dB on the test set, the highest so far, with a full-band PESQ-WB of 3.388. It replaces S30hp + W5 (3.861 / 3.469 / 19.00 dB) and has the same size as the best system (71.9M, offline). The text below describes the systems before queue Z.
- **Best system (A):** the training-free blend of the **S30** soup with the bidirectional spectral branch **W5** (w = 0.25, chosen on val_unseen) gives **PESQ 3.375** on the test set. With the 3 S30 seeds separately, it gives **3.372 ± 0.009**.
  - That is +0.164 over RX_C3 (3.212) and +0.212 over K0extfull (3.164).
  - STOI 0.956, SI-SDR 18.85 dB, COVL 4.15. It has 60.9M parameters and runs offline only.
  - It does not depend on polarity and has the smallest failure tail: 0.4% of the files end up more than 0.1 below K0extfull.
  - **For use, the version with S30hp (14.8, X4)** gives PESQ 3.326, but has the highest PESQ without < 100 Hz (3.469) and the highest SI-SDR (19.00 dB) of all systems, plus DNSMOS OVRL 3.219 against 3.206.
- **Simple, robust model (B):** the **S30 soup** gives **3.335** on the test set (3 seeds: 3.328 ± 0.011).
  - It is a single waveform-only network with 41.4M parameters, no fusion, and streaming-compatible.
  - Recipe: K0 with external noise, plus the metric discriminator (weight 0.3), polarity inversion of the training pair (p = 0.5) and 30 epochs.
  - Caveat: the gain comes from a loss that targets PESQ. STOI, SI-SDR, segSNR and WSS do not improve (14.5).
  - **For use, the S30hp soup (14.8, X4):** the same, with the discriminator only above 100 Hz.
    - PESQ 3.280 (−0.055).
    - The same PESQ-NB as S30 (+0.001): the speech-band gain is the same.
    - SI-SDR +0.21 dB, segSNR +0.28 and WSS −0.19 over S30, with SI-SDR equal to the control's. On vctk_unseen (14.9), SI-SDR (+0.17 dB) and segSNR (+0.25) repeat for 98–100% of the 80 speakers; WSS does not (−0.01).
    - The +0.017 PESQ without < 100 Hz over S30 is the metric S30hp optimizes, and PESQ-NB does not confirm it (14.8.2).
    - DNSMOS OVRL +0.016 over S30 and half the variance between seeds.
- **Polarity (14.3):** models trained with MUSAN babble nearly stop working on inverted input (K0extfull: 3.18 → 1.82; RX_C3: 3.22 → 1.96). Inverting the training pair with p = 0.5 fixes this at no cost. Giving the babble talkers a random sign does not.
- **RX_C3 is no longer recommended.** It loses to the S30 soup by 0.124, depends on polarity, and adding it to the ensemble makes the result worse.
- **Streaming:** S30 loses 0.25 on the test set in streaming. Almost all of the loss comes from the warm-up of the causal input scale (14.7.4).
  - In the first second, the estimated scale is about 3× too small.
  - With a fixed scale or a prior, streaming of the S30 soup goes from 3.164 to 3.335 or 3.325 on val_unseen, without training. Offline gives 3.348.
  - **Confirmed on the test set (14.8, X1):** with the prior, streaming gives 3.323, against 3.335 offline and 3.085 before. The option is in `cleanunet/streaming.py` (`prior_std`, `prior_frames`).
- **Analysis (14.7):**
  - val_unseen ranks systems like the test set: Spearman 0.978 over 52 systems. The old val gives 0.66.
  - Almost 60% of the metric-discriminator gain (+0.099 of +0.170) and all of its SI-SDR loss sit below 100 Hz. The PESQ-WB input filter still lets the 50–100 Hz band through.
  - Without that band in the reference and the output, S30 gains +0.078 over the control and +0.053 over RX_C3, and the blend keeps +0.040 over S30. The ranking of the systems does not change.
- **Queue X (14.8):**
  - **DNSMOS:** it does not confirm the discriminator gain.
    - S30 − control gives OVRL −0.026 and BAK −0.037.
    - The difference comes from the < 100 Hz band: the same content that PESQ-WB rewards sounds like background noise to DNSMOS.
    - DNSMOS is at its ceiling: the control already ties with the clean reference.
  - **S30hp:** it separates the two effects of the discriminator.
    - It keeps the speech-band gain: PESQ-NB +0.067 over the control, the same as S30 (+0.066). Without < 100 Hz it gives +0.094 against +0.077, but that difference does not hold up (14.8.2).
    - It removes ~60% of the effect below 100 Hz. The rest comes from the soft cutoff of the filter.
  - **Complementary analysis (14.8.2):**
    - PESQ-NB (300–3400 Hz) confirms the discriminator's speech-band gain, the same for S30 and S30hp.
    - The test set and val_unseen have 2 speakers each. The part of the gain below 100 Hz varies 5× between the 4 speakers (0.03 to 0.15); the other effects have the same sign for all 4.
  - **Recommendation:** S30 and S30 + W5 for the benchmark number (PESQ-WB); S30hp and S30hp + W5 for use.
  - **Listening test:** the page (X5) is published and still has no answers (checked on 2026-10-19). Block B decides whether the 50–100 Hz band is audible.
- **Queue Y (14.9):**
  - **80 unseen speakers (vctk_unseen):** the numbers are within ≤ 0.03 of the test set, with the same ranking. The discriminator's PESQ-NB gain (+0.055) appears for 100% of the speakers, the same for S30 and S30hp. The blend with W5 improves every metric for 85–100% of the speakers.
  - **S30hp gains no PESQ over S30 in any band** (PESQ-NB −0.001; without < 100 Hz +0.008, for 51% of the speakers). It gains SI-SDR and segSNR.
  - **Blend weight of S30hp + W5:** w = 0.25, as for S30. **Blend with the causal branch (Bext):** only +0.011 to +0.018 on the test set, a third of W5; the streaming blend drops to low priority.
- **Queue BB (14.10):** stage A and S30 at batch 128 and W5 at batch 1024, with square-root learning-rate scaling and the same epochs.
  - Stage A is unchanged. S30 loses 0.045 on the test set, mostly the part of the discriminator gain below 100 Hz. W5 underfits with 1,100 steps: the blend gains +0.022 instead of +0.040.
  - The full system loses 0.052 to 0.063 on the three sets, with no real saving of time. Keep batch 16.

Summary of queue V (2026-10-15):

- **Answer of V1:** a training-free blend of K0extfull with Bext (w = 0.25) gives **3.209 ± 0.005** on the test set, tied with RX_C3 (3.212 ± 0.008). The learned fusion adds nothing on new noise; the recommended offline system becomes the blend.
- **Complementary analysis (12.11):** RX_C3 has a failure tail that the blend does not have: 5.9% of the files end up more than 0.1 below K0extfull, against 0.5%. This tail does not show in SI-SDR or STOI. The PESQ of single files is unstable, but the means do not change with a robust PESQ. The waveform average of RX_C3 and the blend gives **3.238** on the test set and 3.253 on val_unseen.
- **V2–V7:** condition dropout did not reduce the dependency; no streaming fusion and no distillation passed δ_min; no new spectral branch passed the threshold (the bidirectional one came closest); the `noise`-only recipe gave +0.007 on the test set; with 30 epochs K0ext improves by +0.018 and RX_C3 not at all.

Summary before queue V:

- **Best test result:** **PESQ 3.212 ± 0.008** (3 seeds) ✔, with the **RX_C3** hybrid: P3 + S5 + P8, spectral branch Bext and training with external noise (MUSAN).
  - That is +0.048 over the waveform-only control with the same training (K0extfull, 3.164) and +0.127 over the old control (K0full, 3.085).
  - The gain appears for all 3 seeds, all 5 test noises and every SNR range.
  - RX_C3 runs offline only (S5 and P8), has 53.8M parameters (against 41.4M) and a dependency of 2.06: with the condition of another file, PESQ drops by 2 points.
- **Largest single gain:** training the waveform branch with external noise gives +0.079 on the test set, at no inference cost ✔.
- **Streaming:** the only streaming-compatible fusion (P3) loses 0.013 on the test set, even with Bext ✔. The best streaming model is still waveform-only (K0ext).
- **What unlocked the fusion:** a spectral branch that generalizes to new noise. With B++, trained only on VB-DEMAND, the hybrids lost (P3: −0.074) or tied (C3: −0.014) on the test set (Phase 6) ✔.
- **Methodological lesson:** against the control, the old validation set (with the same 10 noise types as training) got the sign of the test result wrong in 4 of 5 comparisons. val_unseen (unseen noise) got all 5 right ✔.
  - Selection now uses val_unseen (section 3.4).
  - The rankings of Phases 1–4 do not hold for new noise.
- **Most important open question:** how much of the RX_C3 gain comes from the learned fusion, and how much would a simple training-free blend of the two models give? This is experiment V1 (section 13). **Answered: nothing (section 12.9).**

**Original plan (2026-10-13):**

- **Problem.** The hybrid CleanUNet2-Mamba ties with the waveform-only model on the test set: PESQ 3.096 against 3.095 ✔. The spectral branch adds 11M parameters and doubles inference time with no gain.
- **Probable cause.** The whole output of CleanSpecNet reaches CleanUNet as a **scalar gain on the waveform**. That gain is produced by an 8-parameter FiLM, which training leaves almost at zero ✔ (section 1.2).
- **Implementation finding.** CleanSpecNet itself barely cleans: the magnitude it estimates, resynthesized with the noisy phase, stays close to the noisy audio ✔ (sections 1.4 and 6.1). Without a good spectral branch, no fusion has anything to pass on.
- **Goal.** Find a form of fusion that makes the hybrid beat the waveform-only model, screened with short ~10 min trainings per variant ◇, and **improve the CleanSpecNet architecture** (track N, section 6) so that it brings useful information.
- **Method:**
  1. A **common short protocol** (section 3): it starts from the two trained branches, trains only the fusion with the branches frozen, then everything together for 5 epochs.
  2. **One axis at a time**, each phase starting from the winner of the previous one: where to inject (Phase 1), what to inject (Phase 2), how to combine (Phase 3) and how to train (Phase 4).
  3. Every variant is compared with a **waveform-only control that gets exactly the same extra training**. Other controls separate "information from CleanSpecNet" from "more parameters": noisy spectrum, shuffled condition and oracle.
  4. In parallel with the fusion, **track N** improves CleanSpecNet: first measured alone (cheap), then inside the hybrid. The best spectral branch (B++) is used from Phase 2 on.
  5. The finalists are confirmed with 3 seeds and the full protocol. The test set is used only once, at the end.
- **Cost:** ~75 runs, ~21 GPU hours in total ◇ (about three nights of queue). Almost every variant does without the `SpecUpsampler`. Without it, a hybrid step costs ~1.3× a waveform-only step ✔, instead of ~8×.

---

## 1. Starting point

### 1.1 Test results (824 files) ✔

| Model | Init | Params | PESQ-WB | STOI | SI-SDR | COVL |
|---|---|---|---|---|---|---|
| Waveform-only, Mamba E8 (stage A) | DNS | 41.4M | 3.095 | 0.953 | 18.57 | 3.90 |
| Mamba hybrid, stages A→B→C1→C2 | DNS | 52.6M | 3.096 | 0.954 | 18.57 | 3.91 |
| Mamba hybrid, single training | DNS | 52.6M | 3.089 | 0.954 | 18.55 | 3.89 |
| Waveform-only, Mamba | scratch, 30 ep. | 41.4M | 2.668 | 0.938 | 16.61 | 3.46 |
| Mamba hybrid | scratch, 30 ep. | 52.6M | 2.650 | 0.939 | 16.38 | 3.45 |
| Transformer hybrid (E0) | scratch, 30 ep. | 58.6M | 2.595 | 0.938 | 16.20 | 3.38 |

### 1.2 Why the current conditioning does not work ✔

```
|STFT| (513 x T) -> CleanSpecNet -> S_hat (513 x T)
  -> SpecUpsampler: 2x ConvTranspose2d (same weights for every band) -> mean over 513 bands -> c(t), 1 channel
  -> FiLM: x'(t) = (1 + gamma(t)) * x(t) + beta(t),  [gamma, beta] = Conv1d(1 -> 2, k=3)(c)   <- 8 parameters
  -> CleanUNet(x')
```

1. **Information bottleneck.** The mean over frequency leaves a single channel, so no band gets its own weight.
2. **Weak operator.** A scalar gain on the waveform does not filter frequencies: it only changes the volume over time.
3. **Weights near zero.** After training, the RMS of the `film_gen` weights goes from 0.005 (stage C2) to 0.06 (training from scratch), and that of the bias is 0.02. (The zero initialization itself does not block learning: the last layer gets a gradient from the 1st step, and the earlier ones soon after.)
4. **Direct evidence.** Stage C1 (only the "glue" trains, with the branches frozen) ended with a validation PESQ of 2.7548, practically the starting point (2.755).

### 1.3 Architecture facts that guide the fusion ✔

| Point of CleanUNet E8 | Stride (samples) | Rate | Channels |
|---|---|---|---|
| encoder level 1 | 2 | 8 kHz | 64 |
| encoder level 2 | 4 | 4 kHz | 128 |
| encoder level 3 | 8 | 2 kHz | 256 |
| encoder level 4 | 16 | 1 kHz | 512 |
| encoder levels 5–8 | 32–256 | 500–62.5 Hz | 768 |
| bottleneck | 256 | 62.5 Hz | 768 → 512 (3 Mamba layers) → 768 |

- **STFT hop (256) = total U-Net stride (2⁸).** STFT frame *j* and bottleneck token *j* move together. At a level of stride 2^ℓ, each frame covers 256/2^ℓ positions.
- **`spec_hidden`**, the output of the CleanSpecNet backbone, has 512 dimensions: the same `d_model` as the bottleneck.
- **Lookahead.** Token *j* sees the samples [256j, 256j+766), i.e. 48 ms. With `alignment: center`, frame *j* uses [256j−384, 256j+640).
  - Fusing frame *j* into token *j* **does not increase** the lookahead.
  - Fusing frame *j*+1 increases it by ~8 ms.
  - With `alignment: causal`, frame *j* uses only [256j−768, 256j+256).
- **Cost per step** (forward + backward, batch 2):

  | Variant | Time |
  |---|---|
  | Waveform-only | 18 ms |
  | CleanSpecNet only | 6 ms |
  | Hybrid without upsampler | 24 ms |
  | Full hybrid | 145 ms |

- **Time per epoch, including validation:** waveform-only 0.85 min; full hybrid 6.5 min; CleanSpecNet only ~0.2 min.

### 1.4 Implementation finding: the trained CleanSpecNet barely cleans ✔

Measured with `scripts/diagnose_spec_branch.py` on 30 validation files. The error is |·|^0.3 of the magnitude against the clean one, in a domain common to all models. "+ noisy phase" is the PESQ of the estimated magnitude resynthesized with the phase of the noisy audio.

| System | Magnitude error | Error at SNR ≥ 12.5 dB | PESQ (+ noisy phase) |
|---|---|---|---|
| Noisy audio | 0.137 | 0.081 | 1.57 (the signal itself) |
| Waveform branch (stage A) | **0.047** | 0.037 | 2.84 (model output) |
| CleanSpecNet of stage B (10 epochs) | 0.117 | 0.116 | 1.14 |
| Clean magnitude (oracle) | 0 | 0 | 3.31 |

In the log1p metric, the CleanSpecNets of all hybrids trained so far (C2, DNS, scratch, E0) ended between 0.058 and 0.067, against 0.064 for the noisy audio, and all give PESQ ~1.2 with the noisy phase. This means:
- **The spectral branch never had useful information to offer the waveform branch**, which explains much of the tie in section 1.1.
- At high SNR it makes the input **worse**: its error is above that of the noisy audio.

Probable causes:
- **Direct output without a shortcut from the input:** the network must rebuild the 513 bands from a 64-channel bottleneck.
- **Almost linear log1p compression:** the normalized magnitudes are small, so log1p barely compresses and the low-energy bands hardly weigh in the loss.
- **Generalization to long sequences:** in a quick test with mask output, the error grew in the stretches beyond 2 s (the length of the training crops); processed in 2 s blocks, the error dropped.

**Consequence for the plan:** D0 stopped being "train B for longer" and became a **comparison of 7 CleanSpecNet candidates** (Phase 0, section 5), and the CleanSpecNet architecture got its own track (section 6). The implemented options are `cleanspecnet_params.output_mode` (`direct`, `residual`, `mask`) and `spectrogram.compression` (`log1p`, `pow` with exponent 0.3). B+ is chosen by the PESQ of the estimated magnitude with the noisy phase, on whole utterances.

### 1.5 What the results teach about screening ✔

- **Differences below 0.02 are not reliable.** Trained from scratch, the Mamba hybrid was ahead of the waveform-only model on validation (2.284 against 2.277) and behind on the test set (2.650 against 2.668).
- **The ranking early in a training from scratch is misleading.** At epoch 10 of 30, the Mamba hybrid was 3rd of 4 on validation and finished 1st. That is why screening uses fine-tuning from trained branches, not short trainings from scratch.
- **At the end of the cosine schedule, validation is stable.** In the last 3 epochs of stage C2, the validation PESQ varied by 0.002. Using the **final** checkpoint (LR → 0), rather than the "best epoch", avoids picking the maximum of a noisy series.
- **Short trainings are valid in this protocol.** Every variant starts exactly equal to stage A, because the fusion is initialized as the identity. The difference between variants measures only what the fusion adds.
- **Variance between seeds ✔.** δ_min was 0.02 in every case.

  | Control | val | val_unseen | test |
  |---|---|---|---|
  | K0 (short protocol) | σ = 0.005 | σ = 0.001 | — |
  | K0ext (short protocol) | σ = 0.004 | σ = 0.002 | — |
  | K0full (full protocol) | σ = 0.002 | σ = 0.001 | σ = 0.007 |
  | K0extfull (full protocol) | σ = 0.004 | σ = 0.004 | σ = 0.008 |

- **The old validation set does not predict new noise ✔** (update of 2026-10-15). It has the same 10 noise types as training.
  - Over the 50 short variants, its ranking was the opposite of val_unseen's (Spearman −0.62).
  - Against the control, it got the sign of the test set wrong in 4 of 5 comparisons.
- **val_unseen gets the sign right, but not the size.** It got the sign right in all 5 comparisons, but exaggerated the losses of Phase 6: −0.16 on it, against −0.07 and −0.01 on the test set.
- **To rank hybrids against each other, both validation sets agreed with the test set:** C3 was above P3 in Phase 6 and in Phase RT.

---

## 2. Hypotheses

| # | Hypothesis | Test | Phase |
|---|---|---|---|
| H1 | The current conditioning carries no information: replacing the condition with that of another file does not change the result | D3 | 0 |
| H2 | Injecting per-channel features at the frame rate (bottleneck) makes CleanSpecNet useful: a gain over the control **and** a condition dependency > 0 | P3, P4 | 1 |
| H3 | Injecting at the fine levels (encoder, decoder, skips) helps more at low SNR, where per-band information matters | P1, P2, P6, P7, per SNR range | 1 |
| H4 | What the spectral branch *removed* (the mask Ŝ − log1p\|X\|) is more useful than the estimated clean spectrum | S2 × S3 | 2 |
| H5 | The gain grows with the quality of CleanSpecNet, and the oracle (true clean spectrum) gives the ceiling | Q1, Q2, K3 | 2 |
| H6 | A fusion that also depends on the CleanUNet content (concatenation, cross-attention) beats FiLM, which depends only on the condition | O1–O7 | 3 |
| H7 | The benefit is larger when the waveform branch is weaker (training from scratch, E6 model) | Z1, Z2 | 5 |
| H8 | The 64-channel front end of CleanSpecNet is an information bottleneck: widening it reduces the magnitude error, especially at high SNR | N1a, N1b | N |
| H9 | Modelling frequency with shared weights (dual-path TF or sub-band) gives the spectral branch a quality jump with fewer parameters | N2, N3 | N |
| H10 | A better stand-alone CleanSpecNet increases the hybrid's gain (monotonic relation between stand-alone PESQ and Δ over K0) | N-d, Q1, Q2 | N |
| H11 | Training with longer sequences removes the error drift after 2 s | N7a | N |

---

## 3. Short protocol (PC)

### 3.1 Starting point
- **Waveform branch:** stage A (`logs/stageA_cleanunet_mamba/checkpoints/best-epoch016-pesq2.755.ckpt`, validation PESQ 2.755) ✔.
- **Spectral branch: B+**, the best of the 7 D0 candidates (section 1.4), retrained for 100 epochs (~20 min ◇). Its settings (output mode and compression) apply to every variant; they are stored in `results/abl/bplus.json`.
- **Fusion initialized as the identity**, with the last layer zeroed. At step 0, every variant reproduces stage A exactly (verified by a unit test).
- **`conditioning_type: none`:** no `SpecUpsampler` and no FiLM on the waveform. The exception is the variants that use the original conditioning (F0 = K1).

### 3.2 Stages

| Stage | What trains | Length | LR | Loss |
|---|---|---|---|---|
| PC-1: fusion warm-up | only the fusion (branches frozen) | 1000 steps (~1.5 epochs) | 1e-3, 5% warm-up, cosine | waveform (L1 + MR-STFT) |
| PC-2: joint training | everything | 5 epochs (3375 steps) | fusion 1e-4; branches 2e-5 (`lr_scale` 0.2); cosine down to 1% | hybrid: waveform + 1.0 × spectral L1 of CleanSpecNet |

Batch 16, 2 s segments and the augmentation of `base.yaml`.

**Cost per variant** ◇, without the upsampler:

| Part | Time |
|---|---|
| PC-1 | ~2 min |
| PC-2 | ~6 min |
| Evaluations (normal and with the shuffled condition) | ~2 min |
| **Total** | **~10 min** |

With the original upsampler (only the F0 reference), a variant takes ~45 min, or ~15 min with A0 from `IMPROVEMENTS_CLAUDE.md`.

### 3.3 Controls

| ID | Control | Question |
|---|---|---|
| K0 | Waveform-only (stage A) with the same PC-2: LR 2e-5, 5 epochs, waveform loss. **3 seeds** | How much does the extra training alone give? What is the standard deviation between seeds (σ)? |
| K1 = F0 | Current hybrid (FiLM on the waveform), with the full PC | Where does the current conditioning stand under the same protocol? |
| K2 | Best fusion fed with the **noisy** spectrum (CleanSpecNet off) | Does the gain come from the cleaning done by CleanSpecNet, or just from a time-frequency view of the input? |
| K3 | Best fusion fed with the **true clean** spectrum (oracle, for training and evaluation) | What is the ceiling of this form of fusion? |
| — | Condition of **another file** at evaluation (no extra training) | Does the model really depend on the condition? |

### 3.4 Metrics and decision rule
- **Main metric:** validation PESQ-WB (770 files, speakers p226/p287, whole utterances) of the **final** PC-2 checkpoint.
- **Update (2026-10-15): the main metric becomes PESQ-WB on val_unseen.**
  - val_unseen uses the same 770 clean files, mixed with 5 DEMAND environments that VB-DEMAND does not use: field, hallway, park, river and washing.
  - The old validation set stays as a guard.
  - The reason is in section 1.5.
  - The test set has already been used twice (Phases 6 and RT). From now on it is kept for the final confirmation only.
- **Guard metrics:**
  - SI-SDR: may not drop by more than 0.3 dB relative to K0;
  - STOI.
- **Diagnostics, recorded for every variant:**
  - **dependency** = PESQ(normal) − PESQ(with the condition of another file, cropped or repeated to the right length);
  - PESQ per input SNR: 0, 5, 10 and 15 dB, the SNRs of the training set;
  - extra parameters, ms per step, peak memory and lookahead.
- **Threshold:** δ_min = max(0.02, 2σ), with σ measured on K0 (Phase 0).
- **Two-step promotion:**
  1. Run 1 seed. If Δ = PESQ − mean(K0) ≥ δ_min **and** the dependency ≥ 0.02, run a 2nd seed.
  2. Promote if the mean of the two seeds ≥ δ_min **and** the 95% interval of the per-file paired bootstrap (against K0) excludes zero.
- **Tie** (difference < δ_min): keep the cheapest variant in parameters, time and lookahead.
- **Multiple comparisons:** with ~50 variants evaluated on the same validation set, the largest Δ obtained by chance alone is around 2.5σ ◇. That is why nothing enters the final report without passing Phase 6.

---

## 4. Design space

**Injection points, named by their stride:**

| Name | Where it is |
|---|---|
| `enc_s{2…256}` | output of an encoder block |
| `skip_s*` | skip connection |
| `bott_pre` | after `tsfm_conv1`, before the Mamba layers; this is where the current `bottleneck_hook` acts ⚑ |
| `bott_post` | after the Mamba layers, before `tsfm_conv2` |
| `dec_s*` | input of a decoder block, after adding the skip |
| `out` | output; late fusion in the time-frequency domain |

| Axis | Options | Default (until decided) |
|---|---|---|
| **Where (P)** | input (current); fine encoder levels; `bott_pre`; `bott_post`; coarse or fine decoder; skips; `out` | decided in Phase 1 |
| **What (S)** | `spec_hidden` (512); Ŝ (513 bands); mask Ŝ − log1p\|X\|; concatenation; time-domain estimate iSTFT(Ŝ, noisy phase). Controls: noisy, oracle | `spec_hidden` |
| **How (O)** | operators O1–O9 in the table below | O2 (bounded FiLM) |
| **Generator (G)** | `Linear` only; causal trunk (Conv k=3 → GELU → Conv 1×1, 256 channels) + one head per point; larger trunk | 256-channel trunk + heads |
| **Resampling (R)** | repeat the frame; linear interpolation between frames *j*−1 and *j* (does not increase the lookahead); learned `ConvTranspose1d` | repeat |
| **Alignment (L)** | shift of −1, 0 or +1 frame; STFT `center` or `causal` | 0, `center` |
| **Training (T)** | warm-up of 0, 1000 or 3000 steps; fusion LR; gradient into CleanSpecNet (stop-gradient, weight λ_spec); initialization | 1000 steps; LR 1e-3 / 1e-4; end to end with λ_spec = 1; only the last layer zeroed |

### 4.1 Operators

Notation: *h* are the CleanUNet features at the point (B, C, T_ℓ); *c* is the condition resampled to T_ℓ; *g* is the generator; Z(·) is a layer whose weights are zero at initialization.

| ID | Operator | Formula | Starts as the identity? | Reference |
|---|---|---|---|---|
| O0 | Scalar FiLM on the waveform (current) | x' = (1+γ)·x + β, one γ and one β per sample | yes (whole generator zeroed) | CleanUNet2 |
| O1 | Per-channel FiLM | h' = (1+γ)⊙h + β, with [γ; β] = Z(g(c)) | yes | FiLM (Perez et al., 2018) |
| O2 | Bounded FiLM | h' = (1 + tanh γ)⊙h + β | yes | — |
| O3 | adaLN-Zero | h' = h + α⊙[(1+γ)⊙Norm(h) + β], with α, γ, β = Z(g(c)) | yes | DiT (Peebles & Xie, 2023) |
| O4 | Addition with a zero conv | h' = h + Z(g(c)) | yes | ControlNet (Zhang et al., 2023) |
| O5 | Concatenation + residual MLP | h' = h + Z(MLP([h; g(c)])) | yes | — |
| O6 | Gated cross-attention, at the frame rate only | h' = h + tanh(a)·Attn(q = h, kv = c of frames j−W…j), a = 0 at the start | yes | Flamingo (tanh gate); HT-Demucs (attention between branches) |
| O7 | Two-way exchange between the Mamba stacks of the two branches | at every layer: h_wave += Z(f(h_spec)) and h_spec += Z(f(h_wave)) | yes | Hybrid Demucs / HT-Demucs |
| O8 | Late fusion in the TF domain | ŷ' = iSTFT(M⊙STFT(ŷ)), with M = 1 + tanh(Z(net([log\|Y\|, Ŝ, log\|X\|]))) | yes | — |
| O9 | Gate on the skips | skip' = skip⊙(1 + tanh(Z(g(c)))) | yes | Attention U-Net (Oktay et al., 2018) |

Differences that motivate Phase 3:
- **O1 and O2** modulate *h* as a function of the condition only.
- **O5 and O6** combine *h* and *c*, so the correction can depend on what CleanUNet has already extracted (H6).
- **O7** also lets CleanSpecNet use the features of the waveform branch.

---

## 5. Phases and experiments

### Phase 0: infrastructure, diagnostics and noise floor (~1.7 h ◇)

| ID | What | Cost | Purpose |
|---|---|---|---|
| D0 | 7 CleanSpecNet candidates, 30 epochs each: output {direct, residual, mask} × compression {log1p, pow 0.3}, plus mask/pow with 4 s crops. Selection by the PESQ of the estimated magnitude + noisy phase (whole validation set). The winner is retrained for 100 epochs (B+) | ~1 h | Spectral branch used throughout the plan (section 1.4) |
| D1 | Quality of the spectral branch alone: iSTFT(expm1(Ŝ) · noisy phase) for B and B+. Ceiling: clean magnitude with noisy phase | ~5 min, no training | Places CleanSpecNet between the noisy audio and the oracle |
| D2 | Complementarity. (a) Log-magnitude error per band and per SNR for the noisy audio, Ŝ and \|STFT(ŷ_A)\|. (b) "Best of the two in each bin" ceiling, with the phase of ŷ_A. (c) Training-free blend exp((1−w)·log\|Y_A\| + w·log\|Ŝ\|), w ∈ {0.25, 0.5} | ~10 min, no training | If Ŝ is not better than the waveform-branch output in any region, fusion has little to add. Informs P8 |
| D3 | Dependency of the hybrids already trained (C2, DNS, scratch): evaluate with the shuffled condition and with the fusion turned off | ~5 min | Tests H1 (expected: ΔPESQ ≈ 0) |
| K0 | Waveform-only control with PC-2, seeds 1, 2 and 3 | 3 × ~5 min | σ and δ_min |
| K1 | F0 (current hybrid) with the full PC | ~45 min (~15 with A0) | Reference |
| I0 | Sanity: at step 0, every new variant reproduces stage A | unit tests | Guarantees a fair comparison |

### Phase 1: where to inject (~2 h ◇)

Fixed in this phase: signal `spec_hidden`, operator O2, 256-channel trunk, frame repetition, shift 0.

| ID | Points | Extra params ◇ | Hypothesis or reason |
|---|---|---|---|
| P1 | `enc_s2` | ~0.5M | The 1st encoder conv is a learned filter bank: a per-channel gain there amounts to a per-band gain |
| P2 | `enc_s2`, `s4`, `s8`, `s16` | ~1.0M | H3 |
| P3 | `bott_pre` ⚑ (the current `BottleneckFusion`, without the waveform FiLM) | ~0.7M | H2; the cheapest variant |
| P4 | `bott_post` | ~0.7M | The information enters after the temporal modelling and goes straight to the decoder |
| P5 | `dec_s256`, `s128`, `s64`, `s32` | ~2.0M | Decoder, coarse levels |
| P6 | `dec_s16`, `s8`, `s4`, `s2` | ~1.0M | H3; the levels that rebuild fine detail |
| P7 | Gate on the 8 skips (O9) | ~1.5M | Noise leaks through the skips, which the bottleneck does not control |
| P8 | `out`: late TF fusion (O8) | ~1.5M | Uses the complementarity measured in D2. The STFT/iSTFT raises the total lookahead to ~112 ms (1790 samples), so it is offline only |
| P9 | Combination of the 2 best among P1 to P8 | — | Do the gains add up? |

### Phase 2: what to inject (~2 h ◇), at the best point P\*

| ID | Condition | Question |
|---|---|---|
| S1 | `spec_hidden` (winner of Phase 1) | — |
| S2 | Ŝ (513 bands, with a learned projection) | Internal features or the output of CleanSpecNet? |
| S3 | Mask m = Ŝ − log1p\|X\| | H4 |
| S4 | [`spec_hidden`; m] | Do the two complement each other? |
| S5 | Time-domain estimate, iSTFT(CleanSpecNet magnitude · e^{j∠X}), injected at the 1st encoder level through its own strided conv with a zeroed output. It is not a 2nd channel of `encoder.0.0`, so that it can train in PC-1 with CleanUNet frozen | Does a physical signal at the sample rate work better? The iSTFT raises the total lookahead to ~112 ms |
| K2 | log1p\|X\| (noisy, CleanSpecNet off) | Control: just the time-frequency view |
| K3 | Clean log1p\|S\| (oracle), same projection as S2 | Ceiling (H5) |
| Q1 | S1 with the CleanSpecNet of stage B (10 epochs) instead of B+ | Gain as a function of the spectral-branch quality (H5) |
| Q2 | S1 with the bidirectional CleanSpecNet (`cleanunet2_mamba_e8_bidir` ⚑, trained like B+) | Practical offline ceiling; +~20 min to train the branch |

If track N (section 6) runs first, Phase 2 uses B++ instead of B+, and Q1 and Q2 are folded into N-d.

### Phase 3: how to combine (~2.5 h ◇), with P\* and S\*

| ID | Variant | Condition to run |
|---|---|---|
| O1 | Per-channel FiLM, unbounded | — |
| O3 | adaLN-Zero | — |
| O4 | Addition with a zero conv | — |
| O5 | Concatenation + residual MLP | — |
| O6 | Gated causal cross-attention, window W = 8 frames (128 ms) | P\* at the frame rate |
| O7 | Two-way exchange between the Mamba stacks (exploratory; ~150 lines of code) | P\* in the bottleneck |
| G1 | `Linear`-only generator | — |
| G3 | 512-channel trunk, k = 5 | — |
| R2 | Linear interpolation between frames *j*−1 and *j* | P\* with fine levels (risk of audible modulation at 62.5 Hz) |
| R3 | Learned depthwise `ConvTranspose1d` | P\* with fine levels |
| L−1, L+1 | Shift of −1 or +1 frame (+1 adds ~8 ms of lookahead) | P\* at the frame rate |
| Lc | `causal` STFT | Measures the cost of making the spectral branch streaming-ready |

### Phase 4: how to train the fusion (~1.3 h ◇)

| ID | Variant | Question |
|---|---|---|
| T1 | No warm-up (straight to PC-2) | Is PC-1 needed? |
| T2 | 3000-step warm-up | Does more warm-up help? |
| T3 | Fusion LR of 5e-4 in PC-2 | Does the fusion learn too slowly? |
| T4 | Stop-gradient on the condition: CleanSpecNet trains only through the spectral loss | Does the waveform gradient help or hurt CleanSpecNet? |
| T5 | λ_spec = 0: CleanSpecNet becomes a free feature extractor | Is the spectral loss needed? |
| T6 | λ_spec = 0.3 | Intermediate point |
| T7 | Last layer with a small random initialization (N(0, 10⁻³)) instead of zero (`zero_init: small`) | Does an immediate gradient into the trunk speed up learning of the fusion? |
| T8 | Oracle curriculum in PC-1: use the clean condition with probability p, from 0.5 to 0 (exploratory) | Teaches the fusion to read the condition before the real condition gets noisy |

### Phase 5: regimes with a weaker waveform branch (~2 h ◇)

| ID | Experiment | Cost ◇ | Note |
|---|---|---|---|
| Z1 | From scratch, 30 epochs: best fusion × waveform-only (✔ val 2.277 / test 2.668) × F0 hybrid (✔ val 2.284 / test 2.650) | ~40 min (1.2 min/epoch) | Deliberately long training, because rankings early in a training from scratch are misleading (1.5) |
| Z2 | E6 (27.2M parameters, 12 ms lookahead): stage A-E6 from the DNS E6 (20 epochs), then PC with the best fusion × PC control | ~40 min | Needs the `causal` STFT (lookahead ≤ 16 ms). With the centred STFT, the lookahead would go from 12 to ~40 ms |
| Z3 | (Optional) lightweight model `cleanunet_mamba_light` from scratch, with and without fusion | ~30 min | Check the compatibility between the hop and the total stride first |

### Phase 6: confirmation (~3 h ◇)

- **Who:** the 2 best configurations and K0.
- **Full protocol:** stage C1 (2000 steps) + stage C2 (10 epochs) from A + B+, with **3 seeds** each.
- **Test set** (824 files), only once:
  - mean ± standard deviation between seeds;
  - per-file paired Wilcoxon test, with Holm correction;
  - results per SNR range;
  - condition dependency;
  - parameters, RTF and lookahead.
- **If the variant is causal:** record the total lookahead. Streaming of the hybrid does not exist yet (E3 of `IMPROVEMENTS_CLAUDE.md`).

### Phases added during execution (C, U, R and RT) ✔

| Phase | What | Why |
|---|---|---|
| C | Offline combinations on top of P3, in the short protocol: C1 = P3 + S5, C2 = P3 + P8, C3 = P3 + S5 + P8 | S5 and P8 tied with P3 (Phase 2 and N-x). Do the gains add up? |
| 6 with `--finalists auto` | P3 + B++ and the best offline configuration of Phase C (C3), with 3 seeds, the full protocol and the test set | Confirmation |
| U | Every run with a validation result is re-evaluated, without training, on **val_unseen** (`scripts/make_val_unseen.py`): speakers p226/p287 mixed with the 5 DEMAND environments VB-DEMAND does not use, at the test SNRs | The validation gains did not show up on the test set |
| R | Spectral branch trained with **external noise**, then K0ext and the RX hybrids with the same noise (details below) | Does a spectral branch that generalizes make the fusion generalize? |
| RT | RX_S5P8 (S5 + P8 without P3) in the short protocol; then the full protocol of Phase 6, with 3 seeds, for K0extfull, RX_P3 and RX_C3, evaluated on test, val and val_unseen | Confirmation of Phase R on the test set |

**External noise (`noise_bank.py`, option `data.ext_noise`):**
- With probability p, the noise of the crop is replaced by an excerpt from MUSAN (CC BY 4.0).
- The SNR is drawn between −5 and 20 dB, with the speech level measured per P.56.
- The excerpt comes from three sources, with weights 0.6 / 0.25 / 0.15:
  - `noise`: 6.2 h;
  - babble of 3 to 7 talkers, built from `speech` (60 h);
  - `music`: 43 h.
- The DEMAND environments are left out: both the 5 of the test set and the 5 of val_unseen.

**Phase R, step by step:**
1. Train two spectral branches, with p = 0.5 and p = 0.8, and pick by the mean stand-alone PESQ on val and val_unseen.
2. Retrain the winner for 100 epochs. This is **Bext**.
3. Train K0ext: K0 with the same noise, 3 seeds.
4. Train the hybrids RX_P3, RX_S5 and RX_C3, with Bext and the same noise.
5. Train RX_P3vb: with Bext, but only with the VB-DEMAND data.

---

## 6. Track N: CleanSpecNet architecture

D1 (section 1.4) showed that the fusion has nothing to pass on as long as CleanSpecNet does not clean. This track improves the spectral branch **itself**. Each variant is first measured alone (~10 min of training and ~2 min of evaluation), and only the best ones go into the hybrid.

**Status on 2026-10-13:** N0–N9 implemented and tested (`tests/test_specnet.py`, 28 tests). N10 was not implemented, because it depends on downloading the DNS data. The track runs automatically with `scripts/run_ablation.py --phases N Nd` (section 6.5).

**Update on 2026-10-15:**
- The track ran on 10/13, and the results are in section 12.2. B++ is N1b with 100 epochs.
- The spectral branch with external noise (Bext, Phase R) uses the same architecture.
- N10 is still not implemented. The external noise came from MUSAN, not from DNS.

### 6.1 Diagnosis of the current architecture ✔

```
compressed |X| (513 x T)
  -> input_layer: Conv1d(513 -> 513, k=3)                                     0.79M
  -> 5 blocks: Conv1d(-> 64, k=4) -> ReLU -> Conv1d(64 -> 128, k=4) -> GLU     0.36M   <- 64 channels
  -> Linear(64 -> 512)                                                        0.03M
  -> 5 Mamba layers (d = 512, causal)                                         9.22M (82%)
  -> output_layer: Conv1d(512 -> 513, k=3)                                    0.79M
  = 11.2M parameters (Mamba version)
```

| Problem | Evidence | Consequence |
|---|---|---|
| **64-channel bottleneck before the temporal modelling.** Each 513-band frame becomes 64 numbers in the 1st block; the 512-dimensional backbone receives a projection of rank ≤ 64 | code ✔ | The fine harmonic structure does not get through: at 15.6 Hz per band, the harmonics of a voice with F0 of 100–300 Hz are every ~6–20 bands. The output tends to a smooth envelope |
| **Frequency treated as channels.** There are no weights shared along frequency: a pattern (harmonic, formant) has to be learned at each position separately | code ✔ | Poor data efficiency, on a corpus of only 9.4 h |
| **82% of the parameters in the backbone**, which only receives the 64-dimensional projection | count ✔ | Badly distributed capacity |
| **Direct output** and **almost linear log1p compression** | section 1.4 ✔ | Handled by D0 (`residual`/`mask` modes, `pow` compression) |
| **Magnitude only**, no phase at the input or the output | code ✔ | The branch cannot produce a complete signal on its own; the noisy phase limits S5 and P8 |
| **Drift on long sequences** (mask variant, 3 epochs) | section 1.4 ✔ | The error grows after 2 s, the length of the training crops |
| **64 ms window** (n_fft 1024) | configuration ✔ | Good frequency resolution, but it smears transients in time (pre-echo) |

**First result of D0** (30 validation files). It is the same CleanSpecNet, with LR 5e-4 and 30 epochs instead of 1e-4 and 10:

| System | Magnitude error | PESQ | STOI | SI-SDR |
|---|---|---|---|---|
| Noisy audio | 0.137 | 1.57 | 0.858 | 6.87 |
| Old CleanSpecNet B + noisy phase | 0.117 | 1.14 | 0.674 | −3.13 |
| D0 `direct_log1p` + noisy phase | 0.093 | 1.79 | 0.826 | 6.15 |
| Waveform branch (stage A) | 0.047 | 2.84 | 0.928 | 17.71 |
| Clean magnitude + noisy phase (ceiling) | 0 | 3.31 | 0.974 | 18.93 |

More training helps, but the result stays close to the noisy audio, with STOI and SI-SDR even below it. The problem is structural, not just optimization.

### 6.2 Proposals

| ID | Proposal | Change | Hypothesis | Implementation | Training ◇ |
|---|---|---|---|---|---|
| N0 | Output mode and compression | `direct`/`residual`/`mask` × `log1p`/`pow` | Input shortcut + effective compression | ✔ done (D0, running) | ~10 min per candidate |
| N1a | Wider front end | `conv_hidden_dim` 64 → 256 and `num_conv_layers` 5 → 2 | H8 | ⚑ configuration only | ~10 min |
| N1b | Per-frame encoder | Replace the conv blocks with `Linear(513 → 512)` + a causal depthwise conv over time | H8, simpler version | ✔ `frontend: linear` | ~10 min |
| N2 | **Dual-path time-frequency Mamba** | 2-D encoder (B, C, F, T) with a 4× reduction in frequency; 4 blocks with Mamba **causal in time** (per band) and Mamba **bidirectional in frequency** (per frame, which does not break causality); 2-D decoder with skips from the encoder and a bounded mask. 16 channels at full resolution and 64 at the reduced levels | H9; 0.98M parameters ✔ | ✔ `cleanunet/tf_mamba.py` | ~2 min/epoch ✔ (~60 min) |
| N3 | Full band + sub-band | Current model (with N1) + a small model shared across bands, which sees band *f* ± *k* and the full-band output; the output is a correction of the full-band estimate, zero at the start | H9, lighter alternative | ✔ `cleanunet/subband.py` | ~1 min/epoch ✔ (~30 min) |
| N4 | Output with phase | Magnitude decoder (mask) + phase decoder (atan2 of two outputs). Losses: magnitude, complex spectrum, anti-wrapping phase (instantaneous, group delay and instantaneous frequency, weighted by the clean magnitude) and consistency | The branch produces a complete signal; S5 uses the predicted phase | ✔ `predict_phase: true` + `loss.spec_terms` | — |
| N5 | Input with phase | [\|X\|^0.3, cos∠X, sin∠X] | The noisy phase carries temporal cues (group delay, instantaneous frequency) | ✔ `input_features: mag_phase`. In the 1-D model, the 1st conv gets 3 × 513 channels: 19.4M parameters instead of 11.2M | — |
| N6a | 32 ms window | n_fft = win = 512 and hop 256 (257 bands): still 1 frame per token | Less pre-echo and fewer bands (cheaper). The frame looks only 384 samples ahead, instead of 640 | ⚑ configuration only | ~8 min |
| N6b | Hop of 128 | n_fft 512 and hop 128; the fusion aggregates 2 frames per token | Better time resolution in the spectral branch | ✔ mean of the frames within each position (`fusion.py`) | ~15 min |
| N7a | Long sequences | 6 s crops made of 3 concatenated utterances (10 ms crossfade); 10 epochs with batch 6, ≈ the same amount of audio as 30 epochs of 2 s crops | H11 | ✔ `data.concat_utterances` | ~15 min |
| N7b | Causal input normalization | Divide the magnitude by its running mean over time (mean over frequency), as in FullSubNet | Statistics stable over time | ✔ `input_norm: cumulative` | ~10 min |
| N8a | Time-domain loss | L1 + MR-STFT of iSTFT(\|Ŝ\|·e^{j∠X}) against the clean signal (weight 0.2), added to the spectral L1 | Directly optimizes what D1 measures | ✔ `loss.spec_terms: {l1: 1, wave: 0.2}` | ~12 min |
| N8b | Complex and consistency losses | Together with N4: `{l1: 1, phase: 0.3, complex: 0.1, consistency: 0.1}` (+ `wave: 0.2` if N8a helps) | — | ✔ | — |
| N9 | Scale | Two points per architecture: 1-D (8 layers; d = 256), TF (6 blocks; 96 channels), sub-band (3 layers; d = 64) | Redistribute capacity after N1/N2 | ✔ `--with_nc` | 2 × 10–60 min |
| N10 | More data (optional) | Pretrain CleanSpecNet on the DNS Challenge, where the waveform branch already came from | The data asymmetry between the branches (hundreds of hours × 9.4 h) limits the spectral branch | large download ◇ | hours |

References for N2–N5: SEMamba (Chao et al., SLT 2024), MP-SENet (Lu et al., Interspeech 2023), FullSubNet (Hao et al., ICASSP 2021), DPCRN (Le et al., Interspeech 2021) and TF-GridNet (Wang et al., 2023).

**Design notes:**
- **Causality.** In N2 and N3, frequency can be bidirectional, because it only uses the current frame. Time stays causal, so the branch remains streaming-compatible. The causality test (perturb future frames and check the past ones) applies to every variant.
- **Interface with the fusion.** A 2-D variant (N2) needs one vector per frame for `spec_hidden`. Proposal ◇: mean over ~16 sub-bands and a projection from C·16 to 512. The `spec` and `mask` signals work unchanged.
- **Alignment.** N6a keeps hop 256, i.e. 1 frame per token. N6b requires aggregating frames in the fusion.

### 6.3 Track protocol (PN)

1. **Stand-alone training** of CleanSpecNet, as in stage B: 30 epochs, LR 5e-4 with cosine decay, spectral loss (plus the N8 terms where relevant) and the output and compression settings of B+.
2. **Stand-alone evaluation** on the 770 validation files (~2 min), with `scripts/diagnose_spec_branch.py`:
   - PESQ, STOI and SI-SDR of the estimated magnitude with the noisy phase (for N4, also with the predicted phase);
   - magnitude error per band and per SNR range;
   - **drift**: error after 4 s divided by the error in the first 2 s, on the files longer than 4 s;
   - **complementarity** with stage A: the energy-weighted fraction of bins where Ŝ is closer to the clean signal than \|STFT(ŷ_A)\|, and the "best of the two" ceiling.
3. **Promotion:** the 2 best by stand-alone PESQ move on to the hybrid test. Tie: complementarity decides, then cost.
4. **Hybrid test (N-d):** each promoted CleanSpecNet goes into the winning fusion of Phase 1, with the short protocol of section 3, and is compared with K0 and with the same fusion using B+. **This is the final criterion**: a CleanSpecNet that is better alone only matters if it increases the hybrid's gain.

**Targets on validation** ◇:

| Measure | Best so far | Target |
|---|---|---|
| Stand-alone PESQ (+ noisy phase) | 1.79 | ≥ 2.3 |
| Magnitude error per SNR range | above the noisy audio at high SNR | below the noisy audio in every range |
| Drift | > 1 (mask, 3 epochs) | ≤ 1.1 |
| "Best of the two" ceiling | 3.13 with the old B (30 files) | ≥ waveform branch + 0.2 |
| Hybrid gain over K0 | — | ≥ δ_min, with dependency > 0.02 |

### 6.4 Stages

| Stage | Variants | GPU ◇ | Depends on |
|---|---|---|---|
| N-a: configuration and small changes | N1a, N1b, N6a, N7a, N7b, N8a, one factor at a time on top of B+ | ~1.5 h | B+ (Phase 0) |
| N-b: new architectures | N2 and N3 with the best options of N-a; then N4 + N5 on the better of the two | ~3 h | N-a |
| N-c: scale and data (optional) | N9; N10 if DNS data is available | ~0.5 h (+ N10) | N-b |
| N-d: hybrid test | The 2 best CleanSpecNets in the winning fusion of Phase 1 (short protocol); repeat the 2–3 best points of Phase 1 with the new branch | ~1 h | Phase 1 and N-b |

**Position in the schedule:** after Phase 1 and before Phase 2. The best CleanSpecNet of the track (**B++**) becomes the spectral branch of Phases 2–6, and Q1 and Q2 of Phase 2 are folded into N-d.

### 6.5 Automatic execution (`scripts/run_ablation.py --phases N Nd`)

| Step | Rule |
|---|---|
| Reference | The winning D0 candidate with the **same budget** as the N variants (30 epochs), evaluated by the same script |
| N-a | Each option is trained alone on top of B+. An option is **kept** if the stand-alone PESQ rises by ≥ 0.05 over the reference |
| N-b | If 2 or more options are kept, train the 1-D model with all of them (`N_a_combo`). N2 and N3 get the generic options that were kept (N6a, N7a, N7b, N8a); N3 also gets the best front end. N4 and N4+N5 are trained on the best architecture so far |
| N-c | Only with `--with_nc`: the two scale variants of the best architecture |
| B++ | The best variant, if it beats the reference by ≥ 0.05, is retrained with 10/3 of its budget (like B+: 30 → 100 epochs) and stored in `results/abl/bplusplus.json`. Otherwise B++ is B+ itself |
| N-d | The 3 best points of Phase 1 with B++ (`--winner auto` picks the winner by validation PESQ), plus the 2nd best spectral branch (30-epoch checkpoint, no retraining) at the winning point. Comparison with K0 and with the same points using B+ in `results/abl/phaseNd.md` |

Stand-alone results are in `results/abl/N/<variant>/diagnostics.json`, with PESQ, STOI and SI-SDR, error per band and per SNR range, drift and complementarity.

---

## 7. Decision tree

| Observation | Conclusion | Next step |
|---|---|---|
| D3: dependency ≈ 0 in the current hybrids | H1 confirmed | Go on to Phase 1 |
| Some Phase-1 variant ≥ δ_min, with dependency > 0 | The right fusion works | Phases 2–4 refine the winner |
| No variant passes, but the K3 oracle gives ≥ +0.1 | The limit is the quality of CleanSpecNet, not the fusion | Track N (section 6), then repeat the best points of Phase 1 and Phase 2 with B++ |
| No variant passes and K3 gives < +0.05 | On this dataset, the waveform branch (E8 + DNS) already extracts that information by itself | Phase 5. If it also fails: report the negative result and simplify (remove the spectral branch, or use it only as a teacher, E1) |
| K2 (noisy spectrum) ≈ best variant | The gain comes from the time-frequency view, not from the cleaning done by CleanSpecNet | Replace CleanSpecNet with a light TF encoder |
| Gain only at low SNR | Useful, but VB-DEMAND has little low-SNR material | Also evaluate on harder data (DNS test set) |
| B+ already reaches stand-alone PESQ ≥ 2.3 | D0 solved the essentials | Reduced track N: N-d and, if time allows, N2 |
| Stand-alone PESQ rises a lot (N2/N3), but the hybrid gain stays < δ_min | The spectral branch got good, but brings no new information or the fusion does not use it | Look at the complementarity (D2). If high: Phase 3 (operators) and P8. If low: the waveform branch already knows the same |
| No track-N variant passes a stand-alone PESQ of ~2 | The limit lies outside the architecture | Revisit STFT, normalization and loss (N6–N8) and check the pipeline before trying larger architectures |
| The validation gains do not show up on the test set (Phase 6) ✔ | The validation set has the same noises as training, and the fusion over-trusts a spectral branch that only saw VB-DEMAND | Validate with unseen noise (Phase U); train the spectral branch and the hybrid with external noise (Phase R) |
| With external noise, the hybrid gains on both validation sets ✔ | Fusion works when the spectral branch generalizes | Confirm on the test set (Phase RT) |
| The training-free blend K0ext + Bext ties with RX_C3 (V1) | The learned fusion is not justified | Offline system = K0ext + Bext + blend. Fusion research only for what the blend cannot do: streaming and distillation (V6) |
| RX_C3 beats the blend by ≥ δ_min on val_unseen (V1) | The learned fusion adds something | Continue with RX_C3: condition dropout (V2) and a better spectral branch (V4) |

**Path taken (10/13 to 10/15):**
1. D3 showed dependency ≈ 0 in the old hybrids, which confirmed H1.
2. Phase 1, with B+, gave gains below δ_min.
3. Track N produced B++.
4. N-d gave +0.06 on validation.
5. Phases 2–4 gave only ties.
6. Phase C reached +0.12 on validation, with C3.
7. Phase 6 showed that this gain does not transfer to the test set.
8. Phase U built and applied the validation set with unseen noise.
9. Phase R trained with external noise.
10. Phase RT confirmed RX_C3 on the test set: +0.048 over K0extfull.

---

## 8. Implementation

**Status on 2026-10-15:**
- All phases are implemented and have been run: 0–6, N, N-d, N-x, C, U, R and RT.
- The test suite collects 115 tests:

  | File | Tests |
  |---|---|
  | `test_fusion.py` | 36 |
  | `test_specnet.py` | 30 |
  | `test_models.py` | 29 |
  | `test_cleanumamba_compat.py` | 7 |
  | `test_training_pipeline.py` | 6 |
  | `test_noise_bank.py` | 4 |
  | `test_val_unseen.py` | 3 |

- Nothing had been committed on `feature/mamba` yet.

(Until 2026-10-13, Phases 0 and 1 were in the queue and track N started automatically after them.)

| To | Command |
|---|---|
| Run Phases 0 and 1 (already running) | `setsid nohup python scripts/run_ablation.py --phases 0 1 > logs/abl/queue.log 2>&1 &` |
| Run Phases 2–4 after choosing the winner | `python scripts/run_ablation.py --phases 2 3 4 --winner P3_bott_pre [--signal mask] [--operator adaln_zero]` |
| Run track N and the hybrid test | `python scripts/run_ablation.py --phases N Nd --winner auto [--with_nc]` |
| Run Phase 6 | `python scripts/run_ablation.py --phases 6 --finalists <id1> <id2>` (or `--finalists auto --winner P3_bott_pre`) |
| Run the offline combinations | `python scripts/run_ablation.py --phases C --winner P3_bott_pre` |
| Re-evaluate everything on val_unseen | `python scripts/run_ablation.py --phases U` (reports `results/abl/U_*.md`) |
| Run the external-noise training and its confirmation | `python scripts/run_ablation.py --phases R RT --winner P3_bott_pre` (reports `results/abl/phaseR*.md`, `phaseRT*.md`) |
| Generate val_unseen | `python scripts/make_val_unseen.py` (writes `val_unseen/` in the dataset directory and `filelists/val_unseen.csv`) |
| See the commands without running them | add `--dry_run` |
| Compare with K0 | `python scripts/compare_runs.py --ref results/abl/K0_s{1,2,3} --runs 'results/abl/P*'` (the queue already writes `results/abl/phase<N>.md`) |
| Diagnose spectral branches | `python scripts/diagnose_spec_branch.py --wave_ckpt <A> --spec_ckpt name=<ckpt> ...` |
| Evaluate the condition dependency | `python evaluate.py --checkpoint <ckpt> --filelist filelists/val.csv --fast --condition shuffle\|off` |

Where the results are:
- checkpoints and logs in `logs/abl/<id>_pc1|_pc2/`;
- evaluations in `results/abl/<id>/val` and `val_shuffle`, with the variant definition in `variant.json`;
- the queue is restartable: completed steps are skipped and interrupted trainings resume from `last.ckpt`.

### 8.1 Files (implemented)

| File | Change |
|---|---|
| `cleanunet/fusion.py` (new) | `CondTrunk` (causal trunk at the frame rate); `FusionPoint` with the operators `film`, `film_tanh`, `adaln_zero`, `add`, `gate`, `concat_mlp`, `xattn`; resampling `repeat`/`linear`/`convT` and `shift`; `FusionHub` (per-point hooks); `InputEstimateFusion` (S5); `BranchExchange` (O7, Mamba stacks in lockstep) |
| `cleanunet/postfilter.py` (new) | O8: bounded mask in the STFT domain on the CleanUNet output |
| `cleanunet/cleanunet.py` | `fusion_points()`; `forward(..., hooks=None, bottleneck_runner=None)` with hooks at `enc_s*`, `skip_s*`, `bott_pre`, `bott_post`, `dec_s*`. `bottleneck_hook` remains (= `bott_pre`) ⚑ |
| `cleanunet/cleanspecnet.py` | `output_mode: direct \| residual \| mask`; `forward` split into `encode()` and `head()` (needed for O7) |
| `cleanunet/spectral.py` | `complex_stft`, `istft` (own overlap-add, inverts the STFT with an error < 1e-6) and `decompress` |
| `cleanunet/cleanunet2.py` | `conditioning_type: none` (no upsampler and no waveform FiLM); `fusion:` block; signals `spec_hidden`, `spec`, `mask`, `noisy_spec`, `oracle_spec`, `oracle_mask` (or a list); `forward(..., clean=, cond_noisy=, cond_clean=, fusion_enabled=)`; `oracle_prob` for the T8 curriculum |
| `lightning_modules/cleanunet_module.py` | Passes `clean` to the model; spectral loss only when CleanSpecNet runs; `fusion_curriculum: {oracle_prob, steps}`. `lr_scale` already accepts `fusion_hub`, `postfilter`, `input_estimate`, `exchange` ⚑ |
| `config_utils.py` | Overrides `model.architecture.<key>=...` applied after resolving the architecture YAML |
| `evaluate.py` | `--condition {normal,shuffle,off}`; `--fast`; architecture read from the Lightning checkpoint; per-file input SNR in `metrics.csv` and a summary per SNR range |
| `scripts/run_ablation.py` (new) | Restartable queue of Phases 0–6 (waits for free GPU memory, resumes after OOM) |
| `scripts/diagnose_spec_branch.py` (new) | D0 (choice of B+), D1 and D2 |
| `scripts/compare_runs.py` (new) | Per-file paired Δ against K0, 95% bootstrap, Wilcoxon + Holm, dependency, Δ per SNR range |
| `configs/model/fusion/*.yaml` | `base.yaml`, `base_e6.yaml`, `F0_waveform_film`, `P1`–`P8`, `S5_input_estimate`, `O7_exchange` |
| `configs/ablation/*.yaml` | `stageB_plus` (D0/B+), `pc1`, `pc2`, `k0` |
| `tests/test_fusion.py` (new) | Identity at the start for 21 variants; gradient into the fusion at the 1st step; 766-sample lookahead preserved with the E8 geometry; oracle and condition from another file; STFT inversion; loading of the pretrained branches |
| `noise_bank.py` (new) | P.56 active speech level (`active_speech_power`), `mix_at_snr` and `NoiseBank`: excerpts of MUSAN `noise`, babble and `music`, with weights |
| `dataset.py`, `lightning_modules/data_module.py` | `data.ext_noise: {dir, p, snr_db, weights, babble_talkers}` (training only; off by default) |
| `scripts/make_val_unseen.py` (new) | val_unseen: the validation speakers with the 5 DEMAND environments VB-DEMAND does not use, at the test SNRs (P.56, seed 0) |
| `scripts/run_ablation.py` | Phases C, U, R and RT; `--finalists auto`; `variant(..., splits=...)`; tables of mean ± standard deviation per configuration (`phase*_seeds*.md`) |
| `tests/test_noise_bank.py`, `tests/test_val_unseen.py` (new) | The P.56 active speech level ignores pauses; mixing at the requested SNR; `NoiseBank` draws every source at the requested length and refuses a missing source; the dataset replaces the noise only in training crops; Spearman of Phase U |

### 8.2 Configuration: each variant is just a YAML

```yaml
# configs/model/fusion/p3_hidden_film.yaml
base: ../cleanunet2_mamba_e8.yaml
conditioning_type: none            # no SpecUpsampler, no waveform FiLM
fusion:
  signal: spec_hidden              # spec_hidden | spec | mask | noisy_spec | oracle_spec | [spec_hidden, mask]
  trunk: {channels: 256, kernel: 3, layers: 2, causal: true}
  resample: repeat                 # repeat | linear (frames j-1..j) | convT
  shift: 0                         # frames; +1 adds ~8 ms of lookahead
  zero_init: last                  # last | small
  points:
    bott_pre: {op: film_tanh}      # film | film_tanh | adaln_zero | add | concat_mlp | xattn | gate
```

```yaml
# configs/ablation/pc1.yaml - short protocol, step 1: fusion warm-up with both branches frozen
base: ../base.yaml
trainer: {max_steps: 1000, max_epochs: -1}
model:
  init:
    cleanunet_checkpoint: logs/stageA_cleanunet_mamba/checkpoints/best-epoch016-pesq2.755.ckpt
    cleanspecnet_checkpoint: logs/abl/Bplus/last.ckpt
  train_cleanunet: false
  train_cleanspecnet: false
loss: {mode: waveform}
optim: {lr: 1.0e-3}
```

```yaml
# configs/ablation/pc2.yaml - short protocol, step 2: joint fine-tuning (init = last.ckpt of PC-1)
base: ../base.yaml
trainer: {max_epochs: 5}
loss: {mode: hybrid}
optim:
  lr: 1.0e-4
  lr_scale: {clean_unet: 0.2, clean_spec_net: 0.2}
```

Each run gets `model.architecture=../model/fusion/<id>.yaml`, the B+ options (`model.architecture.cleanspecnet_params.output_mode=...`, `model.architecture.spectrogram.compression=...`), `seed=<n>` and its own directories on the command line; `scripts/run_ablation.py` assembles all of this. K0 uses `configs/ablation/k0.yaml` (waveform-only, 5 epochs, LR 2e-5, waveform loss).

### 8.3 Files of track N (implemented)

| File | Change | Proposals |
|---|---|---|
| `cleanunet/cleanspecnet.py` | `frontend: conv \| linear`; `feature_groups` (input with phase); `input_norm: none \| cumulative`; `predict_phase`; `run()` returns `spec`, `hidden` and `phase`; shared output and normalization functions | N0, N1b, N4, N5, N7b |
| `cleanunet/tf_mamba.py` (new) | `TFMambaSpecNet`: 2-D encoder/decoder causal in time, Mamba blocks over time (causal) + frequency (bidirectional), `hidden` per sub-band | N2 |
| `cleanunet/subband.py` (new) | `SubbandSpecNet`: full band (1-D CleanSpecNet) + a sub-band model shared across bands | N3 |
| `cleanunet/cleanunet2.py` | `cleanspecnet_params.architecture: conv1d \| tf_mamba \| subband`; `cleanspecnet_params.input_features: mag \| mag_phase`; `spec_features()`, `spec_branch()`, `spec_to_wave()`, `target_stft()`; `phase` output | N2–N5 |
| `losses.py` | `anti_wrapping`, `phase_losses` | N4 |
| `lightning_modules/cleanunet_module.py` | `loss.spec_terms: {l1, mse, wave, phase, complex, consistency}` (default `{l1: 1}` = the original loss) | N4, N8 |
| `dataset.py`, `lightning_modules/data_module.py` | `data.concat_utterances`, `data.crossfade` | N7a |
| `cleanunet/fusion.py` | Mean of the frames within each position when hop < stride | N6b |
| `scripts/diagnose_spec_branch.py` | Each model's own STFT; drift; complementarity; resynthesis with the predicted phase | PN protocol |
| `scripts/run_ablation.py` | Phases `N` and `Nd`; B++ in Phases 2–6; `--winner auto`; `--with_nc` | PN protocol |
| `configs/model/cleanunet2_mamba_e8_tfspec.yaml`, `cleanunet2_mamba_e8_subband.yaml` | Presets with N2 and N3 | N2, N3 |
| `tests/test_specnet.py` (new) | Causality in time of every variant (with and without phase), initial identity of the residual and sub-band modes, arbitrary band counts, cumulative normalization, phase losses, spectral-loss terms (with a gradient into the phase head), frame aggregation in the fusion and concatenated examples | all |

**Measured cost per step** (batch 16, 2 s, bf16, with another training on the GPU) ✔:

| Spectral branch | Parameters | ms/step | Peak memory |
|---|---|---|---|
| Current 1-D (B+) | 11.2M | 13 | 0.5 GB |
| N1a (256-channel front end) | 12.8M | 13 | 0.5 GB |
| N1b (linear front end) | 10.3M | 11 | 0.5 GB |
| N2 (dual-path TF) | 0.98M | 173 | 6.3 GB |
| N2 with phase (N4) | 1.0M | 194 | 6.8 GB |
| N3 (sub-band) | 11.2M | 92 | 3.5 GB |
| 1-D with phase at the input and the output (N4 + N5) | 19.4M | 11 | 0.7 GB |

For the 1-D models, an epoch (~20 s) is dominated by data loading, not by the network.

---

## 9. Budget

| Phase | Runs | GPU ◇ |
|---|---|---|
| 0 | D0–D3, K0 × 3, K1 | ~1.7 h (~1 h with A0) |
| 1 | P1–P9, plus the 2nd seed of those that pass | ~2 h |
| 2 | S2–S5, K2, K3, Q1, Q2 | ~2 h |
| 3 | ~12 variants | ~2.5 h |
| 4 | T1–T8 | ~1.3 h |
| 5 | Z1, Z2 (+ Z3) | ~2 h |
| 6 | 3 configurations × 3 seeds, full protocol + test | ~3 h |
| N | N-a (6 variants), N-b (3–5), retraining of B++, N-d (4), N-c (optional) | ~6–8 h (N2 costs ~1 h; B++ with N2 ~3.5 h) |
| **Total** | **~75** | **~21 h** |

Each phase fits in an overnight queue. Phases 0 and 1 together take ~4 h and already answer the main question: is there any form of fusion that works with the current spectral branch? Track N answers the next question: with a good spectral branch, does fusion become worthwhile?

**Executed ✔ (10/13 to 10/15, 10:16):**
- 163 trainings, counting PC-1 and PC-2 separately, and 311 evaluations.
- Each queue fit in half a day or a night.
- A short variant took ~7–10 min; one seed of the full protocol, ~10–15 min.
- The GPU was shared with another user. Only Phase R spent ~2 h waiting for free memory.

---

## 10. Risks and precautions

- **Small effect.** With the waveform branch already at 3.09 after DNS and fine-tuning, realistic gains should be ≤ 0.05 ◇. Seeds, controls and the dependency measure are indispensable. Measured ✔: RX_C3 gained +0.048 over K0extfull on the test set.
- **Selection on validation.** With many variants, the validation set ends up "used" by the selection. Only Phase 6 (3 seeds + test) counts for the report.
  - **It happened ✔, and in a worse way than expected.** The problem was not just the chance of many comparisons: the validation set has the same noise types as training and rewarded variants that do not generalize (section 1.5).
- **Repeated use of the test set.** The test set has already entered two decisions (Phases 6 and RT). Selection now uses val_unseen, and the test set is kept for a single final confirmation of the next finalists.
- **High dependency.** RX_C3 loses 2 PESQ points with the condition of another file. If the spectral branch fails on a noise it does not know, there is no safety net (V2, section 13).
- **External data and comparison with the literature.**
  - The waveform branch comes from DNS, and K0ext and the RX models use MUSAN.
  - When comparing with published VB-DEMAND numbers, separate "VB-DEMAND only" from "with external data".
  - K0full (3.085) is the control without external noise.
- **Overfitting.** VB-DEMAND has only 9.4 h of audio. Larger modules (O6, O7, P5) may memorize the training set; watch the gap between training and validation.
- **Lookahead and latency.** Record them for every variant. P8, S5 (iSTFT) and L+1 increase the lookahead, so they are valid for offline use only or must be reported separately.
- **Frame-rate artefacts.** A constant γ for 16 ms at the fine levels can create audible modulation at 62.5 Hz. That is why variants R2/R3 exist, and it is worth listening to the examples the training already logs (`log_audio_examples`).
- **Shared GPU.** The queue waits for free memory and resumes from `last.ckpt` (already implemented ⚑).
- **Cost of N2/N3.** Per-band (time) and per-frame (frequency) sequences multiply the number of Mamba sequences. Measure an epoch before queuing several variants; reduce frequency (4×) and the number of bands (N6a) if needed.
- **Good but redundant spectral branch.** A strong CleanSpecNet may learn the same thing as the waveform branch. The complementarity rate (PN protocol) shows this before the hybrid test.
- **Comparability.** K0 and every variant start from the same stage-A checkpoint, with the same number of steps and the same LR in the waveform branch.

---

## 11. Relation to `IMPROVEMENTS_CLAUDE.md`

| Proposal there | In this plan |
|---|---|
| A1 (multi-channel upsampler) | Replaced by P1 and S2 with O1: the same idea, without the upsampler |
| A2 (bottleneck fusion) | P3/P4 with O1/O5 and L±1 |
| A3 (multi-scale fusion) | P2, P5, P6 |
| A4 (spectral post-filter) | P8 / O8 |
| A5 (controls) | K0–K3 and the dependency measure |
| C4 (gated skips) | P7 / O9 |
| B1 (residual or mask output) | N0. In D0, the mask with log1p won |
| B2 (dual-path TF Mamba backbone) | N2 |
| B3 (input representation) | N0 (`pow` compression), N5 (phase), N6 (window and hop) |
| B4 (separate causality, bidirectional version) | Q2; in N2, bidirectional frequency with causal time |
| B5 (phase estimation) | N4 |

- **A0** is now needed only for the F0 reference (K1), the only one that still uses the original upsampler.
- Group B became track N (section 6), which runs right after Phase 1.

---

## 12. Results log ✔

Conventions of this section:
- Δ is the PESQ-WB difference against the mean of the seeds of the indicated control.
- "Dep." is the dependency: normal PESQ minus PESQ with the condition of another file.
- The full tables, with 95% intervals, Holm p-values and SNR ranges, are in `results/abl/*.md`.

### 12.1 Phases 0 and 1: spectral branch B+, short protocol, old val

- **K0** (3 seeds): 2.760 on val (σ = 0.005, hence δ_min = 0.02) and 3.092 on val_unseen.
- **D0 → B+:** the winner was the mask output with log1p, stand-alone PESQ (with the noisy phase) of 1.97. The other candidates:

  | Candidate | Stand-alone PESQ |
  |---|---|
  | `mask_pow` | 1.97 |
  | `mask_pow` with 4 s crops | 1.94 |
  | `residual_pow` | 1.92 |
  | `direct_log1p` | 1.89 |
  | `residual_log1p` | 1.75 |
  | `direct_pow` | 1.31 |

- **D3 and K1:** the original fusion (F0) has dependency 0.000 and Δ of −0.003. This confirms H1.
- **Phase 1:** with B+, no point reached δ_min.

| ID | Point | Δ val | Dep. |
|---|---|---|---|
| P3 | `bott_pre` | +0.008 | 0.01 |
| P2 | fine encoder | +0.007 | 0.01 |
| P5 | coarse decoder | +0.004 | 0.04 |
| P4 | `bott_post` | +0.004 | 0.02 |
| P6 | fine decoder | +0.003 | 0.00 |
| P8 | TF post-filter | +0.003 | 1.18 |
| P1 | `enc_s2` | −0.001 | 0.00 |
| P7 | gates on the skips | −0.005 | 0.00 |

The combination of points (P9) was done later, in Phase C.

### 12.2 Track N: spectral branch alone, old val, estimated magnitude with the noisy phase

| Variant | Stand-alone PESQ | Note |
|---|---|---|
| Reference: B+ with 30 epochs | 1.97 | |
| N1a, 256-channel front end | 2.34 | |
| **N1b, linear front end** | **2.49** | Kept |
| N6a, 32 ms window | 2.09 | |
| N7a, 6 s crops | 1.73 | |
| N7b, cumulative normalization | 2.07 | |
| N8a, time-domain loss | 1.89 | |
| N-a, combined options | 2.48 | |
| N2, dual-path TF | 2.43 | 0.98M parameters, ~13× slower per step |
| N3, sub-band | 2.48 | |
| N4, phase at the output | diverged (NaN) | Fixed with a bounded-gradient atan2: N4fix 2.31 (2.07 with the predicted phase) |
| N4 + N5, phase at the input and the output | 1.85 | N45fix 2.22 (2.27 with the predicted phase) |
| **B++ = N1b with 100 epochs** | **2.556** | See below |

Measurements of B++:
- drift of 0.97;
- "best of the two" ceiling of 3.145;
- the training-free blend with stage A (w = 0.5) gives 2.864, +0.10 over stage A.

All the validation targets of the PN protocol were met (section 6.3):
- the stand-alone PESQ passed 2.3;
- the magnitude error was below the noisy audio's in every SNR range;
- the drift was below 1.1;
- the ceiling was 0.39 above the waveform branch;
- in the hybrid, N-d P3 gave +0.059, with a dependency of 0.37.

### 12.3 Hybrids with B++: short protocol, 1 seed, Δ against K0

| ID | What | Δ val | Δ val_unseen | Dep. (val) |
|---|---|---|---|---|
| K3 | Oracle: clean spectrum at P3 | +0.267 | +0.208 | 1.09 |
| C3 | P3 + S5 + P8 | +0.121 | −0.127 | 1.70 |
| C1 | P3 + S5 | +0.073 | −0.128 | 1.15 |
| S5 | Time-domain estimate at `enc_s2` | +0.071 | −0.015 | 1.30 |
| L+1 | P3 shifted by +1 frame | +0.066 | −0.109 | 0.39 |
| C2 | P3 + P8 | +0.064 | −0.123 | 1.54 |
| N-d P2 | Fine encoder | +0.061 | −0.101 | 0.27 |
| G1 / G3 | Linear generator / 512-channel trunk | +0.061 / +0.057 | −0.117 / −0.123 | 0.45 / 0.39 |
| **N-d P3** | `bott_pre`, the base of Phases 2–4 | +0.059 | −0.116 | 0.37 |
| T1–T8 | Fusion training | +0.040 to +0.058 | −0.102 to −0.146 | 0.30 to 0.44 |
| O1, O3, O4, O5 | Operators | +0.045 to +0.055 | −0.114 to −0.134 | 0.36 to 0.45 |
| S2 / S4 | Ŝ / [`spec_hidden`; mask] | +0.055 / +0.053 | −0.073 / −0.141 | 0.24 / 0.36 |
| N-d P5 | Coarse decoder | +0.051 | −0.176 | 0.68 |
| Q2 | Bidirectional CleanSpecNet | +0.045 | −0.070 | 0.62 |
| N-x P8 | Post-filter | +0.037 | −0.089 | 1.55 |
| O6 | Cross-attention | +0.023 | −0.056 | 0.17 |
| N-d, 2nd branch | Combined N-a (30 epochs) at P3 | +0.021 | −0.032 | 0.05 |
| L−1 / Lc | −1 frame / causal STFT | +0.018 / +0.012 | −0.118 / −0.056 | 0.25 / 0.17 |
| S3 | Mask | +0.016 | −0.025 | 0.06 |
| O7 | Exchange between the Mamba stacks | +0.001 | −0.021 | 0.02 |
| K2 | Noisy spectrum | −0.002 | −0.006 | 0.00 |

Over the 50 short runs, the Spearman correlation between the Δ on val and those on val_unseen is −0.62. The variants that gained most on val lost most on new noise.

### 12.4 Phase 6 (B++, VB-DEMAND data only, full protocol, 3 seeds) and Phase 5

| Config. | val | val_unseen | Test | Dep. (test) |
|---|---|---|---|---|
| K0full | 2.767 | 3.099 | 3.085 ± 0.007 | — |
| P3 + B++ | 2.822 (+0.055) | 2.940 (−0.158) | 3.011 ± 0.007 (−0.074) | 0.32 |
| C3 + B++ | 2.893 (+0.126) | 2.944 (−0.154) | 3.071 ± 0.003 (−0.014) | 1.98 |

Diagnosis on the test set:
- With the condition turned off, P3 (seed 1) rises from 3.018 to 3.107. The B++ condition hurts on the test set.
- From val to test, B++ alone rises by +0.19 and the waveform branch by +0.33.
- On val, the gains of P3 and C3 were concentrated in babble (+0.15 and +0.29), meeting and ssn. The training babble is made of VCTK voices.
- On the test set, P3 loses on all 5 noises. C3 loses 0.07 at SNR < 2.5 dB.

Phase 5:
- **Z1** (from scratch, 30 epochs, test): P3 + B++ scored 2.656, against 2.668 for waveform-only and 2.650 for the old hybrid. It is a tie.
- **Z2** (E6): +0.017 on val and −0.039 on val_unseen.

### 12.5 Phase R: external noise, short protocol

Spectral branch alone, with the noisy phase:

| Branch | val | val_unseen | Training-free blend with stage A, w = 0.5 (val / val_unseen) |
|---|---|---|---|
| Stage A (waveform branch, reference) | 2.760 | 3.101 | — |
| B++ (VB-DEMAND, 100 epochs) | 2.556 | 2.627 | 2.864 / 3.039 |
| R_ext50 (p = 0.5, 30 epochs) | 2.415 | 2.880 | 2.785 / 3.163 |
| R_ext80 (p = 0.8, 30 epochs) | 2.379 | 2.867 | 2.765 / 3.157 |
| **Bext = R_ext50 with 100 epochs** | 2.483 | **2.902** | 2.818 / **3.177** |

Bext on val_unseen, compared with stage A:
- The log-magnitude error is close: 0.047 against 0.043.
- Bext is closer to the clean signal in 61% of the bins, weighted by energy.
- The "best of the two" ceiling reaches 3.443, i.e. +0.34.
- The blend with w = 0.25 gives 3.168.

K0ext (3 seeds), against K0, scores −0.011 on val and +0.077 on val_unseen. The gain comes mostly from park and washing.

Hybrids (1 seed), with Δ against K0ext:

| ID | What | Δ val | Δ val_unseen | Dep. (val / val_unseen) |
|---|---|---|---|---|
| RX_C3 | P3 + S5 + P8 | +0.079 | +0.052 | 1.56 / 1.95 |
| RX_S5P8 (Phase RT) | S5 + P8, without P3 | +0.063 | +0.023 | 1.61 / 1.99 |
| RX_S5 | S5 | +0.044 | +0.029 | 1.17 / 1.53 |
| RX_P3 | P3 | +0.024 | +0.012 | 0.09 / 0.11 |
| RX_P3vb | P3 with Bext, but the hybrid trained only on VB-DEMAND (Δ against K0) | +0.019 | +0.002 | 0.06 / 0.03 |

Per noise type, RX_C3 gains:
- on the 5 val_unseen environments, from +0.04 to +0.06;
- on the 10 val noises, from +0.04 (car) to +0.15 (babble).

C3 with B++ lost on all 5 val_unseen environments.

### 12.6 Phase RT: external noise and Bext, full protocol, 3 seeds

| Config. | Params | val | val_unseen | **Test** | STOI | SI-SDR | CSIG | CBAK | COVL | Dep. (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| K0full (VB-DEMAND) | 41.4M | 2.767 | 3.099 | 3.085 ± 0.007 | 0.953 | 18.48 | 4.57 | 3.66 | 3.89 | — |
| K0extfull | 41.4M | 2.754 | 3.185 | 3.164 ± 0.008 | 0.953 | 18.72 | 4.61 | 3.71 | 3.96 | — |
| RX_P3 | 52.4M | 2.777 | 3.181 | 3.151 ± 0.005 | 0.953 | 18.64 | 4.60 | 3.70 | 3.95 | 0.22 |
| **RX_C3** | 53.8M | 2.830 | 3.220 | **3.212 ± 0.008** | 0.954 | 18.96 | 4.64 | 3.75 | 4.01 | 2.06 |

**RX_C3 against K0extfull, on the test set: +0.048.**
- Per seed: +0.056, +0.048 and +0.039, all with Holm p < 1e-30.
- Per noise: from +0.044 to +0.054 on each of the 5.
- Per SNR: the smallest gain is at SNR ≥ 12.5 dB (+0.02 to +0.03). At SNR < 2.5 dB the gain is +0.03 to +0.05; with B++, C3 lost 0.07 in that range.

**RX_P3 against K0extfull: −0.013.**
- Per seed: −0.008, −0.014 and −0.017, significant for only one.
- It loses on all 5 noises.

**K0extfull against K0full: +0.079.** It gains on all 5 test noises, from +0.06 to +0.10.

### 12.7 Validation sets against the test set

| Comparison | Δ val | Δ val_unseen | Δ test |
|---|---|---|---|
| P3 + B++ − K0full (Phase 6) | +0.055 | −0.158 | −0.074 |
| C3 + B++ − K0full (Phase 6) | +0.126 | −0.154 | −0.014 |
| K0extfull − K0full | −0.013 | +0.086 | +0.079 |
| RX_P3 − K0extfull | +0.022 | −0.004 | −0.013 |
| RX_C3 − K0extfull | +0.076 | +0.035 | +0.048 |
| K0 `noise` recipe − K0extfull (VF) | +0.008 | +0.015 | +0.007 |
| Blend K0extfull + Bext − RX_C3 (V1) | −0.019 | +0.009 | −0.002 |

### 12.8 Analysis (2026-10-15)

1. **Noise diversity was the main lever, more than the architecture.**
   - On its own, external noise in the waveform branch gave +0.079 on the test set. That is more than any fusion variant of Phases 1–4, and none of those held up with new noise.
   - The spectral branch needed the same remedy: its stand-alone PESQ on val_unseen went from 2.63 to 2.90.
   - The hybrid also needs diverse noise in its own training. RX_P3vb uses Bext, but was trained only on VB-DEMAND, and stayed at +0.002.
2. **Fusion only helps when the spectral branch generalizes, and the gain comes from the offline fusions.**
   - S5 and P8 use the estimate of the spectral branch directly.
   - P3 alone loses on the test set, but inside C3 it contributes: without it, C3 loses 0.017 on val and 0.029 on val_unseen (1 seed).
3. **The two branches complement each other a lot, and the fusion exploits little of that.** On val_unseen, against stage A:
   - Bext is more accurate in 61% of the bins;
   - the "best of the two" ceiling is +0.34 above the waveform branch;
   - the K3 oracle gives +0.21;
   - RX_C3 gets +0.035 over K0extfull.
4. **From the numbers that already exist, a training-free blend gives as much as the learned fusion.**
   - With stage A, the blend gives +0.076 on val_unseen (Bext, w = 0.5) and +0.051 on the test set (B++, w = 0.25).
   - RX_C3 gives +0.035 and +0.048 over K0extfull.
   - The bases differ (stage A × K0extfull), and the complementarity may drop with a stronger base. That is why V1 comes before anything else.
5. **RX_C3 is fragile.** With the condition of another file, it loses 2.06 points. Nothing guarantees its behaviour on a noise where Bext fails.
6. **Streaming:** no streaming-compatible fusion gains on the test set. The best model with 48 ms of lookahead is K0ext.
7. **Method:** only val_unseen followed the sign of the test set (section 12.7). The rankings of Phases 1–4 were made on the old val and should not guide new choices.

### 12.9 Queue V (section 13), run on 2026-10-15, 14:31–17:55 ✔

`scripts/run_ablation.py --phases V1 V3 V2 V5 V6 V4 V7 VF --winner P3_bott_pre`. Tables in `results/abl/V*.md`; decisions in `results/abl/V*.json`. Δ in PESQ-WB; the control is the short K0ext (val_unseen 3.169, val 2.749), unless stated otherwise.

**V1: training-free blend × learned fusion (full protocol, 3 seeds).** The weight chosen on val_unseen was w = 0.25 (3.229, against 3.218 with w = 0.5).

| System | Parameters | val | val_unseen | test |
|---|---|---|---|---|
| K0extfull | 41.4M | 2.754 ± 0.004 | 3.185 ± 0.004 | 3.164 ± 0.008 |
| Blend K0extfull + Bext (w = 0.25) | 51.7M | 2.811 ± 0.004 | **3.229 ± 0.003** | 3.209 ± 0.005 |
| RX_C3 | 53.8M | **2.830 ± 0.016** | 3.220 ± 0.018 | **3.212 ± 0.008** |

- Blend − RX_C3: −0.019 on val, +0.009 on val_unseen, −0.002 on the test set (on the test set, the 95% CI of each seed includes zero). **`fusion_justified = false`.**
- On the test set, the blend beats K0extfull for all 3 seeds (+0.040 to +0.049) and has the same CSIG/COVL as RX_C3, with less variance between seeds.
- The learned fusion only beats the blend on the noise types seen in training (old val).
- Short protocol: blend K0ext + Bext +0.045 to +0.048 on val_unseen, against +0.052 for the short RX_C3. **RX_P8** alone gives +0.004.

**V3: noise recipe of the waveform branch (short K0).**

| Recipe | Δ val_unseen | Δ val |
|---|---|---|
| `noise` only (MUSAN noise, p = 0.5) | **+0.027** | +0.007 |
| p = 0.8 | +0.018 | −0.003 |
| SNR 0–20 dB | +0.013 | +0.007 |
| more babble | +0.002 | +0.002 |
| p = 0.3 | −0.010 | +0.010 |

- `noise` only, 3 short seeds: +0.020 on val_unseen. It passed and went to VF.
- **VF (full protocol, 3 seeds):** 3.171 ± 0.006 on the test set against 3.164 for K0extfull (+0.007; seeds +0.002 / +0.014 / +0.005, only one significant). On val_unseen, +0.015. A real but small gain: not worth changing the base recipe.

**V2: condition dropout in RX_C3 (short).**

| Run | val_unseen | with the condition of another file | with the fusion turned off | dependency |
|---|---|---|---|---|
| RX_C3 | 3.221 | 1.269 | 3.057 | 1.95 |
| p = 0.15 | 3.209 | 1.309 | **3.196** | 1.90 |
| p = 0.30 | 3.207 | 1.365 | 3.185 | 1.84 |

- It loses 0.012–0.014 of the Δ and **does not reduce the dependency** (a wrong condition still drops PESQ to ~1.3). Rejected.
- Side effect: with the fusion turned off, the model trained with p = 0.15 scores 3.196, above K0ext (3.169). The waveform branch learns to work without a condition, but not to ignore a wrong one.

**V5: streaming-compatible fusions (short, Bext, external noise).** None passed: S2 +0.014, P2 +0.014, P3 +0.012, G1 +0.008, O4 +0.007, P3 with dropout +0.003, P5 +0.000. All below δ_min on val_unseen, despite +0.018 to +0.029 on the old val.

**V6: distillation of RX_C3 into the waveform-only model (short).** λ = 1: +0.020 on val_unseen, +0.011 on val; λ = 0.5: +0.016 / +0.010. Just below δ_min; a gain of the same size as the `noise` recipe of V3, which gave only +0.007 on the test set.

**V4: spectral branch alone (PN protocol, external noise).** Reference: R_ext50, stand-alone PESQ 2.880 on val_unseen.

| Branch | alone (val_unseen) | blend with stage A, w = 0.5 (val_unseen) |
|---|---|---|
| R_ext50 | 2.880 | 3.163 |
| R_ext50_final (= Bext) | 2.902 | 3.177 |
| N3 sub-band | 2.875 | 3.146 |
| deep (8 layers) | 2.876 | 3.161 |
| **bidirectional** | **2.915** | **3.190** |

None reached the +0.05 threshold. The bidirectional one, with only the short training, already beats the final Bext (+0.013 alone and in the blend); it is not causal.

**V7: 30 epochs (1 seed, seed 1).**

| Run | 10 epochs val_unseen | 30 epochs val_unseen |
|---|---|---|
| K0extfull | 3.182 | 3.200 (+0.018) |
| RX_C3 | 3.238 | 3.238 (+0.000) |

The waveform branch still improves with more epochs; RX_C3 has already saturated. The gap drops from +0.056 to +0.038.

### 12.10 Analysis of queue V

1. **The learned fusion is not justified on new noise.** The training-free blend of the two separately trained models ties with RX_C3 on the test set (−0.002) and on val_unseen (+0.009), with fewer parameters, no joint training and without the fragility of a dependency of 2. RX_C3 only has an edge on the noises seen in training (old val, +0.019), which suggests that joint training fits the fusion to the training noises.
2. **The recommended offline system** becomes **K0extfull + Bext + blend (w = 0.25)**: 3.209 ± 0.005 on the test set. The ways to improve it are the two separate branches: the spectral branch (the bidirectional one is the strongest candidate, if whole-utterance latency is acceptable) and the waveform branch (more epochs: +0.018 with 30).
3. **Streaming:** no causal fusion, and no distillation, passed δ_min. The best streaming model is still K0ext, possibly with 30 epochs and the `noise` recipe (small gains that may add up; not tested together).
4. **Condition dropout** teaches the model to work without a condition, not to resist a wrong one. Reducing the dependency would require training with corrupted conditions (from another file or with noise), which makes little sense given item 1.
5. **Method:** the old val stayed optimistic about everything involving fusion (V5: +0.018 to +0.029 on val, ~0 on val_unseen; V1: it favours RX_C3). val_unseen overestimated the V3 `noise` recipe (+0.015 against +0.007 on the test set), but got the sign right; the blend − RX_C3 difference stayed within the noise on both sets.

### 12.11 Complementary analysis of queue V (2026-10-15, evening) ✔

Done with the regenerated outputs of the 4 systems (K0extfull, RX_C3, blend, K0 with the `noise` recipe; 3 seeds; test and val_unseen). The outputs reproduce the PESQ values of the tables exactly. The scripts were kept out of the repository.

**Per noise (mean of the 3 seeds, Δ against K0extfull):**

| Noise | Blend | RX_C3 | Blend − RX_C3 |
|---|---|---|---|
| test: bus | +0.018 | +0.044 | −0.026 |
| test: cafe / living / office / psquare | +0.046 to +0.060 | +0.044 to +0.054 | −0.001 to +0.010 |
| val_unseen: washing | +0.020 | +0.046 | −0.026 |
| val_unseen: river / park | +0.059 / +0.063 | +0.024 / +0.044 | +0.035 / +0.020 |
| val_unseen: field / hallway | +0.039 / +0.040 | +0.037 / +0.025 | +0.002 / +0.015 |

RX_C3 only has an edge on stationary low-frequency noise (bus and washing machine). On natural non-stationary noise (river, park), the blend is better.

**Failure tail of RX_C3.** Per file, against K0extfull:

| | Blend (test) | RX_C3 (test) | Blend (val_unseen) | RX_C3 (val_unseen) |
|---|---|---|---|---|
| Files with Δ < −0.1 | 0.5% | 5.9% | 0.4% | 8.3% |
| 1st percentile of Δ | −0.08 | −0.47 | −0.09 | −0.47 |

- The median gain is the same for both (~+0.04 to +0.05). RX_C3 wins more often by a little and sometimes loses by a lot.
- On the tail files, the other metrics barely get worse: ΔSI-SDR −0.02 dB and ΔSTOI −0.005. On the other files, ΔSI-SDR is +0.26 dB.
- The failure already appears in PC-1 (only the fusion trained, branches frozen). It comes from the fusion, not from the joint fine-tuning.

**PESQ instability.** On the worst files, PESQ-WB changes by up to 1.6 points under neutral transformations:
- Prepending 100 samples of silence to both the clean and the processed signal: on p257_196, RX_C3 goes from 2.59 to 4.21.
- Replacing 3 ms (50 samples) of RX_C3 with those of K0: recovers 1.6 points. The difference between the two in that stretch is ~35 dB below the speech.
- K0 is affected too: on p257_030, its PESQ drops from 2.53 to 1.76 with the same extra silence.

With PESQ computed as the mean over 5 amounts of leading silence (0, 40, 100, 160 and 240 samples):
- **The means and the conclusions do not change.** Blend − RX_C3 = −0.002 on the test set and +0.007 on val_unseen. RX_C3 has +0.049 and the blend +0.047 over K0extfull on the test set. The `noise` recipe has +0.005 on the test set and +0.015 on val_unseen.
- **The RX_C3 tail remains:** 5.6% of the files on the test set and 8.1% on val_unseen. About half of them fail for all 3 seeds (21 on the test set, 22 on val_unseen).
- **The test set is much less stable than val_unseen:** 4.5% of the files vary by more than 0.5 between the amounts of silence, against 0.2% on val_unseen.

Conclusion: the tail is real for PESQ, but it does not show in SI-SDR or STOI. Only listening to the 21 test files that fail for all 3 seeds can tell whether it is audible.

**Variance between seeds (standard deviation of PESQ):**

| | val_unseen | test |
|---|---|---|
| Blend | 0.003 | 0.005 |
| RX_C3 | 0.018 | 0.008 |

**Waveform averages (training-free, weights 0.5 / 0.5):**

| Combination | test | val_unseen |
|---|---|---|
| RX_C3 + blend | **3.238 ± 0.004** | **3.253 ± 0.010** |
| RX_C3 + K0extfull | 3.220 ± 0.002 | 3.238 ± 0.010 |
| Blend + K0 `noise` recipe | 3.207 ± 0.006 | 3.231 ± 0.002 |

- The RX_C3 + blend average gains +0.026 over RX_C3 on the test set. The two sets agree, and no weight was tuned.
- The per-file gains of the blend and of RX_C3 have a correlation of only 0.60 (test) and 0.45 (val_unseen). The errors of the two systems differ enough for the average to compensate them.
- Two almost identical waveform models (K0ext and K0 with the `noise` recipe) add nothing.
- Cost: K0 + RX_C3 + Bext, ~105M parameters and two passes, offline. It is an ensemble, not an architecture.

---

## 13. Next experiments (proposal of 2026-10-15)

**Status:** V1–V7 run on 2026-10-15 (section 12.9). V4d and V8 were not run: they require downloading another corpus, and the obvious candidate (ESC-50) comes from Freesound, like MUSAN, whose IDs are not listed, so there is no way to guarantee unseen noise.

**Rules for all:**
- Selection uses val_unseen. On the old val, a variant may not drop by more than 0.04 (2·δ_min) against the control.
- Δ is always against the control with the same training: K0ext, or the K0 of a new recipe.
- Screening uses 1 seed; the full protocol, 3.
- The test set is kept for the final confirmation only.

### Priority 1: next queue, little or no code

| ID | Experiment | Question | Cost ◇ |
|---|---|---|---|
| **V1** | **Ensemble control.** Training-free blend of K0extfull (3 seeds) with Bext, w = 0.25 and 0.5, with the phase of the waveform branch (`scripts/diagnose_spec_branch.py --wave_ckpt logs/abl/K0extfull_s*/last.ckpt --spec_ckpt Bext=logs/abl/N_R_ext50_final/last.ckpt`), on val, val_unseen and, with the chosen w, the test set. Also a short run of **RX_P8**: only P8, with Bext and the external noise | Does the learned fusion (RX_C3) beat a simple blend of the same two models? | ~1 h, mostly inference |
| **V3** | **Noise recipe of the waveform branch.** Short K0 with p = 0.3 and 0.8 (today 0.5); SNR from 0 to 20 dB (today −5 to 20); weights `noise` only and more babble (0.4 / 0.4 / 0.2; today 0.6 / 0.25 / 0.15) | p = 0.5 was chosen by the spectral branch. Which recipe is best for the waveform branch? | ~6 short runs, ~45 min |

**Decisions:**
- **V1, if the blend is within δ_min of RX_C3 on val_unseen:** the learned fusion is not justified.
  - The offline system becomes K0ext + Bext + blend, without joint training, with the two branches trained separately.
  - Fusion research is kept only for what the blend cannot do: streaming (V5) and distillation (V6).
- **V1, if RX_C3 is ≥ δ_min above the blend:** the learned fusion adds something. Continue with V2 and V4 on RX_C3.
- **V3, if a recipe beats K0ext by ≥ δ_min:** it becomes the base of the next hybrids. Bext may need to be retrained with it.

### Priority 2: new code, or dependent on V1/V3

| ID | Experiment | Question / hypothesis | Cost ◇ |
|---|---|---|---|
| **V2** | **Condition dropout in RX_C3.** With probability p per batch, the training step runs with `fusion_enabled=False` (P3, S5 and P8 off), with p = 0.15 and 0.3. The spectral branch keeps getting the spectral loss. If it works, a per-example version (a per-item mask on each fusion path) comes next | Does the dependency drop from ~2 to below 0.5 without losing the Δ? With the fusion off, PESQ should stay close to K0ext instead of collapsing. The regularization may even help on val_unseen | ~1 h of code and tests, plus ~30 min of GPU |
| **V4** | **Better spectral branch for the offline system**, in the PN protocol with external noise, chosen by the stand-alone PESQ on val_unseen and then tested in RX_C3. Candidates: (a) N3 sub-band, which tends to generalize better and tied with N1b on the old val; (b) scale (N9) and more epochs; (c) bidirectional CleanSpecNet, only for whole-utterance mode, which is another latency class; (d) another noise corpus besides MUSAN, without DEMAND, which requires a download | The K3 oracle (+0.21) and the "best of the two" ceiling (+0.34) show there is room. Does a better Bext increase the hybrid's gain (H10)? | ~1–3 h per candidate |
| **V5** | **New screening of the streaming-compatible fusions on val_unseen**, with Bext and the external noise: P2, P5, S2, O4/G1, Lc and P3 with condition dropout | Is there a streaming-compatible fusion that gains on new noise? Only worth it if a streaming hybrid is a goal | ~8 short runs, ~1.5 h |
| **V6** | **Distillation of RX_C3 into the waveform-only model.** The teacher is the frozen RX_C3; the student starts from K0extfull. Loss: waveform against the clean signal + λ · L1 against the teacher output, with the external noise | Does part of the +0.048 pass to a 41.4M-parameter model, with 48 ms of lookahead and no spectral branch? It is the most likely path to a streaming model better than K0ext | ~1–2 h of code, plus ~30 min of GPU |
| **V7** | **Longer training:** K0ext and RX_C3 with 30 epochs in the full protocol, 1 seed each | External noise makes training harder. Does the RX_C3 gain grow or vanish with more epochs? | ~2–3 h |
| V8 (optional) | **Second unseen-noise set**, for example NOISEX-92 or ESC-50, with the test speakers | val_unseen is now used for selection, so an independent set strengthens the final conclusion | Download, plus ~10 min |

**Final confirmation:** the winners of V1–V7, with 3 seeds, the full protocol and the test set used once.

**What not to repeat:** screenings of operator, injection point or fusion training chosen on the old val.

**Suggested order:**
1. One queue with V1 and V3, which need no code.
2. While it runs, implement V2. Then run V2.
3. Choose between V4 and V6 depending on the result of V1.

---

## 14. Best system and simplest robust model (queue W, 2026-10-15)

**Goals:**
- **(A) Best result:** the highest test PESQ, offline, with no cost limit.
- **(B) Simplest robust model:** a single waveform-only network (41.4M parameters, that of K0ext), with no fusion and streaming-compatible. Robust means:
  - stable between seeds;
  - no tail of failed files;
  - good on new noise, including at low SNR.

**Starting point (test, 3 seeds):**

| System | Parameters | Test |
|---|---|---|
| K0extfull (waveform-only, 10 epochs) | 41.4M | 3.164 ± 0.008 |
| Blend K0extfull + Bext (w = 0.25) | 51.7M | 3.209 ± 0.005 |
| RX_C3 | 53.8M | 3.212 ± 0.008 |
| Output average RX_C3 + blend (12.11) | ~105M | 3.238 ± 0.004 |

**Rules:**
- **Selection** on val_unseen, with the old val as a guard (−0.04) and δ_min = 0.02, as in section 13.
- **Guards against "PESQ hacking"** for the new losses, on val_unseen: ΔSTOI ≥ −0.003 and ΔSI-SDR ≥ −0.5 dB.
- **TTA** is accepted only if it gains ≥ 0.01 on val_unseen, because it multiplies the cost.
- **Soups** are always evaluated, because they cost nothing.
- **Test set** only in the final phase (WF).

### 14.1 Experiments

| ID | Experiment | Question / decision | Cost ◇ |
|---|---|---|---|
| **W0** | **val_unseen_low** set: the val_unseen noises at −5 and 0 dB (770 files, 77 per condition; a mixture that would clip is rescaled together with a copy of the clean signal) | Robustness stress test | done |
| **W1** | **No training**, on existing checkpoints: (a) **weight soup** of the 3 seeds of K0extfull and of K0ext; (b) mean of the outputs of the 3 seeds, as the soup's ceiling; (c) **TTA** with sign inversion and with 4 input shifts (0, 64, 128 and 192 samples) on K0extfull; (d) blend weight w with Bext (0.15 / 0.25 / 0.35 / 0.5); (e) weight α of RX_C3 in the RX_C3 + blend average (0.3 to 0.7) | Free improvements? Chooses w, α and the TTA → `W1.json` | ~30 min |
| **W2** | **Loss of the waveform branch** (short K0, seed 1, external noise): (a) MSE of the compressed magnitude \|X\|^0.3 (n_fft 400) with weights 10 and 30; (b) **metric discriminator** (MetricGAN+, as in CMGAN and MP-SENet) with weights 0.3 and 1.0 | Does a loss better aligned with perception improve PESQ on new noise at no inference cost? | ~35 min |
| **W3** | **Distillation of the RX_C3 + blend average** (the best system, α and w from W1) into the waveform-only model, λ = 1 and 2 (short K0) | Does a better teacher without the RX_C3 tail pass on more gain than V6 (+0.020)? | ~15 min |
| **W4** | **Final recipe of the simple model:** the winners of W2 and W3 and, if both exist, their combination (short). The best one that passes is kept; with no winner, the recipe is plain K0ext | → `W4.json` | ~10 min |
| **W5** | **Bidirectional Bext** with the Bext budget (100 epochs; V4_bidir was the only candidate above Bext) and the blends with K0extfull (4 weights) | Chooses the spectral branch and w for the offline system → `W5.json` | ~50 min |
| **WF** | **Final, 3 seeds, interleaved, evaluated on test, val_unseen, val_unseen_low and val:** (B) K0ext with 30 epochs (control; seed 1 is V7_K0ext30) and the W4 recipe with 30 epochs (S30), plus the soups; (A) blend of S30 with the W5 spectral branch and the average α·RX_C3 + (1−α)·blend (with the W1 TTA), next to the average of analysis 12.11 | Final numbers for both goals | ~3–4 h |
| **WR** | **Robustness report** (`results/abl/W_robustness.md`): per system, standard deviation between seeds, robust test PESQ, % of files > 0.1 below K0extfull, worst noise type, val_unseen_low and streaming of the waveform-only models | Which model is the most robust? | ~40 min |

**Calibration of the loss weights.** Measured with K0extfull_s1 on training crops with external noise, by gradient norm:
- **Base (L1 + MR-STFT):** the gradient comes almost entirely from MR-STFT.
- **Compressed magnitude:** each unit of weight gives ~0.044 times the gradient of the base, so weights 10 and 30 amount to ~0.45 and ~1.3 times.
- **Generator of the metric discriminator** (after 500 discriminator steps): ~0.3 times the base, so weights 0.3 and 1.0 amount to ~9% and ~30%.
- **Compressed complex term:** left out, because the gradient explodes on bins with almost no energy (7.6 times that of the base).

**Training cost** (K0 with 16 crops of 2 s per batch): base ~16 it/s; distillation with the ensemble teacher 12 it/s and 4.3 GB; metric discriminator ~6 it/s (the PESQ of the 16 crops runs in a pool of 8 processes).

### 14.2 Implementation

| File | Change |
|---|---|
| `losses.py` | `CompressedMagnitudeLoss` |
| `cleanunet/metric_disc.py` | `MetricDiscriminator`, `batch_pesq` (process pool), generator and discriminator losses |
| `cleanunet/blend.py` | Per-example blend, used by `evaluate.py` and by the distillation teachers |
| `lightning_modules/cleanunet_module.py` | `loss.cmag`; `loss.metric_gan` (manual optimization, two optimizers, gradient clipping in the module, `mgan_step` saved in the checkpoint); `distill.teachers` (weighted average of teachers, each with an optional blend) |
| `train.py` | No `gradient_clip_val` in the Trainer when the module uses manual optimization |
| `evaluate.py` | `--tta flip/shift/flipshift`, `--robust_pesq` (column `pesq_wb_r`), reusable writing functions |
| `evaluation/metrics.py` | `robust_pesq_wb` (mean over 0, 40, 100, 160 and 240 samples of leading silence) |
| `scripts/make_soup.py` | Soup of Lightning checkpoints with the same architecture |
| `scripts/ensemble_outputs.py` | Weighted average of saved audio, with metrics in the `evaluate.py` format |
| `scripts/robustness_report.py` | Robustness report from a JSON of systems |
| `scripts/make_val_unseen.py` | `--rescale` for low SNRs |
| `scripts/run_ablation.py` | Phases W0–W5, WF and WR |
| `tests/test_w_experiments.py` | 14 tests, including the polarity ones (14.3); the whole suite passes (134) |

**Validation before the launch:**
- Dry run with 13 trainings, 198 evaluations and 75 scripts.
- Test of the decision rules with fake results, in two scenarios (with and without winners; GAN blocked by the SI-SDR guard).
- GPU smoke tests of the metric discriminator (500 steps: the discriminator loss drops from 0.41 to 0.01) and of the distillation with the ensemble teacher.

**Execution:** `logs/abl/run_W.sh` → `logs/abl/queue_W.log`, launched on 2026-10-15 at 21:11.

**First result:** the soup of the 3 K0extfull seeds gives 2.760 on val and 3.189 on val_unseen, against seed means of 2.754 and 3.185. It beats every single seed, at the cost of one model.

### 14.3 Finding during the queue: polarity dependence (2026-10-15, 21:30) ✔

The sign-inversion TTA dropped K0extfull by 0.24 on val_unseen (W1). This led to a direct test: the same model with the input inverted and the output inverted back. Measured on 300 val_unseen files for K0extfull and stage A, and on 150 for the others:

| Model | Training noise | Normal PESQ | PESQ with inverted input |
|---|---|---|---|
| Stage A | VB-DEMAND only | 3.095 | 3.095 |
| K0_s1 (5 epochs) / K0full_s1 (10 epochs) | VB-DEMAND only | 3.097 / 3.109 | 3.074 / 3.008 |
| V3_noise | MUSAN `noise` only | 3.207 | 3.190 |
| K0ext_s1 (5 epochs) | MUSAN noise / babble / music | 3.181 | **1.976** |
| K0extfull_s1 (10 epochs) | same | 3.187 | **1.494** (STOI 0.57; SI-SDR 0.95 dB) |
| V3_babble | more babble | 3.180 | **1.528** |

**Interpretation:**
- The MUSAN babble is made of speech from another corpus, and each talker comes in with the original polarity of the recording. Since the speech waveform is asymmetric, the model learned a shortcut: keep the speech with the VCTK polarity and remove the other one.
- With the input inverted, the target speech takes on the "babble polarity" and gets erased.
- The VB-DEMAND test set does not reveal this, because its polarity is constant.
- The babble of VB-DEMAND itself is made of VCTK voices, which is why K0 with VB data only barely suffers. Even so, it loses 0.10 with 10 epochs.

**Consequences:**
- Every waveform model trained with the default external noise inherits the shortcut: K0ext, K0extfull, RX_C3 and the blends based on them. Bext is invariant by construction, because it uses only the magnitude.
- For goal (B), robustness, this must be fixed.

**Fixes implemented** (off by default, so as not to change the runs of the queue in progress):
- `augment_polarity: p`, in the module: inverts the whole training pair with probability p. The distillation teachers always see the original polarity; the sign is applied to their output.
- `data.ext_noise.random_sign: true`, in `noise_bank.py`: gives a random sign to each babble talker and each noise or music excerpt.
- `evaluate.py --invert`: evaluates with the inverted input.
- Tests: the suite passes (134).

**Phase WP** (inserted before W5 and WF; the queue was restarted at 21:45 without losing work):
- **Candidates** (short K0, seed 1): `WP_flip` (pair inversion, p = 0.5), `WP_sign` (random sign on the sources), `WP_both` (both). Also V3_noise (no babble or music; 3 seeds).
- **Rule:** the loss with the inverted input must be ≤ 0.05, the Δ on val_unseen against K0ext must be ≥ −δ_min, and val acts as a guard. Among those that pass, the best Δ wins. A 1-seed winner is confirmed with seeds 2 and 3.
- **Effect on the finals (WF):** the chosen fix goes into the control (K0ext30p, 30 epochs, 3 new seeds) and into the S30 recipe. V7_K0ext30 (without the fix) stays as a reference.
- **Effect on WR:** the report gets a "PESQ with inverted input" column on val_unseen, for every system.
- **Expectation:** the random sign alone should not be enough, because the target speech would still always have the same polarity.

### 14.4 Results of queue W (2026-10-15 21:45 → 2026-10-16 04:43) ✔

The queue finished with no failures. S30_s1 crashed once with a segfault in native code, near epoch 24, and was resumed from the last checkpoint. Its numbers lie between those of the other two seeds.

**W1 (no training, val_unseen):**
- **Soup** of the 3 K0extfull seeds: +0.005 (3.189 against the mean 3.185), at the cost of one model. The **mean of the outputs** of the 3 seeds gives +0.020, at the cost of three.
- **TTA rejected:** sign inversion loses 0.24 (the polarity effect, 14.3) and the shifts lose 0.02.
- Blend K0extfull + Bext: w = 0.35 (3.230). RX_C3 + blend average: α = 0.4 (3.254).

**W2–W4** (short K0, seed 1; Δ against K0ext with 3 seeds, which has val 2.749 and val_unseen 3.169):

| Variant | Δ val_unseen | Δ val | ΔSTOI | ΔSI-SDR | Decision |
|---|---|---|---|---|---|
| Compressed magnitude, weight 10 | +0.030 | +0.029 | +0.0003 | +0.04 dB | passes |
| Compressed magnitude, weight 30 | +0.029 | +0.030 | +0.0006 | −0.01 dB | passes |
| **Metric discriminator, weight 0.3** | **+0.068** | +0.061 | 0.0000 | −0.41 dB | **W2 winner** |
| Metric discriminator, weight 1.0 | +0.099 | +0.090 | 0.0000 | −1.07 dB | blocked by the SI-SDR guard |
| Ensemble distillation (W3), λ = 1 | +0.026 | +0.022 | +0.0005 | +0.16 dB | passes |
| Ensemble distillation (W3), λ = 2 | +0.030 | +0.025 | +0.0004 | +0.15 dB | W3 winner (V6 gave +0.020) |
| Discriminator 0.3 + distillation 2 (W4) | +0.056 | +0.048 | +0.0007 | +0.13 dB | loses to the discriminator alone |

- **Recipe of the simple model** (`W4.json`): metric discriminator with weight 0.3.
- The discriminator gain is uniform over the SNR ranges (+0.06 to +0.07). The combination with distillation recovers the SI-SDR, but loses 0.012 PESQ.

**WP: polarity fix** (short K0; the loss is the normal PESQ minus the PESQ with the inverted input, on val_unseen):

| Variant | Seeds | Δ val_unseen | Δ val | Loss with the inverted input | Decision |
|---|---|---|---|---|---|
| K0ext, no fix | 3 | — | — | 0.87 | — |
| **WP_flip** (pair inversion, p = 0.5) | 1 → 3 | +0.000 → +0.001 | −0.008 → −0.004 | −0.011 | **winner** |
| WP_sign (random sign on the sources) | 1 | −0.005 | −0.001 | 0.90 | does not fix it |
| WP_both (both) | 1 | −0.005 | −0.008 | −0.007 | passes, but loses to WP_flip |
| V3_noise (no babble or music) | 3 | +0.020 | +0.007 | 0.08 (seed 3: 0.18) | does not pass |

- **The expectation of 14.3 was confirmed:** a random sign on the sources is not enough, because the target speech still always has the VCTK polarity. Only inverting the whole pair removes the shortcut.
- **The fix is free:** +0.001 on val_unseen and −0.004 on val, with 3 seeds.
- **Removing the babble does not solve it either:** V3_noise still depends on polarity for one of the 3 seeds.

**W5: bidirectional spectral branch with 100 epochs** (`W5_bidir_final`):
- Alone, it gives 2.967 on val_unseen (Bext: 2.902; short V4_bidir: 2.915) and 2.552 on val (Bext: 2.483).
- In the blends with K0extfull (3 seeds, val_unseen), the best weight is w = 0.5, with 3.249. Bext gives 3.230 with w = 0.35: +0.019.
- It is bidirectional, hence offline only.

**WF: finals** (30 epochs, 3 interleaved seeds; the polarity fix goes into the control and S30; the test set has 824 files):

| System | Parameters | Test | STOI | SI-SDR | COVL | val_unseen | val_unseen_low | val |
|---|---|---|---|---|---|---|---|---|
| K0extfull (10 epochs, starting point) | 41.4M | 3.164 ± 0.008 | 0.9534 | 18.72 | 3.96 | 3.185 | 2.434 | 2.754 |
| K0ext30p (30 epochs + polarity, control) | 41.4M | 3.161 ± 0.001 | 0.9542 | 18.71 | 3.96 | 3.201 | 2.445 | 2.757 |
| K0ext30p soup | 41.4M | 3.165 | 0.9543 | 18.74 | 3.97 | 3.204 | 2.452 | 2.762 |
| **S30** (discriminator 0.3 + polarity, 30 epochs) | 41.4M | **3.328 ± 0.011** | 0.9542 | 18.44 | 4.10 | 3.344 | 2.565 | 2.890 |
| **S30 soup** | 41.4M | **3.335** | 0.9544 | 18.47 | 4.10 | 3.348 | 2.572 | 2.897 |
| Blend S30 + W5, w = 0.5 (weight of W5) | 60.9M | 3.353 ± 0.006 | 0.9569 | 19.05 | 4.13 | 3.345 | 2.631 | 2.946 |
| **Blend S30 + W5, w = 0.25** (chosen on val_unseen, 14.5) | 60.9M | **3.372 ± 0.009** | 0.9562 | 18.83 | 4.15 | 3.370 | 2.626 | 2.945 |
| **S30 soup + W5, w = 0.25** | 60.9M | **3.375** | 0.9564 | 18.85 | 4.15 | 3.373 | 2.631 | 2.951 |
| RX_C3 | 53.8M | 3.212 ± 0.008 | 0.9536 | 18.96 | 4.01 | 3.220 | 2.514 | 2.830 |
| 0.4·RX_C3 + 0.6·blend S30 (w = 0.5) | ~115M | 3.335 ± 0.008 | 0.9561 | 19.13 | 4.12 | 3.333 | 2.633 | 2.943 |
| Average RX_C3 + blend K0extfull (12.11) | ~105M | 3.238 ± 0.004 | 0.9541 | 18.97 | 4.03 | 3.253 | 2.535 | 2.851 |

- **The metric discriminator is the big gain:** S30 is +0.167 above the control with the same training (K0ext30p). The gain appears for all 3 seeds, all 5 test noises (+0.15 to +0.18) and every SNR range (+0.15 to +0.19).
- **30 epochs (with the polarity fix) do not change the test result:** K0ext30p is at 3.161, against 3.164 for K0extfull. On val_unseen, they gain +0.016.
- **The soups gain +0.003 to +0.007 over the seed mean on every set, at the cost of one model.**
- **RX_C3 hurts the ensemble:** with it, the result drops from 3.353 to 3.335 on the test set. It only complemented the old blend, built with K0extfull.
- **On the use of the test set:** w = 0.5 was evaluated on the test set within WF. w = 0.25 was chosen afterwards, on val_unseen only, and evaluated on the test set once.

**WR: robustness** (full report in `results/abl/W_robustness_final.md`; the queue's report is in `W_robustness.md`):

| System | Test files > 0.1 below K0extfull | Worst 1% of the per-file Δ (test) | Loss with inverted input (val_unseen) | Test in streaming (offline → streaming) |
|---|---|---|---|---|
| K0extfull | — | — | **1.36** | 3.154 → 2.868 (seed 1) |
| K0ext30, no fix (seed 1) | — | — | **1.98** | — |
| K0ext30p | 4.1% | −0.361 | 0.000 | 3.163 → 2.911 (seed 1) |
| K0ext30p soup | 3.9% | −0.212 | −0.001 | 3.165 → 2.920 |
| S30 | 0.6% | −0.063 | +0.001 | 3.331 → 3.065 (seed 1) |
| **S30 soup** | 0.6% | −0.057 | 0.000 | **3.335 → 3.085** |
| Blend S30 + W5, w = 0.25 | **0.2%** | **−0.026** | 0.000 | offline only |
| S30 soup + W5, w = 0.25 | 0.4% | −0.026 | −0.001 | offline only |
| RX_C3 | 5.9% | −0.472 | **1.26** | offline only |
| 0.4·RX_C3 + 0.6·blend S30 | 0.8% | −0.084 | 0.10 | offline only |
| Average 12.11 | 1.6% | −0.163 | **1.19** | offline only |

- **Polarity:** every system with K0extfull or RX_C3 inherits the shortcut. With the fix, no system loses more than 0.002.
- **Failure tail:** S30 and the blends with it have almost no tail. K0ext30p has a 4% tail against K0extfull without changing the mean; S30, trained the same way plus the discriminator, has none.
- **Low SNR (val_unseen_low, −5 and 0 dB):** the S30 soup gains +0.138 over K0extfull, and the blends gain +0.19 to +0.20. Each system's worst noise is still above K0extfull.
- **Streaming:** every waveform model loses 0.25 to 0.29 on the test set (see 14.5).

### 14.5 Analyses after the queue (2026-10-16) ✔

**Blend weight for S30.** The w = 0.5 of W5 was chosen with K0extfull. Redone for S30 without training, on val_unseen (mean of the 3 seeds; val acts as a guard):

| w | 0 (S30) | 0.15 | **0.25** | 0.35 | 0.5 |
|---|---|---|---|---|---|
| val_unseen | 3.344 | 3.367 | **3.370** | 3.365 | 3.345 |
| val | 2.890 | 2.932 | 2.945 | 2.951 | 2.946 |
| STOI (val_unseen) | 0.9236 | 0.9256 | 0.9266 | 0.9273 | 0.9280 |
| SI-SDR (val_unseen) | 18.34 | 18.59 | 18.72 | 18.82 | 18.92 |

- With S30, which is already stronger in PESQ, the ideal weight of the spectral branch drops. w = 0.25 gains +0.026 over S30 alone (above δ_min) and passes the val guard.
- On the test set, w = 0.25 gives 3.372 ± 0.009: +0.044 over S30 and +0.019 over w = 0.5 (per-file Wilcoxon: p = 3·10⁻⁶⁰ and 2·10⁻²²). STOI and SI-SDR rise with w, so w = 0.5 is the choice when they matter more than PESQ.
- val_unseen again got the direction of the test result right.

**Does the discriminator gain show outside PESQ?** Seed means on the test set. Lower is better for WSS and LLR.

| System | PESQ-WB | PESQ-NB | STOI | SI-SDR | segSNR | WSS | LLR |
|---|---|---|---|---|---|---|---|
| K0ext30p | 3.161 | 3.755 | 0.9542 | 18.71 | 9.40 | 3.88 | 0.289 |
| S30 | 3.328 | 3.821 | 0.9542 | 18.44 | 9.04 | 4.07 | 0.291 |
| Blend S30 + W5, w = 0.25 | 3.372 | 3.859 | 0.9562 | 18.83 | 9.38 | 3.66 | 0.267 |
| Blend S30 + W5, w = 0.5 | 3.353 | 3.853 | 0.9569 | 19.05 | 9.62 | 3.36 | 0.269 |
| RX_C3 | 3.212 | 3.781 | 0.9536 | 18.96 | 9.64 | 3.76 | 0.281 |

- **In S30, the gain stays in PESQ.** Against K0ext30p, PESQ-WB rises by 0.167 and PESQ-NB by only 0.066. STOI does not change, and SI-SDR (−0.27 dB), segSNR (−0.36 dB) and WSS (+0.19) get slightly worse.
- **CSIG, CBAK and COVL are not independent evidence.** All three are linear combinations that include PESQ (0.603, 0.478 and 0.805 per PESQ point). With that term removed, S30 is 0.03 lower in CSIG, 0.02 lower in CBAK and equal in COVL.
- **The blends improve everything at once.** They gain PESQ, STOI, WSS and LLR. SI-SDR and segSNR return to the K0ext30p level with w = 0.25 and go above it with w = 0.5.
- **Conclusion:** the PESQ of S30 is robust to a shift of the signal (robust PESQ 3.329 against 3.328), but confirming that the improvement is perceptible requires a measure other than PESQ: a listening test or a non-intrusive MOS predictor (DNSMOS, UTMOS).

**Origin of the streaming loss.** S30 soup, val_unseen, frame by frame with `cleanunet/streaming.py`:

| Input normalization | PESQ | STOI | SI-SDR |
|---|---|---|---|
| Offline (reference) | 3.348 | 0.9239 | 18.38 |
| Streaming, current causal normalization (running mean of the per-frame standard deviation) | 3.164 | 0.9136 | 17.21 |
| Streaming with the whole-file normalization (non-causal) | 3.339 | 0.9239 | 18.38 |

- The frame-by-frame processing is practically exact (−0.009). **Almost all of the loss comes from the causal estimate of the scale**, as already seen with stage A (−0.16).
- **Why:** at the start of the file there is little signal, the estimated scale is small and the speech reaches the model amplified beyond what it saw in training. The running mean takes a while to adjust.
- A variant with the running RMS was tested on 9 files and did not help (3.171 against 3.189).

### 14.6 Conclusions and next steps

**(A) Best result: S30 soup + W5 spectral branch, blend with w = 0.25.**
- **Test:** PESQ 3.375; STOI 0.956; SI-SDR 18.85 dB; CSIG 4.72; CBAK 3.81; COVL 4.15.
- **Cost:** 60.9M parameters, offline, no learned fusion. These are two separately trained networks, combined by the magnitude blend of `cleanunet/blend.py`.
- **Robustness:** it does not depend on polarity, its failure tail is 0.4% and it scores 2.631 on val_unseen_low.
- **Variant:** with w = 0.5, it trades 0.02 PESQ for +0.2 dB SI-SDR and +0.0007 STOI.

**(B) Simple, robust model: S30 soup.**
- **Architecture and use:** a single waveform-only network with 41.4M parameters, streaming-compatible.
- **Test:** 3.335 (3 seeds: 3.328 ± 0.011).
- **Robustness:** it does not depend on polarity, its failure tail is 0.6% and it gains +0.138 at low SNR.
- **Recipe:** `configs/ablation/k0.yaml` with `trainer.max_epochs=30`, `data.ext_noise` (MUSAN, p = 0.5, SNR from −5 to 20 dB), `augment_polarity=0.5` and `loss.metric_gan={weight: 0.3}`. Then `scripts/make_soup.py` over the 3 seeds.
- **Caveats:** the gain is in PESQ (14.5), and streaming loses 0.25 because of the normalization.
  - Half of the gain sits in the 50–100 Hz band (14.7.2).
  - The streaming loss has a training-free fix on val_unseen (14.7.4).
- **Conservative alternative**, without a loss that targets PESQ: the K0ext30p soup, with 3.165 and the other metrics equal to those of K0extfull.

**Proposed next steps:**
1. **Normalization for streaming** (goal B). *Updated in 14.7.4: a fixed scale or a running estimate with a prior solves it without training on val_unseen. The text below is kept for the record.* The most promising option is to fine-tune S30 for a few epochs with the causal normalization simulated in training: the same per-frame running scale of `streaming.py`, applied to the crop. The alternative is a better causal estimator, for example with a prior and a fast attack, evaluated on val_unseen. Target: recover most of the 0.18 on val_unseen.
2. **Perceptual validation of the metric discriminator:** DNSMOS or UTMOS (downloading the models follows the untrusted-files rule) or a short ABX listening test comparing S30 with K0ext30p.
3. **Causal spectral branch** for a streaming blend. W5 is bidirectional.
4. **Training stability with the discriminator:** there was 1 segfault in native code in ~4 h of training with the PESQ pool (`fork`). Switching to `forkserver` is a cheap option if it happens again.

### 14.7 Analysis of the results (2026-10-16) ✔

Scripts and data: `results/abl/W_analysis/` (see the README). On the test set, only systems that had already been evaluated on it are included, and no new choice was made with it. The streaming diagnostics use val_unseen.

**14.7.1 Is the protocol reliable?**
- **val_unseen predicts the test set.** Over the 52 systems evaluated on both, Spearman is 0.978 and Pearson 0.951 (test ≈ 0.80 · val_unseen + 0.65, with a residual of 0.031). The old val, with the training noises, gives Spearman 0.66 on the same systems. Changing the selection criterion (section 10) was right.
- **The effects are well above the noise.** 95% bootstrap intervals over the test files, with the seed mean:

| Comparison | ΔPESQ | 95% CI | Files that improve | ΔSI-SDR |
|---|---|---|---|---|
| S30 − K0ext30p (the discriminator) | +0.167 | [+0.157, +0.178] | 96% | −0.27 dB (worse for 96%) |
| Blend w 0.25 − S30 | +0.044 | [+0.037, +0.051] | 75% | +0.39 dB (better for 99%) |
| S30 − RX_C3 | +0.117 | [+0.104, +0.130] | 84% | −0.52 dB |
| K0ext30p − K0extfull (30 epochs + polarity) | −0.002 | [−0.009, +0.004] | 53% | −0.01 dB |
| S30 soup − seed mean | +0.007 | [+0.005, +0.010] | 68% | +0.04 dB |

- **30 epochs alone do not help on the test set** (−0.002 on test, +0.016 on val_unseen). What changed S30 was the loss.
- **The discriminator gain is uniform:** +0.15 to +0.19 over the 4 SNR levels and +0.15 (bus) to +0.18 (office) over the noises.
- **Speakers.** The gain varies by speaker without following gender.
  - On the test set: p232 (M) +0.13 and p257 (F) +0.20.
  - On val_unseen: p226 (M) +0.21 and p287 (F) +0.09.
  - With 2 speakers per split, generalization to new speakers is the weakest point of the benchmark.
- **The discriminator increases the variance between seeds:** ±0.011 for S30 against ±0.0013 for K0ext30p. The soup absorbs that variance.

**14.7.2 What the discriminator changed in the signal.** Attribution on the test set: start from the output of the K0ext30p soup and replace one time-frequency region at a time with the output of the S30 soup (STFT 512/128, then iSTFT). The total gain is +0.170.

| Region replaced | PESQ | Δ | SI-SDR |
|---|---|---|---|
| None (K0ext30p) | 3.165 | — | 18.74 |
| 0–100 Hz | 3.264 | **+0.099** | **18.49** |
| 100–200 Hz / 200–300 Hz | 3.182 / 3.172 | +0.017 / +0.006 | 18.72 / 18.74 |
| 300–3400 Hz | 3.213 | +0.048 | 18.74 |
| Above 3.4 kHz | 3.169 | +0.004 | 18.74 |
| Frames 0 to 20 dB below the maximum | 3.215 | +0.049 | 18.66 |
| Frames 20 to 40 dB below (weak speech, transitions) | 3.277 | **+0.111** | 18.59 |
| Frames more than 40 dB below (pauses) | 3.172 | +0.007 | 18.71 |
| Magnitude only / phase only | 3.313 / 3.191 | +0.148 / +0.026 | 18.52 / 18.73 |
| Everything (S30) | 3.335 | +0.170 | 18.47 |

- **Almost 60% of the gain and all of the SI-SDR loss are below 100 Hz**, mostly in the weak-speech frames: the < 300 Hz band of those frames alone is worth +0.078. The reverse swap confirms it: S30 with the 0–100 Hz band of K0ext30p loses 0.097.
- **Why this band weighs so much in PESQ-WB.**
  - The input filter of P.862.2 (`pesq` package) is a 2nd-order high-pass: −12 dB at 50 Hz and −3 dB at 100 Hz, relative to the passband.
  - The level alignment uses only the power above 300 Hz.
  - So the content between 50 and 100 Hz goes straight into the disturbance.
  - Below 50 Hz it does not matter: removing that content from the outputs barely changes the difference between the systems (+0.162).
- **It is a hard band with little speech.**
  - 88% of the energy of the test noise is below 100 Hz. The input SNR is −16 dB in 0–50 Hz and −0.2 dB in 50–100 Hz.
  - In the clean reference, 50–100 Hz holds 1.6% of the energy for p232 (male F0 region) and 0.15% for p257.
- **It is not more noise suppression.** By the shadow filter (G = Y/X applied to the speech and to the noise), S30 attenuates the noise slightly less (9.7 against 9.9 dB). It also leaves more residual in the pauses: −13.1 against −14.2 dB relative to the input.
- **Evaluation without the band below 100 Hz.** Zero-phase high-pass at 100 Hz on the reference and the output:

| System | PESQ | PESQ without < 100 Hz | SI-SDR without < 100 Hz |
|---|---|---|---|
| K0ext30p soup | 3.165 | 3.337 | 25.96 |
| RX_C3 (3 seeds) | 3.211 | 3.363 | 25.94 |
| S30 soup | 3.335 | 3.415 | 25.89 |
| S30 soup + W5, w 0.25 | 3.375 | 3.455 | 26.11 |

  - **S30 − K0ext30p** drops from +0.170 to **+0.078** [+0.073, +0.084]. 92% of the files improve, with only −0.07 dB of SI-SDR. Per speaker: +0.072 and +0.083.
  - **S30 − RX_C3** drops from +0.124 to +0.053 [+0.045, +0.061].
  - **Blend − S30** stays the same (+0.040): the gain of the blend is in the speech band.
  - **Blend − RX_C3** gives +0.092 [+0.084, +0.100], and SI-SDR now favours the blend (+0.17 dB).
  - The ranking of the systems does not change.
- **SI-SDR on this benchmark is dominated by what lies below 100 Hz.**
  - Without that band, SI-SDR goes from ~18.7 to ~26 dB for every system, and the differences between them drop to ≤ 0.2 dB.
  - RX_C3's SI-SDR edge (+0.5 dB) comes almost entirely from that band: without it, the difference to S30 drops to 0.04 dB.
  - A zero-phase high-pass at 50 Hz on the output only gives +0.8 to +0.9 dB of SI-SDR to K0ext30p and S30, without changing PESQ.
  - A causal high-pass ruins SI-SDR because of its phase: with 4th order at 50 Hz, it drops to ~5 dB.
- **Reading.**
  - Half of the discriminator gain (+0.08) is an improvement in the speech band, with no SI-SDR cost above 100 Hz.
  - The other half (+0.09) lies between 50 and 100 Hz, which PESQ-WB weighs heavily and whose perceptual relevance is small. Phone and laptop speakers do not even reproduce that band.
  - The listening test should look at this band, with headphones.

**14.7.3 Blend with W5.**
- **The fixed weight is almost optimal.** On val_unseen, the best per-file w (between 0 and 0.5) would give 3.398, only +0.028 over the fixed w = 0.25. A rule based on the true SNR would give +0.002. A learned gate has little room.
- **The spectral branch helps more at low SNR.**
  - On val_unseen, the best w is 0.35 at 2.5 dB and 0.15 at 12.5 and 17.5 dB.
  - On the test set, the blend gains +0.061 at 2.5 dB and +0.027 at 17.5 dB.
  - On val_unseen_low, it gains +0.061.
- **The PESQ gain of the blend comes from the speech frames and is spread over the bands.** The largest piece is +0.014, in 300–3400 Hz, and the pauses contribute only +0.002. The SI-SDR recovery (+0.38 dB), on the other hand, comes almost entirely from < 100 Hz (+0.32).

**14.7.4 Streaming.** S30 soup, val_unseen.
- **The network tolerates a static scale error well.** Offline, with the input scale multiplied by c:

| c | 0.5 | 0.71 | 0.85 | 1 | 1.18 | 1.41 | 2 |
|---|---|---|---|---|---|---|---|
| ΔPESQ | −0.058 | −0.014 | −0.003 | 0 (3.348) | −0.002 | −0.011 | −0.042 |

- **The running estimate is very wrong at the start.** Relative to the standard deviation of the whole file, it is (median, with the 10–90 percentile range):
  - 0.29 (0.14–0.63) at 100 ms;
  - 0.30 at 500 ms;
  - 0.78 (0.69–0.87) at the end.

  The sentences have a median length of 2.3 to 2.6 s, so a good part of each file comes in amplified about 3×.
- **The final bias does not explain the loss.** Offline, fixing the final ratio per file, the loss is −0.010. With the geometric mean of the trajectory, it is −0.031.
- **The loss is in the first second.** SNR per segment, running estimate against oracle:
  - 0.5–1 s: 15.4 against 20.3 dB;
  - 1–1.5 s: 18.6 against 19.8 dB;
  - after 2 s: 12.3 against 12.5 dB.
- **Training-free fixes.** Frame-by-frame streaming; the prior values come from val, not from val_unseen:

| Input scale | PESQ | STOI | SI-SDR |
|---|---|---|---|
| Offline | 3.348 | 0.9239 | 18.38 |
| Running, as it is today | 3.164 | 0.9136 | 17.21 |
| Whole file (oracle, non-causal) | 3.339 | 0.9239 | 18.38 |
| **Fixed** (0.075, the val median) | **3.335** | 0.9235 | 18.25 |
| **Running with a prior** (62 pseudo-frames, ≈ 1 s, at 0.060: the median of the final estimate on val) | **3.325** | 0.9232 | 18.51 |

- **The fixed scale only works with a calibrated input level.** On VB-DEMAND, the standard deviation of the input varies only 1.8–2× between the 10th and 90th percentiles. The version with a prior stays adaptive.
- **Fine-tuning with the simulated causal normalization (14.6, step 1) is no longer needed on this benchmark.** A single confirmation on the test set is still missing; streaming there currently gives 3.085.

**14.7.5 Consequences.**
- **Choices (A) and (B) stand.** The ranking of the systems survives the exclusion of < 100 Hz.
- **When reporting the discriminator gain**, separate the speech-band part (+0.08) from the 50–100 Hz part (+0.09). It is worth reporting PESQ without < 100 Hz as a secondary metric.
- **(B) becomes viable in streaming** at almost offline quality and without training, if the test set confirms it.

Next steps, in order:
1. **Confirm on the test set, once, the streaming of the S30 soup** with the running scale with a prior, and with the fixed one as a reference (~5 min). Then bring the prior into `cleanunet/streaming.py` as an option.
2. **Short ABX listening test** of S30 against K0ext30p, with headphones, paying attention to the 50–100 Hz band. And/or DNSMOS/UTMOS, following the untrusted-files rule.
3. **If the 50–100 Hz part is judged irrelevant**, train the discriminator with PESQ computed on signals high-passed above 100 Hz, so the loss targets only the speech band.
4. **Causal spectral branch** for a streaming blend (unchanged).
5. **Learned gate for the blend weight:** the ceiling is +0.03, so the priority is low.

### 14.8 Queue X: the next steps of 14.7 (2026-10-16)

User request: "run the proposed experiments". There were three fronts: streaming with a prior on the test set and the option in the code; perceptual validation (DNSMOS and a listening page); and the metric discriminator restricted to the band above 100 Hz (S30hp).

**Implementation.** Tests in `tests/test_x_experiments.py` (6 new); the suite has 140 tests and passes.
- **Streaming with a prior.** `cleanunet/streaming.py`: `StreamingCleanUNet` and `stream_denoise` got `prior_std` and `prior_frames`.
  - The running mean of the per-frame standard deviation starts from `prior_frames` pseudo-frames at `prior_std`.
  - `prior_frames = inf` fixes the scale.
  - Without these parameters, the result is bit-exact with the previous one.
  - Options in `evaluate.py`: `--stream_prior_std` and `--stream_prior_frames`.
- **PESQ without < 100 Hz.** `evaluation/metrics.py` got `highpass` (4th-order Butterworth, forward and backward, zero phase) and the metric `pesq_wb_hp`: PESQ-WB with the reference and the estimate high-passed above 100 Hz. Enabled with `evaluate.py --band_limited`.
- **Restricted discriminator, `loss.metric_gan.highpass_hz`** (`cleanunet/metric_disc.py` and the module).
  - The discriminator target becomes the PESQ of the two filtered signals.
  - The discriminator sees the signals filtered by the same |H|², applied through the FFT, differentiably.
  - So the metric loss does not act below 100 Hz.
- **DNSMOS.** `evaluation/dnsmos.py` re-implements Microsoft's `dnsmos_local.py`: P.835 (SIG, BAK, OVRL) and P.808, with 9.01 s windows. The downloaded code is not executed.
  - The ONNX models are in `/home/fred/DATASETS/DNSMOS_dl_20261016/` (sha256 `269fbebd…` for `sig_bak_ovr.onnx`, `9246480c…` for `model_v8.onnx`).
  - The script runs with `python -I`.
  - On the test set, the clean reference gives OVRL 3.221, SIG 3.511, BAK 4.038 and P808 3.533. The noisy audio gives 2.697, 3.346, 3.126 and 3.053: the known values for VB-DEMAND.
- **Launchers and outputs.**
  - `logs/abl/run_X.sh`: S30hp.
  - `logs/abl/run_X_eval.sh`: streaming and PESQ without < 100 Hz for the existing systems, in new `*_bl` directories.
  - `logs/abl/run_X_dnsmos.sh`: DNSMOS.
  - Log in `logs/abl/queue_X.log`, audio in `results/abl/X_audio/` and DNSMOS in `results/abl/X_dnsmos/`.
  - Analysis scripts, report (`x_report.txt`) and the listening page: `results/abl/X_analysis/` (see the README).

**X1. Streaming with a prior (test set, one run per configuration).** The prior values come from val, without tuning on val_unseen or the test set:
- prior: 0.0604 with 62 pseudo-frames (≈ 1 s), the median of the final estimate on val;
- fixed scale: 0.075, the median of the whole-file standard deviation on val.

On val_unseen, the implementation reproduces the diagnostic of 14.7.4: 3.3253 and 3.3346.

| Soup | Offline | Current streaming | With a prior | Fixed scale |
|---|---|---|---|---|
| S30, PESQ | 3.335 | 3.085 | **3.323** | 3.317 |
| S30, STOI / SI-SDR | 0.9544 / 18.47 | 0.9416 / 17.26 | 0.9536 / 18.70 | 0.9546 / 18.39 |
| K0ext30p, PESQ | 3.165 | 2.920 | **3.151** | 3.142 |
| K0ext30p, STOI / SI-SDR | 0.9543 / 18.74 | 0.9405 / 17.34 | 0.9535 / 18.89 | 0.9544 / 18.68 |

- **The prior recovers ~95% of the loss for both models**, without training. The fixed scale recovers ~93%. On val_unseen the order between the two is reversed (S30: 3.325 against 3.335), but both stay within ≤ 0.03 of offline.
- **Recommended: the prior**, which stays adaptive. The fixed scale depends on a calibrated input level.
- **(B) in streaming scores 3.323 on the test set**, against 3.335 offline (STOI 0.9536, SI-SDR 18.70). Step 1 of 14.6 (fine-tuning with causal normalization) is no longer needed.

**X2. DNSMOS on the test set (824 files).** Per-file paired differences with a 95% bootstrap CI, and the share of files that improve:

| System | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| Clean reference | 3.511 | 4.038 | 3.221 | 3.533 |
| Noisy | 3.346 | 3.126 | 2.697 | 3.053 |
| K0ext30p soup (control) | 3.512 | 4.045 | 3.225 | 3.437 |
| S30 soup | 3.503 | 4.008 | 3.198 | 3.447 |
| S30 with the < 100 Hz band of the control | 3.505 | 4.044 | 3.217 | 3.436 |
| Control with the < 100 Hz band of S30 | 3.509 | 4.010 | 3.204 | 3.448 |
| S30 soup + W5, w 0.25 | 3.503 | 4.026 | 3.206 | 3.461 |
| RX_C3 (seed 1) | 3.512 | 4.036 | 3.219 | 3.461 |
| S30 soup, current streaming | 3.475 | 4.043 | 3.193 | 3.449 |
| S30 soup, streaming with a prior | 3.499 | 4.030 | 3.205 | 3.443 |

- **DNSMOS does not confirm the discriminator gain.** S30 − control:
  - OVRL −0.026 [−0.029, −0.024], with only 22% of the files improving;
  - BAK −0.037 [−0.040, −0.034];
  - SIG −0.009;
  - P808 +0.010 [+0.008, +0.012].
- **The difference comes from the band below 100 Hz.** Giving S30 the < 100 Hz band of the control back recovers +0.019 OVRL and +0.035 BAK (91% of the files). Putting the S30 band into the control takes away −0.020 and −0.035. Above 100 Hz, S30 and the control are practically equal in DNSMOS (OVRL −0.008). The same 50–100 Hz content that yields +0.10 PESQ-WB sounds to DNSMOS like a little more background noise.
- **DNSMOS is close to its ceiling.** The control already ties with the clean reference (3.225 against 3.221). The differences between systems (≤ 0.03) are small next to the gain over the noisy audio (+0.5) and probably below what a listener notices. That is why DNSMOS does not decide; the listening test does.
- **The other pairs have small effects.**
  - Blend − S30: OVRL +0.008, BAK +0.018.
  - S30 − RX_C3: OVRL −0.021.
  - Streaming with a prior − current streaming: OVRL +0.012, SIG +0.024. The warm-up of the scale was hurting the speech signal.

**X3. PESQ without < 100 Hz, 3 seeds (`*_bl`).** Mean ± sample standard deviation between the seeds:

| System | Test: PESQ | Test: without < 100 Hz | val_unseen: PESQ | val_unseen: without < 100 Hz |
|---|---|---|---|---|
| K0ext30p | 3.161 ± 0.001 | 3.333 ± 0.002 | 3.201 ± 0.002 | 3.381 ± 0.002 |
| RX_C3 | 3.211 ± 0.008 | 3.363 ± 0.005 | 3.220 ± 0.018 | 3.382 ± 0.013 |
| S30 | 3.328 ± 0.011 | 3.410 ± 0.006 | 3.344 ± 0.012 | 3.443 ± 0.008 |
| S30 + W5, w 0.25 | 3.372 ± 0.009 | 3.452 ± 0.006 | 3.370 ± 0.010 | 3.472 ± 0.006 |

- **S30 − K0ext30p without < 100 Hz:** +0.077 [+0.072, +0.082] on the test set, with 93% of the files improving, and +0.062 on val_unseen. With the full PESQ, it was +0.167 and +0.143.
- **The soups give the same numbers as the analysis of 14.7.2**, which confirms the implementation.

**X4. Discriminator restricted to the band above 100 Hz (S30hp).**
- **Recipe:** that of S30 with `loss.metric_gan={weight: 0.3, highpass_hz: 100}`, 3 seeds and soup.
- **Training:** the 3 seeds trained with no failures or resumes.
- **Evaluation:** with `--band_limited`. ± is the sample standard deviation between the seeds.

| Test | PESQ | PESQ without < 100 Hz | SI-SDR | PESQ-NB | CSIG / CBAK / COVL | segSNR | WSS |
|---|---|---|---|---|---|---|---|
| K0ext30p (control) | 3.161 ± 0.001 | 3.333 ± 0.002 | 18.71 | 3.755 | 4.61 / 3.71 / 3.96 | 9.40 | 3.88 |
| S30 | 3.328 ± 0.011 | 3.410 ± 0.006 | 18.44 | 3.821 | 4.68 / 3.77 / 4.10 | 9.04 | 4.07 |
| **S30hp** | 3.274 ± 0.005 | **3.427 ± 0.004** | 18.65 | 3.822 | 4.66 / 3.76 / 4.05 | 9.32 | 3.88 |
| S30 soup | 3.335 | 3.415 | 18.47 | 3.825 | 4.69 / 3.77 / 4.10 | 9.07 | 4.06 |
| S30hp soup | 3.280 | 3.430 | 18.68 | 3.825 | 4.67 / 3.76 / 4.06 | 9.35 | 3.86 |
| S30 soup + W5, w 0.25 | **3.375** | 3.455 | 18.85 | 3.862 | 4.72 / 3.81 / 4.15 | 9.41 | 3.65 |
| S30hp soup + W5, w 0.25 | 3.326 | **3.469** | **19.00** | 3.861 | 4.71 / 3.80 / 4.11 | 9.61 | **3.52** |

STOI is 0.954 without W5 and 0.956 with it in every row. Paired differences on the test set, per file and with the seed mean:

| Comparison | PESQ | PESQ without < 100 Hz | SI-SDR |
|---|---|---|---|
| S30hp − S30 | −0.054 [−0.062, −0.047], 18% improve | **+0.017** [+0.013, +0.021], 63% | **+0.21 dB**, 94% |
| S30hp − K0ext30p | +0.113 [+0.106, +0.120], 95% | +0.094 [+0.089, +0.099], 93% | −0.06 dB, 23% |
| S30 − K0ext30p (reference) | +0.167 [+0.157, +0.178], 96% | +0.077 [+0.072, +0.082], 93% | −0.27 dB, 4% |
| Soups with W5: S30hp − S30 | −0.049 [−0.056, −0.043], 18% | +0.014 [+0.012, +0.017], 64% | +0.15 dB, 93% |

On val_unseen the pattern repeats:
- S30hp − S30: −0.052, +0.014 and +0.21 dB.
- S30hp − control: +0.091, +0.076 and +0.00 dB.

- **The restriction worked, but not completely.**
  - **SI-SDR returns to the control's:** −0.06 dB on the test set and +0.00 on val_unseen, against −0.27 and −0.21 dB for S30. segSNR and WSS also return to the control level.
  - **The 50–100 Hz error also returns.** The error power relative to that of the reference, summed over the test set, is −12.5 dB, against −12.6 dB for the control and −12.0 dB for S30.
  - **A piece below 100 Hz remains.** By the band swap of 14.7.2 (`x_hp_swap.py`):
    - put into the control, the < 100 Hz band of S30hp is still worth +0.041 PESQ (that of S30 was worth +0.099);
    - replacing it with the control's takes 0.043 away from S30hp (for S30, it took 0.097).
  - **That residual comes from the soft cutoff.**
    - The filter (4th-order Butterworth, applied forward and backward) attenuates only 6 dB at 100 Hz, 8.5 dB at 94 Hz and 17 dB at 80 Hz.
    - The STFT window carries part of what lies just above 100 Hz into the bins below it.
    - PESQ without < 100 Hz uses the same filter and also counts that transition.
- **Without < 100 Hz, S30hp is +0.094 above the control, against +0.077 for S30.** It was written here that the speech-band gain had increased. **Revised in 14.8.2:** PESQ-NB gives the same gain to both, and the filtered PESQ is the very target of the S30hp discriminator.
- **The cost is the full PESQ-WB.**
  - It is −0.054 relative to S30. CSIG and COVL drop with it, because they are regressions on PESQ.
  - Even so, they stay +0.113 above the control, with 95% of the files improving.
- **Robustness.**
  - **Smaller variance between seeds:** ± 0.005, against ± 0.011 for S30.
  - **No polarity dependence:** 3.295 on val_unseen and 3.299 with the inverted input.
  - **Low SNR:** 2.534 on val_unseen_low, against 2.565 for S30 and 2.445 for the control.
  - **Failure tail:** 1.1% of the files are more than 0.1 below K0extfull (0.8% for the soup). S30 has 0.6% and the K0ext30p control 4.1%.
  - **Streaming with a prior:** the soup gives 3.273 on the test set, −0.007 from offline. S30 is −0.012 from offline.
- **DNSMOS on the test set:**

| System | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| K0ext30p soup (control) | 3.512 | 4.045 | 3.225 | 3.437 |
| S30 soup | 3.503 | 4.008 | 3.198 | 3.447 |
| S30hp soup | 3.508 | 4.034 | 3.215 | 3.438 |
| S30hp soup + W5, w 0.25 | 3.507 | 4.046 | 3.219 | 3.450 |
| S30hp soup, streaming with a prior | 3.501 | 4.050 | 3.218 | 3.428 |

  - **S30hp − S30:**
    - OVRL +0.016 [+0.014, +0.018], with 73% of the files improving;
    - BAK +0.025, with 82%;
    - P808 −0.009.
  - **S30hp − control:** OVRL −0.010 [−0.012, −0.008], BAK −0.012, P808 +0.001 (not significant).
  - **S30hp recovers ~60% of the S30 deficit**, in the same proportion as the drop of the band-swap effect (from +0.099 to +0.041). What remains follows the residual below 100 Hz.
  - **With W5:** the S30hp blend − the S30 blend gives OVRL +0.013 and BAK +0.019.
  - **DNSMOS sees no discriminator gain, not even in the speech band** (S30hp − control: OVRL −0.010, P808 +0.001). Since the control already ties with the clean reference, it cannot separate systems at this level. Here it only serves to detect degradation, such as the BAK below 100 Hz.

**X5. Blind listening page:** https://claude.ai/artifact/GqL1QYn6mBzMSc4KHuijXm (private, for listening with headphones).
- **Content:** 22 trials, with test files as WAV.
  - **Block A:** 8 preference trials, S30 soup against the K0ext30p soup.
  - **Block B:** 8 ABX trials, S30 against S30 with the < 100 Hz band of the control. Tests whether that band is audible.
  - **Block C:** 6 preference trials, the blend with W5 against S30.
- **File selection:** between the 75th and 95th percentiles of the per-file ΔPESQ (robust PESQ in blocks A and C), balanced by speaker. The position of each system and the ABX reference were drawn once, when the trials were generated.
- **Answers:** one document per answer in the `respostas` ("answers") collection of the artifact database (uid, trial, block, file, noise, SNR, order, answer, correct or choice, time). To read: ArtifactData `list`.
- **Files:** `listen_audio.py` generates the WAVs, and the page is in `results/abl/X_analysis/listen/`.
- **Status on 2026-10-16, 11:15:** no answers yet.

**14.8.1 Conclusions of queue X.**
1. **Streaming solved without training.**
   - With the running scale with a prior (`--stream_prior_std 0.0604 --stream_prior_frames 62`), streaming is within ≤ 0.014 of offline on the test set: S30 3.323, S30hp 3.273, K0ext30p 3.151. Before, it lost 0.25.
   - The DNSMOS of streaming with a prior is equal to that of offline or slightly above (OVRL +0.003 to +0.007).
2. **The metric discriminator has two effects.**
   - **In the speech band**, a PESQ gain that holds without < 100 Hz and in PESQ-NB (+0.066 for S30, +0.067 for S30hp), with no SI-SDR cost.
   - **In 50–100 Hz**, content that PESQ-WB rewards (+0.10). It costs SI-SDR (−0.27 dB), and DNSMOS reads it as background noise (BAK −0.035).
   - S30hp separates the two: it keeps the first and removes ~60% of the second.
3. **Recommendation.**
   - **For the benchmark number (PESQ-WB)**, the S30 soup + W5 (3.375) and the S30 soup (3.335) remain, as in 14.6. The caveat is explicit: ~0.09 of this gain is the 50–100 Hz band.
   - **For use, the S30hp soup as (B) and the S30hp soup + W5 as (A).** They are better on every other metric.
     - **(B)** PESQ 3.280; without < 100 Hz 3.430; SI-SDR 18.68 dB; DNSMOS OVRL 3.215. The variance between seeds is half that of S30.
     - **(A)** PESQ 3.326; without < 100 Hz 3.469 and SI-SDR 19.00 dB, the highest of all systems; WSS 3.52, the lowest; DNSMOS OVRL 3.219.
   - The choice between S30 and S30hp is a question about the 50–100 Hz band, and that is what block B of the listening test tests.
     - If the band is not audible, S30hp keeps the advantage on the remaining metrics.
     - If it is audible and the S30 version is preferred, PESQ-WB was right, and S30 is the choice.
4. **DNSMOS is at its ceiling on this benchmark:** it detects degradation, but it does not rank systems that already match the clean reference.

**Next steps:**
1. **Listening test** (up to the user): the X5 page takes ~10 min, with headphones.
2. **Steeper or higher cutoff in the discriminator** (8th order, or 150 Hz), to remove the +0.04 residual below 100 Hz: ~3 h for 3 seeds. Only worth it if the listening test shows the band matters.
3. **Causal spectral branch**, for a streaming blend (unchanged). Not part of queue X.
4. **Learned gate for the blend weight:** ceiling of +0.03, low priority. Not part of queue X.

**14.8.2 Complementary analysis (2026-10-16, evening).** No training and no new inference: only the per-file results already stored. Script `results/abl/X_analysis/x_extra.py`, output `x_extra.txt`.

**PESQ-NB as a check on the speech band.** PESQ-NB only evaluates 300–3400 Hz and is the target of neither discriminator. It is the same algorithm, so it does not replace the listening test. Paired differences on the test set, per file and with the seed mean:

| Comparison | PESQ-WB | Without < 100 Hz | PESQ-NB | SI-SDR | segSNR | WSS (lower is better) |
|---|---|---|---|---|---|---|
| S30 − K0ext30p | +0.167 | +0.077 | +0.066 [+0.062, +0.070], 94% | −0.27 dB | −0.36 | +0.19 |
| S30hp − K0ext30p | +0.113 | +0.094 | +0.067 [+0.063, +0.071], 95% | −0.06 dB | −0.08 | −0.00 |
| S30hp − S30 | −0.054 | +0.017 | +0.001 [−0.001, +0.002], 46% | +0.21 dB | +0.28 | −0.19 |

- **The discriminator's speech-band gain is real and equal for both models:** +0.066 and +0.067 PESQ-NB over the control. It appears for the 3 seeds (t of 28 and 66 between the per-seed means), the 5 noises and the 4 SNRs of the test set.
- **The +0.017 of S30hp over S30 without < 100 Hz does not appear in PESQ-NB.** The filtered PESQ is the target of the S30hp discriminator, as PESQ-WB is that of S30, and each model gains on the metric it optimizes. S30hp keeps the speech-band gain of S30, without increasing it.
- **The advantages of S30hp over S30 outside PESQ:** SI-SDR +0.21 dB, segSNR +0.28, WSS −0.19, DNSMOS OVRL +0.016 and BAK +0.025, with half the variance between seeds.
- **Against the control, S30hp ties on everything that is not PESQ:**
  - equal STOI, equal WSS (−0.003), SI-SDR −0.06 dB on the test set and +0.00 on val_unseen, LLR +0.003;
  - the gain is only in PESQ: NB +0.067, WB +0.113;
  - DNSMOS OVRL is −0.010.

**Speakers.** The test set has 2 speakers (p232 and p257) and val_unseen another 2 (p226 and p287). The per-file CIs treat ~800 files as independent and do not measure the variation between speakers. Means per speaker:

| | p232 (test) | p257 (test) | p226 (val_unseen) | p287 (val_unseen) |
|---|---|---|---|---|
| S30 − K0, PESQ-WB | +0.128 | +0.203 | +0.208 | +0.088 |
| S30 − K0, without < 100 Hz | +0.070 | +0.083 | +0.062 | +0.062 |
| Part below 100 Hz (the difference of the two rows) | 0.058 | 0.120 | 0.146 | 0.026 |
| S30hp − S30, without < 100 Hz | +0.003 | +0.030 | +0.049 | −0.015 |
| S30hp − S30, SI-SDR | +0.31 dB | +0.13 dB | +0.22 dB | +0.20 dB |
| S30hp − K0, without < 100 Hz | +0.073 | +0.113 | +0.111 | +0.047 |
| Energy of the clean reference in 50–100 Hz (dB of the total) | −12.4 | −25.9 | −12.1 | −8.3 |

- **Stable across the 4 speakers:**
  - the gain without < 100 Hz of S30 and S30hp over the control;
  - the PESQ-WB loss and the SI-SDR gain of S30hp over S30.
- **The part below 100 Hz varies 5×** between the speakers, from 0.026 to 0.146.
- **The +0.017 of S30hp over S30 without < 100 Hz goes from −0.015 to +0.049.** It follows the part below 100 Hz: it is larger where S30 gained more below 100 Hz (p226 and p257). This reinforces that it comes from the transition of the soft cutoff, not from the speech band.
- **Within each split, the speaker with less 50–100 Hz energy in the reference has the larger effect below 100 Hz:** p257 on the test set and p226 on val_unseen.
  - For p257, a higher voice (energy concentrated in 200–300 Hz), the reference has almost no energy in that band (−25.9 dB of the total).
  - The 50–100 Hz error of every system is as large as the reference itself (−0.4 to −0.9 dB): what PESQ-WB rewards there is not reproduced speech.
  - With 4 speakers, this is only a hint.

**Noise and SNR.**
- Every effect has the same sign across the 5 noises and the 4 SNRs of the test set, and the 5 noises of val_unseen.
- The DNSMOS deficit of S30 grows with the noise:
  - OVRL −0.016 at 17.5 dB and −0.037 at 2.5 dB;
  - BAK −0.019 and −0.057.
- S30hp halves that deficit at each SNR. This is consistent with the 50–100 Hz content of S30 being residual noise.

**Level in 50–100 Hz.**
- S30 does not put more energy in the band: median −0.07 dB per file, against the control. The error in the band rises by +0.6 dB.
- Per file, neither the level difference nor the error difference in 50–100 Hz predicts the part of the PESQ gain below 100 Hz (|r| ≤ 0.11).
- The mechanism inside PESQ remains open.

**Training.** The training curves of S30 and S30hp (batch PESQ, discriminator loss, waveform loss) have the same shape, with no instability. Both still rise by ~0.01 PESQ over the last 10 epochs.

**Consequences.**
1. **The recommendation of 14.8.1 does not change, but its justification does.** S30hp has the same speech-band PESQ gain as S30 (PESQ-NB) and is better in SI-SDR, segSNR, WSS and DNSMOS. STOI and LLR stay equal. It does not have "more speech-band gain".
2. **When reporting the discriminator's speech-band gain, use PESQ-NB**, not PESQ without < 100 Hz. PESQ-NB is the target of neither discriminator and has no cutoff transition.
3. **Listening test.** Block A compares S30 with the control and mixes the two effects. Since the recommendation for use is S30hp, it would be worth replacing block A with S30hp against the control, which isolates the speech-band gain. There are no answers yet, so the change loses no data. Up to the user.
4. **When generalizing, count speakers, not files.** Effects that change sign between the 4 speakers (like the +0.017) should not be reported with the per-file CI alone.

### 14.9 Queue Y: training-free checks and a set with 80 speakers (2026-10-16 to 19) ✔

User request: run items 1 to 3 proposed after 14.8.2 and download VCTK for item 7. There was no training.
- **Launchers:** `logs/abl/run_Y.sh` (Y1–Y3) and `logs/abl/run_YV.sh` (vctk_unseen). Log in `logs/abl/queue_Y.log`.
- **Analysis:** `results/abl/Y_analysis/` (README, `y_report.txt`, `y_vctk.txt`).
- **New directories:** `<run>/val_unseen_full` and `<run>/test_full` (all metrics plus PESQ without < 100 Hz) and `<run>/vctk_unseen`. The blends are `WS_*` (W5) and `WB_*` (Bext).
- **Execution:** Y1–Y3 on 2026-10-16, 19:39–20:11, with no failures. The evaluation on vctk_unseen stopped when the session ended (10/16, 20:13) and was resumed on 10/19, 14:54–15:00, with no failures.

**Y1. Blend weight with W5 for S30hp** (val_unseen, mean of the 3 seeds):

| w | 0.15 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| S30hp + W5: PESQ | 3.314 | **3.318** | 3.316 | 3.301 |
| S30hp + W5: PESQ-NB | 3.835 | 3.845 | 3.849 | 3.846 |
| S30hp + W5: SI-SDR | 18.76 | 18.87 | 18.95 | 19.02 |
| S30 + W5: PESQ | 3.367 | **3.370** | 3.365 | 3.345 |

- **w = 0.25 remains the choice for both.** The curve is flat between 0.25 and 0.35: 0.35 trades 0.002 PESQ-WB for +0.004 PESQ-NB and +0.08 dB SI-SDR.
- The recipe of (A) does not change.

**Y2. PESQ-NB on val_unseen** (per-file paired differences, seed mean):

| Comparison | PESQ-NB | p226 | p287 | Without < 100 Hz: p226 / p287 |
|---|---|---|---|---|
| S30 − control | +0.054 [+0.050, +0.058] | +0.052 | +0.055 | +0.062 / +0.062 |
| S30hp − control | +0.053 [+0.049, +0.058] | +0.052 | +0.055 | +0.111 / +0.047 |
| S30hp − S30 | −0.001 [−0.002, +0.001] | −0.000 | −0.001 | +0.049 / −0.015 |

- **It confirms 14.8.2 on the other 2 speakers:** the speech-band gain is the same for both models and barely varies between speakers.
- PESQ without < 100 Hz, the target of S30hp, is what varies.

**Y3. Blend with Bext: offline ceiling of a streaming blend.**
- **Bext** (`N_R_ext50_final`) is the spectral branch with a unidirectional backbone and causal convolutions. The STFT (1024/256) is still centred.
- **Weight chosen on val_unseen:** w = 0.15 for S30 and for S30hp.
- **Gain of the blend over the model alone**, on the test set and with the soups:

| Blend | PESQ-WB | PESQ-NB | SI-SDR | STOI |
|---|---|---|---|---|
| S30 + W5 (w 0.25) | +0.040 | +0.036 | +0.38 dB | +0.002 |
| S30 + Bext (w 0.15) | +0.011 | +0.014 | +0.12 dB | +0.001 |
| S30hp + W5 (w 0.25) | +0.046 | +0.037 | +0.32 dB | +0.002 |
| S30hp + Bext (w 0.15) | +0.018 | +0.015 | +0.09 dB | +0.001 |

- **The causal branch gives about a third of the gain of the bidirectional one.** And that is the offline ceiling: in streaming, the causal normalization and the STFT latency would take away more.
- **A streaming blend would give at most +0.01 to +0.02 PESQ**, with +20M parameters and new code. The "causal spectral branch" step drops to low priority.

**YV. vctk_unseen: 80 unseen speakers.**
- **Construction** (details in the README of `Y_analysis`):
  - source: VCTK 0.92 (CC BY 4.0, MD5 checked), microphone 1;
  - speakers: the 80 outside VB-DEMAND (48 women and 32 men; mostly American, Scottish, Irish and Canadian accents), 10 sentences each;
  - processing: 16 kHz and peak 0.493, as in VB-DEMAND (`scripts/prepare_vctk_clean.py`, with `python -I`);
  - noise: the 5 DEMAND environments of val_unseen, at the test SNRs (P.56), with 40 files per condition (`scripts/make_val_unseen.py --seed 2`).
- **Calibration on p232 and p257:**
  - microphone 1 reproduces VB-DEMAND (correlation 0.994; microphone 2 gives 0.946);
  - the spectrum below 200 Hz matches within 0.4 dB;
  - the 0.92 files have on average 0.26 s more leading silence.
- **Difficulty:** the PESQ of the noisy audio is 1.971 (on the test set, 1.97).
- **STOI:** the absolute value is low (0.83). Some speakers have strong rumble below 50 Hz in the clean recording, and it disturbs STOI's silent-frame removal. Paired differences are not affected.

| System (soup) | vctk_unseen: PESQ | Without < 100 Hz | PESQ-NB | SI-SDR | Test: PESQ |
|---|---|---|---|---|---|
| Control K0ext30p | 3.153 | 3.373 | 3.773 | 17.48 | 3.165 |
| S30 | 3.325 | 3.447 | 3.828 | 17.30 | 3.335 |
| S30hp | 3.256 | 3.453 | 3.827 | 17.47 | 3.280 |
| S30 + W5 (w 0.25) | **3.365** | 3.483 | **3.862** | 17.63 | 3.375 |
| S30hp + W5 (w 0.25) | 3.300 | **3.489** | 3.861 | **17.74** | 3.326 |

With the 3 seeds (mean ± sample standard deviation): K0ext30p 3.148 ± 0.005; S30 3.321 ± 0.013; S30hp 3.252 ± 0.005.

**Paired differences.** 95% CI by bootstrap over speakers, and the share of the 80 speakers that improve. For WSS, lower is better.

| Comparison | PESQ-WB | Without < 100 Hz | PESQ-NB | SI-SDR | segSNR | WSS |
|---|---|---|---|---|---|---|
| S30 − control | +0.173 [+0.159, +0.187], 100% | +0.074, 100% | +0.055 [+0.051, +0.059], 100% | −0.17 dB, 4% | −0.29, 0% | +0.08 [+0.04, +0.12], 40% |
| S30hp − control | +0.104 [+0.094, +0.115], 100% | +0.082, 98% | +0.055 [+0.050, +0.059], 100% | −0.00 dB, 48% | −0.04, 20% | +0.08 [+0.03, +0.12], 36% |
| S30hp − S30 | −0.069 [−0.078, −0.060], 0% | +0.008 [+0.002, +0.015], 51% | −0.001 [−0.003, +0.001], 49% | **+0.17 dB**, 98% | **+0.25**, 100% | −0.01 [−0.06, +0.06], 56% |
| S30 + W5 − S30 (soups) | +0.040 [+0.030, +0.051], 89% | +0.036, 90% | +0.034, 99% | +0.33 dB, 94% | +0.31, 98% | −0.32, 85% |
| S30hp + W5 − S30hp (soups) | +0.044 [+0.034, +0.055], 90% | +0.036, 90% | +0.034, 100% | +0.28 dB, 94% | +0.24, 91% | −0.29, 88% |

The STOI of the blends rises by +0.007, for 99% of the speakers; in the other comparisons it stays equal. The effects with a clear sign (PESQ-WB and PESQ-NB over the control; PESQ-WB, SI-SDR and segSNR of S30hp − S30) have the same sign across the 5 noises, the 4 SNRs and both genders.

**14.9.1 Conclusions.**
1. **The test results hold for 80 speakers.**
   - The absolute numbers and the differences are within ≤ 0.03 of the test set, with the same ranking of systems.
   - With 80 speakers, the per-speaker CI is almost equal to the per-file CI. The problem of 14.8.2 was having only 2 speakers per set.
2. **The discriminator's speech-band gain is general.**
   - PESQ-NB +0.055 over the control, the same for S30 and S30hp, positive for all 80 speakers (minimum +0.018).
   - It falls with the SNR: +0.070 at 2.5 dB and +0.039 at 17.5 dB.
3. **S30hp gains no PESQ over S30 in any band.**
   - PESQ-NB −0.001 (49% of the speakers) and without < 100 Hz +0.008 (51%).
   - Its advantages are SI-SDR (+0.17 dB, 98% of the speakers) and segSNR (+0.25, 100%), and it ties with the control in SI-SDR.
   - The WSS of the test set does not repeat: −0.01 against S30, and both are +0.08 above the control.
4. **The blend with W5 is the most robust improvement:** every metric improves, for 85–100% of the speakers (PESQ-NB for 99–100%, STOI for 99%).
5. **The part of the S30 gain below 100 Hz is the main source of variation between speakers.**
   - It goes from −0.005 to +0.298, with a mean of +0.099.
   - The standard deviation of the gain between speakers is 0.066 in PESQ-WB, 0.025 without < 100 Hz and 0.019 in PESQ-NB.
   - It is larger for the speakers with less 50–100 Hz energy in the clean speech: Spearman −0.28 (p = 0.01), and −0.45 in the S30hp residual. The hint of 14.8.2 is confirmed, but the correlation is moderate.
   - For women it is +0.110; for men, +0.082. The PESQ-NB gain is the same for both (+0.055).
   - **In S30hp − S30, the gain without < 100 Hz appears where S30 gained more below 100 Hz** (Spearman −0.73). Since PESQ-NB does not change (standard deviation between speakers of 0.008), that gain comes from the cutoff transition or from outside 300–3400 Hz, not from the middle of the speech band.
6. **The recommendation of 14.8.1 does not change.**
   - For the benchmark number: the S30 soup + W5 (test 3.375; vctk_unseen 3.365) and the S30 soup (3.335; 3.325).
   - For use: the S30hp versions. They have the same speech-band gain, without the 50–100 Hz effect and without the SI-SDR loss. The cost is −0.07 PESQ-WB (and CSIG, CBAK and COVL, which follow PESQ).
   - The listening test still decides between the two.
7. **To report the discriminator gain**, use PESQ-NB next to PESQ-WB: +0.055 over 80 speakers, against +0.17 PESQ-WB that varies from +0.05 to +0.37 between speakers.

**Next steps:**
1. **Listening test** (up to the user): still no answers on 2026-10-19. The suggestion to change block A to S30hp against the control still stands.
2. **Independent set for the final report.** vctk_unseen uses the val_unseen noises, which were used to choose the weights. Repeating it with the 5 DEMAND environments of the test set (download from Zenodo) would give a set on which no choice was made. Optional.
3. **DNSMOS on vctk_unseen**, to see the effect on BAK with 80 speakers: ~20 min, optional.
4. **Low priority:** the steeper cutoff (only if the listening test shows the band matters), the causal spectral branch (Y3) and the learned gate.

### 14.10 Queue BB: the S30 + W5 system retrained with the largest batch that fits the GPU (2026-10-19) ✔

User request: run the full training of the S30 + W5 system again, with a batch size that uses all the GPU memory.
- **Launcher:** `logs/abl/run_BB.sh`. Log in `logs/abl/queue_BB.log`.
- **Analysis:** `results/abl/BB_analysis/` (README, `bb_report.py`, `bb_report.txt`). The batch-16 stage A checkpoint
  was also scored on val_unseen and test with all metrics, in `results/abl/stageA_old/`.
- **Execution:** 2026-10-19, 15:16–18:51, with no failures.
- **Batch sizes.** The peaks were measured with a few training steps on the RTX 4090 (24.5 GB). About 10–15% of the
  memory was left free, because the GPU is sometimes shared.

| Stage | Batch | Peak memory | Optimizer steps | Learning rate | Wall time |
|---|---|---|---|---|---|
| Stage A (20 epochs) | 16 → 128 | 19.8 GB | 13,520 → 1,700 | 5.0e-5 → 1.414e-4 | 27 → 15 min |
| W5 (100 epochs) | 16 → 1024 | 22.0 GB | 67,600 → 1,100 | 5.0e-4 → 4.0e-3 | 28 → 50 min |
| S30, per seed (30 epochs) | 16 → 128 | 20.9 GB | 20,280 → 2,550 | 2.0e-5 → 5.657e-5 | 58–95 → 43 min |

- **Same recipe otherwise:** data, epochs, seeds, checkpoint selection and the soup are those of the batch-16 system.
- **Learning rates** follow the square-root rule for AdamW, lr × √(B / 16). The discriminator learning rate is scaled
  the same way (5e-4 → 1.414e-3), and its warm-up keeps the number of examples (300 → 38 steps).
- **The batch-16 wall times are not clean:** some of those runs shared the GPU, and S30 s1 crashed once and resumed.
- **W5 is slower at batch 1024 than at 16.** Data loading (MUSAN mixing and the STFT of 1024 clips per step) is the
  bottleneck, not the GPU.

**Results** (PESQ-WB; the soups, and the blends with w = 0.25):

| System | val_unseen | Test | vctk_unseen |
|---|---|---|---|
| Stage A, batch 16 | 3.101 | 3.095 | — |
| Stage A, batch 128 | 3.097 | 3.093 | — |
| S30 soup, batch 16 | 3.348 | 3.335 | 3.325 |
| S30 soup, batch 128 | 3.295 | 3.290 | 3.288 |
| S30 soup + W5, batch 16 | **3.373** | **3.375** | **3.365** |
| S30 soup + W5, large batch | 3.313 | 3.312 | 3.313 |
| Large-batch S30 soup + batch-16 W5 | 3.326 | 3.334 | 3.331 |
| Batch-16 S30 soup + large-batch W5 | 3.357 | 3.352 | 3.347 |

With the 3 seeds (mean ± sample standard deviation), on the test set: S30 3.328 ± 0.011 at batch 16 and
3.283 ± 0.013 at batch 128; S30 + W5 3.372 ± 0.009 and 3.308 ± 0.012.

**Paired differences of the full system** (soups, large batch − batch 16). 95% CI per file; on vctk_unseen, the CI by
bootstrap over speakers and the share of the 80 speakers that improve.

| Set | PESQ-WB | Without < 100 Hz | PESQ-NB | SI-SDR | CBAK |
|---|---|---|---|---|---|
| val_unseen | −0.060 [−0.065, −0.055] | −0.015 | −0.029 | −0.09 dB | −0.031 |
| Test | −0.063 [−0.071, −0.055] | −0.022 | −0.028 | −0.12 dB | −0.035 |
| vctk_unseen | −0.052 [−0.060, −0.045], 8% | −0.015, 24% | −0.025, 2% | −0.16 dB, 16% | −0.028, 11% |

**Attribution with cross blends** (soups, w = 0.25; change of PESQ-WB against the batch-16 system):

| Set | Waveform branch | Spectral branch | Both | Interaction |
|---|---|---|---|---|
| val_unseen | −0.047 | −0.016 | −0.060 | +0.003 |
| Test | −0.042 | −0.023 | −0.063 | +0.002 |
| vctk_unseen | −0.034 | −0.018 | −0.052 | +0.000 |

**14.10.1 Findings.**
1. **Stage A loses nothing at batch 128.**
   - PESQ −0.004 on val_unseen and −0.002 on the test set, within the CI; SI-SDR −0.06 to −0.09 dB.
   - 8× fewer steps are enough to fine-tune from the DNS checkpoint with best-checkpoint selection, and it took 15 min
     instead of 27.
2. **W5 underfits at batch 1024:** 1,100 steps instead of 67,600.
   - Its validation loss is 0.0183 at epoch 100, still falling. Batch 16 reached that value near epoch 25 (0.0184)
     and ended at 0.0170.
   - Alone (estimated magnitude with the noisy phase, val_unseen): PESQ 2.886 against 2.967. The log-magnitude error
     is 5–8% higher in every band.
   - The gain of the blend falls from +0.040 to +0.022 on the test set (val_unseen +0.024 → +0.018; vctk_unseen
     +0.040 → +0.024).
   - Blend weight: 0.15 and 0.25 tie on val_unseen (3.309, mean of the seeds), and 0.25 was kept.
3. **S30 at batch 128 loses most of the discriminator gain below 100 Hz.**
   - Gain over the control K0ext30p soup on val_unseen: PESQ-WB +0.144 at batch 16 and +0.091 at batch 128.
     Without < 100 Hz it is +0.064 and +0.061; PESQ-NB +0.053 and +0.035.
   - Paired, PESQ without < 100 Hz changes by −0.003 (val_unseen), −0.006 (test) and +0.000 (vctk_unseen, 50% of the
     speakers). The PESQ-WB loss sits in the band that S30hp leaves out: the large-batch S30 has the PESQ-WB of S30hp
     (3.295 on val_unseen for both), but not its SI-SDR gain.
   - There are smaller losses in the speech band too: PESQ-NB −0.016 to −0.018 (−0.009 on vctk_unseen), SI-SDR −0.04
     to −0.12 dB, CBAK −0.016 to −0.026.
   - Training curves (mean of the seeds): the PESQ of the training outputs scored for the discriminator is about 0.06
     lower at every epoch (2.853 against 2.911 at epoch 30), and the discriminator loss stays higher (0.0050 against
     0.0040). With 8× fewer updates, the generator does not get as far.
   - The spread between seeds (sd 0.013 against 0.011) and the gain of the soup (+0.007) do not change.
4. **The two losses add up** (interaction ≤ 0.003): 65–78% from the waveform branch and the rest from the spectral
   branch, on all three sets.
5. **The time saved is small.** Training took 3 h 14 min, against about 4 h 27 min at batch 16. But the batch-16 time
   includes GPU sharing and a crash, and W5 got slower.

**14.10.2 Conclusions.**
1. **Keep batch 16.** The recommended systems and numbers of 14.8 and 14.9 do not change. The large-batch checkpoints
   stay in `logs/abl/BB_*`, but they are not recommended.
2. **The square-root rule with the same number of epochs is not enough** when the step count falls by 8–64×. Only
   stage A tolerates it (fine-tuning from a pretrained model, with best-checkpoint selection).
3. **If a larger batch is ever needed** (e.g. on a bigger GPU), the options are below. None was run, and they are low
   priority, because batch 16 already fits and the time saving is small.
   - Stage A at batch 128 works as is and saves 12 min.
   - W5 at batch 128 or 256 (8,400 or 4,200 steps), with more epochs. Data loading caps the speed anyway, so first
     precompute the STFTs or add workers.
   - S30 needs about the same number of generator and discriminator updates. For example, batch 32–64 with more
     epochs, checking that the PESQ of the training outputs reaches ~2.91 by the end.

### 14.11 Queue Z: experiments for a better final system (2026-10-19)

User request: "run the planned experiments", after the proposals that followed 14.10. Item 7 of the proposals (a
spectral branch with a phase decoder, MP-SENet / SEMamba style) was left out: it needs new code and 4–8 h of GPU per
run, and our earlier attempt (N4, with the anti-wrapping losses) stayed below the magnitude-only branch. The listening
test still needs listeners.

- **Launchers:** `logs/abl/run_Z.sh` (part 1: Z1 and Z2) and the parts that follow, all logging to
  `logs/abl/queue_Z.log`. The queue runs in the tmux session `queueZ`.
- **Code:**
  - `loss.mode: blend` (`lightning_modules/cleanunet_module.py`): the output is the blend of `cleanunet/blend.py`
    (`blend_with_spec`, now differentiable) of a frozen waveform branch with the spectral branch. The waveform
    losses and the metric discriminator act on the blend, and the spectral loss keeps S an estimate of the clean
    magnitude. Config `configs/ablation/z5_blend.yaml`, architecture `configs/model/cleanunet2_mamba_e8_w5blend.yaml`.
  - Unidirectional Mamba weights now load into a bidirectional stack (`cleanunet/checkpoint_io.py`,
    `expand_to_bidirectional`): both copies get the weights and the backward output projection starts at zero, so
    the model starts as the causal one.
  - `scripts/ensemble_outputs.py --band_limited`.
  - Tests in `tests/test_z_experiments.py` (6 new); the suite has 141 passing tests on CPU (plus the CUDA-only
    checkpoint test).

**Experiments and decision rules** (fixed before the results; selection on val_unseen, the test set and vctk_unseen
only for the systems that pass):

| Id | Experiment | Compared with | Passes if |
|---|---|---|---|
| Z1 | Mean of the outputs of the 3 seeds (S30, S30hp, with and without W5) | The weight soups | PESQ-WB +0.010 (it triples the cost, as the TTA rule of W1) |
| Z2 | Short protocol (5 epochs, seeds 1 and 2): S30 + compressed magnitude (weight 10); S30hp with metric weights 0.5 and 1.0, and S30hp + compressed magnitude | The short S30 (weight 0.3) and S30hp (0.3) with the same seeds | Mean Δ ≥ +0.010 in PESQ-WB (S30 track) or PESQ-NB (S30hp track), with STOI ≥ control − 0.003 and SI-SDR ≥ control − 0.5 dB (control: the short K0ext with polarity, WP_flip) |
| Z3 | Finals of the Z2 winners: 3 seeds, 30 epochs, soup, blend with W5 (w on val_unseen) | S30 / S30hp | Paired CI of the soups above 0 in the track metric |
| Z5 | W5 fine-tuned for the blend (20 epochs, lr 1e-4, discriminator 0.3; with highpass 100 Hz for S30hp) on the frozen S30 / S30hp soup | The training-free blend with the original W5 | Δ ≥ +0.010 in the track metric, STOI ≥ −0.003, SI-SDR ≥ −0.5 dB |
| Z6 | S30 with a bidirectional Mamba bottleneck (52.4M parameters), seed 1, then seeds 2–3 if it passes | The 3 S30 seeds (3.344 ± 0.012 on val_unseen) | PESQ-WB +0.015 over the seed mean, same guards |
| Z7 | Bidirectional bottleneck + S30hp with weight 1.0, 3 seeds, only if the Z6 and Z3 finals pass (added 10-20, before its results) | The Z3 soup | Paired CI of the soups above 0 in PESQ-NB, same guards |
| Z4 | 60 epochs instead of 30, on the best waveform recipe, seed 1 first | Its 30-epoch seeds | +0.015 in the track metric |

**Z1 result (val_unseen):** the mean of the outputs gains only +0.004 to +0.005 over the soups, and STOI drops by
0.0003 to 0.0006. Rejected.

| System | Soup | Mean of the 3 outputs |
|---|---|---|
| S30 | 3.348 | 3.354 |
| S30 + W5 | 3.373 | 3.378 |
| S30hp | 3.295 | 3.299 |
| S30hp + W5 | 3.320 | 3.324 |

**Z2 result (val_unseen, mean of seeds 1 and 2; Δ paired against the reference with the same seeds):**

| Recipe (5 epochs) | PESQ-WB | PESQ-WB > 100 Hz | PESQ-NB | SI-SDR | STOI | Δ track metric [95% CI] | Decision |
|---|---|---|---|---|---|---|---|
| S30, metric weight 0.3 (reference) | 3.236 | 3.405 | 3.774 | 18.22 | 0.9226 | | |
| S30 + compressed magnitude (weight 10) | 3.234 | 3.412 | 3.780 | 18.36 | 0.9232 | PESQ-WB −0.002 [−0.005, +0.001] | rejected |
| S30hp, weight 0.3 (reference) | 3.205 | 3.403 | 3.771 | 18.41 | 0.9228 | | |
| S30hp, weight 0.5 | 3.216 | 3.415 | 3.773 | 18.46 | 0.9229 | PESQ-NB +0.002 [+0.000, +0.005] | rejected |
| S30hp, weight 1.0 | 3.225 | 3.423 | 3.782 | 18.37 | 0.9229 | PESQ-NB +0.011 [+0.009, +0.014] | **passes** |
| S30hp + compressed magnitude | 3.211 | 3.406 | 3.780 | 18.42 | 0.9233 | PESQ-NB +0.009 [+0.007, +0.011] | rejected (just below) |

All six recipes pass the guards (STOI ≥ 0.9197, SI-SDR ≥ 18.05 dB).

- The compressed-magnitude loss raises SI-SDR (+0.14 dB with S30), PESQ-NB and PESQ-WB above 100 Hz (+0.006 and
  +0.007), but not the full-band PESQ-WB of the benchmark.
- With the discriminator above 100 Hz, a larger weight helps, and the trend is monotonic over 0.3, 0.5 and 1.0, so
  1.0 is at the edge of the tested range. At weight 1.0 the S30hp recipe gets back part of the full-band PESQ-WB that
  the high-pass costs (3.225 against 3.236 for S30, instead of 3.205), for −0.04 dB of SI-SDR.

**Part 3** (`logs/abl/run_Z_p3.sh`, after part 2). The branches are decided by `results/abl/Z_analysis/z_decide.py`
with the rules above; the guards of Z3–Z6 are relative to the compared system.

- Z3: final of S30hp with weight 1.0 (`Z3_hp1.0`: 3 seeds, 30 epochs, soup, blend with W5, w picked on val_unseen).
- Z6: seeds 2–3, soup and blend if seed 1 passes.
- Z5: for each waveform soup where seed 1 passes, seeds 2–3 of the fine-tuning, a soup of the three fine-tuned
  spectral branches (they share the W5 start) and its blend, w picked on val_unseen.
- Z4, one run per track (the plan said "the best waveform recipe"; the two tracks have different metrics): for the
  S30 track, the bidirectional bottleneck if its 3-seed soup beats the S30 soup (paired CI above 0), else S30; for
  the S30hp track, weight 1.0 if Z3 passes, else S30hp. Seed 1 first against the mean of that recipe's 30-epoch
  seeds; seeds 2–3, soup and blend if it passes. The cosine schedule spans the 60 epochs.

**Part 2** ran on 2026-10-19 22:14 → 10-20 00:25 with no failures (Z6 seed 1: 62 min; each Z5 run: 33 min).

**Z5 result** (val_unseen, seed 1; fine-tuned W5 against the original, both blended with the frozen soup): rejected
on both soups.

| Waveform soup (track metric) | Original W5, w 0.25 | Fine-tuned W5, w 0.15 / 0.25 / 0.35 | Δ at w 0.25 [95% CI] | Δ SI-SDR |
|---|---|---|---|---|
| S30 (PESQ-WB) | 3.373 | 3.369 / 3.356 / 3.331 | −0.016 [−0.021, −0.012] | +0.21 dB |
| S30hp (PESQ-NB) | 3.847 | 3.843 / 3.850 / 3.847 | +0.003 [−0.000, +0.006] | +0.22 dB |

- Training through the blend trades PESQ for SI-SDR: PESQ-WB above 100 Hz falls by 0.013 on both soups. The
  waveform losses on the blended output outweigh the discriminator at 0.3. As with the learned post-filter RX_P8 (V1),
  a learned stage does not beat the fixed blend of a branch trained only for its magnitude.

**Z6 result** (val_unseen): the bidirectional bottleneck passes by a wide margin.

| System | PESQ-WB | PESQ-WB > 100 Hz | PESQ-NB | SI-SDR | STOI |
|---|---|---|---|---|---|
| S30, mean of the 3 seeds | 3.344 | 3.443 | 3.810 | 18.34 | 0.9236 |
| S30 bidirectional, seed 1 | 3.373 | 3.472 | 3.846 | 18.82 | 0.9285 |
| Δ [95% CI] | +0.029 [+0.021, +0.037] | +0.029 | +0.036 | +0.47 dB | +0.005 |
| S30 + W5 (w 0.25), mean of the 3 seeds | 3.370 | 3.472 | 3.846 | 18.72 | 0.9266 |
| S30 bidirectional, seed 1, + W5 (w 0.25) | 3.387 | 3.488 | 3.871 | 18.99 | 0.9299 |

- The gain shows on every metric, so it does not come from the discriminator alone.
- On its own, one bidirectional seed equals the S30 soup + W5 system (3.373): one model of 52.4M parameters instead
  of two (60.9M).
- The W5 blend adds less to it (+0.014 against +0.026 for the S30 seeds): part of what W5 contributes is future
  context.
- The bottleneck sees the whole utterance, so the model is offline, like every system with W5.

**Order and Z7.** Part 3 was restarted on 10-20 at 00:30 to run the Z6 final first and to add **Z7**. The first start
had run 1 epoch of Z3_hp1.0_s1, which was moved to `logs/abl/_aborted/`.

- Z7 trains the two winners together: the bidirectional bottleneck plus weight 1.0 with the high-pass.
- Its rule was fixed before any of its results. It runs only if both the Z6 and the Z3 finals pass. It passes if its
  soup beats the Z3 soup in PESQ-NB (paired CI above 0, same guards). If it passes, it becomes the S30hp-track recipe
  for Z4.

**Z6 final** (10-20 00:29 → 02:46): the bidirectional S30 passes. Seeds 3.373 / 3.376 / 3.379 on val_unseen (S30:
3.349 / 3.330 / 3.353), so the seed spread is also a quarter of S30's. The blend weight picked on val_unseen is
w 0.15 (summed PESQ-WB of the 3 seeds). **New best system: bidirectional S30 soup + W5 (w 0.15), test 3.421**
(previous 3.375). The bidirectional soup alone, one 52.4M model, scores 3.399 on the test set.

| System | val_unseen | test | vctk_unseen | Params |
|---|---|---|---|---|
| S30 soup | 3.348 | 3.335 | 3.325 | 41.4M |
| Bidirectional S30 soup | 3.383 | 3.399 | 3.375 | 52.4M |
| Δ, PESQ-WB [95% CI] | +0.035 [+0.026, +0.042] | +0.064 [+0.054, +0.074] | +0.050 [+0.038, +0.062] | |
| S30 soup + W5 (w 0.25) | 3.373 | 3.375 | 3.365 | 60.9M |
| Bidirectional S30 soup + W5 (w 0.15) | **3.398** | **3.421** | **3.395** | 71.9M |
| Δ, PESQ-WB [95% CI] | +0.025 [+0.018, +0.032] | +0.046 [+0.038, +0.054] | +0.030 [+0.020, +0.040] | |

- The gain is not confined to PESQ. Without the blend, on the test set: PESQ-NB +0.053, SI-SDR +0.63 dB, STOI
  +0.003. On vctk_unseen: STOI +0.013 (76 of 80 speakers), SI-SDR +0.44 dB.
- **By speaker.** All 4 speakers of val_unseen and test improve, alone and with W5. On vctk_unseen, 64 of 80
  speakers improve in PESQ-WB alone (speaker-level CI [+0.037, +0.063]) and 65 of 80 with W5 ([+0.018, +0.042]).
  The worst speaker is −0.075 alone and −0.24 with W5.
- **By file.** Files more than 0.1 below the reference: 6.8% on test alone, 6.3% with W5. Files more than 0.1
  above: 33% and 25%.
- **Causality.** It is an offline model. The causal S30 soup (3.335, 3.323 when streaming with the scale prior,
  14.8) stays the streaming option.

**Z3 final** (10-20 02:46 → 05:50): S30hp with metric weight 1.0 passes. The blend weight picked on val_unseen is
w 0.35, by summed PESQ-NB; S30hp kept 0.25, picked by PESQ-WB in 14.9. Cells give PESQ-NB / PESQ-WB.

| System | val_unseen | test | vctk_unseen |
|---|---|---|---|
| S30hp soup | 3.812 / 3.295 | 3.825 / 3.280 | 3.827 / 3.256 |
| S30hp weight 1.0 soup | 3.831 / 3.330 | 3.848 / 3.316 | 3.848 / 3.294 |
| Δ PESQ-NB [95% CI] | +0.019 [+0.017, +0.021] | +0.023 [+0.021, +0.025] | +0.021 [+0.018, +0.025] |
| S30hp soup + W5 (w 0.25) | 3.847 / 3.320 | 3.861 / 3.326 | 3.861 / 3.300 |
| S30hp weight 1.0 soup + W5 (w 0.35) | 3.864 / 3.340 | 3.880 / 3.351 | 3.878 / 3.327 |

- **Other metrics.** PESQ-WB above 100 Hz gains +0.020 / +0.027 / +0.028. SI-SDR and STOI do not change (SI-SDR
  +0.01 to +0.05 dB). With the high-pass, a larger discriminator weight therefore improves the speech band without
  the SI-SDR cost that weight 1.0 had on the full band (−1.07 dB in W2).
- **By speaker.** All 4 speakers of val_unseen and test improve. On vctk_unseen, 77 of 80 speakers improve in
  PESQ-NB (speaker-level CI [+0.017, +0.027]).
- **Against the S30 track (test, PESQ-NB).** This system + W5 scores 3.880. The bidirectional S30 + W5 scores
  3.894 and S30 + W5 scores 3.862.
- Z7 (bidirectional + weight 1.0) started at 05:51.

**Z7 final** (10-20 05:51 → 09:15): the bidirectional bottleneck with metric weight 1.0 above 100 Hz passes. Seeds
3.862 / 3.858 / 3.866 in PESQ-NB on val_unseen (Z3: 3.826 / 3.822 / 3.834). The blend weight picked on val_unseen is
w 0.25 (summed PESQ-NB of the 3 seeds: 11.633 / 11.646 / 11.642 for w 0.15 / 0.25 / 0.35). Cells give PESQ-NB /
PESQ-WB.

| System | val_unseen | test | vctk_unseen |
|---|---|---|---|
| S30hp weight 1.0 soup (Z3) | 3.831 / 3.330 | 3.848 / 3.316 | 3.848 / 3.294 |
| Bidirectional S30hp weight 1.0 soup (Z7) | 3.868 / 3.346 | 3.897 / 3.378 | 3.881 / 3.333 |
| Δ PESQ-NB [95% CI] | +0.037 [+0.031, +0.043] | +0.049 [+0.042, +0.056] | +0.033 [+0.026, +0.040] |
| Z3 soup + W5 (w 0.35) | 3.864 / 3.340 | 3.880 / 3.351 | 3.878 / 3.327 |
| Z7 soup + W5 (w 0.25) | **3.886** / 3.354 | **3.913** / 3.388 | **3.900** / 3.352 |
| Bidirectional S30 soup + W5 (Z6, w 0.15) | 3.869 / **3.398** | 3.894 / **3.421** | 3.884 / **3.395** |

- **Against Z3.** As with Z6 over S30, the gain shows on every metric: SI-SDR +0.45 / +0.51 / +0.33 dB and STOI
  +0.006 / +0.003 / +0.013. All 4 speakers of val_unseen and test improve in PESQ-NB. On vctk_unseen, 69 of 80 do
  (speaker-level CI [+0.027, +0.039], worst −0.034), and STOI improves for 76 of 80.
- **Against Z6.** Z7 gains in the speech band and loses full-band PESQ-WB, the same split as S30hp against S30.
  - Soups: PESQ-NB +0.016 / +0.019 / +0.016 (72 of 80 speakers), PESQ-WB above 100 Hz +0.039 / +0.049 / +0.030,
    SI-SDR +0.14 / +0.11 / +0.10 dB (75 of 80), STOI unchanged; PESQ-WB −0.037 / −0.021 / −0.042 (21 of 80 speakers
    improve).
  - With W5: PESQ-NB +0.017 / +0.019 / +0.017 (74 of 80), SI-SDR +0.15 / +0.12 / +0.10 dB; PESQ-WB −0.044 / −0.033 /
    −0.043.
- **Systems after the finals** (both 71.9M and offline): the bidirectional S30 soup + W5 for the PESQ-WB benchmark
  (3.421), and the bidirectional S30hp weight-1.0 soup + W5 for use. On the test set the latter has the highest
  PESQ-NB (3.913), PESQ-WB above 100 Hz (3.531) and SI-SDR (19.32 dB) measured so far, and STOI 0.958 like Z6 + W5.
- **Z4** therefore runs on the two bidirectional recipes: `Z4_bi_e60` (against the Z6 seeds, PESQ-WB) from 09:15,
  then `Z4_bihp1.0_e60` (against the Z7 seeds, PESQ-NB).

**Z4 screens** (10-20 09:15 → 13:22): 60 epochs instead of 30 (the cosine schedule spans the 60 epochs), seed 1, on
both bidirectional recipes. val_unseen; the Δ are paired per file, with 95% CIs.

| Run | PESQ-WB | PESQ-NB | PESQ-WB > 100 Hz | SI-SDR (dB) | STOI |
|---|---|---|---|---|---|
| Bidirectional S30, mean of the 3 seeds at 30 epochs (Z6) | 3.376 | 3.846 | 3.473 | 18.86 | 0.9286 |
| Bidirectional S30 soup (Z6) | 3.383 | 3.852 | 3.480 | 18.90 | 0.9291 |
| Bidirectional S30, 60 epochs, seed 1 (`Z4_bi_e60_s1`) | **3.397** | 3.851 | 3.484 | 18.91 | 0.9282 |
| Bidirectional S30hp weight 1.0, mean of the 3 seeds at 30 epochs (Z7) | 3.329 | 3.862 | 3.509 | 18.97 | 0.9286 |
| Bidirectional S30hp weight 1.0 soup (Z7) | 3.346 | 3.868 | 3.519 | 19.04 | 0.9293 |
| Bidirectional S30hp weight 1.0, 60 epochs, seed 1 (`Z4_bihp1.0_e60_s1`) | 3.385 | **3.875** | **3.528** | 18.98 | 0.9284 |

- **Track A passes.** PESQ-WB +0.021 [+0.016, +0.026] over the mean of the Z6 seeds (rule: +0.015), +0.024 over the
  same seed at 30 epochs; PESQ-NB +0.005, above 100 Hz +0.011, SI-SDR +0.05 dB, STOI −0.0004 (guards ok). The single
  60-epoch model is above the 3-seed Z6 soup in PESQ-WB (+0.014 [+0.009, +0.020]) and level with it in PESQ-NB and
  SI-SDR, with STOI −0.0009. With W5 (w 0.25) it scores PESQ-WB 3.409 on val_unseen, against 3.398 for the
  benchmark system (Z6 soup + W5, w 0.15).
- **Track B does not pass.** PESQ-NB +0.013 [+0.010, +0.017] over the mean of the Z7 seeds, below the +0.015 of the
  rule (guards ok). It gains more in PESQ-WB (+0.056; the Z7 seeds spread over 3.299–3.355) and above 100 Hz
  (+0.020), with SI-SDR +0.01 dB. Against the Z7 soup: PESQ-NB +0.007 [+0.003, +0.011], PESQ-WB +0.038, SI-SDR
  −0.06 dB, STOI −0.0009. With W5 (w 0.25): PESQ-NB 3.894 against 3.886 for the Z7 soup + W5.
- **Training curves.** The PESQ that the metric discriminator measures on training batches still rises at the end
  (5-epoch means over epochs 46–60: 3.070 / 3.075 / 3.082 for track A, 3.174 / 3.178 / 3.184 for track B), by less
  than 0.01 per 5 epochs as the learning rate reaches its floor. The waveform loss falls the same way.
- The final of track A (`Z4_bi_e60`, seeds 2–3 at 60 epochs, soup, blend with w picked on val_unseen) started at
  13:22. Track B gets no final.

**Z4 final** (10-20 13:22 → 19:31): seeds 2–3 of the bidirectional S30 recipe at 60 epochs (seed 1 from the
screen), soup and blend. The trainings ended at 17:23. Part 3 then waited for GPU memory held by another user's
process (8.4 GB, GPU idle), so it was stopped at 19:18. Part 4 (`logs/abl/run_Z_p4.sh`: the same functions, with a
GPU check on free memory) ran the evaluations and the rule of track B. The soup passes the soup rule against the
30-epoch soup: PESQ-WB +0.023 [+0.018, +0.027] on val_unseen. The blend weight picked on val_unseen is w 0.15
(summed PESQ-WB of the 3 seeds: 10.243 / 10.238 / 10.210 for w 0.15 / 0.25 / 0.35). Cells give PESQ-WB / PESQ-NB.

| System | val_unseen | test | vctk_unseen | Params |
|---|---|---|---|---|
| Bidirectional S30 soup, 30 epochs (Z6) | 3.383 / 3.852 | 3.399 / 3.878 | 3.375 / 3.866 | 52.4M |
| Bidirectional S30 seeds, 60 epochs | 3.397–3.402 / 3.848–3.854 | 3.413–3.427 / 3.876–3.884 | 3.387–3.389 / 3.863–3.865 | 52.4M |
| Bidirectional S30 soup, 60 epochs (Z4) | 3.406 / 3.855 | 3.428 / 3.885 | 3.397 / 3.868 | 52.4M |
| Z6 soup + W5 (w 0.15) | 3.398 / 3.869 | 3.421 / 3.894 | 3.395 / 3.884 | 71.9M |
| **Z4 soup + W5 (w 0.15)** | **3.419** / 3.873 | **3.446** / 3.902 | **3.414** / 3.888 | 71.9M |
| Z7 soup + W5 (w 0.25) | 3.354 / **3.886** | 3.388 / **3.913** | 3.352 / **3.900** | 71.9M |

- **Against the 30-epoch soup (Z6).** PESQ-WB +0.023 / +0.029 / +0.022 (val_unseen / test / vctk_unseen; 65 of
  80 speakers improve). Above 100 Hz the gain is +0.011 / +0.018 / +0.008 and in PESQ-NB +0.004 / +0.008 / +0.002
  (the last not significant), so a third to two thirds of it lies below 100 Hz. SI-SDR +0.13 / +0.05 / −0.01 dB,
  STOI unchanged.
- **With W5.** Against Z6 + W5: PESQ-WB +0.021 / +0.025 / +0.019 (65 of 80 speakers), PESQ-NB +0.004 / +0.008 /
  +0.004, SI-SDR +0.11 / +0.04 / +0.00 dB. On the test set, CSIG 4.754, CBAK 3.878 and COVL 4.208 are the highest
  so far, and STOI is 0.958 as for Z6 + W5 and Z7 + W5.
- **One network above every earlier blend.** The 60-epoch soup alone (52.4M) scores 3.428 on the test set, against
  3.421 for Z6 + W5 (71.9M) and 3.375 for S30 + W5 (60.9M).
- **Soup over seeds.** On all three sets the soup gains PESQ-WB +0.008 and SI-SDR +0.11 dB over seed 1; the seeds
  spread over 0.002–0.014 in PESQ-WB.
- **Against Z7 + W5 (the system for use).** PESQ-WB +0.058 on the test set, but PESQ-NB −0.012, above 100 Hz
  −0.021 and SI-SDR −0.08 dB; on vctk_unseen, 18 of 80 speakers improve in PESQ-NB. The split between a benchmark
  system and a system for use stays.

**Conclusions of queue Z.**

1. **The bidirectional bottleneck is the largest gain of the queue** (Z6): on the test set, PESQ-WB +0.064 and
   SI-SDR +0.63 dB over S30, for 11.0M more parameters and no streaming. Unidirectional stage A weights load into
   both directions with the backward output projections zeroed, so no new pretraining is needed.
2. **60 epochs help the bidirectional recipe** (Z4): PESQ-WB +0.029 on the test set over 30 epochs, partly below
   100 Hz. On the weight-1.0 recipe the PESQ-NB gain (+0.013 for seed 1) stayed just below the rule.
3. **Metric weight 1.0 above 100 Hz improves the speech band** (Z3): PESQ-NB +0.023 over S30hp on the test set;
   with the bidirectional bottleneck (Z7), +0.049 more.
4. **Rejected:** output ensembles (Z1, +0.005 over the soups), the compressed-magnitude loss and metric weight 0.5
   (Z2), W5 fine-tuned through the blend (Z5), and the Z4 final of the weight-1.0 recipe.
5. **Systems:**
   - PESQ-WB benchmark: the 60-epoch bidirectional S30 soup + W5 (w 0.15), test 3.446, vctk_unseen 3.414 (71.9M,
     offline); as one network, the 60-epoch bidirectional S30 soup, 3.428 (52.4M, offline).
   - Use: the bidirectional S30hp weight-1.0 soup + W5 (w 0.25), test PESQ-NB 3.913, PESQ-WB above 100 Hz 3.531,
     SI-SDR 19.32 dB (71.9M, offline).
   - Streaming (41.4M): the S30 soup for PESQ-WB (3.335) and the S30hp weight-1.0 soup (Z3) for PESQ-NB (3.848).
     Z3 has not been evaluated in streaming yet.
6. **Not run:** item 7 (phase decoder) and the listening test (the page still has no answers). Candidates for a
   next queue: the final of 60 epochs on the weight-1.0 recipe (just below the rule), a streaming check of Z3, and
   longer training of the bidirectional recipe (the training PESQ still rose at epoch 60).
