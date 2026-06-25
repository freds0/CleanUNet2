#!/usr/bin/env bash
# ============================================================================
# train_all_fusion_options.sh
#
# Trains the 3 WavLM -> CleanUNet fusion options sequentially, each as the full
# two-stage pipeline (Stage-1 with WavLM -> Stage-2 latent distillation), for
# 100 epochs per stage (set in the configs). A one-time WavLM embedding
# extraction runs first.
#
#   Options: cross_attention_film | cvae_bottleneck | hierarchical_multiscale
#
# Usage:
#   bash train_all_fusion_options.sh                      # all three options
#   bash train_all_fusion_options.sh cvae_bottleneck      # a single option
#   DEVICE=cuda PYTHON=python bash train_all_fusion_options.sh
#
# Requirements:
#   - dataset at /raid/user_fredoliveira/DATASETS/VoiceBank-DEMAND-16k
#   - a CUDA GPU
#   - internet on first run (downloads microsoft/wavlm-large)
#
# Notes:
#   - The extraction step builds a shared ALL-layer WavLM cache (wavlm_embeddings_raw).
#     It is consumed only by configs with use_preextracted_embeddings: true.
#     The provided configs train on-the-fly (required by hierarchical_multiscale,
#     which needs the live per-layer extractor), so the cache is optional there —
#     flip use_preextracted_embeddings to true in the cross_attention/cvae configs
#     to reuse it for faster epochs.
# ============================================================================

set -euo pipefail

# Run from the repository root (directory of this script).
cd "$(dirname "$0")"

ALL_OPTIONS=("cross_attention_film" "cvae_bottleneck" "hierarchical_multiscale")
if [ "$#" -ge 1 ]; then
    OPTIONS=("$@")
else
    OPTIONS=("${ALL_OPTIONS[@]}")
fi

DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"

log() { echo -e "\n============================================================"; echo "  $*"; echo -e "============================================================\n"; }

# ----------------------------------------------------------------------------
# 1) Pre-extract WavLM embeddings ONCE (shared, dataset-dependent only).
# ----------------------------------------------------------------------------
log "Extracting WavLM embeddings (shared all-layer cache)"
"$PYTHON" extract_wavlm_embeddings.py \
    --config configs/stage1_cross_attention_film.yaml \
    --device "$DEVICE"

# ----------------------------------------------------------------------------
# 2) Train each fusion option: Stage-1 -> Stage-2.
# ----------------------------------------------------------------------------
for opt in "${OPTIONS[@]}"; do
    s1="configs/stage1_${opt}.yaml"
    s2="configs/stage2_${opt}.yaml"

    if [ ! -f "$s1" ] || [ ! -f "$s2" ]; then
        echo "[skip] missing config(s) for option '$opt' ($s1 / $s2)"
        continue
    fi

    log "[$opt] STAGE 1 (WavLM fusion)"
    "$PYTHON" train.py --config "$s1" --stage 1

    log "[$opt] STAGE 2 (latent distillation, no extractor)"
    "$PYTHON" train.py --config "$s2" --stage 2

    log "[$opt] COMPLETE"
done

log "All requested fusion options finished."
