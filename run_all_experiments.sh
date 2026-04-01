#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/experiments"
cd "$SCRIPT_DIR"
RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--skip-existing"; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inicio do sweep" | tee -a experiments/sweep.log
python run_experiments.py --epochs 30 --val-every 5 $RESUME_FLAG 2>&1 | tee -a experiments/sweep.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SWEEP FINALIZADO" | tee -a experiments/sweep.log
echo "TensorBoard: tensorboard --logdir ${SCRIPT_DIR}/experiments"
