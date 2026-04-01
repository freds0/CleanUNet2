#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# Executa todos os 33 experimentos de hyperparameter sweep do CleanUNet2.
#
# Uso:
#   bash run_all_experiments.sh            # roda tudo
#   bash run_all_experiments.sh --resume   # pula experimentos ja concluidos
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
EPOCHS=30
VAL_EVERY=5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/experiments/sweep.log"
RESULTS_FILE="${SCRIPT_DIR}/experiments/results_summary.csv"
RESUME_FLAG=""

if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--skip-existing"
    echo "[INFO] Modo resume: experimentos ja concluidos serao pulados."
fi

# ---------------------------------------------------------------------------
# Grupos na ordem de execucao
# ---------------------------------------------------------------------------
GROUPS=(
    "lr"
    "conditioning"
    "loss_type"
    "loss_weights"
    "stft_weights"
    "batch_size"
    "precision"
    "grad_clip"
    "combo"
)

# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    local msg="[$(timestamp)] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

separator() {
    echo "======================================================================"
}

# ---------------------------------------------------------------------------
# Preparacao
# ---------------------------------------------------------------------------
mkdir -p "${SCRIPT_DIR}/experiments"

log "============================================================"
log "  CleanUNet2 - Hyperparameter Sweep Completo"
log "  Epocas por experimento: ${EPOCHS}"
log "  Validacao a cada: ${VAL_EVERY} epocas"
log "  Inicio: $(timestamp)"
log "============================================================"

# Gera todos os configs antes de iniciar
log "Gerando configs para todos os experimentos..."
cd "$SCRIPT_DIR"
python run_experiments.py --generate-only --epochs "$EPOCHS" --val-every "$VAL_EVERY" 2>&1 | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Execucao grupo a grupo
# ---------------------------------------------------------------------------
TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_SKIP=0
FAILED_EXPS=()

for group in "${GROUPS[@]}"; do
    separator
    log ">>> GRUPO: ${group}"
    separator

    # Captura a saida para contar resultados
    set +e
    python run_experiments.py \
        --epochs "$EPOCHS" \
        --val-every "$VAL_EVERY" \
        --group "$group" \
        $RESUME_FLAG \
        2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [[ $exit_code -ne 0 ]]; then
        log "[AVISO] Grupo '${group}' terminou com erros (exit code: ${exit_code})"
        FAILED_EXPS+=("grupo:${group}")
        ((TOTAL_FAIL++)) || true
    else
        log "[OK] Grupo '${group}' concluido."
        ((TOTAL_OK++)) || true
    fi

    log ""
done

# ---------------------------------------------------------------------------
# Coleta de resultados finais
# ---------------------------------------------------------------------------
separator
log "Coletando resultados de todos os experimentos..."
separator

python run_experiments.py --collect-only 2>&1 | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
separator
log ""
log "============================================================"
log "  SWEEP FINALIZADO"
log "  Grupos OK:    ${TOTAL_OK} / ${#GROUPS[@]}"
log "  Grupos FAIL:  ${TOTAL_FAIL} / ${#GROUPS[@]}"
log "  Fim: $(timestamp)"
log "============================================================"

if [[ ${#FAILED_EXPS[@]} -gt 0 ]]; then
    log ""
    log "Experimentos/grupos com falha:"
    for f in "${FAILED_EXPS[@]}"; do
        log "  - $f"
    done
fi

log ""
log "Resultados em: ${RESULTS_FILE}"
log "Logs em:       ${LOG_FILE}"
log "TensorBoard:   tensorboard --logdir ${SCRIPT_DIR}/experiments"
