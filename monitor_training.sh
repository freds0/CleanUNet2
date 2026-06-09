#!/bin/bash

# Script para monitorar o progresso do treinamento

OUTPUT_FILE="$1"

if [ -z "$OUTPUT_FILE" ]; then
    echo "Uso: $0 <arquivo_de_output>"
    exit 1
fi

echo "=== Monitorando Treinamento ==="
echo ""

# Verificar se o arquivo existe
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Arquivo não encontrado: $OUTPUT_FILE"
    exit 1
fi

# Extrair informações de época
echo "--- Progresso das Épocas ---"
grep -E "Epoch [0-9]+:" "$OUTPUT_FILE" | tail -10

echo ""
echo "--- Últimas Métricas de Validação ---"
grep -E "(val_loss|val/pesq|val/stoi|val/si_sdr)" "$OUTPUT_FILE" | tail -20

echo ""
echo "--- Últimas Linhas do Log ---"
tail -30 "$OUTPUT_FILE"
