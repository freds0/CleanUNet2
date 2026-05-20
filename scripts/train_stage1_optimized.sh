#!/bin/bash
# Script para treinar Stage-1 com configuração otimizada
# Stage-1: Training with X-Vectors/Wav2Vec2

echo "=========================================="
echo "CleanUNet2 - Stage 1 Training (Optimized)"
echo "=========================================="
echo ""
echo "This script trains the model with speaker embeddings (X-Vectors or Wav2Vec2)"
echo ""

# Ativa o ambiente virtual se necessário
# source /path/to/venv/bin/activate

# Define variáveis de ambiente para PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ==========================================
# Pre-extract Embeddings (if needed)
# ==========================================
echo "Checking for cached Wav2Vec2 embeddings..."

EMBEDDING_CACHE_DIR="wav2vec2_embeddings_stage1_cache"
EXISTING_EMBEDDINGS=$(ls -1 "$EMBEDDING_CACHE_DIR"/*.pt 2>/dev/null | wc -l)

if [ "$EXISTING_EMBEDDINGS" -gt 0 ]; then
    echo "✓ Found $EXISTING_EMBEDDINGS cached embeddings"
    echo "  Training will use cached embeddings for faster performance"
    echo ""
else
    echo "No cached embeddings found."
    echo ""
    echo "⚠️  RECOMMENDATION: Pre-extract embeddings for faster training"
    echo "   Run: bash scripts/preextract_all_embeddings.sh"
    echo ""
    read -p "Continue without cached embeddings? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting. Please extract embeddings first."
        exit 0
    fi
    echo "Continuing with on-the-fly embedding extraction (slower)..."
    echo ""
fi

# Configuração
CONFIG_FILE="configs/train_stage1_optimized.yaml"

# Verifica se o arquivo de configuração existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo "Starting training..."
echo ""

# Executa o treinamento
python train_xvector.py \
    --config "$CONFIG_FILE" \
    --stage stage1

echo ""
echo "Training completed!"
echo "Model: Wav2Vec2 XLS-R 2B (facebook/wav2vec2-xls-r-2b)"
echo "Checkpoints saved to: experiments/exp_wav2vec2_xlsr2b_stage1/checkpoints"
echo "Latents saved to: stored_latents_wav2vec2_stage1"
echo ""
echo "To monitor training with TensorBoard:"
echo "  tensorboard --logdir experiments/exp_wav2vec2_xlsr2b_stage1"
echo ""
echo "Next step: Train Stage-2 using the best checkpoint from Stage-1"
echo "  bash scripts/train_stage2_optimized.sh"
