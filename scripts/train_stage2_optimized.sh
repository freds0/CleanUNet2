#!/bin/bash
# Script para treinar Stage-2 com configuração otimizada
# Stage-2: Training without X-Vectors/Wav2Vec2 (Latent Replication)

echo "=========================================="
echo "CleanUNet2 - Stage 2 Training (Optimized)"
echo "=========================================="
echo ""
echo "This script trains the model to replicate Stage-1 latents without speaker embeddings"
echo ""

# Ativa o ambiente virtual se necessário
# source /path/to/venv/bin/activate

# Define variáveis de ambiente para PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Configuração
CONFIG_FILE="configs/train_stage2_optimized.yaml"

# Verifica se o arquivo de configuração existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Verifica se os latents do Stage-1 existem
if [ ! -d "stored_latents_wav2vec2_stage1" ]; then
    echo "ERROR: Stage-1 latents not found!"
    echo "Please train Stage-1 first using: bash scripts/train_stage1_optimized.sh"
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo ""
echo "⚠️  IMPORTANT: Before running Stage-2, update the Stage-1 checkpoint path in the config file:"
echo "    stage1_checkpoint: experiments/exp_wav2vec2_xlsr2b_stage1/checkpoints/best-epoch-XX-valY.YYYY.ckpt"
echo ""
read -p "Have you updated the Stage-1 checkpoint path in $CONFIG_FILE? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please update the config file and run this script again."
    exit 1
fi

echo "Starting training..."
echo ""

# Executa o treinamento
python train_xvector.py \
    --config "$CONFIG_FILE" \
    --stage stage2

echo ""
echo "Training completed!"
echo "Model: Wav2Vec2 XLS-R 2B (without embeddings at inference)"
echo "Checkpoints saved to: experiments/exp_wav2vec2_xlsr2b_stage2/checkpoints"
echo ""
echo "To monitor training with TensorBoard:"
echo "  tensorboard --logdir experiments/exp_wav2vec2_xlsr2b_stage2"
echo ""
echo "Next step: Use the Stage-2 checkpoint for fast inference without speaker embeddings"
