#!/bin/bash
# Script to pre-extract Wav2Vec2 embeddings for both train and validation sets

echo "=========================================="
echo "Pre-extracting All Wav2Vec2 Embeddings"
echo "=========================================="
echo ""
echo "This script will extract embeddings for:"
echo "  1. Training set"
echo "  2. Validation set"
echo ""

# Extract training embeddings
echo "=========================================="
echo "Step 1/2: Extracting training embeddings"
echo "=========================================="
bash scripts/preextract_embeddings_train.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Training embedding extraction failed!"
    exit 1
fi

echo ""
echo ""

# Extract validation embeddings
echo "=========================================="
echo "Step 2/2: Extracting validation embeddings"
echo "=========================================="
bash scripts/preextract_embeddings_val.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Validation embedding extraction failed!"
    exit 1
fi

echo ""
echo ""
echo "=========================================="
echo "✓ All embeddings extracted successfully!"
echo "=========================================="
echo ""
echo "Cache directory: wav2vec2_embeddings_stage1_cache"
echo "Total embeddings: $(ls -1 wav2vec2_embeddings_stage1_cache/*.pt 2>/dev/null | wc -l)"
echo ""
echo "Next steps:"
echo "1. Start Stage 1 training:"
echo "   bash scripts/train_stage1_optimized.sh"
echo ""
echo "2. Or submit SLURM job:"
echo "   sbatch slurm_train_stage1_optimized.slurm"
echo ""
echo "Training will now load embeddings from cache instead of"
echo "extracting them on-the-fly, significantly speeding up training!"
