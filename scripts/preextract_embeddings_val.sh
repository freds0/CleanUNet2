#!/bin/bash
# Script to pre-extract Wav2Vec2 embeddings for validation set

echo "=========================================="
echo "Pre-extracting Wav2Vec2 Embeddings (Val)"
echo "=========================================="
echo ""

# Configuration
DATA_DIR=""  # Paths in filelists are absolute
VAL_LIST="filelists/test.csv"
OUTPUT_DIR="wav2vec2_embeddings_stage1_cache"  # Same directory as training
MODEL="facebook/wav2vec2-xls-r-2b"
POOLING="self_attention"
BATCH_SIZE=16
NUM_WORKERS=8

# Check if file list exists
if [ ! -f "$VAL_LIST" ]; then
    echo "ERROR: Validation file list not found: $VAL_LIST"
    exit 1
fi

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  File list: $VAL_LIST"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL"
echo "  Pooling: $POOLING"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting extraction..."
echo ""

# Run extraction
python scripts/preextract_wav2vec2_embeddings.py \
    --data_dir "$DATA_DIR" \
    --file_list "$VAL_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --pooling_method "$POOLING" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Extraction completed successfully!"
    echo "=========================================="
    echo ""
    echo "Embeddings saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary:"
    echo "  Total cached embeddings: $(ls -1 $OUTPUT_DIR/*.pt 2>/dev/null | wc -l)"
    echo ""
    echo "Next steps:"
    echo "1. Start training with cached embeddings:"
    echo "   bash scripts/train_stage1_optimized.sh"
    echo ""
    echo "2. Or submit SLURM job:"
    echo "   sbatch slurm_train_stage1_optimized.slurm"
else
    echo ""
    echo "=========================================="
    echo "✗ Extraction failed with exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
