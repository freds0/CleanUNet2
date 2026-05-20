#!/bin/bash
# Script to pre-extract Wav2Vec2 embeddings for training set

echo "=========================================="
echo "Pre-extracting Wav2Vec2 Embeddings (Train)"
echo "=========================================="
echo ""

# Configuration
DATA_DIR=""  # Paths in filelists are absolute
TRAIN_LIST="filelists/train.csv"
OUTPUT_DIR="wav2vec2_embeddings_stage1_cache"
MODEL="facebook/wav2vec2-xls-r-2b"
POOLING="self_attention"
BATCH_SIZE=16
NUM_WORKERS=8

# Check if file list exists
if [ ! -f "$TRAIN_LIST" ]; then
    echo "ERROR: Training file list not found: $TRAIN_LIST"
    exit 1
fi

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  File list: $TRAIN_LIST"
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
    --file_list "$TRAIN_LIST" \
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
    echo "Next steps:"
    echo "1. Extract validation embeddings:"
    echo "   bash scripts/preextract_embeddings_val.sh"
    echo ""
    echo "2. Start training with cached embeddings:"
    echo "   bash scripts/train_stage1_optimized.sh"
else
    echo ""
    echo "=========================================="
    echo "✗ Extraction failed with exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
