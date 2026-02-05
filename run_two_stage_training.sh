#!/bin/bash
# run_two_stage_training.sh
# Automated script for running two-stage CleanUNet2 training with X-Vectors

set -e  # Exit on error

echo "=========================================="
echo "CleanUNet2 Two-Stage Training Pipeline"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if config files exist
STAGE1_CONFIG="configs/train_xvector_vanilla_stage1.yaml"
STAGE2_CONFIG="configs/train_xvector_vanilla_stage2.yaml"

if [ ! -f "$STAGE1_CONFIG" ]; then
    print_error "Stage-1 config not found: $STAGE1_CONFIG"
    exit 1
fi

if [ ! -f "$STAGE2_CONFIG" ]; then
    print_error "Stage-2 config not found: $STAGE2_CONFIG"
    exit 1
fi

# Parse command line arguments
STAGE="all"  # Default: run both stages
RESUME=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage <stage>    Run specific stage: 'stage1', 'stage2', or 'all' (default: all)"
            echo "  --resume           Resume training from last checkpoint"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Run both stages from scratch"
            echo "  $0 --stage stage1         # Run only Stage-1"
            echo "  $0 --stage stage2         # Run only Stage-2"
            echo "  $0 --stage stage1 --resume  # Resume Stage-1 training"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate stage argument
if [[ "$STAGE" != "all" && "$STAGE" != "stage1" && "$STAGE" != "stage2" ]]; then
    print_error "Invalid stage: $STAGE (must be 'stage1', 'stage2', or 'all')"
    exit 1
fi

# ========================================
# Stage-1: Training with X-Vectors
# ========================================

if [[ "$STAGE" == "all" || "$STAGE" == "stage1" ]]; then
    echo ""
    print_info "========================================"
    print_info "STAGE-1: Training with X-Vectors"
    print_info "========================================"
    echo ""

    # Check if resuming
    RESUME_FLAG=""
    if [ $RESUME -eq 1 ]; then
        LAST_CKPT="logs/stage1_checkpoints/last.ckpt"
        if [ -f "$LAST_CKPT" ]; then
            print_info "Resuming Stage-1 from: $LAST_CKPT"
            # Note: PyTorch Lightning handles resume via trainer config
            print_warning "To resume, uncomment 'resume_from_checkpoint' in $STAGE1_CONFIG"
        else
            print_warning "Last checkpoint not found. Starting from scratch."
        fi
    fi

    # Run Stage-1 training
    print_info "Starting Stage-1 training..."
    print_info "Config: $STAGE1_CONFIG"
    print_info "Logs: logs/stage1/"
    print_info "Checkpoints: logs/stage1_checkpoints/"
    print_info "Latents: stored_latents_stage1/"
    echo ""

    python train_xvector.py \
        --config "$STAGE1_CONFIG" \
        --stage stage1

    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        print_info "Stage-1 training completed successfully!"
    else
        print_error "Stage-1 training failed!"
        exit 1
    fi

    # Check if latents were saved
    LATENTS_DIR="stored_latents_stage1"
    if [ ! -d "$LATENTS_DIR" ] || [ -z "$(ls -A $LATENTS_DIR)" ]; then
        print_error "No latents found in $LATENTS_DIR"
        print_error "Stage-2 requires latents from Stage-1 validation"
        exit 1
    else
        LATENT_COUNT=$(ls -1 $LATENTS_DIR/val_batch_*.pt 2>/dev/null | wc -l)
        print_info "Found $LATENT_COUNT latent files in $LATENTS_DIR"
    fi

    echo ""
fi

# ========================================
# Stage-2: Training without X-Vectors
# ========================================

if [[ "$STAGE" == "all" || "$STAGE" == "stage2" ]]; then
    echo ""
    print_info "========================================"
    print_info "STAGE-2: Training without X-Vectors"
    print_info "========================================"
    echo ""

    # Verify Stage-1 checkpoint exists
    STAGE1_CKPT=$(grep "stage1_checkpoint:" "$STAGE2_CONFIG" | awk '{print $2}' | tr -d '"')

    if [ -z "$STAGE1_CKPT" ] || [ "$STAGE1_CKPT" == "null" ]; then
        print_error "Stage-2 config missing 'stage1_checkpoint' path"
        print_error "Please set stage1_checkpoint in $STAGE2_CONFIG"
        echo ""
        print_info "Example:"
        print_info "  stage1_checkpoint: \"logs/stage1_checkpoints/stage1-best.ckpt\""
        exit 1
    fi

    # Try to find best Stage-1 checkpoint automatically
    STAGE1_BEST=$(ls -t logs/stage1_checkpoints/stage1-best-*.ckpt 2>/dev/null | head -1)

    if [ ! -f "$STAGE1_CKPT" ]; then
        if [ -n "$STAGE1_BEST" ]; then
            print_warning "Configured checkpoint not found: $STAGE1_CKPT"
            print_info "Found alternative: $STAGE1_BEST"
            print_info "Please update $STAGE2_CONFIG with the correct path"
        else
            print_error "Stage-1 checkpoint not found: $STAGE1_CKPT"
            print_error "Please run Stage-1 first or provide valid checkpoint path"
        fi
        exit 1
    fi

    # Verify latents directory exists
    LATENTS_DIR="stored_latents_stage1"
    if [ ! -d "$LATENTS_DIR" ] || [ -z "$(ls -A $LATENTS_DIR)" ]; then
        print_error "Latents directory not found or empty: $LATENTS_DIR"
        print_error "Please run Stage-1 validation first to generate latents"
        exit 1
    fi

    # Check if resuming
    if [ $RESUME -eq 1 ]; then
        LAST_CKPT="logs/stage2_checkpoints/last.ckpt"
        if [ -f "$LAST_CKPT" ]; then
            print_info "Resuming Stage-2 from: $LAST_CKPT"
            print_warning "To resume, uncomment 'resume_from_checkpoint' in $STAGE2_CONFIG"
        else
            print_warning "Last checkpoint not found. Starting from scratch."
        fi
    fi

    # Run Stage-2 training
    print_info "Starting Stage-2 training..."
    print_info "Config: $STAGE2_CONFIG"
    print_info "Stage-1 checkpoint: $STAGE1_CKPT"
    print_info "Logs: logs/stage2/"
    print_info "Checkpoints: logs/stage2_checkpoints/"
    echo ""

    python train_xvector.py \
        --config "$STAGE2_CONFIG" \
        --stage stage2

    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        print_info "Stage-2 training completed successfully!"
    else
        print_error "Stage-2 training failed!"
        exit 1
    fi

    echo ""
fi

# ========================================
# Training Complete
# ========================================

echo ""
print_info "========================================"
print_info "Training Pipeline Complete!"
print_info "========================================"
echo ""

if [[ "$STAGE" == "all" || "$STAGE" == "stage2" ]]; then
    # Find best Stage-2 checkpoint
    STAGE2_BEST=$(ls -t logs/stage2_checkpoints/stage2-best-*.ckpt 2>/dev/null | head -1)

    if [ -n "$STAGE2_BEST" ]; then
        print_info "Best Stage-2 checkpoint: $STAGE2_BEST"
        echo ""
        print_info "To run inference:"
        echo "  1. Update configs/inference_xvector.yaml:"
        echo "     checkpoint_path: \"$STAGE2_BEST\""
        echo ""
        echo "  2. Run inference:"
        echo "     python inference_xvector.py --config configs/inference_xvector.yaml"
    fi
fi

echo ""
print_info "View training logs:"
echo "  tensorboard --logdir logs/"
echo ""

print_info "Checkpoints:"
if [[ "$STAGE" == "all" || "$STAGE" == "stage1" ]]; then
    echo "  Stage-1: logs/stage1_checkpoints/"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "stage2" ]]; then
    echo "  Stage-2: logs/stage2_checkpoints/"
fi
echo ""

exit 0
