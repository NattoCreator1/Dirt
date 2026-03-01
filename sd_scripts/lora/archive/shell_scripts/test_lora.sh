#!/bin/bash
#
# Quick Test Script for Trained LoRA Model
#
# Usage:
#   bash test_lora.sh [options]
#

set -e

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_MODEL="/home/yf/models/sd2_1_base"
OUTPUT_BASE="/home/yf/soiling_project/sd_scripts/lora/test_output"

# Find latest training output
LATEST_TRAINING=$(ls -td /home/yf/soiling_project/sd_scripts/lora/output/*_lora_v1 2>/dev/null | head -1)

if [ -z "$LATEST_TRAINING" ]; then
    echo "Error: No training output found!"
    exit 1
fi

# Default to final checkpoint
LORA_PATH="$LATEST_TRAINING/checkpoint-3000"
if [ ! -d "$LORA_PATH" ]; then
    LORA_PATH="$LATEST_TRAINING/final_checkpoint"
fi

# Generation settings
NUM_SAMPLES=1
NUM_INFERENCE_STEPS=25
GUIDANCE_SCALE=7.5
LORA_SCALE=1.0

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_STEP="$2"
            LORA_PATH="$LATEST_TRAINING/checkpoint-$CHECKPOINT_STEP"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --lora-scale)
            LORA_SCALE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash test_lora.sh [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint N    Use checkpoint at step N (default: 3000)"
            echo "  --samples N        Generate N samples per prompt (default: 1)"
            echo "  --steps N          Number of inference steps (default: 25)"
            echo "  --guidance N       CFG scale (default: 7.5)"
            echo "  --lora-scale N     LoRA adapter weight (default: 1.0)"
            echo "  --output PATH      Output directory (default: ./test_output)"
            echo "  --help             Show this message"
            echo ""
            echo "Example:"
            echo "  bash test_lora.sh --checkpoint 2500 --samples 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$OUTPUT_BASE/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Display Configuration
# ============================================================================

echo "=========================================="
echo "LoRA Testing Configuration"
echo "=========================================="
echo ""
echo "Base Model:        $BASE_MODEL"
echo "LoRA Path:         $LORA_PATH"
echo "LoRA Scale:        $LORA_SCALE"
echo "Output Directory:  $OUTPUT_DIR"
echo ""
echo "Generation Settings:"
echo "  Samples:         $NUM_SAMPLES"
echo "  Inference Steps: $NUM_INFERENCE_STEPS"
echo "  Guidance Scale:  $GUIDANCE_SCALE"
echo ""
echo "=========================================="
echo ""

# ============================================================================
# Run Generation
# ============================================================================

cd /home/yf/soiling_project/sd_scripts/lora

python 04_test_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --lora_scale "$LORA_SCALE" \
    --width 512 \
    --height 512

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  ls -la $OUTPUT_DIR/"
echo ""
