#!/bin/bash
#
# Test Script for LoRA on SD2.1 Inpainting
#
# This script tests the trained LoRA weights on SD2.1 Inpainting pipeline
# to generate realistic soiling effects with mask constraints.
#
# Usage:
#   bash test_lora_inpaint.sh [options]
#

set -e

# ============================================================================
# Configuration
# ============================================================================

# Paths
INPAINTING_MODEL="/home/yf/models/sd2_inpaint"
OUTPUT_BASE="/home/yf/soiling_project/sd_scripts/lora/inpainting_test_output"

# Find latest training output
LATEST_TRAINING=$(ls -td /home/yf/soiling_project/sd_scripts/lora/output/*_lora_v1 2>/dev/null | head -1)

if [ -z "$LATEST_TRAINING" ]; then
    echo "Error: No training output found!"
    echo "Please run LoRA training first."
    exit 1
fi

# Default to final checkpoint
LORA_PATH="$LATEST_TRAINING/checkpoint-3000"
if [ ! -d "$LORA_PATH" ]; then
    LORA_PATH="$LATEST_TRAINING/final_checkpoint"
fi

# Default input directories (user can override)
# For testing, we can use some clean samples from the dataset
CLEAN_IMAGE_DIR="/home/yf/soiling_project/sd_scripts/lora/training_data/clean_samples"
MASK_DIR="/home/yf/soiling_project/sd_scripts/lora/training_data/mask_samples"

# Generation settings (optimized for background preservation)
STRENGTH=0.35                    # Lower = preserve more original background
GUIDANCE_SCALE=5.5               # Moderate guidance
NUM_INFERENCE_STEPS=30           # Balanced quality/speed
LORA_SCALE=1.0

# Test configuration
SOILING_CLASS="auto"             # auto-detect from mask
SEVERITY="noticeable"            # noticeable or severe
NUM_SAMPLES=1

# Quality control
SAVE_SPILL_VIZ=true              # Save difference map
COMPUTE_QC_METRICS=true

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
        --clean-dir)
            CLEAN_IMAGE_DIR="$2"
            shift 2
            ;;
        --mask-dir)
            MASK_DIR="$2"
            shift 2
            ;;
        --strength)
            STRENGTH="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --lora-scale)
            LORA_SCALE="$2"
            shift 2
            ;;
        --class)
            SOILING_CLASS="$2"
            shift 2
            ;;
        --severity)
            SEVERITY="$2"
            shift 2
            ;;
        --all)
            SOILING_CLASS="all"
            SEVERITY="both"
            shift
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --no-spill-viz)
            SAVE_SPILL_VIZ=false
            shift
            ;;
        --no-qc)
            COMPUTE_QC_METRICS=false
            shift
            ;;
        --help)
            echo "Usage: bash test_lora_inpaint.sh [options]"
            echo ""
            echo "Test trained LoRA on SD2.1 Inpainting pipeline"
            echo ""
            echo "Options:"
            echo "  --checkpoint N     Use checkpoint at step N (default: 3000)"
            echo "  --clean-dir PATH   Directory with clean road images"
            echo "  --mask-dir PATH    Directory with corresponding masks"
            echo "  --strength N       Inpainting strength 0.0-1.0 (default: 0.35)"
            echo "  --guidance N       CFG scale (default: 5.5)"
            echo "  --steps N          Inference steps (default: 30)"
            echo "  --lora-scale N     LoRA adapter weight (default: 1.0)"
            echo "  --class C1/C2/C3   Soiling class (default: auto)"
            echo "  --severity mild/moderate/severe (default: noticeable)"
            echo "  --all              Test all classes and severities"
            echo "  --output PATH      Output directory (default: ./inpainting_test_output)"
            echo "  --no-spill-viz     Don't save spill visualization"
            echo "  --no-qc            Don't compute QC metrics"
            echo "  --help             Show this message"
            echo ""
            echo "Example:"
            echo "  bash test_lora_inpaint.sh --checkpoint 2500 --all"
            echo ""
            echo "Note: This tests the TWO-PHASE approach:"
            echo "  1. LoRA trained on SD2.1 Base (learns soiling appearance)"
            echo "  2. LoRA loaded on SD2.1 Inpainting (provides spatial constraint)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

echo "=========================================="
echo "LoRA + SD2.1 Inpainting Test"
echo "=========================================="
echo ""

# Check inpainting model
if [ ! -d "$INPAINTING_MODEL" ]; then
    echo "Error: Inpainting model not found: $INPAINTING_MODEL"
    exit 1
fi

# Check LoRA
if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA checkpoint not found: $LORA_PATH"
    exit 1
fi

# Check input directories
if [ ! -d "$CLEAN_IMAGE_DIR" ]; then
    echo "Warning: Clean image directory not found: $CLEAN_IMAGE_DIR"
    echo "Creating test data from training set..."
    # Create from training data
    mkdir -p "$CLEAN_IMAGE_DIR"
    mkdir -p "$MASK_DIR"
    # Copy a few samples
    find /home/yf/soiling_project/sd_scripts/lora/training_data/dreambooth_format/train -name "*.jpg" -type f | head -5 | while read f; do
        cp "$f" "$CLEAN_IMAGE_DIR/"
        base=$(basename "$f" .jpg)
        # Create a dummy mask if needed (or use original mask if available)
        if [ -f "/home/yf/soiling_project/sd_scripts/lora/training_data/masks/${base}.png" ]; then
            cp "/home/yf/soiling_project/sd_scripts/lora/training_data/masks/${base}.png" "$MASK_DIR/"
        fi
    done
fi

if [ -z "$(ls -A $CLEAN_IMAGE_DIR 2>/dev/null)" ]; then
    echo "Error: No clean images found in $CLEAN_IMAGE_DIR"
    exit 1
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "Configuration:"
echo "  Inpainting Model: $INPAINTING_MODEL"
echo "  LoRA Path:        $LORA_PATH"
echo "  LoRA Scale:       $LORA_SCALE"
echo "  Clean Images:     $CLEAN_IMAGE_DIR"
echo "  Masks:            $MASK_DIR"
echo ""
echo "Generation Settings:"
echo "  Strength:         $STRENGTH"
echo "  Guidance Scale:   $GUIDANCE_SCALE"
echo "  Inference Steps:  $NUM_INFERENCE_STEPS"
echo ""
echo "Test Config:"
echo "  Soiling Class:    $SOILING_CLASS"
echo "  Severity:         $SEVERITY"
echo "  QC Metrics:       $COMPUTE_QC_METRICS"
echo "  Spill Viz:        $SAVE_SPILL_VIZ"
echo ""
echo "=========================================="
echo ""

# ============================================================================
# Run Generation
# ============================================================================

cd /home/yf/soiling_project/sd_scripts/lora

python 05_test_lora_inpaint.py \
    --inpainting_model "$INPAINTING_MODEL" \
    --lora_path "$LORA_PATH" \
    --clean_image_dir "$CLEAN_IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_BASE" \
    --strength "$STRENGTH" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --lora_scale "$LORA_SCALE" \
    --soiling_class "$SOILING_CLASS" \
    --severity "$SEVERITY" \
    --num_samples "$NUM_SAMPLES" \
    $( [ "$SAVE_SPILL_VIZ" = true ] && echo "--save_spill_viz" ) \
    $( [ "$COMPUTE_QC_METRICS" = true ] && echo "--compute_qc_metrics" )

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo "Output saved to: $(ls -t $OUTPUT_BASE | head -1)"
echo ""
echo "View results:"
echo "  ls -la $OUTPUT_BASE/$(ls -t $OUTPUT_BASE | head -1)/"
echo ""
