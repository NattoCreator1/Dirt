#!/bin/bash
#
# LoRA Training Script for Lens Soiling Effects
#
# This script launches LoRA fine-tuning on Stable Diffusion 2.1
# using the training data prepared in Phase 2.
#
# Usage:
#   bash train_lora.sh

set -e

# ============================================================================
# Configuration
# ============================================================================

# Paths
TRAIN_DATA_DIR="/home/yf/soiling_project/sd_scripts/lora/training_data/dreambooth_format/train"
OUTPUT_DIR="/home/yf/soiling_project/sd_scripts/lora/output/$(date +%Y%m%d_%H%M%S)_lora_v1"
LOGS_DIR="/home/yf/soiling_project/sd_scripts/lora/logs"

# Model settings
PRETRAINED_MODEL="/home/yf/models/sd2_1_base"
IMAGE_SIZE=512

# Training hyperparameters (optimized for 3200 samples)
MAX_TRAIN_STEPS=3000          # ~1 epoch for 3200 samples with batch_size=4
BATCH_SIZE=4                   # Adjust based on GPU memory
GRADIENT_ACCUMULATION=1        # Effective batch size = 4
LEARNING_RATE=1e-4             # Standard LoRA learning rate
LR_SCHEDULER="constant"        # Constant LR with warmup
LR_WARMUP_STEPS=500            # 500 step warmup (~15% of training)

# LoRA settings
LORA_RANK=16                   # Rank for low-rank adaptation
LORA_ALPHA=32                  # Alpha scaling (typically 2x rank)
LORA_DROPOUT=0.1               # Dropout for regularization

# Checkpointing
CHECKPOINT_STEPS=500           # Save checkpoint every 500 steps
SAVE_STEPS=500
LOG_STEPS=50                   # Log every 50 steps

# Hardware
MIXED_PRECISION="fp16"         # Use fp16 for memory efficiency
SEED=42

# ============================================================================
# Training Script
# ============================================================================

echo "=========================================="
echo "LoRA Training for Lens Soiling Effects"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base Model:      $PRETRAINED_MODEL"
echo "  Training Data:   $TRAIN_DATA_DIR"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Image Size:      ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Max Steps:       $MAX_TRAIN_STEPS"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Learning Rate:   $LEARNING_RATE"
echo "  LoRA Rank:       $LORA_RANK"
echo "  LoRA Alpha:      $LORA_ALPHA"
echo "  Mixed Precision: $MIXED_PRECISION"
echo ""
echo "=========================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Check if training data exists
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "Error: Training data directory not found: $TRAIN_DATA_DIR"
    exit 1
fi

# Count training samples
NUM_SAMPLES=$(find "$TRAIN_DATA_DIR" -name "*.jpg" | wc -l)
echo "Found $NUM_SAMPLES training samples"

if [ "$NUM_SAMPLES" -eq 0 ]; then
    echo "Error: No training images found in $TRAIN_DATA_DIR"
    exit 1
fi

# Estimated training time
# For SD2.1 LoRA with batch_size=4 on typical GPU:
# ~0.5-1 second per step
ESTIMATED_TIME=$((MAX_TRAIN_STEPS / 2))  # Very rough estimate in seconds
ESTIMATED_MINUTES=$((ESTIMATED_TIME / 60))

echo ""
echo "Estimated training time: ~${ESTIMATED_MINUTES} minutes (varies by hardware)"
echo ""
echo "=========================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/training_config.txt" << EOF
LoRA Training Configuration
============================================

Base Model:        ${PRETRAINED_MODEL}
Training Data:     ${TRAIN_DATA_DIR}
Output Directory:  ${OUTPUT_DIR}
Image Size:        ${IMAGE_SIZE}x${IMAGE_SIZE}

Max Training Steps:        ${MAX_TRAIN_STEPS}
Batch Size:                ${BATCH_SIZE}
Gradient Accumulation:     ${GRADIENT_ACCUMULATION}
Learning Rate:             ${LEARNING_RATE}
LR Scheduler:              ${LR_SCHEDULER}
LR Warmup Steps:           ${LR_WARMUP_STEPS}

LoRA Rank:                 ${LORA_RANK}
LoRA Alpha:                ${LORA_ALPHA}
LoRA Dropout:              ${LORA_DROPOUT}

Mixed Precision:           ${MIXED_PRECISION}
Seed:                      ${SEED}

Number of Samples:         ${NUM_SAMPLES}
Steps per Epoch:           $((NUM_SAMPLES / BATCH_SIZE))

Timestamp:                 $(date -Iseconds)
EOF

# Activate accelerate config if not already configured
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "Note: Accelerate not configured. Using default single-GPU configuration."
fi

# Launch training
# Note: This requires the diffusers library with LoRA support
# Install with: pip install diffusers transformers accelerate peft

echo "Launching training..."
echo ""

accelerate launch --mixed_precision="$MIXED_PRECISION" \
    train_lora_diffusers.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --train_data_dir="$TRAIN_DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --image_size=$IMAGE_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="$LR_SCHEDULER" \
    --lr_warmup_steps=$LR_WARMUP_STEPS \
    --lora_rank=$LORA_RANK \
    --lora_alpha=$LORA_ALPHA \
    --lora_dropout=$LORA_DROPOUT \
    --checkpointing_steps=$CHECKPOINT_STEPS \
    --save_steps=$SAVE_STEPS \
    --logging_dir="$LOGS_DIR" \
    --seed=$SEED

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Checkpoints saved at steps:"
echo "  - ${CHECKPOINT_STEPS}, ${CHECKPOINT_STEPS}*2, ${CHECKPOINT_STEPS}*3, ..."
echo ""
echo "To use the trained LoRA:"
echo "  python -c \"from diffusers import StableDiffusionPipeline; import torch; pipe = StableDiffusionPipeline.from_pretrained('${PRETRAINED_MODEL}', torch_dtype=torch.float16); pipe.load_lora_weights('${OUTPUT_DIR}/checkpoint-${MAX_TRAIN_STEPS}')\""
echo ""
