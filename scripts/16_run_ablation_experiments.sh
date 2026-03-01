#!/bin/bash
# Run label definition ablation experiments for Severity Score
# This script trains baseline models with different severity targets to validate the design
# Experiments run sequentially - each waits for the previous to complete

set -e  # Exit on error

PROJECT_DIR="/home/yf/soiling_project"
cd "$PROJECT_DIR"

# Common arguments
INDEX_CSV="dataset/woodscape_processed/meta/labels_index_ablation.csv"
IMG_ROOT="."
LABELS_TILE_DIR="dataset/woodscape_processed/labels_tile"
TRAIN_SPLIT="train"
VAL_SPLIT="val"
TEST_SPLIT="test"

# Training hyperparameters (same for all experiments)
EPOCHS=40
BATCH_SIZE=32
LR=3e-4
WEIGHT_DECAY=1e-2
LAMBDA_GLOB=1.0
MU_CONS=0.1  # Use consistency loss
SEED=42
NUM_WORKERS=4

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Type 1: Label Definition Ablation Experiments              ║"
echo "║     Sequential Training (each waits for previous)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to run a single experiment
run_experiment() {
    local target=$1
    local name=$2
    local num=$3

    echo "[$num/5] Training: $name"
    echo "      Target: $target"
    echo ""

    PYTHONPATH=/home/yf/soiling_project:$PYTHONPATH python baseline/train_baseline.py \
        --index_csv "$INDEX_CSV" \
        --img_root "$IMG_ROOT" \
        --labels_tile_dir "$LABELS_TILE_DIR" \
        --train_split "$TRAIN_SPLIT" \
        --val_split "$VAL_SPLIT" \
        --global_target "$target" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --lambda_glob $LAMBDA_GLOB \
        --mu_cons $MU_CONS \
        --seed $SEED \
        --num_workers $NUM_WORKERS \
        --run_name "ablation_$target" \
        --out_root "baseline/runs/ablation_label_def"

    echo ""
    echo "✅ [$num/5] Completed: $name"
    echo "────────────────────────────────────────────────────────────────"
    echo ""
}

# Run experiments sequentially
run_experiment "s" "Simple Mean (Baseline)" "1"
run_experiment "S_op_only" "Opacity-Aware Only" "2"
run_experiment "S_op_sp" "Opacity + Spatial (no dominance)" "3"
run_experiment "S_full" "Full Severity Score (eta=0.9)" "4"
run_experiment "S_full_eta00" "Full Severity Score (eta=0)" "5"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     All ablation experiments completed!                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: baseline/runs/ablation_label_def/"
echo ""
echo "Next steps:"
echo "  1. Evaluate results:"
echo "     python scripts/17_evaluate_ablation_results.py"
echo ""
echo "  2. Generate comparison table"
echo ""
echo "  3. Analyze to validate Severity Score design"
echo ""