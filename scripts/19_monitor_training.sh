#!/bin/bash
# Real-time training progress monitor for ablation experiments
# Usage: bash scripts/19_monitor_training.sh

PROJECT_DIR="/home/yf/soiling_project"
ABLATION_DIR="$PROJECT_DIR/baseline/runs/ablation_label_def"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Type 1: Label Definition Ablation Experiments             ║"
echo "║       Training Progress Monitor                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Severity targets and their names
declare -A TARGETS=(
    ["s"]="Simple Mean (Baseline)"
    ["S_op_only"]="Opacity-Aware Only"
    ["S_op_sp"]="Opacity + Spatial"
    ["S_full"]="Full Severity Score"
    ["S_full_eta00"]="Full (eta=0, no transparency discount)"
)

# Check each experiment
for target in "${!TARGETS[@]}"; do
    name="${TARGETS[$target]}"
    run_dir="$ABLATION_DIR/ablation_$target"

    if [ ! -d "$run_dir" ]; then
        status="⏳ Not started"
        epoch="-"
        loss="-"
        val_mae="-"
    else
        # Check if training is running
        if pgrep -f "ablation_$target" > /dev/null; then
            status="🔄 Training"
        elif [ -f "$run_dir/ckpt_best.pth" ]; then
            status="✅ Completed"
        else
            status="⚠️ Failed or stopped"
        fi

        # Get latest metrics
        if [ -f "$run_dir/metrics.csv" ]; then
            latest=$(tail -n 2 "$run_dir/metrics.csv" | head -n 1)
            if [ "$latest" != "epoch,train_loss,val_tile_mae,val_glob_mae,val_glob_rmse,val_gap_mae,rho_shat_sagg,rho_shat_gt" ]; then
                epoch=$(echo "$latest" | cut -d',' -f1)
                loss=$(echo "$latest" | cut -d',' -f2)
                val_mae=$(echo "$latest" | cut -d',' -f4)
            else
                epoch="-"
                loss="-"
                val_mae="-"
            fi
        fi
    fi

    printf "%-30s | %-15s | Epoch: %-3s | Loss: %-6s | Val MAE: %-6s\n" \
        "$name" "$status" "$epoch" "$loss" "$val_mae"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU info not available"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to exit. Refresh every 10 seconds..."
echo ""

# Auto-refresh mode
while true; do
    sleep 10
    clear

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       Type 1: Label Definition Ablation Experiments             ║"
    echo "║       Training Progress Monitor (Auto-refresh)                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    date "+%Y-%m-%d %H:%M:%S"
    echo ""

    for target in "${!TARGETS[@]}"; do
        name="${TARGETS[$target]}"
        run_dir="$ABLATION_DIR/ablation_$target"

        if [ ! -d "$run_dir" ]; then
            status="⏳ Not started"
            epoch="-"
            loss="-"
            val_mae="-"
        else
            if pgrep -f "global_target $target" > /dev/null; then
                status="🔄 Training"
            elif [ -f "$run_dir/ckpt_best.pth" ]; then
                status="✅ Completed"
            else
                status="⚠️ Failed"
            fi

            if [ -f "$run_dir/metrics.csv" ]; then
                latest=$(tail -n 2 "$run_dir/metrics.csv" | head -n 1)
                if [ "$latest" != "epoch,train_loss,val_tile_mae,val_glob_mae,val_glob_rmse,val_gap_mae,rho_shat_sagg,rho_shat_gt" ]; then
                    epoch=$(echo "$latest" | cut -d',' -f1)
                    loss=$(echo "$latest" | cut -d',' -f2)
                    val_mae=$(echo "$latest" | cut -d',' -f4)
                else
                    epoch="-"
                    loss="-"
                    val_mae="-"
                fi
            fi
        fi

        printf "%-30s | %-15s | Epoch: %-3s | Loss: %-6s | Val MAE: %-6s\n" \
            "$name" "$status" "$epoch" "$loss" "$val_mae"
    done

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU info not available"
    echo "══════════════════════════════════════════════════════════════"
done