#!/usr/bin/env python3
"""
One-shot training progress checker for ablation experiments.
Shows current status without auto-refresh.

Usage: python scripts/20_check_progress.py
"""
import os
import time
from pathlib import Path

import pandas as pd


def check_training_status():
    """Check status of all ablation experiments."""
    proj = Path.home() / "soiling_project"
    ablation_dir = proj / "baseline/runs/ablation_label_def"

    targets = {
        "s": "Simple Mean (Baseline)",
        "S_op_only": "Opacity-Aware Only",
        "S_op_sp": "Opacity + Spatial",
        "S_full": "Full Severity Score",
        "S_full_eta00": "Full (eta=0, no transparency discount)",
    }

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Type 1: Label Definition Ablation Experiments              ║")
    print("║     Training Progress                                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    for target, name in targets.items():
        run_dir = ablation_dir / f"ablation_{target}"

        if not run_dir.exists():
            status = "⏳ Not started"
            epoch = "-"
            loss = "-"
            val_mae = "-"
            best_val_mae = "-"
        else:
            # Check if training is running
            result = os.popen(f"pgrep -f 'global_target {target}'").read()
            is_running = bool(result.strip())

            # Check for completion
            ckpt_best = run_dir / "ckpt_best.pth"
            if is_running:
                status = "🔄 Training"
            elif ckpt_best.exists():
                status = "✅ Completed"
            else:
                status = "⚠️ Failed or stopped"

            # Get latest metrics
            metrics_file = run_dir / "metrics.csv"
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        epoch = int(latest['epoch'])
                        loss = f"{latest['train_loss']:.4f}"
                        val_mae = f"{latest['val_glob_mae']:.4f}"
                        best_val_mae = f"{df['val_glob_mae'].min():.4f}"
                    else:
                        epoch = "-"
                        loss = "-"
                        val_mae = "-"
                        best_val_mae = "-"
                except Exception as e:
                    epoch = "-"
                    loss = f"Error: {e}"
                    val_mae = "-"
                    best_val_mae = "-"
            else:
                epoch = "-"
                loss = "-"
                val_mae = "-"
                best_val_mae = "-"

        print(f"{name:<45} | {status:<15} | Epoch: {epoch:<3} | Loss: {loss:<8} | Val MAE: {val_mae:<8} | Best: {best_val_mae}")

    print()
    print("══════════════════════════════════════════════════════════════")
    print("GPU Status:")
    gpu_info = os.popen("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null").read()
    if gpu_info.strip():
        print(f"  GPU: {gpu_info.strip()}")
    else:
        print("  GPU info not available")
    print("══════════════════════════════════════════════════════════════")
    print()
    print(f"Checked at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Commands:")
    print("  - Monitor continuously: watch -n 5 python scripts/20_check_progress.py")
    print("  - Run next experiment: bash scripts/16_run_ablation_experiments.sh")


if __name__ == "__main__":
    check_training_status()