#!/usr/bin/env python3
"""
Compare Model A and Model B on External Test Set

This script compares:
1. Spearman correlation with occlusion levels
2. Per-level prediction statistics
3. Distribution of S_hat and S_agg
4. Monotonicity analysis
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def load_eval_results(run_dir):
    """Load evaluation results from a run directory."""
    run_dir = Path(run_dir)

    # Load metrics
    metrics_path = run_dir / "eval_Test_Ext_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Load aggregated results (with multi-level column headers)
    agg_path = run_dir / "eval_Test_Ext_aggregated.csv"
    agg_df = pd.read_csv(agg_path, header=[0, 1], index_col=0)

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]

    # Load predictions (optional, for detailed analysis)
    pred_path = run_dir / "eval_Test_Ext_predictions.csv"
    pred_df = pd.read_csv(pred_path)

    return metrics, agg_df, pred_df


def print_comparison(metrics_a, agg_a, metrics_b, agg_b):
    """Print comparison between Model A and Model B."""

    print("=" * 80)
    print("External Test Set: Model A vs Model B 對比分析")
    print("=" * 80)
    print()

    # Overall metrics
    print("一、整體指標對比")
    print("-" * 80)
    print(f"{'指標':<30} {'Model A (有 L_cons)':<20} {'Model B (無 L_cons)':<20} {'差異 (B-A)'}")
    print("-" * 80)

    rho_a = metrics_a.get("spearman_rho", 0)
    rho_b = metrics_b.get("spearman_rho", 0)
    print(f"{'Spearman rho (S_hat vs level)':<30} {rho_a:<20.4f} {rho_b:<20.4f} {rho_b - rho_a:+.4f}")

    mono_a = metrics_a.get("monotonic", False)
    mono_b = metrics_b.get("monotonic", False)
    print(f"{'Monotonic (5>4>3>2>1)':<30} {str(mono_a):<20} {str(mono_b):<20} {str(mono_b != mono_a)}")
    print()

    # Per-level comparison
    print("二、按 Occlusion Level 的預測對比 (S_hat)")
    print("-" * 80)
    print(f"{'Level':<8} {'標籤':<16} {'樣本數':<10} {'Model A':<12} {'Model B':<12} {'差異 (B-A)'}")
    print("-" * 80)

    level_names = {
        5: "occlusion_a (最重)",
        4: "occlusion_b",
        3: "occlusion_c",
        2: "occlusion_d",
        1: "occlusion_e (最輕)",
    }

    for level in [5, 4, 3, 2, 1]:
        count = int(agg_a.loc[level, "S_hat_count"])
        mean_a = agg_a.loc[level, "S_hat_mean"]
        mean_b = agg_b.loc[level, "S_hat_mean"]
        diff = mean_b - mean_a
        name = level_names.get(level, f"level_{level}")
        print(f"{level:<8} {name:<16} {count:<10} {mean_a:<12.4f} {mean_b:<12.4f} {diff:+.4f}")
    print()

    # Check monotonicity
    print("三、單調性分析")
    print("-" * 80)

    def check_monotonic(agg_df, model_name):
        means = [agg_df.loc[l, "S_hat_mean"] for l in [5, 4, 3, 2, 1]]
        is_mono = all(means[i] >= means[i+1] for i in range(4))
        print(f"\n{model_name}:")
        print(f"  Level 5: {means[0]:.4f}")
        print(f"  Level 4: {means[1]:.4f} (diff: {means[1]-means[0]:+.4f})")
        print(f"  Level 3: {means[2]:.4f} (diff: {means[2]-means[1]:+.4f})")
        print(f"  Level 2: {means[3]:.4f} (diff: {means[3]-means[2]:+.4f})")
        print(f"  Level 1: {means[4]:.4f} (diff: {means[4]-means[3]:+.4f})")
        print(f"  單調性: {'✅ 滿足 5>4>3>2>1' if is_mono else '❌ 不滿足'}")
        return is_mono

    check_monotonic(agg_a, "Model A (有 L_cons)")
    print()
    check_monotonic(agg_b, "Model B (無 L_cons)")
    print()

    # S_agg comparison
    print("四、Aggregator 預測對比 (S_agg)")
    print("-" * 80)
    print(f"{'Level':<8} {'Model A':<12} {'Model B':<12} {'差異 (B-A)'}")
    print("-" * 80)

    for level in [5, 4, 3, 2, 1]:
        mean_a = agg_a.loc[level, "S_agg_mean"]
        mean_b = agg_b.loc[level, "S_agg_mean"]
        diff = mean_b - mean_a
        print(f"{level:<8} {mean_a:<12.4f} {mean_b:<12.4f} {diff:+.4f}")
    print()

    # Internal consistency (gap between S_hat and S_agg)
    print("五、內部一致性分析 (|S_hat - S_agg|)")
    print("-" * 80)

    for level in [5, 4, 3, 2, 1]:
        gap_a = abs(agg_a.loc[level, "S_hat_mean"] - agg_a.loc[level, "S_agg_mean"])
        gap_b = abs(agg_b.loc[level, "S_hat_mean"] - agg_b.loc[level, "S_agg_mean"])
        diff = gap_b - gap_a
        name = level_names.get(level, f"level_{level}")
        print(f"Level {level} ({name:<14}): Model A={gap_a:.4f}, Model B={gap_b:.4f}, 差異={diff:+.4f}")
    print()


def create_comparison_plots(agg_a, agg_b, out_dir):
    """Create comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("External Test: Model A vs Model B", fontsize=16)

    levels = [5, 4, 3, 2, 1]
    level_labels = ["5 (a)", "4 (b)", "3 (c)", "2 (d)", "1 (e)"]

    # Plot 1: S_hat comparison
    ax = axes[0, 0]
    x = np.arange(len(levels))
    width = 0.35

    s_hat_a = [agg_a.loc[l, "S_hat_mean"] for l in levels]
    s_hat_b = [agg_b.loc[l, "S_hat_mean"] for l in levels]

    bars_a = ax.bar(x - width/2, s_hat_a, width, label='Model A (有 L_cons)', alpha=0.8)
    bars_b = ax.bar(x + width/2, s_hat_b, width, label='Model B (無 L_cons)', alpha=0.8)

    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('S_hat Mean')
    ax.set_title('Global Head 預測對比')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 2: S_agg comparison
    ax = axes[0, 1]
    s_agg_a = [agg_a.loc[l, "S_agg_mean"] for l in levels]
    s_agg_b = [agg_b.loc[l, "S_agg_mean"] for l in levels]

    bars_a = ax.bar(x - width/2, s_agg_a, width, label='Model A', alpha=0.8)
    bars_b = ax.bar(x + width/2, s_agg_b, width, label='Model B', alpha=0.8)

    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('S_agg Mean')
    ax.set_title('Aggregator 預測對比')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 3: Difference (B - A)
    ax = axes[1, 0]
    diff_s_hat = [s_hat_b[i] - s_hat_a[i] for i in range(len(levels))]
    colors = ['green' if d >= 0 else 'red' for d in diff_s_hat]

    ax.bar(x, diff_s_hat, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Difference (Model B - Model A)')
    ax.set_title('S_hat 差異')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Standard deviation comparison
    ax = axes[1, 1]
    std_a = [agg_a.loc[l, "S_hat_std"] for l in levels]
    std_b = [agg_b.loc[l, "S_hat_std"] for l in levels]

    ax.plot(x, std_a, marker='o', label='Model A', linewidth=2, markersize=8)
    ax.plot(x, std_b, marker='s', label='Model B', linewidth=2, markersize=8)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('S_hat Std')
    ax.set_title('預測標準差對比 (越小越穩定)')
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "external_test_comparison.png", dpi=150, bbox_inches='tight')
    print(f"圖表已保存至: {out_dir / 'external_test_comparison.png'}")


def create_distribution_comparison(pred_a, pred_b, out_dir):
    """Create distribution comparison plots."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: S_hat distribution
    ax = axes[0]
    ax.hist(pred_a["S_hat"], bins=50, alpha=0.5, label='Model A', density=True)
    ax.hist(pred_b["S_hat"], bins=50, alpha=0.5, label='Model B', density=True)
    ax.set_xlabel('S_hat')
    ax.set_ylabel('Density')
    ax.set_title('S_hat 分布對比')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: S_agg distribution
    ax = axes[1]
    ax.hist(pred_a["S_agg"], bins=50, alpha=0.5, label='Model A', density=True)
    ax.hist(pred_b["S_agg"], bins=50, alpha=0.5, label='Model B', density=True)
    ax.set_xlabel('S_agg')
    ax.set_ylabel('Density')
    ax.set_title('S_agg 分布對比')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "external_test_distributions.png", dpi=150, bbox_inches='tight')
    print(f"分布圖已保存至: {out_dir / 'external_test_distributions.png'}")


def main():
    ap = argparse.ArgumentParser(description="Compare Model A and Model B on External Test")
    ap.add_argument("--run_a", default="baseline/runs/model1_r18_640x480",
                    help="Path to Model A run directory")
    ap.add_argument("--run_b", default="baseline/runs/model1_r18_NO_CONS",
                    help="Path to Model B run directory")
    ap.add_argument("--out_dir", default="docs/experiments",
                    help="Output directory for comparison results")
    ap.add_argument("--plots", action="store_true", help="Generate comparison plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # Load results
    print("Loading Model A results...")
    metrics_a, agg_a, pred_a = load_eval_results(args.run_a)
    print("Loading Model B results...")
    metrics_b, agg_b, pred_b = load_eval_results(args.run_b)
    print()

    # Print comparison
    print_comparison(metrics_a, agg_a, metrics_b, agg_b)

    # Create plots if requested
    if args.plots:
        print("\n生成對比圖表...")
        try:
            create_comparison_plots(agg_a, agg_b, out_dir)
            create_distribution_comparison(pred_a, pred_b, out_dir)
        except Exception as e:
            print(f"生成圖表時出錯: {e}")
            print("請確保已安裝 matplotlib 和 seaborn")

    # Save comparison summary to JSON
    summary = {
        "model_a_spearman_rho": metrics_a.get("spearman_rho", 0),
        "model_b_spearman_rho": metrics_b.get("spearman_rho", 0),
        "spearman_rho_diff": metrics_b.get("spearman_rho", 0) - metrics_a.get("spearman_rho", 0),
        "model_a_monotonic": metrics_a.get("monotonic", False),
        "model_b_monotonic": metrics_b.get("monotonic", False),
        "per_level_s_hat": {
            f"level_{l}": {
                "model_a": float(agg_a.loc[l, "S_hat_mean"]),
                "model_b": float(agg_b.loc[l, "S_hat_mean"]),
                "diff": float(agg_b.loc[l, "S_hat_mean"] - agg_a.loc[l, "S_hat_mean"]),
            } for l in [5, 4, 3, 2, 1]
        },
        "per_level_s_agg": {
            f"level_{l}": {
                "model_a": float(agg_a.loc[l, "S_agg_mean"]),
                "model_b": float(agg_b.loc[l, "S_agg_mean"]),
                "diff": float(agg_b.loc[l, "S_agg_mean"] - agg_a.loc[l, "S_agg_mean"]),
            } for l in [5, 4, 3, 2, 1]
        },
    }

    summary_path = out_dir / "external_test_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n對比摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()
