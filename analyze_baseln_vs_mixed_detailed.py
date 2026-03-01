#!/usr/bin/env python3
"""
Baseline vs Mixed(989) 详细对比分析

分析两个模型在 WoodScape Test 和 External Test 上的详细表现
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from pathlib import Path

def load_test_real_predictions():
    """加载Test_Real预测结果"""
    baseline_pred = pd.read_csv('baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Real_predictions.csv')
    mixed_pred = pd.read_csv('baseline/runs/mixed_ws4000_sd989_baseline_filtered/eval_Test_Real_predictions.csv')

    baseline_metrics = json.load(open('baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Real_metrics.json'))
    mixed_metrics = json.load(open('baseline/runs/mixed_ws4000_sd989_baseline_filtered/eval_Test_Real_metrics.json'))

    return baseline_pred, mixed_pred, baseline_metrics, mixed_metrics

def load_test_ext_predictions():
    """加载Test_Ext预测结果"""
    baseline_pred = pd.read_csv('baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Ext_predictions.csv')
    mixed_pred = pd.read_csv('baseline/runs/mixed_ws4000_sd989_baseline_filtered/eval_Test_Ext_predictions.csv')

    baseline_metrics = json.load(open('baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Ext_metrics.json'))
    mixed_metrics = json.load(open('baseline/runs/mixed_ws4000_sd989_baseline_filtered/eval_Test_Ext_metrics.json'))

    return baseline_pred, mixed_pred, baseline_metrics, mixed_metrics

def analyze_error_distribution(predictions, model_name):
    """分析误差分布"""
    predictions['bias'] = predictions['S_hat'] - predictions['S_gt']
    predictions['abs_error'] = np.abs(predictions['bias'])

    print(f"\n  {model_name} 误差分布分析:")
    print(f"    误差均值: {predictions['abs_error'].mean():.6f}")
    print(f"    误差中位数: {predictions['abs_error'].median():.6f}")
    print(f"    误差标准差: {predictions['abs_error'].std():.6f}")
    print(f"    偏置均值: {predictions['bias'].mean():.6f}")
    print(f"    偏置中位数: {predictions['bias'].median():.6f}")

    # 误差分位数
    print(f"\n    误差分位数:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"      {p}%: {predictions['abs_error'].quantile(p/100):.6f}")

    # 高误差样本比例
    for threshold in [0.01, 0.02, 0.05, 0.10]:
        ratio = (predictions['abs_error'] > threshold).mean() * 100
        print(f"    误差 > {threshold}: {ratio:.2f}%")

    return predictions

def analyze_by_s_range(predictions, model_name):
    """按S值范围分析"""
    predictions['S_bin'] = pd.cut(predictions['S_gt'],
                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])

    print(f"\n  {model_name} 按S值区间分析:")
    for interval, group in predictions.groupby('S_bin'):
        n = len(group)
        s_mean = group['S_gt'].mean()
        error_mean = group['abs_error'].mean()
        bias_mean = group['bias'].mean()
        s_hat_mean = group['S_hat'].mean()
        print(f"    {interval}: n={n:4d}, S_gt={s_mean:.3f}, S_hat={s_hat_mean:.3f}, error={error_mean:.4f}, bias={bias_mean:+.4f}")

def compare_test_real():
    """对比Test_Real结果"""
    print("=" * 80)
    print("WoodScape Test Set (Test_Real) 详细对比")
    print("=" * 80)

    baseline_pred, mixed_pred, baseline_metrics, mixed_metrics = load_test_real_predictions()

    # 添加误差列
    baseline_pred = analyze_error_distribution(baseline_pred, "Baseline")
    mixed_pred = analyze_error_distribution(mixed_pred, "Mixed (989)")

    # 按S值范围分析
    baseline_pred = analyze_by_s_range(baseline_pred, "Baseline")
    mixed_pred = analyze_by_s_range(mixed_pred, "Mixed (989)")

    # 详细指标对比
    print("\n" + "=" * 80)
    print("详细指标对比")
    print("=" * 80)

    metrics_comparison = [
        ("Tile MAE", baseline_metrics['tile_mae'], mixed_metrics['tile_mae']),
        ("Global MAE", baseline_metrics['glob_mae'], mixed_metrics['glob_mae']),
        ("Global RMSE", baseline_metrics['glob_rmse'], mixed_metrics['glob_rmse']),
        ("Gap MAE (|S_hat - S_agg|)", baseline_metrics['gap_mae'], mixed_metrics['gap_mae']),
        ("ρ (S_hat vs S_agg)", baseline_metrics['rho_shat_sagg'], mixed_metrics['rho_shat_sagg']),
        ("ρ (S_hat vs S_gt)", baseline_metrics['rho_shat_gt'], mixed_metrics['rho_shat_gt']),
        ("Level Accuracy", baseline_metrics['level_accuracy'], mixed_metrics['level_accuracy']),
    ]

    print("\n  {:<30} {:>12} {:>12} {:>12}".format("指标", "Baseline", "Mixed(989)", "差异"))
    print("  " + "-" * 70)
    for name, base_val, mixed_val in metrics_comparison:
        diff = mixed_val - base_val
        indicator = "✓" if abs(diff) < 0.001 else ("↑" if diff > 0 else "↓")
        print(f"  {name:<30} {base_val:>12.6f} {mixed_val:>12.6f} {diff:>+11.6f} {indicator}")

    # Level accuracy 详细对比
    print("\n  Level Accuracy 详细对比:")
    print("  " + "-" * 70)
    print("  {:<20} {:>12} {:>12} {:>12}".format("Level", "Baseline", "Mixed(989)", "差异"))
    print("  " + "-" * 70)
    for i in range(1, 4):
        key = f'level_{i}_acc'
        base_val = baseline_metrics[key]
        mixed_val = mixed_metrics[key]
        diff = mixed_val - base_val
        print(f"  Level {i:<15} {base_val:>12.6f} {mixed_val:>12.6f} {diff:>+12.6f}")

def compare_test_ext():
    """对比Test_Ext结果"""
    print("\n\n" + "=" * 80)
    print("External Test Set (Test_Ext) 详细对比")
    print("=" * 80)

    baseline_pred, mixed_pred, baseline_metrics, mixed_metrics = load_test_ext_predictions()

    # Spearman ρ 对比
    print("\n  Spearman ρ:")
    print(f"    Baseline:     {baseline_metrics['spearman_rho']:.6f}")
    print(f"    Mixed (989):  {mixed_metrics['spearman_rho']:.6f}")
    print(f"    差异:         {mixed_metrics['spearman_rho'] - baseline_metrics['spearman_rho']:+.6f}")

    # 按等级分析
    print("\n  按等级分析 (S_hat 均值):")
    print("  " + "-" * 70)
    print("  {:<10} {:>15} {:>15} {:>15}".format("等级", "Baseline", "Mixed(989)", "差异"))
    print("  " + "-" * 70)
    for level in [1, 2, 3, 4, 5]:
        base_val = baseline_metrics['level_means'][str(level)]
        mixed_val = mixed_metrics['level_means'][str(level)]
        diff = mixed_val - base_val
        print(f"  Level {level:<6} {base_val:>15.6f} {mixed_val:>15.6f} {diff:>+15.6f}")

    # 单调性检查
    print(f"\n  单调性:")
    print(f"    Baseline:     {baseline_metrics['monotonic']}")
    print(f"    Mixed (989):  {mixed_metrics['monotonic']}")

    # 等级分离度分析
    base_levels = [baseline_metrics['level_means'][str(i)] for i in range(1, 6)]
    mixed_levels = [mixed_metrics['level_means'][str(i)] for i in range(1, 6)]

    print(f"\n  等级分离度 (相邻等级差距):")
    print("  " + "-" * 70)
    print("  {:<15} {:>15} {:>15}".format("相邻等级", "Baseline", "Mixed(989)"))
    print("  " + "-" * 70)
    for i in range(4):
        base_gap = base_levels[i+1] - base_levels[i]
        mixed_gap = mixed_levels[i+1] - mixed_levels[i]
        print(f"  Level {i+1}->{i+2:<6} {base_gap:>15.6f} {mixed_gap:>15.6f}")

def analyze_prediction_patterns():
    """分析预测模式差异"""
    print("\n\n" + "=" * 80)
    print("预测模式分析")
    print("=" * 80)

    baseline_pred, mixed_pred, _, _ = load_test_real_predictions()

    # 合并预测结果
    merged = pd.merge(
        baseline_pred[['image_path', 'S_hat', 'S_gt']].rename(columns={'S_hat': 'S_hat_baseline'}),
        mixed_pred[['image_path', 'S_hat']].rename(columns={'S_hat': 'S_hat_mixed'}),
        on='image_path'
    )

    merged['error_baseline'] = np.abs(merged['S_hat_baseline'] - merged['S_gt'])
    merged['error_mixed'] = np.abs(merged['S_hat_mixed'] - merged['S_gt'])
    merged['diff'] = merged['error_mixed'] - merged['error_baseline']

    # 统计哪个模型更好
    n_mixed_better = (merged['diff'] < 0).sum()
    n_baseline_better = (merged['diff'] > 0).sum()
    n_equal = (merged['diff'] == 0).sum()

    print(f"\n  样本级对比 (共{len(merged)}个样本):")
    print(f"    Mixed 更好:    {n_mixed_better:4d} ({n_mixed_better/len(merged)*100:.1f}%)")
    print(f"    Baseline 更好: {n_baseline_better:4d} ({n_baseline_better/len(merged)*100:.1f}%)")
    print(f"    相等:          {n_equal:4d} ({n_equal/len(merged)*100:.1f}%)")

    # 误差差异分布
    print(f"\n  误差差异分布:")
    print(f"    均值: {merged['diff'].mean():.6f}")
    print(f"    中位数: {merged['diff'].median():.6f}")
    print(f"    标准差: {merged['diff'].std():.6f}")

    # 找出Mixed显著更好的样本
    print(f"\n  Mixed显著更好的样本 (误差降低 > 0.02):")
    better_samples = merged[merged['diff'] < -0.02].sort_values('diff')
    print(f"    数量: {len(better_samples)}")
    if len(better_samples) > 0:
        print(f"    平均误差降低: {better_samples['diff'].mean():.6f}")
        print(f"    最大的5个改进:")
        for _, row in better_samples.head(5).iterrows():
            print(f"      {row['image_path']}: Baseline_error={row['error_baseline']:.4f}, Mixed_error={row['error_mixed']:.4f}, diff={row['diff']:.4f}")

    # 找出Baseline显著更好的样本
    print(f"\n  Baseline显著更好的样本 (误差降低 > 0.02):")
    worse_samples = merged[merged['diff'] > 0.02].sort_values('diff', ascending=False)
    print(f"    数量: {len(worse_samples)}")
    if len(worse_samples) > 0:
        print(f"    平均误差增加: {worse_samples['diff'].mean():.6f}")
        print(f"    最大的5个退化:")
        for _, row in worse_samples.head(5).iterrows():
            print(f"      {row['image_path']}: Baseline_error={row['error_baseline']:.4f}, Mixed_error={row['error_mixed']:.4f}, diff={row['diff']:.4f}")

def main():
    print("=" * 80)
    print("Baseline vs Mixed(989) 详细对比分析")
    print("=" * 80)

    compare_test_real()
    compare_test_ext()
    analyze_prediction_patterns()

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
