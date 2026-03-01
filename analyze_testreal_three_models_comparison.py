#!/usr/bin/env python3
"""分析WoodScape Test_Real各等级内部预测结果 - Baseline vs Mixed 989 vs Mixed 1260对比"""

import pandas as pd
import numpy as np
from scipy import stats

# 读取预测结果
baseline_df = pd.read_csv('baseline/runs/model1_r18_640x480/eval_Test_Real_predictions.csv')
mixed989_df = pd.read_csv('baseline/runs/mixed_ws4000_sd989_baseline_filtered/eval_Test_Real_predictions.csv')
mixed1260_df = pd.read_csv('baseline/runs/mixed_ws4000_sd1260_conservative_final/eval_Test_Real_predictions.csv')

# 添加global_level分组（基于S_gt）
def get_level(s):
    if s < 0.25:
        return 1
    elif s < 0.5:
        return 2
    elif s < 0.75:
        return 3
    else:
        return 4

baseline_df['level'] = baseline_df['S_gt'].apply(get_level)
mixed989_df['level'] = mixed989_df['S_gt'].apply(get_level)
mixed1260_df['level'] = mixed1260_df['S_gt'].apply(get_level)

# 重命名列
baseline_df = baseline_df.rename(columns={'S_hat': 'S_hat_baseline', 'S_agg': 'S_agg_baseline', 'error': 'error_baseline'})
mixed989_df = mixed989_df.rename(columns={'S_hat': 'S_hat_989', 'S_agg': 'S_agg_989', 'error': 'error_989'})
mixed1260_df = mixed1260_df.rename(columns={'S_hat': 'S_hat_1260', 'S_agg': 'S_agg_1260', 'error': 'error_1260'})

# 合并数据
merged = pd.merge(baseline_df[['image_path', 'level', 'S_gt', 'S_hat_baseline', 'S_agg_baseline', 'error_baseline']],
                  mixed989_df[['image_path', 'S_hat_989', 'S_agg_989', 'error_989']],
                  on='image_path')
merged = pd.merge(merged,
                  mixed1260_df[['image_path', 'S_hat_1260', 'S_agg_1260', 'error_1260']],
                  on='image_path')

print("=" * 110)
print("WoodScape Test_Real 各等级内部预测结果对比分析 (Baseline vs Mixed 989 vs Mixed 1260)")
print("=" * 110)

# 总体指标对比
print(f"\n【总体指标对比】")
print(f"{'指标':<25} {'Baseline':<15} {'Mixed 989':<15} {'Mixed 1260':<15}")
print("-" * 70)
b_mae_all = merged['error_baseline'].abs().mean()
m989_mae_all = merged['error_989'].abs().mean()
m1260_mae_all = merged['error_1260'].abs().mean()
print(f"{'Global MAE':<25} {b_mae_all:<15.4f} {m989_mae_all:<15.4f} {m1260_mae_all:<15.4f}")
print(f"{'Global RMSE':<25} {np.sqrt((merged['error_baseline']**2).mean()):<15.4f} {np.sqrt((merged['error_989']**2).mean()):<15.4f} {np.sqrt((merged['error_1260']**2).mean()):<15.4f}")
print(f"{'Spearman ρ (S_hat, S_gt)':<25} {merged['S_hat_baseline'].corr(merged['S_gt'], method='spearman'):<15.4f} {merged['S_hat_989'].corr(merged['S_gt'], method='spearman'):<15.4f} {merged['S_hat_1260'].corr(merged['S_gt'], method='spearman'):<15.4f}")
print(f"{'Pearson r (S_hat, S_gt)':<25} {merged['S_hat_baseline'].corr(merged['S_gt']):<15.4f} {merged['S_hat_989'].corr(merged['S_gt']):<15.4f} {merged['S_hat_1260'].corr(merged['S_gt']):<15.4f}")

# 偏差分析
print(f"\n【偏差分析 (S_hat - S_gt)】")
b_bias = merged['S_hat_baseline'].mean() - merged['S_gt'].mean()
m989_bias = merged['S_hat_989'].mean() - merged['S_gt'].mean()
m1260_bias = merged['S_hat_1260'].mean() - merged['S_gt'].mean()
print(f"{'Mean Bias':<25} {b_bias:<15.4f} {m989_bias:<15.4f} {m1260_bias:<15.4f}")

# 按等级分组分析
for level in sorted(merged['level'].unique()):
    level_data = merged[merged['level'] == level]

    # 计算S_gt范围
    s_min = level_data['S_gt'].min()
    s_max = level_data['S_gt'].max()
    s_mean = level_data['S_gt'].mean()

    print(f"\n{'='*110}")
    print(f"Level {level} (S_gt ∈ [{s_min:.3f}, {s_max:.3f}], mean={s_mean:.3f}, N={len(level_data)})")
    print("=" * 110)

    # MAE对比
    print(f"\n【预测误差对比】")
    print(f"{'指标':<12} {'Baseline':<14} {'Mixed 989':<14} {'Mixed 1260':<14} {'989-BL':<12} {'1260-BL':<12}")
    print("-" * 80)

    b_mae = level_data['error_baseline'].abs().mean()
    m989_mae = level_data['error_989'].abs().mean()
    m1260_mae = level_data['error_1260'].abs().mean()
    b_bias_l = (level_data['S_hat_baseline'] - level_data['S_gt']).mean()
    m989_bias_l = (level_data['S_hat_989'] - level_data['S_gt']).mean()
    m1260_bias_l = (level_data['S_hat_1260'] - level_data['S_gt']).mean()

    print(f"{'MAE':<12} {b_mae:<14.4f} {m989_mae:<14.4f} {m1260_mae:<14.4f} {m989_mae-b_mae:+.4f}   {m1260_mae-b_mae:+.4f}")
    print(f"{'Bias':<12} {b_bias_l:<14.4f} {m989_bias_l:<14.4f} {m1260_bias_l:<14.4f} {m989_bias_l-b_bias_l:+.4f}   {m1260_bias_l-b_bias_l:+.4f}")

    # S_hat统计对比
    print(f"\n【S_hat统计对比】")
    b_mean = level_data['S_hat_baseline'].mean()
    m989_mean = level_data['S_hat_989'].mean()
    m1260_mean = level_data['S_hat_1260'].mean()
    b_std = level_data['S_hat_baseline'].std()
    m989_std = level_data['S_hat_989'].std()
    m1260_std = level_data['S_hat_1260'].std()
    b_median = level_data['S_hat_baseline'].median()
    m989_median = level_data['S_hat_989'].median()
    m1260_median = level_data['S_hat_1260'].median()

    print(f"{'指标':<12} {'Baseline':<14} {'Mixed 989':<14} {'Mixed 1260':<14} {'989-BL':<12} {'1260-BL':<12}")
    print("-" * 80)
    print(f"{'Mean':<12} {b_mean:<14.4f} {m989_mean:<14.4f} {m1260_mean:<14.4f} {m989_mean-b_mean:+.4f}   {m1260_mean-b_mean:+.4f}")
    print(f"{'Std':<12} {b_std:<14.4f} {m989_std:<14.4f} {m1260_std:<14.4f} {m989_std-b_std:+.4f}   {m1260_std-b_std:+.4f}")
    print(f"{'Median':<12} {b_median:<14.4f} {m989_median:<14.4f} {m1260_median:<14.4f} {m989_median-b_median:+.4f}   {m1260_median-b_median:+.4f}")

    # S_gt统计
    print(f"\n【S_gt (ground truth) 统计】")
    print(f"Mean: {level_data['S_gt'].mean():.4f}")
    print(f"Std:  {level_data['S_gt'].std():.4f}")
    print(f"Min:  {level_data['S_gt'].min():.4f}")
    print(f"Max:  {level_data['S_gt'].max():.4f}")

    # 分位数对比
    print(f"\n【S_hat分位数分布对比】")
    print(f"{'分位数':<10} {'S_gt':<14} {'Baseline':<14} {'Mixed 989':<14} {'Mixed 1260':<14}")
    print("-" * 70)
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        gt_q = level_data['S_gt'].quantile(q)
        b_q = level_data['S_hat_baseline'].quantile(q)
        m989_q = level_data['S_hat_989'].quantile(q)
        m1260_q = level_data['S_hat_1260'].quantile(q)
        print(f"Q{int(q*100):<7} {gt_q:<14.4f} {b_q:<14.4f} {m989_q:<14.4f} {m1260_q:<14.4f}")

    # 预测范围分析
    print(f"\n【预测范围】")
    b_range = level_data['S_hat_baseline'].max() - level_data['S_hat_baseline'].min()
    m989_range = level_data['S_hat_989'].max() - level_data['S_hat_989'].min()
    m1260_range = level_data['S_hat_1260'].max() - level_data['S_hat_1260'].min()
    print(f"{'模型':<15} {'Min':<12} {'Max':<12} {'Range':<12}")
    print("-" * 50)
    print(f"{'S_gt (truth)':<15} {level_data['S_gt'].min():<12.4f} {level_data['S_gt'].max():<12.4f} {level_data['S_gt'].max()-level_data['S_gt'].min():<12.4f}")
    print(f"{'Baseline':<15} {level_data['S_hat_baseline'].min():<12.4f} {level_data['S_hat_baseline'].max():<12.4f} {b_range:<12.4f}")
    print(f"{'Mixed 989':<15} {level_data['S_hat_989'].min():<12.4f} {level_data['S_hat_989'].max():<12.4f} {m989_range:<12.4f}")
    print(f"{'Mixed 1260':<15} {level_data['S_hat_1260'].min():<12.4f} {level_data['S_hat_1260'].max():<12.4f} {m1260_range:<12.4f}")

    # 区间分布
    print(f"\n【预测值区间分布】")
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    gt_hist, _ = np.histogram(level_data['S_gt'], bins=bins)
    b_hist, _ = np.histogram(level_data['S_hat_baseline'], bins=bins)
    m989_hist, _ = np.histogram(level_data['S_hat_989'], bins=bins)
    m1260_hist, _ = np.histogram(level_data['S_hat_1260'], bins=bins)
    print(f"{'区间':<15} {'S_gt':<10} {'Baseline':<10} {'Mixed 989':<10} {'Mixed 1260':<10}")
    print("-" * 60)
    for i in range(len(bins)-1):
        print(f"[{bins[i]:.2f}, {bins[i+1]:.2f}){'':<7} {gt_hist[i]:<10} {b_hist[i]:<10} {m989_hist[i]:<10} {m1260_hist[i]:<10}")

    # 误差分布
    print(f"\n【误差绝对值分布】")
    bins_err = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
    b_err_hist, _ = np.histogram(level_data['error_baseline'].abs(), bins=bins_err)
    m989_err_hist, _ = np.histogram(level_data['error_989'].abs(), bins=bins_err)
    m1260_err_hist, _ = np.histogram(level_data['error_1260'].abs(), bins=bins_err)
    print(f"{'误差区间':<15} {'Baseline':<10} {'Mixed 989':<10} {'Mixed 1260':<10}")
    print("-" * 50)
    for i in range(len(bins_err)-1):
        print(f"[{bins_err[i]:.2f}, {bins_err[i+1]:.2f}){'':<7} {b_err_hist[i]:<10} {m989_err_hist[i]:<10} {m1260_err_hist[i]:<10}")

    # S_hat vs S_agg 一致性
    print(f"\n【S_hat vs S_agg 一致性】")
    b_gap = (level_data['S_hat_baseline'] - level_data['S_agg_baseline']).abs().mean()
    m989_gap = (level_data['S_hat_989'] - level_data['S_agg_989']).abs().mean()
    m1260_gap = (level_data['S_hat_1260'] - level_data['S_agg_1260']).abs().mean()
    b_rho = level_data['S_hat_baseline'].corr(level_data['S_agg_baseline'])
    m989_rho = level_data['S_hat_989'].corr(level_data['S_agg_989'])
    m1260_rho = level_data['S_hat_1260'].corr(level_data['S_agg_1260'])
    print(f"{'模型':<15} {'|S_hat - S_agg|':<18} {'corr(S_hat, S_agg)'}")
    print("-" * 50)
    print(f"{'Baseline':<15} {b_gap:<18.4f} {b_rho:.4f}")
    print(f"{'Mixed 989':<15} {m989_gap:<18.4f} {m989_rho:.4f}")
    print(f"{'Mixed 1260':<15} {m1260_gap:<18.4f} {m1260_rho:.4f}")

# 总体趋势对比
print(f"\n{'='*110}")
print("总体趋势对比")
print("=" * 110)

print(f"\n【各等级MAE趋势】")
print(f"{'Level':<8} {'Baseline':<12} {'Mixed 989':<12} {'Mixed 1260':<12} {'989-BL':<10} {'1260-BL':<10}")
print("-" * 70)
for level in sorted(merged['level'].unique()):
    level_data = merged[merged['level'] == level]
    b_mae = level_data['error_baseline'].abs().mean()
    m989_mae = level_data['error_989'].abs().mean()
    m1260_mae = level_data['error_1260'].abs().mean()
    s_mean = level_data['S_gt'].mean()
    print(f"{level:<8} ({s_mean:.2f}){b_mae:<12.4f} {m989_mae:<12.4f} {m1260_mae:<12.4f} {m989_mae-b_mae:+.4f}   {m1260_mae-b_mae:+.4f}")

print(f"\n【各等级Bias趋势】")
print(f"{'Level':<8} {'Baseline':<12} {'Mixed 989':<12} {'Mixed 1260':<12} {'989-BL':<10} {'1260-BL':<10}")
print("-" * 70)
for level in sorted(merged['level'].unique()):
    level_data = merged[merged['level'] == level]
    b_bias = (level_data['S_hat_baseline'] - level_data['S_gt']).mean()
    m989_bias = (level_data['S_hat_989'] - level_data['S_gt']).mean()
    m1260_bias = (level_data['S_hat_1260'] - level_data['S_gt']).mean()
    print(f"{level:<8} {b_bias:<12.4f} {m989_bias:<12.4f} {m1260_bias:<12.4f} {m989_bias-b_bias:+.4f}   {m1260_bias-b_bias:+.4f}")

print(f"\n【各等级S_hat均值趋势】")
print(f"{'Level':<8} {'S_gt_mean':<12} {'Baseline':<12} {'Mixed 989':<12} {'Mixed 1260':<12}")
print("-" * 60)
for level in sorted(merged['level'].unique()):
    level_data = merged[merged['level'] == level]
    s_mean = level_data['S_gt'].mean()
    b_mean = level_data['S_hat_baseline'].mean()
    m989_mean = level_data['S_hat_989'].mean()
    m1260_mean = level_data['S_hat_1260'].mean()
    print(f"{level:<8} {s_mean:<12.4f} {b_mean:<12.4f} {m989_mean:<12.4f} {m1260_mean:<12.4f}")

# 各等级样本数
print(f"\n【各等级样本数分布】")
for level in sorted(merged['level'].unique()):
    level_data = merged[merged['level'] == level]
    print(f"Level {level}: N={len(level_data)}")

print(f"\n{'='*110}")
