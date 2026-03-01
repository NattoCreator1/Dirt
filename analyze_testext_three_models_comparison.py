#!/usr/bin/env python3
"""分析Test_Ext各等级内部预测结果 - Baseline vs Mixed 989 vs Mixed 1260对比"""

import pandas as pd
import numpy as np
from scipy import stats

# 读取预测结果
baseline_df = pd.read_csv('baseline/runs/model1_r18_640x480/eval_Test_Ext_predictions.csv')
mixed989_df = pd.read_csv('baseline/runs/mixed_ws4000_sd989_baseline_filtered/eval_Test_Ext_predictions.csv')
mixed1260_df = pd.read_csv('baseline/runs/mixed_ws4000_sd1260_conservative_final/eval_Test_Ext_predictions.csv')

# 重命名列
baseline_df = baseline_df.rename(columns={'S_hat': 'S_hat_baseline', 'S_agg': 'S_agg_baseline'})
mixed989_df = mixed989_df.rename(columns={'S_hat': 'S_hat_989', 'S_agg': 'S_agg_989'})
mixed1260_df = mixed1260_df.rename(columns={'S_hat': 'S_hat_1260', 'S_agg': 'S_agg_1260'})

# 合并数据
merged = pd.merge(baseline_df[['image_path', 'occlusion_level', 'S_hat_baseline', 'S_agg_baseline']],
                  mixed989_df[['image_path', 'S_hat_989', 'S_agg_989']],
                  on='image_path')
merged = pd.merge(merged,
                  mixed1260_df[['image_path', 'S_hat_1260', 'S_agg_1260']],
                  on='image_path')

print("=" * 100)
print("Test_Ext 各等级内部预测结果对比分析 (Baseline vs Mixed 989 vs Mixed 1260)")
print("=" * 100)

# 按等级分组分析
for level in sorted(merged['occlusion_level'].unique()):
    level_data = merged[merged['occlusion_level'] == level]

    print(f"\n{'='*100}")
    print(f"Level {level} (N={len(level_data)})")
    print(f"{'='*100}")

    # 基本统计对比
    print(f"\n【S_hat统计对比】")
    print(f"{'指标':<12} {'Baseline':<14} {'Mixed 989':<14} {'Mixed 1260':<14} {'989-BL':<12} {'1260-BL':<12}")
    print(f"{'-'*80}")

    b_mean = level_data['S_hat_baseline'].mean()
    m989_mean = level_data['S_hat_989'].mean()
    m1260_mean = level_data['S_hat_1260'].mean()
    b_std = level_data['S_hat_baseline'].std()
    m989_std = level_data['S_hat_989'].std()
    m1260_std = level_data['S_hat_1260'].std()
    b_median = level_data['S_hat_baseline'].median()
    m989_median = level_data['S_hat_989'].median()
    m1260_median = level_data['S_hat_1260'].median()

    print(f"{'Mean':<12} {b_mean:<14.4f} {m989_mean:<14.4f} {m1260_mean:<14.4f} {m989_mean-b_mean:+.4f}   {m1260_mean-b_mean:+.4f}")
    print(f"{'Std':<12} {b_std:<14.4f} {m989_std:<14.4f} {m1260_std:<14.4f} {m989_std-b_std:+.4f}   {m1260_std-b_std:+.4f}")
    print(f"{'Median':<12} {b_median:<14.4f} {m989_median:<14.4f} {m1260_median:<14.4f} {m989_median-b_median:+.4f}   {m1260_median-b_median:+.4f}")
    print(f"{'Min':<12} {level_data['S_hat_baseline'].min():<14.4f} {level_data['S_hat_989'].min():<14.4f} {level_data['S_hat_1260'].min():<14.4f} {level_data['S_hat_989'].min()-level_data['S_hat_baseline'].min():+.4f}   {level_data['S_hat_1260'].min()-level_data['S_hat_baseline'].min():+.4f}")
    print(f"{'Max':<12} {level_data['S_hat_baseline'].max():<14.4f} {level_data['S_hat_989'].max():<14.4f} {level_data['S_hat_1260'].max():<14.4f} {level_data['S_hat_989'].max()-level_data['S_hat_baseline'].max():+.4f}   {level_data['S_hat_1260'].max()-level_data['S_hat_baseline'].max():+.4f}")

    # 分位数对比
    print(f"\n【分位数分布对比】")
    print(f"{'分位数':<10} {'Baseline':<14} {'Mixed 989':<14} {'Mixed 1260':<14}")
    print(f"{'-'*55}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        b_q = level_data['S_hat_baseline'].quantile(q)
        m989_q = level_data['S_hat_989'].quantile(q)
        m1260_q = level_data['S_hat_1260'].quantile(q)
        print(f"Q{int(q*100):<7} {b_q:<14.4f} {m989_q:<14.4f} {m1260_q:<14.4f}")

    # 预测范围分析
    print(f"\n【预测范围】")
    b_range = level_data['S_hat_baseline'].max() - level_data['S_hat_baseline'].min()
    m989_range = level_data['S_hat_989'].max() - level_data['S_hat_989'].min()
    m1260_range = level_data['S_hat_1260'].max() - level_data['S_hat_1260'].min()
    print(f"Baseline:   [{level_data['S_hat_baseline'].min():.4f}, {level_data['S_hat_baseline'].max():.4f}] (跨度={b_range:.4f})")
    print(f"Mixed 989:  [{level_data['S_hat_989'].min():.4f}, {level_data['S_hat_989'].max():.4f}] (跨度={m989_range:.4f})")
    print(f"Mixed 1260: [{level_data['S_hat_1260'].min():.4f}, {level_data['S_hat_1260'].max():.4f}] (跨度={m1260_range:.4f})")

    # 区间分布
    print(f"\n【预测值区间分布】")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    b_hist, _ = np.histogram(level_data['S_hat_baseline'], bins=bins)
    m989_hist, _ = np.histogram(level_data['S_hat_989'], bins=bins)
    m1260_hist, _ = np.histogram(level_data['S_hat_1260'], bins=bins)
    print(f"{'区间':<12} {'Baseline':<12} {'Mixed 989':<12} {'Mixed 1260':<12}")
    print(f"{'-'*50}")
    for i in range(len(bins)-1):
        print(f"[{bins[i]:.1f}, {bins[i+1]:.1f}){'':<5} {b_hist[i]:<12} {m989_hist[i]:<12} {m1260_hist[i]:<12}")

    # S_hat vs S_agg 一致性
    print(f"\n【S_hat vs S_agg 一致性】")
    b_gap = (level_data['S_hat_baseline'] - level_data['S_agg_baseline']).abs().mean()
    m989_gap = (level_data['S_hat_989'] - level_data['S_agg_989']).abs().mean()
    m1260_gap = (level_data['S_hat_1260'] - level_data['S_agg_1260']).abs().mean()
    b_rho = level_data['S_hat_baseline'].corr(level_data['S_agg_baseline'])
    m989_rho = level_data['S_hat_989'].corr(level_data['S_agg_989'])
    m1260_rho = level_data['S_hat_1260'].corr(level_data['S_agg_1260'])
    print(f"{'模型':<12} {'|S_hat - S_agg|':<18} {'corr(S_hat, S_agg)'}")
    print(f"{'-'*50}")
    print(f"{'Baseline':<12} {b_gap:<18.4f} {b_rho:.4f}")
    print(f"{'Mixed 989':<12} {m989_gap:<18.4f} {m989_rho:.4f}")
    print(f"{'Mixed 1260':<12} {m1260_gap:<18.4f} {m1260_rho:.4f}")

# 总体趋势对比
print(f"\n{'='*100}")
print("总体趋势对比")
print(f"{'='*100}")

print(f"\n【各等级S_hat均值趋势】")
print(f"{'Level':<8} {'Baseline':<12} {'Mixed 989':<12} {'Mixed 1260':<12} {'989-BL':<10} {'1260-BL':<10}")
print(f"{'-'*70}")
for level in sorted(merged['occlusion_level'].unique()):
    level_data = merged[merged['occlusion_level'] == level]
    b_mean = level_data['S_hat_baseline'].mean()
    m989_mean = level_data['S_hat_989'].mean()
    m1260_mean = level_data['S_hat_1260'].mean()
    print(f"{level:<8} {b_mean:<12.4f} {m989_mean:<12.4f} {m1260_mean:<12.4f} {m989_mean-b_mean:+.4f}   {m1260_mean-b_mean:+.4f}")

print(f"\n【各等级S_hat标准差趋势】")
print(f"{'Level':<8} {'Baseline':<12} {'Mixed 989':<12} {'Mixed 1260':<12} {'989-BL':<10} {'1260-BL':<10}")
print(f"{'-'*70}")
for level in sorted(merged['occlusion_level'].unique()):
    level_data = merged[merged['occlusion_level'] == level]
    b_std = level_data['S_hat_baseline'].std()
    m989_std = level_data['S_hat_989'].std()
    m1260_std = level_data['S_hat_1260'].std()
    print(f"{level:<8} {b_std:<12.4f} {m989_std:<12.4f} {m1260_std:<12.4f} {m989_std-b_std:+.4f}   {m1260_std-b_std:+.4f}")

# 单调性检验
print(f"\n【等级间增量（单调性检验）】")
level_order = sorted(merged['occlusion_level'].unique())
print(f"{'Level':<8} {'Baseline Δ':<12} {'Mixed 989 Δ':<12} {'Mixed 1260 Δ':<12}")
print(f"{'-'*50}")
for i in range(len(level_order)-1):
    curr, next_l = level_order[i], level_order[i+1]
    b_curr = merged[merged['occlusion_level'] == curr]['S_hat_baseline'].mean()
    b_next = merged[merged['occlusion_level'] == next_l]['S_hat_baseline'].mean()
    m989_curr = merged[merged['occlusion_level'] == curr]['S_hat_989'].mean()
    m989_next = merged[merged['occlusion_level'] == next_l]['S_hat_989'].mean()
    m1260_curr = merged[merged['occlusion_level'] == curr]['S_hat_1260'].mean()
    m1260_next = merged[merged['occlusion_level'] == next_l]['S_hat_1260'].mean()
    print(f"{curr}→{next_l:<5} {b_next-b_curr:+12.4f} {m989_next-m989_curr:+12.4f} {m1260_next-m1260_curr:+12.4f}")

# Spearman相关
print(f"\n【Spearman等级相关性分析】")
b_level_means = [merged[merged['occlusion_level'] == l]['S_hat_baseline'].mean() for l in level_order]
m989_level_means = [merged[merged['occlusion_level'] == l]['S_hat_989'].mean() for l in level_order]
m1260_level_means = [merged[merged['occlusion_level'] == l]['S_hat_1260'].mean() for l in level_order]
b_rho, _ = stats.spearmanr(level_order, b_level_means)
m989_rho, _ = stats.spearmanr(level_order, m989_level_means)
m1260_rho, _ = stats.spearmanr(level_order, m1260_level_means)
print(f"Baseline:   Spearman ρ(level, S_hat_mean) = {b_rho:.4f}")
print(f"Mixed 989:  Spearman ρ(level, S_hat_mean) = {m989_rho:.4f}")
print(f"Mixed 1260: Spearman ρ(level, S_hat_mean) = {m1260_rho:.4f}")

print(f"\n{'='*100}")
