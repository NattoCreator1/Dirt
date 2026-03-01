#!/usr/bin/env python3
"""分析Test_Ext各等级内部的预测结果分布差异"""

import pandas as pd
import numpy as np
from scipy import stats

# 读取预测结果
baseline_df = pd.read_csv('baseline/runs/model1_r18_640x480/eval_Test_Ext_predictions.csv')
mixed_df = pd.read_csv('baseline/runs/mixed_ws4000_sd1260_conservative_final/eval_Test_Ext_predictions.csv')

# 重命名列以区分
baseline_df = baseline_df.rename(columns={'S_hat': 'S_hat_baseline', 'S_agg': 'S_agg_baseline'})
mixed_df = mixed_df.rename(columns={'S_hat': 'S_hat_mixed', 'S_agg': 'S_agg_mixed'})

# 合并数据
merged = pd.merge(baseline_df[['image_path', 'occlusion_level', 'S_hat_baseline', 'S_agg_baseline']],
                  mixed_df[['image_path', 'S_hat_mixed', 'S_agg_mixed']],
                  on='image_path')

print("=" * 80)
print("Test_Ext 各等级内部预测结果对比分析")
print("=" * 80)

# 按等级分组分析
for level in sorted(merged['occlusion_level'].unique()):
    level_data = merged[merged['occlusion_level'] == level]

    print(f"\n{'='*80}")
    print(f"Level {level} (N={len(level_data)})")
    print(f"{'='*80}")

    # 基本统计
    print(f"\n【基本统计】")
    print(f"{'指标':<20} {'Baseline':<15} {'Mixed 1260':<15} {'差异':<15}")
    print(f"{'-'*70}")

    for metric, name in [('S_hat_baseline', 'Mean S_hat'),
                         ('S_hat_mixed', ''),
                         ('', 'Diff'),
                         ('S_hat_baseline', 'Std S_hat'),
                         ('S_hat_mixed', ''),
                         ('', 'Diff'),
                         ('S_hat_baseline', 'Median S_hat'),
                         ('S_hat_mixed', ''),
                         ('', 'Diff'),
                         ('S_hat_baseline', 'Min S_hat'),
                         ('S_hat_mixed', ''),
                         ('', 'Diff'),
                         ('S_hat_baseline', 'Max S_hat'),
                         ('S_hat_mixed', ''),
                         ('', 'Diff')]:
        if name == '':
            if metric == '':
                print(f"{'':20} {level_data['S_hat_mixed'].mean():<15.4f}")
            else:
                print(f"{'':20} {level_data[metric].std():<15.4f}" if 'Std' in metric else
                      f"{'':20} {level_data[metric].median():<15.4f}" if 'Median' in metric else
                      f"{'':20} {level_data[metric].min():<15.4f}" if 'Min' in metric else
                      f"{'':20} {level_data[metric].max():<15.4f}")
        elif name == 'Diff':
            b_vals = level_data['S_hat_baseline']
            m_vals = level_data['S_hat_mixed']
            print(f"{'':20} {m_vals.mean() - b_vals.mean():<15.4f}" if metric == '' else
                  f"{'':20} {m_vals.std() - b_vals.std():<15.4f}" if 'Std' in metric else
                  f"{'':20} {m_vals.median() - b_vals.median():<15.4f}" if 'Median' in metric else
                  f"{'':20} {m_vals.min() - b_vals.min():<15.4f}" if 'Min' in metric else
                  f"{'':20} {m_vals.max() - b_vals.max():<15.4f}")
        else:
            print(f"{name:<20} {level_data[metric].mean():<15.4f}", end='')
            if 'S_hat' in metric and 'baseline' in metric:
                print()

    # 重新组织统计表格
    print(f"\n【S_hat统计对比】")
    b_mean = level_data['S_hat_baseline'].mean()
    m_mean = level_data['S_hat_mixed'].mean()
    b_std = level_data['S_hat_baseline'].std()
    m_std = level_data['S_hat_mixed'].std()
    b_median = level_data['S_hat_baseline'].median()
    m_median = level_data['S_hat_mixed'].median()

    print(f"{'指标':<15} {'Baseline':<15} {'Mixed 1260':<15} {'差异(Mixed-BL)'}")
    print(f"{'-'*60}")
    print(f"{'Mean':<15} {b_mean:<15.4f} {m_mean:<15.4f} {m_mean-b_mean:+.4f}")
    print(f"{'Std':<15} {b_std:<15.4f} {m_std:<15.4f} {m_std-b_std:+.4f}")
    print(f"{'Median':<15} {b_median:<15.4f} {m_median:<15.4f} {m_median-b_median:+.4f}")
    print(f"{'Min':<15} {level_data['S_hat_baseline'].min():<15.4f} {level_data['S_hat_mixed'].min():<15.4f} {level_data['S_hat_mixed'].min()-level_data['S_hat_baseline'].min():+.4f}")
    print(f"{'Max':<15} {level_data['S_hat_baseline'].max():<15.4f} {level_data['S_hat_mixed'].max():<15.4f} {level_data['S_hat_mixed'].max()-level_data['S_hat_baseline'].max():+.4f}")

    # 分布特征
    print(f"\n【分布特征】")
    print(f"{'分位数':<10} {'Baseline':<15} {'Mixed 1260':<15}")
    print(f"{'-'*40}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        b_q = level_data['S_hat_baseline'].quantile(q)
        m_q = level_data['S_hat_mixed'].quantile(q)
        print(f"Q{int(q*100):<7} {b_q:<15.4f} {m_q:<15.4f}")

    # 预测范围分析
    print(f"\n【预测范围分布】")
    b_range = level_data['S_hat_baseline'].max() - level_data['S_hat_baseline'].min()
    m_range = level_data['S_hat_mixed'].max() - level_data['S_hat_mixed'].min()
    print(f"Baseline 范围: [{level_data['S_hat_baseline'].min():.4f}, {level_data['S_hat_baseline'].max():.4f}] (跨度={b_range:.4f})")
    print(f"Mixed 1260 范围: [{level_data['S_hat_mixed'].min():.4f}, {level_data['S_hat_mixed'].max():.4f}] (跨度={m_range:.4f})")

    # 区间分布
    print(f"\n【预测值区间分布】")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    b_hist, _ = np.histogram(level_data['S_hat_baseline'], bins=bins)
    m_hist, _ = np.histogram(level_data['S_hat_mixed'], bins=bins)
    print(f"{'区间':<15} {'Baseline数量':<15} {'Mixed 1260数量':<15}")
    print(f"{'-'*50}")
    for i in range(len(bins)-1):
        print(f"[{bins[i]:.1f}, {bins[i+1]:.1f}){'':<8} {b_hist[i]:<15} {m_hist[i]:<15}")

    # S_hat vs S_agg 一致性
    print(f"\n【S_hat vs S_agg 一致性】")
    b_gap = (level_data['S_hat_baseline'] - level_data['S_agg_baseline']).abs().mean()
    m_gap = (level_data['S_hat_mixed'] - level_data['S_agg_mixed']).abs().mean()
    b_rho = level_data['S_hat_baseline'].corr(level_data['S_agg_baseline'])
    m_rho = level_data['S_hat_mixed'].corr(level_data['S_agg_mixed'])
    print(f"Baseline: |S_hat - S_agg| = {b_gap:.4f}, corr(S_hat, S_agg) = {b_rho:.4f}")
    print(f"Mixed 1260: |S_hat - S_agg| = {m_gap:.4f}, corr(S_hat, S_agg) = {m_rho:.4f}")

# 总体统计
print(f"\n{'='*80}")
print("总体统计")
print(f"{'='*80}")

print(f"\n【各等级S_hat均值趋势】")
print(f"{'Level':<10} {'Baseline':<15} {'Mixed 1260':<15} {'差异':<15}")
print(f"{'-'*60}")
for level in sorted(merged['occlusion_level'].unique()):
    level_data = merged[merged['occlusion_level'] == level]
    b_mean = level_data['S_hat_baseline'].mean()
    m_mean = level_data['S_hat_mixed'].mean()
    print(f"{level:<10} {b_mean:<15.4f} {m_mean:<15.4f} {m_mean-b_mean:+.4f}")

print(f"\n【各等级S_hat标准差趋势】")
print(f"{'Level':<10} {'Baseline':<15} {'Mixed 1260':<15} {'差异':<15}")
print(f"{'-'*60}")
for level in sorted(merged['occlusion_level'].unique()):
    level_data = merged[merged['occlusion_level'] == level]
    b_std = level_data['S_hat_baseline'].std()
    m_std = level_data['S_hat_mixed'].std()
    print(f"{level:<10} {b_std:<15.4f} {m_std:<15.4f} {m_std-b_std:+.4f}")

# Spearman相关分析
print(f"\n【Spearman等级相关性分析】")
level_order = sorted(merged['occlusion_level'].unique())
b_level_means = [merged[merged['occlusion_level'] == l]['S_hat_baseline'].mean() for l in level_order]
m_level_means = [merged[merged['occlusion_level'] == l]['S_hat_mixed'].mean() for l in level_order]
b_rho, _ = stats.spearmanr(level_order, b_level_means)
m_rho, _ = stats.spearmanr(level_order, m_level_means)
print(f"Baseline: Spearman ρ(level, S_hat_mean) = {b_rho:.4f}")
print(f"Mixed 1260: Spearman ρ(level, S_hat_mean) = {m_rho:.4f}")

# 单调性检验
print(f"\n【单调性检验】")
print(f"{'Level':<10} {'→ Next Δ':<20} {'Baseline':<15} {'Mixed 1260':<15}")
print(f"{'-'*60}")
for i in range(len(level_order)-1):
    curr, next_l = level_order[i], level_order[i+1]
    b_curr = merged[merged['occlusion_level'] == curr]['S_hat_baseline'].mean()
    b_next = merged[merged['occlusion_level'] == next_l]['S_hat_baseline'].mean()
    m_curr = merged[merged['occlusion_level'] == curr]['S_hat_mixed'].mean()
    m_next = merged[merged['occlusion_level'] == next_l]['S_hat_mixed'].mean()
    print(f"{curr}→{next_l:<5} {b_next-b_curr:+.4f} / {m_next-m_curr:+.4f}")

print(f"\n{'='*80}")
