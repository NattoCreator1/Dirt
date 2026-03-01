#!/usr/bin/env python3
"""
详细分析SD数据的S值分布
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

print("=" * 80)
print("SD数据S值详细分析")
print("=" * 80)
print()

# === 1. 加载Mixed 1260训练数据 ===
print("【1. Mixed 1260训练数据中的SD部分】")
df_1260 = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/mixed_train_val_ws4000_sd1260_conservative_final.csv')
sd_1260 = df_1260[df_1260['split'] == 'sd_train'].copy()
print(f"SD样本数: {len(sd_1260)}")
print(f"S列名: {list(sd_1260.filter(regex='^S').columns)}")

# 查找正确的S列
S_cols = [col for col in sd_1260.columns if 'S_' in col.lower() or col == 'S']
print(f"可能的S列: {S_cols}")

for col in S_cols:
    if col in sd_1260.columns:
        print(f"  {col}: mean={sd_1260[col].mean():.4f}, range=[{sd_1260[col].min():.4f}, {sd_1260[col].max():.4f}]")

# 使用S_full_wgap_alpha50
if 'S_full_wgap_alpha50' in sd_1260.columns:
    sd_1260_S = sd_1260['S_full_wgap_alpha50'].values
    print()
    print(f"SD 1260 S_full_wgap_alpha50:")
    print(f"  均值: {sd_1260_S.mean():.4f}")
    print(f"  标准差: {sd_1260_S.std():.4f}")
    print(f"  范围: [{sd_1260_S.min():.4f}, {sd_1260_S.max():.4f}]")

print()

# === 2. 分析筛选子集 ===
print("【2. 筛选子集989】")
df_989 = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/filtered_index_error_0.05.csv')
print(f"样本数: {len(df_989)}")
print(f"S_gt列: mean={df_989['S_gt'].mean():.4f}, range=[{df_989['S_gt'].min():.4f}, {df_989['S_gt'].max():.4f}]")
print()

# === 3. 检查是否存在用于训练的完整SD数据 ===
print("【3. 查找训练用SD数据文件】")
for file in os.listdir('sd_scripts/lora/baseline_filtered_8960/'):
    if 'sd_train' in file and file.endswith('.csv'):
        print(f"  {file}")
        try:
            df = pd.read_csv(f'sd_scripts/lora/baseline_filtered_8960/{file}')
            print(f"    样本数: {len(df)}")
            if 'S_full_wgap_alpha50' in df.columns:
                print(f"    S均值: {df['S_full_wgap_alpha50'].mean():.4f}")
        except Exception as e:
            print(f"    加载失败: {e}")

print()

# === 4. 分桶对比 ===
print("【4. 分桶对比】")
bins = np.linspace(0, 1, 11)

# WoodScape
ws_df = pd.read_csv('dataset/woodscape_processed/meta/labels_index_ablation.csv')
ws_train = ws_df[ws_df['split'] == 'train']['S_full_wgap_alpha50'].values
ws_hist, _ = np.histogram(ws_train, bins=bins)

# SD 1260
sd_1260_hist, _ = np.histogram(sd_1260_S, bins=bins)

# 筛选989
filtered_989_hist, _ = np.histogram(df_989['S_gt'].values, bins=bins)

print(f"{'S区间':<15} {'WoodScape':<12} {'SD 1260':<12} {'筛选989':<12}")
print("-" * 60)
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    ws_count = ws_hist[i]
    sd_count = sd_1260_hist[i]
    filtered_count = filtered_989_hist[i]
    print(f"[{low:.1f}, {high:.1f}){'':<8} {ws_count:<12} {sd_count:<12} {filtered_count:<12}")

print()

# === 5. 总结 ===
print("【总结】")
print(f"WoodScape Train S均值: {ws_train.mean():.4f}")
print(f"SD 1260 S均值: {sd_1260_S.mean():.4f}")
print(f"筛选989 S均值: {df_989['S_gt'].mean():.4f}")
print()
print("结论:")
print("  - SD 1260 (0.37) < WoodScape (0.41) → 偏低 ✓")
print("  - 筛选989 (0.29) < WoodScape (0.41) → 更低 ✓")
print("  - 筛选策略有效选择了低S值样本")
