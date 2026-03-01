#!/usr/bin/env python3
"""
阶段0：SD数据校准诊断分析（完整版）

目的：
1. 从NPZ文件中加载SD数据的S值
2. 分析SD与WoodScape的S分布差异
3. 输出诊断报告
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

print("=" * 80)
print("阶段0：SD数据校准诊断分析（完整版）")
print("=" * 80)
print()

# === 配置 ===
SD_1260_CSV = 'sd_scripts/lora/baseline_filtered_8960/sd_train_1260_conservative.csv'
WS_ABLATION_CSV = 'dataset/woodscape_processed/meta/labels_index_ablation.csv'
OUTPUT_DIR = Path('sd_calibration_work/phase0_diagnostic')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("【数据路径】")
print(f"SD 1260: {SD_1260_CSV}")
print(f"WoodScape: {WS_ABLATION_CSV}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# === 1. 加载WoodScape数据 ===
print("【1. 加载WoodScape数据】")
ws_df = pd.read_csv(WS_ABLATION_CSV)
ws_train = ws_df[ws_df['split'] == 'train'].copy()
ws_S = ws_train['S_full_wgap_alpha50'].values

print(f"WoodScape Train: {len(ws_train)} 条")
print(f"  S_full_wgap_alpha50 均值: {ws_S.mean():.4f}")
print(f"  S_full_wgap_alpha50 标准差: {ws_S.std():.4f}")
print(f"  S_full_wgap_alpha50 范围: [{ws_S.min():.4f}, {ws_S.max():.4f}]")
print()

# === 2. 加载SD 1260数据（从NPZ）===
print("【2. 加载SD 1260数据（从NPZ文件）】")
sd_df = pd.read_csv(SD_1260_CSV)
print(f"SD 1260 CSV: {len(sd_df)} 条")

def load_S_from_npz(npz_path):
    """从NPZ文件中加载S值"""
    try:
        data = np.load(npz_path)
        # 按优先级查找S值
        if 'S_full_wgap_alpha50' in data:
            return data['S_full_wgap_alpha50']
        elif 'S_full' in data:
            return data['S_full']
        elif 'global_score' in data:
            return data['global_score']
        else:
            return None
    except Exception as e:
        print(f"Warning: 加载 {npz_path} 失败: {e}")
        return None

# 加载SD S值
print("加载S值（可能需要几分钟）...")
sd_S_values = []
valid_indices = []

for idx, row in sd_df.iterrows():
    npz_path = row['npz_path']
    s_val = load_S_from_npz(npz_path)
    if s_val is not None:
        sd_S_values.append(s_val)
        valid_indices.append(idx)

sd_S_values = np.array(sd_S_values)
print(f"成功加载: {len(sd_S_values)}/{len(sd_df)} 条")

if len(sd_S_values) > 0:
    print(f"SD 1260 S均值: {sd_S_values.mean():.4f}")
    print(f"SD 1260 S标准差: {sd_S_values.std():.4f}")
    print(f"SD 1260 S范围: [{sd_S_values.min():.4f}, {sd_S_values.max():.4f}]")
else:
    print("ERROR: 无法加载SD S值")
    sys.exit(1)
print()

# === 3. 分布对比分析 ===
print()
print("=" * 80)
print("【3. 分布对比分析】")
print("=" * 80)

# KS检验
ks_stat, ks_pval = stats.ks_2samp(ws_S, sd_S_values)
print(f"KS检验:")
print(f"  统计量: {ks_stat:.4f}")
print(f"  p值: {ks_pval:.6f}")
print(f"  结论: 分布差异{'显著 (p<0.001)' if ks_pval < 0.001 else '不显著'}")
print()

# 分桶对比
bins = np.linspace(0, 1, 11)
ws_hist, _ = np.histogram(ws_S, bins=bins)
sd_hist, _ = np.histogram(sd_S_values, bins=bins)

print(f"{'S区间':<12} {'WoodScape':<12} {'SD 1260':<12} {'SD/WS':<10}")
print("-" * 50)
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    ws_count = ws_hist[i]
    sd_count = sd_hist[i]
    ratio = sd_count / ws_count if ws_count > 0 else 0
    print(f"[{low:.1f}, {high:.1f}){'':<5} {ws_count:<12} {sd_count:<12} {ratio:.2f}")
print()

# === 4. 诊断总结 ===
print()
print("=" * 80)
print("【4. 诊断总结】")
print("=" * 80)

print()
print("【数据集规模】")
print(f"  WoodScape Train: {len(ws_train)}")
print(f"  SD 1260: {len(sd_S_values)}")
print()

print("【S分布对比】")
print(f"  WoodScape S均值: {ws_S.mean():.4f}")
print(f"  SD 1260 S均值: {sd_S_values.mean():.4f}")
diff = sd_S_values.mean() - ws_S.mean()
print(f"  差异: {diff:+.4f}")

if diff < -0.05:
    print(f"  结论: SD显著偏低 ({diff:.4f} < -0.05) ✓")
    print("  → 需要校准")
elif diff > 0.05:
    print(f"  结论: SD显著偏高 ({diff:.4f} > 0.05)")
    print("  → 需要反向校准（降低权重）或重新评估")
else:
    print(f"  结论: 分布接近 (|diff| < 0.05)")
    print("  → 可能不需要校准")
print()

print("【分布差异检验】")
if ks_pval < 0.001:
    print("  ✓ KS检验显著 (p < 0.001)")
    print("  → 分布存在显著差异")
else:
    print(f"  - KS检验{'显著' if ks_pval < 0.05 else '不显著'} (p = {ks_pval:.4f})")

print()
print("=" * 80)
print("【下一步决策】")
print("=" * 80)

if diff < -0.05:
    print("✓ SD均值偏低，需要校准")
    print("  → 进入阶段1：实施分桶Importance Sampling")
    print("  权重策略: 低S区间降低权重，高S区间提高权重")
elif diff > 0.05:
    print("✓ SD均值偏高，需要反向校准")
    print("  → 进入阶段1：实施分桶Importance Sampling")
    print("  权重策略: 低S区间提高权重，高S区间降低权重")
else:
    print("- SD均值接近WoodScape，可能不需要校准")
    print("  → 仍可进入阶段1验证")

print()
print("【输出文件】")
distribution_data = {
    'woodscape': {
        'n_samples': int(len(ws_train)),
        's_mean': float(ws_S.mean()),
        's_std': float(ws_S.std()),
        's_min': float(ws_S.min()),
        's_max': float(ws_S.max()),
    },
    'sd_1260': {
        'n_samples': int(len(sd_S_values)),
        's_mean': float(sd_S_values.mean()),
        's_std': float(sd_S_values.std()),
        's_min': float(sd_S_values.min()),
        's_max': float(sd_S_values.max()),
    },
    'ks_test': {
        'statistic': float(ks_stat),
        'p_value': float(ks_pval),
        'significant': bool(ks_pval < 0.001),
    },
    'bins': bins.tolist(),
    'woodscape_hist': ws_hist.tolist(),
    'sd_1260_hist': sd_hist.tolist(),
    'recommendation': 'calibration_needed' if abs(diff) > 0.05 else 'optional',
}

with open(OUTPUT_DIR / 'diagnostic_report.json', 'w') as f:
    json.dump(distribution_data, f, indent=2)

print(f"诊断报告已保存: {OUTPUT_DIR / 'diagnostic_report.json'}")

print()
print("=" * 80)
print("诊断分析完成！")
print("=" * 80)
