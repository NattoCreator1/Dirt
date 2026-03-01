#!/usr/bin/env python3
"""
阶段0补充：分析SD全量8960数据的S分布
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
print("阶段0补充：SD全量8960分布分析")
print("=" * 80)
print()

# 配置
SD_8960_CSV = 'sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv'
WS_ABLATION_CSV = 'dataset/woodscape_processed/meta/labels_index_ablation.csv'
OUTPUT_DIR = Path('sd_calibration_work/phase0_diagnostic')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("【数据路径】")
print(f"SD 8960: {SD_8960_CSV}")
print(f"WoodScape: {WS_ABLATION_CSV}")
print()

# === 1. 加载WoodScape数据 ===
print("【1. 加载WoodScape数据】")
ws_df = pd.read_csv(WS_ABLATION_CSV)
ws_train = ws_df[ws_df['split'] == 'train'].copy()
ws_S = ws_train['S_full_wgap_alpha50'].values

print(f"WoodScape Train: {len(ws_train)} 条")
print(f"  S均值: {ws_S.mean():.4f}")
print(f"  S标准差: {ws_S.std():.4f}")
print(f"  S范围: [{ws_S.min():.4f}, {ws_S.max():.4f}]")
print()

# === 2. 加载SD 8960数据（从NPZ）===
print("【2. 加载SD 8960数据（从NPZ文件）】")
sd_df = pd.read_csv(SD_8960_CSV)
print(f"SD 8960 CSV: {len(sd_df)} 条")

def load_S_from_npz(npz_path):
    """从NPZ文件中加载S值"""
    try:
        data = np.load(npz_path)
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
    if idx % 1000 == 0:
        print(f"  进度: {idx}/{len(sd_df)}")
    npz_path = row['npz_path']
    s_val = load_S_from_npz(npz_path)
    if s_val is not None:
        sd_S_values.append(s_val)
        valid_indices.append(idx)

sd_S_values = np.array(sd_S_values)
print(f"成功加载: {len(sd_S_values)}/{len(sd_df)} 条")

if len(sd_S_values) > 0:
    print(f"SD 8960 S均值: {sd_S_values.mean():.4f}")
    print(f"SD 8960 S标准差: {sd_S_values.std():.4f}")
    print(f"SD 8960 S范围: [{sd_S_values.min():.4f}, {sd_S_values.max():.4f}]")
else:
    print("ERROR: 无法加载SD S值")
    sys.exit(1)
print()

# === 3. 分布对比分析 ===
print()
print("=" * 80)
print("【3. 分布对比分析：WoodScape vs SD 8960】")
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

print(f"{'S区间':<12} {'WoodScape':<12} {'SD 8960':<12} {'SD/WS':<10}")
print("-" * 50)
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    ws_count = ws_hist[i]
    sd_count = sd_hist[i]
    ratio = sd_count / ws_count if ws_count > 0 else 0
    print(f"[{low:.1f}, {high:.1f}){'':<5} {ws_count:<12} {sd_count:<12} {ratio:.2f}")
print()

# === 4. 与SD 1260对比 ===
print("=" * 80)
print("【4. 与SD 1260对比】")
print("=" * 80)

# 加载之前的1260数据
with open(OUTPUT_DIR / 'diagnostic_report.json', 'r') as f:
    diag_1260 = json.load(f)

sd_1260_hist = np.array(diag_1260['sd_1260_hist'])
sd_1260_mean = diag_1260['sd_1260']['s_mean']

print(f"{'S区间':<12} {'WoodScape':<12} {'SD 1260':<12} {'SD 8960':<12}")
print("-" * 50)
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    ws_count = ws_hist[i]
    sd_1260_count = sd_1260_hist[i]
    sd_8960_count = sd_hist[i]
    print(f"[{low:.1f}, {high:.1f}){'':<5} {ws_count:<12} {sd_1260_count:<12} {sd_8960_count:<12}")
print()

print(f"{'数据集':<15} {'样本数':<10} {'S均值':<10} {'S标准差':<10}")
print("-" * 50)
print(f"{'WoodScape':<15} {len(ws_S):<10} {ws_S.mean():<10.4f} {ws_S.std():<10.4f}")
print(f"{'SD 1260':<15} {1260:<10} {sd_1260_mean:<10.4f} {diag_1260['sd_1260']['s_std']:<10.4f}")
print(f"{'SD 8960':<15} {len(sd_S_values):<10} {sd_S_values.mean():<10.4f} {sd_S_values.std():<10.4f}")
print()

# === 5. 诊断总结 ===
print()
print("=" * 80)
print("【5. 诊断总结】")
print("=" * 80)

print()
print("【关键发现】")
print(f"1. SD 8960 S均值: {sd_S_values.mean():.4f}")
print(f"   vs WoodScape ({ws_S.mean():.4f}): {sd_S_values.mean() - ws_S.mean():+.4f}")
print(f"   vs SD 1260 ({sd_1260_mean:.4f}): {sd_S_values.mean() - sd_1260_mean:+.4f}")
print()

if sd_S_values.mean() < ws_S.mean() - 0.05:
    print("2. 结论: SD 8960均值显著偏低 ✓")
    print("   → 需要校准（提高高S区间权重）")
elif sd_S_values.mean() > ws_S.mean() + 0.05:
    print("2. 结论: SD 8960均值显著偏高")
    print("   → 需要反向校准（降低高S区间权重）")
else:
    print("2. 结论: SD 8960均值与WoodScape接近")
    print("   → 可能不需要强烈校准")

print()
print("【校准建议】")
print("- 对于SD 8960，应基于8960自己的分布计算权重")
print("- 而非使用SD 1260的分布")
print()

# 保存结果
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
    'sd_8960': {
        'n_samples': int(len(sd_S_values)),
        's_mean': float(sd_S_values.mean()),
        's_std': float(sd_S_values.std()),
        's_min': float(sd_S_values.min()),
        's_max': float(sd_S_values.max()),
    },
    'sd_1260': {
        'n_samples': 1260,
        's_mean': sd_1260_mean,
        's_std': diag_1260['sd_1260']['s_std'],
        's_min': diag_1260['sd_1260']['s_min'],
        's_max': diag_1260['sd_1260']['s_max'],
    },
    'ks_test_ws_sd8960': {
        'statistic': float(ks_stat),
        'p_value': float(ks_pval),
        'significant': bool(ks_pval < 0.001),
    },
    'bins': bins.tolist(),
    'woodscape_hist': ws_hist.tolist(),
    'sd_8960_hist': sd_hist.tolist(),
    'sd_1260_hist': sd_1260_hist.tolist(),
}

with open(OUTPUT_DIR / 'diagnostic_report_8960.json', 'w') as f:
    json.dump(distribution_data, f, indent=2)

print(f"诊断报告已保存: {OUTPUT_DIR / 'diagnostic_report_8960.json'}")

print()
print("=" * 80)
print("分析完成！")
print("=" * 80)
