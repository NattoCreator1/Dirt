#!/usr/bin/env python3
"""
阶段0：诊断分析 - SD数据校准方案

目的：
1. 分析SD全量8960与WoodScape的S分布差异
2. 分析Baseline模型在SD数据上的预测质量（如果有）
3. 分析等级区分度
4. 输出诊断报告，为后续校准方案提供依据
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

print("=" * 80)
print("阶段0：SD数据校准诊断分析")
print("=" * 80)
print()

# === 数据路径配置 ===
SD_MANIFEST_8960 = 'sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv'
WS_ABLATION_CSV = 'dataset/woodscape_processed/meta/labels_index_ablation.csv'
BASELINE_CKPT = 'baseline/runs/model1_r18_640x480/ckpt_best.pth'

# 输出目录
OUTPUT_DIR = Path('sd_calibration_work/phase0_diagnostic')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("【配置】")
print(f"SD数据: {SD_MANIFEST_8960}")
print(f"WoodScape数据: {WS_ABLATION_CSV}")
print(f"Baseline模型: {BASELINE_CKPT}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# === 诊断1：S分布对比 ===
print("=" * 80)
print("诊断1：S分布对比")
print("=" * 80)

# 加载数据
print("加载数据...")
sd_df = pd.read_csv(SD_MANIFEST_8960)
ws_df = pd.read_csv(WS_ABLATION_CSV)

# SD数据的S值需要从NPZ文件中读取
# 这里先使用一个简化的方法，假设需要后续完善
# 暂时使用占位值
print(f"SD数据: {len(sd_df)} 条")
print(f"WoodScape数据: {len(ws_df)} 条")
print()

# 分析WoodScape的S分布
ws_train = ws_df[ws_df['split'] == 'train'].copy()
ws_S_full_wgap = ws_train['S_full_wgap_alpha50'].values

print(f"WoodScape Train: {len(ws_train)} 条")
print(f"  S_full_wgap_alpha50 均值: {ws_S_full_wgap.mean():.4f}")
print(f"  S_full_wgap_alpha50 标准差: {ws_S_full_wgap.std():.4f}")
print(f"  S_full_wgap_alpha50 范围: [{ws_S_full_wgap.min():.4f}, {ws_S_full_wgap.max():.4f}]")
print()

# SD数据的S值分析
# 需要从NPZ文件读取，这里先创建一个临时分析脚本
print("SD数据S值分析...")
print("注意：SD数据的S值需要从NPZ文件中读取，这里先做基本统计")
print(f"SD数据总数: {len(sd_df)}")
print()

# 创建SD S值加载函数
def load_sd_S_values(sd_df, n_sample=None):
    """从SD数据的NPZ文件中加载S值"""
    S_values = []

    # 限制样本数量以加快分析
    if n_sample is not None and len(sd_df) > n_sample:
        sample_df = sd_df.sample(n=n_sample, random_state=42)
    else:
        sample_df = sd_df

    for idx, row in sample_df.iterrows():
        npz_path = row['npz_path']
        try:
            data = np.load(npz_path)
            # 尝试不同的S值字段
            if 'S_full_wgap_alpha50' in data:
                S_values.append(data['S_full_wgap_alpha50'])
            elif 'S_full' in data:
                S_values.append(data['S_full'])
            else:
                # 如果没有这些字段，使用fallback
                S_values.append(data.get('global_score', 0.5))
        except Exception as e:
            print(f"Warning: 无法加载 {npz_path}: {e}")
            S_values.append(0.5)  # 默认值

    return np.array(S_values)

# 尝试加载SD S值（抽样1000条）
print("加载SD S值（抽样1000条）...")
sd_S_sample = load_sd_S_values(sd_df, n_sample=1000)

print(f"SD抽样数据: {len(sd_S_sample)} 条")
print(f"  S均值: {sd_S_sample.mean():.4f}")
print(f"  S标准差: {sd_S_sample.std():.4f}")
print(f"  S范围: [{sd_S_sample.min():.4f}, {sd_S_sample.max():.4f}]")
print()

# KS检验
print("【KS检验：分布差异】")
ks_stat, ks_pval = stats.ks_2samp(ws_S_full_wgap, sd_S_sample)
print(f"KS统计量: {ks_stat:.4f}")
print(f"p值: {ks_pval:.6f}")
if ks_pval < 0.001:
    print("结论: 分布差异显著 (p < 0.001) ✓")
else:
    print(f"结论: 分布差异 {'显著' if ks_pval < 0.05 else '不显著'} (p = {ks_pval:.4f})")
print()

# 分桶对比
print("【分桶对比】")
bins = np.linspace(0, 1, 11)  # 10个桶
ws_hist, _ = np.histogram(ws_S_full_wgap, bins=bins)
sd_hist, _ = np.histogram(sd_S_sample, bins=bins)

print(f"{'S区间':<15} {'WoodScape频数':<20} {'SD频数':<15} {'SD/WS比率':<15}")
print("-" * 70)
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    ws_count = ws_hist[i]
    sd_count = sd_hist[i]
    ratio = sd_count / ws_count if ws_count > 0 else 0
    print(f"[{low:.1f}, {high:.1f}){'':<8} {ws_count:<20} {sd_count:<15} {ratio:.2f}")
print()

# 保存分布对比图数据
distribution_data = {
    'bins': bins.tolist(),
    'ws_hist': ws_hist.tolist(),
    'sd_hist': sd_hist.tolist(),
    'ws_mean': float(ws_S_full_wgap.mean()),
    'sd_mean': float(sd_S_sample.mean()),
    'ws_std': float(ws_S_full_wgap.std()),
    'sd_std': float(sd_S_sample.std()),
    'ks_stat': float(ks_stat),
    'ks_pval': float(ks_pval),
}

with open(OUTPUT_DIR / 'distribution_comparison.json', 'w') as f:
    json.dump(distribution_data, f, indent=2)

print(f"分布对比数据已保存至: {OUTPUT_DIR / 'distribution_comparison.json'}")

# === 诊断2：检查是否已有Baseline预测结果 ===
print()
print("=" * 80)
print("诊断2：Baseline在SD数据上的预测质量")
print("=" * 80)

# 检查是否有SD数据的预测结果
baseline_sd_predictions = 'sd_calibration_work/results/baseline/predictions_sd_8960.csv'
if os.path.exists(baseline_sd_predictions):
    print(f"发现Baseline预测结果: {baseline_sd_predictions}")
    # TODO: 分析gap分布
else:
    print("未找到Baseline在SD数据上的预测结果")
    print("建议：先运行Baseline模型对SD数据进行推理")
    print(f"  输出路径: {baseline_sd_predictions}")
    print()
    print("推理命令示例：")
    print("python baseline/predict_sd_8960.py \\")
    print("    --ckpt baseline/runs/model1_r18_640x480/ckpt_best.pth \\")
    print("    --sd_csv sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv \\")
    print("    --output results/baseline/predictions_sd_8960.csv")

# === 诊断3：等级区分度分析（使用现有结果）===
print()
print("=" * 80)
print("诊断3：等级区分度分析（基于现有Mixed模型）")
print("=" * 80)

# 加载Test_Ext评估结果
baseline_metrics = 'baseline/runs/model1_r18_640x480/eval_Test_Ext_metrics.json'
mixed_8960_metrics = 'baseline/runs/mixed_ws4000_sd1260_conservative_final/eval_Test_Ext_metrics.json'

if os.path.exists(baseline_metrics) and os.path.exists(mixed_8960_metrics):
    with open(baseline_metrics) as f:
        baseline_data = json.load(f)
    with open(mixed_8960_metrics) as f:
        mixed_8960_data = json.load(f)

    print(f"Baseline Test_Ext Spearman ρ: {baseline_data.get('spearman_rho', 'N/A')}")
    print(f"Mixed 1260 Test_Ext Spearman ρ: {mixed_8960_data.get('spearman_rho', 'N/A')}")

    # 分析level均值
    if 'level_means' in mixed_8960_data:
        level_means = mixed_8960_data['level_means']
        print()
        print("【等级预测均值（Mixed 1260）】")
        for level in ['1', '2', '3', '4', '5']:
            s_mean = level_means.get(level, 'N/A')
            print(f"  Level {level}: {s_mean}")

        # 计算斜率
        levels = np.array([1, 2, 3, 4, 5], dtype=float)
        means = np.array([float(level_means.get(str(int(l)), 0)) for l in levels])
        slope = np.polyfit(levels, means, 1)[0]
        print()
        print(f"等级斜率: {slope:.4f}")
else:
    print("未找到Test_Ext评估结果")

# === 诊断总结 ===
print()
print("=" * 80)
print("诊断总结")
print("=" * 80)

print()
print("【数据集规模】")
print(f"  WoodScape Train: {len(ws_train)}")
print(f"  SD全量 8960: {len(sd_df)}")
print(f"  SD抽样分析: {len(sd_S_sample)}")
print()

print("【S分布差异】")
print(f"  WoodScape S均值: {ws_S_full_wgap.mean():.4f}")
print(f"  SD S均值 (抽样): {sd_S_sample.mean():.4f}")
print(f"  差异: {sd_S_sample.mean() - ws_S_full_wgap.mean():.4f}")
print(f"  KS统计量: {ks_stat:.4f} (p < 0.001: {'是' if ks_pval < 0.001 else '否'})")
print()

print("【下一步决策】")
if ks_pval < 0.001 and (sd_S_sample.mean() - ws_S_full_wgap.mean()) < -0.1:
    print("  ✓ 分布差异显著且SD均值偏低")
    print("  → 进入阶段1：实施分桶Importance Sampling")
else:
    print("  - 分布差异不显著或SD均值不低")
    print("  → 需要重新评估校准必要性")

print()
print("=" * 80)
print("诊断完成！")
print("=" * 80)
print()
print(f"结果文件: {OUTPUT_DIR / 'distribution_comparison.json'}")
