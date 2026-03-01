#!/usr/bin/env python3
"""
可视化SD与WoodScape的S分布对比
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 加载诊断报告
with open('sd_calibration_work/phase0_diagnostic/diagnostic_report.json', 'r') as f:
    data = json.load(f)

bins = np.array(data['bins'])
ws_hist = np.array(data['woodscape_hist'])
sd_hist = np.array(data['sd_1260_hist'])

# 计算密度
ws_total = ws_hist.sum()
sd_total = sd_hist.sum()
ws_density = ws_hist / ws_total
sd_density = sd_hist / sd_total

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# === 1. 频数直方图 ===
ax1 = axes[0, 0]
bin_centers = (bins[:-1] + bins[1:]) / 2
width = 0.04

ax1.bar(bin_centers - width/2, ws_hist, width, label='WoodScape', alpha=0.7, color='#2E86AB')
ax1.bar(bin_centers + width/2, sd_hist, width, label='SD 1260', alpha=0.7, color='#A23B72')
ax1.set_xlabel('S值', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.set_title('频数分布对比', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# === 2. 密度对比 ===
ax2 = axes[0, 1]
ax2.plot(bin_centers, ws_density, 'o-', label='WoodScape', linewidth=2, markersize=6, color='#2E86AB')
ax2.plot(bin_centers, sd_density, 's-', label='SD 1260', linewidth=2, markersize=6, color='#A23B72')
ax2.fill_between(bin_centers, ws_density, alpha=0.2, color='#2E86AB')
ax2.fill_between(bin_centers, sd_density, alpha=0.2, color='#A23B72')
ax2.set_xlabel('S值', fontsize=12)
ax2.set_ylabel('密度', fontsize=12)
ax2.set_title('密度分布对比', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# === 3. 比率分析 ===
ax3 = axes[1, 0]
ratio = np.zeros_like(sd_hist, dtype=float)
valid_mask = ws_hist > 0
ratio[valid_mask] = sd_hist[valid_mask] / ws_hist[valid_mask]
ratio[~valid_mask] = np.nan

colors = ['#FF6B6B' if r > 1 else '#4ECDC4' if r < 1 else '#95E1D3' for r in ratio]
bars = ax3.bar(bin_centers, ratio, width=0.08, color=colors, edgecolor='black', linewidth=0.5)
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='平衡线 (SD/WS=1)')
ax3.set_xlabel('S区间', fontsize=12)
ax3.set_ylabel('SD / WoodScape 比率', fontsize=12)
ax3.set_title('分桶比率分析 (理想值=1)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 标注数值
for i, (center, r) in enumerate(zip(bin_centers, ratio)):
    if not np.isnan(r):
        ax3.text(center, r + 0.1, f'{r:.2f}', ha='center', va='bottom', fontsize=8)

# === 4. 累积分布 ===
ax4 = axes[1, 1]
ws_cdf = np.cumsum(ws_density)
sd_cdf = np.cumsum(sd_density)

ax4.plot(bin_centers, ws_cdf, 'o-', label='WoodScape', linewidth=2, markersize=6, color='#2E86AB')
ax4.plot(bin_centers, sd_cdf, 's-', label='SD 1260', linewidth=2, markersize=6, color='#A23B72')
ax4.set_xlabel('S值', fontsize=12)
ax4.set_ylabel('累积概率', fontsize=12)
ax4.set_title('累积分布函数 (CDF)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_ylim([0, 1.1])

# 添加统计信息
stats_text = f"""
统计摘要:
WoodScape: 均值={data['woodscape']['s_mean']:.4f}, 标准差={data['woodscape']['s_std']:.4f}
SD 1260:   均值={data['sd_1260']['s_mean']:.4f}, 标准差={data['sd_1260']['s_std']:.4f}
KS检验:    统计量={data['ks_test']['statistic']:.4f}, p<1e-150
"""
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.08, 1, 1])

# 保存图像
output_path = Path('sd_calibration_work/phase0_diagnostic/distribution_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"图像已保存: {output_path}")

plt.close()

print()
print("=" * 80)
print("可视化分析完成")
print("=" * 80)
print()
print("【关键观察】")
print("1. SD数据集中在[0.2, 0.4]区间，WoodScape分布更均匀")
print("2. SD数据在高S区间(>0.5)严重不足")
print("3. SD/WoodScape比率: 低区间(<0.3)不足，中区间(0.3-0.4)过量，高区间(>0.4)严重不足")
print("4. CDF显示SD分布整体左移，累积速度更快")
print()
print("【校准建议】")
print("- 提高低S区间([0, 0.2))权重: ×1.9 ~ ×21")
print("- 降低中S区间([0.3, 0.4))权重: ×0.8")
print("- 大幅提高高S区间([0.4, 0.6))权重: ×5 ~ ×290")
print()
