#!/usr/bin/env python3
"""
分层筛选策略分析

分析8960条SD数据的分布，设计分层筛选方案：
- 低S区间: 严格阈值 (error < 0.05)
- 高S区间: 放宽阈值 (error < 0.10 或 0.15)
"""

import pandas as pd
import numpy as np

# 读取筛选后的989条数据
filtered_989 = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/filtered_index_error_0.05.csv')

# 读取完整的batch_896f manifest
batch_896f = pd.read_csv('synthetic_soiling/batch_896f_10masks/manifest_20260218_205715.csv')

# 读取NPZ manifest (获取S_full_wgap_alpha50标签)
npz_manifest = pd.read_csv('sd_scripts/lora/accepted_20260224_164715/npz/manifest_npz.csv')

# 获取resize后的图像路径映射
resized_dir = 'synthetic_soiling/batch_896f_10masks/resized_640x480/'
resized_images = []
for orig_file in batch_896f['output_filename']:
    resized_path = f"{resized_dir}{orig_file}"
    resized_images.append(resized_path)

batch_896f['image_path'] = resized_images
batch_896f['S_gt'] = batch_896f['mask_S_full']  # 使用mask_S_full作为S_gt

print("=" * 80)
print("分层筛选策略分析")
print("=" * 80)

print("\n=== 当前筛选 (error < 0.05) ===")
print(f"总数: {len(filtered_989)}")
print(f"S均值: {filtered_989['S_gt'].mean():.4f}")
print(f"S范围: [{filtered_989['S_gt'].min():.4f}, {filtered_989['S_gt'].max():.4f}]")

print("\n=== 完整8960条数据分布 ===")
print(f"总数: {len(batch_896f)}")
print(f"S均值: {batch_896f['S_gt'].mean():.4f}")
print(f"S范围: [{batch_896f['S_gt'].min():.4f}, {batch_896f['S_gt'].max():.4f}]")

# 获取已筛选样本的文件名集合
selected_filenames = set(filtered_989['output_filename'].values)

# 标记未筛选的样本
batch_896f['is_selected'] = batch_896f['output_filename'].isin(selected_filenames)

print("\n=== 按S区间分析筛选情况 ===")
batch_896f['S_bin'] = pd.cut(batch_896f['S_gt'], bins=[0, 0.2, 0.3, 0.4, 0.5, 1.0],
                               labels=['0-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5+'])

for interval, group in batch_896f.groupby('S_bin'):
    total = len(group)
    selected = group['is_selected'].sum()
    s_mean = group['S_gt'].mean()
    selected_ratio = selected / total if total > 0 else 0
    print(f"{interval}: 总数={total:4d}, 已选={selected:4d} ({selected_ratio*100:.1f}%), S_mean={s_mean:.3f}")

# 分析未筛选的高S样本
print("\n=== 未筛选的高S样本 (S >= 0.4) ===")
high_s_unselected = batch_896f[(batch_896f['S_gt'] >= 0.4) & (~batch_896f['is_selected'])]
print(f"数量: {len(high_s_unselected)}")
print(f"S均值: {high_s_unselected['S_gt'].mean():.4f}")
print(f"S范围: [{high_s_unselected['S_gt'].min():.4f}, {high_s_unselected['S_gt'].max():.4f}]")

# 设计分层筛选策略
print("\n=== 分层筛选策略建议 ===")
print("策略: 根据S值区间使用不同的误差阈值")
print()
print("S区间         | 阈值   | 预期新增 | 理由")
print("--------------|--------|---------|------")
print("[0, 0.3]      | < 0.05 | ~0      | 当前已覆盖良好")
print("[0.3, 0.5]    | < 0.10 | ~50-100 | 适度放宽，引入中等S")
print("[0.5, 1.0]    | < 0.15 | ~20-50  | 大幅放宽，引入高S")

# 模拟分层筛选效果
# 假设未筛选样本的误差分布与S值相关
def estimate_error_for_s(s_value):
    """根据S值估计可能的baseline误差"""
    # 低S: 误差较小
    # 高S: 误差可能较大（标签质量下降）
    base_error = 0.02 + s_value * 0.15
    return base_error

print("\n=== 模拟分层筛选效果 ===")
for s_min, s_max, threshold in [(0.3, 0.5, 0.10), (0.5, 1.0, 0.15)]:
    candidates = batch_896f[(batch_896f['S_gt'] >= s_min) & (batch_896f['S_gt'] < s_max)]
    unselected = candidates[~candidates['is_selected']]
    print(f"S区间 [{s_min}, {s_max}): 候选={len(unselected)}个, 建议阈值={threshold}")

# 输出未筛选的高S样本列表（前20个）
print("\n=== 前20个未筛选的高S样本 ===")
high_s_samples = high_s_unselected[['output_filename', 'S_gt', 'mask_coverage']].head(20)
for _, row in high_s_samples.iterrows():
    print(f"  {row['output_filename']}: S={row['S_gt']:.3f}, coverage={row['mask_coverage']:.3f}")
