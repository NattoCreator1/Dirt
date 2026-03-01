#!/usr/bin/env python3
"""
保守渐进式筛选 (方案A)

渐进式策略：
- [0, 0.25]: 严格阈值 < 0.05
- [0.25, 0.35]: 放宽到 < 0.07
- [0.35, 0.45]: 放宽到 < 0.08
- [0.45, 1.0]: 放宽到 < 0.10

目标：新增约300条样本，保持数据结构稳定
"""

import sys
import os
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from torch.utils.data import DataLoader

# 加载当前已筛选的989条数据
filtered_989 = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/filtered_index_error_0.05.csv')

# 读取NPZ manifest
npz_manifest = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv')

# 构建image_path
resized_dir = 'synthetic_soiling/batch_896f_10masks/resized_640x480/'
npz_manifest['image_path'] = resized_dir + npz_manifest['original_filename']

# 从NPZ文件加载S_gt
def load_s_from_npz(npz_path):
    data = np.load(npz_path)
    return float(data['S_full_wgap_alpha50'])

npz_manifest['S_gt'] = npz_manifest['npz_path'].apply(load_s_from_npz)

# 获取已筛选样本的集合 - 从image_path中提取文件名
selected_filenames = set(filtered_989['image_path'].str.split('/').str[-1].values)
npz_manifest['is_selected'] = npz_manifest['original_filename'].isin(selected_filenames)

print("=" * 80)
print("保守渐进式筛选 (方案A)")
print("=" * 80)
print(f"\n当前已筛选: {npz_manifest['is_selected'].sum()} 条")
print(f"未筛选: {(~npz_manifest['is_selected']).sum()} 条")

# 未筛选样本
unselected = npz_manifest[~npz_manifest['is_selected']].copy()
print(f"未筛选且有NPZ标签的样本: {len(unselected)} 条")

# 创建临时index文件用于baseline推理
temp_index_path = 'sd_scripts/lora/baseline_filtered_8960/unselected_temp_index.csv'
unselected[['image_path', 'npz_path', 'S_gt']].to_csv(temp_index_path, index=False)

# 创建dataset和dataloader
spec = WoodscapeIndexSpec(
    index_csv=temp_index_path,
    img_root='.',
    labels_tile_dir=None,
    split_col=None,
    split_value=None,
    global_target='S_full_wgap_alpha50',
    resize_w=640,
    resize_h=480,
    augmentation=None
)

dataset = WoodscapeTileDataset(spec)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=False)

# 加载baseline模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaselineDualHead(pretrained=False).to(device)
ckpt = torch.load('baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth',
                      map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

print(f"\n开始Baseline推理...")
print(f"设备: {device}")
print(f"样本数: {len(dataset)}")

# 推理
all_shat = []
all_img_paths = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Baseline inference"):
        x = batch["image"].to(device)
        img_paths = batch["img_path"]

        out = model(x)
        shat = out["S_hat"].cpu().numpy().flatten()

        all_shat.extend(shat)
        all_img_paths.extend(img_paths)

# 创建结果DataFrame
unselected_results = pd.DataFrame({
    'image_path': all_img_paths,
    'S_hat_baseline': all_shat
})

# 合并S_gt和npz_path
unselected_results = unselected_results.merge(
    unselected[['image_path', 'S_gt', 'npz_path', 'original_filename']],
    on='image_path',
    how='left'
)

# 计算误差
unselected_results['abs_error'] = np.abs(unselected_results['S_hat_baseline'] - unselected_results['S_gt'])
unselected_results['bias'] = unselected_results['S_hat_baseline'] - unselected_results['S_gt']

print("\n=== 保守渐进式筛选 (方案A) ===")

# 应用渐进式阈值
def get_conservative_threshold(s_gt):
    if s_gt < 0.25:
        return 0.05
    elif s_gt < 0.35:
        return 0.07
    elif s_gt < 0.45:
        return 0.08
    else:
        return 0.10

unselected_results['threshold'] = unselected_results['S_gt'].apply(get_conservative_threshold)
unselected_results['passed'] = unselected_results['abs_error'] < unselected_results['threshold']

# 统计
total = len(unselected_results)
passed = unselected_results['passed'].sum()
print(f"总数: {total}")
print(f"通过筛选: {passed} ({passed/total*100:.1f}%)")

# 按S区间统计
print("\n按S区间统计:")
unselected_results['S_bin'] = pd.cut(unselected_results['S_gt'], bins=[0, 0.25, 0.35, 0.45, 1.0],
                                     labels=['0-0.25', '0.25-0.35', '0.35-0.45', '0.45+'])
for interval, group in unselected_results.groupby('S_bin'):
    n = len(group)
    passed_n = group['passed'].sum()
    s_mean = group['S_gt'].mean()
    threshold = group['threshold'].iloc[0]
    print(f"  {interval}: 通过={passed_n}/{n} ({passed_n/n*100:.1f}%), S_mean={s_mean:.3f}, 阈值={threshold}")

# 获取通过筛选的样本
conservative_filtered = unselected_results[unselected_results['passed']].copy()

# 合并原始989条和新增样本
print(f"\n=== 合并数据 ===")
print(f"原始989条: S_mean={filtered_989['S_gt'].mean():.4f}, 范围=[{filtered_989['S_gt'].min():.4f}, {filtered_989['S_gt'].max():.4f}]")
print(f"新增样本: {len(conservative_filtered)}条, S_mean={conservative_filtered['S_gt'].mean():.4f}, 范围=[{conservative_filtered['S_gt'].min():.4f}, {conservative_filtered['S_gt'].max():.4f}]")

# 准备合并格式
conservative_filtered_export = conservative_filtered[['image_path', 'S_gt', 'S_hat_baseline', 'abs_error', 'bias', 'original_filename']].copy()
conservative_filtered_export.columns = ['image_path', 'S_gt', 'S_hat_baseline', 'error', 'bias', 'output_filename']

# 合并
original_989_export = filtered_989[['image_path', 'S_gt', 'S_hat_baseline', 'error', 'bias', 'output_filename']].copy()

combined = pd.concat([original_989_export, conservative_filtered_export], ignore_index=True)

print(f"\n合并后总计: {len(combined)} 条")
print(f"  S_mean: {combined['S_gt'].mean():.4f}")
print(f"  S范围: [{combined['S_gt'].min():.4f}, {combined['S_gt'].max():.4f}]")
print(f"  SD/WS比例: {len(combined)}/{4000} = 1:{4000/len(combined):.1f}")

# 保存结果
output_dir = 'sd_scripts/lora/baseline_filtered_8960'
os.makedirs(output_dir, exist_ok=True)

output_path = f'{output_dir}/filtered_index_conservative.csv'
combined.to_csv(output_path, index=False)
print(f"\n已保存至: {output_path}")

# 保存新增样本详情
conservative_output_path = f'{output_dir}/conservative_added_samples.csv'
conservative_filtered_export.to_csv(conservative_output_path, index=False)
print(f"新增样本已保存至: {conservative_output_path}")

# 输出S分布对比
print(f"\n=== S分布对比 ===")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.8', '0.8-1.0']

print("\n原始989条:")
hist1, _ = np.histogram(filtered_989['S_gt'], bins=bins)
for label, count in zip(labels, hist1):
    if count > 0:
        print(f"  {label}: {count}")

print("\n新增样本:")
hist2, _ = np.histogram(conservative_filtered['S_gt'], bins=bins)
for label, count in zip(labels, hist2):
    if count > 0:
        print(f"  {label}: {count}")

print("\n合并后:")
hist_combined, _ = np.histogram(combined['S_gt'], bins=bins)
for label, count in zip(labels, hist_combined):
    if count > 0:
        print(f"  {label}: {count}")

print("\n完成!")
