#!/usr/bin/env python3
"""
分层筛选实现

步骤:
1. 加载当前已筛选的989条数据 (error < 0.05)
2. 获取剩余未筛选样本的baseline预测
3. 根据S值应用分层阈值筛选:
   - S < 0.3: 保持 error < 0.05
   - 0.3 <= S < 0.5: 放宽到 error < 0.10
   - S >= 0.5: 放宽到 error < 0.15
4. 合并数据并保存
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

# 读取完整的batch_896f manifest
batch_896f = pd.read_csv('synthetic_soiling/batch_896f_10masks/manifest_20260218_205715.csv')

# 构建image_path
resized_dir = 'synthetic_soiling/batch_896f_10masks/resized_640x480/'
batch_896f['image_path'] = resized_dir + batch_896f['output_filename']

# 读取NPZ manifest获取S_full_wgap_alpha50标签
npz_manifest = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv')

# 合并NPZ路径
batch_896f = batch_896f.merge(
    npz_manifest[['original_filename', 'npz_path']],
    left_on='output_filename',
    right_on='original_filename',
    how='inner'
)  # 使用inner只保留有NPZ的样本

# 从NPZ文件加载S_gt
import numpy as np

def load_s_from_npz(npz_path):
    data = np.load(npz_path)
    return float(data['S_full_wgap_alpha50'])

batch_896f['S_gt'] = batch_896f['npz_path'].apply(load_s_from_npz)

# 获取已筛选样本的集合
selected_filenames = set(filtered_989['output_filename'].values)
batch_896f['is_selected'] = batch_896f['output_filename'].isin(selected_filenames)

print("=" * 80)
print("分层筛选实现")
print("=" * 80)
print(f"\n已筛选: {batch_896f['is_selected'].sum()} 条")
print(f"未筛选: {(~batch_896f['is_selected']).sum()} 条")

# 创建临时index文件用于baseline推理
unselected = batch_896f[~batch_896f['is_selected']].copy()
unselected = unselected[['image_path', 'npz_path', 'S_gt']].dropna()

if len(unselected) == 0:
    print("没有未筛选的样本，退出。")
    sys.exit(0)

print(f"\n未筛选且有NPZ标签的样本: {len(unselected)} 条")

# 保存临时index
temp_index_path = 'sd_scripts/lora/baseline_filtered_8960/unselected_temp_index.csv'
unselected.to_csv(temp_index_path, index=False)

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
ckpt = torch.load('baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth', map_location=device)
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

# 合并S_gt
unselected_results = unselected_results.merge(
    unselected[['image_path', 'S_gt', 'npz_path']],
    on='image_path',
    how='left'
)

# 计算误差
unselected_results['abs_error'] = np.abs(unselected_results['S_hat_baseline'] - unselected_results['S_gt'])
unselected_results['bias'] = unselected_results['S_hat_baseline'] - unselected_results['S_gt']

# 提取output_filename
unselected_results['output_filename'] = unselected_results['image_path'].str.split('/').str[-1]

print("\n=== 分层筛选结果 ===")

# 应用分层阈值
def get_tiered_threshold(s_gt):
    if s_gt < 0.3:
        return 0.05
    elif s_gt < 0.5:
        return 0.10
    else:
        return 0.15

unselected_results['threshold'] = unselected_results['S_gt'].apply(get_tiered_threshold)
unselected_results['passed'] = unselected_results['abs_error'] < unselected_results['threshold']

# 统计
total = len(unselected_results)
passed = unselected_results['passed'].sum()
print(f"总数: {total}")
print(f"通过分层筛选: {passed} ({passed/total*100:.1f}%)")

# 按S区间统计
print("\n按S区间统计:")
unselected_results['S_bin'] = pd.cut(unselected_results['S_gt'], bins=[0, 0.3, 0.5, 1.0],
                                     labels=['0-0.3', '0.3-0.5', '0.5+'])
for interval, group in unselected_results.groupby('S_bin'):
    n = len(group)
    passed_n = group['passed'].sum()
    s_mean = group['S_gt'].mean()
    print(f"  {interval}: 通过={passed_n}/{n} ({passed_n/n*100:.1f}%), S_mean={s_mean:.3f}")

# 获取通过筛选的样本
tiered_filtered = unselected_results[unselected_results['passed']].copy()

# 合并原始989条和新增样本
print(f"\n=== 合并数据 ===")
print(f"原始989条: S_mean={filtered_989['S_gt'].mean():.4f}, 范围=[{filtered_989['S_gt'].min():.4f}, {filtered_989['S_gt'].max():.4f}]")
print(f"新增样本: {len(tiered_filtered)}条, S_mean={tiered_filtered['S_gt'].mean():.4f}, 范围=[{tiered_filtered['S_gt'].min():.4f}, {tiered_filtered['S_gt'].max():.4f}]")

# 准备合并格式
tiered_filtered_export = tiered_filtered[['image_path', 'S_gt', 'S_hat_baseline', 'abs_error', 'bias', 'output_filename']].copy()
tiered_filtered_export.columns = ['image_path', 'S_gt', 'S_hat_baseline', 'error', 'bias', 'output_filename']
tiered_filtered_export['original_filename'] = tiered_filtered_export['output_filename']

# 合并
original_989_export = filtered_989[['image_path', 'S_gt', 'S_hat_baseline', 'error', 'bias', 'output_filename']].copy()
original_989_export['original_filename'] = original_989_export['output_filename']

combined = pd.concat([original_989_export, tiered_filtered_export], ignore_index=True)

print(f"\n合并后总计: {len(combined)} 条")
print(f"  S_mean: {combined['S_gt'].mean():.4f}")
print(f"  S范围: [{combined['S_gt'].min():.4f}, {combined['S_gt'].max():.4f}]")

# 保存结果
output_dir = 'sd_scripts/lora/baseline_filtered_8960'
os.makedirs(output_dir, exist_ok=True)

output_path = f'{output_dir}/filtered_index_tiered.csv'
combined.to_csv(output_path, index=False)
print(f"\n已保存至: {output_path}")

# 同时保存新增样本的详细信息
tiered_output_path = f'{output_dir}/tiered_added_samples.csv'
tiered_filtered_export.to_csv(tiered_output_path, index=False)
print(f"新增样本已保存至: {tiered_output_path}")

print("\n完成!")
