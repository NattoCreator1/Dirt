#!/usr/bin/env python3
"""
修正训练数据准备：移除s列，让数据集类统一从NPZ加载标签
"""

import sys
import os
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import pandas as pd
import numpy as np

print("=" * 80)
print("准备Mixed训练数据 - 修正版")
print("=" * 80)

# ========== 1. 准备SD数据 ==========
sd_1260 = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/filtered_index_conservative.csv')
npz_manifest = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv')

sd_1260['filename'] = sd_1260['image_path'].str.split('/').str[-1]
sd_1260 = sd_1260.merge(
    npz_manifest[['original_filename', 'npz_path']],
    left_on='filename',
    right_on='original_filename',
    how='left'
)

# 计算global_level（从S_full_wgap_alpha50）
def load_s_full_wgap_alpha50(npz_path):
    data = np.load(npz_path)
    return float(data['S_full_wgap_alpha50'])

sd_1260['S_full_wgap_alpha50'] = sd_1260['npz_path'].apply(load_s_full_wgap_alpha50)

def get_global_level(s):
    if s < 0.2:
        return 1
    elif s < 0.4:
        return 2
    elif s < 0.6:
        return 3
    else:
        return 4

sd_1260['global_level'] = sd_1260['S_full_wgap_alpha50'].apply(get_global_level)
sd_1260['global_bin'] = 1

# 准备训练格式 - 不包含s列！
sd_train = pd.DataFrame({
    'rgb_path': sd_1260['image_path'],
    'npz_path': sd_1260['npz_path'],
    'split': 'train',
    'global_level': sd_1260['global_level'],
    'global_bin': sd_1260['global_bin']
})

print(f"\nSD 1260条:")
print(f"  S_full_wgap_alpha50均值: {sd_1260['S_full_wgap_alpha50'].mean():.4f}")

# ========== 2. 准备WoodScape数据 ==========
ws_all = pd.read_csv('dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv')
ws_train = ws_all[ws_all['split'] == 'train'].copy()
ws_val = ws_all[ws_all['split'] == 'val'].copy()

# 移除s列，让数据集从NPZ加载
ws_train_clean = ws_train[['rgb_path', 'npz_path', 'split', 'global_level', 'global_bin']].copy()
ws_val_clean = ws_val[['rgb_path', 'npz_path', 'split', 'global_level', 'global_bin']].copy()

print(f"\nWoodScape:")
print(f"  训练集: {len(ws_train_clean)}")
print(f"  验证集: {len(ws_val_clean)}")

# ========== 3. 合并数据 ==========
combined = pd.concat([ws_train_clean, ws_val_clean, sd_train], ignore_index=True)

print(f"\n合并后数据集:")
print(f"  总数: {len(combined)}")

# ========== 4. 保存文件 ==========
output_dir = 'sd_scripts/lora/baseline_filtered_8960'
os.makedirs(output_dir, exist_ok=True)

output_path = f'{output_dir}/mixed_train_val_ws4000_sd1260_conservative_fixed.csv'
combined.to_csv(output_path, index=False)
print(f"\n已保存至: {output_path}")

print("\n注意：数据集类将统一从NPZ加载S_full_wgap_alpha50标签")
print("WoodScape NPZ会fallback到global_score，SD NPZ使用S_full_wgap_alpha50")
