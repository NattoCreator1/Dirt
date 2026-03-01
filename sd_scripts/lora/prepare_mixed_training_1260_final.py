#!/usr/bin/env python3
"""
最终修正版：使用ablation CSV的S_full_wgap_alpha50列
"""

import sys
import os
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import pandas as pd
import numpy as np

print("=" * 80)
print("准备Mixed训练数据 - 最终修正版")
print("=" * 80)

# ========== 1. 读取ablation CSV (包含S_full_wgap_alpha50) ==========
ablation = pd.read_csv('dataset/woodscape_processed/meta/labels_index_ablation.csv')

# 分离train和val
ws_train = ablation[ablation['split'] == 'train'].copy()
ws_val = ablation[ablation['split'] == 'val'].copy()

print(f"\nWoodScape (from ablation CSV):")
print(f"  训练集: {len(ws_train)}, S_full_wgap_alpha50均值: {ws_train['S_full_wgap_alpha50'].mean():.4f}")
print(f"  验证集: {len(ws_val)}, S_full_wgap_alpha50均值: {ws_val['S_full_wgap_alpha50'].mean():.4f}")

# ========== 2. 准备SD数据 ==========
sd_1260 = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/filtered_index_conservative.csv')
npz_manifest = pd.read_csv('sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv')

sd_1260['filename'] = sd_1260['image_path'].str.split('/').str[-1]
sd_1260 = sd_1260.merge(
    npz_manifest[['original_filename', 'npz_path']],
    left_on='filename',
    right_on='original_filename',
    how='left'
)

# 从NPZ加载S_full_wgap_alpha50
def load_s_full_wgap_alpha50(npz_path):
    data = np.load(npz_path)
    return float(data['S_full_wgap_alpha50'])

sd_1260['S_full_wgap_alpha50'] = sd_1260['npz_path'].apply(load_s_full_wgap_alpha50)

# 计算s列（与ablation CSV保持一致）
sd_1260['s'] = sd_1260['S_full_wgap_alpha50']

# 计算global_level
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
sd_1260['split'] = 'train'

print(f"\nSD 1260:")
print(f"  S_full_wgap_alpha50均值: {sd_1260['S_full_wgap_alpha50'].mean():.4f}")

# ========== 3. 准备SD数据格式（与ablation CSV列对齐） ==========
sd_train = pd.DataFrame({
    'rgb_path': sd_1260['image_path'],
    'npz_path': sd_1260['npz_path'],
    'split': 'train',
    'S': sd_1260['S_full_wgap_alpha50'],
    'S_full_wgap_alpha50': sd_1260['S_full_wgap_alpha50'],
    's': sd_1260['s'],
    'global_level': sd_1260['global_level'],
    'global_bin': sd_1260['global_bin']
})

# ========== 4. 准备WoodScape数据格式 ==========
# 先添加s列
ws_train['s'] = ws_train['S_full_wgap_alpha50']
ws_val['s'] = ws_val['S_full_wgap_alpha50']

ws_columns = ['rgb_path', 'npz_path', 'split', 'S', 'S_full_wgap_alpha50', 's', 'global_level', 'global_bin']
ws_train_final = ws_train[ws_columns].copy()
ws_val_final = ws_val[ws_columns].copy()

# ========== 5. 合并数据 ==========
combined = pd.concat([ws_train_final, ws_val_final, sd_train], ignore_index=True)

print(f"\n合并后数据集:")
print(f"  总数: {len(combined)}")
print(f"  S_full_wgap_alpha50均值: {combined['S_full_wgap_alpha50'].mean():.4f}")

# ========== 6. 保存文件 ==========
output_dir = 'sd_scripts/lora/baseline_filtered_8960'
os.makedirs(output_dir, exist_ok=True)

output_path = f'{output_dir}/mixed_train_val_ws4000_sd1260_conservative_final.csv'
combined.to_csv(output_path, index=False)
print(f"\n已保存至: {output_path}")

print("\n注意：数据集类将优先使用CSV中的S_full_wgap_alpha50列")
print("WoodScape和SD数据都使用相同的标签尺度")
