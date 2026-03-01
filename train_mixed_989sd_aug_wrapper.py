#!/usr/bin/env python3
"""训练 Mixed (989 SD) + 相机域随机化模型"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 导入训练模块并运行
import importlib.util
spec = importlib.util.spec_from_file_location("train_augmented", "baseline/train_augmented.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

# 训练配置 - WoodScape (4000) + SD (989) = 4989
sys.argv = [
    'train_augmented.py',
    '--index_csv', 'dataset/woodscape_processed/meta/labels_index_mixed_989sd_20260224_202844.csv',
    '--img_root', '.',
    '--split_col', 'split',
    '--train_split', 'train',
    '--val_split', 'val',
    '--global_target', 'S_full_wgap_alpha50',
    '--epochs', '40',
    '--batch_size', '16',
    '--lr', '3e-4',
    '--weight_decay', '1e-2',
    '--lambda_glob', '1.0',
    '--mu_cons', '0.1',
    '--seed', '42',
    '--num_workers', '4',
    '--run_name', 'mixed_ws4000_sd989_baseline_filtered_aug',
    '--out_root', 'baseline/runs',
    '--aug_enable',
    '--aug_color_prob', '0.8',
    '--aug_noise_prob', '0.7',
    '--aug_blur_prob', '0.4',
    '--aug_compression_prob', '0.5',
    '--aug_lens_prob', '0.4',
    '--aug_resolution_prob', '0.3',
]

if __name__ == '__main__':
    train_module.main()
