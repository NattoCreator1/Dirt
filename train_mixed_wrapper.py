#!/usr/bin/env python3
"""混合训练wrapper脚本"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 导入训练模块并运行
import baseline.train_baseline as train_module

# 获取命令行参数（除去脚本名）
import sys
sys.argv = [
    'train_baseline.py',
    '--index_csv', 'dataset/woodscape_processed/meta/labels_index_merged_20260224_182712.csv',
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
    '--run_name', 'mixed_ws578_sd_wgap_alpha50',
    '--out_root', 'baseline/runs/mixed_experiment',
]

if __name__ == '__main__':
    train_module.main()
