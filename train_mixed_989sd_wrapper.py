#!/usr/bin/env python3
"""混合训练 Wrapper (WoodScape + Baseline筛选的989张SD数据)"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 导入训练模块并运行
import importlib.util
spec = importlib.util.spec_from_file_location("train_baseline", "baseline/train_baseline.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

# 训练配置
sys.argv = [
    'train_baseline.py',
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
    '--run_name', 'mixed_ws4000_sd989_baseline_filtered',
    '--out_root', 'baseline/runs',
    '--seed', '42',
    '--num_workers', '4',
]

if __name__ == '__main__':
    train_module.main()
