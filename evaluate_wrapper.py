#!/usr/bin/env python3
"""评估wrapper脚本"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 直接运行脚本
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_baseline", "scripts/09_evaluate_baseline.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

# 获取命令行参数（除去脚本名）
sys.argv = [
    'evaluate_wrapper.py',
    '--ckpt', 'baseline/runs/mixed_experiment/mixed_ws578_sd_wgap_alpha50/ckpt_best.pth',
    '--test_set', 'both',
    '--index_csv', 'dataset/woodscape_processed/meta/labels_index_ablation.csv',
    '--img_root', '.',
    '--split_col', 'split',
    '--test_split', 'test',
    '--global_target', 'S_full_wgap_alpha50',
    '--ext_csv', 'dataset/external_test_washed_processed/test_ext.csv',
    '--ext_img_root', '.',
    '--batch_size', '32',
    '--num_workers', '4',
]

if __name__ == '__main__':
    eval_module.main()
