#!/usr/bin/env python3
"""筛选wrapper脚本"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 导入筛选模块并运行
import importlib.util
spec = importlib.util.spec_from_file_location("filter_sd_with_baseline", "sd_scripts/lora/filter_sd_with_baseline.py")
filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_module)

# 获取命令行参数（除去脚本名）
sys.argv = [
    'filter_wrapper.py',
    '--baseline_ckpt', 'baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth',
    '--sd_index_csv', 'synthetic_soiling/batch_896f_10masks/manifest.csv',
    '--npz_manifest', 'synthetic_soiling/batch_896f_10masks/npz/manifest_npz.csv',
    '--img_root', '.',
    '--output_dir', 'sd_scripts/lora/baseline_filtered_8960',
    '--batch_size', '16',
    '--num_workers', '2',
    '--strategy', 'error_threshold',
    '--error_threshold', '0.05',
]

if __name__ == '__main__':
    filter_module.main()
