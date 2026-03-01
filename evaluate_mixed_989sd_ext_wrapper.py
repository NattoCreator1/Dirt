#!/usr/bin/env python3
"""评估 Mixed (989 SD) 模型 - External Test Set"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# 导入评估模块并运行
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_baseline", "scripts/09_evaluate_baseline.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

# 评估配置 - Test_Ext (External Test Set)
sys.argv = [
    'evaluate_baseline.py',
    '--test_set', 'external',
    '--ckpt', 'baseline/runs/mixed_ws4000_sd989_baseline_filtered/ckpt_best.pth',
    '--ext_csv', 'dataset/external_test_washed_processed/test_ext.csv',
    '--ext_img_root', '.',
    '--ext_label_col', 'ext_level',
    '--global_target', 'S_full_wgap_alpha50',
    '--out_dir', 'baseline/runs/mixed_ws4000_sd989_baseline_filtered',
]

if __name__ == '__main__':
    eval_module.main()
