#!/usr/bin/env python3
"""
准备混合训练索引 (WoodScape + Baseline筛选的989张SD数据)

Author: SD Experiment Team
Date: 2026-02-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime


def main():
    print("=" * 70)
    print("准备混合训练索引")
    print("=" * 70)

    # 1. 读取 WoodScape 索引
    print("\n1. 读取 WoodScape 索引")
    ws_index = 'dataset/woodscape_processed/meta/labels_index_ablation.csv'
    ws_df = pd.read_csv(ws_index)
    print(f"  WoodScape 样本数: {len(ws_df)}")
    print(f"  分隔分布:")
    print(ws_df['split'].value_counts().to_string())

    # 2. 读取 Baseline 筛选的 SD 索引
    print("\n2. 读取 Baseline 筛选的 SD 索引")
    sd_filtered = 'sd_scripts/lora/baseline_filtered_8960/filtered_index_error_0.05.csv'
    sd_df = pd.read_csv(sd_filtered)
    print(f"  SD 筛选样本数: {len(sd_df)}")

    # 从 image_path 提取 filename
    sd_df['filename'] = sd_df['image_path'].str.split('/').str[-1].str.replace('.png', '.npz')

    # 3. 准备 SD 数据的 NPZ 路径
    print("\n3. 准备 SD 数据的 NPZ 路径")
    npz_manifest = 'synthetic_soiling/batch_896f_10masks/npz/manifest_npz.csv'
    npz_df = pd.read_csv(npz_manifest)
    print(f"  NPZ manifest 样本数: {len(npz_df)}")

    # 从 npz_path 提取 filename 用于匹配
    npz_df['filename'] = npz_df['npz_path'].str.split('/').str[-1]

    # 合并 NPZ 路径
    sd_merged = pd.merge(
        sd_df[['image_path', 'S_gt', 'filename']],
        npz_df[['filename', 'npz_path', 'original_filename']],
        on='filename',
        how='inner'
    )
    print(f"  合并后 SD 样本数: {len(sd_merged)}")

    # 4. 创建 SD 训练索引（添加必要列）
    print("\n4. 创建 SD 训练索引")

    # 构建完整的 SD 索引，匹配 WoodScape 格式
    sd_train_df = pd.DataFrame()

    # 基本列
    sd_train_df['rgb_path'] = sd_merged['image_path'].values
    sd_train_df['npz_path'] = sd_merged['npz_path'].values
    sd_train_df['split'] = 'train'  # SD 数据全部用于训练
    # original_filename 来自 NPZ manifest
    sd_train_df['original_filename'] = sd_merged['original_filename'].str.replace('.npz', '.png').values

    # S_full_wgap_alpha50 列（来自 S_gt）
    sd_train_df['S_full_wgap_alpha50'] = sd_merged['S_gt'].values

    # 创建其他 S 列（使用 S_full_wgap_alpha50 作为占位符）
    # 注意：SD 数据只有 S_full_wgap_alpha50 是可靠的
    for col in ['S', 'S_op_only', 'S_op_sp', 'S_full', 'S_full_eta00', 'S_op', 'S_sp',
                'S_dom_eta00', 'S_dom_eta09', 'S_op_wgap', 'S_sp_wgap', 'S_dom_wgap']:
        if col not in sd_train_df.columns:
            sd_train_df[col] = sd_train_df['S_full_wgap_alpha50'].values

    # global_level 和 global_bin（基于 S_full_wgap_alpha50 计算）
    s_vals = sd_train_df['S_full_wgap_alpha50'].values
    sd_train_df['global_level'] = pd.cut(s_vals, bins=[-np.inf, 0.15, 0.35, 0.6, np.inf],
                                          labels=[0, 1, 2, 3]).astype(int)
    sd_train_df['global_bin'] = pd.cut(s_vals, bins=[-np.inf, 0.15, np.inf],
                                        labels=[0, 1]).astype(int)

    print(f"  SD 训练索引样本数: {len(sd_train_df)}")
    print(f"  S_full_wgap_alpha50 均值: {sd_train_df['S_full_wgap_alpha50'].mean():.4f}")
    print(f"  S_full_wgap_alpha50 范围: [{sd_train_df['S_full_wgap_alpha50'].min():.4f}, {sd_train_df['S_full_wgap_alpha50'].max():.4f}]")

    # 5. 合并 WoodScape 和 SD 数据
    print("\n5. 合并 WoodScape 和 SD 数据")

    # 选择 WoodScape 的 train/val 样本
    ws_train_val = ws_df[ws_df['split'].isin(['train', 'val'])].copy()
    print(f"  WoodScape train+val 样本数: {len(ws_train_val)}")

    # 合并
    merged_df = pd.concat([ws_train_val, sd_train_df], ignore_index=True)
    print(f"  合并后总样本数: {len(merged_df)}")

    print(f"\n分隔分布:")
    print(merged_df['split'].value_counts().to_string())

    # 6. 保存合并后的索引
    print("\n6. 保存合并后的索引")

    output_dir = Path('dataset/woodscape_processed/meta')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f'labels_index_mixed_989sd_{timestamp}.csv'

    merged_df.to_csv(output_path, index=False)
    print(f"  已保存: {output_path}")

    # 7. 创建统计摘要
    print("\n7. 统计摘要")

    stats = {
        'timestamp': timestamp,
        'output_path': str(output_path),
        'woodscape_train_val': len(ws_train_val),
        'sd_baseline_filtered': len(sd_train_df),
        'total_samples': len(merged_df),
        'woodscape_s_mean': float(ws_train_val['S_full_wgap_alpha50'].mean()),
        'sd_s_mean': float(sd_train_df['S_full_wgap_alpha50'].mean()),
        'split_distribution': merged_df['split'].value_counts().to_dict(),
    }

    stats_path = output_dir / f'mixed_989sd_stats_{timestamp}.json'
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  统计摘要已保存: {stats_path}")

    print("\n" + "=" * 70)
    print("准备完成!")
    print("=" * 70)
    print(f"\n下一步: 使用合并后的索引进行混合训练")
    print(f"  --index_csv {output_path}")
    print(f"  --global_target S_full_wgap_alpha50")
    print(f"  --train_split train")
    print(f"  --val_split val")


if __name__ == "__main__":
    main()
