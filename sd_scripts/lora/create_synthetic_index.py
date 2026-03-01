#!/usr/bin/env python3
"""
为SD合成数据创建训练索引CSV

功能:
1. 从manifest和NPZ manifest读取数据
2. 生成与WoodScape格式对齐的索引CSV
3. 支持train/val split配置

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="为SD合成数据创建训练索引CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--manifest_csv", type=str, required=True,
                       help="Accepted图像manifest文件路径")
    parser.add_argument("--npz_manifest", type=str, default=None,
                       help="NPZ manifest路径 (默认：<manifest_dir>/npz/manifest_npz.csv)")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="图像目录 (默认：从manifest推断)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录 (默认：与manifest同目录)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--use_resized", action="store_true",
                       help="使用resized_640x480图像而非原始512x512")

    return parser.parse_args()


def create_synthetic_index(args):
    """创建合成数据索引CSV"""
    print("=" * 70)
    print("创建合成数据训练索引")
    print("=" * 70)

    # 读取manifest
    print(f"\n读取manifest: {args.manifest_csv}")
    manifest_df = pd.read_csv(args.manifest_csv)
    print(f"样本数量: {len(manifest_df)}")

    # 读取NPZ manifest
    manifest_dir = Path(args.manifest_csv).parent
    if args.npz_manifest:
        npz_manifest_path = Path(args.npz_manifest)
    else:
        npz_manifest_path = manifest_dir / "npz" / "manifest_npz.csv"

    print(f"读取NPZ manifest: {npz_manifest_path}")
    npz_manifest_df = pd.read_csv(npz_manifest_path)
    print(f"NPZ样本数量: {len(npz_manifest_df)}")

    # 确定图像目录
    if args.image_dir:
        if args.use_resized:
            image_dir = Path(args.image_dir) / "resized_640x480"
        else:
            image_dir = Path(args.image_dir) / "original_512x512"
    else:
        if args.use_resized:
            image_dir = manifest_dir / "resized_640x480"
        else:
            image_dir = manifest_dir / "original_512x512"

    print(f"图像目录: {image_dir}")

    # 合并manifest
    print("\n合并manifest数据...")
    merged_df = pd.merge(
        manifest_df,
        npz_manifest_df[['original_filename', 'S_full', 'S_full_wgap_alpha50', 's']],
        on='original_filename',
        how='inner'
    )

    # 使用S_full_wgap_alpha50作为主要的S值（与baseline对齐）
    merged_df['S'] = merged_df['S_full_wgap_alpha50']

    # 确定global_level (基于S_full)
    def get_global_level(s_full):
        if s_full < 0.15:
            return 1
        elif s_full < 0.35:
            return 2
        elif s_full < 0.60:
            return 3
        else:
            return 4

    merged_df['global_level'] = merged_df['S_full'].apply(get_global_level)

    # 计算global_bin (使用level的bin表示，这里简化为1表示强监督数据)
    merged_df['global_bin'] = 1

    # 设置split (train/val)
    rng = np.random.default_rng(args.seed)
    n_samples = len(merged_df)
    n_val = int(n_samples * args.val_ratio)

    # 随机打乱
    indices = rng.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    merged_df['split'] = 'train'
    merged_df.loc[merged_df.index[val_indices], 'split'] = 'val'

    print(f"训练集: {len(train_indices)} 张")
    print(f"验证集: {len(val_indices)} 张")

    # 构建索引记录
    print("\n构建索引记录...")
    index_records = []

    for idx, row in merged_df.iterrows():
        # 确定图像文件名
        if args.use_resized:
            image_filename = row['resized_filename']
        else:
            image_filename = row['original_filename']

        # 确定NPZ文件名
        npz_filename = row['original_filename'].replace('.png', '.npz')

        # 构建路径
        rgb_path = image_dir / image_filename
        npz_path = manifest_dir / "npz" / npz_filename

        index_records.append({
            'rgb_path': str(rgb_path),
            'npz_path': str(npz_path),
            'split': row['split'],
            'S': row['S'],  # 使用S（已设置为S_full_wgap_alpha50）
            's': row['s'],
            'global_level': row['global_level'],
            'global_bin': row['global_bin'],
            'original_filename': row['original_filename'],
            'mask_file_id': row['mask_file_id'],
            'spill_rate': row['spill_rate'],
        })

    # 创建索引DataFrame
    index_df = pd.DataFrame(index_records)

    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = manifest_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存索引文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存完整索引
    index_path = output_dir / f"index_synthetic_{timestamp}.csv"
    index_df.to_csv(index_path, index=False)
    print(f"\n完整索引已保存: {index_path}")

    # 保存训练集和验证集分开的版本
    train_df = index_df[index_df['split'] == 'train']
    val_df = index_df[index_df['split'] == 'val']

    train_path = output_dir / f"index_synthetic_train_{timestamp}.csv"
    val_path = output_dir / f"index_synthetic_val_{timestamp}.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"训练集索引: {train_path}")
    print(f"验证集索引: {val_path}")

    # 打印统计信息
    print("\n" + "=" * 70)
    print("索引统计")
    print("=" * 70)

    print(f"\n总体S_full分布:")
    print(f"  平均: {index_df['S'].mean():.4f}")
    print(f"  中位数: {index_df['S'].median():.4f}")
    print(f"  范围: [{index_df['S'].min():.4f}, {index_df['S'].max():.4f}]")

    print(f"\n按split统计:")
    for split in ['train', 'val']:
        split_df = index_df[index_df['split'] == split]
        print(f"  {split}: n={len(split_df)}, S平均={split_df['S'].mean():.4f}")

    print(f"\n按global_level分布:")
    level_counts = index_df['global_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"  Level {level}: {count} ({count/len(index_df)*100:.1f}%)")

    print(f"\n按类别分布:")
    class_counts = index_df['mask_file_id'].str.extract(r'(\d+)_\w+')[0].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"  Camera {cls}: {count}")

    return output_dir, index_path


def main():
    args = parse_args()

    print("=" * 70)
    print("合成数据训练索引创建工具")
    print("=" * 70)
    print(f"Manifest: {args.manifest_csv}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"使用resized图像: {args.use_resized}")
    print(f"随机种子: {args.seed}")
    print("=" * 70)

    output_dir, index_path = create_synthetic_index(args)

    print(f"\n✅ 索引文件已创建: {index_path}")
    print(f"   输出目录: {output_dir}")


if __name__ == "__main__":
    main()
