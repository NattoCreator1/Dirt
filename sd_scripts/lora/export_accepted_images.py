#!/usr/bin/env python3
"""
导出Accepted图像并生成多分辨率版本

功能:
1. 从筛选结果CSV中读取accepted样本
2. 复制原始512x512图像
3. 生成640x480版本（与baseline对齐）
4. 生成新的manifest文件

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="导出Accepted图像并生成多分辨率版本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--filtered_csv", type=str, required=True,
                       help="筛选结果CSV文件路径")
    parser.add_argument("--source_images", type=str,
                       default="synthetic_soiling/batch_896f_10masks/images",
                       help="源图像目录")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认：accepted_<timestamp>）")
    parser.add_argument("--resize_to", type=str, nargs=2, default=["640", "480"],
                       help="目标分辨率 宽 高")

    return parser.parse_args()


def export_accepted_images(args):
    """导出accepted图像"""
    print("=" * 70)
    print("导出Accepted图像")
    print("=" * 70)

    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("sd_scripts/lora") / f"accepted_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    original_dir = output_dir / "original_512x512"
    resized_dir = output_dir / f"resized_{args.resize_to[0]}x{args.resize_to[1]}"
    original_dir.mkdir(exist_ok=True)
    resized_dir.mkdir(exist_ok=True)

    # 读取筛选结果
    print(f"\n读取筛选结果: {args.filtered_csv}")
    df = pd.read_csv(args.filtered_csv)

    # 筛选accepted
    accepted = df[df['filter_status'] == 'accepted'].copy()
    print(f"Accepted样本: {len(accepted)} 张")

    # 目标分辨率
    target_w, target_h = int(args.resize_to[0]), int(args.resize_to[1])
    print(f"目标分辨率: {target_w}x{target_h}")

    # 源图像目录
    source_dir = Path(args.source_images)
    print(f"源图像目录: {source_dir}")

    # 处理每张图像
    manifest = []
    skipped = []

    for idx, row in tqdm(accepted.iterrows(), total=len(accepted), desc="导出图像"):
        filename = row['output_filename']
        src_path = source_dir / filename

        if not src_path.exists():
            skipped.append(filename)
            continue

        # 读取图像
        img = cv2.imread(str(src_path))
        if img is None:
            skipped.append(filename)
            continue

        # 保存原始512x512版本
        dst_original = original_dir / filename
        shutil.copy2(src_path, dst_original)

        # 生成640x480版本
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized_filename = filename.replace('.png', f'_{target_w}x{target_h}.png')
        dst_resized = resized_dir / resized_filename
        cv2.imwrite(str(dst_resized), img_resized)

        # 记录到manifest
        manifest_entry = {
            'original_filename': filename,
            'resized_filename': resized_filename,
            'original_path': str(dst_original),
            'resized_path': str(dst_resized),
            'filter_status': 'accepted',
            'clean_file_id': row.get('clean_file_id', ''),
            'clean_camera': row.get('clean_camera', ''),
            'mask_file_id': row.get('mask_file_id', ''),
            'spill_rate': row.get('spill_rate', ''),
            'coverage': row.get('coverage', ''),
            'mask_dominant_class': row.get('mask_dominant_class', ''),
            'caption': row.get('caption', ''),
        }
        manifest.append(manifest_entry)

    # 保存manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = output_dir / "manifest_accepted.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # 打印摘要
    print("\n" + "=" * 70)
    print("导出完成!")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"成功导出: {len(manifest)} 张")
    print(f"跳过: {len(skipped)} 张")

    if skipped:
        print(f"\n跳过的文件:")
        for f in skipped[:5]:
            print(f"  - {f}")
        if len(skipped) > 5:
            print(f"  ... 还有 {len(skipped)-5} 个")

    print(f"\n目录结构:")
    print(f"  {original_dir}/")
    print(f"  {resized_dir}/")
    print(f"  {manifest_path}")

    return output_dir, len(manifest)


def main():
    args = parse_args()

    print("=" * 70)
    print("导出Accepted图像工具")
    print("=" * 70)
    print(f"筛选结果: {args.filtered_csv}")
    print(f"源图像: {args.source_images}")
    print(f"目标分辨率: {args.resize_to[0]}x{args.resize_to[1]}")
    print("=" * 70)

    output_dir, count = export_accepted_images(args)

    print(f"\n✅ 成功导出 {count} 张Accepted图像到: {output_dir}")


if __name__ == "__main__":
    main()
