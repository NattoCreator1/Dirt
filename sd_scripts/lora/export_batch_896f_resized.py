#!/usr/bin/env python3
"""
导出 batch_896f_10masks 的 resized 图像 (512x512 → 640x480)
并生成对应的 manifest

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import cv2
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="导出 batch_896f_10masks 的 resized 图像")
    parser.add_argument("--manifest", default="synthetic_soiling/batch_896f_10masks/manifest.csv",
                       help="原始 manifest 文件")
    parser.add_argument("--image_dir", default="synthetic_soiling/batch_896f_10masks/images",
                       help="原始图像目录")
    parser.add_argument("--output_dir", default="synthetic_soiling/batch_896f_10masks/resized_640x480",
                       help="输出目录")
    parser.add_argument("--target_w", type=int, default=640,
                       help="目标宽度")
    parser.add_argument("--target_h", type=int, default=480,
                       help="目标高度")

    args = parser.parse_args()

    # 读取 manifest
    print(f"读取 manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    print(f"  样本数: {len(df)}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir)

    # 处理每张图像
    print(f"\n开始导出 resized 图像...")
    print(f"  源目录: {args.image_dir}")
    print(f"  目标目录: {args.output_dir}")
    print(f"  目标分辨率: {args.target_w}x{args.target_h}")

    resized_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=100):
        # 原始图像路径
        src_path = image_dir / row["output_filename"]

        if not src_path.exists():
            print(f"  警告: 图像不存在: {src_path}")
            continue

        # 读取图像
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  警告: 无法读取图像: {src_path}")
            continue

        # Resize
        img_resized = cv2.resize(img, (args.target_w, args.target_h),
                                 interpolation=cv2.INTER_AREA)

        # 保存
        dst_path = output_dir / row["output_filename"]
        cv2.imwrite(str(dst_path), img_resized)

        resized_records.append({
            "original_filename": row["output_filename"],
            "resized_path": str(dst_path),
        })

    # 保存 resized manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resized_manifest_path = output_dir / f"manifest_resized_{timestamp}.csv"
    resized_df = pd.DataFrame(resized_records)
    resized_df.to_csv(resized_manifest_path, index=False)

    print(f"\n完成!")
    print(f"  导出图像数: {len(resized_records)}")
    print(f"  Resized manifest: {resized_manifest_path}")

    # 验证
    print(f"\n验证:")
    print(f"  输出目录文件数: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
