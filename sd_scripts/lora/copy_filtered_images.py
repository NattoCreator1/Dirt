#!/usr/bin/env python3
"""
复制筛选后的合成图像

根据筛选结果CSV，将accepted的图像复制到指定目录。
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd


def copy_filtered_images(csv_path, output_dir, status='accepted', dry_run=False):
    """
    复制筛选后的图像

    Args:
        csv_path: 筛选结果CSV文件路径
        output_dir: 输出目录
        status: 要复制的筛选状态 ('accepted' 或 'rejected')
        dry_run: 预演模式，不实际复制文件
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    # 读取筛选结果
    df = pd.read_csv(csv_path)

    # 筛选指定状态的图像
    filtered = df[df['filter_status'] == status]

    print(f"=" * 60)
    print(f"复制筛选结果")
    print(f"=" * 60)
    print(f"输入CSV: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"筛选状态: {status}")
    print(f"图像数量: {len(filtered)}")
    print(f"模式: {'预演 (不实际复制)' if dry_run else '实际复制'}")
    print(f"=" * 60)

    if len(filtered) == 0:
        print("没有找到匹配的图像")
        return

    # 创建输出目录
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 复制图像
    source_dir = csv_path.parent / "images"

    copied = 0
    skipped = 0

    for _, row in filtered.iterrows():
        filename = row['output_filename']
        src = source_dir / filename
        dst = output_dir / filename

        if not src.exists():
            print(f"⚠️ 跳过 (源文件不存在): {filename}")
            skipped += 1
            continue

        if not dry_run:
            shutil.copy2(src, dst)

        copied += 1
        if copied <= 10 or copied % 50 == 0:
            print(f"  [{copied}/{len(filtered)}] {filename}")

    print(f"\n完成!")
    print(f"  成功复制: {copied}")
    print(f"  跳过: {skipped}")
    print(f"  输出目录: {output_dir}")

    # 生成简要统计
    print(f"\n质量指标统计:")
    if 'spill_rate' in filtered.columns:
        print(f"  Spill Rate - 平均: {filtered['spill_rate'].mean():.3f}, 中位数: {filtered['spill_rate'].median():.3f}")
    if 'mask_dominant_class' in filtered.columns:
        print(f"  类别分布:")
        for cls, count in filtered['mask_dominant_class'].value_counts().sort_index().items():
            print(f"    C{int(cls)}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="复制筛选后的合成图像",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", type=str, required=True,
                       help="筛选结果CSV文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--status", type=str, default="accepted",
                       choices=["accepted", "rejected"],
                       help="要复制的筛选状态")
    parser.add_argument("--dry_run", action="store_true",
                       help="预演模式，不实际复制文件")

    args = parser.parse_args()

    copy_filtered_images(args.csv, args.output_dir, args.status, args.dry_run)


if __name__ == "__main__":
    main()
