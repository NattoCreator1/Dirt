#!/usr/bin/env python3
"""
使用手动标注的 mask 将 SD 脏污图像转换为 RGBA 叠加层

输入：
- SD 生成的脏污图像（RGB）
- 手动标注的 mask（黑白图像，白色=脏污，黑色=透明）

输出：
- RGBA 脏污层（RGB 来自 SD 图像，Alpha 来自 mask）
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2


def ensure_dir(p: str):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def load_mask(mask_path: str, target_size: tuple = None) -> np.ndarray:
    """
    加载手动标注的 mask

    Args:
        mask_path: mask 文件路径
        target_size: 目标尺寸 (width, height)，None 表示保持原尺寸

    Returns:
        alpha 通道 (H, W), uint8 [0, 255]
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # 确保是二值图像
    # 假设白色 (255) = 脏污，黑色 (0) = 透明
    # 如果使用其他颜色，可以调整阈值
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Resize 到目标尺寸
    if target_size is not None:
        binary = cv2.resize(binary, target_size, interpolation=cv2.INTER_NEAREST)

    return binary


def convert_single_image(synthetic_path: str, mask_path: str, output_path: str) -> bool:
    """
    转换单个合成图像为 RGBA 叠加层

    Args:
        synthetic_path: SD 生成的脏污图像路径（RGB）
        mask_path: 手动标注的 mask 路径（黑白灰度图）
        output_path: 输出 RGBA 路径

    Returns:
        是否成功
    """
    # 读取 SD 生成的脏污图
    synthetic_bgr = cv2.imread(synthetic_path, cv2.IMREAD_COLOR)
    if synthetic_bgr is None:
        return False

    # 读取手动标注的 mask
    alpha = load_mask(mask_path, (synthetic_bgr.shape[1], synthetic_bgr.shape[0]))
    if alpha is None:
        return False

    # 转换为 RGBA
    synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([synthetic_rgb, alpha])  # (H, W, 4)

    # 保存
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, rgba)

    return True


class ManualMaskConverter:
    """使用手动标注 mask 批量转换脏污层"""

    def __init__(self,
                 annotation_dir: str,
                 output_dir: str,
                 copy_without_mask: bool = False):
        """
        Args:
            annotation_dir: 标注目录（包含原图和 _mask.png）
            output_dir: RGBA 输出目录
            copy_without_mask: 是否复制没有 mask 的图像（保持原RGB）
        """
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.copy_without_mask = copy_without_mask

        # 加载映射表
        mapping_path = os.path.join(annotation_dir, "file_mapping.csv")
        self.mapping_df = pd.read_csv(mapping_path, encoding='utf-8-sig')

        # 创建输出目录
        ensure_dir(output_dir)

        # 统计
        self.stats = {
            'total': len(self.mapping_df),
            'with_mask': 0,
            'without_mask': 0,
            'error': 0,
        }

    def convert_all(self):
        """批量转换"""
        print(f"开始转换 {len(self.mapping_df)} 个脏污层...")
        print(f"标注目录: {self.annotation_dir}")
        print(f"输出目录: {self.output_dir}")
        print()

        for _, row in tqdm(self.mapping_df.iterrows(), total=len(self.mapping_df), desc="转换进度"):
            annotation_file = row['annotation_file']
            original_file = row['original_file']

            # 路径
            image_path = os.path.join(self.annotation_dir, annotation_file)
            mask_path = os.path.join(self.annotation_dir, annotation_file.replace('.png', '_mask.png'))

            # 输出路径（使用原始文件名）
            output_path = os.path.join(self.output_dir, original_file)

            # 检查 mask 是否存在
            if os.path.exists(mask_path):
                # 有 mask：转换
                success = convert_single_image(image_path, mask_path, output_path)
                if success:
                    self.stats['with_mask'] += 1
                else:
                    self.stats['error'] += 1
            else:
                # 没有 mask
                self.stats['without_mask'] += 1
                if self.copy_without_mask:
                    # 复制原图（保持 RGB）
                    import shutil
                    shutil.copy2(image_path, output_path)

        self._print_summary()

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("转换完成")
        print("="*60)
        print(f"总数: {self.stats['total']}")
        print(f"有 mask (已转换): {self.stats['with_mask']}")
        print(f"无 mask: {self.stats['without_mask']}")
        print(f"错误: {self.stats['error']}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="使用手动标注的 mask 将 SD 脏污图转换为 RGBA 透明叠加层",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用流程:
  1. 运行 prepare_for_manual_annotation.py 准备标注素材
  2. 手动为每张图像标注脏污区域（保存为 _mask.png）
  3. 运行此脚本进行批量转换

Mask 格式:
  - 黑色 (#000000): 透明区域（alpha=0）
  - 白色 (#FFFFFF): 脏污区域（alpha=255）
  - 支持灰度边缘实现羽化效果

输出 RGBA 图像:
  - RGB: SD 生成的脏污颜色
  - A: 根据 mask 设置的透明度
        """
    )

    ap.add_argument("--annotation_dir", type=str,
                    default="dataset/sd_temporal_training/for_annotation",
                    help="标注目录（包含原图和 _mask.png）")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/dirt_layers_manual_rgba",
                    help="RGBA 输出目录")
    ap.add_argument("--copy_without_mask", action="store_true",
                    help="复制没有 mask 的图像（保持 RGB）")

    args = ap.parse_args()

    converter = ManualMaskConverter(
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        copy_without_mask=args.copy_without_mask,
    )

    converter.convert_all()


if __name__ == "__main__":
    main()
