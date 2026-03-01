#!/usr/bin/env python3
"""
使用 rgbLabels 将 SD 生成的脏污图像转换为 RGBA 透明叠加层

核心思路：
- SD 生成的图像是完整的 RGB（背景+脏污）
- rgbLabels 中：黑色=干净（透明），彩色=脏污（保留）
- 直接用 rgbLabels 生成 alpha 通道，而不是计算像素差异

rgbLabels 颜色编码（Woodscape 标准）：
- [0, 0, 0]     (黑色)  = Clean      → alpha = 0
- [0, 255, 0]   (绿色)  = Transparent → alpha = 85 (轻度)
- [0, 0, 255]   (蓝色)  = Semi-transparent → alpha = 170 (中度)
- [255, 0, 0]   (红色)  = Opaque     → alpha = 255 (完全不透明)

作者: soiling_project
日期: 2026-02-27
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import cv2


# rgbLabels 颜色编码
RGBLABELS_COLORS = {
    (0, 0, 0): (0, 'clean'),          # 黑色 → 完全透明
    (0, 255, 0): (1, 'transparent'),  # 绿色 → 透明脏污
    (0, 0, 255): (2, 'semi'),         # 蓝色 → 半透明
    (255, 0, 0): (3, 'opaque'),       # 红色 → 不透明
}

# Alpha 值设置（可调整）
ALPHA_VALUES = {
    0: 0,     # clean → 完全透明
    1: 255,   # transparent → 不透明（保留脏污颜色）
    2: 255,   # semi → 不透明
    3: 255,   # opaque → 不透明
}


def parse_synthetic_filename(filename: str) -> dict:
    """
    解析合成脏污图像的文件名，提取 mask_id

    输入格式: {base_clean_name}_{frame_id}_{view}_s{sample}_640x480.png
    例如: ..._2521_FV_s000_640x480.png

    返回: {'mask_id': '2521_FV', 'view': 'FV', 'frame_id': '2521'}
    """
    # 移除后缀
    base = filename.replace('_640x480.png', '').replace('.png', '')

    # 从右边分割: _s000 _{view} _{frame_id} _{base}
    parts = base.rsplit('_', 3)
    if len(parts) < 4:
        return None

    sample_str = parts[-1]  # s000
    view = parts[-2]         # FV/MVL/MVR/RV
    frame_id = parts[-3]     # 2521

    if not sample_str.startswith('s'):
        return None

    mask_id = f"{frame_id}_{view}"
    return {
        'mask_id': mask_id,
        'view': view,
        'frame_id': frame_id,
    }


def find_rgblabel_path(mask_id: str, rgblabels_dir: str) -> str:
    """
    查找 rgbLabels 文件路径

    Args:
        mask_id: 如 '2521_FV'
        rgblabels_dir: rgbLabels 目录

    Returns:
        完整路径，找不到返回 None
    """
    path = os.path.join(rgblabels_dir, f"{mask_id}.png")
    if os.path.exists(path):
        return path
    return None


def rgblabels_to_alpha(rgblabel_bgr: np.ndarray) -> np.ndarray:
    """
    将 rgbLabels (BGR) 转换为 alpha 通道

    Args:
        rgblabel_bgr: rgbLabels 图像 (H, W, 3), BGR, uint8

    Returns:
        alpha: Alpha 通道 (H, W), uint8 [0, 255]
    """
    # BGR → RGB
    rgb = cv2.cvtColor(rgblabel_bgr, cv2.COLOR_BGR2RGB)

    # 初始化 alpha
    h, w = rgb.shape[:2]
    alpha = np.zeros((h, w), dtype=np.uint8)

    # 按颜色设置 alpha
    for color_bgr, (class_id, name) in RGBLABELS_COLORS.items():
        color_rgb = color_bgr[::-1]  # BGR → RGB
        mask = (rgb == color_rgb).all(axis=2)
        alpha[mask] = ALPHA_VALUES[class_id]

    return alpha


def convert_single_image(synthetic_path: str,
                        rgblabel_path: str,
                        output_path: str,
                        resize_rgblabel: bool = True) -> bool:
    """
    转换单个合成图像为 RGBA 叠加层

    Args:
        synthetic_path: 合成图像路径 (640x480 RGB)
        rgblabel_path: rgbLabels 路径 (1280x960 RGB)
        output_path: 输出 RGBA 路径
        resize_rgblabel: 是否 resize rgbLabels 到合成图尺寸

    Returns:
        是否成功
    """
    # 读取合成图
    synthetic_bgr = cv2.imread(synthetic_path, cv2.IMREAD_COLOR)
    if synthetic_bgr is None:
        return False

    # 读取 rgbLabels
    rgblabel_bgr = cv2.imread(rgblabel_path, cv2.IMREAD_COLOR)
    if rgblabel_bgr is None:
        return False

    # Resize rgbLabels 到合成图尺寸（如果需要）
    if resize_rgblabel and rgblabel_bgr.shape[:2] != synthetic_bgr.shape[:2]:
        rgblabel_bgr = cv2.resize(rgblabel_bgr,
                                  (synthetic_bgr.shape[1], synthetic_bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

    # 生成 alpha 通道
    alpha = rgblabels_to_alpha(rgblabel_bgr)

    # 转换为 RGBA
    synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([synthetic_rgb, alpha])  # (H, W, 4)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, rgba)

    return True


class DirtLayerConverterRGBLabels:
    """使用 rgbLabels 批量转换脏污层"""

    def __init__(self,
                 dirt_manifest_path: str,
                 dirt_source_dir: str,
                 rgblabels_dir: str,
                 output_dir: str,
                 alpha_values: dict = None):
        """
        Args:
            dirt_manifest_path: 脏污层清单文件
            dirt_source_dir: 脏污层源图像目录（RGB 合成图）
            rgblabels_dir: rgbLabels 目录
            output_dir: RGBA 输出目录
            alpha_values: 自定义 alpha 值 {class_id: alpha_value}
        """
        self.dirt_manifest_path = dirt_manifest_path
        self.dirt_source_dir = dirt_source_dir
        self.rgblabels_dir = rgblabels_dir
        self.output_dir = output_dir

        # 更新 alpha 值
        if alpha_values:
            global ALPHA_VALUES
            ALPHA_VALUES.update(alpha_values)

        # 加载清单
        self.df = pd.read_csv(dirt_manifest_path, encoding='utf-8-sig')

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 统计
        self.stats = {
            'total': len(self.df),
            'success': 0,
            'parse_failed': 0,
            'rgblabel_not_found': 0,
            'read_error': 0,
        }

    def convert_all(self):
        """批量转换"""
        print(f"开始转换 {len(self.df)} 个脏污层...")
        print(f"rgbLabels 目录: {self.rgblabels_dir}")
        print(f"Alpha 值设置: {ALPHA_VALUES}")
        print()

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="转换进度"):
            filename = row['filename']

            # 解析文件名
            info = parse_synthetic_filename(filename)
            if info is None:
                self.stats['parse_failed'] += 1
                continue

            # 源路径
            source_path = os.path.join(self.dirt_source_dir, filename)
            if not os.path.exists(source_path):
                if filename.endswith('.png'):
                    base = filename[:-4]
                    source_path = os.path.join(self.dirt_source_dir, base + '_640x480.png')

            if not os.path.exists(source_path):
                self.stats['read_error'] += 1
                continue

            # 查找 rgbLabels
            rgblabel_path = find_rgblabel_path(info['mask_id'], self.rgblabels_dir)
            if rgblabel_path is None:
                self.stats['rgblabel_not_found'] += 1
                continue

            # 输出路径
            output_path = os.path.join(self.output_dir, filename)

            # 转换
            success = convert_single_image(source_path, rgblabel_path, output_path)

            if success:
                self.stats['success'] += 1
            else:
                self.stats['read_error'] += 1

        self._print_summary()

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("转换完成")
        print("="*60)
        print(f"总数: {self.stats['total']}")
        print(f"成功: {self.stats['success']} ({self.stats['success']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"解析失败: {self.stats['parse_failed']}")
        print(f"rgbLabels 未找到: {self.stats['rgblabel_not_found']}")
        print(f"读取错误: {self.stats['read_error']}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="使用 rgbLabels 将 SD RGB 脏污图转换为 RGBA 透明叠加层",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
转换原理:
  SD inpainting 生成完整 RGB 图像（背景+脏污）。
  Woodscape rgbLabels 使用颜色编码标注脏污区域：
    - 黑色 [0,0,0]     → Clean      → Alpha = 0 (完全透明)
    - 绿色 [0,255,0]   → Transparent → Alpha = 255
    - 蓝色 [0,0,255]   → Semi-transparent → Alpha = 255
    - 红色 [255,0,0]   → Opaque     → Alpha = 255

  直接用 rgbLabels 生成 alpha，无需计算像素差异。

  输出 RGBA 图像：
    - RGB = 合成图的颜色（脏污颜色）
    - A = 根据 rgbLabels 设置的透明度
        """
    )

    ap.add_argument("--dirt_manifest", type=str,
                    default="dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv",
                    help="脏污层清单文件")
    ap.add_argument("--dirt_source_dir", type=str,
                    default="sd_scripts/lora/accepted_20260224_164715/resized_640x480",
                    help="脏污层源图像目录（RGB 合成图）")
    ap.add_argument("--rgblabels_dir", type=str,
                    default="dataset/woodscape_raw/train/rgbLabels",
                    help="rgbLabels 目录")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/dirt_layers_rgba",
                    help="RGBA 输出目录")

    ap.add_argument("--max_samples", type=int, default=None,
                    help="最大处理样本数（测试用）")

    args = ap.parse_args()

    converter = DirtLayerConverterRGBLabels(
        dirt_manifest_path=args.dirt_manifest,
        dirt_source_dir=args.dirt_source_dir,
        rgblabels_dir=args.rgblabels_dir,
        output_dir=args.output_dir,
    )

    # 可选：限制样本数
    if args.max_samples:
        converter.df = converter.df.head(args.max_samples)
        print(f"测试模式: 仅处理 {args.max_samples} 个样本")

    converter.convert_all()


if __name__ == "__main__":
    main()
