#!/usr/bin/env python3
"""
从 SD inpainting 生成结果中提取纯净的脏污残差层

核心思想：
1. SD 生成时使用"强力降语义"的干净底图（模糊/灰度/低分辨率）
2. 提取残差：Residual = I_SD - I_clean_base
3. 残差层只包含"新增的脏污纹理"，不包含原始背景语义
4. 合成时：I_output = I_target_clean + Alpha ⊙ Residual

优点：
- 从根本上解决"另一条道路"的语义泄漏问题
- 不需要手动标注 mask
- 可批量处理大量数据

作者: soiling_project
日期: 2026-02-27
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional

import cv2


def ensure_dir(p: str):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def create_degraded_base(clean_image: np.ndarray,
                         method: str = 'blur_gray') -> np.ndarray:
    """
    创建降语义的干净底图

    目标：降低底图的视觉语义，使残差只包含"新增的脏污"

    Args:
        clean_image: 原始干净图像 (H, W, 3), BGR, uint8
        method: 降语义方法
            - 'blur_gray': 强模糊 + 转灰度 + 还原
            - 'blur': 仅强模糊
            - 'downsample': 下采样再上采样
            - 'flat': 纯灰色底图

    Returns:
        降语义底图 (H, W, 3), BGR, uint8
    """
    h, w = clean_image.shape[:2]

    if method == 'blur_gray':
        # 方法1：强模糊 + 转灰度 + 还原为RGB
        # 最有效：彻底破坏结构，保留亮度
        gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
        # 强模糊
        blurred = cv2.GaussianBlur(gray, (41, 41), 10)
        # 还原为 BGR
        degraded = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    elif method == 'blur':
        # 方法2：仅强模糊
        degraded = cv2.GaussianBlur(clean_image, (41, 41), 10)

    elif method == 'downsample':
        # 方法3：下采样再上采样（丢失高频细节）
        small = cv2.resize(clean_image, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    elif method == 'flat':
        # 方法4：纯灰色底图
        gray_mean = int(clean_image.astype(np.float32).mean())
        degraded = np.full_like(clean_image, gray_mean)

    else:
        degraded = clean_image.copy()

    return degraded


def extract_residual_layer(sd_image: np.ndarray,
                           clean_base: np.ndarray,
                           rgblabel_mask: Optional[np.ndarray] = None,
                           method: str = 'diff') -> np.ndarray:
    """
    从 SD 生成图中提取脏污残差层

    Args:
        sd_image: SD inpainting 生成的脏污图 (H, W, 3), BGR
        clean_base: 降语义的干净底图 (H, W, 3), BGR
        rgblabel_mask: rgbLabels mask (H, W), 0=clean, 1=dirt (可选)
        method: 提取方法
            - 'diff': 简单差值 Residual = I_SD - I_base
            - 'guided': 引导滤波保留细节
            - 'mask_only': 仅保留 mask 区域的残差

    Returns:
        残差层 (H, W, 3), float32 (可能有负值)
    """
    # 确保尺寸一致
    if sd_image.shape != clean_base.shape:
        clean_base = cv2.resize(clean_base,
                               (sd_image.shape[1], sd_image.shape[0]),
                               interpolation=cv2.INTER_AREA)

    # 转换到 float
    sd_float = sd_image.astype(np.float32)
    base_float = clean_base.astype(np.float32)

    if method == 'diff':
        # 简单差值
        residual = sd_float - base_float

    elif method == 'mask_only':
        # 仅在 mask 区域提取残差
        if rgblabel_mask is None:
            # 没有 mask，使用全部
            residual = sd_float - base_float
        else:
            # Resize mask 到匹配
            if rgblabel_mask.shape[:2] != sd_float.shape[:2]:
                rgblabel_mask = cv2.resize(rgblabel_mask,
                                          (sd_float.shape[1], sd_float.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

            # 创建 mask (H, W, 1)
            mask_3ch = rgblabel_mask[:, :, np.newaxis]

            # mask 外残差为 0
            residual = (sd_float - base_float) * mask_3ch

    elif method == 'guided':
        # 引导滤波保留细节（需要 mask）
        if rgblabel_mask is not None:
            if rgblabel_mask.shape[:2] != sd_float.shape[:2]:
                rgblabel_mask = cv2.resize(rgblabel_mask,
                                          (sd_float.shape[1], sd_float.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

            # 使用引导滤波平滑残差
            residual = sd_float - base_float
            # 对每个通道分别滤波
            residual_filtered = np.zeros_like(residual)
            for c in range(3):
                residual_filtered[:, :, c] = cv2.ximgproc.guidedFilter(
                    base_float[:, :, c],  # guide
                    residual[:, :, c],     # src
                    radius=8,
                    eps=1e-3
                )
            residual = residual_filtered
        else:
            residual = sd_float - base_float

    else:
        residual = sd_float - base_float

    return residual


def residual_to_rgba(residual: np.ndarray,
                     alpha_mask: np.ndarray,
                     output_mode: str = 'residual') -> np.ndarray:
    """
    将残差层转换为 RGBA 图像

    Args:
        residual: 残差层 (H, W, 3), float32
        alpha_mask: Alpha mask (H, W), uint8 [0, 255]
        output_mode: 输出模式
            - 'residual': RGB = 残差值 (可能为负)
            - 'absolute': RGB = 绝对值
            - 'clamp': RGB = 截断到 [0, 255]
            - 'sd_color': RGB = SD 生成图的颜色

    Returns:
        RGBA 图像 (H, W, 4), uint8
    """
    h, w = residual.shape[:2]

    if output_mode == 'residual':
        # 保留残差值（有偏移以便存储）
        # 残差范围约 [-255, 255]，加 128 映射到 [0, 255]
        rgb = np.clip(residual + 128, 0, 255).astype(np.uint8)

    elif output_mode == 'absolute':
        # 绝对值
        rgb = np.abs(residual).astype(np.uint8)

    elif output_mode == 'clamp':
        # 截断到 [0, 255]
        rgb = np.clip(residual, 0, 255).astype(np.uint8)

    elif output_mode == 'sd_color':
        # 使用 SD 生成图的颜色（残差只用于 mask）
        # 这个模式需要在调用时传入原始 sd_image
        raise NotImplementedError("Use 'compose_direct' instead")

    else:
        rgb = np.clip(residual + 128, 0, 255).astype(np.uint8)

    # 确保 alpha 是 (H, W)
    if alpha_mask.ndim == 3:
        alpha_mask = alpha_mask[:, :, 0]

    # 组合 RGBA
    rgba = np.dstack([rgb, alpha_mask.astype(np.uint8)])

    return rgba


def compose_with_residual(target_clean: np.ndarray,
                          residual: np.ndarray,
                          alpha: np.ndarray,
                          strength: float = 1.0) -> np.ndarray:
    """
    直接使用残差层合成到目标干净帧

    公式: I_output = I_clean + strength * Alpha ⊙ Residual

    Args:
        target_clean: 目标干净帧 (H, W, 3), uint8
        residual: 残差层 (H, W, 3), float32
        alpha: Alpha mask (H, W), float32 [0, 1]
        strength: 强度系数

    Returns:
        合成图像 (H, W, 3), uint8
    """
    # 确保尺寸一致
    if residual.shape[:2] != target_clean.shape[:2]:
        residual = cv2.resize(residual,
                             (target_clean.shape[1], target_clean.shape[0]),
                             interpolation=cv2.INTER_AREA)
    if alpha.shape[:2] != target_clean.shape[:2]:
        alpha = cv2.resize(alpha,
                          (target_clean.shape[1], target_clean.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # 扩展 alpha 到 3 通道
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    # 合成
    clean_float = target_clean.astype(np.float32)
    composed = clean_float + strength * alpha * residual
    composed = np.clip(composed, 0, 255).astype(np.uint8)

    return composed


class ResidualLayerExtractor:
    """批量提取残差层"""

    def __init__(self,
                 sd_dir: str,
                 clean_dir: str,
                 rgblabels_dir: str,
                 output_dir: str,
                 degrade_method: str = 'blur_gray',
                 extract_method: str = 'diff',
                 output_mode: str = 'residual'):
        """
        Args:
            sd_dir: SD 生成图像目录
            clean_dir: 对应干净帧目录
            rgblabels_dir: rgbLabels 目录
            output_dir: 输出目录
            degrade_method: 降语义方法
            extract_method: 残差提取方法
            output_mode: 输出模式
        """
        self.sd_dir = sd_dir
        self.clean_dir = clean_dir
        self.rgblabels_dir = rgblabels_dir
        self.output_dir = output_dir
        self.degrade_method = degrade_method
        self.extract_method = extract_method
        self.output_mode = output_mode

        ensure_dir(output_dir)

        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
        }

    def parse_sd_filename(self, filename: str) -> Optional[Dict]:
        """解析 SD 文件名，提取 mask_id"""
        # 移除后缀
        base = filename.replace('_640x480.png', '').replace('.png', '')
        parts = base.rsplit('_', 3)

        if len(parts) < 4:
            return None

        sample_str = parts[-1]  # s000
        view = parts[-2]         # FV/MVL/MVR/RV
        frame_id = parts[-3]     # 2521

        if not sample_str.startswith('s'):
            return None

        mask_id = f"{frame_id}_{view}"
        return {'mask_id': mask_id, 'view': view, 'frame_id': frame_id}

    def process_single(self, sd_filename: str, clean_filename: str) -> bool:
        """处理单个文件"""
        # 读取 SD 图像
        sd_path = os.path.join(self.sd_dir, sd_filename)
        sd_image = cv2.imread(sd_path)
        if sd_image is None:
            return False

        # 读取干净图像
        clean_path = os.path.join(self.clean_dir, clean_filename)
        clean_image = cv2.imread(clean_path)
        if clean_image is None:
            return False

        # 创建降语义底图
        degraded_base = create_degraded_base(clean_image, self.degrade_method)

        # 读取 rgbLabels mask
        info = self.parse_sd_filename(sd_filename)
        rgblabel_mask = None
        if info:
            rgblabel_path = os.path.join(self.rgblabels_dir, f"{info['mask_id']}.png")
            if os.path.exists(rgblabel_path):
                rgblabel = cv2.imread(rgblabel_path)
                rgblabel_rgb = cv2.cvtColor(rgblabel, cv2.COLOR_BGR2RGB)
                # 转换为 binary mask
                rgblabel_mask = (rgblabel_rgb.sum(axis=2) > 0).astype(np.uint8) * 255

        # 提取残差
        residual = extract_residual_layer(
            sd_image, degraded_base, rgblabel_mask, self.extract_method
        )

        # 生成 alpha mask
        if rgblabel_mask is not None:
            alpha_mask = rgblabel_mask
        else:
            # 没有 mask，根据残差强度生成
            residual_strength = np.abs(residual).sum(axis=2)
            alpha_mask = (residual_strength > 20).astype(np.uint8) * 255

        # 转换为 RGBA
        rgba = residual_to_rgba(residual, alpha_mask, self.output_mode)

        # 保存
        output_path = os.path.join(self.output_dir, sd_filename)
        cv2.imwrite(output_path, rgba)

        return True

    def process_batch(self, manifest_path: str):
        """批量处理"""
        df = pd.read_csv(manifest_path, encoding='utf-8-sig')
        self.stats['total'] = len(df)

        print(f"开始提取残差层...")
        print(f"  SD 目录: {self.sd_dir}")
        print(f"  干净帧目录: {self.clean_dir}")
        print(f"  降语义方法: {self.degrade_method}")
        print(f"  提取方法: {self.extract_method}")
        print()

        for _, row in tqdm(df.iterrows(), total=len(df), desc="提取进度"):
            sd_filename = row['filename']

            # 查找对应的干净帧
            # 从 source_path 提取
            if 'source_path' in row:
                # 使用清单中的源路径
                pass

            # 简化：使用文件名匹配
            # SD: ..._2521_FV_s000.png
            # Clean: ..._2521_FV_...jpg

            # 尝试直接处理
            success = self.process_single(sd_filename, sd_filename)

            if success:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

        self._print_summary()

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("残差层提取完成")
        print("="*60)
        print(f"总数: {self.stats['total']}")
        print(f"成功: {self.stats['success']}")
        print(f"失败: {self.stats['failed']}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="从 SD inpainting 结果中提取纯净的脏污残差层",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
核心思想:
  通过"降语义底图 + 残差提取"，获得不含背景语义的纯净脏污层。

  公式:
    I_degraded = Degrade(I_clean)
    Residual = I_SD - I_degraded
    I_output = I_target + Alpha ⊙ Residual

  这样即使 SD 生成时带背景重绘，也会被残差化抑制。
        """
    )

    ap.add_argument("--sd_dir", type=str,
                    default="sd_scripts/lora/accepted_20260224_164715/resized_640x480",
                    help="SD 生成图像目录")
    ap.add_argument("--clean_dir", type=str,
                    default="dataset/my_clean_frames_4by3",
                    help="对应干净帧目录")
    ap.add_argument("--rgblabels_dir", type=str,
                    default="dataset/woodscape_raw/train/rgbLabels",
                    help="rgbLabels 目录")
    ap.add_argument("--manifest", type=str,
                    default="dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv",
                    help="清单文件")

    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/dirt_layers_residual",
                    help="输出目录")

    ap.add_argument("--degrade_method", type=str,
                    choices=['blur_gray', 'blur', 'downsample', 'flat'],
                    default='blur_gray',
                    help="降语义方法")
    ap.add_argument("--extract_method", type=str,
                    choices=['diff', 'guided', 'mask_only'],
                    default='diff',
                    help="残差提取方法")
    ap.add_argument("--output_mode", type=str,
                    choices=['residual', 'absolute', 'clamp'],
                    default='residual',
                    help="输出模式")

    args = ap.parse_args()

    extractor = ResidualLayerExtractor(
        sd_dir=args.sd_dir,
        clean_dir=args.clean_dir,
        rgblabels_dir=args.rgblabels_dir,
        output_dir=args.output_dir,
        degrade_method=args.degrade_method,
        extract_method=args.extract_method,
        output_mode=args.output_mode,
    )

    extractor.process_batch(args.manifest)


if __name__ == "__main__":
    main()
