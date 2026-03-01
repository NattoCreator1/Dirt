#!/usr/bin/env python3
"""
将 SD 生成的 RGB 脏污图像转换为 RGBA 透明叠加层

问题：SD inpainting 生成的是完整的 RGB 图像（背景+脏污），不是透明叠加层
解决：通过与原始干净帧对比，提取纯脏污层并生成 alpha 通道

转换策略：
1. 解析文件名，找到对应的干净帧
2. 计算合成图与干净图的差异
3. 生成 alpha 通道（差异大的区域 = 脏污）
4. 输出 RGBA 图像（RGB=脏污颜色，A=透明度）

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


def parse_synthetic_filename(filename: str) -> dict:
    """
    解析合成脏污图像的文件名

    输入格式: {base_clean_name}_{frame_id}_{view}_s{sample}_640x480.png
    例如: 1751703857935_1751704212892_01_merged_seg1_final_level_max5_20250705162827_20250705162835_f_t0000000000_f000000000_2521_FV_s000_640x480.png

    返回: {
        'clean_filename': '{base_clean_name}.jpg',
        'frame_id': '2521',
        'view': 'FV',
        'sample': '000'
    }
    """
    # 移除 .png 或 _640x480.png 后缀
    base = filename.replace('_640x480.png', '').replace('.png', '')

    # 模式: {clean_name}_{frame_id}_{view}_s{sample}
    # 其中 frame_id 是数字，view 是 FV/MVL/MVR/RV 等，s{sample} 是 s000
    parts = base.rsplit('_', 3)  # 从右边分割3次

    if len(parts) < 4:
        return None

    # parts[-1] = s000 (sample)
    # parts[-2] = FV (view)
    # parts[-3] = 2521 (frame_id)
    # parts[:-3] = clean_name

    sample_str = parts[-1]
    view = parts[-2]
    frame_id = parts[-3]
    clean_name = '_'.join(parts[:-3])

    # 验证格式
    if not sample_str.startswith('s'):
        return None

    return {
        'clean_filename': clean_name + '.jpg',
        'frame_id': frame_id,
        'view': view,
        'sample': sample_str[1:],  # 去掉 's'
    }


def find_clean_image(clean_filename: str, clean_base_dir: str) -> str:
    """
    查找干净帧图像

    优先顺序:
    1. clean_base_dir/{view}/{clean_filename}
    2. clean_base_dir/{view_lower}/{clean_filename}

    Returns:
        完整路径，找不到返回 None
    """
    # 尝试从 manifest 获取的原始路径
    clean_path = os.path.join(clean_base_dir, clean_filename)
    if os.path.exists(clean_path):
        return clean_path

    # 尝试根据 view 分目录
    # clean_filename 格式: ..._{view}_t{time}_f{frame}.jpg
    # 需要提取 view

    # 使用正则提取 view (lf, f, rf, lr, r, rr, FV, MVL, MVR, RV 等)
    match = re.search(r'_([lfFrR][vV]|[lLrR][fF])_t\d+_f\d+\.jpg$', clean_filename)
    if not match:
        # 尝试其他模式
        match = re.search(r'_([a-z]{1,2})_t\d+.*\.jpg$', clean_filename)

    if match:
        view = match.group(1).lower()
        # 视角映射: FV->f, RV->r, MVL->lf, MVR->lr
        view_map = {
            'fv': 'f',
            'rv': 'r',
            'mvl': 'lf',
            'mvr': 'lr',
        }
        view_dir = view_map.get(view, view)

        path_by_view = os.path.join(clean_base_dir, view_dir, clean_filename)
        if os.path.exists(path_by_view):
            return path_by_view

    return None


def compute_alpha_channel(synthetic_bgr: np.ndarray,
                          clean_bgr: np.ndarray,
                          diff_threshold: float = 15.0) -> np.ndarray:
    """
    通过对比合成图和干净图，计算 alpha 通道

    Args:
        synthetic_bgr: 合成脏污图 (H, W, 3), BGR, uint8
        clean_bgr: 干净背景图 (H, W, 3), BGR, uint8
        diff_threshold: 差异阈值，小于此值视为无脏污

    Returns:
        alpha: Alpha 通道 (H, W), uint8 [0, 255]
    """
    # 确保尺寸一致
    if synthetic_bgr.shape != clean_bgr.shape:
        clean_bgr = cv2.resize(clean_bgr,
                               (synthetic_bgr.shape[1], synthetic_bgr.shape[0]),
                               interpolation=cv2.INTER_AREA)

    # 计算每像素 L1 距离
    diff = np.abs(synthetic_bgr.astype(np.float32) - clean_bgr.astype(np.float32))
    diff_l1 = diff.mean(axis=2)  # (H, W)

    # 转换为 alpha: 差异大 = 不透明
    # 使用 sigmoid 平滑过渡
    # alpha = 255 * sigmoid((diff - threshold) / scale)
    scale = 10.0
    alpha = 255.0 / (1.0 + np.exp(-(diff_l1 - diff_threshold) / scale))
    alpha = alpha.astype(np.uint8)

    return alpha


def convert_single_image(synthetic_path: str,
                        clean_base_dir: str,
                        output_path: str,
                        diff_threshold: float = 15.0,
                        blur_alpha: bool = True) -> bool:
    """
    转换单个合成图像为 RGBA 叠加层

    Args:
        synthetic_path: 合成图像路径
        clean_base_dir: 干净帧基准目录
        output_path: 输出 RGBA 路径
        diff_threshold: 差异阈值
        blur_alpha: 是否模糊 alpha 边缘

    Returns:
        是否成功
    """
    filename = os.path.basename(synthetic_path)

    # 1. 解析文件名
    info = parse_synthetic_filename(filename)
    if info is None:
        return False

    # 2. 查找干净帧
    clean_path = find_clean_image(info['clean_filename'], clean_base_dir)
    if clean_path is None:
        return False

    # 3. 读取图像
    synthetic_bgr = cv2.imread(synthetic_path, cv2.IMREAD_COLOR)
    clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)

    if synthetic_bgr is None or clean_bgr is None:
        return False

    # 4. 计算 alpha
    alpha = compute_alpha_channel(synthetic_bgr, clean_bgr, diff_threshold)

    # 5. 可选: 模糊 alpha 边缘
    if blur_alpha:
        from scipy.ndimage import gaussian_filter
        alpha = gaussian_filter(alpha.astype(float), sigma=1.0).astype(np.uint8)

    # 6. 转换为 RGBA
    synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([synthetic_rgb, alpha])  # (H, W, 4)

    # 7. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, rgba)

    return True


class DirtLayerConverter:
    """批量转换脏污层"""

    def __init__(self,
                 dirt_manifest_path: str,
                 dirt_source_dir: str,
                 clean_base_dir: str,
                 output_dir: str,
                 diff_threshold: float = 15.0,
                 blur_alpha: bool = True):
        """
        Args:
            dirt_manifest_path: 脏污层清单文件
            dirt_source_dir: 脏污层源图像目录（RGB 合成图）
            clean_base_dir: 干净帧基准目录
            output_dir: RGBA 输出目录
            diff_threshold: 差异阈值
            blur_alpha: 是否模糊 alpha
        """
        self.dirt_manifest_path = dirt_manifest_path
        self.dirt_source_dir = dirt_source_dir
        self.clean_base_dir = clean_base_dir
        self.output_dir = output_dir
        self.diff_threshold = diff_threshold
        self.blur_alpha = blur_alpha

        # 加载清单
        self.df = pd.read_csv(dirt_manifest_path, encoding='utf-8-sig')

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 统计
        self.stats = {
            'total': len(self.df),
            'success': 0,
            'failed_parse': 0,
            'clean_not_found': 0,
            'read_error': 0,
        }

    def convert_all(self):
        """批量转换"""
        print(f"开始转换 {len(self.df)} 个脏污层...")
        print(f"干净帧目录: {self.clean_base_dir}")
        print(f"差异阈值: {self.diff_threshold}")
        print()

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="转换进度"):
            filename = row['filename']

            # 源路径（可能有 _640x480 后缀）
            source_path = os.path.join(self.dirt_source_dir, filename)
            if not os.path.exists(source_path):
                # 尝试添加后缀
                if filename.endswith('.png'):
                    base = filename[:-4]
                    source_path = os.path.join(self.dirt_source_dir, base + '_640x480.png')

            if not os.path.exists(source_path):
                self.stats['failed_parse'] += 1
                continue

            # 输出路径（保持文件名）
            output_path = os.path.join(self.output_dir, filename)

            # 转换
            success = convert_single_image(
                source_path,
                self.clean_base_dir,
                output_path,
                self.diff_threshold,
                self.blur_alpha
            )

            if success:
                self.stats['success'] += 1
            else:
                self.stats['failed_parse'] += 1

        self._print_summary()

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("转换完成")
        print("="*60)
        print(f"总数: {self.stats['total']}")
        print(f"成功: {self.stats['success']} ({self.stats['success']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"失败: {self.stats['failed_parse']}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="将 SD RGB 脏污图转换为 RGBA 透明叠加层",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
转换原理:
  SD inpainting 生成的是完整 RGB 图像（背景+脏污），不是透明叠加。
  通过与原始干净帧对比，计算差异来生成 alpha 通道。

  Alpha 生成:
    alpha = 255 * sigmoid((|synthetic - clean| - threshold) / scale)

  输出格式:
    RGBA 图像，其中:
    - RGB = 脏污颜色（来自合成图）
    - A = 透明度（根据差异计算）

使用建议:
  1. 先在小批量测试参数 --diff_threshold
  2. 检查输出的 alpha 通道质量
  3. 调整参数后批量处理全部数据
        """
    )

    ap.add_argument("--dirt_manifest", type=str,
                    default="dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv",
                    help="脏污层清单文件")
    ap.add_argument("--dirt_source_dir", type=str,
                    default="sd_scripts/lora/accepted_20260224_164715/resized_640x480",
                    help="脏污层源图像目录（RGB 合成图）")
    ap.add_argument("--clean_base_dir", type=str,
                    default="dataset/my_clean_frames_4by3",
                    help="干净帧基准目录")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/dirt_layers_rgba",
                    help="RGBA 输出目录")

    ap.add_argument("--diff_threshold", type=float, default=15.0,
                    help="差异阈值，越小越敏感 (0-255)")
    ap.add_argument("--no_blur_alpha", action="store_true",
                    help="不模糊 alpha 边缘")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="最大处理样本数（测试用）")

    args = ap.parse_args()

    converter = DirtLayerConverter(
        dirt_manifest_path=args.dirt_manifest,
        dirt_source_dir=args.dirt_source_dir,
        clean_base_dir=args.clean_base_dir,
        output_dir=args.output_dir,
        diff_threshold=args.diff_threshold,
        blur_alpha=not args.no_blur_alpha,
    )

    # 可选：限制样本数
    if args.max_samples:
        converter.df = converter.df.head(args.max_samples)
        print(f"测试模式: 仅处理 {args.max_samples} 个样本")

    converter.convert_all()


if __name__ == "__main__":
    main()
