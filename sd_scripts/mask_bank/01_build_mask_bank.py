#!/usr/bin/env python3
"""
阶段1：构建 MaskBank

从 WoodScape Train/Val 提取 mask 形态库，包括：
1. 预处理到 640×480
2. 提取统计信息（覆盖率、类别比例、形态参数）
3. 生成 manifest 和采样接口
4. 严格数据隔离：仅使用 WoodScape Train 和 Val，不使用 Test
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2
from collections import defaultdict


# ============================================================================
# 配置
# ============================================================================

PROJECT_ROOT = Path("/home/yf/soiling_project")
DATASET_ROOT = PROJECT_ROOT / "dataset"

# 数据路径
WOODSCAPE_RAW = DATASET_ROOT / "woodscape_raw"
WOODSCAPE_PROCESSED = DATASET_ROOT / "woodscape_processed"
LABELS_INDEX = WOODSCAPE_PROCESSED / "meta" / "labels_index_ablation.csv"
LABELS_TILE_DIR = WOODSCAPE_PROCESSED / "labels_tile"

# MaskBank 输出路径
MASK_BANK_ROOT = DATASET_ROOT / "mask_bank"

# 目标分辨率
TARGET_RESOLUTION = (640, 480)

# WoodScape 原始分辨率
WS_WIDTH, WS_HEIGHT = 1920, 1080


# ============================================================================
# 统计信息提取
# ============================================================================

def extract_mask_statistics(mask: np.ndarray) -> Dict:
    """
    提取 mask 统计信息

    Args:
        mask: [H, W] 4类mask (0=clean, 1=trans, 2=semi, 3=opaque)

    Returns:
        统计信息字典
    """
    H, W = mask.shape
    total_pixels = H * W

    # 类别占比
    class_counts = np.bincount(mask.flatten(), minlength=4)
    class_ratios = class_counts / total_pixels

    # clean占比
    clean_ratio = class_ratios[0]

    # 覆盖率（有脏污区域占比）
    coverage = 1.0 - clean_ratio

    # 各类别覆盖率
    trans_ratio = class_ratios[1]
    semi_ratio = class_ratios[2]
    opaque_ratio = class_ratios[3]

    # 主导类别（跳过clean）
    if class_counts[1:].sum() == 0:
        dominant_class = 0  # 全是clean
    else:
        dominant_class = np.argmax(class_counts[1:]) + 1

    # 连通区域分析
    mask_binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    num_regions = num_labels - 1  # 减去背景
    if num_regions > 0:
        # 最大区域占比
        max_region_size = stats[1:, cv2.CC_STAT_AREA].max() / total_pixels
    else:
        max_region_size = 0.0

    # 边界统计（脏污是否接近图像边界）
    border_margin = 20
    border_mask = np.zeros_like(mask)
    border_mask[:border_margin, :] = 1
    border_mask[-border_margin:, :] = 1
    border_mask[:, :border_margin] = 1
    border_mask[:, -border_margin:] = 1

    border_dirty_ratio = (mask[border_mask == 1] > 0).sum() / (border_mask == 1).sum()

    # 中心区域脏污比例
    center_h, center_w = H // 4, W // 4
    center_mask = np.zeros_like(mask)
    center_mask[center_h:center_h*3, center_w:center_w*3] = 1
    center_dirty_ratio = (mask[center_mask == 1] > 0).sum() / (center_mask == 1).sum()

    return {
        'clean_ratio': float(clean_ratio),
        'trans_ratio': float(trans_ratio),
        'semi_ratio': float(semi_ratio),
        'opaque_ratio': float(opaque_ratio),
        'coverage': float(coverage),
        'dominant_class': int(dominant_class),
        'num_regions': int(num_regions),
        'max_region_size': float(max_region_size),
        'border_dirty_ratio': float(border_dirty_ratio),
        'center_dirty_ratio': float(center_dirty_ratio),
        'class_counts': class_counts.tolist(),
    }


def get_tile_coverage_from_npz(npz_path: Path) -> np.ndarray:
    """
    从现有 .npz 文件读取 tile coverage

    Args:
        npz_path: .npz 文件路径

    Returns:
        [8, 8, 4] tile coverage
    """
    data = np.load(npz_path)
    return data['tile_cov']


def compute_severity_from_tile_cov(
    tile_cov: np.ndarray,
    class_weights: Tuple = (0.0, 0.15, 0.50, 1.0),
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
    eta_trans: float = 0.9,
    spatial_mode: str = "gaussian",
    spatial_sigma: float = 0.5,
) -> Dict:
    """
    从 tile coverage 计算 Severity Score

    Args:
        tile_cov: [8, 8, 4] tile coverage
        class_weights: (w_clean, w_trans, w_semi, w_opaque)
        alpha, beta, gamma: 融合系数
        eta_trans: transparent折扣因子
        spatial_mode: spatial weight mode
        spatial_sigma: spatial weight sigma

    Returns:
        Severity分数字典
    """
    H, W = 8, 8
    w = np.array(class_weights, dtype=np.float32)

    # S_op: Opacity-aware coverage
    p = tile_cov.mean(axis=(0, 1))  # [4]
    S_op = float((w * p).sum())

    # S_sp: Spatial weighted
    if spatial_mode == "gaussian":
        # 生成高斯权重（中心高，边缘低）
        y = np.arange(H)
        x = np.arange(W)
        xx, yy = np.meshgrid(x, y)
        cx, cy = W / 2 - 0.5, H / 2 - 0.5
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * spatial_sigma * H)**2)
        Wmap = gaussian / gaussian.sum()
    else:
        Wmap = np.ones((H, W)) / (H * W)

    # 加权tile覆盖率
    tile_weights = tile_cov @ w  # [8, 8]
    S_sp = float((Wmap * tile_weights).sum())

    # S_dom: Dominance
    # 找出最严重的tile
    tile_severity = tile_cov @ w
    S_dom = float(tile_severity.max())

    # S_full: 融合
    S_full = alpha * S_op + beta * S_sp + gamma * S_dom

    return {
        'S_op': S_op,
        'S_sp': S_sp,
        'S_dom': S_dom,
        'S_full': float(np.clip(S_full, 0.0, 1.0)),
    }


# ============================================================================
# Mask 提取与预处理
# ============================================================================

def get_mask_from_gt_label(gt_label_path: Path) -> np.ndarray:
    """
    从 WoodScape gt label 文件读取 mask

    WoodScape 原始标签值:
        0: clean
        1: transparent
        2: semi-transparent
        3: opaque

    直接使用 4 类标签: 0, 1, 2, 3
    """
    mask = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 mask: {gt_label_path}")

    # WoodScape 标签已经是 0-3 的 4 类，直接转换为 int32
    mask_4class = mask.astype(np.int32)

    return mask_4class


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整 mask 到目标分辨率

    使用最近邻插值保持类别标签
    """
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


# ============================================================================
# MaskBank 构建
# ============================================================================

def build_mask_bank_for_split(
    split_name: str,
    labels_df: pd.DataFrame,
    output_dir: Path,
    target_resolution: Tuple[int, int] = (640, 480),
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    为特定 split 构建 MaskBank

    Args:
        split_name: 'train' 或 'val'
        labels_df: 标签索引 DataFrame
        output_dir: 输出目录
        target_resolution: 目标分辨率

    Returns:
        (manifest列表, 统计信息DataFrame)
    """
    print(f"\n{'='*60}")
    print(f"构建 MaskBank_{split_name.upper()}")
    print(f"{'='*60}")

    # 筛选当前 split
    split_df = labels_df[labels_df['split'] == split_name].copy()
    n_samples = len(split_df)
    print(f"样本数: {n_samples}")

    # 创建输出目录
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    statistics = []

    for idx, row in tqdm(split_df.iterrows(), total=n_samples, desc=f"Processing {split_name}"):
        # 获取文件名
        rgb_path = Path(row['rgb_path'])
        file_id = rgb_path.stem  # 如 "0001_FV"

        # 读取现有 tile coverage
        npz_path = Path(row['npz_path'])
        if not npz_path.exists():
            print(f"警告: NPZ 文件不存在: {npz_path}")
            continue

        tile_cov = get_tile_coverage_from_npz(npz_path)

        # 尝试从原始 GT label 读取 mask
        # WoodScape raw 结构: 只有 train 和 test 目录
        # val 是从 train 中划分出来的，所以 val 样本需要从 train 目录读取
        if split_name == 'val':
            gt_label_path = WOODSCAPE_RAW / 'train' / "gtLabels" / f"{file_id}.png"
        else:
            gt_label_path = WOODSCAPE_RAW / split_name / "gtLabels" / f"{file_id}.png"

        if gt_label_path.exists():
            # 从原始 GT 读取
            mask = get_mask_from_gt_label(gt_label_path)
            # 调整分辨率
            mask_resized = resize_mask(mask, target_resolution)
            source_type = "raw_gt"
        else:
            # 如果原始 GT 不存在，从 tile coverage 重建
            # 将 8x8 tile 上采样到目标分辨率
            # 每个 tile 是 80x60 像素 (640/8=80, 480/8=60)
            mask_upsampled = np.zeros(target_resolution, dtype=np.int32)
            for i in range(8):
                for j in range(8):
                    # 获取该 tile 的主导类别
                    tile_class = np.argmax(tile_cov[i, j])
                    y0, y1 = i * 60, (i + 1) * 60
                    x0, x1 = j * 80, (j + 1) * 80
                    mask_upsampled[y0:y1, x0:x1] = tile_class

            mask_resized = mask_upsampled
            source_type = "reconstructed_from_tile"

        # 保存预处理后的 mask
        output_path = processed_dir / f"{file_id}_mask.png"
        cv2.imwrite(str(output_path), mask_resized.astype(np.uint8))

        # 提取统计信息
        stats = extract_mask_statistics(mask_resized)
        stats['file_id'] = file_id
        stats['original_path'] = str(gt_label_path)
        stats['processed_path'] = str(output_path)
        stats['source_type'] = source_type
        stats['npz_path'] = str(npz_path)
        statistics.append(stats)

        # 计算 Severity Score
        severity = compute_severity_from_tile_cov(tile_cov)

        manifest.append({
            'file_id': file_id,
            'original_path': str(gt_label_path),
            'processed_path': str(output_path),
            'npz_path': str(npz_path),
            'source_type': source_type,
            'split': split_name,
            # 统计信息
            'coverage': stats['coverage'],
            'clean_ratio': stats['clean_ratio'],
            'trans_ratio': stats['trans_ratio'],
            'semi_ratio': stats['semi_ratio'],
            'opaque_ratio': stats['opaque_ratio'],
            'dominant_class': stats['dominant_class'],
            'num_regions': stats['num_regions'],
            # Severity Score
            'S_op': severity['S_op'],
            'S_sp': severity['S_sp'],
            'S_dom': severity['S_dom'],
            'S_full': severity['S_full'],
        })

    # 保存 manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_csv = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\n保存 manifest: {manifest_csv}")
    print(f"  总计: {len(manifest_df)} 样本")

    # 保存统计信息
    stats_df = pd.DataFrame(statistics)
    stats_csv = output_dir / "statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"保存统计信息: {stats_csv}")

    # 打印统计摘要
    print(f"\n统计摘要:")
    print(f"  覆盖率分布:")
    print(f"    min: {manifest_df['coverage'].min():.4f}")
    print(f"    max: {manifest_df['coverage'].max():.4f}")
    print(f"    mean: {manifest_df['coverage'].mean():.4f}")
    print(f"    median: {manifest_df['coverage'].median():.4f}")
    print(f"\n  主导类别分布:")
    print(f"    {manifest_df['dominant_class'].value_counts().to_dict()}")

    return manifest, stats_df


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*60)
    print("MaskBank 构建 - 阶段1")
    print("="*60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据根目录: {DATASET_ROOT}")

    # 加载标签索引
    if not LABELS_INDEX.exists():
        raise FileNotFoundError(f"标签索引不存在: {LABELS_INDEX}")

    labels_df = pd.read_csv(LABELS_INDEX)
    print(f"\n加载标签索引: {len(labels_df)} 样本")
    print(f"  划分: {labels_df['split'].value_counts().to_dict()}")

    # 构建 MaskBank_Train
    train_manifest, train_stats = build_mask_bank_for_split(
        split_name='train',
        labels_df=labels_df,
        output_dir=MASK_BANK_ROOT / "train",
        target_resolution=TARGET_RESOLUTION,
    )

    # 构建 MaskBank_Val
    val_manifest, val_stats = build_mask_bank_for_split(
        split_name='val',
        labels_df=labels_df,
        output_dir=MASK_BANK_ROOT / "val",
        target_resolution=TARGET_RESOLUTION,
    )

    # 生成总体报告
    print("\n" + "="*60)
    print("MaskBank 构建完成")
    print("="*60)

    print(f"\n目录结构:")
    print(f"  {MASK_BANK_ROOT}/")
    print(f"    ├── train/")
    print(f"    │   ├── processed/  ({len(train_manifest)} masks)")
    print(f"    │   ├── manifest.csv")
    print(f"    │   └── statistics.csv")
    print(f"    └── val/")
    print(f"        ├── processed/  ({len(val_manifest)} masks)")
    print(f"        ├── manifest.csv")
    print(f"        └── statistics.csv")

    print(f"\n数据隔离检查:")
    # 从 rgb_path 提取 file_id 进行检查
    test_df = labels_df[labels_df['split'] == 'test'].copy()
    test_df['file_id'] = test_df['rgb_path'].apply(lambda x: Path(x).stem)

    mask_file_ids = set([m['file_id'] for m in train_manifest + val_manifest])
    test_used = test_df['file_id'].isin(mask_file_ids).sum()

    if test_used > 0:
        print(f"  ✗ 警告: 发现 {test_used} 个 test 样本被误用!")
    else:
        print(f"  ✓ Test 数据未被使用")

    print(f"\n下一步:")
    print(f"  python scripts/02_generate_synthetic.py")

    print("\n✓ MaskBank 构建完成")


if __name__ == "__main__":
    main()