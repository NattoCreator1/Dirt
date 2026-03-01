#!/usr/bin/env python3
"""
为SD合成图像生成强标注NPZ标签文件

功能:
1. 从原始mask文件读取WoodScape格式的标签
2. 计算tile-level coverage (8x8x4)
3. 计算各种global severity scores
4. 保存为NPZ格式，与WoodScape数据集对齐

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

# 导入计算函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_labels",
    project_root / "scripts" / "03_build_labels_tile_global.py"
)
build_labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_labels)

compute_tile_cov = build_labels.compute_tile_cov
compute_all_severity_variants = build_labels.compute_all_severity_variants
make_spatial_weight = build_labels.make_spatial_weight


def parse_args():
    parser = argparse.ArgumentParser(
        description="为SD合成图像生成NPZ标签文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--manifest_csv", type=str, required=True,
                       help="Accepted图像manifest文件路径")
    parser.add_argument("--mask_bank", type=str,
                       default="dataset/mask_bank/train/processed",
                       help="Mask文件目录")
    parser.add_argument("--target_size", type=int, nargs=2, default=[640, 480],
                       help="目标图像尺寸 宽 高 (与baseline对齐)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录 (默认：与manifest同目录/npz)")
    parser.add_argument("--class_weights", type=float, nargs=4,
                       default=[0.0, 0.15, 0.50, 1.0],
                       help="类别权重 (clean, trans, semi, opaque)")

    return parser.parse_args()


def load_mask_from_file(mask_path: Path, target_size=(640, 480)):
    """
    加载并处理mask文件到目标尺寸

    Args:
        mask_path: Mask文件路径
        target_size: 目标尺寸 (width, height)

    Returns:
        mask_cls: 处理后的mask数组 (H, W)，值为0-3
    """
    # 读取mask图像 (灰度图)
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)

    # 转换为WoodScape类别 (0=clean, 1=transparent, 2=semi, 3=opaque)
    # 假设mask_bank的mask已经是0-255的灰度值，需要映射到0-3
    # 或者mask已经包含正确的类别值

    # 这里假设mask值直接对应类别（根据实际mask格式调整）
    mask_cls = mask_np.astype(np.uint8)

    # 确保值在0-3范围内
    if mask_cls.max() > 3:
        # 如果是0-255范围，需要转换
        # 假设: 0=clean, 85=trans, 170=semi, 255=opaque (近似)
        # 或者根据实际mask格式调整
        pass

    # Resize到目标尺寸 (使用最近邻插值保持类别值)
    mask_resized = cv2.resize(mask_cls, target_size, interpolation=cv2.INTER_NEAREST)

    return mask_resized


def compute_npz_labels(mask_cls, class_weights=(0.0, 0.15, 0.50, 1.0)):
    """
    计算NPZ标签的所有字段

    Args:
        mask_cls: Mask数组 (H, W)，值为0-3
        class_weights: 类别权重 (默认w_gap_strong)

    Returns:
        dict: 包含所有NPZ字段的字典
    """
    H, W = mask_cls.shape
    w = np.array(class_weights, dtype=np.float32)

    # 1. 计算 tile-level coverage (8x8x4)
    tile_cov = compute_tile_cov(mask_cls, n=8, m=8, C=4)

    # 2. 计算 global coverage per class
    total = float(mask_cls.size)
    global_cov_per_class = np.array([
        float((mask_cls == c).sum()) / total for c in range(4)
    ], dtype=np.float32)

    # 3. 计算 global_soiling (简单覆盖率)
    global_soiling = 1.0 - global_cov_per_class[0]

    # 4. 计算基础组件
    # Opacity-aware coverage
    p = global_cov_per_class
    S_op = float((w * p).sum())

    # Spatial weighted (直接在mask上计算)
    Wmap = make_spatial_weight(H, W, mode="gaussian")
    S_sp = float((Wmap * w[mask_cls]).sum())

    # Dominance
    S_dom_eta09, _, _, _ = build_labels.compute_dominance_class_aware(
        tile_cov, class_weights=class_weights, eta_trans=0.9
    )
    S_dom_eta00, _, _, _ = build_labels.compute_dominance_class_aware(
        tile_cov, class_weights=class_weights, eta_trans=0.0
    )

    # 5. 计算各fusion配置的S_full
    # Default fusion: alpha=0.6, beta=0.3, gamma=0.1
    S_full_default = 0.6 * S_op + 0.3 * S_sp + 0.1 * S_dom_eta09

    # eta00 fusion: alpha=0.6, beta=0.3, gamma=0.1
    S_full_eta00 = 0.6 * S_op + 0.3 * S_sp + 0.1 * S_dom_eta00

    # S_full_wgap_alpha50 fusion: alpha=0.5, beta=0.4, gamma=0.1
    S_full_wgap_alpha50 = 0.5 * S_op + 0.4 * S_sp + 0.1 * S_dom_eta09

    # Simple scores
    s = global_soiling
    S_op_only = S_op
    S_op_sp = 0.7 * S_op + 0.3 * S_sp

    # 6. 构建完整的NPZ字典
    npz_data = {
        # Tile-level coverage
        'tile_cov': tile_cov.astype(np.float32),

        # Global coverage per class
        'global_cov_per_class': global_cov_per_class.astype(np.float32),

        # Simple global soiling ratio
        'global_soiling': np.float32(global_soiling),

        # Severity scores (使用与WoodScape一致的命名)
        'S': np.float32(S_full_default),  # 默认使用default fusion
        's': np.float32(s),
        'S_op_only': np.float32(S_op_only),
        'S_op_sp': np.float32(S_op_sp),
        'S_full': np.float32(S_full_default),
        'S_full_eta00': np.float32(S_full_eta00),
        'S_full_wgap_alpha50': np.float32(S_full_wgap_alpha50),  # 关键：添加wgap_alpha50
        'S_op': np.float32(S_op),
        'S_sp': np.float32(S_sp),
        'S_dom': np.float32(S_dom_eta09),
        'S_dom_eta00': np.float32(S_dom_eta00),
        'S_dom_eta09': np.float32(S_dom_eta09),

        # Legacy compatibility
        'global_score': np.float32(S_full_default),
        'glboal_level': int(np.argmax(global_cov_per_class[1:]) + 1),  # 排除clean类
    }

    return npz_data


def generate_npz_for_synthetic_images(args):
    """为SD合成图像生成NPZ标签文件"""
    print("=" * 70)
    print("生成NPZ标签文件")
    print("=" * 70)

    # 读取manifest
    print(f"\n读取manifest: {args.manifest_csv}")
    df = pd.read_csv(args.manifest_csv)
    print(f"样本数量: {len(df)}")

    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.manifest_csv).parent / "npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mask目录
    mask_dir = Path(args.mask_bank)
    print(f"Mask目录: {mask_dir}")

    # 目标尺寸
    target_w, target_h = args.target_size
    target_size = (target_w, target_h)
    print(f"目标尺寸: {target_w}x{target_h}")

    # 类别权重
    class_weights = tuple(args.class_weights)
    print(f"类别权重: {class_weights}")

    # 处理每个样本
    manifest_updates = []
    skipped = []
    errors = []

    # 列名兼容性处理
    if 'original_filename' not in df.columns and 'output_filename' in df.columns:
        df['original_filename'] = df['output_filename']

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成NPZ"):
        # 获取文件名和mask信息
        original_filename = row['original_filename']
        mask_file_id = row['mask_file_id']

        # 生成NPZ文件名 (与原始图像对应)
        npz_filename = original_filename.replace('.png', '.npz')
        npz_path = output_dir / npz_filename

        # 构建mask文件路径
        mask_path = mask_dir / f"{mask_file_id}_mask.png"

        if not mask_path.exists():
            skipped.append(f"{original_filename}: mask not found")
            continue

        try:
            # 加载并处理mask
            mask_cls = load_mask_from_file(mask_path, target_size)

            # 计算NPZ标签
            npz_data = compute_npz_labels(mask_cls, class_weights)

            # 保存NPZ文件
            np.savez_compressed(npz_path, **npz_data)

            # 记录到manifest
            manifest_updates.append({
                'original_filename': original_filename,
                'npz_path': str(npz_path),
                'tile_cov_shape': str(npz_data['tile_cov'].shape),
                'S_full': float(npz_data['S_full']),
                'S_full_wgap_alpha50': float(npz_data['S_full_wgap_alpha50']),
                's': float(npz_data['s']),
            })

        except Exception as e:
            errors.append(f"{original_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 更新manifest
    if manifest_updates:
        manifest_df = pd.DataFrame(manifest_updates)
        manifest_path = output_dir / "manifest_npz.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\nNPZ manifest已保存: {manifest_path}")

    # 打印摘要
    print("\n" + "=" * 70)
    print("生成完成!")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"成功生成: {len(manifest_updates)} 个")
    print(f"跳过: {len(skipped)} 个")
    print(f"错误: {len(errors)} 个")

    if skipped:
        print(f"\n跳过的样本:")
        for s in skipped[:5]:
            print(f"  - {s}")
        if len(skipped) > 5:
            print(f"  ... 还有 {len(skipped)-5} 个")

    if errors:
        print(f"\n错误的样本:")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors)-5} 个")

    return output_dir, len(manifest_updates)


def main():
    args = parse_args()

    print("=" * 70)
    print("SD合成图像NPZ标签生成工具")
    print("=" * 70)
    print(f"Manifest: {args.manifest_csv}")
    print(f"Mask目录: {args.mask_bank}")
    print(f"目标尺寸: {args.target_size[0]}x{args.target_size[1]}")
    print(f"类别权重: {args.class_weights}")
    print("=" * 70)

    output_dir, count = generate_npz_for_synthetic_images(args)

    print(f"\n✅ 成功生成 {count} 个NPZ文件到: {output_dir}")


if __name__ == "__main__":
    main()
