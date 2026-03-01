#!/usr/bin/env python3
"""
LoRA Training Evaluation Metrics (v1.2)

评估指标面板：Spill Rate、Mask 区域真实性、可控性等

设计目标：评估"只在约束区域生成更真实的脏污层，同时尽量不污染背景"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from sklearn.metrics import pairwise_distances


def compute_spill_rate(
    synthetic_image: np.ndarray,
    clean_image: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    计算 Spill Rate（背景改写比例）

    定义：mask 外的像素变化占总变化的比例

    Args:
        synthetic_image: [H, W, 3] 生成图像
        clean_image: [H, W, 3] 干净底图
        mask: [H, W] mask (1=dirty区域, 0=clean区域)

    Returns:
        spill_rate: 0.0-1.0 之间，越低越好
    """
    # 计算全图差异
    diff_full = np.abs(synthetic_image.astype(float) - clean_image.astype(float))
    diff_full_norm = np.linalg.norm(diff_full, axis=2)  # [H, W]

    total_change = diff_full_norm.sum()

    if total_change == 0:
        return 0.0

    # 计算 mask 外的变化
    mask_bool = (mask > 0)
    diff_outside = diff_full_norm.copy()
    diff_outside[mask_bool] = 0

    outside_change = diff_outside.sum()

    spill_rate = outside_change / total_change
    return float(spill_rate)


def compute_background_metrics(
    synthetic_image: np.ndarray,
    clean_image: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    计算背景区域的质量指标

    Args:
        synthetic_image: [H, W, 3] 生成图像
        clean_image: [H, W, 3] 干净底图
        mask: [H, W] mask

    Returns:
        包含 PSNR、SSIM 等指标的字典
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

    mask_bool = (mask > 0)
    mask_complement = ~mask_bool

    # 只在 mask 外计算
    if mask_complement.sum() == 0:
        return {
            "psnr": float('inf'),
            "ssim": 1.0,
        }

    # 提取背景区域
    synth_bg = synthetic_image[mask_complement]
    clean_bg = clean_image[mask_complement]

    # PSNR
    try:
        psnr = peak_signal_noise_ratio(clean_bg, synth_bg, data_range=255)
    except:
        psnr = float('inf')

    # SSIM（需要对整图计算，然后只在背景区域比较）
    # 创建 mask 外的权重
    weight = np.zeros_like(mask, dtype=float)
    weight[mask_complement] = 1.0

    ssim_value = 0.0
    for c in range(3):
        ssim_c = ssim(
            synthetic_image[:, :, c],
            clean_image[:, :, c],
            data_range=255,
        )
        ssim_value += ssim_c
    ssim_value /= 3

    return {
        "psnr": float(psnr),
        "ssim": float(ssim_value),
    }


def compute_mask_region_statistics(
    image: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    计算掩码区域的统计特征

    Args:
        image: [H, W, 3] 图像
        mask: [H, W] mask

    Returns:
        包含颜色直方图、梯度能量等统计的字典
    """
    mask_bool = (mask > 0)

    if mask_bool.sum() == 0:
        return {
            "color_histogram": np.zeros((256, 3)),
            "gradient_energy": 0.0,
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
        }

    # 提取 mask 内像素
    masked_pixels = image[mask_bool]

    # 颜色直方图
    histograms = []
    for c in range(3):
        hist, _ = np.histogram(masked_pixels[:, c], bins=256, range=(0, 256))
        histograms.append(hist)
    color_histogram = np.stack(histograms, axis=1)  # [256, 3]

    # 梯度能量（纹理特征）
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_energy = gradient_magnitude[mask_bool].mean()

    # 亮度统计
    mean_brightness = masked_pixels.mean()
    std_brightness = masked_pixels.std()

    return {
        "color_histogram": color_histogram,
        "gradient_energy": float(gradient_energy),
        "mean_brightness": float(mean_brightness),
        "std_brightness": float(std_brightness),
    }


def compute_distribution_distance(
    stats1: Dict[str, np.ndarray],
    stats2: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    计算两组统计特征之间的距离

    Args:
        stats1, stats2: compute_mask_region_statistics 返回的统计字典

    Returns:
        各种距离指标
    """
    # 颜色直方图距离 (EMD 近似)
    hist1_flat = stats1["color_histogram"].reshape(-1)
    hist2_flat = stats2["color_histogram"].reshape(-1)

    # 归一化
    hist1_flat = hist1_flat / (hist1_flat.sum() + 1e-10)
    hist2_flat = hist2_flat / (hist2_flat.sum() + 1e-10)

    # L1 距离
    hist_l1 = np.abs(hist1_flat - hist2_flat).sum()

    # 梯度能量差异
    grad_diff = abs(stats1["gradient_energy"] - stats2["gradient_energy"])

    # 亮度差异
    brightness_diff = abs(stats1["mean_brightness"] - stats2["mean_brightness"])

    return {
        "color_histogram_l1": float(hist_l1),
        "gradient_energy_diff": float(grad_diff),
        "brightness_diff": float(brightness_diff),
    }


def evaluate_controllability_grid(
    generator,
    clean_image: np.ndarray,
    mask: np.ndarray,
    class_list: List[int] = [1, 2, 3],
    severity_prompts: List[str] = ["mild", "moderate", "severe"],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    生成可控性网格图

    固定 clean 图和 mask，变化 class 和 severity prompt

    Returns:
        包含网格图和元数据的字典
    """
    grid_rows = len(severity_prompts)
    grid_cols = len(class_list)

    grid_images = np.zeros((grid_rows, grid_cols, *clean_image.shape), dtype=np.uint8)
    grid_metadata = []

    for row_idx, severity in enumerate(severity_prompts):
        for col_idx, target_class in enumerate(class_list):
            # 构造 prompt
            class_name = {1: "transparent", 2: "semi-transparent", 3: "opaque"}[target_class]
            prompt = f"{severity} {class_name} soiling layer on camera lens"

            # 生成
            result = generator.generate(
                clean_image=clean_image,
                target_mask=mask,
                prompt=prompt,
                target_class=target_class,
                seed=seed + row_idx * grid_cols + col_idx,
            )

            if result is not None:
                grid_images[row_idx, col_idx] = result["synthetic_image"]

            grid_metadata.append({
                "row": row_idx,
                "col": col_idx,
                "severity": severity,
                "target_class": target_class,
                "prompt": prompt,
                "success": result is not None,
            })

    return {
        "grid_images": grid_images,
        "grid_metadata": grid_metadata,
        "class_list": class_list,
        "severity_prompts": severity_prompts,
    }


def save_controllability_grid(
    grid_data: Dict[str, any],
    output_path: Path,
    figsize: Tuple[int, int] = (15, 15),
):
    """
    保存可控性网格图
    """
    import matplotlib.pyplot as plt

    grid_images = grid_data["grid_images"]
    grid_metadata = grid_data["grid_metadata"]
    class_list = grid_data["class_list"]
    severity_prompts = grid_data["severity_prompts"]

    fig, axes = plt.subplots(len(severity_prompts), len(class_list), figsize=figsize)

    for row_idx, severity in enumerate(severity_prompts):
        for col_idx, target_class in enumerate(class_list):
            ax = axes[row_idx, col_idx]
            ax.imshow(grid_images[row_idx, col_idx])
            ax.set_title(f"{class_list[col_idx]}: {severity}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved controllability grid to: {output_path}")


def evaluate_dataset(
    synthetic_dir: Path,
    clean_dir: Path,
    output_dir: Path,
) -> Dict[str, any]:
    """
    评估整个数据集的质量指标

    Args:
        synthetic_dir: 合成图像目录
        clean_dir: 干净底图目录
        output_dir: 输出目录

    Returns:
        汇总的评估结果
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "spill_rates": [],
        "background_psnr": [],
        "background_ssim": [],
        "sample_count": 0,
    }

    # 遍历合成图像
    synthetic_files = list(synthetic_dir.glob("*.png")) + list(synthetic_dir.glob("*.jpg"))

    print(f"Found {len(synthetic_files)} synthetic images")

    for synth_path in synthetic_files:
        # 查找对应的干净底图和 mask
        sample_id = synth_path.stem
        clean_path = clean_dir / f"{sample_id}.png"
        mask_path = synthetic_dir.parent / "masks" / f"{sample_id}_mask.png"

        if not clean_path.exists() or not mask_path.exists():
            continue

        try:
            # 加载数据
            synth_img = cv2.imread(str(synth_path))
            synth_img = cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB)

            clean_img = cv2.imread(str(clean_path))
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # 计算指标
            spill_rate = compute_spill_rate(synth_img, clean_img, mask)
            bg_metrics = compute_background_metrics(synth_img, clean_img, mask)

            results["spill_rates"].append(spill_rate)
            results["background_psnr"].append(bg_metrics["psnr"])
            results["background_ssim"].append(bg_metrics["ssim"])
            results["sample_count"] += 1

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue

    # 汇总统计
    if results["sample_count"] > 0:
        summary = {
            "spill_rate": {
                "mean": np.mean(results["spill_rates"]),
                "median": np.median(results["spill_rates"]),
                "std": np.std(results["spill_rates"]),
                "max": np.max(results["spill_rates"]),
            },
            "background_psnr": {
                "mean": np.mean(results["background_psnr"]),
                "median": np.median(results["background_psnr"]),
            },
            "background_ssim": {
                "mean": np.mean(results["background_ssim"]),
                "median": np.median(results["background_ssim"]),
            },
            "sample_count": results["sample_count"],
        }

        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)
        print(f"Samples evaluated: {summary['sample_count']}")
        print(f"\nSpill Rate (lower is better, target < 0.05):")
        print(f"  Mean: {summary['spill_rate']['mean']:.4f}")
        print(f"  Median: {summary['spill_rate']['median']:.4f}")
        print(f"  Std: {summary['spill_rate']['std']:.4f}")
        print(f"  Max: {summary['spill_rate']['max']:.4f}")
        print(f"\nBackground Quality:")
        print(f"  PSNR: {summary['background_psnr']['mean']:.2f} dB")
        print(f"  SSIM: {summary['background_ssim']['mean']:.4f}")

        # 保存结果
        import json
        output_path = output_dir / "evaluation_summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\n✓ Saved summary to: {output_path}")

        return summary
    else:
        print("No samples evaluated!")
        return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LoRA training results"
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        required=True,
        help="Directory containing synthetic images"
    )
    parser.add_argument(
        "--clean-dir",
        type=str,
        required=True,
        help="Directory containing clean base images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    evaluate_dataset(
        synthetic_dir=Path(args.synthetic_dir),
        clean_dir=Path(args.clean_dir),
        output_dir=Path(args.output_dir),
    )
