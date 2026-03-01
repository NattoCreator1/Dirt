#!/usr/bin/env python3
"""
使用 Baseline 模型筛选 SD 合成数据

策略：
1. 用表现良好的 baseline 模型对所有 8960 张 SD 图像进行预测
2. 计算 S_hat (baseline预测) vs S_gt (NPZ标注) 的误差
3. 筛选标准：
   - 误差小于阈值的样本（baseline 和标注一致）
   - 或者根据误差分布选择高质量样本

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2


class ImageOnlySDataset:
    """仅加载图像的数据集（用于SD合成数据预测）"""
    def __init__(self, index_csv, img_root=".", resize_w=640, resize_h=480):
        self.df = pd.read_csv(index_csv)
        self.img_root = img_root
        self.resize_w = resize_w
        self.resize_h = resize_h

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Detect image path column
        for col in ["rgb_path", "image_path", "image", "img_path", "path", "filename"]:
            if col in self.df.columns:
                self.img_col = col
                break
        else:
            raise RuntimeError(f"Cannot find image path column in {index_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]

        # Handle relative paths
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()

        return {
            "image": x,
            "img_path": img_path,
            "original_filename": row.get("original_filename", ""),
        }


@torch.no_grad()
def predict_with_baseline(model, loader, device):
    """用 baseline 模型预测 SD 数据的严重程度"""
    model.eval()

    all_paths = []
    all_shat = []
    all_sagg = []

    for batch in tqdm(loader, desc="Predicting with baseline", ncols=100):
        x = batch["image"].to(device, non_blocking=True)
        paths = batch["img_path"]

        out = model(x)

        all_shat.append(out["S_hat"].detach().cpu().numpy())
        all_sagg.append(out["S_agg"].detach().cpu().numpy())
        all_paths.extend(paths)

    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)

    return {
        "image_path": all_paths,
        "S_hat_baseline": shat,
        "S_agg_baseline": sagg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="使用 Baseline 模型筛选 SD 合成数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--baseline_ckpt", required=True,
                       help="Baseline 模型 checkpoint 路径")
    parser.add_argument("--sd_index_csv", required=True,
                       help="SD 合成数据索引 CSV (包含所有 8960 张)")
    parser.add_argument("--npz_manifest", required=True,
                       help="NPZ manifest (包含 S_full_wgap_alpha50 标注)")
    parser.add_argument("--img_root", default=".",
                       help="图像根目录")
    parser.add_argument("--output_dir", default="sd_scripts/lora/baseline_filtered",
                       help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")

    # 筛选策略
    parser.add_argument("--strategy", choices=["error_threshold", "percentile", "bias_corrected"],
                       default="error_threshold",
                       help="筛选策略")
    parser.add_argument("--error_threshold", type=float, default=0.05,
                       help="误差阈值 (|S_hat - S_gt| < threshold)")
    parser.add_argument("--percentile", type=int, default=50,
                       help="保留误差最小的前 N%% 样本")
    parser.add_argument("--bias_correction", action="store_true",
                       help="是否使用偏置校正 (baseline bias = -0.002)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 baseline 模型
    print(f"\n加载 baseline 模型: {args.baseline_ckpt}")
    from baseline.models.baseline_dualhead import BaselineDualHead

    ckpt = torch.load(args.baseline_ckpt, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  模型 epoch: {ckpt.get('epoch', 'unknown')}")

    # 读取 SD 索引
    print(f"\n读取 SD 索引: {args.sd_index_csv}")
    sd_df = pd.read_csv(args.sd_index_csv)
    print(f"  SD 样本数: {len(sd_df)}")

    # 列名兼容性处理: output_filename -> original_filename
    if 'original_filename' not in sd_df.columns and 'output_filename' in sd_df.columns:
        sd_df['original_filename'] = sd_df['output_filename']

    # 读取 NPZ manifest
    print(f"\n读取 NPZ manifest: {args.npz_manifest}")
    npz_df = pd.read_csv(args.npz_manifest)
    print(f"  NPZ 样本数: {len(npz_df)}")

    # 创建临时索引文件（包含 image_path 列供预测使用）
    temp_index_path = "/tmp/temp_sd_index_for_filtering.csv"
    merged_df = pd.merge(
        sd_df,
        npz_df[["original_filename", "S_full_wgap_alpha50"]],
        on="original_filename",
        how="inner"
    )

    # 确定/构造 image_path 列
    img_col = None
    for col in ["rgb_path", "image_path", "image", "img_path"]:
        if col in merged_df.columns:
            img_col = col
            break

    if img_col is None:
        # 如果没有图像路径列，从 output_filename 构造
        if "output_filename" in merged_df.columns:
            # 使用 resized 图像目录
            img_dir = Path(args.sd_index_csv).parent / "resized_640x480"
            merged_df["rgb_path"] = merged_df["output_filename"].apply(
                lambda x: str(img_dir / x)
            )
            img_col = "rgb_path"
            print(f"  构造图像路径列: 使用 resized_640x480 目录")
        else:
            raise RuntimeError("Cannot find or construct image path column")
    else:
        # 重命名为 image_path 供数据集使用
        merged_df.rename(columns={img_col: "rgb_path"}, inplace=True)

    # 保存临时索引
    merged_df[["rgb_path", "original_filename", "S_full_wgap_alpha50"]].to_csv(temp_index_path, index=False)
    print(f"\n创建临时索引: {temp_index_path}")
    print(f"  合并后样本数: {len(merged_df)}")

    # 创建数据集
    dataset = ImageOnlySDataset(temp_index_path, args.img_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # 预测
    print("\n开始预测...")
    predictions = predict_with_baseline(model, loader, device)

    # 合并预测结果和标注
    results_df = pd.DataFrame({
        "image_path": predictions["image_path"],
        "S_gt": merged_df["S_full_wgap_alpha50"].values,
        "S_hat_baseline": predictions["S_hat_baseline"],
        "S_agg_baseline": predictions["S_agg_baseline"],
    })

    # 计算误差
    results_df["error"] = np.abs(results_df["S_hat_baseline"] - results_df["S_gt"])
    results_df["bias"] = results_df["S_hat_baseline"] - results_df["S_gt"]

    # 分析误差分布
    print("\n" + "=" * 70)
    print("误差分布分析")
    print("=" * 70)
    print(f"误差均值: {results_df['error'].mean():.4f}")
    print(f"误差中位数: {results_df['error'].median():.4f}")
    print(f"误差标准差: {results_df['error'].std():.4f}")
    print(f"误差范围: [{results_df['error'].min():.4f}, {results_df['error'].max():.4f}]")

    print(f"\n偏置分析:")
    print(f"偏置均值: {results_df['bias'].mean():.4f}")
    print(f"  (正值 = baseline 高估, 负值 = baseline 低估)")

    # 误差分位数
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  误差 {p}% 分位数: {results_df['error'].quantile(p/100):.4f}")

    # 根据策略筛选
    print("\n" + "=" * 70)
    print(f"筛选策略: {args.strategy}")
    print("=" * 70)

    if args.strategy == "error_threshold":
        # 误差阈值筛选
        threshold = args.error_threshold
        filtered_df = results_df[results_df["error"] < threshold].copy()
        print(f"误差阈值: {threshold}")
        print(f"筛选前: {len(results_df)} 张")
        print(f"筛选后: {len(filtered_df)} 张 ({len(filtered_df)/len(results_df)*100:.1f}%)")

    elif args.strategy == "percentile":
        # 百分位数筛选
        percentile = args.percentile
        threshold = results_df["error"].quantile(percentile / 100)
        filtered_df = results_df[results_df["error"] <= threshold].copy()
        print(f"百分位数: {percentile}%")
        print(f"误差阈值: {threshold}")
        print(f"筛选前: {len(results_df)} 张")
        print(f"筛选后: {len(filtered_df)} 张 ({len(filtered_df)/len(results_df)*100:.1f}%)")

    elif args.strategy == "bias_corrected":
        # 偏置校正筛选 (考虑 baseline 的 -0.002 偏置)
        BASELINE_BIAS = -0.002
        corrected_error = np.abs(results_df["bias"] - BASELINE_BIAS)
        threshold = args.error_threshold
        filtered_df = results_df[corrected_error < threshold].copy()
        print(f"Baseline 偏置: {BASELINE_BIAS:.4f}")
        print(f"校正后误差阈值: {threshold}")
        print(f"筛选前: {len(results_df)} 张")
        print(f"筛选后: {len(filtered_df)} 张 ({len(filtered_df)/len(results_df)*100:.1f}%)")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存筛选后的索引
    filtered_index_path = output_dir / f"filtered_index_{args.strategy}_{timestamp}.csv"

    # 合并原始信息
    filtered_with_info = pd.merge(
        filtered_df[["image_path", "S_gt"]],
        sd_df,
        left_on="image_path",
        right_on="rgb_path",
        how="left"
    )

    filtered_with_info.to_csv(filtered_index_path, index=False)
    print(f"\n筛选后索引已保存: {filtered_index_path}")

    # 保存完整预测结果
    all_results_path = output_dir / f"all_predictions_{timestamp}.csv"
    results_df.to_csv(all_results_path, index=False)
    print(f"所有预测结果已保存: {all_results_path}")

    # 保存筛选统计
    stats = {
        "strategy": args.strategy,
        "total_samples": len(results_df),
        "filtered_samples": len(filtered_df),
        "filter_ratio": len(filtered_df) / len(results_df),
        "error_mean": float(results_df["error"].mean()),
        "error_median": float(results_df["error"].median()),
        "bias_mean": float(results_df["bias"].mean()),
        "timestamp": timestamp,
    }

    if args.strategy == "error_threshold":
        stats["threshold"] = args.error_threshold
    elif args.strategy == "percentile":
        stats["percentile"] = args.percentile
        stats["threshold"] = float(threshold)

    stats_path = output_dir / f"filter_stats_{timestamp}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"筛选统计已保存: {stats_path}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"\n下一步:")
    print(f"1. 检查筛选后的样本数量: {len(filtered_df)}")
    print(f"2. 可视化检查部分样本质量")
    print(f"3. 如果满意，可以用筛选后的索引重新训练")


if __name__ == "__main__":
    main()
