#!/usr/bin/env python3
"""
Detailed Tile-level Prediction Accuracy Analysis

This script analyzes:
1. Overall tile MAE
2. Per-class MAE
3. Per-tile error distribution
4. Visual comparison of predictions vs ground truth
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from baseline.models.baseline_dualhead import BaselineDualHead


@torch.no_grad()
def analyze_tile_accuracy(model, loader, device):
    """
    Analyze tile-level prediction accuracy in detail.

    Returns:
    - Overall tile MAE
    - Per-class MAE
    - Error statistics
    - Sample predictions for visualization
    """
    model.eval()

    sum_mae = 0.0
    sum_mae_per_class = np.zeros(4)
    count = 0

    # Collect per-tile errors for distribution analysis
    all_tile_errors = []

    # Store some samples for visualization
    samples = []

    for batch in tqdm(loader, desc="Analyzing tile accuracy", ncols=100):
        x = batch["image"].to(device, non_blocking=True)
        tile_gt = batch["tile_cov"].to(device, non_blocking=True)

        out = model(x)
        G_hat = out["G_hat"]  # [B,4,8,8]

        # Compute MAE per sample, per class
        mae = torch.mean(torch.abs(G_hat - tile_gt.permute(0, 3, 1, 2).contiguous()), dim=[1, 2, 3])  # [B]
        class_mae = torch.mean(torch.abs(G_hat - tile_gt.permute(0, 3, 1, 2).contiguous()), dim=[2, 3, 0])  # [4]

        sum_mae += float(mae.sum().cpu())
        sum_mae_per_class += class_mae.sum(dim=0).cpu().numpy()
        count += x.size(0)

        all_tile_errors.append(mae.view(-1).cpu().numpy())

        # Store some samples
        if len(samples) < 10:
            for i in range(min(5, x.size(0))):
                samples.append({
                    "image_path": batch["img_path"][i] if "img_path" in batch else [""],
                    "G_gt": tile_gt[i].permute(2, 0, 1).cpu().numpy(),  # [4,8,8]
                    "G_hat": G_hat[i].cpu().numpy(),  # [4,8,8]
                    "tile_mae": mae[i].cpu().numpy(),
                })

    # Overall metrics
    overall_mae = sum_mae / (count * 64 * 4)  # 8*8*4 = 256
    per_class_mae = sum_mae_per_class / (count * 64)  # 8*8 = 64 tiles per class

    # Error distribution
    all_errors = np.concatenate(all_tile_errors)

    metrics = {
        "overall_tile_mae": overall_mae,
        "per_class_mae": per_class_mae.tolist(),  # [clean, transparent, semi-transparent, opaque]
        "error_mean": float(np.mean(all_errors)),
        "error_std": float(np.std(all_errors)),
        "error_min": float(np.min(all_errors)),
        "error_max": float(np.max(all_errors)),
        "error_p25": float(np.percentile(all_errors, 25)),
        "error_p50": float(np.percentile(all_errors, 50)),
        "error_p75": float(np.percentile(all_errors, 75)),
        "error_p95": float(np.percentile(all_errors, 95)),
        "error_p99": float(np.percentile(all_errors, 99)),
    }

    return metrics, samples


def print_class_names():
    class_names = ["clean (0)", "transparent (1)", "semi-transparent (2)", "opaque (3)"]
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")


def main():
    ap = argparse.ArgumentParser(description="Analyze tile-level prediction accuracy")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint")
    ap.add_argument("--index_csv", default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv")
    ap.add_argument("--img_root", default=".")
    ap.add_argument("--labels_tile_dir", default=None)
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--global_target", choices=["S", "s"], default="S")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")

    # Setup dataset
    spec = WoodscapeIndexSpec(
        index_csv=args.index_csv,
        img_root=args.img_root,
        labels_tile_dir=args.labels_tile_dir,
        split_col=args.split_col,
        split_value=args.test_split,
        global_target=args.global_target,
    )
    ds = WoodscapeTileDataset(spec)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    print(f"Test set size: {len(ds)}")

    # Analyze
    metrics, samples = analyze_tile_accuracy(model, loader, device)

    # Print results
    print("\n" + "="*70)
    print("Tile 預測準確性詳細分析")
    print("="*70)

    print_class_names()
    print()

    print("整體指標:")
    print(f"  Overall Tile MAE: {metrics['overall_tile_mae']:.4f}")
    print(f"  Error mean:         {metrics['error_mean']:.6f}")
    print(f"  Error std:          {metrics['error_std']:.6f}")
    print()

    print("按類別的 MAE:")
    for i, name in enumerate(["clean", "transparent", "semi-transparent", "opaque"]):
        print(f"  Class {i} ({name:16s}): {metrics['per_class_mae'][i]:.4f}")
    print()

    print("誤差分布:")
    print(f"  Min:   {metrics['error_min']:.6f}")
    print(f"  25%:   {metrics['error_p25']:.6f}")
    print(f"  50% (中位數): {metrics['error_p50']:.6f}")
    print(f"  75%:   {metrics['error_p75']:.6f}")
    print(f"  95%:   {metrics['error_p95']:.6f}")
    print(f"  99%:   {metrics['error_p99']:.6f}")
    print(f"  Max:   {metrics['error_max']:.6f}")
    print()

    # Interpretation
    print("解讀:")
    print(f"   • 平均誤差約 {metrics['overall_tile_mae']*100:.2f}% 每個覆蓋率值的誤差")
    print(f"  • 50% 的 tile 預測誤差小於 {metrics['error_p50']*100:.2f}%")
    print(f"  • 95% 的 tile 預測誤差小於 {metrics['error_p95']*100:.2f}%")
    print()

    # Save results
    out_dir = os.path.dirname(args.ckpt)
    out_path = os.path.join(out_dir, "tile_accuracy_analysis.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {out_path}")

    # Show detailed comparison for first sample
    print("\n樣本詳細對比 (第一個樣本):")
    if samples:
        s = samples[0]
        G_gt = s["G_gt"]  # [4,8,8]
        G_hat = s["G_hat"]  # [4,8,8]

        print(f"  Image: {s['image_path']}")
        print(f"  Tile MAE: {s['tile_mae']:.4f}")
        print()
        print("  Ground Truth vs Prediction (按類別):")
        for i, name in enumerate(["clean", "transparent", "semi-transparent", "opaque"]):
            mae = np.mean(np.abs(G_hat[i, :, :] - G_gt[i, :, :]))
            print(f"    {name:16s}: MAE={mae:.4f}")
        print()
        print("  最差 10 個的 tile (按總誤差排序):")
        # Reshape to [64, 4] for per-tile analysis
        G_hat_flat = G_hat.reshape(4, -1).T  # [64, 4]
        G_gt_flat = G_gt.reshape(4, -1).T  # [64, 4]
        tile_errors = np.abs(G_hat_flat - G_gt_flat).sum(axis=-1)  # [64]
        worst_10 = np.argsort(tile_errors)[-10:]
        for idx in worst_10:
            err = tile_errors[idx]
            err_by_class = np.abs(G_hat_flat[idx, :] - G_gt_flat[idx, :])
            worst_class = np.argmax(err_by_class)
            print(f"    Tile {idx:2d}: error={err:.4f}, worst_class={worst_class}")


if __name__ == "__main__":
    main()
