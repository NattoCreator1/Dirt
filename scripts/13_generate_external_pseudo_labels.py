#!/usr/bin/env python3
"""
Generate Pseudo-Labels for External Test Set

This script:
1. Runs inference on external_test using trained baseline model
2. Saves tile-level coverage predictions (G_hat) as npz files
3. Saves global predictions (S_hat, S_agg) to a new index CSV
4. Creates a "pseudo-labeled" external test set compatible with Woodscape format

Usage:
    python scripts/13_generate_external_pseudo_labels.py \
        --ckpt baseline/runs/model1_r18_640x480/ckpt_best.pth \
        --ext_csv dataset/my_external_test/test_ext.csv \
        --out_root dataset/external_test_pseudo
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baseline.models.baseline_dualhead import BaselineDualHead


class ExternalTestDataset(Dataset):
    """Dataset for external test images (no labels, just images)."""

    def __init__(self, csv_path: str, img_root: str = ".", resize_w: int = 640, resize_h: int = 480):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.resize_w = resize_w
        self.resize_h = resize_h

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = self.img_root / img_path
        else:
            img_path = Path(img_path)

        # Read image using cv2
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Resize
        img = cv2.resize(img, (self.resize_w, self.resize_h))

        # BGR to RGB
        img = img[:, :, ::-1]

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(img).float(),
            "image_id": row["image_id"],
            "image_path": str(img_path),
            "orig_label": row.get("ext_level", -1),  # Original weak label
        }


@torch.no_grad()
def generate_pseudo_labels(model, loader, device, out_dir):
    """Generate pseudo-labels for external test set."""

    model.eval()

    results = []

    # Tile labels directory
    tile_dir = Path(out_dir) / "labels_tile"
    tile_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Generating pseudo-labels", ncols=100):
        x = batch["image"].to(device, non_blocking=True)
        image_ids = batch["image_id"]
        image_paths = batch["image_path"]
        orig_labels = batch["orig_label"]

        # Forward pass
        out = model(x)
        G_hat = out["G_hat"]  # [B, 4, 8, 8]
        S_hat = out["S_hat"]  # [B, 1]
        S_agg = out["S_agg"]  # [B, 1]

        # Process each sample
        for i in range(x.size(0)):
            image_id = image_ids[i]
            image_path = image_paths[i]

            # G_hat: [4, 8, 8] -> transpose to [8, 8, 4] for saving
            G_hat_np = G_hat[i].cpu().numpy().transpose(1, 2, 0)  # [8, 8, 4]

            # Save tile labels as npz
            tile_path = tile_dir / f"{image_id}.npz"
            np.savez_compressed(
                tile_path,
                tile_cov=G_hat_np,  # [8, 8, 4]
            )

            # Collect results for CSV
            results.append({
                "image_id": image_id,
                "image_path": image_path,
                "tile_label_path": str(tile_path),
                # Original weak label (for reference)
                "orig_occlusion_level": int(orig_labels[i].cpu().numpy()) if orig_labels[i] >= 0 else -1,
                # New pseudo labels (from model)
                "pseudo_S": float(S_hat[i].cpu().numpy()),
                "pseudo_S_agg": float(S_agg[i].cpu().numpy()),
                # Per-class average coverage (for analysis)
                "pseudo_avg_clean": float(G_hat_np[:, :, 0].mean()),
                "pseudo_avg_transparent": float(G_hat_np[:, :, 1].mean()),
                "pseudo_avg_semi_transparent": float(G_hat_np[:, :, 2].mean()),
                "pseudo_avg_opaque": float(G_hat_np[:, :, 3].mean()),
            })

    return results


def main():
    ap = argparse.ArgumentParser(description="Generate pseudo-labels for external test set")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint")
    ap.add_argument("--ext_csv", default="dataset/my_external_test/test_ext.csv",
                    help="Path to external test CSV")
    ap.add_argument("--img_root", default=".", help="Root directory for images")
    ap.add_argument("--out_root", default="dataset/external_test_pseudo",
                    help="Output directory for pseudo-labeled data")
    ap.add_argument("--resize_w", type=int, default=640)
    ap.add_argument("--resize_h", type=int, default=480)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")

    # Create dataset and loader
    print(f"Loading external test data from: {args.ext_csv}")
    ds = ExternalTestDataset(args.ext_csv, args.img_root, args.resize_w, args.resize_h)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    print(f"External test set size: {len(ds)}")

    # Generate pseudo-labels
    print("\nGenerating pseudo-labels...")
    results = generate_pseudo_labels(model, loader, device, args.out_root)

    # Save results to CSV
    out_csv = out_dir / "test_ext_pseudo.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results)} pseudo-labeled samples to: {out_csv}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Pseudo-Label Statistics")
    print("=" * 70)

    print(f"\nS (Global Head prediction):")
    print(f"  Mean:   {df['pseudo_S'].mean():.4f}")
    print(f"  Std:    {df['pseudo_S'].std():.4f}")
    print(f"  Min:    {df['pseudo_S'].min():.4f}")
    print(f"  Max:    {df['pseudo_S'].max():.4f}")

    print(f"\nS_agg (Aggregator prediction):")
    print(f"  Mean:   {df['pseudo_S_agg'].mean():.4f}")
    print(f"  Std:    {df['pseudo_S_agg'].std():.4f}")
    print(f"  Min:    {df['pseudo_S_agg'].min():.4f}")
    print(f"  Max:    {df['pseudo_S_agg'].max():.4f}")

    print(f"\nGap |S - S_agg|:")
    gap = np.abs(df['pseudo_S'] - df['pseudo_S_agg'])
    print(f"  Mean:   {gap.mean():.4f}")
    print(f"  Std:    {gap.std():.4f}")

    print(f"\nPer-class average coverage:")
    print(f"  Clean:             {df['pseudo_avg_clean'].mean():.4f}")
    print(f"  Transparent:       {df['pseudo_avg_transparent'].mean():.4f}")
    print(f"  Semi-transparent:  {df['pseudo_avg_semi_transparent'].mean():.4f}")
    print(f"  Opaque:            {df['pseudo_avg_opaque'].mean():.4f}")

    # Print statistics by original occlusion level
    if "orig_occlusion_level" in df.columns:
        print("\n" + "=" * 70)
        print("Statistics by Original Occlusion Level")
        print("=" * 70)

        level_names = {
            5: "occlusion_a (heaviest)",
            4: "occlusion_b",
            3: "occlusion_c",
            2: "occlusion_d",
            1: "occlusion_e (lightest)",
        }

        for level in sorted(df["orig_occlusion_level"].unique()):
            if level < 1:
                continue
            subset = df[df["orig_occlusion_level"] == level]
            name = level_names.get(level, f"level_{level}")
            print(f"\nLevel {level} ({name}), n={len(subset)}:")
            print(f"  S_mean:     {subset['pseudo_S'].mean():.4f} ± {subset['pseudo_S'].std():.4f}")
            print(f"  S_agg_mean: {subset['pseudo_S_agg'].mean():.4f} ± {subset['pseudo_S_agg'].std():.4f}")

    # Save summary
    summary = {
        "checkpoint": str(args.ckpt),
        "num_samples": len(results),
        "output_dir": str(args.out_root),
        "statistics": {
            "S_mean": float(df['pseudo_S'].mean()),
            "S_std": float(df['pseudo_S'].std()),
            "S_min": float(df['pseudo_S'].min()),
            "S_max": float(df['pseudo_S'].max()),
            "S_agg_mean": float(df['pseudo_S_agg'].mean()),
            "S_agg_std": float(df['pseudo_S_agg'].std()),
            "S_agg_min": float(df['pseudo_S_agg'].min()),
            "S_agg_max": float(df['pseudo_S_agg'].max()),
            "gap_mean": float(gap.mean()),
        },
        "per_class_coverage": {
            "clean": float(df['pseudo_avg_clean'].mean()),
            "transparent": float(df['pseudo_avg_transparent'].mean()),
            "semi_transparent": float(df['pseudo_avg_semi_transparent'].mean()),
            "opaque": float(df['pseudo_avg_opaque'].mean()),
        },
    }

    summary_path = out_dir / "pseudo_label_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("DONE! Pseudo-labeled external test set created.")
    print("=" * 70)
    print(f"\nOutput structure:")
    print(f"  {out_dir}/")
    print(f"    ├── labels_tile/          # Tile-level coverage labels (npz files)")
    print(f"    ├── test_ext_pseudo.csv   # Index file with pseudo labels")
    print(f"    └── pseudo_label_summary.json")


if __name__ == "__main__":
    main()
