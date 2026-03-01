#!/usr/bin/env python3
"""评估 Mixed(8960, D-only) 模型 - Test_Ext"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import cv2

from baseline.models.baseline_dualhead import BaselineDualHead


class ImageOnlyDataset:
    """Minimal dataset that only loads images."""
    def __init__(self, index_csv, img_root=".", resize_w=640, resize_h=480):
        self.df = pd.read_csv(index_csv)
        self.img_root = img_root
        self.resize_w = resize_w
        self.resize_h = resize_h

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Auto-detect image path column
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

        return {"image": x, "img_path": img_path}


def spearmanr(a, b):
    """Compute Spearman correlation."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    ra = a.argsort().argsort().astype(np.float32)
    rb = b.argsort().argsort().astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


def main():
    ckpt_path = 'baseline/runs/mixed_ws4000_sd8960_d_only_correct/ckpt_last.pth'
    index_csv = 'dataset/external_test_washed_processed/test_ext.csv'
    out_dir = 'baseline/runs/mixed_ws4000_sd8960_d_only_correct'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("Model loaded")

    # Create test dataset
    dataset = ImageOnlyDataset(index_csv=index_csv, img_root='.')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate
    model.eval()
    all_S_hat = []
    all_levels = []
    all_img_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Test_Ext"):
            images = batch["image"].to(device)

            outputs = model(images)
            S_hat = outputs['S_hat'].cpu().numpy()

            all_S_hat.append(S_hat)
            all_img_paths.extend(batch["img_path"])

    all_S_hat = np.concatenate(all_S_hat).reshape(-1)

    # Load occlusion levels from CSV
    df = pd.read_csv(index_csv)
    # Auto-detect level column
    level_col = None
    for col in df.columns:
        if 'level' in col.lower() or col == 'occlusion':
            level_col = col
            break
    if level_col is None:
        level_col = df.columns[-1]  # Assume last column

    all_levels = df[level_col].values

    # The CSV uses numeric levels where 5=a (highest), 1=e (lowest)
    # For Spearman correlation, we can use numeric levels directly
    # (higher number = higher occlusion = higher S expected)
    rho = spearmanr(all_S_hat, all_levels)

    # Compute per-level statistics
    # Map numeric levels to letter labels for display
    level_map = {5: 'a', 4: 'b', 3: 'c', 2: 'd', 1: 'e'}
    per_level_stats = {}
    for num_level, letter in level_map.items():
        mask = all_levels == num_level
        if mask.sum() > 0:
            level_S = all_S_hat[mask]
            per_level_stats[letter] = {
                "count": int(mask.sum()),
                "S_mean": float(level_S.mean()),
                "S_std": float(level_S.std()),
            }

    results = {
        "rho": float(rho),
        "per_level_stats": per_level_stats,
    }

    # Save results
    with open(os.path.join(out_dir, 'eval_Test_Ext_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("Test_Ext Results")
    print("="*50)
    print(f"ρ (S_hat, occlusion level): {results['rho']:.4f}")
    print("\nPer-level statistics:")
    for level, stats in results["per_level_stats"].items():
        print(f"  Level {level}: n={stats['count']}, S_mean={stats['S_mean']:.4f}, S_std={stats['S_std']:.4f}")
    print(f"\nSaved to: {os.path.join(out_dir, 'eval_Test_Ext_metrics.json')}")


if __name__ == '__main__':
    main()
