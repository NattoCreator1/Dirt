#!/usr/bin/env python3
"""评估 Mixed(8960, D-only) 模型 - Test_Real - 兼容自定义checkpoint格式"""

import sys
import os

# 确保在项目根目录
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import cv2

from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from baseline.models.baseline_dualhead import BaselineDualHead


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
    index_csv = 'dataset/woodscape_processed/meta/labels_index_ablation.csv'
    out_dir = 'baseline/runs/mixed_ws4000_sd8960_d_only_correct'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint - handle both formats
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)

    # Handle different checkpoint formats
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get('epoch', 'unknown')
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get('step', 'unknown')
    else:
        model.load_state_dict(ckpt)
        epoch = 'unknown'
    print(f"Loaded model from {epoch}")

    # Create test dataset
    spec = WoodscapeIndexSpec(
        index_csv=index_csv,
        img_root='.',
        split_col='split',
        split_value='test',
        global_target='S_full_wgap_alpha50'
    )
    dataset = WoodscapeTileDataset(spec)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate
    model.eval()
    all_S_hat = []
    all_S_gt = []
    all_G_hat = []
    all_G_gt = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Test_Real"):
            images = batch["image"].to(device)
            tile_cov = batch["tile_cov"].to(device)
            S_gt = batch["global_score"].to(device)

            outputs = model(images)
            G_hat = outputs['G_hat'].cpu()
            S_hat = outputs['S_hat'].cpu()

            all_S_hat.append(S_hat)
            all_S_gt.append(S_gt)
            all_G_hat.append(G_hat)
            all_G_gt.append(tile_cov)

    all_S_hat = torch.cat(all_S_hat).cpu().numpy()
    all_S_gt = torch.cat(all_S_gt).cpu().numpy()
    all_G_hat = torch.cat(all_G_hat).cpu().numpy()
    all_G_gt = torch.cat(all_G_gt).cpu().numpy()

    # Compute metrics
    G_tgt = all_G_gt.transpose(0, 3, 1, 2)  # [B,8,8,4] -> [B,4,8,8]
    tile_mae = np.mean(np.abs(all_G_hat - G_tgt))

    glob_mae = np.mean(np.abs(all_S_hat - all_S_gt))
    glob_rmse = np.sqrt(np.mean((all_S_hat - all_S_gt) ** 2))

    S_agg = all_S_hat  # Use S_hat directly
    gap_mae = np.mean(np.abs(all_S_hat - S_agg))

    rho_shat_gt = spearmanr(all_S_hat.reshape(-1), all_S_gt.reshape(-1))

    results = {
        "tile_mae": float(tile_mae),
        "glob_mae": float(glob_mae),
        "glob_rmse": float(glob_rmse),
        "gap_mae": float(gap_mae),
        "rho_shat_gt": float(rho_shat_gt),
    }

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'eval_Test_Real_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("Test_Real Results")
    print("="*50)
    print(f"Tile MAE:      {results['tile_mae']:.4f}")
    print(f"Global MAE:    {results['glob_mae']:.4f}")
    print(f"Global RMSE:   {results['glob_rmse']:.4f}")
    print(f"Gap MAE:       {results['gap_mae']:.4f}")
    print(f"ρ(S_hat, GT): {results['rho_shat_gt']:.4f}")
    print(f"\nSaved to: {os.path.join(out_dir, 'eval_Test_Real_metrics.json')}")


if __name__ == '__main__':
    main()
