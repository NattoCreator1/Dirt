#!/usr/bin/env python3
"""
SD Tile-Only Training (保险策略)

Key Strategy:
- SD samples train ONLY the tile head (local supervision)
- WoodScape samples train BOTH tile and global heads
- Global head scale is anchored entirely on real (WoodScape) data

Intuition: SD8960 may have risk for global head scale learning,
but it still has value for tile head's "local dirt coverage/category texture".
This maximizes SD's local supervision while completely anchoring "cross-domain scale"
on real data.
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.models.baseline_dualhead import BaselineDualHead
from sd_calibration_work.phase2_consistency_gate.train_a_plus_only_hardened import MixedWoodscapeDataset, spearmanr


def train_one_epoch(model, loader, optimizer, device, step_offset, lambda_glob=1.0):
    """
    Train for one epoch with SD tile-only strategy.

    SD samples:
        - Tile head: FULL loss weight
        - Global head: ZERO loss weight (no scale learning from SD)

    WoodScape samples:
        - Tile head: FULL loss weight
        - Global head: FULL loss weight
    """
    model.train()
    sum_tile = 0.0
    sum_glob = 0.0
    sum_total = 0.0
    n_batches = 0

    # Track SD vs Real contributions
    sd_tile_loss_sum = 0.0
    real_tile_loss_sum = 0.0
    real_glob_loss_sum = 0.0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        tile_cov = batch["tile_cov"].to(device)
        global_scores = batch["global_score"].to(device)
        is_sd_list = batch["is_sd"]
        batch_size = images.shape[0]

        # Forward pass
        outputs = model(images)
        G_hat = outputs['G_hat']  # [B,4,8,8]
        S_hat = outputs['S_hat']  # [B,1]
        S_agg = outputs['S_agg']  # [B,1]

        # Tile loss (all samples, FULL weight - both SD and WoodScape)
        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()  # [B,4,8,8]
        tile_loss_per_sample = F.l1_loss(G_hat, G_tgt, reduction='none').mean(dim=(1,2,3))  # [B]
        L_tile = tile_loss_per_sample.mean()

        # Separate tile loss by source for monitoring
        for i, is_sd in enumerate(is_sd_list):
            if is_sd:
                sd_tile_loss_sum += tile_loss_per_sample[i].item()
            else:
                real_tile_loss_sum += tile_loss_per_sample[i].item()

        # Global loss: ONLY WoodScape samples contribute
        # SD samples have ZERO global loss weight
        glob_loss_per_sample = F.l1_loss(S_hat, global_scores, reduction='none').squeeze(-1)  # [B]

        # Zero out SD samples' global loss
        for i, is_sd in enumerate(is_sd_list):
            if is_sd:
                glob_loss_per_sample[i] = 0.0
            else:
                real_glob_loss_sum += glob_loss_per_sample[i].item()

        # Average over non-zero samples (WoodScape only for global)
        n_real = sum(1 for is_sd in is_sd_list if not is_sd)
        L_glob = glob_loss_per_sample.sum() / max(n_real, 1)

        # Total loss
        L_total = L_tile + lambda_glob * L_glob

        # Backward
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        # Metrics
        sum_tile += L_tile.item()
        sum_glob += L_glob.item()
        sum_total += L_total.item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{L_total.item():.4f}",
            "tile": f"{L_tile.item():.4f}",
            "glob": f"{L_glob.item():.4f}",
            "n_real": f"{n_real}/{batch_size}"
        })

    return {
        "loss": sum_total / n_batches,
        "loss_tile": sum_tile / n_batches,
        "loss_glob": sum_glob / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    sum_tile = 0.0
    sum_glob = 0.0
    n = 0

    all_shat = []
    all_sagg = []
    all_gt = []

    for batch in tqdm(loader, desc="Evaluating", ncols=80):
        images = batch["image"].to(device)
        tile_cov = batch["tile_cov"].to(device)
        global_scores = batch["global_score"].to(device)

        outputs = model(images)
        G_hat = outputs['G_hat']
        S_hat = outputs['S_hat']
        S_agg = outputs['S_agg']

        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()
        tile_mae = torch.mean(torch.abs(G_hat - G_tgt))
        glob_mae = torch.mean(torch.abs(S_hat - global_scores))

        bs = images.shape[0]
        sum_tile += float(tile_mae.cpu()) * bs
        sum_glob += float(glob_mae.cpu()) * bs
        n += bs

        all_shat.append(S_hat.cpu().numpy())
        all_sagg.append(S_agg.cpu().numpy())
        all_gt.append(global_scores.cpu().numpy())

    shat = np.concatenate(all_shat).reshape(-1)
    sagg = np.concatenate(all_sagg).reshape(-1)
    gt = np.concatenate(all_gt).reshape(-1)

    rho_shat_gt = spearmanr(shat, gt)
    rho_shat_sagg = spearmanr(shat, sagg)

    return {
        "tile_mae": sum_tile / n,
        "glob_mae": sum_glob / n,
        "rho_shat_gt": rho_shat_gt,
        "rho_shat_sagg": rho_shat_sagg,
    }


def main():
    parser = argparse.ArgumentParser(description="SD Tile-Only Training (Insurance Policy)")
    parser.add_argument("--mixed_csv", type=str, default="sd_calibration_work/phase1_importance_sampling/mixed_ws4000_sd8960.csv")
    parser.add_argument("--img_root", type=str, default=".")
    parser.add_argument("--run_name", type=str, default="mixed_ws4000_sd8960_tile_only")
    parser.add_argument("--out_root", type=str, default="baseline/runs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda_glob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=2000)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save args
    args_path = os.path.join(out_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print(f"Loading mixed dataset: {args.mixed_csv}")
    dataset = MixedWoodscapeDataset(
        mixed_csv=args.mixed_csv,
        img_root=args.img_root,
        global_target="S_full_wgap_alpha50"
    )
    print(f"Dataset size: {len(dataset)}")

    # Create dataloaders
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create model
    model = BaselineDualHead(pretrained=False).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\nSD Tile-Only Training (Insurance Policy)")
    print(f"  SD samples: Train tile head ONLY (no global loss)")
    print(f"  WoodScape samples: Train BOTH tile and global heads")
    print(f"  Global head scale anchored entirely on WoodScape data")
    print(f"  Total steps: {args.total_steps}")
    print(f"  λ_glob: {args.lambda_glob}")

    # Metrics CSV
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("step,loss,loss_tile,loss_glob\n")

    # Training loop
    step = 0

    while step < args.total_steps:
        metrics = train_one_epoch(
            model, loader, optimizer, device,
            step_offset=step, lambda_glob=args.lambda_glob
        )
        step += len(loader)

        # Log metrics
        with open(metrics_path, "a") as f:
            f.write(f"{step},{metrics['loss']:.4f},{metrics['loss_tile']:.4f},{metrics['loss_glob']:.4f}\n")

        # Print progress
        if step % args.log_interval == 0 or step >= args.total_steps:
            print(f"\nStep {step}/{args.total_steps}")
            print(f"  Loss: {metrics['loss']:.4f} (Tile: {metrics['loss_tile']:.4f}, Glob: {metrics['loss_glob']:.4f})")

        # Save checkpoint
        if step % args.save_interval == 0 or step >= args.total_steps:
            ckpt_path = os.path.join(out_dir, f"ckpt_{step}.pth")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            # Save last checkpoint
            last_path = os.path.join(out_dir, "ckpt_last.pth")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, last_path)

    print(f"\nTraining complete!")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
