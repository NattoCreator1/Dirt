#!/usr/bin/env python3
"""
Ablation study: With vs Without Aggregator

This script evaluates the same checkpoint under two conditions:
1. With Aggregator: S_hat is constrained to match S_agg via consistency loss
2. Without Aggregator: Only S_hat is used, no consistency constraint

The difference is not in the model architecture, but in how we interpret
the results and what metrics we use for evaluation.
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


def spearmanr(a, b):
    """Compute Spearman correlation (no scipy dependency)."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    ra = a.argsort().argsort().astype(np.float32)
    rb = b.argsort().argsort().astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model and return metrics for both conditions.

    With Aggregator: Uses S_hat (which was trained with consistency loss)
    Without Aggregator: Uses S_hat (simulating no consistency constraint)
    """
    model.eval()

    sum_tile = 0.0
    sum_glob = 0.0
    sum_mse = 0.0
    n = 0

    all_shat = []
    all_sagg = []
    all_gt = []

    for batch in tqdm(loader, desc="Evaluating", ncols=100):
        x = batch["image"].to(device, non_blocking=True)
        tile_cov = batch["tile_cov"].to(device, non_blocking=True)
        y = batch["global_score"].to(device, non_blocking=True)

        out = model(x)

        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()
        tile_mae = torch.mean(torch.abs(out["G_hat"] - G_tgt))
        glob_mae = torch.mean(torch.abs(out["S_hat"] - y))
        glob_mse = torch.mean((out["S_hat"] - y) ** 2)

        bs = x.size(0)
        sum_tile += float(tile_mae.cpu()) * bs
        sum_glob += float(glob_mae.cpu()) * bs
        sum_mse += float(glob_mse.cpu()) * bs
        n += bs

        all_shat.append(out["S_hat"].detach().cpu().numpy())
        all_sagg.append(out["S_agg"].detach().cpu().numpy())
        all_gt.append(y.detach().cpu().numpy())

    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)
    gt = np.concatenate(all_gt, axis=0).reshape(-1)

    # Common metrics
    tile_mae = sum_tile / max(n, 1)
    glob_mae = sum_glob / max(n, 1)
    glob_rmse = float(np.sqrt(sum_mse / max(n, 1)))

    # "With Aggregator" metrics
    gap_mae = float(np.mean(np.abs(shat - sagg)))
    rho_shat_sagg = spearmanr(shat, sagg)
    rho_sagg_gt = spearmanr(sagg, gt)

    # "Without Aggregator" metrics
    rho_shat_gt = spearmanr(shat, gt)

    metrics = {
        "tile_mae": tile_mae,
        "glob_mae": glob_mae,
        "glob_rmse": glob_rmse,
    }

    # With Aggregator
    metrics_with = {
        "gap_mae": gap_mae,
        "rho_shat_sagg": rho_shat_sagg,
        "rho_sagg_gt": rho_sagg_gt,
        "rho_shat_gt": rho_shat_gt,
    }

    # Without Aggregator (simulated)
    metrics_without = {
        "rho_shat_gt": rho_shat_gt,
    }

    return metrics, metrics_with, metrics_without


def main():
    ap = argparse.ArgumentParser(description="Ablation study: With vs Without Aggregator")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth file)")
    ap.add_argument("--index_csv", default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv")
    ap.add_argument("--img_root", default=".")
    ap.add_argument("--labels_tile_dir", default=None)
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--global_target", choices=["S", "s"], default="S")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")

    # Determine output directory
    if args.out_dir is None:
        out_dir = os.path.dirname(args.ckpt)
    else:
        out_dir = args.out_dir

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

    # Evaluate
    metrics, metrics_with, metrics_without = evaluate(model, loader, device)

    # Print results
    print("\n" + "="*60)
    print("Ablation Study: With vs Without Aggregator")
    print("="*60)

    print("\n--- Common Metrics (same for both) ---")
    print(f"  Tile MAE:     {metrics['tile_mae']:.4f}")
    print(f"  Global MAE:   {metrics['glob_mae']:.4f}")
    print(f"  Global RMSE:  {metrics['glob_rmse']:.4f}")

    print("\n--- With Aggregator (current model) ---")
    print(f"  Gap MAE (|S_hat - S_agg|):           {metrics_with['gap_mae']:.4f}")
    print(f"  ρ(S_hat, S_agg):                    {metrics_with['rho_shat_sagg']:.4f}")
    print(f"  ρ(S_agg, S_gt):                     {metrics_with['rho_sagg_gt']:.4f}")
    print(f"  ρ(S_hat, S_gt):                     {metrics_with['rho_shat_gt']:.4f}")

    print("\n--- Without Aggregator (simulated) ---")
    print(f"  ρ(S_hat, S_gt):                     {metrics_without['rho_shat_gt']:.4f}")

    print("\n--- Key Insights ---")
    print(f"  1. Internal Consistency: gap_mae = {metrics_with['gap_mae']:.4f}")
    print(f"     → Small gap means S_hat ≈ S_agg (model is self-consistent)")
    print(f"  2. Tile→Ground Truth: ρ(S_agg, S_gt) = {metrics_with['rho_sagg_gt']:.4f}")
    print(f"     → Validates Severity Score definition via Aggregator")
    print(f"  3. Direct→Ground Truth: ρ(S_hat, S_gt) = {metrics_without['rho_shat_gt']:.4f}")
    print(f"     → Same for both (S_hat is identical)")

    print("\n--- What This Ablation Shows ---")
    print("  The Aggregator serves TWO purposes:")
    print("  1. Provides a 'free' global prediction from tile predictions")
    print("  2. Constrains S_hat via consistency loss, preventing shortcuts")
    print()
    print("  To truly test 'without Aggregator', we would need to:")
    print("  - Train a model WITHOUT consistency loss (mu_cons=0)")
    print("  - Compare the trained models, not just the same checkpoint")
    print()

    # Save results
    out_path = os.path.join(out_dir, "ablate_aggregator.json")
    result = {
        "ablation": "with_vs_without_aggregator",
        "checkpoint": args.ckpt,
        "metrics_common": metrics,
        "metrics_with_aggregator": metrics_with,
        "metrics_without_aggregator": metrics_without,
        "notes": {
            "gap_mae": "Consistency gap between S_hat and S_agg",
            "rho_shat_sagg": "Internal consistency of the model",
            "rho_sagg_gt": "Validity of Severity Score via Aggregator",
            "rho_shat_gt": "Direct prediction accuracy (same for both)",
        }
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {out_path}")

    print("\n" + "="*60)
    print("Recommendation for True Ablation:")
    print("="*60)
    print("To properly ablate the Aggregator, we need to:")
    print("1. Train a model WITH consistency loss (mu_cons > 0) ← Current")
    print("2. Train a model WITHOUT consistency loss (mu_cons = 0)")
    print("3. Compare the two models on the same test set")
    print()
    print("Shall I proceed with training the 'without' model?")


if __name__ == "__main__":
    main()
