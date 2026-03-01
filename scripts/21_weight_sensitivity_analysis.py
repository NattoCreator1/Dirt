#!/usr/bin/env python3
"""
Post-hoc weight sensitivity analysis for Severity Score variants.

This script evaluates how different weight configurations (alpha, beta, gamma) and
class weights (w) affect external test performance WITHOUT retraining.

Methodology:
1. Load existing trained models' tile predictions (G_hat)
2. Recompute S variants with different weight configurations
3. Evaluate on external test set to measure sensitivity

Usage:
    python scripts/21_weight_sensitivity_analysis.py
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman correlation."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


class ExternalTestDataset:
    """Dataset for external test images with weak labels."""

    def __init__(self, csv_path: str, img_root: str = ".", resize_w: int = 640, resize_h: int = 480):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)

        import cv2
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.cv2 = cv2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = self.img_root / img_path

        img = self.cv2.imread(str(img_path))
        img = self.cv2.resize(img, (self.resize_w, self.resize_h))
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(img).float(),
            "ext_level": torch.tensor(row["ext_level"], dtype=torch.float32),
        }


def compute_spatial_weight(H: int, W: int, mode: str = "gaussian") -> np.ndarray:
    """Compute spatial importance weight map."""
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(x, y)

    if mode == "gaussian":
        sigma = 0.5
        Wmap = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    elif mode == "linear":
        Wmap = 1.0 - np.sqrt(xx**2 + yy**2) / 1.5
        Wmap = np.clip(Wmap, 0, 1)
    else:
        Wmap = np.ones((H, W))

    return Wmap.astype(np.float32)


def compute_dominance_score(tile_cov: np.ndarray, class_weights: Tuple, eta_trans: float) -> float:
    """Compute dominance score for a single tile."""
    t = 1.0 - tile_cov[..., 0]  # non-clean coverage

    if len(t.shape) == 1:
        # Single tile
        t = t.reshape(1, -1)

    N, M, C = tile_cov.shape
    w = np.array(class_weights)

    # Weighted coverage per tile
    wc = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            wc[i, j] = (w * tile_cov[i, j]).sum() / (w.sum() + 1e-12)

    # Transparency ratio
    trans = (tile_cov[..., 1] + tile_cov[..., 2]).sum() / (tile_cov[..., :3].sum() + 1e-12)

    # Find max tile
    max_idx = np.unravel_index(wc.argmax(), wc.shape)
    dom_raw = wc[max_idx]

    # Transparency discount
    factor = 1.0 - eta_trans * trans
    S_dom = float(dom_raw * factor)

    return S_dom


def recompute_s_from_tile_cov(tile_cov: np.ndarray,
                               class_weights: Tuple = (0.0, 0.33, 0.66, 1.0),
                               alpha: float = 0.6,
                               beta: float = 0.3,
                               gamma: float = 0.1,
                               eta_trans: float = 0.9) -> Dict[str, float]:
    """
    Recompute all S variants from tile coverage predictions.

    Args:
        tile_cov: (8, 8, 4) array of predicted tile coverage
        class_weights: (w_clean, w_trans, w_semi, w_opaque)
        alpha, beta, gamma: fusion coefficients
        eta_trans: transparency discount for dominance

    Returns:
        Dictionary with all S variant scores
    """
    H, W = 8, 8  # tile grid size
    w = np.array(class_weights, dtype=np.float32)

    # S_op: Opacity-aware coverage
    p = tile_cov.mean(axis=(0, 1))  # average per-class coverage
    S_op = float((w * p).sum())

    # S_sp: Spatial weighted
    Wmap = compute_spatial_weight(H, W, mode="gaussian")
    tile_weights = tile_cov @ w  # (8, 8) weighted coverage per tile
    S_sp = float((Wmap * tile_weights).sum() / Wmap.sum())

    # S_dom: Dominance
    S_dom = compute_dominance_score(tile_cov, class_weights, eta_trans)

    # S_full: fused score
    S_full = alpha * S_op + beta * S_sp + gamma * S_dom

    return {
        "S_op": S_op,
        "S_sp": S_sp,
        "S_dom": S_dom,
        "S_full": S_full,
        "S_op_only": S_op,  # Opacity-only variant
        "S_op_sp": 0.7 * S_op + 0.3 * S_sp,  # No dominance
    }


def analyze_weight_sensitivity(model_paths: Dict[str, str],
                                ext_csv: str,
                                ext_img_root: str,
                                device: torch.device,
                                weight_configs: List[Dict]) -> pd.DataFrame:
    """
    Analyze sensitivity to different weight configurations.

    Args:
        model_paths: Dict mapping model name to checkpoint path
        ext_csv: Path to external test CSV
        ext_img_root: Root dir for images
        device: torch device
        weight_configs: List of weight configuration dicts

    Returns:
        DataFrame with sensitivity analysis results
    """
    from baseline.models.baseline_dualhead import BaselineDualHead

    # Create external test dataset
    ds = ExternalTestDataset(ext_csv, ext_img_root)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                       num_workers=2, pin_memory=False)

    results = []

    for model_name, ckpt_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*60}")

        # Load model
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BaselineDualHead(pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # Collect predictions
        all_tile_covs = []
        all_ext_levels = []

        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                y_ext = batch["ext_level"]

                out = model(x)
                G_hat = out["G_hat"]  # [B, 4, 8, 8]

                for i in range(x.size(0)):
                    tile_cov = G_hat[i].permute(1, 2, 0).cpu().numpy()  # [8, 8, 4]
                    # Normalize to sum to 1 per tile
                    tile_cov = tile_cov / (tile_cov.sum(axis=2, keepdims=True) + 1e-12)
                    all_tile_covs.append(tile_cov)
                    all_ext_levels.append(y_ext[i].item())

        all_ext_levels = np.array(all_ext_levels)

        # Test each weight configuration
        for config in weight_configs:
            config_name = config["name"]

            # Recompute S for all samples with this config
            all_s_recomputed = []

            for tile_cov in all_tile_covs:
                s_dict = recompute_s_from_tile_cov(
                    tile_cov,
                    class_weights=config.get("class_weights", (0.0, 0.33, 0.66, 1.0)),
                    alpha=config.get("alpha", 0.6),
                    beta=config.get("beta", 0.3),
                    gamma=config.get("gamma", 0.1),
                    eta_trans=config.get("eta_trans", 0.9)
                )
                all_s_recomputed.append(s_dict["S_full"])

            all_s_recomputed = np.array(all_s_recomputed)

            # Scale to 1-5 for comparison
            s_scaled = all_s_recomputed * 4.0 + 1.0

            # Compute metrics
            mae = float(np.mean(np.abs(s_scaled - all_ext_levels)))
            rmse = float(np.sqrt(np.mean((s_scaled - all_ext_levels) ** 2)))
            rho = spearmanr(s_scaled, all_ext_levels.astype(np.float32))

            results.append({
                "model": model_name,
                "config": config_name,
                "alpha": config.get("alpha", 0.6),
                "beta": config.get("beta", 0.3),
                "gamma": config.get("gamma", 0.1),
                "ext_mae": mae,
                "ext_rmse": rmse,
                "ext_rho": rho,
            })

            print(f"  {config_name:<30} | MAE={mae:.4f} | RMSE={rmse:.4f} | ρ={rho:.4f}")

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Weight sensitivity analysis for Severity Score")
    ap.add_argument("--ablation_dir", default="baseline/runs/ablation_label_def",
                    help="Directory containing ablation experiment results")
    ap.add_argument("--ext_csv", default="dataset/external_test_washed_processed/test_ext.csv")
    ap.add_argument("--ext_img_root", default=".")
    ap.add_argument("--out_csv", default="baseline/runs/ablation_label_def/weight_sensitivity.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ablation_dir = Path(args.ablation_dir)

    # Find model checkpoints
    model_paths = {}
    for target in ["s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00"]:
        ckpt_dir = ablation_dir / f"ablation_{target}"
        if ckpt_dir.exists():
            ckpt_path = ckpt_dir / "ckpt_best.pth"
            if ckpt_path.exists():
                model_paths[target] = str(ckpt_path)

    print(f"\nFound {len(model_paths)} models to analyze")

    # Define weight configurations to test

    # Scheme A: Adjust fusion coefficients (alpha, beta, gamma)
    fusion_configs = [
        {
            "name": "Default (0.6, 0.3, 0.1)",
            "alpha": 0.6, "beta": 0.3, "gamma": 0.1,
            "class_weights": (0.0, 0.33, 0.66, 1.0),
        },
        {
            "name": "Op75 (0.75, 0.20, 0.05)",
            "alpha": 0.75, "beta": 0.20, "gamma": 0.05,
            "class_weights": (0.0, 0.33, 0.66, 1.0),
        },
        {
            "name": "Op85 (0.85, 0.10, 0.05)",
            "alpha": 0.85, "beta": 0.10, "gamma": 0.05,
            "class_weights": (0.0, 0.33, 0.66, 1.0),
        },
        {
            "name": "Op90 (0.90, 0.05, 0.05)",
            "alpha": 0.90, "beta": 0.05, "gamma": 0.05,
            "class_weights": (0.0, 0.33, 0.66, 1.0),
        },
    ]

    # Scheme B: Adjust class weights (w)
    class_weight_configs = [
        {
            "name": "Default w",
            "alpha": 0.6, "beta": 0.3, "gamma": 0.1,
            "class_weights": (0.0, 0.33, 0.66, 1.0),
        },
        {
            "name": "w_gap_mild",
            "alpha": 0.6, "beta": 0.3, "gamma": 0.1,
            "class_weights": (0.0, 0.25, 0.60, 1.0),
        },
        {
            "name": "w_gap_strong",
            "alpha": 0.6, "beta": 0.3, "gamma": 0.1,
            "class_weights": (0.0, 0.15, 0.50, 1.0),
        },
    ]

    # Combine all configs
    all_configs = fusion_configs + class_weight_configs

    print(f"\nTesting {len(all_configs)} weight configurations...")

    # Run analysis
    df = analyze_weight_sensitivity(model_paths, args.ext_csv, args.ext_img_root, device, all_configs)

    # Save results
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Print summary
    print("\n" + "=" * 140)
    print("Weight Sensitivity Analysis Summary")
    print("=" * 140)

    print("\nScheme A: Fusion Coefficient Sensitivity")
    print("-" * 140)
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        fusion_df = model_df[model_df["config"].str.contains("Op|Default", na=False)]
        for _, row in fusion_df.iterrows():
            delta_rho = row["ext_rho"] - fusion_df[fusion_df["config"] == "Default (0.6, 0.3, 0.1)"]["ext_rho"].values[0]
            print(f"  {row['config']:<25} | ρ={row['ext_rho']:.4f} | Δρ={delta_rho:+.4f}")

    print("\n" + "-" * 140)
    print("\nScheme B: Class Weight Sensitivity")
    print("-" * 140)
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        weight_df = model_df[model_df["config"].str.contains("w_gap|Default w", na=False)]
        for _, row in weight_df.iterrows():
            delta_rho = row["ext_rho"] - weight_df[weight_df["config"] == "Default w"]["ext_rho"].values[0]
            print(f"  {row['config']:<25} | ρ={row['ext_rho']:.4f} | Δρ={delta_rho:+.4f}")

    print("\n" + "=" * 140)
    print("Conclusions:")
    print("  - Positive Δρ indicates better alignment with external test ranking")
    print("  - Use results to determine optimal (alpha, beta, gamma) or class_weights")
    print("  - If Op85/Op90 shows significant improvement, consider full retraining")
    print("  - Spatial and Dominance components may still provide value for lane probe")
    print("=" * 140)


if __name__ == "__main__":
    main()
