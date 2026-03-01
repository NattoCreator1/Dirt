#!/usr/bin/env python3
"""
Add S_full_wgap_alpha50 column to ablation labels CSV.

This script computes a new Severity Score variant that combines:
1. Class weights: w = (0, 0.15, 0.50, 1.0)  [w_gap_strong]
2. Fusion coefficients: α=0.5, β=0.4, γ=0.1  [alpha50]

The hypothesis:
- w_gap_strong strengthens opaque influence across ALL components
- Lower alpha (0.6 -> 0.5) gives more relative weight to Spatial and Dominance
- This "mixed adjustment" validates the two weight systems concept

Usage:
    python scripts/23_add_wgap_alpha50_labels.py
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def make_spatial_weight(H: int, W: int, mode: str = "gaussian") -> np.ndarray:
    """Compute spatial importance weight map."""
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(x, y)

    if mode == "gaussian":
        sigma = 0.5
        Wmap = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    else:
        Wmap = np.ones((H, W))

    return Wmap.astype(np.float32)


def compute_dominance_score(
    tile_cov: np.ndarray,
    class_weights: Tuple,
    eta_trans: float
) -> float:
    """Compute dominance score."""
    t = 1.0 - tile_cov[..., 0]

    if len(t.shape) == 1:
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


def compute_severity_from_tile_cov(
    tile_cov: np.ndarray,
    class_weights: Tuple,
    alpha: float,
    beta: float,
    gamma: float,
    eta_trans: float = 0.9
) -> dict:
    """
    Compute all Severity Score components from tile coverage.

    Args:
        tile_cov: (8, 8, 4) array - tile coverage per class
        class_weights: (w_clean, w_trans, w_semi, w_opaque)
        alpha, beta, gamma: fusion coefficients
        eta_trans: transparency discount for dominance

    Returns:
        dict with S_op, S_sp, S_dom, S_full
    """
    H, W = 8, 8  # tile grid size
    w = np.array(class_weights, dtype=np.float32)

    # S_op: Opacity-aware coverage
    p = tile_cov.mean(axis=(0, 1))  # average per-class coverage
    S_op = float((w * p).sum())

    # S_sp: Spatial weighted
    Wmap = make_spatial_weight(H, W, mode="gaussian")
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
        "S_full": float(np.clip(S_full, 0.0, 1.0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Add S_full_wgap_alpha50 to ablation CSV")
    ap.add_argument("--ablation_csv",
                    default="dataset/woodscape_processed/meta/labels_index_ablation.csv",
                    help="Path to ablation labels CSV")
    ap.add_argument("--labels_tile_dir",
                    default="dataset/woodscape_processed/labels_tile",
                    help="Directory containing tile labels (npz files)")
    ap.add_argument("--out_csv",
                    default="dataset/woodscape_processed/meta/labels_index_ablation.csv",
                    help="Output CSV (default: overwrite input)")
    args = ap.parse_args()

    # Configuration for S_full_wgap_alpha50
    CLASS_WEIGHTS = (0.0, 0.15, 0.50, 1.0)  # w_gap_strong
    ALPHA, BETA, GAMMA = 0.5, 0.4, 0.1  # alpha50 (lower alpha, higher beta)
    ETA_TRANS = 0.9

    print("=" * 70)
    print("Adding S_full_wgap_alpha50 to ablation labels")
    print("=" * 70)
    print(f"Class weights: {CLASS_WEIGHTS}")
    print(f"Fusion: α={ALPHA}, β={BETA}, γ={GAMMA}")
    print("=" * 70)

    # Load ablation CSV
    print(f"\nLoading: {args.ablation_csv}")
    df = pd.read_csv(args.ablation_csv)
    print(f"  {len(df)} samples")

    # Check if column already exists
    if "S_full_wgap_alpha50" in df.columns:
        print("  Column 'S_full_wgap_alpha50' already exists, will recompute")

    # Get npz_path column
    npz_col = None
    for col in ["npz_path", "label_npz", "tile_npz"]:
        if col in df.columns:
            npz_col = col
            break

    if npz_col is None:
        raise ValueError("Cannot find npz_path column in CSV")

    labels_dir = Path(args.labels_tile_dir)

    # Compute new scores
    s_full_wgap_alpha50 = []
    s_op_wgap = []
    s_sp_wgap = []
    s_dom_wgap = []

    print("\nComputing S_full_wgap_alpha50 from tile_cov...")
    for _, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        npz_path = labels_dir / Path(row[npz_col]).name

        if not npz_path.exists():
            # Try using stem from other columns
            stem = Path(row[npz_col]).stem
            npz_path = labels_dir / f"{stem}.npz"

        if not npz_path.exists():
            print(f"Warning: {npz_path} not found")
            s_full_wgap_alpha50.append(np.nan)
            s_op_wgap.append(np.nan)
            s_sp_wgap.append(np.nan)
            s_dom_wgap.append(np.nan)
            continue

        # Load tile coverage
        data = np.load(npz_path)
        tile_cov = data["tile_cov"]  # (8, 8, 4)

        # Normalize to sum to 1 per tile (should already be normalized)
        tile_cov = tile_cov / (tile_cov.sum(axis=2, keepdims=True) + 1e-12)

        # Compute severity with new weights
        scores = compute_severity_from_tile_cov(
            tile_cov=tile_cov,
            class_weights=CLASS_WEIGHTS,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            eta_trans=ETA_TRANS,
        )

        s_full_wgap_alpha50.append(scores["S_full"])
        s_op_wgap.append(scores["S_op"])
        s_sp_wgap.append(scores["S_sp"])
        s_dom_wgap.append(scores["S_dom"])

    # Add new columns
    df["S_full_wgap_alpha50"] = s_full_wgap_alpha50
    df["S_op_wgap"] = s_op_wgap
    df["S_sp_wgap"] = s_sp_wgap
    df["S_dom_wgap"] = s_dom_wgap

    # Reorder columns - put new variant after existing severity variants
    base_cols = ["rgb_path", "npz_path", "split"]
    severity_cols = ["S", "s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00",
                     "S_full_wgap_alpha50",  # New variant
                     "S_op", "S_sp", "S_dom_eta00", "S_dom_eta09",
                     "S_op_wgap", "S_sp_wgap", "S_dom_wgap"]  # New components
    other_cols = ["global_level", "global_bin"]

    cols = base_cols + [c for c in severity_cols if c in df.columns] + other_cols
    df = df[cols]

    # Save updated CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*70}")
    print(f"Saved updated CSV: {out_path}")
    print(f"{'='*70}")

    # Print statistics
    print("\nSeverity Score Statistics:")
    print(f"  S_full (default):       mean={df['S_full'].mean():.4f}, std={df['S_full'].std():.4f}")
    print(f"  S_full_wgap_alpha50:    mean={df['S_full_wgap_alpha50'].mean():.4f}, std={df['S_full_wgap_alpha50'].std():.4f}")

    # Compute correlation
    valid_mask = df["S_full"].notna() & df["S_full_wgap_alpha50"].notna()
    if valid_mask.sum() > 0:
        corr = df.loc[valid_mask, "S_full"].corr(df.loc[valid_mask, "S_full_wgap_alpha50"])
        print(f"\nCorrelation (S_full vs S_full_wgap_alpha50): {corr:.4f}")

    # Component analysis
    print("\nComponent Statistics (w_gap_strong):")
    print(f"  S_op_wgap:   mean={df['S_op_wgap'].mean():.4f}, std={df['S_op_wgap'].std():.4f}")
    print(f"  S_sp_wgap:   mean={df['S_sp_wgap'].mean():.4f}, std={df['S_sp_wgap'].std():.4f}")
    print(f"  S_dom_wgap:  mean={df['S_dom_wgap'].mean():.4f}, std={df['S_dom_wgap'].std():.4f}")

    # Sample rows
    print("\nSample rows:")
    print(df[["rgb_path", "S_full", "S_full_wgap_alpha50"]].head(3).to_string())
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()