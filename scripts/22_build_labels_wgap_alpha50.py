#!/usr/bin/env python3
"""
Generate WoodScape tile + global labels with w_gap_strong + alpha=0.5 configuration.

This script generates labels using:
- Class weights: w = (0, 0.15, 0.50, 1.0)  [w_gap_strong]
- Fusion coefficients: α=0.5, β=0.4, γ=0.1  [alpha50 with redistributed beta/gamma]

Based on weight sensitivity analysis findings:
1. w_gap_strong strengthens opaque influence across ALL components
2. Lowering alpha (0.6 -> 0.5) gives more relative weight to Spatial and Dominance
3. This "mixed adjustment" strategy should validate the two weight systems concept

Usage:
    python scripts/22_build_labels_wgap_alpha50.py
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

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
    elif mode == "linear":
        Wmap = 1.0 - np.sqrt(xx**2 + yy**2) / 1.5
        Wmap = np.clip(Wmap, 0, 1)
    else:
        Wmap = np.ones((H, W))

    return Wmap.astype(np.float32)


def compute_dominance_class_aware(
    tile_cov: np.ndarray,
    class_weights: Tuple = (0.0, 0.33, 0.66, 1.0),
    eta_trans: float = 0.9
) -> Tuple[float, float, float, Tuple[int, int]]:
    """
    Compute dominance score (concentrated blockage).

    Returns:
        S_dom: final dominance score with transparency discount
        dom_raw: raw max tile coverage (before transparency discount)
        trans_ratio: transparency ratio in the whole image
        dom_ij: (i, j) location of the dominant tile
    """
    t = 1.0 - tile_cov[..., 0]

    if len(t.shape) == 1:
        t = t.reshape(1, -1)

    N, M, C = tile_cov.shape
    w = np.array(class_weights)

    wc = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            wc[i, j] = (w * tile_cov[i, j]).sum() / (w.sum() + 1e-12)

    trans = (tile_cov[..., 1] + tile_cov[..., 2]).sum() / (tile_cov[..., :3].sum() + 1e-12)

    max_idx = np.unravel_index(wc.argmax(), wc.shape)
    dom_raw = wc[max_idx]

    factor = 1.0 - eta_trans * trans
    S_dom = float(dom_raw * factor)

    return S_dom, dom_raw, trans_ratio, (i, j)


def compute_severity_score(
    mask: np.ndarray,
    tile_cov: np.ndarray,
    class_weights: Tuple = (0.0, 0.15, 0.50, 1.0),
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
    eta_trans: float = 0.9,
    spatial_mode: str = "gaussian",
) -> Dict[str, float]:
    """
    Compute Severity Score with custom class weights and fusion coefficients.

    Args:
        mask: (H, W) int64 in {0, 1, 2, 3}
        tile_cov: (8, 8, 4) float32
        class_weights: (w_clean, w_trans, w_semi, w_opaque)
        alpha, beta, gamma: fusion coefficients
        eta_trans: transparency discount for dominance

    Returns:
        Dict with S_full, S_op, S_sp, S_dom
    """
    H, W = mask.shape
    w = np.array(class_weights, dtype=np.float32)

    # S_op: Opacity-aware coverage
    p = np.array([(mask == c).sum() / mask.size for c in range(4)], dtype=np.float32)
    S_op = float((w * p).sum())

    # S_sp: Spatial weighted
    Wmap = make_spatial_weight(H, W, mode=spatial_mode)
    S_sp = float((Wmap * w[mask]).sum())

    # S_dom: Dominance
    S_dom, _, _, _ = compute_dominance_class_aware(
        tile_cov, class_weights=class_weights, eta_trans=eta_trans
    )

    # S_full: Fused score
    S_full = alpha * S_op + beta * S_sp + gamma * S_dom
    S_full = float(np.clip(S_full, 0.0, 1.0))

    return {
        "S_full": S_full,
        "S_op": S_op,
        "S_sp": S_sp,
        "S_dom": S_dom,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate labels with w_gap_strong + alpha50 config")
    ap.add_argument("--index_csv", default="dataset/woodscape_processed/meta/labels_index.csv",
                    help="Path to original labels index")
    ap.add_argument("--img_root", default="dataset/woodscape_raw",
                    help="Root dir for images")
    ap.add_argument("--labels_tile_dir", default="dataset/woodscape_processed/labels_tile",
                    help="Directory containing tile labels")
    ap.add_argument("--out_csv", default="dataset/woodscape_processed/meta/labels_index_wgap_alpha50.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    # Configuration: w_gap_strong + alpha50
    CLASS_WEIGHTS = (0.0, 0.15, 0.50, 1.0)  # w_gap_strong
    ALPHA, BETA, GAMMA = 0.5, 0.4, 0.1  # alpha50 (redistributed from 0.6, 0.3, 0.1)
    ETA_TRANS = 0.9

    print("=" * 70)
    print("Generating labels with w_gap_strong + alpha50 configuration")
    print("=" * 70)
    print(f"Class weights: {CLASS_WEIGHTS}")
    print(f"Fusion coefficients: α={ALPHA}, β={BETA}, γ={GAMMA}")
    print(f"Transparency discount: η={ETA_TRANS}")
    print("=" * 70)

    # Load original index
    df = pd.read_csv(args.index_csv)
    print(f"\nLoaded {len(df)} samples from {args.index_csv}")

    # Load tile labels
    label_dir = Path(args.labels_tile_dir)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        stem = Path(row["rel_img"]).stem
        npz_path = label_dir / f"{stem}.npz"

        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping")
            continue

        # Load tile labels
        data = np.load(npz_path)
        tile_cov = data["tile_cov"]  # (8, 8, 4)

        # Load mask to compute S_op, S_sp from scratch
        # For now, we can use the tile_cov to estimate
        # tile_cov[i,j,c] = proportion of class c in tile (i,j)

        # Reconstruct class proportions from tile_cov
        H_tiles, W_tiles, C = tile_cov.shape
        total_pixels = H_tiles * W_tiles

        # Average per-class coverage
        p = tile_cov.mean(axis=(0, 1))  # (4,)

        # For S_sp, we need the actual mask or an approximation
        # Since we don't have the mask, we'll use a weighted sum
        # This is an approximation - ideally we'd have the original mask

        # Approximate: use tile_cov to compute spatial weights
        # Create a pseudo-mask by "unfolding" tile_cov
        mask_approx = np.zeros((H_tiles * 8, W_tiles * 8), dtype=np.int64)
        for i in range(H_tiles):
            for j in range(W_tiles):
                # For each tile, assign majority class to all pixels
                majority_class = int(tile_cov[i, j].argmax())
                mask_approx[i*8:(i+1)*8, j*8:(j+1)*8] = majority_class

        # Compute severity scores
        scores = compute_severity_score(
            mask=mask_approx,
            tile_cov=tile_cov,
            class_weights=CLASS_WEIGHTS,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            eta_trans=ETA_TRANS,
            spatial_mode="gaussian",
        )

        rows.append({
            "rel_img": row["rel_img"],
            "label_npz": row["label_npz"],
            "split": row.get("split", "unknown"),
            "global_level": row.get("global_level", -1),
            "global_bin": row.get("global_bin", -1),
            # New severity score with w_gap_strong + alpha50
            "S": scores["S_full"],
            "S_full_wgap_alpha50": scores["S_full"],
            # Components for analysis
            "S_op_wgap": scores["S_op"],
            "S_sp_wgap": scores["S_sp"],
            "S_dom_wgap": scores["S_dom"],
        })

    # Create output DataFrame
    df_out = pd.DataFrame(rows)

    # Save to CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*70}")
    print(f"Saved {len(df_out)} samples to: {out_path}")
    print(f"{'='*70}")

    # Print statistics
    print(f"\nSeverity Score Statistics (w_gap_strong + alpha50):")
    print(f"  S_full: mean={df_out['S_full_wgap_alpha50'].mean():.4f}, "
          f"std={df_out['S_full_wgap_alpha50'].std():.4f}")
    print(f"  S_op:   mean={df_out['S_op_wgap'].mean():.4f}, "
          f"std={df_out['S_op_wgap'].std():.4f}")
    print(f"  S_sp:   mean={df_out['S_sp_wgap'].mean():.4f}, "
          f"std={df_out['S_sp_wgap'].std():.4f}")
    print(f"  S_dom:  mean={df_out['S_dom_wgap'].mean():.4f}, "
          f"std={df_out['S_dom_wgap'].std():.4f}")

    print(f"\nSplit distribution:")
    for split, count in df_out["split"].value_counts().items():
        print(f"  {split}: {count}")

    print(f"\nSample rows:")
    print(df_out.head(3).to_string())
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()