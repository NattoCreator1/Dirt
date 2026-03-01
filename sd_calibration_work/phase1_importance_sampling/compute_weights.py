#!/usr/bin/env python3
"""
Compute Importance Weights for SD Data

Purpose:
1. Load diagnostic data (distribution comparison)
2. Compute histogram-based weights for each SD sample
3. Save weights and metadata for training

Usage:
    python compute_weights.py \
        --sd_csv sd_scripts/lora/baseline_filtered_8960/sd_train_1260_conservative.csv \
        --output sd_calibration_work/phase1_importance_sampling/weights.npz
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')


def compute_bin_weights(
    woodscape_hist: np.ndarray,
    sd_hist: np.ndarray,
    min_weight: float = 0.1,
    max_weight: float = 50.0
) -> np.ndarray:
    """
    Compute importance weights for each bin.

    weight[bin] = woodscape_hist[bin] / sd_hist[bin]

    Args:
        woodscape_hist: Histogram counts for WoodScape
        sd_hist: Histogram counts for SD
        min_weight: Minimum weight (clipped)
        max_weight: Maximum weight (clipped)

    Returns:
        Array of weights for each bin
    """
    n_bins = len(woodscape_hist)
    weights = np.zeros(n_bins)

    for i in range(n_bins):
        if sd_hist[i] > 0:
            weights[i] = woodscape_hist[i] / sd_hist[i]
        else:
            # If SD has no samples in this bin, use max weight
            weights[i] = max_weight

    # Clip weights to reasonable range
    weights = np.clip(weights, min_weight, max_weight)

    return weights


def load_S_from_npz(npz_path: str) -> float:
    """Load S value from NPZ file."""
    try:
        data = np.load(npz_path)
        if 'S_full_wgap_alpha50' in data:
            return float(data['S_full_wgap_alpha50'])
        elif 'S_full' in data:
            return float(data['S_full'])
        elif 'global_score' in data:
            return float(data['global_score'])
        else:
            return 0.5  # Default
    except Exception as e:
        print(f"Warning: Failed to load {npz_path}: {e}")
        return 0.5


def assign_weights_to_samples(
    sd_df: pd.DataFrame,
    bin_edges: np.ndarray,
    bin_weights: np.ndarray
) -> np.ndarray:
    """
    Assign weight to each SD sample based on its S value.

    Args:
        sd_df: SD dataframe with npz_path column
        bin_edges: Bin edge values
        bin_weights: Weight for each bin

    Returns:
        Array of weights (same length as sd_df)
    """
    n_samples = len(sd_df)
    weights = np.ones(n_samples)
    s_values = np.zeros(n_samples)

    print("Loading S values from NPZ files...")
    for idx, row in sd_df.iterrows():
        s_val = load_S_from_npz(row['npz_path'])
        s_values[idx] = s_val

        # Find bin
        bin_idx = np.digitize(s_val, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_weights) - 1)
        weights[idx] = bin_weights[bin_idx]

    print(f"Loaded {n_samples} S values")
    print(f"  S range: [{s_values.min():.4f}, {s_values.max():.4f}]")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    return weights, s_values


def main():
    parser = argparse.ArgumentParser(description="Compute importance weights for SD data")
    parser.add_argument('--sd_csv', type=str, required=True, help='SD training CSV')
    parser.add_argument('--diagnostic_json', type=str,
                        default='sd_calibration_work/phase0_diagnostic/diagnostic_report_8960.json',
                        help='Diagnostic report JSON (use 8960 version for full SD dataset)')
    parser.add_argument('--output', type=str,
                        default='sd_calibration_work/phase1_importance_sampling/weights.npz',
                        help='Output weights file')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins')
    parser.add_argument('--min_weight', type=float, default=0.1, help='Minimum weight')
    parser.add_argument('--max_weight', type=float, default=50.0, help='Maximum weight')
    parser.add_argument('--real_ratio', type=float, default=0.8, help='Real:SD ratio in training')

    args = parser.parse_args()

    print("=" * 80)
    print("Compute Importance Weights for SD Data")
    print("=" * 80)
    print()

    # Load diagnostic data
    print("【1. Load diagnostic data】")
    print(f"Diagnostic file: {args.diagnostic_json}")
    with open(args.diagnostic_json, 'r') as f:
        diagnostic = json.load(f)

    bins = np.array(diagnostic['bins'])
    woodscape_hist = np.array(diagnostic['woodscape_hist'])

    # Use sd_8960_hist if available, otherwise sd_1260_hist
    if 'sd_8960_hist' in diagnostic:
        sd_hist = np.array(diagnostic['sd_8960_hist'])
        print(f"Using SD 8960 histogram")
    elif 'sd_1260_hist' in diagnostic:
        sd_hist = np.array(diagnostic['sd_1260_hist'])
        print(f"Using SD 1260 histogram")
    else:
        raise ValueError("No SD histogram found in diagnostic file")

    print(f"WoodScape samples: {woodscape_hist.sum()}")
    print(f"SD samples: {sd_hist.sum()}")
    print()

    # Compute bin weights
    print("【2. Compute bin weights】")
    bin_weights = compute_bin_weights(
        woodscape_hist,
        sd_hist,
        min_weight=args.min_weight,
        max_weight=args.max_weight
    )

    print("Bin weights:")
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        ws_count = woodscape_hist[i]
        sd_count = sd_hist[i]
        raw_weight = ws_count / sd_count if sd_count > 0 else args.max_weight
        final_weight = bin_weights[i]
        print(f"  [{low:.1f}, {high:.1f}): "
              f"WS={ws_count}, SD={sd_count}, "
              f"raw={raw_weight:.2f}, final={final_weight:.2f}")
    print()

    # Load SD data
    print("【3. Load SD data】")
    print(f"SD CSV: {args.sd_csv}")
    sd_df = pd.read_csv(args.sd_csv)
    print(f"SD samples: {len(sd_df)}")
    print()

    # Assign weights to samples
    print("【4. Assign weights to samples】")
    weights, s_values = assign_weights_to_samples(sd_df, bins, bin_weights)
    print()

    # Summary statistics
    print("【5. Weight summary】")
    print(f"Total samples: {len(weights)}")
    print(f"Weight mean: {weights.mean():.4f}")
    print(f"Weight std: {weights.std():.4f}")
    print(f"Weight min: {weights.min():.4f}")
    print(f"Weight max: {weights.max():.4f}")
    print()

    # Weight distribution
    weight_bins = [0, 0.5, 1, 2, 5, 10, 20, 50, np.inf]
    weight_hist, _ = np.histogram(weights, bins=weight_bins)
    print("Weight distribution:")
    for i in range(len(weight_bins) - 1):
        low, high = weight_bins[i], weight_bins[i+1]
        count = weight_hist[i]
        pct = count / len(weights) * 100
        if high == np.inf:
            print(f"  [{low:.1f}, inf): {count} ({pct:.1f}%)")
        else:
            print(f"  [{low:.1f}, {high:.1f}): {count} ({pct:.1f}%)")
    print()

    # Save weights
    print("【6. Save weights】")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        weights=weights,
        s_values=s_values,
        bin_edges=bins,
        bin_weights=bin_weights,
        woodscape_hist=woodscape_hist,
        sd_hist=sd_hist,
        real_ratio=args.real_ratio
    )

    # Also save metadata
    metadata = {
        'n_samples': int(len(weights)),
        'weight_mean': float(weights.mean()),
        'weight_std': float(weights.std()),
        'weight_min': float(weights.min()),
        'weight_max': float(weights.max()),
        's_mean': float(s_values.mean()),
        's_std': float(s_values.std()),
        'n_bins': args.n_bins,
        'real_ratio': args.real_ratio,
        'min_weight': args.min_weight,
        'max_weight': args.max_weight,
    }

    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Weights saved: {output_path}")
    print(f"Metadata saved: {metadata_path}")
    print()

    print("=" * 80)
    print("Weight computation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review weight distribution")
    print("2. Run training with weights:")
    print(f"   python train_d_only.py --weights {output_path}")


if __name__ == "__main__":
    main()
