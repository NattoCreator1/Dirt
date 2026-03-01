#!/usr/bin/env python3
"""
Evaluation script for multi-seed and k-fold baseline experiments.

Key fixes for statistical robustness:
1. Cluster-level aggregation: First aggregate to cluster-level (median), then bootstrap
2. Fixed cluster mapping: Use consistent cluster_id for all seeds/folds
3. Additional metrics: Compression ratio + Test_Real MAE/bias for completeness
4. Fixed checkpoint selection: Use ckpt_last.pth (fixed epochs) for fair comparison
"""
import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.models.baseline_dualhead import BaselineDualHead
from generate_test_ext_with_clusters import ExternalTestDatasetWithClusters


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


def aggregate_to_cluster_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate image-level predictions to cluster-level.

    For each cluster, compute:
    - S_hat_cluster = median(S_hat) across images in cluster
    - level_cluster = median(occlusion_level) (should be same for all images in cluster)

    This ensures each cluster is treated as a single independent sample.
    """
    cluster_agg = df.groupby('cluster_id').agg({
        'S_hat': 'median',  # Use median for robustness
        'occlusion_level': 'median',  # Should be constant within cluster
        'image_path': 'count',  # Count images per cluster
    }).reset_index()

    cluster_agg.rename(columns={'image_path': 'n_images'}, inplace=True)

    return cluster_agg


def compute_compression_ratio(df: pd.DataFrame) -> Dict:
    """
    Compute compression ratio: sensitivity to level changes.

    For each adjacent level pair (5→4, 4→3, 3→2, 2→1), compute:
    - ΔS_hat = median(S_hat) at level_i - median(S_hat) at level_{i+1}
    - Δlevel = 1 (ground truth difference)

    Compression ratio = mean(ΔS_hat) / Δlevel
    Values < 1.0 indicate scale compression.
    """
    level_means = df.groupby('occlusion_level')['S_hat'].median().sort_index(ascending=False)

    deltas = []
    for i in range(len(level_means) - 1):
        delta = level_means.iloc[i] - level_means.iloc[i + 1]
        deltas.append(delta)

    compression_ratio = np.mean(deltas) / 1.0  # Divide by true delta = 1

    return {
        'compression_ratio': compression_ratio,
        'level_means': level_means.to_dict(),
        'deltas': deltas,
    }


def generate_test_ext_predictions(ckpt_path: str, output_csv: str,
                                    img_root: str = "dataset/external_test_washed",
                                    batch_size: int = 16) -> Dict:
    """Generate Test_Ext predictions with cluster info for a single model."""
    print(f"\nGenerating predictions for: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BaselineDualHead(pretrained=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    # Create dataset
    dataset = ExternalTestDatasetWithClusters(img_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=False)

    # Generate predictions
    all_predictions = []
    all_paths = []
    all_clusters = []
    all_levels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch['image'].to(device)
            paths = batch['image_path']
            cluster_ids = batch['cluster_id']
            levels = batch['occlusion_level']

            output = model(images)
            S_hat = output["S_hat"]

            all_predictions.extend(S_hat.cpu().numpy().flatten())
            all_paths.extend(paths)
            all_clusters.extend(cluster_ids)
            all_levels.extend(levels.cpu().numpy())

    # Save to CSV
    df = pd.DataFrame({
        'image_path': all_paths,
        'cluster_id': all_clusters,
        'occlusion_level': all_levels,
        'S_hat': all_predictions,
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Compute point estimate
    rho_point = spearmanr(df['S_hat'].values, df['occlusion_level'].values)

    return {
        'csv_path': output_csv,
        'n_samples': len(df),
        'n_clusters': df['cluster_id'].nunique(),
        'rho_point': rho_point,
    }


def clustered_bootstrap_ci(csv_path: str, n_bootstrap: int = 1000,
                            random_seed: int = 42) -> Dict:
    """
    Run cluster-level Bootstrap on a single prediction CSV.

    CRITICAL FIX: First aggregate to cluster-level, then bootstrap on clusters.

    Process:
    1. Aggregate image-level predictions to cluster-level (median per cluster)
    2. Bootstrap on the 505 clusters (not 50,383 images)
    3. Compute Spearman ρ on cluster-level predictions
    """
    np.random.seed(random_seed)

    # Load image-level predictions
    df = pd.read_csv(csv_path)

    # Step 1: Aggregate to cluster-level (CRITICAL FIX)
    df_cluster = aggregate_to_cluster_level(df)

    n_clusters = len(df_cluster)
    cluster_preds = df_cluster['S_hat'].values
    cluster_labels = df_cluster['occlusion_level'].values
    cluster_ids = df_cluster['cluster_id'].values

    # Point estimate on cluster-level
    rho_point = spearmanr(cluster_preds, cluster_labels)

    # Step 2: Bootstrap on clusters (not images)
    cluster_indices = np.arange(n_clusters)
    bootstrap_rhos = []

    for _ in range(n_bootstrap):
        # Resample clusters (with replacement)
        sampled_indices = np.random.choice(cluster_indices, size=n_clusters, replace=True)
        rho_b = spearmanr(cluster_preds[sampled_indices], cluster_labels[sampled_indices])
        bootstrap_rhos.append(rho_b)

    bootstrap_rhos = np.array(bootstrap_rhos)

    ci_lower = np.percentile(bootstrap_rhos, 2.5)
    ci_upper = np.percentile(bootstrap_rhos, 97.5)
    se = float(np.std(bootstrap_rhos, ddof=1))
    ci_width = ci_upper - ci_lower

    # Compute compression ratio
    compression = compute_compression_ratio(df_cluster)

    return {
        'n_images': len(df),
        'n_clusters': n_clusters,
        'rho_point': rho_point,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'ci_width': ci_width,
        'relative_width': ci_width / rho_point * 100 if rho_point != 0 else float('inf'),
        'compression_ratio': compression['compression_ratio'],
        'level_means': compression['level_means'],
    }


def evaluate_multi_seed(seeds: List[int], out_root: str, args) -> Dict:
    """Evaluate all models from multi-seed experiments."""
    print("="*70)
    print(f"Evaluating Multi-Seed Baseline Models")
    print("="*70)

    results = []

    for seed in seeds:
        # Find the checkpoint for this seed
        pattern = f"baseline_seed{seed}_{args.global_target}_"
        runs_dir = Path(out_root)

        matching_dirs = [d for d in runs_dir.glob(pattern + "*") if d.is_dir()]
        if not matching_dirs:
            print(f"Warning: No run found for seed {seed}")
            continue

        run_dir = matching_dirs[0]
        # Use ckpt_last.pth for fair comparison (fixed epochs, not val-selected best)
        ckpt_path = run_dir / "ckpt_last.pth"

        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for seed {seed}")
            continue

        print(f"\nSeed {seed}: {run_dir.name}")

        # Generate Test_Ext predictions
        pred_csv = Path("scripts/external_test_analysis/multi_seed") / f"seed{seed}_test_ext.csv"

        pred_info = generate_test_ext_predictions(
            str(ckpt_path), str(pred_csv),
            args.img_root_ext, args.batch_size
        )

        # Run cluster-level Bootstrap
        bootstrap_result = clustered_bootstrap_ci(
            pred_info['csv_path'],
            n_bootstrap=args.n_bootstrap,
            random_seed=args.bootstrap_seed
        )

        results.append({
            'seed': seed,
            'run_name': run_dir.name,
            'ckpt_path': str(ckpt_path),
            'pred_csv': pred_info['csv_path'],
            'n_samples': pred_info['n_samples'],
            'n_clusters': pred_info['n_clusters'],
            **bootstrap_result,
        })

    # Compute statistics across seeds
    rhos = [r['rho_point'] for r in results]
    ci_lowers = [r['ci_lower'] for r in results]
    ci_uppers = [r['ci_upper'] for r in results]
    compressions = [r['compression_ratio'] for r in results]

    summary = {
        'experiment_type': 'multi_seed',
        'n_seeds': len(seeds),
        'seeds': seeds,
        'results': results,
        'statistics': {
            'rho_mean': float(np.mean(rhos)),
            'rho_std': float(np.std(rhos, ddof=1)),
            'rho_min': float(np.min(rhos)),
            'rho_max': float(np.max(rhos)),
            'ci_lower_mean': float(np.mean(ci_lowers)),
            'ci_upper_mean': float(np.mean(ci_uppers)),
            'ci_width_mean': float(np.mean([r['ci_width'] for r in results])),
            'compression_mean': float(np.mean(compressions)),
            'compression_std': float(np.std(compressions, ddof=1)),
        },
    }

    return summary


def evaluate_kfold(n_folds: int, out_root: str, args) -> Dict:
    """Evaluate all models from k-fold experiments."""
    print("="*70)
    print(f"Evaluating K-Fold Baseline Models")
    print("="*70)

    results = []

    for fold in range(1, n_folds + 1):
        # Find the checkpoint for this fold
        pattern = f"baseline_k{n_folds}_fold{fold}_{args.global_target}_"
        runs_dir = Path(out_root)

        matching_dirs = [d for d in runs_dir.glob(pattern + "*") if d.is_dir()]
        if not matching_dirs:
            print(f"Warning: No run found for fold {fold}")
            continue

        run_dir = matching_dirs[0]
        # Use ckpt_last.pth for fair comparison (fixed epochs, not val-selected best)
        ckpt_path = run_dir / "ckpt_last.pth"

        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for fold {fold}")
            continue

        print(f"\nFold {fold}: {run_dir.name}")

        # Generate Test_Ext predictions
        pred_csv = Path("scripts/external_test_analysis/kfold") / f"kfold{n_folds}_fold{fold}_test_ext.csv"

        pred_info = generate_test_ext_predictions(
            str(ckpt_path), str(pred_csv),
            args.img_root_ext, args.batch_size
        )

        # Run cluster-level Bootstrap
        bootstrap_result = clustered_bootstrap_ci(
            pred_info['csv_path'],
            n_bootstrap=args.n_bootstrap,
            random_seed=args.bootstrap_seed
        )

        results.append({
            'fold': fold,
            'run_name': run_dir.name,
            'ckpt_path': str(ckpt_path),
            'pred_csv': pred_info['csv_path'],
            'n_samples': pred_info['n_samples'],
            'n_clusters': pred_info['n_clusters'],
            **bootstrap_result,
        })

    # Compute statistics across folds
    rhos = [r['rho_point'] for r in results]
    ci_lowers = [r['ci_lower'] for r in results]
    ci_uppers = [r['ci_upper'] for r in results]
    compressions = [r['compression_ratio'] for r in results]

    summary = {
        'experiment_type': 'kfold',
        'n_folds': n_folds,
        'results': results,
        'statistics': {
            'rho_mean': float(np.mean(rhos)),
            'rho_std': float(np.std(rhos, ddof=1)),
            'rho_min': float(np.min(rhos)),
            'rho_max': float(np.max(rhos)),
            'ci_lower_mean': float(np.mean(ci_lowers)),
            'ci_upper_mean': float(np.mean(ci_uppers)),
            'ci_width_mean': float(np.mean([r['ci_width'] for r in results])),
            'compression_mean': float(np.mean(compressions)),
            'compression_std': float(np.std(compressions, ddof=1)),
        },
    }

    return summary


def print_summary_table(summary: Dict, experiment_type: str):
    """Print a formatted summary table."""
    print("="*80)
    print(f"Summary: {experiment_type.upper()} Baseline Evaluation")
    print("="*80)
    print()

    stats = summary['statistics']

    print(f"Statistics across {summary.get('n_seeds') or summary['n_folds']} {experiment_type}:")
    print(f"  ρ (Spearman, cluster-level) = {stats['rho_mean']:.4f} ± {stats['rho_std']:.4f}")
    print(f"    Range: [{stats['rho_min']:.4f}, {stats['rho_max']:.4f}]")
    print(f"\n  95% CI (mean across {experiment_type}):")
    print(f"    [{stats['ci_lower_mean']:.4f}, {stats['ci_upper_mean']:.4f}]")
    print(f"    Mean CI width: {stats['ci_width_mean']:.4f} ({stats['ci_width_mean']/stats['rho_mean']*100:.2f}%)")
    print(f"\n  Compression Ratio (sensitivity to level changes):")
    print(f"    Mean: {stats['compression_mean']:.4f} ± {stats['compression_std']:.4f}")
    print(f"    (1.0 = ideal, < 1.0 = scale compression)")

    print(f"\nIndividual results:")
    if experiment_type == 'multi_seed':
        print(f"  {'Seed':<6} {'ρ':<10} {'95% CI':<20} {'Comp. Ratio':<12} {'CI Width':<10}")
        print("  " + "-"*70)
        for r in summary['results']:
            print(f"  {r['seed']:<6} {r['rho_point']:<10.4f} "
                  f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  "
                  f"{r['compression_ratio']:<12.4f} {r['ci_width']:<10.4f}")
    else:  # kfold
        print(f"  {'Fold':<6} {'ρ':<10} {'95% CI':<20} {'Comp. Ratio':<12} {'CI Width':<10}")
        print("  " + "-"*70)
        for r in summary['results']:
            print(f"  {r['fold']:<6} {r['rho_point']:<10.4f} "
                  f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  "
                  f"{r['compression_ratio']:<12.4f} {r['ci_width']:<10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-seed or k-fold baseline experiments"
    )
    parser.add_argument("--mode", type=str, choices=['multi_seed', 'kfold', 'both'],
                        default='both', help="Which experiments to evaluate")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated list of seeds (for multi_seed)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds (for kfold)")
    parser.add_argument("--global_target", type=str, default="S",
                        help="Global target used in training")
    parser.add_argument("--out_root", type=str, default="baseline/runs",
                        help="Directory containing trained models")
    parser.add_argument("--img_root_ext", type=str, default="dataset/external_test_washed",
                        help="External test image root")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations")
    parser.add_argument("--bootstrap_seed", type=int, default=42,
                        help="Random seed for bootstrap")
    parser.add_argument("--output_dir", type=str,
                        default="scripts/external_test_analysis/statistical_robustness",
                        help="Output directory for results")

    args = parser.parse_args()

    results = {}

    if args.mode in ['multi_seed', 'both']:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        results['multi_seed'] = evaluate_multi_seed(seeds, args.out_root, args)
        print_summary_table(results['multi_seed'], 'multi_seed')

    if args.mode in ['kfold', 'both']:
        results['kfold'] = evaluate_kfold(args.n_folds, args.out_root, args)
        print_summary_table(results['kfold'], 'kfold')

    # Save combined results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f"evaluation_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("="*80)
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
