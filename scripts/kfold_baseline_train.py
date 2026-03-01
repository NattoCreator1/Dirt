#!/usr/bin/env python3
"""
K-fold cross-validation for baseline model.

Performs k-fold cross-validation on the combined train+val set (4000 samples)
to assess the stability of baseline performance given the small dataset size.
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
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def create_kfold_splits(index_csv: str, n_folds: int, random_seed: int,
                        output_dir: str) -> List[Dict]:
    """
    Create k-fold splits from the index CSV.

    Combines train+val splits and creates k-fold partitions.
    Test set is kept separate and not used in CV.
    """
    print(f"\nCreating {n_folds}-fold splits...")

    df = pd.read_csv(index_csv)

    # Filter to train+val only (exclude test)
    df_trainval = df[df['split'].isin(['train', 'val'])].copy()
    df_test = df[df['split'] == 'test'].copy()

    print(f"  Train+Val samples: {len(df_trainval)}")
    print(f"  Test samples: {len(df_test)} (kept separate)")

    # Get unique filenames for splitting (to avoid splitting same image from different views)
    # Extract base filename (without view suffix)
    df_trainval['base_name'] = df_trainval['rgb_path'].apply(
        lambda x: Path(x).stem.rsplit('_', 1)[0] if '_' in Path(x).stem else Path(x).stem
    )

    # Get unique base names
    unique_names = df_trainval['base_name'].unique()
    print(f"  Unique base names: {len(unique_names)}")

    # Create KFold split on unique names
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_info = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_names), 1):
        train_names = set(unique_names[train_idx])
        val_names = set(unique_names[val_idx])

        # Assign splits based on base names
        df_trainval['kfold_split'] = df_trainval['base_name'].map(
            lambda x: f'kfold{fold_idx}_val' if x in val_names else f'kfold{fold_idx}_train'
        )

        # Combine with test set
        df_combined = pd.concat([
            df_trainval,
            df_test.assign(kfold_split='test')
        ], ignore_index=True)

        # Save fold-specific index CSV
        fold_csv_path = output_path / f"labels_index_kfold{n_folds}_fold{fold_idx}.csv"
        df_combined.to_csv(fold_csv_path, index=False)

        # Count samples
        n_train = (df_combined['kfold_split'] == f'kfold{fold_idx}_train').sum()
        n_val = (df_combined['kfold_split'] == f'kfold{fold_idx}_val').sum()
        n_test = (df_combined['kfold_split'] == 'test').sum()

        fold_info.append({
            'fold': fold_idx,
            'csv_path': str(fold_csv_path),
            'n_train': int(n_train),
            'n_val': int(n_val),
            'n_test': int(n_test),
        })

        print(f"  Fold {fold_idx}: train={n_train}, val={n_val}, test={n_test}")

    # Save fold summary
    summary_path = output_path / f"kfold{n_folds}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'n_folds': n_folds,
            'random_seed': random_seed,
            'total_trainval': len(df_trainval),
            'total_test': len(df_test),
            'folds': fold_info,
        }, f, indent=2)

    print(f"\nK-fold splits created in: {output_dir}")
    print(f"Summary saved to: {summary_path}")

    return fold_info


def train_single_fold(fold_info: Dict, args: argparse.Namespace) -> Dict:
    """Train a single baseline model for one fold."""
    fold = fold_info['fold']
    fold_csv = fold_info['csv_path']

    print(f"\n{'='*70}")
    print(f"Training Fold {fold}/{args.n_folds}")
    print(f"{'='*70}")
    print(f"  Train: {fold_info['n_train']}, Val: {fold_info['n_val']}, Test: {fold_info['n_test']}")
    print(f"  CSV: {fold_csv}\n")

    run_name = f"baseline_k{args.n_folds}_fold{fold}_{args.global_target}_{time.strftime('%Y%m%d_%H%M%S')}"

    cmd = [
        "python", "-u", "baseline/train_baseline.py",
        "--index_csv", fold_csv,
        "--img_root", args.img_root,
        "--labels_tile_dir", args.labels_tile_dir,
        "--split_col", "kfold_split",
        "--train_split", f"kfold{fold}_train",
        "--val_split", f"kfold{fold}_val",
        "--test_split", "test",
        "--global_target", args.global_target,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda_glob", str(args.lambda_glob),
        "--mu_cons", str(args.mu_cons),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--run_name", run_name,
        "--out_root", args.out_root,
    ]

    if args.amp:
        cmd.append("--amp")

    # Set up environment for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/yf/soiling_project'

    start_time = time.time()
    result = subprocess.run(cmd, check=True, cwd='/home/yf/soiling_project', env=env)
    elapsed = time.time() - start_time

    print(f"Fold {fold} training completed in {elapsed/3600:.2f} hours")

    ckpt_path = Path(args.out_root) / run_name / "ckpt_best.pth"
    return {
        'fold': fold,
        'run_name': run_name,
        'ckpt_path': str(ckpt_path),
        'n_train': fold_info['n_train'],
        'n_val': fold_info['n_val'],
        'n_test': fold_info['n_test'],
        'training_time_hours': elapsed / 3600,
    }


def main():
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation for baseline model"
    )
    # Data arguments
    parser.add_argument("--index_csv", type=str,
                        default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv",
                        help="Path to original index CSV")
    parser.add_argument("--img_root", type=str, default=".",
                        help="Image root directory (use '.' since CSV paths are already relative)")
    parser.add_argument("--labels_tile_dir", type=str, default="dataset/woodscape_processed/labels_tile")
    parser.add_argument("--kfold_output_dir", type=str,
                        default="dataset/woodscape_processed/meta/kfold_splits",
                        help="Directory to save k-fold split CSVs")

    # Model arguments
    parser.add_argument("--global_target", type=str, default="S",
                        choices=["S", "s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00", "S_full_wgap_alpha50"])

    # Training arguments
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda_glob", type=float, default=1.0)
    parser.add_argument("--mu_cons", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out_root", type=str, default="baseline/runs")

    # K-fold arguments
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for k-fold splitting")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation for Baseline Model")
    print(f"{'='*70}")
    print(f"  K = {args.n_folds}")
    print(f"  Split random seed = {args.seed}")
    print(f"  Epochs per fold = {args.epochs}")
    print(f"  Estimated total time: {args.n_folds * args.epochs * 2 / 60:.0f} minutes\n")

    # Step 1: Create k-fold splits
    fold_info_list = create_kfold_splits(
        args.index_csv, args.n_folds, args.seed, args.kfold_output_dir
    )

    # Step 2: Train each fold
    results = []
    for fold_info in fold_info_list:
        result = train_single_fold(fold_info, args)
        results.append(result)

    # Save summary
    summary = {
        'n_folds': args.n_folds,
        'split_seed': args.seed,
        'args': vars(args),
        'folds': results,
    }

    summary_path = Path(args.out_root) / f"kfold{args.n_folds}_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"K-fold cross-validation completed!")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTrained models:")
    for r in results:
        print(f"  Fold {r['fold']}: {r['run_name']}")
        print(f"    Checkpoint: {r['ckpt_path']}")

    print(f"\nNext step: Run evaluation on Test_Real and Test_Ext")
    print(f"  python scripts/kfold_evaluate.py \\")
    print(f"    --n_folds {args.n_folds} \\")
    print(f"    --out_root {args.out_root}")


if __name__ == "__main__":
    main()
