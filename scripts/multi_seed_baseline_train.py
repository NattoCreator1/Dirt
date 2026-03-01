#!/usr/bin/env python3
"""
Multi-seed baseline training for statistical robustness.

Trains the baseline model with multiple random seeds to assess
the stability of performance metrics given the small WoodScape dataset.
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


def train_single_seed(seed: int, args: argparse.Namespace) -> Dict:
    """Train a single baseline model with given seed."""
    print(f"\n{'='*70}")
    print(f"Training Baseline with seed {seed}")
    print(f"{'='*70}\n")

    run_name = f"baseline_seed{seed}_{args.global_target}_{time.strftime('%Y%m%d_%H%M%S')}"

    cmd = [
        "python", "-u", "baseline/train_baseline.py",
        "--index_csv", args.index_csv,
        "--img_root", args.img_root,
        "--labels_tile_dir", args.labels_tile_dir,
        "--split_col", args.split_col,
        "--train_split", args.train_split,
        "--val_split", args.val_split,
        "--test_split", args.test_split if args.test_split else "",
        "--global_target", args.global_target,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda_glob", str(args.lambda_glob),
        "--mu_cons", str(args.mu_cons),
        "--seed", str(seed),
        "--num_workers", str(args.num_workers),
        "--run_name", run_name,
        "--out_root", args.out_root,
    ]

    if args.amp:
        cmd.append("--amp")

    print(f"Command: {' '.join(cmd)}\n")

    # Set up environment for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/yf/soiling_project'

    start_time = time.time()
    result = subprocess.run(cmd, check=True, cwd='/home/yf/soiling_project', env=env)
    elapsed = time.time() - start_time

    print(f"Seed {seed} training completed in {elapsed/3600:.2f} hours")

    # Return info about the trained model
    ckpt_path = Path(args.out_root) / run_name / "ckpt_best.pth"
    return {
        'seed': seed,
        'run_name': run_name,
        'ckpt_path': str(ckpt_path),
        'training_time_hours': elapsed / 3600,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed baseline training for statistical robustness"
    )
    # Data arguments
    parser.add_argument("--index_csv", type=str,
                        default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv",
                        help="Path to index CSV")
    parser.add_argument("--img_root", type=str, default=".",
                        help="Image root directory (use '.' since CSV paths are already relative)")
    parser.add_argument("--labels_tile_dir", type=str, default="dataset/woodscape_processed/labels_tile")
    parser.add_argument("--split_col", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default=None)

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

    # Multi-seed arguments
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated list of random seeds")

    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"\nWill train with {len(seeds)} seeds: {seeds}")
    print(f"Total estimated time: {len(seeds) * args.epochs * 2 / 60:.0f} minutes\n")

    # Train with each seed
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Starting seed {seed}")
        result = train_single_seed(seed, args)
        results.append(result)

    # Save summary
    summary = {
        'n_seeds': len(seeds),
        'seeds': seeds,
        'args': vars(args),
        'runs': results,
    }

    summary_path = Path(args.out_root) / f"multi_seed_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Multi-seed training completed!")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTrained models:")
    for r in results:
        print(f"  Seed {r['seed']}: {r['run_name']}")
        print(f"    Checkpoint: {r['ckpt_path']}")

    print(f"\nNext step: Run evaluation on Test_Real and Test_Ext")
    print(f"  python scripts/multi_seed_evaluate.py \\")
    print(f"    --seeds {args.seeds} \\")
    print(f"    --out_root {args.out_root}")


if __name__ == "__main__":
    main()
