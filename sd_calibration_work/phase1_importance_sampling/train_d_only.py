#!/usr/bin/env python3
"""
Phase 1: Train Mixed(8960, D-only) with Importance Sampling

Purpose:
- Train mixed WoodScape + SD model using binned importance sampling
- Use two-layer sampler to maintain real:sd ratio while applying weights
- Train for fixed 10,000 steps for fair comparison

Usage:
    python train_d_only.py \
        --weights weights.npz \
        --run_name mixed_ws4000_sd8960_d_only
"""

import sys
import os
import argparse
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Import project modules
from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.datasets.woodscape_tile_dataset import WoodscapeTileDataset
from baseline.utils.metrics import compute_metrics
from sd_calibration_work.phase1_importance_sampling.two_layer_sampler import create_two_layer_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mixed model with D-only calibration")

    # Data paths
    parser.add_argument('--ws_csv', type=str,
                        default='dataset/woodscape_processed/meta/labels_index_ablation.csv',
                        help='WoodScape index CSV')
    parser.add_argument('--sd_csv', type=str,
                        default='sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv',
                        help='SD index CSV')
    parser.add_argument('--img_root', type=str, default='.',
                        help='Image root directory')
    parser.add_argument('--weights', type=str, required=True,
                        help='Importance weights file (NPZ)')

    # Training parameters
    parser.add_argument('--real_ratio', type=float, default=0.76,
                        help='Target real:total ratio (4000/5260 ≈ 0.76)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--total_steps', type=int, default=10000,
                        help='Total training steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--lambda_glob', type=float, default=1.0,
                        help='Global loss weight')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--run_name', type=str, required=True,
                        help='Run name for output directory')
    parser.add_argument('--out_root', type=str, default='baseline/runs',
                        help='Output root directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval (steps)')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='Save interval (steps)')

    return parser.parse_args()


def create_mixed_dataset(args):
    """
    Create mixed dataset by concatenating WoodScape and SD data.

    Returns:
        dataset, n_real, n_sd
    """
    print("=" * 80)
    print("Creating Mixed Dataset")
    print("=" * 80)
    print()

    # Load WoodScape
    print("【1. Load WoodScape】")
    ws_df = pd.read_csv(args.ws_csv)
    ws_train = ws_df[ws_df['split'] == 'train'].copy()
    print(f"WoodScape Train: {len(ws_train)}")

    # Load SD
    print("【2. Load SD】")
    sd_df = pd.read_csv(args.sd_csv)
    print(f"SD: {len(sd_df)}")

    # Select SD subset (8960 or all)
    # For now, use all SD samples
    print(f"Using all {len(sd_df)} SD samples")

    # Combine into single dataframe
    # Add source column
    ws_train['source'] = 'real'
    sd_df['source'] = 'sd'

    # Concatenate
    mixed_df = pd.concat([ws_train, sd_df], ignore_index=True)
    n_real = len(ws_train)
    n_sd = len(sd_df)

    print()
    print(f"【3. Mixed Dataset】")
    print(f"Total samples: {len(mixed_df)}")
    print(f"  Real: {n_real} ({n_real/len(mixed_df)*100:.1f}%)")
    print(f"  SD: {n_sd} ({n_sd/len(mixed_df)*100:.1f}%)")
    print(f"  Target real ratio: {args.real_ratio:.2f}")
    print()

    return mixed_df, n_real, n_sd


def train_one_step(model, batch, criterion_tile, criterion_glob, args, device):
    """Train one batch."""
    images = batch['image'].to(device)
    G_gt = batch['tile_cov'].to(device)  # [B, 8, 8, 4]
    S_gt = batch['S'].to(device)  # [B, 1] or [B]

    # Forward
    with torch.cuda.amp.autocast(enabled=args.amp):
        G_hat, S_hat = model(images)

        # Tile loss (all samples)
        L_tile = criterion_tile(G_hat, G_gt)

        # Global loss (all samples)
        # For D-only, we use S_gt directly (no teacher)
        # The importance sampling is handled by the sampler
        S_gt = S_gt.unsqueeze(1) if S_gt.dim() == 1 else S_gt
        L_glob = criterion_glob(S_hat, S_gt)

        # Total loss
        L_total = L_tile + args.lambda_glob * L_glob

    return L_total, L_tile, L_glob


def evaluate(model, dataloader, device, args):
    """Evaluate model."""
    model.eval()

    all_S_hat = []
    all_S_gt = []
    all_G_hat = []
    all_G_gt = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            G_gt = batch['tile_cov'].to(device)
            S_gt = batch['S'].to(device)

            G_hat, S_hat = model(images)

            all_S_hat.append(S_hat.cpu())
            all_S_gt.append(S_gt.cpu())
            all_G_hat.append(G_hat.cpu())
            all_G_gt.append(G_gt.cpu())

    all_S_hat = torch.cat(all_S_hat).numpy()
    all_S_gt = torch.cat(all_S_gt).numpy()
    all_G_hat = torch.cat(all_G_hat).numpy()
    all_G_gt = torch.cat(all_G_gt).numpy()

    metrics = compute_metrics(all_S_hat, all_S_gt, all_G_hat, all_G_gt)

    return metrics


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.out_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Phase 1: Train Mixed(8960, D-only) with Importance Sampling")
    print("=" * 80)
    print()
    print(f"Run name: {args.run_name}")
    print(f"Output dir: {output_dir}")
    print(f"Total steps: {args.total_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Real ratio: {args.real_ratio}")
    print()

    # Create dataset
    mixed_df, n_real, n_sd = create_mixed_dataset(args)

    # Create dataset instance
    print("【4. Create Dataset Instance】")
    dataset = WoodscapeTileDataset(
        index_csv=None,  # We'll pass df directly
        img_root=args.img_root,
        split_col=None,
        train_split=None,
        val_split=None,
        dataframe=mixed_df,
        global_target='S'
    )
    print(f"Dataset size: {len(dataset)}")
    print()

    # Load weights
    print("【5. Load Importance Weights】")
    weights_data = np.load(args.weights)
    sd_weights = weights_data['weights']
    print(f"SD weights shape: {sd_weights.shape}")
    print(f"SD weights range: [{sd_weights.min():.4f}, {sd_weights.max():.4f}]")
    print()

    # Create dataloader with two-layer sampling
    print("【6. Create DataLoader with Two-Layer Sampling】")
    train_loader = create_two_layer_dataloader(
        dataset=dataset,
        n_real=n_real,
        n_sd=n_sd,
        sd_weights=sd_weights,
        real_ratio=args.real_ratio,
        batch_size=args.batch_size,
        num_workers=4,
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train loader created")
    print()

    # Create model
    print("【7. Create Model】")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = BaselineDualHead(
        backbone='resnet18',
        pretrained=True,
        backbone_freeze_epochs=0
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_tile = nn.KLDivLoss(reduction='batchmean')
    criterion_glob = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Training loop
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print()

    model.train()
    step = 0
    best_val_loss = float('inf')

    metrics_history = []

    pbar = tqdm(total=args.total_steps, desc="Training")

    while step < args.total_steps:
        for batch in train_loader:
            if step >= args.total_steps:
                break

            # Train
            optimizer.zero_grad()

            loss, loss_tile, loss_glob = train_one_step(
                model, batch, criterion_tile, criterion_glob, args, device
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log
            if step % args.log_interval == 0:
                metric = {
                    'step': step,
                    'loss': float(loss),
                    'loss_tile': float(loss_tile),
                    'loss_glob': float(loss_glob),
                }
                metrics_history.append(metric)

                tqdm.write(f"Step {step}: loss={loss:.4f}, tile={loss_tile:.4f}, glob={loss_glob:.4f}")

            # Save
            if (step + 1) % args.save_interval == 0 or step == 0:
                ckpt_path = output_dir / f'ckpt_step_{step}.pth'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': float(loss),
                }, ckpt_path)
                tqdm.write(f"Saved checkpoint: {ckpt_path}")

            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

    pbar.close()

    # Save final checkpoint
    final_ckpt_path = output_dir / 'ckpt_last.pth'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(loss),
    }, final_ckpt_path)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print()
    print(f"Final step: {step}")
    print(f"Output dir: {output_dir}")
    print()
    print("Next steps:")
    print("1. Evaluate on Test_Real:")
    print(f"   python baseline/eval.py --ckpt {final_ckpt_path} --split test")
    print("2. Evaluate on Test_Ext:")
    print(f"   python baseline/eval_ext.py --ckpt {final_ckpt_path}")


if __name__ == "__main__":
    main()
