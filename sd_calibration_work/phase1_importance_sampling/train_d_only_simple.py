#!/usr/bin/env python3
"""
Phase 1: Train Mixed(8960, D-only) with Importance Sampling

Simplified version using existing baseline training infrastructure.
"""

import sys
import os
import argparse
import random
import csv
import json
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.losses import criterion


class MixedWoodscapeDataset(WoodscapeTileDataset):
    """Custom dataset that properly handles SD data with missing S values in CSV."""

    def __getitem__(self, idx: int):
        result = super().__getitem__(idx)

        # Check if global_score is NaN (happens for SD data when CSV has NaN)
        import numpy as np
        gs = result['global_score']

        # Handle both tensor and scalar cases
        needs_fixing = False
        if isinstance(gs, torch.Tensor):
            needs_fixing = torch.isnan(gs).any()
        elif isinstance(gs, float):
            needs_fixing = np.isnan(gs)

        if needs_fixing:
            # Manually load from NPZ
            row = self.df.iloc[idx]
            npz_path = row[self.npz_col] if self.npz_col else None
            if npz_path and str(npz_path) != 'nan':
                npz_data = np.load(npz_path)
                # Try to load S_full_wgap_alpha50 first, then fallback
                if 'S_full_wgap_alpha50' in npz_data:
                    s_val = float(npz_data['S_full_wgap_alpha50'])
                elif 'global_score' in npz_data:
                    s_val = float(npz_data['global_score'])
                else:
                    s_val = 0.5
                # Store as tensor to match original format
                result['global_score'] = torch.tensor(s_val, dtype=torch.float32)

        return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spearmanr_np(a, b):
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
    model.eval()
    sum_tile = 0.0
    sum_glob = 0.0
    sum_mse = 0.0
    n = 0
    all_shat = []
    all_gt = []

    for batch in loader:
        images = batch['image'].to(device)
        tile_gt = batch['tile_cov'].to(device)
        s_gt = batch['S'].to(device)

        tile_hat, s_hat = model(images)

        # Tile loss
        tile_hat_log = torch.log(tile_hat + 1e-10)
        sum_tile += criterion['tile'](tile_hat_log, tile_gt).item() * images.size(0)

        # Global loss
        s_gt = s_gt.unsqueeze(1) if s_gt.dim() == 1 else s_gt
        s_hat = s_hat.unsqueeze(1) if s_hat.dim() == 1 else s_hat
        diff = s_hat - s_gt
        sum_glob += torch.abs(diff).sum().item()
        sum_mse += (diff ** 2).sum().item()

        n += images.size(0)

        all_shat.append(s_hat.cpu().numpy())
        all_gt.append(s_gt.cpu().numpy())

    metrics = {
        'tile_mae': sum_tile / n,
        'glob_mae': sum_glob / n,
        'glob_rmse': np.sqrt(sum_mse / n),
    }

    all_shat = np.concatenate(all_shat).reshape(-1)
    all_gt = np.concatenate(all_gt).reshape(-1)
    metrics['spearman'] = spearmanr_np(all_shat, all_gt)

    return metrics


class TwoLayerImportanceSampler(Sampler):
    """Two-layer sampler: maintain real:sd ratio + weighted SD sampling."""

    def __init__(self, n_real, n_sd, sd_weights, real_ratio=0.76,
                 batch_size=32, generator=None):
        self.n_real = n_real
        self.n_sd = n_sd
        self.n_total = n_real + n_sd
        self.real_ratio = real_ratio
        self.batch_size = batch_size
        self.generator = generator
        self.sd_weights = sd_weights

        # Pre-generate batches
        self.batches = self._generate_batches(10000)

    def _generate_batches(self, n_batches):
        batches = []
        n_real_batch = int(self.batch_size * self.real_ratio)
        n_sd_batch = self.batch_size - n_real_batch

        # Create SD weighted sampler (iterable)
        sd_sampler_iter = iter(WeightedRandomSampler(
            weights=self.sd_weights,
            num_samples=self.n_sd,
            replacement=True,
            generator=self.generator
        ))

        for _ in range(n_batches):
            # Sample real
            real_indices = torch.randint(
                0, self.n_real, (n_real_batch,), generator=self.generator
            ).tolist()

            # Sample SD with weights
            sd_indices = []
            for _ in range(n_sd_batch):
                try:
                    sd_idx = next(sd_sampler_iter)
                except StopIteration:
                    # Recreate iterator if exhausted
                    sd_sampler_iter = iter(WeightedRandomSampler(
                        weights=self.sd_weights,
                        num_samples=self.n_sd,
                        replacement=True,
                        generator=self.generator
                    ))
                    sd_idx = next(sd_sampler_iter)
                sd_indices.append(self.n_real + sd_idx)

            # Combine and shuffle
            batch = real_indices + sd_indices
            perm = torch.randperm(len(batch), generator=self.generator).tolist()
            batch = [batch[i] for i in perm]
            batches.append(batch)

        return batches

    def __iter__(self):
        while True:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_csv', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img_root', type=str, default='.')
    parser.add_argument('--real_ratio', type=float, default=0.76)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--out_root', type=str, default='baseline/runs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lambda_glob', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.out_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Phase 1: Train Mixed(8960, D-only) with Importance Sampling")
    print("=" * 80)
    print(f"Run: {args.run_name}")
    print(f"Total steps: {args.total_steps}")
    print()

    # Load mixed CSV
    print("【1. Load mixed dataset】")
    mixed_df = pd.read_csv(args.mixed_csv)
    n_real = (mixed_df['source'] == 'woodscape').sum()
    n_sd = (mixed_df['source'] == 'sd').sum()
    print(f"Total: {len(mixed_df)} (Real: {n_real}, SD: {n_sd})")
    print()

    # Load weights
    print("【2. Load SD weights】")
    weights_data = np.load(args.weights)
    sd_weights = torch.from_numpy(weights_data['weights']).float()
    print(f"SD weights: n={len(sd_weights)}, mean={sd_weights.mean():.4f}")
    print()

    # Create dataset spec
    print("【3. Create dataset】")
    spec = WoodscapeIndexSpec(
        index_csv=args.mixed_csv,
        img_root=args.img_root,
        split_col='split',
        split_value='train',
        global_target='S_full_wgap_alpha50'
    )

    # Use custom dataset class that handles SD data properly
    dataset = MixedWoodscapeDataset(spec=spec)
    print(f"Dataset size: {len(dataset)}")
    print()

    # Create sampler and dataloader
    print("【4. Create sampler and dataloader】")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = TwoLayerImportanceSampler(
        n_real=int(n_real),
        n_sd=int(n_sd),
        sd_weights=sd_weights,
        real_ratio=args.real_ratio,
        batch_size=args.batch_size,
        generator=generator
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    print("DataLoader created")
    print()

    # Create model
    print("【5. Create model】")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = BaselineDualHead(pretrained=True).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print()

    model.train()
    step = 0
    metrics_history = []
    best_val_loss = float('inf')

    pbar = tqdm(total=args.total_steps, desc="Training")

    for batch_idx in range(100000):
        for batch in train_loader:
            if step >= args.total_steps:
                break

            # Debug: check batch keys on first iteration
            if step == 0:
                print(f"Batch keys: {batch.keys()}")

            images = batch['image'].to(device)
            tile_gt = batch['tile_cov'].to(device)

            # Get S value - dataset returns it as 'global_score'
            s_gt = batch['global_score'].to(device)

            # Forward - model returns dict
            outputs = model(images)

            # Prepare targets dict for criterion
            s_gt = s_gt.unsqueeze(1) if s_gt.dim() == 1 else s_gt
            targets = {
                'tile_cov': tile_gt,      # [B,8,8,4]
                'global_score': s_gt,     # [B,1]
            }

            # Compute loss using criterion function
            loss, loss_logs = criterion(
                outputs=outputs,
                targets=targets,
                lambda_glob=args.lambda_glob,
                mu_cons=0.0
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            if step % args.log_interval == 0:
                metrics_history.append({
                    'step': step,
                    'loss': float(loss),
                    'loss_tile': loss_logs['l_tile'],
                    'loss_glob': loss_logs['l_glob'],
                })
                tqdm.write(f"Step {step}: loss={loss:.4f}, tile={loss_logs['l_tile']:.4f}, glob={loss_logs['l_glob']:.4f}")

            # Save
            if (step + 1) % args.save_interval == 0 or step == 0:
                ckpt_path = output_dir / f'ckpt_step_{step}.pth'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                tqdm.write(f"Saved: {ckpt_path}")

            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        if step >= args.total_steps:
            break

    pbar.close()

    # Save final
    final_ckpt = output_dir / 'ckpt_last.pth'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_ckpt)

    # Save metrics
    with open(output_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'loss', 'loss_tile', 'loss_glob'])
        writer.writeheader()
        writer.writerows(metrics_history)

    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
