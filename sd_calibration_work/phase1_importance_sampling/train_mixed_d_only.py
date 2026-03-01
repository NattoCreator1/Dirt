#!/usr/bin/env python3
"""
Phase 1: Train Mixed(8960, D-only) with Importance Sampling

This script trains a mixed WoodScape + SD model using importance sampling.
It integrates with the existing baseline training infrastructure.

Usage:
    python train_mixed_d_only.py \
        --mixed_csv mixed_ws4000_sd8960.csv \
        --weights weights_8960_correct.npz \
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
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cv2

# Import project modules
from baseline.models.baseline_dualhead import BaselineDualHead


class MixedWoodscapeDataset(Dataset):
    """
    Wrapper for mixed WoodScape + SD dataset.
    Handles NaN values in CSV by loading from NPZ.
    """
    def __init__(self, dataframe, img_root: str = '.', global_target: str = 'S'):
        self.df = dataframe.reset_index(drop=True)
        self.img_root = img_root
        self.global_target = global_target

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _load_npz(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row['rgb_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()

        # Load NPZ
        npz_path = row['npz_path']
        if not os.path.isabs(npz_path):
            npz_path = os.path.join(self.img_root, npz_path)

        npz = self._load_npz(npz_path)
        tile_cov = torch.from_numpy(npz["tile_cov"].astype(np.float32))

        # Load global score - handle NaN in CSV
        gt_key = self.global_target
        global_score = None

        # Check if CSV column exists and value is valid (not NaN)
        if gt_key in self.df.columns:
            val = row.get(gt_key)
            if pd.notna(val):
                global_score = float(val)

        # If CSV value is NaN or missing, load from NPZ
        if global_score is None:
            if gt_key in npz:
                global_score = float(npz[gt_key])
            elif "S" in npz:
                global_score = float(npz["S"])
            elif "global_score" in npz:
                global_score = float(npz["global_score"])
            else:
                raise RuntimeError(f"Cannot find global target '{gt_key}' in NPZ")

        y = torch.tensor([global_score], dtype=torch.float32)

        return {
            "image": x,
            "tile_cov": tile_cov,
            "S": y.squeeze(),  # Return as scalar for consistency
        }


class TwoLayerImportanceSampler(Sampler):
    """
    Two-layer sampler:
    1. First layer: Choose between real and SD by ratio
    2. Second layer: Apply weighted sampling to SD subset

    Index mapping:
    - Real samples: [0, n_real-1]
    - SD samples: [n_real, n_real+n_sd-1]
    """

    def __init__(self, n_real, n_sd, sd_weights, real_ratio=0.76, batch_size=32, generator=None):
        self.n_real = n_real
        self.n_sd = n_sd
        self.n_total = n_real + n_sd
        self.real_ratio = real_ratio
        self.batch_size = batch_size
        self.generator = generator

        # Create SD weighted sampler
        from torch.utils.data.sampler import WeightedRandomSampler
        self.sd_sampler = list(WeightedRandomSampler(
            weights=sd_weights,
            num_samples=len(sd_weights),
            replacement=True,
            generator=generator
        ))

        # Pre-generate batches
        self.batches = self._generate_batches()

    def _generate_batches(self, n_batches=1000):
        """Pre-generate batches."""
        batches = []
        n_real_batch = int(self.batch_size * self.real_ratio)
        n_sd_batch = self.batch_size - n_real_batch

        for _ in range(n_batches):
            # Sample real
            real_indices = torch.randint(
                0, self.n_real, (n_real_batch,), generator=self.generator
            ).tolist()

            # Sample SD with weights
            sd_indices = []
            for _ in range(n_sd_batch):
                if len(self.sd_sampler) == 0:
                    # Refill
                    from torch.utils.data.sampler import WeightedRandomSampler
                    self.sd_sampler = list(WeightedRandomSampler(
                        weights=torch.ones(self.n_sd),
                        num_samples=self.n_sd,
                        replacement=True,
                        generator=self.generator
                    ))
                sd_idx = self.sd_sampler.pop(0)
                sd_indices.append(self.n_real + sd_idx)  # Offset to global index

            # Combine and shuffle
            batch = real_indices + sd_indices
            perm = torch.randperm(len(batch), generator=self.generator).tolist()
            batch = [batch[i] for i in perm]
            batches.append(batch)

        return batches

    def __iter__(self):
        """Iterate over batches infinitely."""
        while True:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--mixed_csv', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img_root', type=str, default='.')
    parser.add_argument('--real_ratio', type=float, default=0.76)

    # Training
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--out_root', type=str, default='baseline/runs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lambda_glob', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    # Logging
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=2000)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.out_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Phase 1: Train Mixed(8960, D-only) with Importance Sampling")
    print("=" * 80)
    print()
    print(f"Run: {args.run_name}")
    print(f"Mixed CSV: {args.mixed_csv}")
    print(f"Weights: {args.weights}")
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
    sd_weights = weights_data['weights']
    print(f"SD weights: {len(sd_weights)}, mean={sd_weights.mean():.4f}")
    print()

    # Create dataset
    print("【3. Create dataset】")
    dataset = MixedWoodscapeDataset(
        dataframe=mixed_df,
        img_root=args.img_root,
        global_target='S_full_wgap_alpha50'
    )
    print(f"Dataset size: {len(dataset)}")
    print()

    # Create sampler and dataloader
    print("【4. Create sampler and dataloader】")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = TwoLayerImportanceSampler(
        n_real=n_real,
        n_sd=n_sd,
        sd_weights=sd_weights,
        real_ratio=args.real_ratio,
        batch_size=args.batch_size,
        generator=generator
    )

    def collate_fn(batch):
        """Custom collate function."""
        images = torch.stack([item['image'] for item in batch])
        tile_cov = torch.stack([item['tile_cov'] for item in batch])

        # S values are now guaranteed to be valid (loaded from NPZ if CSV was NaN)
        S_values = [item['S'] for item in batch]
        S_tensor = torch.stack(S_values)  # Stack as scalars

        return {
            'image': images,
            'tile_cov': tile_cov,
            'S': S_tensor,
        }

    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4
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

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use MAE (L1) loss for both tile and global, matching baseline/losses.py
    import torch.nn.functional as F
    criterion_tile = F.l1_loss
    criterion_glob = F.l1_loss

    # Training loop
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print()

    model.train()
    step = 0
    metrics_history = []

    pbar = tqdm(total=args.total_steps, desc="Training")

    for batch_idx in range(100000):  # Large number to cover all steps
        for batch in train_loader:
            if step >= args.total_steps:
                break

            # Move to device
            images = batch['image'].to(device)
            G_gt = batch['tile_cov'].to(device)  # [B,8,8,4]
            S_gt = batch['S'].to(device).unsqueeze(1)  # [B,1]

            # Forward
            outputs = model(images)
            G_hat = outputs['G_hat']  # [B,4,8,8]
            S_hat = outputs['S_hat']  # [B,1]

            # Loss: Use MAE (L1) matching baseline/losses.py
            # G_gt is [B,8,8,4], need to permute to [B,4,8,8]
            G_gt_permuted = G_gt.permute(0, 3, 1, 2).contiguous()  # [B,4,8,8]
            L_tile = criterion_tile(G_hat, G_gt_permuted)  # MAE for tiles
            L_glob = criterion_glob(S_hat, S_gt)  # MAE for global
            L_total = L_tile + args.lambda_glob * L_glob

            # Backward
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            # Log
            if step % args.log_interval == 0:
                metrics_history.append({
                    'step': step,
                    'loss': float(L_total),
                    'loss_tile': float(L_tile),
                    'loss_glob': float(L_glob),
                })
                tqdm.write(f"Step {step}: loss={L_total:.4f}, tile={L_tile:.4f}, glob={L_glob:.4f}")

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
            pbar.set_postfix({'loss': f'{L_total:.4f}'})

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
    pd.DataFrame(metrics_history).to_csv(output_dir / 'metrics.csv', index=False)

    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Final checkpoint: {final_ckpt}")
    print()
    print("Next: Evaluate on Test_Real and Test_Ext")


if __name__ == "__main__":
    main()
