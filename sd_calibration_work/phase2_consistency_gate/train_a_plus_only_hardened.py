#!/usr/bin/env python3
"""
Phase 2: A+-only Training with Hardening (一致性门控 + 硬化策略)

Key Hardening Modifications:
1. Gate only applies to SD subset's global head
2. SD's global loss weight multiplied by α_sd < 1 (e.g., 0.4)
3. SD tile head trains normally (full weight)
4. Track and report pass_rate_sd

Intuition: Even gated SD samples may not align perfectly with external "level semantics";
let them only provide a "light nudge" to the global head.
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset


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


class MixedWoodscapeDataset(Dataset):
    """
    Mixed dataset with WoodScape + SD samples.
    Handles NaN values in CSV by loading from NPZ.
    """
    def __init__(self, mixed_csv, img_root=".", global_target="S_full_wgap_alpha50"):
        self.df = pd.read_csv(mixed_csv)
        self.img_root = img_root
        self.global_target = global_target

        # Identify columns
        self.img_col = None
        for col in ["rgb_path", "image_path", "image", "img_path", "path", "filename"]:
            if col in self.df.columns:
                self.img_col = col
                break
        if self.img_col is None:
            raise RuntimeError(f"Cannot find image path column in {mixed_csv}")

        self.npz_col = None
        for col in ["npz_path", "tile_npz", "label_npz", "npz", "tile_path"]:
            if col in self.df.columns:
                self.npz_col = col
                break

        self.source_col = None
        for col in ["source", "domain", "dataset"]:
            if col in self.df.columns:
                self.source_col = col
                break

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Load global_target from NPZ for NaN entries
        self._load_targets_from_npz()

    def _load_targets_from_npz(self):
        """Load S_full_wgap_alpha50 from NPZ for rows where CSV has NaN."""
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            val = row.get(self.global_target, None)
            if pd.isna(val):
                # Load from NPZ
                npz_path = row[self.npz_col] if self.npz_col else self._get_npz_path(row)
                if os.path.exists(npz_path):
                    npz = np.load(npz_path, allow_pickle=True)
                    if self.global_target in npz:
                        self.df.at[idx, self.global_target] = float(npz[self.global_target])
                    elif "S_full" in npz:
                        # Fallback: S_full_wgap_alpha50 might be stored as S_full
                        self.df.at[idx, self.global_target] = float(npz["S_full"])

    def _get_npz_path(self, row):
        """Derive NPZ path from image path."""
        img_path = row[self.img_col]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return f"dataset/woodscape_processed/labels_tile/{stem}.npz"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)

        # Load image
        import cv2
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)

        # Normalize
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # [3,H,W]

        # Load NPZ for tile labels
        npz_path = row[self.npz_col] if self.npz_col and not pd.isna(row[self.npz_col]) else self._get_npz_path(row)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        if "tile_cov" not in npz:
            raise RuntimeError(f"'tile_cov' not in npz: {npz_path}")
        tile_cov = npz["tile_cov"].astype(np.float32)  # (8,8,4)
        tile_cov_t = torch.from_numpy(tile_cov)        # [8,8,4]

        # Global score
        global_score = float(row[self.global_target])
        y = torch.tensor([global_score], dtype=torch.float32)  # [1]

        # Source indicator (WoodScape or SD)
        is_sd = False
        if self.source_col:
            is_sd = str(row[self.source_col]).lower() in ['sd', 'synthetic', 'syn']

        return {
            "image": x,
            "tile_cov": tile_cov_t,
            "global_score": y,
            "is_sd": is_sd,
            "img_path": img_path,
        }


class ConsistencyGate:
    """
    A+-only: Consistency gate for filtering SD samples based on prediction consistency.

    Gate criterion: |S_hat - S_agg| < threshold
    Only SD samples that pass the gate contribute to global head training.
    All samples (including SD) always contribute to tile head training.

    Hardening: SD samples have reduced global loss weight (α_sd < 1).
    """
    def __init__(self, threshold=0.05, alpha_sd=0.4, warmup_steps=1000):
        self.threshold = threshold
        self.alpha_sd = alpha_sd  # SD global loss weight multiplier (< 1)
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Statistics
        self.total_sd = 0
        self.passed_sd = 0
        self.pass_rate_history = []

    def __call__(self, S_hat, S_agg, is_sd_list, step=None):
        """
        Apply consistency gate to a batch.

        Args:
            S_hat: [B, 1] Global head predictions
            S_agg: [B, 1] Aggregated predictions from tile head
            is_sd_list: List[B] of bool indicating SD samples
            step: Current training step (optional)

        Returns:
            lambda_weights: [B, 1] Loss weights for global head (1.0 for WoodScape,
                           α_sd for gated SD, 0.0 for filtered SD)
            gate_info: Dict with gate statistics
        """
        if step is not None:
            self.current_step = step

        B = S_hat.shape[0]
        lambda_weights = torch.ones(B, 1, device=S_hat.device)

        # During warmup, no gating
        if self.current_step < self.warmup_steps:
            return lambda_weights, {"phase": "warmup"}

        # Compute consistency gap
        gap = torch.abs(S_hat - S_agg).squeeze(-1)  # [B]

        # Apply gate only to SD samples
        for i, is_sd in enumerate(is_sd_list):
            if is_sd:
                self.total_sd += 1
                if gap[i].item() < self.threshold:
                    self.passed_sd += 1
                    lambda_weights[i] = self.alpha_sd  # Passed SD: reduced weight
                else:
                    lambda_weights[i] = 0.0  # Failed SD: no global loss

        # Compute pass rate
        pass_rate = self.passed_sd / max(self.total_sd, 1)
        self.pass_rate_history.append(pass_rate)

        gate_info = {
            "phase": "gating",
            "total_sd": self.total_sd,
            "passed_sd": self.passed_sd,
            "pass_rate": pass_rate,
            "threshold": self.threshold,
            "alpha_sd": self.alpha_sd,
        }

        return lambda_weights, gate_info


def train_one_epoch(model, loader, optimizer, device, gate, step_offset, lambda_glob=1.0):
    """Train for one epoch."""
    model.train()
    sum_tile = 0.0
    sum_glob = 0.0
    sum_total = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        step = step_offset + n_batches

        images = batch["image"].to(device)
        tile_cov = batch["tile_cov"].to(device)
        global_scores = batch["global_score"].to(device)
        is_sd_list = batch["is_sd"]
        batch_size = images.shape[0]

        # Forward pass
        outputs = model(images)
        G_hat = outputs['G_hat']  # [B,4,8,8]
        S_hat = outputs['S_hat']  # [B,1]
        S_agg = outputs['S_agg']  # [B,1]

        # Tile loss (all samples, full weight)
        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()  # [B,4,8,8]
        L_tile = F.l1_loss(G_hat, G_tgt)

        # Apply consistency gate for global loss
        lambda_weights, gate_info = gate(S_hat, S_agg, is_sd_list, step=step)
        lambda_weights = lambda_weights.to(device)

        # Global loss with gate weights
        base_glob_loss = F.l1_loss(S_hat, global_scores, reduction='none')  # [B,1]
        L_glob = (base_glob_loss * lambda_weights).sum() / max(lambda_weights.sum(), 1e-8)

        # Total loss
        L_total = L_tile + lambda_glob * L_glob

        # Backward
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        # Metrics
        sum_tile += L_tile.item()
        sum_glob += L_glob.item()
        sum_total += L_total.item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{L_total.item():.4f}",
            "tile": f"{L_tile.item():.4f}",
            "glob": f"{L_glob.item():.4f}",
            "pass_rate": f"{gate_info.get('pass_rate', 1.0):.2f}" if gate_info.get('phase') == 'gating' else "warmup"
        })

    return {
        "loss": sum_total / n_batches,
        "loss_tile": sum_tile / n_batches,
        "loss_glob": sum_glob / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    sum_tile = 0.0
    sum_glob = 0.0
    n = 0

    all_shat = []
    all_sagg = []
    all_gt = []

    for batch in tqdm(loader, desc="Evaluating", ncols=80):
        images = batch["image"].to(device)
        tile_cov = batch["tile_cov"].to(device)
        global_scores = batch["global_score"].to(device)

        outputs = model(images)
        G_hat = outputs['G_hat']
        S_hat = outputs['S_hat']
        S_agg = outputs['S_agg']

        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()
        tile_mae = torch.mean(torch.abs(G_hat - G_tgt))
        glob_mae = torch.mean(torch.abs(S_hat - global_scores))

        bs = images.shape[0]
        sum_tile += float(tile_mae.cpu()) * bs
        sum_glob += float(glob_mae.cpu()) * bs
        n += bs

        all_shat.append(S_hat.cpu().numpy())
        all_sagg.append(S_agg.cpu().numpy())
        all_gt.append(global_scores.cpu().numpy())

    shat = np.concatenate(all_shat).reshape(-1)
    sagg = np.concatenate(all_sagg).reshape(-1)
    gt = np.concatenate(all_gt).reshape(-1)

    rho_shat_gt = spearmanr(shat, gt)
    rho_shat_sagg = spearmanr(shat, sagg)

    return {
        "tile_mae": sum_tile / n,
        "glob_mae": sum_glob / n,
        "rho_shat_gt": rho_shat_gt,
        "rho_shat_sagg": rho_shat_sagg,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: A+-only training with hardening")
    parser.add_argument("--mixed_csv", type=str, default="sd_calibration_work/phase1_importance_sampling/mixed_ws4000_sd8960.csv")
    parser.add_argument("--img_root", type=str, default=".")
    parser.add_argument("--real_ratio", type=float, default=0.76, help="Expected WoodScape ratio in batch")
    parser.add_argument("--run_name", type=str, default="mixed_ws4000_sd8960_a_plus_only_hardened")
    parser.add_argument("--out_root", type=str, default="baseline/runs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda_glob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--gate_threshold", type=float, default=0.05)
    parser.add_argument("--alpha_sd", type=float, default=0.4, help="SD global loss weight multiplier (< 1)")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save args
    args_path = os.path.join(out_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print(f"Loading mixed dataset: {args.mixed_csv}")
    dataset = MixedWoodscapeDataset(
        mixed_csv=args.mixed_csv,
        img_root=args.img_root,
        global_target="S_full_wgap_alpha50"
    )
    print(f"Dataset size: {len(dataset)}")

    # Create dataloaders (train only, no val split for mixed training)
    # Disable pin_memory to avoid CUDA OOM issues
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)

    # Create model
    model = BaselineDualHead(pretrained=False).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Consistency gate
    gate = ConsistencyGate(
        threshold=args.gate_threshold,
        alpha_sd=args.alpha_sd,
        warmup_steps=args.warmup_steps
    )

    print(f"\nPhase 2: A+-only Training (Hardened)")
    print(f"  Gate threshold: {args.gate_threshold}")
    print(f"  α_sd (SD global weight): {args.alpha_sd}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  λ_glob: {args.lambda_glob}")

    # Metrics CSV
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("step,loss,loss_tile,loss_glob,pass_rate_sd\n")

    # Training loop
    step = 0
    best_glob_mae = float('inf')

    while step < args.total_steps:
        metrics = train_one_epoch(
            model, loader, optimizer, device, gate,
            step_offset=step, lambda_glob=args.lambda_glob
        )
        step += len(loader)

        # Log metrics
        pass_rate = gate.pass_rate_history[-1] if gate.pass_rate_history else 1.0
        with open(metrics_path, "a") as f:
            f.write(f"{step},{metrics['loss']:.4f},{metrics['loss_tile']:.4f},{metrics['loss_glob']:.4f},{pass_rate:.4f}\n")

        # Print progress
        if step % args.log_interval == 0 or step >= args.total_steps:
            print(f"\nStep {step}/{args.total_steps}")
            print(f"  Loss: {metrics['loss']:.4f} (Tile: {metrics['loss_tile']:.4f}, Glob: {metrics['loss_glob']:.4f})")
            print(f"  Gate pass rate: {pass_rate:.2%}")

        # Save checkpoint
        if step % args.save_interval == 0 or step >= args.total_steps:
            ckpt_path = os.path.join(out_dir, f"ckpt_{step}.pth")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gate": {
                    "total_sd": gate.total_sd,
                    "passed_sd": gate.passed_sd,
                    "pass_rate": pass_rate,
                },
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            # Save last checkpoint
            last_path = os.path.join(out_dir, "ckpt_last.pth")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gate": {
                    "total_sd": gate.total_sd,
                    "passed_sd": gate.passed_sd,
                    "pass_rate": pass_rate,
                },
                "args": vars(args),
            }, last_path)

    print(f"\nTraining complete!")
    print(f"Final gate statistics:")
    print(f"  Total SD samples evaluated: {gate.total_sd}")
    final_pass_rate = gate.passed_sd / max(gate.total_sd, 1)
    print(f"  Passed gate: {gate.passed_sd} ({final_pass_rate:.2%})")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
