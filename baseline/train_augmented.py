#!/usr/bin/env python3
"""
训练脚本 - 带相机域随机化增强 (Training with Camera Domain Randomization)

这是train_baseline.py的扩展版本，专门用于在线相机域随机化数据增强的训练。

使用场景:
1. WoodScape + SD合成数据混合训练时，添加相机域随机化
2. 纯WoodScape数据训练时，添加相机域随机化（对比实验）
3. 任何需要域随机化增强的训练场景

关键区别:
- train_baseline.py: 无数据增强
- train_augmented.py: 包含在线相机域随机化

Author: Claude Code Assistant
Date: 2026-02-24
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.losses import criterion
from baseline.data_augmentation import CameraDomainRandomization


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spearmanr_np(a, b):
    # no scipy dependency
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
    sum_gap = 0.0
    n = 0

    all_shat = []
    all_sagg = []
    all_gt = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        tile_cov = batch["tile_cov"].to(device, non_blocking=True)
        y = batch["global_score"].to(device, non_blocking=True)

        out = model(x)

        # tile mae
        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()
        tile_mae = torch.mean(torch.abs(out["G_hat"] - G_tgt))

        # global mae/rmse
        glob_mae = torch.mean(torch.abs(out["S_hat"] - y))
        glob_mse = torch.mean((out["S_hat"] - y) ** 2)

        # gap
        gap = torch.mean(torch.abs(out["S_hat"] - out["S_agg"]))

        bs = x.size(0)
        sum_tile += float(tile_mae.cpu()) * bs
        sum_glob += float(glob_mae.cpu()) * bs
        sum_mse += float(glob_mse.cpu()) * bs
        sum_gap += float(gap.cpu()) * bs
        n += bs

        all_shat.append(out["S_hat"].detach().cpu().numpy())
        all_sagg.append(out["S_agg"].detach().cpu().numpy())
        all_gt.append(y.detach().cpu().numpy())

    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)
    gt = np.concatenate(all_gt, axis=0).reshape(-1)

    metrics = {
        "tile_mae": sum_tile / max(n, 1),
        "glob_mae": sum_glob / max(n, 1),
        "glob_rmse": float(np.sqrt(sum_mse / max(n, 1))),
        "gap_mae": sum_gap / max(n, 1),
        "rho_shat_sagg": spearmanr_np(shat, sagg),
        "rho_shat_gt": spearmanr_np(shat, gt),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Train with Camera Domain Randomization augmentation"
    )
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--img_root", default="dataset/woodscape_raw")
    ap.add_argument("--labels_tile_dir", default=None)
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--train_split", required=True)
    ap.add_argument("--val_split", required=True)
    ap.add_argument("--test_split", default=None)
    ap.add_argument("--global_target",
                    choices=["S", "s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00", "S_full_wgap_alpha50"],
                    default="S",
                    help="Severity target for ablation experiments")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--lambda_glob", type=float, default=1.0)
    ap.add_argument("--mu_cons", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_root", default="baseline/runs")
    ap.add_argument("--amp", action="store_true")

    # === Camera Domain Randomization arguments ===
    ap.add_argument("--aug_enable", action="store_true", default=True,
                    help="Enable camera domain randomization (default: True for this script)")
    ap.add_argument("--aug_color_prob", type=float, default=0.8,
                    help="Probability of color/exposure randomization (default: 0.8)")
    ap.add_argument("--aug_noise_prob", type=float, default=0.7,
                    help="Probability of noise addition (default: 0.7)")
    ap.add_argument("--aug_blur_prob", type=float, default=0.4,
                    help="Probability of blur addition (default: 0.4)")
    ap.add_argument("--aug_compression_prob", type=float, default=0.5,
                    help="Probability of compression artifacts (default: 0.5)")
    ap.add_argument("--aug_lens_prob", type=float, default=0.4,
                    help="Probability of lens effects (default: 0.4)")
    ap.add_argument("--aug_resolution_prob", type=float, default=0.3,
                    help="Probability of resolution randomization (default: 0.3)")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update run_name to include augmentation info
    aug_suffix = "_aug" if args.aug_enable else ""
    run_name = args.run_name or f"augmented_{args.global_target}_mu{args.mu_cons}_{time.strftime('%Y%m%d_%H%M%S')}{aug_suffix}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # === Create augmentation for training only ===
    augmentation = None
    if args.aug_enable:
        augmentation = CameraDomainRandomization(
            enable=True,
            color_prob=args.aug_color_prob,
            noise_prob=args.aug_noise_prob,
            blur_prob=args.aug_blur_prob,
            compression_prob=args.aug_compression_prob,
            lens_prob=args.aug_lens_prob,
            resolution_prob=args.aug_resolution_prob,
            seed=args.seed,
        )
        print(f"[Camera Domain Randomization ENABLED]")
        print(f"  color_prob={args.aug_color_prob}, noise_prob={args.aug_noise_prob}")
        print(f"  blur_prob={args.aug_blur_prob}, compression_prob={args.aug_compression_prob}")
        print(f"  lens_prob={args.aug_lens_prob}, resolution_prob={args.aug_resolution_prob}")

    # === datasets ===
    # Training set: WITH augmentation
    spec_tr = WoodscapeIndexSpec(
        index_csv=args.index_csv,
        img_root=args.img_root,
        labels_tile_dir=args.labels_tile_dir,
        split_col=args.split_col,
        split_value=args.train_split,
        global_target=args.global_target,
        augmentation=augmentation  # Augmentation ONLY for training
    )

    # Validation set: NO augmentation
    spec_va = WoodscapeIndexSpec(
        index_csv=args.index_csv,
        img_root=args.img_root,
        labels_tile_dir=args.labels_tile_dir,
        split_col=args.split_col,
        split_value=args.val_split,
        global_target=args.global_target,
        augmentation=None  # NO augmentation for validation
    )

    ds_tr = WoodscapeTileDataset(spec_tr)
    ds_va = WoodscapeTileDataset(spec_va)

    print(f"[Dataset] Train: {len(ds_tr)} samples, Val: {len(ds_va)} samples")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model
    model = BaselineDualHead(pretrained=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # logging
    with open(os.path.join(out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_tile_mae", "val_glob_mae", "val_glob_rmse", "val_gap_mae", "rho_shat_sagg", "rho_shat_gt"])

    best_val = 1e9
    ckpt_best = os.path.join(out_dir, "ckpt_best.pth")
    ckpt_last = os.path.join(out_dir, "ckpt_last.pth")

    print(f"[Training] Starting training for {args.epochs} epochs...")
    print(f"[Output] {out_dir}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(dl_tr, desc=f"epoch {ep}/{args.epochs}", ncols=120)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            tile_cov = batch["tile_cov"].to(device, non_blocking=True)
            y = batch["global_score"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                out = model(x)
                loss, _ = criterion(out, {"tile_cov": tile_cov, "global_score": y},
                                    lambda_glob=args.lambda_glob, mu_cons=args.mu_cons)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        train_loss = float(np.mean(losses)) if losses else 0.0

        val_metrics = evaluate(model, dl_va, device)

        # save ckpt
        torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_last)
        if val_metrics["glob_mae"] < best_val:
            best_val = val_metrics["glob_mae"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val_glob_mae": best_val}, ckpt_best)

        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                ep, train_loss,
                val_metrics["tile_mae"], val_metrics["glob_mae"], val_metrics["glob_rmse"],
                val_metrics["gap_mae"], val_metrics["rho_shat_sagg"], val_metrics["rho_shat_gt"]
            ])

        print(f"[val] ep={ep} train_loss={train_loss:.4f} "
              f"tile_mae={val_metrics['tile_mae']:.4f} glob_mae={val_metrics['glob_mae']:.4f} "
              f"gap={val_metrics['gap_mae']:.4f} rho(shat,sagg)={val_metrics['rho_shat_sagg']:.4f}")

    print("DONE:", out_dir)
    print("BEST_CKPT:", ckpt_best)


if __name__ == "__main__":
    main()
