#!/usr/bin/env python3
"""
Evaluate all label definition ablation experiments and generate comparison table.

This script evaluates trained ablation models on both Test_Real (WoodScape)
and Test_Ext (external washed) to generate comparison tables for the paper.

Usage:
    python scripts/17_evaluate_ablation_results.py --ablation_dir baseline/runs/ablation_label_def
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np

from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from baseline.models.baseline_dualhead import BaselineDualHead


class ExternalTestDataset:
    """Dataset for external test images with weak labels (ext_level 1-5)."""

    def __init__(self, csv_path: str, img_root: str = ".", resize_w: int = 640, resize_h: int = 480):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.resize_w = resize_w
        self.resize_h = resize_h

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = self.img_root / img_path
        else:
            img_path = Path(img_path)

        # Read image using cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Resize
        img = cv2.resize(img, (self.resize_w, self.resize_h))

        # BGR to RGB
        img = img[:, :, ::-1]

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(img).float(),
            "ext_level": torch.tensor(row["ext_level"], dtype=torch.float32),
        }


def spearmanr(a, b):
    """Compute Spearman correlation (no scipy dependency)."""
    # Convert to numpy if tensor
    import numpy as np
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    a = a.reshape(-1)
    b = b.reshape(-1)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def evaluate_with_gt(model, loader, device, global_target="S"):
    """Evaluate on dataset with ground truth tile + global labels."""
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

        G_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()
        tile_mae = torch.mean(torch.abs(out["G_hat"] - G_tgt))
        glob_mae = torch.mean(torch.abs(out["S_hat"] - y))
        glob_mse = torch.mean((out["S_hat"] - y) ** 2)
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

    import numpy as np
    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)
    gt = np.concatenate(all_gt, axis=0).reshape(-1)

    metrics = {
        "tile_mae": sum_tile / max(n, 1),
        "glob_mae": sum_glob / max(n, 1),
        "glob_rmse": float(np.sqrt(sum_mse / max(n, 1))),
        "gap_mae": sum_gap / max(n, 1),
        "rho_shat_sagg": spearmanr(shat, sagg),
        "rho_shat_gt": spearmanr(shat, gt),
    }
    return metrics


def load_and_evaluate(ckpt_path, index_csv, img_root, labels_tile_dir,
                      split_col, split_value, global_target, device, batch_size=32):
    """Load checkpoint and evaluate on specified dataset."""
    if not os.path.exists(ckpt_path):
        return None

    print(f"  Evaluating {os.path.basename(ckpt_path)}...")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    # Create dataset
    spec = WoodscapeIndexSpec(
        index_csv=index_csv,
        img_root=img_root,
        labels_tile_dir=labels_tile_dir,
        split_col=split_col,
        split_value=split_value,
        global_target=global_target,
    )
    ds = WoodscapeTileDataset(spec)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)

    # Evaluate
    metrics = evaluate_with_gt(model, loader, device, global_target)
    metrics["epoch"] = ckpt.get("epoch", -1)
    return metrics


def load_and_evaluate_external(ckpt_path, ext_csv, ext_img_root, device, batch_size=32):
    """Load checkpoint and evaluate on external test set (weak labels)."""
    if not os.path.exists(ckpt_path):
        return None
    if not os.path.exists(ext_csv):
        print(f"  WARNING: External test CSV not found: {ext_csv}")
        return None

    print(f"  Evaluating on external test...")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    # Evaluate with smaller batch size for large external test set
    metrics = evaluate_external_test(model, ext_csv, ext_img_root, device, batch_size=16)
    return metrics


@torch.no_grad()
def evaluate_external_test(model, ext_csv, ext_img_root, device, batch_size=16):
    """Evaluate on external test set with weak labels (ext_level 1-5)."""
    model.eval()

    # Create dataset
    ds = ExternalTestDataset(ext_csv, ext_img_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=False)

    all_shat = []
    all_sagg = []
    all_ext_level = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_ext = batch["ext_level"]  # Weak label: 1-5 scale

        out = model(x)

        all_shat.append(out["S_hat"].detach().cpu().numpy())
        all_sagg.append(out["S_agg"].detach().cpu().numpy())
        all_ext_level.append(y_ext.numpy())

    import numpy as np
    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)
    ext_level = np.concatenate(all_ext_level, axis=0).reshape(-1)

    # Normalize predictions to 1-5 scale for comparison
    # Since S_hat is in [0, 1], we scale to [1, 5]
    shat_scaled = shat * 4.0 + 1.0
    sagg_scaled = sagg * 4.0 + 1.0

    # Compute metrics
    mae_shat = float(np.mean(np.abs(shat_scaled - ext_level)))
    mae_sagg = float(np.mean(np.abs(sagg_scaled - ext_level)))
    rmse_shat = float(np.sqrt(np.mean((shat_scaled - ext_level) ** 2)))
    rmse_sagg = float(np.sqrt(np.mean((sagg_scaled - ext_level) ** 2)))

    # Compute Spearman correlation
    rho_shat = spearmanr(torch.from_numpy(shat_scaled), torch.from_numpy(ext_level.astype(np.float32)))
    rho_sagg = spearmanr(torch.from_numpy(sagg_scaled), torch.from_numpy(ext_level.astype(np.float32)))

    return {
        "ext_mae_shat": mae_shat,
        "ext_mae_sagg": mae_sagg,
        "ext_rmse_shat": rmse_shat,
        "ext_rmse_sagg": rmse_sagg,
        "ext_rho_shat": rho_shat,
        "ext_rho_sagg": rho_sagg,
        "ext_n": len(ext_level),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate ablation experiments")
    ap.add_argument("--ablation_dir", default="baseline/runs/ablation_label_def",
                    help="Directory containing ablation experiment results")
    ap.add_argument("--index_csv", default="dataset/woodscape_processed/meta/labels_index_ablation.csv",
                    help="Path to index CSV")
    ap.add_argument("--img_root", default="dataset/woodscape_raw")
    ap.add_argument("--labels_tile_dir", default="dataset/woodscape_processed/labels_tile")
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--ext_csv", default="dataset/external_test_washed_processed/test_ext.csv")
    ap.add_argument("--ext_img_root", default=".")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_csv", default="baseline/runs/ablation_label_def/comparison_table.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ablation_dir = Path(args.ablation_dir)

    # Severity targets to evaluate
    severity_targets = {
        "s": "Simple Mean (1 - clean_ratio)",
        "S_op_only": "Opacity-Aware Only",
        "S_op_sp": "Opacity + Spatial",
        "S_full": "Full Severity Score (eta=0.9)",
        "S_full_eta00": "Full Severity Score (eta=0)",
    }

    # Collect results
    results = []

    for target_key, target_name in severity_targets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {target_name}")
        print(f"{'='*60}")

        # Find checkpoint directories matching this target
        # First try exact match, then try with suffix
        ckpt_dir = ablation_dir / f"ablation_{target_key}"
        if not ckpt_dir.exists():
            ckpt_dirs = sorted(ablation_dir.glob(f"ablation_{target_key}_*"))
            if not ckpt_dirs:
                print(f"  WARNING: No checkpoint found for {target_key}")
                continue
            ckpt_dir = ckpt_dirs[-1]

        ckpt_path = ckpt_dir / "ckpt_best.pth"

        if not ckpt_path.exists():
            print(f"  WARNING: ckpt_best.pth not found in {ckpt_dir}")
            continue

        # Evaluate on Val_Real
        val_metrics = load_and_evaluate(
            str(ckpt_path), args.index_csv, args.img_root, args.labels_tile_dir,
            args.split_col, args.val_split, target_key, device, args.batch_size
        )

        # Evaluate on Test_Real
        test_metrics = load_and_evaluate(
            str(ckpt_path), args.index_csv, args.img_root, args.labels_tile_dir,
            args.split_col, args.test_split, target_key, device, args.batch_size
        )

        # Evaluate on External Test (weak labels)
        ext_metrics = load_and_evaluate_external(
            str(ckpt_path), args.ext_csv, args.ext_img_root, device
        )

        # Load run args
        run_args_path = ckpt_dir / "run_args.json"
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

        result = {
            "severity_target": target_key,
            "target_name": target_name,
            "run_dir": str(ckpt_dir),
            "epoch": val_metrics["epoch"] if val_metrics else -1,
            # Val_Real metrics
            "val_tile_mae": val_metrics["tile_mae"] if val_metrics else None,
            "val_glob_mae": val_metrics["glob_mae"] if val_metrics else None,
            "val_glob_rmse": val_metrics["glob_rmse"] if val_metrics else None,
            "val_gap_mae": val_metrics["gap_mae"] if val_metrics else None,
            "val_rho_shat_gt": val_metrics["rho_shat_gt"] if val_metrics else None,
            # Test_Real metrics
            "test_tile_mae": test_metrics["tile_mae"] if test_metrics else None,
            "test_glob_mae": test_metrics["glob_mae"] if test_metrics else None,
            "test_glob_rmse": test_metrics["glob_rmse"] if test_metrics else None,
            "test_gap_mae": test_metrics["gap_mae"] if test_metrics else None,
            "test_rho_shat_gt": test_metrics["rho_shat_gt"] if test_metrics else None,
            # External test metrics (weak labels, 1-5 scale)
            "ext_mae_shat": ext_metrics["ext_mae_shat"] if ext_metrics else None,
            "ext_mae_sagg": ext_metrics["ext_mae_sagg"] if ext_metrics else None,
            "ext_rmse_shat": ext_metrics["ext_rmse_shat"] if ext_metrics else None,
            "ext_rmse_sagg": ext_metrics["ext_rmse_sagg"] if ext_metrics else None,
            "ext_rho_shat": ext_metrics["ext_rho_shat"] if ext_metrics else None,
            "ext_rho_sagg": ext_metrics["ext_rho_sagg"] if ext_metrics else None,
            "ext_n": ext_metrics["ext_n"] if ext_metrics else None,
        }

        results.append(result)

    # Create comparison table
    df = pd.DataFrame(results)

    # Save to CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Comparison table saved to: {out_path}")
    print(f"{'='*60}")

    # Print formatted table
    print("\n" + "=" * 140)
    print("Label Definition Ablation Results:")
    print("=" * 140)

    # Table 1: WoodScape results (Val + Test)
    print("\n1. WoodScape Strong Labels (Val: 800 samples, Test: 1000 samples):")
    print("-" * 140)
    print(f"{'Severity Target':<35} | {'Val MAE':<10} | {'Val RMSE':<10} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test ρ':<10}")
    print("-" * 140)

    for _, row in df.iterrows():
        print(f"{row['target_name']:<35} | {row['val_glob_mae']:.4f}   | {row['val_glob_rmse']:.4f}   | {row['test_glob_mae']:.4f}   | {row['test_glob_rmse']:.4f}   | {row['test_rho_shat_gt']:.4f}")

    print("-" * 140)

    # Table 2: External test results (weak labels, 1-5 scale)
    if df["ext_mae_shat"].notna().any():
        print("\n2. External Test Weak Labels (~50k samples, ext_level 1-5 scale):")
        print("-" * 160)
        print(f"{'Severity Target':<35} | {'MAE (S_hat)':<12} | {'RMSE (S_hat)':<12} | {'MAE (S_agg)':<12} | {'RMSE (S_agg)':<12} | {'ρ (S_hat)':<10} | {'ρ (S_agg)':<10} | {'Δρ':<8}")
        print("-" * 160)

        for _, row in df.iterrows():
            if pd.notna(row['ext_mae_shat']):
                delta_rho = row['ext_rho_shat'] - row['ext_rho_sagg']
                print(f"{row['target_name']:<35} | {row['ext_mae_shat']:.4f}       | {row['ext_rmse_shat']:.4f}       | {row['ext_mae_sagg']:.4f}       | {row['ext_rmse_sagg']:.4f}       | {row['ext_rho_shat']:.4f}      | {row['ext_rho_sagg']:.4f}      | {delta_rho:+.4f}")

        print("-" * 160)

    # Print analysis
    print("\nAnalysis:")
    print("-" * 140)

    if len(df) > 0:
        best_val = df.loc[df["val_glob_mae"].idxmin()]
        best_test = df.loc[df["test_glob_mae"].idxmin()]
        best_rho = df.loc[df["test_rho_shat_gt"].idxmax()]

        print(f"Best validation MAE:  {best_val['target_name']} ({best_val['val_glob_mae']:.4f})")
        print(f"Best test MAE:       {best_test['target_name']} ({best_test['test_glob_mae']:.4f})")
        print(f"Best test ρ:         {best_rho['target_name']} ({best_rho['test_rho_shat_gt']:.4f})")

        if df["ext_mae_shat"].notna().any():
            best_ext = df.loc[df["ext_mae_shat"].idxmin()]
            best_ext_rho = df.loc[df["ext_rho_shat"].idxmax()]
            best_ext_rho_sagg = df.loc[df["ext_rho_sagg"].idxmax()]
            print(f"Best external MAE (S_hat):  {best_ext['target_name']} ({best_ext['ext_mae_shat']:.4f})")
            print(f"Best external ρ (S_hat):    {best_ext_rho['target_name']} ({best_ext_rho['ext_rho_shat']:.4f})")
            print(f"Best external ρ (S_agg):    {best_ext_rho_sagg['target_name']} ({best_ext_rho_sagg['ext_rho_sagg']:.4f})")

            # Consistency analysis: compare S_hat vs S_agg
            print("\nConsistency Analysis (S_hat vs S_agg on External Test):")
            print("-" * 140)
            for _, row in df.iterrows():
                if pd.notna(row['ext_rho_shat']):
                    delta_rho = row['ext_rho_shat'] - row['ext_rho_sagg']
                    delta_mae = row['ext_mae_shat'] - row['ext_mae_sagg']
                    consistency = "更稳定" if abs(delta_rho) < 0.02 else ("S_hat更好" if delta_rho > 0 else "S_agg更好")
                    print(f"  {row['target_name']:<35} | Δρ={delta_rho:+.4f}, ΔMAE={delta_mae:+.4f} -> {consistency}")
            print("-" * 140)

    print("\nConclusions:")
    print("  - Lower MAE/RMSE = better regression accuracy")
    print("  - Higher ρ (Spearman) = better ranking ability")
    print("  - WoodScape: Strong labels (full mask annotations)")
    print("  - External: Weak labels (1-5 severity scale)")
    print("  - Compare S_full vs s to validate Severity Score design")
    print("  - Compare S_full vs S_op_only to validate spatial component")
    print("  - Compare S_full vs S_op_sp to validate dominance component")
    print("  - Compare S_full vs S_full_eta00 to validate transparency discount")
    print("  - External test validates generalization to real-world images")
    print("  - Δρ = ρ(S_hat) - ρ(S_agg): smaller absolute value = better consistency")


if __name__ == "__main__":
    main()
