#!/usr/bin/env python3
"""
Evaluate a trained baseline model on test sets.

Supports:
1. WoodScape test set (Test_Real): with full ground truth (tile + global)
   - Metrics: Tile MAE, Global MAE/RMSE, Level Accuracy, Consistency Gap
2. External test set (Test_Ext): with only occlusion level labels (a-e)
   - Metrics: Spearman correlation (S_hat vs occlusion level), Boxplot separation
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset
from baseline.models.baseline_dualhead import BaselineDualHead


class ImageOnlyDataset:
    """Minimal dataset that only loads images (no labels)."""
    def __init__(self, index_csv, img_root=".", resize_w=640, resize_h=480):
        import cv2
        import pandas as pd

        self.df = pd.read_csv(index_csv)
        self.img_root = img_root
        self.resize_w = resize_w
        self.resize_h = resize_h

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Auto-detect image path column
        for col in ["rgb_path", "image_path", "image", "img_path", "path", "filename"]:
            if col in self.df.columns:
                self.img_col = col
                break
        else:
            raise RuntimeError(f"Cannot find image path column in {index_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]

        # Handle relative paths
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()

        return {"image": x, "img_path": img_path}


def spearmanr(a, b):
    """Compute Spearman correlation (no scipy dependency)."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    ra = a.argsort().argsort().astype(np.float32)
    rb = b.argsort().argsort().astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


def compute_level_accuracy(S_pred, S_gt, thresholds=None):
    """
    Compute 4-level classification accuracy using thresholds.

    Levels: 0 (clean), 1 (light), 2 (moderate), 3 (heavy)
    Default thresholds based on training set quantiles.
    """
    if thresholds is None:
        # Default thresholds from Step 1.2.5 (re-binning)
        b0, b1, b2 = 0.01, 0.33, 0.66  # Will be loaded from trained model config if available
        # Fallback: use simple quartiles on predictions
        b1 = np.percentile(S_gt, 33)
        b2 = np.percentile(S_gt, 66)

    def to_level(S):
        L = np.zeros_like(S, dtype=int)
        # L=0 for clean (s <= b0), L=1-3 based on S for soiled samples
        L[S > b0] = 1
        L[S > b1] = 2
        L[S > b2] = 3
        return L

    L_pred = to_level(S_pred)
    L_gt = to_level(S_gt)

    accuracy = (L_pred == L_gt).mean()

    # Per-level accuracy
    level_acc = {}
    for l in range(4):
        mask = L_gt == l
        if mask.sum() > 0:
            level_acc[f"level_{l}_acc"] = (L_pred[mask] == L_gt[mask]).mean()

    return accuracy, level_acc


@torch.no_grad()
def evaluate_with_gt(model, loader, device, thresholds_path=None):
    """
    Evaluate on dataset with ground truth tile + global labels.

    Metrics:
    - Tile MAE: coverage prediction error
    - Global MAE/RMSE: severity regression error
    - Level Accuracy: 4-level classification accuracy
    - Consistency Gap: difference between S_hat and S_agg
    """
    model.eval()
    sum_tile = 0.0
    sum_glob = 0.0
    sum_mse = 0.0
    sum_gap = 0.0
    n = 0

    all_shat = []
    all_sagg = []
    all_gt = []
    all_paths = []
    all_Ghat = []  # Store tile predictions for visualization

    for batch in tqdm(loader, desc="Evaluating Test_Real", ncols=100):
        x = batch["image"].to(device, non_blocking=True)
        tile_cov = batch["tile_cov"].to(device, non_blocking=True)
        y = batch["global_score"].to(device, non_blocking=True)
        paths = batch.get("img_path", batch.get("npz_path"))

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
        all_paths.extend(paths)
        all_Ghat.append(out["G_hat"].detach().cpu().numpy())

    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)
    gt = np.concatenate(all_gt, axis=0).reshape(-1)

    # Load re-binning thresholds if available
    thresholds = None
    if thresholds_path and os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            th = json.load(f)
            thresholds = (th.get("b0", 0.01), th.get("b1", 0.33), th.get("b2", 0.66))

    # Compute level accuracy
    level_acc, per_level_acc = compute_level_accuracy(shat, gt, thresholds)

    metrics = {
        "tile_mae": sum_tile / max(n, 1),
        "glob_mae": sum_glob / max(n, 1),
        "glob_rmse": float(np.sqrt(sum_mse / max(n, 1))),
        "gap_mae": sum_gap / max(n, 1),  # Consistency Gap
        "level_accuracy": level_acc,
        "rho_shat_sagg": spearmanr(shat, sagg),
        "rho_shat_gt": spearmanr(shat, gt),
    }
    metrics.update(per_level_acc)

    predictions = pd.DataFrame({
        "image_path": all_paths,
        "S_hat": shat,
        "S_agg": sagg,
        "S_gt": gt,
        "error": np.abs(shat - gt),
    })

    return metrics, predictions


@torch.no_grad()
def evaluate_external(model, index_csv, img_root, device, label_col="ext_level", batch_size=16):
    """
    Evaluate on external test set (Test_Ext).

    Since this dataset only has weak ordinal labels (occlusion level 1-5),
    we evaluate:
    1. Spearman correlation between S_hat and occlusion level
    2. Separability: mean S_hat per level, check monotonicity
    3. Boxplot data for visualization
    """
    model.eval()

    # Load labels from CSV
    df = pd.read_csv(index_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in index CSV")

    labels = df[label_col].values

    # Create image-only dataset
    dataset = ImageOnlyDataset(index_csv, img_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_shat = []
    all_sagg = []
    all_paths = []

    for batch in tqdm(loader, desc="Evaluating Test_Ext", ncols=100):
        x = batch["image"].to(device, non_blocking=True)
        paths = batch["img_path"]

        out = model(x)

        all_shat.append(out["S_hat"].detach().cpu().numpy())
        all_sagg.append(out["S_agg"].detach().cpu().numpy())
        all_paths.extend(paths)

    shat = np.concatenate(all_shat, axis=0).reshape(-1)
    sagg = np.concatenate(all_sagg, axis=0).reshape(-1)

    predictions = pd.DataFrame({
        "image_path": all_paths,
        "occlusion_level": labels,
        "S_hat": shat,
        "S_agg": sagg,
    })

    # Aggregate by occlusion level
    agg = predictions.groupby("occlusion_level").agg({
        "S_hat": ["count", "mean", "std", "min", "max"],
        "S_agg": ["mean", "std"],
    }).round(4)

    # KEY METRIC: Spearman correlation between S_hat and occlusion level
    # Higher correlation = better generalization (model can rank severity)
    rho = spearmanr(shat, labels)

    # Check monotonicity: heavier occlusion (higher level) should have higher S_hat
    level_stats = predictions.groupby("occlusion_level")["S_hat"].agg(["mean", "std", "count"])
    level_means = level_stats["mean"].sort_index(ascending=False)  # 5,4,3,2,1
    monotonic = all(level_means.iloc[i] >= level_means.iloc[i+1] for i in range(len(level_means)-1))

    metrics = {
        "spearman_rho": rho,
        "monotonic": monotonic,
        "level_means": level_means.to_dict(),
    }

    return metrics, agg, predictions


def main():
    ap = argparse.ArgumentParser(description="Evaluate baseline model on test sets")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth file)")
    ap.add_argument("--test_set", choices=["woodscape", "external", "both"], default="both",
                    help="Which test set to evaluate on")

    # WoodScape test set arguments
    ap.add_argument("--index_csv", default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv",
                    help="Path to index CSV (for WoodScape)")
    ap.add_argument("--img_root", default="dataset/woodscape_raw")
    ap.add_argument("--labels_tile_dir", default=None)
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--test_split", default="test", help="Split value for WoodScape test set")
    ap.add_argument("--global_target", choices=["S", "s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00", "S_full_wgap_alpha50"], default="S",
                    help="Severity target for ablation experiments")
    ap.add_argument("--thresholds_path", default="dataset/woodscape_processed/meta/rebinned_thresholds.json",
                    help="Path to re-binning thresholds for level accuracy")

    # External test set arguments
    ap.add_argument("--ext_csv", default="dataset/my_external_test/test_ext.csv",
                    help="Path to external test CSV")
    ap.add_argument("--ext_img_root", default=".",
                    help="Root dir for resolving external test image paths")
    ap.add_argument("--ext_label_col", default="ext_level",
                    help="Column name for occlusion level in external CSV")

    # Other arguments
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for results (defaults to ckpt dir)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")

    # Determine output directory
    if args.out_dir is None:
        out_dir = os.path.dirname(args.ckpt)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Run evaluation
    if args.test_set in ["woodscape", "both"]:
        print("\n" + "="*50)
        print("Test_Real: WoodScape Test Set (In-domain, Full Labels)")
        print("="*50)

        spec = WoodscapeIndexSpec(
            index_csv=args.index_csv,
            img_root=args.img_root,
            labels_tile_dir=args.labels_tile_dir,
            split_col=args.split_col,
            split_value=args.test_split,
            global_target=args.global_target,
        )
        ds = WoodscapeTileDataset(spec)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
        print(f"Test set size: {len(ds)}")

        metrics, predictions = evaluate_with_gt(model, loader, device, args.thresholds_path)

        print("\n--- Metrics ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Save results
        metrics_path = os.path.join(out_dir, "eval_Test_Real_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

        pred_path = os.path.join(out_dir, "eval_Test_Real_predictions.csv")
        predictions.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")

        # Analysis: worst predictions
        print("\n--- Worst Predictions (Top 10) ---")
        worst = predictions.nlargest(10, "error")
        for _, row in worst.iterrows():
            print(f"  error={row['error']:.4f} | S_gt={row['S_gt']:.3f} | S_hat={row['S_hat']:.3f} | {os.path.basename(row['image_path'])}")

    if args.test_set in ["external", "both"]:
        print("\n" + "="*50)
        print("Test_Ext: External Test Set (Out-of-domain, Weak Labels)")
        print("="*50)

        if not os.path.exists(args.ext_csv):
            print(f"WARNING: External CSV not found: {args.ext_csv}")
            print("Skipping external evaluation.")
        else:
            metrics, agg, predictions = evaluate_external(
                model, args.ext_csv, args.ext_img_root, device,
                label_col=args.ext_label_col, batch_size=args.batch_size
            )

            print("\n--- Metrics ---")
            print(f"  Spearman rho (S_hat vs occlusion_level): {metrics['spearman_rho']:.4f}")
            print(f"  Monotonic (5 > 4 > 3 > 2 > 1): {metrics['monotonic']}")

            print("\n--- Aggregated by Occlusion Level ---")
            print(agg)
            print("\n  Note: ext_level 5=occlusion_a (heaviest), 1=occlusion_e (lightest)")
            print(f"  Interpretation: Higher rho = better generalization")

            # Save results
            metrics_path = os.path.join(out_dir, "eval_Test_Ext_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {metrics_path}")

            agg_path = os.path.join(out_dir, "eval_Test_Ext_aggregated.csv")
            agg.to_csv(agg_path)
            print(f"Aggregated results saved to: {agg_path}")

            pred_path = os.path.join(out_dir, "eval_Test_Ext_predictions.csv")
            predictions.to_csv(pred_path, index=False)
            print(f"Full predictions saved to: {pred_path}")

    print("\n" + "="*50)
    print("DONE!")
    print("="*50)


if __name__ == "__main__":
    main()
