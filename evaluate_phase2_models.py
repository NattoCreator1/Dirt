#!/usr/bin/env python3
"""
Evaluate Phase 2 models (A+-only and Tile-only) on Test_Real and Test_Ext
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2

from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset


class ImageOnlyDataset:
    """Minimal dataset that only loads images (for Test_Ext without labels)."""
    def __init__(self, index_csv, img_root=".", resize_w=640, resize_h=480):
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
    """Compute Spearman correlation."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    ra = a.argsort().argsort().astype(np.float32)
    rb = b.argsort().argsort().astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def evaluate_model(model, loader, device, split_name=""):
    """Evaluate model on a dataset."""
    model.eval()
    sum_tile = 0.0
    sum_glob = 0.0
    n = 0

    all_shat = []
    all_sagg = []
    all_gt = []
    all_ghat = []

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", ncols=80):
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
        all_ghat.append(G_hat.cpu().numpy())

    shat = np.concatenate(all_shat).reshape(-1)
    sagg = np.concatenate(all_sagg).reshape(-1)
    gt = np.concatenate(all_gt).reshape(-1)
    ghat = np.concatenate(all_ghat)  # [N,4,8,8]

    rho_shat_gt = spearmanr(shat, gt)
    rho_shat_sagg = spearmanr(shat, sagg)

    return {
        "tile_mae": sum_tile / n,
        "glob_mae": sum_glob / n,
        "rho_shat_gt": rho_shat_gt,
        "rho_shat_sagg": rho_shat_sagg,
        "S_hat_mean": float(shat.mean()),
        "S_hat_std": float(shat.std()),
    }


def evaluate_test_ext(model, test_ext_csv, img_root, device, batch_size=16):
    """Evaluate model on Test_Ext with per-level statistics."""
    model.eval()

    # Load labels from CSV
    df = pd.read_csv(test_ext_csv)
    if "ext_level" not in df.columns:
        raise ValueError(f"ext_level column not found in {test_ext_csv}")

    labels = df["ext_level"].values

    # Create image-only dataset
    dataset = ImageOnlyDataset(test_ext_csv, img_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    all_shat = []
    all_sagg = []

    for batch in tqdm(loader, desc="Evaluating Test_Ext", ncols=80):
        images = batch["image"].to(device)
        outputs = model(images)
        S_hat = outputs['S_hat']
        S_agg = outputs['S_agg']

        all_shat.append(S_hat.detach().cpu().numpy())
        all_sagg.append(S_agg.detach().cpu().numpy())

    shat = np.concatenate(all_shat).reshape(-1)
    sagg = np.concatenate(all_sagg).reshape(-1)

    # Spearman correlation between S_hat and occlusion level
    rho = spearmanr(shat, labels)

    # Per-level statistics (level 5=a, 4=b, 3=c, 2=d, 1=e)
    level_means = {}
    for level in sorted(np.unique(labels), reverse=True):  # 5,4,3,2,1
        mask = labels == level
        if mask.sum() > 0:
            level_shat = shat[mask]
            level_means[str(int(level))] = float(level_shat.mean())

    # Check monotonicity (higher level = higher S_hat)
    sorted_levels = sorted(level_means.keys(), key=lambda x: int(x), reverse=True)  # 5,4,3,2,1
    monotonic = all(level_means[sorted_levels[i]] >= level_means[sorted_levels[i+1]]
                     for i in range(len(sorted_levels)-1))

    return {
        "spearman_rho": rho,
        "monotonic": monotonic,
        "level_means": level_means,
        "S_hat_mean": float(shat.mean()),
        "S_hat_std": float(shat.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 models")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model paths to evaluate (e.g., runs/mixed_ws4000_sd8960_a_plus_only_hardened/ckpt_last.pth)")
    parser.add_argument("--test_real_csv", type=str,
                        default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv")
    parser.add_argument("--test_ext_csv", type=str,
                        default="dataset/external_test_washed_processed/test_ext.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    for model_path in args.models:
        model_name = os.path.basename(os.path.dirname(model_path))
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {model_path}")
        print(f"{'='*60}")

        # Load model
        ckpt = torch.load(model_path, map_location=device)
        model = BaselineDualHead(pretrained=False).to(device)

        # Handle different checkpoint formats
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        model_results = {}

        # Evaluate on Test_Real
        if os.path.exists(args.test_real_csv):
            print(f"\n--- Test_Real Evaluation ---")
            spec_real = WoodscapeIndexSpec(
                index_csv=args.test_real_csv,
                img_root=".",
                split_col="split",
                split_value="test",
                global_target="S_full_wgap_alpha50"
            )
            test_real_dataset = WoodscapeTileDataset(spec_real)
            test_real_loader = DataLoader(
                test_real_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=2, pin_memory=False
            )

            test_real_metrics = evaluate_model(model, test_real_loader, device, "Test_Real")
            print(f"  Tile MAE:  {test_real_metrics['tile_mae']:.4f}")
            print(f"  Global MAE: {test_real_metrics['glob_mae']:.4f}")
            print(f"  ρ(S_hat, GT): {test_real_metrics['rho_shat_gt']:.4f}")
            print(f"  ρ(S_hat, S_agg): {test_real_metrics['rho_shat_sagg']:.4f}")
            print(f"  S_hat mean: {test_real_metrics['S_hat_mean']:.4f}")

            model_results["test_real"] = test_real_metrics

        # Evaluate on Test_Ext
        if os.path.exists(args.test_ext_csv):
            print(f"\n--- Test_Ext Evaluation ---")
            test_ext_metrics = evaluate_test_ext(
                model, args.test_ext_csv, ".", device, args.batch_size
            )
            print(f"  Spearman ρ: {test_ext_metrics['spearman_rho']:.4f}")
            print(f"  Monotonic: {test_ext_metrics['monotonic']}")
            print(f"  Level means:")
            for level, mean_val in sorted(test_ext_metrics['level_means'].items(),
                                         key=lambda x: int(x[0]), reverse=True):
                label_map = {"5": "a", "4": "b", "3": "c", "2": "d", "1": "e"}
                label = label_map.get(level, level)
                print(f"    Level {label}: {mean_val:.4f}")

            model_results["test_ext"] = test_ext_metrics

        results[model_name] = model_results

    # Save results
    results_path = "sd_calibration_work/phase2_consistency_gate/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
