#!/usr/bin/env python3
"""
生成带簇信息的 Test_Ext 预测 CSV（修正版）

簇 = 子文件夹（例如 occlusion_a/LJVA4PALXNT130670_20240220090334718_split_3/）
"""
import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from baseline.models.baseline_dualhead import BaselineDualHead


class ExternalTestDatasetWithClusters(Dataset):
    """Dataset that uses subdirectory as cluster ID."""
    def __init__(self, img_root="dataset/external_test_washed"):
        self.img_root = Path(img_root)

        # Map level name to numeric value
        self.level_map = {'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1}

        # Collect all images with their cluster info
        self.samples = []
        for level_dir in sorted(self.img_root.glob("occlusion_*")):
            level_name = level_dir.name.split('_')[-1]
            if level_name in self.level_map:
                level_value = self.level_map[level_name]
                
                # Each subdirectory is a cluster
                for cluster_dir in level_dir.glob("*"):
                    if not cluster_dir.is_dir():
                        continue
                    
                    # Use relative path from img_root as cluster ID
                    cluster_id = str(cluster_dir.relative_to(self.img_root))
                    
                    for img_path in cluster_dir.glob("*.jpg"):
                        self.samples.append((str(img_path), cluster_id, level_value))

        print(f"Found {len(self.samples)} images")
        
        # Print cluster statistics
        cluster_df = pd.DataFrame(self.samples, columns=['path', 'cluster', 'level'])
        n_clusters = cluster_df['cluster'].nunique()
        print(f"Number of clusters: {n_clusters}")
        print(f"Average images per cluster: {len(self.samples) / n_clusters:.1f}")

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cluster_id, level = self.samples[idx]

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((640, 480), Image.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1))

        return {
            'image': img,
            'image_path': img_path,
            'cluster_id': cluster_id,
            'occlusion_level': level,
        }


def evaluate_and_save_predictions(model, dataset, device, output_csv, batch_size=16):
    """Evaluate model and save predictions with cluster info."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=False)

    all_predictions = []
    all_paths = []
    all_clusters = []
    all_levels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            images = batch['image'].to(device)
            paths = batch['image_path']
            cluster_ids = batch['cluster_id']
            levels = batch['occlusion_level']

            # Forward pass
            output = model(images)
            S_hat = output["S_hat"]

            # Store results
            all_predictions.extend(S_hat.cpu().numpy().flatten())
            all_paths.extend(paths)
            all_clusters.extend(cluster_ids)
            all_levels.extend(levels.cpu().numpy())

    # Create DataFrame and save
    df = pd.DataFrame({
        'image_path': all_paths,
        'cluster_id': all_clusters,
        'occlusion_level': all_levels,
        'S_hat': all_predictions,
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions to {output_csv}")

    # Print cluster statistics
    print(f"\n簇统计:")
    cluster_counts = df.groupby('cluster_id').size()
    print(f"  总簇数: {len(cluster_counts)}")
    print(f"  平均每簇: {len(df) / len(cluster_counts):.1f} 张")
    print(f"  簇大小分布:")
    print(f"    最小: {cluster_counts.min()}")
    print(f"    最大: {cluster_counts.max()}")
    print(f"    中位数: {cluster_counts.median():.1f}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Test_Ext predictions with cluster info")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--img_root", type=str,
                        default="dataset/external_test_washed",
                        help="Root directory for external test images")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create model
    model = BaselineDualHead(pretrained=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    print(f"Loaded model")

    # Create dataset
    print(f"Loading Test_Ext from {args.img_root}")
    dataset = ExternalTestDatasetWithClusters(args.img_root)
    print(f"Dataset size: {len(dataset)}")

    # Evaluate and save
    evaluate_and_save_predictions(model, dataset, device, args.output_csv, args.batch_size)


if __name__ == "__main__":
    main()
