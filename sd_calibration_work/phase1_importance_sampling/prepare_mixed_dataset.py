#!/usr/bin/env python3
"""
Prepare Mixed Dataset (WoodScape + SD)

Purpose:
1. Load WoodScape train data and SD data
2. Combine into single CSV for training
3. Save mixed dataset manifest

Usage:
    python prepare_mixed_dataset.py
"""

import sys
import os
import pandas as pd
from pathlib import Path

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

# Paths
WS_CSV = 'dataset/woodscape_processed/meta/labels_index_ablation.csv'
SD_CSV = 'sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv'
OUTPUT_DIR = Path('sd_calibration_work/phase1_importance_sampling')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Prepare Mixed Dataset (WoodScape + SD 8960)")
print("=" * 80)
print()

# Load WoodScape
print("【1. Load WoodScape】")
ws_df = pd.read_csv(WS_CSV)
ws_train = ws_df[ws_df['split'] == 'train'].copy()
print(f"WoodScape Train: {len(ws_train)}")

# Check columns
print(f"WoodScape columns: {list(ws_train.columns)}")

# For WoodScape, we need the required columns
# Typical columns: rgb_path, npz_path, split, S, s, S_full_wgap_alpha50, etc.
required_cols = ['rgb_path', 'npz_path', 'S_full_wgap_alpha50']
for col in required_cols:
    if col not in ws_train.columns:
        print(f"Warning: {col} not in WoodScape CSV")

# Load SD
print()
print("【2. Load SD】")
sd_df = pd.read_csv(SD_CSV)
print(f"SD: {len(sd_df)}")
print(f"SD columns: {list(sd_df.columns)}")

# For SD, we need to check the columns too
# SD CSV might have different column names
# Need to map to WoodScape format

# Add source column
ws_train['source'] = 'woodscape'
sd_df['source'] = 'sd'

# Standardize column names for SD
# SD CSV likely has: image_path, npz_path
# Need to rename to match Woodscape format
sd_renamed = sd_df.rename(columns={'image_path': 'rgb_path'})

# For SD, S_full_wgap_alpha50 will be loaded from NPZ during training
# So we don't need it in the CSV

# Check common columns
print()
print("【3. Standardize columns】")
print("WoodScape columns:", list(ws_train.columns)[:10], "...")
print("SD columns:", list(sd_renamed.columns)[:10], "...")

# Select common columns
# IMPORTANT: Don't include S_full_wgap_alpha50 for SD data
# This forces the dataset class to load it from NPZ files
ws_cols = ['rgb_path', 'npz_path', 'source', 'S_full_wgap_alpha50']
sd_cols = ['rgb_path', 'npz_path', 'source']  # No S column - will load from NPZ

# Create subset dataframes
ws_subset = ws_train[ws_cols].copy()
sd_subset = sd_renamed[sd_cols].copy()

print()
print("【4. Create mixed dataset】")
print(f"WoodScape: {len(ws_subset)}")
print(f"SD: {len(sd_subset)}")

# Concatenate - pandas will fill missing columns with NaN
# But since we didn't include S_full_wgap_alpha50 in sd_cols, it won't be in SD rows
mixed_df = pd.concat([ws_subset, sd_subset], ignore_index=True)
print(f"Mixed total: {len(mixed_df)}")

# Add split column (all train)
mixed_df['split'] = 'train'

# Add index column
mixed_df['idx'] = range(len(mixed_df))

print()
print("【5. Save mixed dataset】")
output_csv = OUTPUT_DIR / 'mixed_ws4000_sd8960.csv'
mixed_df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
print()

# Print summary
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Total samples: {len(mixed_df)}")
print(f"  WoodScape: {len(ws_subset)} ({len(ws_subset)/len(mixed_df)*100:.1f}%)")
print(f"  SD: {len(sd_subset)} ({len(sd_subset)/len(mixed_df)*100:.1f}%)")
print()
print(f"Output file: {output_csv}")
print()
print("Next steps:")
print(f"1. Weights: {OUTPUT_DIR / 'weights_8960_correct.npz'}")
print(f"2. Train: python train_d_only.py --mixed_csv {output_csv}")
