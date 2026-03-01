#!/usr/bin/env python3
"""
Merge the original rebinned baseline CSV with new ablation severity variants.

This script reads the original labels_index_rebinned_baseline.csv and adds
the new severity variants (s, S_op_only, S_op_sp, S_full, S_full_eta00) from
the .npz files to create a complete CSV for ablation experiments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    proj = Path.home() / "soiling_project"
    meta_root = proj / "dataset/woodscape_processed/meta"

    # Original CSV with proper columns
    orig_csv = meta_root / "labels_index_rebinned_baseline.csv"
    # New CSV with severity variants but missing rgb_path
    new_csv = meta_root / "labels_index.csv"
    # Output CSV with both
    out_csv = meta_root / "labels_index_ablation.csv"

    print(f"Reading original CSV: {orig_csv}")
    df_orig = pd.read_csv(orig_csv)
    print(f"  Original: {len(df_orig)} rows")

    print(f"Reading new CSV: {new_csv}")
    df_new = pd.read_csv(new_csv)
    print(f"  New: {len(df_new)} rows")

    # Get the stem from npz_path to match
    df_orig['stem'] = df_orig['npz_path'].apply(lambda x: Path(x).stem)
    df_new['stem'] = df_new['label_npz'].apply(lambda x: Path(x).stem)

    # Merge on stem
    df_merged = df_orig.merge(
        df_new[['stem', 's', 'S_op_only', 'S_op_sp', 'S_full', 'S_full_eta00',
                'S_op', 'S_sp', 'S_dom_eta00', 'S_dom_eta09']],
        on='stem',
        how='left'
    )

    # Drop the temporary stem column
    df_merged = df_merged.drop(columns=['stem'])

    # Reorder columns for clarity (only use columns that exist)
    base_cols = ['rgb_path', 'npz_path', 'split']
    severity_cols = ['S', 's', 'S_op_only', 'S_op_sp', 'S_full', 'S_full_eta00',
                    'S_op', 'S_sp', 'S_dom_eta00', 'S_dom_eta09']
    other_cols = ['global_level', 'global_bin']

    # Build column list from what actually exists
    cols = base_cols + [c for c in severity_cols if c in df_merged.columns] + other_cols
    df_merged = df_merged[cols]

    # Save merged CSV
    df_merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Saved merged CSV: {out_csv}")
    print(f"  Total: {len(df_merged)} rows")

    # Verify all severity variants are present
    print("\nSeverity variant statistics:")
    for col in ['S', 's', 'S_op_only', 'S_op_sp', 'S_full', 'S_full_eta00']:
        if col in df_merged.columns:
            print(f"  {col}: mean={df_merged[col].mean():.4f}, std={df_merged[col].std():.4f}")

    print("\nSample rows:")
    print(df_merged.head(3).to_string())


if __name__ == "__main__":
    main()