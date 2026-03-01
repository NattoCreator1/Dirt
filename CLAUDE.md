# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based machine learning project for detecting and quantifying camera lens soiling/dirt contamination using the WoodScape dataset. The model uses a dual-head ResNet18 architecture that simultaneously predicts:
- **Tile-level classification**: 8x8 grid with 4-class coverage predictions (clean/transparent/semi-transparent/opaque)
- **Global severity score**: Single value [0,1] representing overall soiling severity

### The 4-Class Labeling System

| Class | Name        | GT Value | RGB Color (visualization) |
|-------|-------------|----------|---------------------------|
| 0     | Clean       | 0        | [0, 0, 0] (black)         |
| 1     | Transparent | 1        | [0, 255, 0] (green)       |
| 2     | Semi-transparent | 2    | [0, 0, 255] (blue)        |
| 3     | Opaque      | 3        | [255, 0, 0] (red)         |

Tile coverage outputs are distributions over 4 classes that sum to 1 per tile.

### Global Severity Scores

- **S**: Composite severity score (considers opacity weights, spatial position, dominance pooling)
- **s**: Simple soiling ratio = 1 - clean_ratio

## Architecture

### BaselineDualHead Model (`baseline/models/baseline_dualhead.py`)

```
Input [B,3,640,480]
    ↓
ResNet18 backbone (pretrained on ImageNet)
    ↓
Feature map [B,512,H,W]
    ↓
    ├─→ AdaptiveAvgPool(8x8) → Conv2d(512→4) → Softmax → G_hat [B,4,8,8]
    │                                                   (tile head)
    │
    └─→ AdaptiveAvgPool → FC(512→1) → Sigmoid → S_hat [B,1]
                                                    (global head)

G_hat → SeverityAggregator → S_agg [B,1]
```

### SeverityAggregator

Combines tile predictions into a global score using three components:
- **Opacity-aware mean** (α=0.6): Weighted average using opacity weights [0.0, 0.33, 0.66, 1.0]
- **Spatial-weighted sum** (β=0.3): Gaussian-weighted spatial importance (center-biased)
- **Dominance pooling** (γ=0.1): Softmax over tiles with transparency adjustment

### Partial Supervision Design

The model supports training with mixed supervision:
- **Strong supervision** (D_strong): WoodScape data with tile-level labels → both tile_head and global_head trained
- **Weak supervision** (D_weak): External data without tile labels → only global_head trained

Loss: `L_total = L_tile + λ * L_global + μ * L_cons`

### Temporal Modeling (Planned Extension)

**Core Insight**: Soiling is fixed relative to the camera lens, while the scene changes as the vehicle moves. This physical property can be leveraged via temporal modeling to improve soiling detection accuracy.

**Three-Stage Training Strategy**:

1. **Stage 1** (Completed): Single-frame strong supervision pre-training
   - Data: Train_Real (3200) + Val_Real (800)
   - Output: Pre-trained backbone + dual-head model
   - Baseline: Tile MAE=0.0396, Global MAE=0.0187

2. **Stage 2** (Planned): Temporal module + SD_seq training
   - Data: SD_seq synthetic sequences with strong temporal labels
   - New loss: `L_temp = (1/B(T-1)) · Σ |(Ŝ_{t+1} - Ŝ_t) - (S'_{t+1} - S'_t)|`
   - Output: Temporal-aware model

3. **Stage 3** (Planned): Mixed fine-tuning
   - Data: Train_Real (single-frame) + SD_seq (temporal) mixed by batch
   - Goal: Maintain strong annotation accuracy while gaining temporal robustness

**Temporal Module Options**:

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| LSTM/GRU | Mature, handles long sequences | More parameters, slower | Long-sequence modeling |
| Temporal Conv | Lightweight, fast inference | Short sequences only | Real-time applications |
| Temporal Attn | Interpretable, parallel | Higher compute cost | When interpretability needed |

**Test Strategy**:
- Test_Real: Single-frame mode (seq_len=1) → verify static accuracy maintained
- Test_Ext: Temporal mode → verify generalization improvement
- Test_Syn: Temporal mode → verify temporal learning works as expected

See `docs/temporal_module_design.md` for detailed design documentation.

## Data Pipeline

Execute scripts in numerical order for initial dataset setup:

```bash
# 1. Verify dataset integrity
python scripts/00_sanity_check.py

# 2. Create CSV manifest from raw WoodScape data
python scripts/01_manifest_woodscape.py

# 3. Split dataset into train/val/test (70/15/15)
# Creates: Test_Real/Val_Real/Train_Real = 1000/800/3200
python scripts/02_split_by_list.py

# 4. Generate tile-level (8x8x4) and global labels
python scripts/03_build_labels_tile_global.py

# 5. Re-bin global severity levels (optional)
python scripts/04_rebin_global_level.py

# 6. Build final index CSV for baseline training
python scripts/08_build_baseline_index_from_splits.py
```

### Custom Data Processing

```bash
# Extract frames from 6-view videos
python scripts/05_extract_clean_framesNsplit6.py \
    --video_dir dataset/my_clean_video_raw \
    --out_frames_dir dataset/my_clean_frames \
    --out_manifest_dir dataset/my_clean_manifests \
    --delta_t 1.0

# Crop frames to 4:3 aspect ratio
python scripts/06_crop_clean_frames_to_4by3.py \
    --in_root dataset/my_clean_frames \
    --out_root dataset/my_clean_frames_4by3

# Resize to 640x480
python scripts/resize_4by3_to_640480.py \
    --in_root dataset/my_clean_frames_4by3 \
    --out_root dataset/my_clean_frames_4by3_640x480

# Prepare external test set (occlusion levels a-e)
python scripts/07_prepare_external_testset.py \
    --in_root dataset/my_soiling_raw \
    --out_root dataset/my_external_test
```

## Training

### Basic Training Command

```bash
python baseline/train_baseline.py \
    --index_csv dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv \
    --img_root . \
    --split_col split \
    --train_split train \
    --val_split val \
    --global_target S \
    --epochs 40 \
    --batch_size 32 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --lambda_glob 1.0 \
    --run_name model1_r18_640x480 \
    --out_root baseline/runs
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--index_csv` | Required | Path to index CSV with columns for image paths, npz paths, split |
| `--img_root` | dataset/woodscape_raw | Root directory for resolving relative image paths |
| `--labels_tile_dir` | None | Directory for NPZ files (auto-derived if CSV has npz_path column) |
| `--split_col` | None | Column name for splits (auto-detects: split/subset/domain/set) |
| `--train_split` | Required | Value for training split (e.g., "train", "Train_Real") |
| `--val_split` | Required | Value for validation split |
| `--global_target` | "S" | Global target: "S" (composite) or "s" (simple ratio) |
| `--epochs` | 40 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 3e-4 | Learning rate for AdamW |
| `--weight_decay` | 1e-2 | AdamW weight decay |
| `--lambda_glob` | 1.0 | Weight for global loss |
| `--mu_cons` | 0.0 | Weight for consistency loss (S_hat vs S_agg) |
| `--amp` | False | Enable automatic mixed precision (AMP) |
| `--seed` | 42 | Random seed |
| `--num_workers` | 4 | DataLoader workers |

### Training Outputs

Saved to `baseline/runs/{run_name}/`:
- `ckpt_best.pth`: Best model checkpoint (by validation global MAE)
- `ckpt_last.pth`: Latest checkpoint
- `metrics.csv`: Per-epoch metrics
- `run_args.json`: Training configuration

### Metrics Tracked

- `train_loss`, `val_loss`: Total training/validation loss
- `val_tile_mae`: Tile coverage MAE
- `val_glob_mae`, `val_glob_rmse`: Global score MAE/RMSE
- `val_gap_mae`: Consistency gap between S_hat (global head) and S_agg (aggregator)
- `rho_shat_sagg`: Spearman correlation between predicted and aggregated scores
- `rho_shat_gt`: Spearman correlation between predicted and ground truth

## Dataset Structure

```
dataset/
├── woodscape_raw/              # Original WoodScape dataset
│   ├── train/                  # 4000 training samples
│   │   ├── gtLabels/           # PNG segmentation masks (values 0-3)
│   │   ├── rbgImages/          # RGB images (note: typo in original)
│   │   └── rbgLabels/          # RGB visualization of labels
│   └── test/                   # 1000 official test samples (KEEP INDEPENDENT)
│
├── woodscape_processed/        # Processed dataset
│   ├── labels_tile/            # NPZ files with tile_cov [8,8,4] + global scores
│   └── meta/                   # Metadata CSVs
│       ├── manifest_woodscape.csv
│       ├── labels_index.csv
│       ├── labels_index_rebinned.csv
│       ├── labels_index_rebinned_baseline.csv
│       └── splits/
│           ├── train_real.csv
│           ├── val_real.csv
│           └── test_real.csv
│
├── my_clean_frames/            # Custom clean frame data
├── my_clean_frames_4by3/       # 4:3 cropped clean frames
└── my_external_test/           # External test set with occlusion levels a-e
```

## Index CSV Format

The dataset uses CSV index files for flexible data splitting. Example columns:
- `rgb_path` or `image_path`: Relative or absolute path to RGB image
- `npz_path` or `tile_npz`: Path to NPZ file with tile labels
- `split`: Train/val/test split designation
- `S`, `s`: Global severity scores
- `global_level`, `global_bin`: Discretized severity levels

The dataset class (`WoodscapeTileDataset`) auto-detects column names by pattern matching.

## Implementation Notes

- **Path resolution**: Relative paths in index CSV are resolved relative to `--img_root`
- **NPZ loading**: If CSV lacks npz_path column, NPZ files are derived from `labels_tile_dir/{stem}.npz`
- **Normalization**: Uses ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **Image resize**: Default 640x480 using INTER_AREA
- **No formal test suite**: Validation is built into training loop via `evaluate()`
