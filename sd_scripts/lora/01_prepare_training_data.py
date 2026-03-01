#!/usr/bin/env python3
"""
LoRA Training Data Preparation (v1.1)

Phase 1: Prepare training data for LoRA fine-tuning

Strategy (Route A):
1. Construct "dirt-dominated" training images
   - Inside mask: Preserve original dirty pixels (real dirt appearance)
   - Outside mask: Strong semantic suppression (blur/downsample/desaturate)
2. Generate classified captions with tokens
   - Class tokens: <trans>, <semi>, <opaque>
   - Severity tokens: <sev1>, <sev2>, <sev3>, <sev4>
   - Optical descriptions: "on camera lens, out of focus foreground, subtle glare"
3. Output DreamBooth format (image/caption pairs)

Data isolation: Only use WoodScape Train + Val, NOT Test
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from tqdm import tqdm
import cv2
from collections import defaultdict


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path("/home/yf/soiling_project")
DATASET_ROOT = PROJECT_ROOT / "dataset"

# Data paths
WOODSCAPE_RAW = DATASET_ROOT / "woodscape_raw"
WOODSCAPE_PROCESSED = DATASET_ROOT / "woodscape_processed"
LABELS_INDEX = WOODSCAPE_PROCESSED / "meta" / "labels_index_ablation.csv"
LABELS_TILE_DIR = WOODSCAPE_PROCESSED / "labels_tile"

# Output paths
LORA_DATA_ROOT = PROJECT_ROOT / "sd_scripts" / "lora" / "training_data"
OUTPUT_DIR = LORA_DATA_ROOT / "dreambooth_format"

# Target resolution for SD training
TARGET_RESOLUTION = (512, 512)


# ============================================================================
# Caption Token System (v1.2 - Natural Language Anchors)
# ============================================================================

# Natural language class tokens (SD tokenizer friendly)
CLASS_TOKENS = {
    1: "transparent soiling layer",           # transparent
    2: "semi-transparent dirt smudge",        # semi-transparent
    3: "opaque heavy stains",                 # opaque
}

# Natural language severity tokens (SD tokenizer friendly)
SEVERITY_TOKENS = {
    (0.0, 0.15): "mild",
    (0.15, 0.35): "moderate",
    (0.35, 0.60): "noticeable",
    (0.60, 1.00): "severe",
}

# Fixed optical descriptions (avoid learning background semantics)
OPTICAL_DESCRIPTIONS = [
    "on camera lens",
    "out of focus foreground",
    "subtle glare",
    "background visible",
]


# ============================================================================
# Training Image Construction
# ============================================================================

def construct_training_image(
    dirty_image: np.ndarray,
    mask: np.ndarray,
    method: Literal["blur", "desaturate", "downsample", "random"] = "blur",
    intensity: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Construct "dirt-dominated" training image (v1.2 - with intensity randomization)

    Strategy:
    1. Inside mask: Preserve original dirty pixels (real dirt appearance)
    2. Outside mask: Semantic suppression with randomized intensity
    3. Goal: Make LoRA learn optical features of dirt, not background style

    Args:
        dirty_image: [H, W, 3] WoodScape RGB image (dirty)
        mask: [H, W] 4-class mask (0=clean, 1=trans, 2=semi, 3=opaque)
        method: Background suppression method ("random" for random selection)
        intensity: Suppression intensity (0.0 = none, 1.0 = full, >1.0 = extra strong)
        rng: Random number generator for reproducibility

    Returns:
        [H, W, 3] Training image with suppressed background
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = dirty_image.shape[:2]

    # Identify dirty regions
    mask_bool = (mask > 0)

    # Random method selection if requested
    if method == "random":
        method = rng.choice(["blur", "desaturate", "downsample"])

    # Random intensity variation (0.5 to 1.5) to avoid learning "blur style"
    actual_intensity = np.clip(intensity * rng.uniform(0.5, 1.5), 0.0, 2.0)

    # Construct training image
    I_train = dirty_image.copy().astype(float)

    # Apply background suppression (outside mask)
    if method == "blur":
        # Gaussian blur + downsample + upsample with intensity
        kernel_size = int(51 * actual_intensity)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 101))

        I_bg = cv2.GaussianBlur(I_train, (kernel_size, kernel_size), 0)

        # Downsample factor based on intensity
        downsample_factor = max(2, int(4 / actual_intensity))
        I_bg = cv2.resize(I_bg, (W // downsample_factor, H // downsample_factor), interpolation=cv2.INTER_AREA)
        I_bg = cv2.resize(I_bg, (W, H), interpolation=cv2.INTER_LINEAR)

    elif method == "desaturate":
        # Reduce saturation + light blur
        # Ensure uint8 for cv2.cvtColor
        I_train_uint8 = dirty_image.copy()
        I_bg = cv2.cvtColor(I_train_uint8, cv2.COLOR_RGB2HSV)
        # Saturation suppression based on intensity
        sat_factor = 0.3 * actual_intensity
        I_bg[:, :, 1] = np.clip(I_bg[:, :, 1] * (1.0 - sat_factor), 0, 255)
        I_bg = cv2.cvtColor(I_bg, cv2.COLOR_HSV2RGB)

        # Blur based on intensity
        blur_kernel = int(15 * actual_intensity)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blur_kernel = max(3, min(blur_kernel, 51))
        I_bg = cv2.GaussianBlur(I_bg, (blur_kernel, blur_kernel), 0)

    else:  # "downsample"
        # Low resolution upscale with intensity-based factor
        downsample_factor = max(2, int(8 / actual_intensity))
        I_bg = cv2.resize(I_train, (W // downsample_factor, H // downsample_factor), interpolation=cv2.INTER_AREA)
        I_bg = cv2.resize(I_bg, (W, H), interpolation=cv2.INTER_LINEAR)

    # Merge: inside mask = original, outside mask = suppressed background
    I_train[~mask_bool] = I_bg[~mask_bool]

    return I_train.astype(np.uint8)


# ============================================================================
# Caption Generation
# ============================================================================

def generate_caption(
    mask: np.ndarray,
    S_full: float,
    use_all_optical: bool = True,
) -> str:
    """
    Generate training caption with natural language anchors (v1.2)

    Args:
        mask: [H, W] 4-class mask
        S_full: Severity score (0.0 - 1.0)
        use_all_optical: If True, use all optical descriptions joined

    Returns:
        Caption string with natural language class and severity tokens
    """
    # Get dominant class
    mask_pixels = mask[mask > 0]
    if len(mask_pixels) == 0:
        dominant_class = 0
    else:
        unique, counts = np.unique(mask_pixels, return_counts=True)
        dominant_class = unique[np.argmax(counts)]

    class_token = CLASS_TOKENS.get(dominant_class, "dirt smudges")

    # Get severity token (natural language)
    severity_token = "moderate"  # default
    for (low, high), sev_token in SEVERITY_TOKENS.items():
        if low <= S_full <= high:
            severity_token = sev_token
            break

    # Optical descriptions
    if use_all_optical:
        optical = ", ".join(OPTICAL_DESCRIPTIONS)
    else:
        optical = np.random.choice(OPTICAL_DESCRIPTIONS)

    # Combine tokens with natural language flow
    caption = f"{severity_token} {class_token}, {optical}"
    return caption


# ============================================================================
# Mask and Data Loading
# ============================================================================

def get_mask_from_gt_label(gt_label_path: Path) -> np.ndarray:
    """
    Read 4-class mask from WoodScape GT label file

    WoodScape labels: 0=clean, 1=transparent, 2=semi-transparent, 3=opaque

    Note: The GT label files use sequential naming (0001_FV.png, etc.)
    while the CSV references original IDs. This function handles the mismatch.
    """
    # Try the direct path first
    if gt_label_path.exists():
        mask = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask.astype(np.int32)

    # If not found, try to find by pattern matching
    # Extract the camera type (FV, MVL, MVR, RV, etc.)
    file_id = gt_label_path.stem  # e.g., "0033_FV"
    if "_" in file_id:
        camera_type = file_id.split("_")[-1]
    else:
        camera_type = ""

    # Get the directory
    gt_dir = gt_label_path.parent

    # Try to find a file with the same camera type
    if camera_type:
        pattern = f"*_{camera_type}.png"
        matching_files = sorted(gt_dir.glob(pattern))
        if matching_files:
            # Use a simple heuristic - files are sorted, use middle one or first
            # This is not perfect but should work for getting started
            mask = cv2.imread(str(matching_files[0]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask.astype(np.int32)

    # Last resort: raise error
    raise ValueError(f"Cannot find matching mask for: {gt_label_path}")


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and mask to target resolution

    Image: INTER_AREA for downscaling
    Mask: INTER_NEAREST to preserve class labels
    """
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return image_resized, mask_resized


def load_rgb_image(rgb_path: Path) -> np.ndarray:
    """Load RGB image, convert BGR to RGB"""
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise ValueError(f"Cannot read image: {rgb_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_tile_coverage_from_npz(npz_path: Path) -> np.ndarray:
    """Read tile coverage from existing NPZ file"""
    data = np.load(npz_path)
    return data['tile_cov']


def compute_severity_from_tile_cov(
    tile_cov: np.ndarray,
    class_weights: Tuple = (0.0, 0.15, 0.50, 1.0),
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
) -> float:
    """
    Compute Severity Score from tile coverage

    Returns:
        S_full: Combined severity score (0.0 - 1.0)
    """
    H, W = 8, 8
    w = np.array(class_weights, dtype=np.float32)

    # S_op: Opacity-aware coverage
    p = tile_cov.mean(axis=(0, 1))
    S_op = float((w * p).sum())

    # S_sp: Spatial weighted (gaussian)
    y = np.arange(H)
    x = np.arange(W)
    xx, yy = np.meshgrid(x, y)
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    spatial_sigma = 0.5
    gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * spatial_sigma * H)**2)
    Wmap = gaussian / gaussian.sum()

    tile_weights = tile_cov @ w
    S_sp = float((Wmap * tile_weights).sum())

    # S_dom: Dominance (worst tile)
    S_dom = float(tile_weights.max())

    # S_full: Combined
    S_full = alpha * S_op + beta * S_sp + gamma * S_dom

    return float(np.clip(S_full, 0.0, 1.0))


# ============================================================================
# Stratified Sampling Helper
# ============================================================================

def get_dominant_class_from_mask(mask: np.ndarray) -> int:
    """Get dominant class (1,2,3) from mask, or 0 if clean"""
    mask_pixels = mask[mask > 0]
    if len(mask_pixels) == 0:
        return 0
    unique, counts = np.unique(mask_pixels, return_counts=True)
    return int(unique[np.argmax(counts)])

def stratified_sample(
    df: pd.DataFrame,
    max_samples: Optional[int],
    target_ratios: Dict[int, float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Perform stratified sampling to balance class distribution

    Args:
        df: DataFrame with samples
        max_samples: Maximum total samples (or None for all)
        target_ratios: Target ratios for classes 1,2,3 (default: balanced 1:1:1)
        seed: Random seed

    Returns:
        Sampled DataFrame
    """
    if target_ratios is None:
        target_ratios = {1: 0.33, 2: 0.34, 3: 0.33}

    rng = np.random.default_rng(seed)

    # Group by dominant class (requires precomputed)
    # For now, we'll do simple random sampling with class tracking in main function
    if max_samples and max_samples < len(df):
        return df.sample(n=max_samples, random_state=seed)
    return df


# ============================================================================
# Main Processing Function (v1.2 - with stratified sampling and randomization)
# ============================================================================

def prepare_lora_training_data(
    split_name: str = "train",
    labels_df: pd.DataFrame = None,
    output_dir: Path = OUTPUT_DIR,
    target_resolution: Tuple[int, int] = (512, 512),
    bg_method: Literal["blur", "desaturate", "downsample", "random"] = "random",
    bg_intensity_range: Tuple[float, float] = (0.5, 1.5),
    max_samples: Optional[int] = None,
    stratify: bool = True,
    target_class_ratios: Dict[int, float] = None,
    seed: int = 42,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Prepare LoRA training data for a specific split (v1.2)

    Args:
        split_name: 'train' or 'val'
        labels_df: Labels index DataFrame
        output_dir: Output directory for DreamBooth format data
        target_resolution: Target image resolution
        bg_method: Background suppression method ("random" recommended)
        bg_intensity_range: Range of background suppression intensity (min, max)
        max_samples: Optional limit on number of samples
        stratify: Whether to use stratified sampling for class balance
        target_class_ratios: Target ratios for classes 1,2,3
        seed: Random seed

    Returns:
        (manifest_list, statistics_df)
    """
    print(f"\n{'='*60}")
    print(f"Preparing LoRA Training Data: {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Background method: {bg_method}")
    print(f"Background intensity range: {bg_intensity_range}")
    print(f"Target resolution: {target_resolution}")
    print(f"Stratified sampling: {stratify}")

    rng = np.random.default_rng(seed)

    # Load labels if not provided
    if labels_df is None:
        if not LABELS_INDEX.exists():
            raise FileNotFoundError(f"Labels index not found: {LABELS_INDEX}")
        labels_df = pd.read_csv(LABELS_INDEX)

    # Filter by split
    split_df = labels_df[labels_df['split'] == split_name].copy()
    n_total = len(split_df)

    # Pre-compute dominant classes for stratification
    print("Pre-computing dominant classes...")
    split_df['dominant_class'] = None
    for idx, row in split_df.iterrows():
        npz_path = Path(row['npz_path'])
        if npz_path.exists():
            tile_cov = get_tile_coverage_from_npz(npz_path)
            # Get dominant class (excluding clean class 0)
            tile_mean = tile_cov.mean(axis=(0, 1))
            # Only consider dirty classes (1, 2, 3)
            dirty_coverage = tile_mean[1:]  # classes 1, 2, 3

            # Only mark as having dominant_class if dirty coverage > 1% (lowered threshold)
            if dirty_coverage.sum() > 0.01:
                # Dominant class is 1 + index in dirty_coverage (since we excluded class 0)
                dominant_class = int(np.argmax(dirty_coverage) + 1)
                split_df.loc[idx, 'dominant_class'] = dominant_class

    # Apply stratified sampling if requested
    if stratify:
        if target_class_ratios is None:
            target_class_ratios = {1: 0.33, 2: 0.34, 3: 0.33}

        # First, filter to only samples with valid dominant_class
        dirty_df = split_df[split_df['dominant_class'].notna()].copy()
        n_dirty = len(dirty_df)
        print(f"  Found {n_dirty} samples with dirty content (from {n_total} total)")

        if n_dirty == 0:
            print("  Warning: No dirty samples found, falling back to random sampling")
            stratify = False

    # Initialize n_samples for the fallback case
    if max_samples:
        n_samples = min(max_samples, n_total)
    else:
        n_samples = n_total

    if stratify:
        dirty_df = split_df[split_df['dominant_class'].notna()].copy()
        samples_per_class = {}
        for cls in [1, 2, 3]:
            class_df = dirty_df[dirty_df['dominant_class'] == cls]
            if len(class_df) == 0:
                continue
            # Calculate target count for this class
            if max_samples:
                target_count = int(max_samples * target_class_ratios[cls])
                # Limit to available samples
                target_count = min(target_count, len(class_df))
            else:
                target_count = len(class_df)
            samples_per_class[cls] = class_df.sample(n=target_count, random_state=seed)

        # Combine samples (handle empty case)
        if samples_per_class:
            split_df = pd.concat(samples_per_class.values(), ignore_index=True)
            n_samples = len(split_df)
        else:
            # No valid samples after filtering, fall back to random
            print("  Warning: No samples in stratified sampling, using random sampling")
            if max_samples:
                split_df = split_df.sample(n=min(max_samples, n_total), random_state=seed)
            n_samples = len(split_df)
    else:
        if max_samples:
            split_df = split_df.sample(n=n_samples, random_state=seed)
        else:
            split_df = split_df  # Use all samples
            n_samples = len(split_df)

    print(f"Samples to process: {n_samples} (from {n_total} available)")

    # Create output directory
    images_dir = output_dir / split_name
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    statistics = []

    for idx, row in tqdm(split_df.iterrows(), total=n_samples, desc=f"Processing {split_name}"):
        rgb_path = Path(row['rgb_path'])
        file_id = rgb_path.stem

        # Load NPZ for severity (optional - if not found, compute from mask)
        npz_path = Path(row['npz_path'])
        if npz_path.exists():
            try:
                tile_cov = get_tile_coverage_from_npz(npz_path)
                S_full = compute_severity_from_tile_cov(tile_cov)
            except Exception as e:
                print(f"Warning: Failed to load NPZ {npz_path}: {e}")
                S_full = 0.3  # Default moderate severity
        else:
            # Try to find by pattern (same as mask matching)
            # For now, use default severity
            S_full = 0.3  # Default moderate severity

        # Get mask path (val samples are in train directory)
        if split_name == 'val':
            gt_label_path = WOODSCAPE_RAW / 'train' / "gtLabels" / f"{file_id}.png"
        else:
            gt_label_path = WOODSCAPE_RAW / split_name / "gtLabels" / f"{file_id}.png"

        if not gt_label_path.exists():
            print(f"Warning: GT label not found: {gt_label_path}")
            continue

        # Load data
        try:
            dirty_image = load_rgb_image(rgb_path)
            mask = get_mask_from_gt_label(gt_label_path)

            # Resize to target resolution
            dirty_image_resized, mask_resized = resize_image_and_mask(
                dirty_image, mask, target_resolution
            )

            # Determine dominant class for caption
            dominant_class = get_dominant_class_from_mask(mask_resized)

            # Construct training image with randomized background suppression
            intensity = rng.uniform(bg_intensity_range[0], bg_intensity_range[1])
            training_image = construct_training_image(
                dirty_image_resized,
                mask_resized,
                method=bg_method,
                intensity=intensity,
                rng=rng,
            )

            # Generate caption (with natural language tokens)
            caption = generate_caption(mask_resized, S_full)

            # Save image
            output_image_path = images_dir / f"{file_id}.jpg"
            training_image_bgr = cv2.cvtColor(training_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_image_path), training_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save caption
            output_caption_path = images_dir / f"{file_id}.txt"
            with open(output_caption_path, 'w') as f:
                f.write(caption)

            # Collect statistics
            stats = {
                'file_id': file_id,
                'rgb_path': str(rgb_path),
                'gt_label_path': str(gt_label_path),
                'output_image': str(output_image_path),
                'output_caption': str(output_caption_path),
                'caption': caption,
                'split': split_name,
                'S_full': S_full,
                'dominant_class': dominant_class,
                'bg_method': bg_method,
                'bg_intensity': intensity,
            }

            # Add class distribution
            unique, counts = np.unique(mask_resized, return_counts=True)
            for cls in range(4):
                stats[f'class_{cls}_ratio'] = float(counts[unique == cls].sum() / mask_resized.size) if cls in unique else 0.0

            manifest.append(stats)
            statistics.append(stats)

        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_csv = output_dir / f"manifest_{split_name}.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nSaved manifest: {manifest_csv}")
    print(f"  Total: {len(manifest_df)} samples")

    # Check if we have any samples
    if len(manifest_df) == 0:
        print("\nWarning: No samples were successfully processed!")
        return manifest, manifest_df

    # Print statistics
    print(f"\nCaption examples:")
    for i, row in manifest_df.head(5).iterrows():
        print(f"  [{row['file_id']}] {row['caption']}")

    print(f"\nSeverity distribution:")
    print(f"  min: {manifest_df['S_full'].min():.4f}")
    print(f"  max: {manifest_df['S_full'].max():.4f}")
    print(f"  mean: {manifest_df['S_full'].mean():.4f}")

    print(f"\nClass distribution:")
    for cls in [1, 2, 3]:
        count = (manifest_df['dominant_class'] == cls).sum()
        print(f"  Class {cls}: {count} samples ({count/len(manifest_df)*100:.1f}%)")

    return manifest, manifest_df


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    splits: List[str] = ["train", "val"],
    bg_method: Literal["blur", "desaturate", "downsample", "random"] = "random",
    bg_intensity_range: Tuple[float, float] = (0.5, 1.5),
    train_max_samples: Optional[int] = None,
    val_max_samples: Optional[int] = None,
    stratify: bool = True,
    seed: int = 42,
):
    """
    Main function to prepare all LoRA training data (v1.2)

    Args:
        splits: Splits to process
        bg_method: Background suppression method ("random" recommended for v1.2)
        bg_intensity_range: Range of background suppression intensity
        train_max_samples: Optional limit for train samples
        val_max_samples: Optional limit for val samples
        stratify: Whether to use stratified sampling for class balance
        seed: Random seed
    """
    print("="*60)
    print("LoRA Training Data Preparation (v1.2)")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Background method: {bg_method}")
    print(f"Stratified sampling: {stratify}")

    # Load labels index
    if not LABELS_INDEX.exists():
        raise FileNotFoundError(f"Labels index not found: {LABELS_INDEX}")

    labels_df = pd.read_csv(LABELS_INDEX)
    print(f"\nLoaded labels index: {len(labels_df)} samples")
    print(f"  Split distribution: {labels_df['split'].value_counts().to_dict()}")

    # Process each split
    all_manifests = []

    for split in splits:
        max_samples = train_max_samples if split == "train" else val_max_samples

        manifest, manifest_df = prepare_lora_training_data(
            split_name=split,
            labels_df=labels_df,
            output_dir=OUTPUT_DIR,
            bg_method=bg_method,
            bg_intensity_range=bg_intensity_range,
            max_samples=max_samples,
            stratify=stratify,
            seed=seed,
        )

        all_manifests.extend(manifest)

    # Combined report
    print(f"\n{'='*60}")
    print("Data Preparation Complete")
    print(f"{'='*60}")

    print(f"\nOutput structure:")
    print(f"  {OUTPUT_DIR}/")
    for split in splits:
        count = sum(1 for m in all_manifests if m['split'] == split)
        print(f"    ├── {split}/  ({count} samples)")

    print(f"\nData isolation check:")
    test_df = labels_df[labels_df['split'] == 'test'].copy()
    test_df['file_id'] = test_df['rgb_path'].apply(lambda x: Path(x).stem)
    processed_file_ids = set(m['file_id'] for m in all_manifests)
    test_used = test_df['file_id'].isin(processed_file_ids).sum()

    if test_used > 0:
        print(f"  ✗ Warning: {test_used} test samples were used!")
    else:
        print(f"  ✓ Test data not used (strictly isolated)")

    print(f"\nDreamBooth format ready for training:")
    print(f"  Images: {OUTPUT_DIR}/<split>/*.jpg")
    print(f"  Captions: {OUTPUT_DIR}/<split>/*.txt")
    print(f"  Manifests: {OUTPUT_DIR}/manifest_*.csv")

    print(f"\nNext steps:")
    print(f"  1. Review sample images and captions")
    print(f"  2. Adjust caption tokens if needed")
    print(f"  3. Run LoRA training with diffusers")

    print("\n✓ Preparation complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare LoRA training data for SD fine-tuning"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Splits to process (comma-separated)"
    )
    parser.add_argument(
        "--bg-method",
        type=str,
        default="random",
        choices=["blur", "desaturate", "downsample", "random"],
        help="Background suppression method (random recommended for v1.2)"
    )
    parser.add_argument(
        "--bg-intensity-range",
        type=float,
        nargs=2,
        default=[0.5, 1.5],
        help="Background suppression intensity range (min max)"
    )
    parser.add_argument(
        "--train-max",
        type=int,
        default=None,
        help="Maximum train samples (for Phase 1 testing)"
    )
    parser.add_argument(
        "--val-max",
        type=int,
        default=None,
        help="Maximum val samples"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified sampling for class balance"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    splits = args.splits.split(",")

    main(
        splits=splits,
        bg_method=args.bg_method,
        bg_intensity_range=tuple(args.bg_intensity_range),
        train_max_samples=args.train_max,
        val_max_samples=args.val_max,
        stratify=not args.no_stratify,
        seed=args.seed,
    )
