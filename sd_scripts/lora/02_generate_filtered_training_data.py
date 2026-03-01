#!/usr/bin/env python3
"""
Generate training data from filtered manifest

This script regenerates training data using only the samples that passed
quality filtering. It maintains the same configuration as the original
preparation but uses the filtered sample list.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from 01_prepare_training_data
import importlib.util
spec = importlib.util.spec_from_file_location("prepare_training_data", Path(__file__).parent / "01_prepare_training_data.py")
prepare_training_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_training_data)

load_rgb_image = prepare_training_data.load_rgb_image
get_mask_from_gt_label = prepare_training_data.get_mask_from_gt_label
resize_image_and_mask = prepare_training_data.resize_image_and_mask
construct_training_image = prepare_training_data.construct_training_image
generate_caption = prepare_training_data.generate_caption
compute_severity_score = prepare_training_data.compute_severity_from_tile_cov
get_dominant_class_from_mask = prepare_training_data.get_dominant_class_from_mask

import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_filtered_training_data(
    filtered_manifest_path: Path,
    output_dir: Path,
    target_resolution: tuple = (512, 512),
    bg_method: str = "random",
    bg_intensity_range: tuple = (0.5, 1.5),
    seed: int = 42,
):
    """
    Generate training data from filtered manifest

    Args:
        filtered_manifest_path: Path to filtered manifest CSV
        output_dir: Output directory for training data
        target_resolution: Target image resolution
        bg_method: Background suppression method
        bg_intensity_range: Background intensity range
        seed: Random seed
    """
    print("=" * 60)
    print("Generate Filtered Training Data")
    print("=" * 60)

    # Load filtered manifest
    filtered_df = pd.read_csv(filtered_manifest_path)
    n_samples = len(filtered_df)

    print(f"Filtered samples: {n_samples}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    images_dir = output_dir / "train"
    images_dir.mkdir(parents=True, exist_ok=True)

    # RNG
    rng = np.random.default_rng(seed)

    # Statistics
    stats = {
        'total': n_samples,
        'by_class': {1: 0, 2: 0, 3: 0},
        'by_severity': {'mild': 0, 'moderate': 0, 'noticeable': 0, 'severe': 0},
    }

    manifest = []

    # Process each sample
    for idx, row in tqdm(filtered_df.iterrows(), total=n_samples, desc="Generating"):
        file_id = row['file_id']
        rgb_path = Path(row['rgb_path'])
        gt_label_path = Path(row['gt_label_path'])

        if not rgb_path.exists() or not gt_label_path.exists():
            print(f"Warning: Missing files for {file_id}")
            continue

        # Load data
        dirty_image = load_rgb_image(rgb_path)
        mask = get_mask_from_gt_label(gt_label_path)

        # Resize
        dirty_image_resized, mask_resized = resize_image_and_mask(
            dirty_image, mask, target_resolution
        )

        # Compute severity score and get class ratios from NPZ
        npz_path = Path(row['npz_path'])
        class_ratios = np.array([0.25, 0.25, 0.25, 0.25])  # Default fallback

        if npz_path.exists():
            try:
                # Read S_full directly from NPZ (most accurate)
                S_full = get_s_full_from_npz(npz_path)
                # Read class ratios
                class_ratios = get_class_ratios_from_npz(npz_path)
            except Exception as e:
                print(f"Warning: Error reading NPZ for {file_id}: {e}")
                # Fallback to tile_cov method
                tile_cov = get_tile_coverage_from_npz(npz_path)
                S_full = compute_severity_score(tile_cov)
        else:
            S_full = row.get('S_full', 0.5)

        # Determine dominant class
        dominant_class = get_dominant_class_from_mask(mask_resized)

        # Construct training image
        intensity = rng.uniform(bg_intensity_range[0], bg_intensity_range[1])
        training_image = construct_training_image(
            dirty_image_resized,
            mask_resized,
            method=bg_method,
            intensity=intensity,
            rng=rng,
        )

        # Generate caption
        caption = generate_caption(mask_resized, S_full)

        # Save image
        output_image_path = images_dir / f"{file_id}.jpg"
        train_img_bgr = cv2.cvtColor(training_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_image_path), train_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save caption
        output_caption_path = images_dir / f"{file_id}.txt"
        with open(output_caption_path, 'w') as f:
            f.write(caption)

        # Update statistics
        stats['by_class'][dominant_class] += 1

        # Severity bucket
        if S_full < 0.15:
            stats['by_severity']['mild'] += 1
        elif S_full < 0.35:
            stats['by_severity']['moderate'] += 1
        elif S_full < 0.60:
            stats['by_severity']['noticeable'] += 1
        else:
            stats['by_severity']['severe'] += 1

        # Add to manifest
        manifest.append({
            'file_id': file_id,
            'output_image': str(output_image_path),
            'output_caption': str(output_caption_path),
            'caption': caption,
            'S_full': S_full,
            'dominant_class': dominant_class,
            'bg_method': bg_method,
            'class_0_ratio': float(class_ratios[0]),
            'class_1_ratio': float(class_ratios[1]),
            'class_2_ratio': float(class_ratios[2]),
            'class_3_ratio': float(class_ratios[3]),
        })

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = output_dir / "manifest_train.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # Print report
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"\nClass distribution:")
    for cls, count in stats['by_class'].items():
        print(f"  Class {cls}: {count} ({count/stats['total']*100:.1f}%)")
    print(f"\nSeverity distribution:")
    for sev, count in stats['by_severity'].items():
        print(f"  {sev}: {count} ({count/stats['total']*100:.1f}%)")
    print(f"\nManifest saved: {manifest_path}")
    print("=" * 60)


def get_tile_coverage_from_npz(npz_path: Path) -> np.ndarray:
    """Load tile coverage from NPZ file"""
    data = np.load(npz_path)
    # Fixed: correct key is 'tile_cov' not 'tile_coverage'
    return data['tile_cov']


def get_class_ratios_from_npz(npz_path: Path) -> np.ndarray:
    """Load class ratios from NPZ file"""
    data = np.load(npz_path)
    # global_cov_per_class has shape (4,) containing ratios for classes 0-3
    return data['global_cov_per_class']


def get_s_full_from_npz(npz_path: Path) -> float:
    """Load S_full score from NPZ file"""
    data = np.load(npz_path)
    return float(data['S_full'])


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(
        description="Generate training data from filtered manifest"
    )
    parser.add_argument(
        "--filtered-manifest",
        type=str,
        required=True,
        help="Path to filtered manifest CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: filtered_manifest_dir/filtered_training_data)"
    )
    parser.add_argument(
        "--bg-method",
        type=str,
        default="random",
        choices=["blur", "desaturate", "downsample", "random"],
        help="Background suppression method"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        filtered_manifest_path = Path(args.filtered_manifest)
        output_dir = filtered_manifest_path.parent / "filtered_training_data"

    generate_filtered_training_data(
        filtered_manifest_path=Path(args.filtered_manifest),
        output_dir=output_dir,
        bg_method=args.bg_method,
        seed=args.seed,
    )
