#!/usr/bin/env python3
"""
Visualize prepared LoRA training data

Inspect:
1. Original dirty image
2. Training image (with background suppression)
3. Mask overlay
4. Caption text
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_training_samples(
    manifest_path: Path,
    output_dir: Path,
    num_samples: int = 10,
    seed: int = 42,
):
    """
    Visualize training samples from manifest

    Args:
        manifest_path: Path to manifest CSV
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
        seed: Random seed for sampling
    """
    np.random.seed(seed)

    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} samples from {manifest_path}")

    # Sample randomly
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization for each sample
    for idx, row in sample_df.iterrows():
        file_id = row['file_id']
        print(f"Visualizing {file_id}...")

        # Load training image
        train_img_path = Path(row['output_image'])
        if not train_img_path.exists():
            print(f"  Warning: Image not found: {train_img_path}")
            continue

        train_img = cv2.imread(str(train_img_path))
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

        # Load mask for overlay (if available)
        gt_label_path = Path(row.get('gt_label_path', ''))
        overlay_img = train_img.copy()

        if gt_label_path.exists():
            mask = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize mask to match training image
                h, w = train_img.shape[:2]
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Create overlay
                mask_bool = (mask_resized > 0)
                overlay_img[mask_bool] = overlay_img[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5

        # Create figure
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig)

        # Training image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(train_img)
        ax1.set_title(f"Training Image\n{file_id}", fontsize=12)
        ax1.axis('off')

        # Mask overlay
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(overlay_img)
        ax2.set_title("Mask Overlay (Red=Dirty)", fontsize=12)
        ax2.axis('off')

        # Caption and stats
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        caption = row.get('caption', 'N/A')
        S_full = row.get('S_full', 0)
        bg_method = row.get('bg_method', 'N/A')

        info_text = f"""
Caption:
{caption}

Stats:
• Severity (S_full): {S_full:.4f}
• BG Method: {bg_method}

Class Ratios:
• Clean (0): {row.get('class_0_ratio', 0):.2%}
• Trans (1): {row.get('class_1_ratio', 0):.2%}
• Semi (2): {row.get('class_2_ratio', 0):.2%}
• Opaque (3): {row.get('class_3_ratio', 0):.2%}
        """.strip()

        ax3.text(0.05, 0.95, info_text,
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment='top',
                family='monospace')

        plt.tight_layout()

        # Save
        output_path = output_dir / f"{file_id}_training_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

    print(f"\n✓ Visualizations saved to: {output_dir}")


def compare_bg_methods(
    manifest_paths: dict,
    output_dir: Path,
    file_id: str = None,
):
    """
    Compare different background suppression methods side-by-side

    Args:
        manifest_paths: Dict of {method: manifest_path}
        output_dir: Output directory
        file_id: Specific file_id to compare (None = random)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests
    dfs = {}
    for method, path in manifest_paths.items():
        if path.exists():
            dfs[method] = pd.read_csv(path)
            print(f"Loaded {method}: {len(dfs[method])} samples")

    if not dfs:
        print("No manifests found!")
        return

    # Pick a common file_id
    if file_id is None:
        # Find a file_id present in all manifests
        common_ids = set(dfs[list(dfs.keys())[0]]['file_id'])
        for method_df in dfs.values():
            common_ids &= set(method_df['file_id'])

        if not common_ids:
            print("No common file_ids found across manifests")
            return

        file_id = list(common_ids)[0]

    print(f"Comparing methods for file_id: {file_id}")

    # Create comparison figure
    n_methods = len(dfs)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 8))

    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    for idx, (method, df) in enumerate(dfs.items()):
        row = df[df['file_id'] == file_id]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        # Load training image
        train_img_path = Path(row['output_image'])
        train_img = cv2.imread(str(train_img_path))
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

        # Load mask overlay
        gt_label_path = Path(row.get('gt_label_path', ''))
        overlay_img = train_img.copy()

        if gt_label_path.exists():
            mask = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                h, w = train_img.shape[:2]
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bool = (mask_resized > 0)
                overlay_img[mask_bool] = overlay_img[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5

        # Plot training image
        axes[0, idx].imshow(train_img)
        axes[0, idx].set_title(f"{method.upper()}\n{file_id}", fontsize=12)
        axes[0, idx].axis('off')

        # Plot overlay
        axes[1, idx].imshow(overlay_img)
        axes[1, idx].set_title("Mask Overlay", fontsize=12)
        axes[1, idx].axis('off')

    plt.tight_layout()

    output_path = output_dir / f"{file_id}_bg_method_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize LoRA training data")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Visualize samples command
    viz_parser = subparsers.add_parser("visualize", help="Visualize training samples")
    viz_parser.add_argument("--manifest", type=str, required=True,
                          help="Path to manifest CSV")
    viz_parser.add_argument("--output", type=str, required=True,
                          help="Output directory")
    viz_parser.add_argument("--num-samples", type=int, default=10,
                          help="Number of samples to visualize")

    # Compare methods command
    cmp_parser = subparsers.add_parser("compare", help="Compare background methods")
    cmp_parser.add_argument("--blur", type=str, help="Manifest for blur method")
    cmp_parser.add_argument("--desaturate", type=str, help="Manifest for desaturate method")
    cmp_parser.add_argument("--downsample", type=str, help="Manifest for downsample method")
    cmp_parser.add_argument("--output", type=str, required=True,
                          help="Output directory")
    cmp_parser.add_argument("--file-id", type=str, help="Specific file_id to compare")

    args = parser.parse_args()

    if args.command == "visualize":
        visualize_training_samples(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output),
            num_samples=args.num_samples,
        )
    elif args.command == "compare":
        manifest_paths = {}
        if args.blur:
            manifest_paths['blur'] = Path(args.blur)
        if args.desaturate:
            manifest_paths['desaturate'] = Path(args.desaturate)
        if args.downsample:
            manifest_paths['downsample'] = Path(args.downsample)

        compare_bg_methods(
            manifest_paths=manifest_paths,
            output_dir=Path(args.output),
            file_id=args.file_id,
        )
    else:
        parser.print_help()
