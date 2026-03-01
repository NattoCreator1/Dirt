#!/usr/bin/env python3
"""
Command-line Quality Filter for LoRA Training Data (WSL Compatible)

This tool provides a simple CLI interface for filtering training samples.
It generates preview images that can be viewed in an external image viewer,
then uses command-line input for accept/reject decisions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from typing import List, Dict, Set
import json
from datetime import datetime


class CLIQualityFilter:
    """
    Command-line quality filter for training data (WSL compatible)
    """

    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path,
        preview_dir: Path = None,
        batch_size: int = 20,
    ):
        """
        Initialize the CLI filter

        Args:
            manifest_path: Path to manifest CSV
            output_dir: Output directory for results
            preview_dir: Directory for preview images
            batch_size: Number of samples to review per batch
        """
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.batch_size = batch_size

        # Preview directories
        if preview_dir is None:
            self.preview_base_dir = output_dir / "previews"
        else:
            self.preview_base_dir = preview_dir

        # Subdirectories for organization
        self.preview_current = self.preview_base_dir / "current"
        self.preview_accepted = self.preview_base_dir / "accepted"
        self.preview_rejected = self.preview_base_dir / "rejected"

        # Create all directories
        self.preview_current.mkdir(parents=True, exist_ok=True)
        self.preview_accepted.mkdir(parents=True, exist_ok=True)
        self.preview_rejected.mkdir(parents=True, exist_ok=True)

        # Load manifest
        self.df = pd.read_csv(manifest_path)
        self.total_samples = len(self.df)
        self.current_idx = 0

        # Tracking
        self.accepted: Set[str] = set()
        self.rejected: Set[str] = set()
        self.reasons: Dict[str, str] = {}

        # Progress file
        self.progress_file = output_dir / "filter_progress.json"

        # Load previous progress if exists
        self._load_progress()

        # Create output directory (preview subdirectories already created above)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_progress(self):
        """Load previous filtering progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.accepted = set(data.get('accepted', []))
                self.rejected = set(data.get('rejected', []))
                self.reasons = data.get('reasons', {})

                # Find first unreviewed sample
                reviewed = self.accepted | self.rejected
                for i, row in self.df.iterrows():
                    if row['file_id'] not in reviewed:
                        self.current_idx = i
                        break
                else:
                    self.current_idx = len(self.df)

            print(f"Loaded progress: {len(self.accepted)} accepted, {len(self.rejected)} rejected")
        else:
            print("Starting new filtering session")

    def _save_progress(self):
        """Save current progress"""
        data = {
            'accepted': list(self.accepted),
            'rejected': list(self.rejected),
            'reasons': self.reasons,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _create_preview_image(self, sample: Dict, img_size: int = 400) -> np.ndarray:
        """Create preview image for sample"""
        # Load training image
        train_img_path = Path(sample['output_image'])
        if not train_img_path.exists():
            return None

        train_img = cv2.imread(str(train_img_path))
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

        # Load mask for overlay
        gt_label_path = Path(sample.get('gt_label_path', ''))
        overlay_img = train_img.copy()

        if gt_label_path.exists():
            mask = cv2.imread(str(gt_label_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                h, w = train_img.shape[:2]
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Create overlay with semi-transparent red
                mask_bool = (mask_resized > 0)
                overlay_img = overlay_img.astype(float)
                overlay_img[mask_bool] = overlay_img[mask_bool] * 0.6 + np.array([255, 0, 0]) * 0.4
                overlay_img = overlay_img.astype(np.uint8)

        # Resize images
        h, w = train_img.shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(train_img, (new_w, new_h))
        overlay_resized = cv2.resize(overlay_img, (new_w, new_h))

        # Create side-by-side view
        display = np.hstack([img_resized, overlay_resized])

        # Add info panel
        panel_h = 80
        panel = np.ones((panel_h, display.shape[1], 3), dtype=np.uint8) * 250

        # Add text info using cv2.putText
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 20

        # File ID and class info
        text = f"ID: {sample['file_id']}"
        cv2.putText(panel, text, (10, y_offset), font, 0.5, (0, 0, 0), 1)

        y_offset += 20
        text = f"C1:{sample.get('class_1_ratio', 0):.1%} C2:{sample.get('class_2_ratio', 0):.1%} C3:{sample.get('class_3_ratio', 0):.1%}"
        cv2.putText(panel, text, (10, y_offset), font, 0.4, (0, 0, 0), 1)

        y_offset += 20
        text = f"Sev: {sample.get('S_full', 0):.2f} | {sample.get('caption', '')[:50]}"
        cv2.putText(panel, text, (10, y_offset), font, 0.35, (0, 0, 150), 1)

        # Combine
        result = np.vstack([display, panel])
        return result

    def _generate_batch_previews(self, start_idx: int) -> List[Dict]:
        """Generate preview images for a batch"""
        batch_samples = []

        # Clear current directory
        for existing_file in self.preview_current.glob("*_preview.jpg"):
            existing_file.unlink()

        end_idx = min(start_idx + self.batch_size, self.total_samples)

        for idx in range(start_idx, end_idx):
            row = self.df.iloc[idx]
            file_id = row['file_id']

            # Skip if already reviewed
            if file_id in self.accepted or file_id in self.rejected:
                continue

            # Create preview
            preview_img = self._create_preview_image(row)
            if preview_img is not None:
                preview_path = self.preview_current / f"{file_id}_preview.jpg"
                cv2.imwrite(str(preview_path), cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))

                batch_samples.append({
                    'idx': idx,
                    'file_id': file_id,
                    'preview_path': preview_path,
                    'row': row,
                })

        return batch_samples

    def _review_batch(self, batch_samples: List[Dict]) -> None:
        """Review a batch of samples"""
        if not batch_samples:
            print("No new samples in this batch.")
            return

        print("\n" + "=" * 60)
        print(f"Batch Preview ({len(batch_samples)} samples)")
        print("=" * 60)
        print(f"\nPreview images saved to: {self.preview_current}")
        print("  - Current batch: previews/current/")
        print("  - Accepted: previews/accepted/")
        print("  - Rejected: previews/rejected/")
        print("\nOpen the preview images in your image viewer, then enter decisions below.")
        print("\nCommands:")
        print("  a <file_id>   - Accept sample")
        print("  r <file_id>   - Reject sample")
        print("  ra            - Review all (accept all in batch)")
        print("  rr            - Reject all (reject all in batch)")
        print("  s             - Skip batch (review later)")
        print("  q             - Quit and save")
        print("\nFor rejection, you'll be asked for a reason:")
        print("  1 - Mask inaccurate")
        print("  2 - Image quality poor")
        print("  3 - No visible dirt")
        print("  4 - Other")
        print("=" * 60)

        # Show list of file IDs in batch
        print("\nSamples in this batch:")
        for i, sample in enumerate(batch_samples, 1):
            print(f"  {i}. {sample['file_id']}")

        while True:
            cmd = input(f"\nCommand (a/r/ra/rr/s/q) [{len(batch_samples)} samples]: ").strip()

            # Convert command part to lowercase, preserve file_id case
            cmd_lower = cmd.lower()

            if cmd_lower == 'q':
                # Clean up current directory before quitting
                for preview_file in self.preview_current.glob("*_preview.jpg"):
                    preview_file.unlink()
                self.current_idx = batch_samples[-1]['idx'] + 1
                return 'quit'

            elif cmd_lower == 's':
                # Clean up current directory before skipping
                for preview_file in self.preview_current.glob("*_preview.jpg"):
                    preview_file.unlink()
                self.current_idx = batch_samples[-1]['idx'] + 1
                return 'skip'

            elif cmd_lower == 'ra':
                # Accept all
                for sample in batch_samples:
                    self.accepted.add(sample['file_id'])
                    print(f"  Accepted: {sample['file_id']}")

                    # Move preview to accepted directory
                    preview_src = sample['preview_path']
                    preview_dst = self.preview_accepted / f"{sample['file_id']}_preview.jpg"
                    if preview_src.exists():
                        preview_src.rename(preview_dst)

                self.current_idx = batch_samples[-1]['idx'] + 1
                return 'done'

            elif cmd_lower == 'rr':
                # Reject all with reason
                reason = input("Rejection reason (1-4): ").strip()
                reason_map = {'1': 'mask_inaccurate', '2': 'poor_quality', '3': 'no_dirt', '4': 'other'}
                reason_code = reason_map.get(reason, 'other')

                for sample in batch_samples:
                    self.rejected.add(sample['file_id'])
                    self.reasons[sample['file_id']] = reason_code
                    print(f"  Rejected: {sample['file_id']} ({reason_code})")

                    # Move preview to rejected directory
                    preview_src = sample['preview_path']
                    preview_dst = self.preview_rejected / f"{sample['file_id']}_preview.jpg"
                    if preview_src.exists():
                        preview_src.rename(preview_dst)

                self.current_idx = batch_samples[-1]['idx'] + 1
                return 'done'

            elif cmd_lower.startswith('a '):
                # Accept specific
                file_id = cmd[2:].strip()  # Keep original case
                sample = next((s for s in batch_samples if s['file_id'] == file_id), None)
                if sample:
                    self.accepted.add(file_id)
                    print(f"  Accepted: {file_id}")

                    # Move preview to accepted directory
                    preview_src = sample['preview_path']
                    preview_dst = self.preview_accepted / f"{file_id}_preview.jpg"
                    if preview_src.exists():
                        preview_src.rename(preview_dst)

                    # Remove from batch
                    batch_samples.remove(sample)
                    if not batch_samples:
                        return 'done'
                else:
                    print(f"  Error: file_id '{file_id}' not found in batch")

            elif cmd_lower.startswith('r '):
                # Reject specific
                file_id = cmd[2:].strip()  # Keep original case
                sample = next((s for s in batch_samples if s['file_id'] == file_id), None)
                if sample:
                    reason = input(f"Rejection reason for {file_id} (1-4): ").strip()
                    reason_map = {'1': 'mask_inaccurate', '2': 'poor_quality', '3': 'no_dirt', '4': 'other'}
                    reason_code = reason_map.get(reason, 'other')

                    self.rejected.add(file_id)
                    self.reasons[file_id] = reason_code
                    print(f"  Rejected: {file_id} ({reason_code})")

                    # Move preview to rejected directory
                    preview_src = sample['preview_path']
                    preview_dst = self.preview_rejected / f"{file_id}_preview.jpg"
                    if preview_src.exists():
                        preview_src.rename(preview_dst)

                    # Remove from batch
                    batch_samples.remove(sample)
                    if not batch_samples:
                        return 'done'
                else:
                    print(f"  Error: file_id '{file_id}' not found in batch")

            else:
                print("  Unknown command. Try again.")

    def run(self):
        """Run CLI filtering session"""
        print("=" * 60)
        print("CLI Quality Filter (WSL Compatible)")
        print("=" * 60)
        print(f"Total samples: {self.total_samples}")
        print(f"Already reviewed: {len(self.accepted) + len(self.rejected)}")
        print(f"Remaining: {self.total_samples - len(self.accepted) - len(self.rejected)}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 60)

        while self.current_idx < self.total_samples:
            # Generate batch previews
            batch_samples = self._generate_batch_previews(self.current_idx)

            if not batch_samples:
                self.current_idx += self.batch_size
                continue

            # Review batch
            result = self._review_batch(batch_samples)

            if result == 'quit':
                print("\nSaving progress and exiting...")
                self._save_progress()
                break

            # Save progress after each batch
            self._save_progress()

        # Final save and report
        self._save_progress()
        self._generate_report()

    def _generate_report(self):
        """Generate filtering report and filtered manifest"""
        print("\n" + "=" * 60)
        print("Filtering Report")
        print("=" * 60)

        reviewed = len(self.accepted) + len(self.rejected)
        acceptance_rate = len(self.accepted) / reviewed * 100 if reviewed > 0 else 0

        print(f"Total samples: {self.total_samples}")
        print(f"Reviewed: {reviewed}")
        print(f"Accepted: {len(self.accepted)} ({acceptance_rate:.1f}%)")
        print(f"Rejected: {len(self.rejected)} ({100 - acceptance_rate:.1f}%)")

        # Rejection reasons
        if self.rejected:
            print("\nRejection reasons:")
            reason_counts = {}
            for reason in self.reasons.values():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count}")

        # Generate filtered manifest
        filtered_df = self.df[self.df['file_id'].isin(self.accepted)].copy()

        # Save filtered manifest
        filtered_manifest_path = self.output_dir / "manifest_filtered.csv"
        filtered_df.to_csv(filtered_manifest_path, index=False)
        print(f"\nFiltered manifest saved: {filtered_manifest_path}")
        print(f"Filtered samples: {len(filtered_df)}")

        # Save rejected list
        rejected_df = self.df[self.df['file_id'].isin(self.rejected)].copy()
        rejected_manifest_path = self.output_dir / "manifest_rejected.csv"
        rejected_df.to_csv(rejected_manifest_path, index=False)
        print(f"Rejected manifest saved: {rejected_manifest_path}")

        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Command-line quality filter for LoRA training data (WSL compatible)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./filter_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Directory for preview images (default: output_dir/previews)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of samples per batch (default: 20)"
    )

    args = parser.parse_args()

    filter_tool = CLIQualityFilter(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output),
        preview_dir=Path(args.preview_dir) if args.preview_dir else None,
        batch_size=args.batch_size,
    )

    filter_tool.run()


if __name__ == "__main__":
    main()
