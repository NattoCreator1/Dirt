#!/usr/bin/env python3
"""
Interactive Quality Filter for LoRA Training Data

This tool provides an interactive interface to manually review and filter
training samples based on mask accuracy and image quality.

Features:
- Keyboard navigation (arrows, space to mark, q to quit)
- Display original image + mask overlay + caption
- Save filtering results to CSV
- Generate filtered manifest for training
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from typing import List, Dict, Set
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


class InteractiveQualityFilter:
    """
    Interactive quality filter for training data
    """

    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path,
        save_interval: int = 10,
    ):
        """
        Initialize the filter

        Args:
            manifest_path: Path to manifest CSV
            output_dir: Output directory for results
            save_interval: Save progress every N samples
        """
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.save_interval = save_interval

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

        # For button interaction
        self.current_decision = None

        # Load previous progress if exists
        self._load_progress()

        # Create output directory
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

    def _load_sample(self, idx: int) -> Dict:
        """Load sample data for display"""
        if idx >= len(self.df):
            return None

        row = self.df.iloc[idx]

        # Load training image
        train_img_path = Path(row['output_image'])
        if not train_img_path.exists():
            return None

        train_img = cv2.imread(str(train_img_path))
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

        # Load mask for overlay
        gt_label_path = Path(row.get('gt_label_path', ''))
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

        return {
            'image': train_img,
            'overlay': overlay_img,
            'file_id': row['file_id'],
            'caption': row.get('caption', 'N/A'),
            'S_full': row.get('S_full', 0),
            'bg_method': row.get('bg_method', 'N/A'),
            'class_0_ratio': row.get('class_0_ratio', 0),
            'class_1_ratio': row.get('class_1_ratio', 0),
            'class_2_ratio': row.get('class_2_ratio', 0),
            'class_3_ratio': row.get('class_3_ratio', 0),
        }

    def _create_display_image(self, sample: Dict, img_size: int = 800) -> np.ndarray:
        """Create combined display image"""
        # Resize images
        h, w = sample['image'].shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(sample['image'], (new_w, new_h))
        overlay_resized = cv2.resize(sample['overlay'], (new_w, new_h))

        # Create side-by-side view
        display = np.hstack([img_resized, overlay_resized])

        # Add info panel
        panel_h = 120
        panel = np.ones((panel_h, display.shape[1], 3), dtype=np.uint8) * 255

        # Add text info
        info_lines = [
            f"File: {sample['file_id']}",
            f"Class Ratios: C0:{sample['class_0_ratio']:.1%} C1:{sample['class_1_ratio']:.1%} C2:{sample['class_2_ratio']:.1%} C3:{sample['class_3_ratio']:.1%}",
            f"Severity: {sample['S_full']:.3f} | BG Method: {sample['bg_method']}",
        ]

        y_offset = 20
        for line in info_lines:
            cv2.putText(panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25

        # Add caption (wrapped)
        caption = sample['caption']
        if len(caption) > 80:
            caption = caption[:77] + "..."
        cv2.putText(panel, f"Caption: {caption}", (10, y_offset + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        # Add instructions
        cv2.putText(panel, "[A]ccept [R]eject [S]kip [Q]uit",
                   (display.shape[1] - 350, panel_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Combine
        result = np.vstack([display, panel])
        return result

    def _show_sample(self, sample: Dict) -> str:
        """
        Display sample and get user input using matplotlib

        Returns:
            'accept', 'reject', 'skip', 'prev', 'next', or 'quit'
        """
        display_img = self._create_display_image(sample, img_size=600)

        # Create figure with buttons
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.canvas.manager.set_window_title(f"Quality Filter [{self.current_idx + 1}/{self.total_samples}]")

        # Display image
        ax.imshow(display_img)
        ax.axis('off')

        # Reset decision
        self.current_decision = None

        # Button callbacks
        def accept(event):
            self.current_decision = 'accept'
            plt.close(fig)

        def reject(event):
            self.current_decision = 'reject'
            plt.close(fig)

        def skip(event):
            self.current_decision = 'skip'
            plt.close(fig)

        def prev_sample(event):
            self.current_decision = 'prev'
            plt.close(fig)

        def next_sample(event):
            self.current_decision = 'next'
            plt.close(fig)

        def quit_session(event):
            self.current_decision = 'quit'
            plt.close(fig)

        # Add buttons
        ax_accept = plt.axes([0.15, 0.02, 0.12, 0.05])
        ax_reject = plt.axes([0.28, 0.02, 0.12, 0.05])
        ax_skip = plt.axes([0.41, 0.02, 0.12, 0.05])
        ax_prev = plt.axes([0.54, 0.02, 0.12, 0.05])
        ax_next = plt.axes([0.67, 0.02, 0.12, 0.05])
        ax_quit = plt.axes([0.80, 0.02, 0.12, 0.05])

        btn_accept = Button(ax_accept, 'Accept (A)')
        btn_reject = Button(ax_reject, 'Reject (R)')
        btn_skip = Button(ax_skip, 'Skip (S)')
        btn_prev = Button(ax_prev, '◄ Prev')
        btn_next = Button(ax_next, 'Next ►')
        btn_quit = Button(ax_quit, 'Quit (Q)')

        btn_accept.on_clicked(lambda x: accept(x))
        btn_reject.on_clicked(lambda x: reject(x))
        btn_skip.on_clicked(lambda x: skip(x))
        btn_prev.on_clicked(lambda x: prev_sample(x))
        btn_next.on_clicked(lambda x: next_sample(x))
        btn_quit.on_clicked(lambda x: quit_session(x))

        # Key press handler
        def on_key(event):
            if event.key == 'a' or event.key == 'A':
                self.current_decision = 'accept'
                plt.close(fig)
            elif event.key == 'r' or event.key == 'R':
                self.current_decision = 'reject'
                plt.close(fig)
            elif event.key == 's' or event.key == 'S':
                self.current_decision = 'skip'
                plt.close(fig)
            elif event.key == 'q' or event.key == 'Q' or event.key == 'escape':
                self.current_decision = 'quit'
                plt.close(fig)
            elif event.key == 'left':
                self.current_decision = 'prev'
                plt.close(fig)
            elif event.key == 'right':
                self.current_decision = 'next'
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

        return self.current_decision if self.current_decision else 'skip'

    def run(self):
        """Run interactive filtering session"""
        print("=" * 60)
        print("Interactive Quality Filter")
        print("=" * 60)
        print(f"Total samples: {self.total_samples}")
        print(f"Already reviewed: {len(self.accepted) + len(self.rejected)}")
        print(f"Remaining: {self.total_samples - len(self.accepted) - len(self.rejected)}")
        print("\nControls:")
        print("  A - Accept sample")
        print("  R - Reject sample")
        print("  S - Skip sample (review later)")
        print("  Left/Right - Navigate")
        print("  Q/ESC - Quit and save")
        print("=" * 60)

        rejected_reasons = {}

        while self.current_idx < self.total_samples:
            sample = self._load_sample(self.current_idx)

            if sample is None:
                print(f"Warning: Could not load sample at index {self.current_idx}")
                self.current_idx += 1
                continue

            file_id = sample['file_id']

            # Skip if already reviewed
            if file_id in self.accepted or file_id in self.rejected:
                self.current_idx += 1
                continue

            # Show sample and get decision
            decision = self._show_sample(sample)

            if decision == 'accept':
                self.accepted.add(file_id)
                print(f"[{self.current_idx + 1}/{self.total_samples}] ACCEPTED: {file_id}")
                self.current_idx += 1

            elif decision == 'reject':
                self.rejected.add(file_id)
                print(f"[{self.current_idx + 1}/{self.total_samples}] REJECTED: {file_id}")

                # Ask for reason
                print("\nReason for rejection:")
                print("  1 - Mask inaccurate (dirty regions wrong)")
                print("  2 - Image quality poor (blurry, noisy)")
                print("  3 - No visible dirt")
                print("  4 - Other")

                reason_key = input("Enter reason (1-4): ").strip()
                reason_map = {
                    '1': 'mask_inaccurate',
                    '2': 'poor_quality',
                    '3': 'no_dirt',
                    '4': 'other',
                }
                if reason_key in reason_map:
                    self.reasons[file_id] = reason_map[reason_key]

                self.current_idx += 1

            elif decision == 'skip':
                print(f"[{self.current_idx + 1}/{self.total_samples}] SKIPPED: {file_id}")
                self.current_idx += 1

            elif decision == 'prev':
                self.current_idx = max(0, self.current_idx - 1)

            elif decision == 'next':
                self.current_idx = min(self.total_samples - 1, self.current_idx + 1)

            elif decision == 'quit':
                print("\nSaving progress and exiting...")
                self._save_progress()
                break

            # Save progress periodically
            if len(self.accepted) + len(self.rejected) % self.save_interval == 0:
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
        description="Interactive quality filter for LoRA training data"
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
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N samples"
    )

    args = parser.parse_args()

    filter_tool = InteractiveQualityFilter(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output),
        save_interval=args.save_interval,
    )

    filter_tool.run()


if __name__ == "__main__":
    main()
