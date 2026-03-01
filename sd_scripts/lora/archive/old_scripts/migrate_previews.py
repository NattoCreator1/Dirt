#!/usr/bin/env python3
"""
Migrate existing preview images to organized directory structure
"""

import json
from pathlib import Path

# Paths
progress_file = Path("training_data/filter_results/filter_progress.json")
previews_dir = Path("training_data/filter_results/previews")
accepted_dir = previews_dir / "accepted"
rejected_dir = previews_dir / "rejected"
current_dir = previews_dir / "current"

# Create subdirectories
accepted_dir.mkdir(parents=True, exist_ok=True)
rejected_dir.mkdir(parents=True, exist_ok=True)
current_dir.mkdir(parents=True, exist_ok=True)

# Load progress
with open(progress_file) as f:
    data = json.load(f)
    accepted = set(data.get('accepted', []))
    rejected = set(data.get('rejected', []))

print(f"Found {len(accepted)} accepted, {len(rejected)} rejected samples")

# Find and move preview images
moved = 0
for preview_file in previews_dir.glob("*_preview.jpg"):
    if preview_file.parent in [accepted_dir, rejected_dir, current_dir]:
        continue  # Skip already organized
    
    file_id = preview_file.stem.replace("_preview", "")
    
    if file_id in accepted:
        dest = accepted_dir / preview_file.name
        preview_file.rename(dest)
        print(f"  Moved {file_id} -> accepted/")
        moved += 1
    elif file_id in rejected:
        dest = rejected_dir / preview_file.name
        preview_file.rename(dest)
        print(f"  Moved {file_id} -> rejected/")
        moved += 1

print(f"\nMoved {moved} preview images")
