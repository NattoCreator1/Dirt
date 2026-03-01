"""
Preprocess the washed external test dataset (external_test_washed)

The washed dataset has redefined occlusion levels that are more consistent
with the Severity Score definition.

New level definitions (from README.md.txt):
- Occlusion_a (level 5): 由泥巴导致的"全局遮挡"："大面积"蜂窝状或黑色区域
- Occlusion_b (level 4): 全局遮挡，但脏污的不透明程度与区域没有occlusion_a那么严重
- Occlusion_c (level 3):
    1. 全局遮挡，但不透明程度比occlusion_b更轻微
    2. 虽然存在部分区域清晰，但基本还是近乎全局的黑点遮挡
    3. 部分遮挡，遮挡脏污明显，不透明程度较高，主要趋近于画面中央
- Occlusion_d (level 2):
    1. 全局浅色遮挡或模糊 + 局部点状小面积的泥巴类脏污遮挡
    2. 部分遮挡，遮挡脏污更偏向半透明，或脏污区域更偏向于画面边缘
- Occlusion_e (level 1): 全局很浅的遮挡或带有少量黑斑，不影响整体道路目标轮廓识别
"""

import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

LEVEL_MAP = {  # letter -> ext_level (bigger = more severe)
    "a": 5,
    "b": 4,
    "c": 3,
    "d": 2,
    "e": 1,
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

def read_image_unicode(path: Path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

def write_jpg_unicode(path: Path, bgr, quality: int = 95) -> bool:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", bgr, params)
    if not ok:
        return False
    ensure_dir(path.parent)
    enc.tofile(str(path))
    return True

def center_crop_to_ratio(bgr, target_ratio: float):
    h, w = bgr.shape[:2]
    cur = w / h
    if abs(cur - target_ratio) < 1e-6:
        return bgr, (0, 0, w, h)

    if cur > target_ratio:
        # too wide -> crop width
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        x1 = x0 + new_w
        y0, y1 = 0, h
    else:
        # too tall -> crop height
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        y1 = y0 + new_h
        x0, x1 = 0, w

    crop = bgr[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)

def resize_to(bgr, tw: int, th: int):
    h, w = bgr.shape[:2]
    if (w, h) == (tw, th):
        return bgr
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_CUBIC
    return cv2.resize(bgr, (tw, th), interpolation=interp)

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Preprocess washed external test dataset"
    )

    ap.add_argument("--in_root", required=True,
                    help="Path to external_test_washed directory")
    ap.add_argument("--out_root", required=True,
                    help="Path to output directory (e.g., dataset/external_test_washed_processed)")

    ap.add_argument("--target_w", type=int, default=640)
    ap.add_argument("--target_h", type=int, default=480)
    ap.add_argument("--target_ratio", type=str, default="4:3")
    ap.add_argument("--jpeg_quality", type=int, default=95)

    ap.add_argument("--min_size", type=int, default=64,
                    help="Filter out abnormally small images")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    img_root = out_root / "images"

    ensure_dir(out_root)
    for letter in "abcde":
        ensure_dir(img_root / f"occlusion_{letter}")

    # Parse ratio
    import re
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", args.target_ratio)
    if not m:
        raise ValueError("target_ratio format should be like 4:3")
    rw, rh = int(m.group(1)), int(m.group(2))
    target_ratio = rw / rh

    rows = []
    bad_rows = []
    counts = {f"occlusion_{k}": 0 for k in "abcde"}

    # Process each occlusion folder directly
    for letter in "abcde":
        label_raw = f"occlusion_{letter}"
        ext_level = LEVEL_MAP[letter]

        occ_dir = in_root / label_raw
        if not occ_dir.exists():
            print(f"Warning: {occ_dir} does not exist, skipping...")
            continue

        # Collect all images in this occlusion folder (recursive)
        img_paths = [p for p in occ_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        img_paths = sorted(img_paths)

        print(f"Processing {label_raw}: {len(img_paths)} images")

        for p in tqdm(img_paths, desc=label_raw, leave=False):
            img = read_image_unicode(p)
            if img is None:
                bad_rows.append({
                    "orig_path": str(p),
                    "reason": "read_fail",
                    "label_raw": label_raw,
                })
                continue

            h0, w0 = img.shape[:2]
            if min(h0, w0) < args.min_size:
                bad_rows.append({
                    "orig_path": str(p),
                    "reason": f"too_small({w0}x{h0})",
                    "label_raw": label_raw,
                })
                continue

            # 1) Center crop to target ratio (default 4:3)
            crop, (x0, y0, x1, y1) = center_crop_to_ratio(img, target_ratio)
            h1, w1 = crop.shape[:2]

            # 2) Resize to target size (default 640x480)
            out_img = resize_to(crop, args.target_w, args.target_h)
            h2, w2 = out_img.shape[:2]

            # Output filename
            uid = sha1_12(str(p))
            name = f"{label_raw}_{uid}.jpg"
            out_path = img_root / label_raw / name

            if out_path.exists() and (not args.overwrite):
                pass  # Already exists, skip writing but still record
            else:
                ok = write_jpg_unicode(out_path, out_img, quality=args.jpeg_quality)
                if not ok:
                    bad_rows.append({
                        "orig_path": str(p),
                        "reason": "write_fail",
                        "label_raw": label_raw,
                    })
                    continue

            counts[label_raw] += 1

            rows.append({
                "image_id": name[:-4],
                "image_path": str(out_path).replace("\\", "/"),
                "label_raw": label_raw,
                "ext_level": int(ext_level),
                "orig_path": str(p).replace("\\", "/"),
                "orig_w": int(w0),
                "orig_h": int(h0),
                "crop_x0": int(x0),
                "crop_y0": int(y0),
                "crop_x1": int(x1),
                "crop_y1": int(y1),
                "crop_w": int(w1),
                "crop_h": int(h1),
                "final_w": int(w2),
                "final_h": int(h2),
                "preprocess": f"center_crop({args.target_ratio})+resize({args.target_w}x{args.target_h})",
            })

    # Save outputs
    df = pd.DataFrame(rows)
    df_bad = pd.DataFrame(bad_rows)

    out_csv = out_root / "test_ext.csv"
    out_bad = out_root / "test_ext_bad_files.csv"
    out_sum = out_root / "test_ext_summary.json"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if len(df_bad) > 0:
        df_bad.to_csv(out_bad, index=False, encoding="utf-8-sig")

    summary = {
        "in_root": str(in_root),
        "out_root": str(out_root),
        "target_ratio": args.target_ratio,
        "target_size": [args.target_w, args.target_h],
        "jpeg_quality": args.jpeg_quality,
        "num_images": int(len(df)),
        "num_bad_files": int(len(df_bad)),
        "counts_by_level": counts,
        "level_definitions": {
            "occlusion_a (5)": "Mud-induced global occlusion with large honeycomb or black areas",
            "occlusion_b (4)": "Global occlusion with less severe opacity and area than occlusion_a",
            "occlusion_c (3)": "Global occlusion with lighter opacity, or partial occlusion with severe dirt in center",
            "occlusion_d (2)": "Global light occlusion/blur with small local mud spots, or partial semi-transparent dirt",
            "occlusion_e (1)": "Very light global occlusion or small black spots, road contours still recognizable",
        }
    }

    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput CSV: {out_csv}")
    print(f"Total images: {len(df)}")

if __name__ == "__main__":
    main()
