import os
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

VIEWS = {"lf", "f", "rf", "lr", "r", "rr"}
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
    # 对包含特殊字符路径更稳：np.fromfile + imdecode
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

def infer_view_id(p: Path):
    parts = {x.lower() for x in p.parts}
    hit = [v for v in VIEWS if v in parts]
    return hit[0] if hit else ""

def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_root", required=True, help="dataset/my_soiling_raw")
    ap.add_argument("--out_root", required=True, help="dataset/my_external_test")

    ap.add_argument("--target_w", type=int, default=640)
    ap.add_argument("--target_h", type=int, default=480)
    ap.add_argument("--target_ratio", type=str, default="4:3", help="如 4:3")
    ap.add_argument("--jpeg_quality", type=int, default=95)

    ap.add_argument("--min_size", type=int, default=64, help="过滤异常小图")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    img_root = out_root / "images"

    ensure_dir(out_root)
    for letter in "abcde":
        ensure_dir(img_root / f"occlusion_{letter}")

    # parse ratio
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", args.target_ratio)
    if not m:
        raise ValueError("target_ratio format should be like 4:3")
    rw, rh = int(m.group(1)), int(m.group(2))
    target_ratio = rw / rh

    # find all occlusion folders
    # 允许大小写：occlusion_A, OCCLUSION_b ...
    occ_pat = re.compile(r"occlusion_([a-e])$", re.IGNORECASE)

    rows = []
    bad_rows = []
    counts = {f"occlusion_{k}": 0 for k in "abcde"}

    # 遍历 in_root 下所有目录，找到名为 occlusion_[a-e] 的目录
    occ_dirs = []
    for d in in_root.rglob("*"):
        if d.is_dir():
            mm = occ_pat.match(d.name)
            if mm:
                occ_dirs.append(d)
    occ_dirs = sorted(set(occ_dirs))

    for occ_dir in tqdm(occ_dirs, desc="occlusion_dirs"):
        mm = occ_pat.match(occ_dir.name)
        if not mm:
            continue
        letter = mm.group(1).lower()
        ext_level = LEVEL_MAP[letter]
        label_raw = f"occlusion_{letter}"

        # group_id：取 in_root 下的一级目录名（如 dataset_20240304）
        try:
            rel = occ_dir.relative_to(in_root)
            group_id = rel.parts[0] if len(rel.parts) > 0 else ""
        except Exception:
            group_id = ""

        # 收集该 occlusion_dir 下所有图片（递归）
        img_paths = [p for p in occ_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        img_paths = sorted(img_paths)

        for p in tqdm(img_paths, desc=f"{label_raw}", leave=False):
            img = read_image_unicode(p)
            if img is None:
                bad_rows.append({
                    "orig_path": str(p),
                    "reason": "read_fail",
                    "label_raw": label_raw,
                    "group_id": group_id,
                })
                continue

            h0, w0 = img.shape[:2]
            if min(h0, w0) < args.min_size:
                bad_rows.append({
                    "orig_path": str(p),
                    "reason": f"too_small({w0}x{h0})",
                    "label_raw": label_raw,
                    "group_id": group_id,
                })
                continue

            # 1) center crop to target ratio (default 4:3)
            crop, (x0, y0, x1, y1) = center_crop_to_ratio(img, target_ratio)
            h1, w1 = crop.shape[:2]

            # 2) resize to target size (default 640x480)
            out_img = resize_to(crop, args.target_w, args.target_h)
            h2, w2 = out_img.shape[:2]

            # output filename (stable, collision-free)
            view_id = infer_view_id(p)
            rel_path_str = str(p.relative_to(in_root)).replace("\\", "/")
            uid = sha1_12(rel_path_str)
            name = f"{label_raw}_{group_id}_{view_id}_{uid}.jpg".replace("__", "_").strip("_")
            out_path = img_root / label_raw / name

            if out_path.exists() and (not args.overwrite):
                # 已存在则跳过写入，但仍记录到 manifest（保证可复现）
                pass
            else:
                ok = write_jpg_unicode(out_path, out_img, quality=args.jpeg_quality)
                if not ok:
                    bad_rows.append({
                        "orig_path": str(p),
                        "reason": "write_fail",
                        "label_raw": label_raw,
                        "group_id": group_id,
                    })
                    continue

            counts[label_raw] += 1

            rows.append({
                "image_id": name[:-4],
                "image_path": str(out_path).replace("\\", "/"),
                "label_raw": label_raw,
                "ext_level": int(ext_level),      # 1~5 (bigger = more severe)
                "group_id": group_id,
                "view_id": view_id,
                "orig_path": str(p).replace("\\", "/"),
                "orig_w": int(w0), "orig_h": int(h0),
                "crop_x0": int(x0), "crop_y0": int(y0), "crop_x1": int(x1), "crop_y1": int(y1),
                "crop_w": int(w1), "crop_h": int(h1),
                "final_w": int(w2), "final_h": int(h2),
                "preprocess": f"center_crop({args.target_ratio})+resize({args.target_w}x{args.target_h})",
            })

    df = pd.DataFrame(rows)
    df_bad = pd.DataFrame(bad_rows)

    out_csv = out_root / "test_ext.csv"
    out_bad = out_root / "test_ext_bad_files.csv"
    out_sum = out_root / "test_ext_summary.json"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_bad.to_csv(out_bad, index=False, encoding="utf-8-sig")

    summary = {
        "in_root": str(in_root),
        "out_root": str(out_root),
        "target_ratio": args.target_ratio,
        "target_size": [args.target_w, args.target_h],
        "jpeg_quality": args.jpeg_quality,
        "num_occlusion_dirs": len(occ_dirs),
        "num_images_written_or_indexed": int(len(df)),
        "num_bad_files": int(len(df_bad)),
        "counts_by_level": counts,
    }
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
