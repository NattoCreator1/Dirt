import os
import glob
import argparse
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

VIEWS = ["lf", "f", "rf", "lr", "r", "rr"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def center_crop_to_ratio(img, target_w: int, target_h: int):
    """
    按目标宽高比做中心裁剪（不拉伸）。
    target_w/target_h 只用于定义目标 ratio，不要求输出就是 target_w,target_h。
    """
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    cur_ratio = w / h

    if abs(cur_ratio - target_ratio) < 1e-6:
        return img, (0, 0, w, h)

    if cur_ratio > target_ratio:
        # 太宽：裁宽
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        x1 = x0 + new_w
        y0, y1 = 0, h
    else:
        # 太高：裁高
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        y1 = y0 + new_h
        x0, x1 = 0, w

    crop = img[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)

def resize_if_needed(img, tw, th):
    if tw is None or th is None:
        return img
    h, w = img.shape[:2]
    if (w, h) == (tw, th):
        return img
    # 下采样用 AREA，上采样用 CUBIC
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_CUBIC
    return cv2.resize(img, (tw, th), interpolation=interp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="输入根目录，例如 dataset/my_clean_frames")
    ap.add_argument("--out_root", required=True, help="输出根目录，例如 dataset/my_clean_frames_4by3")
    ap.add_argument("--exts", default="jpg,jpeg,png", help="逗号分隔后缀")
    ap.add_argument("--target_w", type=int, default=None, help="可选：裁剪后再 resize 到该宽")
    ap.add_argument("--target_h", type=int, default=None, help="可选：裁剪后再 resize 到该高")
    ap.add_argument("--overwrite", action="store_true", help="覆盖输出已存在文件")
    ap.add_argument("--write_manifest", action="store_true", help="输出 manifest_4by3.csv")
    args = ap.parse_args()

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    ensure_dir(args.out_root)
    for v in VIEWS:
        ensure_dir(os.path.join(args.out_root, v))

    rows = []
    total = 0
    bad_read = 0

    for v in VIEWS:
        in_dir = os.path.join(args.in_root, v)
        out_dir = os.path.join(args.out_root, v)
        files = []
        for e in exts:
            files += glob.glob(os.path.join(in_dir, f"*.{e}"))
            files += glob.glob(os.path.join(in_dir, f"*.{e.upper()}"))
        files = sorted(files)

        for fp in tqdm(files, desc=f"view={v}", leave=False):
            # img = cv2.imdecode(
            #     # 兼容含特殊字符路径
            #     open(fp, "rb").read(),
            #     cv2.IMREAD_COLOR
            # )
            buf = np.fromfile(fp, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                bad_read += 1
                continue

            h0, w0 = img.shape[:2]

            # 先裁到 4:3（以 4/3 为 ratio）
            crop, (x0, y0, x1, y1) = center_crop_to_ratio(img, 4, 3)

            # 如果你确认输入必为 640x360，也可强制裁成 480x360：
            # crop = img[:, 80:560]
            # x0,y0,x1,y1 = 80,0,560,360

            h1, w1 = crop.shape[:2]

            # 可选：再统一 resize 到 target_w,target_h
            out_img = resize_if_needed(crop, args.target_w, args.target_h)
            h2, w2 = out_img.shape[:2]

            out_name = os.path.basename(fp)
            out_fp = os.path.join(out_dir, out_name)

            if (not args.overwrite) and os.path.exists(out_fp):
                continue

            ok, buf = cv2.imencode(os.path.splitext(out_fp)[1], out_img)
            if not ok:
                continue
            with open(out_fp, "wb") as f:
                f.write(buf.tobytes())

            total += 1
            if args.write_manifest:
                rows.append({
                    "view_id": v,
                    "orig_path": fp.replace("\\", "/"),
                    "out_path": out_fp.replace("\\", "/"),
                    "orig_w": w0, "orig_h": h0,
                    "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                    "crop_w": w1, "crop_h": h1,
                    "final_w": w2, "final_h": h2,
                })

    print(f"processed: {total}, bad_read: {bad_read}")

    if args.write_manifest:
        mani_dir = os.path.join(os.path.dirname(args.out_root), "my_clean_manifests")
        ensure_dir(mani_dir)
        out_csv = os.path.join(mani_dir, "manifest_clean_4by3.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"manifest saved: {out_csv}")

if __name__ == "__main__":
    main()
