import os
import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm

VIEWS = ["lf", "f", "rf", "lr", "r", "rr"]
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_image_unicode(fp: str):
    buf = np.fromfile(fp, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def write_image_unicode(fp: str, img, quality=95):
    ensure_dir(os.path.dirname(fp))
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", img, params)
    if not ok:
        return False
    enc.tofile(fp)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="dataset/my_clean_frames_4by3")
    ap.add_argument("--out_root", required=True, help="dataset/my_clean_frames_4by3_640x480")
    ap.add_argument("--target_w", type=int, default=640)
    ap.add_argument("--target_h", type=int, default=480)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for v in VIEWS:
        ensure_dir(os.path.join(args.out_root, v))

    total = 0
    bad = 0

    for v in VIEWS:
        in_dir = os.path.join(args.in_root, v)
        out_dir = os.path.join(args.out_root, v)

        files = []
        for e in EXTS:
            files += glob.glob(os.path.join(in_dir, f"*{e}"))
            files += glob.glob(os.path.join(in_dir, f"*{e.upper()}"))
        files = sorted(files)

        for fp in tqdm(files, desc=f"view={v}"):
            name = os.path.splitext(os.path.basename(fp))[0] + ".jpg"
            out_fp = os.path.join(out_dir, name)

            if (not args.overwrite) and os.path.exists(out_fp):
                continue

            img = read_image_unicode(fp)
            if img is None:
                bad += 1
                continue

            h, w = img.shape[:2]

            # 若输入已经是 640x480，则直接写（或跳过）
            if (w, h) == (args.target_w, args.target_h):
                out_img = img
            else:
                # 这里假设输入为 4:3（例如 480x360）
                interp = cv2.INTER_CUBIC  # 上采样更合适
                out_img = cv2.resize(img, (args.target_w, args.target_h), interpolation=interp)

            if not write_image_unicode(out_fp, out_img, quality=args.jpeg_quality):
                bad += 1
                continue

            total += 1

    print(f"done. written: {total}, bad: {bad}")

if __name__ == "__main__":
    main()
