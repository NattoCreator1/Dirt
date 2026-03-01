import os
import re
import glob
import json
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize_stem(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

def ahash64(bgr, hash_size=8) -> int:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small >= avg).astype(np.uint8).flatten()
    h = 0
    for i, b in enumerate(bits):
        h |= (int(b) << i)
    return h

def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

def calc_metrics(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap.var())

    sat_hi = float((gray >= 250).mean())
    sat_lo = float((gray <= 5).mean())
    mean_gray = float(gray.mean())

    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float((edges > 0).mean())

    return sharp, mean_gray, sat_hi, sat_lo, edge_ratio

def pass_rules(sharp, mean_gray, sat_hi, sat_lo, edge_ratio, cfg):
    if sharp < cfg["min_sharpness"]:
        return False, "blur"
    if sat_hi > cfg["max_sat_hi"]:
        return False, "overexposed"
    if sat_lo > cfg["max_sat_lo"]:
        return False, "underexposed"
    if not (cfg["min_mean_gray"] <= mean_gray <= cfg["max_mean_gray"]):
        return False, "bad_brightness"
    if not (cfg["min_edge_ratio"] <= edge_ratio <= cfg["max_edge_ratio"]):
        return False, "bad_edges"
    return True, "ok"

def split_views(mosaic_bgr):
    """
    默认按 2 行 3 列切分，返回 dict:
    {view_id: (crop_bgr, (x0,y0,x1,y1))}
    view_id:
      lf, f, rf, lr, r, rr
    """
    H, W = mosaic_bgr.shape[:2]
    wc = W // 3
    hc = H // 2

    # 若不是整除，会丢掉右侧/底部不足部分；对 1920x720 无影响
    rois = {
        "lf": (0*wc, 0*hc, 1*wc, 1*hc),
        "f":  (1*wc, 0*hc, 2*wc, 1*hc),
        "rf": (2*wc, 0*hc, 3*wc, 1*hc),
        "lr": (0*wc, 1*hc, 1*wc, 2*hc),
        "r":  (1*wc, 1*hc, 2*wc, 2*hc),
        "rr": (2*wc, 1*hc, 3*wc, 2*hc),
    }

    out = {}
    for vid, (x0, y0, x1, y1) in rois.items():
        crop = mosaic_bgr[y0:y1, x0:x1]
        out[vid] = (crop, (x0, y0, x1, y1))
    return out

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--video_dir", required=True, help="原始环视视频目录")
    ap.add_argument("--out_frames_dir", required=True, help="输出干净子图目录（会按 view_id 建子目录）")
    ap.add_argument("--out_manifest_dir", required=True, help="输出 manifest 目录")

    ap.add_argument("--delta_t", type=float, default=1.0, help="抽帧间隔(秒)")
    ap.add_argument("--jpeg_quality", type=int, default=95)

    # 去重：每个 (video_id, view_id) 独立维护 last_keep_hash
    ap.add_argument("--dup_ham_min", type=int, default=8, help="aHash 去重汉明距离阈值")

    # 过滤阈值
    ap.add_argument("--min_sharpness", type=float, default=80.0)
    ap.add_argument("--max_sat_hi", type=float, default=0.08)
    ap.add_argument("--max_sat_lo", type=float, default=0.08)
    ap.add_argument("--min_mean_gray", type=float, default=35.0)
    ap.add_argument("--max_mean_gray", type=float, default=210.0)
    ap.add_argument("--min_edge_ratio", type=float, default=0.01)
    ap.add_argument("--max_edge_ratio", type=float, default=0.25)

    args = ap.parse_args()

    ensure_dir(args.out_frames_dir)
    ensure_dir(args.out_manifest_dir)
    for v in ["lf", "f", "rf", "lr", "r", "rr"]:
        ensure_dir(os.path.join(args.out_frames_dir, v))

    cfg = {
        "delta_t": args.delta_t,
        "dup_ham_min": args.dup_ham_min,
        "min_sharpness": args.min_sharpness,
        "max_sat_hi": args.max_sat_hi,
        "max_sat_lo": args.max_sat_lo,
        "min_mean_gray": args.min_mean_gray,
        "max_mean_gray": args.max_mean_gray,
        "min_edge_ratio": args.min_edge_ratio,
        "max_edge_ratio": args.max_edge_ratio,
        "jpeg_quality": args.jpeg_quality,
        "layout": "2rowsx3cols_split",
        "view_ids": ["lf", "f", "rf", "lr", "r", "rr"],
    }

    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.MP4", "*.MOV", "*.MKV", "*.AVI")
    video_paths = []
    for ext in exts:
        video_paths += glob.glob(os.path.join(args.video_dir, ext))
    video_paths = sorted(video_paths)

    rows_keep = []
    rows_rej = []

    total_seen_mosaic = 0
    total_seen_views = 0
    total_keep_views = 0

    # 去重状态：key=(video_id, view_id) -> last_keep_hash
    last_hash = {}

    for vp in tqdm(video_paths, desc="videos"):
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            rows_rej.append({"video_path": vp, "reason": "cannot_open"})
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            fps = 30.0

        step = max(1, int(round(fps * args.delta_t)))
        raw_stem = os.path.splitext(os.path.basename(vp))[0]
        video_id = sanitize_stem(raw_stem)

        frame_idx = 0
        while True:
            ok, mosaic = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            total_seen_mosaic += 1
            Hm, Wm = mosaic.shape[:2]
            ts_ms = int(round(1000.0 * frame_idx / fps))

            views = split_views(mosaic)

            for view_id, (crop, (x0, y0, x1, y1)) in views.items():
                total_seen_views += 1

                Hv, Wv = crop.shape[:2]
                sharp, mean_gray, sat_hi, sat_lo, edge_ratio = calc_metrics(crop)
                passed, reason = pass_rules(sharp, mean_gray, sat_hi, sat_lo, edge_ratio, cfg)

                hash64 = ahash64(crop)

                # 按 (video_id, view_id) 去重
                k = (video_id, view_id)
                if passed and (k in last_hash):
                    dH = hamming64(hash64, last_hash[k])
                    if dH < args.dup_ham_min:
                        passed = False
                        reason = f"duplicate(dH={dH})"

                if passed:
                    fname = f"{video_id}_{view_id}_t{ts_ms:010d}_f{frame_idx:09d}.jpg"
                    abs_path = os.path.join(args.out_frames_dir, view_id, fname)

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
                    buf = cv2.imencode(".jpg", crop, encode_param)[1]
                    buf.tofile(abs_path)

                    last_hash[k] = hash64
                    total_keep_views += 1

                    rows_keep.append({
                        "image_id": os.path.splitext(fname)[0],
                        "image_path": os.path.relpath(abs_path, start=os.path.dirname(args.out_manifest_dir)).replace("\\", "/"),
                        "video_id": video_id,
                        "view_id": view_id,
                        "frame_idx": int(frame_idx),
                        "timestamp_ms": int(ts_ms),
                        "mosaic_width": int(Wm),
                        "mosaic_height": int(Hm),
                        "crop_x0": int(x0), "crop_y0": int(y0), "crop_x1": int(x1), "crop_y1": int(y1),
                        "width": int(Wv),
                        "height": int(Hv),
                        "fps": float(fps),
                        "sharpness": sharp,
                        "mean_gray": mean_gray,
                        "sat_hi_ratio": sat_hi,
                        "sat_lo_ratio": sat_lo,
                        "edge_ratio": edge_ratio,
                        "hash64": int(hash64),
                        "video_path": vp.replace("\\", "/"),
                    })
                else:
                    rows_rej.append({
                        "video_id": video_id,
                        "view_id": view_id,
                        "frame_idx": int(frame_idx),
                        "timestamp_ms": int(ts_ms),
                        "reason": reason,
                        "mosaic_width": int(Wm),
                        "mosaic_height": int(Hm),
                        "crop_x0": int(x0), "crop_y0": int(y0), "crop_x1": int(x1), "crop_y1": int(y1),
                        "width": int(Wv),
                        "height": int(Hv),
                        "fps": float(fps),
                        "sharpness": sharp,
                        "mean_gray": mean_gray,
                        "sat_hi_ratio": sat_hi,
                        "sat_lo_ratio": sat_lo,
                        "edge_ratio": edge_ratio,
                        "hash64": int(hash64),
                        "video_path": vp.replace("\\", "/"),
                    })

            frame_idx += 1

        cap.release()

    df_keep = pd.DataFrame(rows_keep)
    df_rej = pd.DataFrame(rows_rej)

    keep_csv = os.path.join(args.out_manifest_dir, "manifest_clean.csv")
    rej_csv = os.path.join(args.out_manifest_dir, "manifest_clean_reject.csv")
    rep_json = os.path.join(args.out_manifest_dir, "manifest_clean_report.json")

    df_keep.to_csv(keep_csv, index=False, encoding="utf-8-sig")
    df_rej.to_csv(rej_csv, index=False, encoding="utf-8-sig")

    report = {
        "videos": len(video_paths),
        "total_seen_mosaic_frames": int(total_seen_mosaic),
        "total_seen_view_images": int(total_seen_views),
        "total_keep_view_images": int(total_keep_views),
        "keep_ratio_views": float(total_keep_views / max(1, total_seen_views)),
        "cfg": cfg,
        "video_dir": args.video_dir,
        "out_frames_dir": args.out_frames_dir,
        "out_manifest_dir": args.out_manifest_dir,
    }
    with open(rep_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
