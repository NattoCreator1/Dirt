# scripts/03_build_labels_tile_global.py
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def img_rel_to_gt_abs(raw_root: Path, rel_img: str) -> Path:
    p = Path(rel_img)  # e.g. train/rbgImages/0000_FV.png
    parts = list(p.parts)
    parts = [("gtLabels" if x in ("rbgImages", "rgbImages") else x) for x in parts]
    gt_rel = Path(*parts)
    gt_abs = raw_root / gt_rel
    return gt_abs

def compute_tile_cov(mask_cls, n=8, m=8, C=4):
    h, w = mask_cls.shape
    out = np.zeros((n, m, C), dtype=np.float32)
    for i in range(n):
        y0 = int(i * h / n); y1 = int((i + 1) * h / n)
        for j in range(m):
            x0 = int(j * w / m); x1 = int((j + 1) * w / m)
            patch = mask_cls[y0:y1, x0:x1]
            denom = float(max(1, patch.size))
            for c in range(C):
                out[i, j, c] = float((patch == c).sum()) / denom
    return out

# 新增利用高斯中心权重的 空间权重图 W(x, y)
def make_spatial_weight(h, w, mode="gaussian", sigma=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yn = (yy - cy) / (h * sigma)
    xn = (xx - cx) / (w * sigma)

    if mode == "gaussian":
        W = np.exp(-(xn * xn + yn * yn))
    elif mode == "uniform":
        W = np.ones((h, w), dtype=np.float32)
    else:
        raise ValueError(f"Unknown spatial weight mode: {mode}")
    
    W = W / (W.sum() + 1e-12)
    return W

import numpy as np

def compute_dominance_class_aware(tile_cov,
                                 class_weights=(0.0, 0.33, 0.66, 1.0),
                                 eta_trans=0.9,
                                 eps=1e-8):
    """
    tile_cov: (N,M,4), sum over last dim = 1
    class_weights: w0..w3, with w0=0
    eta_trans: max discount strength for transparent dominance
               eta_trans=0.9 -> pure transparent tile gets factor 0.1
    return:
      S_dom, dom_tile_score, trans_ratio, (i,j)
    """
    w = np.array(class_weights, dtype=np.float32)  # (4,)

    # per-tile weighted severity (ignore clean because w0=0)
    s_tile = (tile_cov * w[None, None, :]).sum(axis=-1)  # (N,M)

    # locate worst tile
    idx = int(np.argmax(s_tile))
    N, M = s_tile.shape
    i, j = idx // M, idx % M
    dom_tile_score = float(s_tile[i, j])

    # transparent ratio inside soiling area of the dominant tile
    soiling_part = float(tile_cov[i, j, 1] + tile_cov[i, j, 2] + tile_cov[i, j, 3])
    trans_ratio = float(tile_cov[i, j, 1] / (soiling_part + eps))

    # smooth discount: when trans_ratio=1, factor=1-eta_trans
    factor = float(1.0 - eta_trans * trans_ratio)
    factor = float(np.clip(factor, 0.0, 1.0))

    S_dom = float(dom_tile_score * factor)
    return S_dom, dom_tile_score, trans_ratio, (i, j)


# 新增计算连续严重度评分S
def compute_global_severity(mask, tile_cov, class_weights=(0.0, 0.33, 0.66, 1.0),
                            alpha=0.6, beta=0.3, gamma=0.1,
                            spatial_mode="gaussian", eta_trans=0.9):
    """
    mask: (H, W) int64 in {0, 1, 2, 3}
    tile_cov: (N, M, 4) float32, sum over last dim = 1
    return : glboal_score S in [0, 1] (approximately)
    """
    H, W = mask.shape
    w = np.array(class_weights, dtype=np.float32)

    # part one: Opacity-aware coverage score
    total = float(mask.size)
    p = np.array([(mask==c).sum() / total for c in range(4)], dtype=np.float32)
    S_op = float((w * p).sum())

    # part two: Spatial weighted score
    Wmap = make_spatial_weight(H, W, mode=spatial_mode)
    S_sp = float((Wmap * w[mask]).sum())

    # part three: Dominance / concentrated blockage
    t = 1.0 - tile_cov[..., 0]
    S_dom, dom_raw, r_tr, dom_ij = compute_dominance_class_aware(
        tile_cov,
        class_weights=(0.0, 0.33, 0.66, 1.0),
        eta_trans=eta_trans
)


    # Fuse
    S = alpha * S_op + beta * S_sp + gamma * S_dom
    S = float(np.clip(S, 0.0, 1.0))

    return S, S_op, S_sp, S_dom


def compute_all_severity_variants(mask, tile_cov,
                                    class_weights=(0.0, 0.33, 0.66, 1.0),
                                    spatial_mode="gaussian"):
    """
    Compute all Severity Score variants for ablation experiments.

    Returns dictionary with variants:
        - s: Simple mean (1 - clean_ratio) = global_soiling
        - S_op_only: Opacity-aware coverage only
        - S_op_sp: Opacity + Spatial (no dominance)
        - S_full: Complete Severity Score (alpha=0.6, beta=0.3, gamma=0.1)
        - S_full_eta00: Full with eta=0 (no transparency discount)
    """
    H, W = mask.shape
    w = np.array(class_weights, dtype=np.float32)

    # Simple mean (baseline): s = 1 - clean_ratio
    clean_ratio = float((mask == 0).sum() / mask.size)
    s = 1.0 - clean_ratio

    # Opacity-aware coverage score (common component)
    p = np.array([(mask == c).sum() / mask.size for c in range(4)], dtype=np.float32)
    S_op = float((w * p).sum())

    # Spatial weighted score (common component)
    Wmap = make_spatial_weight(H, W, mode=spatial_mode)
    S_sp = float((Wmap * w[mask]).sum())

    # Dominance with different eta values
    S_dom_eta00, _, _, _ = compute_dominance_class_aware(
        tile_cov, class_weights=class_weights, eta_trans=0.0
    )
    S_dom_eta09, _, _, _ = compute_dominance_class_aware(
        tile_cov, class_weights=class_weights, eta_trans=0.9
    )

    # Variant 1: Simple mean only
    S_simple = s

    # Variant 2: Opacity-aware only
    S_op_only = S_op

    # Variant 3: Opacity + Spatial (no dominance)
    S_op_sp = 0.7 * S_op + 0.3 * S_sp

    # Variant 4: Full with eta=0.9 (default)
    S_full = 0.6 * S_op + 0.3 * S_sp + 0.1 * S_dom_eta09

    # Variant 5: Full with eta=0 (no transparency discount)
    S_full_eta00 = 0.6 * S_op + 0.3 * S_sp + 0.1 * S_dom_eta00

    return {
        "s": np.float32(S_simple),
        "S_op_only": np.float32(S_op_only),
        "S_op_sp": np.float32(S_op_sp),
        "S_full": np.float32(S_full),
        "S_full_eta00": np.float32(S_full_eta00),
        # Also save components for analysis
        "S_op": np.float32(S_op),
        "S_sp": np.float32(S_sp),
        "S_dom_eta00": np.float32(S_dom_eta00),
        "S_dom_eta09": np.float32(S_dom_eta09),
    }

# 新增：映射以上复合评分至 global level
def score_to_level(S, bins=(0.02, 0.08, 0.20)):
    b1, b2, b3 = bins
    if S <= b1:
        return 0
    elif S <= b2:
        return 1
    elif S <= b3:
        return 2
    else:
        return 3


def main():
    proj = Path.home() / "soiling_project"
    raw_root = proj / "dataset/woodscape_raw"
    out_root = proj / "dataset/woodscape_processed"
    meta_root = out_root / "meta"
    manifest_path = meta_root / "manifest_woodscape.csv"

    n, m, C = 8, 8, 4

    df = pd.read_csv(manifest_path)
    df = df[(df["ok_img"] == True) & (df["ok_gt"] == True) & (df["ok_label_values_0_3"] == True)].copy()

    label_dir = out_root / "labels_tile"
    label_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="build labels"):
        rel_img = r["rel_img"]
        img_abs = raw_root / rel_img
        gt_abs = img_rel_to_gt_abs(raw_root, rel_img)
        if not gt_abs.exists():
            raise FileNotFoundError(f"Missing gt: {gt_abs}")

        mask = np.array(Image.open(gt_abs))
        if mask.ndim == 3:
            mask = np.array(Image.open(gt_abs).convert("L"))
        mask = mask.astype(np.int64)

        # 安全检查：只允许 0..3
        u = np.unique(mask)
        if np.any((u < 0) | (u > 3)):
            raise ValueError(f"Unexpected label values {u} in {gt_abs}")

        tile_cov = compute_tile_cov(mask, n=n, m=m, C=C)

        # tile_cov sanity：每 tile 四类和应接近 1
        s = tile_cov.sum(axis=-1)
        if not np.allclose(s, 1.0, atol=1e-4):
            raise ValueError(f"Tile cov sum != 1 for {rel_img}")

        # 0131 -> 仅通过tile脏污和以及一个预定的soiling_thr来定义 global_bin 为0或1对于下游任务的权重影响太大，粒度过粗；
        # -> 考虑更全面的全局标签制定体系：
        # -> 结合脏污透明度、全局脏污覆盖面积、脏污覆盖的空间位置重要性 以及 脏污覆盖的形态集中度来合成每一张脏污图像的全局最终评分
        
        # 1.1 透明度危害权重: w_3 > w_2 > w_1 > w_0 == 0
        # 1.2 空间位置重要性: 例如，对于车载镜头来说，简单点，画面上方1/3空间脏污危害性更小(主要的检测目标出现在上方画面区域的可能性较低), 目前考虑用高斯空间权重
        # 1.3 形态集中度：从 tile_cov 得到的每个tile的 非clean 覆盖率，找到最高覆盖率的tile, 它是否出现"大块的遮挡", 对下游感知影响较大；

        total = float(mask.size)
        # global_cov_per_class = np.array([(mask == c).sum() / total for c in range(C)], dtype=np.float32)
        # global_soiling = float(1.0 - global_cov_per_class[0])
        # global_bin = int(global_soiling > soiling_thr)
        global_cov_per_class = np.array([(mask==c).sum() / total for c in range(C)], dtype=np.float32)
        global_soiling = float(1.0 - global_cov_per_class[0])

        # Compute all Severity Score variants for ablation experiments
        severity_variants = compute_all_severity_variants(
            mask=mask,
            tile_cov=tile_cov,
            class_weights=(0.0, 0.33, 0.66, 1.0),
            spatial_mode="gaussian"
        )

        # For backward compatibility, use S_full as default global_score
        global_score = float(severity_variants["S_full"])
        global_level = score_to_level(global_score, bins=(0.02, 0.08, 0.20))
        global_bin = int(global_score > 0.10)

        out_npz = label_dir / (Path(rel_img).stem + ".npz")
        np.savez_compressed(
            out_npz,
            tile_cov=tile_cov.astype(np.float32),
            global_cov_per_class=global_cov_per_class,
            global_soiling=np.float32(global_soiling),
            # Default: S (backward compatibility, same as S_full)
            S=np.float32(global_score),
            s=severity_variants["s"],  # Simple mean (baseline)
            S_op_only=severity_variants["S_op_only"],  # Opacity-only
            S_op_sp=severity_variants["S_op_sp"],  # Opacity + Spatial
            S_full=severity_variants["S_full"],  # Full Severity Score (default)
            S_full_eta00=severity_variants["S_full_eta00"],  # Full with eta=0
            # Component scores (for analysis)
            S_op=severity_variants["S_op"],
            S_sp=severity_variants["S_sp"],
            S_dom_eta00=severity_variants["S_dom_eta00"],
            S_dom_eta09=severity_variants["S_dom_eta09"],
            # Legacy fields for backward compatibility
            global_score=np.float32(global_score),
            glboal_level=np.int64(global_level),
            S_dom=severity_variants["S_dom_eta09"],  # Default uses eta=0.9
            rel_img=rel_img,
            rel_gt=gt_abs.relative_to(raw_root).as_posix(),
        )

        rows.append({
            "rel_img": rel_img,
            "label_npz": out_npz.relative_to(out_root).as_posix(),
            "global_soiling": global_soiling,
            "global_score": global_score,
            "global_level": global_level,
            "global_bin": global_bin,
            # Add severity variants to CSV for analysis
            "s": float(severity_variants["s"]),
            "S_op_only": float(severity_variants["S_op_only"]),
            "S_op_sp": float(severity_variants["S_op_sp"]),
            "S_full": float(severity_variants["S_full"]),
            "S_full_eta00": float(severity_variants["S_full_eta00"]),
            "S_op": float(severity_variants["S_op"]),
            "S_sp": float(severity_variants["S_sp"]),
            "S_dom_eta00": float(severity_variants["S_dom_eta00"]),
            "S_dom_eta09": float(severity_variants["S_dom_eta09"]),
        })

    out_index = meta_root / "labels_index.csv"
    pd.DataFrame(rows).to_csv(out_index, index=False, encoding="utf-8-sig")
    print(f"saved: {out_index}")

if __name__ == "__main__":
    main()
