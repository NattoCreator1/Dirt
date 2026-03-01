# scripts/08_build_baseline_index_from_splits.py
from pathlib import Path
import os
import pandas as pd

def norm_rel(p: str) -> str:
    """统一 rel_img/rel_path 的格式，方便集合匹配。"""
    s = str(p).replace("\\", "/").strip()
    s = s.lstrip("./")
    # 如果不小心带了 dataset/woodscape_raw 前缀，去掉
    prefix = "dataset/woodscape_raw/"
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s

def norm_rgb_path(rel_img: str) -> str:
    rel = norm_rel(rel_img)
    # 变成从项目根目录可直接访问的路径
    return os.path.normpath(os.path.join("dataset/woodscape_raw", rel))

def norm_npz_path(p: str, meta_root: Path) -> str:
    s = str(p).replace("\\", "/").strip().lstrip("./")
    if os.path.isabs(s):
        return s
    if s.startswith("dataset/"):
        return os.path.normpath(s)
    if s.startswith("woodscape_processed/"):
        return os.path.normpath(os.path.join("dataset", s))
    # 默认按 meta 目录相对路径处理（最稳）
    return os.path.normpath(str(meta_root / s))

def npz_from_rgb_path(rgb_path: str) -> str:
    stem = os.path.splitext(os.path.basename(rgb_path))[0]
    return os.path.normpath(os.path.join("dataset/woodscape_processed/labels_tile", stem + ".npz"))


def load_split_set(csv_path: Path) -> set:
    df = pd.read_csv(csv_path)
    if "rel_path" not in df.columns:
        raise RuntimeError(f"{csv_path} missing column rel_path")
    return set(df["rel_path"].map(norm_rel).tolist())

def main():
    meta_root = Path("dataset/woodscape_processed/meta")
    src_index = meta_root / "labels_index_rebinned.csv"
    split_dir = meta_root / "splits"

    train_csv = split_dir / "train_real.csv"
    val_csv   = split_dir / "val_real.csv"
    test_csv  = split_dir / "test_real.csv"

    out_csv = meta_root / "labels_index_rebinned_baseline.csv"
    missing_csv = meta_root / "labels_index_rebinned_baseline_missing.csv"

    df = pd.read_csv(src_index)

    # 必要列检查
    need_cols = ["rel_img", "label_npz", "global_soiling", "global_score"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"{src_index} missing column: {c}")

    train_set = load_split_set(train_csv)
    val_set   = load_split_set(val_csv)
    test_set  = load_split_set(test_csv)

    # 统一 rel key
    df["rel_key"] = df["rel_img"].map(norm_rel)

    # 赋 split
    df["split"] = "drop"
    df.loc[df["rel_key"].isin(train_set), "split"] = "train"
    df.loc[df["rel_key"].isin(val_set),   "split"] = "val"
    df.loc[df["rel_key"].isin(test_set),  "split"] = "test"

    # 生成 Baseline 训练脚本更容易识别的列名
    df["rgb_path"] = df["rel_img"].map(norm_rgb_path)
    df["npz_path"] = df["rgb_path"].map(npz_from_rgb_path)


    # 标准化 global 目标字段
    df["S"] = df["global_score"].astype(float)
    df["s"] = df["global_soiling"].astype(float)

    # 输出统计
    print("Split counts:")
    print(df["split"].value_counts())

    # 把 drop 的样本另存（便于排查）
    df_missing = df[df["split"] == "drop"].copy()
    if len(df_missing) > 0:
        df_missing[["rel_img", "label_npz"]].to_csv(missing_csv, index=False, encoding="utf-8-sig")
        print("WARNING: some samples not found in split lists, wrote:", missing_csv)

    # 仅导出训练需要的列（其余保留也行，这里尽量精简）
    keep = ["rgb_path", "npz_path", "split", "S", "s"]
    # 如果你后续要用 global_bin/global_level 做分类头或采样，就一起带上
    for extra in ["global_level", "global_bin"]:
        if extra in df.columns:
            keep.append(extra)

    df_out = df[df["split"] != "drop"][keep].copy()
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("WROTE:", out_csv)
    print("Example rows:")
    print(df_out.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
