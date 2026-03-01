# scripts/02_split_by_list.py
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    meta_root = Path.home() / "soiling_project/dataset/woodscape_processed/meta"
    manifest_path = meta_root / "manifest_woodscape.csv"
    split_dir = meta_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    # 过滤坏样本
    df = df[(df["ok_img"] == True) & (df["ok_gt"] == True) & (df["ok_label_values_0_3"] == True)].copy()

    train_all = df[df["subset"] == "train"].copy()
    test_all  = df[df["subset"] == "test"].copy()

    # 固定官方 test
    test_out = test_all[["rel_img"]].rename(columns={"rel_img": "rel_path"})
    test_out.to_csv(split_dir / "test_real.csv", index=False, encoding="utf-8-sig")

    # 从 train(4000) 内抽 val=800，余下 3200 为 train_real
    seed = 42
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_all))
    rng.shuffle(idx)

    n_val = 800
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    val_out = train_all.iloc[val_idx][["rel_img"]].rename(columns={"rel_img": "rel_path"})
    trn_out = train_all.iloc[trn_idx][["rel_img"]].rename(columns={"rel_img": "rel_path"})

    val_out.to_csv(split_dir / "val_real.csv", index=False, encoding="utf-8-sig")
    trn_out.to_csv(split_dir / "train_real.csv", index=False, encoding="utf-8-sig")

    print("Split summary:")
    print(f"train_real: {len(trn_out)}")
    print(f"val_real:   {len(val_out)}")
    print(f"test_real:  {len(test_out)}")

if __name__ == "__main__":
    main()
