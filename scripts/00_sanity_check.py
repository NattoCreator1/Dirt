# scripts/00_sanity_check.py
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    proj = Path.home() / "soiling_project"
    out_root = proj / "dataset/woodscape_processed"
    meta = out_root / "meta"
    splits = meta / "splits"

    idx = pd.read_csv(meta / "labels_index.csv")
    idx_map = dict(zip(idx["rel_img"], idx["label_npz"]))

    for name in ["train_real", "val_real", "test_real"]:
        sp = pd.read_csv(splits / f"{name}.csv")
        missing = [rel for rel in sp["rel_path"].tolist() if rel not in idx_map]
        print(name, "n=", len(sp), "missing_labels=", len(missing))
        if missing:
            print("example missing:", missing[:5])

    # 抽样检查若干 npz
    sample = idx.sample(n=min(50, len(idx)), random_state=0)
    for rel_img, npz_rel in zip(sample["rel_img"], sample["label_npz"]):
        z = np.load(out_root / npz_rel)
        tile = z["tile_cov"]
        if tile.ndim != 3 or tile.shape[-1] != 4:
            raise ValueError(f"bad tile_cov shape {tile.shape} for {rel_img}")
        s = tile.sum(axis=-1)
        if not np.allclose(s, 1.0, atol=1e-4):
            raise ValueError(f"tile sum != 1 for {rel_img}, min={s.min()}, max={s.max()}")
    print("sanity check passed")

if __name__ == "__main__":
    main()
