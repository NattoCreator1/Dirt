# scripts/01_make_manifest.py
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png"}

def find_dir(raw_root: Path, subset: str, name1: str, name2: str):
    c1 = raw_root / subset / name1
    c2 = raw_root / subset / name2
    if c1.exists(): return c1
    if c2.exists(): return c2
    raise FileNotFoundError(f"Missing {subset}/{name1} or {subset}/{name2}")

def main():
    raw_root = Path.home() / "soiling_project/dataset/woodscape_raw"
    out_csv = Path.home() / "soiling_project/dataset/woodscape_processed/meta/manifest_woodscape.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for subset in ["train", "test"]:
        img_root = find_dir(raw_root, subset, "rbgImages", "rgbImages")
        gt_root  = raw_root / subset / "gtLabels"
        if not gt_root.exists():
            raise FileNotFoundError(f"Missing {subset}/gtLabels")

        img_paths = [p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        for p in tqdm(img_paths, desc=f"scan {subset}"):
            rel_img = p.relative_to(raw_root).as_posix()
            gt_path = gt_root / p.name
            rel_gt  = gt_path.relative_to(raw_root).as_posix()

            ok_img = True
            w = h = None
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                ok_img = False

            ok_gt = gt_path.exists()
            # 只在 gt 存在时做最基本的 unique 校验（读取会稍慢，但能避免返工）
            ok_val = None
            if ok_gt:
                arr = np.array(Image.open(gt_path))
                u = np.unique(arr)
                ok_val = bool(np.all((u >= 0) & (u <= 3)))

            rows.append({
                "subset": subset,
                "rel_img": rel_img,
                "rel_gt": rel_gt if ok_gt else "",
                "img_name": p.name,
                "width": w,
                "height": h,
                "ok_img": ok_img,
                "ok_gt": ok_gt,
                "ok_label_values_0_3": ok_val,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"saved: {out_csv}")
    print("Counts by subset:")
    print(df.groupby("subset").size())
    print("Missing gtLabels:")
    print(df[df["ok_gt"] == False].groupby("subset").size())
    print("Bad label values (not in 0..3):")
    print(df[df["ok_label_values_0_3"] == False].groupby("subset").size())

if __name__ == "__main__":
    main()
