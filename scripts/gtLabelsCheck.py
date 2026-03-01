import numpy as np
from PIL import Image
from pathlib import Path
import random

raw_root = Path.home() /"soiling_project/dataset/woodscape_raw"
gt_paths = list((raw_root/"train"/"gtLabels").rglob("*.png"))
random.shuffle(gt_paths)

for p in gt_paths[:20]:
    arr = np.array(Image.open(p))
    u = np.unique(arr)
    print(p.name, u[:20], "...", "len=", len(u))