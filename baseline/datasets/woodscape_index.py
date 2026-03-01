import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import augmentation module
from ..data_augmentation import CameraDomainRandomization


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for k in candidates:
        if k in s:
            return k
    return None


def _resolve_path(p: str, base: Optional[str]) -> str:
    if p is None:
        return None
    p = str(p)
    if os.path.isabs(p):
        return p
    if base is None:
        return p
    return os.path.normpath(os.path.join(base, p))


@dataclass
class WoodscapeIndexSpec:
    index_csv: str
    img_root: str                           # usually dataset/woodscape_raw
    labels_tile_dir: Optional[str] = None   # usually dataset/woodscape_processed/labels_tile
    split_col: Optional[str] = None         # auto-detect if None
    split_value: Optional[str] = None       # e.g. "train"/"val"/"test" or "Train_Real"/...
    global_target: str = "S"                # "S" (default full), "s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00"
    resize_w: int = 640
    resize_h: int = 480

    # Data augmentation
    augmentation: Optional[Callable] = None  # Callable for online data augmentation

    # Valid severity targets for ablation experiments
    VALID_SEVERITY_TARGETS = {
        "S", "s", "S_op_only", "S_op_sp", "S_full", "S_full_eta00",
        "S_full_wgap_alpha50"  # Two weight systems experiment: w_gap_strong + alpha=0.5
    }


class WoodscapeTileDataset(Dataset):
    """
    One-row-per-image index CSV dataset.
    Expects to be able to locate:
      - RGB image path (absolute or relative)
      - tile npz path OR derive from labels_tile_dir + stem + .npz
      - global score column 'S'/'s' OR load from npz if present
    """
    def __init__(self, spec: WoodscapeIndexSpec):
        self.spec = spec
        df = pd.read_csv(spec.index_csv)

        # auto-detect split column if not given
        if spec.split_col is None:
            split_col = _pick_col(df.columns.tolist(), ["split", "subset", "domain", "set"])
        else:
            split_col = spec.split_col
        self.split_col = split_col

        if self.split_col is not None and spec.split_value is not None:
            df = df[df[self.split_col].astype(str) == str(spec.split_value)].copy()

        # pick image path column
        img_col = _pick_col(df.columns.tolist(), [
            "rgb_path", "rgb", "image_path", "image", "img_path", "path",
            "rgb_relpath", "rgb_rel", "rgb_file", "filename", "fname"
        ])
        if img_col is None:
            raise RuntimeError(f"Cannot find image path column in {spec.index_csv}")
        self.img_col = img_col

        # pick npz column (optional)
        npz_col = _pick_col(df.columns.tolist(), ["npz_path", "tile_npz", "label_npz", "npz", "tile_path"])
        self.npz_col = npz_col

        self.df = df.reset_index(drop=True)

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if spec.global_target not in WoodscapeIndexSpec.VALID_SEVERITY_TARGETS:
            raise ValueError(f"global_target must be one of {WoodscapeIndexSpec.VALID_SEVERITY_TARGETS}, got '{spec.global_target}'")

    def __len__(self):
        return len(self.df)

    def _load_npz(self, npz_path: str) -> Dict[str, Any]:
        d = np.load(npz_path, allow_pickle=True)
        out = {k: d[k] for k in d.files}
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # resolve image path
        img_path = _resolve_path(row[self.img_col], self.spec.img_root)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.spec.resize_w, self.spec.resize_h), interpolation=cv2.INTER_AREA)

        # Apply data augmentation (if enabled)
        # Note: Augmentation is applied BEFORE normalization
        # This ensures augmentation operates on raw pixel values [0, 255]
        if self.spec.augmentation is not None:
            rgb = self.spec.augmentation(rgb)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # [3,H,W]

        # resolve npz path
        npz_path = None
        if self.npz_col is not None and str(row[self.npz_col]) != "nan":
            npz_path = _resolve_path(row[self.npz_col], None)
        else:
            if self.spec.labels_tile_dir is None:
                raise RuntimeError("labels_tile_dir is None and no npz_path column found.")
            stem = os.path.splitext(os.path.basename(img_path))[0]
            npz_path = os.path.join(self.spec.labels_tile_dir, stem + ".npz")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        npz = self._load_npz(npz_path)

        if "tile_cov" not in npz:
            raise RuntimeError(f"'tile_cov' not in npz: {npz_path}")
        tile_cov = npz["tile_cov"].astype(np.float32)  # (8,8,4)
        tile_cov_t = torch.from_numpy(tile_cov)        # [8,8,4]

        # global score: load from npz using severity_target
        gt_key = self.spec.global_target
        global_score = None

        # Try CSV first for backward compatibility
        if gt_key in self.df.columns:
            global_score = float(row[gt_key])
        elif gt_key in npz:
            global_score = float(npz[gt_key])
        elif "S" in npz and gt_key == "S":
            # Fallback: if "S" is requested but not in npz, use global_score
            global_score = float(npz.get("global_score", npz["S"]))
        elif "global_score" in npz:
            # Final fallback: use global_score for any target
            global_score = float(npz["global_score"])
        else:
            raise RuntimeError(f"Global target '{gt_key}' not found in CSV columns nor NPZ keys. Available NPZ keys: {list(npz.keys())}")

        y = torch.tensor([global_score], dtype=torch.float32)  # [1]

        return {
            "image": x,
            "tile_cov": tile_cov_t,
            "global_score": y,
            "img_path": img_path,
            "npz_path": npz_path,
        }
