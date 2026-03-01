#!/usr/bin/env python3
"""
Qualitative Analysis and Cleaning of External Test Labels

This script performs QUALITATIVE analysis based on Severity Score definition:
1. Samples images from each occlusion level for visual inspection
2. Provides evaluation guidelines based on Severity Score dimensions
3. Records manual assessment results
4. Supports remapping based on qualitative analysis

Severity Score Dimensions (for qualitative assessment):
    S = 0.6 * S_op + 0.3 * S_sp + 0.1 * S_dom

    Qualitative criteria:
    1. Opacity (S_op): Opaque > Semi-transparent > Transparent > Clean
    2. Spatial (S_sp): Center pollution > Edge pollution
    3. Dominance (S_dom): Concentrated severe > Distributed mild

Expected monotonicity: Level 5 > 4 > 3 > 2 > 1
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


def load_and_resize_image(img_path: str, target_size: tuple = (320, 240)):
    """Load and resize image for preview."""
    try:
        img = Image.open(img_path)
        img = img.resize(target_size)
        return img
    except Exception as e:
        print(f"Warning: Failed to load {img_path}: {e}")
        return None


def sample_images_by_level(csv_path: str, img_root: str, n_samples: int = 20) -> Dict:
    """Sample images from each occlusion level."""

    df = pd.read_csv(csv_path)

    samples = {}

    for level in sorted(df["ext_level"].unique()):
        level_df = df[df["ext_level"] == level]

        # Sample up to n_samples
        sampled = level_df.sample(n=min(n_samples, len(level_df)), random_state=42)

        samples[level] = {
            "total": len(level_df),
            "sampled": len(sampled),
            "images": sampled[["image_id", "image_path"]].to_dict("records"),
        }

    return samples


def create_inspection_sheet(samples: Dict, csv_path: str, out_dir: Path):
    """Create an inspection sheet for manual assessment."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataframe for reference
    df = pd.read_csv(csv_path)

    level_names = {
        5: "occlusion_a (heaviest)",
        4: "occlusion_b",
        3: "occlusion_c",
        2: "occlusion_d",
        1: "occlusion_e (lightest)",
    }

    # Create inspection file
    inspection_file = out_dir / "inspection_sheet.md"

    with open(inspection_file, "w", encoding="utf-8") as f:
        f.write("# External Test 標籤清洗檢查表\n\n")
        f.write("## Severity Score 定義（用於定性評估）\n\n")
        f.write("```\n")
        f.write("S = 0.6 * S_op + 0.3 * S_sp + 0.1 * S_dom\n\n")
        f.write("定性評估維度：\n")
        f.write("1. Opacity (S_op, 權重0.6): 不透明度\n")
        f.write("   - Opaque (不透明) > Semi-transparent (半透明) > Transparent (透明) > Clean (清潔)\n\n")
        f.write("2. Spatial (S_sp, 權重0.3): 空間位置\n")
        f.write("   - 中心髒污 > 邊緣髒污\n\n")
        f.write("3. Dominance (S_dom, 權重0.1): 主導性\n")
        f.write("   - 局部嚴重髒污 > 均勻輕微髒污\n")
        f.write("```\n\n")

        f.write("## 預期單調性\n\n")
        f.write("Level 5 > 4 > 3 > 2 > 1\n\n")
        f.write("| Level | 標籤 | 預期特徵 |\n")
        f.write("|-------|------|----------|\n")
        f.write("| 5 | occlusion_a | 大量 opaque，中心嚴重髒污 |\n")
        f.write("| 4 | occlusion_b | 較多 opaque 或 semi-transparent |\n")
        f.write("| 3 | occlusion_c | 中等混合，semi-transparent 為主 |\n")
        f.write("| 2 | occlusion_d | 輕微，transparent 為主 |\n")
        f.write("| 1 | occlusion_e | 最輕，主要是 clean |\n\n")

        f.write("## 檢查說明\n\n")
        f.write("對每個 level 的抽樣圖片進行視覺檢查，評估：\n")
        f.write("1. 該 level 的整體嚴重程度是否符合預期\n")
        f.write("2. 是否存在明顯錯誤分類的樣本\n")
        f.write("3. 記錄需要重新映射的樣本\n\n")

        f.write("---\n\n")

        for level in sorted(samples.keys()):
            level_data = samples[level]
            name = level_names.get(level, f"level_{level}")

            f.write(f"## Level {level}: {name}\n\n")
            f.write(f"**總樣本數**: {level_data['total']}\n")
            f.write(f"**抽樣數**: {level_data['sampled']}\n\n")

            f.write("### 檢查要點\n")
            if level == 5:
                f.write("- [ ] 是否以 opaque 髒污為主\n")
                f.write("- [ ] 是否有明顯的中心髒污\n")
                f.write("- [ ] 整體嚴重程度是否為最高\n\n")
            elif level == 4:
                f.write("- [ ] 是否有較多的 opaque 或 semi-transparent\n")
                f.write("- [ ] 嚴重程度是否低於 Level 5\n")
                f.write("- [ ] 是否高於 Level 3\n\n")
            elif level == 3:
                f.write("- [ ] 是否以 semi-transparent 為主\n")
                f.write("- [ ] 嚴重程度是否中等\n")
                f.write("- [ ] **關鍵檢查**: 是否有部分樣本應該歸入 Level 2\n\n")
            elif level == 2:
                f.write("- [ ] 是否以 transparent 為主\n")
                f.write("- [ ] 嚴重程度是否較輕\n")
                f.write("- [ ] **關鍵檢查**: 是否有部分樣本嚴重程度高於 Level 3\n\n")
            elif level == 1:
                f.write("- [ ] 是否主要是 clean\n")
                f.write("- [ ] 嚴重程度是否最低\n\n")

            f.write("### 樣本列表\n\n")
            f.write("| image_id | image_path | 評估 | 備註 |\n")
            f.write("|----------|------------|------|------|\n")

            for img in level_data["images"]:
                f.write(f"| {img['image_id']} | {img['image_path']} | | |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("## 重映射決策\n\n")
        f.write("根據檢查結果，記錄需要重新映射的規則：\n\n")
        f.write("```json\n")
        f.write("{\n")
        f.write('  "remap_rules": [\n')
        f.write('    {"from": "occlusion_c", "to": "occlusion_d", "reason": "..."},\n')
        f.write('    {"from": "occlusion_d", "to": "occlusion_c", "reason": "..."}\n')
        f.write('  ]\n')
        f.write("}\n")
        f.write("```\n")

    print(f"檢查表已生成: {inspection_file}")
    return inspection_file


def create_visual_montage(samples: Dict, img_root: str, out_dir: Path):
    """Create visual montages for each level."""

    out_dir = Path(out_dir)
    montage_dir = out_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)

    level_names = {
        5: "level5_occlusion_a",
        4: "level4_occlusion_b",
        3: "level3_occlusion_c",
        2: "level2_occlusion_d",
        1: "level1_occlusion_e",
    }

    for level in sorted(samples.keys()):
        level_data = samples[level]
        images = level_data["images"]

        # Load images
        imgs = []
        for img_info in images[:16]:  # Max 16 images per montage
            img_path = img_info["image_path"]
            if not os.path.isabs(img_path):
                img_path = Path(img_root) / img_path
            else:
                img_path = Path(img_path)

            img = load_and_resize_image(str(img_path))
            if img is not None:
                imgs.append((img_info["image_id"], img))

        if not imgs:
            continue

        # Create montage (4x4 grid)
        n_rows = 4
        n_cols = 4
        thumb_w, thumb_h = 160, 120

        montage = Image.new("RGB", (n_cols * thumb_w, n_rows * thumb_h), (128, 128, 128))

        for idx, (img_id, img) in enumerate(imgs):
            row = idx // n_cols
            col = idx % n_cols
            img = img.resize((thumb_w, thumb_h))

            # Add label
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)

            # Simple label (no font required)
            label = f"{img_id[:12]}..."
            draw.rectangle([(0, 0), (len(label)*6, 12)], fill=(0, 0, 0))
            draw.text((2, 0), label, fill=(255, 255, 255))

            montage.paste(img, (col * thumb_w, row * thumb_h))

        montage_path = montage_dir / f"{level_names[level]}.jpg"
        montage.save(montage_path, quality=90)
        print(f"  蒙太奇已保存: {montage_path}")


def print_analysis_summary(df: pd.DataFrame):
    """Print summary statistics for qualitative analysis."""

    level_names = {
        5: "occlusion_a (heaviest)",
        4: "occlusion_b",
        3: "occlusion_c",
        2: "occlusion_d",
        1: "occlusion_e (lightest)",
    }

    print("\n" + "=" * 80)
    print("External Test 標籤定性分析")
    print("=" * 80)
    print()

    print("Severity Score 定義（定性評估維度）：")
    print("  1. Opacity (權重0.6): Opaque > Semi-transparent > Transparent > Clean")
    print("  2. Spatial (權重0.3): 中心髒污 > 邊緣髒污")
    print("  3. Dominance (權重0.1): 局部嚴重 > 均勻輕微")
    print()

    print("預期單調性: Level 5 > 4 > 3 > 2 > 1")
    print()

    print("-" * 80)
    print(f"{'Level':<8} {'標籤':<25} {'樣本數':<10} {'比例':<10}")
    print("-" * 80)

    total = len(df)
    for level in [5, 4, 3, 2, 1]:
        count = len(df[df["ext_level"] == level])
        ratio = count / total * 100 if total > 0 else 0
        name = level_names.get(level, f"level_{level}")
        print(f"{level:<8} {name:<25} {count:<10} {ratio:<10.1f}%")

    print()

    print("定性檢查要點：")
    print()
    print("Level 5 (occlusion_a):")
    print("  - 應該是嚴重程度最高的")
    print("  - 應該包含大量 opaque 髒污")
    print("  - 中心區域應該有明顯髒污")
    print()
    print("Level 3 vs Level 2 (關鍵檢查):")
    print("  - **Level 3 應該比 Level 2 更嚴重**")
    print("  - 檢查 occlusion_c 和 occlusion_d 的定義是否符合 Severity Score")
    print("  - 如果 occlusion_d 比 occlusion_c 更嚴重，需要重映射")
    print()


def main():
    ap = argparse.ArgumentParser(description="Qualitative analysis of external test labels")
    ap.add_argument("--ext_csv", default="dataset/my_external_test/test_ext.csv",
                    help="Path to external test CSV")
    ap.add_argument("--img_root", default=".", help="Root directory for images")
    ap.add_argument("--out_dir", default="dataset/external_test_qualitative",
                    help="Output directory for analysis")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="Number of samples to inspect per level")
    ap.add_argument("--create_montages", action="store_true",
                    help="Create visual montages for each level")
    args = ap.parse_args()

    # Load dataframe for summary
    df = pd.read_csv(args.ext_csv)

    # Print summary
    print_analysis_summary(df)

    # Sample images
    print(f"\n從每個 level 抽取 {args.n_samples} 張圖片...")
    samples = sample_images_by_level(args.ext_csv, args.img_root, args.n_samples)

    # Create inspection sheet
    print("\n生成檢查表...")
    inspection_file = create_inspection_sheet(samples, args.ext_csv, args.out_dir)

    # Create visual montages if requested
    if args.create_montages:
        print("\n生成視覺蒙太奇...")
        create_visual_montage(samples, args.img_root, args.out_dir)

    print("\n" + "=" * 80)
    print("定性分析準備完成！")
    print("=" * 80)
    print()
    print("下一步：")
    print(f"1. 打開檢查表: {inspection_file}")
    print("2. 查看抽樣圖片，根據 Severity Score 定義進行人工評估")
    print("3. 記錄需要重映射的樣本和規則")
    if args.create_montages:
        print(f"4. 查看蒙太奇圖: {args.out_dir}/montages/")
    print()


if __name__ == "__main__":
    main()
