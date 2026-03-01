#!/usr/bin/env python3
"""
阶段0：数据状态检查脚本

检查所有相关数据集的状态，为SD合成数据增强实验做准备。
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
import numpy as np


# 项目根目录
PROJECT_ROOT = Path("/home/yf/soiling_project")
DATASET_ROOT = PROJECT_ROOT / "dataset"


def count_files(directory: Path, pattern: str = "*.png") -> int:
    """统计目录中符合模式的文件数量"""
    if not directory.exists():
        return 0
    return len(list(directory.rglob(pattern)))


def check_woodscape_raw() -> Dict:
    """检查WoodScape原始数据"""
    print("\n" + "="*60)
    print("1. WoodScape 原始数据 (woodscape_raw)")
    print("="*60)

    ws_raw = DATASET_ROOT / "woodscape_raw"

    result = {
        "train_rgb": 0,
        "train_gt": 0,
        "test_rgb": 0,
        "test_gt": 0,
    }

    for split in ["train", "test"]:
        rgb_dir = ws_raw / split / "rgbImages"
        gt_dir = ws_raw / split / "gtLabels"

        rgb_count = count_files(rgb_dir)
        gt_count = count_files(gt_dir)

        result[f"{split}_rgb"] = rgb_count
        result[f"{split}_gt"] = gt_count

        print(f"\n{split.upper()}:")
        print(f"  RGB Images: {rgb_count}")
        print(f"  GT Labels:   {gt_count}")

    # 检查数据对齐
    train_aligned = result["train_rgb"] == result["train_gt"]
    test_aligned = result["test_rgb"] == result["test_gt"]
    print(f"\n数据对齐检查:")
    print(f"  Train: {'✓ 对齐' if train_aligned else '✗ 不对齐'}")
    print(f"  Test:  {'✓ 对齐' if test_aligned else '✗ 不对齐'}")

    return result


def check_woodscape_processed() -> Dict:
    """检查WoodScape处理后数据"""
    print("\n" + "="*60)
    print("2. WoodScape 处理后数据 (woodscape_processed)")
    print("="*60)

    ws_proc = DATASET_ROOT / "woodscape_processed"

    result = {
        "has_meta": False,
        "has_labels_tile": False,
        "has_splits": False,
        "index_files": [],
        "split_info": {},
    }

    # 检查meta目录
    meta_dir = ws_proc / "meta"
    if meta_dir.exists():
        result["has_meta"] = True
        print(f"\n✓ meta 目录存在")

        # 查找索引文件
        index_files = [
            "labels_index.csv",
            "labels_index_rebinned_baseline.csv",
            "labels_index_ablation.csv",
        ]

        for idx_file in index_files:
            file_path = meta_dir / idx_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                result["index_files"].append(idx_file)
                print(f"\n  {idx_file}:")
                print(f"    样本数: {len(df)}")

                # 检查是否有split列
                if "split" in df.columns:
                    split_counts = df["split"].value_counts().to_dict()
                    result["split_info"][idx_file] = split_counts
                    print(f"    划分: {split_counts}")

    # 检查labels_tile目录
    labels_tile_dir = ws_proc / "labels_tile"
    if labels_tile_dir.exists():
        result["has_labels_tile"] = True
        tile_count = count_files(labels_tile_dir, "*.npz")
        print(f"\n✓ labels_tile 目录存在 ({tile_count} 个 .npz 文件)")

    return result


def check_clean_data() -> Dict:
    """检查Clean数据（自采干净帧）"""
    print("\n" + "="*60)
    print("3. Clean 数据 (自采干净帧)")
    print("="*60)

    result = {}

    # 检查原始clean帧
    clean_frames_dir = DATASET_ROOT / "my_clean_frames"
    clean_4by3_dir = DATASET_ROOT / "my_clean_frames_4by3"

    if clean_4by3_dir.exists():
        total_count = 0
        split_counts = {}

        # 统计各方向数量
        for direction in clean_4by3_dir.iterdir():
            if direction.is_dir():
                count = count_files(direction, "*")
                split_counts[direction.name] = count
                total_count += count

        result["clean_4by3_total"] = total_count
        result["clean_4by3_by_direction"] = split_counts

        print(f"\n✓ my_clean_frames_4by3/ 存在")
        print(f"  总帧数: {total_count}")
        print(f"  各方向:")
        for direction, count in split_counts.items():
            print(f"    {direction}: {count}")

    # 检查manifest文件
    manifests_dir = DATASET_ROOT / "my_clean_manifests"
    if manifests_dir.exists():
        manifest_files = list(manifests_dir.glob("*.csv"))
        result["manifest_files"] = [f.name for f in manifest_files]
        print(f"\n✓ my_clean_manifests/ 存在")
        print(f"  Manifest文件: {len(manifest_files)}")
        for mf in manifest_files:
            print(f"    - {mf.name}")

    return result


def check_external_test() -> Dict:
    """检查External Test数据"""
    print("\n" + "="*60)
    print("4. External Test 数据 (自采弱标注)")
    print("="*60)

    result = {}

    # 检查external_test_washed
    ext_washed = DATASET_ROOT / "external_test_washed"
    if ext_washed.exists():
        occlusion_levels = [d for d in ext_washed.iterdir() if d.is_dir()]
        result["occlusion_levels"] = [d.name for d in occlusion_levels]

        total_count = 0
        level_counts = {}

        for level_dir in occlusion_levels:
            count = count_files(level_dir, "*.jpg")
            level_counts[level_dir.name] = count
            total_count += count

        result["total_count"] = total_count
        result["level_counts"] = level_counts

        print(f"\n✓ external_test_washed/ 存在")
        print(f"  总样本数: {total_count}")
        print(f"  各遮挡级别:")
        for level, count in level_counts.items():
            print(f"    {level}: {count}")

    # 检查处理后的CSV
    ext_proc = DATASET_ROOT / "external_test_washed_processed"
    if ext_proc.exists():
        csv_file = ext_proc / "test_ext.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            result["test_ext_rows"] = len(df)
            print(f"\n✓ external_test_washed_processed/test_ext.csv 存在")
            print(f"  样本数: {len(df)}")
            print(f"  列: {list(df.columns)}")

    return result


def check_baseline_models() -> Dict:
    """检查Baseline模型"""
    print("\n" + "="*60)
    print("5. Baseline 模型检查")
    print("="*60)

    result = {}

    # 检查ablation实验模型
    ablation_dir = PROJECT_ROOT / "baseline" / "runs" / "ablation_label_def"
    if ablation_dir.exists():
        experiments = [d for d in ablation_dir.iterdir() if d.is_dir()]
        result["ablation_experiments"] = [d.name for d in experiments]

        print(f"\n✓ ablation_label_def/ 存在")
        print(f"  实验数: {len(experiments)}")

        for exp_dir in experiments:
            # 查找checkpoint
            ckpt_files = list(exp_dir.glob("*.pth"))
            if ckpt_files:
                print(f"    - {exp_dir.name}: {len(ckpt_files)} checkpoints")

            # 查找训练日志
            log_file = exp_dir / "metrics.csv"
            if log_file.exists():
                print(f"      (有训练日志)")

    # 检查主模型
    main_runs = PROJECT_ROOT / "baseline" / "runs"
    if main_runs.exists():
        all_runs = [d for d in main_runs.iterdir() if d.is_dir()]
        result["all_runs"] = [d.name for d in all_runs]

    return result


def check_sd_requirements() -> Dict:
    """检查SD合成实验所需条件"""
    print("\n" + "="*60)
    print("6. SD 合成实验准备情况")
    print("="*60)

    result = {
        "ready": False,
        "missing": [],
        "warnings": [],
    }

    # 检查MaskBank目录
    mask_bank_dir = DATASET_ROOT / "mask_bank"
    if not mask_bank_dir.exists():
        result["missing"].append("mask_bank/ 目录不存在，需要创建")
    else:
        print(f"\n✓ mask_bank/ 目录已存在")

    # 检查合成数据目录
    synthetic_dir = DATASET_ROOT / "synthetic_soiling"
    if not synthetic_dir.exists():
        result["missing"].append("synthetic_soiling/ 目录不存在，需要创建")
    else:
        print(f"✓ synthetic_soiling/ 目录已存在")

    # 检查clean数据是否足够
    clean_4by3_dir = DATASET_ROOT / "my_clean_frames_4by3"
    if clean_4by3_dir.exists():
        clean_count = count_files(clean_4by3_dir, "*")
        if clean_count < 50000:
            result["warnings"].append(f"Clean数据量较少 ({clean_count} 帧不足以支撑80k训练集目标)")
        else:
            print(f"✓ Clean数据充足 ({clean_count} 帧)")

    # 检查配置文件
    config_file = PROJECT_ROOT / "severity_config.json"
    if not config_file.exists():
        result["missing"].append("severity_config.json 不存在，需要创建")
    else:
        print(f"✓ severity_config.json 已存在")

    result["ready"] = len(result["missing"]) == 0

    if result["warnings"]:
        print(f"\n⚠ 警告:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    if result["missing"]:
        print(f"\n✗ 缺失:")
        for missing in result["missing"]:
            print(f"  - {missing}")

    return result


def print_summary(woodscape: Dict, processed: Dict, clean: Dict,
                  external: Dict, baseline: Dict, sd_req: Dict):
    """打印总结"""
    print("\n" + "="*60)
    print("数据状态总结")
    print("="*60)

    print(f"\nWoodScape 原始数据:")
    print(f"  Train: {woodscape['train_rgb']} 样本")
    print(f"  Test:  {woodscape['test_rgb']} 样本")

    if processed["index_files"]:
        print(f"\nWoodScape 处理后数据:")
        print(f"  索引文件: {len(processed['index_files'])} 个")
        for idx_file in processed["index_files"]:
            if idx_file in processed["split_info"]:
                print(f"    - {idx_file}: {processed['split_info'][idx_file]}")

    if clean:
        print(f"\nClean 数据:")
        print(f"  总帧数: {clean.get('clean_4by3_total', 0)}")

    if external:
        print(f"\nExternal Test:")
        print(f"  总样本数: {external.get('total_count', 0)}")

    print(f"\nSD 合成实验准备:")
    if sd_req["ready"]:
        print(f"  ✓ 所需目录和配置文件已就绪")
    else:
        print(f"  ✗ 需要创建 {len(sd_req['missing'])} 项")

    if sd_req["warnings"]:
        print(f"  ⚠ 有 {len(sd_req['warnings'])} 个警告")

    print("\n" + "="*60)
    print("建议下一步操作:")
    print("="*60)

    if not sd_req["ready"]:
        print("\n1. 创建缺失的目录和配置文件")
        print("   python sd_scripts/00_initialize_sd_structure.py")

    if sd_req["warnings"]:
        print("\n2. 解决警告事项")

    print("\n3. 构建MaskBank (阶段1)")
    print("   python sd_scripts/mask_bank/01_build_mask_bank.py")

    print("\n4. 实现合成生成器 (阶段2)")
    print("   python sd_scripts/generation/02_generate_synthetic.py")


def main():
    """主函数"""
    print("="*60)
    print("SD 合成数据增强 - 阶段0: 数据状态检查")
    print("="*60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据根目录: {DATASET_ROOT}")

    # 执行各项检查
    woodscape = check_woodscape_raw()
    processed = check_woodscape_processed()
    clean = check_clean_data()
    external = check_external_test()
    baseline = check_baseline_models()
    sd_req = check_sd_requirements()

    # 打印总结
    print_summary(woodscape, processed, clean, external, baseline, sd_req)

    print("\n✓ 数据状态检查完成")


if __name__ == "__main__":
    main()
