#!/usr/bin/env python3
"""
阶段0：初始化SD合成数据增强实验的目录结构和配置文件

创建所需的目录、配置文件，并生成统一的 severity_config.json
"""

import json
from pathlib import Path
import shutil

# 项目根目录
PROJECT_ROOT = Path("/home/yf/soiling_project")
DATASET_ROOT = PROJECT_ROOT / "dataset"


# ============================================================================
# 目录结构定义
# ============================================================================

DIRECTORY_STRUCTURE = {
    # MaskBank 目录
    "mask_bank": {
        "train": {
            "processed": "",  # 存放预处理后的mask
            "visualizations": "",  # 存放mask可视化图（可选）
        },
        "val": {
            "processed": "",
            "visualizations": "",
        },
    },

    # 合成数据目录
    "synthetic_soiling": {
        "v1.0_wgap_alpha50": {
            "npz": "",  # 存放 .npz 文件
            "images": "",  # 存放生成的图像（可选，用于可视化）
            "manifests": "",  # 存放清单文件
            "quality_report": "",  # 存放质量报告
        },
    },

    # CycleGAN 目录（可选，用于 q_D 计算）
    "cycle_gan": {
        "data": {
            "trainA": "",  # Clean域
            "trainB": "",  # WoodScape域
            "testA": "",
            "testB": "",
        },
        "models": "",
        "logs": "",
        "checkpoints": "",
    },

    # 实验运行目录
    "runs": {
        "sd_enhancement": {
            "exp001_s_baseline": "",
            "exp002_S_baseline": "",
            "exp003_s_filtered": "",
            "exp004_S_filtered": "",
        },
    },

    # 脚本目录
    "sd_scripts": {
        "mask_bank": "",
        "generation": "",
        "quality_control": "",
        "training": "",
    },
}


# ============================================================================
# Severity Score 配置（基于两权重系统验证结论）
# ============================================================================

SEVERITY_CONFIG = {
    "version": "v1.0_wgap_alpha50",
    "description": "Converged Severity Score for SD enhancement experiments",
    "reference": "docs/experiments/2025-02-09_two_weight_systems_validation.md",
    "creation_date": "2025-02-09",

    # 类别权重（w_gap_strong）
    "class_weights": {
        "clean": 0.0,
        "transparent": 0.15,
        "semi_transparent": 0.50,
        "opaque": 1.0,
        "_comment": "w_gap_strong: 扩大opaque与其它类的间隔"
    },

    # 融合系数（alpha50）
    "fusion_coeffs": {
        "alpha": 0.5,  # Opacity-aware
        "beta": 0.4,   # Spatial
        "gamma": 0.1,  # Dominance
        "_comment": "alpha50: 降低α，提高β，给予Spatial更多权重"
    },

    # 其他参数
    "eta_trans": 0.9,  # 透明度折扣因子
    "spatial": {
        "mode": "gaussian",
        "sigma": 0.5,
    },

    # Severity Score 公式
    "formula": "S_full = alpha * S_op + beta * S_sp + gamma * S_dom",
}


# ============================================================================
# 合成生成配置
# ============================================================================

GENERATION_CONFIG = {
    "version": "v1.0",
    "creation_date": "2025-02-09",

    # 数据源配置
    "data_sources": {
        "clean_frames_dir": "dataset/my_clean_frames_4by3",
        "mask_bank_train": "dataset/mask_bank/train",
        "mask_bank_val": "dataset/mask_bank/val",
        "woodscape_train": "dataset/woodscape_raw/train",
        "woodscape_val": "dataset/woodscape_processed/meta/splits/val.txt",
    },

    # 目标分辨率
    "target_resolution": [640, 480],
    "aspect_ratio": "4:3",

    # 生成策略
    "generation_strategy": {
        "allow_clean_reuse": True,  # 允许同一clean帧与不同mask组合
        "target_syn_samples": 10000,  # 目标合成样本数（调整后）
        "coverage_distribution": "uniform",  # uniform, real, custom
        "coverage_bins": 10,
    },

    # 质量控制阈值
    "quality_thresholds": {
        "background_ssim_min": 0.95,
        "background_diff_max": 0.05,
        "mask_diff_min": 0.1,
        "boundary_artifact_max": 0.1,
    },

    # q_D + q_T 配置
    "quality_control": {
        "use_qD": True,
        "use_qT": True,
        "qD_threshold": 0.5,
        "qT_threshold": 0.5,
        "cycle_gan_optional": True,  # 可选，计算成本高
    },
}


# ============================================================================
# 实验配置模板
# ============================================================================

EXPERIMENT_CONFIG_TEMPLATE = {
    "experiment_id": "",  # 待填充
    "target": "",  # "s" 或 "S_full_wgap_alpha50"
    "severity_config": "v1.0_wgap_alpha50",
    "generation_config": "v1.0",

    "data_config": {
        "real_source": "Train_Real",
        "syn_source": "Train_Syn",
        "real_samples": 3200,
        "syn_samples": 0,  # 待填充
        "sampling_strategy": "stratified",  # stratified, uniform, weighted
        "synthetic_config": "",  # "naive", "filtered", "weighted"
    },

    "model_config": {
        "backbone": "resnet18",
        "tile_size": 8,
        "pretrained": True,
        "dual_head": True,
    },

    "training_config": {
        "epochs": 40,
        "batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "seed": 42,

        "loss_weights": {
            "lambda_tile": 1.0,
            "lambda_global": 1.0,
            "mu_cons": 0.1,
        },
        "use_consistency_loss": True,
    },

    "quality_config": {
        "background_ssim_threshold": 0.95,
        "background_diff_threshold": 0.05,
        "qD_threshold": 0.5,
        "qT_threshold": 0.5,
        "use_q_filter": False,  # 根据实验组设置
    },

    "evaluation": {
        "woodscape_test": "dataset/woodscape_processed/meta/splits/test.txt",
        "external_test": "dataset/external_test_washed_processed/test_ext.csv",
        "mapping_rule": "scale_to_1_5",  # S ∈ [0,1] → [1,5]
    },
}


# ============================================================================
# 实验组定义
# ============================================================================

EXPERIMENT_GROUPS = {
    "G1": {
        "name": "Baseline",
        "description": "仅使用真实数据训练",
        "data_config": {
            "use_synthetic": False,
            "real_samples": 3200,
        },
        "priority": "P0",
    },
    "G2": {
        "name": "Filtered",
        "description": "真实数据 + 质量筛选后的合成数据",
        "data_config": {
            "use_synthetic": True,
            "syn_samples": 5000,
            "filtering": "hard_constraints_qD_qT",
        },
        "priority": "P0",
    },
    "G3": {
        "name": "Weighted",
        "description": "真实数据 + 动态加权的合成数据",
        "data_config": {
            "use_synthetic": True,
            "syn_samples": 5000,
            "sampling": "soft_weighting",
        },
        "priority": "P1",
    },
    "G4": {
        "name": "Naïve",
        "description": "真实数据 + 无筛选合成数据（对照）",
        "data_config": {
            "use_synthetic": True,
            "syn_samples": 5000,
            "filtering": "none",
        },
        "priority": "P2",
    },
}


def create_directory_structure(base_path: Path, structure: dict, prefix: str = ""):
    """递归创建目录结构"""
    for name, content in structure.items():
        full_path = base_path / name

        if isinstance(content, dict):
            # 这是一个目录，包含子目录
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {prefix}{name}/")
            create_directory_structure(full_path, content, prefix + "  ")
        else:
            # 这是一个空目录（叶子节点）
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {prefix}{name}/")


def save_config_files():
    """保存配置文件"""
    print("\n" + "="*60)
    print("创建配置文件")
    print("="*60)

    # 保存 severity_config.json 到项目根目录
    severity_config_path = PROJECT_ROOT / "severity_config.json"
    with open(severity_config_path, "w", encoding="utf-8") as f:
        json.dump(SEVERITY_CONFIG, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {severity_config_path.relative_to(PROJECT_ROOT)}/")

    # 保存 generation_config.json 到 dataset/
    generation_config_path = DATASET_ROOT / "generation_config.json"
    with open(generation_config_path, "w", encoding="utf-8") as f:
        json.dump(GENERATION_CONFIG, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {generation_config_path.relative_to(PROJECT_ROOT)}/")

    # 保存实验组配置
    experiment_groups_path = PROJECT_ROOT / "experiment_groups.json"
    with open(experiment_groups_path, "w", encoding="utf-8") as f:
        json.dump(EXPERIMENT_GROUPS, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {experiment_groups_path.relative_to(PROJECT_ROOT)}/")

    # 保存实验配置模板
    template_path = PROJECT_ROOT / "experiment_config_template.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(EXPERIMENT_CONFIG_TEMPLATE, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {template_path.relative_to(PROJECT_ROOT)}/")


def create_readme_files():
    """创建各目录的 README 文件"""
    print("\n" + "="*60)
    print("创建 README 文件")
    print("="*60)

    # MaskBank README
    mask_bank_readme = DATASET_ROOT / "mask_bank" / "README.md"
    with open(mask_bank_readme, "w", encoding="utf-8") as f:
        f.write("""# MaskBank - 脏污形态库

## 目录结构

- `train/`: 从 WoodScape Train 提取的 mask 形态库 (3,200 samples)
- `val/`: 从 WoodScape Val 提取的 mask 形态库 (800 samples)

## 严格数据隔离

- **禁止使用 WoodScape Test 的 mask**
- train/ 仅用于训练侧 mask 采样
- val/ 仅用于生成质量抽检和参数调优

## 文件格式

- `processed/`: 预处理后的 4 类 mask (640×480)
- `manifest.csv`: mask 清单文件
- `statistics.csv`: mask 统计信息

## 构建

运行 `python scripts/01_build_mask_bank.py` 构建 MaskBank。
""")
    print(f"  ✓ {mask_bank_readme.relative_to(PROJECT_ROOT)}/")

    # Synthetic_soiling README
    synthetic_readme = DATASET_ROOT / "synthetic_soiling" / "README.md"
    with open(synthetic_readme, "w", encoding="utf-8") as f:
        f.write("""# 合成脏污数据

## 目录结构

按 Severity 配置版本组织，如 `v1.0_wgap_alpha50/`

### 子目录

- `npz/`: 合成数据 .npz 文件（包含图像、mask、标签）
- `manifests/`: 数据清单文件
- `quality_report/`: 质量报告和诊断图表

## 文件格式

每个 .npz 文件包含：
- `image`: [H, W, 3] uint8 - 合成脏污图像
- `mask`: [H, W] uint8 - 对应的 mask (4类)
- `tile_cov`: [8, 8, 4] float32 - tile 覆盖率标签
- `s_simple`: float - 简单严重度 (1 - clean_ratio)
- `S_full`: float - 完整 Severity Score
- `S_op`, `S_sp`, `S_dom`: float - 各组件分数
- `generation_params_json`: str - 生成参数（JSON字符串）

## 生成

运行 `python scripts/02_generate_synthetic.py` 生成合成数据。
""")
    print(f"  ✓ {synthetic_readme.relative_to(PROJECT_ROOT)}/")


def print_summary():
    """打印总结"""
    print("\n" + "="*60)
    print("初始化完成总结")
    print("="*60)

    print("\n创建的目录结构:")
    print("  dataset/")
    print("    ├── mask_bank/")
    print("    │   ├── train/")
    print("    │   └── val/")
    print("    ├── synthetic_soiling/")
    print("    │   └── v1.0_wgap_alpha50/")
    print("    └── cycle_gan/ (可选)")
    print("  runs/")
    print("    └── sd_enhancement/")
    print("  sd_scripts/")

    print("\n创建的配置文件:")
    print("  ├── severity_config.json")
    print("  ├── generation_config.json")
    print("  ├── experiment_groups.json")
    print("  └── experiment_config_template.json")

    print("\n下一步操作:")
    print("  1. 构建 MaskBank:")
    print("     python scripts/01_build_mask_bank.py")
    print("\n  2. 生成合成数据:")
    print("     python scripts/02_generate_synthetic.py")
    print("\n  3. 训练模型:")
    print("     python baseline/train_baseline.py --index_csv ...")

    print("\n" + "="*60)


def main():
    """主函数"""
    print("="*60)
    print("SD 合成数据增强 - 阶段0: 初始化目录结构和配置")
    print("="*60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据根目录: {DATASET_ROOT}")

    # 创建目录结构
    print("\n" + "="*60)
    print("创建目录结构")
    print("="*60)

    create_directory_structure(DATASET_ROOT, DIRECTORY_STRUCTURE)
    create_directory_structure(PROJECT_ROOT / "runs", DIRECTORY_STRUCTURE["runs"])
    create_directory_structure(PROJECT_ROOT / "sd_scripts", DIRECTORY_STRUCTURE["sd_scripts"])

    # 保存配置文件
    save_config_files()

    # 创建 README 文件
    create_readme_files()

    # 打印总结
    print_summary()

    print("\n✓ 初始化完成")


if __name__ == "__main__":
    main()
