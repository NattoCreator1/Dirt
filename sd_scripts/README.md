# SD 合成数据增强脚本

本目录包含 SD（Synthetic Data）合成数据增强实验的所有脚本。

---

## 目录结构

```
sd_scripts/
├── README.md                          # 本文件
├── 00_check_data_status.py           # 数据状态检查
├── 00_initialize_sd_structure.py     # 初始化目录结构和配置
│
├── mask_bank/                         # MaskBank 构建相关
│   └── 01_build_mask_bank.py         # 从 WoodScape 构建 MaskBank
│
├── generation/                         # 合成数据生成相关
│   └── (待添加)
│
├── quality_control/                    # 质量控制相关
│   ├── compute_qD.py                 # 计算域相似度 q_D (CycleGAN)
│   ├── compute_qT.py                 # 计算任务一致性 q_T (Teacher)
│   └── filter_synthetic.py            # 筛选/加权合成数据
│
└── training/                           # 模型训练相关
    ├── train_sd_enhanced.py           # 训练 SD 增强模型
    └── evaluate_sd_experiments.py     # 评估 SD 实验结果
```

---

## 脚本使用流程

### 阶段0：数据检查与初始化

```bash
# 1. 检查数据状态
python sd_scripts/00_check_data_status.py

# 2. 初始化目录结构和配置文件
python sd_scripts/00_initialize_sd_structure.py
```

### 阶段1：构建 MaskBank

```bash
# 从 WoodScape Train/Val 构建 MaskBank
python sd_scripts/mask_bank/01_build_mask_bank.py
```

### 阶段2：生成合成数据

```bash
# 生成合成脏污数据（待实现）
python sd_scripts/generation/02_generate_synthetic.py
```

### 阶段3：质量控制（q_D + q_T）

```bash
# 计算 q_D（可选，需要 CycleGAN）
python sd_scripts/quality_control/compute_qD.py

# 计算 q_T
python sd_scripts/quality_control/compute_qT.py

# 筛选合成数据
python sd_scripts/quality_control/filter_synthetic.py
```

### 阶段4：模型训练

```bash
# 训练 SD 增强模型
python sd_scripts/training/train_sd_enhanced.py
```

---

## 与原有 scripts/ 的区别

| 目录 | 用途 |
|------|------|
| `scripts/` | Baseline 实验相关脚本（00-13 编号） |
| `sd_scripts/` | SD 合成数据增强实验脚本 |

---

## 配置文件

- `severity_config.json` - Severity Score 配置（项目根目录）
- `dataset/generation_config.json` - 合成数据生成配置
- `experiment_groups.json` - 实验组定义