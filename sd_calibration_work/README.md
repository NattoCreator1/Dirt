# SD数据校准工作目录

**创建日期**: 2026-02-25
**目的**: 实施SD数据校准方案

---

## 目录结构

```
sd_calibration_work/
├── README.md                          # 本文件
├── phase0_diagnostic/                  # 阶段0：诊断分析
│   ├── analyze_distribution.py        # 分布分析脚本
│   └── diagnostic_report.md           # 诊断报告
├── phase1_importance_sampling/        # 阶段1：分桶Importance Sampling
│   ├── two_layer_sampler.py
│   ├── compute_weights.py
│   └── train_d_only.py
├── phase2_consistency_gate/           # 阶段2：一致性门控
│   ├── gate_weights.py
│   └── train_a_only.py
├── results/                            # 所有实验结果
│   ├── baseline/
│   ├── mixed_8960_no_cal/
│   ├── mixed_8960_d_only/
│   ├── mixed_8960_a_only/
│   ├── mixed_8960_d_a/
│   ├── mixed_989_no_cal/
│   └── mixed_1260_no_cal/
└── logs/                               # 训练日志
```

---

## 数据集配置

| 数据集 | 路径 | 数量 | S均值 |
|--------|------|------|-------|
| WoodScape Train | `dataset/woodscape_processed/meta/labels_index_ablation.csv` | 4000 | 0.408 |
| SD全量 8960 | `sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv` | 8960 | ~0.29 (需确认) |
| SD筛选 989 | 筛选子集 | 989 | 0.267 |
| SD筛选 1260 | 筛选子集 | 1260 | 0.287 |

---

## 实验组

| 实验组 | SD数据 | 校准 | 状态 |
|--------|--------|------|------|
| Baseline | - | - | 对照 |
| Mixed(8960, no cal) | 全量8960 | 无 | 主实验起点 |
| Mixed(8960, D-only) | 全量8960 | Importance Sampling | 主实验 |
| Mixed(8960, A+-only) | 全量8960 | 一致性门控 | 备选 |
| Mixed(989, no cal) | 筛选989 | 无 | 对照 |
| Mixed(1260, no cal) | 筛选1260 | 无 | 对照 |

---

## 当前进度

- [x] 工作目录创建
- [x] 阶段0：诊断分析 ✓ (2026-02-25完成)
- [x] 阶段0补充：SD 8960分析 ⚠️ (关键发现！)
- [x] 阶段1：D-only实现 ✓ (代码完成)
- [ ] 阶段1：D-only训练 (待执行)
- [ ] 阶段2：A+-only实现
- [ ] 对照实验
- [ ] 结果汇总

### 阶段0完成总结

**SD 1260诊断**:
- SD 1260 S均值: 0.2866 (显著低于WoodScape 0.4081)
- 差异: -0.1215 (< -0.05阈值)

**⚠️ SD 8960关键发现**:
- SD 8960 S均值: 0.4658 (**高于**WoodScape 0.4081!)
- 差异: +0.0576
- **结论**: SD 1260是偏低的有偏子集，SD 8960全量数据实际上偏高

**校准策略调整**:
- SD 8960需要**反向校准**（降低权重而非提高）
- 所有权重 < 1 (mean=0.36)
- 因为SD 8960在各区间都超过WoodScape样本数

**输出文件**:
- `phase0_diagnostic/diagnostic_report.json` - SD 1260统计
- `phase0_diagnostic/diagnostic_report_8960.json` - SD 8960统计
- `phase0_diagnostic/diagnostic_report.md` - 详细报告
- `phase0_diagnostic/distribution_comparison.png` - 可视化对比

### 阶段1实现完成

**已实现**:
- ✅ Two-layer sampler (`two_layer_sampler.py`)
- ✅ Weight computation (`compute_weights.py`)
- ✅ Mixed dataset prep (`prepare_mixed_dataset.py`)
- ✅ Training script (`train_mixed_d_only.py`)

**输出文件**:
- `phase1_importance_sampling/weights_8960_correct.npz` - 正确权重
- `phase1_importance_sampling/mixed_ws4000_sd8960.csv` - 混合数据集

**训练命令**:
```bash
python sd_calibration_work/phase1_importance_sampling/train_mixed_d_only.py \
    --mixed_csv sd_calibration_work/phase1_importance_sampling/mixed_ws4000_sd8960.csv \
    --weights sd_calibration_work/phase1_importance_sampling/weights_8960_correct.npz \
    --run_name mixed_ws4000_sd8960_d_only \
    --total_steps 10000
```

---

## 快速导航

```bash
# 阶段0诊断
cd phase0_diagnostic && python analyze_distribution.py

# 阶段1训练
cd phase1_importance_sampling && python train_d_only.py

# 查看结果
ls results/
```
