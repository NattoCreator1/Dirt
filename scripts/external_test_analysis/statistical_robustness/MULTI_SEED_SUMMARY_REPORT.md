# Multi-Seed Baseline Evaluation Report

**日期**: 2026-02-27
**实验类型**: Multi-seed training for statistical robustness
**目标**: 验证 baseline 性能对随机初始化的稳定性

---

## 一、实验设置

| 参数 | 值 |
|------|-----|
| 随机种子 | 42, 123, 456 |
| 训练集 | WoodScape Train_Real (3,200) |
| 验证集 | WoodScape Val_Real (800) |
| 测试集 | External Test (50,383 images, 505 clusters) |
| Checkpoint | **ckpt_last.pth** (固定 40 epochs) |
| Bootstrap | 簇层面, 1000 次重采样 |

---

## 二、核心结果

### 2.1 Spearman ρ (簇层面 Bootstrap)

| Seed | ρ 点估计 | 95% CI | CI 宽度 | 压缩比 |
|------|---------|--------|---------|--------|
| 42 | 0.7145 | [0.6378, 0.7405] | 0.1027 | 0.081 |
| 123 | 0.7421 | [0.6935, 0.7732] | 0.0796 | 0.081 |
| 456 | 0.7249 | [0.6674, 0.7577] | 0.0903 | 0.067 |

**统计摘要**:
- **ρ (mean ± std)**: 0.7272 ± 0.0138
- **ρ 范围**: [0.7145, 0.7421]
- **CI 重叠度**: 100% (所有 seeds 的 CI 有重叠)
- **Max-Min 差异**: 0.0276

### 2.2 95% CI 可视化

```
Seed 42:  [0.6378 |======|0.7145|======| 0.7405]
Seed 123: [0.6935 |=====|0.7421|======| 0.7732]
Seed 456: [0.6674 |=====|0.7249|======| 0.7577]
          ↑-------+-------+-------↑
          共同重叠区域
```

---

## 三、稳健性评估

### 3.1 成功标准检验

| 标准 | 目标 | 实际结果 | 状态 |
|------|------|----------|------|
| ρ 稳定性 | std < 0.03 | 0.0138 | ✅ **通过** |
| CI 重叠度 | 100% | 100% | ✅ **通过** |
| 无离群值 | Max-Min < 0.10 | 0.0276 | ✅ **通过** |
| 压缩比一致性 | std < 0.10 | 0.008 | ✅ **通过** |

### 3.2 关键发现

1. **低方差**: ρ_std = 0.0138，远小于目标 0.03
   - 说明 baseline 性能对随机初始化不敏感

2. **CI 完全重叠**: 所有三个 seeds 的 95% CI 都有显著重叠
   - 统计上无法区分三个模型的性能
   - 证明了结果的稳定性

3. **压缩比一致**: 三个 seeds 的压缩比都在 0.067-0.081 范围内
   - 说明标尺压缩现象是稳定的，不是随机波动

---

## 四、与原 Baseline 对比

| 模型 | Checkpoint | ρ (簇级) | CI | 说明 |
|------|-----------|----------|-----|------|
| **原 Baseline** | ckpt_best | 0.7446 | [0.6739, 0.7662] | Seed 42, 按 val MAE 选 |
| **Multi-seed Mean** | ckpt_last | 0.7272 | - | 3 seeds 均值 |

**说明**:
- Multi-seed 使用 `ckpt_last.pth` (固定 40 epochs)
- 原 Baseline 使用 `ckpt_best.pth` (按 val MAE 选择)
- Multi-seed 的 ρ 略低是因为使用 last checkpoint 而非 best
- 但 Multi-seed 展示了**跨种子的稳定性**，这是核心贡献

---

## 五、结论

Multi-seed 实验证明:

1. **Baseline 性能稳定**: 跨 3 个不同随机种子，Spearman ρ 的标准差仅为 0.0138

2. **统计上无法区分**: 三个模型的 95% CI 完全重叠，说明性能差异不显著

3. **标尺压缩稳定**: 压缩比在 seeds 间保持一致 (std = 0.008)

4. **方法论价值**: 使用 **ckpt_last.pth** (固定 epochs) 比 **ckpt_best.pth** 更严谨
   - 避免"挑最好"的偏差
   - 更真实反映训练稳定性
   - 审稿人更认可

---

## 六、文件位置

| 文件 | 路径 |
|------|------|
| 预测 CSV | `scripts/external_test_analysis/multi_seed/seed{42,123,456}_test_ext.csv` |
| Bootstrap 结果 | `scripts/external_test_analysis/statistical_robustness/seed*_bootstrap.json` |
| 综合结果 | `scripts/external_test_analysis/statistical_robustness/evaluation_results_*.json` |

---

**报告生成时间**: 2026-02-27
**状态**: ✅ Multi-seed 实验完成
**下一步**: K-fold 交叉验证实验 (可选)
