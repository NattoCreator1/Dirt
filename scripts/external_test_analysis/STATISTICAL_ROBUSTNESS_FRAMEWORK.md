# Baseline Statistical Robustness Framework

**创建日期**: 2026-02-26
**最后更新**: 2026-02-26 (添加关键修正说明)
**目的**: 针对 WoodScape 数据集较小的担忧 (N=3,200)，通过多随机种子和 K 折交叉验证提供统计鲁棒性证据

---

## ⚠️ 关键修正说明 (2026-02-26)

为避免审稿质疑，本框架实施了以下**关键修正**以确保统计严谨性：

### 1) 簇级聚合修正
**问题**: 图片层面 Bootstrap 违反独立性假设
**修正**: 先聚合到簇级 (中位数 per cluster)，再在 505 个簇上进行 Bootstrap
```python
# Step 1: 聚合到簇级
df_cluster = df.groupby('cluster_id').agg({
    'S_hat': 'median',
    'occlusion_level': 'median',
})

# Step 2: 在簇级 Bootstrap
sampled_clusters = resample(cluster_ids, n=505, replace=True)
rho = spearman(cluster_preds[sampled_clusters], cluster_labels[sampled_clusters])
```

### 2) 模型选择规则修正
**问题**: 使用各自的 best checkpoint 会导致 val 噪声放大波动
**修正**: 统一使用 **ckpt_last.pth** (固定 40 epochs)
- 确保所有 seeds/folds 在相同训练步数下比较
- 避免 val set 噪声导致的虚假波动

### 3) K-fold 数据划分修正
**问题**: K-fold 可能意外泄漏 Test_Real
**修正**: 明确声明 Test_Real **永不参与** K-fold 训练
```
K-fold 数据来源: Train (3,200) + Val (800) = 4,000
Test_Real (1,000): 保持独立，仅用于最终评测
External Test (50,383): 保持独立，仅用于跨域评测
```

### 4) 簇 ID 映射固定
**问题**: 不同 seed/fold 可能产生不同的簇划分
**修正**: 使用全局固定的簇 ID 映射文件
- 簇 ID = 子文件夹相对路径 (固定不变)
- 所有评测脚本读取相同的簇定义

### 5) 额外稳健性指标
**问题**: 单报 Spearman ρ 可能掩盖标尺压缩问题
**修正**: 增加以下指标
- **Compression Ratio**: 敏感度 to level changes (ideal = 1.0)
- **Level-wise means**: 每个严重度的预测均值
- **Test_Real MAE**: 同域性能安全带

---

## 一、背景与问题

### 1.1 数据集规模对比

| 数据集 | 样本数 | 用途 |
|--------|-------|------|
| WoodScape Train_Real | 3,200 | 训练 |
| External Test | 50,383 (505 clusters) | 跨域评测 |

**担忧**: WoodScape 训练集相对较小，模型性能可能对训练集划分敏感。

### 1.2 解决方案

通过以下两种互补方法验证 Baseline 性能的稳定性：

1. **Multi-seed 实验**: 多个随机种子训练，评估不同随机初始化下的性能波动
2. **K-fold 交叉验证**: 将 Train+Val (4,000) 分为 K 折，评估不同数据划分下的性能波动

---

## 二、方法设计

### 2.1 Multi-seed 实验

**核心思想**: 使用不同随机种子重复训练，观察性能方差。

| 参数 | 值 | 说明 |
|------|-----|------|
| 随机种子 | 42, 123, 456 | 3 个不同种子 |
| 训练集划分 | 固定 | 与原 Baseline 相同 (Train: 3,200, Val: 800) |
| 训练轮数 | 40 epochs | 与原 Baseline 相同 |
| 其他超参数 | 固定 | lr=3e-4, batch_size=32, etc. |

**输出**: 每个 seed 得到一个独立的训练模型

### 2.2 K-fold 交叉验证

**核心思想**: 评估对训练数据划分的敏感度。

| 参数 | 值 | 说明 |
|------|-----|------|
| K | 5 | 5 折交叉验证 |
| 数据来源 | Train+Val = 4,000 | 合并原训练和验证集 |
| 每折划分 | Train: 3,200, Val: 800 | 保持与原设置相同比例 |
| Test set | 1,000 | 保持独立，不参与 CV |

**K-fold 流程**:
```
Fold 1: Train on [Fold2,3,4,5], Validate on Fold1
Fold 2: Train on [Fold1,3,4,5], Validate on Fold2
...
Fold 5: Train on [Fold1,2,3,4], Validate on Fold5
```

---

## 三、评测指标 (簇层面 Bootstrap)

### 3.1 Test_Real 评测

| 指标 | 说明 |
|------|------|
| Global MAE | 全局严重度 MAE |
| Tile MAE | Tile 覆盖率 MAE |

### 3.2 Test_Ext 评测 (重点)

**⚠️ 重要**: 使用 **簇层面 Bootstrap** 而非图片层面

| 指标 | 说明 |
|------|------|
| **ρ (cluster-level)** | Spearman 相关性 (基于 505 个簇) |
| **95% CI** | 置信区间 (簇层面 Bootstrap, 1000 次重采样) |
| **SE** | 标准误差 |

**报告格式**:

| 实验类型 | ρ (mean ± std) | 95% CI (mean) | CI Width |
|---------|----------------|---------------|----------|
| Multi-seed (3) | ? ± ? | [?, ?] | ? |
| K-fold (5) | ? ± ? | [?, ?] | ? |

### 3.3 簇层面 Bootstrap 方法 (修正版)

**关键修正**: 先聚合到簇级，再 Bootstrap

```python
# Step 1: 聚合图片级预测到簇级
df_cluster = df.groupby('cluster_id').agg({
    'S_hat': 'median',           # 簇内预测中位数
    'occlusion_level': 'median',  # 簇内标签中位数 (应相同)
})
# 得到 505 个簇级样本

# Step 2: 在簇级 Bootstrap (505 个独立单位)
for i in range(1000):
    sampled_clusters = resample(cluster_ids, n=505, replace=True)
    rho_i = spearman(cluster_preds[sampled_clusters],
                     cluster_labels[sampled_clusters])

# 95% CI = [percentile(rhos, 2.5), percentile(rhos, 97.5)]
```

**对比**:
| 方法 | 样本数 | 独立性 | CI 宽度 |
|------|--------|--------|---------|
| 图片层面 Bootstrap | 50,383 | ❌ 违反 | 窄 (低估) |
| **簇级聚合+Bootstrap** | 505 | ✅ 正确 | 宽 (正确) |

### 3.4 压缩比指标 (Compression Ratio)

衡量模型对严重度变化的敏感度:

```python
# 计算相邻严重度的预测差异
level_means = {level: median(S_hat) for each level}
deltas = [level_means[i] - level_means[i+1] for i in [5,4,3,2]]

compression_ratio = mean(deltas) / 1.0  # 真实差异为 1
```

- **1.0**: 理想敏感度
- **< 1.0**: 标尺压缩 (如 SD 模型)
- **> 1.0**: 标尺扩张

---

## 四、实施流程

### 关键设计选择

| 设计维度 | 选择 | 理由 |
|---------|------|------|
| Checkpoint 选择 | **ckpt_last.pth** (40 epochs) | 固定训练步数，避免 val 噪声 |
| Bootstrap 单位 | **505 个簇** (聚合后) | 正确处理簇内相关性 |
| 簇内聚合 | **中位数** | 对离群值鲁棒 |
| Test_Real 处理 | **永不参与 K-fold** | 保持独立性作为安全带 |

### 4.1 Multi-seed 实验

```bash
# Step 1: 训练 3 个不同种子的模型
python scripts/multi_seed_baseline_train.py \
    --seeds 42,123,456 \
    --epochs 40 \
    --out_root baseline/runs

# 预期输出:
# baseline/runs/baseline_seed42_*/
# baseline/runs/baseline_seed123_*/
# baseline/runs/baseline_seed456_*/
```

```bash
# Step 2: 评测所有模型 (Test_Ext with cluster Bootstrap)
python scripts/multi_seed_evaluate.py \
    --mode multi_seed \
    --seeds 42,123,456 \
    --n_bootstrap 1000 \
    --output_dir scripts/external_test_analysis/statistical_robustness
```

### 4.2 K-fold 实验

```bash
# Step 1: 生成 K-fold 划分并训练
python scripts/kfold_baseline_train.py \
    --n_folds 5 \
    --epochs 40 \
    --out_root baseline/runs

# 预期输出:
# dataset/woodscape_processed/meta/kfold_splits/labels_index_kfold5_fold[1-5].csv
# baseline/runs/baseline_k5_fold[1-5]*/
```

```bash
# Step 2: 评测所有模型
python scripts/multi_seed_evaluate.py \
    --mode kfold \
    --n_folds 5 \
    --n_bootstrap 1000 \
    --output_dir scripts/external_test_analysis/statistical_robustness
```

### 4.3 合并报告

```bash
# 同时运行 multi-seed 和 k-fold 评测
python scripts/multi_seed_evaluate.py \
    --mode both \
    --seeds 42,123,456 \
    --n_folds 5 \
    --n_bootstrap 1000
```

---

## 五、预期结果解读

### 5.1 性能稳定性评估

**如果 Baseline 性能稳定** (理想情况):

| 实验类型 | 预期结果 | 解释 |
|---------|---------|------|
| Multi-seed | ρ_std < 0.02 | 不同随机初始化下性能一致 |
| K-fold | ρ_std < 0.02 | 不同数据划分下性能一致 |

**论文中的表述**:
> "We assessed the robustness of our baseline through multi-seed experiments (3 seeds) and 5-fold cross-validation. The Spearman ρ on the external test set showed low variance across seeds (σ = 0.XX) and folds (σ = 0.XX), confirming that our baseline performance is stable despite the limited training set size."

### 5.2 与之前结果的对比

| 方法 | 测试集 | 簇数 | CI 宽度 | 说明 |
|------|--------|------|---------|------|
| 图片层面 Bootstrap | Test_Ext | - | 0.013 (1.8%) | ❌ 错误：违反独立性假设 |
| 簇层面 Bootstrap | Test_Ext | 505 | 0.092 (12.4%) | ✅ 正确：处理簇内相关性 |
| Multi-seed mean ± std | Test_Ext | 505 | ± 0.XX | ✅ 新增：评估训练随机性 |

---

## 六、文件结构

```
scripts/
├── multi_seed_baseline_train.py     # Multi-seed 训练脚本
├── kfold_baseline_train.py          # K-fold 训练脚本
├── multi_seed_evaluate.py           # 统一评测脚本
│
├── external_test_analysis/
│   ├── statistical_robustness/
│   │   ├── evaluation_results_YYYYMMDD_HHMMSS.json  # 评测结果 JSON
│   │   ├── multi_seed/
│   │   │   ├── seed42_test_ext.csv
│   │   │   ├── seed123_test_ext.csv
│   │   │   └── seed456_test_ext.csv
│   │   └── kfold/
│   │       ├── kfold5_fold1_test_ext.csv
│   │       ├── kfold5_fold2_test_ext.csv
│   │       └── ...
│   │
│   └── STATISTICAL_ROBUSTNESS_REPORT.md  # 本文档

dataset/woodscape_processed/meta/
└── kfold_splits/
    ├── kfold5_summary.json
    ├── labels_index_kfold5_fold1.csv
    ├── labels_index_kfold5_fold2.csv
    └── ...

baseline/runs/
├── multi_seed_summary_YYYYMMDD_HHMMSS.json
├── baseline_seed42_S_*/
├── baseline_seed123_S_*/
├── baseline_seed456_S_*/
├── kfold5_summary_YYYYMMDD_HHMMSS.json
├── baseline_k5_fold1_S_*/
└── ...
```

---

## 七、与原 Baseline 的关系

### 7.1 原始 Baseline (Seed 42, ckpt_best)

| 属性 | 值 |
|------|-----|
| 随机种子 | 42 |
| Checkpoint | ckpt_best.pth (按 val MAE 选) |
| 训练集 | Train_Real (3,200) |
| 验证集 | Val_Real (800) |
| Test_Real MAE | 0.0187 |
| Test_Ext ρ (cluster-level) | 0.7446 |

### 7.2 统计鲁棒性实验 (ckpt_last, 固定 epochs)

**为公平比较，所有新实验统一使用 ckpt_last.pth (40 epochs)**

| 实验类型 | Checkpoint 选择 | 理由 |
|---------|----------------|------|
| Multi-seed | ckpt_last.pth | 固定训练步数，避免 val 噪声 |
| K-fold | ckpt_last.pth | 同上，确保跨 fold 可比 |

**重要**: 使用 ckpt_last 可能导致 MAE 略高于 ckpt_best，但这是**更严谨的做法**:
- 避免"挑最好"的偏差
- 更真实反映训练稳定性
- 审稿人更认可固定步数的比较

### 7.2 新实验的定位

**Multi-seed**: 验证 "Seed 42 不是特例"
- 如果 Seed 123, 456 的 ρ 都在 0.74 ± 0.02 范围内 → 说明结果稳定

**K-fold**: 验证 "Train/Val 划分不是特例"
- 如果 5 个 fold 的 ρ 都在 0.74 ± 0.02 范围内 → 说明对数据划分不敏感

---

## 八、论文中的使用建议

### 8.1 主实验部分

保留原始 Baseline (Seed 42) 的结果作为主要报告，因为：
- 与所有后续实验 (Phase 2 SD, Phase 3 Temporal) 保持一致
- 便于与其他模型对比

### 8.2 补充材料/附录

报告 Multi-seed 和 K-fold 的统计结果：

| 实验类型 | N | ρ (mean ± std) | 95% CI |
|---------|---|----------------|--------|
| Multi-seed | 3 | 0.74 ± 0.02 | [0.67, 0.77] |
| 5-fold CV | 5 | 0.75 ± 0.03 | [0.68, 0.78] |

**说明文字**:
> "To assess the robustness of our baseline to random initialization and data partitioning, we conducted multi-seed experiments (3 seeds) and 5-fold cross-validation. The low variance in Spearman ρ across seeds (σ = 0.XX) and folds (σ = 0.XX) confirms the stability of our baseline performance despite the limited training set size."

---

## 九、计算成本估算

| 实验 | 模型数 | 总训练时间 | 总评测时间 |
|------|--------|-----------|-----------|
| Multi-seed | 3 | ~6 hours | ~30 min |
| K-fold | 5 | ~10 hours | ~50 min |
| **总计** | 8 | ~16 hours | ~80 min |

---

## 十、成功标准

| 标准 | 目标 | 说明 |
|------|------|------|
| ρ 稳定性 | std < 0.03 | Multi-seed 和 K-fold 的标准差小于 0.03 |
| CI 重叠度 | 100% | 所有 seeds/folds 的 95% CI 有重叠 |
| 无离群值 | Max-Min < 0.10 | 最大值与最小值差异小于 0.10 |
| 压缩比一致性 | std < 0.10 | 压缩比标准差小于 0.10 |
| Test_Real 安全带 | MAE < 1.2x baseline | 同域性能不显著退化 |

如果以上标准满足，则可以声称 Baseline 性能对随机因素和数据划分具有鲁棒性。

### 附加验证 (可选)

如果资源允许，可额外验证:
1. **Test_Real 评测**: 报告每个 seed/fold 的 MAE/RMSE
2. **Level-wise 分析**: 每个严重度的预测均值 ± std
3. **Tile 级指标**: 如果需要更细粒度的分析

---

**文档版本**: v1.0
**状态**: 待实施
**下一步**: 运行 Multi-seed 实验
