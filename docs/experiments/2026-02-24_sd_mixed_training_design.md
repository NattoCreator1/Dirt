# SD合成数据混合训练实验设计

**实验日期**: 2026-02-24
**实验类型**: SD合成数据增强验证
**目的**: 验证人工筛选的SD合成数据能否提升模型跨域泛化能力

---

## 一、实验设计原则

### 1.1 控制变量

| 变量 | 对照组 | 实验组 | 说明 |
|------|--------|--------|------|
| **模型架构** | BaselineDualHead | BaselineDualHead | 相同 |
| **Severity定义** | S_full_wgap_alpha50 | S_full_wgap_alpha50 | 相同 |
| **类别权重** | (0, 0.15, 0.50, 1.0) | (0, 0.15, 0.50, 1.0) | 相同 |
| **融合系数** | (α=0.5, β=0.4, γ=0.1) | (α=0.5, β=0.4, γ=0.1) | 相同 |
| **训练参数** | 40 epochs, lr=3e-4 | 40 epochs, lr=3e-4 | 相同 |
| **数据来源** | WoodScape Only | WoodScape + SD Synthetic | **不同** |

### 1.2 数据配置

| 数据集 | 训练集 | 验证集 | 测试集 |
|--------|--------|--------|--------|
| **WoodScape** | 3200 | 800 | 1000 (Test_Real) |
| **SD Synthetic** | 578 | 102 | - |
| **合计 (实验组)** | 3778 | 902 | 1000 |

**SD数据比例**:
- 训练集: 578/3778 ≈ 15.3%
- 验证集: 102/902 ≈ 11.3%

---

## 二、实验配置

### 2.1 对照组 (Baseline)

```bash
python baseline/train_baseline.py \
  --index_csv dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv \
  --img_root . \
  --split_col split \
  --train_split train \
  --val_split val \
  --global_target S \
  --epochs 40 \
  --batch_size 16 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --lambda_glob 1.0 \
  --mu_cons 0.1 \
  --run_name baseline_ws_only_wgap_alpha50 \
  --out_root baseline/runs/mixed_experiment
```

**注意**: 使用 `--global_target S` (对应NPZ中的S_full_wgap_alpha50)

### 2.2 实验组 (WoodScape + SD Synthetic)

**步骤1**: 合并索引

```bash
python sd_scripts/lora/merge_for_mixed_training.py \
  --woodscape_index dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv \
  --synthetic_index sd_scripts/lora/accepted_20260224_164715/index_synthetic_20260224_182316.csv \
  --output_dir dataset/woodscape_processed/meta \
  --seed 42
```

**步骤2**: 混合训练

```bash
python baseline/train_baseline.py \
  --index_csv dataset/woodscape_processed/meta/labels_index_merged_YYYYMMDD_HHMMSS.csv \
  --img_root . \
  --split_col split \
  --train_split train \
  --val_split val \
  --global_target S \
  --epochs 40 \
  --batch_size 16 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --lambda_glob 1.0 \
  --mu_cons 0.1 \
  --run_name mixed_ws_sd_wgap_alpha50 \
  --out_root baseline/runs/mixed_experiment
```

---

## 三、评估策略

### 3.1 评估数据集

| 数据集 | 类型 | 目的 |
|--------|------|------|
| **WoodScape Val** | 同域验证 | 验证基础性能保持 |
| **WoodScape Test** | 同域测试 | 最终性能对比 |
| **External Test** | 跨域测试 | 验证跨域泛化提升 |
| **SD Synthetic Val** | 合成数据验证 | 验证合成数据拟合 |

### 3.2 评估指标

| 指标 | 说明 | 重要性 |
|------|------|--------|
| **MAE (S_hat)** | 全局预测误差 | 主要 |
| **RMSE (S_hat)** | 均方根误差 | 次要 |
| **Spearman ρ (S_hat)** | 排序一致性 | 主要 |
| **Tile MAE** | Tile级预测误差 | 次要 |
| **Δρ = ρ(S_hat) - ρ(S_agg)** | 双头学习有效性 | 分析 |

### 3.3 成功标准

| 标准 | 阈值 | 说明 |
|------|------|------|
| **同域性能保持** | ΔMAE ≤ 0.02 | WoodScape Test性能不显著下降 |
| **跨域性能提升** | Δρ ≥ 0.01 | External Test排序一致性提升 |
| **合成数据拟合** | MAE合理 | SD Synthetic Val误差不过大 |

---

## 四、预期结果与分析

### 4.1 预期假设

**假设1**: SD合成数据通过人工筛选，剔除了低质量样本，能够提供额外的跨域信息。

**假设2**: SD合成数据使用WoodScape训练集的mask但应用到新的干净帧（camera f），这种跨域组合有助于模型学习脏污模式的不变性。

**假设3**: 15%的SD合成数据比例不会显著影响同域性能，但能够在跨域测试上带来明显提升。

### 4.2 可能的结果

| 结果 | 解释 | 后续行动 |
|------|------|----------|
| **跨域显著提升，同域保持** | SD数据有效，验证假设 | 继续分析，准备论文 |
| **跨域提升，同域略降** | 存在trade-off，权衡取舍 | 调整SD数据比例 |
| **无显著差异** | SD数据无效或需更多数据 | 分析失败原因 |
| **性能下降** | 负迁移，SD数据质量/分布问题 | 重新筛选SD数据 |

### 4.3 错误分析维度

如果实验效果不理想，需要分析：

1. **SD数据质量**: 可视化查看SD生成的脏污效果是否真实
2. **数据分布差异**: SD数据的S值分布与WoodScape的差异
3. **领域gap**: SD合成图像的风格与真实图像的差异
4. **训练策略**: 是否需要调整混合比例或训练策略

---

## 五、实验记录模板

### 5.1 对照组结果

```
Model: BaselineDualHead + S_full_wgap_alpha50
Data: WoodScape Only (3200 train + 800 val)

WoodScape Test:
  MAE: _____, RMSE: _____, ρ: _____

External Test:
  MAE: _____, RMSE: _____, ρ: ___
```

### 5.2 实验组结果

```
Model: BaselineDualHead + S_full_wgap_alpha50
Data: WoodScape (3200) + SD Synthetic (578) train + 800/102 val

WoodScape Test:
  MAE: _____, RMSE: _____, ρ: _____
  ΔMAE vs 对照: _____

External Test:
  MAE: _____, RMSE: _____, ρ: _____
  ΔMAE vs 对照: _____, Δρ vs 对照: ___
```

---

## 六、后续工作

### 6.1 短期

1. **运行对照组实验** (如果需要重新训练)
2. **运行实验组实验**
3. **对比分析结果**
4. **可视化分析**: 展示SD数据和真实数据的对比

### 6.2 中期

1. **不同混合比例实验**: 测试5%, 10%, 20%, 30%的SD数据比例
2. **错误分析**: 分析预测错误的样本特征
3. **可视化验证**: 人工检查SD数据质量和模型预测结果

### 6.3 长期

1. **时序模块**: 结合SD合成数据进行时序建模
2. **更多SD数据**: 扩大筛选规模，获取更多高质量合成数据

---

**文档版本**: v1.0
**创建日期**: 2026-02-24
