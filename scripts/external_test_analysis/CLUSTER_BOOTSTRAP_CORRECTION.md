# External Test Bootstrap 分析修正报告

## 问题发现

原始 `external_test_washed_processed` 版本的文件名丢失了序列信息。经过检查原始 `external_test_washed` 目录，发现：

### 数据结构
```
external_test_washed/
├── occlusion_a/
│   ├── LJVA4PALXNT130670_20240220090334718_split_3/  (簇1: 100张)
│   ├── 10000003_01_20250924104911033_3_a/            (簇2: 291张)
│   └── ... (共15个子文件夹)
├── occlusion_b/
│   └── ... (共115个子文件夹)
...
```

**关键发现**：
- 总共 **505 个子文件夹（簇）**
- 每个簇平均 **99.8 张图片**
- **同一簇内的图片来自同一视频序列，高度相关**

### 统计影响

| Bootstrap 方法 | 采样单元数 | CI 宽度 | 相对宽度 |
|---------------|----------:|--------:|---------:|
| **图片层面（错误）** | 50,383 张 | 0.0094 | 1.31% |
| **簇层面（正确）** | 505 簇 | **0.0923** | **12.40%** |

**结论**：图片层面的 CI 被**低估了约 10 倍**！

## 修正后的统计结论

### Baseline 模型（簇层面 Bootstrap）

| 指标 | 值 |
|------|-----|
| 点估计 | 0.7446 |
| 95% CI | [0.6739, 0.7662] |
| 标准误差 | 0.0233 |
| CI 宽度 | 0.0923 (12.40%) |

### 与图片层面的对比

之前的图片层面分析报告的 CI 宽度：
- Baseline: 0.0094 (1.31%)
- SD989: 0.0093 (1.30%)

这些 CI **过于狭窄**，没有正确反映统计不确定性。

## 对之前结论的影响

### 需要修正的结论

1. **"SD989 的下降是统计显著的"** → 需要重新评估
   - 之前：Δρ = -0.008, 95% CI [-0.0119, -0.0076]
   - CI 完全位于 0 之外
   - 但这个 CI 是基于错误的图片层面 Bootstrap

2. **"Phase 2 模型的下降是统计显著的"** → 需要重新评估
   - 需要用簇层面 Bootstrap 重新计算

### 正确的统计推断框架

1. **有效样本数是 505（簇数），而非 50,383（图片数）**
2. **Bootstrap 应该在簇层面进行**
3. **统计功效远低于之前估计**

## 下一步行动

1. ✅ 已修正：创建簇层面的预测 CSV
2. ⏳ 待完成：为其他模型（SD989, A+-only, Tile-only）生成簇层面预测
3. ⏳ 待完成：重新计算所有模型的簇层面 CI
4. ⏳ 待完成：更新论文中的统计结论

## 技术细节

### 簇 ID 定义
```python
cluster_id = "occlusion_a/LJVA4PALXNT130670_20240220090334718_split_3/"
```

使用子文件夹的相对路径作为簇 ID，确保：
- 同一视频序列的所有图片属于同一簇
- 不同脏污等级的相同相机是不同的簇（因为子文件夹路径不同）

### Bootstrap 流程
```python
for i in range(n_bootstrap):
    # 重采样簇（而非图片）
    sampled_clusters = random.choice(clusters, n_clusters, replace=True)
    # 每个簇内的所有图片都包含在内
    sampled_images = concat([cluster[seq_id] for seq_id in sampled_clusters])
    # 计算 Spearman ρ
    rho_b = spearman(predictions[sampled_images], labels[sampled_images])
```

## 文件清单

- `scripts/generate_test_ext_with_clusters.py` - 生成带簇信息的预测
- `scripts/cluster_bootstrap_corrected.py` - 簇层面 Bootstrap 分析
- `baseline/runs/.../eval_Test_Ext_predictions_clustered.csv` - 带簇信息的预测数据

---
**报告日期**: 2025-02-26
**修正原因**: External Test 数据集的簇内相关性问题
