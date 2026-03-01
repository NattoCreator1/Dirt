# Phase 3 MVP: 时序稳定性学习 - 最小可发表闭环

**创建日期**: 2026-02-26
**基于**: PHASE3_DETAILED_PLAN.md 的减法优化
**核心定位**: 用最小工作量验证"SD 用于时序稳定性学习"的可行性

---

## 一、核心主张（论文结论）

**一句话总结**：

> SD 不适合直接监督学习跨域严重度标尺，但可作为构造"前景静止、背景变化"序列的工具，通过时序一致性训练提升预测稳定性，同时不破坏静态评测。

---

## 二、必须保留的最小闭环

这四块决定论文主张是否站得住：

| 组件 | 作用 | 状态 | 证据 |
|------|------|------|------|
| **Cluster-level 外部评测** | 避免重复样本质疑 | ✓ 已完成 | CI 放大 10-15x |
| **SD 静态混训失效证据** | 支撑战略转型 | ✓ 已完成 | **A+-only: Δρ = -0.097** |
| **时序数据构造定义** | 物理先验支撑 | 待实现 | 固定脏污 + 变化背景 |
| **时序稳定性指标** | 证明时序模块价值 | 待实现 | J_G 降低 > 20% |

**为什么选择 A+-only 而非 D-only？**

| 对比 | A+-only | D-only | 推荐 |
|------|---------|--------|------|
| ρ (簇层面) | 0.6478 | 0.570 | A+-only |
| Δρ vs Baseline | **-0.097** | **-0.175** | A+-only |
| 下降幅度 | 13% | 23% | A+-only 更适中 |
| 审稿人感知 | "门控仍有下降" | "SD 完全不可靠" | A+-only |
| 叙述角度 | 即使筛选仍有害 | 混入即崩塌 | A+-only |

**核心论点**：
> 即使通过严格门控（pass rate = 5.77%），SD 样本参与训练仍导致跨域排序能力显著下降（Δρ = -0.097, 95% CI [-0.137, -0.054]，不覆盖 0）。这证明 SD 的失效不只是"质量问题"，而是更深层的方法论问题。

---

## 三、实验设计（MVP 版本）

### 3.1 模型架构（极简版）

```
Baseline (已训练)
    ↓
冻结 backbone + global head
    ↓
新增: Temporal Module (1D Conv, 轻量)
    ↓
输出: 时序感知的预测
```

**关键设计决策**：
- ✅ **冻结** backbone + global head（防止 SD 污染标尺）
- ✅ 只训练 temporal module（最小参数更新）
- ❌ 删 Stage 3 端到端微调（高成本、高风险）

### 3.2 训练流程（单一版本）

| 参数 | 值 | 说明 |
|------|-----|------|
| **初始化** | Baseline 权重 | 从 real-only 预训练开始 |
| **冻结** | backbone + global head | 防止 SD 影响 |
| **训练** | 仅 temporal module | ~40K 参数 |
| **数据** | SD_seq_HQ（300 条序列） | 只用 HQ，不做 LQ ablation |
| **损失** | L_anchor + λ·L_tempG | 见下方 |
| **优化器** | AdamW, lr=3e-4 | 标准配置 |

**损失函数**（简化版）：

```python
L_total = L_anchor + λ · L_tempG

# Anchor 损失：防止输出偏离 baseline
L_anchor = MSE(Ŝ_temporal, Ŝ_baseline)

# Tile 时序一致性（核心）
L_tempG = (1 / (T-1)) · Σ ||Ĝ_{t+1} - Ĝ_t||₁
```

**删除**：
- ❌ L_tempS（全局稳定性，避免影响 global 标尺）
- ❌ 复杂的多权重组合

### 3.3 对照组（最小集）

| 对照 | 描述 | 目的 |
|------|------|------|
| **Baseline** | 无 temporal module | 参考基准 |
| **Temporal** | + temporal module | 主要结果 |

**可选**（如果资源允许）：
- Temporal w/o anchor：证明 anchor 的必要性

### 3.4 数据构造（精简版）

**素材**：
- 脏污层：HQ 300（已有人工筛选）
- 背景：真实干净视频帧（需收集）

**序列参数**：
- 长度 T = 16（足够体现背景变化）
- 数量：300 条（1:1 对应 HQ 300）
- 生成：固定脏污层 + 滑动窗口采样背景

**QC 标准**（最小集）：
- Spill（mask 外污染率）: < 5%
- 人工审查：HQ 300 已完成

**删除**：
- ❌ HQ vs LQ ablation（放到 Future Work）
- ❌ 复杂的对比度/透明度统计

---

## 四、评测体系（简化版）

### 4.1 静态性能安全带

| 测试集 | 指标 | 基线 | MVP 目标 |
|--------|------|------:|----------:|
| WoodScape Test_Real | Global MAE | 0.0187 | < 1.2x |
| External Test_Ext | ρ (cluster-level) | 0.7446 | 不显著下降 |

**统计报告**：
- Cluster-level ρ：给 95% CI
- 其他指标：点估计即可

### 4.2 时序稳定性主指标

**核心指标**（只报告这一个）：

```
J_G = (1 / (T-1)) · Σ ||Ĝ_{t+1} - Ĝ_t||₁
```

**物理含义**：Tile 级预测的帧间抖动

**工程意义**：
- J_G 越小 → 脏污区域预测越稳定
- 直接对应"误报抖动"的用户体验问题

**报告格式**：

| 模型 | J_G (mean ± std) | vs Baseline |
|------|-----------------:|------------:|
| Baseline | ? | - |
| Temporal | ? | ↓ 降低 % |

**删除**：
- ❌ J_S（全局稳定性，避免混淆）
- ❌ F_Alarm（报警抖动率，概念重叠）
- ❌ 多余的 Bootstrap CI（只对 cluster ρ 做）

### 4.3 可视化（最小集）

1. **时序对比图**：Baseline vs Temporal 的 Ŝ_t 序列
2. **抖动分布图**：J_G 的直方图对比

---

## 五、实施计划（2周版）

### Week 1: 数据 + 模型

| 任务 | 工作量 | 输出 |
|------|-------|------|
| 背景视频收集 | 1 天 | `background_videos/` |
| 序列生成脚本 | 0.5 天 | `generate_sd_sequences.py` |
| 生成 300 条序列 | 0.5 天 | `sd_seq_hq_300/` |
| Temporal module 实现 | 0.5 天 | `models/temporal_conv.py` |
| 训练脚本 | 0.5 天 | `train_phase3_mvp.py` |

### Week 2: 训练 + 评测

| 任务 | 工作量 | 输出 |
|------|-------|------|
| Temporal 模块训练 | 1 天 | `ckpt_temporal.pth` |
| 静态评测 | 0.5 天 | MAE, ρ 结果 |
| 时序稳定性评测 | 0.5 天 | J_G 结果 |
| 可视化 + 分析 | 1 天 | 图表 + 报告 |
| 文档整理 | 1 天 | 论文草稿 |

---

## 六、论文结构（MVP 版）

```
Section 1: Introduction
  - 车载镜头脏污检测的重要性
  - 现有方法的局限性（单帧、静态）
  - 我们的贡献（见下方）

Section 2: Related Work
  - 脏污检测
  - 时序建模
  - 合成数据增强

Section 3: Method
  3.1 Severity Score + Aggregator
  3.2 Why SD Augmentation Failed
      - Cluster-level Bootstrap 发现
      - 失效机制分析（一个代表性案例）
  3.3 Temporal Stability Learning
      - 物理先验：脏污固定、背景变化
      - 数据构造：固定脏污层 + 变化背景
      - 时序模块 + 冻结策略

Section 4: Experiments
  4.1 实验设置
  4.2 SD 静态增强失效（A+-only 案例）
      - 门控通过率 5.77%
      - Δρ = -0.097（显著下降）
      - 论点：即使筛选仍有害
  4.3 时序稳定性主结果
      - J_G 降低
      - 静态性能保持
  4.4 讨论

Section 5: Conclusion
  - SD 不适合直接监督，但可用于时序稳定性
  - 未来工作（HQ vs LQ, Stage 3, 等等）
```

---

## 七、删减优先级总结

| 优先级 | 删减项 | 原因 | 保留位置 |
|-------|-------|------|----------|
| **1** | Stage 3 端到端微调 | 高成本、高风险、易污染标尺 | Future Work |
| **2** | HQ vs LQ ablation | 工作量大、对核心结论贡献小 | Future Work |
| **3** | 复杂 QC 指标 | 实施耗时、边际收益低 | 只保留 spill + 人工 |
| **4** | L_tempS | 避免影响 global 标尺 | 只用 L_tempG |
| **5** | 过多统计 CI | 分散注意力 | 只对 cluster ρ 做 |
| **6** | 多对照组 | 单一对比足够 | 可选：w/o anchor |

---

## 八、成功标准（简化版）

| 优先级 | 指标 | 目标 |
|-------|------|------|
| **主要** | J_G 降低 | > 20% vs Baseline |
| **主要** | Test_Ext ρ | 不显著下降（CI 重叠） |
| 次要 | Test_Real MAE | < 1.2x Baseline |

**保底选项**（即使主要目标未达成）：

1. ✓ **Cluster-level Bootstrap** 方法论贡献
2. ✓ **SD 静态增强失效**的经验贡献
3. ✓ **时序稳定性作为新评估维度**的问题定义

---

## 九、与完整版文档的对比

| 方面 | 完整版 (PHASE3_DETAILED_PLAN.md) | MVP 版 (本文档) |
|------|--------------------------------|-----------------|
| 训练阶段 | 3 个（Stage 1/2/3） | 1 个（冻结策略） |
| 数据集 | HQ + LQ ablation | 仅 HQ 300 |
| QC 指标 | 4+ 项 | 2 项（spill + 人工） |
| 时序损失 | L_tempS + L_tempG | 仅 L_tempG |
| 对照组 | 3-4 个 | 2 个（Baseline + Temporal） |
| 统计报告 | 全部 CI | 简化（关键指标） |
| 实施周期 | 3 周 | 2 周 |

---

## 十、关键代码框架

### 10.1 Temporal Module（极简版）

```python
class TemporalConv1D(nn.Module):
    """轻量时序模块"""
    def __init__(self, in_channels=512, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                             padding=kernel_size//2)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # [B, H, W, T, C]
        x = x.reshape(B*H*W, C, T)
        x = self.conv(x)
        x = F.relu(x)
        x = x.reshape(B, H, W, T, C)
        x = x.permute(0, 3, 4, 1, 2)  # [B, T, C, H, W]
        return x.mean(dim=1)  # [B, C, H, W]
```

### 10.2 训练循环（关键部分）

```python
# 冻结 backbone 和 global head
for name, param in model.backbone.named_parameters():
    param.requires_grad = False
for name, param in model.global_head.named_parameters():
    param.requires_grad = False

# 只训练 temporal module
optimizer = AdamW(model.temporal_module.parameters(), lr=3e-4)

for batch in dataloader:
    # batch: [B, T, ...]

    # Baseline 参考（frozen，无梯度）
    with torch.no_grad():
        S_baseline = model.global_head(model.backbone(batch[:, 0]))

    # Temporal 预测
    outputs = model(batch)  # [B, T, ...]
    G_hat = outputs['G_hat']  # [B, T, 4, 8, 8]

    # Anchor 损失
    S_temporal = outputs['S_hat'][:, -1]  # 最后一帧
    loss_anchor = F.mse_loss(S_temporal, S_baseline)

    # Tile 时序一致性
    loss_tempG = 0
    for t in range(batch.shape[1] - 1):
        loss_tempG += F.l1_loss(G_hat[:, t+1], G_hat[:, t])
    loss_tempG = loss_tempG / (batch.shape[1] - 1)

    # 总损失
    loss = loss_anchor + 0.1 * loss_tempG

    loss.backward()
    optimizer.step()
```

### 10.3 评测脚本（稳定性）

```python
def evaluate_stability(model, sequence_dataloader):
    """计算时序稳定性 J_G"""
    j_g_values = []

    model.eval()
    with torch.no_grad():
        for sequence_batch in sequence_dataloader:
            outputs = model(sequence_batch)
            G_hat = outputs['G_hat']  # [B, T, 4, 8, 8]

            for b in range(G_hat.shape[0]):
                for t in range(G_hat.shape[1] - 1):
                    j_g = F.l1_loss(G_hat[b, t+1], G_hat[b, t])
                    j_g_values.append(j_g.item())

    return {
        'J_G_mean': np.mean(j_g_values),
        'J_G_std': np.std(j_g_values),
    }
```

---

**文档版本**: v1.0 (MVP)
**状态**: 待实施
**下一步**: 开始收集背景视频数据
