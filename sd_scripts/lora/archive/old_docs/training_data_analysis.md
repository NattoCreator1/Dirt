# LoRA训练数据分析报告

## 问题发现（2026-02-18）

### 核心问题：Caption与图像内容不匹配

**训练数据统计**：
- 总样本数: 3200
- 多类别样本: 3200 (100%)
- 单类别样本: 0 (0%)

### 典型例子

| 样本ID | C1比例 | C2比例 | C3比例 | Caption | 实际内容 |
|--------|--------|--------|--------|---------|----------|
| 0001_FV | 36.2% | 63.6% | 0% | "transparent soiling layer" | C1+C2混合 |
| 0015_MVR | 0% | 44.4% | 55.6% | "transparent soiling layer" | C2+C3混合 |

### 问题影响

当推理时使用"water droplets"（期望纯C1）：
- LoRA生成的是C1+C2/C3的混合外观
- 结果：water drops图像中出现"轻度泥巴"类型的脏污

---

## 解决方案

### 方案A：纯化训练数据（推荐）

**策略**：只使用单一类别占绝对主导的样本

```python
# 纯度阈值
PURITY_THRESHOLD = 0.85  # 单一类占比>85%

# 筛选
pure_samples = df[
    (df['class_1_ratio'] > 0.85) |  # 纯C1
    (df['class_2_ratio'] > 0.85) |  # 纯C2
    (df['class_3_ratio'] > 0.85)    # 纯C3
]
```

**优点**：
- Caption与图像内容一致
- LoRA学习到纯粹的类别特征
- 推理时的物质类型更准确

**缺点**：
- 训练样本数减少（可能<1000）
- 需要重新训练LoRA

### 方案B：修改Caption策略

**策略**：在Caption中说明是混合类型

```python
# 检测多类别
if num_classes > 1:
    class_token = f"mixed soiling with {class_names}"
else:
    class_token = CLASS_TOKENS[dominant_class]
```

**优点**：
- 保留所有训练数据
- Caption更准确

**缺点**：
- 推理时也需要使用"mixed"prompt
- 失去了单一类别的控制能力

### 方案C：接受现状

**理解**：这是WoodScape数据的特点
- 真实世界的镜头脏污本身就是多类别混合的
- "水滴"和"泥巴"在现实中往往同时存在
- 当前的区分程度已经"够用"

**策略**：
- 不重新训练LoRA
- 在推理时使用物质锚定prompt
- 接受生成的"混合外观"

---

## 推荐行动

### 短期（当前实验阶段）
1. **接受方案C**：继续使用当前LoRA
2. **专注于可控制的维度**：覆盖率、位置、严重度
3. **开始批量生成**：验证下游任务效果

### 长期（如果需要改进）
1. **实施方案A**：纯化训练数据
2. **重新训练LoRA**：使用纯类别样本
3. **对比效果**：验证是否提升

---

## 数据统计

### 当前训练数据类别分布

| Dominant Class | 样本数 | 比例 |
|----------------|--------|------|
| C1 (transparent) | 509 | 15.9% |
| C2 (semi) | 713 | 22.3% |
| C3 (opaque) | 1978 | 61.8% |

### Caption分布

| Caption | 样本数 |
|---------|--------|
| noticeable opaque heavy stains | 995 |
| severe opaque heavy stains | 983 |
| noticeable semi-transparent dirt smudge | 367 |
| severe semi-transparent dirt smudge | 346 |
| noticeable transparent soiling layer | 272 |
| severe transparent soiling layer | 237 |
