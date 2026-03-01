# SD数据校准总体方案

**文档版本**: v1.0 Final
**创建日期**: 2026-02-25
**目的**: 解决SD合成数据增强导致的标尺漂移和系统性低估问题

---

## 目录

1. [背景与问题](#一背景与问题)
2. [方案概述](#二方案概述)
3. [阶段0：诊断分析](#三阶段0诊断分析)
4. [阶段1：分桶Importance Sampling](#四阶段1分桶importance-sampling)
5. [阶段2：一致性门控伪标签](#五阶段2一致性门控伪标签)
6. [对照实验设计](#六对照实验设计)
7. [实施路线图](#七实施路线图)
8. [技术细节说明](#八技术细节说明)
9. [常见问题](#九常见问题)

---

## 一、背景与问题

### 1.1 问题描述

在使用SD合成数据增强训练Mixed模型时，我们观察到以下现象：

| 现象 | Baseline | Mixed 989 | Mixed 1260 |
|------|----------|-----------|------------|
| Test_Real MAE | 0.0187 | 0.0188 | 0.0187 |
| Test_Real **Bias** | +0.004 | **-0.094** | **-0.093** |
| Test_Ext ρ | 0.718 | 0.710 | 0.670 |

**核心问题**：Mixed模型虽然MAE与Baseline相当，但存在**系统性负偏差**，且跨域性能下降。

### 1.2 根本原因分析

经过分析，问题的根本原因是**SD数据的严重度分布与真实数据不匹配**：

```
WoodScape S均值: 0.408
SD全量 8960 S均值:   ~0.29  (需诊断确认)
SD筛选子集 S均值:   0.267~0.287  ← 筛选后进一步偏低
```

这导致：
1. **边际分布偏移**：训练数据中低S值样本过多
2. **标尺漂移**：模型学习到的S标尺偏向低值区间
3. **推理结果**：系统性低估，等级区分度下降

**关键决策**：校准方案以**全量8960张SD数据**为起点，而非筛选后的989/1260子集。

**理由**：
- 校准方案本身用于解决分布偏移问题，可直接用于全量数据
- 全量数据量更大，增强效果可能更好
- 筛选子集(989/1260)作为对照组，用于验证筛选策略的必要性

### 1.3 解决思路

不改变SD数据本身，而是通过**校准**来纠正分布偏移和标尺漂移：

1. **Importance Sampling**：调整SD样本的采样权重，使其S分布接近WoodScape
2. **一致性门控**：用可靠的teacher模型对SD样本进行重新标注，并过滤掉不可信样本

---

## 二、方案概述

### 2.1 方案优先级

```
阶段0: 诊断分析
    ↓
    ├─ 确认问题原因
    ↓
阶段1: 分桶Importance Sampling（D-only）
    ↓
    ├─ 评估：bias恢复？ρ保持？
    │   ↓ Yes → 成功
    │   ↓ No
    ↓
阶段2: 一致性门控伪标签（A+-only）
    ↓
    ├─ 评估：进一步改善？
    │   ↓ Yes → 成功
    │   ↓ No
    ↓
阶段3: D + A+ 结合（最后手段）
```

### 2.2 核心设计原则

1. **低风险优先**：先用最简单的方法验证假设
2. **变量解耦**：每个方案单独验证，确保可归因
3. **完全可逆**：所有修改都是可撤回的
4. **保持稳定**：不引入训练不稳定因素

---

## 三、阶段0：诊断分析

### 3.1 诊断目的

在实施任何校准之前，先通过诊断分析确认：
- SD数据与WoodScape的分布差异有多大？
- Baseline模型在SD数据上的预测质量如何？
- 问题主要来自分布偏移还是标尺不一致？

### 3.2 诊断清单

#### 诊断1：S分布对比

```python
# 计算SD和WoodScape的S分布直方图
sd_hist, bins = np.histogram(sd_S_gt, bins=20, range=(0, 1))
ws_hist, _ = np.histogram(ws_S_gt, bins=bins)

# Kolmogorov-Smirnov检验
from scipy.stats import ks_2samp
ks_stat, ks_pval = ks_2samp(sd_S_gt, ws_S_gt)

print(f"KS统计量: {ks_stat:.4f}, p值: {ks_pval:.4f}")
```

**输出示例**：
```
SD S均值: 0.267
WoodScape S均值: 0.408
KS统计量: 0.35, p值 < 0.001  ← 显著差异
```

#### 诊断2：Teacher内部一致性

```python
# 计算Baseline模型在SD数据上的gap
gap = np.abs(S_hat_base - S_tilde_base)

# 按S区间统计gap的分布
bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
for low, high in bins:
    mask = (sd_S_gt >= low) & (sd_S_gt < high)
    gap_dist = gap[mask]
    epsilon = np.percentile(gap_dist, 90)  # 90分位数作为阈值
    print(f"S区间[{low}, {high}): gap均值={gap_dist.mean():.4f}, ε(90%)={epsilon:.4f}")
```

**输出示例**：
```
S区间[0, 0.25): gap均值=0.02, ε(90%)=0.04
S区间[0.25, 0.5): gap均值=0.03, ε(90%)=0.06
S区间[0.5, 0.75): gap均值=0.04, ε(90%)=0.08
S区间[0.75, 1.0): gap均值=0.05, ε(90%)=0.10
```

#### 诊断3：等级区分度分析

```python
# 在Test_Ext上计算各等级的S_hat均值
for level in [1, 2, 3, 4, 5]:
    baseline_mean = baseline_S_hat[test_ext_level == level].mean()
    mixed_mean = mixed_S_hat[test_ext_level == level].mean()
    print(f"Level {level}: Baseline={baseline_mean:.3f}, Mixed={mixed_mean:.3f}")

# 计算斜率（区分度）
baseline_slope = np.polyfit([1,2,3,4,5], baseline_means, 1)[0]
mixed_slope = np.polyfit([1,2,3,4,5], mixed_means, 1)[0]
compression = baseline_slope / mixed_slope
print(f"斜率压缩比: {compression:.2f} (应>1表示Mixed区分度下降)")
```

**输出示例**：
```
Level 1: Baseline=0.518, Mixed=0.322
Level 2: Baseline=0.615, Mixed=0.380
Level 3: Baseline=0.695, Mixed=0.433
Level 4: Baseline=0.786, Mixed=0.505
Level 5: Baseline=0.902, Mixed=0.645
斜率压缩比: 1.45  ← Mixed区分度显著下降
```

#### 诊断4：视觉抽样检查

```python
# 从SD数据中随机抽样，人工检查视觉严重度
n_sample = 50
samples = random_sample(sd_dataset, n_sample, stratify_by=S_gt)

# 对每个样本，检查：
# - 视觉上的脏污程度
# - S_gt标签是否合理
# - 是否存在"视觉轻污但S_gt偏高"的情况
```

### 3.3 诊断输出

诊断完成后，输出以下信息：

| 诊断项 | 结果 | 决策 |
|--------|------|------|
| 分布偏移程度 | KS=0.35, p<0.001 | 显著偏移，需校准 |
| Teacher一致性 | gap均值=0.03, ε=0.04-0.10 | 可用于门控 |
| 等级区分度 | 压缩比=1.45 | 显著下降 |
| 视觉检查 | 90%样本标签合理 | SD数据可用 |

**决策**：进入阶段1，实施Importance Sampling

---

## 四、阶段1：分桶Importance Sampling

### 4.1 方法原理

**核心思想**：让SD数据在训练中的采样分布接近WoodScape的S分布。

```
原始SD采样:  均匀采样 → S分布偏低
校准后采样:   加权采样 → S分布接近WoodScape
```

**权重计算公式**：
```
weight(S_bin) = WoodScape频率(S_bin) / SD频率(S_bin)
```

### 4.2 实现步骤

#### 步骤1：分桶

```python
import numpy as np

# 准备数据
# 全量SD数据：8960张
sd_S_gt = load_sd_labels('sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv')
# WoodScape数据：4000张（train）
ws_S_gt = load_ws_labels('dataset/woodscape_processed/meta/labels_index_ablation.csv')
# 只使用train split
ws_S_gt = ws_S_gt[ws_S_gt['split'] == 'train']['S_full_wgap_alpha50']

# 将S值分成K个桶（K=10或20）
K = 10
bins = np.linspace(0, 1, K + 1)
# bins = [0.0, 0.1, 0.2, ..., 1.0]

# 统计每个桶的频率
sd_hist, _ = np.histogram(sd_S_gt, bins=bins)
ws_hist, _ = np.histogram(ws_S_gt, bins=bins)

print(f"SD分布: {sd_hist}")
print(f"WS分布: {ws_hist}")
```

**输出示例**：
```
桶编号    [0.0, 0.1)  [0.1, 0.2)  [0.2, 0.3)  ...
SD频率:    150         300         250      ...  (偏低集中在低S)
WS频率:    50          100         200      ...  (更均匀分布)
```

#### 步骤2：计算权重

```python
# 计算权重（加平滑避免除零）
delta = 1e-6
weights = (ws_hist + delta) / (sd_hist + delta)

# 截断权重（避免极端值）
w_min, w_max = 0.5, 2.0
weights = np.clip(weights, w_min, w_max)

print(f"校准权重: {weights}")
```

**输出示例**：
```
桶编号        0     1     2     3    ...
权重:      0.67   0.67   1.0   1.5  ...
含义:      ↓降低  ↓降低  不变  ↑提高
```

#### 步骤3：分配样本权重

```python
# 根据每个SD样本的S值，分配对应的权重
sample_weights = np.zeros(len(sd_S_gt))

for i, s in enumerate(sd_S_gt):
    bin_idx = np.digitize(s, bins) - 1
    sample_weights[i] = weights[bin_idx]

# sample_weights[i] 表示第i个SD样本的采样权重
```

### 4.3 两层采样器实现

#### 为什么需要两层采样？

直接在混合数据集上用WeightedRandomSampler会有问题：
- 会改变real:sd的比例
- 会导致训练不稳定

**解决方案**：两层采样
1. 第一层：按固定比例(80/20)选择real或sd子集
2. 第二层：在sd子集内按权重采样

#### 完整代码

```python
from torch.utils.data import Sampler, WeightedRandomSampler, DataLoader
import random

class TwoLayerImportanceSampler(Sampler):
    """
    两层采样器：
    - 第一层：按real_ratio选择real或sd
    - 第二层：sd子集内按权重采样

    参数:
        n_real: real样本数量
        n_sd: sd样本数量
        sd_weights: sd样本的权重 [n_sd]
        real_ratio: 每次采样选择real的概率（默认0.8）
    """
    def __init__(self, n_real, n_sd, sd_weights, real_ratio=0.8):
        self.n_real = n_real
        self.n_sd = n_sd
        self.n_total = n_real + n_sd
        self.real_ratio = real_ratio

        # 创建SD采样器并缓存
        self.sd_sampler = WeightedRandomSampler(
            sd_weights,
            num_samples=len(sd_weights),
            replacement=True
        )
        self.sd_indices_pool = list(self.sd_sampler)
        self.sd_ptr = 0

    def __iter__(self):
        while True:
            # 第一层：选择real或sd
            if random.random() < self.real_ratio:
                # 采样real样本 [0, n_real-1]
                idx = random.randint(0, self.n_real - 1)
                yield idx
            else:
                # 采样sd样本 [n_real, n_real+n_sd-1]
                sd_local_idx = self.sd_indices_pool[self.sd_ptr]
                self.sd_ptr = (self.sd_ptr + 1) % len(self.sd_indices_pool)
                # 映射到全局索引
                yield self.n_real + sd_local_idx

    def __len__(self):
        # 返回一个epoch的样本数
        return self.n_total


class MixedDataset(torch.utils.data.Dataset):
    """
    混合数据集：
    - 索引 [0, n_real-1] 对应 real 样本
    - 索引 [n_real, n_total-1] 对应 sd 样本
    """
    def __init__(self, real_dataset, sd_dataset):
        self.real_dataset = real_dataset
        self.sd_dataset = sd_dataset
        self.n_real = len(real_dataset)
        self.n_sd = len(sd_dataset)
        self.offset = self.n_real

    def __getitem__(self, idx):
        if idx < self.n_real:
            # Real样本
            return self.real_dataset[idx]
        else:
            # SD样本
            sd_idx = idx - self.offset
            return self.sd_dataset[sd_idx]

    def __len__(self):
        return self.n_real + self.n_sd
```

#### 使用方法

```python
# 准备数据
real_dataset = WoodscapeDataset(...)      # 4000 samples
sd_dataset = FullSDDataset(...)           # 8960 samples (全量)
sample_weights = compute_sd_weights(sd_dataset)  # 从4.2节计算

# 创建混合数据集和采样器
mixed_dataset = MixedDataset(real_dataset, sd_dataset)
sampler = TwoLayerImportanceSampler(
    n_real=len(real_dataset),
    n_sd=len(sd_dataset),
    sd_weights=sample_weights,
    real_ratio=0.8  # 保持80% real, 20% sd
)

# 数据集规模：Real(4000) + SD(8960) = 12960
# 或使用 real_ratio = 4000/12960 ≈ 0.31 保持原始比例

# 创建DataLoader
train_loader = DataLoader(
    mixed_dataset,
    sampler=sampler,
    batch_size=16,
    # 注意：使用自定义sampler时不要设置shuffle=True
)
```

### 4.4 训练配置（确保对照组可比）

```python
# 使用固定总步数训练，确保对照组可比
TRAINING_CONFIG = {
    'baseline': {
        'n_samples': 4000,
        'batch_size': 16,
        'steps_per_epoch': 250,  # 4000/16
        'total_steps': 10000,    # 固定总步数
    },
    'mixed_8960': {  # 主实验：全量8960 + 校准
        'n_samples': 12960,  # WS 4000 + SD 8960
        'batch_size': 16,
        'steps_per_epoch': 810,  # 12960/16
        'total_steps': 10000,    # 相同总步数
    },
    'mixed_989': {   # 对照：筛选子集989
        'n_samples': 4989,
        'batch_size': 16,
        'steps_per_epoch': 312,
        'total_steps': 10000,
    },
    'mixed_1260': {  # 对照：筛选子集1260
        'n_samples': 5260,
        'batch_size': 16,
        'steps_per_epoch': 329,
        'total_steps': 10000,
    }
}

def train_fixed_steps(model, train_loader, total_steps=10000):
    """按固定总步数训练"""
    step = 0
    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            loss = model.train_step(batch)
            step += 1
```

**关键点**：
- ✅ 使用相同总步数，确保可比
- ✅ 不依赖epoch定义（避免混淆）
- ✅ 所有实验组统一用10000步

### 4.5 评估指标

| 指标 | 期望效果 | 失败标准 |
|------|----------|----------|
| Test_Real bias | → 0 | 仍<-0.05 |
| Test_Real MAE | ≤ 0.019 | >0.020 |
| Test_Ext ρ | ≥ 0.71 | <0.70 (硬约束) |
| Level斜率比 | → 1.0 | 仍<1.2 |

**成功标准**：满足≥3项且硬约束满足 → 采用D-only

---

## 五、阶段2：一致性门控伪标签

### 5.1 方法原理

**核心思想**：用可靠的Baseline模型作为teacher，对SD样本进行重新标注，并通过一致性门控过滤掉不可信样本。

```
原始SD标签: S_gt (可能偏低)
Teacher预测: S_hat_base, S_tilde_base (来自Baseline)
门控条件:   |S_hat_base - S_tilde_base| < ε  (一致性良好)
校准标签:   S_cal = λ*S_hat_base + (1-λ)*S_tilde_base
```

### 5.2 数据驱动的门控阈值

```python
# 在WoodScape train/val上统计gap分布
gap = np.abs(S_hat_base - S_tilde_base)

# 自适应分桶（保证每桶至少N≥200样本）
def adaptive_bins(S, min_samples=200):
    bins = []
    current_start = 0.0
    while current_start < 1.0:
        next_end = min(current_start + 0.25, 1.0)
        n_samples = ((S >= current_start) & (S < next_end)).sum()

        if n_samples >= min_samples or next_end == 1.0:
            bins.append((current_start, next_end))
            current_start = next_end
        else:
            # 样本不足，扩大bin
            next_end = min(current_start + 0.50, 1.0)
            bins.append((current_start, next_end))
            current_start = next_end
    return bins

bins = adaptive_bins(S_ws, min_samples=200)

# 计算每个bin的阈值
epsilon = {}  # {bin_range: threshold}
for low, high in bins:
    mask = (S_ws >= low) & (S_ws < high)
    gap_dist = gap[mask]
    epsilon[(low, high)] = np.percentile(gap_dist, 90)
```

**输出示例**：
```
S区间[0.0, 0.25): ε=0.04
S区间[0.25, 0.5): ε=0.06
S区间[0.5, 0.75): ε=0.08
S区间[0.75, 1.0]: ε=0.10
```

### 5.3 软门控函数

```python
def lambda_gap(gap, epsilon, tau):
    """
    软门控函数

    参数:
        gap: |S_hat_base - S_tilde_base|
        epsilon: 该S区间的阈值
        tau: 过渡参数，设为0.1×epsilon

    返回:
        lam: 门控权重 [0, 1]
             - gap << epsilon: lam → 1 (完全信任)
             - gap ≈ epsilon: lam 平滑过渡
             - gap >> epsilon: lam → 0 (不信任)
    """
    return torch.sigmoid((epsilon - gap) / tau)

# 使用示例
tau = 0.1 * epsilon  # 过渡区约为阈值的10%
lam = lambda_gap(gap, epsilon, tau)
```

### 5.4 仅对SD子集应用门控

**关键设计**：门控只作用于SD样本，Real样本保持原标签

```python
def compute_gate_weights(batch):
    """
    计算门控权重和校准标签

    Real样本: lam=1, S_cal=S_gt (不使用teacher)
    SD样本: lam=gate, S_cal=teacher加权
    """
    B = len(batch.images)
    lam = torch.ones(B)  # 初始化为1
    S_cal = batch.S_gt.clone()  # 初始化为原标签

    # 找出SD样本
    is_sd = batch.is_sd  # [B] bool tensor
    sd_indices = torch.where(is_sd)[0]

    if len(sd_indices) > 0:
        # 只对SD样本计算门控
        sd_gap = torch.abs(
            batch.S_hat_base[sd_indices] -
            batch.S_tilde_base[sd_indices]
        )
        sd_epsilon = batch.epsilon[sd_indices]
        sd_tau = 0.1 * sd_epsilon

        # 软门控
        sd_lam = torch.sigmoid((sd_epsilon - sd_gap) / sd_tau)

        # 最小阈值
        lambda_min = 0.1
        sd_lam = sd_lam * (sd_lam >= lambda_min).float()

        # 校准标签
        sd_S_cal = sd_lam * batch.S_hat_base[sd_indices] + \
                   (1 - sd_lam) * batch.S_tilde_base[sd_indices]

        # 更新SD样本
        lam[sd_indices] = sd_lam
        S_cal[sd_indices] = sd_S_cal

    return lam, S_cal
```

### 5.5 Mixed Loss实现

```python
def mixed_loss_forward(batch, model, lambda_glob=1.0):
    """
    Mixed模型损失函数

    参数:
        batch: 包含
            - images: [B, ...]
            - S_gt: [B]
            - is_sd: [B] bool
            - S_hat_base: [B] (仅SD需要)
            - S_tilde_base: [B] (仅SD需要)
        model: Mixed模型
        lambda_glob: global loss权重
    """
    # Tile loss (所有样本都训练)
    G_hat = model.tile_head(batch.images)
    L_tile = F.mse_loss(G_hat, batch.G_gt)

    # 计算门控
    lam, S_cal = compute_gate_weights(batch)

    # Global loss
    S_hat = model.global_head(batch.images)
    per_sample_loss = (S_hat - S_cal) ** 2

    # 按lam加权，并归一化
    weighted_loss = lam * per_sample_loss
    L_glob = torch.sum(weighted_loss) / (torch.sum(lam) + 1e-8)

    # 总损失
    L_total = L_tile + lambda_glob * L_glob

    # 监控指标（仅统计SD子集）
    is_sd = batch.is_sd
    metrics = {
        'loss': L_total.item(),
        'tile_loss': L_tile.item(),
        'glob_loss': L_glob.item(),
        'pass_rate_sd': (lam[is_sd] >= 0.1).float().mean().item(),
        'lam_mean_sd': lam[is_sd].mean().item(),
        'lam_std_sd': lam[is_sd].std().item(),
    }

    return L_total, metrics
```

### 5.6 Batch数据准备

```python
class MixedDataBatch:
    """混合训练的Batch数据结构"""
    def __init__(self, images, G_gt, S_gt, is_sd,
                 S_hat_base=None, S_tilde_base=None, epsilon=None):
        self.images = images
        self.G_gt = G_gt
        self.S_gt = S_gt
        self.is_sd = is_sd  # [B] bool tensor
        self.S_hat_base = S_hat_base
        self.S_tilde_base = S_tilde_base
        self.epsilon = epsilon


def collate_fn(batch_list):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch_list])
    G_gt = torch.stack([item['G_gt'] for item in batch_list])
    S_gt = torch.tensor([item['S_gt'] for item in batch_list])
    is_sd = torch.tensor([item['is_sd'] for item in batch_list], dtype=torch.bool)

    # SD样本需要的额外信息
    S_hat_base = torch.tensor([item.get('S_hat_base', item['S_gt']) for item in batch_list])
    S_tilde_base = torch.tensor([item.get('S_tilde_base', item['S_gt']) for item in batch_list])
    epsilon = torch.tensor([item.get('epsilon', 0.05) for item in batch_list])

    return MixedDataBatch(images, G_gt, S_gt, is_sd, S_hat_base, S_tilde_base, epsilon)
```

---

## 六、对照实验设计

### 6.1 实验组

| 实验组 | SD数据 | 配置 | 目的 |
|--------|--------|------|------|
| **Baseline** | - | WoodScape only | 基准 |
| **Mixed(8960, no cal)** | 全量8960 | 无校准 | 验证全量数据的问题 |
| **Mixed(8960, D-only)** | 全量8960 | +Importance Sampling | **主实验** |
| **Mixed(8960, A+-only)** | 全量8960 | +一致性门控 | 主实验备选 |
| **Mixed(8960, D+A+)** | 全量8960 | 两者结合 | 主实验备选 |
| Mixed(989, no cal) | 筛选989 | 无校准 | 对照：筛选效果 |
| Mixed(1260, no cal) | 筛选1260 | 无校准 | 对照：筛选效果 |

**关键对比**：
- 8960 vs 989/1260：验证全量数据+校准 是否优于 筛选子集
- D-only vs no cal：验证校准是否有效
- A+ vs D：验证哪种校准更有效

### 6.2 统一评估面板

| 指标 | Baseline | Mixed(8960, no cal) | Mixed(8960, D-only) | Mixed(8960, A+) | Mixed(989) | Mixed(1260) |
|------|----------|---------------------|---------------------|----------------|------------|------------|
| Test_Real bias | ~0 | ? | ? | ? | -0.09 | -0.09 |
| Test_Real MAE | 0.019 | ? | ≤0.019? | ≤0.019? | 0.019 | 0.019 |
| Test_Ext ρ | 0.72 | ? | ≥0.70? | ≥0.70? | 0.71 | 0.67 |
| Level斜率比 | 1.0 | ? | →1.0? | →1.0? | 0.8 | 0.7 |
| Pass rate SD | - | - | - | 记录 | - | - |

### 6.3 硬约束

**Test_Ext ρ ≥ 0.70** 必须满足，否则认为方案失败

### 6.4 核心对比

**主要对比**：Mixed(8960, D-only) vs Mixed(989/1260)
- 如果8960+校准 **优于** 筛选子集 → 说明全量数据+校准更优
- 如果8960+校准 **相当于** 筛选子集 → 说明校准可以替代筛选
- 如果8960+校准 **差于** 筛选子集 → 说明筛选仍必要

---

## 七、实施路线图

### 7.1 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| 阶段0 | 诊断分析实现 | 2小时 |
| 阶段1 | D-only实现+训练+评估 | 0.5天 |
| 阶段2 | A+-only实现+训练+评估 | 1天 |
| 阶段3 | D+A+实现+训练+评估 | 0.5天 |
| 总计 | - | 2-3天 |

### 7.2 实施流程

```
┌─────────────────────────────────────────────┐
│ 阶段0: 诊断分析                              │
│ - 实现诊断脚本                              │
│ - 运行诊断，输出报告                        │
│ - 确认问题原因                              │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 阶段1: D-only                              │
│ - 实现TwoLayerImportanceSampler             │
│ - 训练Mixed模型(10000步)                   │
│ - 评估Test_Real + Test_Ext                  │
│ - 判断: 满足标准？                          │
└──────────────┬──────────────────────────────┘
               │
               ▼ (如果D-only不足)
┌─────────────────────────────────────────────┐
│ 阶段2: A+-only                              │
│ - 实现软门控伪标签                          │
│ - 修改MixedDataset添加is_sd标记             │
│ - 训练Mixed模型(10000步)                   │
│ - 评估Test_Real + Test_Ext                  │
│ - 判断: 满足标准？                          │
└──────────────┬──────────────────────────────┘
               │
               ▼ (如果A+-only不足)
┌─────────────────────────────────────────────┐
│ 阶段3: D+A+                                 │
│ - 结合两种方法                              │
│ - 训练Mixed模型(10000步)                   │
│ - 评估Test_Real + Test_Ext                  │
│ - 输出最终结论                              │
└─────────────────────────────────────────────┘
```

---

## 八、技术细节说明

### 8.1 为什么用分桶而不是连续密度估计？

**问题**：连续密度估计（如KDE）在边界处容易不稳定

**解决方案**：分桶+平滑
```
weight = (ws_hist + δ) / (sd_hist + δ)
```

其中δ=1e-6是平滑参数，避免除零。

### 8.2 为什么要截断权重？

**问题**：某些桶的SD样本很少，权重会爆炸

**解决方案**：截断到[0.5, 2.0]
```
weight = clip(weight, 0.5, 2.0)
```

这样既保持了分布调整的效果，又避免了极端值。

### 8.3 为什么要软门控而不是硬门控？

**问题**：硬门控（gap<ε ? pass : reject）对阈值敏感

**解决方案**：软门控
```
lam = sigmoid((ε - gap) / τ)
```

优点：
- 平滑过渡，训练更稳定
- 可解释：τ控制过渡区宽度
- 鲁棒：对ε的微小变化不敏感

### 8.4 为什么Real样本不参与门控？

**逻辑**：
- 门控的目的是过滤"不可信的SD样本"
- Real样本来自WoodScape，标签可靠，不需要过滤
- 若Real样本也参与门控，会削弱真实监督强度

### 8.5 为什么使用固定总步数？

**问题**：不同实验组的数据量不同，按epoch训练不可比

**解决方案**：固定总步数
```
Baseline: 4000样本 → 250 steps/epoch → 40 epochs = 10000步
Mixed: 4989样本 → 312 steps/epoch → 32 epochs = 10000步
```

确保所有实验组看到相同数量的梯度更新。

---

## 九、常见问题

### Q1: 如果D-only已经有效，还需要尝试A+吗？

**A**: 不需要。D-only有效说明问题主要是分布偏移，采用D-only即可。

### Q2: 权重截断范围[0.5, 2.0]是如何确定的？

**A**: 这是经验值：
- 下界0.5：最低保留50%采样，避免完全丢弃某些桶
- 上界2.0：最高2倍采样，避免某些桶过度

可以调整为[0.3, 3.0]更激进，或[0.7, 1.5]更保守。

### Q3: τ=0.1×ε的含义是什么？

**A**: τ控制软门控的过渡区宽度：
- τ=0.1×ε：过渡区宽度约为阈值的10%
- 例如ε=0.05时，τ=0.005
- gap在[0.045, 0.055]范围内会平滑过渡

### Q4: 如何判断SD数据的视觉严重度？

**A**: 抽样检查：
1. 随机选择50张SD图像
2. 按S_gt分成5个等级
3. 人工检查每个等级的视觉脏污程度
4. 确认S_gt是否与视觉一致

### Q5: 如果所有校准方案都无效怎么办？

**A**: 说明问题不在分布偏移，可能需要：
1. 重新审视SD数据生成质量
2. 调整SD筛选策略
3. 考虑其他校准方法（如分位数匹配）

### Q6: bootstrap置信区间如何计算？

**A**:
```python
from scipy.stats import bootstrap

def rho_metric(data, axis):
    return spearmanr(data[:,0], data[:,1])[0]

# data: [N, 2] = [S_gt, S_hat]
res = bootstrap(data, rho_metric, n_bootstrap=1000)
ci_low, ci_high = res.confidence_interval.alpha(0.95)
print(f"ρ = {rho:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
```

### Q7: 如何监控训练稳定性？

**A**: 关注以下指标：
- pass_rate_sd: 应稳定在50%-90%
- lam_mean_sd: 应稳定在0.5-0.8
- loss: 应平稳下降，无突变
- grad_norm: 应在正常范围

### Q8: 能否同时使用D和A+？

**A**: 可以，但需要注意：
- D的权重要保守（如[0.7, 1.5]）
- A的pass率不能太低（>50%）
- 监控最终参与训练的SD样本数量

---

## 附录：数据集对比说明

### 数据集全景

| 数据集 | 数量 | S均值 | S范围 | 用途 |
|--------|------|-------|-------|------|
| WoodScape Train | 4000 | 0.408 | [0.0, 1.0] | 真实数据 |
| **SD全量 8960** | **8960** | **~0.29** | **[0.0, ~1.0]** | **主实验载体** |
| SD筛选989 | 989 | 0.267 | [0.055, 0.493] | 对照：严格筛选 |
| SD筛选1260 | 1260 | 0.287 | [0.064, 0.543] | 对照：保守筛选 |

### 为什么用全量8960作为主实验载体？

**理由1：校准方案的本质**
- 校准方案本身就是用来解决"分布偏移"问题的
- 如果校准有效，理应能处理全量数据的分布问题
- 不需要事前筛选，直接校准即可

**理由2：数据量优势**
- 全量8960 > 筛选989/1260
- 更多数据意味着更强的增强效果
- 训练更稳定

**理由3：验证筛选的必要性**
- 通过对比实验可以验证：
  - 如果8960+校准 **优于** 989/1260 → 筛选不必要
  - 如果8960+校准 **相当于** 989/1260 → 校准可替代筛选
  - 如果8960+校准 **差于** 989/1260 → 筛选仍必要

### 实验组的关系

```
                    ┌─────────────────┐
                    │   WoodScape    │
                    │     (4000)      │
                    └────────┬────────┘
                             │
            ┌────────────┼────────────┐
            │            │            │
    ┌───────▼────┐  ┌───▼────┐  ┌───▼────┐
    │  Baseline  │  │ Mixed  │  │ Mixed  │
    │   (4000)   │  │ (8960) │  │ (989)  │
    └────────────┘  │无校准  │  │无校准  │
                   └────┬───┘  └────────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
      ┌─────▼─────┐ ┌───▼─────┐ ┌──▼─────┐
      │  D-only   │ │  A+-only│ │  D+A+   │
      └───────────┘ └─────────┘ └─────────┘
           (主实验)

对照组：
- Mixed(989, no cal)   → 验证筛选效果
- Mixed(1260, no cal)  → 验证保守筛选效果
```

### 核心假设

**假设1：校准方案有效**
- 如果假设成立：Mixed(8960, D-only) 优于 Mixed(8960, no cal)
- 如果假设不成立：Mixed(8960, D-only) 相当于/差于 Mixed(8960, no cal)

**假设2：校准可替代筛选**
- 如果假设成立：Mixed(8960, D-only) ≈ 优于 Mixed(989/1260)
- 如果假设不成立：Mixed(8960, D-only) 差于 Mixed(989/1260)

**假设3：全量数据+校准是最优策略**
- 如果假设成立：Mixed(8960, D-only) 优于所有对照组
- 如果假设不成立：筛选子集可能仍有优势

### 决策树

```
开始: Mixed(8960, no cal) 表现如何？
    |
    ├─ 已经很好 (bias≈0, ρ≥0.71)
    │   → 无需校准，直接使用全量数据
    │
    ├─ 存在问题 (bias<-0.05, ρ<0.70)
    │   ↓
    │   尝试 D-only 校准
    │   ↓
    │   D-only 有效？
    │   ├─ Yes → 采用 Mixed(8960, D-only)
    │   │       对比 Mixed(989/1260)
    │   │       若优于 → 全量+校准最优
    │   │       若相当 → 校准可替代筛选
    │   │       若差于 → 筛选仍必要
    │   │
    │   └─ No → 尝试 A+-only 或 D+A+
    │
    └─ 最终选择表现最好的方案
```

---

### A. 关键代码文件

```
项目结构:
├── analyze_sd_distribution.py          # 阶段0诊断
├── two_layer_sampler.py                 # 阶段1采样器
├── mixed_dataset_with_gate.py           # 阶段2数据集
├── train_fixed_steps.py                 # 固定步数训练
└── evaluate_all_experiments.py          # 对照实验评估
```

### B. 参考配置

```yaml
# data.yaml
data:
  real:
    n_samples: 4000
    s_mean: 0.408
    source: woodscape_processed/meta/labels_index_ablation.csv
  sd_full:
    n_samples: 8960
    s_mean: ~0.29  # 需诊断确认
    source: sd_scripts/lora/baseline_filtered_8960/npz_manifest_8960.csv
  sd_989:
    n_samples: 989
    s_mean: 0.267
    source: 筛选子集
  sd_1260:
    n_samples: 1260
    s_mean: 0.287
    source: 保守筛选子集

# sampler.yaml
sampler:
  real_ratio: 0.8
  n_bins: 10
  w_min: 0.5
  w_max: 2.0

# gate.yaml
gate:
  tau_ratio: 0.1  # tau = 0.1 * epsilon
  lambda_min: 0.1
  epsilon_percentile: 90

# training.yaml
training:
  batch_size: 16
  total_steps: 10000
  eval_interval: 500
```

### C. 预期输出示例

```
=== 阶段0诊断报告 ===
SD全量8960 S均值: 0.29
WoodScape S均值: 0.41
KS统计量: 0.30, p < 0.001
分布偏移: 显著
Teacher一致性: gap均值=0.03, ε∈[0.04, 0.10]
等级区分度: 压缩比=1.45
结论: 进入阶段1

=== 阶段1 D-only结果 ===
Test_Real bias: 0.012 ✅
Test_Real MAE: 0.0188 ✅
Test_Ext ρ: 0.712 ✅
Level斜率比: 1.05 ✅
结论: D-only有效，采用

=== 对照实验总结 ===
Baseline:        ρ=0.718, bias=0.004
Mixed(8960):     ρ=0.680, bias=-0.08
Mixed(989):      ρ=0.710, bias=-0.09
Mixed(1260):     ρ=0.670, bias=-0.09
Mixed(8960, D):  ρ=0.715, bias=0.01 ← 成功

=== 核心发现 ===
1. 全量8960 + 校准 ≈ 筛选989 + 校准
2. 校准可以替代严格筛选，保留更多数据
3. 全量数据 + 校准是最优策略
```

---

**文档结束**

本文档提供了SD数据校准的完整方案，从问题分析到实施细节，可直接用于工程实施。
