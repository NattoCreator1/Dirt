# Phase 1.2 实施总结 - LoRA 训练数据准备优化

**版本**: v1.2
**日期**: 2025-02-10
**状态**: Phase 1 优化完成，Phase 2 就绪

---

## 修正概述

基于详细的可行性评估和用户反馈，Phase 1 数据准备脚本已完成以下关键修正：

### 高优先级修正（必须）

| 问题 | v1.1 方案 | v1.2 修正 | 状态 |
|------|----------|----------|------|
| **Special Token** | `<trans>`, `<semi>`, `<opaque>`, `<sev1-4>` | 自然语言锚点 | ✅ |
| **背景抑制随机化** | 固定强度 | 随机强度混合 (0.5-1.5x) | ✅ |

### 中优先级修正（建议）

| 问题 | v1.1 方案 | v1.2 修正 | 状态 |
|------|----------|----------|------|
| **类别不平衡** | 自然分布 (opaque 62%) | 分层采样 (1:1:1) | ✅ |
| **UNet 兼容性验证** | 未验证 | 结构一致性确认 | ✅ |
| **评估指标** | 全图 FID | Spill Rate + Mask 区域指标 | ✅ |

---

## 详细修正说明

### 1. Special Token 改为自然语言锚点

**问题**：`<opaque>` 等特殊 token 不在 SD 词表中，模型可能无法学习控制信号

**修正**：
```python
# v1.1 (问题)
CLASS_TOKENS = {
    1: "<trans>",
    2: "<semi>",
    3: "<opaque>",
}
SEVERITY_TOKENS = {
    (0.0, 0.15): "<sev1>",
    ...
}

# v1.2 (修正)
CLASS_TOKENS = {
    1: "transparent soiling layer",
    2: "semi-transparent dirt smudge",
    3: "opaque heavy stains",
}
SEVERITY_TOKENS = {
    (0.0, 0.15): "mild",
    (0.15, 0.35): "moderate",
    (0.35, 0.60): "noticeable",
    (0.60, 1.00): "severe",
}
```

**Caption 示例**：
- v1.1: `<opaque> <sev3> on camera lens, out of focus foreground, subtle glare`
- v1.2: `noticeable opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible`

### 2. 背景抑制强度随机化

**问题**：固定强度可能导致 LoRA 学到"背景模糊风格"

**修正**：
```python
def construct_training_image(
    dirty_image, mask,
    method: Literal["blur", "desaturate", "downsample", "random"] = "random",
    intensity: float = 1.0,  # 基础强度
    rng: np.random.Generator = None,
):
    # 随机方法选择
    if method == "random":
        method = rng.choice(["blur", "desaturate", "downsample"])

    # 随机强度变化 (0.5 - 1.5)
    actual_intensity = np.clip(intensity * rng.uniform(0.5, 1.5), 0.0, 2.0)
    ...
```

**效果**：
- 一部分样本背景抑制很强（压语义）
- 一部分样本抑制很弱（保真实统计）
- 三种方法随机混合

### 3. 分层采样解决类别不平衡

**问题**：自然分布 opaque 占 62%，trans 仅 9%

**修正**：
```python
# 预计算每个样本的主导类别
# 按目标比例 (1:1:1) 分层采样
if stratify:
    samples_per_class = {}
    for cls in [1, 2, 3]:
        class_df = split_df[split_df['dominant_class'] == cls]
        target_count = int(max_samples * target_class_ratios[cls])
        samples_per_class[cls] = class_df.sample(n=target_count, random_state=seed)
```

### 4. UNet 结构一致性验证

**验证结果**：✓ **COMPATIBLE**

```
Critical Configuration Comparison:
✓ block_out_channels             | base: [320, 640, 1280, 1280] | inpaint: [320, 640, 1280, 1280]
✓ layers_per_block               | base: 2 | inpaint: 2
✓ cross_attention_dim            | base: 1024 | inpaint: 1024
✓ attention_head_dim             | base: [5, 10, 20, 20] | inpaint: [5, 10, 20, 20]
✓ down_block_types               | base: [CrossAttnDownBlock2D, ...]
✓ up_block_types                 | base: [UpBlock2D, CrossAttnUpBlock2D, ...]
✓ use_linear_projection          | base: True | inpaint: True
✓ norm_num_groups                | base: 32 | inpaint: 32

Expected Differences:
✓ in_channels: base=4, inpaint=9 (Expected difference)

Attention Layer Analysis:
SD2.1 base: 13 total (6 down + 6 up + 1 mid)
SD2-inpainting: 13 total (6 down + 6 up + 1 mid)
```

**结论**：LoRA 注入在 attention 层可以在两个模型间迁移

---

## 新增脚本文件

| 文件 | 功能 |
|------|------|
| [`01_prepare_training_data.py`](sd_scripts/lora/01_prepare_training_data.py) | 主数据准备脚本（v1.2 优化版） |
| [`verify_unet_offline.py`](sd_scripts/lora/verify_unet_offline.py) | UNet 兼容性离线验证 |
| [`evaluate_metrics.py`](sd_scripts/lora/evaluate_metrics.py) | 评估指标面板（Spill Rate 等） |
| [`visualize_training_data.py`](sd_scripts/lora/visualize_training_data.py) | 可视化工具 |

---

## 使用方法

### 数据准备（v1.2）

```bash
# 准备训练数据（推荐配置）
python sd_scripts/lora/01_prepare_training_data.py \
    --splits train \
    --bg-method random \
    --bg-intensity-range 0.5 1.5 \
    --train-max 3000 \
    --seed 42

# 可视化样本
python sd_scripts/lora/visualize_training_data.py visualize \
    --manifest sd_scripts/lora/training_data/dreambooth_format/manifest_train.csv \
    --output sd_scripts/lora/training_data/v1.2_visual_check \
    --num-samples 20
```

### UNet 兼容性验证

```bash
python sd_scripts/lora/verify_unet_offline.py
```

### 评估指标计算

```bash
python sd_scripts/lora/evaluate_metrics.py \
    --synthetic-dir /path/to/synthetic/images \
    --clean-dir /path/to/clean/images \
    --output-dir /path/to/evaluation/results
```

---

## 评估指标面板（v1.2）

### 1) 背景污染（必须）

**Spill Rate**：
```
spill = Σ(mask外变化) / Σ(总变化)
目标: < 0.05
```

**Background PSNR / SSIM**：
只在 mask 外计算，直接衡量背景是否保持

### 2) 脏污区域真实性（建议至少一种）

- 颜色直方图距离（mask 内）
- 梯度能量对比（纹理特征）
- 亮度统计对比

### 3) 可控性（必须做图）

固定 clean 图 + 固定 mask 的 3×3 网格：
- 列: trans / semi / opaque（class）
- 行: mild / moderate / severe（severity）

---

## Phase 2 准备工作

### 训练配置（v1.2）

| 参数 | 值 | 说明 |
|------|-----|------|
| Base Model | `stabilityai/stable-diffusion-2-1-base` | 文生图模式 |
| Rank | 16 | 提高容量 |
| Alpha | 32 | LoRA 缩放因子 |
| Learning Rate | 1e-4 | 稳定学习率 |
| Batch Size | 2 | 取决于显存 |
| Gradient Accumulation | 4 | 模拟更大 batch |
| Mixed Precision | FP16 | 加速训练 |

### 训练命令模板

```bash
accelerate launch --mixed_precision="fp16" \
    scripts/train_lora_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --instance_data_dir="./sd_scripts/lora/training_data/dreambooth_format/train" \
    --output_dir="./sd_scripts/lora/weights" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=5000 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --seed=42 \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --mixed_precision="fp16" \
    --save_steps=500 \
    --validation_prompt="noticeable opaque heavy stains, on camera lens" \
    --num_validation_images=10
```

---

## 下一步行动

### Phase 2a: 小规模验证（推荐先执行）

1. **准备验证数据集**
   ```bash
   python sd_scripts/lora/01_prepare_training_data.py \
       --splits train \
       --bg-method random \
       --train-max 100
   ```

2. **检查样本质量**
   ```bash
   python sd_scripts/lora/visualize_training_data.py visualize \
       --manifest sd_scripts/lora/training_data/dreambooth_format/manifest_train.csv \
       --output sd_scripts/lora/training_data/phase2a_check \
       --num-samples 20
   ```

3. **确认 caption 格式正确**
   - 类别 token 是否为自然语言
   - 严重度 token 是否为 mild/moderate/severe

### Phase 2b: 完整训练

验证通过后，执行完整训练：
- 训练集：3000 样本（分层采样）
- 验证集：800 样本
- 训练步数：5000
- 每个 checkpoint 生成验证样本

---

## 文件位置总结

```
sd_scripts/lora/
├── 01_prepare_training_data.py      # v1.2 主脚本
├── visualize_training_data.py       # 可视化工具
├── verify_unet_offline.py           # UNet 兼容性验证
├── evaluate_metrics.py              # 评估指标面板
├── docs/
│   ├── lora_training_proposal.md     # v1.0 原始方案
│   ├── lora_training_proposal_v1.1.md  # v1.1 修正方案
│   └── phase1_summary.md            # Phase 1 总结
└── training_data/
    ├── dreambooth_format/           # v1.1 输出（100 样本）
    └── visual_samples/              # 可视化样本
```

---

**文档版本**: v1.2
**创建日期**: 2025-02-10
**状态**: Phase 1 优化完成，Phase 2 就绪
**下一步**: 运行 Phase 2a 小规模验证

**关键改进总结**：
1. Special Token 改为 SD tokenizer 友好的自然语言
2. 背景抑制随机化避免学习"模糊风格"
3. 分层采样确保类别平衡
4. UNet 结构一致性验证通过
5. 完整的评估指标面板（Spill Rate 等）
