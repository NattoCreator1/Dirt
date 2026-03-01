# LoRA 微调 SD Inpainting 方案

**文档版本**: v1.0
**创建日期**: 2025-02-10
**目的**: 通过 LoRA 微调让 SD2-inpainting 学习镜头脏污的真实视觉特征

---

## 一、方案概述

### 1.1 问题分析

**当前 SD2-inpainting 的局限性**：
- 模型从未见过"镜头脏污"这种特定视觉模式
- Prompt 只能描述语义（"dirt"、"smudge"），无法传达光学效果
- 生成的"脏污"是纹理而非真实的散射、模糊、色散效果

**LoRA 解决方案**：
- 在真实脏污数据上微调 SD 模型
- 学习镜头脏污的真实视觉特征（光学效应）
- 保留 SD 的生成多样性，同时增加真实感

### 1.2 技术路线

```
WoodScape 真实数据 → LoRA 训练 → SD2-inpainting + LoRA
     (clean, dirty, mask)        ↓              ↓
                              学习脏污特征      生成真实感脏污
```

---

## 二、训练数据准备

### 2.1 数据来源

| 来源 | 内容 | 路径 |
|------|------|------|
| WoodScape Train | RGB 图像 + GT Labels | `dataset/woodscape_raw/train/` |
| WoodScape Val | RGB 图像 + GT Labels | `dataset/woodscape_raw/train/` (val 从 train 划分) |

**数据隔离**：
- 训练集：仅使用 WoodScape Train + Val
- 测试集：WoodScape Test 完全隔离，用于最终评估

### 2.2 数据格式

每个训练样本需要三元组：
```
{
    "clean_image": RGB 图像 (640x480),    # 干净底图或合成
    "dirty_image": RGB 图像 (640x480),    # 真实脏污图像
    "mask": 4类 mask (640x480),            # 0=clean, 1=trans, 2=semi, 3=opaque
    "file_id": 唯一标识符
}
```

### 2.3 数据预处理

```python
# 数据预处理脚本
# sd_scripts/lora/01_prepare_training_data.py

def prepare_training_samples(
    woodscape_rgb_dir: Path,
    woodscape_gt_dir: Path,
    output_dir: Path,
    target_resolution: Tuple[int, int] = (512, 512),  # SD 原生分辨率
):
    """
    准备 LoRA 训练数据

    处理流程:
    1. 读取 WoodScape RGB 图像
    2. 读取对应的 GT Labels (4类 mask)
    3. Resize 到 512x512 (SD 原生分辨率)
    4. 生成训练三元组 (clean, dirty, mask)
    5. 保存为训练格式
    """
    # TODO: 实现
```

### 2.4 数据增强

为增加训练数据多样性：
- **几何增强**: 随机裁剪、旋转、翻转
- **颜色增强**: 轻微亮度/对比度调整
- **Mask 增强**: 轻微腐蚀/膨胀

---

## 三、LoRA 训练配置

### 3.1 LoRA 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Rank | 4-8 | LoRA 秩，控制表达能力 |
| Alpha | 8-16 | LoRA 缩放因子 |
| Target Modules | ["UNet"] | 微调目标模块 |
| Dropout | 0.1 | 防止过拟合 |

### 3.2 训练超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Batch Size | 1-4 | 取决于显存 |
| Learning Rate | 1e-4 - 5e-4 | AdamW 优化器 |
| Epochs | 10-20 | 根据收敛情况调整 |
| Warmup Steps | 500 | 学习率预热 |
| Gradient Accumulation | 4-8 | 模拟更大 batch |
| Mixed Precision | FP16 | 加速训练 |

### 3.3 训练 Prompt

训练时使用简单的通用 prompt：
```
"dirty camera lens with soiling"
```

测试时使用类别特定的 prompt（保持现有设计）。

### 3.4 训练脚本框架

```python
# sd_scripts/lora/02_train_lora.py

import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import LoraConfig, get_peft_model

def train_lora(
    train_data_dir: Path,
    output_dir: Path,
    base_model_id: str = "/home/yf/models/sd2_inpaint",
    rank: int = 8,
    epochs: int = 15,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
):
    """
    LoRA 训练主函数
    """
    # 1. 加载基础模型
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
    )

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        inference_mode=False,
    )

    # 3. 注入 LoRA
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    # 4. 训练循环
    # TODO: 实现训练逻辑

    # 5. 保存 LoRA 权重
    pipeline.unet.save_pretrained(output_dir / "lora_weights")
```

---

## 四、验证方案

### 4.1 验证阶段

| 阶段 | 规模 | 目的 |
|------|------|------|
| **Phase 1: 小规模验证** | 100 样本, 5 epochs | 验证方案可行性 |
| **Phase 2: 中等规模** | 1000 样本, 10 epochs | 优化超参数 |
| **Phase 3: 完整训练** | 全量数据, 15 epochs | 最终模型 |

### 4.2 评估指标

**定量指标**：
1. **视觉质量**：生成样本的视觉真实感
2. **质量控制通过率**：使用现有 6 项质量指标
3. **FID Score**：与真实脏污数据的分布距离
4. **Mask IoU**：生成区域与 mask 的一致性

**定性评估**：
- 人工视觉检查
- 与原始 SD2-inpainting 对比

### 4.3 对比实验

| 方法 | 描述 | 预期效果 |
|------|------|----------|
| Baseline | 原始 SD2-inpainting | 差（已验证） |
| +Prompt Engineering | 优化 prompt | 中等改善 |
| +LoRA (本方案) | LoRA 微调 | 显著改善 |

---

## 五、集成到生成流程

### 5.1 推理时加载 LoRA

```python
# sd_scripts/generation/02_generate_synthetic_with_lora.py

from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

class SDInpaintingSoilingGeneratorWithLoRA:
    def __init__(
        self,
        base_model_id: str = "/home/yf/models/sd2_inpaint",
        lora_weights_path: str = "/path/to/lora_weights",
        device: str = "cuda",
    ):
        # 加载基础模型
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
        ).to(device)

        # 加载 LoRA 权重
        self.pipeline.unet = PeftModel.from_pretrained(
            self.pipeline.unet,
            lora_weights_path
        )

    def generate(self, clean_image, mask, prompt, **kwargs):
        # 保持现有接口不变
        return self.pipeline(
            prompt=prompt,
            image=clean_image,
            mask_image=mask,
            **kwargs
        )
```

### 5.2 兼容性

- 保持现有 `02_generate_synthetic.py` 的质量控制流程
- NPZ 文件格式保持不变
- Manifest 记录 LoRA 版本信息

---

## 六、成本与时间评估

### 6.1 计算资源

| 资源 | 需求 | 备注 |
|------|------|------|
| GPU | 推荐 24GB+ 显存 | RTX 4090 / A100 |
| 存储 | ~50GB | 模型 + 数据 + 输出 |
| 内存 | 32GB+ | 训练时加载大批量数据 |

### 6.2 时间估算

| 阶段 | 数据量 | 训练时间 |
|------|--------|----------|
| Phase 1 验证 | 100 样本 | ~30 分钟 |
| Phase 2 中等 | 1000 样本 | ~3 小时 |
| Phase 3 完整 | 5000+ 样本 | ~15 小时 |

### 6.3 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 过拟合 | 中 | 中 | 数据增强、早停、Dropout |
| 训练不收敛 | 低 | 高 | 学习率调整、warmup |
| 质量未提升 | 中 | 高 | Phase 1 快速验证 |

---

## 七、实施计划

### 7.1 里程碑

```
Week 1: 数据准备 + Phase 1 验证
├── 准备训练数据 (100 样本)
├── 配置训练环境
└── 运行 Phase 1 训练 + 评估

Week 2: Phase 2 优化 + 完整训练
├── 扩展训练数据 (1000 样本)
├── 超参数优化
└── 完整训练

Week 3: 集成 + 评估
├── 集成到生成流程
├── 大规模生成测试
└── 文档整理
```

### 7.2 下一步行动

1. **确认方案可行性**：技术评审
2. **准备训练数据**：实现 `01_prepare_training_data.py`
3. **配置训练环境**：安装依赖、测试脚本
4. **Phase 1 验证**：快速验证 LoRA 是否有效

---

## 八、技术栈

```
训练框架:
├── diffusers (Hugging Face)
├── peft (LoRA 支持)
├── transformers
└── accelerate (分布式训练)

数据处理:
├── opencv-python
├── numpy, pandas
└── Pillow
```

---

## 九、附录：LoRA 原理简介

### 9.1 什么是 LoRA

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法：
- 冻结预训练模型权重
- 在特定层添加低秩分解矩阵
- 只训练新增参数（通常 <1% 原模型大小）

### 9.2 为什么选择 LoRA

| 优势 | 说明 |
|------|------|
| 参数高效 | 只训练 ~0.5% 参数 |
| 内存友好 | 可在单卡上训练 |
| 可插拔 | 推理时动态加载/卸载 |
| 保留知识 | 保持 SD 原能力 |

---

**文档版本**: v1.0
**最后更新**: 2025-02-10
**状态**: 待评审
