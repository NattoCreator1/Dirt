# LoRA 微调 SD Inpainting 方案 (v1.1)

**文档版本**: v1.2
**创建日期**: 2025-02-09
**修订日期**: 2025-02-17
**目的**: 通过 LoRA 让 SD 学习"镜头脏污外观"，再挂载到 inpainting 使用

---

## 修订记录

| 版本 | 日期 | 修订内容 | 修订者 |
|------|------|----------|--------|
| v1.0 | 2025-02-10 | 初始版本（含三元组假设等结构性问题） | Claude |
| v1.1 | 2025-02-10 | **核心重构**：路线A - "脏污外观LoRA + inpainting集成" | Claude |
| v1.2 | 2026-02-17 | **关键修正**：自然语言token + 背景随机化 + 分层采样 + UNet验证 | Claude |

**v1.1 核心修订**：
1. **放弃三元组假设**：不需要 clean/dirty 配对，改用 dirty + mask 直接训练
2. **训练数据重构**：mask 外强降语义，让脏污成为主信号
3. **训练 base 改用 SD2.1 文生图**：推理时再挂到 SD2-inpainting
4. **Prompt 分类化**：类别 token + 程度 token + 光学描述，避免概念塌缩
5. **Rank 上调**：r=8-16（镜头薄层纹理容量需求更高）

---

## 一、方案概述（v1.1 修正版）

### 1.1 核心问题修正

**v1.0 的三处硬伤**：
| 问题 | v1.0 方案 | v1.1 修正 |
|------|----------|------------|
| 数据配对 | clean/dirty/mask 三元组（不成立） | dirty + mask 直接训练，无配对需求 |
| 训练输入 | 未处理 9 通道 inpainting 输入 | 文生图训练（4通道），推理时挂载 LoRA |
| Prompt 概念 | 单一泛化 prompt 导致塌缩 | 分类 token + 光学描述，保持可控 |

### 1.2 技术路线 A：脏污外观 LoRA → Inpainting 集成

```
WoodScape dirty 图像 + mask
         ↓
    构造"脏污主导"训练图（mask外强降语义）
         ↓
    SD2.1 base + LoRA 训练（文生图模式）
         ↓
    学习"镜头脏污外观"（光学纹理）
         ↓
    推理时挂载 LoRA 到 SD2-inpainting
         ↓
    生成真实感脏污（保持 mask 约束）
```

---

## 二、训练数据准备（v1.1）

### 2.1 数据来源

| 数据 | 路径 | 用途 |
|------|------|------|
| WoodScape Train/Val RGB | `dataset/woodscape_raw/train/rgb/` | 训练图源 |
| WoodScape Train/Val GT | `dataset/woodscape_raw/train/gtLabels/` | Mask 来源 |
| 标签索引 | `dataset/woodscape_processed/meta/labels_index_ablation.csv` | 元数据 |

### 2.2 训练图构造：让"脏污"成为主信号

对每张 dirty 图像 $I$ 与 mask $M$，构造训练图 $I_{train}$：

```python
def construct_training_image(
    dirty_image: np.ndarray,  # [H, W, 3] WoodScape RGB
    mask: np.ndarray,           # [H, W] 4类 mask
    target_resolution: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    构造"脏污主导"训练图

    策略：
    1. mask 内：保留原始脏污像素（真实脏污外观）
    2. mask 外：强降语义（模糊/低分辨率/降饱和）
    3. 目标：让 LoRA 学习脏污的光学特征，而非背景语义
    """
    H, W = dirty_image.shape[:2]

    # 1. 识别脏污区域
    mask_bool = (mask > 0)

    # 2. 构造训练图
    I_train = dirty_image.copy().astype(float)

    # 3. mask 外强降语义（三选一或组合）
    if method == "blur":
        # 高斯模糊 + 降采样
        I_bg = cv2.GaussianBlur(I_train, (51, 51), 0)
        I_bg = cv2.resize(I_bg, (W//4, H//4), interpolation=cv2.INTER_AREA)
        I_bg = cv2.resize(I_bg, (W, H), interpolation=cv2.INTER_LINEAR)
    elif method == "desaturate":
        # 降饱和 + 轻微模糊
        I_bg = cv2.cvtColor(I_train, cv2.COLOR_RGB2HSV)
        I_bg[:, :, 1] = I_bg[:, :, 1] * 0.3  # 压低饱和度
        I_bg = cv2.cvtColor(I_bg, cv2.COLOR_HSV2RGB)
        I_bg = cv2.GaussianBlur(I_bg, (15, 15), 0)
    else:  # "downsample"
        # 低分辨率上采样
        I_bg = cv2.resize(I_train, (W//8, H//8), interpolation=cv2.INTER_AREA)
        I_bg = cv2.resize(I_bg, (W, H), interpolation=cv2.INTER_LINEAR)

    # 4. 合并：mask 内保留原图，mask 外使用降语义背景
    I_train[~mask_bool] = I_bg[~mask_bool]

    return I_train.astype(np.uint8)
```

### 2.3 Caption Token 设计（分类 + 程度 + 光学）

**避免概念塌缩的关键**：分类 token + 程度 token + 固定光学描述

```python
# 类别 token
CLASS_TOKENS = {
    1: "<trans>",           # transparent
    2: "<semi>",            # semi-transparent
    3: "<opaque>",          # opaque
}

# 程度 token（从 Severity Score 分桶）
SEVERITY_TOKENS = {
    (0.0, 0.15): "<sev1>",
    (0.15, 0.35): "<sev2>",
    (0.35, 0.60): "<sev3>",
    (0.60, 1.00): "<sev4>",
}

# 固定光学描述（避免学习背景语义）
OPTICAL_DESCRIPTIONS = [
    "on camera lens",
    "out of focus foreground",
    "subtle glare",
    "background visible",
]

def generate_caption(
    mask: np.ndarray,           # [H, W] 4类 mask
    S_full: float,               # Severity Score
) -> str:
    """生成训练 caption"""
    # 主导类别
    unique, counts = np.unique(mask[mask > 0], return_counts=True)
    if len(unique) == 0:
        dominant_class = 0
    else:
        dominant_class = unique[np.argmax(counts)]

    class_token = CLASS_TOKENS.get(dominant_class, "")

    # 程度 token
    for (low, high), sev_token in SEVERITY_TOKENS.items():
        if low <= S_full <= high:
            severity_token = sev_token
            break
    else:
        severity_token = "<sev2>"

    # 光学描述（随机选择）
    optical = np.random.choice(OPTICAL_DESCRIPTIONS)

    # 组合
    caption = f"{class_token} {severity_token} {optical}"
    return caption

# 示例：
# "<opaque> <sev3> on camera lens, out of focus foreground, subtle glare"
# "<trans> <sev1> on camera lens, background visible"
```

---

## 三、LoRA 训练配置（v1.1 修正）

### 3.1 训练参数修正

| 参数 | v1.0 值 | v1.1 值 | 说明 |
|------|---------|---------|------|
| Base Model | `/home/yf/models/sd2_inpaint` | `stabilityai/stable-diffusion-2-1` | 改用文生图 base |
| Rank | 4-8 | **8-16** | 提高容量，先试 16 |
| Alpha | rank × 2 | 32 | LoRA 缩放因子 |
| Learning Rate | 2e-4 | **1e-4** | 更稳的学习率 |
| Training Mode | Inpainting | **Text-to-Image** | 文生图训练 |
| Resolution | 512×512 | 512×512 | 保持 SD 原生分辨率 |

### 3.2 训练脚本（复用 diffusers）

```bash
# 使用 diffusers 官方 LoRA 训练脚本
# 参考：https://huggingface.co/docs/diffusers/training/lora

# 安装依赖
pip install diffusers[torch] accelerate transformers

# 训练命令（示例）
accelerate launch --mixed_precision="fp16" \
    scripts/train_lora_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --instance_data_dir="./lora_training_data" \
    --output_dir="./lora_weights" \
    --resolution=512 \
    --center_crop \
    --random_flip \
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
    --dataloader_num_workers=4 \
    --save_steps=500 \
    --validation_steps=100 \
    --validation_prompt="<opaque> <sev3> on camera lens, out of focus foreground, subtle glare, background visible" \
    --num_validation_images=10 \
    --logging_dir="./logs"
```

### 3.3 数据格式（DreamBooth 格式）

```
lora_training_data/
├── 0001_xxxx.jpg         # 训练图（512x512，脏污主导）
├── 0001_xxxx.txt         # caption: "<opaque> <sev3> on camera lens, ..."
├── 0002_xxxx.jpg
├── 0002_xxxx.txt
└── ...
```

---

## 四、推理集成（v1.1）

### 4.1 推理时加载 LoRA

```python
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

class SDInpaintingWithLoRA:
    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-2-1",
        lora_weights_path: str = "/path/to/lora_weights",
        device: str = "cuda",
    ):
        # 1. 加载 SD2-inpainting pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            f"{base_model_id}-inpainting",
            torch_dtype=torch.float16,
        ).to(device)

        # 2. 加载 LoRA 权重到 UNet
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

### 4.2 与现有流程集成

- 保持 `02_generate_synthetic.py` 的质量控制流程
- 只需修改模型初始化部分
- NPZ 文件格式保持不变

---

## 五、评估指标（v1.1 修正）

### 5.1 新增/替换指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| **Spill Rate** | mask 外像素变化能量 | < 0.05 |
| **分类型可控性** | 固定底图 + 不同类型 prompt 的网格对比 | 视觉上有明显差异 |
| **Mask 内统计匹配** | 颜色/频谱/边缘特征与真实脏污的距离 | 低于阈值 |
| **质量通过率** | 6 项质量指标通过率 | > 20%（baseline） |

### 5.2 保留指标

- 人工定性对比面板（必须）
- 6 项质量控制通过率

### 5.3 弱化指标

- 全图 FID → 改为 mask 区域 FID
- Mask IoU → 仅作为辅助参考

---

## 六、实施计划（v1.1）

### Phase 1: 数据准备（1-2天）
- [ ] 实现训练图构造脚本
- [ ] 生成 caption token 文件
- [ ] 转换为 DreamBooth 格式

### Phase 2: LoRA 训练（2-3天）
- [ ] 配置 diffusers 训练环境
- [ ] Phase 1 小规模验证（100 样本，5 epochs）
- [ ] Phase 2 中等规模（1000 样本，10 epochs）

### Phase 3: 集成验证（1-2天）
- [ ] 修改 `02_generate_synthetic.py` 集成 LoRA
- [ ] 运行对比实验
- [ ] 评估 v1.0 vs v1.1 效果差异

---

## 七、技术栈（v1.1）

```
训练:
├── diffusers (Hugging Face)
├── peft (LoRA 支持)
├── accelerate (分布式训练)
└── transformers

数据处理:
├── opencv-python
├── numpy, pandas
└── Pillow
```

---

## 八、关键修正对比

| 方面 | v1.0 | v1.1 |
|------|------|------|
| **数据假设** | clean/dirty/mask 三元组 | dirty + mask 直接训练 |
| **训练模式** | Inpainting 9 通道 | 文生图 4 通道 |
| **Base 模型** | SD2-inpainting | SD2.1 文生图 |
| **LoRA Rank** | 4-8 | 8-16 |
| **Prompt** | 单一泛化 | 分类 + 程度 token |
| **训练目标** | 不成立的配对映射 | 脏污外观建模 |

---

**文档版本**: v1.2
**最后更新**: 2026-02-17
**状态**: Phase 1.2 完成 - 关键修正已实施并验证
**实施状态**:
- [x] Phase 1: 数据准备脚本实现
- [x] Phase 1.1: Special Token 改为自然语言锚点
- [x] Phase 1.2: 背景抑制强度随机化
- [x] Phase 1.2: 分层采样解决类别不平衡
- [x] Phase 1.2: UNet 结构一致性验证（通过）
- [x] Phase 1.2: 评估指标面板（Spill Rate）
- [ ] Phase 2: LoRA训练环境配置
- [ ] Phase 3: 集成验证

**实施位置**:
- 主脚本: `sd_scripts/lora/01_prepare_training_data.py` (v1.2)
- 可视化: `sd_scripts/lora/visualize_training_data.py`
- UNet验证: `sd_scripts/lora/verify_unet_offline.py`
- 评估指标: `sd_scripts/lora/evaluate_metrics.py`
- 数据输出: `sd_scripts/lora/training_data/dreambooth_format/`
- 总结文档: `sd_scripts/lora/phase1.2_summary.md`

**下一步**: Phase 2a 小规模验证（100样本），检查 caption 格式和样本质量
