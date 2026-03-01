# SD合成数据生成指南

## 实验结论（2026-02-18）

经过系统测试，确认SD2.1 Inpainting在当前设置下的能力边界：

### ✅ 可以控制的维度
1. **严重程度** (Severity): mild < noticeable < severe
2. **覆盖率** (Coverage): 通过mask大小控制
3. **位置** (Position): 通过mask位置控制
4. **脏污外观** (Appearance): LoRA学到的基础镜头脏污特征

### ❌ 无法有效控制的维度
1. **物质类型** (Material): water/grease/mud无区别
2. **细微纹理差异**: C1/C2/C3之间区分度有限

---

## 生成配置建议

### 推荐配置
```python
# 模型配置
model = "SD2.1 Inpainting"
lora_path = "checkpoint-3000"  # 继续使用LoRA，提供基础脏污外观

# 生成参数
strength = 0.45-0.50          # 提高strength以获得更明显的效果
guidance_scale = 6.0-7.0      # 保持适中的guidance
num_inference_steps = 30      # 标准步数

# Prompt系统
caption_templates = {
    "C1_mild": "mild transparent soiling layer, on camera lens, out of focus foreground",
    "C1_noticeable": "noticeable transparent soiling layer, on camera lens, out of focus foreground",
    "C1_severe": "severe transparent soiling layer, on camera lens, out of focus foreground",
    # ... 类似地定义C2, C3
}

# Mask策略
- 使用完整MaskBank (3200 masks)
- 确保覆盖率的多样性 (low/medium/high)
- 确保位置的多样性
```

### 生成策略
1. **单类别策略**: 每个样本只含一个脏污类别 (C1/C2/C3)
2. **类别平衡**: 按stratified sampling确保1:1:1
3. **覆盖率分层**: 确保不同覆盖率的样本均匀分布
4. **严重度分层**: mild/noticeable/severe各占1/3

---

## 关键理解

**当前二阶段SD方案的核心价值**：
- 不是"完美还原真实脏污"
- 而是**提供多样化的训练数据增强**
- 帮助模型学习"镜头脏污"的**基础概念**

**质量标准**：
- ✅ 背景不被污染 (spill_rate < 0.3)
- ✅ 脏污区域有合理的失焦效果
- ✅ 不同严重度有视觉差异
- ⚠️ 物质类型不区分 (接受现实)

---

## 下一步行动

1. **批量生成Train_Syn数据集**
   - 目标: ~10,000 合成样本
   - 使用推荐配置
   - 记录详细manifest

2. **质量筛选**
   - 使用spill_rate < 0.3 作为硬约束
   - 可选: SSIM检查

3. **模型训练**
   - G1: Baseline (Train_Real only)
   - G2: Real + Syn (筛选后)
   - 对比External Test ρ

4. **评估决策**
   - 如果External ρ提升 > 1.5% → 成功
   - 如果External ρ提升 < 1.0% → 考虑其他改进方向

---

## 实验决策记录（2026-02-18 晚）

### Caption修复实验 - 结论：回滚

**实验过程**：
1. 发现WoodScape的dominant_class标注与实际像素分布不一致
2. 编写`fix_captions.py`基于实际像素比例重新计算dominant_class
3. 修复后分布：C1=0, C2=839, C3=2361
4. 生成审阅报告供用户检查

**用户结论**：
- 修复后的captions反而**更不一致**于图像实际内容
- 决定**信任WoodScape原始标注**
- 回滚所有caption修改

**原因分析**：
- WoodScape的标注基于"视觉主导性"而非"像素占比"
- 例如：少数几个明显水滴可能比大面积轻微污渍更"主导"
- 这是符合人眼感知的标注逻辑

### 新实验方案：双轨并行

**方案2.1：基于当前LoRA的全量生成 + 人工筛选**
```
步骤：
1. 使用当前checkpoint-3000 LoRA
2. 对所有clean帧 + mask生成脏污图像
3. 人工筛选出"拟真"的合成样本
4. 筛选后样本用作训练数据增强

优势：快速获得大量合成数据
风险：需要大量人工筛选工作
```

**方案2.2：筛选WoodScape典型样本 + 重训LoRA**
```
步骤：
1. 从WoodScape人工筛选"非常典型、面积大"的脏污样本
2. 用筛选子集重新训练LoRA
3. 用新LoRA生成更高质量的合成数据

筛选标准：
- 脏污非常明显（不受背景纹理影响）
- 占据面积大（>30%）
- 类型特征清晰（水滴/油渍/泥污）
```

### 执行计划

| 阶段 | 任务 | 优先级 | 预计产出 |
|------|------|--------|----------|
| A-1 | 编写批量生成脚本 | 高 | 全量合成数据脚本 |
| A-2 | 执行全量生成 | 高 | ~10,000 合成样本 |
| A-3 | 人工筛选合成数据 | 中 | ~1,000-3,000 高质量样本 |
| B-1 | 设计WoodScape筛选工具 | 中 | 可视化筛选界面 |
| B-2 | 筛选典型样本 | 中 | ~500-1,000 典型样本 |
| B-3 | 重新训练LoRA | 低 | 新LoRA checkpoint |
| C-1 | 对比实验 | 低 | 实验对比报告 |

### 实施状态

- [x] 回滚caption修改
- [x] 恢复原始captions和manifest
- [x] 编写批量生成脚本
- [x] 执行全量生成 (8960张图像)
- [ ] 人工筛选合成数据

---

## 批量生成超参数配置（batch_896f_10masks）

### 生成参数详解

| 参数 | 当前值 | 出发点 | 优势 | 劣势 | 影响 |
|------|--------|--------|------|------|------|
| **strength** | 0.45-0.50 | 平衡效果强度与背景保留 | 能生成明显的脏污效果 | 导致较高的spill_rate (均值0.26) | 直接影响生成图像与原图的差异程度；值越大，mask外区域被修改的风险越高 |
| **guidance_scale** | 6.0-7.0 | 标准SD推荐范围 | 对prompt遵循度适中 | 可能限制生成多样性 | 控制模型对caption的依从程度；过高可能导致过度风格化 |
| **num_inference_steps** | 30 | 质量/速度平衡 | 良好的生成质量 | 推理时间较长 | 影响生成细节质量；更多步数通常更好但有边际递减 |
| **mask_blur_radius** | 3 | 软化mask边缘 | 自然过渡，减少硬边 | 增加spill-over风险 | 高斯模糊sigma值；越大边缘越软，但spill范围越大 |
| **mask_dilate_iter** | 1 | 轻微扩展mask | 确保脏污覆盖完整 | 进一步增加spill | 形态学膨胀迭代次数；扩展mask白色区域 |
| **resolution** | 512×512 | SD2.1原生分辨率 | 最佳模型性能 | 需resize到640×480用于训练 | 影响生成质量和后续处理 |

### 质量控制指标

| 指标 | 定义 | 当前批次统计 | 筛选阈值建议 |
|------|------|-------------|-------------|
| **spill_rate** | mask外像素变化占比 | μ=0.26, σ=0.31, max=0.998 | < 0.3 (严格), < 0.5 (宽松) |
| **mask_coverage** | mask区域占比 | 多样化分布 | > 0.05 (避免过小) |
| **dominant_class** | WoodScape类别标签 | C1/C2/C3分布 | 基于原始标注 |

### Spill Rate 分析

**当前配置问题**：
- 约25%的样本spill_rate > 0.5，意味着背景被显著修改
- 这违反了inpainting的核心假设：只在mask区域生成内容

**根本原因**：
1. `strength=0.45-0.50` 过高，导致整个latent空间被大幅修改
2. `blur_radius=3` 造成软边，spill扩展到mask外
3. SD Inpainting在高strength时对mask约束力减弱

**优化建议**：
```
当前配置（问题配置）:
  strength = [0.45, 0.50]
  blur_radius = 3
  → spill_rate均值 ≈ 0.26

建议配置（降低spill）:
  strength = [0.30, 0.40]  # 降低20-30%
  blur_radius = 1          # 减少边缘模糊
  → 预期spill_rate均值 < 0.15
```

### Caption设计

**当前模板**（基于WoodScape dominant_class）：
```python
CAPTION_TEMPLATES = {
    1: {  # Transparent soiling
        "noticeable": "noticeable transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",
    },
    2: {  # Semi-transparent smudge
        "noticeable": "noticeable semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",
    },
    3: {  # Opaque stains
        "noticeable": "noticeable opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",
    },
}
```

**关键元素**：
- `"on camera lens"` - 明确位置
- `"out of focus foreground"` - 引导失焦效果
- `"subtle glare"` - 增加真实感
- `"background visible"` - 强调透明度

**Negative Prompt**：
```
"blurry, low quality, distorted, watermark, text, error, corrupted, ugly, camera, lens, sharp focus, crystal clear"
```
- 避免"sharp focus"确保失焦效果
- 避免"crystal clear"确保有脏污感

### 数据生成策略

**当前批次**（batch_896f_10masks）：
- 896张clean帧 × 10个随机mask = 8960张图像
- 每张clean图像随机采样10个不同mask
- 确保mask多样性（类别、位置、覆盖率）

**筛选策略**：
1. **硬约束**：spill_rate < 0.3（背景变化<30%）
2. **软约束**：视觉检查脏污自然度
3. **预期筛选率**：~1.7-2.2%（约150-200张高质量样本）

### 论文撰写要点

**方法描述**：
```
We employ Stable Diffusion 2.1 Inpainting [cite] with a custom LoRA adapter
trained on WoodScape dataset to generate synthetic soiling patterns. The
generation process uses:
- Input: clean camera frames + binary soiling masks
- Strength: 0.45-0.50 (controls modification intensity)
- Guidance scale: 6.0-7.0 (prompt adherence)
- Inference steps: 30 (quality/efficiency balance)
- Resolution: 512×512 (SD native), resized to 640×480 for training

The mask is pre-processed with Gaussian blur (σ=3) and binary dilation (1 iteration)
to create smooth edges. A quality control metric (spill_rate) measures the
proportion of pixel changes outside the mask region, with samples exceeding
0.3 being filtered out to ensure background preservation.
```

**消融实验建议**：
| 实验 | strength | blur_radius | 预期效果 |
|------|----------|-------------|----------|
| A1 | 0.45-0.50 | 3 | 当前配置（高spill） |
| A2 | 0.30-0.40 | 3 | 降低spill，但脏污效果弱 |
| A3 | 0.30-0.40 | 1 | 最低spill，但硬边不自然 |
| A4 | 0.35-0.45 | 2 | 平衡配置 |

### 参数权衡分析

**降低 Strength 的负面影响**：
- strength=0.30-0.40 → 脏污效果变弱，透明度增加，特征不清晰
- 权衡：牺牲效果强度换取背景保留

**降低 blur_radius 的负面影响**：
- blur_radius=1 → 硬边效应，融合度差，真实感下降
- 权衡：牺牲自然过渡换取精确边界

### 后处理修复方案（待尝试）

**核心思想**：生成后用原始图像强制替换mask外区域，完全消除spill

```python
def force_background_preservation(generated, original, mask):
    """强制保留mask外区域（后处理修复）

    原理：
    - 使用高strength生成明显的脏污效果
    - 生成后用原始图像替换mask外区域
    - 保留mask内的软边缘过渡效果

    Args:
        generated: SD生成的图像 (PIL.Image)
        original: 原始干净帧 (PIL.Image)
        mask: 原始binary mask (numpy array, 0=outside, 1=inside)

    Returns:
        修复后的图像 (PIL.Image)
    """
    import numpy as np
    from PIL import Image

    # 转换为numpy
    gen_np = np.array(generated).astype(float)
    orig_np = np.array(original).astype(float)

    # 创建3通道mask
    if mask.ndim == 2:
        mask_3ch = np.stack([mask] * 3, axis=-1)
    else:
        mask_3ch = mask

    # 确保mask是0-1范围
    if mask_3ch.max() > 1:
        mask_3ch = mask_3ch / 255.0

    # 混合：mask内用生成图，mask外用原图
    result_np = gen_np * mask_3ch + orig_np * (1 - mask_3ch)

    return Image.fromarray(result_np.astype(np.uint8))


# 在generate_single_sample中添加
def generate_single_sample_with_postfix(self, ...):
    """生成样本 + 后处理修复"""
    # ... 原有生成代码 ...
    result_image = result.images[0]

    # 后处理：强制保留背景
    if self.args.force_background:
        result_image = force_background_preservation(
            result_image,
            clean_image,
            mask_array  # 使用原始binary mask
        )

    # 重新计算spill_rate（应该接近0）
    metrics = compute_quality_metrics(clean_image, result_image, mask_array, clean_image)
```

**添加命令行参数**：
```python
parser.add_argument("--force_background", action="store_true",
                   help="后处理：强制替换mask外区域为原始图像")
```

**方案优势**：
| 优势 | 说明 |
|------|------|
| ✅ 完全消除spill | spill_rate理论上为0 |
| ✅ 保持高strength | 脏污效果明显 |
| ✅ 保持软边缘 | blur_radius=3的自然过渡 |
| ✅ 实现简单 | 只需几行代码 |

**方案劣势**：
| 劣势 | 说明 |
|------|------|
| ⚠️ 硬边界 | mask边缘可能仍有轻微接缝 |
| ⚠️ 额外步骤 | 需要后处理计算 |

**实验验证步骤**：
1. 对当前batch_896f_10masks应用后处理
2. 检查修复后图像的视觉效果
3. 对比spill_rate变化
4. 如果效果好，下次生成直接启用

---

## Tile级覆盖率标注生成

### 关键发现

**问题**：当前批量生成脚本仅生成合成图像，未生成训练所需的tile级标注（tile_cov）。

**影响**：缺少tile_cov标注会导致训练时抛出异常：
```python
RuntimeError: 'tile_cov' not in npz
```

**解决方案**：后处理阶段从mask生成tile_cov标注。每个合成图像在manifest中都有对应的mask_file_id，可基于mask计算tile级覆盖率。

### Tile标注生成算法

**核心思想**：将512×512的mask分割为8×8网格，每个tile大小64×64，统计每个tile内各类别的覆盖率。

**输入**：
- `mask`: (512, 512) numpy array, 值为0/1/2/3表示4个类别
  - 0: clean
  - 1: transparent soiling (C1)
  - 2: semi-transparent smudge (C2)
  - 3: opaque stains (C3)

**输出**：
- `tile_cov`: (8, 8, 4) numpy array, 每个tile对4个类别的覆盖率

**算法步骤**：
```python
def generate_tile_cov_from_mask(mask, tile_size=64):
    """从mask生成tile级覆盖率标注

    Args:
        mask: (H, W) numpy array, 值为0/1/2/3表示类别
        tile_size: tile大小，默认64 (512/8=64)

    Returns:
        tile_cov: (8, 8, 4) numpy array, float32, 范围[0,1]
    """
    H, W = mask.shape
    assert H == W == 512, f"Mask size must be 512x512, got {H}x{W}"

    n_tiles = H // tile_size  # 8
    tile_cov = np.zeros((n_tiles, n_tiles, 4), dtype=np.float32)

    for i in range(n_tiles):
        for j in range(n_tiles):
            # 提取当前tile的mask区域
            tile_mask = mask[i*tile_size:(i+1)*tile_size,
                           j*tile_size:(j+1)*tile_size]

            # 计算tile内各类别的像素占比
            total_pixels = tile_mask.size

            for c in range(4):
                count_c = (tile_mask == c).sum()
                tile_cov[i, j, c] = count_c / total_pixels

    return tile_cov
```

### 完整后处理流程

**输入数据**：
1. `filtered.csv`: 筛选工具输出的CSV，包含accepted图像
2. `manifest.csv`: 生成时的manifest，包含mask_file_id映射
3. `MaskBank/`: 原始mask文件目录
4. `Train_Syn/raw/images/`: 合成图像目录

**输出数据**：
1. `Train_Syn/labels_tile/*.npz`: 每个图像的tile标注文件
2. `Train_Syn/index.csv`: 训练索引CSV

**流程图**：
```
filtered.csv (accepted图像)
    ↓
读取manifest获取mask_file_id
    ↓
加载对应mask (512×512, 类别标注)
    ↓
生成tile_cov (8×8×4)
    ↓
计算global_score (S_full)
    ↓
保存.npz文件
    ↓
生成训练索引CSV
```

### NPZ文件格式规范

**必须包含的字段**：
```python
{
    'tile_cov': (8, 8, 4) float32,  # tile级4类别覆盖率
    'S_full': float,                  # global severity score
    'mask_file_id': str,              # 原始mask ID（用于追溯）
    'dominant_class': int,            # 主导类别 (1/2/3)
    'mask_coverage': float,           # mask覆盖率
    'spill_rate': float,              # 生成时的spill_rate
    'caption': str,                   # 生成时的caption
}
```

**示例代码**：
```python
def save_training_npz(output_path, tile_cov, metadata):
    """保存训练格式的.npz文件

    Args:
        output_path: 输出.npz文件路径
        tile_cov: (8,8,4) tile级覆盖率
        metadata: dict包含其他元数据
    """
    np.savez_compressed(
        output_path,
        tile_cov=tile_cov.astype(np.float32),
        S_full=float(metadata.get('S_full', tile_cov[:,:,1:].mean())),
        mask_file_id=str(metadata.get('mask_file_id', '')),
        dominant_class=int(metadata.get('dominant_class', 1)),
        mask_coverage=float(metadata.get('mask_coverage', 0.0)),
        spill_rate=float(metadata.get('spill_rate', 0.0)),
        caption=str(metadata.get('caption', '')),
    )
```

### Global Score计算

**定义**：`S_full`表示图像整体脏污严重度，基于tile_cov加权平均。

**计算公式**：
```python
def compute_global_score(tile_cov, method='weighted_mean'):
    """计算global severity score

    Args:
        tile_cov: (8,8,4) numpy array
        method: 计算方法
            - 'weighted_mean': 加权平均 (权重: w=[0, 0.33, 0.66, 1.0])
            - 'mean': 直接平均
            - 'max': 最大值

    Returns:
        S_full: float, 范围[0,1]
    """
    if method == 'weighted_mean':
        w = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
        # 每个tile的加权分数
        tile_scores = (tile_cov * w).sum(axis=-1)  # (8,8)
        # 全局平均
        S_full = tile_scores.mean()
    elif method == 'mean':
        # 排除clean类别的平均覆盖率
        S_full = tile_cov[:, :, 1:].mean()
    elif method == 'max':
        tile_scores = tile_cov[:, :, 1:].sum(axis=-1)  # 所有脏污类别覆盖率
        S_full = tile_scores.max()
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(np.clip(S_full, 0.0, 1.0))
```

### 完整后处理脚本框架

```python
#!/usr/bin/env python3
"""
后处理筛选后的合成图像，生成训练格式的标注文件

Usage:
    python postprocess_synthetic_data.py \
        --filtered_csv results/filtered_results.csv \
        --manifest_csv batch_896f_10masks/manifest.csv \
        --mask_bank data/MaskBank \
        --image_dir batch_896f_10masks/images \
        --output_dir Train_Syn \
        --force_background
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil


def load_mask(mask_bank_dir, mask_file_id):
    """加载mask文件

    Args:
        mask_bank_dir: MaskBank目录
        mask_file_id: mask文件ID (如 'mask_00012')

    Returns:
        mask: (512, 512) numpy array, 值为0/1/2/3
    """
    # 假设mask以PNG格式存储，灰度值表示类别
    mask_path = Path(mask_bank_dir) / f"{mask_file_id}.png"
    mask = Image.open(mask_path)
    mask_np = np.array(mask)

    # 确保是类别标注
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    return mask_np


def postprocess_single_sample(row, mask_bank_dir, image_dir, output_dir,
                              force_background=False):
    """处理单个样本

    Args:
        row: CSV行数据
        mask_bank_dir: MaskBank目录
        image_dir: 合成图像目录
        output_dir: 输出目录
        force_background: 是否强制保留背景

    Returns:
        success: bool
    """
    try:
        # 1. 读取合成图像
        img_path = Path(image_dir) / row['output_filename']
        if not img_path.exists():
            print(f"⚠️ 图像不存在: {img_path}")
            return False

        # 2. 读取mask
        mask_file_id = row['mask_file_id']
        mask = load_mask(mask_bank_dir, mask_file_id)

        # 3. 生成tile_cov
        tile_cov = generate_tile_cov_from_mask(mask)

        # 4. 计算global_score
        S_full = compute_global_score(tile_cov, method='weighted_mean')

        # 5. (可选) 强制保留背景
        if force_background and 'original_image_path' in row:
            # 需要manifest中保存原图路径
            original = Image.open(row['original_image_path'])
            generated = Image.open(img_path)
            result = force_background_preservation(generated, original, mask)
            # 保存修复后的图像
            result.save(img_path)

        # 6. 保存.npz
        stem = Path(row['output_filename']).stem
        npz_path = Path(output_dir) / 'labels_tile' / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'S_full': S_full,
            'mask_file_id': mask_file_id,
            'dominant_class': row.get('mask_dominant_class', 1),
            'mask_coverage': row.get('mask_coverage', 0.0),
            'spill_rate': row.get('spill_rate', 0.0),
            'caption': row.get('caption', ''),
        }
        save_training_npz(npz_path, tile_cov, metadata)

        # 7. 复制图像到输出目录
        output_img_path = Path(output_dir) / 'images' / row['output_filename']
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, output_img_path)

        return True

    except Exception as e:
        print(f"❌ 处理失败 {row['output_filename']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="后处理合成数据生成训练格式")
    parser.add_argument("--filtered_csv", required=True, help="筛选结果CSV")
    parser.add_argument("--manifest_csv", required=True, help="生成manifest CSV")
    parser.add_argument("--mask_bank", required=True, help="MaskBank目录")
    parser.add_argument("--image_dir", required=True, help="合成图像目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--force_background", action="store_true",
                       help="强制保留mask外背景区域")
    parser.add_argument("--status", default="accepted",
                       choices=["accepted", "rejected"],
                       help="处理哪种状态的图像")

    args = parser.parse_args()

    # 读取筛选结果
    df = pd.read_csv(args.filtered_csv)
    df_filtered = df[df['filter_status'] == args.status].copy()

    print(f"处理 {len(df_filtered)} 张图像...")

    # 读取manifest获取额外信息
    manifest = pd.read_csv(args.manifest_csv)
    manifest_dict = manifest.set_index('output_filename').to_dict('index')

    # 合并manifest信息
    for idx, row in df_filtered.iterrows():
        filename = row['output_filename']
        if filename in manifest_dict:
            for k, v in manifest_dict[filename].items():
                if k not in row:
                    df_filtered.at[idx, k] = v

    # 处理每个样本
    success_count = 0
    for _, row in df_filtered.iterrows():
        if postprocess_single_sample(row, args.mask_bank, args.image_dir,
                                     args.output_dir, args.force_background):
            success_count += 1

    print(f"\n完成! 成功处理: {success_count}/{len(df_filtered)}")

    # 生成训练索引CSV
    generate_training_index(args.output_dir)


def generate_training_index(output_dir):
    """生成训练索引CSV"""
    labels_dir = Path(output_dir) / 'labels_tile'
    images_dir = Path(output_dir) / 'images'

    index_data = []
    for npz_path in labels_dir.glob('*.npz'):
        stem = npz_path.stem
        img_path = images_dir / f"{stem}.png"

        if img_path.exists():
            index_data.append({
                'rgb_path': str(img_path),
                'tile_npz': str(npz_path),
                'split': 'train',  # 默认为训练集
                'domain': 'synthetic',
            })

    df_index = pd.DataFrame(index_data)
    index_path = Path(output_dir) / 'index.csv'
    df_index.to_csv(index_path, index=False)
    print(f"训练索引已保存: {index_path}")


if __name__ == "__main__":
    main()
```

### 后处理执行步骤

**阶段1：准备**
```bash
# 确保筛选完成
cd /home/yf/soiling_project/sd_scripts/lora

# 检查筛选结果
python -c "import pandas as pd; df=pd.read_csv('results/filtered_results.csv'); print(df['filter_status'].value_counts())"
```

**阶段2：执行后处理**
```bash
# 处理accepted图像
python postprocess_synthetic_data.py \
    --filtered_csv results/filtered_results.csv \
    --manifest_csv batch_896f_10masks/manifest.csv \
    --mask_bank ../../data/MaskBank \
    --image_dir batch_896f_10masks/images \
    --output_dir ../../baseline/datasets/Train_Syn \
    --status accepted

# 可选：应用背景修复
python postprocess_synthetic_data.py \
    ... (同上) \
    --force_background
```

**阶段3：验证输出**
```bash
# 检查.npz文件格式
python -c "
import numpy as np
p = np.load('Train_Syn/labels_tile/sample.npz')
print('Keys:', list(p.keys()))
print('tile_cov shape:', p['tile_cov'].shape)
print('S_full:', p['S_full'])
"

# 检查训练索引
python -c "
import pandas as pd
df = pd.read_csv('Train_Syn/index.csv')
print(f'Total samples: {len(df)}')
print(df.head())
"
```

### 数据质量检查

**必要检查**：
1. ✅ tile_cov shape正确 (8, 8, 4)
2. ✅ 每个类别覆盖率范围[0, 1]
3. ✅ 每个tile的4类别和≤1.0
4. ✅ S_full范围[0, 1]
5. ✅ 图像文件和npz文件一一对应

**统计指标**：
```python
def analyze_tile_statistics(labels_dir):
    """分析tile标注统计"""
    tile_covs = []
    global_scores = []

    for npz_path in Path(labels_dir).glob('*.npz'):
        data = np.load(npz_path)
        tile_covs.append(data['tile_cov'])
        global_scores.append(data['S_full'])

    tile_covs = np.stack(tile_covs)  # (N, 8, 8, 4)

    stats = {
        'n_samples': len(tile_covs),
        'mean_global_score': np.mean(global_scores),
        'mean_coverage_per_class': tile_covs.mean(axis=(0, 1, 2)),  # (4,)
        'spatial_distribution': tile_covs.mean(axis=0),  # (8, 8, 4) 空间分布
    }

    return stats
```

### 集成到训练流程

**配置WoodscapeIndexSpec**：
```python
from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset

# 合成数据集配置
syn_spec = WoodscapeIndexSpec(
    index_csv="datasets/Train_Syn/index.csv",
    img_root="datasets/Train_Syn/images",
    labels_tile_dir="datasets/Train_Syn/labels_tile",
    split_col=None,  # 合成数据全部用于训练
    split_value=None,
    global_target="S_full",
    resize_w=640,
    resize_h=480,
)

syn_dataset = WoodscapeTileDataset(syn_spec)
print(f"合成数据集大小: {len(syn_dataset)}")
```

**数据增强配置**：
```python
# 合并真实+合成数据
from torch.utils.data import ConcatDataset

real_dataset = WoodscapeTileDataset(real_spec)
syn_dataset = WoodscapeTileDataset(syn_spec)

combined_dataset = ConcatDataset([real_dataset, syn_dataset])
print(f"合并数据集大小: {len(combined_dataset)}")
```

---
