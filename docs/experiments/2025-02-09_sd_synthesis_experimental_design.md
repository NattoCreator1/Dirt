# SD合成数据实验设计方案 (v4.0)

**设计日期**: 2025-02-09
**更新日期**: 2026-02-18
**版本**: v4.0 (两阶段LoRA + SD Inpainting 方案)
**实验类型**: 数据增强 - 合成脏污数据生成与验证
**目的**: 基于mask和干净帧图像生成合成脏污数据，验证其对模型性能的提升效果

**方案说明**: 本文档描述的是**两阶段LoRA方案**：
1. 训练阶段：SD2.1 Base + LoRA 学习"镜头脏污外观材质"
2. 推理阶段：SD2.1 Inpainting + LoRA 在mask约束下生成脏污效果

---

## 文档修订记录

| 版本 | 日期 | 修订内容 | 修订者 |
|------|------|----------|--------|
| v1.0 | 2025-02-09 | 初始版本 | Claude |
| v2.0 | 2025-02-09 | 根据反馈第一次修订 | Claude |
| v3.0 | 2025-02-09 | 根据核查结果第二次修订 | Claude |
| v3.1 | 2025-02-09 | 同步静态 SD Inpainting 方案实现 | Claude |
| v4.0 | 2026-02-18 | 整合两阶段LoRA方案，更正clean数据集路径 | Claude |

**v4.0主要修订**:
1. 新增第三章：两阶段LoRA方案（SD2.1 Base训练 + SD2.1 Inpainting推理）
2. 更新Clean数据集描述：使用 `dataset/my_clean_frames_4by3/` (5,258 samples)
3. 明确LoRA技术验证结果（128 keys成功加载到Inpainting模型）
4. 补充Caption控制、类别不平衡处理、证据面板设计
5. 与论文实验主线的衔接说明

**v3.1主要修订**:
1. 同步静态 SD Inpainting 生成器实现（`02_generate_synthetic.py`）
2. 更新单类别 mask 策略说明
3. 更新类别采样比例配置
4. 更新版本绑定机制实现
5. 更新质量控制指标体系（6项指标）
6. 更新 NPZ 文件元数据字段

**v3.0主要修订**:
1. 统一External数据集来源描述（自采弱标注）
2. 明确指标主次定位：Spearman ρ为主，MAE/RMSE为辅助
3. 修正FID和相关系数为数据集级诊断指标
4. 明确clean底图优先使用自采视频
5. 补充MaskBank构建细节 
6. 增加q_D+q_T训练成本评估
7. 确保训练损失与标签形式一致
8. 补充报表分离工作量

---

## 一、两权重系统收敛后的实验范围

### 1.1 收敛策略

基于两权重系统验证实验的结论，将Severity目标收敛为**两条主线对照**：

| 目标类型 | 目标名称 | 用途 | 配置 |
|----------|----------|------|------|
| **控制组** | `s = 1 - clean_ratio` | 传统面积强度基线 | Simple Mean |
| **主实验组** | `S_full_wgap_alpha50` | SD增强与时序的统一主目标 | w=(0,0.15,0.50,1.0), α=0.5,β=0.4,γ=0.1 |

### 1.2 其他权重组合的定位

其余权重组合（S_op_only, S_op_sp, S_full, S_full_eta00）定位为**敏感性分析与选择依据**：

| 定位 | 内容 |
|------|------|
| **论文正文** | 只呈现 s vs S_full_wgap_alpha50 的对比 |
| **附录/讨论** | 引用敏感性分析结果作为设计依据 |
| **SD实验** | 不再扩展维度，避免指标解释混杂 |

### 1.3 统一配置版本化

所有实验使用同一套 Severity 定义配置，落盘保存为 `severity_config.json`：

```json
{
  "version": "v1.0_wgap_alpha50",
  "description": "Converged Severity Score for SD enhancement experiments",
  "class_weights": [0.0, 0.15, 0.50, 1.0],
  "fusion_coeffs": {
    "alpha": 0.5,
    "beta": 0.4,
    "gamma": 0.1
  },
  "eta_trans": 0.9,
  "spatial_mode": "gaussian",
  "spatial_sigma": 0.5
}
```

---

## 二、数据集统一描述与严格隔离

### 2.1 数据集统一描述

#### 2.1.1 WoodScape (真实车载镜头数据)

| 属性 | 值 |
|------|-----|
| **来源** | WoodScape车载摄像头采集 |
| **标注类型** | 强标注（完整像素级mask） |
| **训练集** | 3,200 samples |
| **验证集** | 800 samples |
| **测试集** | 1,000 samples |
| **分辨率** | 640×480, 4:3 |
| **用途** | 模型训练、同域评估 |

#### 2.1.2 External Test (自采弱标注外部泛化测试集)

| 属性 | 值 |
|------|-----|
| **来源** | 自采视频抽帧，人工标注 |
| **标注类型** | 弱标注（1-5级别人工标注） |
| **样本量** | ~50,000 samples |
| **标注内容** | `ext_level`: 整体观感严重度等级 |
| **目录** | `dataset/my_external_test/` |
| **用途** | 跨域泛化评测 **（不用于训练）** |

**关键说明**: External Test 为自采数据集，非互联网收集。其弱标注特性意味着：
- **主要评估指标**: Spearman ρ（排序一致性）
- **辅助统计**: MAE, RMSE（固定映射规则，需谨慎解释）

#### 2.1.3 Clean 数据集 (自采环视视频)

| 属性 | 值 |
|------|-----|
| **来源** | 自采环视视频抽帧 |
| **选择标准** | 干净镜头（无脏污） |
| **分辨率** | 640×480, 4:3（与WoodScape对齐） |
| **实际样本量** | 5,258 frames |
| **目录** | `dataset/my_clean_frames_4by3/` |
| **清单文件** | `dataset/my_clean_manifests/manifest_clean_4by3.csv` |
| **用途** | 合成数据底图（生成Train_Syn） |

**数据结构**:
```
manifest_clean_4by3.csv字段说明:
- view_id: 视角类型 (lf=左前, rf=右前, lr=左后, rr=右后)
- out_path: 最终4:3裁剪后的图像路径
- final_w, final_h: 最终分辨率 (640×480)
```

**选择理由**:
- 避免引入不同数据集的相机几何差异（鱼眼畸变、视场角等）
- 统一成像参数和色彩风格
- 降低q_D筛选的复杂度
- 已完成4:3裁剪，与WoodScape分辨率对齐

### 2.2 数据划分规则

#### 2.2.1 合成数据侧：训练用 vs 测试用

| 数据集 | 用途 | 来源限制 |
|--------|------|----------|
| `Train_Syn` | 训练侧合成数据 | Clean_Train + MaskBank_Train |
| `Test_Syn` | 合成域分析（可选） | Clean_Test + MaskBank_Val |
| `Val_Real` | 真实域验证集 | 真实WoodScape Val |
| `Test_Real` | 真实域测试集 | 真实WoodScape Test，**独立** |

**关键规则**:
1. `Train_Syn` **不进入** `Val_Real` 与 `Test_Real` 主评测集
2. 合成域分析如需存在，单独建立 `Test_Syn`，单表汇报
3. `Test_Real` 仅用于最终评测，不参与任何训练侧策略

#### 2.2.2 Mask Bank 严格隔离

| Mask Bank | 来源 | 用途 | 样本量 |
|-----------|------|------|--------|
| `MaskBank_Train` | WoodScape **Train only** | 训练侧mask形态库 | 3,200 |
| `MaskBank_Val` | WoodScape **Val only** | 生成质量抽检 | 800 |
| `MaskBank_Test` | WoodScape **Test** | **禁用** | - |

**严格规定**:
1. **训练侧 mask bank**：仅使用 `Train_Real` 提取形态库与统计
2. **生成质量抽检**：可使用 `Val_Real` 做人工抽检与质量统计
3. `Test_Real` **不参与**: mask bank 构建、纹理库提取、提示词挑选、阈值拟合等任何会反向影响训练侧生成策略的环节

#### 2.2.3 Clean 底图划分

| Clean 数据集 | 用途 | 来源 | 样本量 |
|-------------|------|------|--------|
| `Clean_Train` | 生成 `Train_Syn` | 自采视频 train | ~80,000 |
| `Clean_Test` | 生成 `Test_Syn` 与质量验证 | 自采视频 val/test | ~20,000 |
| `Clean_Val` | 生成参数调优（可选） | 自采视频 val 抽取 | ~500 |

**关键要求**:
1. 清单中保留 `source_video` 与 `frame_id` 字段
2. 确保同一原视频片段不跨集合
3. 禁止使用 WoodScape 同一场景的干净帧作为 clean 底图

### 2.3 信息泄露检查清单

每次实验前必须确认：

- [ ] `Test_Real` mask 未用于 MaskBank_Train
- [ ] `Test_Real` 场景未用于 Clean_Train
- [ ] 合成数据的生成参数未在 `Test_Real` 上拟合
- [ ] 质量过滤阈值未在 `Test_Real` 上优化
- [ ] `Test_Syn` 与 `Test_Real` 评测结果分离报告
- [ ] q_D 阈判别器未在 `Test_Real` 上训练
- [ ] q_T 阈值未在 `Test_Real` 上选择

---

## 三、两阶段LoRA方案（SD2.1 Base + SD2.1 Inpainting）

> **核心思路**: 用 **SD2.1 base + LoRA** 学到"镜头脏污层的真实外观/光学特征"，再把该 LoRA **挂载到 SD2 inpainting 生成器**上，使得在给定 *clean 图 + 约束区域（来自 gtLabels 的支持域）* 时，生成的脏污层更接近真实世界（半透明、失焦、散射/高光环、颗粒边界等），同时尽量不污染背景。

### 3.1 总体架构

```
训练阶段:                推理阶段:
SD2.1 Base              SD2.1 Inpainting
    ↓                          ↓
+ LoRA Training        + Load LoRA
    ↓                          ↓
Learn "soiling       Clean Image + Mask → Soiling Effect
appearance"            (only in masked region)
```

### 3.2 为什么要拆成 "SD2.1 base 训练 + SD2 inpaint 推理"

#### 3.2.1 训练用 SD2.1 base（文生图）

* **训练工具链成熟**: 直接使用标准 LoRA 文生图训练流程，工程工作量最小
* **训练目标更明确**: 让 LoRA 学"脏污外观材质"，不要求真实 clean/dirty 配对
* **本地模型支持**: 使用本地SD2.1 base (`/home/yf/models/sd2_1_base`)

#### 3.2.2 推理用 SD2 inpainting

* **生成时需要"只在指定区域改变"**: inpainting 的 mask 约束最可靠
* **已有生成器可复用**: 仅需新增"加载 LoRA"功能
* **本地模型支持**: 使用本地SD2.1 Inpainting (`/home/yf/models/sd2_inpaint`)

#### 3.2.3 关键技术点

> **LoRA 注入在 attention 层（`to_q/to_k/to_v/to_out`）时**，即使 base 与 inpaint 的 UNet 输入通道不同（4 vs 9），attention 结构一致时迁移仍可用。

**验证结果** (2026-02-18):
- ✅ 成功加载 128 个 LoRA keys（attention层权重）
- ✅ 跳过 4 个不兼容层（conv_in/conv_out）
- ✅ LoRA权重在Inpainting模型中正常工作

### 3.3 数据与监督信号的核心设计

#### 3.3.1 避免"三元组陷阱"

WoodScape 通常没有严格配对的 (I_clean, I_dirty)，因此不做"映射学习"。

**改为**: 从真实脏污图像中**提取脏污主信号**，构造训练图，使 LoRA 学"镜头脏污外观"。

#### 3.3.2 训练图构造（soiling-focused）

对每个样本的 dirty 图 (I) 与 mask (M)：

| 区域 | 处理方式 | 目的 |
|------|----------|------|
| **mask 内** | 保持原像素 | 保留真实脏污外观 |
| **mask 外** | blur/downsample/desaturate | 降语义，减少 LoRA 学道路背景 |

得到训练图 (I_train)，用于 LoRA 文生图训练。

#### 3.3.3 Caption 控制（自然语言锚点）

避免 `<opaque>` 这类不稳定 special token，采用自然语言可控描述：

**类别锚点**（来自 mask 主导类别）:
| 类别 | Token |
|------|-------|
| C1 | `transparent soiling layer` |
| C2 | `semi-transparent dirt smudge` |
| C3 | `opaque heavy stains` |

**严重度锚点**（由 Severity Score 分桶）:
| 严重度 | Token |
|--------|-------|
| mild | `mild / moderate / severity level 1/2/3` |
| noticeable | `noticeable` |
| severe | `severe` |

**光学短语**（固定保留）:
```
on camera lens, out of focus foreground, subtle glare, background visible
```

**完整Caption示例**:
```
noticeable transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible
severe opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible
```

#### 3.3.4 类别不平衡处理（训练采样层面）

WoodScape 的 opaque 占比高（61.8%），会导致生成偏厚重。

**解决方案**: Stratified sampling / oversampling 让 trans/semi/opaque 在训练中接近 1:1:1。

### 3.4 LoRA 训练配置

#### 3.4.1 已完成训练 (v1)

| 参数 | 值 | 说明 |
|------|-----|------|
| **Base 模型** | `/home/yf/models/sd2_1_base` | 本地SD2.1 base |
| **Rank** | 16 | 纹理/光学细节需要容量 |
| **Alpha** | 32 | 2× rank，标准实践 |
| **Target Layers** | `to_q/to_k/to_v/to_out` | Attention层 |
| **Learning Rate** | 1e-4 | 标准LoRA学习率 |
| **Training Steps** | 3,000 | ~1 epoch for 3200 samples |
| **Batch Size** | 4 | GPU内存平衡 |
| **Dropout** | 0.1 | 轻度正则化 |
| **Output** | `/home/yf/soiling_project/sd_scripts/lora/output/20260217_182102_lora_v1/` | Checkpoints @ 500/1000/1500/2000/2500/3000 |

#### 3.4.2 LoRA 验证结果

| 检查项 | 结果 | 说明 |
|--------|------|------|
| **Loss收敛** | ✅ | 3000步完成训练 |
| **生成质量** | ✅ | 可生成脏污效果 |
| **Inpainting迁移** | ✅ | 128 keys成功加载 |
| **尺寸兼容性** | ✅ | 正确跳过conv_in/conv_out |

### 3.5 把 LoRA 集成到 SD2 inpainting 生成器

#### 3.5.1 生成输入

| 输入 | 来源 | 说明 |
|------|------|------|
| **clean 底图** | `dataset/my_clean_frames_4by3/` | 自采干净帧 (5,258 samples) |
| **约束区域** | WoodScape GT labels | 作为支持域 |
| **Mask软化** | 边界羽化 + 形态扰动 | 得到 alpha，减少贴纸感 |

**Mask软化处理**:
- 边界羽化（blur）
- 轻微形态扰动（dilate/erode）
- 内部调制（噪声/条纹）使其更像真实脏污形态

> **重要**: gtLabels 的块状直边是"发生区域约束"，不是最终真实边界；把它软化是减少贴纸感的关键。

#### 3.5.2 推理逻辑

* **仍用 inpainting**，只是 **加载 LoRA 权重**
* **参数范围建议**（起点）:

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `strength` | 0.25 ~ 0.45 | 太大背景会被重画 |
| `guidance_scale` | 4 ~ 7 | 太高容易贴纸感 |
| `steps` | 25 ~ 40 | 质量与速度平衡 |

#### 3.5.3 输出与记录

保存生成图、对应 mask/alpha、prompt、LoRA 版本、推理超参到 manifest，便于后续做数据集版本对齐与论文复现。

### 3.6 优化效果评估：证据面板

#### 3.6.1 必做定量指标

**1. Spill Rate（背景污染率）**

衡量 mask 外区域被改写的比例，越低越好：
```
spill = Σ_{(x,y)∉M} |I'(x,y)-I(x,y)|₁ / Σ_{x,y} |I'(x,y)-I(x,y)|₁
```

**2. Mask 内统计匹配**

在 M 内比较生成与真实脏污的颜色/纹理统计距离（直方图、梯度能量、频谱能量等），反映"像不像真实脏污层"。

**3. QC 通过率**

沿用现有 6 项质量指标，作为工程可用性门槛。

#### 3.6.2 必做图例输出

**1. 可控性网格图**（最关键）

固定同一 clean 图 + 同一 mask，生成 3×3 网格：
- 类型（trans/semi/opaque）× 严重度（mild/moderate/severe）

**2. 对比面板**

Baseline（无 LoRA） vs +LoRA：同一输入、同一超参，直观看差异。

**3. Spill 可视化**

输出 |I'-I| 的热力图（尤其是 mask 外），证明背景未被重画。

### 3.7 与论文实验主线的衔接

当 LoRA+inpaint 生成器稳定后，它在论文里扮演的是**"合成数据增强模块"**，后续在检测模型训练侧可以做：

* **无增强 vs 有增强**: Real only vs Real+Syn
* **Syn 不筛选 vs 筛选**: 按质量指标或任务一致性指标筛选
* **在增强成立的前提下**，再比较：
  * (s) vs (S)
  * 是否启用 Aggregator+L_cons
  * 下游车道线 probe 的关联性/分桶退化规律

这样 SD/LoRA 贡献与标签/结构贡献在实验设计上是**正交可分**的。

### 3.8 一句话概括

用 **SD2.1 base 训练一个"镜头脏污外观 LoRA"**（不依赖 clean/dirty 配对，靠 mask 抑制背景语义），再把该 LoRA **挂到 SD2 inpainting 生成器**里，让生成过程在 **gtLabels 支持域约束**下只改局部，从而得到更真实、可控、可版本化的合成脏污数据。

---

## 四、MaskBank 构建细节（补充）

### 4.1 MaskBank 标准化流程

#### 4.1.1 数据预处理

```python
def build_mask_bank_train(
    woodscape_train_dir: Path,
    target_resolution: Tuple[int, int] = (640, 480),
    output_dir: Path = Path("dataset/mask_bank_train/"),
) -> Dict:
    """
    构建标准化MaskBank_Train

    流程:
    1. 从WoodScape Train读取mask
    2. 统一分辨率到640×480
    3. 提取形态参数
    4. 计算统计信息
    """
    from pathlib import Path
    import cv2
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    stats = []

    # 读取WoodScape Train masks
    mask_files = list(Path(woodscape_train_dir).glob("*.png"))

    print(f"Found {len(mask_files)} masks in WoodScape Train")

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        # 读取mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        # 统一分辨率
        mask_resized = cv2.resize(mask, target_resolution,
                               interpolation=cv2.INTER_NEAREST)

        # 归一化为4类
        # WoodScape原始: 0=clean, 125=trans, 200=semi, 255=opaque
        mask_4class = np.zeros_like(mask_resized, dtype=np.int32)
        mask_4class[mask_resized < 50] = 0  # clean
        mask_4class[(mask_resized >= 50) & (mask_resized < 150)] = 1  # trans
        mask_4class[(mask_resized >= 150) & (mask_resized < 230)] = 2  # semi
        mask_4class[mask_resized >= 230] = 3  # opaque

        # 保存预处理后的mask
        output_file = output_dir / f"{mask_file.stem}_processed.png"
        cv2.imwrite(str(output_file), mask_4class)

        # 提取统计信息
        stats.append(extract_mask_statistics(mask_4class))

        masks.append({
            'file_id': mask_file.stem,
            'original_path': str(mask_file),
            'processed_path': str(output_file),
            'resolution': target_resolution,
        })

    # 保存清单
    manifest = pd.DataFrame(masks)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    # 保存统计信息
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "statistics.csv", index=False)

    print(f"\nMaskBank_Train built: {len(masks)} masks")
    print(f"Saved to: {output_dir}")

    return {
        'masks': masks,
        'statistics': stats_df,
        'output_dir': output_dir,
    }


def extract_mask_statistics(mask: np.ndarray) -> Dict:
    """
    提取mask统计信息

    返回:
        {
            'clean_ratio': float,      # clean区域占比
            'class_ratios': Tuple,       # 各类别占比
            'coverage': float,          # 有脏污区域占比
            'dominant_class': int,      # 主导类别
            'num_regions': int,         # 连通区域数量
            'dominance_region_size': float,  # 最大区域占比
        }
    """
    H, W = mask.shape

    # 类别占比
    total_pixels = H * W
    class_counts = np.bincount(mask.flatten(), minlength=4)
    class_ratios = tuple(class_counts / total_pixels)

    # clean占比
    clean_ratio = class_ratios[0]

    # 覆盖率（有脏污区域占比）
    coverage = 1.0 - clean_ratio

    # 主导类别
    dominant_class = np.argmax(class_counts[1:]) + 1  # skip clean

    # 连通区域数量
    num_clean, num_regions = cv2.connectedComponents(
        (mask > 0).astype(np.uint8)
    )

    # 最大区域占比
    max_region_size = num_regions.max() / total_pixels if num_regions.size > 0 else 0

    return {
        'clean_ratio': float(clean_ratio),
        'class_ratios': tuple(class_ratios),
        'coverage': float(coverage),
        'dominant_class': int(dominant_class),
        'num_regions': int(num_regions),
        'max_region_size': float(max_region_size),
    }
```

#### 3.1.2 形态参数提取

```python
def extract_morphology_parameters(
    mask: np.ndarray,
    class_id: int,
) -> Dict:
    """
    提取形态参数

    参数:
        mask: [H, W] 4类mask
        class_id: 关注的类别 (1, 2, 3)

    返回:
        {
            'morph_type': str,      # drop, smear, block, streak
            'morph_params': Dict,   # 形态参数
            'area': float,           # 区域面积
            'aspect_ratio': float,  # 长宽比
            'eccentricity': float,   # 离心率
            'solidity': float,      # 实心度
        }
    """
    # 提取目标类别的区域
    binary_mask = (mask == class_id).astype(np.uint8)

    if binary_mask.sum() == 0:
        return None

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels == 0:
        return None

    # 找到最大区域
    max_label = np.argmax([s[cv2.CC_STAT_AREA] for s in stats]) + 1

    # 创建最大区域的mask
    largest_region = (labels == max_label).astype(np.uint8)

    # 计算形状特征
    contours, _ = cv2.findContours(largest_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = contours[0]
    area = cv2.contourArea(cnt)

    # 边界矩形
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 1.0

    # 凸包
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0

    # 离心率
    if cnt.shape[0] >= 5:
        ellipse = cv2.fitEllipse(cnt)
        eccentricity = ellipse[2]
    else:
        eccentricity = 0.0

    # 判断形态类型
    morph_type = classify_morphology(
        area, aspect_ratio, eccentricity, solidity
    )

    return {
        'morph_type': morph_type,
        'morph_params': {
            'area': float(area),
            'aspect_ratio': float(aspect_ratio),
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'centroid_x': float(x + w/2),
            'centroid_y': float(y + h/2),
        },
    }


def classify_morphology(
    area: float,
    aspect_ratio: float,
    eccentricity: float,
    solidity: float,
) -> str:
    """
    根据形状特征分类形态类型

    规则:
    - drop: 面积小，接近圆形
    - smear: 面积中等，高长宽比
    - block: 面积大，实心度高
    - streak: 面积中等，极高长宽比
    """
    if area < 500:  # 小面积
        return "drop"
    elif aspect_ratio > 5.0:  # 极高长宽比
        return "streak"
    elif solidity > 0.8:  # 高实心度
        return "block"
    else:
        return "smear"
```

#### 3.1.3 覆盖率分布控制

```python
class CoverageSampler:
    """
    覆盖率分层采样器

    确保合成样本覆盖不同的强度区间
    """
    def __init__(
        self,
        mask_bank_dir: Path,
        n_bins: int = 10,
        target_distribution: str = "uniform",  # uniform, real, custom
    ):
        self.mask_bank_dir = mask_bank_dir
        self.n_bins = n_bins
        self.target_distribution = target_distribution

        # 加载mask统计
        manifest = pd.read_csv(mask_bank_dir / "manifest.csv")
        stats = pd.read_csv(mask_bank_dir / "statistics.csv")

        # 计算覆盖率分布
        coverages = stats['coverage'].values

        if target_distribution == "real":
            # 使用真实数据的分布
            hist, bin_edges = np.histogram(coverages, bins=n_bins)
            self.bin_probs = hist / hist.sum()
        elif target_distribution == "uniform":
            # 均匀分布
            self.bin_probs = np.ones(n_bins) / n_bins
        else:
            # 自定义分布
            self.bin_probs = target_distribution

        # 分箱边界
        self.bin_edges = np.linspace(0, 1, n_bins + 1)

        # 按覆盖率分箱
        self.masks_by_bin = self._organize_masks_by_coverage(manifest, stats)

    def _organize_masks_by_coverage(
        self,
        manifest: pd.DataFrame,
        stats: pd.DataFrame
    ) -> Dict[int, List[Dict]]:
        """按覆盖率分箱"""
        masks_by_bin = {i: [] for i in range(self.n_bins)}

        for _, row in manifest.iterrows():
            stat_row = stats[stats['file_id'] == row['file_id']]
            if len(stat_row) == 0:
                continue

            coverage = stat_row['coverage'].iloc[0]
            bin_idx = self._get_bin_index(coverage)

            masks_by_bin[bin_idx].append({
                'file_id': row['file_id'],
                'path': row['processed_path'],
                'coverage': coverage,
            })

        return masks_by_bin

    def _get_bin_index(self, coverage: float) -> int:
        """获取覆盖率对应的bin索引"""
        bin_idx = int(coverage * self.n_bins)
        return min(bin_idx, self.n_bins - 1)

    def sample_mask(self, n_samples: int = 1) -> List[str]:
        """
        按目标分布采样mask

        返回: mask路径列表
        """
        # 确定每个bin的采样数量
        samples_per_bin = np.random.multinomial(
            n_samples, self.bin_probs
        )

        sampled_masks = []

        for bin_idx, count in enumerate(samples_per_bin):
            available = self.masks_by_bin[bin_idx]

            if count > len(available):
                # 如果需要的数量超过可用数量，重复采样
                indices = np.random.choice(len(available), size=count)
            else:
                indices = np.random.choice(len(available), size=count, replace=False)

            for idx in indices:
                sampled_masks.append(available[idx]['path'])

        return sampled_masks
```

### 4.2 MaskBank 使用接口

```python
class MaskBank:
    """
    MaskBank使用接口
    """
    def __init__(self, mask_bank_dir: Path):
        self.mask_bank_dir = mask_bank_dir

        # 加载清单和统计
        self.manifest = pd.read_csv(mask_bank_dir / "manifest.csv")
        self.statistics = pd.read_csv(mask_bank_dir / "statistics.csv")

        # 初始化采样器
        self.coverage_sampler = CoverageSampler(
            mask_bank_dir,
            n_bins=10,
            target_distribution="uniform"  # 可调
        )

    def sample_masks(
        self,
        n_samples: int,
        coverage_range: Tuple[float, float] = None,
        class_id: int = None,
        morph_type: str = None,
    ) -> List[Dict]:
        """
        采样mask

        参数:
            n_samples: 采样数量
            coverage_range: (min_coverage, max_coverage)
            class_id: 类别过滤 (1, 2, 3)
            morph_type: 形态类型过滤 (drop, smear, block, streak)
        """
        # 从coverage采样器获取基础样本
        mask_paths = self.coverage_sampler.sample_masks(n_samples * 2)  # 采样多一些以便过滤

        # 应用过滤条件
        sampled = []
        for path in mask_paths:
            # 获取统计信息
            stat_row = self.statistics[self.statistics['file_id'] == Path(path).stem]

            if len(stat_row) == 0:
                continue

            coverage = stat_row['coverage'].iloc[0]
            dominant_class = stat_row['dominant_class'].iloc[0]

            # 过滤条件
            if coverage_range is not None:
                if not (coverage_range[0] <= coverage <= coverage_range[1]):
                    continue

            if class_id is not None and dominant_class != class_id:
                continue

            # TODO: 添加morph_type过滤

            sampled.append({
                'path': path,
                'coverage': coverage,
                'dominant_class': dominant_class,
            })

            if len(sampled) >= n_samples:
                break

        return sampled[:n_samples]
```

---

## 五、合成生成方式（Inpainting优先）

> **方案类型**: 静态 SD Inpainting（不考虑时序 embedding）
>
> **核心特点**: 单帧独立生成，使用 Stable Diffusion Inpainting 模型

### 5.1 核心原则

**合成目标**: 背景不被改写，mask 内生成脏污外观，保证标签天然对齐

**设计原则**: "生成控制量即标签来源"
- 每个合成样本只包含一个脏污类别 c ∈ {1,2,3}
- mask 内所有像素统一为类别 c，与 prompt 一致
- 标签侧也只含 0 与 c，确保闭环一致性

### 5.2 Inpainting 生成实现（静态方案）

```python
class SDInpaintingSoilingGenerator:
    """
    基于 Stable Diffusion Inpainting 的脏污生成器（静态版本）

    核心特性:
    1. 单类别 mask 策略: 每个样本只包含一个目标类别
    2. 版本绑定机制: Severity 配置与阈值版本严格匹配
    3. 质量控制体系: 6 项指标确保生成质量
    4. 异常容错机制: 单个样本失败不终止整批生成
    """
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        severity_config_path: Path = SEVERITY_CONFIG_PATH,
    ):
        self.device = device
        self.torch_dtype = torch_dtype

        # 加载 Severity 配置（版本化）
        self.severity_config = load_severity_config(severity_config_path)
        severity_version = self.severity_config.get("version", "unknown")

        # 加载分箱阈值（版本绑定）
        self.rebinned_thresholds = load_rebinned_thresholds(severity_version)

        # 提取配置参数
        self.class_weights = self._parse_class_weights()  # 兼容 list/dict
        self.alpha = self.severity_config["fusion_coeffs"]["alpha"]
        self.beta = self.severity_config["fusion_coeffs"]["beta"]
        self.gamma = self.severity_config["fusion_coeffs"]["gamma"]
        self.eta_trans = self.severity_config["eta_trans"]
        self.sigma = self.severity_config["spatial"]["sigma"]

        # 加载 SD Inpainting 模型
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to(device)

        # 质量控制阈值
        self.background_ssim_threshold = 0.95
        self.background_diff_threshold = 0.05

        # 类别相关的 mask_diff 阈值
        self.mask_diff_min_thresholds = {
            1: 0.02,  # transparent: 允许较小变化
            2: 0.05,  # semi_transparent: 中等变化
            3: 0.10,  # opaque: 需要明显变化
        }

    def generate(
        self,
        clean_image: np.ndarray,
        target_mask: np.ndarray,
        prompt: str,
        target_class: int,
        negative_prompt: str = NEGATIVE_PROMPT,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        生成合成脏污图像（单类别版本）

        设计原则：
        - 每个合成样本只包含一个脏污类别 c ∈ {1,2,3}
        - mask 内所有像素统一为类别 c，与 prompt 一致
        - 标签侧也只含 0 与 c，确保"生成控制量即标签来源"的闭环

        返回:
            成功时返回结果字典，失败返回 None
        """
        # 1. 转换为单类别 mask
        single_class_mask = self._convert_to_single_class_mask(target_mask, target_class)

        # 2. 准备 PIL 图像
        clean_pil = Image.fromarray(clean_image.astype(np.uint8))
        mask_binary = (single_class_mask > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_binary)

        # 3. SD Inpainting 生成
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                image=clean_pil,
                mask_image=mask_pil,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        synthetic_image = np.array(result.images[0])

        # 4. 质量检查
        quality_result = self._quality_check(
            clean_image, synthetic_image, single_class_mask
        )

        if not quality_result["passed"]:
            return None

        # 5. 计算 Severity Score
        tile_cov = self._compute_tile_coverage(single_class_mask)
        severity = compute_severity_from_tile_cov(
            tile_cov,
            class_weights=self.class_weights,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            eta_trans=self.eta_trans,
            sigma=self.sigma,
        )

        # 6. 计算 L (Level) - 版本严格绑定
        thresholds_source = self.rebinned_thresholds.get("source", "unknown")
        if thresholds_source == "version_specific":
            L = compute_L(severity["s_simple"], severity["S_full"], self.rebinned_thresholds)
            L_unavailable_reason = None
        else:
            L = -1  # 标记为不可用
            L_unavailable_reason = f"thresholds_source={thresholds_source}"

        return {
            "synthetic_image": synthetic_image.astype(np.uint8),
            "clean_image": clean_image.astype(np.uint8),
            "target_mask": single_class_mask.astype(np.uint8),
            "target_class": int(target_class),
            "tile_cov": tile_cov.astype(np.float32),
            # Severity Score
            "S_op": severity["S_op"],
            "S_sp": severity["S_sp"],
            "S_dom": severity["S_dom"],
            "S_full": severity["S_full"],
            "s_simple": severity["s_simple"],
            "clean_ratio": severity["clean_ratio"],
            # L (Level)
            "L": int(L),
            "L_unavailable_reason": L_unavailable_reason,
            "thresholds_source": thresholds_source,
            # 元数据
            "prompt": prompt,
            "seed": seed,
            "quality": quality_result,
            "severity_config_version": self.severity_config.get("version", "unknown"),
        }

    def _quality_check(
        self,
        clean_image: np.ndarray,
        synthetic_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Dict:
        """
        质量检查（6 项指标）

        1. Background SSIM > 0.95: 背景一致性
        2. Background Diff < 0.05: 背景像素差异
        3. Mask Diff (类别相关): mask 内变化幅度
        4. Boundary Artifact Score < 30.0: 边界伪影检测
        5. Mask Coverage <= 0.8: 覆盖率上限
        """
        mask_bool = (target_mask > 0)
        unique_values = np.unique(target_mask[target_mask > 0])
        target_class = int(unique_values[0]) if len(unique_values) > 0 else None

        result = {
            "passed": True,
            "background_ssim": 1.0,
            "background_diff": 0.0,
            "mask_diff": 0.0,
            "boundary_artifact_score": 0.0,
        }

        # 背景一致性检查
        synthetic_masked = synthetic_image.copy()
        if mask_bool.any():
            synthetic_masked[mask_bool] = clean_image[mask_bool]  # 修复链式索引 bug

        # 计算全图 SSIM
        ssim_values = []
        for c in range(3):
            ssim_c = ssim(
                synthetic_masked[:, :, c],
                clean_image[:, :, c],
                data_range=255
            )
            ssim_values.append(ssim_c)
        background_ssim = np.mean(ssim_values)
        result["background_ssim"] = float(background_ssim)

        if background_ssim < 0.95:
            result["passed"] = False
            result["fail_reason"] = f"背景 SSIM 过低: {background_ssim:.4f}"

        # 背景像素差异
        mask_complement = ~mask_bool
        if mask_complement.sum() > 0:
            background_diff = np.abs(
                synthetic_image[mask_complement].astype(float) -
                clean_image[mask_complement].astype(float)
            ).mean() / 255.0
            result["background_diff"] = float(background_diff)

            if background_diff > 0.05:
                result["passed"] = False
                if "fail_reason" not in result:
                    result["fail_reason"] = f"背景差异过大: {background_diff:.4f}"

        # mask 内变化幅度（类别相关阈值）
        if mask_bool.sum() > 0:
            mask_diff = np.abs(
                synthetic_image[mask_bool].astype(float) -
                clean_image[mask_bool].astype(float)
            ).mean() / 255.0
            result["mask_diff"] = float(mask_diff)

            min_threshold = self.mask_diff_min_thresholds.get(target_class, 0.05)
            if mask_diff < min_threshold:
                result["passed"] = False
                if "fail_reason" not in result:
                    result["fail_reason"] = f"mask 内变化不足: {mask_diff:.4f} < {min_threshold:.4f}"

        # 边界带伪影检测
        boundary_artifact_score = self._check_boundary_artifacts(
            clean_image, synthetic_image, target_mask
        )
        result["boundary_artifact_score"] = float(boundary_artifact_score)

        if boundary_artifact_score > 30.0:
            result["passed"] = False
            if "fail_reason" not in result:
                result["fail_reason"] = f"边界带伪影异常: {boundary_artifact_score:.2f}"

        # Mask 覆盖率上限检查
        mask_coverage = float(mask_bool.sum() / mask_bool.size)
        result["mask_coverage"] = mask_coverage

        if mask_coverage > 0.8:
            result["passed"] = False
            if "fail_reason" not in result:
                result["fail_reason"] = f"mask 覆盖率过高: {mask_coverage:.2f} > 0.8"

        return result

    def _convert_to_single_class_mask(
        self,
        mask: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        将多类别 mask 转换为单类别 mask

        转换策略：所有脏污像素 (mask>0) 统一赋值为 target_class
        """
        if target_class not in (1, 2, 3):
            raise ValueError(f"target_class 必须为 1, 2, 3，当前为 {target_class}")

        single_class_mask = np.where(mask > 0, target_class, 0).astype(np.uint8)
        return single_class_mask
```

### 5.3 类别采样比例配置

```python
# 类别采样比例（用于控制合成数据的类别分布）
# 按预设比例采样 target_class，而非被动依赖 MaskBank 的类别比例
CLASS_PROPORTIONS = {
    1: 0.33,  # transparent
    2: 0.34,  # semi_transparent
    3: 0.33,  # opaque
}
# 确保总和为 1
_total = sum(CLASS_PROPORTIONS.values())
CLASS_PROPORTIONS = {k: v / _total for k, v in CLASS_PROPORTIONS.items()}

# 使用方式
target_class = np.random.choice(
    list(CLASS_PROPORTIONS.keys()),
    p=list(CLASS_PROPORTIONS.values())
)

# 根据目标类别选择 prompt
PROMPT_TEMPLATES = {
    "transparent": [
        "light dirt smudges on camera lens",
        "transparent dirt spots on lens",
    ],
    "semi_transparent": [
        "semi-transparent dirt on camera lens",
        "foggy smudges on lens surface",
    ],
    "opaque": [
        "heavy dirt stains on camera lens",
        "opaque dirt patches blocking camera view",
        "mud and dirt covering camera lens",
    ],
}
```

### 5.4 版本绑定机制

```python
def load_rebinned_thresholds(severity_version: str = "unknown") -> Dict:
    """
    加载训练集自适应分箱阈值（严格版本匹配模式）

    阈值文件按 Severity 版本命名绑定，严格策略：
    - 必须加载版本特定文件：rebinned_thresholds__{version}.json
    - 若版本特定文件不存在，L 将被标记为不可用
    - 避免使用错误阈值导致的 L 语义漂移

    Returns:
        {"b0": float, "b1": float, "b2": float, "version": str, "source": str}
        source ∈ {"version_specific", "missing_versioned", "default"}
    """
    meta_dir = DATASET_ROOT / "woodscape_processed" / "meta"
    version_specific_path = meta_dir / f"rebinned_thresholds__{severity_version}.json"

    if version_specific_path.exists():
        with open(version_specific_path, "r") as f:
            thresholds = json.load(f)
        thresholds["source"] = "version_specific"
        return thresholds
    else:
        # 版本特定文件不存在，L 标记为不可用
        print(f"⚠️ 未找到版本特定阈值文件，L 将被标记为不可用")
        return {
            "b0": 0.01,
            "b1": 0.10,
            "b2": 0.20,
            "version": "default",
            "source": "missing_versioned"
        }

# L 计算时检查版本
thresholds_source = self.rebinned_thresholds.get("source", "unknown")
if thresholds_source == "version_specific":
    L = compute_L(severity["s_simple"], severity["S_full"], self.rebinned_thresholds)
else:
    L = -1  # 标记为不可用
```

### 5.5 多类别混合生成（备选方案）

> **优先级**: 低 - 单类别策略验证后再考虑
>
> **触发条件**: 若单类别合成数据效果不足，或需要更贴近真实场景的多类别混合样本

#### 设计思路

```
当前单类别策略:
├── 优点: Prompt 匹配明确、质量控制可控
└── 限制: 与真实多类别混合场景有差异

多类别混合策略:
├── 方案 A: 分区域 Inpainting（推荐）
│   └── 按 mask 连通区域分别生成，保持"区域-类别-prompt"闭环
├── 方案 B: 混合 Prompt
│   └── 构建描述多类别的组合 prompt
└── 方案 C: 迭代 Inpainting
    └── 按类别优先级依次生成（先 opaque，再 semi，最后 trans）
```

#### 方案 A: 分区域 Inpainting（推荐）

```python
def generate_multiclass_by_region(
    clean_image: np.ndarray,
    target_mask: np.ndarray,  # 原始多类别 mask
    generator: SDInpaintingSoilingGenerator,
) -> Dict:
    """
    多类别合成：按连通区域分别生成

    保持"生成控制量即标签来源"的区域级闭环
    """
    synthetic_image = clean_image.copy().astype(float)
    final_mask = np.zeros_like(target_mask)

    # 对每个类别分别处理
    for class_id in [1, 2, 3]:
        class_mask = (target_mask == class_id)

        if class_mask.sum() == 0:
            continue

        # 获取该类别的连通区域
        num_labels, labels = cv2.connectedComponents(
            class_mask.astype(np.uint8)
        )

        # 对每个连通区域单独 inpainting
        for label in range(1, num_labels):
            region_mask = (labels == label)

            # 使用该类别对应的 prompt
            prompt = select_prompt_for_class(class_id)

            # 局部 inpainting（扩展 mask 边界避免硬边界）
            expanded_mask = expand_mask_boundary(region_mask, pixels=5)

            # 生成
            result = generator.generate(
                clean_image=synthetic_image.astype(np.uint8),
                target_mask=expanded_mask.astype(np.uint8),
                prompt=prompt,
                target_class=class_id,
            )

            if result is not None:
                synthetic_image[region_mask] = result['synthetic_image'][region_mask]
                final_mask[region_mask] = class_id

    return {
        'synthetic_image': synthetic_image.astype(np.uint8),
        'target_mask': final_mask,
    }
```

**保留单类别策略的理由**:
1. 真实数据已提供多类别混合样本
2. 合成数据主要用于补充严重度覆盖
3. 先验证简单方案有效，再增加复杂度

---

### 5.6 分层合成（备选方案）

```python
def generate_synthetic_soiling_layered(
    clean_image: np.ndarray,
    target_mask: np.ndarray,
    texture_bank: Dict,
    class_weights: Tuple = (0.0, 0.15, 0.50, 1.0),
) -> Dict:
    """
    分层合成生成（备选方案）

    分层建模表达式:
        I' = I_clean ⊙ (1-M) + blend(T, I_clean) ⊙ M
    """
    H, W = clean_image.shape[:2]
    M = target_mask

    # 结果初始化（从clean图像开始）
    synthetic_image = clean_image.copy().astype(float)

    # 对每个类别应用不同的纹理
    for class_id in [1, 2, 3]:  # skip clean (0)
        class_mask = (M == class_id)

        if class_mask.sum() == 0:
            continue

        # 选择对应类别的纹理
        texture = sample_texture_from_bank(texture_bank, class_id)

        # 应用纹理
        synthetic_region = apply_texture_to_region(
            synthetic_image,
            class_mask,
            texture,
            class_weights=class_weights,
        )

        # 更新合成图像
        synthetic_image[class_mask] = synthetic_region

    # 转换回uint8
    synthetic_image = np.clip(synthetic_image, 0, 255).astype(np.uint8)

    return {
        'synthetic_image': synthetic_image,
        'synthetic_mask': target_mask,
        'clean_image': clean_image,
    }
```

---

## 六、质量控制模块（修订）

### 6.1 质量评估指标体系

#### 5.1.1 主要筛选指标（硬约束）

| 指标 | 计算方法 | 阈值 | 说明 |
|------|----------|------|------|
| **背景一致性SSIM** | SSIM(I'\\M, I_clean\\M) | > 0.95 | mask外背景未被重绘 |
| **背景像素差异** | \|I' - I_clean\| on Ω\\M | < 0.05 | mask外像素差异小 |
| **边界带伪影检测** | 梯度异常统计 | < threshold | mask边界无破碎 |
| **mask内变化幅度** | \|I' - I_clean\| on M | > 0.1 | mask内有有效脏污生成 |

**注意**: "mask IoU" 不作为主筛选指标（在inpainting流程中判别力不足）

#### 5.1.2 数据集级诊断指标

| 指标 | 计算方法 | 用途 |
|------|----------|------|
| **FID** | 整个数据集的特征分布距离 | 诊断整体分布 |
| **Severity分布一致性** | 直方图/KS检验 | 诊断Severity分布匹配 |
| **LPIPS** | 样本间感知距离分布 | 诊断多样性 |

**注意**: 这些指标按**数据集或批次**计算，不用于单样本阈值筛选

### 6.2 q_D + q_T 过滤加权闭环

#### 5.2.1 域相似度 q_D

**实现方式**: CycleGAN判别器

```python
# CycleGAN配置（严格数据隔离）
CYCLE_GAN_CONFIG = {
    'domain_A': 'clean',           # 干净域（自采视频）
    'domain_B': 'woodsc_real',    # WoodScape真实域

    # 严格数据隔离
    'domain_A_train': 'Clean_Train',
    'domain_A_test': 'Clean_Test',
    'domain_B_train': 'Train_Real',    # 仅使用WoodScape Train
    'domain_B_test': 'Val_Real',       # 使用Val作为测试集，不使用Test

    # 禁止使用 Test_Real
    'forbidden_sources': ['Test_Real'],
}

# 训练CycleGAN
cycle_gan = train_cycle_gan(CYCLE_GAN_CONFIG)

# 提取判别器作为q_D计算器
domain_discriminator = cycle_gan.discriminator_B


def compute_q_D_batch(
    synthetic_batch: torch.Tensor,  # [B, 3, H, W]
    domain_discriminator: nn.Module,
) -> np.ndarray:
    """
    批量计算域相似度 q_D

    返回: [B] 每个样本的q_D值
    """
    with torch.no_grad():
        # 判别器输出
        d_output = domain_discriminator(synthetic_batch)
        # d_output: [B, 1] ∈ [0, 1]
        # 接近0.5表示在边界，接近0或1表示偏向某一域

        # 转换为相似度（越接近0.5越相似）
        q_D = 1.0 - np.abs(d_output.cpu().numpy().flatten() - 0.5) * 2
        q_D = np.clip(q_D, 0.0, 1.0)

    return q_D
```

**训练成本估算**:
- CycleGAN训练时间: ~2-3天
- 推理成本: 可忽略（单次前向传播）

#### 5.2.2 任务一致性 q_T

**实现方式**: Baseline Teacher

```python
# 使用S_full_wgap_alpha50的baseline模型作为teacher
TEACHER_CONFIG = {
    'checkpoint': 'baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth',
    'device': 'cuda',
}

teacher_model = BaselineDualHead(pretrained=False)
teacher_model.load_state_dict(torch.load(TEACHER_CONFIG['checkpoint'])['model'])
teacher_model.to(TEACHER_CONFIG['device'])
teacher_model.eval()


def compute_q_T_batch(
    synthetic_batch: torch.Tensor,  # [B, 3, H, W]
    synthetic_tile_cov: np.ndarray,  # [B, 8, 8, 4] 真实tile coverage
    teacher_model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    批量计算任务一致性 q_T

    返回: [B] 每个样本的q_T值
    """
    B = synthetic_batch.shape[0]
    q_T_list = []

    with torch.no_grad():
        # Teacher预测
        outputs = teacher_model(synthetic_batch.to(device))
        G_hat = outputs['G_hat'].cpu()  # [B, 4, 8, 8]

    for i in range(B):
        # 计算预测与真实的距离
        G_star = synthetic_tile_cov[i]  # [8, 8, 4]
        G_hat_i = G_hat[i].permute(1, 2, 0).numpy()  # [8, 8, 4]

        # L2距离
        dist = np.linalg.norm(G_hat_i - G_star)

        # 转换为一致性分数（距离越小，一致性越高）
        tau = 0.5  # 温度参数
        q_T = np.exp(-dist / tau)

        q_T_list.append(q_T)

    return np.array(q_T_list)
```

**训练成本估算**:
- Teacher推理: 与正常推理一致，成本可忽略
- 批量计算可显著提高效率

#### 5.2.3 过滤策略

**策略1: 硬过滤**

```python
def filter_synthetic_data_batched(
    samples: List[Dict],
    domain_discriminator: nn.Module,
    teacher_model: nn.Module,
    device: torch.device,
    q_D_threshold: float = 0.5,
    q_T_threshold: float = 0.5,
    batch_size: int = 32,
) -> List[Dict]:
    """
    批量计算q_D和q_T并筛选
    """
    filtered = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        if len(batch) == 0:
            break

        # 准备batch
        images = np.stack([s['synthetic_image'] for s in batch])
        images = torch.from_numpy(images).float().permute(0, 3, 1).to(device) / 255.0
        tile_covs = np.array([s['tile_cov'] for s in batch])

        # 计算q_D
        q_D = compute_q_D_batch(images, domain_discriminator)

        # 计算q_T
        q_T = compute_q_T_batch(images, tile_covs, teacher_model, device)

        # 筛选
        for j, sample in enumerate(batch):
            sample['q_D'] = float(q_D[j])
            sample['q_T'] = float(q_T[j])

            if q_D[j] >= q_D_threshold and q_T[j] >= q_T_threshold:
                filtered.append(sample)

    return filtered
```

**策略2: Soft加权（训练时）**

```python
def compute_sample_weights(
    samples: List[Dict],
    domain_discriminator: nn.Module,
    teacher_model: nn.Module,
    device: torch.device,
    alpha: float = 0.5,  # q_D 和 q_T 的相对权重
    batch_size: int = 32,
) -> np.ndarray:
    """
    计算样本权重（用于训练时加权采样）
    """
    n_samples = len(samples)
    weights = np.zeros(n_samples)

    for i in range(0, n_samples, batch_size):
        batch = samples[i:i+batch_size]
        if len(batch) == 0:
            break

        # 准备batch
        images = np.stack([s['synthetic_image'] for s in batch])
        images = torch.from_numpy(images).float().permute(0, 3, 1).to(device) / 255.0
        tile_covs = np.array([s['tile_cov'] for s in batch])

        # 计算q_D和q_T
        q_D = compute_q_D_batch(images, domain_discriminator)
        q_T = compute_q_T_batch(images, tile_covs, teacher_model, device)

        # 组合权重
        batch_weights = alpha * q_D + (1 - alpha) * q_T
        weights[i:i+len(batch)] = batch_weights

    return weights
```

### 6.3 单调性验证实验

为验证合成数据闭环的有效性：

```
Real only < Real + Syn (Naïve) < Real + Syn (q_D only) < Real + Syn (q_D + q_T)
```

**实验设计**:

| 实验组 | 数据组成 | 预期 External ρ | 说明 |
|--------|----------|-----------------|------|
| Baseline | Train_Real | 0.7177 | 基线 |
| Naïve | Train_Real + Train_Syn (无筛选) | ≥ 0.7177 | 验证合成数据基本有效性 |
| q_D only | Train_Real + Train_Syn (q_D筛选) | > Naïve | 验证域相似度筛选有效 |
| q_D + q_T | Train_Real + Train_Syn (q_D+q_T筛选) | > q_D only | 验证任务一致性筛选有效 |

---

## 七、评估指标主次定位（修订）

### 7.1 主要评估指标

| 评估维度 | WoodScape | External | 说明 |
|----------|----------|----------|------|
| **排序一致性** | ✅ | **✅** | **Spearman ρ**（主要） |
| **双头一致性** | ✅ | ✅ | **Δρ = ρ(S_hat) - ρ(S_agg)** |
| **功能解释** | ✅ | - | Lane Probe相关性（未来工作） |

### 7.2 辅助统计指标

| 评估维度 | WoodScape | External | 说明 |
|----------|----------|----------|------|
| **数值准确性** | MAE, RMSE | MAE, RMSE | 固定映射规则，需谨慎解释 |
| **排序一致性** | Spearman ρ | Spearman ρ | 主要指标 |

### 7.3 指标解释边界

**External Test MAE/RMSE 解释**:

```
External Test 的 MAE 和 RMSE 是在固定映射规则下的统计量：
  预测值 S ∈ [0, 1] 通过线性映射到 [1, 5]：S_mapped = 4*S + 1

这些指标：
1. 可以用于同一实验组内的相对对比
2. 不宜跨不同映射方式比较绝对数值
3. 不应被解读为"弱标签的数值真值"

主要评估指标仍然是 Spearman ρ（排序一致性）
```

### 7.4 实验结果汇报表格

**主要表格**（用于论文正文）:

| 实验组 | WoodScape ρ | External ρ | WoodScape Δρ | External Δρ |
|--------|-------------|------------|-------------|-------------|
| **s - Baseline** | - | 0.2177 | - | - |
| **s - Filtered** | - | 0.22xx | - | Δ vs Baseline |
| **S_full - Baseline** | 0.9919 | 0.7177 | +0.0166 | - |
| **S_full - Filtered** | 0.99xx | 0.73xx | - | Δ vs Baseline |

**辅助统计表**（附录）:

| 实验组 | WoodScape MAE | External MAE | External RMSE |
|--------|-------------|-------------|---------------|
| **s - Baseline** | 0.0199 | 2.2450 | 2.5637 |
| **s - Filtered** | 0.01xx | 2.1xxx | 2.4xxx |
| **S_full - Baseline** | 0.0199 | 0.7394 | 0.9065 |
| **S_full - Filtered** | 0.01xx | 0.7xxx | 0.9xxx |

---

## 八、训练配置与损失函数对齐

### 8.1 统一模型配置

```python
MODEL_CONFIG = {
    'backbone': 'resnet18',
    'tile_size': 8,
    'pretrained': True,
}
```

### 8.2 统一训练配置

```python
TRAINING_CONFIG = {
    'epochs': 40,
    'batch_size': 16,
    'lr': 3e-4,
    'weight_decay': 1e-2,
    'seed': 42,

    # 损失权重
    'lambda_glob': 1.0,      # 全局损失权重
    'mu_cons': 0.1,          # 一致性损失权重（启用时记录）
    'use_consistency_loss': True,  # 是否启用一致性损失
}
```

### 8.3 损失函数定义（与baseline对齐）

#### 7.3.1 Tile回归损失

```python
def compute_tile_loss(
    G_hat: torch.Tensor,  # [B, 4, 8, 8] 预测tile coverage
    G_star: torch.Tensor,  # [B, 4, 8, 8] 真实tile coverage
) -> torch.Tensor:
    """
    Tile回归损失 - 使用SmoothL1/L1

    与baseline一致，标签为连续覆盖率向量
    """
    # 确保归一化
    G_hat_norm = G_hat / (G_hat.sum(dim=1, keepdim=True) + 1e-12)
    G_star_norm = G_star / (G_star.sum(dim=1, keepdim=True) + 1e-12)

    # 使用SmoothL1损失
    loss_tile = F.smooth_l1_loss(G_hat_norm, G_star_norm)

    return loss_tile
```

**关键说明**: 标签G为连续覆盖率向量 `[B, 4, 8, 8]`，使用回归损失而非交叉熵

#### 7.3.2 全局回归损失

```python
def compute_global_loss(
    S_hat: torch.Tensor,  # [B] 预测severity
    S_star: torch.Tensor,  # [B] 真实severity
) -> torch.Tensor:
    """
    全局回归损失 - 使用SmoothL1/L1
    """
    loss_global = F.smooth_l1_loss(S_hat, S_star)
    return loss_global
```

#### 7.3.3 一致性损失

```python
def compute_consistency_loss(
    S_hat: torch.Tensor,     # [B] 全局头预测
    S_agg: torch.Tensor,     # [B] 聚合头预测
    mu_cons: float = 0.1,
) -> torch.Tensor:
    """
    一致性损失 - 约束 S_hat ≈ S_agg

    由 Ĝ 聚合得 Ṡ，约束 Ŝ ≈ Ṡ
    """
    loss_cons = F.smooth_l1_loss(S_hat, S_agg)
    return mu_cons * loss_cons
```

---

## 九、实验组设计（简化）

### 9.1 实验组矩阵

在"只对比 s 与 S_full_wgap_alpha50"的收敛策略下，每个目标各自跑同一套增强消融主线：

| 实验组 | s | S_full_wgap_alpha50 | 说明 | 优先级 |
|--------|---|---------------------|------|--------|
| **G1: Baseline** | Real only | Real only | 基线 | P0 |
| **G2: Filtered** | Real + Syn (硬约束+q_D+q_T) | Real + Syn (硬约束+q_D+q_T) | 质量筛选 | **P0** |
| **G3: Weighted** | Real + Syn (soft weight) | Real + Syn (soft weight) | 动态加权 | P1 |
| **G4: Naïve** | Real + Syn (无筛选) | Real + Syn (无筛选) | 对照（可选） | P2 |
| **G5: Syn Only** | Syn only | Syn only | 对照（可选） | P2 |

### 9.2 成功标准

| 指标 | Baseline | 目标 | 提升幅度 |
|------|----------|------|----------|
| **External ρ (S)** | 0.2177 | ≥ 0.2200 | +1.1% |
| **External ρ (S_full)** | 0.7177 | ≥ 0.7300 | +1.7% |
| **External MAE (S)** | 2.2450 | ≤ 2.1000 | -6.5% |
| **External MAE (S_full)** | 0.7394 | ≤ 0.7000 | -5.3% |

---

## 十、复现与产物清单

### 10.1 每次合成数据版本必产物

#### 9.1.1 清单文件: `manifest_syn.csv`

```csv
sample_id,clean_source,clean_video,clean_frame_id,mask_source,mask_id,c,pi,morph_type,morph_params,q_D,q_T,seed,severity_config_version
syn_000001,clean_train,video_001,000456,ws_Train,ws_000123,3,0.35,drop,"{size:15}",0.78,0.82,42,v1.0_wgap_alpha50
syn_000002,clean_train,video_002,000789,ws_Train,ws_000456,2,0.52,smear,"{length:50,angle:30}",0.65,0.71,43,v1.0_wgap_alpha50
```

#### 9.1.2 配置文件: `severity_config.json`

```json
{
  "version": "v1.0_wgap_alpha50",
  "description": "Converged Severity Score for SD enhancement experiments",
  "class_weights": [0.0, 0.15, 0.50, 1.0],
  "fusion_coeffs": {"alpha": 0.5, "beta": 0.4, "gamma": 0.1},
  "eta_trans": 0.9,
  "spatial": {"mode": "gaussian", "sigma": 0.5},
  "reference": "2025-02-09_two_weight_systems_validation.md",
  "creation_date": "2025-02-09"
}
```

#### 9.1.3 .npz文件内容

```python
np.savez_compressed(
    f'{sample_id}.npz',
    # 图像与标签
    image=image_synthetic.astype(np.uint8),        # [H, W, 3]
    mask=mask_synthetic.astype(np.uint8),          # [H, W] 单类别 mask (0 或 target_class)
    tile_cov=tile_coverage.astype(np.float32),     # [8, 8, 4]

    # 双目标标签
    s_simple=np.float32(s_score),                  # float: 1 - clean_ratio
    S_full=np.float32(S_score),                    # float: Severity Score

    # Severity组件（用于分析）
    S_op=np.float32(S_op_score),                   # Opacity-aware coverage
    S_sp=np.float32(S_sp_score),                   # Spatial weighted
    S_dom=np.float32(S_dom_score),                 # Dominance with transparent discount

    # L (Level) - 训练集自适应分箱
    L=int(L_value),                                # int ∈ {0, 1, 2, 3, -1}

    # 元数据字段（用于后续质量控制和实验追踪）
    target_class=int(target_class),                # 目标类别 c ∈ {1, 2, 3}
    severity_config_version="v1.0_wgap_alpha50",   # Severity 配置版本
    thresholds_source="version_specific",          # 阈值来源（missing_versioned 表示 L 不可用）
)
```

**说明**:
- `mask`: 单类别 mask，值为 0 或 target_class（1/2/3）
- `L = -1`: 表示阈值版本不匹配，L 值不可用，不应用于分层采样
- `target_class`: 用于控制类别分布和分析
- `thresholds_source`: 追踪 L 值的可靠性

### 10.2 每次训练实验必产物

#### 9.2.1 训练配置: `training_config.json`

```json
{
  "experiment_id": "exp001_S_filtered",
  "target": "S_full_wgap_alpha50",
  "data_config": {
    "real_source": "Train_Real",
    "syn_source": "Train_Syn_Filtered",
    "real_samples": 3200,
    "syn_samples": 5000,
    "sampling_strategy": "stratified",
  },
  "model_config": {
    "backbone": "resnet18",
    "tile_size": 8,
    "pretrained": true
  },
  "training_config": {
    "epochs": 40,
    "batch_size": 16,
    "lr": 0.0003,
    "weight_decay": 0.01,
    "lambda_glob": 1.0,
    "mu_cons": 0.1,
    "use_consistency_loss": true,
    "seed": 42
  },
  "quality_config": {
    "background_ssim_threshold": 0.95,
    "background_diff_threshold": 0.05,
    "q_D_threshold": 0.5,
    "q_T_threshold": 0.5,
    "use_q_filter": true
  },
  "severity_config": "v1.0_wgap_alpha50",
  "mapping_rule": "scale_to_1_5"
}
```

#### 9.2.2 训练日志: `training_log.csv`

```csv
epoch,train_loss,train_tile_mae,train_glob_mae,val_loss,val_tile_mae,val_glob_mae,ws_mae,ws_rho,ext_mae,ext_rho,lr
1,0.1234,0.0456,0.0234,0.1456,0.0512,0.0298,0.0221,0.9856,0.0003,0.0003
2,0.0987,0.0398,0.0189,0.1234,0.0445,0.0245,0.0198,0.9878,0.0003,0.0003
...
```

#### 9.2.3 评估结果: `evaluation_results.json`

```json
{
  "experiment_id": "exp001_S_filtered",
  "target": "S_full_wgap_alpha50",
  "woodscape_test": {
    "mae_s_hat": 0.0199,
    "rmse_s_hat": 0.0320,
    "rho_s_hat": 0.9919,
    "mae_s_agg": 0.0245,
    "rmse_s_agg": 0.0389,
    "rho_s_agg": 0.9752,
    "delta_rho": 0.0167
  },
  "external_test": {
    "mae_s_hat": 0.7394,
    "rmse_s_hat": 0.9065,
    "rho_s_hat": 0.7177,
    "mae_s_agg": 1.1575,
    "rmse_s_agg": 1.3651,
    "rho_s_agg": 0.6617,
    "delta_rho": 0.0560,
    "mapping_rule": "scale_to_1_5",
    "note": "MAE/RMSE are auxiliary statistics, not ground truth"
  },
  "evaluation_date": "2025-02-09",
  "random_seed": 42
}
```

### 10.3 产物目录结构

```
dataset/
├── my_external_test/                    # 自采外部测试集（统一命名）
│   └── test_ext.csv
├── clean_frames/                           # 自采干净帧
│   ├── train/
│   ├── val/
│   └── test/
├── woodscape_processed/
│   ├── meta/
│   │   ├── rebinned_thresholds.json      # rebinning阈值（仅Train）
│   │   └── labels_index_ablation.csv    # 标签索引
│   └── ...
└── synthetic_soiling_v1.0_wgap_alpha50/
    ├── manifest_syn.csv                    # 样本清单
    ├── severity_config.json               # Severity配置
    ├── generation_config.json             # 生成配置
    ├── npz/                               # .npz文件
    │   ├── syn_000001.npz
    │   └── ...
    ├── quality_report.md                  # 质量报告
    └── diagnosis/                         # 数据集级诊断
        ├── fid_score.json                 # FID分数
        └── severity_distribution.png       # 分布对比图

runs/
└── sd_enhancement/
    ├── exp001_s_baseline/
    │   ├── training_config.json
    │   ├── training_log.csv
    │   ├── evaluation_results.json
    │   └── ckpt_best.pth
    ├── exp002_S_filtered/
    ├── exp003_S_weighted/
    └── exp004_S_baseline/
    ├── exp005_S_filtered/
    └── exp006_S_weighted/

mask_bank/
├── train/                                 # MaskBank_Train
│   ├── manifest.csv
│   ├── statistics.csv
│   └── processed/
└── val/                                   # MaskBank_Val
    ├── manifest.csv
    ├── statistics.csv
    └── processed/

cycle_gan/
├── models/
│   ├── discriminator_B.pth               # 域判别器（用于q_D）
│   └── ...
└── logs/
    └── training_log.csv
```

---

## 十一、实施计划（修订）

### 11.1 阶段划分

| 阶段 | 任务 | 预计时间 | 产出 | 依赖 |
|------|------|----------|------|------|
| **阶段0** | 配置准备与数据划分 | 1-2天 | 配置文件、数据清单 | - |
| **阶段1** | 自采Clean数据处理 | 2-3天 | Clean_Train/Test/Val | - |
| **阶段2** | MaskBank_Train构建 | 2-3天 | MaskBank_Train | 阶段1 |
| **阶段3** | CycleGAN训练（可选） | 2-3天 | 域判别器 | 阶段2 |
| **阶段4** | 生成算法实现 | 5-7天 | 生成器代码 | 阶段2 |
| **阶段5** | 批量生成 | 2-3天 | Train_Syn (~10k) | 阶段4 |
| **阶段6** | q_D+q_T计算与筛选 | 2-3天 | Train_Syn_Filtered | 阶段5, 阶段3 |
| **阶段7** | 模型训练 | 7-10天 | 4个P0实验组 | 阶段6 |
| **阶段8** | 评估分析与报告 | 3-5天 | 实验报告 | 阶段7 |
| **总计** | - | **26-41天** | - | - |

**时间估算说明**:
- 基础版本（不含CycleGAN）: 26-32天
- 完整版本（含CycleGAN）: 32-41天

### 11.2 关键里程碑

- **M0**: 完成配置准备与数据划分 (Day 2)
- **M1**: 完成MaskBank_Train构建 (Day 7)
- **M2**: 完成生成算法实现 (Day 14)
- **M3**: 完成数据生成和质量检查 (Day 19)
- **M4**: 完成P0实验组训练 (Day 29)
- **M5**: 完成评估分析报告 (Day 34)

---

## 十二、风险与缓解措施

### 12.1 主要风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 自采clean数据不足 | 需要使用公开数据集 | 中 | 优先准备自采数据，准备备选方案 |
| CycleGAN训练不稳定 | q_D计算不准确 | 中 | 可选，简化为特征空间距离 |
| 生成质量不合格 | 合格样本率 < 70% | 中 | 调整生成算法或筛选阈值 |
| 性能提升不明显 | External ρ 无提升 | 中 | 分析原因，尝试其他增强方式 |
| 过拟合合成数据 | WoodScape ρ 下降 | 中 | 加强正则化，调整数据配比 |

### 12.2 失败预案

| 失败场景 | 检测方法 | 应对策略 |
|----------|----------|----------|
| 生成质量不合格 | 人工评分 < 3.0 或 合格率 < 70% | 改进生成算法或调整阈值 |
| 合成数据无效果 | External ρ 无提升或下降 | 分析原因，尝试其他增强方式 |
| 过拟合合成数据 | WoodScape ρ 下降 > 0.005 | 加强正则化，调整数据配比 |
| 计算资源不足 | 训练时间过长 | 优先完成P0实验，合并实验组 |

---

## 十三、附录

### A. 参考文档

- [Baseline实验完整性评估](2025-02-09_baseline_completeness_evaluation.md)
- [两权重系统验证](2025-02-09_two_weight_systems_validation.md)
- [权重敏感性分析](2025-02-08_weight_sensitivity_analysis.md)
- [时序模块设计](temporal_module_design.md)
- [LoRA训练v1实验记录](2026-02-17_lora_training_v1.md) - SD2.1 Base + LoRA训练详情

### B. v4.0 修订说明

**v4.0主要修正** (2026-02-18):
1. **新增第三章：两阶段LoRA方案** - SD2.1 Base训练 + SD2.1 Inpainting推理
2. **更新Clean数据集描述** - 使用 `dataset/my_clean_frames_4by3/` (5,258 samples)
3. **明确LoRA技术验证** - 128 keys成功加载到Inpainting模型
4. **补充Caption控制** - 自然语言锚点系统
5. **补充类别不平衡处理** - Stratified sampling策略
6. **补充证据面板设计** - Spill Rate、可控性网格图、对比面板
7. **与论文实验主线衔接** - 正交可分的实验设计

**v3.0主要修正**:
1. **统一数据集描述**: External Test 明确为自采弱标注测试集
2. **指标主次定位**: Spearman ρ 为主指标，MAE/RMSE 为辅助统计
3. **修正FID实现**: 明确为数据集级诊断指标，不用于单样本筛选
4. **优先自采clean域**: 避免引入额外风格差异
5. **补齐MaskBank构建细节**: 标准化流程、形态参数提取、覆盖率采样
6. **增加q_D+q_T训练成本**: CycleGAN 2-3天，Teacher推理可忽略
7. **确保损失函数一致**: Tile使用回归损失，避免交叉熵冲突
8. **补充报表分离工作量**: Test_Syn独立报表维护

**关键改进**:
- 逻辑成立性：避免概念混淆和实现层面的矛盾
- 工程可实现性：修正了不切实际的指标定义
- 复现完整性：补充了必要的实现细节和产物清单
- LoRA方案可行性：验证了Base训练+Inpainting推理的技术路径

---

**文档版本**: v4.0
**修订日期**: 2026-02-18
**设计者**: Claude Code Assistant
**审核状态**: 待审核
**下一步**: 完善LoRA+Inpainting生成器，准备生成Test_Syn数据集

**方案类型**: 两阶段LoRA + SD Inpainting
**训练状态**: LoRA v1训练完成 (3000 steps)
**推理状态**: LoRA+Inpainting验证通过
