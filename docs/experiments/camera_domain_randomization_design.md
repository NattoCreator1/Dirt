# 相机域随机化 (Camera Domain Randomization) 设计方案

**目标**: 在SD合成数据上模拟真实的相机成像链差异，覆盖External Test与Woodscape之间的域差

---

## 一、问题分析

### 1.1 当前域差来源分析

External Test 与 Woodscape 的域差可能包括：

| 维度 | Woodscape | External | 差异 |
|------|-----------|----------|------|
| **相机类型** | 车载相机 | 多种相机 | 传感器响应不同 |
| **ISP处理** | 未知但固定 | 各不相同 | 颜色映射、噪声处理 |
| **压缩** | 可能压缩 | 不同编码 | 伪影、质量损失 |
| **环境条件** | 受控 | 野外 | 曝光、天气变化 |
| **镜头** | 特定镜头 | 各种镜头 | 晕畸变、暗角 |

### 1.2 当前SD合成的域覆盖

| 可变因素 | 当前状态 | 覆盖External域差 |
|---------|---------|---------------|
| 污渍纹理 | ✓ LoRA学习 | ✓ 部分 |
| 污渍位置 | ✓ 随机mask | ✓ 部分 |
| 相机域 | ✗ 固定 | ✗ 缺失 |

---

## 二、相机域随机化策略

### 2.1 颜色与曝光随机化

**目的**：模拟不同相机和成像条件下的色彩表现

```python
def randomize_color_and_exposure(image):
    """
    颜色和曝光随机化
    """
    # 1. Gamma 变化 (模拟不同ISP的gamma校正)
    gamma = np.random.uniform(0.8, 1.2)
    image = np.power(image / 255.0, gamma) * 255.0

    # 2. 色温偏移 (模拟白平衡差异)
    # 在RGB空间添加轻微的色调偏移
    temp_shift = np.random.uniform(-0.05, 0.05)  # -5000K to +5000K
    image = apply_color_temperature(image, temp_shift)

    # 3. 整体曝光调整 (模拟曝光差异)
    exposure = np.random.uniform(0.8, 1.2)
    image = np.clip(image * exposure, 0, 255)

    # 4. 局部过曝/欠曝 (模拟复杂光照)
    # 随机选择区域进行曝光调整
    if np.random.random() < 0.3:  # 30%概率
        # 创建局部曝光掩膜
        mask = create_random_blob_mask(image.shape[:2])
        exposure_factor = np.random.choice([0.7, 1.3])
        image = apply_local_exposure(image, mask, exposure_factor)

    return image
```

### 2.2 噪声添加

**目的**：模拟不同传感器和ISO条件下的噪声

```python
def add_realistic_noise(image):
    """
    添加真实的相机噪声
    """
    h, w = image.shape[:2]

    # 1. Shot noise (Poisson-Gaussian混合模型)
    # 模拟光子计数噪声
    iso_gain = np.random.uniform(1.0, 3.0)  # ISO增益

    # 读取噪声
    shot_noise = np.random.poisson(image * iso_gain) / iso_gain - image
    read_noise = np.random.normal(0, 5.0, image.shape)  # 读取电路热噪声

    noise_image = image + shot_noise + read_noise

    # 2. 色噪 (高频彩色噪声)
    if np.random.random() < 0.5:
        # 在高频区域添加色噪
        noise = np.random.normal(0, 3.0, image.shape)
        noise = cv2.GaussianBlur(noise, (3, 3), 0)
        noise_image = noise_image + noise * 0.5

    return np.clip(noise_image, 0, 255).astype(np.uint8)
```

### 2.3 模糊随机化

**目的**：模拟不同对焦条件和运动模糊

```python
def add_random_blur(image):
    """
    添加随机模糊
    """
    blur_type = np.random.choice(['defocus', 'motion', 'none'],
                                    p=[0.3, 0.3, 0.4])

    if blur_type == 'defocus':
        # 散焦模糊（模拟对焦不准）
        kernel_size = np.random.choice([3, 5, 7])
        sigma = np.random.uniform(0.5, 2.0)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    elif blur_type == 'motion':
        # 运动模糊（模拟相机抖动或物体运动）
        kernel_size = np.random.randint(5, 15)
        angle = np.random.uniform(0, 180)

        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1
        kernel = cv2.warpAffine(kernel,
                                 cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2),
                                                         angle, 1.0),
                                 (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()

        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = image

    return blurred
```

### 2.4 压缩伪影

**目的**：模拟JPEG压缩和编码链损失

```python
def add_compression_artifacts(image):
    """
    添加压缩伪影
    """
    # 随机JPEG质量因子
    quality = np.random.randint(70, 95)

    # 编码再解码
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image.astype(np.uint8), encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # 添加色度子采样伪影（模拟4:2:0采样）
    if np.random.random() < 0.3:
        # 下采样再上采样
        h, w = image.shape[:2]
        small = cv2.resize(decoded, (w//2, h//2), interpolation=cv2.INTER_AREA)
        decoded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return decoded
```

### 2.5 镜头效果

**目的**：模拟光学镜头的各种效应

```python
def add_lens_effects(image):
    """
    添加镜头效果
    """
    # 1. Vignetting (暗角)
    if np.random.random() < 0.5:
        h, w = image.shape[:2]
        Y, X = np.ogrid(h, w)
        # 创建vignette效果
        center_x, center_y = w/2, h/2
        vignette_strength = np.random.uniform(0.1, 0.4)

        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)

        vignette = 1 - vignette_strength * (distance / max_distance)**2
        vignette = np.clip(vignette, 0.2, 1.0)

        # 应用到RGB
        image = image * vignette[..., np.newaxis]

    # 2. Chromatic Aberration (色差)
    if np.random.random() < 0.3:
        # 简化的色差模拟（RGB通道不同程度偏移）
        image = add_chromatic_aberration(image)

    return np.clip(image, 0, 255).astype(np.uint8)
```

### 2.6 分辨率链随机化

**目的**：模拟不同的ISP处理链

```python
def random_resolution_chain(image):
    """
    随机分辨率处理链
    """
    # 随机下采样倍数
    downscale_factor = np.random.choice([2, 4])
    h, w = image.shape[:2]

    # 下采样
    small = cv2.resize(image, (w//downscale_factor, h//downscale_factor),
                      interpolation=cv2.INTER_AREA)

    # 随机上采样方法
    up_methods = [
        (cv2.INTER_LINEAR, "linear"),
        (cv2.INTER_CUBIC, "cubic"),
        (cv2.INTER_LANCZOS4, "lanczos"),
    ]
    up_method, _ = np.random.choice(up_methods, p=[0.4, 0.3, 0.3])

    # 上采样
    restored = cv2.resize(small, (w, h), interpolation=up_method)

    return restored
```

---

## 三、完整实现

### 3.1 统一接口

```python
class CameraDomainRandomizer:
    """相机域随机化器"""

    def __init__(self,
                 color_prob=0.8,      # 颜色/曝光随机化概率
                 noise_prob=0.7,      # 噪声添加概率
                 blur_prob=0.5,       # 模糊添加概率
                 compression_prob=0.6, # 压缩伪影概率
                 lens_prob=0.5,       # 镜头效果概率
                 resolution_prob=0.5): # 分辨率随机化概率

        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.compression_prob = compression_prob
        self.lens_prob = lens_prob
        self.resolution_prob = resolution_prob

    def __call__(self, image):
        """
        对图像应用随机相机域变换

        Args:
            image: RGB图像, HxWx3, uint8, [0, 255]

        Returns:
            变换后的图像
        """
        # 1. 颜色/曝光
        if np.random.random() < self.color_prob:
            image = self.randomize_color_and_exposure(image)

        # 2. 噪声
        if np.random.random() < self.noise_prob:
            image = self.add_realistic_noise(image)

        # 3. 模糊
        if np.random.random() < self.blur_prob:
            image = self.add_random_blur(image)

        # 4. 压缩伪影
        if np.random.random() < self.compression_prob:
            image = self.add_compression_artifacts(image)

        # 5. 镜头效果
        if np.random.random() < self.lens_prob:
            image = self.add_lens_effects(image)

        # 6. 分辨率链
        if np.random.random() < self.resolution_prob:
            image = self.random_resolution_chain(image)

        return np.clip(image, 0, 255).astype(np.uint8)
```

### 3.2 集成到SD生成流程

```python
# 在现有SD生成流程中的集成点

# 原流程:
# Clean Frame + Mask → LoRA Inpainting → Dirty Image

# 新流程:
# Clean Frame + Mask → LoRA Inpainting → Dirty Image → **Camera Domain Randomizer** → Final Image
```

---

## 四、实施步骤

### Phase 1: 基础实现 (1-2天)

1. 实现各个随机化函数
2. 创建CameraDomainRandomizer类
3. 单元测试各个模块

### Phase 2: 集成和调优 (2-3天)

1. 集成到SD生成流程
2. 调整随机化强度和概率
3. 视觉检查生成结果

### Phase 3: 批量生成 (1天)

1. 对现有的989张筛选样本进行相机域随机化
2. 或者对8960张原始SD图像进行相机域随机化后重新筛选

### Phase 4: 训练验证 (1天)

1. 使用相机域随机化后的数据重新训练
2. 评估External Test性能
3. 对比分析改进效果

---

## 五、参数建议

### 5.1 随机化强度

| 变换 | 参数范围 | 说明 |
|------|---------|------|
| Gamma | 0.8-1.2 | 温和调整 |
| 色温偏移 | ±0.05 | 轻微色调变化 |
| ISO增益 | 1.0-3.0 | 模拟不同ISO |
| JPEG质量 | 70-95 | 有损压缩 |
| 模糊核 | 3-15 | 中等强度 |

### 5.2 应用概率

建议初始使用中等概率，避免过度随机化：
- color_prob: 0.8
- noise_prob: 0.7
- blur_prob: 0.5
- compression_prob: 0.6
- lens_prob: 0.5
- resolution_prob: 0.5

---

## 六、预期效果与验证

### 6.1 预期改进

1. **跨域泛化**：External ρ 提升
2. **鲁棒性**：模型对成像条件变化更不敏感
3. **标尺一致性**：减少预测值整体下移

### 6.2 验证方法

1. **定量验证**：External Test Spearman ρ
2. **定性验证**：可视化检查生成的图像
3. **消融验证**：单独测试每个随机化模块的贡献

---

## 七、与现有工作的整合

### 7.1 与筛选策略的配合

相机域随机化后，需要调整筛选策略：

1. **保留几何一致性筛选**（spill低、mask adherence好）
2. **放宽预测一致性要求**：因为域随机化后baseline预测会更不准
3. **确保数据多样性**：按S值、模糊类型等分层抽样

### 7.2 与alpha形态软化的配合

文档提到的4.2策略可以一并实施：
- 相机域随机化：覆盖成像链域差
- Alpha形态软化：解决视觉-标签不匹配
- 两者结合：更真实的合成数据

---

## 八、后续实验计划

1. **Baseline测试**：先对现有989张样本应用相机域随机化，观察效果
2. **全量生成**：对8960张原始SD图像重新生成（带域随机化）
3. **对比实验**：域随机化前后的跨域性能对比
4. **消融实验**：单独测试各个随机化模块的贡献
