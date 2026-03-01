# 手动标注脏污区域工作流程

## 问题背景

SD inpainting 生成的脏污纹理位置与 rgbLabels mask 不完全一致，导致合成质量下降。

**解决方案**：手动标注每张 SD 脏污图像中的真正脏污区域。

---

## 工作流程

### 步骤 1：准备标注素材

```bash
python scripts/temporal/prepare_for_manual_annotation.py
```

输出：`dataset/sd_temporal_training/for_annotation/`
- `0000.png` ~ `0529.png` - 530 张 SD 生成的脏污图像
- `file_mapping.csv` - 文件名映射表
- `README_标注说明.txt` - 标注说明

---

### 步骤 2：手动标注脏污区域

**工具**：Photoshop、GIMP 或其他图像编辑软件

**操作方法**：

1. 打开图像（如 `0000.png`）
2. 新建图层，使用以下规则绘制 mask：
   - **白色 (#FFFFFF)** = 脏污区域（保留 SD 生成的纹理）
   - **黑色 (#000000)** = 透明区域（显示干净背景）
3. 边缘可以适当羽化，使合成更自然
4. 导出标注图层为 `0000_mask.png`

**标注规范**：

| 区域 | 颜色 | Alpha 值 | 说明 |
|------|------|----------|------|
| 脏污 | 白色 | 255 | 保留 SD 生成的脏污纹理 |
| 透明 | 黑色 | 0 | 显示干净背景 |
| 边缘 | 灰度 | 1-254 | 实现羽化效果 |

**示例**：
```
原图: 0000.png
标注: 0000_mask.png
```

---

### 步骤 3：批量转换为 RGBA

```bash
python scripts/temporal/compose_with_manual_masks.py
```

输出：`dataset/sd_temporal_training/dirt_layers_manual_rgba/`
- 530 张 RGBA 脏污层
- RGB 来自 SD 生成的脏污图像
- Alpha 来自手动标注的 mask

---

### 步骤 4：时序合成

```bash
python scripts/temporal/compose_sd_temporal.py \
    --dirt_layer_dir dataset/sd_temporal_training/dirt_layers_manual_rgba \
    --dirt_manifest dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv \
    --sequence_manifest dataset/temporal_sequences/metadata/sequence_manifest.csv \
    --sequence_base_dir dataset/temporal_sequences/raw_sequences \
    --output_dir dataset/sd_temporal_training/composed_sequences \
    --metadata_dir dataset/sd_temporal_training/composition_metadata \
    --num_sequences 1000 \
    --reuse_per_bg 3
```

---

## 批处理技巧

### Photoshop Action 批量处理

1. 创建 Action：
   - 打开图像
   - 新建图层
   - 使用画笔标注脏污区域
   - 隐藏原图图层
   - 导出标注图层为 `{filename}_mask.png`
   - 关闭文件

2. 批量运行：
   - 文件 → 自动 → 批处理
   - 选择源文件夹和目标文件夹

### 参考信息

标注时可参考 `file_mapping.csv` 中的信息：
- `mask_class`: 脏污类型（1=Transparent, 2=Semi, 3=Opaque）
- `coverage`: 覆盖度参考值
- `spill_rate`: Spill rate

---

## 验证

完成后可以运行测试验证合成效果：

```bash
python scripts/temporal/test_multiple_composition.py
```

修改脚本中的 `dirt_dir` 指向 `dataset/sd_temporal_training/dirt_layers_manual_rgba`

---

## 文件结构

```
dataset/sd_temporal_training/
├── for_annotation/              # 标注素材
│   ├── 0000.png ~ 0529.png
│   ├── 0000_mask.png ~ 0529_mask.png  # 手动标注的 mask
│   ├── file_mapping.csv
│   └── README_标注说明.txt
│
├── dirt_layers_manual_rgba/     # 转换后的 RGBA 脏污层
│   ├── {original_filename}.png  # 530 个 RGBA 文件
│
└── composed_sequences/          # 最终合成序列
    └── ...
```
