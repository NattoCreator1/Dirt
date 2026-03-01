#!/usr/bin/env python3
"""
准备手动标注素材

将 530 张 SD 脏污图像复制到标注目录，生成标注说明文件
"""

import os
import shutil
import pandas as pd
from pathlib import Path

# 路径配置
dirt_dir = "dataset/sd_temporal_training/dirt_layers"
manifest_path = "dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv"
output_dir = "dataset/sd_temporal_training/for_annotation"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载清单
df = pd.read_csv(manifest_path, encoding='utf-8-sig')

print(f"准备 {len(df)} 张图像用于手动标注...")
print(f"输出目录: {output_dir}")

# 创建标注记录文件
annotation_records = []

# 复制图像
for idx, row in df.iterrows():
    filename = row['filename']

    # 源路径（可能有 _640x480 后缀）
    src_path = os.path.join(dirt_dir, filename)
    if not os.path.exists(src_path):
        if filename.endswith('.png'):
            base = filename[:-4]
            src_path = os.path.join(dirt_dir, base + '_640x480.png')

    if not os.path.exists(src_path):
        print(f"警告: 找不到 {filename}")
        continue

    # 目标路径（简化文件名）
    new_filename = f"{idx:04d}.png"
    dst_path = os.path.join(output_dir, new_filename)

    shutil.copy2(src_path, dst_path)

    # 记录映射关系
    annotation_records.append({
        'annotation_file': new_filename,
        'original_file': filename,
        'mask_class': row['mask_class'],
        'mask_class_name': row['mask_class_name'],
        'view': row['view'],
        'coverage': row['coverage'],
        'spill_rate': row['spill_rate'],
    })

# 保存映射表
mapping_df = pd.DataFrame(annotation_records)
mapping_path = os.path.join(output_dir, "file_mapping.csv")
mapping_df.to_csv(mapping_path, index=False, encoding='utf-8-sig')

# 创建标注说明文件
readme_path = os.path.join(output_dir, "README_标注说明.txt")
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write("""
手动标注说明
============

任务：为每张 SD 生成的脏污图像标注真正的脏污区域

标注方法：
--------
1. 使用 Photoshop/GIMP 等图像编辑软件打开图像
2. 新建一个图层，用白色填充真正脏污的区域
3. 非脏污区域保持黑色
4. 将标注图层另存为独立的 PNG 文件（与原图同名的 _mask.png）

输出格式：
--------
原图: 0000.png
标注: 0000_mask.png (黑白图像，白色=脏污，黑色=透明)

标注规范：
--------
- 白色 (#FFFFFF): 脏污区域（将保留 SD 生成的脏污纹理）
- 黑色 (#000000): 透明区域（将显示干净背景）
- 边缘可以适当羽化，使合成更自然

文件对应关系：
--------
详见 file_mapping.csv
- annotation_file: 标注文件名（0000.png）
- original_file: 原始文件名
- mask_class: 脏污类型（1=Transparent, 2=Semi, 3=Opaque）
- coverage: 覆盖度参考值

完成标注后：
-----------
将所有 _mask.png 文件放在同一目录下，然后运行转换脚本：
  python scripts/temporal/compose_with_manual_masks.py
""")

print(f"\\n完成！")
print(f"  - 复制了 {len(annotation_records)} 张图像到 {output_dir}/")
print(f"  - 映射表: {mapping_path}")
print(f"  - 标注说明: {readme_path}")
