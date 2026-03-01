#!/usr/bin/env python3
"""
测试多个 RGBA 脏污层的合成效果

随机选择多个脏污层与干净帧进行合成，验证效果
"""

import os
import cv2
import numpy as np
import pandas as pd

# 路径配置
dirt_dir = "dataset/sd_temporal_training/dirt_layers_rgba"
bg_dir = "dataset/temporal_sequences/raw_sequences"
manifest_path = "dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv"
output_dir = "scripts/temporal/test_composition_multiple"

os.makedirs(output_dir, exist_ok=True)

# 加载清单
df = pd.read_csv(manifest_path, encoding='utf-8-sig')

print("=" * 60)
print(f"测试 {min(10, len(df))} 个脏污层的合成效果")
print("=" * 60)

# 找一些干净背景帧
bg_frames = {}
for view in ['f', 'lf', 'rf', 'lr', 'r', 'rr']:
    view_dir = os.path.join(bg_dir, view)
    if os.path.exists(view_dir):
        seq_dirs = [d for d in os.listdir(view_dir) if os.path.isdir(os.path.join(view_dir, d))][:3]
        for seq_dir in seq_dirs:
            frame_path = os.path.join(view_dir, seq_dir, "frame_0000.jpg")
            if os.path.exists(frame_path):
                if view not in bg_frames:
                    bg_frames[view] = frame_path
                    break

print(f"\n找到干净背景帧: {list(bg_frames.keys())}")

# 测试多个样本
test_count = 0
max_tests = 10

for idx, row in df.iterrows():
    if test_count >= max_tests:
        break

    filename = row['filename']
    dirt_path = os.path.join(dirt_dir, filename)

    if not os.path.exists(dirt_path):
        if filename.endswith('.png'):
            base = filename[:-4]
            dirt_path = os.path.join(dirt_dir, base + '_640x480.png')

    if not os.path.exists(dirt_path):
        continue

    # 读取脏污层
    dirt_img = cv2.imread(dirt_path, cv2.IMREAD_UNCHANGED)
    if dirt_img is None or dirt_img.shape[2] != 4:
        continue

    # 选择合适的背景帧
    view = row['view'].lower()
    view_map = {'fv': 'f', 'rv': 'r', 'mvl': 'lf', 'mvr': 'lr'}
    bg_view = view_map.get(view, view)

    if bg_view not in bg_frames:
        bg_view = list(bg_frames.keys())[0]

    bg_path = bg_frames[bg_view]
    bg = cv2.imread(bg_path)

    if bg is None:
        continue

    # Resize 确保尺寸一致
    if dirt_img.shape[:2] != bg.shape[:2]:
        dirt_resized = cv2.resize(dirt_img, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        dirt_resized = dirt_img

    # Alpha blending
    alpha = dirt_resized[:, :, 3:4].astype(np.float32) / 255.0
    foreground = dirt_resized[:, :, :3].astype(np.float32)
    bg_float = bg.astype(np.float32)

    composed = (1 - alpha) * bg_float + alpha * foreground
    composed = np.clip(composed, 0, 255).astype(np.uint8)

    # 保存结果
    base_name = filename.replace('.png', '')
    output_base = os.path.join(output_dir, f"{test_count:02d}_{base_name}")

    cv2.imwrite(f"{output_base}_bg.jpg", bg)
    cv2.imwrite(f"{output_base}_dirt.png", dirt_resized)
    cv2.imwrite(f"{output_base}_composed.jpg", composed)

    # 统计
    alpha_channel = dirt_resized[:, :, 3]
    transparent_ratio = (alpha_channel == 0).sum() / alpha_channel.size * 100

    print(f"\n样本 {test_count + 1}: {filename[:50]}...")
    print(f"  视角: {row['view']}, 脏污类型: {row['mask_class_name']}")
    print(f"  透明区域: {transparent_ratio:.1f}%, 不透明区域: {100-transparent_ratio:.1f}%")

    test_count += 1

print(f"\n" + "=" * 60)
print(f"测试完成! 共 {test_count} 个样本")
print(f"结果保存在: {output_dir}/")
print("=" * 60)
