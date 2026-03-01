#!/usr/bin/env python3
"""
测试 RGBA 转换和合成效果

验证：
1. RGBA 脏污层的 alpha 通道是否正确（黑色区域=透明）
2. 合成效果是否符合预期（干净帧背景可见）
"""

import os
import cv2
import numpy as np

# 测试文件路径
dirt_rgba_path = "dataset/sd_temporal_training/dirt_layers_rgba/1751703857935_1751704212892_01_merged_seg1_final_level_max5_20250705162827_20250705162835_f_t0000000000_f000000000_2521_FV_s000.png"
rgblabel_path = "dataset/woodscape_raw/train/rgbLabels/2521_FV.png"

# 找一个干净背景帧
bg_dir = "dataset/temporal_sequences/raw_sequences/f"
bg_seq_dirs = [d for d in os.listdir(bg_dir) if os.path.isdir(os.path.join(bg_dir, d))]
if bg_seq_dirs:
    bg_frame_path = os.path.join(bg_dir, bg_seq_dirs[0], "frame_0000.jpg")
else:
    print("找不到干净背景帧")
    bg_frame_path = None

print("=" * 60)
print("RGBA 转换测试")
print("=" * 60)

# 1. 检查 RGBA 脏污层
print("\n1. 检查 RGBA 脏污层:")
dirt_img = cv2.imread(dirt_rgba_path, cv2.IMREAD_UNCHANGED)
print(f"   形状: {dirt_img.shape}")
if dirt_img.shape[2] == 4:
    alpha = dirt_img[:, :, 3]
    print(f"   Alpha 范围: [{alpha.min()}, {alpha.max()}]")
    print(f"   Alpha=0 (透明) 占比: {(alpha == 0).sum() / alpha.size * 100:.1f}%")
    print(f"   Alpha=255 (不透明) 占比: {(alpha == 255).sum() / alpha.size * 100:.1f}%")
else:
    print(f"   错误: 不是 RGBA 格式！通道数: {dirt_img.shape[2]}")

# 2. 检查 rgbLabels
print("\n2. 检查 rgbLabels:")
rgblabel = cv2.imread(rgblabel_path)
rgblabel_rgb = cv2.cvtColor(rgblabel, cv2.COLOR_BGR2RGB)
if rgblabel is not None:
    # Resize 到 640x480 以便对比
    rgblabel_small = cv2.resize(rgblabel_rgb, (640, 480), interpolation=cv2.INTER_NEAREST)

    black = (rgblabel_small == [0, 0, 0]).all(axis=2)
    print(f"   黑色 (干净) 区域占比: {black.sum() / black.size * 100:.1f}%")

    # 验证对应关系
    if dirt_img.shape[2] == 4:
        # 检查 rgbLabels 黑色区域是否对应 alpha=0
        alpha_zero_in_black = (alpha[black] == 0).sum()
        total_black = black.sum()
        print(f"   rgbLabels 黑色区域中 alpha=0 的比例: {alpha_zero_in_black / max(1, total_black) * 100:.1f}%")

# 3. 测试合成效果
if bg_frame_path and os.path.exists(bg_frame_path):
    print("\n3. 测试合成效果:")
    bg = cv2.imread(bg_frame_path)
    if bg is not None:
        print(f"   背景帧形状: {bg.shape}")

        # 确保尺寸一致
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
        os.makedirs("scripts/temporal/test_output", exist_ok=True)
        cv2.imwrite("scripts/temporal/test_output/bg_original.jpg", bg)
        cv2.imwrite("scripts/temporal/test_output/dirt_rgba.png", dirt_resized)
        cv2.imwrite("scripts/temporal/test_output/composed.jpg", composed)
        print(f"   已保存测试结果到: scripts/temporal/test_output/")
        print(f"   - bg_original.jpg: 原始干净背景")
        print(f"   - dirt_rgba.png: RGBA 脏污层")
        print(f"   - composed.jpg: 合成结果")

        # 检查合成结果
        # 在 alpha=0 的区域，合成结果应该接近原始背景
        alpha_mask = dirt_resized[:, :, 3]
        diff_in_transparent = np.abs(composed.astype(np.float32) - bg.astype(np.float32))
        diff_in_transparent = diff_in_transparent[alpha_mask == 0]
        print(f"   透明区域与原图的差异 (平均): {diff_in_transparent.mean():.2f} (应该接近 0)")

print("\n" + "=" * 60)
