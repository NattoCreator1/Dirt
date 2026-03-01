#!/usr/bin/env python3
"""
批量测试残差层提取方法，生成多样本对比图

对比方法A（直接叠加）与方法B（残差叠加）在多个样本上的效果
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# 路径配置
sd_dir = "sd_scripts/lora/accepted_20260224_164715/resized_640x480"
rgblabels_dir = "dataset/woodscape_raw/train/rgbLabels"
bg_base_dir = "dataset/temporal_sequences/raw_sequences"
manifest_path = "dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv"

output_dir = "scripts/temporal/test_residual_multiple"
os.makedirs(output_dir, exist_ok=True)

# 加载清单
df = pd.read_csv(manifest_path, encoding='utf-8-sig')

# 找干净背景帧
bg_frames = {}
for view in ['f', 'lf', 'rf', 'lr', 'r', 'rr']:
    view_dir = os.path.join(bg_base_dir, view)
    if os.path.exists(view_dir):
        seq_dirs = [d for d in os.listdir(view_dir) if os.path.isdir(os.path.join(view_dir, d))][:3]
        for seq_dir in seq_dirs:
            frame_path = os.path.join(view_dir, seq_dir, "frame_0000.jpg")
            if os.path.exists(frame_path):
                if view not in bg_frames:
                    bg_frames[view] = frame_path
                    break

def create_degraded_base(image):
    """创建降语义底图"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (41, 41), 10)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

def get_mask_id(filename):
    """从文件名提取 mask_id"""
    base = filename.replace('_640x480.png', '').replace('.png', '')
    parts = base.rsplit('_', 3)
    if len(parts) < 4:
        return None
    frame_id = parts[-3]
    view = parts[-2]
    return f"{frame_id}_{view}"

def gradient_energy(img):
    """计算梯度能量"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2).mean()

print("=" * 70)
print("残差层方法 - 多样本对比测试")
print("=" * 70)

# 测试多个样本
max_samples = 15
results = []

for idx, row in df.iterrows():
    if idx >= max_samples:
        break

    filename = row['filename']
    sd_path = os.path.join(sd_dir, filename)

    if not os.path.exists(sd_path):
        if filename.endswith('.png'):
            base = filename[:-4]
            sd_path = os.path.join(sd_dir, base + '_640x480.png')

    if not os.path.exists(sd_path):
        continue

    # 读取 SD 图像
    sd_image = cv2.imread(sd_path)
    if sd_image is None:
        continue

    # 创建降语义底图
    degraded_base = create_degraded_base(sd_image)

    # 提取残差
    residual = sd_image.astype(np.float32) - degraded_base.astype(np.float32)

    # 读取 rgbLabels mask
    mask_id = get_mask_id(filename)
    alpha_mask = None

    if mask_id:
        rgblabel_path = os.path.join(rgblabels_dir, f"{mask_id}.png")
        if os.path.exists(rgblabel_path):
            rgblabel = cv2.imread(rgblabel_path)
            rgblabel_rgb = cv2.cvtColor(rgblabel, cv2.COLOR_BGR2RGB)
            rgblabel_small = cv2.resize(rgblabel_rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
            alpha_mask = (rgblabel_small.sum(axis=2) > 0).astype(np.float32)[:, :, np.newaxis]

    # 如果没有 mask，使用残差强度生成
    if alpha_mask is None:
        residual_strength = np.abs(residual).sum(axis=2)
        alpha_mask = (residual_strength > 20).astype(np.float32)[:, :, np.newaxis]

    # 找对应的干净背景
    view = row['view'].lower()
    view_map = {'fv': 'f', 'rv': 'r', 'mvl': 'lf', 'mvr': 'lr'}
    bg_view = view_map.get(view, 'f')

    bg_path = bg_frames.get(bg_view, list(bg_frames.values())[0])
    bg_frame = cv2.imread(bg_path)

    if bg_frame is None:
        continue

    # 确保尺寸一致
    if bg_frame.shape[:2] != sd_image.shape[:2]:
        bg_frame = cv2.resize(bg_frame, (sd_image.shape[1], sd_image.shape[0]))

    # 方法A: 直接叠加
    composed_a = ((1 - alpha_mask) * bg_frame.astype(np.float32) +
                  alpha_mask * sd_image.astype(np.float32))
    composed_a = np.clip(composed_a, 0, 255).astype(np.uint8)

    # 方法B: 残差叠加
    composed_b = bg_frame.astype(np.float32) + alpha_mask * residual
    composed_b = np.clip(composed_b, 0, 255).astype(np.uint8)

    # 计算指标
    ge_bg = gradient_energy(bg_frame)
    ge_a = gradient_energy(composed_a)
    ge_b = gradient_energy(composed_b)

    # 在脏污区域计算与背景的差异
    dirty_mask = (alpha_mask[:, :, 0] > 0.5)
    if dirty_mask.sum() > 0:
        diff_a = np.abs(composed_a.astype(np.float32) - bg_frame.astype(np.float32))
        diff_b = np.abs(composed_b.astype(np.float32) - bg_frame.astype(np.float32))

        diff_in_dirt_a = diff_a[dirty_mask].mean()
        diff_in_dirt_b = diff_b[dirty_mask].mean()

        # 在透明区域计算差异（应该接近0）
        clean_mask = ~dirty_mask
        leak_a = diff_a[clean_mask].mean()
        leak_b = diff_b[clean_mask].mean()
    else:
        diff_in_dirt_a = diff_in_dirt_b = leak_a = leak_b = 0

    results.append({
        'idx': idx,
        'filename': filename[:40],
        'mask_class': row['mask_class_name'],
        'dirt_ratio': alpha_mask.mean() * 100,
        'ge_bg': ge_bg,
        'ge_a': ge_a,
        'ge_b': ge_b,
        'ge_ratio_a': ge_a / max(ge_bg, 0.1),
        'ge_ratio_b': ge_b / max(ge_bg, 0.1),
        'diff_dirt_a': diff_in_dirt_a,
        'diff_dirt_b': diff_in_dirt_b,
        'leak_a': leak_a,
        'leak_b': leak_b,
    })

    # 保存对比图
    base_name = f"{idx:02d}_{row['mask_class_name']}"

    # 单图对比
    comparison_single = np.hstack([
        cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(composed_a, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(composed_b, cv2.COLOR_BGR2RGB)
    ])
    comparison_single_bgr = cv2.cvtColor(comparison_single.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/{base_name}_comparison.jpg", comparison_single_bgr)

    # 详细图（4张）
    detail = np.vstack([
        np.hstack([
            cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(sd_image, cv2.COLOR_BGR2RGB),
        ]),
        np.hstack([
            cv2.cvtColor(composed_a, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(composed_b, cv2.COLOR_BGR2RGB),
        ])
    ])
    detail_bgr = cv2.cvtColor(detail.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/{base_name}_detail.jpg", detail_bgr)

    # 残差可视化
    residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/{base_name}_residual.png",
                cv2.cvtColor(residual_vis, cv2.COLOR_BGR2RGB))

    print(f"样本 {idx+1}: {row['mask_class_name']:15s} | "
          f"脏污占比: {alpha_mask.mean()*100:5.1f}% | "
          f"泄漏 A: {leak_a:5.2f} | 泄漏 B: {leak_b:5.2f}")

# 生成汇总统计
print("\n" + "=" * 70)
print("汇总统计")
print("=" * 70)

print(f"\n{'样本':<6} {'类型':<15} {'脏污占比':<10} {'梯度比A':<10} {'梯度比B':<10} {'泄漏A':<10} {'泄漏B':<10}")
print("-" * 80)

for r in results:
    print(f"{r['idx']+1:<6} {r['mask_class']:<15} {r['dirt_ratio']:>6.1f}%    "
          f"{r['ge_ratio_a']:>8.2f}   {r['ge_ratio_b']:>8.2f}   "
          f"{r['leak_a']:>8.2f}   {r['leak_b']:>8.2f}")

# 统计汇总
avg_ge_ratio_a = np.mean([r['ge_ratio_a'] for r in results])
avg_ge_ratio_b = np.mean([r['ge_ratio_b'] for r in results])
avg_leak_a = np.mean([r['leak_a'] for r in results])
avg_leak_b = np.mean([r['leak_b'] for r in results])

print("-" * 80)
print(f"{'平均':<22} {''}  {avg_ge_ratio_a:>8.2f}   {avg_ge_ratio_b:>8.2f}   "
      f"{avg_leak_a:>8.2f}   {avg_leak_b:>8.2f}")

print("\n指标说明:")
print("  梯度比 = 合成图梯度能量 / 背景梯度能量")
print("  泄漏 = 在透明区域的合成差异（越低越好）")
print("\n结论:")
if avg_leak_b < avg_leak_a:
    print(f"  方法B（残差）泄漏更低: {avg_leak_b:.2f} < {avg_leak_a:.2f}")
else:
    print(f"  方法A（直接）泄漏更低: {avg_leak_a:.2f} < {avg_leak_b:.2f}")

print(f"\n结果保存在: {output_dir}/")
print(f"  XX_XXX_comparison.jpg - 三联对比（背景 | 方法A | 方法B）")
print(f"  XX_XXX_detail.jpg - 详细对比（4图）")
print(f"  XX_XXX_residual.png - 残差层可视化")
print("=" * 70)
