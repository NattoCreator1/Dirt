#!/usr/bin/env python3
"""
对 680 张手动筛选的脏污层使用 residual 方法进行合成

流程：
1. 读取 SD 生成的脏污图像
2. 创建降语义底图
3. 提取残差层
4. 增强 residual（解决"过淡/透明感强"问题）
5. 随机选择一张干净背景帧
6. 使用残差方法合成
7. 保存结果供用户筛选

Residual Boost 参数说明：
  - boost_gain: 全局增益系数
  - boost_clip_val: 软饱和阈值
  - boost_gain_pos: 正残差增益（亮的部分）
  - boost_gain_neg: 负残差增益（暗的部分，更像脏污）

推荐起始参数：
  boost_gain_neg=4.0, boost_gain_pos=1.5, boost_clip_val=90
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import cv2


def ensure_dir(p: str):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def create_degraded_base(image):
    """创建降语义底图"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (41, 41), 10)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def extract_residual(sd_image, degraded_base):
    """提取残差层"""
    return sd_image.astype(np.float32) - degraded_base.astype(np.float32)


def boost_residual(residual: np.ndarray,
                   gain: float = 1.0,
                   clip_val: float = 90.0,
                   gain_pos: float = 1.5,
                   gain_neg: float = 4.0) -> np.ndarray:
    """
    增强 residual 幅值，解决"过淡/透明感强"的问题

    - gain_pos / gain_neg: 分别放大正/负残差（负残差=变暗，更像脏污）
    - tanh 软饱和：避免强放大后出现大面积饱和/花屏

    Args:
        residual: 残差层 (H, W, 3), float32, 范围大致在[-255,255]
        gain: 全局增益系数
        clip_val: 软饱和阈值（越大越"重"，但也更容易饱和）
        gain_pos: 正残差增益（亮的部分）
        gain_neg: 负残差增益（暗的部分，更像脏污）

    Returns:
        增强后的残差层 (H, W, 3), float32
    """
    R = residual.astype(np.float32)

    # 分正负增强：让"变暗"更强，减少"发亮假脏污"
    Rp = np.maximum(R, 0.0) * gain_pos
    Rn = np.minimum(R, 0.0) * gain_neg
    R = Rp + Rn

    # 全局再乘一个 gain
    R *= gain

    # 软饱和：把极端值压住，避免合成后大片夹逼到0/255
    R = clip_val * np.tanh(R / clip_val)
    return R


def get_rgblabel_mask(filename, rgblabels_dir):
    """获取 rgbLabels mask"""
    # 解析文件名获取 mask_id
    base = filename.replace('_640x480.png', '').replace('.png', '')
    parts = base.rsplit('_', 3)
    if len(parts) < 4:
        return None

    frame_id = parts[-3]
    view = parts[-2]
    mask_id = f"{frame_id}_{view}"

    rgblabel_path = os.path.join(rgblabels_dir, f"{mask_id}.png")
    if not os.path.exists(rgblabel_path):
        return None

    rgblabel = cv2.imread(rgblabel_path)
    rgblabel_rgb = cv2.cvtColor(rgblabel, cv2.COLOR_BGR2RGB)
    rgblabel_small = cv2.resize(rgblabel_rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
    alpha_mask = (rgblabel_small.sum(axis=2) > 0).astype(np.float32)
    return alpha_mask[:, :, np.newaxis]


def compose_with_residual(bg_frame, residual, alpha_mask, strength=1.0):
    """使用残差方法合成"""
    if residual.shape[:2] != bg_frame.shape[:2]:
        residual = cv2.resize(residual, (bg_frame.shape[1], bg_frame.shape[0]), interpolation=cv2.INTER_AREA)
    if alpha_mask.shape[:2] != bg_frame.shape[:2]:
        alpha_mask = cv2.resize(alpha_mask, (bg_frame.shape[1], bg_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    if alpha_mask.ndim == 2:
        alpha_mask = alpha_mask[:, :, np.newaxis]

    composed = bg_frame.astype(np.float32) + strength * alpha_mask * residual
    return np.clip(composed, 0, 255).astype(np.uint8)


def compute_quality_metrics(composed, bg_frame, alpha_mask):
    """计算质量指标"""
    # 在透明区域检查泄漏
    clean_mask = alpha_mask[:, :, 0] < 0.5
    if clean_mask.sum() > 0:
        leak = np.abs(composed.astype(np.float32) - bg_frame.astype(np.float32))
        leak_score = leak[clean_mask].mean()
    else:
        leak_score = 0.0

    # 残差强度（脏污区域的平均变化）
    dirty_mask = alpha_mask[:, :, 0] >= 0.5
    if dirty_mask.sum() > 0:
        diff = np.abs(composed.astype(np.float32) - bg_frame.astype(np.float32))
        dirt_strength = diff[dirty_mask].mean()
    else:
        dirt_strength = 0.0

    # 脏污占比
    dirt_ratio = dirty_mask.mean()

    return {
        'leak_score': leak_score,
        'dirt_strength': dirt_strength,
        'dirt_ratio': dirt_ratio,
    }


class ResidualCompositor:
    """残差法合成器"""

    def __init__(self,
                 sd_dir: str,
                 clean_bg_dir: str,
                 rgblabels_dir: str,
                 output_dir: str,
                 boost_gain: float = 1.0,
                 boost_clip_val: float = 90.0,
                 boost_gain_pos: float = 1.5,
                 boost_gain_neg: float = 4.0,
                 seed: int = 42):
        self.sd_dir = sd_dir
        self.clean_bg_dir = clean_bg_dir
        self.rgblabels_dir = rgblabels_dir
        self.output_dir = output_dir
        self.boost_gain = boost_gain
        self.boost_clip_val = boost_clip_val
        self.boost_gain_pos = boost_gain_pos
        self.boost_gain_neg = boost_gain_neg
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        ensure_dir(output_dir)

        # 收集所有干净背景帧
        self.bg_frames = self._collect_bg_frames()

        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
        }

        # 结果记录
        self.results = []

    def _collect_bg_frames(self):
        """收集所有干净背景帧"""
        bg_frames = []
        for view in ['f', 'lf', 'rf', 'lr', 'r', 'rr']:
            view_dir = os.path.join(self.clean_bg_dir, view)
            if not os.path.exists(view_dir):
                continue

            for seq_dir in os.listdir(view_dir):
                seq_path = os.path.join(view_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                for frame_file in os.listdir(seq_path):
                    if frame_file.endswith('.jpg'):
                        bg_frames.append({
                            'path': os.path.join(seq_path, frame_file),
                            'view': view,
                        })

        print(f"收集到 {len(bg_frames)} 张干净背景帧")
        return bg_frames

    def process_all(self, max_samples=None):
        """批量处理"""
        # 获取所有 SD 文件
        sd_files = [f for f in os.listdir(self.sd_dir) if f.endswith('.png')]
        self.stats['total'] = len(sd_files)

        if max_samples:
            sd_files = sd_files[:max_samples]
            print(f"测试模式: 仅处理 {max_samples} 个样本")

        print(f"开始处理 {len(sd_files)} 个脏污层...")
        print(f"  SD 目录: {self.sd_dir}")
        print(f"  干净帧目录: {self.clean_bg_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"\nResidual Boost 参数:")
        print(f"  boost_gain={self.boost_gain}")
        print(f"  boost_clip_val={self.boost_clip_val}")
        print(f"  boost_gain_pos={self.boost_gain_pos}")
        print(f"  boost_gain_neg={self.boost_gain_neg}")
        print()

        for idx, filename in enumerate(tqdm(sd_files, desc="处理进度")):
            success = self._process_single(filename, idx)
            if success:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

        # 保存结果清单
        self._save_results()

        self._print_summary()

    def _process_single(self, filename, idx):
        """处理单个文件"""
        # 读取 SD 图像
        sd_path = os.path.join(self.sd_dir, filename)
        sd_image = cv2.imread(sd_path)
        if sd_image is None:
            return False

        # 创建降语义底图
        degraded_base = create_degraded_base(sd_image)

        # 提取残差
        residual = extract_residual(sd_image, degraded_base)

        # 增强 residual（解决"过淡/透明感强"问题）
        residual = boost_residual(
            residual,
            gain=self.boost_gain,
            clip_val=self.boost_clip_val,
            gain_pos=self.boost_gain_pos,
            gain_neg=self.boost_gain_neg,
        )

        # 获取 mask
        alpha_mask = get_rgblabel_mask(filename, self.rgblabels_dir)
        if alpha_mask is None:
            # 使用残差强度生成 mask
            residual_strength = np.abs(residual).sum(axis=2)
            alpha_mask = (residual_strength > 20).astype(np.float32)[:, :, np.newaxis]

        # 随机选择干净背景
        bg_info = random.choice(self.bg_frames)
        bg_frame = cv2.imread(bg_info['path'])

        if bg_frame is None:
            return False

        # Resize 到 640x480
        if bg_frame.shape[:2] != (480, 640):
            bg_frame = cv2.resize(bg_frame, (640, 480), interpolation=cv2.INTER_AREA)

        # 合成
        composed = compose_with_residual(bg_frame, residual, alpha_mask)

        # 计算质量指标
        metrics = compute_quality_metrics(composed, bg_frame, alpha_mask)

        # 保存结果
        base_name = filename.replace('.png', '').replace('_640x480', '')
        output_base = os.path.join(self.output_dir, f"{idx:04d}_{base_name}")

        cv2.imwrite(f"{output_base}_composed.jpg", composed)
        cv2.imwrite(f"{output_base}_bg.jpg", bg_frame)
        cv2.imwrite(f"{output_base}_sd.jpg", sd_image)

        # 保存残差可视化
        residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{output_base}_residual.png",
                    cv2.cvtColor(residual_vis, cv2.COLOR_BGR2RGB))

        # 记录结果
        self.results.append({
            'idx': idx,
            'output_name': f"{idx:04d}_{base_name}",
            'original_file': filename,
            'bg_view': bg_info['view'],
            'dirt_ratio': metrics['dirt_ratio'],
            'leak_score': metrics['leak_score'],
            'dirt_strength': metrics['dirt_strength'],
        })

        return True

    def _save_results(self):
        """保存结果清单"""
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_dir, 'composition_results.csv'), index=False)

        # 按泄漏分数排序，方便筛选
        df_sorted = df.sort_values('leak_score')
        df_sorted.to_csv(os.path.join(self.output_dir, 'results_sorted_by_leak.csv'), index=False)

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("处理完成")
        print("="*60)
        print(f"总数: {self.stats['total']}")
        print(f"成功: {self.stats['success']}")
        print(f"失败: {self.stats['failed']}")
        print(f"\n输出目录: {self.output_dir}")
        print(f"结果清单: composition_results.csv")
        print(f"按泄漏排序: results_sorted_by_leak.csv")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="对 680 张手动筛选的脏污层使用 residual 方法合成"
    )

    ap.add_argument("--sd_dir", type=str,
                    default="sd_scripts/lora/accepted_20260224_164715/resized_640x480",
                    help="SD 生成图像目录（680 张）")
    ap.add_argument("--clean_bg_dir", type=str,
                    default="dataset/temporal_sequences/raw_sequences",
                    help="干净背景帧目录")
    ap.add_argument("--rgblabels_dir", type=str,
                    default="dataset/woodscape_raw/train/rgbLabels",
                    help="rgbLabels 目录")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/residual_composition_680",
                    help="输出目录")

    ap.add_argument("--max_samples", type=int, default=None,
                    help="最大处理样本数（测试用）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")

    # Residual boost 参数（解决"过淡/透明感强"问题）
    ap.add_argument("--boost_gain", type=float, default=1.0,
                    help="残差全局增益系数 (默认: 1.0)")
    ap.add_argument("--boost_clip_val", type=float, default=90.0,
                    help="软饱和阈值，越大越重 (默认: 90.0)")
    ap.add_argument("--boost_gain_pos", type=float, default=1.5,
                    help="正残差增益（亮的部分） (默认: 1.5)")
    ap.add_argument("--boost_gain_neg", type=float, default=4.0,
                    help="负残差增益（暗的部分，更像脏污） (默认: 4.0)")

    args = ap.parse_args()

    compositor = ResidualCompositor(
        sd_dir=args.sd_dir,
        clean_bg_dir=args.clean_bg_dir,
        rgblabels_dir=args.rgblabels_dir,
        output_dir=args.output_dir,
        boost_gain=args.boost_gain,
        boost_clip_val=args.boost_clip_val,
        boost_gain_pos=args.boost_gain_pos,
        boost_gain_neg=args.boost_gain_neg,
        seed=args.seed,
    )

    compositor.process_all(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
