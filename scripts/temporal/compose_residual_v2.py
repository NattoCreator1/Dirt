#!/usr/bin/env python3
"""
对 680 张手动筛选的脏污层使用增强版 residual 方法进行合成

优化内容：
1. 自适应增益 - 根据残差幅值自动调整
2. 低频补偿 - 透射变暗效果
3. 结构泄漏检测 - 防止放大引入语义泄漏
4. 增益上限保护

公式：
  I_output = I_bg + g·Alpha⊕Residual + 低频补偿

作者: soiling_project
日期: 2026-03-01
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def get_rgblabel_mask(filename, rgblabels_dir):
    """获取 rgbLabels mask"""
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


def compute_adaptive_gain(residual, alpha_mask,
                           target_strength=30.0,
                           g_min=0.5, g_max=4.0):
    """
    计算自适应增益

    根据 residual 的 90 分位数幅值来确定增益
    目标是让不同样本的脏污强度趋于一致

    Args:
        residual: 残差层 (H, W, 3)
        alpha_mask: Alpha mask (H, W, 1)
        target_strength: 目标强度（残差幅值的 90 分位数）
        g_min: 最小增益
        g_max: 最大增益（保护上限）

    Returns:
        gain: 增益系数
    """
    # 提取脏污区域的残差幅值
    dirty_mask = alpha_mask[:, :, 0] > 0.5
    if dirty_mask.sum() == 0:
        return 1.0

    # 计算每个像素的残差幅值
    residual_mag = np.abs(residual).sum(axis=2)  # (H, W)
    residual_in_dirt = residual_mag[dirty_mask]

    if len(residual_in_dirt) == 0:
        return 1.0

    # 使用 90 分位数作为强度度量
    current_strength = np.percentile(residual_in_dirt, 90)

    # 计算增益（避免除零）
    if current_strength < 1.0:
        gain = g_max
    else:
        gain = target_strength / current_strength

    # 限制在 [g_min, g_max] 范围内
    gain = np.clip(gain, g_min, g_max)

    return gain


def detect_structure_leakage(residual, alpha_mask, threshold=5.0):
    """
    检测结构泄漏

    通过分析 residual 在脏污区域的结构复杂度来判断是否有泄漏
    如果 residual 包含大量结构（如道路线条），说明可能有泄漏

    Args:
        residual: 残差层 (H, W, 3)
        alpha_mask: Alpha mask (H, W, 1)
        threshold: 泄漏阈值（梯度能量）

    Returns:
        leakage_score: 泄漏分数（越低越好）
        has_leakage: 是否有明显泄漏
    """
    dirty_mask = alpha_mask[:, :, 0] > 0.5
    if dirty_mask.sum() == 0:
        return 0.0, False

    # 计算梯度能量（反映结构复杂度）
    gray_residual = np.abs(residual).sum(axis=2).astype(np.uint8)
    gx = cv2.Sobel(gray_residual, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_residual, cv2.CV_64F, 0, 1, ksize=3)
    gradient_energy = np.sqrt(gx**2 + gy**2)

    # 只统计脏污区域
    energy_in_dirt = gradient_energy[dirty_mask]
    avg_energy = energy_in_dirt.mean()

    return avg_energy, avg_energy > threshold


def add_low_frequency_darkening(bg_frame, alpha_mask, beta=0.3):
    """
    添加低频透射变暗效果

    模拟脏污导致的透射率下降：I = I * (1 - beta * Alpha)

    Args:
        bg_frame: 背景帧 (H, W, 3)
        alpha_mask: Alpha mask (H, W, 1)
        beta: 变暗强度系数 [0, 1]

    Returns:
        变暗后的图像
    """
    if alpha_mask.ndim == 2:
        alpha_mask = alpha_mask[:, :, np.newaxis]

    # I = I * (1 - beta * Alpha)
    darkening_factor = 1.0 - beta * alpha_mask
    darkened = bg_frame.astype(np.float32) * darkening_factor
    return np.clip(darkened, 0, 255).astype(np.uint8)


def add_scattering_blur(bg_frame, alpha_mask, gamma=0.2, kernel_size=15):
    """
    添加散射模糊效果

    模拟脏污导致的散射：I = I + gamma * Alpha * (Blur(I) - I)

    Args:
        bg_frame: 背景帧 (H, W, 3)
        alpha_mask: Alpha mask (H, W, 1)
        gamma: 散射强度系数 [0, 1]
        kernel_size: 模糊核大小

    Returns:
        散射模糊后的图像
    """
    if alpha_mask.ndim == 2:
        alpha_mask = alpha_mask[:, :, np.newaxis]

    # 高斯模糊
    blurred = cv2.GaussianBlur(bg_frame, (kernel_size, kernel_size), 0)

    # I = I + gamma * Alpha * (Blur(I) - I)
    blur_diff = blurred.astype(np.float32) - bg_frame.astype(np.float32)
    scattered = bg_frame.astype(np.float32) + gamma * alpha_mask * blur_diff
    return np.clip(scattered, 0, 255).astype(np.uint8)


def compose_with_residual_enhanced(bg_frame, residual, alpha_mask,
                                   gain=1.0,
                                   use_darkening=True,
                                   beta=0.3,
                                   use_scattering=False,
                                   gamma=0.2):
    """
    使用增强版残差方法合成

    公式:
      I = I_bg + g·Alpha⊕Residual + 低频补偿

    低频补偿选项:
      - 透射变暗: I * (1 - beta * Alpha)
      - 散射模糊: I + gamma * Alpha * (Blur(I) - I)
    """
    h, w = bg_frame.shape[:2]

    # Resize 到匹配
    if residual.shape[:2] != (h, w):
        residual = cv2.resize(residual, (w, h), interpolation=cv2.INTER_AREA)
    if alpha_mask.shape[:2] != (h, w):
        alpha_mask = cv2.resize(alpha_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if alpha_mask.ndim == 2:
        alpha_mask = alpha_mask[:, :, np.newaxis]

    # 基础残差合成
    composed = bg_frame.astype(np.float32) + gain * alpha_mask * residual

    # 添加低频补偿
    composed_int = np.clip(composed, 0, 255).astype(np.uint8)

    if use_darkening:
        composed_int = add_low_frequency_darkening(composed_int, alpha_mask, beta)

    if use_scattering:
        composed_int = add_scattering_blur(composed_int, alpha_mask, gamma)

    return composed_int


def compute_quality_metrics_enhanced(composed, bg_frame, alpha_mask, gain):
    """计算增强版质量指标"""
    clean_mask = alpha_mask[:, :, 0] < 0.5
    dirty_mask = alpha_mask[:, :, 0] >= 0.5

    metrics = {}

    # 1. 泄漏分数
    if clean_mask.sum() > 0:
        leak = np.abs(composed.astype(np.float32) - bg_frame.astype(np.float32))
        metrics['leak_score'] = leak[clean_mask].mean()
    else:
        metrics['leak_score'] = 0.0

    # 2. 脏污强度（增强后）
    if dirty_mask.sum() > 0:
        diff = np.abs(composed.astype(np.float32) - bg_frame.astype(np.float32))
        metrics['dirt_strength'] = diff[dirty_mask].mean()
    else:
        metrics['dirt_strength'] = 0.0

    # 3. 脏污占比
    metrics['dirt_ratio'] = dirty_mask.mean()

    # 4. 增益系数
    metrics['gain'] = gain

    return metrics


class EnhancedResidualCompositor:
    """增强版残差法合成器"""

    def __init__(self,
                 sd_dir: str,
                 clean_bg_dir: str,
                 rgblabels_dir: str,
                 output_dir: str,
                 target_strength: float = 30.0,
                 g_min: float = 0.5,
                 g_max: float = 4.0,
                 use_darkening: bool = True,
                 beta: float = 0.3,
                 use_scattering: bool = False,
                 gamma: float = 0.2,
                 leakage_threshold: float = 5.0,
                 seed: int = 42):
        self.sd_dir = sd_dir
        self.clean_bg_dir = clean_bg_dir
        self.rgblabels_dir = rgblabels_dir
        self.output_dir = output_dir
        self.target_strength = target_strength
        self.g_min = g_min
        self.g_max = g_max
        self.use_darkening = use_darkening
        self.beta = beta
        self.use_scattering = use_scattering
        self.gamma = gamma
        self.leakage_threshold = leakage_threshold
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        ensure_dir(output_dir)
        self.bg_frames = self._collect_bg_frames()

        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'leakage_rejected': 0,
        }
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
        sd_files = [f for f in os.listdir(self.sd_dir) if f.endswith('.png')]
        self.stats['total'] = len(sd_files)

        if max_samples:
            sd_files = sd_files[:max_samples]
            print(f"测试模式: 仅处理 {max_samples} 个样本")

        print(f"开始处理 {len(sd_files)} 个脏污层...")
        print(f"  SD 目录: {self.sd_dir}")
        print(f"  干净帧目录: {self.clean_bg_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"\n参数设置:")
        print(f"  目标强度: {self.target_strength}")
        print(f"  增益范围: [{self.g_min}, {self.g_max}]")
        print(f"  透射变暗: {self.use_darkening} (beta={self.beta})")
        print(f"  散射模糊: {self.use_scattering} (gamma={self.gamma})")
        print(f"  泄漏阈值: {self.leakage_threshold}")
        print()

        for idx, filename in enumerate(tqdm(sd_files, desc="处理进度")):
            success = self._process_single(filename, idx)
            if success == 'rejected':
                self.stats['leakage_rejected'] += 1
            elif success:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

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

        # 获取 mask
        alpha_mask = get_rgblabel_mask(filename, self.rgblabels_dir)
        if alpha_mask is None:
            # 使用残差强度生成 mask
            residual_strength = np.abs(residual).sum(axis=2)
            alpha_mask = (residual_strength > 20).astype(np.float32)[:, :, np.newaxis]

        # 检测结构泄漏
        leakage_score, has_leakage = detect_structure_leakage(
            residual, alpha_mask, self.leakage_threshold
        )

        if has_leakage:
            # 跳过有泄漏的样本
            self.results.append({
                'idx': idx,
                'output_name': f"REJECTED_{idx:04d}",
                'original_file': filename,
                'leakage_score': float(leakage_score),
                'gain': 0.0,
                'dirt_ratio': 0.0,
                'dirt_strength': 0.0,
                'status': 'rejected',
            })
            return 'rejected'

        # 计算自适应增益
        gain = compute_adaptive_gain(
            residual, alpha_mask,
            target_strength=self.target_strength,
            g_min=self.g_min,
            g_max=self.g_max
        )

        # 随机选择干净背景
        bg_info = random.choice(self.bg_frames)
        bg_frame = cv2.imread(bg_info['path'])

        if bg_frame is None:
            return False

        if bg_frame.shape[:2] != (480, 640):
            bg_frame = cv2.resize(bg_frame, (640, 480), interpolation=cv2.INTER_AREA)

        # 合成
        composed = compose_with_residual_enhanced(
            bg_frame, residual, alpha_mask,
            gain=gain,
            use_darkening=self.use_darkening,
            beta=self.beta,
            use_scattering=self.use_scattering,
            gamma=self.gamma
        )

        # 计算质量指标
        metrics = compute_quality_metrics_enhanced(composed, bg_frame, alpha_mask, gain)

        # 保存结果
        base_name = filename.replace('.png', '').replace('_640x480', '')
        output_base = os.path.join(self.output_dir, f"{idx:04d}_{base_name}")

        cv2.imwrite(f"{output_base}_composed.jpg", composed)
        cv2.imwrite(f"{output_base}_bg.jpg", bg_frame)

        # 保存残差可视化（应用增益后）
        residual_vis = np.clip(gain * residual + 128, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{output_base}_residual.png",
                    cv2.cvtColor(residual_vis, cv2.COLOR_BGR2RGB))

        # 记录结果
        self.results.append({
            'idx': idx,
            'output_name': f"{idx:04d}_{base_name}",
            'original_file': filename,
            'bg_view': bg_info['view'],
            'gain': gain,
            'leakage_score': metrics.get('leak_score', 0),
            'dirt_ratio': metrics.get('dirt_ratio', 0),
            'dirt_strength': metrics.get('dirt_strength', 0),
            'status': 'success',
        })

        return True

    def _save_results(self):
        """保存结果清单"""
        df = pd.DataFrame(self.results)

        # 保存全部结果
        df.to_csv(os.path.join(self.output_dir, 'composition_results.csv'), index=False)

        # 只保存成功的
        df_success = df[df['status'] == 'success']
        if len(df_success) > 0:
            df_success.to_csv(os.path.join(self.output_dir, 'results_success_only.csv'), index=False)

            # 按泄漏分数排序
            df_success_sorted = df_success.sort_values('leakage_score')
            df_success_sorted.to_csv(os.path.join(self.output_dir, 'results_sorted_by_leak.csv'), index=False)

            # 按脏污强度排序
            df_success_sorted_strength = df_success.sort_values('dirt_strength', ascending=False)
            df_success_sorted_strength.to_csv(os.path.join(self.output_dir, 'results_sorted_by_strength.csv'), index=False)

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("处理完成")
        print("="*60)
        print(f"总数: {self.stats['total']}")
        print(f"成功: {self.stats['success']}")
        print(f"泄漏拒绝: {self.stats['leakage_rejected']}")
        print(f"失败: {self.stats['failed']}")
        print(f"\n输出目录: {self.output_dir}")
        print(f"结果清单:")
        print(f"  - composition_results.csv (全部)")
        print(f"  - results_success_only.csv (仅成功)")
        print(f"  - results_sorted_by_leak.csv (按泄漏排序)")
        print(f"  - results_sorted_by_strength.csv (按强度排序)")
        print("="*60)

        # 打印增益统计
        if self.results:
            gains = [r['gain'] for r in self.results if r['status'] == 'success']
            if gains:
                print(f"\n增益统计:")
                print(f"  最小: {min(gains):.2f}")
                print(f"  最大: {max(gains):.2f}")
                print(f"  平均: {np.mean(gains):.2f}")
                print(f"  中位数: {np.median(gains):.2f}")


def main():
    ap = argparse.ArgumentParser(
        description="对 680 张手动筛选的脏污层使用增强版 residual 方法合成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
增强内容:
  1. 自适应增益 - 根据残差幅值自动调整增益系数
  2. 透射变暗 - 模拟脏污导致的透射率下降
  3. 结构泄漏检测 - 拒绝有明显泄漏的样本
  4. 增益上限保护 - 防止过度放大

推荐参数:
  target_strength=30.0  -- 目标脏污强度
  g_min=0.5              -- 最小增益
  g_max=4.0              -- 最大增益
  beta=0.3               -- 透射变暗系数
  leakage_threshold=5.0  -- 泄漏检测阈值
        """
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
                    default="dataset/sd_temporal_training/residual_composition_v2",
                    help="输出目录")

    # 增益参数
    ap.add_argument("--target_strength", type=float, default=30.0,
                    help="目标脏污强度（用于自适应增益）")
    ap.add_argument("--g_min", type=float, default=0.5,
                    help="最小增益")
    ap.add_argument("--g_max", type=float, default=4.0,
                    help="最大增益（保护上限）")

    # 低频补偿参数
    ap.add_argument("--use_darkening", action="store_true", default=True,
                    help="启用透射变暗效果")
    ap.add_argument("--beta", type=float, default=0.3,
                    help="透射变暗系数 [0, 1]")
    ap.add_argument("--use_scattering", action="store_true", default=False,
                    help="启用散射模糊效果")
    ap.add_argument("--gamma", type=float, default=0.2,
                    help="散射模糊系数 [0, 1]")

    # 质量控制
    ap.add_argument("--leakage_threshold", type=float, default=5.0,
                    help="泄漏检测阈值")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="最大处理样本数（测试用）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")

    args = ap.parse_args()

    compositor = EnhancedResidualCompositor(
        sd_dir=args.sd_dir,
        clean_bg_dir=args.clean_bg_dir,
        rgblabels_dir=args.rgblabels_dir,
        output_dir=args.output_dir,
        target_strength=args.target_strength,
        g_min=args.g_min,
        g_max=args.g_max,
        use_darkening=args.use_darkening,
        beta=args.beta,
        use_scattering=args.use_scattering,
        gamma=args.gamma,
        leakage_threshold=args.leakage_threshold,
        seed=args.seed,
    )

    compositor.process_all(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
