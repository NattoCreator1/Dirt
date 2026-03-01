#!/usr/bin/env python3
"""
时序训练数据合成脚本

核心设计：
- 同一脏污层作用于整个干净序列的所有帧（确保脏污静止）
- 保存合成参数和 QC 指标（便于后续数据筛选和问题定位）

输入：
- 脏污层目录：dataset/sd_temporal_training/residual_composition_680_heavy2_filtered/
- 干净序列清单：dataset/temporal_sequences/checked/clean_sequences.csv
- 干净序列根目录：dataset/temporal_sequences/raw_sequences/

输出：
- 合成序列目录：dataset/sd_temporal_training/sequences/
- 元数据清单：dataset/sd_temporal_training/sequences_manifest.csv
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random

import cv2


def ensure_dir(p: str):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def create_degraded_base(image):
    """创建降语义底图（高斯模糊 + 灰度）"""
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
    增强 residual 幅值

    Args:
        residual: 残差层 (H, W, 3), float32
        gain: 全局增益系数
        clip_val: 软饱和阈值
        gain_pos: 正残差增益（亮的部分）
        gain_neg: 负残差增益（暗的部分，更像脏污）

    Returns:
        增强后的残差层
    """
    R = residual.astype(np.float32)

    # 分正负增强
    Rp = np.maximum(R, 0.0) * gain_pos
    Rn = np.minimum(R, 0.0) * gain_neg
    R = Rp + Rn

    # 全局增益
    R *= gain

    # 软饱和
    R = clip_val * np.tanh(R / clip_val)
    return R


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


def compute_qc_metrics(composed, bg_frame, alpha_mask):
    """
    计算 QC 指标（用于数据筛选和问题定位）

    Returns:
        dict containing:
        - leak_score: mask 外区域的平均变化（理想情况下应接近 0）
        - dirt_strength: mask 内区域的平均变化
        - dirt_ratio: 脏污区域占比
        - saturation_rate: 饱和像素比例（接近 0 或 255）
        - edge_spill: 边界溢出指标
    """
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

    # 饱和率（接近 0 或 255 的像素比例）
    saturation_mask = (composed < 5) | (composed > 250)
    saturation_rate = saturation_mask.mean()

    # 边界溢出（mask 边缘 3 像素环的平均变化）
    edge_spill = 0.0
    if dirty_mask.sum() > 0:
        # 膨胀 mask 得到边界环
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dirty_mask_u8 = dirty_mask.astype(np.uint8)
        dilated = cv2.dilate(dirty_mask_u8, kernel, iterations=1)
        edge_ring = (dilated > 0) & (dirty_mask == 0)
        if edge_ring.sum() > 0:
            diff = np.abs(composed.astype(np.float32) - bg_frame.astype(np.float32))
            edge_spill = diff[edge_ring].mean()

    return {
        'leak_score': leak_score,
        'dirt_strength': dirt_strength,
        'dirt_ratio': dirt_ratio,
        'saturation_rate': saturation_rate,
        'edge_spill': edge_spill,
    }


class TemporalSequenceComposer:
    """时序序列合成器"""

    def __init__(self,
                 dirt_layers_dir: str,
                 clean_seqs_csv: str,
                 clean_seqs_root: str,
                 rgblabels_dir: str,
                 output_dir: str,
                 boost_gain: float = 1.0,
                 boost_clip_val: float = 90.0,
                 boost_gain_pos: float = 1.5,
                 boost_gain_neg: float = 4.0,
                 strength: float = 1.0,
                 seed: int = 42):
        self.dirt_layers_dir = dirt_layers_dir
        self.clean_seqs_csv = clean_seqs_csv
        self.clean_seqs_root = clean_seqs_root
        self.rgblabels_dir = rgblabels_dir
        self.output_dir = output_dir
        self.boost_gain = boost_gain
        self.boost_clip_val = boost_clip_val
        self.boost_gain_pos = boost_gain_pos
        self.boost_gain_neg = boost_gain_neg
        self.strength = strength
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        ensure_dir(output_dir)

        # 加载干净序列清单
        self.clean_seqs_df = pd.read_csv(clean_seqs_csv)
        print(f"加载了 {len(self.clean_seqs_df)} 个干净序列")

        # 收集脏污层
        self.dirt_layers = self._collect_dirt_layers()

        # 统计
        self.stats = {
            'total_sequences': 0,
            'success_sequences': 0,
            'failed_sequences': 0,
        }

        # 结果记录
        self.results = []

    def _collect_dirt_layers(self):
        """收集所有脏污层"""
        dirt_layers = []

        # 从 residual_composition_680_heavy2_filtered 收集
        for filename in os.listdir(self.dirt_layers_dir):
            if filename.endswith('_composed.jpg'):
                base = filename.replace('_composed.jpg', '')
                sd_file = f"{base}_sd.jpg"

                # 检查对应的 SD 文件是否存在
                sd_path = os.path.join(self.dirt_layers_dir, sd_file)
                if os.path.exists(sd_path):
                    dirt_layers.append({
                        'id': base,
                        'sd_path': sd_path,
                        'composed_path': os.path.join(self.dirt_layers_dir, filename),
                    })

        print(f"收集到 {len(dirt_layers)} 个脏污层")
        return dirt_layers

    def _get_dirt_residual_and_mask(self, dirt_layer_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从脏污层中提取残差和 mask

        这里我们需要从原始 SD 图像重新计算残差，
        因为 compose_with_residual 需要残差层而不是合成结果
        """
        # 找到对应的 dirt layer
        dirt_layer = None
        for dl in self.dirt_layers:
            if dl['id'] == dirt_layer_id:
                dirt_layer = dl
                break

        if dirt_layer is None:
            return None, None

        # 读取 SD 图像
        sd_image = cv2.imread(dirt_layer['sd_path'])
        if sd_image is None:
            return None, None

        # 创建降语义底图
        degraded_base = create_degraded_base(sd_image)

        # 提取残差
        residual = extract_residual(sd_image, degraded_base)

        # 增强 residual
        residual = boost_residual(
            residual,
            gain=self.boost_gain,
            clip_val=self.boost_clip_val,
            gain_pos=self.boost_gain_pos,
            gain_neg=self.boost_gain_neg,
        )

        # 获取 mask（从 SD 文件名推断）
        sd_filename = os.path.basename(dirt_layer['sd_path'])
        alpha_mask = get_rgblabel_mask(sd_filename, self.rgblabels_dir)
        if alpha_mask is None:
            # 使用残差强度生成 mask
            residual_strength = np.abs(residual).sum(axis=2)
            alpha_mask = (residual_strength > 20).astype(np.float32)[:, :, np.newaxis]

        return residual, alpha_mask

    def compose_sequence(self, dirt_layer_id: str, view: str, seq_id: str) -> Optional[Dict]:
        """
        合成一个时序序列

        Args:
            dirt_layer_id: 脏污层 ID
            view: 视角 (f, lf, rf, lr, r, rr)
            seq_id: 序列 ID

        Returns:
            包含合成结果信息的字典，失败返回 None
        """
        # 获取残差和 mask
        residual, alpha_mask = self._get_dirt_residual_and_mask(dirt_layer_id)
        if residual is None:
            return None

        # 获取干净序列路径
        seq_path = os.path.join(self.clean_seqs_root, view, seq_id)
        if not os.path.exists(seq_path):
            return None

        # 获取所有帧
        frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])
        if len(frame_files) == 0:
            return None

        # 创建输出目录
        output_seq_dir = os.path.join(self.output_dir, f"{dirt_layer_id}_{view}_{seq_id}")
        ensure_dir(output_seq_dir)

        # 逐帧合成
        frame_paths = []
        qc_metrics_list = []

        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            bg_frame = cv2.imread(frame_path)

            if bg_frame is None:
                continue

            # Resize 到 640x480
            if bg_frame.shape[:2] != (480, 640):
                bg_frame = cv2.resize(bg_frame, (640, 480), interpolation=cv2.INTER_AREA)

            # 合成
            composed = compose_with_residual(bg_frame, residual, alpha_mask, self.strength)

            # 计算 QC 指标
            qc = compute_qc_metrics(composed, bg_frame, alpha_mask)
            qc_metrics_list.append(qc)

            # 保存
            output_frame_path = os.path.join(output_seq_dir, frame_file)
            cv2.imwrite(output_frame_path, composed)
            frame_paths.append(output_frame_path)

        if len(frame_paths) == 0:
            return None

        # 聚合 QC 指标（取中位数）
        qc_aggregated = {}
        for key in qc_metrics_list[0].keys():
            values = [qc[key] for qc in qc_metrics_list]
            qc_aggregated[key] = np.median(values)

        return {
            'dirt_layer_id': dirt_layer_id,
            'view': view,
            'seq_id': seq_id,
            'num_frames': len(frame_paths),
            'output_dir': output_seq_dir,
            'qc_metrics': qc_aggregated,
        }

    def generate_all(self, max_sequences: Optional[int] = None):
        """
        批量生成所有时序序列

        逻辑：遍历所有干净序列，为每个干净序列分配一个脏污层（循环使用）

        Args:
            max_sequences: 最大生成序列数（测试用）
        """
        num_clean_seqs = len(self.clean_seqs_df)
        num_dirt_layers = len(self.dirt_layers)

        # 每个干净序列分配一个脏污层（循环使用）
        total_possible = num_clean_seqs

        if max_sequences:
            total_possible = min(total_possible, max_sequences)
            print(f"测试模式: 最多生成 {max_sequences} 个序列")

        print(f"\n开始生成时序序列...")
        print(f"  干净序列数: {num_clean_seqs}")
        print(f"  脏污层数: {num_dirt_layers}")
        print(f"  预计总序列数: {total_possible}")
        print(f"  分配策略: 每个干净序列分配 1 个脏污层（循环使用）")
        print(f"\n合成参数:")
        print(f"  boost_gain={self.boost_gain}")
        print(f"  boost_clip_val={self.boost_clip_val}")
        print(f"  boost_gain_pos={self.boost_gain_pos}")
        print(f"  boost_gain_neg={self.boost_gain_neg}")
        print(f"  strength={self.strength}")
        print()

        # 随机打乱干净序列
        clean_seqs_pool = self.clean_seqs_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # 生成序列
        for seq_idx in tqdm(range(total_possible), desc="生成进度"):
            clean_seq = clean_seqs_pool.iloc[seq_idx]
            view = clean_seq['view']
            seq_id = clean_seq['seq_id']

            # 循环分配脏污层
            dirt_layer = self.dirt_layers[seq_idx % num_dirt_layers]
            dirt_layer_id = dirt_layer['id']

            # 合成序列
            result = self.compose_sequence(dirt_layer_id, view, seq_id)
            self.stats['total_sequences'] += 1

            if result is not None:
                self.stats['success_sequences'] += 1

                # 记录结果
                self.results.append({
                    'dirt_layer_id': dirt_layer_id,
                    'view': view,
                    'clean_seq_id': seq_id,
                    'num_frames': result['num_frames'],
                    'output_dir': result['output_dir'],
                    # 合成参数
                    'boost_gain': self.boost_gain,
                    'boost_clip_val': self.boost_clip_val,
                    'boost_gain_pos': self.boost_gain_pos,
                    'boost_gain_neg': self.boost_gain_neg,
                    'strength': self.strength,
                    # QC 指标
                    'qc_leak_score': result['qc_metrics']['leak_score'],
                    'qc_dirt_strength': result['qc_metrics']['dirt_strength'],
                    'qc_dirt_ratio': result['qc_metrics']['dirt_ratio'],
                    'qc_saturation_rate': result['qc_metrics']['saturation_rate'],
                    'qc_edge_spill': result['qc_metrics']['edge_spill'],
                })
            else:
                self.stats['failed_sequences'] += 1

        # 保存结果清单
        self._save_manifest()

        self._print_summary()

    def _save_manifest(self):
        """保存序列清单"""
        df = pd.DataFrame(self.results)
        manifest_path = os.path.join(self.output_dir, 'sequences_manifest.csv')
        df.to_csv(manifest_path, index=False)
        print(f"\n序列清单已保存到: {manifest_path}")

        # 同时保存一份 JSON 格式的完整配置记录
        config = {
            'synthesis_params': {
                'boost_gain': self.boost_gain,
                'boost_clip_val': self.boost_clip_val,
                'boost_gain_pos': self.boost_gain_pos,
                'boost_gain_neg': self.boost_gain_neg,
                'strength': self.strength,
                'seed': self.seed,
            },
            'data_sources': {
                'dirt_layers_dir': self.dirt_layers_dir,
                'clean_seqs_csv': self.clean_seqs_csv,
                'clean_seqs_root': self.clean_seqs_root,
                'rgblabels_dir': self.rgblabels_dir,
            },
            'num_dirt_layers': len(self.dirt_layers),
        }
        config_path = os.path.join(self.output_dir, 'synthesis_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"合成配置已保存到: {config_path}")

    def _print_summary(self):
        """打印统计"""
        print("\n" + "="*60)
        print("生成完成")
        print("="*60)
        print(f"总数: {self.stats['total_sequences']}")
        print(f"成功: {self.stats['success_sequences']}")
        print(f"失败: {self.stats['failed_sequences']}")
        print(f"\n输出目录: {self.output_dir}")
        print(f"序列清单: sequences_manifest.csv")
        print(f"合成配置: synthesis_config.json")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="时序训练数据合成"
    )

    # 输入路径
    ap.add_argument("--dirt_layers_dir", type=str,
                    default="dataset/sd_temporal_training/residual_composition_680_heavy2_filtered",
                    help="筛选后的脏污层目录（174 个高质量样本）")
    ap.add_argument("--clean_seqs_csv", type=str,
                    default="dataset/temporal_sequences/dynamics_analysis/high_dynamic_sequences.csv",
                    help="干净序列清单 CSV（默认使用高动态序列）")
    ap.add_argument("--clean_seqs_root", type=str,
                    default="dataset/temporal_sequences/raw_sequences",
                    help="干净序列根目录")
    ap.add_argument("--rgblabels_dir", type=str,
                    default="dataset/woodscape_raw/train/rgbLabels",
                    help="rgbLabels 目录")

    # 输出路径
    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/sequences",
                    help="输出目录")

    # 合成参数（使用确认的 heavy2 参数）
    ap.add_argument("--boost_gain", type=float, default=1.0,
                    help="残差全局增益系数")
    ap.add_argument("--boost_clip_val", type=float, default=150.0,
                    help="软饱和阈值（heavy2: 150.0）")
    ap.add_argument("--boost_gain_pos", type=float, default=1.5,
                    help="正残差增益（亮的部分）")
    ap.add_argument("--boost_gain_neg", type=float, default=8.0,
                    help="负残差增益（暗的部分，heavy2: 8.0）")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="合成强度系数")

    # 生成控制
    ap.add_argument("--max_sequences", type=int, default=None,
                    help="最大生成序列数（测试用）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")

    args = ap.parse_args()

    compositor = TemporalSequenceComposer(
        dirt_layers_dir=args.dirt_layers_dir,
        clean_seqs_csv=args.clean_seqs_csv,
        clean_seqs_root=args.clean_seqs_root,
        rgblabels_dir=args.rgblabels_dir,
        output_dir=args.output_dir,
        boost_gain=args.boost_gain,
        boost_clip_val=args.boost_clip_val,
        boost_gain_pos=args.boost_gain_pos,
        boost_gain_neg=args.boost_gain_neg,
        strength=args.strength,
        seed=args.seed,
    )

    compositor.generate_all(max_sequences=args.max_sequences)


if __name__ == "__main__":
    main()
