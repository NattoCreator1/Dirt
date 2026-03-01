#!/usr/bin/env python3
"""
SD 时序训练数据合成脚本

将干净背景序列与 SD 脏污层合成为时序训练数据。

核心特性：
1. 固定脏污层：一个序列使用相同的脏污层（脏污静止，背景变化）
2. 背景复用策略：同一段背景可以复制成多条序列，每条配不同脏污层
3. 标签生成：为合成序列生成 tile_cov 和 global_score 标签
4. 分层采样：按脏污层类型进行分层采样
5. 均衡采样：添加 bg_id 用于训练时的片段均衡采样

物理先验：
- 脏污在短时内静止（同一条序列内 mask 不变）
- 背景随车辆移动变化（时序模块学习"脏污不变性"）

作者: soiling_project
日期: 2026-02-27
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import cv2


def ensure_dir(p: str):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def load_dirt_layer_manifest(manifest_path: str) -> pd.DataFrame:
    """加载脏污层清单"""
    return pd.read_csv(manifest_path, encoding='utf-8-sig')


def load_sequence_manifest(manifest_path: str) -> pd.DataFrame:
    """加载序列清单"""
    # Handle UTF-8 BOM
    return pd.read_csv(manifest_path, encoding='utf-8-sig')


@dataclass
class DirtLayer:
    """脏污层数据结构"""
    filename: str
    mask_class: int           # 1=Transparent, 2=Semi-transparent, 3=Opaque
    mask_class_name: str
    spill_rate: float
    coverage: float
    seg: int
    level: int
    view: str
    image: np.ndarray         # RGBA 图像 (H, W, 4)


class SDTemporalComposer:
    """SD时序训练数据合成器

    复用策略说明：
    - 同一段背景序列可以被复用 K 次（reuse_per_bg）
    - 每次复用生成一条独立的训练序列，配不同的脏污层
    - 每条序列内脏污层固定（符合物理先验：短时内脏污静止）
    - K 值建议：2-5，避免过度复用导致模型记住背景

    均衡采样：
    - 输出清单包含 bg_id 字段
    - 训练时可按 bg_id 做均衡采样，避免少数片段主导训练
    """

    MAX_REUSE_PER_BG = 10  # 硬上限：防止过度复用

    def __init__(self,
                 dirt_layer_dir: str,
                 dirt_manifest_path: str,
                 sequence_manifest_path: str,
                 sequence_base_dir: str,
                 output_dir: str,
                 metadata_dir: str,
                 num_sequences: int = 1000,
                 reuse_per_bg: int = 3,
                 stratify_by_mask_class: bool = True,
                 seed: int = 42):
        """
        Args:
            dirt_layer_dir: 脏污层图像目录
            dirt_manifest_path: 脏污层清单文件
            sequence_manifest_path: 背景序列清单文件
            sequence_base_dir: 背景序列基准目录
            output_dir: 输出合成序列目录
            metadata_dir: 元数据输出目录
            num_sequences: 合成序列总数
            reuse_per_bg: 每个背景片段的复用次数 (K)，建议 2-5，硬上限 10
            stratify_by_mask_class: 是否按 mask_class 分层采样
            seed: 随机种子
        """
        # 参数验证
        if reuse_per_bg > self.MAX_REUSE_PER_BG:
            raise ValueError(f"reuse_per_bg ({reuse_per_bg}) 超过硬上限 ({self.MAX_REUSE_PER_BG})，"

                           f"这会导致背景片段过度复用，建议使用 2-5")

        self.dirt_layer_dir = dirt_layer_dir
        self.dirt_manifest_path = dirt_manifest_path
        self.sequence_manifest_path = sequence_manifest_path
        self.sequence_base_dir = sequence_base_dir
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir
        self.num_sequences = num_sequences
        self.reuse_per_bg = reuse_per_bg
        self.stratify_by_mask_class = stratify_by_mask_class
        self.seed = seed

        # 创建目录
        ensure_dir(output_dir)
        ensure_dir(metadata_dir)

        # 加载数据
        self.dirt_df = load_dirt_layer_manifest(dirt_manifest_path)
        self.seq_df = load_sequence_manifest(sequence_manifest_path)

        # 统计信息
        self.stats = {
            "total_sequences": 0,
            "total_frames": 0,
            "mask_class_distribution": {},
            "diff_bin_distribution": {},
            "step_distribution": {},
        }

        # 元数据记录
        self.composition_records = []

        # 设置随机种子
        np.random.seed(seed)

    def _load_dirt_layer(self, filename: str) -> Optional[DirtLayer]:
        """加载单个脏污层"""
        # 从清单中获取元数据
        row = self.dirt_df[self.dirt_df['filename'] == filename]
        if row.empty:
            return None

        row = row.iloc[0]

        # 加载图像
        # 注意：清单中的文件名可能没有 _640x480 后缀，需要尝试两种路径
        img_path = os.path.join(self.dirt_layer_dir, filename)
        if not os.path.exists(img_path):
            # 尝试添加 _640x480 后缀（去掉 .png 后再添加）
            if filename.endswith('.png'):
                base = filename[:-4]
                img_path = os.path.join(self.dirt_layer_dir, base + '_640x480.png')

        if not os.path.exists(img_path):
            return None

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # 确保是 RGBA
        if img.shape[2] == 3:
            # 没有alpha通道，创建一个（假设白色背景为透明）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = (gray < 250).astype(np.uint8) * 255  # 近亮色为透明
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha
        elif img.shape[2] == 4:
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            return None

        return DirtLayer(
            filename=row['filename'],
            mask_class=int(row['mask_class']),
            mask_class_name=row['mask_class_name'],
            spill_rate=float(row['spill_rate']),
            coverage=float(row['coverage']),
            seg=int(row['seg']),
            level=int(row['level']),
            view=str(row['view']),
            image=img,
        )

    def _get_tile_cov_from_dirt(self, dirt: DirtLayer) -> np.ndarray:
        """
        从脏污层生成 8x8x4 的 tile coverage 标签

        这里我们使用简化版本：基于脏污层的 alpha 通道和整体 coverage
        来生成一个全局的 tile distribution

        更精确的做法是对 8x8 每个 tile 分别统计覆盖度
        """
        H, W = dirt.image.shape[:2]
        alpha = dirt.image[:, :, 3]  # [H, W]

        # 计算每个 tile 的平均 alpha 值
        tile_h, tile_w = H // 8, W // 8
        tile_cov = np.zeros((8, 8, 4), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                y0, y1 = i * tile_h, (i + 1) * tile_h
                x0, x1 = j * tile_w, (j + 1) * tile_w

                tile_alpha = alpha[y0:y1, x0:x1]
                coverage = (tile_alpha > 127).astype(np.float32).mean()

                # 根据 mask_class 分配到不同类别
                # Clean (0), Transparent (1), Semi (2), Opaque (3)
                if dirt.mask_class == 1:  # Transparent
                    tile_cov[i, j, 1] = coverage
                    tile_cov[i, j, 0] = 1 - coverage
                elif dirt.mask_class == 2:  # Semi-transparent
                    tile_cov[i, j, 2] = coverage
                    tile_cov[i, j, 0] = 1 - coverage
                elif dirt.mask_class == 3:  # Opaque
                    tile_cov[i, j, 3] = coverage
                    tile_cov[i, j, 0] = 1 - coverage

        # 确保 sum(axis=2) = 1
        tile_cov = tile_cov / (tile_cov.sum(axis=2, keepdims=True) + 1e-8)

        return tile_cov

    def _compose_frame(self, background: np.ndarray, dirt: DirtLayer) -> np.ndarray:
        """
        合成单帧：背景 + 脏污层

        Args:
            background: 背景图像 (H, W, 3), uint8 [0, 255]
            dirt: 脏污层

        Returns:
            合成图像 (H, W, 3), uint8 [0, 255]
        """
        # Resize 脏污层到背景尺寸（如果需要）
        dirt_img = dirt.image
        if dirt_img.shape[:2] != background.shape[:2]:
            dirt_img = cv2.resize(dirt_img, (background.shape[1], background.shape[0]),
                                 interpolation=cv2.INTER_AREA)

        # Alpha blending
        alpha = dirt_img[:, :, 3:4].astype(np.float32) / 255.0
        foreground = dirt_img[:, :, :3].astype(np.float32)
        bg = background.astype(np.float32)

        composed = (1 - alpha) * bg + alpha * foreground
        composed = np.clip(composed, 0, 255).astype(np.uint8)

        return composed

    def _sample_dirt_layer(self, seq_info: Dict) -> Optional[DirtLayer]:
        """
        采样脏污层（可选分层采样）

        Args:
            seq_info: 背景序列信息（用于决定采样策略）

        Returns:
            采样的 DirtLayer，失败返回 None
        """
        if self.stratify_by_mask_class:
            # 按 mask_class 分层采样
            # Transparent : Semi : Opaque ≈ 1 : 8 : 12
            # 1 + 8 + 12 = 21 parts
            probs = {1: 1/21, 2: 8/21, 3: 12/21}
            mask_class = np.random.choice([1, 2, 3], p=[probs[1], probs[2], probs[3]])

            candidates = self.dirt_df[self.dirt_df['mask_class'] == mask_class]
        else:
            candidates = self.dirt_df

        if candidates.empty:
            return None

        # 随机选择一个
        row = candidates.sample(n=1).iloc[0]
        return self._load_dirt_layer(row['filename'])

    def compose_sequences(self):
        """合成时序训练数据"""
        # 计算需要处理的背景序列数
        num_bg_seqs = min(len(self.seq_df), self.num_sequences // self.reuse_per_bg)

        print(f"背景序列总数: {len(self.seq_df)}")
        print(f"计划合成序列数: {self.num_sequences}")
        print(f"每个背景片段复用次数 (K): {self.reuse_per_bg}")
        print(f"需要处理的背景序列数: {num_bg_seqs}")

        # 随机选择背景序列
        selected_seqs = self.seq_df.sample(n=num_bg_seqs, random_state=self.seed)

        for _, seq_row in tqdm(selected_seqs.iterrows(), total=len(selected_seqs), desc="合成序列"):
            for reuse_idx in range(self.reuse_per_bg):
                self._compose_single_sequence(seq_row, reuse_idx)

        # 保存元数据
        self._save_metadata()

        # 打印统计
        self._print_stats()

    def _compose_single_sequence(self, seq_row: pd.Series, reuse_idx: int):
        """合成单个时序序列

        Args:
            seq_row: 背景序列信息
            reuse_idx: 复用索引 (0 到 reuse_per_bg-1)
        """
        # 采样脏污层
        dirt = self._sample_dirt_layer(seq_row)
        if dirt is None:
            return

        # 计算标签
        tile_cov = self._get_tile_cov_from_dirt(dirt)
        global_score = float(dirt.coverage)  # 简化：使用 coverage 作为 global_score

        # 获取背景序列路径
        seq_dir = os.path.join(self.sequence_base_dir, seq_row['view_id'], seq_row['seq_id'])

        if not os.path.exists(seq_dir):
            return

        # bg_id: 用于训练时的均衡采样
        # 使用原始背景序列ID作为 bg_id，同一段背景的不同复用会共享同一个 bg_id
        bg_id = seq_row['seq_id']

        # 生成合成序列ID
        comp_seq_id = f"{bg_id}_reuse{reuse_idx:02d}"

        # 创建输出目录
        output_seq_dir = os.path.join(self.output_dir, comp_seq_id)
        ensure_dir(output_seq_dir)

        # 合成每一帧
        frame_count = 0
        for frame_idx in range(seq_row['seq_length']):
            frame_path = os.path.join(seq_dir, f"frame_{frame_idx:04d}.jpg")
            if not os.path.exists(frame_path):
                continue

            # 读取背景
            background = cv2.imread(frame_path)
            if background is None:
                continue

            # 合成（脏污层在整个序列中固定）
            composed = self._compose_frame(background, dirt)

            # 保存
            output_frame_path = os.path.join(output_seq_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(output_frame_path, composed)
            frame_count += 1

        # 保存标签
        labels_path = os.path.join(output_seq_dir, "labels.npz")
        np.savez_compressed(
            labels_path,
            tile_cov=tile_cov,
            global_score=global_score,
            S=global_score,  # 兼容性
        )

        # 记录元数据（添加 bg_id 字段）
        record = {
            "comp_seq_id": comp_seq_id,
            "bg_id": bg_id,  # 新增：用于训练时均衡采样
            "reuse_idx": reuse_idx,  # 新增：复用索引
            "bg_seq_id": seq_row['seq_id'],
            "dirt_filename": dirt.filename,
            "dirt_mask_class": dirt.mask_class,
            "dirt_mask_class_name": dirt.mask_class_name,
            "dirt_spill_rate": dirt.spill_rate,
            "dirt_coverage": dirt.coverage,
            "bg_view_id": seq_row['view_id'],
            "bg_step": seq_row['step'],
            "bg_diff_bin": seq_row['diff_bin'],
            "bg_duration_sec": seq_row['duration_sec'],
            "seq_length": frame_count,
            "global_score": global_score,
            "output_dir": output_seq_dir,
        }

        self.composition_records.append(record)

        # 更新统计
        self.stats["total_sequences"] += 1
        self.stats["total_frames"] += frame_count
        self.stats["mask_class_distribution"][dirt.mask_class_name] = \
            self.stats["mask_class_distribution"].get(dirt.mask_class_name, 0) + 1
        self.stats["diff_bin_distribution"][seq_row['diff_bin']] = \
            self.stats["diff_bin_distribution"].get(seq_row['diff_bin'], 0) + 1
        self.stats["step_distribution"][seq_row['step']] = \
            self.stats["step_distribution"].get(seq_row['step'], 0) + 1

    def _save_metadata(self):
        """保存元数据"""
        # 保存合成序列清单
        manifest_path = os.path.join(self.metadata_dir, "temporal_composition_manifest.csv")
        df = pd.DataFrame(self.composition_records)
        df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

        # 保存统计信息
        stats_path = os.path.join(self.metadata_dir, "composition_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        # 保存配置
        config_path = os.path.join(self.metadata_dir, "composition_config.json")
        config = {
            "dirt_layer_dir": self.dirt_layer_dir,
            "dirt_manifest_path": self.dirt_manifest_path,
            "sequence_manifest_path": self.sequence_manifest_path,
            "sequence_base_dir": self.sequence_base_dir,
            "num_sequences": self.num_sequences,
            "reuse_per_bg": self.reuse_per_bg,
            "stratify_by_mask_class": self.stratify_by_mask_class,
            "seed": self.seed,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("SD时序数据合成完成")
        print("="*60)
        print(f"合成序列数: {self.stats['total_sequences']}")
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"平均每序列帧数: {self.stats['total_frames'] / max(1, self.stats['total_sequences']):.1f}")

        print("\n脏污类型分布:")
        for mask_name, count in self.stats["mask_class_distribution"].items():
            ratio = count / max(1, self.stats["total_sequences"]) * 100
            print(f"  {mask_name}: {count} ({ratio:.1f}%)")

        print("\n背景帧差分布:")
        for diff_bin, count in sorted(self.stats["diff_bin_distribution"].items()):
            ratio = count / max(1, self.stats["total_sequences"]) * 100
            print(f"  {diff_bin}: {count} ({ratio:.1f}%)")

        print("\n步长分布:")
        for step, count in sorted(self.stats["step_distribution"].items()):
            ratio = count / max(1, self.stats["total_sequences"]) * 100
            print(f"  s={step}: {count} ({ratio:.1f}%)")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="SD时序训练数据合成脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
复用策略说明:
  同一段背景序列可以复制成多条独立训练序列，每条配不同脏污层。
  这符合物理先验：脏污静止（同一条序列内），背景变化。

  参数 reuse_per_bg 控制每个背景片段的复用次数 (K)。
  建议 K=2-5，避免过度复用导致模型记住背景。

  训练时可使用 bg_id 字段做均衡采样，确保不同背景片段贡献均衡。
        """
    )

    ap.add_argument("--dirt_layer_dir", type=str,
                    default="dataset/sd_temporal_training/dirt_layers",
                    help="脏污层图像目录")
    ap.add_argument("--dirt_manifest", type=str,
                    default="dataset/sd_temporal_training/metadata/dirt_layer_manifest.csv",
                    help="脏污层清单文件")
    ap.add_argument("--sequence_manifest", type=str,
                    default="dataset/temporal_sequences/metadata/sequence_manifest.csv",
                    help="背景序列清单文件")
    ap.add_argument("--sequence_base_dir", type=str,
                    default="dataset/temporal_sequences/raw_sequences",
                    help="背景序列基准目录")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/sd_temporal_training/composed_sequences",
                    help="输出合成序列目录")
    ap.add_argument("--metadata_dir", type=str,
                    default="dataset/sd_temporal_training/composition_metadata",
                    help="元数据输出目录")

    ap.add_argument("--num_sequences", type=int, default=1000,
                    help="合成序列总数")
    ap.add_argument("--reuse_per_bg", type=int, default=3,
                    help="每个背景片段的复用次数 (K)，建议 2-5，硬上限 10")
    ap.add_argument("--stratify", action="store_true", default=True,
                    help="按mask_class分层采样")

    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")

    args = ap.parse_args()

    # 创建合成器
    composer = SDTemporalComposer(
        dirt_layer_dir=args.dirt_layer_dir,
        dirt_manifest_path=args.dirt_manifest,
        sequence_manifest_path=args.sequence_manifest,
        sequence_base_dir=args.sequence_base_dir,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        num_sequences=args.num_sequences,
        reuse_per_bg=args.reuse_per_bg,
        stratify_by_mask_class=args.stratify,
        seed=args.seed,
    )

    # 合成序列
    composer.compose_sequences()


if __name__ == "__main__":
    main()
