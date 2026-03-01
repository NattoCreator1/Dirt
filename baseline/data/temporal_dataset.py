"""
时序数据集加载器

用于 Stage 2 时序训练，加载合成的时序序列数据：
- 同一脏污层应用于整个序列的所有帧（脏污静止）
- 背景随时间变化（使用高动态干净序列）

数据格式：
  sequences_final/
    ├── {dirt_layer_id}_{view}_{seq_id}/
    │   ├── frame_0000.jpg
    │   ├── frame_0001.jpg
    │   └── ...
    └── sequences_manifest.csv
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from typing import Dict, List, Optional, Tuple
import random


class TemporalSequenceDataset(Dataset):
    """
    时序序列数据集

    Args:
        manifest_csv: 序列清单 CSV 路径
        seq_root: 序列根目录
        seq_len: 每个序列加载的帧数（-1 表示全部帧）
        transform: 可选的数据增强
    """

    def __init__(self,
                 manifest_csv: str,
                 seq_root: str,
                 seq_len: int = -1,
                 transform=None):
        self.manifest_csv = manifest_csv
        self.seq_root = seq_root
        self.seq_len = seq_len
        self.transform = transform

        # 加载序列清单
        self.df = pd.read_csv(manifest_csv)
        print(f"加载了 {len(self.df)} 个时序序列")

        # 统计序列长度
        if 'num_frames' in self.df.columns:
            self.frame_counts = self.df['num_frames'].values
            print(f"序列帧数范围: [{self.frame_counts.min()}, {self.frame_counts.max()}]")
        else:
            self.frame_counts = None

    def __len__(self) -> int:
        return len(self.df)

    def _load_sequence(self, seq_dir: str) -> np.ndarray:
        """加载一个序列的所有帧"""
        frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])

        frames = []
        for f in frame_files:
            path = os.path.join(seq_dir, f)
            img = cv2.imread(path)
            if img is not None:
                # 统一 resize 到 640x480
                if img.shape[:2] != (480, 640):
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                frames.append(img)

        if len(frames) == 0:
            return None

        # Stack: [T, H, W, C]
        frames = np.stack(frames, axis=0)
        return frames

    def _sample_frames(self, frames: np.ndarray, num_frames: int) -> np.ndarray:
        """
        从序列中采样帧

        Args:
            frames: [T, H, W, C]
            num_frames: 要采样的帧数

        Returns:
            [num_frames, H, W, C]
        """
        total_frames = len(frames)

        if num_frames >= total_frames:
            # 如果要求帧数 >= 总帧数，返回全部（可能重复）
            indices = list(range(total_frames))
            while len(indices) < num_frames:
                indices.append(total_frames - 1)  # 重复最后一帧
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        return frames[indices]

    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本

        Returns:
            dict containing:
                images: [T, C, H, W] Tensor
                seq_id: 序列标识
                seq_dir: 序列目录
        """
        row = self.df.iloc[idx]

        # 检查 output_dir 是否已经是绝对路径或包含完整路径
        output_dir = row['output_dir']
        if os.path.isabs(output_dir):
            # 已经是绝对路径
            seq_dir = output_dir
        elif output_dir.startswith(self.seq_root) or output_dir.startswith('dataset/'):
            # 包含完整路径，直接使用
            seq_dir = output_dir if os.path.isabs(output_dir) else os.path.join(os.getcwd(), output_dir)
        else:
            # 相对路径，需要与 seq_root 拼接
            seq_dir = os.path.join(self.seq_root, output_dir)

        # 加载序列
        frames = self._load_sequence(seq_dir)
        if frames is None:
            raise ValueError(f"无法加载序列: {seq_dir}")

        # 采样帧（如果需要）
        if self.seq_len > 0:
            frames = self._sample_frames(frames, self.seq_len)

        # 转换为 Tensor: [T, H, W, C] -> [T, C, H, W]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # 归一化（ImageNet）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        # 序列标识
        seq_id = f"{row['dirt_layer_id']}_{row['view']}_{row['clean_seq_id']}"

        return {
            'images': frames,  # [T, C, H, W]
            'seq_id': seq_id,
            'seq_dir': row['output_dir'],
        }


class MixedDataLoader:
    """
    混合数据加载器

    混合单帧数据（WoodScape）和时序数据（合成序列）
    支持 Stage 3 混合微调

    Args:
        single_frame_loader: 单帧数据 DataLoader
        temporal_loader: 时序数据 DataLoader
        temporal_ratio: 时序数据占比（0-1）
    """

    def __init__(self,
                 single_frame_loader,
                 temporal_loader,
                 temporal_ratio: float = 0.5,
                 seed: int = 42):
        self.single_loader = single_frame_loader
        self.temporal_loader = temporal_loader
        self.temporal_ratio = temporal_ratio
        self.seed = seed

        # 设置迭代器
        self.single_iter = iter(single_frame_loader)
        self.temporal_iter = iter(temporal_loader)

        self.single_len = len(single_frame_loader)
        self.temporal_len = len(temporal_loader)

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        """
        返回混合批次

        Returns:
            dict containing:
                images: 图像 Tensor
                is_temporal: 是否为时序数据
                ...
        """
        # 根据比例决定使用哪个数据源
        if random.random() < self.temporal_ratio:
            try:
                batch = next(self.temporal_iter)
                batch['is_temporal'] = True
                return batch
            except StopIteration:
                self.temporal_iter = iter(self.temporal_loader)
                batch = next(self.temporal_iter)
                batch['is_temporal'] = True
                return batch
        else:
            try:
                batch = next(self.single_iter)
                batch['is_temporal'] = False
                return batch
            except StopIteration:
                self.single_iter = iter(self.single_loader)
                batch = next(self.single_iter)
                batch['is_temporal'] = False
                return batch

    def __len__(self) -> int:
        # 返回一个 epoch 的迭代次数
        # 这里简化为两个数据源长度的较大值
        return max(self.single_len, self.temporal_len)


def collate_temporal(batch):
    """
    时序数据的 collate 函数

    处理变长序列（如果未来支持）
    """
    # 目前所有序列都是相同长度，直接 stack
    images = torch.stack([item['images'] for item in batch])  # [B, T, C, H, W]

    return {
        'images': images,
        'seq_ids': [item['seq_id'] for item in batch],
        'seq_dirs': [item['seq_dir'] for item in batch],
    }


def collate_mixed(batch):
    """
    混合数据的 collate 函数
    """
    if batch[0].get('is_temporal', False):
        # 时序数据
        return collate_temporal(batch)
    else:
        # 单帧数据（使用现有的 collate 函数）
        from baseline.data.dataset import default_collate
        return default_collate(batch)
