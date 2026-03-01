#!/usr/bin/env python3
"""
Stage 2 时序训练脚本

训练目标：
1. 稳定性：模型在时序上的输出应保持稳定
2. 锚定防坍塌：不偏离 Stage 1 单帧标尺

损失函数：
    L = L_stab + β · L_anchor

    其中：
    - L_stab: 稳定性损失（相邻帧差异或窗口内稳定）
    - L_anchor: 锚定损失（使用 Stage 1 teacher）

数据：
    - 时序序列：482 个合成序列
    - 每个序列 32 帧，同一脏污层（静止）
    - 背景高动态变化
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import json

from torch.utils.data import DataLoader

# 添加路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.temporal_baseline import (
    TemporalBaselineDualHead,
    compute_temporal_loss,
    compute_anchor_loss,
)
from models.baseline_dualhead import BaselineDualHead, SeverityAggregator
from data.temporal_dataset import TemporalSequenceDataset, collate_temporal


class Stage2Trainer:
    """Stage 2 时序训练器"""

    def __init__(self,
                 # 模型参数
                 stage1_ckpt: str,
                 temporal_kwargs: Dict,
                 # 数据参数
                 manifest_csv: str,
                 seq_root: str,
                 seq_len: int = 32,
                 # 训练参数
                 batch_size: int = 8,
                 num_epochs: int = 30,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-2,
                 # 损失参数
                 stability_mode: str = 'window',  # 'adjacent' or 'window'
                 window_beta: float = 1.0,
                 anchor_weight: float = 1.0,
                 # 其他
                 device: str = 'cuda',
                 num_workers: int = 4,
                 seed: int = 42,
                 output_dir: str = 'baseline/runs/temporal_stage2'):

        self.stage1_ckpt = stage1_ckpt
        self.temporal_kwargs = temporal_kwargs
        self.manifest_csv = manifest_csv
        self.seq_root = seq_root
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.stability_mode = stability_mode
        self.window_beta = window_beta
        self.anchor_weight = anchor_weight
        self.device = device
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.seed = seed

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化
        self._init_model()
        self._init_data()
        self._init_optimizer()

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')

        # 保存训练配置
        self._save_config()

    def _init_model(self):
        """初始化模型"""
        print("初始化模型...")

        # Student 模型（时序模型）
        self.model = TemporalBaselineDualHead(
            pretrained=False,  # 不使用预训练，稍后手动加载
            temporal_enabled=True,
            temporal_kwargs=self.temporal_kwargs,
        ).to(self.device)

        # Teacher 模型（Stage 1 baseline，冻结）
        print(f"加载 Stage 1 teacher: {self.stage1_ckpt}")
        self.teacher = BaselineDualHead(pretrained=False).to(self.device)

        checkpoint = torch.load(self.stage1_ckpt, map_location=self.device)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.teacher.load_state_dict(state_dict)
        self.teacher.eval()
        # 冻结 teacher 参数
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 加载 student 的 backbone 权重（从 Stage 1）
        print("加载 Stage 1 backbone 权重到 student...")
        self.model.stem.load_state_dict(self.teacher.stem.state_dict())
        self.model.layer1.load_state_dict(self.teacher.layer1.state_dict())
        self.model.layer2.load_state_dict(self.teacher.layer2.state_dict())
        self.model.layer3.load_state_dict(self.teacher.layer3.state_dict())
        self.model.layer4.load_state_dict(self.teacher.layer4.state_dict())
        self.model.pool8.load_state_dict(self.teacher.pool8.state_dict())
        self.model.tile_head.load_state_dict(self.teacher.tile_head.state_dict())
        self.model.glob_fc.load_state_dict(self.teacher.glob_fc.state_dict())
        self.model.aggregator.load_state_dict(self.teacher.aggregator.state_dict())

        print("模型初始化完成")
        print()

    def _init_data(self):
        """初始化数据加载器"""
        print("初始化数据加载器...")

        # 时序数据集
        train_dataset = TemporalSequenceDataset(
            manifest_csv=self.manifest_csv,
            seq_root=self.seq_root,
            seq_len=self.seq_len,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_temporal,
            pin_memory=False,  # 时序数据内存占用大，禁用 pin_memory
        )

        print(f"时序训练数据: {len(train_dataset)} 个序列")
        print(f"Batch size: {self.batch_size}")
        print(f"每个序列帧数: {self.seq_len}")
        print(f"每个 epoch 迭代次数: {len(self.train_loader)}")
        print()

    def _init_optimizer(self):
        """初始化优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # 学习率调度器（余弦退火）
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6,
        )

    def _save_config(self):
        """保存训练配置"""
        config = {
            'stage1_ckpt': self.stage1_ckpt,
            'temporal_kwargs': self.temporal_kwargs,
            'manifest_csv': self.manifest_csv,
            'seq_root': self.seq_root,
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'stability_mode': self.stability_mode,
            'window_beta': self.window_beta,
            'anchor_weight': self.anchor_weight,
            'device': self.device,
            'seed': self.seed,
        }

        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"训练配置已保存到: {config_path}")
        print()

    def train_epoch(self, epoch: int) -> Dict:
        """训练一个 epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_stab_loss = 0.0
        epoch_anchor_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch in pbar:
            # 获取数据
            images = batch['images'].to(self.device)  # [B, T, C, H, W]
            seq_ids = batch['seq_ids']

            B, T, C, H, W = images.shape

            # Student 前向传播
            output_student = self.model(images)  # dict
            s_hat_student = output_student['S_hat']  # [B, 1]

            # Teacher 前向传播（逐帧）
            s_hat_teacher_list = []
            with torch.no_grad():
                for t in range(T):
                    frame = images[:, t]  # [B, C, H, W]
                    output_teacher = self.teacher(frame)  # dict
                    s_hat_teacher_list.append(output_teacher['S_hat'])  # [B, 1]

            s_hat_teacher = torch.stack(s_hat_teacher_list, dim=1)  # [B, T, 1]

            # 扩展 student 输出以匹配时序维度
            # 注意：s_hat_student 目前只用了最后一帧
            # 为了时序损失，我们需要所有帧的 student 预测
            # 这里简化：让 student 对所有帧都做预测
            with torch.enable_grad():
                s_hat_student_temporal = []
                for t in range(T):
                    frame = images[:, t]  # [B, C, H, W]
                    out = self.model.forward_single_frame(frame)
                    s_hat_student_temporal.append(out['S_hat'])  # [B, 1]

                s_hat_student_temporal = torch.stack(s_hat_student_temporal, dim=1)  # [B, T, 1]

            # 计算损失
            stab_loss = compute_temporal_loss(
                s_hat_student_temporal,
                stability_mode=self.stability_mode,
                beta=self.window_beta,
            )

            anchor_loss = compute_anchor_loss(
                s_hat_student_temporal,
                s_hat_teacher,
            )

            loss = stab_loss + self.anchor_weight * anchor_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_stab_loss += stab_loss.item()
            epoch_anchor_loss += anchor_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'stab': f'{stab_loss.item():.4f}',
                'anchor': f'{anchor_loss.item():.4f}',
            })

        num_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'stab_loss': epoch_stab_loss / num_batches,
            'anchor_loss': epoch_anchor_loss / num_batches,
        }

    def train(self):
        """完整训练流程"""
        print("="*60)
        print("Stage 2 时序训练")
        print("="*60)
        print(f"输出目录: {self.output_dir}")
        print(f"设备: {self.device}")
        print(f"训练 Epochs: {self.num_epochs}")
        print(f"学习率: {self.lr}")
        print(f"权重衰减: {self.weight_decay}")
        print()
        print(f"时序损失: {self.stability_mode}")
        print(f"  - window_beta: {self.window_beta}")
        print(f"锚定权重: {self.anchor_weight}")
        print()

        # 训练历史
        history = {
            'train_loss': [],
            'train_stab_loss': [],
            'train_anchor_loss': [],
            'lr': [],
        }

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch

            # 训练
            metrics = self.train_epoch(epoch)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录
            history['train_loss'].append(metrics['loss'])
            history['train_stab_loss'].append(metrics['stab_loss'])
            history['train_anchor_loss'].append(metrics['anchor_loss'])
            history['lr'].append(current_lr)

            # 打印
            print(f"Epoch {epoch}/{self.num_epochs}: "
                  f"Loss={metrics['loss']:.4f}, "
                  f"Stab={metrics['stab_loss']:.4f}, "
                  f"Anchor={metrics['anchor_loss']:.4f}, "
                  f"LR={current_lr:.6f}")

            # 保存检查点
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self._save_checkpoint(epoch, metrics, is_best=True)

            # 定期保存
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, metrics, is_best=False)

        # 保存训练历史
        self._save_history(history)

        print()
        print("训练完成！")
        print(f"最佳 Loss: {self.best_loss:.4f}")

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_loss': self.best_loss,
            'config': {
                'stage1_ckpt': self.stage1_ckpt,
                'temporal_kwargs': self.temporal_kwargs,
            },
        }

        if is_best:
            path = os.path.join(self.output_dir, 'ckpt_best.pth')
        else:
            path = os.path.join(self.output_dir, f'ckpt_epoch_{epoch}.pth')

        torch.save(checkpoint, path)
        print(f"  → 保存: {path}")

    def _save_history(self, history: Dict):
        """保存训练历史"""
        df = pd.DataFrame(history)
        path = os.path.join(self.output_dir, 'history.csv')
        df.to_csv(path, index=False)
        print(f"\n训练历史已保存到: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Stage 2 时序稳定性训练"
    )

    # 模型参数
    ap.add_argument("--stage1_ckpt", type=str,
                    default="baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth",
                    help="Stage 1 模型路径 (S_full_wgap_alpha50 版本)")
    ap.add_argument("--temporal_hidden_dim", type=int, default=256,
                    help="时序模块隐藏层维度")
    ap.add_argument("--temporal_num_layers", type=int, default=3,
                    help="时序模块层数")
    ap.add_argument("--temporal_kernel_size", type=int, default=3,
                    help="时序模块卷积核大小")
    ap.add_argument("--temporal_dropout", type=float, default=0.1,
                    help="时序模块 Dropout 比例")

    # 数据参数
    ap.add_argument("--manifest_csv", type=str,
                    default="dataset/sd_temporal_training/sequences_final/sequences_manifest.csv",
                    help="时序序列清单")
    ap.add_argument("--seq_root", type=str,
                    default="dataset/sd_temporal_training/sequences_final",
                    help="时序序列根目录")
    ap.add_argument("--seq_len", type=int, default=32,
                    help="每个序列的帧数")

    # 训练参数
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size")
    ap.add_argument("--num_epochs", type=int, default=30,
                    help="训练 Epoch 数")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="学习率")
    ap.add_argument("--weight_decay", type=float, default=1e-2,
                    help="权重衰减")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader workers")

    # 损失参数
    ap.add_argument("--stability_mode", type=str, default="window",
                    choices=["adjacent", "window"],
                    help="稳定性模式：相邻帧差异或窗口内稳定")
    ap.add_argument("--window_beta", type=float, default=1.0,
                    help="窗口模式下的相邻帧权重")
    ap.add_argument("--anchor_weight", type=float, default=1.0,
                    help="锚定损失权重")

    # 其他
    ap.add_argument("--device", type=str, default="cuda",
                    help="设备")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")
    ap.add_argument("--output_dir", type=str,
                    default="baseline/runs/temporal_stage2",
                    help="输出目录")

    args = ap.parse_args()

    # 时序模块参数
    temporal_kwargs = {
        'in_features': 512,
        'hidden_dim': args.temporal_hidden_dim,
        'num_layers': args.temporal_num_layers,
        'kernel_size': args.temporal_kernel_size,
        'dropout': args.temporal_dropout,
    }

    # 创建训练器
    trainer = Stage2Trainer(
        stage1_ckpt=args.stage1_ckpt,
        temporal_kwargs=temporal_kwargs,
        manifest_csv=args.manifest_csv,
        seq_root=args.seq_root,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        stability_mode=args.stability_mode,
        window_beta=args.window_beta,
        anchor_weight=args.anchor_weight,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
