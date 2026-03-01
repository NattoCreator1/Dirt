#!/usr/bin/env python3
"""
混合数据集训练脚本 - WoodScape + SD Synthetic

功能:
1. 支持多个数据源的混合训练
2. 分层采样确保每个batch包含两种数据源
3. 支持自定义混合比例
4. 与baseline train_baseline.py兼容的参数

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from PIL import Image
import cv2

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.models.baseline_dualhead import BaselineDualHead, SeverityAggregator
from baseline.datasets.woodscape_index import WoodscapeTileDataset


class MixedDataset(Dataset):
    """
    混合数据集 - 支持多个数据源

    通过dataset_id区分不同数据源，实现分层采样
    """
    def __init__(self, datasets_dict, sample_ratio=None, seed=42):
        """
        Args:
            datasets_dict: {name: dataset} 字典
            sample_ratio: {name: ratio} 采样比例，None则使用原始数据集大小
            seed: 随机种子
        """
        self.datasets = datasets_dict
        self.dataset_names = list(datasets_dict.keys())
        self.dataset_indices = {}
        self.rng = np.random.default_rng(seed)

        # 计算每个数据集的累积索引
        current_idx = 0
        for name, dataset in datasets_dict.items():
            size = len(dataset)
            self.dataset_indices[name] = {
                'start': current_idx,
                'end': current_idx + size,
                'size': size,
                'dataset': dataset
            }
            current_idx += size

        self.total_size = current_idx

        # 设置采样比例
        if sample_ratio is None:
            # 使用原始数据集大小的比例
            total_original = sum(len(d) for d in datasets_dict.values())
            self.sample_ratio = {
                name: len(d) / total_original
                for name, d in datasets_dict.items()
            }
        else:
            # 使用指定的采样比例
            total_ratio = sum(sample_ratio.values())
            self.sample_ratio = {
                name: ratio / total_ratio
                for name, ratio in sample_ratio.items()
            }

        print(f"\n混合数据集配置:")
        for name in self.dataset_names:
            info = self.dataset_indices[name]
            print(f"  {name}: {info['size']} samples, ratio={self.sample_ratio[name]:.2%}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 找到对应的数据集
        for name, info in self.dataset_indices.items():
            if info['start'] <= idx < info['end']:
                local_idx = idx - info['start']
                sample = info['dataset'].__getitem__(local_idx)
                # 添加dataset_id标识
                sample['dataset_id'] = self.dataset_names.index(name)
                sample['dataset_name'] = name
                return sample

        raise IndexError(f"Index {idx} out of range")


class StratifiedBatchSampler(Sampler):
    """
    分层采样器 - 确保每个batch包含所有数据源

    按照指定比例从每个数据源采样，组成混合batch
    """
    def __init__(self, datasets_dict, batch_size, sample_ratio=None, seed=42, drop_last=True):
        """
        Args:
            datasets_dict: {name: dataset} 字典
            batch_size: 总batch size
            sample_ratio: {name: ratio} 采样比例
            seed: 随机种子
            drop_last: 是否丢弃最后不完整的batch
        """
        self.datasets = datasets_dict
        self.dataset_names = list(datasets_dict.keys())
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        # 计算每个数据源在每个batch中的采样数量
        if sample_ratio is None:
            # 使用原始数据集大小的比例
            total_size = sum(len(d) for d in datasets_dict.values())
            sample_ratio = {
                name: len(d) / total_size
                for name, d in datasets_dict.items()
            }

        # 计算每个batch中各数据源的样本数
        self.batch_counts = {}
        remaining = batch_size
        for name, ratio in sample_ratio.items():
            count = max(1, int(batch_size * ratio))  # 至少1个
            self.batch_counts[name] = count
            remaining -= count

        # 调整以确保总和等于batch_size
        if remaining != 0:
            # 简单调整：加到最大的那个数据源
            max_name = max(self.batch_counts, key=self.batch_counts.get)
            self.batch_counts[max_name] += remaining

        print(f"\n分层采样配置 (batch_size={batch_size}):")
        for name, count in self.batch_counts.items():
            print(f"  {name}: {count} samples/batch")

        # 为每个数据集创建索引列表
        self.indices = {
            name: list(range(len(dataset)))
            for name, dataset in datasets_dict.items()
        }

        # 计算epoch长度（以最小数据源为准）
        min_size = min(len(dataset) for dataset in datasets_dict.values())
        self.epoch_size = min_size // min(self.batch_counts.values())

        if drop_last:
            self.num_batches = self.epoch_size
        else:
            self.num_batches = self.epoch_size + 1

    def __iter__(self):
        # 每个epoch开始时打乱索引
        shuffled_indices = {
            name: self.rng.permutation(indices)
            for name, indices in self.indices.items()
        }

        for batch_idx in range(self.num_batches):
            batch_indices = []
            for name in self.dataset_names:
                count = self.batch_counts[name]
                start_idx = batch_idx * count
                indices = shuffled_indices[name]

                # 循环采样
                for i in range(count):
                    global_idx = (start_idx + i) % len(indices)
                    batch_indices.append((name, indices[global_idx]))

            self.rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


def collate_fn_with_dataset(batch):
    """
    自定义collate函数，处理dataset_id和dataset_name
    """
    # 标准collate
    collated = {}
    for key in batch[0].keys():
        if key == 'dataset_name':
            collated[key] = [item[key] for item in batch]
        else:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = torch.tensor(values)

    return collated


def train_one_epoch(model, dataloader, optimizer, device, loss_config, epoch):
    """单个epoch训练"""
    model.train()

    total_loss = 0.0
    total_tile_loss = 0.0
    total_global_loss = 0.0
    total_cons_loss = 0.0

    loss_fn_tile = nn.CrossEntropyLoss(weight=torch.tensor([0.0, 0.15, 0.50, 1.0]).to(device))
    loss_fn_global = nn.L1Loss()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # 获取数据
        rgb = batch['rgb'].to(device)
        tile_target = batch['tile_cov'].to(device)
        global_target = batch['S'].to(device)

        # 前向传播
        optimizer.zero_grad()

        tile_pred, global_pred = model(rgb)
        s_agg = model.severity_aggregator(tile_pred)

        # 计算损失
        loss_tile = loss_fn_tile(tile_pred.permute(0,2,3,1).reshape(-1,4), tile_target.permute(0,2,3,1).reshape(-1,4))
        loss_global = loss_fn_global(global_pred.squeeze(), global_target)

        if loss_config['mu_cons'] > 0:
            loss_cons = loss_fn_global(s_agg.squeeze(), global_target)
            loss = loss_tile + loss_config['lambda_glob'] * loss_global + loss_config['mu_cons'] * loss_cons
        else:
            loss_cons = torch.tensor(0.0)
            loss = loss_tile + loss_config['lambda_glob'] * loss_global

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_tile_loss += loss_tile.item()
        total_global_loss += loss_global.item()
        total_cons_loss += loss_cons.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'tile_loss': total_tile_loss / n_batches,
        'global_loss': total_global_loss / n_batches,
        'cons_loss': total_cons_loss / n_batches,
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()

    total_loss = 0.0
    tile_mae_list = []
    global_pred_list = []
    global_target_list = []
    s_agg_list = []

    loss_fn_tile = nn.CrossEntropyLoss(weight=torch.tensor([0.0, 0.15, 0.50, 1.0]).to(device))
    loss_fn_global = nn.L1Loss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            rgb = batch['rgb'].to(device)
            tile_target = batch['tile_cov'].to(device)
            global_target = batch['S'].to(device)

            tile_pred, global_pred = model(rgb)
            s_agg = model.severity_aggregator(tile_pred)

            # 计算损失
            loss_tile = loss_fn_tile(tile_pred.permute(0,2,3,1).reshape(-1,4), tile_target.permute(0,2,3,1).reshape(-1,4))
            loss_global = loss_fn_global(global_pred.squeeze(), global_target)
            loss = loss_tile + loss_global

            total_loss += loss.item()

            # 收集预测结果
            tile_pred_flat = tile_pred.permute(0,2,3,1).reshape(-1,4).cpu().numpy()
            tile_target_flat = tile_target.permute(0,2,3,1).reshape(-1,4).cpu().numpy()
            tile_mae = np.abs(tile_pred_flat - tile_target_flat).mean()
            tile_mae_list.append(tile_mae)

            global_pred_list.extend(global_pred.squeeze().cpu().numpy())
            global_target_list.extend(global_target.cpu().numpy())
            s_agg_list.extend(s_agg.squeeze().cpu().numpy())

    n_batches = len(dataloader)

    # 计算指标
    tile_mae = np.mean(tile_mae_list)
    global_mae = np.mean(np.abs(np.array(global_pred_list) - np.array(global_target_list)))
    global_rmse = np.sqrt(np.mean((np.array(global_pred_list) - np.array(global_target_list))**2))
    gap_mae = np.mean(np.abs(np.array(s_agg_list) - np.array(global_pred_list)))

    return {
        'loss': total_loss / n_batches,
        'tile_mae': tile_mae,
        'glob_mae': global_mae,
        'glob_rmse': global_rmse,
        'gap_mae': gap_mae,
    }


def main():
    parser = argparse.ArgumentParser(description="混合数据集训练")

    # WoodScape数据
    parser.add_argument("--woodscape_index", type=str,
                       default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv",
                       help="WoodScape索引CSV")
    parser.add_argument("--woodscape_img_root", type=str, default=".",
                       help="WoodScape图像根目录")
    parser.add_argument("--woodscape_split_col", type=str, default="split",
                       help="WoodScape split列名")

    # SD Synthetic数据
    parser.add_argument("--synthetic_index", type=str,
                       default="sd_scripts/lora/accepted_20260224_164715/index_synthetic_20260224_180255.csv",
                       help="SD合成数据索引CSV")
    parser.add_argument("--synthetic_img_root", type=str, default=".",
                       help="SD合成数据图像根目录")
    parser.add_argument("--synthetic_split_col", type=str, default="split",
                       help="SD合成数据split列名")

    # 混合比例
    parser.add_argument("--woodscape_ratio", type=float, default=0.85,
                       help="WoodScape采样比例")
    parser.add_argument("--synthetic_ratio", type=float, default=0.15,
                       help="SD Synthetic采样比例")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda_glob", type=float, default=1.0)
    parser.add_argument("--mu_cons", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # 输出
    parser.add_argument("--run_name", type=str, default=None,
                       help="运行名称 (默认：自动生成)")
    parser.add_argument("--out_root", type=str, default="baseline/runs_mixed")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mix_info = f"ws{int(args.woodscape_ratio*100)}_sd{int(args.synthetic_ratio*100)}"
        args.run_name = f"mixed_{mix_info}_{timestamp}"

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存运行参数
    run_args_path = out_dir / "run_args.json"
    with open(run_args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集
    print("\n加载数据集...")

    # WoodScape数据集
    woodscape_train = WoodscapeTileDataset(
        index_csv=args.woodscape_index,
        img_root=args.woodscape_img_root,
        split_col=args.woodscape_split_col,
        train_split="train",
        val_split="val",
        global_target="S",
    )
    woodscape_val = WoodscapeTileDataset(
        index_csv=args.woodscape_index,
        img_root=args.woodscape_img_root,
        split_col=args.woodscape_split_col,
        train_split="train",
        val_split="val",
        global_target="S",
    )
    woodscape_val.set_split("val")

    # SD Synthetic数据集
    synthetic_train = WoodscapeTileDataset(
        index_csv=args.synthetic_index,
        img_root=args.synthetic_img_root,
        split_col=args.synthetic_split_col,
        train_split="train",
        val_split="val",
        global_target="S",
    )
    synthetic_val = WoodscapeTileDataset(
        index_csv=args.synthetic_index,
        img_root=args.synthetic_img_root,
        split_col=args.synthetic_split_col,
        train_split="train",
        val_split="val",
        global_target="S",
    )
    synthetic_val.set_split("val")

    print(f"WoodScape训练集: {len(woodscape_train)}")
    print(f"WoodScape验证集: {len(woodscape_val)}")
    print(f"SD Synthetic训练集: {len(synthetic_train)}")
    print(f"SD Synthetic验证集: {len(synthetic_val)}")

    # 创建混合数据集
    train_datasets = {
        'woodscape': woodscape_train,
        'synthetic': synthetic_train,
    }

    # 分层采样
    sample_ratio = {
        'woodscape': args.woodscape_ratio,
        'synthetic': args.synthetic_ratio,
    }

    train_sampler = StratifiedBatchSampler(
        train_datasets,
        batch_size=args.batch_size,
        sample_ratio=sample_ratio,
        seed=args.seed,
        drop_last=True
    )

    # 创建DataLoader
    train_dataset = MixedDataset(train_datasets, sample_ratio=sample_ratio, seed=args.seed)

    def collate_fn(batch):
        return collate_fn_with_dataset(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(woodscape_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    print("\n创建模型...")
    model = BaselineDualHead(
        backbone='resnet18',
        pretrained=True,
        num_classes=4,
        fusion_coeffs={'alpha': 0.5, 'beta': 0.4, 'gamma': 0.1},
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环
    loss_config = {
        'lambda_glob': args.lambda_glob,
        'mu_cons': args.mu_cons,
    }

    metrics_list = []
    best_val_mae = float('inf')

    print(f"\n开始训练...")
    print(f"总epochs: {args.epochs}")
    print(f"每个epoch包含约 {len(train_loader)} 个batch")

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_config, epoch)

        # 验证 (WoodScape)
        val_metrics = evaluate(model, val_loader, device)

        # 打印结果
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Tile MAE: {val_metrics['tile_mae']:.4f}")
        print(f"  Val Global MAE: {val_metrics['glob_mae']:.4f}")
        print(f"  Val Global RMSE: {val_metrics['glob_rmse']:.4f}")

        # 保存指标
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_tile_mae': val_metrics['tile_mae'],
            'val_glob_mae': val_metrics['glob_mae'],
            'val_glob_rmse': val_metrics['glob_rmse'],
        }
        metrics_list.append(metrics)

        # 保存最佳模型
        if val_metrics['glob_mae'] < best_val_mae:
            best_val_mae = val_metrics['glob_mae']
            ckpt_path = out_dir / "ckpt_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ 保存最佳模型 (MAE={best_val_mae:.4f})")

        # 保存最新模型
        ckpt_path = out_dir / "ckpt_last.pth"
        torch.save(model.state_dict(), ckpt_path)

    # 保存指标
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n训练完成!")
    print(f"最佳验证MAE: {best_val_mae:.4f}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
