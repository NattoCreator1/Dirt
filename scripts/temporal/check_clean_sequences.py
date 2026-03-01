#!/usr/bin/env python3
"""
检查干净时序数据中是否存在脏污，并筛选出真正干净的序列

流程：
1. 加载 baseline soiling 检测模型
2. 对所有时序帧进行推理
3. 统计每个序列的平均脏污程度
4. 筛选出真正干净的序列（S < threshold）
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

import cv2


def load_model(model_path: str, device: str = 'cuda') -> nn.Module:
    """加载训练好的模型"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../baseline'))
    from models.baseline_dualhead import BaselineDualHead

    model = BaselineDualHead(pretrained=False)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def analyze_sequences(clean_seqs_dir: str,
                     model: nn.Module,
                     device: str,
                     batch_size: int = 32,
                     s_threshold: float = 0.1) -> pd.DataFrame:
    """
    分析所有时序序列的脏污程度

    Returns:
        DataFrame with columns: [view, seq_id, num_frames, S_mean, S_max, S_std, is_clean]
    """
    views = ['f', 'lf', 'rf', 'lr', 'r', 'rr']
    results = []

    for view in views:
        view_dir = os.path.join(clean_seqs_dir, view)
        if not os.path.exists(view_dir):
            continue

        seq_dirs = [d for d in os.listdir(view_dir) if os.path.isdir(os.path.join(view_dir, d))]

        for seq_dir in tqdm(seq_dirs, desc=f"Analyzing {view}"):
            seq_path = os.path.join(view_dir, seq_dir)
            frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])

            if len(frame_files) == 0:
                continue

            # 收集该序列的所有帧路径
            frame_paths = [os.path.join(seq_path, f) for f in frame_files]

            # 推理获取所有帧的 S 分数
            s_scores = []

            # 批量推理
            for i in range(0, len(frame_paths), batch_size):
                batch_paths = frame_paths[i:i+batch_size]

                # 读取图像
                images = []
                for path in batch_paths:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (640, 480))
                        images.append(img)

                if len(images) == 0:
                    continue

                # 转换为 tensor
                images = np.array(images)
                images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
                images = images.to(device)

                # 归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                images = (images - mean) / std

                with torch.no_grad():
                    output = model(images)
                    s_pred = output['S_hat']  # Use S_hat (global head prediction)
                    s_scores.extend(s_pred.cpu().numpy().flatten())

            if len(s_scores) == 0:
                continue

            s_scores = np.array(s_scores)

            results.append({
                'view': view,
                'seq_id': seq_dir,
                'num_frames': len(frame_files),
                'S_mean': s_scores.mean(),
                'S_max': s_scores.max(),
                'S_min': s_scores.min(),
                'S_std': s_scores.std(),
                'is_clean': s_scores.mean() < s_threshold,
            })

    return pd.DataFrame(results)


def filter_clean_sequences(df: pd.DataFrame,
                            max_s_threshold: float = 0.1,
                            min_frames: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    筛选干净序列

    Args:
        df: 分析结果 DataFrame
        max_s_threshold: 最大允许的平均 S 分数
        min_frames: 最小帧数要求

    Returns:
        clean_df: 干净序列
        dirty_df: 脏污序列
    """
    # 筛选条件
    mask = (
        (df['S_mean'] < max_s_threshold) &
        (df['num_frames'] >= min_frames)
    )

    clean_df = df[mask].sort_values('S_mean')
    dirty_df = df[~mask].sort_values('S_mean', ascending=False)

    return clean_df, dirty_df


def main():
    ap = argparse.ArgumentParser(
        description="检查并筛选干净的时序序列"
    )

    ap.add_argument("--model_path", type=str,
                    default="baseline/runs/model1_r18_640x480/ckpt_best.pth",
                    help="模型路径")
    ap.add_argument("--clean_seqs_dir", type=str,
                    default="dataset/temporal_sequences/raw_sequences",
                    help="干净时序序列目录")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/temporal_sequences/checked",
                    help="输出目录")
    ap.add_argument("--s_threshold", type=float, default=0.1,
                    help="干净阈值（S_mean < threshold 视为干净）")
    ap.add_argument("--min_frames", type=int, default=10,
                    help="最小帧数要求")
    ap.add_argument("--device", type=str, default="cuda",
                    help="设备 (cuda/cpu)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="批大小")

    args = ap.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("干净时序序列检查")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"时序数据目录: {args.clean_seqs_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"干净阈值: S < {args.s_threshold}")
    print(f"设备: {args.device}")
    print()

    # 加载模型
    print("加载模型...")
    model = load_model(args.model_path, args.device)
    print("模型加载完成")
    print()

    # 分析所有序列
    print("开始分析时序序列...")
    df = analyze_sequences(
        args.clean_seqs_dir,
        model,
        args.device,
        batch_size=args.batch_size,
        s_threshold=args.s_threshold
    )

    print()
    print(f"分析完成，共 {len(df)} 个序列")
    print()

    # 筛选干净序列
    clean_df, dirty_df = filter_clean_sequences(
        df,
        max_s_threshold=args.s_threshold,
        min_frames=args.min_frames
    )

    print("="*60)
    print("筛选结果统计")
    print("="*60)
    print(f"总序列数: {len(df)}")
    print(f"干净序列: {len(clean_df)} ({len(clean_df)/len(df)*100:.1f}%)")
    print(f"脏污序列: {len(dirty_df)} ({len(dirty_df)/len(df)*100:.1f}%)")
    print()

    if len(clean_df) > 0:
        print(f"干净序列 S_mean 范围: [{clean_df['S_mean'].min():.3f}, {clean_df['S_mean'].max():.3f}]")
    if len(dirty_df) > 0:
        print(f"脏污序列 S_mean 范围: [{dirty_df['S_mean'].min():.3f}, {dirty_df['S_mean'].max():.3f}]")

    print()
    print(f"各视角统计:")
    for view in ['f', 'lf', 'rf', 'lr', 'r', 'rr']:
        clean_count = len(clean_df[clean_df['view'] == view]) if len(clean_df) > 0 else 0
        dirty_count = len(dirty_df[dirty_df['view'] == view]) if len(dirty_df) > 0 else 0
        total = clean_count + dirty_count
        if total > 0:
            print(f"  {view}: {clean_count}/{total} 干净 ({clean_count/total*100:.1f}%)")

    # 保存结果
    df.to_csv(os.path.join(args.output_dir, 'all_sequences_analysis.csv'), index=False)
    clean_df.to_csv(os.path.join(args.output_dir, 'clean_sequences.csv'), index=False)
    dirty_df.to_csv(os.path.join(args.output_dir, 'dirty_sequences.csv'), index=False)

    print()
    print(f"结果已保存到: {args.output_dir}/")
    print(f"  - all_sequences_analysis.csv: 所有序列分析结果")
    print(f"  - clean_sequences.csv: 干净序列清单")
    print(f"  - dirty_sequences.csv: 脏污序列清单")
    print("="*60)


if __name__ == "__main__":
    import cv2
    main()
