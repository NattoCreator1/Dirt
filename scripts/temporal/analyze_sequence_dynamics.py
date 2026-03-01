#!/usr/bin/env python3
"""
分析干净时序序列的帧间变化程度

目的：筛选出帧间变化足够大的序列，确保时序训练数据有效

指标：
1. 帧间平均差异（相邻帧的光流/像素变化）
2. 首尾帧差异
3. 序列动态性评分（综合指标）
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import cv2


def compute_frame_diff(frame1, frame2, method='pixel'):
    """
    计算两帧之间的差异

    Args:
        frame1, frame2: BGR 图像
        method: 'pixel' (像素级差异) 或 'feature' (特征级差异)

    Returns:
        float: 差异分数
    """
    # 确保尺寸一致
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    if method == 'pixel':
        # 像素级差异（归一化到 [0, 1]）
        diff = cv2.absdiff(frame1, frame2).astype(np.float32)
        diff_score = diff.mean() / 255.0
        return diff_score

    elif method == 'feature':
        # 使用 ORB 特征匹配（更鲁棒）
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(frame1, None)
        kp2, des2 = orb.detectAndCompute(frame2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # 特征点太少，回退到像素差异
            return compute_frame_diff(frame1, frame2, method='pixel')

        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            return 0.0

        # 计算匹配距离（越小越相似）
        distances = [m.distance for m in matches]
        avg_distance = sum(distances) / len(distances)

        # 归一化（ORB 汉明距离范围 [0, 256]）
        return avg_distance / 256.0

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_sequence_dynamics(seq_path: str,
                             sample_rate: int = 4,
                             method: str = 'pixel') -> Dict:
    """
    分析单个序列的动态性

    Args:
        seq_path: 序列目录路径
        sample_rate: 采样率（1=所有帧，4=每4帧采样1次）
        method: 差异计算方法

    Returns:
        dict: 动态性指标
    """
    frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])

    if len(frame_files) < 2:
        return None

    # 采样帧（减少计算量）
    sampled_indices = list(range(0, len(frame_files), sample_rate))
    if len(frame_files) - 1 not in sampled_indices:
        sampled_indices.append(len(frame_files) - 1)  # 确保包含最后一帧

    frames = []
    for idx in sampled_indices:
        path = os.path.join(seq_path, frame_files[idx])
        img = cv2.imread(path)
        if img is not None:
            # 统一 resize 到 640x480
            if img.shape[:2] != (480, 640):
                img = cv2.resize(img, (640, 480))
            frames.append(img)

    if len(frames) < 2:
        return None

    # 计算相邻帧差异
    adjacent_diffs = []
    for i in range(len(frames) - 1):
        diff = compute_frame_diff(frames[i], frames[i + 1], method)
        adjacent_diffs.append(diff)

    # 计算首尾帧差异
    first_last_diff = compute_frame_diff(frames[0], frames[-1], method)

    # 统计指标
    adjacent_diffs = np.array(adjacent_diffs)

    return {
        'num_frames': len(frame_files),
        'num_sampled': len(frames),
        'adjacent_diff_mean': adjacent_diffs.mean(),
        'adjacent_diff_std': adjacent_diffs.std(),
        'adjacent_diff_min': adjacent_diffs.min(),
        'adjacent_diff_max': adjacent_diffs.max(),
        'first_last_diff': first_last_diff,
        # 动态性评分（综合指标）
        'dynamic_score': adjacent_diffs.mean() * 0.7 + first_last_diff * 0.3,
    }


def main():
    ap = argparse.ArgumentParser(
        description="分析干净时序序列的帧间变化程度"
    )

    ap.add_argument("--clean_seqs_csv", type=str,
                    default="dataset/temporal_sequences/checked/clean_sequences.csv",
                    help="干净序列清单 CSV")
    ap.add_argument("--clean_seqs_root", type=str,
                    default="dataset/temporal_sequences/raw_sequences",
                    help="干净序列根目录")
    ap.add_argument("--output_dir", type=str,
                    default="dataset/temporal_sequences/dynamics_analysis",
                    help="输出目录")

    ap.add_argument("--sample_rate", type=int, default=4,
                    help="采样率（1=所有帧，4=每4帧采样1次）")
    ap.add_argument("--method", type=str, default="pixel",
                    choices=["pixel", "feature"],
                    help="差异计算方法")

    # 筛选阈值（用于标记低动态序列）
    ap.add_argument("--min_adjacent_diff", type=float, default=0.02,
                    help="最小相邻帧差异（低于此值标记为低动态）")
    ap.add_argument("--min_first_last_diff", type=float, default=0.05,
                    help="最小首尾帧差异（低于此值标记为低动态）")

    args = ap.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("干净时序序列动态性分析")
    print("="*60)
    print(f"序列清单: {args.clean_seqs_csv}")
    print(f"序列根目录: {args.clean_seqs_root}")
    print(f"采样率: {args.sample_rate}")
    print(f"差异计算方法: {args.method}")
    print(f"低动态阈值: 相邻帧 < {args.min_adjacent_diff}, 首尾 < {args.min_first_last_diff}")
    print()

    # 加载干净序列清单
    df = pd.read_csv(args.clean_seqs_csv)
    print(f"加载了 {len(df)} 个干净序列")
    print()

    # 分析所有序列
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="分析进度"):
        view = row['view']
        seq_id = row['seq_id']

        seq_path = os.path.join(args.clean_seqs_root, view, seq_id)
        if not os.path.exists(seq_path):
            continue

        metrics = analyze_sequence_dynamics(seq_path, args.sample_rate, args.method)
        if metrics is None:
            continue

        results.append({
            'view': view,
            'seq_id': seq_id,
            **metrics,
            # 标记低动态序列
            'is_low_dynamic': (
                metrics['adjacent_diff_mean'] < args.min_adjacent_diff or
                metrics['first_last_diff'] < args.min_first_last_diff
            ),
        })

    # 保存结果
    results_df = pd.DataFrame(results)

    # 保存完整结果
    results_df.to_csv(os.path.join(args.output_dir, 'dynamics_analysis.csv'), index=False)

    # 分别保存高动态和低动态序列
    high_dynamic_df = results_df[~results_df['is_low_dynamic']].sort_values('dynamic_score', ascending=False)
    low_dynamic_df = results_df[results_df['is_low_dynamic']].sort_values('dynamic_score')

    high_dynamic_df.to_csv(os.path.join(args.output_dir, 'high_dynamic_sequences.csv'), index=False)
    low_dynamic_df.to_csv(os.path.join(args.output_dir, 'low_dynamic_sequences.csv'), index=False)

    # 统计
    print()
    print("="*60)
    print("分析完成")
    print("="*60)
    print(f"总序列数: {len(results_df)}")
    print(f"高动态序列: {len(high_dynamic_df)} ({len(high_dynamic_df)/len(results_df)*100:.1f}%)")
    print(f"低动态序列: {len(low_dynamic_df)} ({len(low_dynamic_df)/len(results_df)*100:.1f}%)")
    print()

    if len(high_dynamic_df) > 0:
        print(f"高动态序列 dynamic_score 范围: [{high_dynamic_df['dynamic_score'].min():.4f}, {high_dynamic_df['dynamic_score'].max():.4f}]")
        print(f"  相邻帧差异: [{high_dynamic_df['adjacent_diff_mean'].min():.4f}, {high_dynamic_df['adjacent_diff_mean'].max():.4f}]")
        print(f"  首尾帧差异: [{high_dynamic_df['first_last_diff'].min():.4f}, {high_dynamic_df['first_last_diff'].max():.4f}]")

    if len(low_dynamic_df) > 0:
        print(f"低动态序列 dynamic_score 范围: [{low_dynamic_df['dynamic_score'].min():.4f}, {low_dynamic_df['dynamic_score'].max():.4f}]")
        print(f"  相邻帧差异: [{low_dynamic_df['adjacent_diff_mean'].min():.4f}, {low_dynamic_df['adjacent_diff_mean'].max():.4f}]")
        print(f"  首尾帧差异: [{low_dynamic_df['first_last_diff'].min():.4f}, {low_dynamic_df['first_last_diff'].max():.4f}]")

    print()
    print(f"各视角统计:")
    for view in ['f', 'lf', 'rf', 'lr', 'r', 'rr']:
        view_df = results_df[results_df['view'] == view]
        if len(view_df) > 0:
            high_count = len(view_df[~view_df['is_low_dynamic']])
            low_count = len(view_df[view_df['is_low_dynamic']])
            print(f"  {view}: {high_count}/{len(view_df)} 高动态 ({high_count/len(view_df)*100:.1f}%)")

    print()
    print(f"结果已保存到: {args.output_dir}/")
    print(f"  - dynamics_analysis.csv: 完整分析结果")
    print(f"  - high_dynamic_sequences.csv: 高动态序列清单")
    print(f"  - low_dynamic_sequences.csv: 低动态序列清单")
    print("="*60)


if __name__ == "__main__":
    main()
