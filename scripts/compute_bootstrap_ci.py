#!/usr/bin/env python3
"""
为 External Test Set 计算 Bootstrap 置信区间

用途：
1. 为 ρ 计算 95% 置信区间
2. 为各等级的 E[Ŝ] 计算标准误差
3. 进行单调性检验

输入：模型在 Test_Ext 上的预测结果 CSV
输出：带统计误差的评估结果 + 可视化
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from typing import Dict, List, Tuple
import json


def spearmanr(a, b):
    """Compute Spearman correlation."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    ra = a.argsort().argsort().astype(np.float32)
    rb = b.argsort().argsort().astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


def compute_bootstrap_ci(predictions: np.ndarray, labels: np.ndarray,
                          n_bootstrap: int = 1000,
                          confidence_level: float = 0.95,
                          random_seed: int = 42) -> Dict:
    """
    计算 Spearman ρ 的 Bootstrap 置信区间

    Args:
        predictions: 模型预测值 [N]
        labels: 真实标签 [N]
        n_bootstrap: Bootstrap 重采样次数
        confidence_level: 置信水平 (默认 0.95)
        random_seed: 随机种子

    Returns:
        dict: {
            'point_estimate': 点估计值,
            'ci_lower': 置信区间下界,
            'ci_upper': 置信区间上界,
            'se': 标准误差,
        }
    """
    np.random.seed(random_seed)

    # 点估计
    point_estimate = spearmanr(predictions, labels)

    # 手动 Bootstrap 重采样
    bootstrap_rhos = []
    n = len(predictions)

    for i in range(n_bootstrap):
        # 有放回重采样
        idx = np.random.choice(n, n, replace=True)
        rho = spearmanr(predictions[idx], labels[idx])
        bootstrap_rhos.append(rho)

    bootstrap_rhos = np.array(bootstrap_rhos)

    # 计算 CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrap_rhos, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_rhos, upper_percentile))

    # 标准误差
    se = float(np.std(bootstrap_rhos, ddof=1))

    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
    }


def compute_level_statistics(predictions: np.ndarray, labels: np.ndarray,
                             level_map: Dict = None) -> Dict:
    """
    计算各等级的统计量（均值、标准误差、样本量）

    Args:
        predictions: 模型预测值 [N]
        labels: 真实标签 [N] (1-5)
        level_map: 等级标签映射 {5: 'a', 4: 'b', ...}

    Returns:
        dict: {level: {'count', 'mean', 'se', 'ci_lower', 'ci_upper'}}
    """
    if level_map is None:
        level_map = {5: 'a', 4: 'b', 3: 'c', 2: 'd', 1: 'e'}

    level_stats = {}
    unique_levels = np.unique(labels)

    for level in unique_levels:
        mask = labels == level
        level_preds = predictions[mask]
        n = len(level_preds)

        if n == 0:
            continue

        mean = float(np.mean(level_preds))
        std = float(np.std(level_preds, ddof=1))
        se = std / np.sqrt(n)

        # 95% CI (假设正态分布)
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se

        label = level_map.get(int(level), str(level))

        level_stats[label] = {
            'count': n,
            'mean': mean,
            'std': std,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    return level_stats


def test_monotonicity(level_stats: Dict) -> Dict:
    """
    检验等级均值是否单调递减

    Args:
        level_stats: compute_level_statistics 的输出

    Returns:
        dict: {
            'is_monotonic': bool,
            'differences': 相邻等级差异列表,
        }
    """
    # 按 level 数值降序排列 (a > b > c > d > e)
    ordered_labels = ['a', 'b', 'c', 'd', 'e']

    means = [level_stats[lbl]['mean'] for lbl in ordered_labels if lbl in level_stats]

    differences = []
    is_monotonic = True

    for i in range(len(means) - 1):
        diff = means[i] - means[i+1]
        differences.append(diff)
        if diff <= 0:
            is_monotonic = False

    return {
        'is_monotonic': is_monotonic,
        'differences': differences,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="为 Test_Ext 计算 Bootstrap CI")
    parser.add_argument("--predictions_csv", type=str,
                        default="baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Ext_predictions.csv",
                        help="模型在 Test_Ext 上的预测结果 CSV")
    parser.add_argument("--output_json", type=str,
                        default="baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Ext_bootstrap.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap 重采样次数")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                        help="置信水平")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 读取预测结果
    print(f"读取预测结果: {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)

    # 假设 CSV 有 S_hat 和 ext_level 列
    if 'S_hat' not in df.columns:
        print("错误: CSV 中没有 'S_hat' 列")
        print(f"可用列: {df.columns.tolist()}")
        return

    if 'ext_level' not in df.columns:
        # 尝试其他可能的列名
        for col in df.columns:
            if 'level' in col.lower():
                df['ext_level'] = df[col]
                break
        else:
            print("错误: CSV 中没有 'ext_level' 列")
            print(f"可用列: {df.columns.tolist()}")
            return

    predictions = df['S_hat'].values
    labels = df['ext_level'].values

    print(f"\n样本数: {len(predictions)}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"标签值: {np.unique(labels)}")

    # 计算 Bootstrap CI
    print("\n" + "="*60)
    print("Bootstrap 置信区间计算")
    print("="*60)

    bootstrap_result = compute_bootstrap_ci(
        predictions, labels,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed
    )

    print(f"\nSpearman ρ:")
    print(f"  点估计: {bootstrap_result['point_estimate']:.4f}")
    print(f"  95% CI:  [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print(f"  标准误差: {bootstrap_result['se']:.4f}")
    print(f"  Bootstrap 次数: {bootstrap_result['n_bootstrap']}")

    # 计算各等级统计
    print("\n" + "="*60)
    print("分等级统计分析")
    print("="*60)

    level_stats = compute_level_statistics(predictions, labels)

    print(f"\n{'等级':<10} {'样本数':<10} {'均值':<12} {'标准误':<12} {'95% CI':<20}")
    print("-" * 64)

    for label in ['a', 'b', 'c', 'd', 'e']:
        if label not in level_stats:
            continue
        stats = level_stats[label]
        ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
        print(f"{label:<10} {stats['count']:<10} {stats['mean']:<12.4f} {stats['se']:<12.4f} {ci_str:<20}")

    # 单调性检验
    print("\n" + "="*60)
    print("单调性检验")
    print("="*60)

    monotonicity_result = test_monotonicity(level_stats)

    print(f"\n是否单调递减: {monotonicity_result['is_monotonic']}")
    print(f"\n相邻等级差异:")
    for i, (label1, label2) in enumerate([('a','b'), ('b','c'), ('c','d'), ('d','e')]):
        if label1 in level_stats and label2 in level_stats:
            diff = level_stats[label1]['mean'] - level_stats[label2]['mean']
            print(f"  {label1} - {label2}: {diff:.4f}")

    # 保存结果
    output = {
        'bootstrap': bootstrap_result,
        'level_stats': level_stats,
        'monotonicity': monotonicity_result,
        'n_samples': int(len(predictions)),
    }

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
