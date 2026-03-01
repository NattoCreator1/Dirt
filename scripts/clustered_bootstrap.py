#!/usr/bin/env python3
"""
序列层面 Bootstrap 分析（修正版）

External Test 数据集存在序列内相关性问题：
- 同一序列的图片来自同一视频的连续帧
- 需要在序列层面进行 Bootstrap，而非图片层面

这给出更准确的统计推断。
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
import re
from collections import defaultdict
import json


def extract_sequence_id(filepath):
    """
    从文件路径提取序列ID
    """
    filename = os.path.basename(filepath)

    # 模式1: LJVA... 开头
    match1 = re.match(r'([A-Z0-9]+)', filename)
    if match1 and match1.group(1).startswith('LJ'):
        return match1.group(1)

    # 模式2: 纯数字开头
    match2 = re.match(r'(\d+)', filename)
    if match2:
        return match2.group(1)

    # 模式3: fp 开头
    if filename.startswith('fp'):
        return 'fp'

    return filename.split('_')[0]


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


def compute_clustered_bootstrap(predictions_csv, model_name="",
                                  n_bootstrap=1000,
                                  random_seed=42):
    """
    序列层面的 Bootstrap 分析

    重采样单位是序列（cluster），而非单张图片
    """
    np.random.seed(random_seed)

    # 读取数据
    df = pd.read_csv(predictions_csv)

    # 提取序列ID
    df['sequence_id'] = df['image_path'].apply(extract_sequence_id)

    # 获取每个序列的索引
    sequences = df.groupby('sequence_id').groups

    # 转换为列表形式：每个元素是一个序列的所有索引
    sequence_indices = [list(indices) for indices in sequences.values()]

    n_sequences = len(sequence_indices)
    n_images = len(df)

    print(f"\n模型: {model_name}")
    print(f"总图片数: {n_images}")
    print(f"独立序列数: {n_sequences}")
    print(f"平均每序列: {n_images / n_sequences:.1f} 张")

    # 点估计
    predictions = df['S_hat'].values
    labels = df['occlusion_level'].values
    point_estimate = spearmanr(predictions, labels)

    # 序列层面的 Bootstrap
    bootstrap_rhos = []

    for i in range(n_bootstrap):
        # 重采样序列（而非图片）
        sampled_indices = []
        for _ in range(n_sequences):
            seq_idx = np.random.choice(n_sequences)
            sampled_indices.extend(sequence_indices[seq_idx])

        sampled_indices = np.array(sampled_indices)

        # 计算该重采样的 rho
        rho = spearmanr(predictions[sampled_indices], labels[sampled_indices])
        bootstrap_rhos.append(rho)

    bootstrap_rhos = np.array(bootstrap_rhos)

    # 计算 CI
    alpha = 0.05
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrap_rhos, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_rhos, upper_percentile))
    se = float(np.std(bootstrap_rhos, ddof=1))
    ci_width = ci_upper - ci_lower

    result = {
        'model_name': model_name,
        'n_images': n_images,
        'n_sequences': n_sequences,
        'avg_images_per_sequence': n_images / n_sequences,
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'ci_width': ci_width,
        'relative_width': ci_width / point_estimate * 100,
        'n_bootstrap': n_bootstrap,
    }

    print(f"\n序列层面 Bootstrap 结果:")
    print(f"  点估计: {point_estimate:.4f}")
    print(f"  95% CI:  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  标准误差: {se:.4f}")
    print(f"  CI 宽度: {ci_width:.4f} ({ci_width/point_estimate*100:.2f}% 点估计)")

    return result


def clustered_bootstrap_comparison(csv1, csv2, name1="Model1", name2="Model2",
                                    n_bootstrap=1000, random_seed=42):
    """
    配对序列层面 Bootstrap

    关键：两模型使用相同的序列重采样
    """
    np.random.seed(random_seed)

    # 读取数据
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # 提取序列ID（两数据集应该有相同的图片顺序）
    df1['sequence_id'] = df1['image_path'].apply(extract_sequence_id)
    df2['sequence_id'] = df2['image_path'].apply(extract_sequence_id)

    # 获取序列分组
    sequences = df1.groupby('sequence_id').groups
    sequence_indices = [list(indices) for indices in sequences.values()]

    n_sequences = len(sequence_indices)
    n_images = len(df1)

    print(f"\n{'='*60}")
    print(f"配对序列层面 Bootstrap: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"总图片数: {n_images}")
    print(f"独立序列数: {n_sequences}")

    # 点估计
    preds1 = df1['S_hat'].values
    preds2 = df2['S_hat'].values
    labels = df1['occlusion_level'].values

    rho1_point = spearmanr(preds1, labels)
    rho2_point = spearmanr(preds2, labels)
    delta_point = rho2_point - rho1_point

    print(f"\n点估计:")
    print(f"  {name1} ρ: {rho1_point:.4f}")
    print(f"  {name2} ρ: {rho2_point:.4f}")
    print(f"  Δρ ({name2} - {name1}): {delta_point:+.4f}")

    # 配对 Bootstrap
    bootstrap_rhos1 = []
    bootstrap_rhos2 = []
    bootstrap_deltas = []

    for i in range(n_bootstrap):
        # 重采样序列
        sampled_indices = []
        for _ in range(n_sequences):
            seq_idx = np.random.choice(n_sequences)
            sampled_indices.extend(sequence_indices[seq_idx])

        sampled_indices = np.array(sampled_indices)

        rho1_b = spearmanr(preds1[sampled_indices], labels[sampled_indices])
        rho2_b = spearmanr(preds2[sampled_indices], labels[sampled_indices])
        delta_b = rho2_b - rho1_b

        bootstrap_rhos1.append(rho1_b)
        bootstrap_rhos2.append(rho2_b)
        bootstrap_deltas.append(delta_b)

    bootstrap_rhos1 = np.array(bootstrap_rhos1)
    bootstrap_rhos2 = np.array(bootstrap_rhos2)
    bootstrap_deltas = np.array(bootstrap_deltas)

    # 计算 CI
    lower_pct = 2.5
    upper_pct = 97.5

    rho1_ci_lower = np.percentile(bootstrap_rhos1, lower_pct)
    rho1_ci_upper = np.percentile(bootstrap_rhos1, upper_pct)
    rho2_ci_lower = np.percentile(bootstrap_rhos2, lower_pct)
    rho2_ci_upper = np.percentile(bootstrap_rhos2, upper_pct)
    delta_ci_lower = np.percentile(bootstrap_deltas, lower_pct)
    delta_ci_upper = np.percentile(bootstrap_deltas, upper_pct)
    delta_se = np.std(bootstrap_deltas, ddof=1)

    print(f"\n{'='*60}")
    print(f"配对序列层面 Bootstrap 结果")
    print(f"{'='*60}")

    print(f"\n{name1}:")
    print(f"  95% CI: [{rho1_ci_lower:.4f}, {rho1_ci_upper:.4f}]")

    print(f"\n{name2}:")
    print(f"  95% CI: [{rho2_ci_lower:.4f}, {rho2_ci_upper:.4f}]")

    print(f"\nΔρ ({name2} - {name1}):")
    print(f"  点估计: {delta_point:+.4f}")
    print(f"  95% CI: [{delta_ci_lower:+.4f}, {delta_ci_upper:+.4f}]")
    print(f"  标准误差: {delta_se:.4f}")

    # 判断显著性
    is_significant = (delta_ci_lower > 0) or (delta_ci_upper < 0)
    if is_significant:
        if delta_ci_lower > 0:
            conclusion = f"{name2} 显著优于 {name1}"
        else:
            conclusion = f"{name2} 显著差于 {name1}"
    else:
        conclusion = f"{name2} 与 {name1} 无显著差异"

    print(f"\n统计结论: {conclusion}")

    result = {
        'model1_name': name1,
        'model2_name': name2,
        'n_images': n_images,
        'n_sequences': n_sequences,
        'avg_images_per_sequence': n_images / n_sequences,
        'n_bootstrap': n_bootstrap,

        'model1_rho_point': rho1_point,
        'model2_rho_point': rho2_point,
        'delta_rho_point': delta_point,

        'model1_ci_lower': float(rho1_ci_lower),
        'model1_ci_upper': float(rho1_ci_upper),
        'model2_ci_lower': float(rho2_ci_lower),
        'model2_ci_upper': float(rho2_ci_upper),
        'delta_ci_lower': float(delta_ci_lower),
        'delta_ci_upper': float(delta_ci_upper),
        'delta_se': float(delta_se),

        'is_significant': bool(is_significant),
        'conclusion': conclusion,
    }

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="序列层面 Bootstrap 分析")
    parser.add_argument("--mode", type=str, choices=["single", "paired"], default="single",
                        help="分析模式: single 或 paired")
    parser.add_argument("--csv1", type=str,
                        default="baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/eval_Test_Ext_predictions.csv",
                        help="模型 1 的预测 CSV")
    parser.add_argument("--csv2", type=str,
                        help="模型 2 的预测 CSV (paired 模式需要)")
    parser.add_argument("--name1", type=str, default="Baseline",
                        help="模型 1 名称")
    parser.add_argument("--name2", type=str, default="Model2",
                        help="模型 2 名称")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap 次数")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_json", type=str,
                        default="scripts/external_test_analysis/clustered_bootstrap_results.json",
                        help="输出 JSON 文件")

    args = parser.parse_args()

    if args.mode == "single":
        result = compute_clustered_bootstrap(
            args.csv1,
            model_name=args.name1,
            n_bootstrap=args.n_bootstrap,
            random_seed=args.random_seed
        )

        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {args.output_json}")

    elif args.mode == "paired":
        if not args.csv2:
            raise ValueError("paired 模式需要 --csv2 参数")

        result = clustered_bootstrap_comparison(
            args.csv1, args.csv2,
            name1=args.name1, name2=args.name2,
            n_bootstrap=args.n_bootstrap,
            random_seed=args.random_seed
        )

        output_path = args.output_json.replace('.json', f'_{args.name1}_vs_{args.name2}.json')
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
