#!/usr/bin/env python3
"""
簇层面 Bootstrap 分析（修正版）

External Test 有 505 个簇（子文件夹），每个簇平均 100 张图片。
Bootstrap 应该在簇层面进行，而非图片层面。
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd


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


def clustered_bootstrap_analysis(csv1, csv2=None, name1="Model1", name2=None,
                                  n_bootstrap=1000, random_seed=42):
    """
    簇层面 Bootstrap 分析
    
    Args:
        csv1: 模型1的预测CSV（需包含cluster_id列）
        csv2: 模型2的预测CSV（可选，用于配对分析）
        n_bootstrap: Bootstrap次数
        random_seed: 随机种子
    """
    np.random.seed(random_seed)

    # 读取数据
    df1 = pd.read_csv(csv1)

    # 按簇分组获取索引
    clusters1 = df1.groupby('cluster_id').groups
    cluster_indices1 = [list(indices) for indices in clusters1.values()]

    n_clusters = len(cluster_indices1)
    n_images = len(df1)

    print(f"\n{'='*60}")
    print(f"簇层面 Bootstrap 分析: {name1}")
    print(f"{'='*60}")
    print(f"总图片数: {n_images}")
    print(f"总簇数: {n_clusters}")
    print(f"平均每簇: {n_images / n_clusters:.1f} 张")

    # 点估计
    preds1 = df1['S_hat'].values
    labels1 = df1['occlusion_level'].values
    rho1_point = spearmanr(preds1, labels1)

    # 簇层面 Bootstrap
    bootstrap_rhos1 = []
    for i in range(n_bootstrap):
        # 重采样簇
        sampled_indices = []
        for _ in range(n_clusters):
            cluster_idx = np.random.choice(n_clusters)
            sampled_indices.extend(cluster_indices1[cluster_idx])
        
        sampled_indices = np.array(sampled_indices)
        rho_b = spearmanr(preds1[sampled_indices], labels1[sampled_indices])
        bootstrap_rhos1.append(rho_b)

    bootstrap_rhos1 = np.array(bootstrap_rhos1)

    # 计算 CI
    ci_lower = np.percentile(bootstrap_rhos1, 2.5)
    ci_upper = np.percentile(bootstrap_rhos1, 97.5)
    se = float(np.std(bootstrap_rhos1, ddof=1))
    ci_width = ci_upper - ci_lower

    print(f"\n簇层面 Bootstrap 结果:")
    print(f"  点估计: {rho1_point:.4f}")
    print(f"  95% CI:  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  标准误差: {se:.4f}")
    print(f"  CI 宽度: {ci_width:.4f} ({ci_width/rho1_point*100:.2f}% 点估计)")

    result = {
        'model_name': name1,
        'n_images': n_images,
        'n_clusters': n_clusters,
        'avg_images_per_cluster': n_images / n_clusters,
        'point_estimate': rho1_point,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'ci_width': ci_width,
        'relative_width': ci_width / rho1_point * 100,
    }

    # 如果提供第二个模型，进行配对分析
    if csv2 and name2:
        df2 = pd.read_csv(csv2)
        clusters2 = df2.groupby('cluster_id').groups
        cluster_indices2 = [list(indices) for indices in clusters2.values()]

        preds2 = df2['S_hat'].values
        labels2 = df2['occlusion_level'].values
        rho2_point = spearmanr(preds2, labels2)

        print(f"\n{'='*60}")
        print(f"配对分析: {name1} vs {name2}")
        print(f"{'='*60}")
        print(f"{name1} ρ: {rho1_point:.4f}")
        print(f"{name2} ρ: {rho2_point:.4f}")
        print(f"Δρ ({name2} - {name1}): {rho2_point - rho1_point:+.4f}")

        # 配对 Bootstrap
        bootstrap_rhos2 = []
        bootstrap_deltas = []

        for i in range(n_bootstrap):
            # 使用相同的簇重采样
            sampled_indices = []
            for _ in range(n_clusters):
                cluster_idx = np.random.choice(n_clusters)
                sampled_indices.extend(cluster_indices1[cluster_idx])

            sampled_indices = np.array(sampled_indices)

            rho1_b = spearmanr(preds1[sampled_indices], labels1[sampled_indices])
            rho2_b = spearmanr(preds2[sampled_indices], labels2[sampled_indices])
            delta_b = rho2_b - rho1_b

            bootstrap_rhos2.append(rho2_b)
            bootstrap_deltas.append(delta_b)

        bootstrap_rhos2 = np.array(bootstrap_rhos2)
        bootstrap_deltas = np.array(bootstrap_deltas)

        # 计算 CI
        rho2_ci_lower = np.percentile(bootstrap_rhos2, 2.5)
        rho2_ci_upper = np.percentile(bootstrap_rhos2, 97.5)
        delta_ci_lower = np.percentile(bootstrap_deltas, 2.5)
        delta_ci_upper = np.percentile(bootstrap_deltas, 97.5)
        delta_se = np.std(bootstrap_deltas, ddof=1)

        print(f"\n{name1}:")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        print(f"\n{name2}:")
        print(f"  95% CI: [{rho2_ci_lower:.4f}, {rho2_ci_upper:.4f}]")

        print(f"\nΔρ ({name2} - {name1}):")
        print(f"  点估计: {rho2_point - rho1_point:+.4f}")
        print(f"  95% CI: [{delta_ci_lower:+.4f}, {delta_ci_upper:+.4f}]")
        print(f"  标准误差: {delta_se:.4f}")

        # 判断显著性
        is_significant = (delta_ci_lower > 0) or (delta_ci_upper < 0)
        if is_significant:
            conclusion = f"{name2} 显著{'优于' if delta_ci_lower > 0 else '差于'} {name1}"
        else:
            conclusion = f"{name2} 与 {name1} 无显著差异"

        print(f"\n统计结论: {conclusion}")

        result['paired_analysis'] = {
            'model2_name': name2,
            'model2_rho': rho2_point,
            'delta_rho': rho2_point - rho1_point,
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
    import json

    parser = argparse.ArgumentParser(description="簇层面 Bootstrap 分析")
    parser.add_argument("--csv1", type=str, required=True,
                        help="模型1的预测CSV（需包含cluster_id列）")
    parser.add_argument("--csv2", type=str, default=None,
                        help="模型2的预测CSV（可选，用于配对分析）")
    parser.add_argument("--name1", type=str, default="Model1",
                        help="模型1名称")
    parser.add_argument("--name2", type=str, default="Model2",
                        help="模型2名称")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap次数")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_json", type=str,
                        default="scripts/external_test_analysis/clustered_bootstrap.json",
                        help="输出JSON文件")

    args = parser.parse_args()

    result = clustered_bootstrap_analysis(
        args.csv1, args.csv2,
        name1=args.name1, name2=args.name2 if args.csv2 else None,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed
    )

    # 保存结果
    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
