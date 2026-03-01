#!/usr/bin/env python3
"""
配对 Bootstrap 分析：直接计算两模型之间 Δρ 的置信区间

比分别计算 CI 然后看是否重叠更敏感，更适合论文中的"对比结论"。

用法：
    python scripts/paired_bootstrap_comparison.py \
        --csv1 baseline/runs/.../eval_Test_Ext_predictions.csv \
        --csv2 baseline/runs/.../eval_Test_Ext_predictions.csv \
        --name1 "Baseline" \
        --name2 "SD989"
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
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


def paired_bootstrap_analysis(csv1, csv2, name1="Model1", name2="Model2",
                              n_bootstrap=1000, confidence_level=0.95,
                              random_seed=42, output_dir=None):
    """
    配对 Bootstrap 分析：计算 Δρ = ρ2 - ρ1 的分布

    关键：使用相同的重采样索引，确保两模型在"同一批测试样本"上比较
    """
    np.random.seed(random_seed)

    # 读取预测结果
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # 检测 level 列名
    def get_level_col(df):
        for col in ['ext_level', 'occlusion_level', 'level']:
            if col in df.columns:
                return col
        raise ValueError(f"Cannot find level column in {df.columns.tolist()}")

    level_col = get_level_col(df1)

    # 验证两 CSV 的 level 一致
    if not np.array_equal(df1[level_col].values, df2[level_col].values):
        raise ValueError("Two CSVs have different level labels!")

    labels = df1[level_col].values
    preds1 = df1['S_hat'].values
    preds2 = df2['S_hat'].values
    n = len(labels)

    print(f"\n{'='*60}")
    print(f"配对 Bootstrap 分析: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"样本数: {n}")
    print(f"Bootstrap 次数: {n_bootstrap}")
    print(f"置信水平: {confidence_level * 100}%")

    # 点估计
    rho1_point = spearmanr(preds1, labels)
    rho2_point = spearmanr(preds2, labels)
    delta_rho_point = rho2_point - rho1_point

    print(f"\n点估计:")
    print(f"  {name1} ρ: {rho1_point:.4f}")
    print(f"  {name2} ρ: {rho2_point:.4f}")
    print(f"  Δρ ({name2} - {name1}): {delta_rho_point:+.4f}")

    # 配对 Bootstrap
    print(f"\n运行配对 Bootstrap...")
    bootstrap_rhos1 = []
    bootstrap_rhos2 = []
    bootstrap_deltas = []

    for i in range(n_bootstrap):
        # 同一批索引重采样
        idx = np.random.choice(n, n, replace=True)

        rho1_b = spearmanr(preds1[idx], labels[idx])
        rho2_b = spearmanr(preds2[idx], labels[idx])
        delta_b = rho2_b - rho1_b

        bootstrap_rhos1.append(rho1_b)
        bootstrap_rhos2.append(rho2_b)
        bootstrap_deltas.append(delta_b)

    bootstrap_rhos1 = np.array(bootstrap_rhos1)
    bootstrap_rhos2 = np.array(bootstrap_rhos2)
    bootstrap_deltas = np.array(bootstrap_deltas)

    # 计算 CI
    alpha = 1 - confidence_level
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    # Model 1 CI
    rho1_ci_lower = np.percentile(bootstrap_rhos1, lower_pct)
    rho1_ci_upper = np.percentile(bootstrap_rhos1, upper_pct)

    # Model 2 CI
    rho2_ci_lower = np.percentile(bootstrap_rhos2, lower_pct)
    rho2_ci_upper = np.percentile(bootstrap_rhos2, upper_pct)

    # Δρ CI (配对)
    delta_ci_lower = np.percentile(bootstrap_deltas, lower_pct)
    delta_ci_upper = np.percentile(bootstrap_deltas, upper_pct)
    delta_se = np.std(bootstrap_deltas, ddof=1)

    # 输出结果
    print(f"\n{'='*60}")
    print(f"配对 Bootstrap 结果")
    print(f"{'='*60}")

    print(f"\n{name1}:")
    print(f"  95% CI: [{rho1_ci_lower:.4f}, {rho1_ci_upper:.4f}]")

    print(f"\n{name2}:")
    print(f"  95% CI: [{rho2_ci_lower:.4f}, {rho2_ci_upper:.4f}]")

    print(f"\nΔρ ({name2} - {name1}):")
    print(f"  点估计: {delta_rho_point:+.4f}")
    print(f"  95% CI: [{delta_ci_lower:.4f}, {delta_ci_upper:.4f}]")
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

    # 保存结果
    results = {
        'model1_name': name1,
        'model2_name': name2,
        'n_samples': int(n),
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,

        # 点估计
        'model1_rho_point': rho1_point,
        'model2_rho_point': rho2_point,
        'delta_rho_point': delta_rho_point,

        # CI (配对 bootstrap)
        'model1_ci_lower': float(rho1_ci_lower),
        'model1_ci_upper': float(rho1_ci_upper),
        'model2_ci_lower': float(rho2_ci_lower),
        'model2_ci_upper': float(rho2_ci_upper),
        'delta_ci_lower': float(delta_ci_lower),
        'delta_ci_upper': float(delta_ci_upper),
        'delta_se': float(delta_se),

        # 结论
        'is_significant': bool(is_significant),
        'conclusion': conclusion,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        json_path = output_path / f"paired_bootstrap_{name1}_vs_{name2}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {json_path}")

        # 保存 Markdown 报告
        _save_markdown_report(results, output_path / f"paired_bootstrap_report_{name1}_vs_{name2}.md")

        # 绘图
        _plot_paired_bootstrap(bootstrap_deltas, delta_rho_point, delta_ci_lower, delta_ci_upper,
                               name1, name2, output_path / f"paired_bootstrap_dist_{name1}_vs_{name2}.png")

    return results


def _save_markdown_report(results, output_path):
    """保存 Markdown 报告"""
    report = []
    report.append(f"# 配对 Bootstrap 分析报告\n")
    report.append(f"## {results['model1_name']} vs {results['model2_name']}\n\n")

    report.append(f"### 概况\n")
    report.append(f"- 样本数: {results['n_samples']}\n")
    report.append(f"- Bootstrap 次数: {results['n_bootstrap']}\n")
    report.append(f"- 置信水平: {results['confidence_level'] * 100}%\n\n")

    report.append(f"### 点估计\n")
    report.append(f"| 模型 | ρ |\n")
    report.append(f"|------|-------|\n")
    report.append(f"| {results['model1_name']} | {results['model1_rho_point']:.4f} |\n")
    report.append(f"| {results['model2_name']} | {results['model2_rho_point']:.4f} |\n")
    report.append(f"| Δρ ({results['model2_name']} - {results['model1_name']}) | {results['delta_rho_point']:+.4f} |\n\n")

    report.append(f"### 95% 置信区间 (配对 Bootstrap)\n")
    report.append(f"| 模型 | CI 下界 | CI 上界 |\n")
    report.append(f"|------|--------:|--------:|\n")
    report.append(f"| {results['model1_name']} | {results['model1_ci_lower']:.4f} | {results['model1_ci_upper']:.4f} |\n")
    report.append(f"| {results['model2_name']} | {results['model2_ci_lower']:.4f} | {results['model2_ci_upper']:.4f} |\n\n")

    report.append(f"### Δρ 的置信区间\n")
    report.append(f"| 指标 | 值 |\n")
    report.append(f"|------|-----|\n")
    report.append(f"| 点估计 | {results['delta_rho_point']:+.4f} |\n")
    report.append(f"| 95% CI | [{results['delta_ci_lower']:+.4f}, {results['delta_ci_upper']:+.4f}] |\n")
    report.append(f"| 标准误差 | {results['delta_se']:.4f} |\n\n")

    report.append(f"### 统计结论\n")
    report.append(f"**{results['conclusion']}**\n\n")

    if results['is_significant']:
        if results['delta_ci_lower'] > 0:
            report.append(f"由于 95% CI 完全位于 0 之上，{results['model2_name']} 相对于 {results['model1_name']} 的提升是统计显著的。\n\n")
        else:
            report.append(f"由于 95% CI 完全位于 0 之下，{results['model2_name']} 相对于 {results['model1_name']} 的下降是统计显著的。\n\n")
    else:
        report.append(f"由于 95% CI 包含 0，无法拒绝零假设（两模型性能相同）。\n\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    print(f"报告已保存到: {output_path}")


def _plot_paired_bootstrap(bootstrap_deltas, delta_point, ci_lower, ci_upper,
                           name1, name2, save_path):
    """绘制 Δρ 的 Bootstrap 分布"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    ax.hist(bootstrap_deltas, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # 点估计
    ax.axvline(delta_point, color='red', linestyle='--', linewidth=2,
               label=f'Δρ 点估计: {delta_point:+.4f}')

    # CI
    ax.axvline(ci_lower, color='green', linestyle=':', linewidth=1.5, label=f'95% CI')
    ax.axvline(ci_upper, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='零假设 (Δρ = 0)')

    # 填充 CI 区域
    ax.fill_between([ci_lower, ci_upper], 0, ax.get_ylim()[1],
                     alpha=0.2, color='green', label='95% 置信区间')

    ax.set_xlabel(f'Δρ ({name2} - {name1})', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Paired Bootstrap Distribution: {name1} vs {name2}', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="配对 Bootstrap 分析")
    parser.add_argument("--csv1", type=str, required=True,
                        help="模型 1 的预测 CSV")
    parser.add_argument("--csv2", type=str, required=True,
                        help="模型 2 的预测 CSV")
    parser.add_argument("--name1", type=str, default="Model1",
                        help="模型 1 名称")
    parser.add_argument("--name2", type=str, default="Model2",
                        help="模型 2 名称")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap 次数")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                        help="置信水平")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str,
                        default="scripts/external_test_analysis/paired_bootstrap",
                        help="输出目录")

    args = parser.parse_args()

    paired_bootstrap_analysis(
        args.csv1, args.csv2,
        name1=args.name1, name2=args.name2,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
