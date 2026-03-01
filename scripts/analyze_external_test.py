#!/usr/bin/env python3
"""
External Test Set 完整分析工具

把外部大测试集当作"精密尺"，对模型的外部域行为进行高分辨率分析：

1. Bootstrap 置信区间：回答 "ρ 的差异是否显著"
2. 分等级统计 (E[Ŝ | level])：回答 "标尺是否压缩/漂移"
3. 等级分离度 (Δ)：回答 "各等级是否清晰可分"
4. 斜率 (a) & 压缩比：回答 "模型对标尺变化的敏感度"

输出：
- 详细的统计表格
- 可视化图表（保存为 PNG）
- Markdown 报告（可直接用于论文）
"""

import sys
import os

os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import scipy.stats as stats


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


class ExternalTestAnalyzer:
    """
    外部测试集完整分析器

    功能：
    1. Bootstrap 置信区间
    2. 分等级统计分析
    3. 等级分离度计算
    4. 斜率与压缩比分析
    5. 可视化生成
    """

    def __init__(self, predictions_csv: str,
                 model_name: str = "",
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        """
        Args:
            predictions_csv: 模型预测结果 CSV (需包含 S_hat 和 ext_level 列)
            model_name: 模型名称（用于报告）
            n_bootstrap: Bootstrap 重采样次数
            confidence_level: 置信水平
            random_seed: 随机种子
        """
        self.model_name = model_name
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed

        # 读取数据
        self.df = pd.read_csv(predictions_csv)

        # 检测 level 列名
        if 'ext_level' in self.df.columns:
            level_col = 'ext_level'
        elif 'occlusion_level' in self.df.columns:
            level_col = 'occlusion_level'
        else:
            for col in self.df.columns:
                if 'level' in col.lower():
                    level_col = col
                    break
            else:
                raise ValueError(f"CSV 中没有找到 level 列。可用列: {self.df.columns.tolist()}")

        self.predictions = self.df['S_hat'].values
        self.labels = self.df[level_col].values
        self.n_samples = len(self.predictions)

        print(f"模型: {model_name}")
        print(f"样本数: {self.n_samples}")
        print(f"预测值范围: [{self.predictions.min():.4f}, {self.predictions.max():.4f}]")
        print(f"标签值: {np.unique(self.labels)}")

        # 存储分析结果
        self.results = {}

    def compute_bootstrap_ci(self) -> Dict:
        """计算 Spearman ρ 的 Bootstrap 置信区间"""
        np.random.seed(self.random_seed)

        # 点估计
        point_estimate = spearmanr(self.predictions, self.labels)

        # 手动 Bootstrap 重采样
        bootstrap_rhos = []
        n = self.n_samples

        for i in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            rho = spearmanr(self.predictions[idx], self.labels[idx])
            bootstrap_rhos.append(rho)

        bootstrap_rhos = np.array(bootstrap_rhos)

        # 计算 CI
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = float(np.percentile(bootstrap_rhos, lower_percentile))
        ci_upper = float(np.percentile(bootstrap_rhos, upper_percentile))
        se = float(np.std(bootstrap_rhos, ddof=1))

        # 置信区间宽度
        ci_width = ci_upper - ci_lower

        return {
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se,
            'ci_width': ci_width,
            'relative_width': ci_width / point_estimate * 100,  # 百分比
        }

    def compute_level_statistics(self) -> Dict:
        """
        计算各等级的统计量

        返回:
            {
                'level_<label>': {
                    'count': 样本数,
                    'mean': 均值,
                    'std': 标准差,
                    'se': 标准误差,
                    'ci_lower': 95% CI 下界,
                    'ci_upper': 95% CI 上界,
                }
            }
        """
        level_map = {5: 'a', 4: 'b', 3: 'c', 2: 'd', 1: 'e'}

        level_stats = {}
        unique_levels = np.unique(self.labels)

        for level in unique_levels:
            mask = self.labels == level
            level_preds = self.predictions[mask]
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

            level_stats[f'level_{label}'] = {
                'level_num': int(level),
                'level_label': label,
                'count': n,
                'mean': mean,
                'std': std,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }

        return level_stats

    def compute_separation_and_slope(self) -> Dict:
        """
        计算等级分离度和斜率

        返回:
            {
                'separations': {相邻等级差异},
                'slope': 线性回归斜率,
                'intercept': 截距,
                'r_squared': R²,
                'compression_ratio': 压缩比（相对于 baseline 的斜率比）
            }
        """
        level_map = {5: 'a', 4: 'b', 3: 'c', 2: 'd', 1: 'e'}

        # 按 level 数值降序排列
        ordered_labels = [5, 4, 3, 2, 1]
        means = []
        counts = []

        for level in ordered_labels:
            label = level_map.get(level, str(level))
            key = f'level_{label}'
            if key in self.results['level_stats']:
                level_stat = self.results['level_stats'][key]
                means.append(level_stat['mean'])
                counts.append(level_stat['count'])

        means = np.array(means)
        levels = np.array(ordered_labels, dtype=float)

        # 计算相邻等级分离度
        separations = {}
        for i in range(len(ordered_labels) - 1):
            level_1 = ordered_labels[i]
            level_2 = ordered_labels[i + 1]
            label_1 = level_map[level_1]
            label_2 = level_map[level_2]

            diff = means[i] - means[i + 1]
            separations[f'{label_1}_{label_2}'] = {
                'level_from': level_1,
                'level_to': level_2,
                'difference': diff,
            }

        # 线性回归: mean ~ a * level + b
        # 注意：level 作为自变量，mean 作为因变量
        slope, intercept, r_value, p_value, std_err = stats.linregress(levels, means)

        r_squared = r_value ** 2

        # 计算动态范围 (最高 - 最低)
        dynamic_range = means[0] - means[-1]

        return {
            'separations': separations,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'dynamic_range': dynamic_range,
            'mean_per_level': dict(zip([f'level_{level_map[l]}' for l in ordered_labels], means)),
        }

    def compute_compression_ratio(self, baseline_slope: float) -> Dict:
        """
        计算相对于 baseline 的压缩比

        compression_ratio = baseline_slope / current_slope
        > 1 表示当前模型比 baseline 更压缩
        """
        current_slope = self.results['slope']['slope']

        compression_ratio = baseline_slope / current_slope if current_slope > 0 else float('inf')

        return {
            'baseline_slope': baseline_slope,
            'current_slope': current_slope,
            'compression_ratio': compression_ratio,
            'interpretation': self._interpret_compression(compression_ratio),
        }

    def _interpret_compression(self, ratio: float) -> str:
        """解释压缩比"""
        if ratio > 1.2:
            return "显著压缩 (模型对等级变化敏感度降低)"
        elif ratio > 1.05:
            return "轻微压缩"
        elif ratio > 0.95:
            return "与 baseline 相当"
        elif ratio > 0.8:
            return "轻微扩张"
        else:
            return "显著扩张 (模型对等级变化敏感度提高)"

    def analyze_all(self, baseline_slope: Optional[float] = None) -> Dict:
        """
        执行所有分析
        """
        self.results['bootstrap'] = self.compute_bootstrap_ci()
        self.results['level_stats'] = self.compute_level_statistics()
        self.results['slope'] = self.compute_separation_and_slope()

        if baseline_slope is not None:
            self.results['compression'] = self.compute_compression_ratio(baseline_slope)

        # 生成摘要
        self._generate_summary()

        return self.results

    def _generate_summary(self):
        """生成分析摘要"""
        print("\n" + "="*60)
        print("分析摘要")
        print("="*60)

        # Bootstrap 结果
        boot = self.results['bootstrap']
        print(f"\n1. Spearman ρ (排序相关性)")
        print(f"   点估计: {boot['point_estimate']:.4f}")
        print(f"   95% CI:  [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
        print(f"   标准误差: {boot['se']:.4f}")
        print(f"   CI 宽度: {boot['ci_width']:.4f} ({boot['relative_width']:.2f}% 点估计)")

        # 分等级统计
        print(f"\n2. 分等级统计 (E[Ŝ | level])")
        print(f"\n   {'等级':<10} {'样本数':<10} {'均值':<12} {'标准误':<12} {'95% CI':<20}")
        print("-" * 64)

        for label in ['a', 'b', 'c', 'd', 'e']:
            key = f'level_{label}'
            if key in self.results['level_stats']:
                stats = self.results['level_stats'][key]
                ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                print(f"   {label:<10} {stats['count']:<10} {stats['mean']:<12.4f} {stats['se']:<12.4f} {ci_str:<20}")

        # 等级分离度
        print(f"\n3. 等级分离度 (相邻等级差异)")
        print(f"\n   {'对比':<15} {'差异':<12} {'说明':<30}")
        print("-" * 57)

        for key, val in self.results['slope']['separations'].items():
            diff = val['difference']
            if diff > 0.08:
                status = "✓ 分离良好"
            elif diff > 0.05:
                status = "~ 可区分"
            else:
                status = "✗ 分离不足"

            print(f"   {key:<15} {diff:<12.4f} {status:<30}")

        # 斜率分析
        slope = self.results['slope']['slope']
        r_squared = self.results['slope']['r_squared']
        print(f"\n4. 线性斜率 (Ŝ ~ level 的敏感度)")
        print(f"   斜率 (a): {slope:.4f}")
        print(f"   截距 (b): {self.results['slope']['intercept']:.4f}")
        print(f"   R²: {r_squared:.4f}")

        # 压缩比
        if 'compression' in self.results:
            comp = self.results['compression']
            print(f"\n5. 标尺压缩比 (相对于 baseline)")
            print(f"   Baseline 斜率: {comp['baseline_slope']:.4f}")
            print(f"   当前模型斜率: {comp['current_slope']:.4f}")
            print(f"   压缩比: {comp['compression_ratio']:.2f}")
            print(f"   解释: {comp['interpretation']}")

    def save_report(self, output_path: str):
        """保存 Markdown 格式的完整报告"""
        report = []
        report.append(f"# {self.model_name} - External Test 完整分析报告\n")
        report.append(f"## 1. 概况\n")
        report.append(f"- 样本数: {self.n_samples}\n")
        report.append(f"- Bootstrap 次数: {self.n_bootstrap}\n")
        report.append(f"- 置信水平: {self.confidence_level * 100}%\n")

        # Bootstrap
        boot = self.results['bootstrap']
        report.append(f"## 2. Spearman ρ (排序相关性)\n")
        report.append(f"| 指标 | 值 |\n")
        report.append(f"|------|-----|\n")
        report.append(f"| 点估计 | {boot['point_estimate']:.4f} |\n")
        report.append(f"| 95% CI | [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}] |\n")
        report.append(f"| 标准误差 | {boot['se']:.4f} |\n")
        report.append(f"| CI 宽度 | {boot['ci_width']:.4f} ({boot['relative_width']:.1f}% 点估计) |\n")

        # Level stats
        report.append(f"\n## 3. 分等级统计 (E[Ŝ | level])\n")
        report.append(f"\n### 3.1 统计表\n")
        report.append(f"| 等级 | 样本数 | 均值 | 标准差 | 标准误差 | 95% CI |\n")
        report.append(f"|------|------:|-------:|-------:|--------:|----------|\n")

        for label in ['a', 'b', 'c', 'd', 'e']:
            key = f'level_{label}'
            if key in self.results['level_stats']:
                stats = self.results['level_stats'][key]
                ci_str = f"{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}"
                report.append(f"| {label} | {stats['count']:<6} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['se']:.4f} | {ci_str} |\n")

        # Separations
        report.append(f"\n### 3.2 相邻等级分离度\n")
        report.append(f"| 对比 | 差异 | 状态 |\n")
        report.append(f"|------|------:|------|\n")

        for key, val in self.results['slope']['separations'].items():
            diff = val['difference']
            if diff > 0.08:
                status = "✓ 分离良好"
            elif diff > 0.05:
                status = "~ 可区分"
            else:
                status = "✗ 分离不足"
            report.append(f"| {key} | {diff:.4f} | {status} |\n")

        # Slope
        slope = self.results['slope']['slope']
        report.append(f"\n## 4. 线性斜率分析 (Ŝ ~ level)\n")
        report.append(f"| 指标 | 值 |\n")
        report.append(f"|------|-----|\n")
        report.append(f"| 斜率 (a) | {slope:.4f} |\n")
        report.append(f"| 截距 (b) | {self.results['slope']['intercept']:.4f} |\n")
        report.append(f"| R² | {self.results['slope']['r_squared']:.4f} |\n")
        report.append(f"| P-value | {self.results['slope']['p_value']:.2e} |\n")

        # Compression
        if 'compression' in self.results:
            comp = self.results['compression']
            report.append(f"\n## 5. 标尺压缩比分析\n")
            report.append(f"| 指标 | 值 |\n")
            report.append(f"|------|-----|\n")
            report.append(f"| Baseline 斜率 | {comp['baseline_slope']:.4f} |\n")
            report.append(f"| 当前模型斜率 | {comp['current_slope']:.4f} |\n")
            report.append(f"| 压缩比 | {comp['compression_ratio']:.2f}x |\n")
            report.append(f"| 解释 | {comp['interpretation']} |\n")

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report)

        print(f"\n报告已保存到: {output_path}")

    def visualize(self, output_dir: str):
        """生成可视化图表"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Level-wise 均值折线图 + 误差条
        self._plot_level_means_with_error_bars(output_path / "level_means_errorbar.png")

        # 2. Bootstrap 分布直方图
        self._plot_bootstrap_distribution(output_path / "bootstrap_distribution.png")

        # 3. 等级分离度可视化
        self._plot_separations(output_path / "separations.png")

        print(f"\n可视化图表已保存到: {output_dir}")

    def _plot_level_means_with_error_bars(self, save_path):
        """绘制等级均值折线图（带误差条）"""
        ordered_labels = ['a', 'b', 'c', 'd', 'e']
        means = []
        ses = []
        counts = []

        for label in ordered_labels:
            key = f'level_{label}'
            if key in self.results['level_stats']:
                stats = self.results['level_stats'][key]
                means.append(stats['mean'])
                ses.append(stats['se'] * 1.96)  # 95% CI
                counts.append(stats['count'])

        x_pos = np.arange(len(ordered_labels))

        fig, ax = plt.subplots(figsize=(10, 6))

        # 折线图
        ax.plot(x_pos, means, 'o-', linewidth=2, markersize=8, label='E[Ŝ | level]')
        ax.fill_between(x_pos,
                          [m - se for m, se in zip(means, ses)],
                          [m + se for m, se in zip(means, ses)],
                          alpha=0.3)

        # 样本数量标注
        for i, (x, count) in enumerate(zip(x_pos, counts)):
            ax.text(x, means[i] + ses[i] + 0.01, f'n={count}',
                   ha='center', fontsize=9, color='gray')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(ordered_labels)
        ax.set_xlabel('External Level (a→e: heavy→light)', fontsize=12)
        ax.set_ylabel('E[Ŝ | level]', fontsize=12)
        ax.set_title(f'{self.model_name} - Level-wise Predictions with 95% CI', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_bootstrap_distribution(self, save_path):
        """绘制 Bootstrap 分布直方图"""
        # 重新计算 bootstrap 分布用于可视化
        np.random.seed(self.random_seed)
        bootstrap_rhos = []
        n = self.n_samples

        for i in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            rho = spearmanr(self.predictions[idx], self.labels[idx])
            bootstrap_rhos.append(rho)

        bootstrap_rhos = np.array(bootstrap_rhos)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 直方图
        ax.hist(bootstrap_rhos, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        # 点估计和 CI
        point_est = self.results['bootstrap']['point_estimate']
        ci_lower = self.results['bootstrap']['ci_lower']
        ci_upper = self.results['bootstrap']['ci_upper']

        ax.axvline(point_est, color='red', linestyle='--', linewidth=2, label=f'Point Est: {point_est:.4f}')
        ax.axvline(ci_lower, color='green', linestyle=':', linewidth=1.5, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(ci_upper, color='green', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Spearman ρ (Bootstrap)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{self.model_name} - Bootstrap Distribution (N={self.n_bootstrap})', fontsize=14)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_separations(self, save_path):
        """绘制等级分离度可视化"""
        separations = self.results['slope']['separations']
        pairs = list(separations.keys())
        diffs = [separations[p]['difference'] for p in pairs]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if d > 0.08 else 'orange' if d > 0.05 else 'red' for d in diffs]

        bars = ax.bar(range(len(pairs)), diffs, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs)
        ax.set_ylabel('Mean Difference', fontsize=12)
        ax.set_title(f'{self.model_name} - Adjacent Level Separation', fontsize=14)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='Minimum threshold')

        # 添加数值标注
        for i, (bar, diff) in enumerate(zip(bars, diffs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{diff:.3f}', ha='center', va='bottom')

        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def compare_models(model_predictions: Dict[str, str],
                     baseline_name: str = "ablation_S_full_wgap_alpha50",
                     output_dir: str = "scripts/external_test_analysis"):
    """
    比较多个模型的外部测试集性能

    Args:
        model_predictions: {模型名: predictions_csv路径}
        baseline_name: baseline 模型名（用于计算压缩比）
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 存储所有模型的比较结果
    comparison = {}

    # 首先分析 baseline（获取斜率基准）
    print(f"\n分析 Baseline: {baseline_name}")
    baseline_analyzer = ExternalTestAnalyzer(
        model_predictions[baseline_name],
        model_name=baseline_name
    )
    baseline_analyzer.analyze_all()
    baseline_slope = baseline_analyzer.results['slope']['slope']

    comparison[baseline_name] = baseline_analyzer.results

    # 分析其他模型
    for model_name, csv_path in model_predictions.items():
        if model_name == baseline_name:
            continue

        print(f"\n分析模型: {model_name}")
        analyzer = ExternalTestAnalyzer(
            csv_path,
            model_name=model_name
        )
        analyzer.analyze_all(baseline_slope=baseline_slope)
        comparison[model_name] = analyzer.results

        # 保存单个模型的分析结果
        model_output_dir = output_path / model_name
        model_output_dir.mkdir(exist_ok=True)

        analyzer.save_report(str(model_output_dir / "analysis_report.md"))
        analyzer.visualize(str(model_output_dir / "visualizations"))

    # 生成对比表
    _generate_comparison_table(comparison, baseline_name, output_path)

    return comparison


def _generate_comparison_table(comparison: Dict, baseline_name: str, output_dir: Path):
    """生成模型对比表"""
    comparison_path = output_dir / "comparison_table.md"

    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(f"# External Test 模型对比分析\n\n")

        # 1. Spearman ρ 对比表
        f.write("## 1. Spearman ρ 对比\n\n")
        f.write("| 模型 | ρ 点估计 | 95% CI | CI 宽度 | 与 Baseline 差异 |\n")
        f.write("|------|---------|--------|--------|----------------|\n")

        baseline_rho = comparison[baseline_name]['bootstrap']['point_estimate']
        baseline_ci_lower = comparison[baseline_name]['bootstrap']['ci_lower']
        baseline_ci_upper = comparison[baseline_name]['bootstrap']['ci_upper']

        for model_name, results in comparison.items():
            boot = results['bootstrap']
            rho = boot['point_estimate']
            ci_lower = boot['ci_lower']
            ci_upper = boot['ci_upper']
            ci_width = boot['ci_width']

            # 检查与 baseline CI 是否重叠
            # CI 不重叠的条件: 当前 CI 上界 < baseline 下界 OR 当前 CI 下界 > baseline 上界
            overlap = not (ci_upper < baseline_ci_lower or ci_lower > baseline_ci_upper)
            diff = rho - baseline_rho

            status = "显著差异" if not overlap else "不显著"
            diff_str = f"{diff:+.4f}"

            f.write(f"| {model_name} | {rho:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] | {ci_width:.4f} | {diff_str} ({status}) |\n")

        # 2. 压缩比对比表
        f.write("\n## 2. 标尺压缩比对比\n\n")
        f.write("| 模型 | 斜率 | 压缩比 | 解释 |\n")
        f.write("|------|-------|--------|------|\n")

        for model_name, results in comparison.items():
            if 'compression' in results:
                comp = results['compression']
                f.write(f"| {model_name} | {comp['current_slope']:.4f} | {comp['compression_ratio']:.2f}x | {comp['interpretation']} |\n")

        # 3. 等级均值对比表
        f.write("\n## 3. 等级均值对比表\n\n")
        f.write("| 等级 |")
        for model_name in comparison.keys():
            f.write(f" {model_name} |")
        f.write("\n|------|")
        for model_name in comparison.keys():
            f.write("--------|")
        f.write("\n")

        for label in ['a', 'b', 'c', 'd', 'e']:
            f.write(f"| {label} |")
            for model_name in comparison.keys():
                key = f'level_{label}'
                if key in comparison[model_name]['level_stats']:
                    stats = comparison[model_name]['level_stats'][key]
                    f.write(f" {stats['mean']:.4f} |")
            f.write(" |\n")

        # 4. 等级分离度对比表
        f.write("\n## 4. 等级分离度对比表\n\n")
        f.write("| 模型 | a-b | b-c | c-d | d-e | 平均分离度 |\n")
        f.write("|------|-----|-----|-----|-----|----------|\n")

        for model_name, results in comparison.items():
            separations = results['slope']['separations']
            diffs = [separations[k]['difference'] for k in ['a_b', 'b_c', 'c_d', 'd_e']]
            avg_sep = np.mean(diffs)

            f.write(f"| {model_name} |")
            for diff in diffs:
                f.write(f" {diff:.3f} |")
            f.write(f" {avg_sep:.3f} |\n")

    print(f"\n对比表已保存到: {comparison_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="External Test 完整分析工具")
    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="模型预测结果 CSV")
    parser.add_argument("--model_name", type=str, default="",
                        help="模型名称")
    parser.add_argument("--output_dir", type=str,
                        default="scripts/external_test_analysis",
                        help="输出目录")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap 重采样次数")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                        help="置信水平")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--baseline_slope", type=float, default=None,
                        help="Baseline 斜率（用于计算压缩比）")

    args = parser.parse_args()

    # 创建分析器并分析
    analyzer = ExternalTestAnalyzer(
        predictions_csv=args.predictions_csv,
        model_name=args.model_name or Path(args.predictions_csv).parent.name,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed
    )

    analyzer.analyze_all(baseline_slope=args.baseline_slope)

    # 保存结果
    analyzer.save_report(f"{args.output_dir}/analysis_report.md")
    analyzer.visualize(f"{args.output_dir}/visualizations")

    print(f"\n分析完成！结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
