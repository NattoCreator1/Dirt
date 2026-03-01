#!/usr/bin/env python3
"""
时序模块在 External Test 上的评估脚本

评估框架（遵循物理先验：脏污静止、背景变化）：
1. 簇内稳定性指标（主指标）：J_S(c), V_S(c), F_alarm(c)
2. 簇级排序指标（安全带）：cluster-level ρ

关键原则：
- 簇是时序片段：同一簇 = 同一脏污镜头视频
- 评估单位是簇（不是帧）
- 用簇层面 bootstrap 评估统计显著性

作者: soiling_project
日期: 2026-02-27
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


def extract_frame_index(image_path: str) -> int:
    """
    从图像路径提取帧索引

    示例路径:
    - ..._00225.jpg -> 225
    - 0008_..._00090.jpg -> 90

    Args:
        image_path: 图像路径

    Returns:
        帧索引（用于时序排序）
    """
    # 尝试多种模式
    # 模式1: _数字.jpg (如 _00225.jpg)
    match = re.search(r'_(\d+)\.jpg$', image_path)
    if match:
        return int(match.group(1))

    # 模式2: 文件名末尾数字
    basename = os.path.basename(image_path)
    match = re.search(r'(\d+)$', os.path.splitext(basename)[0])
    if match:
        return int(match.group(1))

    # 默认返回0（无法提取）
    return 0


@dataclass
class ClusterMetrics:
    """单个簇的评估指标"""
    cluster_id: str
    level: int
    num_frames: int

    # Baseline 指标
    baseline_S_values: np.ndarray  # [T] array of S_hat
    baseline_J_S: float            # 帧间抖动
    baseline_V_S: float            # 簇内方差
    baseline_S_median: float       # 中位数（簇级代表值）
    baseline_S_mean: float         # 均值

    # Temporal 指标（如果提供）
    temporal_S_values: Optional[np.ndarray] = None
    temporal_J_S: Optional[float] = None
    temporal_V_S: Optional[float] = None
    temporal_S_median: Optional[float] = None
    temporal_S_mean: Optional[float] = None

    # 报警抖动（如果提供阈值）
    baseline_alarm_flip: Optional[float] = None
    temporal_alarm_flip: Optional[float] = None


class TemporalExternalEvaluator:
    """时序模块外部测试评估器"""

    def __init__(self,
                 baseline_csv: str,
                 temporal_csv: Optional[str] = None,
                 alarm_threshold: Optional[float] = None,
                 output_dir: str = "scripts/temporal/eval_results"):
        """
        Args:
            baseline_csv: baseline 预测结果 CSV (image_path, cluster_id, level, S_hat)
            temporal_csv: temporal 模型预测结果 CSV (同格式)，可选
            alarm_threshold: 报警阈值，用于计算 F_alarm，可选
            output_dir: 输出目录
        """
        self.baseline_csv = baseline_csv
        self.temporal_csv = temporal_csv
        self.alarm_threshold = alarm_threshold
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.baseline_df = self._load_and_prepare(baseline_csv)
        self.temporal_df = self._load_and_prepare(temporal_csv) if temporal_csv else None

        # 聚合到簇级别
        self.cluster_metrics: Dict[str, ClusterMetrics] = {}

        print(f"Baseline 样本数: {len(self.baseline_df)}")
        print(f"Baseline 簇数: {self.baseline_df['cluster_id'].nunique()}")
        if self.temporal_df is not None:
            print(f"Temporal 样本数: {len(self.temporal_df)}")
            print(f"Temporal 簇数: {self.temporal_df['cluster_id'].nunique()}")

    def _load_and_prepare(self, csv_path: str) -> pd.DataFrame:
        """加载并准备数据"""
        df = pd.read_csv(csv_path)

        # 提取帧索引用于排序
        df['frame_idx'] = df['image_path'].apply(extract_frame_index)

        # 确保列名一致
        if 'level' not in df.columns and 'occlusion_level' in df.columns:
            df['level'] = df['occlusion_level']
        if 'S_hat' not in df.columns and 'S' in df.columns:
            df['S_hat'] = df['S']

        return df

    def compute_cluster_metrics(self) -> Dict[str, ClusterMetrics]:
        """计算每个簇的指标"""
        clusters = {}

        for cluster_id, group in self.baseline_df.groupby('cluster_id'):
            # 按帧索引排序（确保时序顺序）
            group = group.sort_values('frame_idx')
            level = int(group['level'].iloc[0])

            S_values = group['S_hat'].values.astype(np.float32)

            # 计算 baseline 指标
            J_S = self._compute_jitter(S_values)
            V_S = self._compute_variance(S_values)
            S_median = float(np.median(S_values))
            S_mean = float(np.mean(S_values))

            metrics = ClusterMetrics(
                cluster_id=cluster_id,
                level=level,
                num_frames=len(S_values),
                baseline_S_values=S_values,
                baseline_J_S=J_S,
                baseline_V_S=V_S,
                baseline_S_median=S_median,
                baseline_S_mean=S_mean,
            )

            # 计算 alarm flip（如果提供阈值）
            if self.alarm_threshold is not None:
                metrics.baseline_alarm_flip = self._compute_alarm_flip(S_values, self.alarm_threshold)

            # 计算 temporal 指标（如果有 temporal 数据）
            if self.temporal_df is not None:
                temp_group = self.temporal_df[self.temporal_df['cluster_id'] == cluster_id]
                if len(temp_group) > 0:
                    temp_group = temp_group.sort_values('frame_idx')
                    temp_S = temp_group['S_hat'].values.astype(np.float32)

                    # 对齐序列长度（取较短的）
                    min_len = min(len(S_values), len(temp_S))
                    if min_len > 0:
                        metrics.temporal_S_values = temp_S[:min_len]
                        metrics.temporal_J_S = self._compute_jitter(temp_S[:min_len])
                        metrics.temporal_V_S = self._compute_variance(temp_S[:min_len])
                        metrics.temporal_S_median = float(np.median(temp_S[:min_len]))
                        metrics.temporal_S_mean = float(np.mean(temp_S[:min_len]))

                        if self.alarm_threshold is not None:
                            metrics.temporal_alarm_flip = self._compute_alarm_flip(temp_S[:min_len], self.alarm_threshold)

            clusters[cluster_id] = metrics

        self.cluster_metrics = clusters
        return clusters

    def _compute_jitter(self, S: np.ndarray) -> float:
        """计算帧间抖动 J_S = mean(|S_{t+1} - S_t|)"""
        if len(S) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(S))))

    def _compute_variance(self, S: np.ndarray) -> float:
        """计算簇内方差 V_S = Var(S)"""
        return float(np.var(S))

    def _compute_alarm_flip(self, S: np.ndarray, threshold: float) -> float:
        """计算报警翻转率 F_alarm = proportion of alarm state changes"""
        if len(S) < 2:
            return 0.0
        alarm = (S >= threshold).astype(int)
        flips = np.sum(np.abs(np.diff(alarm)))
        return float(flips / (len(S) - 1))

    def compute_stability_comparison(self) -> Dict:
        """计算稳定性对比（簇级配对）"""
        if not self.cluster_metrics:
            self.compute_cluster_metrics()

        results = {
            'baseline_J_S': [],
            'temporal_J_S': [],
            'J_S_ratio': [],  # temporal / baseline
            'baseline_V_S': [],
            'temporal_V_S': [],
            'V_S_ratio': [],
        }

        if self.alarm_threshold is not None:
            results.update({
                'baseline_alarm_flip': [],
                'temporal_alarm_flip': [],
                'alarm_flip_ratio': [],
            })

        for cluster_id, m in self.cluster_metrics.items():
            if m.temporal_J_S is None:
                continue

            results['baseline_J_S'].append(m.baseline_J_S)
            results['temporal_J_S'].append(m.temporal_J_S)
            results['J_S_ratio'].append(m.temporal_J_S / (m.baseline_J_S + 1e-8))

            results['baseline_V_S'].append(m.baseline_V_S)
            results['temporal_V_S'].append(m.temporal_V_S)
            results['V_S_ratio'].append(m.temporal_V_S / (m.baseline_V_S + 1e-8))

            if self.alarm_threshold is not None:
                if m.temporal_alarm_flip is not None:
                    results['baseline_alarm_flip'].append(m.baseline_alarm_flip)
                    results['temporal_alarm_flip'].append(m.temporal_alarm_flip)
                    results['alarm_flip_ratio'].append(
                        m.temporal_alarm_flip / (m.baseline_alarm_flip + 1e-8)
                    )

        # 转换为 numpy 数组
        for k, v in results.items():
            results[k] = np.array(v)

        return results

    def compute_cluster_level_ranking(self) -> Dict:
        """计算簇级排序 ρ"""
        if not self.cluster_metrics:
            self.compute_cluster_metrics()

        # 提取簇级代表值（用中位数）
        clusters = []
        for cluster_id, m in self.cluster_metrics.items():
            clusters.append({
                'cluster_id': cluster_id,
                'level': m.level,
                'baseline_S': m.baseline_S_median,
                'temporal_S': m.temporal_S_median if m.temporal_S_median is not None else np.nan,
            })

        df = pd.DataFrame(clusters)

        # 移除 temporal_S 为 NaN 的簇（如果没有 temporal 数据）
        df_valid = df.dropna(subset=['temporal_S'])

        # 计算 Spearman ρ
        from scipy.stats import spearmanr

        rho_baseline, p_baseline = spearmanr(df_valid['level'], df_valid['baseline_S'])
        rho_temporal, p_temporal = spearmanr(df_valid['level'], df_valid['temporal_S'])

        delta_rho = rho_temporal - rho_baseline

        return {
            'num_clusters': len(df_valid),
            'rho_baseline': rho_baseline,
            'rho_temporal': rho_temporal,
            'delta_rho': delta_rho,
            'p_baseline': p_baseline,
            'p_temporal': p_temporal,
        }

    def cluster_level_bootstrap(self,
                                 n_bootstrap: int = 1000,
                                 seed: int = 42) -> Dict:
        """簇层面 bootstrap 评估（采样单元是簇）"""
        if not self.cluster_metrics:
            self.compute_cluster_metrics()

        np.random.seed(seed)

        cluster_ids = list(self.cluster_metrics.keys())
        n_clusters = len(cluster_ids)

        # 用于存储 bootstrap 结果
        J_S_ratios = []
        V_S_ratios = []
        delta_rhos = []

        for i in range(n_bootstrap):
            # 簇层面重采样
            sampled_ids = np.random.choice(cluster_ids, size=n_clusters, replace=True)
            sampled_metrics = [self.cluster_metrics[cid] for cid in sampled_ids]

            # 计算该样本的统计量
            j_ratios = []
            v_ratios = []
            baseline_S = []
            temporal_S = []
            levels = []

            for m in sampled_metrics:
                if m.temporal_J_S is not None:
                    j_ratios.append(m.temporal_J_S / (m.baseline_J_S + 1e-8))
                    v_ratios.append(m.temporal_V_S / (m.baseline_V_S + 1e-8))

                baseline_S.append(m.baseline_S_median)
                levels.append(m.level)
                if m.temporal_S_median is not None:
                    temporal_S.append(m.temporal_S_median)

            # 记录中位数比率
            if j_ratios:
                J_S_ratios.append(np.median(j_ratios))
                V_S_ratios.append(np.median(v_ratios))

            # 计算 delta_rho
            if temporal_S:
                from scipy.stats import spearmanr
                rho_b, _ = spearmanr(levels, baseline_S)
                rho_t, _ = spearmanr(levels, temporal_S)
                delta_rhos.append(rho_t - rho_b)

        # 计算 CI
        def compute_ci(values, alpha=0.05):
            return np.percentile(values, [100*alpha/2, 100*(1-alpha/2)])

        results = {
            'J_S_ratio_median': float(np.median(J_S_ratios)) if J_S_ratios else np.nan,
            'J_S_ratio_ci': compute_ci(J_S_ratios) if J_S_ratios else (np.nan, np.nan),
            'V_S_ratio_median': float(np.median(V_S_ratios)) if V_S_ratios else np.nan,
            'V_S_ratio_ci': compute_ci(V_S_ratios) if V_S_ratios else (np.nan, np.nan),
            'delta_rho_mean': float(np.mean(delta_rhos)) if delta_rhos else np.nan,
            'delta_rho_ci': compute_ci(delta_rhos) if delta_rhos else (np.nan, np.nan),
        }

        return results

    def save_results(self):
        """保存评估结果"""
        # 1. 保存簇级指标表
        clusters_data = []
        for cluster_id, m in self.cluster_metrics.items():
            row = {
                'cluster_id': cluster_id,
                'level': m.level,
                'num_frames': m.num_frames,
                'baseline_J_S': m.baseline_J_S,
                'baseline_V_S': m.baseline_V_S,
                'baseline_S_median': m.baseline_S_median,
                'baseline_S_mean': m.baseline_S_mean,
            }
            if m.temporal_J_S is not None:
                row.update({
                    'temporal_J_S': m.temporal_J_S,
                    'temporal_V_S': m.temporal_V_S,
                    'temporal_S_median': m.temporal_S_median,
                    'temporal_S_mean': m.temporal_S_mean,
                    'J_S_ratio': m.temporal_J_S / (m.baseline_J_S + 1e-8),
                    'V_S_ratio': m.temporal_V_S / (m.baseline_V_S + 1e-8),
                })
            if self.alarm_threshold is not None:
                row['baseline_alarm_flip'] = m.baseline_alarm_flip
                if m.temporal_alarm_flip is not None:
                    row['temporal_alarm_flip'] = m.temporal_alarm_flip
                    row['alarm_flip_ratio'] = m.temporal_alarm_flip / (m.baseline_alarm_flip + 1e-8)

            clusters_data.append(row)

        clusters_df = pd.DataFrame(clusters_data)
        clusters_df.to_csv(self.output_dir / 'cluster_metrics.csv', index=False)

        # 2. 计算并保存统计对比
        stability = self.compute_stability_comparison()
        ranking = self.compute_cluster_level_ranking()
        bootstrap = self.cluster_level_bootstrap()

        # 3. 保存 JSON 报告
        report = {
            'stability_comparison': {
                'num_clusters': len(stability['J_S_ratio']),
                'J_S_ratio_median': float(np.median(stability['J_S_ratio'])) if len(stability['J_S_ratio']) > 0 else None,
                'V_S_ratio_median': float(np.median(stability['V_S_ratio'])) if len(stability['V_S_ratio']) > 0 else None,
            },
            'ranking': ranking,
            'bootstrap': bootstrap,
        }

        if self.alarm_threshold is not None and 'alarm_flip_ratio' in stability:
            if len(stability['alarm_flip_ratio']) > 0:
                report['stability_comparison']['alarm_flip_ratio_median'] = float(np.median(stability['alarm_flip_ratio']))

        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # 4. 生成文本报告
        self._print_report(report, stability)

        print(f"\n结果已保存到: {self.output_dir}")

    def _print_report(self, report: Dict, stability: Dict):
        """打印评估报告"""
        print("\n" + "="*70)
        print("时序模块 External Test 评估报告")
        print("="*70)

        # 稳定性对比
        print("\n【主指标】簇内稳定性对比")
        print("-" * 70)
        if len(stability['J_S_ratio']) > 0:
            print(f"帧间抖动 J_S 比率中位数: {report['stability_comparison']['J_S_ratio_median']:.3f}")
            print(f"  (95% CI): {report['bootstrap']['J_S_ratio_ci'][0]:.3f} - {report['bootstrap']['J_S_ratio_ci'][1]:.3f}")
            print(f"簇内方差 V_S 比率中位数: {report['stability_comparison']['V_S_ratio_median']:.3f}")
            print(f"  (95% CI): {report['bootstrap']['V_S_ratio_ci'][0]:.3f} - {report['bootstrap']['V_S_ratio_ci'][1]:.3f}")

            interpretation = "显著降低" if report['stability_comparison']['J_S_ratio_median'] < 1 else "增加"
            print(f"\n解释: 时序模块{interpretation}了簇内抖动")

        if 'alarm_flip_ratio_median' in report['stability_comparison']:
            print(f"\n报警翻转率比率中位数: {report['stability_comparison']['alarm_flip_ratio_median']:.3f}")

        # 排序指标
        print("\n【安全带】簇级排序对比")
        print("-" * 70)
        print(f"簇数: {report['ranking']['num_clusters']}")
        print(f"Baseline ρ: {report['ranking']['rho_baseline']:.4f}")
        print(f"Temporal ρ: {report['ranking']['rho_temporal']:.4f}")
        print(f"Δρ (Temporal - Baseline): {report['ranking']['delta_rho']:+.4f}")
        print(f"  (95% CI): {report['bootstrap']['delta_rho_ci'][0]:.4f} - {report['bootstrap']['delta_rho_ci'][1]:.4f}")

        # 判断 Δρ 是否显著
        ci_low, ci_high = report['bootstrap']['delta_rho_ci']
        if ci_low <= 0 <= ci_high:
            print("\n解释: Δρ 的 95% CI 包含 0，时序模块没有显著降低簇级排序能力")
        elif ci_low > 0:
            print("\n解释: Δρ 的 95% CI > 0，时序模块显著提升了簇级排序能力")
        else:
            print("\n解释: Δρ 的 95% CI < 0，时序模块显著降低了簇级排序能力（需要注意）")

        print("="*70)


def main():
    ap = argparse.ArgumentParser(description="时序模块 External Test 评估")

    ap.add_argument("--baseline_csv", type=str, required=True,
                    help="Baseline 预测结果 CSV")
    ap.add_argument("--temporal_csv", type=str, default=None,
                    help="Temporal 模型预测结果 CSV（可选）")
    ap.add_argument("--alarm_threshold", type=float, default=None,
                    help="报警阈值，用于计算报警翻转率")
    ap.add_argument("--output_dir", type=str, default="scripts/temporal/eval_results",
                    help="输出目录")
    ap.add_argument("--n_bootstrap", type=int, default=1000,
                    help="Bootstrap 次数")

    args = ap.parse_args()

    evaluator = TemporalExternalEvaluator(
        baseline_csv=args.baseline_csv,
        temporal_csv=args.temporal_csv,
        alarm_threshold=args.alarm_threshold,
        output_dir=args.output_dir,
    )

    evaluator.compute_cluster_metrics()
    evaluator.save_results()


if __name__ == "__main__":
    main()
