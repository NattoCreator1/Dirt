#!/usr/bin/env python3
"""
对比分析手动筛选 (680张) vs Baseline筛选 (989张)

分析两批筛选样本在 baseline 模型下的表现，验证手动筛选是否存在偏差
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import sys

sys.path.insert(0, '/home/yf/soiling_project')

from baseline.models.baseline_dualhead import BaselineDualHead


class ImageOnlySDataset:
    """简单的图像数据集，用于baseline预测"""
    def __init__(self, index_csv, resize_w=640, resize_h=480):
        self.df = pd.read_csv(index_csv)
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['rgb_path']

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()

        return {'image': x, 'img_path': img_path}


@torch.no_grad()
def predict_with_baseline(model, loader, device):
    """用baseline模型预测"""
    model.eval()
    all_shat = []
    all_paths = []

    for batch in tqdm(loader, desc="Predicting", ncols=100):
        x = batch["image"].to(device)
        paths = batch["img_path"]

        out = model(x)
        all_shat.append(out["S_hat"].cpu().numpy())
        all_paths.extend(paths)

    shat = np.concatenate(all_shat, axis=0).reshape(-1)

    return {
        "S_hat": shat,
        "image_path": all_paths,
    }


def main():
    print("=" * 70)
    print("手动筛选 vs Baseline筛选 对比分析")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 baseline 模型
    print(f"\n加载 baseline 模型...")
    ckpt = torch.load(
        'baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth',
        map_location=device
    )
    model = BaselineDualHead(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  Epoch: {ckpt.get('epoch')}")

    # 1. 分析手动筛选的 680 张
    print("\n" + "=" * 70)
    print("1. 手动筛选样本 (680 张)")
    print("=" * 70)

    manual_index = 'sd_scripts/lora/accepted_20260224_164715/index_synthetic_20260224_182316.csv'
    print(f"索引: {manual_index}")

    # 确保使用 S_full_wgap_alpha50
    manual_df = pd.read_csv(manual_index)
    if 'S' in manual_df.columns:
        # 手动筛选使用的是 S 列，需要检查是否对应 S_full_wgap_alpha50
        # 根据 generate_npz_labels.py，S 应该是 S_full_wgap_alpha50
        manual_s = manual_df['S'].values
        print(f"  样本数: {len(manual_df)}")
        print(f"  S 均值: {manual_s.mean():.4f}")
        print(f"  S 范围: [{manual_s.min():.4f}, {manual_s.max():.4f}]")
        manual_gt = manual_s  # 使用 S 列作为 ground truth
    else:
        print(f"  可用列: {manual_df.columns.tolist()}")
        return

    # 预测
    dataset = ImageOnlySDataset(manual_index)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    predictions = predict_with_baseline(model, loader, device)

    # 计算误差
    manual_shat = predictions['S_hat']
    manual_error = np.abs(manual_shat - manual_gt)
    manual_bias = manual_shat - manual_gt

    print(f"\nBaseline 预测结果:")
    print(f"  S_hat 均值: {manual_shat.mean():.4f}")
    print(f"  误差均值: {manual_error.mean():.4f}")
    print(f"  误差中位数: {np.median(manual_error):.4f}")
    print(f"  偏置 (S_hat - S_gt): {manual_bias.mean():.4f}")

    # 2. 从 8960 张中找出手动筛选的 680 张
    print("\n" + "=" * 70)
    print("2. 从8960张中定位手动筛选的680张")
    print("=" *70)

    # 读取完整的8960预测结果（之前保存的）
    all_8960_predictions_path = '/tmp/all_predictions_*.csv'
    import glob as glob
    all_files = glob.glob(all_8960_predictions_path)
    if all_files:
        all_8960_predictions = pd.read_csv(all_files[-1])  # 使用最新的
        print(f"8960张预测结果: {all_files[-1]}")

        # 找出手动筛选的样本在8960中的预测结果
        manual_filtered_in_8960 = all_8960_predictions[
            all_8960_predictions['image_path'].isin(manual_df['rgb_path'])
        ]
        print(f"  在8960预测结果中找到: {len(manual_filtered_in_8960)} 张")
    else:
        print("  未找到8960预测结果文件，重新预测...")

    # 3. Baseline 筛选的 989 张
    print("\n" + "=" * 70)
    print("3. Baseline 筛选样本 (989 张)")
    print("=" *70)

    baseline_filtered_index = 'sd_scripts/lora/baseline_filtered_8960/filtered_index_error_0.05.csv'
    baseline_df = pd.read_csv(baseline_filtered_index)
    print(f"索引: {baseline_filtered_index}")
    print(f"  样本数: {len(baseline_df)}")
    print(f"  S_gt 均值: {baseline_df['S_gt'].mean():.4f}")
    print(f"  S_gt 范围: [{baseline_df['S_gt'].min():.4f}, {baseline_df['S_gt'].max():.4f}]")

    baseline_error = baseline_df['error'].values
    baseline_bias = baseline_df['bias'].values
    print(f"\nBaseline 预测结果:")
    print(f"  误差均值: {baseline_error.mean():.4f}")
    print(f"  偏置均值: {baseline_bias.mean():.4f}")

    # 4. 对比分析
    print("\n" + "=" * 70)
    print("4. 对比分析")
    print("=" * 70)

    print("\n【S值分布对比】")
    print(f"  手动筛选 (680张):   S 均值 = {manual_s.mean():.4f}")
    print(f"  Baseline筛选 (989张):  S_gt 均值 = {baseline_df['S_gt'].mean():.4f}")
    print(f"  差异: {manual_s.mean() - baseline_df['S_gt'].mean():+.4f}")

    print("\n【Baseline 预测误差对比】")
    print(f"  手动筛选 (680张):")
    print(f"    误差均值 = {manual_error.mean():.4f}")
    print(f"    偏置均值 = {manual_bias.mean():+.4f}")
    print(f"  Baseline筛选 (989张):")
    print(f"    误差均值 = {baseline_error.mean():.4f}")
    print(f"    偏置均值 = {baseline_bias.mean():+.4f}")

    # 误差分布对比
    print(f"\n【误差分位数对比】")
    for p in [25, 50, 75, 90, 95]:
        manual_p = np.percentile(manual_error, p)
        baseline_p = np.percentile(baseline_error, p)
        print(f"  {p}% 分位数: 手动={manual_p:.4f} vs Baseline筛选={baseline_p:.4f}")

    # 高误差样本比例
    high_error_threshold = 0.10
    manual_high_ratio = (manual_error > high_error_threshold).mean() * 100
    baseline_high_ratio = (baseline_error > high_error_threshold).mean() * 100
    print(f"\n【高误差样本占比】(error > {high_error_threshold}):")
    print(f"  手动筛选: {manual_high_ratio:.1f}%")
    print(f"  Baseline筛选: {baseline_high_ratio:.1f}%")

    # 5. 误差 vs S_gt 关系
    print(f"\n【手动筛选的误差 vs S_gt】")
    # 按S_gt分桶
    manual_df_with_error = manual_df.copy()
    manual_df_with_error['S_hat'] = manual_shat
    manual_df_with_error['error'] = manual_error

    # 创建S_gt分桶
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    manual_df_with_error['S_bin'] = pd.cut(manual_s, bins=bins, labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])

    print("\n按S值区间的平均误差:")
    for interval, group in manual_df_with_error.groupby('S_bin'):
        print(f"  {interval:}: S均值={group['S'].mean():.3f}, 误差均值={group['error'].mean():.4f}, 样本数={len(group)}")

    # 保存对比结果
    print("\n" + "=" * 70)
    print("5. 保存对比结果")
    print("=" * 70)

    # 保存手动筛选的详细分析
    manual_analysis = pd.DataFrame({
        'S_gt': manual_gt,
        'S_hat': manual_shat,
        'error': manual_error,
        'bias': manual_bias,
        'image_path': predictions['image_path'],
    })

    output_dir = 'sd_scripts/lora/baseline_filtered_8960'
    manual_analysis_path = f'{output_dir}/manual_680_baseline_analysis.csv'
    manual_analysis.to_csv(manual_analysis_path, index=False)
    print(f"手动筛选分析结果已保存: {manual_analysis_path}")

    # 对比摘要
    comparison = {
        'manual_n': len(manual_df),
        'manual_s_mean': float(manual_s.mean()),
        'manual_error_mean': float(manual_error.mean()),
        'manual_bias_mean': float(manual_bias.mean()),
        'baseline_n': len(baseline_df),
        'baseline_s_mean': float(baseline_df['S_gt'].mean()),
        'baseline_error_mean': float(baseline_error.mean()),
        'baseline_bias_mean': float(baseline_bias.mean()),
    }

    import json
    comparison_path = f'{output_dir}/comparison_manual_vs_baseline_680_vs_989.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"对比摘要已保存: {comparison_path}")

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
