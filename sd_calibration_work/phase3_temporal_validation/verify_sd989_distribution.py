#!/usr/bin/env python3
"""
验证 SD 989 作为时序训练数据底座的可行性

对比指标:
1. S 分布的 KS 距离 (Kolmogorov-Smirnov statistic)
2. 在 baseline 模型上的预测误差分布
3. Tile 预测的校准程度

如果 SD989 在这些指标上接近 WoodScape，则适合作为时序训练数据
"""

import sys
import os
os.chdir('/home/yf/soiling_project')
sys.path.insert(0, '/home/yf/soiling_project')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats

from baseline.models.baseline_dualhead import BaselineDualHead
from baseline.datasets.woodscape_index import WoodscapeIndexSpec, WoodscapeTileDataset


def compute_ks_distance(real_values, synth_values):
    """计算两个分布的 KS 距离"""
    ks_stat, p_value = stats.ks_2samp(real_values, synth_values)
    return ks_stat, p_value


def compute_prediction_error(model, loader, device):
    """计算模型在数据集上的预测误差"""
    model.eval()
    errors = []
    s_preds = []
    s_gts = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            global_scores = batch["global_score"].to(device)

            outputs = model(images)
            s_hat = outputs['S_hat']

            errors.append((s_hat - global_scores).cpu().numpy())
            s_preds.append(s_hat.cpu().numpy())
            s_gts.append(global_scores.cpu().numpy())

    errors = np.concatenate(errors).reshape(-1)
    s_preds = np.concatenate(s_preds).reshape(-1)
    s_gts = np.concatenate(s_gts).reshape(-1)

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))  # 正值 = 高估

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "s_mean_pred": float(s_preds.mean()),
        "s_mean_gt": float(s_gts.mean()),
    }


@torch.no_grad()
def compute_tile_calibration(model, loader, device):
    """计算 tile 预测的校准程度"""
    model.eval()
    tile_errors = []

    for batch in loader:
        images = batch["image"].to(device)
        tile_cov = batch["tile_cov"].to(device)

        outputs = model(images)
        g_hat = outputs['G_hat']  # [B,4,8,8]

        g_tgt = tile_cov.permute(0, 3, 1, 2).contiguous()  # [B,4,8,8]
        tile_mae = torch.mean(torch.abs(g_hat - g_tgt), dim=[1,2,3])  # [B]
        tile_errors.append(tile_mae.cpu().numpy())

    tile_errors = np.concatenate(tile_errors)
    return float(tile_errors.mean()), float(tile_errors.std())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 加载 baseline 模型
    ckpt_path = "baseline/runs/ablation_label_def/ablation_S_full_wgap_alpha50/ckpt_best.pth"
    print(f"Loading baseline model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BaselineDualHead(pretrained=False).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # WoodScape 测试集
    print("\n" + "="*60)
    print("WoodScape Test_Real:")
    print("="*60)
    spec_ws = WoodscapeIndexSpec(
        index_csv="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv",
        img_root=".",
        split_col="split",
        split_value="test",
        global_target="S_full_wgap_alpha50"
    )
    dataset_ws = WoodscapeTileDataset(spec_ws)
    loader_ws = DataLoader(dataset_ws, batch_size=32, shuffle=False, num_workers=2)

    # 获取 S 分布
    s_values_ws = []
    for batch in loader_ws:
        s_values_ws.append(batch["global_score"].numpy())
    s_values_ws = np.concatenate(s_values_ws).reshape(-1)

    ws_metrics = compute_prediction_error(model, loader_ws, device)
    ws_tile_mae, ws_tile_std = compute_tile_calibration(model, loader_ws, device)

    print(f"  N = {len(s_values_ws)}")
    print(f"  S distribution: mean={s_values_ws.mean():.4f}, std={s_values_ws.std():.4f}")
    print(f"  Prediction: MAE={ws_metrics['mae']:.4f}, RMSE={ws_metrics['rmse']:.4f}, Bias={ws_metrics['bias']:.4f}")
    print(f"  Tile MAE: {ws_tile_mae:.4f} ± {ws_tile_std:.4f}")

    # SD 989 数据集
    print("\n" + "="*60)
    print("SD 989:")
    print("="*60)

    # 读取 SD 989 CSV (混合数据集，需要分离 SD 样本)
    sd989_csv = "dataset/woodscape_processed/meta/labels_index_mixed_989sd_20260224_202844.csv"
    df_all = pd.read_csv(sd989_csv)
    # SD 样本的 rgb_path 以 "synthetic_soiling/" 开头
    df_sd = df_all[df_all['rgb_path'].str.startswith('synthetic_soiling/', na=False)]
    df_ws = df_all[~df_all['rgb_path'].str.startswith('synthetic_soiling/', na=False)]

    print(f"  Total samples: {len(df_all)}")
    print(f"  SD samples: {len(df_sd)}")
    print(f"  WoodScape samples: {len(df_ws)}")

    print(f"  N = {len(df_sd)}")
    if 'S_full_wgap_alpha50' in df_sd.columns:
        s_values_sd = df_sd['S_full_wgap_alpha50'].values
        print(f"  S distribution: mean={s_values_sd.mean():.4f}, std={s_values_sd.std():.4f}")

        # KS 检验
        ks_stat, p_value = compute_ks_distance(s_values_ws, s_values_sd)
        print(f"  KS distance from WoodScape: {ks_stat:.4f} (p={p_value:.2e})")

        if ks_stat < 0.1:
            print(f"  ✓ SD989 分布与 WoodScape 接近 (KS < 0.1)")
        elif ks_stat < 0.2:
            print(f"  ~ SD989 分布与 WoodScape 中等相似 (0.1 < KS < 0.2)")
        else:
            print(f"  ✗ SD989 分布与 WoodScape 差异较大 (KS > 0.2)")

    # 混合数据集的 S 分布 (用于对比)
    print("\n" + "="*60)
    print("混合数据集对比 (SD8960):")
    print("="*60)
    sd8960_csv = "sd_calibration_work/phase1_importance_sampling/mixed_ws4000_sd8960.csv"
    df_8960 = pd.read_csv(sd8960_csv)
    df_8960_sd = df_8960[df_8960['source'] == 'SD']

    if 'S_full_wgap_alpha50' in df_8960_sd.columns:
        s_values_8960 = df_8960_sd['S_full_wgap_alpha50'].values
        print(f"  N = {len(df_8960_sd)}")
        print(f"  S distribution: mean={s_values_8960.mean():.4f}, std={s_values_8960.std():.4f}")

        ks_stat_8960, _ = compute_ks_distance(s_values_ws, s_values_8960)
        print(f"  KS distance from WoodScape: {ks_stat_8960:.4f}")

    print("\n" + "="*60)
    print("结论:")
    print("="*60)
    print("SD989 作为时序训练数据底座的可行性:")
    print(f"  - 分布相似性: KS={ks_stat:.4f}")
    print(f"  - 样本量: {len(df_sd)} 个 SD989 样本")
    print(f"  - 建议: {'适合作为时序训练数据' if ks_stat < 0.15 else '需要进一步筛选'}")


if __name__ == "__main__":
    main()
