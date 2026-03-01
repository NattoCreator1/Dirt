#!/usr/bin/env python3
"""
合并WoodScape和SD合成数据索引，用于混合训练实验

设计原则：
1. 对照组：WoodScape Only
2. 实验组：WoodScape + SD Synthetic
3. 控制变量：仅改变数据来源，其他参数完全一致

Author: SD Experiment Team
Date: 2026-02-24
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并WoodScape和SD合成数据索引",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--woodscape_index", type=str,
                       default="dataset/woodscape_processed/meta/labels_index_rebinned_baseline.csv",
                       help="WoodScape索引CSV")
    parser.add_argument("--synthetic_index", type=str,
                       default="sd_scripts/lora/accepted_20260224_164715/index_synthetic_20260224_182316.csv",
                       help="SD合成数据索引CSV")
    parser.add_argument("--output_dir", type=str, default="dataset/woodscape_processed/meta",
                       help="输出目录")
    parser.add_argument("--woodscape_split_col", type=str, default="split",
                       help="WoodScape split列名")
    parser.add_argument("--synthetic_split_col", type=str, default="split",
                       help="SD合成数据split列名")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（用于打乱SD数据）")

    return parser.parse_args()


def merge_indexes(args):
    """合并索引文件"""
    print("=" * 70)
    print("合并WoodScape和SD合成数据索引")
    print("=" * 70)

    # 读取索引
    print(f"\n读取WoodScape索引: {args.woodscape_index}")
    ws_df = pd.read_csv(args.woodscape_index)
    print(f"  WoodScape样本: {len(ws_df)}")

    print(f"\n读取SD合成数据索引: {args.synthetic_index}")
    syn_df = pd.read_csv(args.synthetic_index)
    print(f"  SD Synthetic样本: {len(syn_df)}")

    # 检查列名一致性
    print(f"\nWoodScape列: {list(ws_df.columns)}")
    print(f"SD Synthetic列: {list(syn_df.columns)}")

    # 提取split信息
    ws_train = ws_df[ws_df[args.woodscape_split_col] == 'train'].copy()
    ws_val = ws_df[ws_df[args.woodscape_split_col] == 'val'].copy()
    ws_test = ws_df[ws_df[args.woodscape_split_col] == 'test'].copy() if 'test' in ws_df[args.woodscape_split_col].values else pd.DataFrame()

    syn_train = syn_df[syn_df[args.synthetic_split_col] == 'train'].copy()
    syn_val = syn_df[syn_df[args.synthetic_split_col] == 'val'].copy()

    print(f"\nWoodScape: train={len(ws_train)}, val={len(ws_val)}, test={len(ws_test)}")
    print(f"SD Synthetic: train={len(syn_train)}, val={len(syn_val)}")

    # 合并训练集和验证集
    merged_train = pd.concat([ws_train, syn_train], ignore_index=True)
    merged_val = pd.concat([ws_val, syn_val], ignore_index=True)

    # 打乱合并后的数据
    rng = np.random.default_rng(args.seed)
    merged_train = merged_train.iloc[rng.permutation(len(merged_train))].reset_index(drop=True)
    merged_val = merged_val.iloc[rng.permutation(len(merged_val))].reset_index(drop=True)

    # 添加数据源标识
    merged_train['data_source'] = merged_train['rgb_path'].apply(
        lambda x: 'synthetic' if 'accepted_20260224_164715' in str(x) else 'woodscape'
    )
    merged_val['data_source'] = merged_val['rgb_path'].apply(
        lambda x: 'synthetic' if 'accepted_20260224_164715' in str(x) else 'woodscape'
    )

    print(f"\n合并后:")
    print(f"  训练集: {len(merged_train)} (WoodScape: {len(ws_train)}, SD: {len(syn_train)})")
    print(f"  验证集: {len(merged_val)} (WoodScape: {len(ws_val)}, SD: {len(syn_val)})")
    print(f"  SD比例: train={len(syn_train)/len(merged_train)*100:.1f}%, val={len(syn_val)/len(merged_val)*100:.1f}%")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存合并索引
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 完整合并索引（包含train和val）
    merged_all = pd.concat([merged_train, merged_val], ignore_index=True)
    merged_path = output_dir / f"labels_index_merged_{timestamp}.csv"
    merged_all.to_csv(merged_path, index=False)
    print(f"\n完整索引已保存: {merged_path}")

    # 分别保存train和val索引
    train_path = output_dir / f"labels_index_train_merged_{timestamp}.csv"
    val_path = output_dir / f"labels_index_val_merged_{timestamp}.csv"
    test_path = output_dir / f"labels_index_test_ws_{timestamp}.csv"  # WoodScape test集

    merged_train.to_csv(train_path, index=False)
    merged_val.to_csv(val_path, index=False)
    if not ws_test.empty:
        ws_test.to_csv(test_path, index=False)
        print(f"训练集索引: {train_path}")
        print(f"验证集索引: {val_path}")
        print(f"测试集索引: {test_path} (WoodScape only)")
    else:
        print(f"训练集索引: {train_path}")
        print(f"验证集索引: {val_path}")

    # 统计信息
    print("\n" + "=" * 70)
    print("数据分布统计")
    print("=" * 70)

    print(f"\nS值统计:")
    print(f"  WoodScape train: mean={ws_train['S'].mean():.4f}, std={ws_train['S'].std():.4f}")
    print(f"  SD train: mean={syn_train['S'].mean():.4f}, std={syn_train['S'].std():.4f}")
    print(f"  合并train: mean={merged_train['S'].mean():.4f}, std={merged_train['S'].std():.4f}")

    print(f"\n按数据源分布:")
    print(f"  训练集 Woodscape: {(merged_train['data_source']=='woodscape').sum()}")
    print(f"  训练集 Synthetic: {(merged_train['data_source']=='synthetic').sum()}")
    print(f"  验证集 Woodscape: {(merged_val['data_source']=='woodscape').sum()}")
    print(f"  验证集 Synthetic: {(merged_val['data_source']=='synthetic').sum()}")

    return merged_path


def main():
    args = parse_args()

    print("=" * 70)
    print("混合训练索引合并工具")
    print("=" * 70)
    print(f"WoodScape索引: {args.woodscape_index}")
    print(f"SD合成数据索引: {args.synthetic_index}")
    print(f"随机种子: {args.seed}")

    merged_path = merge_indexes(args)

    print(f"\n✅ 索引合并完成: {merged_path}")
    print(f"\n使用方法:")
    print(f"  对照组: --index_csv {args.woodscape_index}")
    print(f"  实验组: --index_csv {merged_path}")


if __name__ == "__main__":
    main()
