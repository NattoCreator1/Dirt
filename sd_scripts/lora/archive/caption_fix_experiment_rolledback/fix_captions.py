#!/usr/bin/env python3
"""
Fix Training Data Captions

基于实际像素分布重新计算dominant_class并更新所有captions。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 数据路径
MANIFEST_PATH = Path("training_data/dreambooth_format/manifest_train.csv")
TRAIN_DIR = Path("training_data/dreambooth_format/train")

# Caption tokens
CLASS_TOKENS = {
    1: "transparent soiling layer",
    2: "semi-transparent dirt smudge",
    3: "opaque heavy stains",
}

SEVERITY_TOKENS = {
    (0.0, 0.15): "mild",
    (0.15, 0.35): "moderate",
    (0.35, 0.60): "noticeable",
    (0.60, 1.00): "severe",
}

OPTICAL = "on camera lens, out of focus foreground, subtle glare, background visible"


def get_actual_dominant_class(row):
    """
    基于实际像素分布重新计算dominant_class

    使用规则：像素占比最高的类别（排除clean类0）
    """
    ratios = {
        1: row.get('class_1_ratio', 0),
        2: row.get('class_2_ratio', 0),
        3: row.get('class_3_ratio', 0),
    }

    # 找最高占比的类别
    dominant_class = max(ratios, key=ratios.get)

    # 如果最高占比<5%，认为是混合
    if ratios[dominant_class] < 0.05:
        return 0

    return dominant_class


def generate_fixed_caption(row):
    """
    基于实际像素分布生成正确的caption
    """
    # 使用重新计算的dominant_class
    actual_dominant = get_actual_dominant_class(row)

    if actual_dominant == 0:
        # 没有明显脏污
        class_token = "light dirt"
    else:
        class_token = CLASS_TOKENS.get(actual_dominant, "dirt smudges")

    # Severity token
    S_full = row.get('S_full', 0.3)
    severity_token = "moderate"
    for (low, high), token in SEVERITY_TOKENS.items():
        if low <= S_full <= high:
            severity_token = token
            break

    return f"{severity_token} {class_token}, {OPTICAL}"


def analyze_dominant_class_mismatch(df):
    """分析标注不一致的情况"""
    print("\n=== Dominant Class不一致分析 ===")

    # 计算实际的dominant_class
    df['actual_dominant_class'] = df.apply(get_actual_dominant_class, axis=1)

    # 统计不一致的情况
    mismatch = df[df['dominant_class'] != df['actual_dominant_class']]
    print(f"标注不一致样本数: {len(mismatch)} / {len(df)} ({len(mismatch)/len(df)*100:.1f}%)")

    # 详细分析
    print("\n原始标注 vs 实际标注:")
    for orig in [1, 2, 3]:
        for actual in [1, 2, 3]:
            count = ((df['dominant_class'] == orig) & (df['actual_dominant_class'] == actual)).sum()
            if count > 0:
                print(f"  {orig} → {actual}: {count} 样本")

    return df


def update_captions(df, dry_run=True):
    """更新captions"""
    print(f"\n=== {'模拟' if dry_run else '实际'}更新Captions ===")

    df['fixed_caption'] = df.apply(generate_fixed_caption, axis=1)

    # 统计caption变化
    changed = df[df['caption'] != df['fixed_caption']]
    print(f"需要更新的caption: {len(changed)} / {len(df)} ({len(changed)/len(df)*100:.1f}%)")

    # 显示变化的例子
    print("\nCaption变化示例:")
    count = 0
    for idx, row in changed.iterrows():
        old_caption = row['caption']
        new_caption = row['fixed_caption']
        orig_dom = row['dominant_class']
        actual_dom = get_actual_dominant_class(row)

        print(f"\n[{row['file_id']}]")
        print(f"  dominant_class: {orig_dom} → {actual_dom}")
        print(f"  C1={row['class_1_ratio']:.3f}, C2={row['class_2_ratio']:.3f}, C3={row['class_3_ratio']:.3f}")
        print(f"  旧: {old_caption}")
        print(f"  新: {new_caption}")

        count += 1
        if count >= 10:
            break

    if not dry_run:
        # 实际更新caption文件
        print(f"\n实际更新caption文件...")
        updated_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row['caption'] != row['fixed_caption']:
                file_id = row['file_id']
                caption_path = TRAIN_DIR / f"{file_id}.txt"

                if caption_path.exists():
                    with open(caption_path, 'w') as f:
                        f.write(row['fixed_caption'])
                    updated_count += 1

        print(f"✅ 已更新 {updated_count} 个caption文件")

        # 更新manifest
        df['caption'] = df['fixed_caption']
        df['dominant_class'] = df['actual_dominant_class']
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"✅ 已更新manifest: {MANIFEST_PATH}")

    return df


def main():
    print("=" * 70)
    print("修复LoRA训练数据Captions")
    print("=" * 70)

    # 读取manifest
    if not MANIFEST_PATH.exists():
        print(f"错误: manifest文件不存在: {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    print(f"读取训练数据: {len(df)} 样本")

    # 分析不一致
    df = analyze_dominant_class_mismatch(df)

    # 模拟更新
    print("\n" + "=" * 70)
    print("第一步：模拟更新（预览变化）")
    print("=" * 70)
    df = update_captions(df, dry_run=True)

    # 询问是否实际更新
    print("\n" + "=" * 70)
    print("是否实际更新caption文件？")
    print("  输入 'yes' 确认更新")
    print("  其他输入取消")
    print("=" * 70)

    # 检查更新后的caption分布
    print("\n=== 更新后的Caption分布 ===")
    print(df['fixed_caption'].value_counts())

    print("\n=== 更新后的类别分布 ===")
    print(df['actual_dominant_class'].value_counts())

    print("\n✅ 模拟完成！请确认是否实际更新。")
    print("   实际更新请运行：python fix_captions.py --apply")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="修复训练数据captions")
    parser.add_argument("--apply", action="store_true", help="实际应用修改")
    args = parser.parse_args()

    main()

    if args.apply:
        df = pd.read_csv(MANIFEST_PATH)
        df = analyze_dominant_class_mismatch(df)
        df = update_captions(df, dry_run=False)
