#!/usr/bin/env python3
"""
恢复Manifest中的原始dominant_class

从original_caption解析并恢复原始的dominant_class值
"""

import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path("training_data/dreambooth_format/manifest_train.csv")

def parse_original_class(caption):
    """从caption解析原始类别"""
    if pd.isna(caption):
        return None
    if 'transparent soiling layer' in caption:
        return 1
    elif 'semi-transparent dirt smudge' in caption:
        return 2
    elif 'opaque heavy stains' in caption:
        return 3
    return None

def restore_manifest():
    print("=" * 70)
    print("恢复Manifest原始dominant_class和caption")
    print("=" * 70)

    # 读取当前manifest
    df = pd.read_csv(MANIFEST_PATH)
    print(f"读取 {len(df)} 个样本")

    # 从备份的原始caption文件解析
    backup_dir = Path("training_data/dreambooth_format/train_backup_original")
    original_captions = {}
    for caption_file in backup_dir.glob("*.txt"):
        file_id = caption_file.stem
        with open(caption_file, 'r') as f:
            original_captions[file_id] = f.read().strip()

    # 从原始caption解析dominant_class并更新caption列
    df['dominant_class'] = df['file_id'].apply(lambda x: parse_original_class(original_captions.get(x)))
    df['caption'] = df['file_id'].map(original_captions)

    # 删除临时列
    if 'actual_dominant_class' in df.columns:
        df = df.drop(columns=['actual_dominant_class'])
    if 'fixed_caption' in df.columns:
        df = df.drop(columns=['fixed_caption'])

    # 保存
    df.to_csv(MANIFEST_PATH, index=False)

    # 验证
    print("\n恢复后的类别分布:")
    print(df['dominant_class'].value_counts().sort_index())

    # 验证caption
    print("\nCaption示例 (第一个样本):")
    first_row = df.iloc[0]
    print(f"  File ID: {first_row['file_id']}")
    print(f"  Dominant Class: {first_row['dominant_class']}")
    print(f"  Caption: {first_row['caption']}")

    print("\n✅ Manifest已恢复")
    print(f"   文件: {MANIFEST_PATH}")

    return df

if __name__ == "__main__":
    restore_manifest()
