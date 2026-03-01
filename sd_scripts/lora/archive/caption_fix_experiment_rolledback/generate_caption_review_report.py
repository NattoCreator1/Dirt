#!/usr/bin/env python3
"""
生成Caption修复审阅报告

创建详细的审阅报告，展示原始caption与修复后caption的对比，
支持按类别、严重度、变化类型筛选。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 路径配置
MANIFEST_PATH = Path("training_data/dreambooth_format/manifest_train.csv")
TRAIN_DIR = Path("training_data/dreambooth_format")
BACKUP_DIR = TRAIN_DIR / "train_backup_original"
REPORT_DIR = Path("docs/caption_review_report")

# 创建报告目录
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 类别名称映射
CLASS_NAMES = {
    0: "clean/mixed",
    1: "transparent",
    2: "semi-transparent",
    3: "opaque",
}

# Severity范围
SEVERITY_NAMES = {
    (0.0, 0.15): "mild",
    (0.15, 0.35): "moderate",
    (0.35, 0.60): "noticeable",
    (0.60, 1.00): "severe",
}

def get_severity_token(S_full):
    """获取严重度token"""
    for (low, high), token in SEVERITY_NAMES.items():
        if low <= S_full <= high:
            return token
    return "moderate"


def load_data():
    """加载manifest和原始captions"""
    df = pd.read_csv(MANIFEST_PATH)

    # 读取原始captions从备份
    original_captions = {}
    for caption_file in BACKUP_DIR.glob("*.txt"):
        file_id = caption_file.stem
        with open(caption_file, 'r') as f:
            original_captions[file_id] = f.read().strip()

    df['original_caption'] = df['file_id'].map(original_captions)

    # 添加severity token列
    df['severity_token'] = df['S_full'].apply(get_severity_token)

    # 恢复原始dominant_class (从original_caption解析)
    # 原始caption格式: "{severity} {class_token}, on camera lens..."
    def parse_original_class(caption):
        if pd.isna(caption):
            return None
        if 'transparent soiling layer' in caption:
            return 1
        elif 'semi-transparent dirt smudge' in caption:
            return 2
        elif 'opaque heavy stains' in caption:
            return 3
        return None

    df['original_dominant_class'] = df['original_caption'].apply(parse_original_class)

    return df


def create_change_summary(df):
    """创建变化摘要"""
    # 识别变化的样本
    changed = df[df['caption'] != df['original_caption']]

    # 按变化类型分类
    class_changes = changed[changed['original_dominant_class'] != changed['actual_dominant_class']]

    # 统计
    summary = {
        "total_samples": len(df),
        "changed_samples": len(changed),
        "unchanged_samples": len(df) - len(changed),
        "change_percentage": len(changed) / len(df) * 100,
        "class_change_samples": len(class_changes),
    }

    # 类别变化统计 - 使用恢复的original_dominant_class
    class_change_matrix = {}
    for orig in [1, 2, 3]:
        for actual in [1, 2, 3]:
            count = ((df['original_dominant_class'] == orig) & (df['actual_dominant_class'] == actual)).sum()
            if count > 0:
                class_change_matrix[f"{orig}→{actual}"] = int(count)

    summary['class_change_matrix'] = class_change_matrix

    # Caption分布对比
    original_dist = df['original_caption'].value_counts().to_dict()
    fixed_dist = df['caption'].value_counts().to_dict()
    summary['original_caption_dist'] = original_dist
    summary['fixed_caption_dist'] = fixed_dist

    # 类别分布对比 - 使用恢复的original_dominant_class
    original_class_dist = df['original_dominant_class'].value_counts().to_dict()
    fixed_class_dist = df['actual_dominant_class'].value_counts().to_dict()
    summary['original_class_dist'] = {str(k): v for k, v in sorted(original_class_dist.items())}
    summary['fixed_class_dist'] = {str(k): v for k, v in sorted(fixed_class_dist.items())}

    return summary, changed


def create_filtered_lists(df):
    """创建各种筛选列表"""
    lists = {}

    # 1. 按变化类型
    df['changed'] = df['caption'] != df['original_caption']
    lists['changed'] = df[df['changed']].sort_values('file_id')
    lists['unchanged'] = df[~df['changed']].sort_values('file_id')

    # 2. 按原始类别分组
    for orig_class in [1, 2, 3]:
        subset = df[df['original_dominant_class'] == orig_class].sort_values('file_id')
        lists[f'original_class_{orig_class}'] = subset

    # 3. 按修复后类别分组
    for fixed_class in [1, 2, 3]:
        subset = df[df['actual_dominant_class'] == fixed_class].sort_values('file_id')
        lists[f'fixed_class_{fixed_class}'] = subset

    # 4. 按严重度分组
    for sev in ['mild', 'moderate', 'noticeable', 'severe']:
        subset = df[df['severity_token'] == sev].sort_values('file_id')
        lists[f'severity_{sev}'] = subset

    # 5. 类别变化详情 - 使用original_dominant_class
    for orig in [1, 2, 3]:
        for actual in [1, 2, 3]:
            subset = df[(df['original_dominant_class'] == orig) & (df['actual_dominant_class'] == actual)]
            if len(subset) > 0:
                lists[f'class_{orig}_to_{actual}'] = subset.sort_values('file_id')

    # 6. 关键变化样本（类别变化大）
    df['class_change_magnitude'] = (df['actual_dominant_class'] - df['original_dominant_class']).abs()
    lists['significant_changes'] = df[df['class_change_magnitude'] >= 2].sort_values('file_id')

    return lists


def generate_markdown_report(summary, changed, lists):
    """生成Markdown审阅报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = []
    lines.append("# Caption修复审阅报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---\n")

    # 1. 执行摘要
    lines.append("## 1. 执行摘要")
    lines.append(f"- **总样本数**: {summary['total_samples']}")
    lines.append(f"- **变化样本数**: {summary['changed_samples']} ({summary['change_percentage']:.1f}%)")
    lines.append(f"- **未变化样本数**: {summary['unchanged_samples']}")
    lines.append(f"- **类别变化样本数**: {summary['class_change_samples']}")
    lines.append("\n**原始Caption已备份至**: `train_backup_original/`")
    lines.append("\n---\n")

    # 2. 类别变化矩阵
    lines.append("## 2. 类别变化矩阵 (原始 → 修复后)")
    lines.append("| 原始 | 修复后 | 样本数 |")
    lines.append("|------|--------|--------|")
    for change, count in sorted(summary['class_change_matrix'].items()):
        orig, actual = change.split('→')
        lines.append(f"| C{orig} ({CLASS_NAMES[int(orig)]}) | C{actual} ({CLASS_NAMES[int(actual)]}) | {count} |")
    lines.append("\n---\n")

    # 3. 类别分布对比
    lines.append("## 3. 类别分布对比")
    lines.append("\n### 原始分布")
    lines.append("| 类别 | 样本数 | 比例 |")
    lines.append("|------|--------|------|")
    total = sum(summary['original_class_dist'].values())
    for cls, count in sorted(summary['original_class_dist'].items(), key=lambda x: int(x[0])):
        cls_int = int(cls)
        pct = count / total * 100
        lines.append(f"| C{cls_int} ({CLASS_NAMES[cls_int]}) | {count} | {pct:.1f}% |")

    lines.append("\n### 修复后分布")
    lines.append("| 类别 | 样本数 | 比例 |")
    lines.append("|------|--------|------|")
    total = sum(summary['fixed_class_dist'].values())
    for cls, count in sorted(summary['fixed_class_dist'].items(), key=lambda x: int(x[0])):
        cls_int = int(cls)
        pct = count / total * 100
        lines.append(f"| C{cls_int} ({CLASS_NAMES[cls_int]}) | {count} | {pct:.1f}% |")
    lines.append("\n---\n")

    # 4. Caption分布对比
    lines.append("## 4. Caption分布对比")
    lines.append("\n### 原始Caption (Top 10)")
    lines.append("| Caption | 样本数 |")
    lines.append("|---------|--------|")
    for caption, count in list(summary['original_caption_dist'].items())[:10]:
        lines.append(f"| {caption[:60]}... | {count} |")

    lines.append("\n### 修复后Caption (Top 10)")
    lines.append("| Caption | 样本数 |")
    lines.append("|---------|--------|")
    for caption, count in list(summary['fixed_caption_dist'].items())[:10]:
        lines.append(f"| {caption[:60]}... | {count} |")
    lines.append("\n---\n")

    # 5. 变化样本详情
    lines.append("## 5. 变化样本详情 (按原始类别分组)")
    lines.append("\n**提示**: 类别比例格式为 `C1=x%, C2=y%, C3=z%`")

    for orig_class in [1, 2, 3]:
        class_changed = lists[f'original_class_{orig_class}']
        class_changed_actual = class_changed[class_changed['caption'] != class_changed['original_caption']]

        if len(class_changed_actual) == 0:
            continue

        lines.append(f"\n### 原始类别 C{orig_class} ({CLASS_NAMES[orig_class]}) 的变化样本")
        lines.append(f"共 {len(class_changed_actual)} 个样本")

        # 按修复后类别分组
        for actual_class in [1, 2, 3]:
            subset = class_changed_actual[class_changed_actual['actual_dominant_class'] == actual_class]
            if len(subset) == 0:
                continue

            lines.append(f"\n#### C{orig_class} → C{actual_class} ({len(subset)} 样本)")
            lines.append("| 文件ID | C1% | C2% | C3% | 严重度 | 原始Caption | 修复Caption |")
            lines.append("|--------|-----|-----|-----|--------|-------------|-------------|")

            for _, row in subset.head(20).iterrows():
                c1 = row['class_1_ratio'] * 100
                c2 = row['class_2_ratio'] * 100
                c3 = row['class_3_ratio'] * 100
                sev = row['severity_token']
                orig_cap = row['original_caption'][:40] + "..."
                new_cap = row['caption'][:40] + "..."
                lines.append(f"| {row['file_id']} | {c1:.0f}% | {c2:.0f}% | {c3:.0f}% | {sev} | {orig_cap} | {new_cap} |")

            if len(subset) > 20:
                lines.append(f"| ... | | | | | 还有 {len(subset) - 20} 个样本 | |")

    lines.append("\n---\n")

    # 6. 关键变化样本
    if len(lists['significant_changes']) > 0:
        lines.append("## 6. 关键变化样本 (类别变化 ≥2)")
        lines.append(f"\n共 {len(lists['significant_changes'])} 个样本")
        lines.append("| 文件ID | C1% | C2% | C3% | 原始类别 | 修复类别 | 原始Caption | 修复Caption |")
        lines.append("|--------|-----|-----|-----|----------|----------|-------------|-------------|")

        for _, row in lists['significant_changes'].head(50).iterrows():
            c1 = row['class_1_ratio'] * 100
            c2 = row['class_2_ratio'] * 100
            c3 = row['class_3_ratio'] * 100
            orig_cls = row['original_dominant_class']
            new_cls = row['actual_dominant_class']
            orig_cap = row['original_caption'][:30] + "..."
            new_cap = row['caption'][:30] + "..."
            lines.append(f"| {row['file_id']} | {c1:.0f}% | {c2:.0f}% | {c3:.0f}% | C{orig_cls} | C{new_cls} | {orig_cap} | {new_cap} |")

    lines.append("\n---\n")

    # 7. 按严重度分组的变化
    lines.append("## 7. 按严重度分组的变化统计")
    for sev in ['mild', 'moderate', 'noticeable', 'severe']:
        subset = lists[f'severity_{sev}']
        changed_subset = subset[subset['caption'] != subset['original_caption']]
        lines.append(f"- **{sev}**: {len(changed_subset)}/{len(subset)} 变化")

    lines.append("\n---\n")

    # 8. 导出说明
    lines.append("## 8. 数据导出")
    lines.append("\n详细的筛选数据已导出为JSON格式:")
    lines.append(f"- `{REPORT_DIR}/review_data.json` - 所有筛选列表")
    lines.append(f"- `{REPORT_DIR}/changed_samples.csv` - 变化样本的CSV")
    lines.append(f"- `{REPORT_DIR}/full_comparison.csv` - 完整对比数据")

    # 写入Markdown
    report_path = REPORT_DIR / f"caption_review_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return report_path


def export_data(df, changed, lists):
    """导出详细数据"""

    # 1. 变化样本CSV
    changed_export = changed[[
        'file_id', 'original_dominant_class', 'actual_dominant_class',
        'class_1_ratio', 'class_2_ratio', 'class_3_ratio', 'S_full',
        'original_caption', 'caption', 'severity_token'
    ]].copy()
    changed_export = changed_export.sort_values('file_id')
    changed_export.to_csv(REPORT_DIR / "changed_samples.csv", index=False)

    # 2. 完整对比CSV
    full_export = df[[
        'file_id', 'original_dominant_class', 'actual_dominant_class',
        'class_1_ratio', 'class_2_ratio', 'class_3_ratio', 'S_full',
        'original_caption', 'caption', 'severity_token'
    ]].copy()
    full_export['changed'] = df['caption'] != df['original_caption']
    full_export = full_export.sort_values('file_id')
    full_export.to_csv(REPORT_DIR / "full_comparison.csv", index=False)

    # 3. JSON格式的筛选列表
    review_data = {
        "summary": {},
        "lists": {}
    }

    # 为每个筛选列表导出关键信息
    for list_name, subset in lists.items():
        if len(subset) == 0:
            continue

        review_data["lists"][list_name] = {
            "count": len(subset),
            "samples": []
        }

        # 只导出变化样本的前100个（减少文件大小）
        for _, row in subset.head(100).iterrows():
            review_data["lists"][list_name]["samples"].append({
                "file_id": row['file_id'],
                "original_class": int(row['original_dominant_class']),
                "fixed_class": int(row['actual_dominant_class']),
                "c1_pct": float(row['class_1_ratio'] * 100),
                "c2_pct": float(row['class_2_ratio'] * 100),
                "c3_pct": float(row['class_3_ratio'] * 100),
                "severity": row['severity_token'],
                "original_caption": row['original_caption'],
                "fixed_caption": row['caption'],
                "changed": row['caption'] != row['original_caption']
            })

    with open(REPORT_DIR / "review_data.json", 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 数据导出完成:")
    print(f"   - changed_samples.csv: {len(changed_export)} 变化样本")
    print(f"   - full_comparison.csv: {len(full_export)} 完整对比")
    print(f"   - review_data.json: {len(review_data['lists'])} 个筛选列表")


def main():
    print("=" * 70)
    print("生成Caption修复审阅报告")
    print("=" * 70)

    # 加载数据
    print("\n加载manifest和原始captions...")
    df = load_data()
    print(f"读取 {len(df)} 个样本")

    # 创建摘要
    print("\n分析变化...")
    summary, changed = create_change_summary(df)

    print(f"总样本: {summary['total_samples']}")
    print(f"变化样本: {summary['changed_samples']} ({summary['change_percentage']:.1f}%)")
    print(f"类别变化: {summary['class_change_samples']}")

    # 创建筛选列表
    print("\n创建筛选列表...")
    lists = create_filtered_lists(df)
    print(f"创建 {len(lists)} 个筛选列表")

    # 生成Markdown报告
    print("\n生成Markdown报告...")
    report_path = generate_markdown_report(summary, changed, lists)
    print(f"报告已生成: {report_path}")

    # 导出数据
    print("\n导出详细数据...")
    export_data(df, changed, lists)

    print("\n" + "=" * 70)
    print("✅ 审阅报告生成完成!")
    print("=" * 70)
    print(f"\n报告目录: {REPORT_DIR}/")
    print(f"主报告: {report_path.name}")
    print(f"\n可用数据文件:")
    print(f"  - caption_review_report_*.md        # 主报告(Markdown)")
    print(f"  - changed_samples.csv                # 变化样本")
    print(f"  - full_comparison.csv                # 完整对比")
    print(f"  - review_data.json                   # 筛选列表")


def create_random_samples(df, n_per_category=10):
    """创建随机抽样用于视觉检查"""
    import shutil

    sample_dir = REPORT_DIR / "random_samples"
    sample_dir.mkdir(exist_ok=True)

    # 按修复后的类别分组抽样
    samples_by_class = {}

    for fixed_class in [1, 2, 3]:
        class_df = df[df['actual_dominant_class'] == fixed_class]
        if len(class_df) == 0:
            continue

        # 随机抽样
        sampled = class_df.sample(n=min(n_per_category, len(class_df)), random_state=42)
        samples_by_class[f'C{fixed_class}'] = sampled

        # 复制图像和caption到样本目录
        class_dir = sample_dir / f"class_{fixed_class}_{CLASS_NAMES[fixed_class].replace(' ', '_')}"
        class_dir.mkdir(exist_ok=True)

        for _, row in sampled.iterrows():
            file_id = row['file_id']

            # 复制图像
            src_img = TRAIN_DIR / f"{file_id}.jpg"
            if src_img.exists():
                dst_img = class_dir / f"{file_id}.jpg"
                shutil.copy2(src_img, dst_img)

            # 创建caption文件
            caption_content = f"""File ID: {file_id}
Original Caption: {row['original_caption']}
Fixed Caption: {row['caption']}

Class Ratios:
  C1 (transparent): {row['class_1_ratio']*100:.1f}%
  C2 (semi-transparent): {row['class_2_ratio']*100:.1f}%
  C3 (opaque): {row['class_3_ratio']*100:.1f}%

Original Dominant Class: C{row['original_dominant_class']} ({CLASS_NAMES.get(row['original_dominant_class'], 'N/A')})
Fixed Dominant Class: C{row['actual_dominant_class']} ({CLASS_NAMES.get(row['actual_dominant_class'], 'N/A')})

Severity: {row['severity_token']} (S_full={row['S_full']:.3f})
"""
            caption_file = class_dir / f"{file_id}_info.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption_content)

    # 创建抽样清单
    manifest_lines = ["# Random Samples for Visual Inspection\n"]
    manifest_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    manifest_lines.append(f"Samples per category: {n_per_category}\n\n")

    for cls_name, sampled in sorted(samples_by_class.items()):
        manifest_lines.append(f"## {cls_name} ({len(sampled)} samples)\n")
        for _, row in sampled.iterrows():
            manifest_lines.append(f"- [{row['file_id']}]({cls_name}_{CLASS_NAMES[int(cls_name[1])].replace(' ', '_')}/{row['file_id']}.jpg)")
            manifest_lines.append(f"  - C1: {row['class_1_ratio']*100:.0f}%, C2: {row['class_2_ratio']*100:.0f}%, C3: {row['class_3_ratio']*100:.0f}%")
            manifest_lines.append(f"  - Caption: {row['caption'][:60]}...\n")

    with open(sample_dir / "samples_manifest.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest_lines))

    print(f"\n✅ 随机抽样已创建:")
    print(f"   目录: {sample_dir}/")
    print(f"   每类样本数: {n_per_category}")
    print(f"   清单: samples_manifest.md")

    return sample_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成Caption修复审阅报告")
    parser.add_argument("--sample", type=int, default=10,
                       help="每类随机抽样数量 (默认: 10)")
    args = parser.parse_args()

    main()

    if args.sample > 0:
        print("\n" + "=" * 70)
        print("创建随机抽样用于视觉检查...")
        print("=" * 70)

        df = load_data()
        create_random_samples(df, n_per_category=args.sample)
