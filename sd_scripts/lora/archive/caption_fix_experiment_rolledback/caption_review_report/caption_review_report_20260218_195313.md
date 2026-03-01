# Caption修复审阅报告

生成时间: 2026-02-18 19:53:13

---

## 1. 执行摘要
- **总样本数**: 3200
- **变化样本数**: 1541 (48.2%)
- **未变化样本数**: 1659
- **类别变化样本数**: 0

**原始Caption已备份至**: `train_backup_original/`

---

## 2. 类别变化矩阵 (原始 → 修复后)
| 原始 | 修复后 | 样本数 |
|------|--------|--------|
| C2 (semi-transparent) | C2 (semi-transparent) | 839 |
| C3 (opaque) | C3 (opaque) | 2361 |

---

## 3. 类别分布对比

### 原始分布
| 类别 | 样本数 | 比例 |
|------|--------|------|
| C2 (semi-transparent) | 839 | 26.2% |
| C3 (opaque) | 2361 | 73.8% |

### 修复后分布
| 类别 | 样本数 | 比例 |
|------|--------|------|
| C2 (semi-transparent) | 839 | 26.2% |
| C3 (opaque) | 2361 | 73.8% |

---

## 4. Caption分布对比

### 原始Caption (Top 10)
| Caption | 样本数 |
|---------|--------|
| noticeable opaque heavy stains, on camera lens, out of focus... | 995 |
| severe opaque heavy stains, on camera lens, out of focus for... | 983 |
| noticeable semi-transparent dirt smudge, on camera lens, out... | 367 |
| severe semi-transparent dirt smudge, on camera lens, out of ... | 346 |
| noticeable transparent soiling layer, on camera lens, out of... | 272 |
| severe transparent soiling layer, on camera lens, out of foc... | 237 |

### 修复后Caption (Top 10)
| Caption | 样本数 |
|---------|--------|
| severe opaque heavy stains, on camera lens, out of focus for... | 1566 |
| noticeable semi-transparent dirt smudge, on camera lens, out... | 839 |
| noticeable opaque heavy stains, on camera lens, out of focus... | 795 |

---

## 5. 变化样本详情 (按原始类别分组)

**提示**: 类别比例格式为 `C1=x%, C2=y%, C3=z%`

### 原始类别 C2 (semi-transparent) 的变化样本
共 642 个样本

#### C2 → C2 (642 样本)
| 文件ID | C1% | C2% | C3% | 严重度 | 原始Caption | 修复Caption |
|--------|-----|-----|-----|--------|-------------|-------------|
| 0001_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0003_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0005_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0007_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0031_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0033_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0035_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0037_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0067_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0069_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0071_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0073_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0097_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0099_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0101_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0103_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0158_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0160_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0162_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0164_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| ... | | | | | 还有 622 个样本 | |

### 原始类别 C3 (opaque) 的变化样本
共 899 个样本

#### C3 → C3 (899 样本)
| 文件ID | C1% | C2% | C3% | 严重度 | 原始Caption | 修复Caption |
|--------|-----|-----|-----|--------|-------------|-------------|
| 0008_MVL | 7% | 4% | 67% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0011_MVL | 7% | 4% | 67% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0013_MVL | 7% | 4% | 67% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0014_MVL | 7% | 4% | 67% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0015_MVR | 0% | 44% | 56% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0017_MVR | 0% | 44% | 56% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0019_MVR | 0% | 44% | 56% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0021_MVR | 0% | 44% | 56% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| 0023_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0025_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0027_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0029_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0057_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0060_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0061_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0063_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0065_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0121_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0123_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0125_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| ... | | | | | 还有 879 个样本 | |

---


---

## 7. 按严重度分组的变化统计
- **mild**: 0/0 变化
- **moderate**: 0/0 变化
- **noticeable**: 958/1634 变化
- **severe**: 583/1566 变化

---

## 8. 数据导出

详细的筛选数据已导出为JSON格式:
- `docs/caption_review_report/review_data.json` - 所有筛选列表
- `docs/caption_review_report/changed_samples.csv` - 变化样本的CSV
- `docs/caption_review_report/full_comparison.csv` - 完整对比数据