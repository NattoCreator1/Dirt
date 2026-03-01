# Caption修复审阅报告

生成时间: 2026-02-18 20:04:21

---

## 1. 执行摘要
- **总样本数**: 3200
- **变化样本数**: 1541 (48.2%)
- **未变化样本数**: 1659
- **类别变化样本数**: 1541

**原始Caption已备份至**: `train_backup_original/`

---

## 2. 类别变化矩阵 (原始 → 修复后)
| 原始 | 修复后 | 样本数 |
|------|--------|--------|
| C1 (transparent) | C2 (semi-transparent) | 126 |
| C1 (transparent) | C3 (opaque) | 383 |
| C2 (semi-transparent) | C2 (semi-transparent) | 197 |
| C2 (semi-transparent) | C3 (opaque) | 516 |
| C3 (opaque) | C2 (semi-transparent) | 516 |
| C3 (opaque) | C3 (opaque) | 1462 |

---

## 3. 类别分布对比

### 原始分布
| 类别 | 样本数 | 比例 |
|------|--------|------|
| C1 (transparent) | 509 | 15.9% |
| C2 (semi-transparent) | 713 | 22.3% |
| C3 (opaque) | 1978 | 61.8% |

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

### 原始类别 C1 (transparent) 的变化样本
共 509 个样本

#### C1 → C2 (126 样本)
| 文件ID | C1% | C2% | C3% | 严重度 | 原始Caption | 修复Caption |
|--------|-----|-----|-----|--------|-------------|-------------|
| 0001_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0003_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0005_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0007_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0253_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0255_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0257_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0259_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0457_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0459_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0461_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0463_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0547_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0549_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0551_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0553_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0781_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0969_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0971_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| 0973_FV | 36% | 64% | 0% | noticeable | noticeable transparent soiling layer, on... | noticeable semi-transparent dirt smudge,... |
| ... | | | | | 还有 106 个样本 | |

#### C1 → C3 (383 样本)
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
| 0121_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0123_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0125_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0150_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0152_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0154_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0156_RV | 0% | 9% | 47% | noticeable | noticeable transparent soiling layer, on... | noticeable opaque heavy stains, on camer... |
| 0199_MVL | 7% | 4% | 67% | severe | severe transparent soiling layer, on cam... | severe opaque heavy stains, on camera le... |
| ... | | | | | 还有 363 个样本 | |

### 原始类别 C2 (semi-transparent) 的变化样本
共 516 个样本

#### C2 → C3 (516 样本)
| 文件ID | C1% | C2% | C3% | 严重度 | 原始Caption | 修复Caption |
|--------|-----|-----|-----|--------|-------------|-------------|
| 0057_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0060_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0061_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0063_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0065_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0215_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0217_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0219_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0221_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0292_MVL | 7% | 4% | 67% | severe | severe semi-transparent dirt smudge, on ... | severe opaque heavy stains, on camera le... |
| 0293_MVL | 7% | 4% | 67% | severe | severe semi-transparent dirt smudge, on ... | severe opaque heavy stains, on camera le... |
| 0295_MVL | 7% | 4% | 67% | severe | severe semi-transparent dirt smudge, on ... | severe opaque heavy stains, on camera le... |
| 0297_MVL | 7% | 4% | 67% | severe | severe semi-transparent dirt smudge, on ... | severe opaque heavy stains, on camera le... |
| 0305_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0307_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0309_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0311_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0449_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0451_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| 0453_RV | 0% | 9% | 47% | noticeable | noticeable semi-transparent dirt smudge,... | noticeable opaque heavy stains, on camer... |
| ... | | | | | 还有 496 个样本 | |

### 原始类别 C3 (opaque) 的变化样本
共 516 个样本

#### C3 → C2 (516 样本)
| 文件ID | C1% | C2% | C3% | 严重度 | 原始Caption | 修复Caption |
|--------|-----|-----|-----|--------|-------------|-------------|
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
| 0223_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0225_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0227_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| 0229_FV | 36% | 64% | 0% | noticeable | noticeable opaque heavy stains, on camer... | noticeable semi-transparent dirt smudge,... |
| ... | | | | | 还有 496 个样本 | |

---

## 6. 关键变化样本 (类别变化 ≥2)

共 383 个样本
| 文件ID | C1% | C2% | C3% | 原始类别 | 修复类别 | 原始Caption | 修复Caption |
|--------|-----|-----|-----|----------|----------|-------------|-------------|
| 0008_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0011_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0013_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0014_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0015_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0017_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0019_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0021_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0023_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0025_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0027_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0029_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0121_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0123_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0125_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0150_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0152_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0154_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0156_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0199_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0200_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0202_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0205_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0206_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0207_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0209_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0211_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0213_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0262_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0264_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0266_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0269_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0271_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0273_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0275_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0299_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0301_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0303_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0317_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0319_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0472_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0474_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0476_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0478_MVR | 0% | 44% | 56% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0480_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0482_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0484_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0486_RV | 0% | 9% | 47% | C1 | C3 | noticeable transparent soiling... | noticeable opaque heavy stains... |
| 0525_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |
| 0528_MVL | 7% | 4% | 67% | C1 | C3 | severe transparent soiling lay... | severe opaque heavy stains, on... |

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