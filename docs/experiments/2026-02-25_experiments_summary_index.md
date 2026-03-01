# 实验文档索引

**更新日期**: 2026-02-25

---

## 一、核心评估文档

### 1.1 三模型全面对比 ⭐
**文件**: `2026-02-25_three_models_comprehensive_comparison_baseline_989_1260.md`

**内容**:
- Baseline vs Mixed 989 vs Mixed 1260 全面对比
- Test_Real详细分析（各等级MAE/Bias/分布）
- Test_Ext详细分析（各等级均值/标准差/区间分布）
- Test_Real vs Test_Ext对比分析
- 核心结论和问题分析

**关键结论**:
- Test_Real MAE: Mixed 1260 (0.0187) ≈ Baseline (0.0187) ✅
- Test_Ext ρ: Baseline (0.718) > Mixed 989 (0.710) > Mixed 1260 (0.670)
- Mixed模型系统性负偏差: Test_Real (-0.09), Test_Ext (-0.20~-0.30)

---

### 1.2 Mixed 1260 (Conservative Final) 评估
**文件**: `2026-02-25_mixed_1260_conservative_final_evaluation.md`

**内容**:
- 保守渐进式筛选（271条新增样本）
- 标签一致性修正（使用ablation CSV的S_full_wgap_alpha50）
- Test_Real/Test_Ext评估结果
- 与之前版本的对比（unfixed/fixed）

**关键结论**:
- Test_Real MAE: 0.0187 ✅ (优于Baseline 0.0192)
- Test_Ext ρ: 0.670 (略低于Mixed 989的0.710)

---

### 1.3 Mixed 989 (Baseline筛选) 评估
**文件**: `2026-02-24_mixed_training_989_baseline_filtered_evaluation.md`

**内容**:
- Baseline筛选方法（误差<0.05）
- 与手动筛选680的对比
- Test_Real/Test_Ext评估结果

**关键结论**:
- Test_Real MAE: 0.0188 ✅ (与Baseline相当)
- Test_Ext ρ: 0.710 (接近Baseline 0.718)
- 验证了Baseline筛选方法的有效性

---

### 1.4 Mixed 989 Augmented 评估
**文件**: `2026-02-25_mixed_989_augmented_evaluation.md`

**内容**:
- Mixed 989的数据增强版本
- Test_Ext评估对比

---

## 二、历史参考文档

### 2.1 手动筛选680失败案例
**文件**: `2026-02-24_mixed_training_989_baseline_filtered_evaluation.md` (包含对比)

**关键结果**:
- Test_Real MAE: 0.0996 ❌ (5x worse than Baseline)
- Test_Ext ρ: 0.171 ❌ (性能崩塌)
- 根本原因: 手动筛选偏向高S值样本，NPZ标签失真

---

## 三、分析脚本索引

### 3.1 三模型对比脚本
| 脚本 | 功能 |
|------|------|
| `analyze_testreal_three_models_comparison.py` | Test_Real三模型详细对比 |
| `analyze_testext_three_models_comparison.py` | Test_Ext三模型详细对比 |

**使用方法**:
```bash
python analyze_testreal_three_models_comparison.py
python analyze_testext_three_models_comparison.py
```

---

## 四、模型文件索引

### 4.1 Checkpoint路径
| 模型 | 路径 |
|------|------|
| Baseline | `baseline/runs/model1_r18_640x480/ckpt_best.pth` |
| Mixed 989 | `baseline/runs/mixed_ws4000_sd989_baseline_filtered/ckpt_best.pth` |
| Mixed 1260 Final | `baseline/runs/mixed_ws4000_sd1260_conservative_final/ckpt_best.pth` |
| 手动680 (失败) | `baseline/runs/mixed_experiment/mixed_ws578_sd_wgap_alpha50/ckpt_best.pth` |

### 4.2 预测结果文件
| 模型 | Test_Real | Test_Ext |
|------|-----------|----------|
| Baseline | `eval_Test_Real_predictions.csv` | `eval_Test_Ext_predictions.csv` |
| Mixed 989 | `eval_Test_Real_predictions.csv` | `eval_Test_Ext_predictions.csv` |
| Mixed 1260 | `eval_Test_Real_predictions.csv` | `eval_Test_Ext_predictions.csv` |

---

## 五、快速查找指南

### 5.1 按指标查找
| 指标 | 最优模型 | 参考文档 |
|------|----------|----------|
| Test_Real MAE | Mixed 1260 (0.0187) | 三模型全面对比 §2.1 |
| Test_Real ρ | Baseline (0.9936) | 三模型全面对比 §2.1 |
| Test_Ext ρ | Baseline (0.718) | 三模型全面对比 §3.1 |
| 无偏性 | Baseline (bias≈0) | 三模型全面对比 §2.1 |

### 5.2 按问题查找
| 问题 | 相关章节 |
|------|----------|
| Mixed模型系统性低估 | 三模型全面对比 §5.2 |
| Level 4预测下移 | 三模型全面对比 §2.2.4 |
| Test_Ext等级对比 | 三模型全面对比 §3.2 |
| 手动筛选失败原因 | Mixed 989评估 §4.1 |

---

**文档维护**: 本索引会随新实验添加而更新
**最后更新**: 2026-02-25