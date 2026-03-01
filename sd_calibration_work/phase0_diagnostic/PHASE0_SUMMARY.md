# Phase 0 Completion Summary

**Date**: 2026-02-25
**Status**: ✅ COMPLETE

---

## Executive Summary

Phase 0 diagnostic analysis confirmed that **SD data requires calibration** before use in mixed training with WoodScape.

**Key Finding**: SD 1260 S_mean (0.2866) is **significantly lower** than WoodScape S_mean (0.4081), with a difference of -0.1215.

---

## Diagnostic Results

### Distribution Comparison

| Metric | WoodScape (3200) | SD 1260 (1260) | Difference |
|--------|------------------|----------------|------------|
| **S Mean** | 0.4081 | 0.2866 | **-0.1215** |
| **S Std** | 0.2399 | 0.0860 | - |
| **S Range** | [0.001, 0.961] | [0.064, 0.543] | - |

### Statistical Test

- **KS Test**: Statistic = 0.4309, p < 1e-150
- **Conclusion**: Distribution difference is **extremely significant**

### Bin-by-Bin Analysis

| S Bin | WoodScape | SD 1260 | SD/WS Ratio | Weight Needed |
|-------|-----------|---------|-------------|---------------|
| [0.0, 0.1) | 299 | 14 | 0.05 | ×21.36 |
| [0.1, 0.2) | 534 | 279 | 0.52 | ×1.91 |
| [0.2, 0.3) | 363 | 309 | 0.85 | ×1.17 |
| [0.3, 0.4) | 466 | 577 | 1.24 | ×0.81 |
| [0.4, 0.5) | 424 | 80 | 0.19 | ×5.30 |
| [0.5, 0.6) | 290 | 1 | 0.00 | ×290.00 |
| [0.6, 1.0) | 824 | 0 | 0.00 | use 0.5 bin weight |

---

## Root Cause Analysis

1. **SD Generation Bias**: Stable Diffusion tends to produce medium-soiling images
2. **Conservative Filtering**: The 1260 selection favored low-S "conservative" samples
3. **Label Scale Inconsistency**: SD NPZ S-values may use different calculation method

---

## Impact on Current Models

Based on `2026-02-25_three_models_comprehensive_comparison_baseline_989_1260.md`:

| Model | Test_Real MAE | Test_Real Bias | Test_Ext ρ |
|-------|---------------|----------------|------------|
| Baseline | 0.0192 | +0.002 | **0.718** |
| Mixed 1260 | 0.0187 | **-0.093** | 0.670 |

**Observed Issues**:
- Systematic negative bias in Mixed models
- Level compression (Level 4 mean: 0.68 → 0.44)
- Reduced rank correlation on Test_Ext

---

## Calibration Strategy

**Phase 1**: Binned Importance Sampling (D-only)

```python
# Weight calculation for each bin
weight[bin] = woodscape_count[bin] / sd_count[bin]

# Example:
# [0.4, 0.5): 424 / 80 = 5.30
# [0.5, 0.6): 290 / 1 = 290.00
```

**Implementation**: Two-layer sampler maintaining real:sd ratio while applying weights to SD subset.

---

## Success Criteria

After calibration, the model should achieve:
- **Test_Real MAE** ≤ 0.019 (maintain current performance)
- **Test_Ext ρ** ≥ 0.70 (recover rank correlation)
- **Test_Real Bias** |bias| < 0.05 (eliminate systematic bias)

---

## Deliverables

✅ `diagnostic_report.json` - Numerical statistics
✅ `diagnostic_report.md` - Comprehensive analysis
✅ `distribution_comparison.png` - Visualization
✅ `diagnose_sd_1260.py` - Diagnostic script
✅ `visualize_distributions.py` - Visualization script

---

## Next Steps

1. ✅ **Phase 0**: Complete diagnostic analysis
2. **Phase 1**: Implement two-layer importance sampler
3. **Phase 1**: Train Mixed(8960, D-only) for 10,000 steps
4. **Phase 1**: Evaluate and compare with Baseline/Mixed 1260
5. **Control**: Run Mixed(8960, no cal) for comparison

---

**Phase 0 completed**: 2026-02-25 20:24
**Ready to proceed to Phase 1**
