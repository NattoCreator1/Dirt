# Phase 1 Status Update

**Date**: 2026-02-25
**Status**: ⚠️ IMPORTANT FINDING - SD 8960 has opposite distribution to SD 1260

---

## Critical Discovery

After analyzing the full SD 8960 dataset, we found:

| Dataset | Samples | S Mean | vs WoodScape |
|---------|---------|--------|--------------|
| WoodScape Train | 3200 | 0.4081 | - |
| SD 1260 | 1260 | 0.2866 | **-0.1215** (lower) |
| SD 8960 | 8960 | 0.4658 | **+0.0576** (higher!) |

**Key Insight**: SD 1260 was a **biased subset** with low S values due to conservative filtering. The full SD 8960 dataset actually has **higher** S values than WoodScape!

---

## Implications for Calibration

### For SD 1260 (if used):
- Need to UP-weight high S bins (weights > 1)
- SD lacks high S samples → boost them

### For SD 8960 (full dataset):
- Need to DOWN-weight all bins (weights < 1)
- SD has 2-12x MORE samples per bin than WoodScape
- Target: reduce SD influence to match WoodScape distribution

---

## Computed Weights for SD 8960

| S Bin | WoodScape | SD 8960 | Weight |
|-------|-----------|---------|--------|
| [0.0, 0.1) | 299 | 717 | 0.42 |
| [0.1, 0.2) | 534 | 1175 | 0.45 |
| [0.2, 0.3) | 363 | 899 | 0.40 |
| [0.3, 0.4) | 466 | 1216 | 0.38 |
| [0.4, 0.5) | 424 | 971 | 0.44 |
| [0.5, 0.6) | 290 | 1077 | 0.27 |
| [0.6, 0.7) | 339 | 782 | 0.43 |
| [0.7, 0.8) | 293 | 904 | 0.32 |
| [0.8, 0.9) | 152 | 706 | 0.22 |
| [0.9, 1.0) | 40 | 513 | 0.10 |

**Weight Statistics**:
- Mean: 0.36
- Range: [0.10, 0.45]
- All weights < 1 (down-weighting)

---

## Files Created

### Phase 0 Diagnostic
- ✅ `diagnose_sd_1260.py` - SD 1260 analysis
- ✅ `diagnose_sd_8960.py` - SD 8960 analysis (key finding!)
- ✅ `diagnostic_report.json` - SD 1260 report
- ✅ `diagnostic_report_8960.json` - SD 8960 report

### Phase 1 Implementation
- ✅ `two_layer_sampler.py` - Two-layer sampling implementation
- ✅ `compute_weights.py` - Weight computation (updated for 8960)
- ✅ `prepare_mixed_dataset.py` - Mixed dataset preparation
- ✅ `train_mixed_d_only.py` - Training script

### Outputs
- ✅ `weights_8960_correct.npz` - Correct weights for SD 8960
- ✅ `mixed_ws4000_sd8960.csv` - Mixed dataset manifest

---

## Ready to Train

To train Mixed(8960, D-only):

```bash
python sd_calibration_work/phase1_importance_sampling/train_mixed_d_only.py \
    --mixed_csv sd_calibration_work/phase1_importance_sampling/mixed_ws4000_sd8960.csv \
    --weights sd_calibration_work/phase1_importance_sampling/weights_8960_correct.npz \
    --run_name mixed_ws4000_sd8960_d_only \
    --total_steps 10000 \
    --real_ratio 0.76
```

---

## Experimental Groups Updated

| Experiment | SD Data | Calibration | Purpose |
|------------|---------|-------------|---------|
| Baseline | - | - | Control |
| Mixed(1260, no cal) | 1260 | None | Current best (S too low) |
| **Mixed(8960, D-only)** | **8960** | **Importance Sampling** | **Main experiment** |
| Mixed(8960, no cal) | 8960 | None | Control (S too high?) |

**Note**: Mixed(8960, no cal) may produce POSITIVE bias (opposite of Mixed 1260) since SD 8960 S_mean > WoodScape.

---

## Updated Success Criteria

After calibration with SD 8960:
- **Test_Real MAE** ≤ 0.019 (maintain performance)
- **Test_Real Bias**: |bias| < 0.05 (eliminate bias, whether negative or positive)
- **Test_Ext ρ** ≥ 0.70 (maintain rank correlation)

**Expected Outcome**:
- D-only should reduce SD's over-representation in high S bins
- Should prevent the model from being overly aggressive in predictions
- May improve Test_Ext performance

---

**Phase 1 Status**: Implementation complete, ready to train
**Next**: Execute training and evaluate results
