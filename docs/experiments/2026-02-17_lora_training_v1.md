# LoRA Training v1: Lens Soiling Effects

**Date:** 2026-02-17
**Experiment ID:** lora_v1
**Status:** Configuration complete, ready to launch

---

## 1. Experiment Overview

### 1.1 Objective
Train a LoRA (Low-Rank Adaptation) adapter on Stable Diffusion 2.1 Base to generate realistic lens soiling and dirt effects for camera imagery.

### 1.2 Base Model
- **Model:** `/home/yf/models/sd2_1_base` (local SD2.1 base model)
- **Resolution:** 512×512
- **Architecture:** Stable Diffusion 2.1 with U-Net and CLIP text encoder
- **Note:** Using local base model (v2-1_512-ema-pruned.safetensors)

### 1.3 Training Method
- **Technique:** LoRA fine-tuning
- **Target Layers:** U-Net attention layers (Q, V projections)
- **Rank:** 16
- **Alpha:** 32

---

## 2. Dataset Specifications

### 2.1 Data Source
- **Dataset:** WoodScape (Volvo Cars authored public dataset)
- **Raw Data Location:** `/dataset/woodscape_raw/train/`
- **Masks:** Direct from WoodScape GT labels (trusted quality)

### 2.2 Training Data Summary

| Metric | Value |
|--------|-------|
| Total Samples | 3,200 |
| Image Resolution | 512×512 |
| Format | DreamBooth (image + caption) |
| Data Directory | `/home/yf/soiling_project/sd_scripts/lora/training_data/dreambooth_format/train` |

### 2.3 Class Distribution

| Class | Description | Count | Percentage |
|-------|-------------|-------|------------|
| C0 | Clean | ~51 | 1.6% |
| C1 | Transparent soiling | ~509 | 15.9% |
| C2 | Semi-transparent soiling | ~714 | 22.3% |
| C3 | Opaque heavy stains | ~1,977 | 61.8% |

### 2.4 Severity Distribution

| Severity | S_full Range | Count | Percentage |
|----------|-------------|-------|------------|
| Mild | < 0.15 | 0 | 0% |
| Moderate | 0.15 - 0.35 | 0 | 0% |
| Noticeable | 0.35 - 0.60 | 1,636 | 51.1% |
| Severe | > 0.60 | 1,564 | 48.9% |

---

## 3. Training Configuration

### 3.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Steps | 3,000 | ~1 epoch for 3200 samples with batch_size=4 |
| Batch Size | 4 | Balanced for GPU memory and training stability |
| Learning Rate | 1e-4 | Standard LoRA learning rate |
| LR Scheduler | Constant | Stable training with warmup |
| LR Warmup Steps | 500 | ~15% of training, stabilizes early training |
| Mixed Precision | FP16 | Memory efficiency |
| Seed | 42 | Reproducibility |

### 3.2 LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank | 16 | Good balance between expressiveness and overfitting |
| Alpha | 32 | 2× rank is standard practice |
| Dropout | 0.1 | Light regularization |

### 3.3 Background Suppression

| Parameter | Value | Description |
|-----------|-------|-------------|
| Method | Random | Randomly selects blur, desaturate, or downsample |
| Intensity Range | 0.5 - 1.5 | Light suppression, preserves some context |

**Note:** Background suppression is applied to clean regions to prevent the model from learning background styles. The intensity is intentionally light to avoid degrading image quality.

---

## 4. Caption System

### 4.1 Natural Language Tokens

The caption system uses descriptive phrases that map to soiling classes:

| Class | Severity Level | Token |
|-------|---------------|-------|
| C1 | Mild/Noticeable | "transparent soiling layer" |
| C1 | Severe | "severe transparent soiling layer" |
| C2 | Mild/Noticeable | "semi-transparent dirt smudge" |
| C2 | Severe | "severe semi-transparent dirt smudge" |
| C3 | Mild/Noticeable | "opaque heavy stains" |
| C3 | Severe | "severe opaque heavy stains" |

### 4.2 Caption Template

```
{severity_token}, on camera lens, out of focus foreground, subtle glare, background visible
```

**Example:**
- "noticeable transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible"
- "severe opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible"

---

## 5. Training Scripts

### 5.1 Python Script
- **Location:** `/home/yf/soiling_project/sd_scripts/lora/03_train_lora.py`
- **Usage:** Configuration script for training parameters

### 5.2 Shell Script
- **Location:** `/home/yf/soiling_project/sd_scripts/lora/train_lora.sh`
- **Usage:** `bash train_lora.sh`

### 5.3 Launch Command
```bash
accelerate launch --mixed_precision=fp16 train_lora_diffusers.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --train_data_dir="/home/yf/soiling_project/sd_scripts/lora/training_data/dreambooth_format/train" \
  --output_dir="./output/lora_v1" \
  --max_train_steps=3000 \
  --train_batch_size=4 \
  --learning_rate=1e-4 \
  --lora_rank=16 \
  --lora_alpha=32
```

---

## 6. Expected Outcomes

### 6.1 Training Metrics to Monitor
- Loss curve convergence
- Checkpoint quality at steps 500, 1000, 1500, 2000, 2500, 3000
- Overfitting indicators

### 6.2 Success Criteria
- Model can generate soiling effects from text prompts
- Generated artifacts match soiling class descriptions
- No significant background style transfer

### 6.3 Next Steps After Training
1. Generate test samples with various prompts
2. Visual inspection of generated soiling effects
3. Quantitative evaluation if ground truth available
4. Iterate on hyperparameters if needed

---

## 7. Important Notes

### 7.1 Mask Quality
- Masks are directly from WoodScape dataset (authoritative source)
- No additional filtering applied
- Trust in dataset annotation quality

### 7.2 Background Suppression
- Light suppression (0.5-1.5 intensity)
- May not be visually obvious in high-coverage samples
- Verified working on sample analysis (94% sharpness reduction in clean areas)

### 7.3 Class Weight System
- Current training uses natural language tokens (captions)
- The two-weight system from `2025-02-09_two_weight_systems_validation.md` applies to the segmentation task, not LoRA training
- LoRA training is unconditional (uses captions only)

---

## 8. Dependencies

```bash
pip install diffusers transformers accelerate peft torch torchvision
```

---

## 9. References

- WoodScape Dataset: [GitHub](https://github.com/woodscape)
- 2025-02-09_two_weight_systems_validation.md: Weight system for segmentation (not used in LoRA training)
- Stable Diffusion 2.1: stabilityai/stable-diffusion-2-1-base

---

## 10. Change Log

| Date | Change |
|------|--------|
| 2026-02-17 | Initial configuration and documentation |
