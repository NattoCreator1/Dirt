# Phase 1 Implementation Summary

## Status: Complete (100-sample test)

### Implemented Scripts

1. **`01_prepare_training_data.py`** - Main data preparation script
   - Constructs "dirt-dominated" training images with background suppression
   - Generates classified captions with tokens
   - Outputs DreamBooth format (image/caption pairs)
   - Supports 3 background suppression methods: blur, desaturate, downsample

2. **`visualize_training_data.py`** - Visualization helper
   - Visualizes training samples with mask overlays
   - Compares different background suppression methods
   - Shows caption and statistics for each sample

### Phase 1 Test Results

**Configuration**:
- Split: train
- Samples: 100 (from 3200 total)
- Background method: blur
- Resolution: 512×512

**Output Distribution** (from 100 samples):
| Class | Count | Percentage |
|-------|-------|------------|
| opaque | 62 | 62% |
| semi | 29 | 29% |
| trans | 9 | 9% |

**Severity Distribution**:
- Range: 0.0631 - 0.9997
- Mean: 0.4458

**Caption Examples**:
- `<opaque> <sev4> on camera lens, out of focus foreground, subtle glare, background visible`
- `<semi> <sev2> on camera lens, out of focus foreground, subtle glare, background visible`
- `<trans> <sev1> on camera lens, out of focus foreground, subtle glare, background visible`

### Observations

1. **Class Imbalance**: The natural WoodScape distribution is heavily skewed toward opaque soiling. For balanced LoRA training, consider:
   - Stratified sampling to ensure ~33% per class
   - Oversampling minority classes (trans/semi)

2. **Caption Consistency**: All captions use the fixed optical description format. The class and severity tokens are correctly applied based on dominant class and S_full score.

3. **Data Isolation**: Test data (1000 samples) is strictly isolated - not used in training preparation.

### Output Files

```
sd_scripts/lora/training_data/dreambooth_format/
├── train/
│   ├── *.jpg         # Training images (512×512, background suppressed)
│   ├── *.txt         # Captions with tokens
│   └── (100 pairs total)
├── manifest_train.csv
└── visual_samples/
    └── *_training_sample.png    # Visualization of 10 sample images
```

### Next Steps

#### Phase 1.5: Full Data Preparation
- [ ] Prepare full training set (3200 samples) with balanced class sampling
- [ ] Prepare validation set (800 samples)
- [ ] Compare background suppression methods (blur vs desaturate vs downsample)
- [ ] Select optimal method based on visual inspection

#### Phase 2: LoRA Training
- [ ] Install diffusers, peft, accelerate dependencies
- [ ] Configure training script with v1.1 parameters:
  - Base model: `stabilityai/stable-diffusion-2-1-base`
  - Rank: 16 (alpha=32)
  - Learning rate: 1e-4
  - Batch size: 2
  - Gradient accumulation: 4
  - Mixed precision: fp16
- [ ] Run small-scale validation (100 samples, 5 epochs)
- [ ] Evaluate validation prompts and adjust

#### Phase 3: Integration
- [ ] Modify `02_generate_synthetic.py` to load LoRA weights
- [ ] Run end-to-end generation comparison (baseline vs LoRA)
- [ ] Evaluate with 6 quality control metrics

### Training Command Template

```bash
accelerate launch --mixed_precision="fp16" \
    scripts/train_lora_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --instance_data_dir="./sd_scripts/lora/training_data/dreambooch_format/train" \
    --output_dir="./sd_scripts/lora/weights" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=5000 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --seed=42 \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --mixed_precision="fp16" \
    --save_steps=500 \
    --validation_prompt="<opaque> <sev3> on camera lens" \
    --num_validation_images=10
```

### Key Design Decisions (v1.1)

1. **Route A Strategy**: Train "dirt appearance LoRA" on text-to-image, then mount to SD2-inpainting for inference
2. **Background Suppression**: Use blur/downsample to reduce background semantics, forcing LoRA to learn dirt features
3. **Classified Captions**: Use `<trans>`, `<semi>`, `<opaque>` tokens to maintain controllability
4. **Severity Tokens**: Use `<sev1>` through `<sev4>` for fine-grained control

---

**Date**: 2025-02-10
**Status**: Phase 1 Complete, Ready for Phase 2
