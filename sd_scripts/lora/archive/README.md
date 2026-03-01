# LoRA Project Archive

This directory contains deprecated scripts, test data, and experimental results that have been archived for reference purposes.

## Archive Contents (2026-02-18)

### caption_fix_experiment_rolledback/
**Reason**: Caption fixing experiment that was rolled back based on user decision to trust WoodScape's original annotations.
- `fix_captions.py` - Script to fix captions based on pixel distribution
- `generate_caption_review_report.py` - Report generation for caption review
- `restore_manifest.py` - Script to restore original manifest
- `caption_review_report/` - Generated review reports

**User Decision**: Trust WoodScape's "visual dominance" labeling over pixel-count based classification.

### test_utilities/
**Reason**: Test and verification utilities no longer needed in active development.
- `create_test_masks.py` - Test mask creation utility
- `verify_unet_compatibility.py` - UNet verification script
- `verify_unet_offline.py` - Offline UNet verification
- `visualize_training_data.py` - Data visualization utility
- `evaluate_metrics.py` - Metrics evaluation script

### old_scripts/
**Reason**: Replaced by newer implementations or integrated into main scripts.
- `filter_training_data.py` - Old filtering implementation
- `filter_training_data_cli.py` - CLI version of filtering
- `migrate_previews.py` - Preview migration utility
- `train_lora_diffusers.py` - Alternative training script (replaced by 03_train_lora.py)

### test_data/
**Reason**: Test data accumulated during development, no longer needed.
- `test_data/` - Small test dataset (580K)
- `test_input/` - Test clean images (74M)
- `test_output/` - Test generation outputs (7.2M)
- `inpainting_test_output/` - Inpainting test results

### old_docs/
**Reason**: Superseded by newer documentation or experimental reports.
- `filtering_guide.md` - Old filtering guide
- `lora_training_proposal.md` - Original training proposal
- `lora_training_proposal_v1.1.md` - Updated training proposal
- `training_data_analysis.md` - Training data analysis
- `phase1_summary.md` - Phase 1 summary
- `phase1.2_summary.md` - Phase 1.2 summary

### shell_scripts/
**Reason**: Wrapper scripts replaced by direct Python script execution.
- `test_lora.sh` - LoRA testing wrapper
- `test_lora_inpaint.sh` - Inpainting test wrapper
- `train_lora.sh` - Training wrapper

### training_data_analysis/
**Reason**: Intermediate analysis results from data exploration.
- `filter_results/` - Data filtering results
- `phase2a_visual_check/` - Visual check samples
- `pure_samples_analysis/` - Pure sample analysis
- `visual_samples/` - Visualization samples

## Active Scripts (Not Archived)

The following scripts remain in the main lora directory and are actively used:

1. `01_prepare_training_data.py` - Training data preparation
2. `02_generate_filtered_training_data.py` - Filtered data generation
3. `03_train_lora.py` - **Main LoRA training script**
4. `04_test_lora.py` - LoRA testing
5. `05_test_lora_inpaint.py` - Inpainting pipeline testing
6. `06_test_material_prompts.py` - Material prompt testing (deprecated but kept for reference)
7. `07_batch_generate_synthetic_data.py` - **Main batch generation script**

## Active Directories

- `docs/` - Active documentation (sd_generation_guide.md)
- `training_data/dreambooth_format/` - Current training data
- `output/` - LoRA model checkpoints
- `logs/` - Training logs

## Recovery

If you need to restore any archived items, they can be copied back from this directory.
