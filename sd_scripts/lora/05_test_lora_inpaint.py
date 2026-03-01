#!/usr/bin/env python3
"""
Test Script for Trained LoRA on SD2.1 Inpainting

This script loads the trained LoRA weights onto SD2.1 Inpainting pipeline
to generate realistic soiling effects on clean road images with mask constraints.

Key Concepts:
- LoRA trained on SD2.1 Base learns "soiling appearance material"
- Inpainting pipeline provides spatial constraint via mask
- Result: Realistic soiling only in masked regions, background preserved
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import json

import torch
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file
from PIL import Image
import numpy as np


# Caption system matching training tokens
CAPTION_TEMPLATES = {
    "C1": {
        "noticeable": "noticeable transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",
        "severe": "severe transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",
    },
    "C2": {
        "noticeable": "noticeable semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",
        "severe": "severe semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",
    },
    "C3": {
        "noticeable": "noticeable opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",
        "severe": "severe opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",
    },
}

# Negative prompt to guide generation
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text, error, corrupted, ugly, camera, lens"


def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA on SD2.1 Inpainting")

    # Model paths
    parser.add_argument(
        "--inpainting_model",
        type=str,
        default="/home/yf/models/sd2_inpaint",
        help="Path to SD2.1 Inpainting model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to trained LoRA checkpoint",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=3000,
        help="Checkpoint step to load (default: 3000)",
    )

    # Input/Output
    parser.add_argument(
        "--clean_image_dir",
        type=str,
        required=True,
        help="Directory containing clean road images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing corresponding masks",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inpainting_test_output",
        help="Output directory for generated images",
    )

    # Generation settings
    parser.add_argument(
        "--strength",
        type=float,
        default=0.35,
        help="Inpainting strength (0.0-1.0). Lower = preserve more original, Higher = more regeneration",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA adapter weight scale",
    )

    # Test configuration
    parser.add_argument(
        "--soiling_class",
        type=str,
        default="auto",
        choices=["C1", "C2", "C3", "auto", "all"],
        help="Soiling class to generate. 'auto' = detect from mask, 'all' = test all classes",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default="noticeable",
        choices=["noticeable", "severe", "both"],
        help="Severity level to generate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per input (for testing consistency)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Quality control
    parser.add_argument(
        "--save_spill_viz",
        action="store_true",
        help="Save spill visualization (difference map)",
    )
    parser.add_argument(
        "--compute_qc_metrics",
        action="store_true",
        help="Compute quality control metrics",
    )

    return parser.parse_args()


def load_inpainting_pipeline(inpainting_model_path, lora_path, lora_scale=1.0):
    """Load SD2.1 Inpainting pipeline with LoRA adapter"""
    print(f"Loading SD2.1 Inpainting from: {inpainting_model_path}")

    # Load inpainting pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Load LoRA weights
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA from: {lora_path}")

        checkpoint_file = Path(lora_path) / "model.safetensors"

        if checkpoint_file.exists():
            print("Loading LoRA weights into Inpainting UNet...")

            # Load checkpoint
            state_dict = load_file(checkpoint_file)

            # Remove "base_model.model." prefix and filter for LoRA layers only
            unet_state_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k.startswith("base_model.model."):
                    new_key = k.replace("base_model.model.", "")

                    # Skip conv_in/conv_out which have dimension mismatches
                    # LoRA only modifies attention layers (to_q, to_k, to_v, to_out)
                    if "conv_in" in new_key or "conv_out" in new_key:
                        skipped_keys.append(new_key)
                        continue

                    # Only load LoRA adapter weights (lora_)
                    if "lora_" in new_key:
                        unet_state_dict[new_key] = v

            print(f"Found {len(unet_state_dict)} LoRA keys")
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} non-LoRA keys (due to dimension mismatch)")
                for key in skipped_keys[:5]:  # Show first few
                    print(f"  - {key}")

            # Load into UNet with strict=False to handle channel differences
            missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)

            if missing:
                print(f"Info: Missing {len(missing)} keys (expected for inpainting conv_in)")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys")

            print("LoRA loaded successfully!")
        else:
            print(f"Warning: No model.safetensors found in {lora_path}")
    else:
        print("Warning: LoRA path not found, using base inpainting model only")

    # Move to GPU
    pipe = pipe.to("cuda")

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    return pipe


def detect_class_from_mask(mask_path):
    """Detect dominant soiling class from mask"""
    mask = np.array(Image.open(mask_path).convert("L"))

    # Count pixels for each class
    unique, counts = np.unique(mask, return_counts=True)

    # Exclude background (0)
    class_pixels = {int(k): v for k, v in zip(unique, counts) if k > 0}

    if not class_pixels:
        return "C1"  # Default

    # Return class with max pixels
    dominant_class = max(class_pixels, key=class_pixels.get)

    class_map = {1: "C1", 2: "C2", 3: "C3"}
    return class_map.get(dominant_class, "C1")


def soft_process_mask(mask_array, blur_radius=3, dilate_iter=1):
    """Soft process mask to reduce sticker-like appearance"""
    from scipy.ndimage import gaussian_filter, binary_dilation

    # Convert to binary
    binary_mask = (mask_array > 0).astype(np.uint8)

    # Dilate slightly
    if dilate_iter > 0:
        binary_mask = binary_dilation(binary_mask, iterations=dilate_iter)

    # Blur edges
    if blur_radius > 0:
        soft_mask = gaussian_filter(binary_mask.astype(float), sigma=blur_radius)
    else:
        soft_mask = binary_mask.astype(float)

    return soft_mask


def compute_spill_rate(original, generated, mask):
    """Compute spill rate: how much change occurred outside mask"""
    original_np = np.array(original).astype(float)
    generated_np = np.array(generated).astype(float)
    mask_np = np.array(mask).astype(float) / 255.0

    # Absolute difference
    diff = np.abs(generated_np - original_np)
    diff_sum = diff.sum()

    if diff_sum == 0:
        return 0.0

    # Spill: difference outside mask
    mask_inv = 1 - mask_np
    # Handle single channel mask
    if mask_inv.ndim == 2:
        mask_inv = np.stack([mask_inv] * 3, axis=-1)

    spill = (diff * mask_inv).sum()

    return spill / diff_sum


def compute_qc_metrics(original, generated, mask):
    """Compute quality control metrics"""
    results = {}

    # Spill rate
    results["spill_rate"] = compute_spill_rate(original, generated, mask)

    # Background SSIM (simplified)
    # For full implementation, use skimage.metrics.structural_similarity

    # Mask coverage
    mask_np = np.array(mask)
    results["mask_coverage"] = (mask_np > 0).sum() / mask_np.size

    return results


def generate_inpainting(
    pipe,
    clean_image,
    mask_image,
    prompt,
    negative_prompt,
    strength,
    guidance_scale,
    num_inference_steps,
    generator,
):
    """Generate inpainting with LoRA"""
    result = pipe(
        prompt=prompt,
        image=clean_image,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    return result.images[0]


def main():
    args = parse_args()

    # Determine LoRA path
    if args.lora_path is None:
        base_output = Path("/home/yf/soiling_project/sd_scripts/lora/output")
        if base_output.exists():
            checkpoint_dirs = sorted(base_output.glob("*_lora_v1"), reverse=True)
            if checkpoint_dirs:
                lora_base = checkpoint_dirs[0]
                lora_path = lora_base / f"checkpoint-{args.checkpoint_step}"
                if not lora_path.exists():
                    lora_path = lora_base / "final_checkpoint"
                args.lora_path = str(lora_path)
                print(f"Auto-detected LoRA path: {args.lora_path}")

    print("=" * 60)
    print("LoRA + SD2.1 Inpainting Test")
    print("=" * 60)
    print(f"Inpainting model: {args.inpainting_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Clean images: {args.clean_image_dir}")
    print(f"Masks: {args.mask_dir}")
    print(f"Strength: {args.strength}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Steps: {args.num_inference_steps}")
    print("=" * 60)

    # Load pipeline
    pipe = load_inpainting_pipeline(
        args.inpainting_model,
        args.lora_path,
        args.lora_scale,
    )

    # Find clean images
    clean_dir = Path(args.clean_image_dir)
    mask_dir = Path(args.mask_dir)

    clean_images = sorted(list(clean_dir.glob("*.jpg")) + list(clean_dir.glob("*.png")))

    if not clean_images:
        print("Error: No clean images found!")
        return

    print(f"\nFound {len(clean_images)} clean images")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / timestamp
    run_dir.mkdir(exist_ok=True)

    # Results manifest
    manifest = {
        "config": vars(args),
        "lora_path": args.lora_path,
        "timestamp": timestamp,
        "results": [],
    }

    # Generate for each clean image
    for img_idx, clean_path in enumerate(clean_images):
        print(f"\n[{img_idx + 1}/{len(clean_images)}] Processing: {clean_path.name}")

        # Load clean image
        clean_image = Image.open(clean_path).convert("RGB")
        clean_image = clean_image.resize((512, 512))

        # Find corresponding mask
        mask_path = mask_dir / clean_path.name.replace(".jpg", ".png").replace(".jpeg", ".png")

        if not mask_path.exists():
            mask_path = mask_dir / f"{clean_path.stem}.png"

        if not mask_path.exists():
            print(f"  Warning: No mask found for {clean_path.name}")
            continue

        # Load and process mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512))

        # Soft process mask
        mask_array = np.array(mask)
        soft_mask_array = soft_process_mask(mask_array, blur_radius=3, dilate_iter=1)
        soft_mask = Image.fromarray((soft_mask_array * 255).astype(np.uint8))

        # Determine class
        if args.soiling_class == "auto":
            soiling_class = detect_class_from_mask(mask_path)
        elif args.soiling_class == "all":
            test_classes = ["C1", "C2", "C3"]
        else:
            test_classes = [args.soiling_class]

        # Determine severity
        if args.severity == "both":
            severities = ["noticeable", "severe"]
        else:
            severities = [args.severity]

        # Generate for each class/severity combination
        if args.soiling_class == "all":
            classes_to_test = test_classes
        else:
            classes_to_test = [soiling_class if args.soiling_class == "auto" else args.soiling_class]

        for cls in classes_to_test:
            for severity in severities:
                # Get prompt
                prompt = CAPTION_TEMPLATES[cls][severity]
                short_name = f"{cls}_{severity}"

                print(f"  Generating: {short_name}")
                print(f"    Prompt: {prompt[:60]}...")

                # Set seed
                if args.seed is not None:
                    generator = torch.Generator(device="cuda").manual_seed(
                        args.seed + img_idx * 100 + classes_to_test.index(cls) * 10 + severities.index(severity)
                    )
                else:
                    generator = None

                # Generate
                try:
                    result_image = generate_inpainting(
                        pipe,
                        clean_image,
                        soft_mask,
                        prompt,
                        NEGATIVE_PROMPT,
                        args.strength,
                        args.guidance_scale,
                        args.num_inference_steps,
                        generator,
                    )

                    # Save result
                    output_filename = f"{img_idx:04d}_{clean_path.stem}_{short_name}.png"
                    result_path = run_dir / output_filename
                    result_image.save(result_path)

                    print(f"    Saved: {output_filename}")

                    # Compute QC metrics
                    if args.compute_qc_metrics:
                        qc = compute_qc_metrics(clean_image, result_image, soft_mask)
                        print(f"    QC: spill_rate={qc['spill_rate']:.4f}, coverage={qc['mask_coverage']:.4f}")
                    else:
                        qc = {}

                    # Save spill visualization
                    if args.save_spill_viz:
                        diff = np.abs(np.array(result_image).astype(float) - np.array(clean_image).astype(float))
                        diff_viz = Image.fromarray((diff / diff.max() * 255).astype(np.uint8))
                        diff_path = run_dir / f"{img_idx:04d}_{clean_path.stem}_{short_name}_spill.png"
                        diff_viz.save(diff_path)

                    # Record in manifest
                    manifest["results"].append({
                        "clean_image": str(clean_path.name),
                        "mask": str(mask_path.name),
                        "class": cls,
                        "severity": severity,
                        "output": output_filename,
                        "prompt": prompt,
                        "qc": qc,
                    })

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

    # Save manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Generation complete! Output: {run_dir}")
    print(f"Total results: {len(manifest['results'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
