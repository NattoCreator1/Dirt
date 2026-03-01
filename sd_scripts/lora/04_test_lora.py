#!/usr/bin/env python3
"""
Test Script for Trained LoRA Model

Generate samples with the trained lens soiling LoRA adapter.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from PIL import Image


# Test prompts based on training caption system
TEST_PROMPTS = [
    # C1 - Transparent soiling
    "noticeable transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",
    "severe transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",

    # C2 - Semi-transparent soiling
    "noticeable semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",
    "severe semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",

    # C3 - Opaque heavy stains
    "noticeable opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",
    "severe opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",

    # Negative prompt (clean lens for comparison)
    "clean camera lens, sharp focus, clear image, high quality",

    # Mixed prompts
    "transparent soiling layer with semi-transparent dirt smudge, on camera lens, out of focus foreground",
    "street scene through dirty camera lens with noticeable transparent soiling layer",
    "city view with severe opaque heavy stains on camera lens, out of focus foreground",
]

# Negative prompt to guide generation
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text, error, corrupted, ugly"


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained LoRA model")

    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/yf/models/sd2_1_base",
        help="Path to base SD2.1 model",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA adapter weight scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to generate at once",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts (overrides default test prompts)",
    )

    return parser.parse_args()


def load_pipeline(base_model_path, lora_path, lora_scale=1.0):
    """Load the SD pipeline with LoRA adapter"""
    print(f"Loading base model from: {base_model_path}")

    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Use DPM++ solver for better quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA from: {lora_path}")

        # Check if this is a checkpoint directory with model.safetensors
        checkpoint_file = Path(lora_path) / "model.safetensors"

        if checkpoint_file.exists():
            # Load the full UNet checkpoint (with LoRA weights)
            print("Loading full UNet checkpoint with LoRA weights...")
            state_dict = load_file(checkpoint_file)

            # Remove "base_model.model." prefix from keys
            unet_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("base_model.model."):
                    new_key = k.replace("base_model.model.", "")
                    unet_state_dict[new_key] = v

            print(f"Found {len(unet_state_dict)} UNet keys")

            # Load state dict with strict=False to allow partial loading
            missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)

            if missing:
                print(f"Warning: Missing {len(missing)} keys in checkpoint")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys in checkpoint")

            print(f"Loaded UNet successfully")
        else:
            # Try standard LoRA loading
            try:
                pipe.load_lora_weights(lora_path)
                print(f"LoRA loaded using standard method")
            except Exception as e:
                print(f"Warning: Could not load LoRA weights: {e}")
                print("Using base model only")
    else:
        print("Warning: LoRA path not found, using base model only")

    # Move to GPU
    pipe = pipe.to("cuda")

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    return pipe


def generate_images(pipe, prompts, output_dir, args):
    """Generate images with the pipeline"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("Generating Images")
    print("=" * 60)

    all_images = []
    all_prompts = []

    for sample_idx in range(args.num_samples):
        for prompt_idx, prompt in enumerate(prompts):
            # Set seed for reproducibility
            if args.seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(args.seed + prompt_idx + sample_idx * 1000)
            else:
                generator = None

            print(f"\n[{sample_idx + 1}/{args.num_samples}] Generating prompt {prompt_idx + 1}/{len(prompts)}")
            print(f"Prompt: {prompt[:80]}...")

            try:
                # Generate image
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    num_images_per_prompt=args.batch_size,
                )

                images = result.images

                # Save images
                for img_idx, image in enumerate(images):
                    # Create filename
                    safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_prompt = safe_prompt.replace(' ', '_')[:50]

                    filename = f"{timestamp}_s{sample_idx+1}_p{prompt_idx+1}_{safe_prompt}.png"
                    filepath = output_path / filename

                    image.save(filepath)
                    print(f"  Saved: {filepath.name}")

                    all_images.append(image)
                    all_prompts.append(prompt)

            except Exception as e:
                print(f"  Error generating image: {e}")
                continue

    # Generate comparison grid
    if len(all_images) > 1:
        print("\nGenerating comparison grid...")
        generate_grid(all_images, all_prompts, output_path / f"{timestamp}_grid.png")

    print(f"\n{'=' * 60}")
    print(f"Generated {len(all_images)} images")
    print(f"Output directory: {output_path}")
    print(f"{'=' * 60}")


def generate_grid(images, prompts, output_path, cols=4):
    """Generate a grid comparison image"""
    from math import ceil

    n_images = len(images)
    rows = ceil(n_images / cols)

    # Assuming all images are same size
    img_width, img_height = images[0].size

    grid_width = img_width * cols
    grid_height = img_height * rows

    # Create grid image
    grid = Image.new("RGB", (grid_width, grid_height), color="white")

    for idx, (img, prompt) in enumerate(zip(images, prompts)):
        row = idx // cols
        col = idx % cols

        # Paste image
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    grid.save(output_path)
    print(f"Grid saved: {output_path}")


def main():
    args = parse_args()

    # Determine LoRA path
    if args.lora_path is None:
        # Try to find the latest training output
        base_output = Path("/home/yf/soiling_project/sd_scripts/lora/output")
        if base_output.exists():
            # Find latest checkpoint directory
            checkpoint_dirs = sorted(base_output.glob("*_lora_v1"), reverse=True)
            if checkpoint_dirs:
                lora_base = checkpoint_dirs[0]
                lora_path = lora_base / f"checkpoint-{args.checkpoint_step}"
                if not lora_path.exists():
                    lora_path = lora_base / "final_checkpoint"
                args.lora_path = str(lora_path)
                print(f"Auto-detected LoRA path: {args.lora_path}")

    # Load prompts
    if args.prompts is None:
        prompts = TEST_PROMPTS
    else:
        prompts = args.prompts

    print("=" * 60)
    print("LoRA Test Script")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"LoRA scale: {args.lora_scale}")
    print(f"Output dir: {args.output_dir}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Image size: {args.width}x{args.height}")
    print("=" * 60)

    # Load pipeline
    pipe = load_pipeline(args.base_model, args.lora_path, args.lora_scale)

    # Generate images
    generate_images(pipe, prompts, args.output_dir, args)


if __name__ == "__main__":
    main()
