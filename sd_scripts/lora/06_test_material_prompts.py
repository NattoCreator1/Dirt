#!/usr/bin/env python3
"""
Material-Based Prompt Test for LoRA+Inpainting

使用具体物质描述替代抽象类别描述，测试SD是否能生成不同类型的脏污效果。

物质锚定:
- C1 (transparent): 水渍、雨滴、指纹
- C2 (semi-transparent): 油渍、雾状污渍
- C3 (opaque): 泥巴、尘土、厚重污垢
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


# ============================================================================
# 物质锚定Prompt系统
# ============================================================================

# 物质锚定Prompt - 替代抽象的类别描述
MATERIAL_PROMPTS = {
    "C1_water_drops": "water droplets on camera lens, clear liquid marks, rain drops, wet spots, on lens surface",
    "C1_fingerprints": "fingerprints on camera lens, oily finger marks, smudge marks, on glass surface",
    "C1_light_dirt": "light dust on camera lens, fine dust particles, light dirt coating, on lens",

    "C2_grease": "greasy smudges on camera lens, oily film, light haze, foggy appearance",
    "C2_oily_haze": "oily haze on lens, semi-transparent film, light dirt layer, cloudy appearance",

    "C3_mud": "mud splatters on camera lens, thick dirt buildup, earth stains, heavy soil",
    "C3_dust_buildup": "thick dust buildup on camera lens, dirt accumulation, heavy grime, opaque stains",
    "C3_dirt": "heavy dirt stains on camera lens, opaque dirt patches, thick grime layer",
}

# 光学短语 - 所有prompt通用
OPTICAL_PHRASE = "out of focus foreground, subtle glare, camera lens photography"

# Negative prompt
NEGATIVE_PROMPT = "text, watermark, signature, clean lens, crystal clear, high definition, sharp focus, artificial"


def parse_args():
    parser = argparse.ArgumentParser(description="Material-based LoRA+Inpainting Test")

    # Model paths
    parser.add_argument("--inpainting_model", type=str,
                       default="/home/yf/models/sd2_inpaint")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=3000)

    # Input/Output
    parser.add_argument("--clean_image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_material_output")

    # Generation settings
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)

    # Test configuration
    parser.add_argument("--material_types", type=str, default="all",
                       help="Comma-separated list or 'all'")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_inpainting_pipeline(inpainting_model_path, lora_path, lora_scale=1.0):
    """加载SD2.1 Inpainting pipeline with LoRA"""
    print(f"Loading SD2.1 Inpainting from: {inpainting_model_path}")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA from: {lora_path}")
        checkpoint_file = Path(lora_path) / "model.safetensors"

        if checkpoint_file.exists():
            state_dict = load_file(checkpoint_file)
            unet_state_dict = {}
            skipped_keys = []

            for k, v in state_dict.items():
                if k.startswith("base_model.model."):
                    new_key = k.replace("base_model.model.", "")
                    if "conv_in" in new_key or "conv_out" in new_key:
                        skipped_keys.append(new_key)
                        continue
                    if "lora_" in new_key:
                        unet_state_dict[new_key] = v

            print(f"Found {len(unet_state_dict)} LoRA keys")
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} non-LoRA keys")

            missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
            print("LoRA loaded successfully!")

    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    return pipe


def soft_process_mask(mask_array, blur_radius=3, dilate_iter=1):
    """软化mask边缘"""
    from scipy.ndimage import gaussian_filter, binary_dilation

    binary_mask = (mask_array > 0).astype(np.uint8)

    if dilate_iter > 0:
        binary_mask = binary_dilation(binary_mask, iterations=dilate_iter)

    if blur_radius > 0:
        soft_mask = gaussian_filter(binary_mask.astype(float), sigma=blur_radius)
    else:
        soft_mask = binary_mask.astype(float)

    return soft_mask


def main():
    args = parse_args()

    # 确定LoRA路径
    if args.lora_path is None:
        base_output = Path("/home/yf/soiling_project/sd_scripts/lora/output")
        checkpoint_dirs = sorted(base_output.glob("*_lora_v1"), reverse=True)
        if checkpoint_dirs:
            lora_base = checkpoint_dirs[0]
            lora_path = lora_base / f"checkpoint-{args.checkpoint_step}"
            if not lora_path.exists():
                lora_path = lora_base / "final_checkpoint"
            args.lora_path = str(lora_path)

    print("=" * 70)
    print("Material-Based Prompt Test for LoRA+Inpainting")
    print("=" * 70)
    print(f"Inpainting model: {args.inpainting_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Clean images: {args.clean_image_dir}")
    print(f"Masks: {args.mask_dir}")
    print(f"Strength: {args.strength}, Guidance: {args.guidance_scale}, Steps: {args.num_inference_steps}")
    print("=" * 70)

    # 加载pipeline
    pipe = load_inpainting_pipeline(
        args.inpainting_model,
        args.lora_path,
    )

    # 确定测试的material类型
    if args.material_types == "all":
        test_materials = list(MATERIAL_PROMPTS.keys())
    else:
        test_materials = args.material_types.split(",")

    print(f"\nTesting {len(test_materials)} material types:")
    for m in test_materials:
        print(f"  - {m}: {MATERIAL_PROMPTS[m][:50]}...")

    # 扫描clean图像和mask
    clean_dir = Path(args.clean_image_dir)
    mask_dir = Path(args.mask_dir)

    clean_images = sorted(list(clean_dir.glob("*.jpg")) + list(clean_dir.glob("*.png")))

    if not clean_images:
        print("Error: No clean images found!")
        return

    print(f"\nFound {len(clean_images)} clean images")

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / timestamp
    run_dir.mkdir(exist_ok=True)

    # 结果manifest
    manifest = {
        "config": vars(args),
        "lora_path": args.lora_path,
        "timestamp": timestamp,
        "material_prompts": MATERIAL_PROMPTS,
        "results": [],
    }

    # 对每个material类型生成
    for img_idx, clean_path in enumerate(clean_images):
        print(f"\n[{img_idx + 1}/{len(clean_images)}] Processing: {clean_path.name}")

        # 加载clean图像
        clean_image = Image.open(clean_path).convert("RGB")
        clean_image = clean_image.resize((512, 512))

        # 找对应mask
        mask_path = mask_dir / f"{clean_path.stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{clean_path.stem.replace('img', 'mask')}.png"

        if not mask_path.exists():
            print(f"  Warning: No mask found for {clean_path.name}")
            continue

        # 加载mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512))
        mask_array = np.array(mask)

        # 获取mask的dominant class
        unique_vals = np.unique(mask_array)
        mask_classes = [v for v in unique_vals if v > 0]

        print(f"  Mask classes: {mask_classes}")

        # 对每个material类型生成
        for material_key in test_materials:
            # 构建完整prompt
            material_prompt = MATERIAL_PROMPTS[material_key]
            full_prompt = f"{material_prompt}, {OPTICAL_PHRASE}"

            short_name = f"{img_idx:04d}_{clean_path.stem}_{material_key}"
            print(f"  Generating: {material_key}")
            print(f"    Prompt: {full_prompt[:80]}...")

            # 软化mask
            soft_mask_array = soft_process_mask(mask_array, blur_radius=3, dilate_iter=1)
            soft_mask = Image.fromarray((soft_mask_array * 255).astype(np.uint8))

            # 设置随机种子
            generator = torch.Generator(device="cuda").manual_seed(
                args.seed + img_idx * 100 + test_materials.index(material_key)
            )

            try:
                # 生成
                with torch.no_grad():
                    result = pipe(
                        prompt=full_prompt,
                        image=clean_image,
                        mask_image=soft_mask,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        strength=args.strength,
                        generator=generator,
                    )

                result_image = result.images[0]

                # 保存结果
                output_filename = f"{short_name}.png"
                result_path = run_dir / output_filename
                result_image.save(result_path)

                print(f"    Saved: {output_filename}")

                # 计算简单的质量指标
                result_np = np.array(result_image)
                clean_np = np.array(clean_image)

                # Spill rate (mask外区域的差异)
                mask_bool = (mask_array > 0)
                diff = np.abs(result_np.astype(float) - clean_np.astype(float))
                total_diff = diff.sum()

                if total_diff > 0:
                    mask_inv = ~mask_bool
                    if mask_inv.ndim == 2:
                        mask_inv = np.stack([mask_inv] * 3, axis=-1)
                    spill = (diff * mask_inv).sum()
                    spill_rate = spill / total_diff
                else:
                    spill_rate = 0.0

                # 记录
                manifest["results"].append({
                    "clean_image": str(clean_path.name),
                    "mask": str(mask_path.name),
                    "material_type": material_key,
                    "prompt": full_prompt,
                    "output": output_filename,
                    "spill_rate": float(spill_rate),
                    "mask_coverage": float(mask_bool.sum() / mask_bool.size),
                })

            except Exception as e:
                print(f"    Error: {e}")
                continue

    # 保存manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Generation complete! Output: {run_dir}")
    print(f"Total results: {len(manifest['results'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
