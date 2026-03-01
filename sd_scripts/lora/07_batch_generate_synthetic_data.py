#!/usr/bin/env python3
"""
Batch Synthetic Soiling Data Generation

使用当前LoRA (checkpoint-3000) 批量生成合成脏污数据。

功能:
- 遍历所有clean帧和mask组合
- 生成带质量指标的合成数据
- 支持断点续传
- 记录详细manifest

Author: SD Experiment Team
Date: 2026-02-18
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import pandas as pd

# ============================================================================
# 配置
# ============================================================================

# Caption模板 (基于原始WoodScape标注)
CAPTION_TEMPLATES = {
    1: {
        "noticeable": "noticeable transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",
        "severe": "severe transparent soiling layer, on camera lens, out of focus foreground, subtle glare, background visible",
    },
    2: {
        "noticeable": "noticeable semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",
        "severe": "severe semi-transparent dirt smudge, on camera lens, out of focus foreground, subtle glare, background visible",
    },
    3: {
        "noticeable": "noticeable opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",
        "severe": "severe opaque heavy stains, on camera lens, out of focus foreground, subtle glare, background visible",
    },
}

NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text, error, corrupted, ugly, camera, lens, sharp focus, crystal clear"

# 默认路径配置
DEFAULT_PATHS = {
    "inpainting_model": "/home/yf/models/sd2_inpaint",
    "lora_base": "/home/yf/soiling_project/sd_scripts/lora/output",
    "clean_frames": "/home/yf/soiling_project/dataset/my_clean_frames",
    "mask_bank": "/home/yf/soiling_project/dataset/mask_bank",
    "output_base": "/home/yf/soiling_project/synthetic_soiling",
}


# ============================================================================
# 工具函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate synthetic soiling data using LoRA+Inpainting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入路径
    parser.add_argument("--clean_frames", type=str, default=DEFAULT_PATHS["clean_frames"],
                       help="Directory containing clean frame images")
    parser.add_argument("--mask_bank", type=str, default=DEFAULT_PATHS["mask_bank"],
                       help="Mask bank base directory")
    parser.add_argument("--mask_dir", type=str, default=None,
                       help="Directory containing masks (default: mask_bank/train/processed)")
    parser.add_argument("--mask_manifest", type=str, default=None,
                       help="Mask manifest CSV (default: mask_bank/train/manifest.csv)")

    # 模型配置
    parser.add_argument("--lora_base", type=str, default=DEFAULT_PATHS["lora_base"],
                       help="Base directory for LoRA checkpoints")
    parser.add_argument("--inpainting_model", type=str, default=DEFAULT_PATHS["inpainting_model"],
                       help="Path to SD2.1 Inpainting model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA checkpoint (default: auto-detect latest)")
    parser.add_argument("--checkpoint_step", type=int, default=3000,
                       help="LoRA checkpoint step")

    # 输出配置
    parser.add_argument("--output_base", type=str, default=DEFAULT_PATHS["output_base"],
                       help="Base output directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: output_base/batch_<timestamp>)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for generation (OOM风险, 建议=1)")

    # 生成参数
    parser.add_argument("--strength", type=float, nargs=2, default=[0.45, 0.50],
                       help="Strength range for random sampling [min, max]")
    parser.add_argument("--guidance_scale", type=float, nargs=2, default=[6.0, 7.0],
                       help="Guidance scale range for random sampling [min, max]")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                       help="Number of denoising steps")
    parser.add_argument("--num_samples_per_pair", type=int, default=1,
                       help="Number of samples per (clean, mask) pair")

    # 数据选择
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to generate (for testing)")
    parser.add_argument("--camera_types", type=str, default="f,lf,l,r,rf",
                       help="Comma-separated camera types to use")
    parser.add_argument("--use_all_masks", action="store_true",
                       help="Use all masks (default: use 3200 train masks)")
    parser.add_argument("--random_masks_per_image", type=int, default=None,
                       help="Number of random masks to use per clean image (e.g., 10 for 8960 images)")

    # 断点续传
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume from specific manifest file")

    # 质量控制
    parser.add_argument("--save_spill_viz", action="store_true",
                       help="Save spill visualization (debug)")
    parser.add_argument("--spill_threshold", type=float, default=0.5,
                       help="Spill rate threshold for warning (0-1)")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    return parser.parse_args()


def get_severity_from_s_full(s_full):
    """根据S_full确定severity token"""
    if s_full < 0.15:
        return "mild"
    elif s_full < 0.35:
        return "moderate"
    elif s_full < 0.60:
        return "noticeable"
    else:
        return "severe"


def load_inpainting_pipeline(inpainting_model_path, lora_path, lora_scale=1.0):
    """加载SD2.1 Inpainting pipeline with LoRA"""
    print(f"\n加载SD2.1 Inpainting: {inpainting_model_path}")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    if lora_path and Path(lora_path).exists():
        print(f"加载LoRA: {lora_path}")
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

            print(f"  找到 {len(unet_state_dict)} LoRA权重")
            pipe.unet.load_state_dict(unet_state_dict, strict=False)
            print("  ✅ LoRA加载成功")
        else:
            print(f"  ⚠️ 未找到: {checkpoint_file}")

    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("  ✅ Pipeline就绪")

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


def compute_spill_rate(original, generated, mask):
    """计算spill_rate: mask外的变化程度"""
    original_np = np.array(original).astype(float)
    generated_np = np.array(generated).astype(float)

    # 将mask转换为binary (0=outside, 1=inside)
    if mask.max() <= 3:  # WoodScape class mask (0,1,2,3)
        mask_binary = (mask > 0).astype(float)
    else:  # 0-255 mask
        mask_binary = (mask > 127).astype(float)

    diff = np.abs(generated_np - original_np)
    diff_sum = diff.sum()

    if diff_sum == 0:
        return 0.0

    mask_inv = 1 - mask_binary
    if mask_inv.ndim == 2:
        mask_inv = np.stack([mask_inv] * 3, axis=-1)

    spill = (diff * mask_inv).sum()
    return spill / diff_sum


def compute_quality_metrics(original, generated, mask, clean_image):
    """计算质量指标"""
    metrics = {}

    # Spill rate
    metrics['spill_rate'] = compute_spill_rate(original, generated, mask)

    # Mask coverage
    mask_bool = (mask > 0)
    metrics['mask_coverage'] = mask_bool.sum() / mask_bool.size

    # 简单亮度统计
    gen_np = np.array(generated)
    metrics['mean_brightness'] = gen_np.mean() / 255.0

    return metrics


# ============================================================================
# 主要生成类
# ============================================================================

class BatchGenerator:
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设置输出目录
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
        else:
            self.output_dir = Path(args.output_base) / f"batch_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # 设置随机种子
        self.rng = np.random.RandomState(args.seed)
        self.torch_seed = args.seed

        # 加载模型
        self.pipe = None
        self.lora_path = self._get_lora_path()

        # 准备数据
        self.clean_images = self._collect_clean_images()
        self.masks = self._collect_masks()

        # Manifest
        self.manifest = []
        self.manifest_path = self.output_dir / "manifest.csv"

        # 断点续传
        if args.resume:
            self._load_resume_state()

    def _get_lora_path(self):
        """自动检测最新的LoRA路径"""
        if self.args.lora_path:
            return self.args.lora_path

        base_output = Path(self.args.lora_base)
        checkpoint_dirs = sorted(base_output.glob("*_lora_v1"), reverse=True)

        if checkpoint_dirs:
            lora_base = checkpoint_dirs[0]
            lora_path = lora_base / f"checkpoint-{self.args.checkpoint_step}"
            if not lora_path.exists():
                lora_path = lora_base / "final_checkpoint"
            print(f"自动检测LoRA路径: {lora_path}")
            return str(lora_path)

        return None

    def _collect_clean_images(self):
        """收集clean帧图像"""
        clean_dir = Path(self.args.clean_frames)
        camera_types = self.args.camera_types.split(",")

        images = []
        for cam_type in camera_types:
            cam_dir = clean_dir / cam_type.strip()
            if cam_dir.exists():
                for img_path in cam_dir.glob("*.jpg"):
                    images.append({
                        "path": img_path,
                        "camera": cam_type.strip(),
                        "file_id": img_path.stem,
                    })

        print(f"\n收集clean帧: {len(images)} 张")
        return images

    def _collect_masks(self):
        """收集mask文件"""
        if self.args.use_all_masks:
            mask_dir = Path(self.args.mask_dir) if self.args.mask_dir else Path(self.args.mask_bank) / "train" / "processed"
            masks = [{"path": p, "file_id": p.stem} for p in mask_dir.glob("*.png")]
        else:
            # 使用manifest
            manifest_path = Path(self.args.mask_manifest) if self.args.mask_manifest else Path(self.args.mask_bank) / "train" / "manifest.csv"
            df = pd.read_csv(manifest_path)

            mask_dir = Path(self.args.mask_bank) / "train" / "processed"

            masks = []
            for _, row in df.iterrows():
                mask_path = mask_dir / f"{row['file_id']}_mask.png"
                if mask_path.exists():
                    masks.append({
                        "path": mask_path,
                        "file_id": row['file_id'],
                        "dominant_class": row.get('dominant_class', 3),
                        "S_full": row.get('S_full', 0.5),
                        "coverage": row.get('coverage', 0.5),
                    })

        print(f"收集mask: {len(masks)} 个")
        return masks

    def _load_resume_state(self):
        """加载断点续传状态"""
        if self.args.resume_from:
            manifest_path = Path(self.args.resume_from)
        else:
            # 查找最新的manifest
            manifests = list(self.output_dir.glob("manifest_*.csv"))
            if manifests:
                manifest_path = max(manifests, key=lambda p: p.stat().st_mtime)
            else:
                print("未找到可恢复的manifest")
                return

        print(f"从manifest恢复: {manifest_path}")
        df = pd.read_csv(manifest_path)

        # 记录已完成的样本
        self.completed_samples = set()
        for _, row in df.iterrows():
            sample_key = f"{row['clean_file_id']}_{row['mask_file_id']}"
            self.completed_samples.add(sample_key)

        print(f"已完成样本: {len(self.completed_samples)}")

        # 重命名旧manifest
        backup_path = manifest_path.with_suffix(".csv.bak")
        manifest_path.rename(backup_path)

    def load_pipeline(self):
        """延迟加载pipeline"""
        if self.pipe is None:
            self.pipe = load_inpainting_pipeline(
                self.args.inpainting_model,
                self.lora_path,
            )

    def generate_single_sample(self, clean_img_path, mask_path, mask_info, sample_idx=0):
        """生成单个样本"""
        # 加载图像
        clean_image = Image.open(clean_img_path).convert("RGB")
        clean_image = clean_image.resize((512, 512))

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((512, 512))
        mask_array = np.array(mask)

        # 软化mask
        soft_mask_array = soft_process_mask(mask_array, blur_radius=3, dilate_iter=1)
        soft_mask = Image.fromarray((soft_mask_array * 255).astype(np.uint8))

        # 确定caption
        dominant_class = mask_info.get('dominant_class', 3)
        S_full = mask_info.get('S_full', 0.5)
        severity = get_severity_from_s_full(S_full)

        if severity in ["mild", "moderate"]:
            severity = "noticeable"  # 使用noticeable作为默认

        caption = CAPTION_TEMPLATES[dominant_class][severity]

        # 随机采样参数
        strength = self.rng.uniform(self.args.strength[0], self.args.strength[1])
        guidance_scale = self.rng.uniform(self.args.guidance_scale[0], self.args.guidance_scale[1])

        # 设置随机种子
        seed = self.torch_seed + hash(f"{clean_img_path}_{mask_path}_{sample_idx}") % 1000000
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # 生成
        with torch.no_grad():
            result = self.pipe(
                prompt=caption,
                image=clean_image,
                mask_image=soft_mask,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        result_image = result.images[0]

        # 计算质量指标
        metrics = compute_quality_metrics(clean_image, result_image, mask_array, clean_image)

        # 保存
        clean_file_id = Path(clean_img_path).stem
        mask_file_id = Path(mask_path).stem.replace("_mask", "")

        output_filename = f"{clean_file_id}_{mask_file_id}_s{sample_idx:03d}.png"
        output_path = self.images_dir / output_filename
        result_image.save(output_path)

        return {
            "output_filename": output_filename,
            "caption": caption,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "seed": seed,
            **metrics,
        }

    def run(self):
        """执行批量生成"""
        self.load_pipeline()

        # 生成任务列表
        tasks = []

        # 随机mask采样模式
        if self.args.random_masks_per_image:
            print(f"\n随机mask采样模式: 每张干净图像使用 {self.args.random_masks_per_image} 个随机mask")
            for clean_img in self.clean_images:
                # 为每张干净图像随机选择mask
                selected_masks = self.rng.choice(
                    self.masks,
                    size=min(self.args.random_masks_per_image, len(self.masks)),
                    replace=False
                )
                for mask in selected_masks:
                    for sample_idx in range(self.args.num_samples_per_pair):
                        sample_key = f"{clean_img['file_id']}_{mask['file_id']}_{sample_idx}"

                        # 检查是否已完成
                        if hasattr(self, 'completed_samples') and sample_key in self.completed_samples:
                            continue

                        tasks.append({
                            "clean_path": clean_img['path'],
                            "clean_file_id": clean_img['file_id'],
                            "clean_camera": clean_img['camera'],
                            "mask_path": mask['path'],
                            "mask_file_id": mask['file_id'],
                            "mask_info": mask,
                            "sample_idx": sample_idx,
                        })
        else:
            # 原有的全量组合模式
            for clean_img in self.clean_images:
                for mask in self.masks:
                    for sample_idx in range(self.args.num_samples_per_pair):
                        sample_key = f"{clean_img['file_id']}_{mask['file_id']}_{sample_idx}"

                        # 检查是否已完成
                        if hasattr(self, 'completed_samples') and sample_key in self.completed_samples:
                            continue

                        tasks.append({
                            "clean_path": clean_img['path'],
                            "clean_file_id": clean_img['file_id'],
                            "clean_camera": clean_img['camera'],
                            "mask_path": mask['path'],
                            "mask_file_id": mask['file_id'],
                            "mask_info": mask,
                            "sample_idx": sample_idx,
                        })

        # 限制样本数
        if self.args.max_samples:
            tasks = tasks[:self.args.max_samples]

        total_tasks = len(tasks)
        print(f"\n总任务数: {total_tasks}")
        print(f"预计输出: {total_tasks} 张图像")

        if total_tasks == 0:
            print("没有任务需要执行")
            return

        # 执行生成
        print("\n开始生成...")
        print("-" * 70)

        for task in tqdm(tasks, desc="生成进度"):
            try:
                result = self.generate_single_sample(
                    task['clean_path'],
                    task['mask_path'],
                    task['mask_info'],
                    task['sample_idx'],
                )

                # 记录到manifest
                manifest_entry = {
                    "clean_file_id": task['clean_file_id'],
                    "clean_camera": task['clean_camera'],
                    "clean_path": str(task['clean_path']),
                    "mask_file_id": task['mask_file_id'],
                    "mask_path": str(task['mask_path']),
                    "mask_dominant_class": task['mask_info'].get('dominant_class', 3),
                    "mask_S_full": task['mask_info'].get('S_full', 0.5),
                    "mask_coverage": task['mask_info'].get('coverage', 0.5),
                    "sample_idx": task['sample_idx'],
                    **result
                }

                self.manifest.append(manifest_entry)

                # 定期保存manifest
                if len(self.manifest) % 100 == 0:
                    self._save_manifest(temp=True)

                # 警告高spill_rate
                if result['spill_rate'] > self.args.spill_threshold:
                    tqdm.write(f"⚠️ 高spill_rate: {result['output_filename']} ({result['spill_rate']:.3f})")

            except Exception as e:
                tqdm.write(f"❌ 错误: {task['clean_file_id']} + {task['mask_file_id']}: {e}")
                continue

        # 保存最终manifest
        self._save_manifest(temp=False)
        self._print_summary()

    def _save_manifest(self, temp=False):
        """保存manifest"""
        df = pd.DataFrame(self.manifest)

        if temp:
            temp_path = self.output_dir / f"manifest_temp_{len(self.manifest)}.csv"
            df.to_csv(temp_path, index=False)
        else:
            # 添加时间戳
            final_path = self.output_dir / f"manifest_{self.timestamp}.csv"
            df.to_csv(final_path, index=False)
            # 也保存一个无时间戳的版本
            df.to_csv(self.manifest_path, index=False)

    def _print_summary(self):
        """打印生成摘要"""
        df = pd.DataFrame(self.manifest)

        print("\n" + "=" * 70)
        print("生成完成!")
        print("=" * 70)
        print(f"总样本数: {len(df)}")
        print(f"输出目录: {self.output_dir}")
        print(f"\nSpill Rate统计:")
        print(f"  平均: {df['spill_rate'].mean():.4f}")
        print(f"  中位数: {df['spill_rate'].median():.4f}")
        print(f"  最大: {df['spill_rate'].max():.4f}")
        print(f"  >0.5: {(df['spill_rate']>0.5).sum()} ({(df['spill_rate']>0.5).sum()/len(df)*100:.1f}%)")
        print(f"\nMask Coverage统计:")
        print(f"  平均: {df['mask_coverage'].mean():.2%}")
        print(f"\n类别分布:")
        print(df['mask_dominant_class'].value_counts().sort_index())


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("批量生成合成脏污数据")
    print("=" * 70)
    print(f"Clean帧: {args.clean_frames}")
    print(f"Mask目录: {args.mask_dir or args.mask_bank + '/train/processed'}")
    print(f"LoRA: {args.lora_path or 'auto-detect'}")
    print(f"输出: {args.output_dir or f'{args.output_base}/batch_<timestamp>'}")
    print(f"Strength范围: {args.strength}")
    print(f"Guidance范围: {args.guidance_scale}")
    print("=" * 70)

    generator = BatchGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
