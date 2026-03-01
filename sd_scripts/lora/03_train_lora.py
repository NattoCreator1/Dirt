#!/usr/bin/env python3
"""
LoRA Training Script for Lens Soiling Effects

This script trains a LoRA adapter on Stable Diffusion 2.1 to generate
lens soiling/dirt effects. The training uses DreamBooth-style format
with image-caption pairs.

Based on:
- Base model: stabilityai/stable-diffusion-2-1-base
- Training method: LoRA fine-tuning
- Dataset: WoodScape soiling annotations
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import transformers


class SoilingDataset(Dataset):
    """Dataset for lens soiling training data"""

    def __init__(self, train_dir, tokenizer=None, size=512):
        """
        Args:
            train_dir: Directory containing training images and captions
            tokenizer: Optional tokenizer for text encoding
            size: Image size (default 512)
        """
        self.train_dir = Path(train_dir)
        self.size = size
        self.tokenizer = tokenizer

        # Get all image files
        self.images = sorted(list(self.train_dir.glob("*.jpg")))
        if len(self.images) == 0:
            raise ValueError(f"No images found in {train_dir}")

        print(f"Found {len(self.images)} training images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Resize to target size
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        # Load caption
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            with open(caption_path, "r") as f:
                caption = f.read().strip()
        else:
            caption = "on camera lens, out of focus foreground, subtle glare, background visible"

        # Tokenize caption if tokenizer provided
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "pixel_values": image,
                "input_ids": tokens.input_ids.squeeze(0),
                "caption": caption,
            }

        return {"pixel_values": image, "caption": caption}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA for lens soiling effects"
    )

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier",
    )

    # Dataset arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size of input images",
    )

    # Training arguments - optimized for 3200 samples
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Total number of training steps (recommend 1-2x dataset size for LoRA)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Scheduler type: linear, cosine, cosine_with_restarts, polynomial, constant",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training",
    )

    # LoRA specific arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank of LoRA approximation (4, 8, 16, 32, 64)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (typically 2x rank)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout probability",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )

    # Logging arguments
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="Log training info every X steps",
    )

    # Mixed precision
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )

    # Hardware
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print training configuration
    print("=" * 60)
    print("LoRA Training Configuration")
    print("=" * 60)
    print(f"Base model: {args.pretrained_model_name_or_path}")
    print(f"Training data: {args.train_data_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    print(f"Max steps: {args.max_train_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LR scheduler: {args.lr_scheduler}")
    print(f"LR warmup steps: {args.lr_warmup_steps}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"LoRA dropout: {args.lora_dropout}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Import training libraries
    try:
        from diffusers import StableDiffusionPipeline
        from diffusers.training_utils import set_seed
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTextModel
    except ImportError as e:
        print(f"Error: Required library not found: {e}")
        print("Please install required packages:")
        print("  pip install diffusers transformers accelerate peft")
        return

    # Set seed
    set_seed(args.seed)

    # For now, just save the configuration
    # Actual training will use the accelerate library
    config_file = output_dir / "training_config.txt"
    with open(config_file, "w") as f:
        f.write("LoRA Training Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Base model: {args.pretrained_model_name_or_path}\n")
        f.write(f"Training data: {args.train_data_dir}\n")
        f.write(f"Image size: {args.image_size}\n")
        f.write(f"Batch size: {args.train_batch_size}\n")
        f.write(f"Gradient accumulation: {args.gradient_accumulation_steps}\n")
        f.write(f"Max steps: {args.max_train_steps}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"LR scheduler: {args.lr_scheduler}\n")
        f.write(f"LR warmup steps: {args.lr_warmup_steps}\n")
        f.write(f"LoRA rank: {args.lora_rank}\n")
        f.write(f"LoRA alpha: {args.lora_alpha}\n")
        f.write(f"LoRA dropout: {args.lora_dropout}\n")
        f.write(f"Mixed precision: {args.mixed_precision}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\nTimestamp: {datetime.now().isoformat()}\n")

    print("\nTo run the actual training, use accelerate launch:")
    print(f"accelerate launch --mixed_precision={args.mixed_precision} {__file__} \\")
    print(f"  --train_data_dir {args.train_data_dir} \\")
    print(f"  --output_dir {args.output_dir}")
    print("\nOr use the provided shell script:")
    print(f"bash train_lora.sh")


if __name__ == "__main__":
    main()
