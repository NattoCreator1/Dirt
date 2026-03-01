#!/usr/bin/env python3
"""
LoRA Training Script using Diffusers Library

This script trains a LoRA adapter on Stable Diffusion 2.1 using
the diffusers library with accelerate for multi-GPU support.

Based on: https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


# Will error if the minimal version of diffusers is not installed
check_min_version("0.25.0")

logger = get_logger(__name__)


class SoilingDataset(Dataset):
    """
    Dataset for lens soiling training data.
    Loads images and captions from DreamBooth format directory.
    """

    def __init__(
        self,
        data_root: str,
        tokenizer: CLIPTokenizer,
        size: int = 512,
        center_crop: bool = False,
    ):
        """Initialize dataset

        Args:
            data_root: Directory containing {file_id}.jpg and {file_id}.txt
            tokenizer: CLIP tokenizer for captions
            size: Image size (default 512)
            center_crop: Whether to center crop images
        """
        self.data_root = Path(data_root)
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # Find all images
        self.images = list(self.data_root.glob("*.jpg"))
        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_root}")

        self.images = sorted(self.images)
        logger.info(f"Found {len(self.images)} training images")

        # Image transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        # Load caption
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = "on camera lens, out of focus foreground"

        # Tokenize caption
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0) if "attention_mask" in inputs else None,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for lens soiling effects")

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/yf/models/sd2_1_base",
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model",
    )

    # Dataset arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size of input images",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images",
    )

    # Training arguments
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Total number of training steps",
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
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for LR scheduler",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="Output directory",
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
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="Log every X steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training",
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    # Setup accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Log configuration
    logger.info("=" * 60)
    logger.info("LoRA Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.pretrained_model_name_or_path}")
    logger.info(f"Training data: {args.train_data_dir}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Batch size: {args.train_batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Max steps: {args.max_train_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info("=" * 60)

    # Load models
    logger.info("Loading models...")

    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Setup LoRA for UNet
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v"],  # Apply to attention layers
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Create dataset
    logger.info("Creating dataset...")
    dataset = SoilingDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.image_size,
        center_crop=args.center_crop,
    )

    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Prepare for training
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Create learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_warmup_steps = args.lr_warmup_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if exists
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.max_train_steps // num_update_steps_per_epoch + 1):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()

                # Log metrics
                if global_step % args.log_steps == 0:
                    avg_loss = train_loss / args.log_steps
                    accelerator.log({"train_loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    train_loss = 0.0

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(Path(args.output_dir) / "final_checkpoint")
        logger.info(f"Saved final checkpoint to {args.output_dir}/final_checkpoint")

    accelerator.end_training()


if __name__ == "__main__":
    import math
    main()
