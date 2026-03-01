#!/usr/bin/env python3
"""
UNet Structure Compatibility Verification (v1.2) - Offline Mode

基于已知配置信息验证 SD2.1 base 与 SD2-inpainting 的 UNet 兼容性

参考配置来源:
- SD2.1 base: stabilityai/stable-diffusion-2-1-base
- SD2-inpainting: stabilityai/stable-diffusion-2-inpainting
"""

import json
from pathlib import Path
from typing import Dict, Any


# SD2.1 base (text-to-image) 的已知 UNet 配置
SD21_BASE_UNET_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "act_fn": "silu",
    "attention_head_dim": [5, 10, 20, 20],
    "block_out_channels": [320, 640, 1280, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 1024,
    "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,  # SD2.1 base 使用 4 通道（噪声 latent）
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    "use_linear_projection": True
}


def verify_unet_compatibility_offline(
    sd2_inpaint_config_path: str = "/home/yf/models/sd2_inpaint/unet/config.json",
) -> Dict[str, Any]:
    """
    基于已知配置验证 UNet 兼容性

    Returns:
        包含验证结果的字典
    """
    print("="*60)
    print("UNet Structure Compatibility Verification (Offline Mode)")
    print("="*60)

    result = {
        "status": "unknown",
        "models": {},
        "compatibility": {},
        "attention_layers": {},
        "conclusion": "",
    }

    # SD2.1 base 配置
    config_base = SD21_BASE_UNET_CONFIG
    result["models"]["sd21_base"] = {
        "in_channels": config_base["in_channels"],
        "out_channels": config_base["out_channels"],
        "sample_size": config_base["sample_size"],
    }

    # 加载 SD2-inpainting 配置
    try:
        with open(sd2_inpaint_config_path, 'r') as f:
            config_inpaint = json.load(f)

        result["models"]["sd2_inpaint"] = {
            "in_channels": config_inpaint["in_channels"],
            "out_channels": config_inpaint["out_channels"],
            "sample_size": config_inpaint["sample_size"],
        }
    except FileNotFoundError:
        print(f"✗ Error: SD2-inpainting config not found at {sd2_inpaint_config_path}")
        result["status"] = "error"
        result["conclusion"] = f"SD2-inpainting config not found at {sd2_inpaint_config_path}"
        return result
    except Exception as e:
        print(f"✗ Error loading SD2-inpainting config: {e}")
        result["status"] = "error"
        result["conclusion"] = f"Error loading config: {e}"
        return result

    # 关键配置参数（LoRA 迁移相关）
    critical_params = [
        "block_out_channels",
        "layers_per_block",
        "cross_attention_dim",
        "attention_head_dim",
        "down_block_types",
        "up_block_types",
        "use_linear_projection",
        "norm_num_groups",
    ]

    print("\n" + "="*60)
    print("Critical Configuration Comparison")
    print("="*60)

    compatibility_results = {}
    for param in critical_params:
        base_val = config_base.get(param)
        inpaint_val = config_inpaint.get(param)

        is_compatible = base_val == inpaint_val
        compatibility_results[param] = {
            "sd21_base": base_val,
            "sd2_inpaint": inpaint_val,
            "compatible": is_compatible,
        }

        status = "✓" if is_compatible else "✗"
        print(f"{status} {param:30s} | base: {base_val} | inpaint: {inpaint_val}")

    result["compatibility"] = compatibility_results

    # 预期差异：in_channels
    print("\n" + "="*60)
    print("Expected Differences")
    print("="*60)
    print(f"✓ in_channels: base={config_base['in_channels']}, inpaint={config_inpaint['in_channels']}")
    print("  (Expected: SD2.1 uses 4 channels, SD2-inpainting uses 9 channels)")
    print("  (LoRA is typically injected in attention layers, not conv_in)")

    # Attention 层分析
    print("\n" + "="*60)
    print("Attention Layer Analysis")
    print("="*60)

    # 计算 attention 层数量
    attn_layers_base = _estimate_attention_layers(config_base)
    attn_layers_inpaint = _estimate_attention_layers(config_inpaint)

    result["attention_layers"]["sd21_base"] = attn_layers_base
    result["attention_layers"]["sd2_inpaint"] = attn_layers_inpaint

    print(f"SD2.1 base attention layers (estimated):")
    print(f"  Total: {attn_layers_base['total']}")
    print(f"  Down blocks: {attn_layers_base['down_blocks']}")
    print(f"  Up blocks: {attn_layers_base['up_blocks']}")
    print(f"  Mid block: {attn_layers_base['mid_block']}")

    print(f"\nSD2-inpainting attention layers (estimated):")
    print(f"  Total: {attn_layers_inpaint['total']}")
    print(f"  Down blocks: {attn_layers_inpaint['down_blocks']}")
    print(f"  Up blocks: {attn_layers_inpaint['up_blocks']}")
    print(f"  Mid block: {attn_layers_inpaint['mid_block']}")

    # 结论
    print("\n" + "="*60)
    print("Conclusion")
    print("="*60)

    # 检查关键参数是否一致
    all_critical_compatible = all(
        v["compatible"] for k, v in compatibility_results.items()
    )

    # 检查 attention 层数量是否一致
    attn_compatible = (
        attn_layers_base["total"] == attn_layers_inpaint["total"] and
        attn_layers_base["down_blocks"] == attn_layers_inpaint["down_blocks"] and
        attn_layers_base["up_blocks"] == attn_layers_inpaint["up_blocks"]
    )

    if all_critical_compatible and attn_compatible:
        result["status"] = "compatible"
        result["conclusion"] = """
✓ COMPATIBLE for LoRA migration

Key findings:
1. All critical attention parameters match
2. Attention layer structure is identical
3. LoRA injected in attention layers should migrate successfully

Recommended approach:
- Train LoRA on SD2.1 base (text-to-image)
- Load LoRA weights on SD2-inpainting UNet at inference time
- LoRA will affect attention layers in both models identically

Technical details:
- Both models use identical attention_head_dim: [5, 10, 20, 20]
- Both models have same block structure: 3 CrossAttnDownBlock + 1 DownBlock
- LoRA typically targets: to_q, to_k, to_v, to_out in attention layers
- These layers have identical structure in both models
"""
    else:
        result["status"] = "incompatible"
        result["conclusion"] = """
✗ INCOMPATIBLE for LoRA migration

Critical differences found that may prevent LoRA migration.
Please review the configuration comparison above.
"""

    print(result["conclusion"])

    return result


def _estimate_attention_layers(config: Dict) -> Dict[str, int]:
    """
    估算 UNet 中的 attention 层数量

    基于 block_types 和 layers_per_block 估算
    """
    down_blocks = 0
    up_blocks = 0
    mid_block = 1  # SD2 系列都有 mid_block attention

    # Down blocks: CrossAttnDownBlock 有 attention，DownBlock 没有
    for block_type in config.get("down_block_types", []):
        if "CrossAttn" in block_type:
            # 每个 CrossAttnDownBlock 有 layers_per_block 个 attention 层
            down_blocks += config.get("layers_per_block", 2)

    # Up blocks: CrossAttnUpBlock 有 attention，UpBlock 没有
    for block_type in config.get("up_block_types", []):
        if "CrossAttn" in block_type:
            # 每个 CrossAttnUpBlock 有 layers_per_block 个 attention 层
            up_blocks += config.get("layers_per_block", 2)

    return {
        "down_blocks": down_blocks,
        "up_blocks": up_blocks,
        "mid_block": mid_block,
        "total": down_blocks + up_blocks + mid_block,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify UNet compatibility for LoRA migration (offline mode)"
    )
    parser.add_argument(
        "--sd2-inpaint-config",
        type=str,
        default="/home/yf/models/sd2_inpaint/unet/config.json",
        help="Path to SD2-inpainting UNet config.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for verification results"
    )

    args = parser.parse_args()

    result = verify_unet_compatibility_offline(
        sd2_inpaint_config_path=args.sd2_inpaint_config,
    )

    # 保存结果到 JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {output_path}")

    return result


if __name__ == "__main__":
    import sys
    result = main()
    sys.exit(0 if result["status"] == "compatible" else 1)
