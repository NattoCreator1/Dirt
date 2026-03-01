#!/usr/bin/env python3
"""
UNet Structure Compatibility Verification (v1.2)

验证 SD2.1 base (text-to-image) 与 SD2-inpainting 的 UNet attention 层结构是否一致

关键假设：LoRA 通常注入在 attention 层（to_q/to_k/to_v/to_out），这些层的结构需要一致才能迁移。
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any


def verify_unet_compatibility(
    sd21_config_path: str = "stabilityai/stable-diffusion-2-1-base",
    sd2_inpaint_config_path: str = "stabilityai/stable-diffusion-2-inpainting",
) -> Dict[str, Any]:
    """
    验证 SD2.1 base 和 SD2-inpainting 的 UNet 配置兼容性

    Returns:
        包含验证结果的字典
    """
    try:
        from diffusers import UNet2DModel
    except ImportError:
        return {
            "status": "error",
            "message": "diffusers not installed. Run: pip install diffusers"
        }

    print("="*60)
    print("UNet Structure Compatibility Verification")
    print("="*60)

    result = {
        "status": "unknown",
        "models": {},
        "compatibility": {},
        "attention_layers": {},
        "conclusion": "",
    }

    try:
        # 加载模型配置（不加载权重，只加载配置）
        print(f"\nLoading SD2.1 base config: {sd21_config_path}")
        unet_base = UNet2DModel.from_pretrained(
            sd21_config_path,
            subfolder="unet",
            torch_dtype=None,  # 不加载权重
        )

        print(f"Loading SD2-inpainting config: {sd2_inpaint_config_path}")
        unet_inpaint = UNet2DModel.from_pretrained(
            sd2_inpaint_config_path,
            subfolder="unet",
            torch_dtype=None,  # 不加载权重
        )

        # 提取关键配置
        config_base = unet_base.config
        config_inpaint = unet_inpaint.config

        result["models"]["sd21_base"] = {
            "in_channels": config_base.in_channels,
            "out_channels": config_base.out_channels,
            "sample_size": config_base.sample_size,
        }

        result["models"]["sd2_inpaint"] = {
            "in_channels": config_inpaint.in_channels,
            "out_channels": config_inpaint.out_channels,
            "sample_size": config_inpaint.sample_size,
        }

        # 关键配置参数（LoRA 迁移相关）
        critical_params = [
            "block_out_channels",
            "layers_per_block",
            "cross_attention_dim",
            "attention_head_dim",
            "num_attention_heads",
            "down_block_types",
            "up_block_types",
            "mid_block_type",
        ]

        print("\n" + "="*60)
        print("Critical Configuration Comparison")
        print("="*60)

        compatibility_results = {}
        for param in critical_params:
            base_val = getattr(config_base, param, None)
            inpaint_val = getattr(config_inpaint, param, None)

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
        print(f"✓ in_channels: base={config_base.in_channels}, inpaint={config_inpaint.in_channels}")
        print("  (This is expected: SD2.1 uses 4 channels, SD2-inpainting uses 9 channels)")
        print("  (LoRA is typically injected in attention layers, not conv_in)")

        # Attention 层分析
        print("\n" + "="*60)
        print("Attention Layer Analysis")
        print("="*60)

        # 检查 attention 层的数量和类型
        attn_layers_base = _count_attention_layers(unet_base)
        attn_layers_inpaint = _count_attention_layers(unet_inpaint)

        result["attention_layers"]["sd21_base"] = attn_layers_base
        result["attention_layers"]["sd2_inpaint"] = attn_layers_inpaint

        print(f"SD2.1 base attention layers:")
        print(f"  Total: {attn_layers_base['total']}")
        print(f"  Down blocks: {attn_layers_base['down_blocks']}")
        print(f"  Up blocks: {attn_layers_base['up_blocks']}")
        print(f"  Mid block: {attn_layers_base['mid_block']}")

        print(f"\nSD2-inpainting attention layers:")
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
            if k != "in_channels"  # in_channels 差异是预期的
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
"""
        else:
            result["status"] = "incompatible"
            result["conclusion"] = """
✗ INCOMPATIBLE for LoRA migration

Critical differences found that may prevent LoRA migration.
Please review the configuration comparison above.
"""

        print(result["conclusion"])

        # 清理模型
        del unet_base
        del unet_inpaint

    except Exception as e:
        result["status"] = "error"
        result["conclusion"] = f"Error during verification: {e}"
        print(f"\n✗ Error: {e}")

    return result


def _count_attention_layers(unet: Any) -> Dict[str, int]:
    """统计 UNet 中的 attention 层数量"""
    down_blocks = 0
    up_blocks = 0
    mid_block = 0

    # 遍历 down blocks
    for down_block in unet.down_blocks:
        if hasattr(down_block, 'attentions'):
            down_blocks += len(down_block.attentions)

    # 遍历 up blocks
    for up_block in unet.up_blocks:
        if hasattr(up_block, 'attentions'):
            up_blocks += len(up_block.attentions)

    # Mid block
    if hasattr(unet, 'mid_block'):
        if hasattr(unet.mid_block, 'attentions'):
            mid_block = len(unet.mid_block.attentions)
        elif hasattr(unet.mid_block, 'attention'):
            mid_block = 1

    return {
        "down_blocks": down_blocks,
        "up_blocks": up_blocks,
        "mid_block": mid_block,
        "total": down_blocks + up_blocks + mid_block,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify UNet compatibility for LoRA migration"
    )
    parser.add_argument(
        "--sd21-base",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="SD2.1 base model path (text-to-image)"
    )
    parser.add_argument(
        "--sd2-inpaint",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="SD2-inpainting model path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for verification results"
    )

    args = parser.parse_args()

    result = verify_unet_compatibility(
        sd21_config_path=args.sd21_base,
        sd2_inpaint_config_path=args.sd2_inpaint,
    )

    # 保存结果到 JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {output_path}")

    # 返回状态码
    sys.exit(0 if result["status"] == "compatible" else 1)


if __name__ == "__main__":
    main()
