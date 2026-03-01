#!/usr/bin/env python3
"""
Create synthetic masks for testing LoRA+Inpainting

Generates different types of soiling masks for testing:
- C1: Transparent soiling (light, scattered pattern)
- C2: Semi-transparent smudge (medium, irregular pattern)
- C3: Opaque stains (heavy, dense pattern)
"""

import numpy as np
from PIL import Image
from pathlib import Path


def create_c1_mask(size=512):
    """Transparent soiling mask - light scattered pattern"""
    mask = np.zeros((size, size), dtype=np.uint8)

    # Scattered small spots
    for _ in range(30):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(10, 30)

        # Create soft circle
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < radius:
                    mask[i, j] = max(mask[i, j], int(80 * (1 - dist/radius)))

    return mask


def create_c2_mask(size=512):
    """Semi-transparent smudge mask - irregular pattern"""
    mask = np.zeros((size, size), dtype=np.uint8)

    # Irregular smudge patches
    for _ in range(5):
        center_x = np.random.randint(50, size - 50)
        center_y = np.random.randint(50, size - 50)

        # Create irregular blob
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < 80:
                    noise = np.random.randn() * 0.3
                    value = int(150 * max(0, (1 - dist/80) + noise))
                    mask[i, j] = max(mask[i, j], min(255, value))

    return mask


def create_c3_mask(size=512):
    """Opaque heavy stains mask - dense heavy pattern"""
    mask = np.zeros((size, size), dtype=np.uint8)

    # Dense heavy patches
    for _ in range(3):
        center_x = np.random.randint(100, size - 100)
        center_y = np.random.randint(100, size - 100)

        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < 60:
                    mask[i, j] = max(mask[i, j], 255)

    return mask


def main():
    output_dir = Path("/home/yf/soiling_project/sd_scripts/lora/test_data/masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate masks for each test image
    for i in range(1, 4):
        # C1 mask
        c1_mask = create_c1_mask()
        c1_img = Image.fromarray(c1_mask, mode='L')
        c1_img.save(output_dir / f"test_{i:03d}_C1.png")

        # C2 mask
        c2_mask = create_c2_mask()
        c2_img = Image.fromarray(c2_mask, mode='L')
        c2_img.save(output_dir / f"test_{i:03d}_C2.png")

        # C3 mask
        c3_mask = create_c3_mask()
        c3_img = Image.fromarray(c3_mask, mode='L')
        c3_img.save(output_dir / f"test_{i:03d}_C3.png")

    print(f"Generated masks in {output_dir}")
    print("  - C1: Transparent soiling (light scattered)")
    print("  - C2: Semi-transparent smudge (irregular)")
    print("  - C3: Opaque heavy stains (dense)")


if __name__ == "__main__":
    main()
