#!/usr/bin/env python3
"""
数据增强模块 (Data Augmentation)

为脏污检测模型提供在线数据增强，包括：
1. 相机域随机化 (Camera Domain Randomization)
2. Photometric变换 (不改变几何，保持mask/标签一致性)

Author: Claude Code Assistant
Date: 2026-02-24
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import torch


class CameraDomainRandomization:
    """
    相机域随机化

    在线数据增强，模拟不同相机/ISP的成像差异。
    仅包含photometric变换，不改变几何结构，确保mask/标签一致性。

    设计原则：
    - 中等强度，避免过度随机化导致视觉-标签不一致
    - 高概率应用，确保每个epoch都见到足够的多样性
    - 仅用于训练，不用于验证/测试
    """

    def __init__(
        self,
        # 总开关
        enable: bool = True,

        # 各类变换概率 (推荐高概率，中等强度)
        color_prob: float = 0.8,       # 颜色/曝光随机化概率
        noise_prob: float = 0.7,       # 噪声添加概率
        blur_prob: float = 0.4,        # 模糊添加概率 (较低，避免过度模糊)
        compression_prob: float = 0.5, # 压缩伪影概率
        lens_prob: float = 0.4,        # 镜头效果概率
        resolution_prob: float = 0.3, # 分辨率随机化概率 (较低，避免质量损失)

        # 随机种子 (用于可重复性)
        seed: Optional[int] = None,
    ):
        """
        初始化相机域随机化

        Args:
            enable: 是否启用增强 (用于验证/测试时关闭)
            color_prob: 颜色/曝光随机化概率
            noise_prob: 噪声添加概率
            blur_prob: 模糊添加概率
            compression_prob: 压缩伪影概率
            lens_prob: 镜头效果概率
            resolution_prob: 分辨率随机化概率
            seed: 随机种子
        """
        self.enable = enable
        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.compression_prob = compression_prob
        self.lens_prob = lens_prob
        self.resolution_prob = resolution_prob

        if seed is not None:
            np.random.seed(seed)

    def _adjust_gamma(self, image: np.ndarray) -> np.ndarray:
        """Gamma校正 (模拟不同ISP的gamma曲线)"""
        # 中等强度: 0.9-1.1 (避免过度调整)
        gamma = np.random.uniform(0.9, 1.1)
        image = image.astype(np.float32) / 255.0
        image = np.power(image, gamma) * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def _adjust_color_temperature(self, image: np.ndarray) -> np.ndarray:
        """色温偏移 (模拟白平衡差异)"""
        # 中等强度: ±0.03
        temp_shift = np.random.uniform(-0.03, 0.03)
        image = image.astype(np.float32) / 255.0

        if temp_shift > 0:
            # 暖色调
            image[:, :, 0] = np.clip(image[:, :, 0] + temp_shift * 0.2, 0, 1)  # R
            image[:, :, 1] = np.clip(image[:, :, 1] - temp_shift * 0.05, 0, 1)  # G
            image[:, :, 2] = np.clip(image[:, :, 2] + temp_shift * 0.15, 0, 1)  # B
        else:
            # 冷色调
            shift = abs(temp_shift)
            image[:, :, 0] = np.clip(image[:, :, 0] - shift * 0.15, 0, 1)  # R
            image[:, :, 1] = np.clip(image[:, :, 1] + shift * 0.05, 0, 1)  # G
            image[:, :, 2] = np.clip(image[:, :, 2] + shift * 0.2, 0, 1)  # B

        return (image * 255).astype(np.uint8)

    def _adjust_exposure(self, image: np.ndarray) -> np.ndarray:
        """曝光调整 (模拟不同曝光条件)"""
        # 中等强度: 0.85-1.15
        exposure = np.random.uniform(0.85, 1.15)
        image = image.astype(np.float32)
        image = np.clip(image * exposure, 0, 255)
        return image.astype(np.uint8)

    def randomize_color_and_exposure(self, image: np.ndarray) -> np.ndarray:
        """颜色和曝光随机化"""
        # 1. Gamma
        image = self._adjust_gamma(image)

        # 2. 色温
        image = self._adjust_color_temperature(image)

        # 3. 整体曝光
        image = self._adjust_exposure(image)

        return image

    def add_realistic_noise(self, image: np.ndarray) -> np.ndarray:
        """
        添加真实相机噪声

        模拟不同传感器和ISO条件下的噪声
        """
        image = image.astype(np.float32)

        # 1. Shot noise (泊松-高斯混合)
        # 中等强度: ISO 1.0-2.0
        iso_gain = np.random.uniform(1.0, 2.0)

        # Shot noise
        scaled = image * iso_gain / 255.0 * 10
        shot_noise = np.random.poisson(np.clip(scaled, 0, 100)) / 10.0 * 255.0 / iso_gain - image

        # Read noise (高斯噪声)
        read_noise = np.random.normal(0, 3.0, image.shape)  # 降低到3.0

        noise_image = image + shot_noise + read_noise

        # 2. 色噪 (轻微)
        if np.random.random() < 0.4:  # 降低概率
            color_noise = np.random.normal(0, 2.0, image.shape)
            color_noise = cv2.GaussianBlur(color_noise, (3, 3), 0)
            noise_image = noise_image + color_noise * 0.3

        return np.clip(noise_image, 0, 255).astype(np.uint8)

    def add_random_blur(self, image: np.ndarray) -> np.ndarray:
        """
        添加随机模糊

        模拟不同对焦条件和运动模糊
        """
        blur_type = np.random.choice(['defocus', 'motion', 'none'],
                                      p=[0.3, 0.2, 0.5])  # 降低概率

        if blur_type == 'defocus':
            # 散焦模糊 (中等强度)
            kernel_size = np.random.choice([3, 5])
            sigma = np.random.uniform(0.5, 1.5)
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        elif blur_type == 'motion':
            # 运动模糊 (轻微)
            kernel_size = np.random.randint(5, 11)
            angle = np.random.uniform(0, 180)

            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0
            kernel = cv2.warpAffine(
                kernel,
                cv2.getRotationMatrix2D(
                    (kernel_size // 2, kernel_size // 2),
                    angle, 1.0
                ),
                (kernel_size, kernel_size)
            )
            kernel = kernel / kernel.sum()

            blurred = cv2.filter2D(image, -1, kernel)
        else:
            blurred = image

        return blurred

    def add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        添加压缩伪影

        模拟JPEG压缩和编码链损失
        """
        # 高质量范围: 75-95
        quality = np.random.randint(75, 95)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image.astype(np.uint8), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # 色度子采样 (降低概率)
        if np.random.random() < 0.2:
            h, w = image.shape[:2]
            small = cv2.resize(decoded, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            decoded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        return decoded

    def add_lens_effects(self, image: np.ndarray) -> np.ndarray:
        """
        添加镜头效果

        模拟光学镜头的各种效应
        """
        image = image.astype(np.float32) / 255.0
        h, w = image.shape[:2]

        # 1. Vignetting (暗角)
        if np.random.random() < 0.5:
            Y, X = np.ogrid[:h, :w]
            center_x, center_y = w / 2, h / 2
            # 轻微暗角: 0.05-0.25
            vignette_strength = np.random.uniform(0.05, 0.25)

            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)

            vignette = 1 - vignette_strength * (distance / max_distance)**2
            vignette = np.clip(vignette, 0.3, 1.0)  # 最低保持30%

            image = image * vignette[..., np.newaxis]

        return np.clip(image * 255, 0, 255).astype(np.uint8)

    def random_resolution_chain(self, image: np.ndarray) -> np.ndarray:
        """
        随机分辨率处理链

        模拟不同的ISP处理链
        """
        h, w = image.shape[:2]

        # 随机下采样倍数
        downscale_factor = np.random.choice([2, 4], p=[0.7, 0.3])  # 2倍更常见

        # 下采样
        small = cv2.resize(
            image,
            (w // downscale_factor, h // downscale_factor),
            interpolation=cv2.INTER_AREA
        )

        # 随机上采样方法
        up_methods = [
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
        ]
        up_method = np.random.choice(up_methods, p=[0.7, 0.3])

        # 上采样
        restored = cv2.resize(small, (w, h), interpolation=up_method)

        return restored

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        对图像应用随机相机域变换

        Args:
            image: RGB图像, HxWx3, uint8, [0, 255]

        Returns:
            变换后的图像 (如果enable=False，返回原图)
        """
        if not self.enable:
            return image

        result = image.copy()

        # 1. 颜色/曝光
        if np.random.random() < self.color_prob:
            result = self.randomize_color_and_exposure(result)

        # 2. 噪声
        if np.random.random() < self.noise_prob:
            result = self.add_realistic_noise(result)

        # 3. 模糊
        if np.random.random() < self.blur_prob:
            result = self.add_random_blur(result)

        # 4. 压缩伪影
        if np.random.random() < self.compression_prob:
            result = self.add_compression_artifacts(result)

        # 5. 镜头效果
        if np.random.random() < self.lens_prob:
            result = self.add_lens_effects(result)

        # 6. 分辨率链
        if np.random.random() < self.resolution_prob:
            result = self.random_resolution_chain(result)

        return result


class Compose:
    """
    组合多个变换

    便于扩展其他类型的增强
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image


# 测试代码
if __name__ == "__main__":
    import os

    # 创建测试输出目录
    os.makedirs("baseline/debug_augmentation", exist_ok=True)

    # 创建随机化器
    randomizer = CameraDomainRandomization(
        enable=True,
        color_prob=0.8,
        noise_prob=0.7,
        blur_prob=0.4,
        compression_prob=0.5,
        lens_prob=0.4,
        resolution_prob=0.3,
        seed=42,
    )

    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 应用增强
    augmented = randomizer(test_image)

    print(f"Original shape: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Augmented shape: {augmented.shape}, dtype: {augmented.dtype}")
    print(f"Augmented range: [{augmented.min()}, {augmented.max()}]")

    # 保存测试结果
    cv2.imwrite("baseline/debug_augmentation/test_original.png",
                cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("baseline/debug_augmentation/test_augmented.png",
                cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print("Test images saved to baseline/debug_augmentation/")
