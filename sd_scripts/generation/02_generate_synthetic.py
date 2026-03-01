#!/usr/bin/env python3
"""
阶段3：SD Inpainting 合成脏污生成器

使用 Stable Diffusion Inpainting 模型生成合成脏污数据。

依赖:
    pip install diffusers transformers accelerate safetensors

使用:
    python sd_scripts/generation/02_generate_synthetic.py --num_samples 100
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline


# ============================================================================
# 配置
# ============================================================================

PROJECT_ROOT = Path("/home/yf/soiling_project")
DATASET_ROOT = PROJECT_ROOT / "dataset"

# 数据路径
# Clean 底图使用最终规格目录（640x480，已过中心裁剪+resize 的几何口径）
CLEAN_FRAMES_DIR = DATASET_ROOT / "my_clean_frames_4by3"
MASK_BANK_TRAIN = DATASET_ROOT / "mask_bank" / "train"
MASK_BANK_VAL = DATASET_ROOT / "mask_bank" / "val"

# 输出路径
OUTPUT_DIR = DATASET_ROOT / "synthetic_soiling" / "v1.0_wgap_alpha50"
OUTPUT_NPZ_DIR = OUTPUT_DIR / "npz"
OUTPUT_MANIFEST_DIR = OUTPUT_DIR / "manifests"

# 目标分辨率
TARGET_RESOLUTION = (640, 480)

# 质量控制阈值
BACKGROUND_SSIM_THRESHOLD = 0.95
BACKGROUND_DIFF_THRESHOLD = 0.05

# 类别相关的 mask 内变化幅度阈值（避免系统性过滤 transparent 类轻度样本）
# transparent 类脏污的合理变化幅度可能显著低于 opaque 类
MASK_DIFF_MIN_THRESHOLDS = {
    1: 0.02,  # transparent: 允许较小变化
    2: 0.05,  # semi_transparent: 中等变化
    3: 0.10,  # opaque: 需要明显变化
}
MASK_DIFF_MIN_THRESHOLD_DEFAULT = 0.05  # 默认阈值（用于无法判断类别的情况）

# 边界带伪影检测参数
BOUNDARY_BAND_WIDTH = 5  # 边界带宽度（像素）
BOUNDARY_GRADIENT_THRESHOLD = 30.0  # 边界带梯度异常阈值

# Mask 覆盖率上限（避免全屏 inpainting 导致背景一致性失效）
MASK_COVERAGE_MAX = 0.8  # 最大覆盖率 80%

# Prompt 模板模式配置
USE_EXPANDED_PROMPTS = True  # True=扩展模板(30-50条/类别), False=简单模板(2-3条/类别)
                              # 注意: 扩展模板为辅助增强，q_D+q_T 筛选比 prompt 扩展更重要

# 类别采样比例（用于控制合成数据的类别分布）
# 按预设比例采样 target_class，而非被动依赖 MaskBank 的类别比例
CLASS_PROPORTIONS = {
    1: 0.33,  # transparent
    2: 0.34,  # semi_transparent
    3: 0.33,  # opaque
}
# 确保总和为 1
_total = sum(CLASS_PROPORTIONS.values())
CLASS_PROPORTIONS = {k: v / _total for k, v in CLASS_PROPORTIONS.items()}

# 加载 Severity 配置
SEVERITY_CONFIG_PATH = PROJECT_ROOT / "severity_config.json"


# ============================================================================
# 配置加载
# ============================================================================

def load_severity_config(config_path: Path = SEVERITY_CONFIG_PATH) -> Dict:
    """
    加载 Severity 配置

    Returns:
        配置字典
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"✓ 加载 Severity 配置: {config.get('version', 'unknown')}")
    print(f"  class_weights: {config.get('class_weights', {})}")
    print(f"  fusion_coeffs: {config.get('fusion_coeffs', {})}")

    return config


def load_rebinned_thresholds(severity_version: str = "unknown") -> Dict:
    """
    加载训练集自适应分箱阈值（严格版本匹配模式）

    阈值文件按 Severity 版本命名绑定，严格策略：
    - 必须加载版本特定文件：rebinned_thresholds__{version}.json
    - 若版本特定文件不存在，L 将被标记为不可用
    - 避免使用错误阈值导致的 L 语义漂移

    Args:
        severity_version: Severity 配置版本号，如 "v1.0_wgap_alpha50"

    Returns:
        {"b0": float, "b1": float, "b2": float, "version": str, "source": str}
        source ∈ {"version_specific", "missing_versioned", "default"}
    """
    meta_dir = DATASET_ROOT / "woodscape_processed" / "meta"

    # 尝试加载版本特定文件
    version_specific_path = meta_dir / f"rebinned_thresholds__{severity_version}.json"

    if version_specific_path.exists():
        with open(version_specific_path, "r") as f:
            thresholds = json.load(f)

        thresholds_version = thresholds.get("version", "unknown")
        thresholds["source"] = "version_specific"

        print(f"✓ 加载分箱阈值 (版本特定 {severity_version}): b0={thresholds['b0']:.4f}, b1={thresholds['b1']:.4f}, b2={thresholds['b2']:.4f}")

        if severity_version != "unknown" and thresholds_version != "unknown":
            if severity_version != thresholds_version:
                print(f"⚠️ 警告: Severity 版本 ({severity_version}) 与阈值文件版本 ({thresholds_version}) 不匹配！")

        return thresholds
    else:
        # 版本特定文件不存在，L 标记为不可用
        print(f"⚠️ 未找到版本特定阈值文件 {version_specific_path.name}")
        print(f"⚠️ L 将被标记为不可用（thresholds_source=missing_versioned），后续请勿使用 L 进行分层采样")

        # 返回默认值但标记为缺失版本
        return {
            "b0": 0.01,
            "b1": 0.10,
            "b2": 0.20,
            "version": "default",
            "source": "missing_versioned"
        }


def compute_L(
    s: float,
    S: float,
    thresholds: Dict[str, float],
) -> int:
    """
    计算等级 L

    使用训练集自适应分箱阈值：
    - 当面积强度 s <= b0 时，L=0（clean）
    - 其余样本根据 S 分位数阈值分到 L=1,2,3

    Args:
        s: 面积强度 s_simple = 1 - clean_ratio [0, 1]
        S: Severity Score S_full [0, 1]
        thresholds: {"b0": float, "b1": float, "b2": float}

    Returns:
        L ∈ {0, 1, 2, 3}
    """
    b0, b1, b2 = thresholds["b0"], thresholds["b1"], thresholds["b2"]

    # 首先用 s（面积强度）判断是否为 clean
    if s <= b0:
        return 0
    # 其余样本用 S（Severity Score）分箱
    elif S <= b1:
        return 1
    elif S <= b2:
        return 2
    else:
        return 3


# ============================================================================
# 提示词模板（扩展版）
# ============================================================================
# 设计说明:
# - 每个类别 10-20 条基础模板，支持随机组合
# - 保持类别与 prompt 的一一对应
# - 注意: 后续 q_D+q_T 筛选比 prompt 扩展更重要，此为辅助增强

# 基础模板组件
_PROMPT_PREFIXES = [
    "",  # 无前缀
    "slight",
    "visible",
    "noticeable",
    "scattered",
]

_PROMPT_DIRT_TYPES = {
    "transparent": [
        "light dirt smudges",
        "transparent dirt spots",
        "faint dirt marks",
        "light smudges",
        "thin dirt layer",
        "slight grime",
        "light dust spots",
        "faint smudges",
        "transparent stains",
        "light dirt film",
        "subtle dirt marks",
        "faint residue",
        "light greasy smudges",
        "thin dirt coating",
        "slight dirt traces",
    ],
    "semi_transparent": [
        "semi-transparent dirt",
        "foggy smudges",
        "hazy dirt spots",
        "misty dirt layer",
        "cloudy smudges",
        "semi-opaque dirt",
        "grime spots",
        "dirty film",
        "smeared dirt",
        "blurry dirt marks",
        "hazy dirt coating",
        "foggy residue",
        "semi-clear smudges",
        "dirt haze",
        "cloudy dirt film",
    ],
    "opaque": [
        "heavy dirt stains",
        "opaque dirt patches",
        "mud and dirt",
        "thick dirt layer",
        "heavy smudges",
        "encrusted dirt",
        "dirt buildup",
        "solid dirt spots",
        "muddy stains",
        "heavy grime",
        "thick dirt coating",
        "opaque smudges",
        "caked dirt",
        "heavy residue",
        "dirt clusters",
    ],
}

_PROMPT_LOCATIONS = [
    "on camera lens",
    "on lens surface",
    "on camera glass",
    "covering lens",
    "on front glass",
    "across camera lens",
]

_PROMPT_SUFFIXES = [
    "",  # 无后缀
    "slightly obscuring view",
    "partially blocking view",
    "visible on camera",
    "on lens surface",
]


def _build_prompt_templates() -> dict:
    """
    构建扩展的 prompt 模板

    通过组合生成多样化 prompt，每个类别约 30-50 条变体
    """
    templates = {
        "generic": [],  # 保留 generic 作为后备
        "transparent": [],
        "semi_transparent": [],
        "opaque": [],
    }

    # 对每个类别生成组合 prompt
    for class_name in ["transparent", "semi_transparent", "opaque"]:
        dirt_types = _PROMPT_DIRT_TYPES[class_name]

        for dirt in dirt_types:
            # 基础组合: dirt + location
            for loc in _PROMPT_LOCATIONS:
                templates[class_name].append(f"{dirt} {loc}")

            # 带前缀的组合: prefix + dirt + location
            for prefix in _PROMPT_PREFIXES:
                if prefix:  # 跳过空前缀
                    for loc in _PROMPT_LOCATIONS[:3]:  # 限制 location 数量
                        templates[class_name].append(f"{prefix} {dirt} {loc}")

            # 带后缀的组合: dirt + location + suffix
            for suffix in _PROMPT_SUFFIXES:
                if suffix:  # 跳过空后缀
                    templates[class_name].append(f"{dirt} {_PROMPT_LOCATIONS[0]}, {suffix}")

    # generic 模板（后备，不区分类别）
    templates["generic"] = [
        "dirt smudges on camera lens",
        "camera lens with dirt and dust",
        "dirty camera lens with smudges",
        "foggy camera lens with dirt spots",
        "lens dirt and grime",
        "camera lens soiling",
        "dirt on lens surface",
    ]

    return templates


# 生成最终模板
PROMPT_TEMPLATES = _build_prompt_templates()


def select_prompt_for_class(target_class: int) -> str:
    """
    根据目标类别随机选择 prompt

    Args:
        target_class: 目标类别 (1=transparent, 2=semi_transparent, 3=opaque)

    Returns:
        随机选择的 prompt 字符串
    """
    class_names = {
        1: "transparent",
        2: "semi_transparent",
        3: "opaque",
    }

    class_name = class_names.get(target_class, "semi_transparent")

    # 根据配置选择模板集
    templates = PROMPT_TEMPLATES if USE_EXPANDED_PROMPTS else PROMPT_TEMPLATES_SIMPLE
    prompts = templates.get(class_name, templates["generic"])

    return str(np.random.choice(prompts))


def _print_prompt_stats():
    """打印 prompt 模板统计信息"""
    print("="*60)
    print("Prompt 模板配置")
    print("="*60)
    print(f"模式: {'扩展模板' if USE_EXPANDED_PROMPTS else '简单模板'}")

    templates = PROMPT_TEMPLATES if USE_EXPANDED_PROMPTS else PROMPT_TEMPLATES_SIMPLE

    for class_name, prompts in templates.items():
        print(f"  {class_name}: {len(prompts)} 条模板")

    if USE_EXPANDED_PROMPTS:
        print("\n注意: 扩展模板为辅助增强，q_D+q_T 筛选比 prompt 扩展更重要")
    print("="*60)


# 模块加载时打印统计信息
_print_prompt_stats()


# 保持向后兼容的原始简单模板（用于对比实验）
PROMPT_TEMPLATES_SIMPLE = {
    "generic": [
        "dirt smudges on camera lens",
        "camera lens with dirt and dust",
        "dirty camera lens with smudges",
        "foggy camera lens with dirt spots",
    ],
    "transparent": [
        "light dirt smudges on camera lens",
        "transparent dirt spots on lens",
    ],
    "semi_transparent": [
        "semi-transparent dirt on camera lens",
        "foggy smudges on lens surface",
    ],
    "opaque": [
        "heavy dirt stains on camera lens",
        "opaque dirt patches blocking camera view",
        "mud and dirt covering camera lens",
    ],
}

NEGATIVE_PROMPT = "text, watermark, signature, clean lens, crystal clear, high definition, sharp focus"


# ============================================================================
# Inpainting 生成器
# ============================================================================

class SDInpaintingSoilingGenerator:
    """
    基于 Stable Diffusion Inpainting 的脏污生成器
    """

    def __init__(
        self,
        model_id: str = "/home/yf/models/sd2_inpaint",  # 本地模型路径
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        severity_config_path: Path = SEVERITY_CONFIG_PATH,
    ):
        """
        初始化 SD Inpainting 管道

        Args:
            model_id: Hugging Face 模型 ID 或本地路径
            device: 运行设备
            torch_dtype: 数据类型
            severity_config_path: Severity 配置文件路径
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # 加载 Severity 配置
        self.severity_config = load_severity_config(severity_config_path)
        severity_version = self.severity_config.get("version", "unknown")

        # 加载分箱阈值（版本绑定）
        self.rebinned_thresholds = load_rebinned_thresholds(severity_version)

        # 提取配置参数
        # class_weights 兼容列表和字典两种格式，固定顺序为 (clean, transparent, semi_transparent, opaque)
        cw = self.severity_config["class_weights"]
        if isinstance(cw, dict):
            # 字典格式：{"clean": 0.0, "transparent": 0.15, ...}
            self.class_weights = (
                cw["clean"],
                cw["transparent"],
                cw["semi_transparent"],
                cw["opaque"],
            )
        elif isinstance(cw, (list, tuple)):
            # 列表格式：[0.0, 0.15, 0.50, 1.0]
            if len(cw) != 4:
                raise ValueError(f"class_weights 长度必须为 4，当前为 {len(cw)}")
            self.class_weights = tuple(cw)
        else:
            raise TypeError(f"class_weights 类型错误: {type(cw)}，期望为 dict 或 list")
        self.alpha = self.severity_config["fusion_coeffs"]["alpha"]
        self.beta = self.severity_config["fusion_coeffs"]["beta"]
        self.gamma = self.severity_config["fusion_coeffs"]["gamma"]
        self.eta_trans = self.severity_config["eta_trans"]
        self.sigma = self.severity_config["spatial"]["sigma"]

        print(f"加载 Stable Diffusion Inpainting 模型: {model_id}")
        print(f"设备: {device}, 数据类型: {torch_dtype}")

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # 禁用安全检查以加快速度
        ).to(device)

        # 优化设置
        if device == "cuda":
            # 启用内存优化
            self.pipeline.enable_attention_slicing()

        print("✓ 模型加载完成")

    def generate(
        self,
        clean_image: np.ndarray,
        target_mask: np.ndarray,
        prompt: str,
        target_class: int,
        negative_prompt: str = NEGATIVE_PROMPT,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        生成合成脏污图像（单类别版本）

        设计原则：
        - 每个合成样本只包含一个脏污类别 c ∈ {1,2,3}
        - mask 内所有像素统一为类别 c，与 prompt 一致
        - 标签侧也只含 0 与 c，确保"生成控制量即标签来源"的闭环

        Args:
            clean_image: [H, W, 3] uint8, 干净图像
            target_mask: [H, W] uint8/int32, 原始 mask (0-3)
            prompt: 提示词
            target_class: 目标类别 c ∈ {1, 2, 3}
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            strength: 变换强度 (0-1)
            seed: 随机种子

        Returns:
            成功时返回结果字典，失败返回 None
        """
        H, W = clean_image.shape[:2]

        # 转换为单类别 mask（使用指定的 target_class）
        single_class_mask = self._convert_to_single_class_mask(target_mask, target_class)

        # 准备 PIL 图像
        clean_pil = Image.fromarray(clean_image.astype(np.uint8))

        # 准备 mask 图像（SD 需要 mask 为白色=修复区域）
        mask_binary = (single_class_mask > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_binary)

        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Inpainting 生成
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                image=clean_pil,
                mask_image=mask_pil,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        synthetic_pil = result.images[0]
        synthetic_image = np.array(synthetic_pil)

        # 确保输出图像尺寸与输入一致（SD 模型可能返回不同尺寸）
        if synthetic_image.shape[:2] != clean_image.shape[:2]:
            synthetic_pil_resized = synthetic_pil.resize(
                (clean_image.shape[1], clean_image.shape[0]),
                Image.Resampling.LANCZOS
            )
            synthetic_image = np.array(synthetic_pil_resized)

        # 质量检查（使用单类别 mask）
        quality_result = self._quality_check(
            clean_image, synthetic_image, single_class_mask
        )

        if not quality_result["passed"]:
            return None

        # 计算 tile coverage（从单类别 mask 生成）
        tile_cov = self._compute_tile_coverage(single_class_mask)

        # 计算 Severity Score（使用加载的配置参数）
        severity = compute_severity_from_tile_cov(
            tile_cov,
            class_weights=self.class_weights,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            eta_trans=self.eta_trans,
            sigma=self.sigma,
        )

        # 计算 L (Level) - 训练集自适应分箱
        # L=0 当 s_simple <= b0，其余按 S_full 分箱
        # 仅在阈值版本匹配时计算 L，否则标记为不可用
        thresholds_source = self.rebinned_thresholds.get("source", "unknown")
        if thresholds_source == "version_specific":
            L = compute_L(severity["s_simple"], severity["S_full"], self.rebinned_thresholds)
            L_unavailable_reason = None
        else:
            L = -1  # 标记为不可用
            L_unavailable_reason = f"thresholds_source={thresholds_source}"

        return {
            "synthetic_image": synthetic_image.astype(np.uint8),
            "clean_image": clean_image.astype(np.uint8),
            "target_mask": single_class_mask.astype(np.uint8),  # 单类别 mask
            "target_class": int(target_class),  # 目标类别 c ∈ {1,2,3}
            "tile_cov": tile_cov.astype(np.float32),
            # Severity Score
            "S_op": severity["S_op"],
            "S_sp": severity["S_sp"],
            "S_dom": severity["S_dom"],
            "S_full": severity["S_full"],
            "s_simple": severity["s_simple"],  # 1 - clean_ratio
            "clean_ratio": severity["clean_ratio"],
            # L (Level) - 用于分层采样
            "L": int(L),
            "L_unavailable_reason": L_unavailable_reason,  # L 不可用的原因
            "thresholds_source": thresholds_source,  # 阈值来源
            # 元数据
            "prompt": prompt,
            "seed": seed,
            "quality": quality_result,
            "severity_config_version": self.severity_config.get("version", "unknown"),
        }

    def _quality_check(
        self,
        clean_image: np.ndarray,
        synthetic_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Dict:
        """
        质量检查

        Args:
            clean_image: 原始干净图像 [H, W, 3]
            synthetic_image: 生成图像 [H, W, 3]
            target_mask: 目标 mask [H, W]（单类别 mask）

        Returns:
            质量检查结果
        """
        H, W = clean_image.shape[:2]
        mask_bool = (target_mask > 0)

        # 获取目标类别（用于类别相关阈值）
        unique_values = np.unique(target_mask[target_mask > 0])
        target_class = int(unique_values[0]) if len(unique_values) > 0 else None

        result = {
            "passed": True,
            "background_ssim": 1.0,
            "background_diff": 0.0,
            "mask_diff": 0.0,
            "boundary_artifact_score": 0.0,
        }

        # 背景一致性检查：用 clean 覆盖 mask 区域后计算全图 SSIM
        # 创建 masked 版本的 synthetic 图像（mask 区域替换为 clean）
        synthetic_masked = synthetic_image.copy()
        if mask_bool.any():
            # 一次性替换所有通道，避免链式索引
            synthetic_masked[mask_bool] = clean_image[mask_bool]

        # 计算全图 SSIM
        from skimage.metrics import structural_similarity as ssim

        # 计算每个通道的 SSIM 并取平均
        ssim_values = []
        for c in range(3):
            ssim_c = ssim(
                synthetic_masked[:, :, c],
                clean_image[:, :, c],
                data_range=255
            )
            ssim_values.append(ssim_c)

        background_ssim = np.mean(ssim_values)
        result["background_ssim"] = float(background_ssim)

        if background_ssim < BACKGROUND_SSIM_THRESHOLD:
            result["passed"] = False
            result["fail_reason"] = f"背景 SSIM 过低: {background_ssim:.4f}"

        # 背景像素差异（mask 外区域）
        mask_complement = ~mask_bool
        if mask_complement.sum() > 0:
            background_diff = np.abs(
                synthetic_image[mask_complement].astype(float) -
                clean_image[mask_complement].astype(float)
            ).mean() / 255.0
            result["background_diff"] = float(background_diff)

            if background_diff > BACKGROUND_DIFF_THRESHOLD:
                result["passed"] = False
                if "fail_reason" not in result:
                    result["fail_reason"] = f"背景差异过大: {background_diff:.4f}"

        # mask 内变化幅度检查（类别相关阈值）
        if mask_bool.sum() > 0:
            mask_diff = np.abs(
                synthetic_image[mask_bool].astype(float) -
                clean_image[mask_bool].astype(float)
            ).mean() / 255.0
            result["mask_diff"] = float(mask_diff)

            # 根据目标类别选择阈值
            if target_class is not None and target_class in MASK_DIFF_MIN_THRESHOLDS:
                min_threshold = MASK_DIFF_MIN_THRESHOLDS[target_class]
            else:
                min_threshold = MASK_DIFF_MIN_THRESHOLD_DEFAULT

            if mask_diff < min_threshold:
                result["passed"] = False
                if "fail_reason" not in result:
                    result["fail_reason"] = f"mask 内变化不足: {mask_diff:.4f} < {min_threshold:.4f} (类别={target_class})"

        # 边界带伪影检测
        boundary_artifact_score = self._check_boundary_artifacts(
            clean_image, synthetic_image, target_mask
        )
        result["boundary_artifact_score"] = float(boundary_artifact_score)

        if boundary_artifact_score > BOUNDARY_GRADIENT_THRESHOLD:
            result["passed"] = False
            if "fail_reason" not in result:
                result["fail_reason"] = f"边界带伪影异常: {boundary_artifact_score:.2f} > {BOUNDARY_GRADIENT_THRESHOLD}"

        # Mask 覆盖率上限检查（避免全屏 inpainting 导致背景一致性失效）
        mask_coverage = float(mask_bool.sum() / mask_bool.size)
        result["mask_coverage"] = mask_coverage

        if mask_coverage > MASK_COVERAGE_MAX:
            result["passed"] = False
            if "fail_reason" not in result:
                result["fail_reason"] = f"mask 覆盖率过高: {mask_coverage:.2f} > {MASK_COVERAGE_MAX}"

        return result

    def _check_boundary_artifacts(
        self,
        clean_image: np.ndarray,
        synthetic_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        """
        边界带伪影检测

        检测 mask 边界带的梯度异常，用于识别光晕、破碎、涂抹等常见 inpainting 伪影

        策略：
        1. 提取 mask 边界带（mask 边缘向外扩展 N 像素）
        2. 计算边界带内 synthetic 与 clean 的梯度差异
        3. 返回平均梯度差异（越大说明伪影越严重）

        Args:
            clean_image: 原始干净图像 [H, W, 3]
            synthetic_image: 生成图像 [H, W, 3]
            target_mask: 目标 mask [H, W]

        Returns:
            边界带伪影分数（越大表示伪影越严重）
        """
        mask_bool = (target_mask > 0)

        if not mask_bool.any():
            return 0.0

        # 使用形态学膨胀获取边界带
        from cv2 import dilate, MORPH_RECT, getStructuringElement

        kernel = getStructuringElement(MORPH_RECT, (BOUNDARY_BAND_WIDTH * 2 + 1, BOUNDARY_BAND_WIDTH * 2 + 1))
        mask_dilated = dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
        mask_dilated = mask_dilated.astype(bool)

        # 边界带 = 膨胀区域 - 原始 mask 区域
        boundary_band = mask_dilated & ~mask_bool

        if not boundary_band.any():
            return 0.0

        # 转换为灰度计算梯度
        clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
        synthetic_gray = cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2GRAY)

        # 使用 Sobel 算子计算梯度
        clean_grad_x = cv2.Sobel(clean_gray, cv2.CV_64F, 1, 0, ksize=3)
        clean_grad_y = cv2.Sobel(clean_gray, cv2.CV_64F, 0, 1, ksize=3)
        clean_grad_mag = np.sqrt(clean_grad_x**2 + clean_grad_y**2)

        synthetic_grad_x = cv2.Sobel(synthetic_gray, cv2.CV_64F, 1, 0, ksize=3)
        synthetic_grad_y = cv2.Sobel(synthetic_gray, cv2.CV_64F, 0, 1, ksize=3)
        synthetic_grad_mag = np.sqrt(synthetic_grad_x**2 + synthetic_grad_y**2)

        # 计算边界带内的梯度差异
        grad_diff = np.abs(synthetic_grad_mag[boundary_band] - clean_grad_mag[boundary_band])

        # 返回平均梯度差异
        artifact_score = float(grad_diff.mean())

        return artifact_score

    def _compute_tile_coverage(self, mask: np.ndarray) -> np.ndarray:
        """
        从 mask 计算 tile coverage

        将 640×480 的 mask 划分为 8×8 个 tile，计算每个 tile 的类别覆盖率

        Args:
            mask: [H, W] mask (0-3)

        Returns:
            [8, 8, 4] tile coverage
        """
        H, W = mask.shape
        tile_h, tile_w = H // 8, W // 8  # 60, 80

        tile_cov = np.zeros((8, 8, 4), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                y0, y1 = i * tile_h, (i + 1) * tile_h
                x0, x1 = j * tile_w, (j + 1) * tile_w

                tile = mask[y0:y1, x0:x1]
                total = tile.size

                for c in range(4):
                    tile_cov[i, j, c] = (tile == c).sum() / total

        return tile_cov

    def _convert_to_single_class_mask(
        self,
        mask: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        将多类别 mask 转换为单类别 mask

        转换策略：所有脏污像素 (mask>0) 统一赋值为 target_class

        Args:
            mask: [H, W] 原始 mask (0-3)
            target_class: 目标类别 c ∈ {1, 2, 3}

        Returns:
            single_class_mask: [H, W] 单类别 mask，值为 0 或 target_class
        """
        if target_class not in (1, 2, 3):
            raise ValueError(f"target_class 必须为 1, 2, 3，当前为 {target_class}")

        # 转换为单类别 mask：所有脏污区域统一为目标类别
        single_class_mask = np.where(mask > 0, target_class, 0).astype(np.uint8)

        return single_class_mask


# ============================================================================
# Severity Score 计算（与 baseline scripts 保持一致）
# ============================================================================

def compute_dominance_class_aware(
    tile_cov: np.ndarray,
    class_weights: Tuple = (0.0, 0.33, 0.66, 1.0),
    eta_trans: float = 0.9,
    eps: float = 1e-8,
) -> Tuple[float, float, float, Tuple[int, int]]:
    """
    计算 dominance 分数（与 scripts/03_build_labels_tile_global.py 一致）

    Args:
        tile_cov: [N, M, 4] tile coverage
        class_weights: 类别权重
        eta_trans: transparent 降权因子
        eps: 数值稳定性

    Returns:
        (S_dom, dom_tile_score, trans_ratio, (i, j))
    """
    w = np.array(class_weights, dtype=np.float32)

    # per-tile weighted severity
    s_tile = (tile_cov * w[None, None, :]).sum(axis=-1)

    # locate worst tile
    idx = int(np.argmax(s_tile))
    N, M = s_tile.shape
    i, j = idx // M, idx % M
    dom_tile_score = float(s_tile[i, j])

    # transparent ratio inside soiling area of the dominant tile
    soiling_part = float(tile_cov[i, j, 1] + tile_cov[i, j, 2] + tile_cov[i, j, 3])
    trans_ratio = float(tile_cov[i, j, 1] / (soiling_part + eps))

    # smooth discount: when trans_ratio=1, factor=1-eta_trans
    factor = float(1.0 - eta_trans * trans_ratio)
    factor = float(np.clip(factor, 0.0, 1.0))

    S_dom = float(dom_tile_score * factor)
    return S_dom, dom_tile_score, trans_ratio, (i, j)


def compute_severity_from_tile_cov(
    tile_cov: np.ndarray,
    class_weights: Tuple = (0.0, 0.15, 0.50, 1.0),
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
    eta_trans: float = 0.9,
    sigma: float = 0.5,
) -> Dict:
    """
    从 tile coverage 计算 Severity Score

    与 baseline 中的 SeverityAggregator.forward() 和 scripts/03 保持一致

    Args:
        tile_cov: [8, 8, 4] tile coverage
        class_weights: (w_clean, w_trans, w_semi, w_opaque)
        alpha, beta, gamma: 融合系数
        eta_trans: transparent 降权因子
        sigma: 空间高斯权重标准差

    Returns:
        Severity分数字典
    """
    N, M, C = tile_cov.shape
    H, W = N, M
    w = np.array(class_weights, dtype=np.float32)

    # S_op: Opacity-aware coverage
    p = tile_cov.mean(axis=(0, 1))  # [4]
    S_op = float((w * p).sum())

    # S_sp: Spatial weighted
    # 空间权重公式 W(x,y) = exp(-(x² + y²) / (2σ²))，其中 x,y ∈ [-1, 1]
    # 对应大纲定义的中心偏置高斯权重，与 baseline SeverityAggregator 一致
    xs = np.linspace(-1.0, 1.0, M)  # 列方向坐标
    ys = np.linspace(-1.0, 1.0, N)  # 行方向坐标
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    W = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    W = W / (W.sum() + 1e-12)  # 归一化

    # tile 级别加权严重度
    s_tile = (tile_cov * w[None, None, :]).sum(axis=-1)  # [N, M]
    S_sp = float((W * s_tile).sum())

    # S_dom: Dominance with transparent discount
    S_dom, dom_raw, r_tr, dom_ij = compute_dominance_class_aware(
        tile_cov,
        class_weights=class_weights,
        eta_trans=eta_trans
    )

    # S_full
    S_full = alpha * S_op + beta * S_sp + gamma * S_dom

    # s_simple: 1 - clean_ratio
    clean_ratio = p[0]
    s_simple = 1.0 - clean_ratio

    return {
        'S_op': S_op,
        'S_sp': S_sp,
        'S_dom': S_dom,
        'S_full': float(np.clip(S_full, 0.0, 1.0)),
        's_simple': float(s_simple),
        'clean_ratio': float(clean_ratio),
    }


# ============================================================================
# 数据采样
# ============================================================================

class DataSampler:
    """数据采样器"""

    def __init__(
        self,
        clean_dir: Path,
        mask_manifest_path: Path,
        target_resolution: Tuple[int, int] = (640, 480),
    ):
        self.clean_dir = clean_dir
        self.target_resolution = target_resolution

        # 加载 mask manifest
        self.mask_df = pd.read_csv(mask_manifest_path)
        # 存储 mask bank 目录用于解析相对路径
        self.mask_bank_dir = mask_manifest_path.parent
        print(f"加载 mask manifest: {len(self.mask_df)} masks")
        print(f"Mask bank 目录: {self.mask_bank_dir}")

        # 扫描 clean 图像
        self.clean_images = []
        for direction in clean_dir.iterdir():
            if direction.is_dir():
                images = list(direction.glob("*.jpg")) + list(direction.glob("*.png"))
                self.clean_images.extend(images)
        print(f"扫描 clean 图像: {len(self.clean_images)} 帧")

    def sample_clean_image(self) -> Tuple[np.ndarray, str, str]:
        """
        随机采样一张 clean 图像

        Returns:
            (img, img_path, view_id)
            - img: RGB 图像
            - img_path: 图像完整路径
            - view_id: 视角目录名，如 lf/f/rf
        """
        img_path = np.random.choice(self.clean_images)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整分辨率
        img = cv2.resize(img, self.target_resolution)

        # 提取视角 ID（父目录名）
        view_id = img_path.parent.name

        return img, str(img_path), view_id

    def sample_mask(self, coverage_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        采样一个 mask

        返回的 mask 保证：
        - 分辨率为 TARGET_RESOLUTION
        - 像素值在 {0, 1, 2, 3} 范围内
        """
        # 根据 coverage 范围筛选
        if coverage_range is not None:
            min_cov, max_cov = coverage_range
            filtered = self.mask_df[
                (self.mask_df['coverage'] >= min_cov) &
                (self.mask_df['coverage'] <= max_cov)
            ]
            if len(filtered) == 0:
                filtered = self.mask_df
        else:
            filtered = self.mask_df

        row = filtered.sample(1).iloc[0]

        # 读取 mask（处理相对/绝对路径）
        processed_path = row['processed_path']
        mask_path = Path(processed_path)

        # 如果是相对路径，尝试相对于 mask bank 目录解析
        if not mask_path.is_absolute():
            # 尝试相对于当前工作目录
            if not mask_path.exists():
                # 尝试相对于 mask bank 目录
                mask_path = self.mask_bank_dir / processed_path
                if not mask_path.exists():
                    raise ValueError(f"无法找到 mask 文件: {processed_path}（尝试了相对路径和 mask bank 目录）")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"无法读取 mask: {row['processed_path']}")

        # 校验分辨率
        if mask.shape != TARGET_RESOLUTION[::-1]:
            mask = cv2.resize(mask, TARGET_RESOLUTION, interpolation=cv2.INTER_NEAREST)

        # 校验像素值范围
        unique_values = np.unique(mask)
        valid_values = {0, 1, 2, 3}

        # 如果 mask 是 0/255 格式，需要映射
        if set(unique_values) & {255, 254, 253, 252}:
            # 假设是 0/255 二值格式，映射回 0-3
            # 或者可能是多阈值格式的残留
            # 为安全起见，拒绝这个样本
            raise ValueError(f"Mask 像素值异常: {unique_values}, 期望 {{0,1,2,3}}")

        # 如果唯一值不在 {0,1,2,3} 中，拒绝
        if not set(unique_values).issubset(valid_values):
            raise ValueError(f"Mask 像素值异常: {unique_values}, 期望 {{0,1,2,3}}")

        return {
            'mask': mask,
            'file_id': row['file_id'],
            'coverage': row['coverage'],
            'S_full': row['S_full'],
        }


# ============================================================================
# 批量生成
# ============================================================================

def generate_batch(
    generator: SDInpaintingSoilingGenerator,
    sampler: DataSampler,
    num_samples: int,
    output_npz_dir: Path,
    seed_start: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    strength: float = 0.8,
    max_attempts: int = None,
) -> List[Dict]:
    """
    批量生成合成数据

    以通过样本数为准，设置最大尝试次数避免无限循环

    Args:
        generator: SD Inpainting 生成器
        sampler: 数据采样器
        num_samples: 目标生成样本数（通过质量检查的数量）
        output_npz_dir: 输出目录
        seed_start: 起始随机种子
        num_inference_steps: 推理步数
        guidance_scale: 引导系数
        strength: 变换强度
        max_attempts: 最大尝试次数，默认为 num_samples * 3

    Returns:
        生成的样本信息列表
    """
    output_npz_dir.mkdir(parents=True, exist_ok=True)

    if max_attempts is None:
        max_attempts = num_samples * 3

    generated = []
    attempt = 0
    sample_idx = 0

    pbar = tqdm(total=num_samples, desc="生成合成数据")

    while len(generated) < num_samples and attempt < max_attempts:
        seed = seed_start + attempt
        attempt += 1

        # 采样数据
        clean_image, clean_path, clean_view_id = sampler.sample_clean_image()

        # 采样 mask（带异常处理，避免单个坏样本终止整批生成）
        try:
            mask_data = sampler.sample_mask()
        except (ValueError, FileNotFoundError, IOError) as e:
            print(f"⚠️ Mask 采样失败，跳过: {e}")
            continue

        # 按预设比例采样目标类别（控制类别分布）
        target_class = np.random.choice(
            list(CLASS_PROPORTIONS.keys()),
            p=list(CLASS_PROPORTIONS.values())
        )

        # 根据目标类别选择提示词（使用扩展模板）
        prompt = select_prompt_for_class(target_class)

        # 生成（使用指定的 target_class）
        result = generator.generate(
            clean_image=clean_image,
            target_mask=mask_data['mask'],
            prompt=prompt,
            target_class=target_class,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )

        if result is None:
            continue

        # 计算单类别 mask 的覆盖率 pi
        mask_coverage_pi = float((result['target_mask'] > 0).sum() / result['target_mask'].size)

        # 保存 .npz 文件
        sample_id = f"syn_{sample_idx:06d}"
        sample_idx += 1
        npz_path = output_npz_dir / f"{sample_id}.npz"

        np.savez_compressed(
            npz_path,
            image=result['synthetic_image'],
            mask=result['target_mask'],
            tile_cov=result['tile_cov'],
            s_simple=result['s_simple'],
            S_full=result['S_full'],
            S_op=result['S_op'],
            S_sp=result['S_sp'],
            S_dom=result['S_dom'],
            L=result['L'],
            # 元数据字段（用于后续质量控制和实验追踪）
            target_class=result['target_class'],
            severity_config_version=result['severity_config_version'],
            thresholds_source=result['thresholds_source'],
        )

        # 记录信息
        generated.append({
            'sample_id': sample_id,
            'clean_source': clean_path,
            'clean_view_id': clean_view_id,  # 视角 ID，用于统计不同视角的合成通过率
            'mask_source': mask_data['file_id'],
            'prompt': result['prompt'],
            'seed': result['seed'],
            'target_class': result['target_class'],
            'S_full': result['S_full'],
            'L': result['L'],
            'mask_coverage_pi': mask_coverage_pi,  # 单类别 mask 覆盖率，用于按覆盖率分层采样
            'severity_config_version': result['severity_config_version'],
            'thresholds_source': result['thresholds_source'],  # 阈值来源，missing_versioned 表示 L 不可用
            'background_ssim': result['quality']['background_ssim'],
            'boundary_artifact_score': result['quality']['boundary_artifact_score'],  # 边界带伪影分数
            # 生成参数（用于复现）
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'strength': strength,
        })

        pbar.update(1)
        pbar.set_postfix({"通过": len(generated), "尝试": attempt})

    pbar.close()

    print(f"\n生成统计: 目标={num_samples}, 通过={len(generated)}, 尝试={attempt}")

    return generated


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SD 合成脏污数据生成")
    parser.add_argument("--num_samples", type=int, default=100, help="生成样本数量")
    parser.add_argument("--model", type=str, default="/home/yf/models/sd2_inpaint", help="SD Inpainting 模型路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小（暂不支持）")
    parser.add_argument("--seed", type=int, default=42)
    # 生成参数
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导系数")
    parser.add_argument("--strength", type=float, default=0.8, help="变换强度")
    parser.add_argument("--max_attempts", type=int, default=None, help="最大尝试次数（默认为 num_samples * 3）")

    args = parser.parse_args()

    print("="*60)
    print("SD 合成脏污数据生成 - 阶段3")
    print("="*60)

    # 初始化生成器
    generator = SDInpaintingSoilingGenerator(
        model_id=args.model,
        device=args.device,
    )

    # 初始化采样器
    sampler = DataSampler(
        clean_dir=CLEAN_FRAMES_DIR,
        mask_manifest_path=MASK_BANK_TRAIN / "manifest.csv",
    )

    # 批量生成
    generated = generate_batch(
        generator=generator,
        sampler=sampler,
        num_samples=args.num_samples,
        output_npz_dir=OUTPUT_NPZ_DIR,
        seed_start=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        max_attempts=args.max_attempts,
    )

    # 保存 manifest
    OUTPUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_MANIFEST_DIR / f"manifest_syn_seed{args.seed}.csv"

    manifest_df = pd.DataFrame(generated)
    manifest_df.to_csv(manifest_path, index=False)

    print("\n" + "="*60)
    print("生成完成")
    print(f"目标数量: {args.num_samples}")
    print(f"通过质量检查: {len(generated)}")
    print(f"输出目录: {OUTPUT_NPZ_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"生成参数: steps={args.num_inference_steps}, guidance={args.guidance_scale}, strength={args.strength}")
    print("="*60)


if __name__ == "__main__":
    main()
