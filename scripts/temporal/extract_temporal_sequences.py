#!/usr/bin/env python3
"""
时序背景片段提取脚本

从环视视频中提取连续帧序列用于时序/稳定性训练。

核心特性：
1. 6视角分割 (2x3 mosaic → 6 individual views)
2. 分辨率转换: 640x360 → 4:3 crop → 640x480
3. 混合步长采样: s ∈ {1, 2, 3, 5}
4. 序列长度: T = 32 帧
5. 帧差分层: 按平均帧差分为小/中/大三档

作者: soiling_project
日期: 2026-02-27
"""

import os
import re
import glob
import json
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def ensure_dir(p: str):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def sanitize_stem(name: str) -> str:
    """清理文件名中的特殊字符"""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def split_views(mosaic_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将 2x3 环视马赛克分割为6个独立视角

    布局:
        [lf]  [f]   [rf]
        [lr]  [r]   [rr]

    Args:
        mosaic_bgr: shape (H, W, 3), 期望 720x1920

    Returns:
        dict: {view_id: crop_bgr}, 每个crop shape ≈ (360, 640, 3)
    """
    H, W = mosaic_bgr.shape[:2]
    wc = W // 3   # 640
    hc = H // 2   # 360

    rois = {
        "lf": (0*wc, 0*hc, 1*wc, 1*hc),
        "f":  (1*wc, 0*hc, 2*wc, 1*hc),
        "rf": (2*wc, 0*hc, 3*wc, 1*hc),
        "lr": (0*wc, 1*hc, 1*wc, 2*hc),
        "r":  (1*wc, 1*hc, 2*wc, 2*hc),
        "rr": (2*wc, 1*hc, 3*wc, 2*hc),
    }

    out = {}
    for vid, (x0, y0, x1, y1) in rois.items():
        out[vid] = mosaic_bgr[y0:y1, x0:x1]
    return out


def crop_to_4by3_and_resize(img_bgr: np.ndarray,
                             target_w: int = 640,
                             target_h: int = 480) -> np.ndarray:
    """
    将图像裁剪到 4:3 比例并 resize 到目标尺寸

    原始尺寸: 640x360 (16:9)
    裁剪策略: 从中心裁剪出 480x360 (4:3)
    目标尺寸: 640x480

    Args:
        img_bgr: 输入图像, shape (H, W, 3)
        target_w: 目标宽度
        target_h: 目标高度

    Returns:
        裁剪并resize后的图像, shape (target_h, target_w, 3)
    """
    h, w = img_bgr.shape[:2]

    # 当前是 640x360 (16:9), 需要裁掉左右两侧得到 4:3
    # 4:3 比例: 360 * 4/3 = 480
    target_ratio = target_w / target_h  # 640 / 480 = 1.333...
    current_ratio = w / h  # 640 / 360 = 1.777...

    if current_ratio > target_ratio:
        # 宽度更大，裁剪左右
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        x1 = x0 + new_w
        cropped = img_bgr[:, x0:x1]
    else:
        # 高度更大，裁剪上下
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        y1 = y0 + new_h
        cropped = img_bgr[y0:y1, :]

    # Resize到目标尺寸
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def compute_frame_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两帧之间的平均L1距离（帧差）

    图像归一化到 [0, 1] 后计算:
        d = mean(|I1 - I2|)

    Args:
        img1: 第一帧, shape (H, W, 3), uint8 [0, 255]
        img2: 第二帧, shape (H, W, 3), uint8 [0, 255]

    Returns:
        帧差均值, float [0, 1]
    """
    # 转换到float并归一化
    f1 = img1.astype(np.float32) / 255.0
    f2 = img2.astype(np.float32) / 255.0

    # L1距离
    diff = np.abs(f1 - f2)
    return float(diff.mean())


class TemporalSequenceExtractor:
    """时序序列提取器"""

    def __init__(self,
                 video_dir: str,
                 output_dir: str,
                 metadata_dir: str,
                 seq_length: int = 32,
                 step_candidates: List[int] = [1, 2, 3, 5],
                 step_probs: Optional[List[float]] = None,
                 min_seq_per_video: int = 5,
                 max_seq_per_video: int = 20,
                 target_w: int = 640,
                 target_h: int = 480,
                 jpeg_quality: int = 95,
                 seed: int = 42):
        """
        Args:
            video_dir: 原始视频目录
            output_dir: 输出序列目录
            metadata_dir: 元数据输出目录
            seq_length: 序列长度T（帧数）
            step_candidates: 候选步长列表
            step_probs: 步长采样概率（None则均匀分布）
            min_seq_per_video: 每个视频最少提取序列数
            max_seq_per_video: 每个视频最多提取序列数
            target_w: 目标宽度
            target_h: 目标高度
            jpeg_quality: JPEG保存质量
            seed: 随机种子
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir
        self.seq_length = seq_length
        self.step_candidates = step_candidates
        self.step_probs = step_probs or ([1.0/len(step_candidates)] * len(step_candidates))
        self.min_seq_per_video = min_seq_per_video
        self.max_seq_per_video = max_seq_per_video
        self.target_w = target_w
        self.target_h = target_h
        self.jpeg_quality = jpeg_quality

        # 归一化概率
        self.step_probs = np.array(self.step_probs) / np.sum(self.step_probs)

        # 创建目录
        ensure_dir(output_dir)
        ensure_dir(metadata_dir)
        for view_id in ["lf", "f", "rf", "lr", "r", "rr"]:
            ensure_dir(os.path.join(output_dir, view_id))

        # 统计信息
        self.stats = {
            "total_videos": 0,
            "total_sequences": 0,
            "total_frames": 0,
            "step_distribution": defaultdict(int),
            "view_distribution": defaultdict(int),
            "frame_diff_bins": {"low": 0, "medium": 0, "high": 0},
        }

        # 元数据记录
        self.sequence_records = []

        # 设置随机种子
        np.random.seed(seed)

    def _get_max_start_frame(self, video_fps: float, step: int, total_frames: int) -> int:
        """
        计算最大起始帧位置

        Args:
            video_fps: 视频帧率
            step: 抽帧步长
            total_frames: 视频总帧数

        Returns:
            最大起始帧位置
        """
        # 序列需要的总帧跨度（原始视频帧数）
        span_frames = (self.seq_length - 1) * step + 1

        if total_frames < span_frames:
            return -1  # 视频太短，无法提取该步长的序列

        return total_frames - span_frames

    def _sample_step(self) -> int:
        """按概率采样步长"""
        return int(np.random.choice(self.step_candidates, p=self.step_probs))

    def _extract_sequence(self,
                          cap: cv2.VideoCapture,
                          view_id: str,
                          start_frame: int,
                          step: int,
                          video_id: str,
                          video_fps: float) -> Optional[Dict]:
        """
        提取单个时序序列

        Args:
            cap: 视频捕获对象
            view_id: 视角ID
            start_frame: 起始帧位置
            step: 抽帧步长
            video_id: 视频ID
            video_fps: 视频帧率

        Returns:
            序列信息dict，失败返回None
        """
        frames = []
        frame_positions = []

        # 提取序列帧
        for i in range(self.seq_length):
            frame_pos = start_frame + i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, mosaic = cap.read()

            if not ret:
                return None  # 读取失败

            # 分割视角
            views = split_views(mosaic)
            if view_id not in views:
                return None

            # 4:3裁剪 + resize
            view_bgr = views[view_id]
            processed = crop_to_4by3_and_resize(view_bgr, self.target_w, self.target_h)
            frames.append(processed)
            frame_positions.append(frame_pos)

        # 计算帧差统计
        frame_diffs = []
        for i in range(len(frames) - 1):
            fd = compute_frame_diff(frames[i], frames[i+1])
            frame_diffs.append(fd)

        mean_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0

        # 分层
        if mean_diff < 0.02:
            diff_bin = "low"
        elif mean_diff < 0.08:
            diff_bin = "medium"
        else:
            diff_bin = "high"

        # 生成序列ID
        seq_id = f"{video_id}_{view_id}_seq{len(self.sequence_records):06d}_s{step}_d{diff_bin[0]}"

        # 保存序列
        seq_dir = os.path.join(self.output_dir, view_id, seq_id)
        ensure_dir(seq_dir)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        for i, frame in enumerate(frames):
            frame_path = os.path.join(seq_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame, encode_param)

        # 记录元数据
        record = {
            "seq_id": seq_id,
            "video_id": video_id,
            "view_id": view_id,
            "start_frame": start_frame,
            "step": step,
            "seq_length": self.seq_length,
            "frame_positions": json.dumps(frame_positions),
            "mean_frame_diff": mean_diff,
            "diff_bin": diff_bin,
            "video_fps": video_fps,
            "duration_sec": (self.seq_length - 1) * step / video_fps,
            "output_dir": seq_dir,
        }

        return record

    def _sample_num_sequences(self, total_frames: int, video_fps: float) -> int:
        """
        根据视频长度决定提取多少序列

        规则:
        - 短视频(< 5秒): min_seq
        - 中等(5-30秒): 按 1序列/2秒
        - 长视频(> 30秒): max_seq
        """
        duration_sec = total_frames / video_fps

        if duration_sec < 5.0:
            return self.min_seq_per_video
        elif duration_sec < 30.0:
            return min(self.max_seq_per_video, max(self.min_seq_per_video, int(duration_sec / 2.0)))
        else:
            return self.max_seq_per_video

    def process_videos(self):
        """处理所有视频"""
        # 查找所有视频文件
        exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.MP4", "*.MOV", "*.MKV", "*.AVI")
        video_paths = []
        for ext in exts:
            video_paths += glob.glob(os.path.join(self.video_dir, ext))
        video_paths = sorted(video_paths)

        self.stats["total_videos"] = len(video_paths)

        if len(video_paths) == 0:
            print(f"警告: 在 {self.video_dir} 中未找到视频文件")
            return

        print(f"找到 {len(video_paths)} 个视频文件")

        for vp in tqdm(video_paths, desc="处理视频"):
            self._process_single_video(vp)

        # 保存元数据
        self._save_metadata()

        # 打印统计
        self._print_stats()

    def _process_single_video(self, video_path: str):
        """处理单个视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告: 无法打开视频 {video_path}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 生成视频ID
        raw_stem = os.path.splitext(os.path.basename(video_path))[0]
        video_id = sanitize_stem(raw_stem)

        # 决定提取多少序列
        num_sequences = self._sample_num_sequences(total_frames, fps)

        # 为每个视角提取序列
        for view_id in ["lf", "f", "rf", "lr", "r", "rr"]:
            for _ in range(num_sequences):
                # 随机采样步长
                step = self._sample_step()

                # 计算最大起始帧
                max_start = self._get_max_start_frame(fps, step, total_frames)
                if max_start < 0:
                    continue  # 视频太短

                # 随机选择起始帧
                start_frame = np.random.randint(0, max_start + 1)

                # 提取序列
                record = self._extract_sequence(
                    cap, view_id, start_frame, step, video_id, fps
                )

                if record:
                    self.sequence_records.append(record)
                    self.stats["total_sequences"] += 1
                    self.stats["total_frames"] += self.seq_length
                    self.stats["step_distribution"][step] += 1
                    self.stats["view_distribution"][view_id] += 1
                    self.stats["frame_diff_bins"][record["diff_bin"]] += 1

        cap.release()

    def _save_metadata(self):
        """保存元数据"""
        # 保存序列清单
        manifest_path = os.path.join(self.metadata_dir, "sequence_manifest.csv")
        df = pd.DataFrame(self.sequence_records)
        df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

        # 保存统计信息
        stats_path = os.path.join(self.metadata_dir, "extraction_stats.json")
        stats_to_save = dict(self.stats)
        stats_to_save["step_distribution"] = dict(stats_to_save["step_distribution"])
        stats_to_save["view_distribution"] = dict(stats_to_save["view_distribution"])

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

        # 保存配置
        config_path = os.path.join(self.metadata_dir, "extraction_config.json")
        config = {
            "seq_length": self.seq_length,
            "step_candidates": self.step_candidates,
            "step_probs": self.step_probs.tolist(),
            "min_seq_per_video": self.min_seq_per_video,
            "max_seq_per_video": self.max_seq_per_video,
            "target_w": self.target_w,
            "target_h": self.target_h,
            "jpeg_quality": self.jpeg_quality,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("时序序列提取完成")
        print("="*60)
        print(f"处理视频数: {self.stats['total_videos']}")
        print(f"提取序列数: {self.stats['total_sequences']}")
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"平均每视频序列数: {self.stats['total_sequences'] / max(1, self.stats['total_videos']):.1f}")
        print("\n步长分布:")
        for step, count in sorted(self.stats["step_distribution"].items()):
            ratio = count / max(1, self.stats["total_sequences"]) * 100
            print(f"  s={step}: {count} ({ratio:.1f}%)")
        print("\n视角分布:")
        for view_id, count in sorted(self.stats["view_distribution"].items()):
            ratio = count / max(1, self.stats["total_sequences"]) * 100
            print(f"  {view_id}: {count} ({ratio:.1f}%)")
        print("\n帧差分层分布:")
        for bin_name, count in [("low", self.stats["frame_diff_bins"]["low"]),
                                 ("medium", self.stats["frame_diff_bins"]["medium"]),
                                 ("high", self.stats["frame_diff_bins"]["high"])]:
            ratio = count / max(1, self.stats["total_sequences"]) * 100
            print(f"  {bin_name}: {count} ({ratio:.1f}%)")
        print("="*60)


def main():
    ap = argparse.ArgumentParser(description="时序背景片段提取脚本")

    ap.add_argument("--video_dir", type=str, default="dataset/my_clean_video_raw",
                    help="原始视频目录")
    ap.add_argument("--output_dir", type=str, default="dataset/temporal_sequences/raw_sequences",
                    help="输出序列目录")
    ap.add_argument("--metadata_dir", type=str, default="dataset/temporal_sequences/metadata",
                    help="元数据输出目录")

    ap.add_argument("--seq_length", type=int, default=32,
                    help="序列长度T（帧数）")
    ap.add_argument("--step_candidates", type=int, nargs="+", default=[1, 2, 3, 5],
                    help="候选步长列表")
    ap.add_argument("--step_probs", type=float, nargs="+", default=None,
                    help="步长采样概率（默认均匀）")
    ap.add_argument("--min_seq_per_video", type=int, default=5,
                    help="每视频最少序列数")
    ap.add_argument("--max_seq_per_video", type=int, default=20,
                    help="每视频最多序列数")

    ap.add_argument("--target_w", type=int, default=640,
                    help="目标宽度")
    ap.add_argument("--target_h", type=int, default=480,
                    help="目标高度")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                    help="JPEG保存质量")

    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")

    args = ap.parse_args()

    # 创建提取器
    extractor = TemporalSequenceExtractor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        seq_length=args.seq_length,
        step_candidates=args.step_candidates,
        step_probs=args.step_probs,
        min_seq_per_video=args.min_seq_per_video,
        max_seq_per_video=args.max_seq_per_video,
        target_w=args.target_w,
        target_h=args.target_h,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
    )

    # 处理视频
    extractor.process_videos()


if __name__ == "__main__":
    main()
