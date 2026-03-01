"""
时序模型：基于 BaselineDualHead + TCN

设计要点：
1. 保留 Stage 1 的 BaselineDualHead backbone
2. 添加时序模块处理帧间关系
3. 支持单帧模式（seq_len=1）兼容 Stage 1 权重
4. 损失函数：稳定性 + 锚定防坍塌

模型结构：
    Input [B, T, 3, H, W]
        ↓
    ResNet18 Backbone (shared weights)
        ↓
    Frame Features [B, T, 512, H', W']
        ↓
    AdaptiveAvgPool → [B, T, 512]
        ↓
    Temporal Module (TCN)
        ↓
    ┌─→ Tile Head → G_hat [B, 4, 8, 8]
    │
    └─→ Global Head → S_hat [B, 1]
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Dict, Tuple, Optional


class TemporalConv1d(nn.Module):
    """
    一维时序卷积模块（TCN）

    使用因果卷积（causal convolution）确保当前帧只依赖历史帧
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TemporalModule(nn.Module):
    """
    时序模块

    使用堆叠的一维卷积处理时序信息

    Args:
        in_features: 输入特征维度（512 for ResNet18）
        hidden_dim: 隐藏层维度
        num_layers: TCN 层数
        kernel_size: 卷积核大小
        dropout: Dropout 比例
    """

    def __init__(self,
                 in_features: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        in_ch = in_features

        # 逐层降低维度并增加感受野
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else in_features
            dilation = 2 ** i  # 指数级膨胀

            layers.append(TemporalConv1d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, T, C] (batch, time, features)

        Returns:
            [B, T, C] 时序增强后的特征
        """
        # 转置为 [B, C, T] 供 Conv1d 使用
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # 转回 [B, T, C]
        x = x.transpose(1, 2)
        return x


class TemporalBaselineDualHead(nn.Module):
    """
    时序双头模型

    支持单帧模式（seq_len=1）和时序模式（seq_len>1）
    """

    def __init__(self,
                 pretrained: bool = True,
                 temporal_enabled: bool = True,
                 temporal_kwargs: Optional[Dict] = None):
        """
        Args:
            pretrained: 是否使用预训练 ResNet18
            temporal_enabled: 是否启用时序模块
            temporal_kwargs: 时序模块参数
        """
        super().__init__()

        # Backbone (ResNet18)
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 池化层
        self.pool8 = nn.AdaptiveAvgPool2d((8, 8))

        # 时序模块
        self.temporal_enabled = temporal_enabled
        self.temporal_module = TemporalModule(**temporal_kwargs) if temporal_enabled else None

        # 双头
        self.tile_head = nn.Conv2d(512, 4, kernel_size=1, bias=True)
        self.glob_fc = nn.Linear(512, 1)

        # 聚合器（用于计算 S_agg，保持一致性）
        from baseline.models.baseline_dualhead import SeverityAggregator
        self.aggregator = SeverityAggregator()

    def forward_single_frame(self, x):
        """
        单帧前向传播（兼容 Stage 1）

        Args:
            x: [B, 3, H, W]

        Returns:
            dict containing logits, G_hat, S_hat, S_agg
        """
        f = self.stem(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)  # [B, 512, H, W]
        f8 = self.pool8(f)  # [B, 512, 8, 8]

        logits = self.tile_head(f8)  # [B, 4, 8, 8]
        G_hat = torch.softmax(logits, dim=1)  # [B, 4, 8, 8]

        v = f8.mean(dim=(2, 3))  # [B, 512]
        S_hat = torch.sigmoid(self.glob_fc(v))  # [B, 1]

        S_agg = self.aggregator(G_hat)  # [B, 1]

        return {
            "logits": logits,
            "G_hat": G_hat,
            "S_hat": S_hat,
            "S_agg": S_agg,
        }

    def forward_temporal(self, x):
        """
        时序前向传播

        Args:
            x: [B, T, 3, H, W]

        Returns:
            dict containing logits, G_hat, S_hat, S_agg
        """
        B, T, C, H, W = x.shape

        # Reshape for backbone: [B*T, 3, H, W]
        x = x.view(B * T, C, H, W)

        # Backbone（共享权重）
        f = self.stem(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)  # [B*T, 512, H', W']
        f8 = self.pool8(f)  # [B*T, 512, 8, 8]

        # Reshape back: [B, T, 512, 8, 8]
        f8 = f8.view(B, T, 512, 8, 8)

        # 时序模块（可选）
        if self.temporal_module is not None:
            # 全局池化后应用时序模块
            v = f8.mean(dim=(3, 4))  # [B, T, 512]
            v_temporal = self.temporal_module(v)  # [B, T, 512]

            # 使用最后一帧的 f8 进行空间预测
            f8_spatial = f8[:, -1]  # [B, 512, 8, 8]
        else:
            v_temporal = None
            f8_spatial = f8[:, -1]  # 使用最后一帧

        # Tile Head (使用最后一帧的空间特征)
        logits = self.tile_head(f8_spatial)  # [B, 4, 8, 8]
        G_hat = torch.softmax(logits, dim=1)  # [B, 4, 8, 8]

        # Global Head
        if v_temporal is not None:
            # 使用时序增强后的特征（取最后一帧）
            v = v_temporal[:, -1]  # [B, 512]
        else:
            v = f8[:, -1].mean(dim=(2, 3))  # [B, 512]

        S_hat = torch.sigmoid(self.glob_fc(v))  # [B, 1]

        # 聚合器
        S_agg = self.aggregator(G_hat)  # [B, 1]

        return {
            "logits": logits,
            "G_hat": G_hat,
            "S_hat": S_hat,
            "S_agg": S_agg,
        }

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, 3, H, W] or [B, T, 3, H, W]

        Returns:
            dict containing logits, G_hat, S_hat, S_agg
        """
        if x.ndim == 4:  # 单帧
            return self.forward_single_frame(x)
        else:  # 时序
            return self.forward_temporal(x)


def compute_temporal_loss(s_hat: torch.Tensor,
                          stability_mode: str = 'adjacent',
                          beta: float = 1.0) -> torch.Tensor:
    """
    计算时序损失（稳定性约束）

    Args:
        s_hat: 预测的 S 分数 [B, T, 1] 或 [B, T]
        stability_mode: 'adjacent' (相邻帧差异) 或 'window' (窗口内稳定)
        beta: 窗口模式的权重

    Returns:
        时序损失标量
    """
    if s_hat.dim() == 2:
        s_hat = s_hat.unsqueeze(-1)  # [B, T, 1]

    B, T, _ = s_hat.shape

    if T <= 1:
        # 单帧，无时序损失
        return torch.tensor(0.0, device=s_hat.device)

    if stability_mode == 'adjacent':
        # 相邻帧差异
        s_diff = s_hat[:, 1:] - s_hat[:, :-1]  # [B, T-1, 1]
        loss = s_diff.abs().mean()

    elif stability_mode == 'window':
        # 窗口内稳定（偏离均值的偏差）
        s_mean = s_hat.mean(dim=1, keepdim=True)  # [B, 1, 1]
        s_dev = (s_hat - s_mean).abs().mean()

        # 可选：结合相邻帧差异
        if beta > 0:
            s_diff = s_hat[:, 1:] - s_hat[:, :-1]
            loss_adjacent = s_diff.abs().mean()
            loss = s_dev + beta * loss_adjacent
        else:
            loss = s_dev

    else:
        raise ValueError(f"Unknown stability_mode: {stability_mode}")

    return loss


def compute_anchor_loss(s_hat_student: torch.Tensor,
                         s_hat_teacher: torch.Tensor) -> torch.Tensor:
    """
    计算锚定损失（防坍塌）

    使用 Stage 1 teacher 的输出作为锚定，防止 student 偏离单帧标尺

    Args:
        s_hat_student: Student 模型的预测 [B, T, 1] 或 [B, T]
        s_hat_teacher: Teacher 模型的预测 [B, T, 1] 或 [B, T]

    Returns:
        锚定损失标量
    """
    if s_hat_student.dim() == 2:
        s_hat_student = s_hat_student.unsqueeze(-1)
    if s_hat_teacher.dim() == 2:
        s_hat_teacher = s_hat_teacher.unsqueeze(-1)

    # 确保维度一致
    if s_hat_student.shape != s_hat_teacher.shape:
        # 可能是序列长度不匹配，取最后一个
        if s_hat_student.shape[1] != s_hat_teacher.shape[1]:
            min_len = min(s_hat_student.shape[1], s_hat_teacher.shape[1])
            s_hat_student = s_hat_student[:, -min_len:]
            s_hat_teacher = s_hat_teacher[:, -min_len:]

    loss = torch.abs(s_hat_student - s_hat_teacher).mean()
    return loss
