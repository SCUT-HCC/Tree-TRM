"""
视觉特征投影层（Visual Projection Layer）
==========================================
实现论文中的 Proj(·) 算子：

  M_L3^{(i,j)} = { Proj(CLIP(F_{i,j,k})) }_{k=1}^{N3}

其中 Proj(·) 是一个可学习的线性投影层，
将 CLIP 图像编码器的输出（视觉特征空间）对齐到文本潜空间，
使得 Phase-3 的跨模态检索（文本查询 vs 视觉节点）在同一向量空间内进行。

设计说明：
----------
1. **VisualProjectionLayer**：核心投影模块，基于 PyTorch nn.Module。
   - 支持恒等初始化（identity init）：当视觉和文本空间维度相同时（如 CLIP 联合空间），
     初始投影为恒等变换，不破坏原有对齐。
   - 支持残差连接（residual）：在维度相同时添加跳跃连接，稳定训练。
   - 层归一化（LayerNorm）：防止投影后的向量量级失衡。

2. **ProjectionManager**：高层管理器，封装批量投影、权重保存/加载。

3. **关键约束**：
   - 推理阶段：投影层通常处于 eval 模式（冻结 BN/Dropout）。
   - 若无预训练权重：使用恒等初始化（同维度）或随机初始化（跨维度），
     后者需要在对齐数据集上微调才能正常工作。
   - L2 归一化：投影后强制进行 L2 归一化，与文本嵌入保持一致。
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VisualProjectionLayer
# ---------------------------------------------------------------------------

class VisualProjectionLayer(nn.Module):
    """可学习的视觉特征投影层 Proj(·)。

    将 CLIP 图像编码器的输出从视觉特征空间映射到文本潜空间，
    实现论文中的跨模态对齐：

    .. math::
        M_{L3}^{(i,j)} = \\{ \\text{Proj}(\\text{CLIP}(F_{i,j,k})) \\}_{k=1}^{N_3}

    网络结构：
    - **同维度**（visual_dim == text_dim）：
      ``LayerNorm → Linear → GELU → Linear``，带残差连接。
      参数初始化为恒等变换，训练时保持稳定。
    - **跨维度**（visual_dim ≠ text_dim）：
      ``Linear（visual→text）→ LayerNorm → GELU → Linear（text→text）``，无残差。
      需要在对齐数据上微调。

    Args:
        visual_dim:  CLIP 图像编码器输出维度（如 ViT-B/32 为 512）。
        text_dim:    目标文本潜空间维度（等于 embed_dim）。
        dropout:     Dropout 概率（推理时为 0）。
        identity_init: 是否将线性层初始化为近似恒等变换（同维度时推荐）。
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 512,
        dropout: float = 0.0,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.same_dim = visual_dim == text_dim

        # 投影网络
        if self.same_dim:
            # 同维度：LN → Linear → GELU → Linear，残差
            self.norm_in = nn.LayerNorm(visual_dim)
            self.fc1 = nn.Linear(visual_dim, visual_dim, bias=True)
            self.fc2 = nn.Linear(visual_dim, text_dim, bias=True)
        else:
            # 跨维度：Linear（升/降维）→ LN → GELU → Linear
            self.fc1 = nn.Linear(visual_dim, text_dim, bias=True)
            self.norm_in = nn.LayerNorm(text_dim)
            self.fc2 = nn.Linear(text_dim, text_dim, bias=True)

        self.norm_out = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if identity_init:
            self._identity_init()

    def _identity_init(self) -> None:
        """将网络参数初始化为（近似）恒等变换。

        对于同维度情形，fc1 初始化为单位矩阵，fc2 初始化为单位矩阵，
        bias 初始化为零。这样在未经训练时，投影层对 CLIP 特征几乎不作改动，
        直接利用 CLIP 的联合对齐空间进行跨模态检索。
        """
        with torch.no_grad():
            if self.same_dim:
                # fc1 → 单位矩阵
                nn.init.eye_(self.fc1.weight)
                nn.init.zeros_(self.fc1.bias)  # type: ignore[arg-type]
                # fc2 → 单位矩阵
                nn.init.eye_(self.fc2.weight)
                nn.init.zeros_(self.fc2.bias)  # type: ignore[arg-type]
            else:
                # 跨维度无法精确恒等，使用 Xavier 均匀初始化
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.zeros_(self.fc1.bias)  # type: ignore[arg-type]
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.zeros_(self.fc2.bias)  # type: ignore[arg-type]

        logger.debug(
            f"VisualProjectionLayer 恒等初始化完成 "
            f"（{self.visual_dim}→{self.text_dim}，same_dim={self.same_dim}）"
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """前向传播。

        Args:
            x:         视觉特征张量，形状 [..., visual_dim]。
            normalize: 是否对输出进行 L2 归一化（推荐 True，与文本嵌入对齐）。

        Returns:
            投影后的特征张量，形状 [..., text_dim]，已 L2 归一化（若 normalize=True）。
        """
        if self.same_dim:
            # 同维度路径：残差结构
            h = self.norm_in(x)
            h = F.gelu(self.fc1(h))
            h = self.dropout(h)
            h = self.fc2(h)
            out = self.norm_out(x + h)  # 残差连接
        else:
            # 跨维度路径
            h = self.fc1(x)             # visual_dim → text_dim
            h = self.norm_in(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = self.fc2(h)
            out = self.norm_out(h)

        if normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def project_numpy(
        self,
        visual_embeds: np.ndarray,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> np.ndarray:
        """对 numpy 嵌入矩阵进行批量投影（推理用接口）。

        Args:
            visual_embeds: 视觉嵌入矩阵，形状 [N, visual_dim]，float32。
            device:        推理设备（"cpu" 或 "cuda"）。
            batch_size:    批处理大小。

        Returns:
            投影后的嵌入矩阵，形状 [N, text_dim]，float32。
        """
        self.eval()
        n = len(visual_embeds)
        if n == 0:
            return np.zeros((0, self.text_dim), dtype=np.float32)

        results = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.from_numpy(
                    visual_embeds[i : i + batch_size].astype(np.float32)
                ).to(device)
                projected = self.forward(batch, normalize=True)
                results.append(projected.cpu().numpy())

        return np.concatenate(results, axis=0)

    def save_checkpoint(self, path: str) -> None:
        """保存投影层权重。

        Args:
            path: 权重文件路径（.pt）。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "visual_dim": self.visual_dim,
                "text_dim": self.text_dim,
            },
            path,
        )
        logger.info(f"投影层权重已保存至 {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "VisualProjectionLayer":
        """从磁盘加载投影层权重。

        Args:
            path:   权重文件路径（.pt）。
            device: 加载设备。

        Returns:
            已加载权重的 VisualProjectionLayer 实例。
        """
        ckpt = torch.load(path, map_location=device)
        visual_dim = ckpt["visual_dim"]
        text_dim = ckpt["text_dim"]
        proj = cls(visual_dim=visual_dim, text_dim=text_dim, identity_init=False)
        proj.load_state_dict(ckpt["state_dict"])
        proj.to(device)
        proj.eval()
        logger.info(f"投影层权重已从 {path} 加载 ({visual_dim}→{text_dim})")
        return proj


# ---------------------------------------------------------------------------
# ProjectionManager
# ---------------------------------------------------------------------------

class ProjectionManager:
    """投影层管理器：封装投影层的创建、加载和批量推理。

    根据配置自动选择正确的维度和初始化策略，
    并提供统一的 numpy 数组接口，屏蔽 PyTorch 细节。

    Args:
        visual_dim:   CLIP 图像编码器输出维度。
        text_dim:     目标文本潜空间维度。
        checkpoint:   预训练权重路径（None 则使用恒等初始化）。
        device:       推理设备。
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 512,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.device = device

        if checkpoint and os.path.isfile(checkpoint):
            self.proj = VisualProjectionLayer.load_checkpoint(checkpoint, device)
            logger.info(f"投影层已从 {checkpoint} 加载预训练权重。")
        else:
            self.proj = VisualProjectionLayer(
                visual_dim=visual_dim,
                text_dim=text_dim,
                identity_init=True,  # 恒等初始化，推理时无损原始 CLIP 空间
            )
            self.proj.to(device)
            self.proj.eval()
            if checkpoint:
                logger.warning(
                    f"投影层权重文件不存在：{checkpoint}，使用恒等初始化。"
                )
            else:
                logger.info("投影层使用恒等初始化（未指定预训练权重）。")

    def project(self, visual_embeds: np.ndarray) -> np.ndarray:
        """对 numpy 视觉嵌入矩阵进行投影。

        Args:
            visual_embeds: 视觉嵌入，形状 [N, visual_dim]，float32。

        Returns:
            投影后的嵌入，形状 [N, text_dim]，L2 归一化，float32。
        """
        return self.proj.project_numpy(
            visual_embeds, device=self.device, batch_size=256
        )

