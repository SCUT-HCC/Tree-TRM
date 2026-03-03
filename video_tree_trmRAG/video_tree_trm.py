"""
Video-Tree-TRM 递归检索引擎（核心模块）
=========================================
实现论文 §3.3 中的"具备模态切换能力的递归双循环推理"。

算法流程（H=3 个周期）：

  初始化：
    q   = CLIP_text(query)        # 查询嵌入，[D]
    z_0 = q                       # 潜在状态初始化为查询嵌入

  Phase 1：粗粒度路由（H=1，文本对文本）
    Scores_L1 = Softmax((q + z_0) · M_L1ᵀ / √D)
    k₁*       = argmax(Scores_L1)
    z_1       = normalize(z_0 + M_L1[k₁*])

  Phase 2：细粒度聚焦（H=2，文本对文本）
    Scores_L2 = Softmax((q + z_1) · M_L2[k₁*]ᵀ / √D)
    k₂*       = argmax(Scores_L2)
    z_2       = normalize(z_1 + M_L2[k₁*][k₂*])

  Phase 3：视觉定位（H=3，文本对图像——跨模态）
    Scores_L3 = Softmax((q + z_2) · M_L3[k₁*,k₂*]ᵀ / √D)
    k₃*       = argmax(Scores_L3)
    F_target  = Frame[k₁*, k₂*, k₃*]

  答案生成：
    answer = VLM(query, F_target)

关键设计：
  1. 「(q + z_h)」：查询向量与潜在状态的加性融合，既保留了原始查询语义，
     又引入了当前检索层积累的上下文信息。
  2. 潜在状态更新「z_{h+1} = normalize(z_h + M_Lh[k_h*])」：
     将选中节点的语义嵌入累加到状态中，驱动下一层的更精确定位。
  3. 模态切换：Phase 1/2 的记忆矩阵为文本嵌入（CLIP text 空间），
     Phase 3 的记忆矩阵为投影后的视觉嵌入（CLIP image → Proj → 同一空间），
     系统无需更换查询机制即可实现跨模态检索。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .video_pyramid import FrameNode, HierarchicalSemanticPyramid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTrace:
    """记录 Video-Tree-TRM 三阶段检索的完整轨迹，用于调试和可解释性分析。

    Attributes:
        query:             用户原始查询文本。
        query_embedding:   查询的文本嵌入，[D]。

        k1_star:           Phase 1 选中的 L1 段索引。
        k2_star:           Phase 2 选中的 L2 片段索引（相对于 L1 段）。
        k3_star:           Phase 3 选中的 L3 帧索引（相对于 L2 片段）。

        l1_scores:         Phase 1 的 Softmax 分数，[N1]。
        l2_scores:         Phase 2 的 Softmax 分数，[N2]。
        l3_scores:         Phase 3 的 Softmax 分数，[N3]。

        z0:                初始潜在状态，[D]。
        z1:                Phase 1 后的潜在状态，[D]。
        z2:                Phase 2 后的潜在状态，[D]。

        segment_summary:   选中的 L1 段摘要文本。
        clip_caption:      选中的 L2 片段描述文本。
        target_frame:      选中的 L3 帧节点（含路径和时间戳）。
        target_timestamp:  目标帧的绝对时间戳（秒）。
    """

    query: str = ""
    query_embedding: Optional[np.ndarray] = None

    k1_star: int = -1
    k2_star: int = -1
    k3_star: int = -1

    l1_scores: Optional[np.ndarray] = None
    l2_scores: Optional[np.ndarray] = None
    l3_scores: Optional[np.ndarray] = None

    z0: Optional[np.ndarray] = None
    z1: Optional[np.ndarray] = None
    z2: Optional[np.ndarray] = None

    segment_summary: str = ""
    clip_caption: str = ""
    target_frame: Optional[FrameNode] = None
    target_timestamp: float = -1.0

    def is_valid(self) -> bool:
        """检查检索是否成功（所有三个索引均合法）。"""
        return self.k1_star >= 0 and self.k2_star >= 0 and self.k3_star >= 0

    def summary(self) -> str:
        """生成人类可读的检索摘要字符串。"""
        if not self.is_valid():
            return "检索失败：未找到有效结果。"
        lines = [
            f"查询：{self.query}",
            f"Phase 1 → L1[{self.k1_star}]：{self.segment_summary[:80]}...",
            f"Phase 2 → L2[{self.k2_star}]：{self.clip_caption[:80]}...",
            f"Phase 3 → L3[{self.k3_star}]：时间戳 {self.target_timestamp:.1f}s，"
            f"路径 {self.target_frame.frame_path if self.target_frame else 'N/A'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree-TRM Attention Kernel
# ---------------------------------------------------------------------------

def tree_trm_attention(
    query_state: np.ndarray,
    memory_matrix: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Tree-TRM 注意力核心计算。

    实现论文中每个阶段的检索计算：

    .. math::
        \\text{Scores} = \\text{Softmax}\\left(
            \\frac{(q + z_h) \\cdot M_{Lh}^\\top}{\\sqrt{D} \\cdot \\tau}
        \\right)

        k_h^* = \\arg\\max(\\text{Scores})

    其中 (q + z_h) 为查询状态（由外层已完成加法传入），
    M_Lh 为当前层的记忆矩阵。

    Args:
        query_state:    当前阶段的复合查询向量 (q + z_h)，形状 [D]，L2 归一化。
        memory_matrix:  当前层的记忆矩阵 M_Lh，形状 [N, D]，L2 归一化。
        temperature:    Softmax 温度系数 τ。τ<1 使分布更尖锐（更确定），τ>1 更平滑。

    Returns:
        三元组 (scores, soft_retrieved, k_star)：
          - scores:        Softmax 分数，形状 [N]，float32。
          - soft_retrieved: 加权求和的软检索向量，形状 [D]，float32（用于训练/解释）。
          - k_star:        argmax 选中的节点索引（硬选择），int。

    Notes:
        - 两个输入均应预先进行 L2 归一化，以使点积等价于余弦相似度。
        - 若 memory_matrix 为空（N=0），返回零分数和空向量，k_star=-1。
    """
    N = memory_matrix.shape[0]
    if N == 0:
        logger.warning("tree_trm_attention：记忆矩阵为空，返回默认值。")
        D = query_state.shape[0]
        return np.zeros(0, dtype=np.float32), np.zeros(D, dtype=np.float32), -1

    D = memory_matrix.shape[1]

    # 缩放点积：(q + z_h) · M_Lh^T / (√D · τ)
    scale = math.sqrt(D) * temperature
    raw_scores = memory_matrix @ query_state / scale  # [N]

    # Softmax（数值稳定：减去最大值）
    raw_scores = raw_scores - raw_scores.max()
    exp_scores = np.exp(raw_scores)
    scores = exp_scores / (exp_scores.sum() + 1e-9)  # [N]

    # 硬选择
    k_star = int(np.argmax(scores))

    # 软检索向量（用于解释或训练）
    soft_retrieved = (scores[:, None] * memory_matrix).sum(axis=0)  # [D]

    return scores.astype(np.float32), soft_retrieved.astype(np.float32), k_star


def update_latent_state(
    z: np.ndarray,
    selected_embed: np.ndarray,
    mode: str = "additive",
    normalize: bool = True,
) -> np.ndarray:
    """更新潜在状态 z_h → z_{h+1}。

    实现论文中的 Update 函数：z_{h+1} = Update(z_h, M_Lh[k_h*])。

    Args:
        z:              当前潜在状态 z_h，形状 [D]。
        selected_embed: 选中节点的嵌入向量 M_Lh[k_h*]，形状 [D]。
        mode:           更新模式：
                          "additive" - z_{h+1} = z_h + selected_embed（论文默认）。
                          "replace"  - z_{h+1} = selected_embed（丢弃历史）。
                          "gated"    - 固定 gate=0.5 加权融合（简化门控）。
        normalize:      是否对结果进行 L2 归一化。

    Returns:
        更新后的潜在状态 z_{h+1}，形状 [D]。
    """
    if mode == "additive":
        new_z = z + selected_embed
    elif mode == "replace":
        new_z = selected_embed.copy()
    elif mode == "gated":
        gate = 0.5  # 固定门控值
        new_z = gate * z + (1.0 - gate) * selected_embed
    else:
        raise ValueError(f"未知的状态更新模式：{mode}。请选择 additive/replace/gated。")

    if normalize:
        norm = np.linalg.norm(new_z)
        if norm > 1e-8:
            new_z = new_z / norm

    return new_z.astype(np.float32)


# ---------------------------------------------------------------------------
# VideoTreeTRM: Core Retrieval Engine
# ---------------------------------------------------------------------------

class VideoTreeTRM:
    """Video-Tree-TRM 三阶段递归检索引擎。

    该类实现论文 §3.3 的核心推理逻辑，给定：
    - 用户查询 q
    - 预构建的分层语义金字塔（HSP）

    执行 H=3 个阶段的由粗到细遍历，输出精确定位到的关键帧 F_target。

    Args:
        embed_dim:         嵌入空间维度（CLIP 输出维度或 Ollama 嵌入维度）。
        text_backend:      文本嵌入后端，"clip" 或 "ollama"。
        clip_model:        CLIP 模型名称（当 text_backend="clip" 时有效）。
        clip_device:       CLIP 推理设备。
        ollama_embed_url:  Ollama 嵌入 API 地址（当 text_backend="ollama" 时有效）。
        ollama_embed_model:Ollama 嵌入模型名称。
        selection_mode:    节点选择模式，"argmax"（硬选择）或 "soft"（软选择）。
        softmax_temperature: Softmax 温度参数。
        state_update_mode: 潜在状态更新模式。
        normalize_state:   是否对潜在状态进行 L2 归一化。
        init_state_mode:   z_0 初始化方式，"query" 或 "zeros"。
        verbose:           是否打印详细推理日志。
    """

    def __init__(
        self,
        embed_dim: int = 512,
        text_backend: str = "clip",
        clip_model: str = "ViT-B/32",
        clip_device: str = "cuda",
        ollama_embed_url: str = "http://localhost:11434/v1/embeddings",
        ollama_embed_model: str = "nomic-embed-text",
        selection_mode: str = "argmax",
        softmax_temperature: float = 1.0,
        state_update_mode: str = "additive",
        normalize_state: bool = True,
        init_state_mode: str = "query",
        verbose: bool = True,
    ) -> None:
        self.embed_dim = embed_dim
        self.text_backend = text_backend
        self.clip_model = clip_model
        self.clip_device = clip_device
        self.ollama_embed_url = ollama_embed_url
        self.ollama_embed_model = ollama_embed_model
        self.selection_mode = selection_mode
        self.softmax_temperature = softmax_temperature
        self.state_update_mode = state_update_mode
        self.normalize_state = normalize_state
        self.init_state_mode = init_state_mode
        self.verbose = verbose

        # 延迟加载的嵌入器
        self._clip_extractor = None
        self._ollama_embedder = None

    # ------------------------------------------------------------------ #
    # Embedding: Query Encoding                                            #
    # ------------------------------------------------------------------ #

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """将查询文本编码为嵌入向量。

        根据 text_backend 选择使用 CLIP text encoder 或 Ollama nomic-embed-text。

        Args:
            text: 查询文本字符串。

        Returns:
            L2 归一化的嵌入向量，形状 [D]，float32。
        """
        if self.text_backend == "clip":
            return self._clip_encode_text(text)
        elif self.text_backend == "ollama":
            return self._ollama_encode_text(text)
        else:
            raise ValueError(f"未知的文本编码后端：{self.text_backend}")

    def _clip_encode_text(self, text: str) -> np.ndarray:
        """使用 CLIP text encoder 编码文本。"""
        from .video_indexer import CLIPFeatureExtractor

        if self._clip_extractor is None:
            self._clip_extractor = CLIPFeatureExtractor(
                model_name=self.clip_model, device=self.clip_device
            )

        embeds = self._clip_extractor.encode_texts([text], normalize=True)
        return embeds[0].astype(np.float32)

    def _ollama_encode_text(self, text: str) -> np.ndarray:
        """使用 Ollama nomic-embed-text 编码文本。"""
        from .video_indexer import OllamaTextEmbedder

        if self._ollama_embedder is None:
            self._ollama_embedder = OllamaTextEmbedder(
                base_url=self.ollama_embed_url,
                model=self.ollama_embed_model,
            )

        embeds = self._ollama_embedder.encode([text])
        return embeds[0].astype(np.float32)

    # ------------------------------------------------------------------ #
    # Phase Implementations                                                #
    # ------------------------------------------------------------------ #

    def _phase1_coarse_routing(
        self,
        q: np.ndarray,
        z0: np.ndarray,
        M_L1: np.ndarray,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Phase 1：粗粒度路由（H=1，文本对文本）。

        在第一层节点（M_L1，全局叙事节点）上执行 Tree-TRM 注意力，
        定位与查询最相关的宏观事件段。

        .. math::
            \\text{Scores}_{L1} = \\text{Softmax}\\left(
                \\frac{(q + z_0) \\cdot M_{L1}^\\top}{\\sqrt{D}}
            \\right)
            \\quad
            k_1^* = \\arg\\max(\\text{Scores}_{L1})
            \\quad
            z_1 = \\text{normalize}(z_0 + M_{L1}[k_1^*])

        Args:
            q:    查询嵌入，[D]，L2 归一化。
            z0:   初始潜在状态，[D]。
            M_L1: L1 节点嵌入矩阵，[N1, D]。

        Returns:
            三元组 (k1_star, scores_l1, z1)：
              - k1_star: 选中的 L1 段索引。
              - scores_l1: Softmax 分数，[N1]。
              - z1: 更新后的潜在状态，[D]。
        """
        # 复合查询向量：q + z_0
        query_state = q + z0
        # L2 归一化（余弦相似度检索）
        norm = np.linalg.norm(query_state)
        if norm > 1e-8:
            query_state = query_state / norm

        scores_l1, _, k1_star = tree_trm_attention(
            query_state, M_L1, self.softmax_temperature
        )

        if k1_star == -1:
            logger.error("Phase 1 失败：M_L1 为空，无法路由。")
            return -1, scores_l1, z0

        # 更新潜在状态：z_1 = normalize(z_0 + M_L1[k1*])
        z1 = update_latent_state(
            z0, M_L1[k1_star], self.state_update_mode, self.normalize_state
        )

        if self.verbose:
            logger.info(
                f"[Phase 1] 粗粒度路由 → L1[{k1_star}] "
                f"（得分 {scores_l1[k1_star]:.4f}，共 {len(scores_l1)} 个候选段）"
            )

        return k1_star, scores_l1, z1

    def _phase2_fine_focusing(
        self,
        q: np.ndarray,
        z1: np.ndarray,
        M_L2_k1: np.ndarray,
        k1_star: int,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Phase 2：细粒度聚焦（H=2，文本对文本）。

        利用更新后的潜在状态 z_1，在第二层节点（M_L2^{(k1*)}，片段级语义节点）
        上执行 Tree-TRM 注意力，锁定到与查询最相关的具体片段。

        .. math::
            k_2^* = \\arg\\max \\left( \\text{Softmax}\\left(
                \\frac{(q + z_1) \\cdot (M_{L2}^{(k_1^*)})^\\top}{\\sqrt{D}}
            \\right) \\right)
            \\quad
            z_2 = \\text{normalize}(z_1 + M_{L2}^{(k_1^*)}[k_2^*])

        Args:
            q:        查询嵌入，[D]，L2 归一化。
            z1:       Phase 1 后的潜在状态，[D]。
            M_L2_k1:  选中 L1 段内的 L2 节点嵌入矩阵，[N2, D]。
            k1_star:  上一阶段选中的 L1 索引（仅用于日志）。

        Returns:
            三元组 (k2_star, scores_l2, z2)：
              - k2_star: 选中的 L2 片段索引（相对于 L1 段）。
              - scores_l2: Softmax 分数，[N2]。
              - z2: 更新后的潜在状态，[D]。
        """
        # 复合查询：q + z_1（此时 z_1 已包含 L1 粗粒度上下文）
        query_state = q + z1
        norm = np.linalg.norm(query_state)
        if norm > 1e-8:
            query_state = query_state / norm

        scores_l2, _, k2_star = tree_trm_attention(
            query_state, M_L2_k1, self.softmax_temperature
        )

        if k2_star == -1:
            logger.error(f"Phase 2 失败：L1[{k1_star}] 下无 L2 节点。")
            return -1, scores_l2, z1

        # 更新潜在状态：z_2 = normalize(z_1 + M_L2[k1*][k2*])
        z2 = update_latent_state(
            z1, M_L2_k1[k2_star], self.state_update_mode, self.normalize_state
        )

        if self.verbose:
            logger.info(
                f"[Phase 2] 细粒度聚焦 → L1[{k1_star}].L2[{k2_star}] "
                f"（得分 {scores_l2[k2_star]:.4f}，共 {len(scores_l2)} 个候选片段）"
            )

        return k2_star, scores_l2, z2

    def _phase3_visual_grounding(
        self,
        q: np.ndarray,
        z2: np.ndarray,
        M_L3_k1_k2: np.ndarray,
        k1_star: int,
        k2_star: int,
    ) -> Tuple[int, np.ndarray]:
        """Phase 3：视觉定位（H=3，文本对图像——跨模态检索）。

        利用已对齐文本空间的潜在状态 z_2，直接与第三层的视觉嵌入节点
        （M_L3^{(k1*,k2*)}，投影后的 CLIP 图像特征）进行交互，
        精确定位包含视觉证据的目标帧。

        此处实现了论文中的「模态切换」（Modality Switching）：
        查询机制从纯语义匹配转变为视觉验证，无需改变注意力公式，
        因为 Proj(CLIP(·)) 已将视觉特征对齐到文本潜空间。

        .. math::
            k_3^* = \\arg\\max \\left( \\text{Softmax}\\left(
                \\frac{(q + z_2) \\cdot (M_{L3}^{(k_1^*, k_2^*)})^\\top}{\\sqrt{D}}
            \\right) \\right)

        Args:
            q:           查询嵌入，[D]，L2 归一化。
            z2:          Phase 2 后的潜在状态，[D]（包含强烈语义预期）。
            M_L3_k1_k2: 选中片段内的 L3 帧视觉嵌入矩阵，[N3, D]（已经过 Proj 投影）。
            k1_star:     上一阶段 L1 索引（仅用于日志）。
            k2_star:     上一阶段 L2 索引（仅用于日志）。

        Returns:
            二元组 (k3_star, scores_l3)：
              - k3_star: 选中的 L3 帧索引（相对于 L2 片段）。
              - scores_l3: Softmax 分数，[N3]。
        """
        # 复合查询：q + z_2（此时 z_2 已积累粗粒度 + 细粒度语义上下文）
        query_state = q + z2
        norm = np.linalg.norm(query_state)
        if norm > 1e-8:
            query_state = query_state / norm

        scores_l3, _, k3_star = tree_trm_attention(
            query_state, M_L3_k1_k2, self.softmax_temperature
        )

        if k3_star == -1:
            logger.error(
                f"Phase 3 失败：L1[{k1_star}].L2[{k2_star}] 下无 L3 帧节点。"
            )
            return -1, scores_l3

        if self.verbose:
            logger.info(
                f"[Phase 3] 视觉定位 → L1[{k1_star}].L2[{k2_star}].L3[{k3_star}] "
                f"（得分 {scores_l3[k3_star]:.4f}，共 {len(scores_l3)} 个候选帧）"
            )

        return k3_star, scores_l3

    # ------------------------------------------------------------------ #
    # Main Retrieval Interface                                             #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        pyramid: HierarchicalSemanticPyramid,
    ) -> RetrievalTrace:
        """执行完整的三阶段递归检索。

        给定用户查询和预构建的分层语义金字塔，运行 H=3 个周期的
        递归双循环推理，返回精确定位到的目标帧及完整检索轨迹。

        Args:
            query:   用户问题文本（自然语言）。
            pyramid: 预构建的 HierarchicalSemanticPyramid 实例。

        Returns:
            RetrievalTrace 实例，包含：
              - 三个阶段选中的索引 (k1*, k2*, k3*)。
              - 每个阶段的 Softmax 分数分布。
              - 中间潜在状态 z0, z1, z2。
              - 目标帧节点 F_target（含图像路径和时间戳）。
              - 各节点的文字描述（摘要/描述）。

        Raises:
            ValueError: 若金字塔不包含任何 L1 节点。
        """
        trace = RetrievalTrace(query=query)

        # ── 0. 查询编码 ────────────────────────────────────────────────
        if self.verbose:
            logger.info(f"开始检索，查询：'{query}'")
        q = self._get_text_embedding(query)
        trace.query_embedding = q

        # ── 0. 初始化潜在状态 z_0 ──────────────────────────────────────
        if self.init_state_mode == "query":
            z0 = q.copy()  # z_0 = 查询嵌入（语义引导初始化）
        else:
            z0 = np.zeros(self.embed_dim, dtype=np.float32)
        trace.z0 = z0

        # ── Phase 1：粗粒度路由 ────────────────────────────────────────
        M_L1 = pyramid.get_l1_embeddings()  # [N1, D]
        if M_L1.shape[0] == 0:
            raise ValueError("金字塔 M_L1 为空，无法进行检索。请先运行 build_pyramid.py。")

        k1_star, l1_scores, z1 = self._phase1_coarse_routing(q, z0, M_L1)
        trace.k1_star = k1_star
        trace.l1_scores = l1_scores
        trace.z1 = z1

        if k1_star < 0:
            logger.error("Phase 1 返回无效索引，检索中止。")
            return trace

        # 记录选中的 L1 摘要
        seg_node = pyramid.get_segment_node(k1_star)
        if seg_node:
            trace.segment_summary = seg_node.summary

        # ── Phase 2：细粒度聚焦 ────────────────────────────────────────
        M_L2_k1 = pyramid.get_l2_embeddings(k1_star)  # [N2, D]
        if M_L2_k1.shape[0] == 0:
            logger.warning(
                f"L1[{k1_star}] 下无 L2 节点，检索降级为仅返回 L1 结果。"
            )
            return trace

        k2_star, l2_scores, z2 = self._phase2_fine_focusing(q, z1, M_L2_k1, k1_star)
        trace.k2_star = k2_star
        trace.l2_scores = l2_scores
        trace.z2 = z2

        if k2_star < 0:
            logger.error("Phase 2 返回无效索引，检索中止。")
            return trace

        # 记录选中的 L2 描述
        clip_node = pyramid.get_clip_node(k1_star, k2_star)
        if clip_node:
            trace.clip_caption = clip_node.caption

        # ── Phase 3：视觉定位（跨模态）────────────────────────────────
        M_L3_k1_k2 = pyramid.get_l3_embeddings(k1_star, k2_star)  # [N3, D]
        if M_L3_k1_k2.shape[0] == 0:
            logger.warning(
                f"L1[{k1_star}].L2[{k2_star}] 下无 L3 帧，检索降级为返回 L2 结果。"
            )
            return trace

        k3_star, l3_scores = self._phase3_visual_grounding(
            q, z2, M_L3_k1_k2, k1_star, k2_star
        )
        trace.k3_star = k3_star
        trace.l3_scores = l3_scores

        if k3_star < 0:
            logger.error("Phase 3 返回无效索引，检索中止。")
            return trace

        # 记录目标帧
        frame_node = pyramid.get_frame_node(k1_star, k2_star, k3_star)
        if frame_node:
            trace.target_frame = frame_node
            trace.target_timestamp = frame_node.timestamp

        # ── 输出摘要 ───────────────────────────────────────────────────
        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info("检索完成：")
            logger.info(trace.summary())
            logger.info(f"{'='*60}")

        return trace

    def batch_retrieve(
        self,
        queries: List[str],
        pyramid: HierarchicalSemanticPyramid,
    ) -> List[RetrievalTrace]:
        """对多个查询批量执行检索。

        注意：当前实现为串行处理（逐条检索），
        主要的计算瓶颈是文本嵌入，可在此处添加批量嵌入优化。

        Args:
            queries: 查询文本列表。
            pyramid: 预构建的 HierarchicalSemanticPyramid 实例。

        Returns:
            RetrievalTrace 列表，与 queries 一一对应。
        """
        traces = []
        for i, query in enumerate(queries):
            logger.info(f"处理查询 {i+1}/{len(queries)}: '{query[:50]}...'")
            trace = self.retrieve(query, pyramid)
            traces.append(trace)
        return traces

