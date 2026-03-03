"""
分层语义金字塔（Hierarchical Semantic Pyramid, HSP）
======================================================
定义三层节点数据结构，并提供序列化/反序列化接口，
使预处理好的金字塔可在磁盘上持久化，推理时快速加载。

数据结构层次：

  HierarchicalSemanticPyramid
  └─ List[SegmentNode]           L1 全局叙事节点（M_L1）
       └─ List[ClipNode]          L2 片段级语义节点（M_L2）
            └─ List[FrameNode]    L3 帧级视觉节点（M_L3）

每个节点存储：
  - 时间元数据（起止时间）
  - 文本 / 视觉嵌入向量（numpy float32，L2 归一化）
  - 原始文本（摘要 / 描述）或帧图像路径

序列化格式（cache_dir/pyramid_name/）：
  metadata.json          -- 金字塔元数据（视频路径、时长、维度等）
  l1_embeddings.npy      -- L1 节点嵌入矩阵 [N1, D]
  l1_metadata.json       -- L1 节点文本 & 时间元数据列表
  l2_embeddings_{i}.npy  -- L2 节点嵌入矩阵 [N2_i, D]（第 i 个 L1 段）
  l2_metadata_{i}.json   -- L2 节点元数据列表
  l3_embeddings_{i}_{j}.npy  -- L3 节点嵌入矩阵 [N3_ij, D]
  l3_metadata_{i}_{j}.json   -- L3 节点元数据列表（含帧路径）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FrameNode:
    """L3 帧级视觉节点。

    代表金字塔的叶子层（最细粒度），保留原始像素级视觉信息。
    每个 FrameNode 对应视频中的一个具体时间点的关键帧。

    Attributes:
        frame_idx:       在所属 L2 片段内的帧序号（从 0 开始）。
        timestamp:       帧的绝对时间戳（秒，相对于视频起始）。
        frame_path:      帧图像文件路径（JPEG）。
        visual_embedding: CLIP 图像编码 + Proj 对齐后的视觉嵌入向量，
                          形状 [D]，L2 归一化，float32。
                          None 表示尚未提取。
    """

    frame_idx: int
    timestamp: float
    frame_path: str
    visual_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """序列化为可 JSON 存储的字典（嵌入向量不在此处存储）。"""
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "frame_path": self.frame_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameNode":
        """从字典反序列化（嵌入向量需单独加载）。"""
        return cls(
            frame_idx=d["frame_idx"],
            timestamp=d["timestamp"],
            frame_path=d["frame_path"],
        )


@dataclass
class ClipNode:
    """L2 片段级语义节点。

    代表金字塔的中间层，连接抽象叙事（L1）与具体视觉证据（L3）。
    每个 ClipNode 对应视频中 10-30 秒的短片段，
    使用 VLM 生成精细描述（如"角色 A 与 B 发生争执并摔门"）。

    Attributes:
        clip_idx:        在所属 L1 段内的片段序号（从 0 开始）。
        start_time:      片段起始时间（秒，绝对时间）。
        end_time:        片段终止时间（秒，绝对时间）。
        caption:         VLM 生成的片段文字描述（D_{i,j}）。
        text_embedding:  caption 的文本嵌入，形状 [D]，L2 归一化，float32。
                         None 表示尚未编码。
        frame_nodes:     所属的 L3 帧节点列表（M_L3^{(i,j)}）。
    """

    clip_idx: int
    start_time: float
    end_time: float
    caption: str = ""
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    frame_nodes: List[FrameNode] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """片段时长（秒）。"""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """序列化（嵌入向量和帧节点单独存储）。"""
        return {
            "clip_idx": self.clip_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "caption": self.caption,
            "num_frames": len(self.frame_nodes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClipNode":
        """从字典反序列化。"""
        return cls(
            clip_idx=d["clip_idx"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            caption=d.get("caption", ""),
        )


@dataclass
class SegmentNode:
    """L1 全局叙事节点。

    代表金字塔的根层（最粗粒度），包含宏观事件的高层语义。
    每个 SegmentNode 对应视频中 10-20 分钟的事件段，
    使用 VLM 生成叙事摘要（如"主角在学校的场景"）。

    Attributes:
        seg_idx:         视频内的段序号（从 0 开始）。
        start_time:      段起始时间（秒，绝对时间）。
        end_time:        段终止时间（秒，绝对时间）。
        summary:         VLM 生成的高层摘要（S_i）。
        text_embedding:  summary 的文本嵌入，形状 [D]，L2 归一化，float32。
                         None 表示尚未编码。
        clip_nodes:      所属的 L2 片段节点列表（M_L2^{(i)}）。
    """

    seg_idx: int
    start_time: float
    end_time: float
    summary: str = ""
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    clip_nodes: List[ClipNode] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """段时长（秒）。"""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """序列化（嵌入向量和子节点单独存储）。"""
        return {
            "seg_idx": self.seg_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "num_clips": len(self.clip_nodes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentNode":
        """从字典反序列化。"""
        return cls(
            seg_idx=d["seg_idx"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            summary=d.get("summary", ""),
        )


# ---------------------------------------------------------------------------
# Hierarchical Semantic Pyramid
# ---------------------------------------------------------------------------

@dataclass
class PyramidMetadata:
    """金字塔元数据（存储于 metadata.json）。"""

    video_path: str
    video_name: str
    video_duration: float
    embed_dim: int
    text_backend: str
    clip_model: str
    num_l1_segments: int
    l1_segment_duration: float
    l2_clip_duration: float
    l3_fps: float
    created_at: str = ""

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PyramidMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


class HierarchicalSemanticPyramid:
    """分层语义金字塔（HSP）容器。

    组织和管理三层节点结构，提供高效的节点访问接口，
    并支持将整个金字塔持久化到磁盘及从磁盘恢复。

    典型工作流::

        # 1. 构建金字塔（由 VideoIndexer 填充节点）
        pyramid = HierarchicalSemanticPyramid(video_path="movie.mp4", ...)
        pyramid.add_segment(seg_node)

        # 2. 保存到磁盘
        pyramid.save("pyramid_cache/movie")

        # 3. 加载并用于推理
        pyramid = HierarchicalSemanticPyramid.load("pyramid_cache/movie")

        # 4. 访问嵌入矩阵（用于 Tree-TRM 检索）
        M_L1 = pyramid.get_l1_embeddings()           # [N1, D]
        M_L2 = pyramid.get_l2_embeddings(k1=2)        # [N2, D]
        M_L3 = pyramid.get_l3_embeddings(k1=2, k2=5)  # [N3, D]

    Attributes:
        video_path:    原始视频文件路径。
        video_name:    视频文件名（不含路径）。
        video_duration: 视频总时长（秒）。
        embed_dim:     嵌入向量维度。
        text_backend:  文本编码后端（"clip" 或 "ollama"）。
        clip_model:    CLIP 模型名称。
        segments:      L1 节点列表（按时间顺序排列）。
    """

    def __init__(
        self,
        video_path: str,
        video_duration: float,
        embed_dim: int,
        text_backend: str = "clip",
        clip_model: str = "ViT-B/32",
        l1_segment_duration: float = 600.0,
        l2_clip_duration: float = 20.0,
        l3_fps: float = 1.0,
    ) -> None:
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.video_duration = video_duration
        self.embed_dim = embed_dim
        self.text_backend = text_backend
        self.clip_model = clip_model
        self.l1_segment_duration = l1_segment_duration
        self.l2_clip_duration = l2_clip_duration
        self.l3_fps = l3_fps
        self.segments: List[SegmentNode] = []

    # ------------------------------------------------------------------ #
    # Node Population                                                      #
    # ------------------------------------------------------------------ #

    def add_segment(self, seg: SegmentNode) -> None:
        """添加一个 L1 段节点。"""
        self.segments.append(seg)

    # ------------------------------------------------------------------ #
    # Embedding Matrix Accessors                                           #
    # ------------------------------------------------------------------ #

    def get_l1_embeddings(self) -> np.ndarray:
        """返回所有 L1 节点的文本嵌入矩阵 M_L1。

        Returns:
            形状 [N1, D] 的 float32 矩阵。若节点无嵌入则行为零向量。
        """
        vecs = []
        for seg in self.segments:
            if seg.text_embedding is not None:
                vecs.append(seg.text_embedding)
            else:
                vecs.append(np.zeros(self.embed_dim, dtype=np.float32))
        if not vecs:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return np.stack(vecs, axis=0)  # [N1, D]

    def get_l2_embeddings(self, k1: int) -> np.ndarray:
        """返回第 k1 个 L1 段内所有 L2 节点的文本嵌入矩阵 M_L2^{(k1)}。

        Args:
            k1: L1 段索引。

        Returns:
            形状 [N2, D] 的 float32 矩阵。
        """
        if k1 < 0 or k1 >= len(self.segments):
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        vecs = []
        for clip in self.segments[k1].clip_nodes:
            if clip.text_embedding is not None:
                vecs.append(clip.text_embedding)
            else:
                vecs.append(np.zeros(self.embed_dim, dtype=np.float32))
        if not vecs:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return np.stack(vecs, axis=0)  # [N2, D]

    def get_l3_embeddings(self, k1: int, k2: int) -> np.ndarray:
        """返回第 (k1, k2) 片段内所有 L3 帧的视觉嵌入矩阵 M_L3^{(k1,k2)}。

        Args:
            k1: L1 段索引。
            k2: L2 片段索引（相对于 L1 段）。

        Returns:
            形状 [N3, D] 的 float32 矩阵。
        """
        if k1 < 0 or k1 >= len(self.segments):
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        seg = self.segments[k1]
        if k2 < 0 or k2 >= len(seg.clip_nodes):
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        vecs = []
        for frame in seg.clip_nodes[k2].frame_nodes:
            if frame.visual_embedding is not None:
                vecs.append(frame.visual_embedding)
            else:
                vecs.append(np.zeros(self.embed_dim, dtype=np.float32))
        if not vecs:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return np.stack(vecs, axis=0)  # [N3, D]

    def get_frame_node(self, k1: int, k2: int, k3: int) -> Optional[FrameNode]:
        """按三级索引精确定位某个 L3 帧节点。

        Args:
            k1: L1 段索引。
            k2: L2 片段索引。
            k3: L3 帧索引。

        Returns:
            对应的 FrameNode；若索引越界则返回 None。
        """
        try:
            return self.segments[k1].clip_nodes[k2].frame_nodes[k3]
        except IndexError:
            return None

    def get_clip_node(self, k1: int, k2: int) -> Optional[ClipNode]:
        """按二级索引定位 L2 片段节点。"""
        try:
            return self.segments[k1].clip_nodes[k2]
        except IndexError:
            return None

    def get_segment_node(self, k1: int) -> Optional[SegmentNode]:
        """按一级索引定位 L1 段节点。"""
        try:
            return self.segments[k1]
        except IndexError:
            return None

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, int]:
        """返回金字塔统计信息。

        Returns:
            字典包含：
              - n_segments: L1 节点数量。
              - n_clips:    L2 节点总数量。
              - n_frames:   L3 节点总数量。
        """
        n_clips = sum(len(seg.clip_nodes) for seg in self.segments)
        n_frames = sum(
            len(clip.frame_nodes)
            for seg in self.segments
            for clip in seg.clip_nodes
        )
        return {
            "n_segments": len(self.segments),
            "n_clips": n_clips,
            "n_frames": n_frames,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"HierarchicalSemanticPyramid("
            f"video='{self.video_name}', "
            f"duration={self.video_duration:.1f}s, "
            f"L1={s['n_segments']}, L2={s['n_clips']}, L3={s['n_frames']}, "
            f"embed_dim={self.embed_dim})"
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, save_dir: str) -> None:
        """将金字塔持久化到磁盘。

        目录结构::

            save_dir/
            ├── metadata.json
            ├── l1_embeddings.npy          [N1, D]
            ├── l1_metadata.json
            ├── l2_embeddings_0.npy        [N2_0, D]
            ├── l2_metadata_0.json
            ├── l3_embeddings_0_0.npy      [N3_00, D]
            ├── l3_metadata_0_0.json
            └── ...

        Args:
            save_dir: 保存目录路径（自动创建）。
        """
        import datetime

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # ── 元数据 ──────────────────────────────────────────────────────
        meta = PyramidMetadata(
            video_path=self.video_path,
            video_name=self.video_name,
            video_duration=self.video_duration,
            embed_dim=self.embed_dim,
            text_backend=self.text_backend,
            clip_model=self.clip_model,
            num_l1_segments=len(self.segments),
            l1_segment_duration=self.l1_segment_duration,
            l2_clip_duration=self.l2_clip_duration,
            l3_fps=self.l3_fps,
            created_at=datetime.datetime.now().isoformat(),
        )
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

        # ── L1 ──────────────────────────────────────────────────────────
        l1_embeds = self.get_l1_embeddings()
        np.save(str(save_path / "l1_embeddings.npy"), l1_embeds)
        l1_meta = [seg.to_dict() for seg in self.segments]
        with open(save_path / "l1_metadata.json", "w", encoding="utf-8") as f:
            json.dump(l1_meta, f, indent=2, ensure_ascii=False)

        # ── L2 & L3 ─────────────────────────────────────────────────────
        for i, seg in enumerate(self.segments):
            l2_embeds = self.get_l2_embeddings(i)
            np.save(str(save_path / f"l2_embeddings_{i}.npy"), l2_embeds)
            l2_meta = [clip.to_dict() for clip in seg.clip_nodes]
            with open(save_path / f"l2_metadata_{i}.json", "w", encoding="utf-8") as f:
                json.dump(l2_meta, f, indent=2, ensure_ascii=False)

            for j, clip in enumerate(seg.clip_nodes):
                l3_embeds = self.get_l3_embeddings(i, j)
                np.save(str(save_path / f"l3_embeddings_{i}_{j}.npy"), l3_embeds)
                l3_meta = [fr.to_dict() for fr in clip.frame_nodes]
                with open(
                    save_path / f"l3_metadata_{i}_{j}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(l3_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"金字塔已保存至 {save_dir}，统计信息：{self.stats()}")

    @classmethod
    def load(cls, load_dir: str) -> "HierarchicalSemanticPyramid":
        """从磁盘加载已序列化的金字塔。

        Args:
            load_dir: 金字塔保存目录路径。

        Returns:
            完整加载的 HierarchicalSemanticPyramid 实例。

        Raises:
            FileNotFoundError: 若目录或关键文件不存在。
        """
        load_path = Path(load_dir)
        if not load_path.exists():
            raise FileNotFoundError(f"金字塔目录不存在：{load_dir}")

        # ── 元数据 ──────────────────────────────────────────────────────
        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = PyramidMetadata.from_dict(meta_dict)

        pyramid = cls(
            video_path=meta.video_path,
            video_duration=meta.video_duration,
            embed_dim=meta.embed_dim,
            text_backend=meta.text_backend,
            clip_model=meta.clip_model,
            l1_segment_duration=meta.l1_segment_duration,
            l2_clip_duration=meta.l2_clip_duration,
            l3_fps=meta.l3_fps,
        )

        # ── L1 ──────────────────────────────────────────────────────────
        l1_embeds: np.ndarray = np.load(str(load_path / "l1_embeddings.npy"))
        with open(load_path / "l1_metadata.json", "r", encoding="utf-8") as f:
            l1_meta_list: list = json.load(f)

        # ── L2 & L3 ─────────────────────────────────────────────────────
        for i, seg_dict in enumerate(l1_meta_list):
            seg = SegmentNode.from_dict(seg_dict)
            if i < l1_embeds.shape[0]:
                seg.text_embedding = l1_embeds[i].astype(np.float32)

            l2_path = load_path / f"l2_embeddings_{i}.npy"
            l2_meta_path = load_path / f"l2_metadata_{i}.json"

            if l2_path.exists() and l2_meta_path.exists():
                l2_embeds: np.ndarray = np.load(str(l2_path))
                with open(l2_meta_path, "r", encoding="utf-8") as f:
                    l2_meta_list: list = json.load(f)

                for j, clip_dict in enumerate(l2_meta_list):
                    clip = ClipNode.from_dict(clip_dict)
                    if j < l2_embeds.shape[0]:
                        clip.text_embedding = l2_embeds[j].astype(np.float32)

                    l3_path = load_path / f"l3_embeddings_{i}_{j}.npy"
                    l3_meta_path = load_path / f"l3_metadata_{i}_{j}.json"

                    if l3_path.exists() and l3_meta_path.exists():
                        l3_embeds: np.ndarray = np.load(str(l3_path))
                        with open(l3_meta_path, "r", encoding="utf-8") as f:
                            l3_meta_list: list = json.load(f)

                        for k, frame_dict in enumerate(l3_meta_list):
                            frame = FrameNode.from_dict(frame_dict)
                            if k < l3_embeds.shape[0]:
                                frame.visual_embedding = l3_embeds[k].astype(np.float32)
                            clip.frame_nodes.append(frame)

                    seg.clip_nodes.append(clip)

            pyramid.segments.append(seg)

        logger.info(f"金字塔已从 {load_dir} 加载，统计信息：{pyramid.stats()}")
        return pyramid

    @staticmethod
    def exists(save_dir: str) -> bool:
        """检查指定目录是否存在有效的已保存金字塔。"""
        p = Path(save_dir)
        return (p / "metadata.json").exists() and (p / "l1_embeddings.npy").exists()

