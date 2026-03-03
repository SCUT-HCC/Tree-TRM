"""
Video-Tree-TRM RAG
==================
专为长视频理解设计的递归式检索增强生成（RAG）框架。

主要模块：
  - config            : 系统配置数据类
  - video_indexer     : 视频帧提取 & VLM 描述生成
  - video_pyramid     : 分层语义金字塔数据结构 & 持久化
  - visual_projection : CLIP 视觉特征 → 文本潜空间对齐投影层
  - video_tree_trm    : 三阶段递归检索引擎（核心）
  - answer_generator  : 基于检索关键帧的 VLM 答案生成
  - pipeline          : 端到端 VideoQA 推理管线
"""

from .config import (
    PyramidConfig,
    EmbeddingConfig,
    VLMConfig,
    AnswerConfig,
    RetrievalConfig,
    VideoTreeTRMConfig,
)
from .video_pyramid import (
    FrameNode,
    ClipNode,
    SegmentNode,
    HierarchicalSemanticPyramid,
)
from .visual_projection import VisualProjectionLayer
from .video_tree_trm import VideoTreeTRM
from .answer_generator import AnswerGenerator
from .pipeline import VideoQAPipeline

__version__ = "1.0.0"
__all__ = [
    "PyramidConfig",
    "EmbeddingConfig",
    "VLMConfig",
    "AnswerConfig",
    "RetrievalConfig",
    "VideoTreeTRMConfig",
    "FrameNode",
    "ClipNode",
    "SegmentNode",
    "HierarchicalSemanticPyramid",
    "VisualProjectionLayer",
    "VideoTreeTRM",
    "AnswerGenerator",
    "VideoQAPipeline",
]

