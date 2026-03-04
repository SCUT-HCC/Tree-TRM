"""
Video-Tree-TRM 端到端推理管线（Pipeline）
==========================================
VideoQAPipeline 是整个 Video-Tree-TRM 系统的统一入口，
将以下各模块串联为一个完整的端到端视频问答工作流：

  ┌──────────────────────────────────────────────────────────┐
  │           VideoQAPipeline                                │
  │                                                          │
  │  ┌──────────────┐   ┌───────────────┐   ┌────────────┐  │
  │  │VideoIndexer  │ → │HierarchicalS  │ → │VideoTreeTRM│  │
  │  │（帧提取/VLM） │   │emanticPyramid │   │（三阶段检索）│  │
  │  └──────────────┘   │（金字塔存储）  │   └────────────┘  │
  │                      └───────────────┘        │          │
  │                                               ↓          │
  │                                    ┌──────────────────┐  │
  │                                    │AnswerGenerator   │  │
  │                                    │（VLM 答案生成）   │  │
  │                                    └──────────────────┘  │
  └──────────────────────────────────────────────────────────┘

两种使用模式：
  1. **完整模式**（含预处理）：从原始视频文件开始，自动构建 HSP 并执行问答。
  2. **仅检索模式**（预处理离线）：加载已构建好的 HSP，直接执行问答。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .answer_generator import AnswerGenerator
from .config import VideoTreeTRMConfig
from .video_indexer import (
    CLIPFeatureExtractor,
    OllamaTextEmbedder,
    QwenKeyframeScorer,
    QwenSceneDetector,
    VideoFrameExtractor,
    VLMDescriptionGenerator,
    get_video_duration,
    sample_representative_frames,
    segment_video,
    segment_video_smart,
)
from .video_pyramid import ClipNode, FrameNode, HierarchicalSemanticPyramid, SegmentNode
from .video_tree_trm import RetrievalTrace, VideoTreeTRM
from .visual_projection import ProjectionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VideoQA Result
# ---------------------------------------------------------------------------

@dataclass
class VideoQAResult:
    """VideoQA 完整结果。

    Attributes:
        query:          用户原始问题。
        answer:         VLM 生成的最终答案。
        trace:          Video-Tree-TRM 检索轨迹（含三级索引和分数分布）。
        video_name:     视频文件名。
        elapsed_sec:    总推理耗时（秒）。
        success:        是否成功完成完整三阶段检索。
    """

    query: str
    answer: str
    trace: Optional[RetrievalTrace] = None
    video_name: str = ""
    elapsed_sec: float = 0.0
    success: bool = False

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 输出）。"""
        return {
            "query": self.query,
            "answer": self.answer,
            "video_name": self.video_name,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "success": self.success,
            "retrieval": {
                "k1_star": self.trace.k1_star if self.trace else -1,
                "k2_star": self.trace.k2_star if self.trace else -1,
                "k3_star": self.trace.k3_star if self.trace else -1,
                "target_timestamp": (
                    self.trace.target_timestamp if self.trace else -1.0
                ),
                "target_frame_path": (
                    self.trace.target_frame.frame_path
                    if (self.trace and self.trace.target_frame)
                    else ""
                ),
                "segment_summary": self.trace.segment_summary if self.trace else "",
                "clip_caption": self.trace.clip_caption if self.trace else "",
            },
        }


# ---------------------------------------------------------------------------
# VideoQAPipeline
# ---------------------------------------------------------------------------

class VideoQAPipeline:
    """Video-Tree-TRM 端到端视频问答管线。

    该类将所有子模块（视频索引、金字塔构建、Tree-TRM 检索、VLM 答案生成）
    整合为一个统一接口，只需提供视频路径和问题即可获得答案。

    Quick Start::

        from video_tree_trmRAG import VideoQAPipeline, VideoTreeTRMConfig

        cfg = VideoTreeTRMConfig.from_dict({
            "vlm": {"backend": "ollama", "ollama_vlm_model": "llava"},
            "answer": {"backend": "deepseek", "deepseek_api_key": "sk-xxx"},
        })

        pipeline = VideoQAPipeline(cfg)

        # 模式 1：完整流程（含预处理）
        result = pipeline.run(
            video_path="movie.mp4",
            query="What is written on the door poster?",
        )
        print(result.answer)

        # 模式 2：仅检索（预处理已完成）
        result = pipeline.run_from_pyramid(
            pyramid_dir="pyramid_cache/movie",
            query="What is written on the door poster?",
        )
        print(result.answer)

    Args:
        config: VideoTreeTRMConfig 实例，控制所有子模块行为。
    """

    def __init__(self, config: VideoTreeTRMConfig) -> None:
        self.config = config
        self._pyramid: Optional[HierarchicalSemanticPyramid] = None

        # 子模块延迟初始化（避免加载不需要的模型）
        self._clip_extractor: Optional[CLIPFeatureExtractor] = None
        self._text_embedder = None
        self._vlm_generator: Optional[VLMDescriptionGenerator] = None
        self._projection_manager: Optional[ProjectionManager] = None
        self._retriever: Optional[VideoTreeTRM] = None
        self._answer_gen: Optional[AnswerGenerator] = None

        self._setup_logging()

    # ------------------------------------------------------------------ #
    # Lazy Initialization                                                  #
    # ------------------------------------------------------------------ #

    def _setup_logging(self) -> None:
        """配置日志级别。"""
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _get_clip_extractor(self) -> CLIPFeatureExtractor:
        if self._clip_extractor is None:
            self._clip_extractor = CLIPFeatureExtractor(
                model_name=self.config.embedding.clip_model,
                device=self.config.embedding.clip_device,
            )
        return self._clip_extractor

    def _get_text_embedder(self):
        """返回文本嵌入器（CLIP text encoder 或 Ollama embedder）。"""
        if self._text_embedder is None:
            if self.config.embedding.text_backend == "clip":
                self._text_embedder = self._get_clip_extractor()
            else:  # ollama
                self._text_embedder = OllamaTextEmbedder(
                    base_url=self.config.embedding.ollama_embed_url,
                    model=self.config.embedding.ollama_embed_model,
                )
        return self._text_embedder

    def _get_vlm_generator(self) -> VLMDescriptionGenerator:
        if self._vlm_generator is None:
            self._vlm_generator = VLMDescriptionGenerator(
                backend=self.config.vlm.backend,
                ollama_chat_url=self.config.vlm.ollama_chat_url,
                ollama_model=self.config.vlm.ollama_vlm_model,
                openai_api_key=self.config.vlm.openai_api_key,
                openai_api_url=self.config.vlm.openai_api_url,
                openai_model=self.config.vlm.openai_vlm_model,
                qwen_api_key=self.config.vlm.qwen_api_key,
                qwen_api_url=self.config.vlm.qwen_api_url,
                qwen_model=self.config.vlm.qwen_vlm_model,
                max_tokens=self.config.vlm.max_tokens,
                temperature=self.config.vlm.temperature,
                timeout=self.config.vlm.timeout,
            )
        return self._vlm_generator

    def _get_projection_manager(self) -> ProjectionManager:
        if self._projection_manager is None:
            emb_cfg = self.config.embedding
            self._projection_manager = ProjectionManager(
                visual_dim=emb_cfg.clip_visual_dim,
                text_dim=emb_cfg.embed_dim,
                checkpoint=emb_cfg.projection_checkpoint,
                device=self.config.device,
            )
        return self._projection_manager

    def _get_retriever(self) -> VideoTreeTRM:
        if self._retriever is None:
            emb_cfg = self.config.embedding
            ret_cfg = self.config.retrieval
            self._retriever = VideoTreeTRM(
                embed_dim=emb_cfg.embed_dim,
                text_backend=emb_cfg.text_backend,
                clip_model=emb_cfg.clip_model,
                clip_device=emb_cfg.clip_device,
                ollama_embed_url=emb_cfg.ollama_embed_url,
                ollama_embed_model=emb_cfg.ollama_embed_model,
                selection_mode=ret_cfg.selection_mode,
                softmax_temperature=ret_cfg.softmax_temperature,
                state_update_mode=ret_cfg.state_update_mode,
                normalize_state=ret_cfg.normalize_state,
                init_state_mode=ret_cfg.init_state_mode,
                verbose=self.config.verbose,
            )
        return self._retriever

    def _get_answer_generator(self) -> AnswerGenerator:
        if self._answer_gen is None:
            ans_cfg = self.config.answer
            self._answer_gen = AnswerGenerator(
                backend=ans_cfg.backend,
                deepseek_api_key=ans_cfg.deepseek_api_key,
                deepseek_url=ans_cfg.deepseek_url,
                deepseek_model=ans_cfg.deepseek_model,
                openai_api_key=ans_cfg.openai_api_key,
                openai_url=ans_cfg.openai_url,
                openai_model=ans_cfg.openai_model,
                ollama_chat_url=ans_cfg.ollama_chat_url,
                ollama_model=ans_cfg.ollama_model,
                qwen_api_key=ans_cfg.qwen_api_key,
                qwen_url=ans_cfg.qwen_api_url,
                qwen_model=ans_cfg.qwen_answer_model,
                max_tokens=ans_cfg.max_tokens,
                temperature=ans_cfg.temperature,
                timeout=ans_cfg.timeout,
                prompt_template=ans_cfg.answer_prompt_template,
                verbose=self.config.verbose,
            )
        return self._answer_gen

    # ------------------------------------------------------------------ #
    # Pyramid Construction                                                 #
    # ------------------------------------------------------------------ #

    def build_pyramid(
        self,
        video_path: str,
        save_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> HierarchicalSemanticPyramid:
        """从原始视频构建分层语义金字塔（HSP）。

        这是离线预处理阶段，包括：
          1. 提取所有帧（L3）
          2. 切分时间段 → L1 / L2 区间
          3. 为每个 L1/L2 段采样代表帧，调用 VLM 生成文字描述
          4. 文本嵌入（L1/L2）和视觉嵌入 + 投影（L3）
          5. 保存金字塔到磁盘

        Args:
            video_path:    视频文件路径。
            save_dir:      金字塔保存目录（None 则使用配置中的 cache_dir）。
            force_rebuild: 是否强制重建（忽略现有缓存）。

        Returns:
            构建完成的 HierarchicalSemanticPyramid 实例。
        """
        pyr_cfg = self.config.pyramid
        emb_cfg = self.config.embedding

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if save_dir is None:
            save_dir = os.path.join(pyr_cfg.cache_dir, video_name)

        # 检查缓存
        if not force_rebuild and HierarchicalSemanticPyramid.exists(save_dir):
            logger.info(f"检测到已有金字塔缓存：{save_dir}，直接加载。")
            pyramid = HierarchicalSemanticPyramid.load(save_dir)
            self._pyramid = pyramid
            return pyramid

        logger.info(f"开始构建金字塔：视频={video_path}")
        t_start = time.time()

        # ── 获取视频时长 ───────────────────────────────────────────────
        duration = get_video_duration(video_path)
        logger.info(f"视频时长：{duration:.1f}s ({duration/60:.1f} min)")

        # ── 初始化工具 ─────────────────────────────────────────────────
        frames_dir = os.path.join(save_dir, pyr_cfg.frames_subdir)
        frame_extractor = VideoFrameExtractor(
            output_dir=frames_dir,
            max_long_side=pyr_cfg.max_frame_long_side,
        )
        vlm_gen = self._get_vlm_generator()
        clip_extractor = self._get_clip_extractor()
        proj_manager = self._get_projection_manager()

        # 文本嵌入函数（根据后端选择）
        def embed_text(text: str) -> list:
            if emb_cfg.text_backend == "clip":
                return clip_extractor.encode_texts([text], normalize=True)
            else:
                emb = OllamaTextEmbedder(
                    emb_cfg.ollama_embed_url, emb_cfg.ollama_embed_model
                )
                return emb.encode([text])

        # ── Qwen 智能组件初始化（可选） ────────────────────────────────
        _qwen_api_key = self.config.vlm.qwen_api_key
        _qwen_api_url = self.config.vlm.qwen_api_url
        _qwen_vlm_model = self.config.vlm.qwen_vlm_model

        qwen_scene_detector: Optional[QwenSceneDetector] = None
        if pyr_cfg.use_qwen_scene_detection:
            if not _qwen_api_key:
                logger.warning(
                    "use_qwen_scene_detection=True 但 qwen_api_key 为空，"
                    "已自动回退到固定时长切分。"
                )
            else:
                qwen_scene_detector = QwenSceneDetector(
                    qwen_api_key=_qwen_api_key,
                    qwen_api_url=_qwen_api_url,
                    qwen_model=_qwen_vlm_model,
                    timeout=self.config.vlm.timeout,
                )
                logger.info("✅ Qwen-VL 场景检测器已启用（智能 L1 切分）。")

        qwen_keyframe_scorer: Optional[QwenKeyframeScorer] = None
        if pyr_cfg.use_qwen_keyframe_scoring:
            if not _qwen_api_key:
                logger.warning(
                    "use_qwen_keyframe_scoring=True 但 qwen_api_key 为空，"
                    "已自动回退到均匀采样。"
                )
            else:
                qwen_keyframe_scorer = QwenKeyframeScorer(
                    qwen_api_key=_qwen_api_key,
                    qwen_api_url=_qwen_api_url,
                    qwen_model=_qwen_vlm_model,
                    timeout=self.config.vlm.timeout,
                )
                logger.info("✅ Qwen-VL 关键帧打分器已启用（Top-K L3 筛选）。")

        # ── 创建金字塔容器 ─────────────────────────────────────────────
        pyramid = HierarchicalSemanticPyramid(
            video_path=video_path,
            video_duration=duration,
            embed_dim=emb_cfg.embed_dim,
            text_backend=emb_cfg.text_backend,
            clip_model=emb_cfg.clip_model,
            l1_segment_duration=pyr_cfg.l1_segment_duration,
            l2_clip_duration=pyr_cfg.l2_clip_duration,
            l3_fps=pyr_cfg.l3_fps,
        )

        # ── 时间分段（固定步长 或 Qwen 场景感知） ─────────────────────
        if qwen_scene_detector is not None:
            logger.info(
                f"使用 Qwen-VL 场景检测切分（探测帧率={pyr_cfg.scene_detection_probe_fps}fps）..."
            )
            # Step 1：以探测帧率均匀采样全局探测帧
            probe_meta = frame_extractor.extract(
                video_path,
                fps=pyr_cfg.scene_detection_probe_fps,
                prefix="scene_probe",
            )
            probe_images_ts = []
            for _, ts, fpath in probe_meta:
                try:
                    from PIL import Image as _PIL
                    probe_images_ts.append((_PIL.open(fpath).convert("RGB"), ts))
                except Exception:
                    pass

            # Step 2：调用 Qwen-VL 检测场景边界
            boundary_timestamps = qwen_scene_detector.detect_boundaries(
                probe_images_ts,
                batch_size=pyr_cfg.scene_detection_batch_size,
            )
            logger.info(
                f"Qwen-VL 检测到 {len(boundary_timestamps)} 个场景边界：{boundary_timestamps}"
            )

            # Step 3：基于场景边界生成语义感知分段
            segments_info = segment_video_smart(
                video_duration=duration,
                boundary_timestamps=boundary_timestamps,
                l1_min_duration=pyr_cfg.l1_min_duration,
                l1_max_duration=pyr_cfg.l1_max_duration,
                l2_duration=pyr_cfg.l2_clip_duration,
                l1_max=pyr_cfg.l1_max_segments,
                l2_max_per_seg=pyr_cfg.l2_max_clips_per_segment,
            )
            logger.info(
                f"场景感知分段：{len(segments_info)} 个 L1 段（最小={pyr_cfg.l1_min_duration}s，"
                f"最大={pyr_cfg.l1_max_duration}s）。"
            )
        else:
            segments_info = segment_video(
                video_duration=duration,
                l1_duration=pyr_cfg.l1_segment_duration,
                l2_duration=pyr_cfg.l2_clip_duration,
                l1_max=pyr_cfg.l1_max_segments,
                l2_max_per_seg=pyr_cfg.l2_max_clips_per_segment,
            )
            logger.info(
                f"固定时长分段：{len(segments_info)} 个 L1 段，"
                f"每段最多 {pyr_cfg.l2_max_clips_per_segment} 个 L2 片段。"
            )

        # ── 逐段构建金字塔 ─────────────────────────────────────────────
        for seg_idx, (l1_start, l1_end, clips_info) in enumerate(segments_info):
            logger.info(
                f"处理 L1[{seg_idx}]：{l1_start:.0f}s – {l1_end:.0f}s "
                f"（{(l1_end-l1_start)/60:.1f} min）"
            )

            # L1 代表帧采样（用于 VLM 摘要）
            l1_rep_meta = frame_extractor.extract(
                video_path,
                fps=pyr_cfg.l1_sample_fps,
                start_sec=l1_start,
                end_sec=l1_end,
                prefix=f"l1_{seg_idx:03d}_rep",
            )
            l1_rep_frames = sample_representative_frames(
                l1_rep_meta, pyr_cfg.l1_max_frames_for_vlm
            )

            # L1 VLM 摘要生成
            summary = vlm_gen.describe(l1_rep_frames, self.config.vlm.summary_prompt)
            logger.info(f"  L1[{seg_idx}] 摘要：{summary[:80]}...")

            # L1 文本嵌入
            l1_emb_arr = embed_text(summary if summary else f"Segment {seg_idx}")
            l1_emb = l1_emb_arr[0].astype("float32") if l1_emb_arr is not None else None

            seg_node = SegmentNode(
                seg_idx=seg_idx,
                start_time=l1_start,
                end_time=l1_end,
                summary=summary,
                text_embedding=l1_emb,
            )

            # ── 逐片段构建 L2/L3 ──────────────────────────────────────
            for clip_idx, (l2_start, l2_end) in enumerate(clips_info):
                # L2 代表帧采样（用于 VLM 描述）
                l2_rep_meta = frame_extractor.extract(
                    video_path,
                    fps=pyr_cfg.l1_sample_fps,  # 低采样率
                    start_sec=l2_start,
                    end_sec=l2_end,
                    prefix=f"l2_{seg_idx:03d}_{clip_idx:04d}_rep",
                )
                l2_rep_frames = sample_representative_frames(
                    l2_rep_meta, pyr_cfg.l2_max_frames_for_vlm
                )

                # L2 VLM 描述生成
                caption = vlm_gen.describe(
                    l2_rep_frames, self.config.vlm.caption_prompt
                )

                # L2 文本嵌入
                l2_emb_arr = embed_text(caption if caption else f"Clip {clip_idx}")
                l2_emb = l2_emb_arr[0].astype("float32") if l2_emb_arr is not None else None

                clip_node = ClipNode(
                    clip_idx=clip_idx,
                    start_time=l2_start,
                    end_time=l2_end,
                    caption=caption,
                    text_embedding=l2_emb,
                )

                # ── L3 帧提取 & CLIP 编码 ──────────────────────────────
                l3_frames_meta = frame_extractor.extract(
                    video_path,
                    fps=pyr_cfg.l3_fps,
                    start_sec=l2_start,
                    end_sec=l2_end,
                    prefix=f"l3_{seg_idx:03d}_{clip_idx:04d}",
                )
                # 硬上限：截断到最大帧数
                l3_frames_meta = l3_frames_meta[: pyr_cfg.l3_max_frames_per_clip]

                if l3_frames_meta:
                    # 加载所有 L3 帧图像
                    from PIL import Image as PILImage

                    l3_images = []
                    for _, _, fpath in l3_frames_meta:
                        try:
                            l3_images.append(PILImage.open(fpath).convert("RGB"))
                        except Exception:
                            l3_images.append(None)  # type: ignore[arg-type]

                    # ── Qwen-VL 关键帧打分筛选（可选） ────────────────
                    if qwen_keyframe_scorer is not None and len(l3_frames_meta) > 1:
                        # 仅对非 None 的帧打分
                        valid_meta = [
                            m for m, img in zip(l3_frames_meta, l3_images)
                            if img is not None
                        ]
                        valid_imgs_for_score = [
                            img for img in l3_images if img is not None
                        ]
                        top_k = max(
                            1,
                            round(len(valid_imgs_for_score) * pyr_cfg.qwen_keyframe_keep_ratio),
                        )
                        filtered_meta, filtered_imgs = qwen_keyframe_scorer.filter_top_k(
                            valid_meta, valid_imgs_for_score, top_k
                        )
                        logger.debug(
                            f"    L3[{seg_idx},{clip_idx}] 关键帧筛选："
                            f"{len(valid_imgs_for_score)} → {len(filtered_imgs)} 帧"
                        )
                        # 用筛选后的帧替换原始列表
                        l3_frames_meta = filtered_meta
                        l3_images_valid = filtered_imgs
                        valid_indices = list(range(len(filtered_imgs)))
                    else:
                        l3_images_valid = [img for img in l3_images if img is not None]
                        valid_indices = [
                            i for i, img in enumerate(l3_images) if img is not None
                        ]

                    # CLIP 视觉编码（批量）
                    if l3_images_valid:
                        raw_visual_embeds = clip_extractor.encode_images(
                            l3_images_valid, normalize=True
                        )
                        # 投影：视觉空间 → 文本空间
                        projected_embeds = proj_manager.project(raw_visual_embeds)
                    else:
                        projected_embeds = None

                    # 构建 L3 帧节点
                    for frame_local_idx, (_, ts, fpath) in enumerate(l3_frames_meta):
                        vis_emb = None
                        if (
                            projected_embeds is not None
                            and frame_local_idx < len(projected_embeds)
                        ):
                            vis_emb = projected_embeds[frame_local_idx]

                        frame_node = FrameNode(
                            frame_idx=frame_local_idx,
                            timestamp=ts,
                            frame_path=fpath,
                            visual_embedding=vis_emb,
                        )
                        clip_node.frame_nodes.append(frame_node)

                seg_node.clip_nodes.append(clip_node)

            pyramid.add_segment(seg_node)

        # ── 保存金字塔 ─────────────────────────────────────────────────
        pyramid.save(save_dir)
        elapsed = time.time() - t_start
        logger.info(
            f"金字塔构建完成！耗时 {elapsed:.1f}s，统计：{pyramid.stats()}"
        )
        self._pyramid = pyramid
        return pyramid

    def load_pyramid(self, pyramid_dir: str) -> HierarchicalSemanticPyramid:
        """从磁盘加载预构建的金字塔。

        Args:
            pyramid_dir: 金字塔目录路径。

        Returns:
            已加载的 HierarchicalSemanticPyramid 实例。
        """
        pyramid = HierarchicalSemanticPyramid.load(pyramid_dir)
        self._pyramid = pyramid
        return pyramid

    # ------------------------------------------------------------------ #
    # Main Run Interface                                                   #
    # ------------------------------------------------------------------ #

    def run(
        self,
        video_path: str,
        query: str,
        pyramid_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> VideoQAResult:
        """完整 VideoQA 流程（含自动预处理）。

        Args:
            video_path:    视频文件路径。
            query:         用户问题。
            pyramid_dir:   金字塔缓存目录（None 则自动生成）。
            force_rebuild: 是否强制重建金字塔。

        Returns:
            VideoQAResult 实例，包含答案和完整检索轨迹。
        """
        t_start = time.time()
        video_name = os.path.basename(video_path)

        # 1. 构建或加载金字塔
        pyramid = self.build_pyramid(
            video_path, save_dir=pyramid_dir, force_rebuild=force_rebuild
        )

        # 2. 执行三阶段检索
        retriever = self._get_retriever()
        trace = retriever.retrieve(query, pyramid)

        # 3. 生成答案
        answer_gen = self._get_answer_generator()
        answer = answer_gen.generate(query, trace, video_name=video_name)

        elapsed = time.time() - t_start
        return VideoQAResult(
            query=query,
            answer=answer,
            trace=trace,
            video_name=video_name,
            elapsed_sec=elapsed,
            success=trace.is_valid(),
        )

    def run_from_pyramid(
        self,
        pyramid_dir: str,
        query: str,
    ) -> VideoQAResult:
        """仅推理模式：从已构建的金字塔执行 VideoQA（跳过预处理）。

        此模式适用于离线预处理已完成的场景，推理速度更快。

        Args:
            pyramid_dir: 金字塔目录路径。
            query:       用户问题。

        Returns:
            VideoQAResult 实例。
        """
        t_start = time.time()

        # 加载金字塔
        pyramid = self.load_pyramid(pyramid_dir)
        video_name = pyramid.video_name

        # 执行三阶段检索
        retriever = self._get_retriever()
        trace = retriever.retrieve(query, pyramid)

        # 生成答案
        answer_gen = self._get_answer_generator()
        answer = answer_gen.generate(query, trace, video_name=video_name)

        elapsed = time.time() - t_start
        return VideoQAResult(
            query=query,
            answer=answer,
            trace=trace,
            video_name=video_name,
            elapsed_sec=elapsed,
            success=trace.is_valid(),
        )

    def run_batch(
        self,
        pyramid_dir: str,
        queries: List[str],
    ) -> List[VideoQAResult]:
        """对同一视频的多个问题批量执行 VideoQA。

        Args:
            pyramid_dir: 金字塔目录路径（预处理已完成）。
            queries:     问题列表。

        Returns:
            VideoQAResult 列表，与 queries 一一对应。
        """
        # 金字塔只加载一次
        pyramid = self.load_pyramid(pyramid_dir)
        video_name = pyramid.video_name
        retriever = self._get_retriever()
        answer_gen = self._get_answer_generator()

        results: List[VideoQAResult] = []
        for i, query in enumerate(queries):
            t_start = time.time()
            logger.info(f"处理问题 {i+1}/{len(queries)}: '{query[:60]}...'")

            trace = retriever.retrieve(query, pyramid)
            answer = answer_gen.generate(query, trace, video_name=video_name)

            elapsed = time.time() - t_start
            results.append(
                VideoQAResult(
                    query=query,
                    answer=answer,
                    trace=trace,
                    video_name=video_name,
                    elapsed_sec=elapsed,
                    success=trace.is_valid(),
                )
            )

        return results

    # ------------------------------------------------------------------ #
    # Config Helpers                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config_file(cls, yaml_path: str) -> "VideoQAPipeline":
        """从 YAML 配置文件创建 Pipeline 实例。

        Args:
            yaml_path: YAML 配置文件路径。

        Returns:
            VideoQAPipeline 实例。
        """
        config = VideoTreeTRMConfig.from_yaml(yaml_path)
        return cls(config)

    @classmethod
    def from_config_dict(cls, d: dict) -> "VideoQAPipeline":
        """从字典创建 Pipeline 实例。

        Args:
            d: 配置字典。

        Returns:
            VideoQAPipeline 实例。
        """
        config = VideoTreeTRMConfig.from_dict(d)
        return cls(config)

