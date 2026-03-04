"""
视频索引器（Video Indexer）
============================
负责从原始视频文件中提取所有必要的原始素材，为分层语义金字塔（HSP）构建做准备：

  1. **帧提取**：以指定帧率从视频中提取关键帧并保存为 JPEG 图像。
  2. **时间分段**：将视频切分为 L1 宏观事件段（10-20 min）和 L2 短片段（10-30 s）。
  3. **代表帧采样**：为每个 L1/L2 段选取用于 VLM 描述的代表帧。
  4. **VLM 描述生成**：
       - L1 节点：调用多模态 VLM 生成高层叙事摘要（Summary）。
       - L2 节点：调用多模态 VLM 生成精细片段描述（Caption）。
  5. **CLIP 视觉特征提取**：对所有 L3 关键帧进行 CLIP 图像编码。

设计原则：
  - 帧提取和 VLM 调用分离，互不依赖，便于单独测试。
  - 所有网络调用均有超时和重试保护。
  - 支持断点续传（已有缓存的帧跳过重新提取）。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """将 PIL.Image 转为 base64 字符串（用于 VLM API 调用）。

    Args:
        image: PIL.Image 对象。
        fmt: 图像格式（JPEG 压缩率更小；PNG 无损）。
        quality: JPEG 压缩质量（0-100）。

    Returns:
        base64 编码的字符串（不含 data URI 前缀）。
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_image(image: Image.Image, max_long_side: int) -> Image.Image:
    """按比例缩放图像，使最长边不超过 max_long_side 像素。

    Args:
        image: 原始 PIL.Image。
        max_long_side: 最长边的像素上限。

    Returns:
        缩放后的 PIL.Image（若已满足要求则原样返回）。
    """
    w, h = image.size
    if max(w, h) <= max_long_side:
        return image
    scale = max_long_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Frame Extraction
# ---------------------------------------------------------------------------

class VideoFrameExtractor:
    """从视频文件中按帧率提取图像帧。

    优先使用 OpenCV（H.264/H.265 等主流格式），若 OpenCV 无法解码（如 AV1 编码视频），
    自动回退到 PyAV（libav，支持 AV1/libdav1d 等更多编解码器）。

    Args:
        output_dir: 保存帧图像的目录。
        max_long_side: 帧图像的最大长边（像素），超出则缩放。
        jpeg_quality: JPEG 保存质量（0-100）。
    """

    def __init__(
        self,
        output_dir: str,
        max_long_side: int = 336,
        jpeg_quality: int = 85,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_long_side = max_long_side
        self.jpeg_quality = jpeg_quality

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        video_path: str,
        fps: float = 1.0,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        prefix: str = "frame",
    ) -> List[Tuple[int, float, str]]:
        """提取视频指定时间区间内的帧。

        自动检测视频编解码器：H.264/H.265 使用 OpenCV；AV1 等格式自动回退到 PyAV。

        Args:
            video_path: 视频文件路径。
            fps: 提取帧率（frames per second）。
            start_sec: 提取起始时间（秒）。
            end_sec: 提取终止时间（秒）。None 表示提取到视频末尾。
            prefix: 输出文件名前缀。

        Returns:
            帧元数据列表，每项为 (frame_idx, timestamp_sec, file_path)。

        Raises:
            FileNotFoundError: 如果视频文件不存在。
            RuntimeError: 如果视频文件无法打开。
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在：{video_path}")

        # 先用 OpenCV 探测，若首帧读取失败则换用 PyAV
        if self._opencv_can_decode(video_path):
            return self._extract_opencv(video_path, fps, start_sec, end_sec, prefix)
        else:
            logger.info(
                f"OpenCV 无法解码 {video_path}，自动切换到 PyAV（AV1/libdav1d 等格式）。"
            )
            return self._extract_pyav(video_path, fps, start_sec, end_sec, prefix)

    # ------------------------------------------------------------------ #
    # Private: codec detection                                             #
    # ------------------------------------------------------------------ #

    def _opencv_can_decode(self, video_path: str) -> bool:
        """测试 OpenCV 能否读取该视频的第一帧（屏蔽 stderr 的硬件加速警告）。"""
        import os
        import sys
        try:
            import cv2
        except ImportError:
            return False
        # 屏蔽 OpenCV/FFmpeg 写入 stderr 的硬件加速警告（如 AV1 不支持硬解）
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except Exception:
            return False
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)

    # ------------------------------------------------------------------ #
    # Private: OpenCV backend                                              #
    # ------------------------------------------------------------------ #

    def _extract_opencv(
        self,
        video_path: str,
        fps: float,
        start_sec: float,
        end_sec: Optional[float],
        prefix: str,
    ) -> List[Tuple[int, float, str]]:
        """使用 OpenCV 提取帧（适用于 H.264 / H.265 等格式）。"""
        import cv2

        cap = cv2.VideoCapture(video_path)
        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration: float = total_frames / video_fps

        end_sec = end_sec if end_sec is not None else video_duration
        end_sec = min(end_sec, video_duration)

        interval = 1.0 / fps
        timestamps = np.arange(start_sec, end_sec, interval)

        frames_meta: List[Tuple[int, float, str]] = []
        frame_counter = 0

        for ts in timestamps:
            frame_pos = int(ts * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, bgr = cap.read()
            if not ret:
                logger.warning(f"时间戳 {ts:.2f}s 处帧读取失败，跳过。")
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = _resize_image(img, self.max_long_side)

            fname = f"{prefix}_{frame_counter:06d}.jpg"
            fpath = str(self.output_dir / fname)
            if not os.path.exists(fpath):
                img.save(fpath, "JPEG", quality=self.jpeg_quality)

            frames_meta.append((frame_counter, float(ts), fpath))
            frame_counter += 1

        cap.release()
        logger.info(
            f"[OpenCV] 从 {video_path} 提取了 {len(frames_meta)} 帧"
            f"（{start_sec:.1f}s – {end_sec:.1f}s，{fps} fps）"
        )
        return frames_meta

    # ------------------------------------------------------------------ #
    # Private: PyAV backend (AV1 / libdav1d fallback)                     #
    # ------------------------------------------------------------------ #

    def _extract_pyav(
        self,
        video_path: str,
        fps: float,
        start_sec: float,
        end_sec: Optional[float],
        prefix: str,
    ) -> List[Tuple[int, float, str]]:
        """使用 PyAV 提取帧（支持 AV1 / libdav1d 等 OpenCV 不支持的格式）。"""
        try:
            import av  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyAV 未安装，无法解码 AV1 视频。请运行：pip install av"
            ) from exc

        container = av.open(video_path)
        video_stream = container.streams.video[0]

        # 获取视频时长
        if container.duration is not None:
            video_duration = float(container.duration) / av.time_base
        else:
            video_duration = float(video_stream.duration * video_stream.time_base)

        end_ts = end_sec if end_sec is not None else video_duration
        end_ts = min(end_ts, video_duration)

        interval = 1.0 / fps
        target_timestamps = list(np.arange(start_sec, end_ts, interval))
        if not target_timestamps:
            container.close()
            return []

        # seek 到起始位置（微秒单位）
        seek_ts = int(start_sec * av.time_base)  # avformat 单位：1/AV_TIME_BASE
        container.seek(seek_ts, any_frame=False, backward=True)

        frames_meta: List[Tuple[int, float, str]] = []
        frame_counter = 0
        target_idx = 0

        prev_frame: Optional[Image.Image] = None
        prev_ts: float = -1.0

        for packet in container.demux(video_stream):
            if target_idx >= len(target_timestamps):
                break
            for av_frame in packet.decode():
                frame_time = float(av_frame.pts * video_stream.time_base)
                if frame_time < start_sec - interval:
                    continue
                if frame_time > end_ts + interval:
                    break

                # 更新 prev_frame（当前最接近 target 的帧）
                img = av_frame.to_image().convert("RGB")
                prev_frame = img
                prev_ts = frame_time

                # 输出所有 target 时间戳 ≤ frame_time 的帧
                while (
                    target_idx < len(target_timestamps)
                    and target_timestamps[target_idx] <= frame_time + interval / 2
                ):
                    target_ts = target_timestamps[target_idx]
                    if prev_frame is not None:
                        out_img = _resize_image(prev_frame, self.max_long_side)
                        fname = f"{prefix}_{frame_counter:06d}.jpg"
                        fpath = str(self.output_dir / fname)
                        if not os.path.exists(fpath):
                            out_img.save(fpath, "JPEG", quality=self.jpeg_quality)
                        frames_meta.append((frame_counter, float(target_ts), fpath))
                        frame_counter += 1
                    target_idx += 1

        # 如果还有剩余目标时间戳（视频末尾），用最后一帧填充
        while target_idx < len(target_timestamps) and prev_frame is not None:
            target_ts = target_timestamps[target_idx]
            out_img = _resize_image(prev_frame, self.max_long_side)
            fname = f"{prefix}_{frame_counter:06d}.jpg"
            fpath = str(self.output_dir / fname)
            if not os.path.exists(fpath):
                out_img.save(fpath, "JPEG", quality=self.jpeg_quality)
            frames_meta.append((frame_counter, float(target_ts), fpath))
            frame_counter += 1
            target_idx += 1

        container.close()
        logger.info(
            f"[PyAV] 从 {video_path} 提取了 {len(frames_meta)} 帧"
            f"（{start_sec:.1f}s – {end_ts:.1f}s，{fps} fps）"
        )
        return frames_meta


# ---------------------------------------------------------------------------
# Temporal Segmentation
# ---------------------------------------------------------------------------

def segment_video(
    video_duration: float,
    l1_duration: float = 600.0,
    l2_duration: float = 20.0,
    l1_max: int = 50,
    l2_max_per_seg: int = 60,
) -> List[Tuple[float, float, List[Tuple[float, float]]]]:
    """将视频切分为两级时间区间。

    Args:
        video_duration: 视频总时长（秒）。
        l1_duration: L1 宏观事件段时长（秒）。
        l2_duration: L2 片段时长（秒）。
        l1_max: 最大 L1 段数量。
        l2_max_per_seg: 每个 L1 段内最大 L2 片段数量。

    Returns:
        嵌套列表，每项为::

            (l1_start, l1_end, [(l2_start, l2_end), ...])

        即每个 L1 段及其下辖的 L2 子片段列表。
    """
    segments: List[Tuple[float, float, List[Tuple[float, float]]]] = []

    l1_starts = np.arange(0.0, video_duration, l1_duration)
    l1_starts = l1_starts[:l1_max]

    for l1_s in l1_starts:
        l1_e = min(float(l1_s) + l1_duration, video_duration)

        # L2 clips within this L1 segment
        clips: List[Tuple[float, float]] = []
        l2_starts = np.arange(l1_s, l1_e, l2_duration)
        l2_starts = l2_starts[:l2_max_per_seg]

        for l2_s in l2_starts:
            l2_e = min(float(l2_s) + l2_duration, l1_e)
            clips.append((float(l2_s), float(l2_e)))

        segments.append((float(l1_s), float(l1_e), clips))

    return segments


def get_video_duration(video_path: str) -> float:
    """获取视频文件总时长（秒）。

    优先使用 OpenCV，若 OpenCV 无法读取（如 AV1 编码）则回退到 PyAV。

    Args:
        video_path: 视频文件路径。

    Returns:
        视频时长（秒）。

    Raises:
        RuntimeError: 若无法读取视频信息。
    """
    # 尝试 OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames > 0 and fps > 0:
                return n_frames / fps
        cap.release()
    except Exception:
        pass

    # 回退 PyAV（支持 AV1 等格式）
    try:
        import av  # type: ignore[import]
        container = av.open(video_path)
        if container.duration is not None:
            duration = float(container.duration) / av.time_base
        else:
            vs = container.streams.video[0]
            duration = float(vs.duration * vs.time_base)
        container.close()
        return duration
    except Exception as exc:
        raise RuntimeError(
            f"无法获取视频时长（OpenCV 和 PyAV 均失败）：{video_path}\n原因：{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Smart Segmentation（基于 Qwen-VL 场景检测）
# ---------------------------------------------------------------------------

def segment_video_smart(
    video_duration: float,
    boundary_timestamps: List[float],
    l1_min_duration: float = 60.0,
    l1_max_duration: float = 1200.0,
    l2_duration: float = 20.0,
    l1_max: int = 50,
    l2_max_per_seg: int = 60,
) -> List[tuple]:
    """基于 Qwen-VL 检测到的场景边界，生成语义感知的时间分段。

    与 segment_video() 的差异：
      - L1 边界由场景转换点决定（而非固定步长），
        语义连贯性更强，避免"切断正在进行的场景"。
      - 过短的段（< l1_min_duration）自动合并，
        过长的段（> l1_max_duration）自动均匀拆分。
      - L2 仍采用固定步长，保持下游行为一致。

    Args:
        video_duration:      视频总时长（秒）。
        boundary_timestamps: Qwen-VL 检测出的场景边界时间戳列表（秒）。
        l1_min_duration:     L1 段最小时长（秒），避免过于碎片化。
        l1_max_duration:     L1 段最大时长（秒），避免超大段。
        l2_duration:         L2 片段固定时长（秒）。
        l1_max:              最大 L1 段数量。
        l2_max_per_seg:      每个 L1 段内最大 L2 数量。

    Returns:
        与 segment_video() 格式相同的嵌套列表::

            [(l1_start, l1_end, [(l2_start, l2_end), ...]), ...]
    """
    # 构建候选 L1 边界（加入视频起止）
    raw = sorted(set([0.0] + list(boundary_timestamps) + [video_duration]))

    # 合并过短的段（贪心前向合并）
    merged_bounds = [0.0]
    for b in raw[1:]:
        if b - merged_bounds[-1] >= l1_min_duration:
            merged_bounds.append(b)
    if merged_bounds[-1] < video_duration:
        merged_bounds.append(video_duration)

    # 拆分过长的段
    l1_intervals: List[tuple] = []
    for i in range(len(merged_bounds) - 1):
        s, e = merged_bounds[i], merged_bounds[i + 1]
        dur = e - s
        if dur > l1_max_duration:
            n_splits = math.ceil(dur / l1_max_duration)
            sub_dur = dur / n_splits
            for k in range(n_splits):
                l1_intervals.append((s + k * sub_dur, s + (k + 1) * sub_dur))
        else:
            l1_intervals.append((s, e))

    # 截断到 l1_max
    l1_intervals = l1_intervals[:l1_max]

    # 为每个 L1 段构建 L2 子片段列表
    segments = []
    for l1_s, l1_e in l1_intervals:
        clips: List[tuple] = []
        l2_starts = np.arange(l1_s, l1_e, l2_duration)
        l2_starts = l2_starts[:l2_max_per_seg]
        for l2_s in l2_starts:
            l2_e = min(float(l2_s) + l2_duration, l1_e)
            clips.append((float(l2_s), float(l2_e)))
        segments.append((float(l1_s), float(l1_e), clips))

    return segments


# ---------------------------------------------------------------------------
# Qwen-VL Scene Detector
# ---------------------------------------------------------------------------

class QwenSceneDetector:
    """使用 Qwen-VL API 检测视频帧序列中的场景边界。

    通过将连续帧序列批量发送给 Qwen-VL，请求其识别相邻帧之间的场景转换点，
    替代固定时长切分策略，使 L1/L2 分段更符合视频内容的语义边界。

    工作流程：
      1. 以 scene_detection_probe_fps 采样探测帧（低帧率，降低 API 调用次数）。
      2. 以 batch_size 为窗口、batch_size-1 为步长滑动批次，发送给 Qwen-VL。
      3. Qwen-VL 以 JSON 数组形式返回批次内发生场景切换的帧索引。
      4. 将批次内局部索引映射回全局时间戳，汇总为场景边界列表。

    Args:
        qwen_api_key: 阿里云百炼 API Key。
        qwen_api_url: 百炼 OpenAI 兼容接口地址。
        qwen_model:   千问视觉模型名称（如 "qwen-vl-plus"）。
        timeout:      API 请求超时（秒）。
    """

    _PROMPT = (
        "You are analyzing a sequence of video frames captured at regular intervals. "
        "The frames are in chronological order: Frame 0, Frame 1, ..., Frame N-1. "
        "Identify which consecutive frame pairs show a SIGNIFICANT scene change "
        "(e.g., new location, major time skip, completely different environment or activity). "
        "Respond ONLY with a valid JSON array of 0-based frame indices after which a major "
        "scene change occurs. Example: if scene changes after Frame 2 and Frame 7, reply: [2, 7] "
        "If no significant scene change is detected, reply: [] "
        "IMPORTANT: Output ONLY the JSON array — no explanation, no extra text."
    )

    def __init__(
        self,
        qwen_api_key: str,
        qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        qwen_model: str = "qwen-vl-plus",
        timeout: int = 90,
    ) -> None:
        self.qwen_api_key = qwen_api_key
        self.qwen_api_url = qwen_api_url
        self.qwen_model = qwen_model
        self.timeout = timeout

    def detect_boundaries(
        self,
        frames_with_timestamps: List[tuple],
        batch_size: int = 8,
    ) -> List[float]:
        """检测场景边界，返回边界时间戳列表。

        Args:
            frames_with_timestamps: (PIL.Image, timestamp_sec) 元组列表，
                                    按时间升序排列。
            batch_size:             每批发送给 Qwen-VL 的帧数（含1帧首尾重叠）。

        Returns:
            场景边界时间戳列表（秒），表示在该时间点之后发生了场景切换。
            空列表表示未检测到场景切换。
        """
        if len(frames_with_timestamps) < 2:
            return []

        images = [f[0] for f in frames_with_timestamps]
        timestamps = [f[1] for f in frames_with_timestamps]
        n = len(images)

        boundary_timestamps: List[float] = []
        step = max(1, batch_size - 1)  # 相邻批次重叠 1 帧，保证连续性
        start = 0

        while start < n - 1:
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]
            batch_ts = timestamps[start:end]

            local_indices = self._call_qwen(batch_imgs)
            for local_idx in local_indices:
                global_idx = start + local_idx
                if 0 <= global_idx < n - 1:
                    # 取两帧时间戳的中点作为边界
                    boundary_timestamps.append(
                        (timestamps[global_idx] + timestamps[global_idx + 1]) / 2.0
                    )

            start += step

        return sorted(set(boundary_timestamps))

    def _call_qwen(self, frames: List[Image.Image]) -> List[int]:
        """单次 API 调用：返回当前批次内的场景边界帧索引列表。"""
        content: list = []
        for frame in frames:
            b64 = _pil_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": self._PROMPT})

        payload = {
            "model": self.qwen_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.qwen_api_url, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\[.*?\]", text, re.DOTALL)
            if m:
                indices = json.loads(m.group())
                n = len(frames)
                return [i for i in indices if isinstance(i, int) and 0 <= i < n - 1]
        except Exception as exc:
            logger.warning(f"[QwenSceneDetector] API 调用失败：{exc}，跳过此批次。")
        return []


# ---------------------------------------------------------------------------
# Qwen-VL Keyframe Scorer
# ---------------------------------------------------------------------------

class QwenKeyframeScorer:
    """使用 Qwen-VL 对 L3 候选帧进行信息密度打分，保留 Top-K 关键帧。

    在固定 fps 均匀采样之后，将一个 L2 clip 内的所有候选帧一次性发给
    Qwen-VL，由其根据动作重要性与视觉信息密度对帧排序，最终仅保留
    前 K 帧作为 L3 节点。这样既减少了冗余帧对检索的干扰，又保留了
    语义最丰富的关键时刻。

    若 Qwen 调用失败，自动兜底为均匀降采样，保证流程不中断。

    Args:
        qwen_api_key: 阿里云百炼 API Key。
        qwen_api_url: 百炼 OpenAI 兼容接口地址。
        qwen_model:   千问视觉模型名称（如 "qwen-vl-plus"）。
        timeout:      API 请求超时（秒）。
    """

    _PROMPT_TMPL = (
        "You are analyzing {n} frames extracted from a short video clip. "
        "Rank each frame by its importance for capturing key actions, "
        "significant visual details, notable objects, or informative content. "
        "Respond ONLY with a valid JSON array of frame indices (0-based) sorted by importance "
        "(most important first), containing exactly the top {top_k} frame indices. "
        "Example (top 3 out of 10 frames): [4, 1, 7] "
        "IMPORTANT: Output ONLY the JSON array — no explanation, no extra text."
    )

    def __init__(
        self,
        qwen_api_key: str,
        qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        qwen_model: str = "qwen-vl-plus",
        timeout: int = 90,
    ) -> None:
        self.qwen_api_key = qwen_api_key
        self.qwen_api_url = qwen_api_url
        self.qwen_model = qwen_model
        self.timeout = timeout

    def filter_top_k(
        self,
        frames_meta: List[tuple],
        images: List[Image.Image],
        top_k: int,
    ) -> tuple:
        """对候选帧打分并返回 Top-K 帧（按原始时序排列）。

        Args:
            frames_meta: 帧元数据列表，每项为 (idx, timestamp, path)。
            images:      对应 PIL.Image 列表（与 frames_meta 一一对应）。
            top_k:       最终保留的帧数量。

        Returns:
            (filtered_frames_meta, filtered_images) 元组，
            均按原始时序（timestamp 升序）排列。
        """
        n = len(images)
        if n <= top_k:
            return frames_meta, images

        ranked = self._call_qwen(images, top_k)
        if not ranked:
            # 兜底：均匀降采样
            indices = [int(round(i * (n - 1) / (top_k - 1))) for i in range(top_k)]
            indices = sorted(set(indices))
        else:
            # 取前 top_k 个，按时序重排（保持时间顺序）
            indices = sorted(set(ranked[:top_k]))

        out_meta = [frames_meta[i] for i in indices if i < n]
        out_imgs = [images[i] for i in indices if i < n]
        return out_meta, out_imgs

    def _call_qwen(self, frames: List[Image.Image], top_k: int) -> List[int]:
        """单次 API 调用：返回 Top-K 帧的索引（按重要性降序）。"""
        n = len(frames)
        prompt = self._PROMPT_TMPL.format(n=n, top_k=min(top_k, n))
        content: list = []
        for frame in frames:
            b64 = _pil_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.qwen_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.qwen_api_url, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\[.*?\]", text, re.DOTALL)
            if m:
                indices = json.loads(m.group())
                return [i for i in indices if isinstance(i, int) and 0 <= i < n]
        except Exception as exc:
            logger.warning(f"[QwenKeyframeScorer] API 调用失败：{exc}，使用均匀采样兜底。")
        return []


# ---------------------------------------------------------------------------
# VLM Description Generator
# ---------------------------------------------------------------------------

class VLMDescriptionGenerator:
    """调用多模态 VLM 为视频段/片段生成文字描述。

    支持两种后端：
      - **ollama**：调用本地 Ollama 服务（llava / bakllava 等）。
      - **openai**：调用 OpenAI GPT-4o / GPT-4V API。
      - **stub**：不调用真实 VLM，返回占位文本（调试用）。

    Args:
        backend: 后端类型，"ollama" | "openai" | "stub"。
        ollama_chat_url: Ollama Chat API 地址。
        ollama_model: Ollama 多模态模型名称。
        openai_api_key: OpenAI API Key（backend="openai" 时使用）。
        openai_api_url: OpenAI API 地址。
        openai_model: OpenAI 模型名称。
        max_tokens: 最大输出 token 数。
        temperature: 生成温度。
        timeout: HTTP 超时时间（秒）。
    """

    def __init__(
        self,
        backend: str = "ollama",
        ollama_chat_url: str = "http://localhost:11434/api/chat",
        ollama_model: str = "llava",
        openai_api_key: str = "",
        openai_api_url: str = "https://api.openai.com/v1/chat/completions",
        openai_model: str = "gpt-4o",
        qwen_api_key: str = "",
        qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        qwen_model: str = "qwen-vl-plus",
        max_tokens: int = 256,
        temperature: float = 0.1,
        timeout: int = 60,
    ) -> None:
        self.backend = backend.lower()
        self.ollama_chat_url = ollama_chat_url
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key
        self.openai_api_url = openai_api_url
        self.openai_model = openai_model
        self.qwen_api_key = qwen_api_key
        self.qwen_api_url = qwen_api_url
        self.qwen_model = qwen_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def describe(
        self,
        frames: List[Image.Image],
        prompt: str,
        retries: int = 2,
    ) -> str:
        """给定一批代表帧和提示词，生成文字描述。

        Args:
            frames: 代表帧列表（PIL.Image）。
            prompt: 文字提示词。
            retries: 失败重试次数。

        Returns:
            VLM 生成的描述文本（失败时返回 "（描述生成失败）"）。
        """
        if self.backend == "stub":
            return f"[STUB] {prompt[:50]}... (帧数={len(frames)})"

        for attempt in range(retries + 1):
            try:
                if self.backend == "ollama":
                    return self._call_ollama(frames, prompt)
                elif self.backend == "openai":
                    return self._call_openai(frames, prompt)
                elif self.backend == "qwen":
                    return self._call_qwen(frames, prompt)
                else:
                    raise ValueError(f"未知 VLM 后端：{self.backend}")
            except Exception as exc:
                logger.warning(
                    f"VLM 调用失败（第 {attempt+1} 次）：{exc}"
                )
                if attempt < retries:
                    time.sleep(2 ** attempt)  # 指数退避

        return "（描述生成失败）"

    # ------------------------------------------------------------------
    # Private: Ollama
    # ------------------------------------------------------------------

    def _call_ollama(self, frames: List[Image.Image], prompt: str) -> str:
        """调用 Ollama VLM API（支持图像输入）。

        Ollama Chat API 格式（带图像）::

            POST /api/chat
            {
              "model": "llava",
              "messages": [{
                "role": "user",
                "content": "...",
                "images": ["<base64>", ...]
              }],
              "stream": false,
              "options": {"temperature": 0.1}
            }
        """
        images_b64 = [_pil_to_base64(f) for f in frames]
        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = requests.post(self.ollama_chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Private: OpenAI
    # ------------------------------------------------------------------

    def _call_openai(self, frames: List[Image.Image], prompt: str) -> str:
        """调用 OpenAI GPT-4o 多模态 API。

        使用 OpenAI Chat Completions API 的 vision 格式，
        将图像嵌入为 base64 data URI。
        """
        content: list = [{"type": "text", "text": prompt}]
        for frame in frames:
            b64 = _pil_to_base64(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                }
            )
        payload = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.openai_api_url, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_qwen(self, frames: List[Image.Image], prompt: str) -> str:
        """调用阿里云百炼千问视觉大模型（Qwen-VL）API。

        百炼提供 OpenAI 兼容接口，图像格式为 base64 data URI。
        多图输入时，所有图像和文本合并到同一 content 列表中。

        参考：https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
        """
        content: list = []
        # Qwen-VL 支持多图：先传图像，再传文本（官方推荐顺序）
        for frame in frames:
            b64 = _pil_to_base64(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.qwen_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.qwen_api_url, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Representative Frame Sampler
# ---------------------------------------------------------------------------

def sample_representative_frames(
    all_frames_meta: List[Tuple[int, float, str]],
    max_frames: int,
) -> List[Image.Image]:
    """从帧元数据列表中均匀采样代表帧并加载为 PIL.Image。

    Args:
        all_frames_meta: 帧元数据列表，每项为 (idx, timestamp, path)。
        max_frames: 最多返回的帧数量。

    Returns:
        PIL.Image 列表（按时间顺序排列）。
    """
    if not all_frames_meta:
        return []

    n = len(all_frames_meta)
    if n <= max_frames:
        indices = list(range(n))
    else:
        # 均匀采样（包含首尾）
        indices = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
        indices = sorted(set(indices))

    images: List[Image.Image] = []
    for idx in indices:
        _, _, fpath = all_frames_meta[idx]
        try:
            images.append(Image.open(fpath).convert("RGB"))
        except Exception as exc:
            logger.warning(f"无法加载帧图像 {fpath}：{exc}")
    return images


# ---------------------------------------------------------------------------
# CLIP Feature Extractor
# ---------------------------------------------------------------------------

class CLIPFeatureExtractor:
    """使用 CLIP 对图像帧进行视觉特征提取。

    提取的视觉嵌入（L3 节点特征）将通过 VisualProjectionLayer
    对齐到文本潜空间，支持 Phase-3 跨模态检索。

    Args:
        model_name: CLIP 模型变体（如 "ViT-B/32"）。
        device: 推理设备（"cuda" 或 "cpu"）。
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._preprocess = None

    def _ensure_loaded(self) -> None:
        """延迟加载 CLIP 模型（避免在不需要时占用 GPU 内存）。"""
        if self._model is not None:
            return
        try:
            import clip  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "请先安装 CLIP：pip install git+https://github.com/openai/CLIP.git"
            ) from exc
        import torch

        device = self._device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU。")
            device = "cpu"

        self._model, self._preprocess = clip.load(self._model_name, device=device)
        self._model.eval()
        self._device = device
        logger.info(f"CLIP 模型 {self._model_name} 已加载到 {device}。")

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """批量编码图像为 CLIP 视觉嵌入。

        Args:
            images: PIL.Image 列表。
            batch_size: 批处理大小（防止 OOM）。
            normalize: 是否对输出进行 L2 归一化。

        Returns:
            视觉嵌入矩阵，形状 [N, D]，dtype float32。
        """
        import torch

        self._ensure_loaded()
        if not images:
            return np.zeros((0, self._get_embed_dim()), dtype=np.float32)

        all_embeds = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack(
                [self._preprocess(img) for img in batch_imgs]  # type: ignore[misc]
            ).to(self._device)

            with torch.no_grad():
                feats = self._model.encode_image(tensors).float()  # type: ignore[union-attr]

            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            all_embeds.append(feats.cpu().numpy())

        return np.concatenate(all_embeds, axis=0)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """批量编码文本为 CLIP 文本嵌入。

        Args:
            texts: 文本字符串列表。
            batch_size: 批处理大小。
            normalize: 是否 L2 归一化。

        Returns:
            文本嵌入矩阵，形状 [N, D]，dtype float32。
        """
        import torch

        try:
            import clip  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "请先安装 CLIP：pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self._get_embed_dim()), dtype=np.float32)

        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(self._device)

            with torch.no_grad():
                feats = self._model.encode_text(tokens).float()  # type: ignore[union-attr]

            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            all_embeds.append(feats.cpu().numpy())

        return np.concatenate(all_embeds, axis=0)

    def _get_embed_dim(self) -> int:
        """返回当前 CLIP 模型的嵌入维度。"""
        _dims = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
        return _dims.get(self._model_name, 512)


# ---------------------------------------------------------------------------
# Ollama Text Embedder
# ---------------------------------------------------------------------------

class OllamaTextEmbedder:
    """使用 Ollama nomic-embed-text 生成文本嵌入。

    用于 text_backend="ollama" 模式下的 L1/L2 节点文本嵌入。

    Args:
        base_url: Ollama OpenAI 兼容嵌入 API 地址。
        model: 嵌入模型名称。
        timeout: HTTP 请求超时（秒）。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1/embeddings",
        model: str = "nomic-embed-text",
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def encode(self, texts: List[str]) -> np.ndarray:
        """批量编码文本为嵌入向量。

        Args:
            texts: 文本字符串列表（允许空列表）。

        Returns:
            嵌入矩阵，形状 [N, D]，dtype float32。
            对于失败或空文本，对应行为全零向量。
        """
        embeds = []
        for text in texts:
            vec = self._encode_single(text)
            embeds.append(vec)
        if not embeds:
            return np.zeros((0, 768), dtype=np.float32)
        return np.array(embeds, dtype=np.float32)

    def _encode_single(self, text: str) -> np.ndarray:
        """编码单条文本。"""
        if not text or not text.strip():
            return np.zeros(768, dtype=np.float32)
        try:
            resp = requests.post(
                self.base_url,
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                vec = resp.json()["data"][0]["embedding"]
                arr = np.array(vec, dtype=np.float32)
                # L2 归一化
                norm = np.linalg.norm(arr)
                if norm > 1e-8:
                    arr = arr / norm
                return arr
            else:
                logger.warning(f"Ollama 嵌入 API 返回 {resp.status_code}：{resp.text[:100]}")
        except Exception as exc:
            logger.warning(f"Ollama 嵌入调用失败：{exc}")
        return np.zeros(768, dtype=np.float32)

