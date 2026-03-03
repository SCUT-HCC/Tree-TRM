"""
答案生成器（Answer Generator）
=================================
实现论文 §3.4 答案生成阶段：

  F_target = Frame[k₁*, k₂*, k₃*]
  answer   = VLM(query, F_target, context_metadata)

Video-Tree-TRM 的分层剪枝实现了高信噪比（High SNR）的信息传递：
VLM 仅需处理单张经过精确定位的关键帧，而非整个视频流，
从而显著降低幻觉率并提升回答准确性。

支持三种答案生成后端：
  1. **DeepSeek**：纯文本模式，将检索到的帧元数据（摘要/描述/时间戳）
     与用户问题一起构造文本提示发送给 DeepSeek API。
     优点：无需本地 GPU，API 调用成本低。
     缺点：无法直接利用视觉信息（仅依赖 VLM 预生成的文字描述）。

  2. **OpenAI GPT-4o**：多模态模式，将关键帧图像（base64 编码）和
     文字提示一起发送给 GPT-4o，实现真正的视觉问答。
     优点：利用强大的视觉理解能力。
     缺点：API 调用成本高，需要 OpenAI API Key。

  3. **Ollama（LLaVA 等）**：本地多模态模式，将关键帧图像
     与提示发送给本地 LLaVA 模型。
     优点：完全本地运行，数据隐私好。
     缺点：需要本地 GPU 和下载 LLaVA 模型权重。
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from .video_tree_trm import RetrievalTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_frame_as_base64(frame_path: str, quality: int = 85) -> Optional[str]:
    """从磁盘加载帧图像并编码为 base64 字符串。

    Args:
        frame_path: 图像文件路径（JPEG）。
        quality:    JPEG 重编码质量（0-100）。

    Returns:
        base64 字符串；若文件不存在或加载失败则返回 None。
    """
    path = Path(frame_path)
    if not path.exists():
        logger.warning(f"帧图像文件不存在：{frame_path}")
        return None
    try:
        img = Image.open(str(path)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.warning(f"无法加载帧图像 {frame_path}：{exc}")
        return None


def _build_text_prompt(
    question: str,
    video_name: str,
    timestamp: float,
    segment_summary: str,
    clip_caption: str,
    prompt_template: str,
) -> str:
    """根据检索上下文构造文本提示词。

    Args:
        question:        用户原始问题。
        video_name:      视频文件名。
        timestamp:       目标帧时间戳（秒）。
        segment_summary: L1 段摘要文本。
        clip_caption:    L2 片段描述文本。
        prompt_template: 提示词模板（含占位符）。

    Returns:
        格式化后的完整提示词字符串。
    """
    return prompt_template.format(
        question=question,
        video_name=video_name,
        timestamp=timestamp,
        segment_summary=segment_summary,
        clip_caption=clip_caption,
    )


# ---------------------------------------------------------------------------
# Answer Generator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """基于检索关键帧的 VLM 答案生成器。

    接收 Video-Tree-TRM 的检索结果（RetrievalTrace），
    将目标帧的视觉信息（图像或文字描述）与用户查询结合，
    调用 VLM 生成最终答案。

    Args:
        backend:              答案生成后端，"deepseek" | "openai" | "ollama"。
        deepseek_api_key:     DeepSeek API Key。
        deepseek_url:         DeepSeek API 地址。
        deepseek_model:       DeepSeek 模型名称。
        openai_api_key:       OpenAI API Key。
        openai_url:           OpenAI API 地址。
        openai_model:         OpenAI 模型名称。
        ollama_chat_url:      Ollama Chat API 地址。
        ollama_model:         Ollama 模型名称（支持视觉的 VLM，如 llava）。
        max_tokens:           最大输出 token 数。
        temperature:          生成温度。
        timeout:              HTTP 超时（秒）。
        prompt_template:      提示词模板（含 {question} 等占位符）。
        verbose:              是否打印详细日志。
    """

    def __init__(
        self,
        backend: str = "deepseek",
        deepseek_api_key: str = "",
        deepseek_url: str = "https://api.deepseek.com/v1/chat/completions",
        deepseek_model: str = "deepseek-chat",
        openai_api_key: str = "",
        openai_url: str = "https://api.openai.com/v1/chat/completions",
        openai_model: str = "gpt-4o",
        ollama_chat_url: str = "http://localhost:11434/api/chat",
        ollama_model: str = "llava",
        qwen_api_key: str = "",
        qwen_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        qwen_model: str = "qwen-vl-plus",
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 90,
        prompt_template: str = (
            "You are an expert video analyst. "
            "A keyframe has been precisely retrieved from a long video based on the user's question.\n\n"
            "User Question: {question}\n\n"
            "Retrieved Keyframe Context:\n"
            "  - Video: {video_name}\n"
            "  - Timestamp: {timestamp:.1f}s\n"
            "  - Segment Summary: {segment_summary}\n"
            "  - Clip Description: {clip_caption}\n\n"
            "Based on the visual evidence in the keyframe and the contextual information above, "
            "provide a clear and accurate answer to the question. "
            "If the question requires identifying specific visual details, describe what you see precisely."
        ),
        verbose: bool = True,
    ) -> None:
        self.backend = backend.lower()
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_url = deepseek_url
        self.deepseek_model = deepseek_model
        self.openai_api_key = openai_api_key
        self.openai_url = openai_url
        self.openai_model = openai_model
        self.ollama_chat_url = ollama_chat_url
        self.ollama_model = ollama_model
        self.qwen_api_key = qwen_api_key
        self.qwen_url = qwen_url
        self.qwen_model = qwen_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.prompt_template = prompt_template
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Main Interface                                                       #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        trace: RetrievalTrace,
        video_name: str = "video",
        retries: int = 2,
    ) -> str:
        """根据检索轨迹生成最终答案。

        Args:
            query:      用户原始问题。
            trace:      Video-Tree-TRM 返回的检索轨迹。
            video_name: 视频文件名（用于提示词构造）。
            retries:    API 调用失败时的重试次数。

        Returns:
            VLM 生成的答案字符串。
            若检索失败则返回含错误信息的字符串。
            若 API 调用多次失败则返回错误原因。
        """
        # 检查检索是否成功
        if not trace.is_valid():
            logger.warning("检索未成功（索引为 -1），无法生成答案。")
            return (
                "抱歉，无法从视频中定位到与该问题相关的关键帧。"
                f"检索轨迹：L1={trace.k1_star}, L2={trace.k2_star}, L3={trace.k3_star}"
            )

        # 构造提示词
        prompt = _build_text_prompt(
            question=query,
            video_name=video_name,
            timestamp=trace.target_timestamp,
            segment_summary=trace.segment_summary,
            clip_caption=trace.clip_caption,
            prompt_template=self.prompt_template,
        )

        # 获取关键帧图像（用于多模态后端）
        frame_b64: Optional[str] = None
        if trace.target_frame and trace.target_frame.frame_path:
            frame_b64 = _load_frame_as_base64(trace.target_frame.frame_path)

        if self.verbose:
            logger.info(
                f"答案生成 | 后端={self.backend} | "
                f"关键帧={'有' if frame_b64 else '无（仅文本上下文）'} | "
                f"时间戳={trace.target_timestamp:.1f}s"
            )

        # 调用后端生成答案
        for attempt in range(retries + 1):
            try:
                if self.backend == "stub":
                    return self._call_stub(prompt, trace)
                elif self.backend == "deepseek":
                    return self._call_deepseek(prompt)
                elif self.backend == "openai":
                    return self._call_openai(prompt, frame_b64)
                elif self.backend == "ollama":
                    return self._call_ollama(prompt, frame_b64)
                elif self.backend == "qwen":
                    return self._call_qwen(prompt, frame_b64)
                else:
                    raise ValueError(f"未知答案生成后端：{self.backend}")
            except Exception as exc:
                logger.warning(f"答案生成失败（第 {attempt+1} 次）：{exc}")
                if attempt < retries:
                    time.sleep(2 ** attempt)

        return f"答案生成失败（已重试 {retries} 次）。请检查 API Key 和网络连接。"

    # ------------------------------------------------------------------ #
    # Private: Backend Implementations                                     #
    # ------------------------------------------------------------------ #

    def _call_stub(self, prompt: str, trace: "RetrievalTrace") -> str:
        """Stub 模式：不调用真实 VLM，直接返回模拟答案（仅用于调试和测试）。"""
        answer = (
            f"[STUB ANSWER] Based on the retrieved keyframe at t={trace.target_timestamp:.1f}s "
            f"(L1[{trace.k1_star}] → L2[{trace.k2_star}] → L3[{trace.k3_star}]): "
            f"The video segment summary indicates '{trace.segment_summary[:80]}...'. "
            f"Clip description: '{trace.clip_caption[:80]}...'. "
            f"(This is a stub response for testing. Use deepseek/openai/ollama for real answers.)"
        )
        if self.verbose:
            logger.info(f"[STUB] 答案生成（调试模式）：{answer[:100]}...")
        return answer

    def _call_deepseek(self, prompt: str) -> str:
        """调用 DeepSeek 纯文本 API 生成答案。

        DeepSeek 当前不支持直接传入图像，但通过精确的文字上下文
        （VLM 生成的摘要/描述 + 时间戳）可以实现高质量的文本问答。
        """
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(
            self.deepseek_url, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        if self.verbose:
            logger.info(f"DeepSeek 答案：{answer[:100]}...")
        return answer

    def _call_openai(self, prompt: str, frame_b64: Optional[str] = None) -> str:
        """调用 OpenAI GPT-4o 多模态 API 生成答案。

        若 frame_b64 不为 None，则将关键帧图像一并传入（真实视觉问答）。
        若为 None，则退化为纯文本模式（仅依赖文字上下文）。
        """
        content: list = [{"type": "text", "text": prompt}]
        if frame_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "high",  # 使用高分辨率模式以获取细节
                    },
                }
            )
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(
            self.openai_url, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        if self.verbose:
            logger.info(f"GPT-4o 答案：{answer[:100]}...")
        return answer

    def _call_ollama(self, prompt: str, frame_b64: Optional[str] = None) -> str:
        """调用 Ollama 本地 VLM（如 LLaVA）生成答案。

        若 frame_b64 不为 None，则以多模态模式调用（支持视觉输入的模型）。
        否则以纯文本模式调用。
        """
        message: dict = {"role": "user", "content": prompt}
        if frame_b64:
            message["images"] = [frame_b64]

        payload = {
            "model": self.ollama_model,
            "messages": [message],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = requests.post(self.ollama_chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        answer = resp.json()["message"]["content"].strip()
        if self.verbose:
            logger.info(f"Ollama 答案：{answer[:100]}...")
        return answer

    def _call_qwen(self, prompt: str, frame_b64: Optional[str] = None) -> str:
        """调用阿里云百炼千问视觉大模型（Qwen-VL）生成答案。

        百炼提供 OpenAI 兼容接口：
          - 若 frame_b64 不为 None，使用多模态 qwen-vl-plus/max 生成真实视觉问答。
          - 若 frame_b64 为 None（帧文件丢失），退化为纯文本模式。

        接口文档：
          https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
        """
        if frame_b64:
            # 多模态：先传图像，再传文本（Qwen-VL 官方推荐顺序）
            content: list = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.qwen_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(self.qwen_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        if self.verbose:
            logger.info(f"Qwen 答案：{answer[:100]}...")
        return answer

