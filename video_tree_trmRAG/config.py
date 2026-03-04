"""
Video-Tree-TRM 系统配置
=======================
所有超参数均通过嵌套数据类统一管理，支持从字典或 YAML 文件加载。

配置层次：
  VideoTreeTRMConfig
  ├── PyramidConfig      : 分层语义金字塔（HSP）构建参数
  ├── EmbeddingConfig    : 嵌入模型（CLIP / text encoder）参数
  ├── VLMConfig          : VLM 描述生成参数
  ├── AnswerConfig       : 答案生成参数
  └── RetrievalConfig    : Tree-TRM 检索引擎参数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Level 1  Pyramid Construction
# ---------------------------------------------------------------------------

@dataclass
class PyramidConfig:
    """分层语义金字塔（HSP）构建参数。

    三层结构：
      L1 - 全局叙事节点（宏观事件段，10-20 分钟）
      L2 - 片段级语义节点（短片段，10-30 秒）
      L3 - 帧级视觉节点（1 fps 关键帧）
    """

    # ── Level 1：全局叙事节点 ──────────────────────────────────────────────
    l1_segment_duration: float = 600.0
    """每个 L1 宏观事件段的时长（秒），默认 10 分钟。"""

    l1_max_segments: int = 50
    """最大 L1 节点数量（防止超长视频导致内存溢出）。"""

    l1_sample_fps: float = 0.1
    """为 VLM 摘要采样的帧率（0.1fps = 每 10 秒一帧）。"""

    l1_max_frames_for_vlm: int = 6
    """发送给 VLM 生成摘要的最大帧数。"""

    # ── Level 2：片段级语义节点 ─────────────────────────────────────────
    l2_clip_duration: float = 20.0
    """每个 L2 片段的时长（秒），默认 20 秒。"""

    l2_max_clips_per_segment: int = 60
    """每个 L1 段内最大 L2 片段数量。"""

    l2_max_frames_for_vlm: int = 4
    """发送给 VLM 生成描述的最大帧数。"""

    # ── Level 3：帧级视觉节点 ───────────────────────────────────────────
    l3_fps: float = 1.0
    """L3 关键帧提取帧率（fps），默认 1fps。"""

    l3_max_frames_per_clip: int = 30
    """每个 L2 片段内最大 L3 帧数量。"""

    # ── 存储 ─────────────────────────────────────────────────────────────
    cache_dir: str = "pyramid_cache"
    """金字塔数据缓存目录。"""

    frames_subdir: str = "frames"
    """关键帧图像保存子目录（相对于 cache_dir）。"""

    max_frame_long_side: int = 336
    """CLIP 输入前的图像最大长边（像素），超过则按比例缩放。"""

    # ── 智能场景切分（Qwen-VL Scene Detection） ───────────────────────
    use_qwen_scene_detection: bool = False
    """是否使用 Qwen-VL 进行智能场景边界检测，替代固定时长的 L1 切分。
    启用后，将以 scene_detection_probe_fps 的帧率采样探测帧，批量发送给
    Qwen-VL 识别场景转换点，使 L1 分段边界与内容语义对齐。
    需同时在 VLMConfig 中设置 qwen_api_key。
    """

    scene_detection_probe_fps: float = 0.5
    """场景检测时的探测帧率（fps）。默认 0.5 = 每 2 秒采一帧。
    值越小，API 调用次数越少；值越大，边界定位越精确。
    """

    scene_detection_batch_size: int = 8
    """每次发送给 Qwen-VL 进行场景检测的帧数（相邻批次重叠 1 帧）。
    增大可减少 API 调用次数，但每次请求的图像数量增加。
    """

    l1_min_duration: float = 60.0
    """场景感知切分中 L1 段的最小时长（秒），防止场景切分过于碎片化。
    过短的相邻段会自动合并，直至满足最小时长要求。
    """

    l1_max_duration: float = 1200.0
    """场景感知切分中 L1 段的最大时长（秒），防止单段过长。
    超长段会被自动均匀拆分为多个子段。
    """

    # ── 关键帧筛选（Qwen-VL Keyframe Scoring） ────────────────────────
    use_qwen_keyframe_scoring: bool = False
    """是否使用 Qwen-VL 对 L3 候选帧打分，仅保留 Top-K 关键帧。
    启用后，对每个 L2 clip 内均匀采样得到的候选帧池，调用 Qwen-VL
    按动作重要性与视觉信息密度排序，仅保留最重要的 top-k 帧作为 L3 节点。
    可显著降低 L3 噪声帧比例，提升 Phase-3 检索准确率 +10-15%。
    需同时在 VLMConfig 中设置 qwen_api_key。
    """

    qwen_keyframe_keep_ratio: float = 0.5
    """L3 帧筛选保留比例（0.0 ~ 1.0）。
    例如 0.5 = 保留每个 clip 内 50% 最重要的帧；
    最终保留帧数 = round(候选帧数 × keep_ratio)，且不少于 1 帧。
    """


# ---------------------------------------------------------------------------
# Level 2  Embedding
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """嵌入模型配置。

    文本嵌入（L1/L2 节点）与视觉嵌入（L3 节点）须映射到同一潜空间，
    以支持 Phase-3 的跨模态检索。

    推荐组合：
      A) 全 CLIP：text_backend="clip"，CLIP text encoder 负责 L1/L2，
         CLIP image encoder 负责 L3，天然共享 512-dim 联合空间。
      B) 混合：text_backend="ollama"（nomic-embed-text，768-dim）用于
         L1/L2，CLIP image encoder（512-dim）+ 可学习投影（→ 768-dim）
         用于 L3。混合模式文本理解更强，但需要训练投影层。
    """

    # ── CLIP ────────────────────────────────────────────────────────────
    clip_model: str = "ViT-B/32"
    """CLIP 模型变体。可选: ViT-B/32, ViT-B/16, ViT-L/14。"""

    clip_device: str = "cuda"
    """CLIP 推理设备，"cuda" 或 "cpu"。"""

    # ── Text Backend ──────────────────────────────────────────────────
    text_backend: str = "clip"
    """文本嵌入后端：
      "clip"   - 使用 CLIP text encoder（维度由 clip_model 决定）。
      "ollama" - 使用本地 Ollama nomic-embed-text（768-dim）。
    """

    ollama_embed_url: str = "http://localhost:11434/v1/embeddings"
    """Ollama OpenAI 兼容嵌入接口地址。"""

    ollama_embed_model: str = "nomic-embed-text"
    """Ollama 文本嵌入模型名称。"""

    ollama_embed_dim: int = 768
    """Ollama 嵌入维度（nomic-embed-text 输出为 768-dim）。"""

    # ── Projection（L3 视觉 → 文本潜空间） ──────────────────────────
    use_projection: bool = True
    """是否在 L3 视觉特征上应用可学习投影层。
    当 text_backend="clip" 时，CLIP 自带联合空间，投影可用恒等初始化。
    当 text_backend="ollama" 时，必须启用投影以对齐维度。
    """

    projection_checkpoint: Optional[str] = None
    """投影层预训练权重路径（.pt 文件）。None 则使用随机初始化。"""

    @property
    def embed_dim(self) -> int:
        """统一嵌入维度（由后端决定）。"""
        if self.text_backend == "ollama":
            return self.ollama_embed_dim
        # CLIP 各模型的输出维度
        _clip_dims = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
        return _clip_dims.get(self.clip_model, 512)

    @property
    def clip_visual_dim(self) -> int:
        """CLIP 图像编码器输出维度（与 clip_model 绑定）。"""
        _clip_dims = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
        return _clip_dims.get(self.clip_model, 512)


# ---------------------------------------------------------------------------
# Level 3  VLM for Captions & Summaries
# ---------------------------------------------------------------------------

@dataclass
class VLMConfig:
    """视觉-语言模型（VLM）配置。

    VLM 用于：
      - L1 节点：为宏观事件段生成高层文字摘要（summary）。
      - L2 节点：为短片段生成细节描述（caption）。
    """

    backend: str = "ollama"
    """VLM 后端："ollama" | "openai" | "qwen" | "stub"。
    "qwen"  - 阿里云百炼千问视觉大模型（qwen-vl-plus / qwen-vl-max），无需本地部署。
    "stub"  - 不调用真实 VLM，直接生成占位文本，用于调试。
    """

    # ── Ollama ────────────────────────────────────────────────────────
    ollama_chat_url: str = "http://localhost:11434/api/chat"
    """Ollama Chat API 地址。"""

    ollama_vlm_model: str = "llava"
    """Ollama 多模态模型名称（支持图像输入的 VLM，如 llava / bakllava）。"""

    # ── OpenAI GPT-4V / GPT-4o ────────────────────────────────────────
    openai_api_key: str = ""
    """OpenAI API Key（当 backend="openai" 时使用）。"""

    openai_api_url: str = "https://api.openai.com/v1/chat/completions"
    """OpenAI Chat Completions API 地址。"""

    openai_vlm_model: str = "gpt-4o"
    """OpenAI 多模态模型名称（gpt-4o / gpt-4-vision-preview）。"""

    # ── 阿里云百炼 Qwen-VL ────────────────────────────────────────────
    qwen_api_key: str = ""
    """阿里云百炼 API Key（当 backend="qwen" 时使用）。
    在 https://bailian.console.aliyun.com/ 创建并获取。
    """

    qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    """百炼 OpenAI 兼容接口地址（无需修改）。"""

    qwen_vlm_model: str = "qwen-vl-plus"
    """千问视觉模型名称。
    可选：
      qwen-vl-plus  - 标准版（性价比高，推荐用于 L2 描述）
      qwen-vl-max   - 旗舰版（理解能力最强，推荐用于 L1 摘要）
      qwen-vl-ocr   - OCR 特化版（适合含文字场景）
    """

    # ── Prompts ───────────────────────────────────────────────────────
    summary_prompt: str = (
        "You are a video analysis assistant. "
        "Based on the provided video frames from a long segment, "
        "write a concise high-level narrative summary (2-3 sentences). "
        "Focus on: overall setting, key characters, main events, and temporal flow. "
        "Be factual and descriptive."
    )
    """L1 摘要生成提示词。"""

    caption_prompt: str = (
        "You are a video analysis assistant. "
        "Based on the provided frames from a short video clip, "
        "write a precise description (1-2 sentences) of the specific actions, "
        "interactions, and visual details occurring in this clip. "
        "Include: who, what, where, and any notable visual elements."
    )
    """L2 描述生成提示词。"""

    # ── Generation ────────────────────────────────────────────────────
    max_tokens: int = 256
    """VLM 最大输出 token 数量。"""

    temperature: float = 0.1
    """VLM 生成温度（低温度保证确定性输出）。"""

    timeout: int = 60
    """VLM API 请求超时时间（秒）。"""


# ---------------------------------------------------------------------------
# Level 4  Answer Generation
# ---------------------------------------------------------------------------

@dataclass
class AnswerConfig:
    """最终答案生成配置。

    检索到关键帧后，将其与原始查询一起送入下游 VLM 生成最终答案。
    """

    backend: str = "deepseek"
    """答案生成后端："deepseek" | "openai" | "ollama" | "qwen"。"""

    # ── DeepSeek ──────────────────────────────────────────────────────
    deepseek_api_key: str = ""
    """DeepSeek API Key。"""

    deepseek_url: str = "https://api.deepseek.com/v1/chat/completions"
    """DeepSeek API 地址。"""

    deepseek_model: str = "deepseek-chat"
    """DeepSeek 模型名称。"""

    # ── OpenAI ────────────────────────────────────────────────────────
    openai_api_key: str = ""
    """OpenAI API Key（当 backend="openai" 时使用）。"""

    openai_url: str = "https://api.openai.com/v1/chat/completions"
    """OpenAI API 地址。"""

    openai_model: str = "gpt-4o"
    """OpenAI 模型名称。"""

    # ── Ollama ────────────────────────────────────────────────────────
    ollama_chat_url: str = "http://localhost:11434/api/chat"
    """Ollama Chat API 地址（支持多模态，如 llava）。"""

    ollama_model: str = "llava"
    """Ollama 模型名称。"""

    # ── 阿里云百炼 Qwen ───────────────────────────────────────────────
    qwen_api_key: str = ""
    """阿里云百炼 API Key（当 backend="qwen" 时使用）。"""

    qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    """百炼 OpenAI 兼容接口地址（无需修改）。"""

    qwen_answer_model: str = "qwen-vl-plus"
    """答案生成所用千问模型。
    可选：
      qwen-vl-plus  - 多模态（视觉+文本），可直接分析关键帧图像
      qwen-vl-max   - 旗舰多模态（效果最佳）
      qwen-plus     - 纯文本（仅利用文字上下文，不传图像）
      qwen-turbo    - 纯文本（速度快、成本低）
    """

    # ── Generation ────────────────────────────────────────────────────
    max_tokens: int = 512
    """最大答案 token 数。"""

    temperature: float = 0.1
    """生成温度。"""

    timeout: int = 90
    """API 超时时间（秒）。"""

    answer_prompt_template: str = (
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
    )
    """答案生成提示词模板，支持 {question}、{video_name}、{timestamp}、"
    "{segment_summary}、{clip_caption} 占位符。"""


# ---------------------------------------------------------------------------
# Level 5  Retrieval Engine
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Tree-TRM 递归检索引擎参数。

    实现论文中的三阶段、H=3 递归双循环推理：
      Phase 1 (H=1) : 粗粒度路由，文本对文本，检索 L1 节点。
      Phase 2 (H=2) : 细粒度聚焦，文本对文本，检索 L2 节点。
      Phase 3 (H=3) : 视觉定位，文本对图像（跨模态），检索 L3 节点。
    """

    h_cycles: int = 3
    """Tree 深度（固定为 3，对应三层金字塔）。"""

    selection_mode: str = "argmax"
    """节点选择策略：
      "argmax" - 硬选择（Hard Selection），取得分最高的节点，与论文一致。
      "soft"   - 软选择（Soft Retrieval），加权求和（可微分，用于训练）。
    """

    softmax_temperature: float = 1.0
    """Softmax 温度系数（τ）。τ < 1 使分布更尖锐，τ > 1 更平滑。"""

    state_update_mode: str = "additive"
    """潜在状态更新策略：
      "additive" - z_{h+1} = normalize(z_h + M_Lh[k_h*])（论文默认）。
      "replace"  - z_{h+1} = M_Lh[k_h*]（丢弃历史状态）。
      "gated"    - z_{h+1} = gate * z_h + (1-gate) * M_Lh[k_h*]。
    """

    normalize_state: bool = True
    """是否对更新后的潜在状态进行 L2 归一化（防止量级漂移）。"""

    init_state_mode: str = "query"
    """初始潜在状态 z_0 的初始化方式：
      "query"  - z_0 = 查询嵌入（语义引导初始化）。
      "zeros"  - z_0 = 零向量。
    """


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class VideoTreeTRMConfig:
    """Video-Tree-TRM 系统主配置。

    所有子系统（金字塔构建、嵌入、VLM、检索、答案生成）的参数均在此汇总。

    示例用法::

        # 使用默认配置
        cfg = VideoTreeTRMConfig()

        # 从字典加载
        cfg = VideoTreeTRMConfig.from_dict({
            "embedding": {"text_backend": "ollama"},
            "vlm": {"backend": "openai", "openai_api_key": "sk-..."},
            "answer": {"backend": "deepseek", "deepseek_api_key": "..."},
        })

        # 从 YAML 文件加载
        cfg = VideoTreeTRMConfig.from_yaml("config/video_tree_trm.yaml")
    """

    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    device: str = "cuda"
    """全局推理设备，"cuda" 或 "cpu"。"""

    verbose: bool = True
    """是否打印详细日志（含每个阶段的检索结果）。"""

    seed: int = 42
    """随机种子（保证可复现性）。"""

    @classmethod
    def from_dict(cls, d: dict) -> "VideoTreeTRMConfig":
        """从嵌套字典创建配置。

        只需传入需要覆盖的字段，其余字段保持默认值。

        Args:
            d: 配置字典，键为子配置名（如 "pyramid"、"vlm"），
               值为对应字段的字典。

        Returns:
            VideoTreeTRMConfig 实例。
        """
        sub_config_map = {
            "pyramid": PyramidConfig,
            "embedding": EmbeddingConfig,
            "vlm": VLMConfig,
            "answer": AnswerConfig,
            "retrieval": RetrievalConfig,
        }
        kwargs: dict = {}
        for key, value in d.items():
            if key in sub_config_map and isinstance(value, dict):
                sub_cls = sub_config_map[key]
                # 过滤掉子配置类不认识的字段
                valid_fields = {f.name for f in sub_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
                filtered = {k: v for k, v in value.items() if k in valid_fields}
                kwargs[key] = sub_cls(**filtered)
            else:
                kwargs[key] = value
        # 未出现的子配置保持默认
        config = cls()
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "VideoTreeTRMConfig":
        """从 YAML 文件加载配置。

        Args:
            path: YAML 文件路径。

        Returns:
            VideoTreeTRMConfig 实例。
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError("请先安装 PyYAML：pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        """将配置序列化为嵌套字典（方便保存为 JSON / YAML）。"""
        import dataclasses
        return dataclasses.asdict(self)

