# Video-Tree-TRM RAG：使用指南

> 本文档详细介绍 Video-Tree-TRM 长视频问答系统的安装、配置与使用方法，涵盖从快速上手到高级配置的全部内容。

---

## 目录

1. [环境要求与依赖安装](#1-环境要求与依赖安装)
2. [系统架构快速预览](#2-系统架构快速预览)
3. [快速上手（5 分钟）](#3-快速上手5-分钟)
4. [离线预处理：构建分层语义金字塔](#4-离线预处理构建分层语义金字塔)
5. [在线推理：视频问答](#5-在线推理视频问答)
6. [Python API 使用指南](#6-python-api-使用指南)
7. [配置详解](#7-配置详解)
8. [后端配置指南](#8-后端配置指南)
9. [输出格式说明](#9-输出格式说明)
10. [性能优化建议](#10-性能优化建议)
11. [常见问题与排错](#11-常见问题与排错)
12. [完整使用场景示例](#12-完整使用场景示例)

---

## 1. 环境要求与依赖安装

### 1.1 硬件要求

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 | 8+ 核 |
| RAM | 8 GB | 16+ GB |
| GPU | 无（纯 CPU 模式） | NVIDIA GPU，6+ GB 显存（用于 CLIP 加速） |
| 磁盘 | 视频文件大小 + 约 100MB（2h视频的金字塔缓存） | SSD 推荐 |

### 1.2 Python 版本

```bash
# 要求 Python 3.8+
python --version
```

### 1.3 安装依赖

**方法一：使用 requirements 文件（推荐）**

```bash
cd /path/to/TinyRecursiveModels-main

# 安装基础依赖
pip install -r video_tree_trmRAG/requirements_video.txt
```

**方法二：手动安装**

```bash
# 1. 核心深度学习框架（GPU 版本，根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CPU-only 版本（速度较慢）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. OpenAI CLIP（官方实现，需从 GitHub 安装）
pip install git+https://github.com/openai/CLIP.git
# 国内网络可用镜像：
# pip install git+https://gitee.com/mirrors/CLIP.git

# 3. 视频处理与工具库
pip install opencv-python>=4.6.0 Pillow>=9.0.0 numpy>=1.21.0 requests>=2.28.0 pyyaml>=6.0 tqdm>=4.64.0
```

### 1.4 可选：本地 VLM 服务（Ollama）

若要使用本地 VLM（LLaVA）生成视频描述：

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. 启动 Ollama 服务
ollama serve

# 3. 拉取多模态 VLM 模型（在另一个终端）
ollama pull llava          # 7B 参数，约 4.5GB
# 或更大的模型（质量更好）：
ollama pull llava:13b      # 13B 参数，约 8GB

# 4. 拉取文本嵌入模型（可选，用于 ollama 文本嵌入后端）
ollama pull nomic-embed-text
```

---

## 2. 系统架构快速预览

```
长视频文件（.mp4/.mkv/...）
        │
        │ [一次性离线预处理]
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │         分层语义金字塔（pyramid_cache/视频名/）           │
 │  L1：全局叙事节点（文本摘要嵌入）    [N1 ≈ 12 个/2h视频] │
 │  L2：片段级语义节点（文本描述嵌入）  [N2 ≈ 360 个/2h视频] │
 │  L3：帧级视觉节点（CLIP视觉嵌入）   [N3 ≈ 7200 个/2h视频] │
 └─────────────────────────────────────────────────────────┘
        │
        │ [每次查询，< 100ms（不含VLM答案生成）]
        ▼
 用户查询 ──► Phase1路由 ──► Phase2聚焦 ──► Phase3视觉定位 ──► 关键帧
                                                                    │
                                                                    ▼
                                                              VLM 答案生成
                                                                    │
                                                                    ▼
                                                               最终答案
```

---

## 3. 快速上手（5 分钟）

### 3.1 调试模式（无需任何 API Key 或 GPU）

使用 `stub` VLM 后端（不调用真实 VLM）和 CPU CLIP 快速验证系统是否正常工作：

```bash
cd /path/to/TinyRecursiveModels-main

# 步骤 1：构建金字塔（stub 模式，秒级完成）
python video_tree_trmRAG/build_pyramid.py \
  --video your_video.mp4 \
  --vlm_backend stub \
  --clip_device cpu \
  --l1_duration 60 \
  --l2_duration 10 \
  --l3_fps 0.5

# 步骤 2：执行问答（Ollama 本地 LLaVA 答案生成）
python video_tree_trmRAG/run_videoqa.py \
  --pyramid_dir pyramid_cache/your_video \
  --query "What is happening in this video?" \
  --answer_backend ollama \
  --answer_ollama_model llava \
  --show_trace
```

### 3.2 生产模式（OpenAI GPT-4o 全链路高质量）

```bash
# 步骤 1：构建高质量金字塔（GPT-4o VLM）
python video_tree_trmRAG/build_pyramid.py \
  --video movie.mp4 \
  --vlm_backend openai \
  --openai_api_key "sk-your-key-here" \
  --vlm_model gpt-4o \
  --l1_duration 600 \
  --l2_duration 20 \
  --l3_fps 1.0

# 步骤 2：视频问答（GPT-4o 答案生成）
python video_tree_trmRAG/run_videoqa.py \
  --pyramid_dir pyramid_cache/movie \
  --query "What is written on the poster on the door?" \
  --answer_backend openai \
  --answer_openai_key "sk-your-key-here" \
  --show_trace
```

---

## 4. 离线预处理：构建分层语义金字塔

### 4.1 `build_pyramid.py` 完整参数说明

```bash
python video_tree_trmRAG/build_pyramid.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `--video` | str | **必填** | 输入视频文件路径 |
| `--output_dir` | str | `pyramid_cache/<视频名>` | 金字塔保存目录 |
| `--force_rebuild` | 标志 | False | 强制重建（忽略已有缓存） |
| `--l1_duration` | float | 600.0 | L1 宏观事件段时长（秒） |
| `--l2_duration` | float | 20.0 | L2 短片段时长（秒） |
| `--l3_fps` | float | 1.0 | L3 关键帧提取帧率 |
| `--l1_max_frames_vlm` | int | 6 | 发给 VLM 生成 L1 摘要的最大帧数 |
| `--l2_max_frames_vlm` | int | 4 | 发给 VLM 生成 L2 描述的最大帧数 |
| `--l3_max_frames` | int | 30 | 每 L2 片段最大 L3 帧数 |
| `--max_frame_size` | int | 336 | 帧图像最大长边（像素） |
| `--vlm_backend` | str | `stub` | VLM 后端：`stub`/`ollama`/`openai` |
| `--ollama_url` | str | `http://localhost:11434/api/chat` | Ollama Chat API 地址 |
| `--vlm_model` | str | `llava` | VLM 模型名称 |
| `--openai_api_key` | str | `""` | OpenAI API Key |
| `--text_backend` | str | `clip` | 文本嵌入后端：`clip`/`ollama` |
| `--clip_model` | str | `ViT-B/32` | CLIP 模型变体 |
| `--clip_device` | str | `cuda` | CLIP 推理设备 |
| `--projection_checkpoint` | str | None | 投影层权重路径 |
| `--verbose` | 标志 | True | 打印详细日志 |

### 4.2 典型使用场景

**场景 A：本地部署，Ollama LLaVA，小视频（< 30 分钟）**

```bash
python video_tree_trmRAG/build_pyramid.py \
  --video short_video.mp4 \
  --vlm_backend ollama \
  --vlm_model llava \
  --l1_duration 300 \
  --l2_duration 15 \
  --l3_fps 1.0 \
  --clip_device cuda
```

**场景 B：云端 GPT-4o，长视频（2+ 小时）**

```bash
python video_tree_trmRAG/build_pyramid.py \
  --video long_movie.mp4 \
  --vlm_backend openai \
  --openai_api_key "sk-..." \
  --vlm_model gpt-4o \
  --l1_duration 600 \
  --l2_duration 20 \
  --l3_fps 1.0 \
  --output_dir /data/pyramids/long_movie
```

**场景 C：快速调试，CPU 模式，无 API Key**

```bash
python video_tree_trmRAG/build_pyramid.py \
  --video test.mp4 \
  --vlm_backend stub \
  --text_backend clip \
  --clip_device cpu \
  --l1_duration 60 \
  --l2_duration 10 \
  --l3_fps 0.5 \
  --l3_max_frames 10
```

### 4.3 构建成功后的输出

```
pyramid_cache/your_video/
├── metadata.json              ← 金字塔元数据（维度、时间参数等）
├── build_summary.json         ← 构建摘要（耗时、节点统计）
├── l1_embeddings.npy          ← L1 嵌入矩阵 [N1, 512]
├── l1_metadata.json           ← L1 节点时间和摘要文本
├── l2_embeddings_0.npy        ← L1[0] 段的 L2 嵌入矩阵 [N2_0, 512]
├── l2_metadata_0.json         ← L1[0] 段的 L2 节点描述
├── l3_embeddings_0_0.npy      ← L1[0].L2[0] 片段的 L3 嵌入矩阵 [N3_00, 512]
├── l3_metadata_0_0.json       ← L1[0].L2[0] 片段的帧信息（含路径）
├── ... (更多 l2/l3 文件)
└── frames/                    ← 所有提取的帧图像（JPEG）
    ├── l1_000_rep_000000.jpg
    ├── l2_000_0000_rep_000000.jpg
    ├── l3_000_0000_000000.jpg
    └── ...
```

---

## 5. 在线推理：视频问答

### 5.1 `run_videoqa.py` 完整参数说明

```bash
python video_tree_trmRAG/run_videoqa.py [OPTIONS]
```

**输入模式（二选一）**

| 参数 | 说明 |
|------|------|
| `--video <path>` | 原始视频路径（触发完整流程，含自动预处理） |
| `--pyramid_dir <path>` | 预构建的金字塔目录（跳过预处理，直接检索） |

**问题输入（二选一）**

| 参数 | 说明 |
|------|------|
| `--query "问题文本"` | 单个问题 |
| `--query_file questions.txt` | 批量问题文件（每行一个问题） |

**答案生成后端**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--answer_backend` | `ollama` | `deepseek`/`openai`/`ollama` |
| `--deepseek_api_key` | `""` | DeepSeek API Key |
| `--deepseek_model` | `deepseek-chat` | DeepSeek 模型名称 |
| `--answer_openai_key` | `""` | OpenAI API Key（答案生成用） |
| `--answer_openai_model` | `gpt-4o` | OpenAI 模型名称 |
| `--answer_ollama_model` | `llava` | Ollama 模型（需支持视觉） |

**输出控制**

| 参数 | 说明 |
|------|------|
| `--output_json <path>` | 将结果保存为 JSON 文件 |
| `--show_trace` | 打印详细检索轨迹（L1/L2/L3 分数分布） |

### 5.2 使用示例

**单问题问答（使用已有金字塔）**

```bash
python video_tree_trmRAG/run_videoqa.py \
  --pyramid_dir pyramid_cache/movie \
  --query "What color is the car in the parking scene?" \
  --answer_backend openai \
  --answer_openai_key "sk-..." \
  --show_trace
```

**批量问题问答（从文件读取）**

```bash
# questions.txt 内容（每行一个问题）：
# What is the main character wearing?
# What happens at the school entrance?
# What is written on the whiteboard?

python video_tree_trmRAG/run_videoqa.py \
  --pyramid_dir pyramid_cache/movie \
  --query_file questions.txt \
  --answer_backend ollama \
  --answer_ollama_model llava \
  --output_json results/movie_answers.json
```

**完整流程（一步完成预处理 + 问答）**

```bash
python video_tree_trmRAG/run_videoqa.py \
  --video movie.mp4 \
  --query "Who is the first character to appear?" \
  --vlm_backend ollama \
  --vlm_model llava \
  --answer_backend ollama \
  --answer_ollama_model llava \
  --pyramid_save_dir pyramid_cache/movie
```

**从 YAML 配置文件运行**

```bash
python video_tree_trmRAG/run_videoqa.py \
  --config config/video_tree_trm.yaml \
  --pyramid_dir pyramid_cache/movie \
  --query "Describe the climax scene."
```

---

## 6. Python API 使用指南

### 6.1 最简使用（3 行代码）

```python
from video_tree_trmRAG import VideoQAPipeline, VideoTreeTRMConfig

pipeline = VideoQAPipeline(VideoTreeTRMConfig())

result = pipeline.run(
    video_path="movie.mp4",
    query="What is written on the door poster?"
)
print(result.answer)
print(f"检索到关键帧时间戳: {result.trace.target_timestamp:.1f}s")
```

### 6.2 完整配置示例

```python
from video_tree_trmRAG import VideoQAPipeline, VideoTreeTRMConfig

# 方式 1：通过字典配置
config = VideoTreeTRMConfig.from_dict({
    "pyramid": {
        "l1_segment_duration": 600.0,  # 10 分钟/段
        "l2_clip_duration": 20.0,       # 20 秒/片段
        "l3_fps": 1.0,                  # 1 帧/秒
        "cache_dir": "my_pyramids",
    },
    "embedding": {
        "clip_model": "ViT-B/32",       # 推荐：ViT-B/32（快）或 ViT-L/14（精）
        "clip_device": "cuda",
        "text_backend": "clip",         # "clip" 或 "ollama"
    },
    "vlm": {
        "backend": "openai",
        "openai_api_key": "sk-your-key",
        "openai_vlm_model": "gpt-4o",
    },
    "answer": {
        "backend": "openai",
        "openai_api_key": "sk-your-key",
        "openai_model": "gpt-4o",
    },
    "retrieval": {
        "selection_mode": "argmax",         # "argmax" 或 "soft"
        "softmax_temperature": 1.0,
        "state_update_mode": "additive",    # "additive"/"replace"/"gated"
        "init_state_mode": "query",         # "query" 或 "zeros"
    },
    "verbose": True,
})

# 方式 2：从 YAML 文件加载（见 §7.3）
# config = VideoTreeTRMConfig.from_yaml("config/video_tree_trm.yaml")

pipeline = VideoQAPipeline(config)
```

### 6.3 两步工作流（推荐生产使用）

```python
from video_tree_trmRAG import VideoQAPipeline, VideoTreeTRMConfig

config = VideoTreeTRMConfig.from_dict({
    "vlm": {"backend": "openai", "openai_api_key": "sk-..."},
    "answer": {"backend": "openai", "openai_api_key": "sk-..."},
})
pipeline = VideoQAPipeline(config)

# 步骤 1：构建金字塔（只需做一次，后续自动从缓存加载）
pyramid = pipeline.build_pyramid(
    video_path="movie.mp4",
    save_dir="pyramid_cache/movie",
    force_rebuild=False,  # True 强制重建
)
print(pyramid)
# 输出：HierarchicalSemanticPyramid(video='movie.mp4', duration=7234.5s,
#                                    L1=12, L2=361, L3=7234, embed_dim=512)

# 步骤 2：多次问答复用同一金字塔
queries = [
    "What is written on the door poster?",
    "Who is the first character to appear?",
    "What happens during the school scene?",
]
results = pipeline.run_batch(
    pyramid_dir="pyramid_cache/movie",
    queries=queries,
)
for result in results:
    print(f"Q: {result.query}")
    print(f"A: {result.answer}")
    print(f"关键帧: {result.trace.target_timestamp:.1f}s ({result.trace.target_frame.frame_path})")
    print()
```

### 6.4 访问检索轨迹

```python
result = pipeline.run_from_pyramid(
    pyramid_dir="pyramid_cache/movie",
    query="What is on the table?",
)

trace = result.trace
if trace.is_valid():
    print(f"Phase 1: 选中 L1[{trace.k1_star}] - {trace.segment_summary[:60]}")
    print(f"Phase 2: 选中 L2[{trace.k2_star}] - {trace.clip_caption[:60]}")
    print(f"Phase 3: 选中 L3[{trace.k3_star}] - 时间戳 {trace.target_timestamp:.1f}s")
    print(f"目标帧路径: {trace.target_frame.frame_path}")
    
    # 查看得分分布（判断检索置信度）
    import numpy as np
    print(f"\nL1 得分（前3名）: {np.sort(trace.l1_scores)[::-1][:3]}")
    print(f"L2 得分（前3名）: {np.sort(trace.l2_scores)[::-1][:3]}")
    
    # 可视化得分分布
    # import matplotlib.pyplot as plt
    # plt.bar(range(len(trace.l1_scores)), trace.l1_scores)
    # plt.title("L1 Retrieval Scores")
    # plt.savefig("l1_scores.png")
```

### 6.5 加载已有金字塔

```python
from video_tree_trmRAG.video_pyramid import HierarchicalSemanticPyramid

# 检查金字塔是否存在
if HierarchicalSemanticPyramid.exists("pyramid_cache/movie"):
    pyramid = HierarchicalSemanticPyramid.load("pyramid_cache/movie")
    
    stats = pyramid.stats()
    print(f"L1 节点数: {stats['n_segments']}")
    print(f"L2 节点数: {stats['n_clips']}")
    print(f"L3 节点数: {stats['n_frames']}")
    
    # 访问特定节点
    seg = pyramid.get_segment_node(0)
    print(f"第一个 L1 节点摘要: {seg.summary}")
    
    clip = pyramid.get_clip_node(0, 0)
    print(f"第一个 L2 节点描述: {clip.caption}")
    print(f"时间范围: {clip.start_time:.1f}s - {clip.end_time:.1f}s")
```

### 6.6 使用 VideoTreeTRM 检索引擎（低层 API）

```python
from video_tree_trmRAG.video_tree_trm import VideoTreeTRM
from video_tree_trmRAG.video_pyramid import HierarchicalSemanticPyramid

# 加载金字塔
pyramid = HierarchicalSemanticPyramid.load("pyramid_cache/movie")

# 创建检索引擎
retriever = VideoTreeTRM(
    embed_dim=512,
    text_backend="clip",
    clip_model="ViT-B/32",
    clip_device="cuda",
    softmax_temperature=0.7,       # 更低的温度 = 更确定的选择
    state_update_mode="additive",
    verbose=True,
)

# 执行检索
trace = retriever.retrieve(
    query="What is written on the door poster?",
    pyramid=pyramid,
)

# 批量检索
traces = retriever.batch_retrieve(
    queries=["Q1", "Q2", "Q3"],
    pyramid=pyramid,
)
```

---

## 7. 配置详解

### 7.1 金字塔结构参数调优

| 参数 | 推荐值 | 说明 | 调整建议 |
|------|-------|------|---------|
| `l1_segment_duration` | 600s（10min） | L1 段时长 | 视频 < 30min 时可设为 180-300s |
| `l2_clip_duration` | 20s | L2 片段时长 | 快速动作视频可设为 5-10s |
| `l3_fps` | 1.0 | L3 帧率 | 高动态视频可设为 2.0，静止场景可设为 0.5 |
| `l1_max_frames_for_vlm` | 6 | L1 VLM 帧数 | 增大可提升摘要质量（但增加成本） |
| `l2_max_frames_for_vlm` | 4 | L2 VLM 帧数 | 同上 |
| `l3_max_frames_per_clip` | 30 | L3 帧数上限 | 防止片段内帧数过多（节省内存） |

### 7.2 检索引擎参数调优

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `selection_mode` | `argmax` | 推理时用 argmax；训练时可改为 `soft` |
| `softmax_temperature` | 1.0 | 降低（0.5）使选择更确定；升高（2.0）使选择更多样 |
| `state_update_mode` | `additive` | 通常无需修改 |
| `normalize_state` | True | 建议保持 True |
| `init_state_mode` | `query` | 用查询嵌入初始化，提供更好的语义引导 |

### 7.3 YAML 配置文件示例

创建 `config/video_tree_trm.yaml`：

```yaml
# Video-Tree-TRM 配置文件

pyramid:
  l1_segment_duration: 600.0
  l2_clip_duration: 20.0
  l3_fps: 1.0
  l1_max_frames_for_vlm: 6
  l2_max_frames_for_vlm: 4
  l3_max_frames_per_clip: 30
  max_frame_long_side: 336
  cache_dir: pyramid_cache

embedding:
  clip_model: "ViT-B/32"     # 选项: ViT-B/32, ViT-B/16, ViT-L/14
  clip_device: "cuda"
  text_backend: "clip"        # 选项: clip, ollama
  use_projection: true
  # projection_checkpoint: "checkpoints/proj_v1.pt"  # 可选

vlm:
  backend: "ollama"            # 选项: ollama, openai, stub
  ollama_chat_url: "http://localhost:11434/api/chat"
  ollama_vlm_model: "llava"
  # backend: "openai"
  # openai_api_key: "sk-..."
  # openai_vlm_model: "gpt-4o"
  max_tokens: 256
  temperature: 0.1
  timeout: 60

answer:
  backend: "ollama"            # 选项: deepseek, openai, ollama
  # deepseek_api_key: "..."
  # openai_api_key: "sk-..."
  ollama_chat_url: "http://localhost:11434/api/chat"
  ollama_model: "llava"
  max_tokens: 512
  temperature: 0.1
  timeout: 90

retrieval:
  h_cycles: 3
  selection_mode: "argmax"
  softmax_temperature: 1.0
  state_update_mode: "additive"
  normalize_state: true
  init_state_mode: "query"

device: "cuda"
verbose: true
seed: 42
```

使用 YAML 配置：

```bash
python video_tree_trmRAG/run_videoqa.py \
  --config config/video_tree_trm.yaml \
  --pyramid_dir pyramid_cache/movie \
  --query "What is the movie about?"
```

---

## 8. 后端配置指南

### 8.1 VLM 后端配置

#### Ollama（本地，推荐入门）

```bash
# 确保 Ollama 服务已启动
ollama serve &

# 验证服务正常
curl http://localhost:11434/api/tags
```

配置参数：
```python
vlm = VLMConfig(
    backend="ollama",
    ollama_chat_url="http://localhost:11434/api/chat",
    ollama_vlm_model="llava",  # 或 "llava:13b"
)
```

#### OpenAI GPT-4o（云端，高质量）

```python
vlm = VLMConfig(
    backend="openai",
    openai_api_key="sk-...",
    openai_vlm_model="gpt-4o",  # 或 "gpt-4-vision-preview"
    openai_api_url="https://api.openai.com/v1/chat/completions",
)
```

> ⚠️ 注意：GPT-4o 按图像数量计费。构建 2h 视频的金字塔需约 372 次图像 API 调用。

#### Stub（调试，无 API 调用）

```python
vlm = VLMConfig(backend="stub")
# 返回占位文本如："[STUB] Summary... (帧数=4)"
```

### 8.2 文本嵌入后端配置

#### CLIP（推荐，零额外成本）

```python
embedding = EmbeddingConfig(
    text_backend="clip",
    clip_model="ViT-B/32",   # 512-dim，快速
    # clip_model="ViT-L/14", # 768-dim，更精准
    clip_device="cuda",
)
```

#### Ollama nomic-embed-text（更强的文本语义）

```bash
ollama pull nomic-embed-text
```

```python
embedding = EmbeddingConfig(
    text_backend="ollama",
    ollama_embed_url="http://localhost:11434/v1/embeddings",
    ollama_embed_model="nomic-embed-text",
    ollama_embed_dim=768,
    # 注意：使用 ollama 文本后端时，视觉投影层需要将 512-dim 映射到 768-dim
    use_projection=True,
)
```

> ⚠️ 注意：ollama 文本后端与 CLIP 视觉后端的维度不同（768 vs 512），投影层会处于"跨维度"模式，未经训练时效果较差。建议使用 CLIP 全链路或训练投影层权重。

### 8.3 答案生成后端配置

#### DeepSeek（低成本，纯文本）

```python
answer = AnswerConfig(
    backend="deepseek",
    deepseek_api_key="sk-...",
    deepseek_model="deepseek-chat",
    deepseek_url="https://api.deepseek.com/v1/chat/completions",
)
```

特点：无法直接处理图像，依赖 VLM 预生成的文字描述（L1 摘要 + L2 描述）回答问题。

#### OpenAI GPT-4o（高质量，多模态）

```python
answer = AnswerConfig(
    backend="openai",
    openai_api_key="sk-...",
    openai_model="gpt-4o",
    openai_url="https://api.openai.com/v1/chat/completions",
)
```

将关键帧图像（base64 编码）直接传入 GPT-4o，可以识别帧中的细节（文字、物体、表情等）。

#### Ollama LLaVA（本地多模态）

```python
answer = AnswerConfig(
    backend="ollama",
    ollama_chat_url="http://localhost:11434/api/chat",
    ollama_model="llava",
)
```

---

## 9. 输出格式说明

### 9.1 终端输出示例

```
======================================================================
📹  视频    : movie.mp4
❓  问题    : What is written on the door poster?
✅  答案    : The poster on the door reads "SILENCE IN THE LIBRARY".
                It appears to be a hand-written sign in black marker.
----------------------------------------------------------------------
🔍  检索轨迹:
   Phase 1 (粗粒度路由)  → L1[3] : The main character arrives at school and...
   Phase 2 (细粒度聚焦)  → L2[12] : The protagonist argues with the teacher and slams the...
   Phase 3 (视觉定位)    → L3[7] : 时间戳 4582.0s
   目标帧路径  : pyramid_cache/movie/frames/l3_003_0012_000007.jpg
⏱️  耗时     : 2.34s
======================================================================
```

### 9.2 JSON 输出格式（`--output_json`）

```json
[
  {
    "query": "What is written on the door poster?",
    "answer": "The poster on the door reads 'SILENCE IN THE LIBRARY'...",
    "video_name": "movie.mp4",
    "elapsed_sec": 2.34,
    "success": true,
    "retrieval": {
      "k1_star": 3,
      "k2_star": 12,
      "k3_star": 7,
      "target_timestamp": 4582.0,
      "target_frame_path": "pyramid_cache/movie/frames/l3_003_0012_000007.jpg",
      "segment_summary": "The main character arrives at school and...",
      "clip_caption": "The protagonist argues with the teacher and slams..."
    }
  }
]
```

---

## 10. 性能优化建议

### 10.1 预处理加速

**并行 VLM 调用**（当前为串行）：

```python
# 当前实现（串行）：
for seg in segments:
    summary = vlm_gen.describe(frames, prompt)

# 可使用 concurrent.futures 并行化（需确保 API 速率限制）：
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    summaries = list(executor.map(lambda f: vlm_gen.describe(f, prompt), frame_batches))
```

**CLIP 批量编码**：当前已支持批量编码，建议 `batch_size=64`（GPU）或 `batch_size=16`（CPU）。

### 10.2 推理加速

推理阶段（不含 VLM 答案生成）的瓶颈是**查询文本编码**（约 5ms）和**点积计算**（< 1ms）。可通过以下方式优化：

```python
# 预计算查询嵌入（批量问题场景）
q_embeds = retriever._get_text_embedding_batch(queries)  # 未来计划实现

# 使用 numpy 矩阵乘法（已实现，不需要额外优化）
scores = memory_matrix @ query_state  # [N, D] @ [D] = [N]
```

### 10.3 存储优化

**内存映射（大规模视频库）**：

```python
# 对于超大型金字塔（> 1GB），可使用 np.memmap 避免全量加载到内存
import numpy as np
l1_embeds = np.memmap("l1_embeddings.npy", dtype="float32", mode="r")
```

**帧图像压缩**：调整 `jpeg_quality` 参数（默认 85）平衡文件大小和图像质量：

```python
PyramidConfig(max_frame_long_side=224)  # 更小的分辨率（牺牲细节）
```

---

## 11. 常见问题与排错

### Q1：`ImportError: No module named 'clip'`

**解决**：CLIP 需要从 GitHub 安装，不在 PyPI 上：

```bash
pip install git+https://github.com/openai/CLIP.git
```

若 GitHub 访问受限：
```bash
pip install git+https://gitee.com/mirrors/CLIP.git
```

---

### Q2：`CUDA 不可用，回退到 CPU`

**原因**：未安装支持 CUDA 的 PyTorch 版本，或 GPU 驱动不匹配。

**解决**：

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 若为 False，重新安装匹配 CUDA 版本的 PyTorch
# 查看 CUDA 版本：
nvidia-smi | grep "CUDA Version"

# 安装对应版本（以 CUDA 11.8 为例）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU 模式下性能说明：CLIP ViT-B/32 在 CPU 上约 50ms/帧（GPU 约 5ms），2h 视频构建金字塔需约 6 分钟（仅 CLIP 部分）。

---

### Q3：Ollama VLM 调用超时

**原因**：Ollama 服务未启动，或模型尚未下载。

**解决**：

```bash
# 检查 Ollama 服务状态
curl http://localhost:11434/api/tags

# 若无响应，启动服务
ollama serve

# 确认模型已下载
ollama list

# 下载缺失模型
ollama pull llava
```

---

### Q4：`金字塔 M_L1 为空，无法进行检索`

**原因**：金字塔构建失败（0 个 L1 节点）或加载了空的金字塔目录。

**诊断**：

```python
from video_tree_trmRAG.video_pyramid import HierarchicalSemanticPyramid

pyramid = HierarchicalSemanticPyramid.load("pyramid_cache/your_video")
print(pyramid)  # 检查 L1=? L2=? L3=? 的数量

# 若 L1=0，说明构建时出错，重新构建：
```

```bash
python video_tree_trmRAG/build_pyramid.py \
  --video your_video.mp4 \
  --vlm_backend stub \
  --clip_device cpu \
  --force_rebuild \
  --verbose
```

---

### Q5：检索结果不准确（答案与视频内容不匹配）

可能原因及解决方案：

1. **VLM 使用了 stub 模式**：stub 模式不生成真实描述，L1/L2 嵌入无实际语义。改用 `ollama` 或 `openai`。

2. **文本嵌入与视觉嵌入不对齐**：若 `text_backend="ollama"` 但未训练投影层，跨模态检索（Phase 3）效果差。改用 `text_backend="clip"`。

3. **CLIP 模型版本不一致**：构建金字塔时使用的 `clip_model` 与推理时不同，导致嵌入空间不匹配。确保两者一致。

4. **Softmax 温度过高**：设置 `softmax_temperature=0.5` 使选择更确定：
   ```python
   retrieval = RetrievalConfig(softmax_temperature=0.5)
   ```

---

### Q6：`FileNotFoundError: 视频文件不存在`

**解决**：确认视频路径正确，且支持 OpenCV 读取的格式（mp4/mkv/avi/mov）：

```bash
# 检查 OpenCV 是否能读取视频
python -c "
import cv2
cap = cv2.VideoCapture('your_video.mp4')
print('Can open:', cap.isOpened())
print('FPS:', cap.get(cv2.CAP_PROP_FPS))
print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
"
```

---

### Q7：内存不足（OOM）

**解决方案**：

```python
# 减少每层节点数量
config = VideoTreeTRMConfig.from_dict({
    "pyramid": {
        "l1_max_segments": 20,          # 限制 L1 最大节点数
        "l2_max_clips_per_segment": 30, # 限制 L2 最大节点数
        "l3_max_frames_per_clip": 15,   # 限制 L3 最大帧数
        "l3_fps": 0.5,                  # 降低 L3 帧率
    }
})
```

---

## 12. 完整使用场景示例

### 场景：电影问答系统（生产级部署）

```python
import json
from pathlib import Path
from video_tree_trmRAG import VideoQAPipeline, VideoTreeTRMConfig

# 1. 配置系统
config = VideoTreeTRMConfig.from_dict({
    "pyramid": {
        "l1_segment_duration": 600.0,
        "l2_clip_duration": 20.0,
        "l3_fps": 1.0,
        "cache_dir": "/data/movie_pyramids",
    },
    "embedding": {
        "clip_model": "ViT-B/16",    # 略高精度
        "clip_device": "cuda",
        "text_backend": "clip",
    },
    "vlm": {
        "backend": "openai",
        "openai_api_key": "sk-...",
        "openai_vlm_model": "gpt-4o",
    },
    "answer": {
        "backend": "openai",
        "openai_api_key": "sk-...",
        "openai_model": "gpt-4o",
    },
    "verbose": False,  # 生产环境关闭详细日志
})

pipeline = VideoQAPipeline(config)

# 2. 批量构建多个电影的金字塔
movies = [
    "/data/movies/inception.mp4",
    "/data/movies/interstellar.mp4",
]

for movie_path in movies:
    video_name = Path(movie_path).stem
    pyramid_dir = f"/data/movie_pyramids/{video_name}"
    
    if not __import__("video_tree_trmRAG").HierarchicalSemanticPyramid.exists(pyramid_dir):
        print(f"构建金字塔: {video_name}...")
        pipeline.build_pyramid(movie_path, save_dir=pyramid_dir)
    else:
        print(f"金字塔已存在: {video_name}")

# 3. 交互式问答
def ask_movie(movie_name: str, question: str) -> dict:
    pyramid_dir = f"/data/movie_pyramids/{movie_name}"
    result = pipeline.run_from_pyramid(
        pyramid_dir=pyramid_dir,
        query=question,
    )
    return result.to_dict()

# 示例查询
answer = ask_movie("inception", "What is the spinning top test?")
print(json.dumps(answer, indent=2, ensure_ascii=False))
```

### 场景：学术视频理解评估

```python
from video_tree_trmRAG import VideoQAPipeline, VideoTreeTRMConfig
import json

# 加载测试集
with open("test_annotations.json") as f:
    test_data = json.load(f)
# 格式: [{"video": "v001.mp4", "question": "...", "answer": "..."}]

config = VideoTreeTRMConfig.from_dict({
    "vlm": {"backend": "stub"},  # 测试集已有金字塔，无需重建
    "answer": {"backend": "ollama", "ollama_model": "llava"},
    "verbose": False,
})
pipeline = VideoQAPipeline(config)

# 批量评估
correct = 0
for item in test_data:
    pyramid_dir = f"pyramid_cache/{item['video'].replace('.mp4', '')}"
    result = pipeline.run_from_pyramid(
        pyramid_dir=pyramid_dir,
        query=item["question"],
    )
    # 简单字符串匹配（实际评估应使用 BLEU/ROUGE 等指标）
    if item["answer"].lower() in result.answer.lower():
        correct += 1

print(f"准确率: {correct}/{len(test_data)} = {correct/len(test_data)*100:.1f}%")
```

---

## 附录：支持的视频格式

Video-Tree-TRM 使用 OpenCV 读取视频，支持所有 OpenCV 支持的格式：

| 格式 | 扩展名 | 备注 |
|------|-------|------|
| MP4 | `.mp4` | 最推荐，广泛支持 |
| MKV | `.mkv` | 高质量容器 |
| AVI | `.avi` | 传统格式 |
| MOV | `.mov` | Apple QuickTime |
| WebM | `.webm` | 网络视频 |
| FLV | `.flv` | 需要编解码器支持 |

---

*文档版本：v1.0.0 | 最后更新：2026-02-25*

