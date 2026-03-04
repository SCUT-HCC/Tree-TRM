# Video-Tree-TRM RAG：代码编写详细规划文档

> 本文档记录 Video-Tree-TRM 长视频理解 RAG 框架的完整工程规划，涵盖问题背景、设计目标、架构决策、模块划分及开发路线图。

---

## 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [核心挑战分析](#2-核心挑战分析)
3. [设计目标与原则](#3-设计目标与原则)
4. [整体架构规划](#4-整体架构规划)
5. [模块划分与职责说明](#5-模块划分与职责说明)
6. [关键技术选型](#6-关键技术选型)
7. [数据结构设计](#7-数据结构设计)
8. [工作流程规划](#8-工作流程规划)
9. [文件目录结构](#9-文件目录结构)
10. [开发阶段与里程碑](#10-开发阶段与里程碑)
11. [性能基准与评估计划](#11-性能基准与评估计划)
12. [已知局限与未来拓展方向](#12-已知局限与未来拓展方向)

---

## 1. 项目背景与动机

### 1.1 从文本 RAG 到视频 RAG

原始 Tree-TRM 框架是一种用于长文本理解的递归推理 RAG 系统。其核心思想是：
- 将长文本文档组织成**层次化记忆树**
- 利用**递归双循环推理**（Dual-Loop Reasoning）逐层压缩搜索空间
- 通过动态路由（Dynamic Routing）定位到与查询最相关的文本片段

然而，长视频理解面临的挑战远比文本更为复杂：
- 视频数据是**多模态**的（视觉帧 + 可选的音频/字幕），不能直接套用文本 RAG 流程
- 长视频（电影、讲座录像、监控）可长达数小时，直接密集采样会产生**难以处理的帧数量**
- 许多关键信息存在于**像素级细节**中（海报文字、表情、手势），仅靠文字描述难以捕捉

### 1.2 迁移设计核心问题

将 Tree-TRM 的检索方式由**文本检索**改造为**长视频检索**，需要解决三个根本问题：

| 问题 | 文本 RAG 的处理方式 | 视频 RAG 需要的处理方式 |
|------|-------------------|----------------------|
| 内容表示 | 文本块 → 文本嵌入 | 视频帧 → 视觉嵌入 + 文字描述 |
| 记忆结构 | 平铺的文本向量库 | 三层分层语义金字塔（时间分层） |
| 检索粒度 | 单一文本模态 | 多粒度：摘要→描述→视觉帧（跨模态） |

---

## 2. 核心挑战分析

### 2.1 时间维度的多尺度性

长视频中的信息存在严格的**时间层次**：
- **宏观叙事层**（分钟~小时）："整部电影的主线情节"
- **场景动作层**（秒~分钟）："主角与老师发生争吵"  
- **像素细节层**（帧级）："门上海报的具体文字"

这三层信息都可能是回答某个问题的关键，且需要**从粗到细**按需检索。

### 2.2 跨模态语义鸿沟

- **文本查询 vs 视觉内容**：用户用文字提问，最终需要对比视觉帧
- **文本嵌入 vs 视觉嵌入**：CLIP 文本编码器输出与图像编码器输出虽在同一空间，但存在模态偏差
- **解决方案**：引入可学习的 **Proj(·) 投影层**，显式对齐视觉特征与文本潜空间

### 2.3 计算效率约束

以 2 小时电影为例（120 分钟）：
- 若以 1fps 密集采样 → **7,200 帧**，每帧 CLIP 编码约 5ms → 需 36 秒
- 若以 0.5fps 采样 + 层次剪枝 → 实际检索仅需访问 **< 100 帧**
- 金字塔结构使检索复杂度从 **O(N_frames)** 降低至 **O(N_L1 + N_L2 + N_L3)**

### 2.4 VLM 调用成本

- L1/L2 层的文字描述生成需要调用 VLM（多模态大模型）
- 对于 120 分钟视频，若 L1=12段、L2=360片段，则需 372 次 VLM 调用
- 通过**离线预处理**策略，将 VLM 调用与推理时间解耦

---

## 3. 设计目标与原则

### 3.1 功能目标

- [x] **完整的视频预处理流程**：帧提取 → VLM 描述 → 嵌入计算 → 金字塔存储
- [x] **三阶段递归检索**：粗粒度路由 → 细粒度聚焦 → 视觉定位
- [x] **跨模态检索支持**：文本查询可直接对比投影后的视觉嵌入
- [x] **多后端支持**：Ollama（本地）/ OpenAI（云端）/ DeepSeek（云端）
- [x] **持久化与断点续传**：金字塔序列化 + 已处理帧跳过重提取
- [x] **端到端推理管线**：单接口完成从视频文件到最终答案的全流程

### 3.2 工程原则

1. **模块解耦**：每个模块（帧提取、嵌入、检索、生成）独立可测试
2. **延迟加载**：只有在需要时才加载 CLIP/VLM 等大模型，节省内存
3. **配置驱动**：所有超参数通过数据类统一管理，支持 YAML 文件覆盖
4. **容错设计**：VLM 调用失败自动重试（指数退避）；帧读取失败跳过
5. **可解释性**：`RetrievalTrace` 完整记录每阶段的分数分布和中间状态

---

## 4. 整体架构规划

```
Video-Tree-TRM RAG 系统架构
═══════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────┐
  │                    离线预处理阶段（Offline）                   │
  │                                                              │
  │  视频文件 ──► VideoFrameExtractor ──► 帧图像（JPEG）          │
  │                      │                                       │
  │                       ──► VLMDescriptionGenerator            │
  │                              │                               │
  │                              ├─ L1 摘要 → CLIPTextEncoder    │
  │                              │           ↓ M_L1 [N1,D]      │
  │                              └─ L2 描述 → CLIPTextEncoder    │
  │                                          ↓ M_L2[i] [N2,D]  │
  │                                                              │
  │  帧图像 ──► CLIPImageEncoder ──► VisualProjectionLayer       │
  │                                          ↓ M_L3[i,j][N3,D] │
  │                                                              │
  │  M_L1 + M_L2 + M_L3 ──► HierarchicalSemanticPyramid.save() │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼（持久化至磁盘）
  ┌──────────────────────────────────────────────────────────────┐
  │                     在线推理阶段（Online）                     │
  │                                                              │
  │  用户查询 q ──► CLIPTextEncoder ──► q_embed [D]              │
  │                                                              │
  │  z_0 = q_embed                                               │
  │                                                              │
  │  Phase 1: TRM-Attention(q+z_0, M_L1) → k1*, z_1            │
  │  Phase 2: TRM-Attention(q+z_1, M_L2[k1*]) → k2*, z_2       │
  │  Phase 3: TRM-Attention(q+z_2, M_L3[k1*,k2*]) → k3*       │
  │                                                              │
  │  F_target = frames/l3_k1_k2_k3.jpg                          │
  │                                                              │
  │  F_target + query ──► AnswerGenerator (VLM) ──► answer      │
  └──────────────────────────────────────────────────────────────┘
```

---

## 5. 模块划分与职责说明

### 5.1 模块清单

| 文件 | 类/函数 | 核心职责 |
|------|--------|---------|
| `config.py` | `VideoTreeTRMConfig` 等 5 个 dataclass | 统一配置管理，支持字典/YAML 加载 |
| `video_pyramid.py` | `HierarchicalSemanticPyramid`, `SegmentNode`, `ClipNode`, `FrameNode` | HSP 三层数据结构，序列化/反序列化 |
| `video_indexer.py` | `VideoFrameExtractor`, `VLMDescriptionGenerator`, `CLIPFeatureExtractor`, `OllamaTextEmbedder`, `QwenSceneDetector`★, `QwenKeyframeScorer`★, `segment_video`, `segment_video_smart`★ | 视频原始素材处理、特征提取；★= v1.1 新增 |
| `visual_projection.py` | `VisualProjectionLayer`, `ProjectionManager` | 视觉特征 → 文本潜空间对齐投影 |
| `video_tree_trm.py` | `VideoTreeTRM`, `RetrievalTrace` | 三阶段递归检索核心引擎 |
| `answer_generator.py` | `AnswerGenerator` | 基于关键帧的 VLM 答案生成 |
| `pipeline.py` | `VideoQAPipeline`, `VideoQAResult` | 端到端统一推理管线；v1.1 新增 Qwen 两条增强分支 |
| `build_pyramid.py` | `main()` | 离线金字塔构建 CLI 工具；v1.1 新增 7 个 Qwen 参数 |
| `run_videoqa.py` | `main()` | 在线视频问答 CLI 工具 |

### 5.2 模块依赖关系

```
config.py
  ↑ (依赖)
  ├── video_pyramid.py
  ├── video_indexer.py
  ├── visual_projection.py
  ├── video_tree_trm.py  ← 依赖 video_pyramid.py
  ├── answer_generator.py ← 依赖 video_tree_trm.py
  └── pipeline.py ← 依赖上述所有模块

build_pyramid.py ← 依赖 config.py, pipeline.py
run_videoqa.py   ← 依赖 config.py, pipeline.py
```

---

## 6. 关键技术选型

### 6.1 嵌入模型

| 模型 | 用途 | 维度 | 优点 | 缺点 |
|------|------|------|------|------|
| CLIP ViT-B/32 | L1/L2 文本嵌入 + L3 视觉嵌入 | 512 | 天然跨模态对齐空间，无需额外训练投影层 | 文本理解能力弱于 LLM-based 嵌入 |
| CLIP ViT-B/16 | 同上，精度更高 | 512 | 比 ViT-B/32 更精细的视觉理解 | 计算量约 2× |
| CLIP ViT-L/14 | 高质量场景 | 768 | 最高精度 | 需要更多显存 |
| Ollama nomic-embed-text | 仅 L1/L2 文本嵌入 | 768 | 强大的中文/英文语义理解 | 无法直接与视觉嵌入对齐，需训练投影层 |

**推荐组合（默认）**：CLIP ViT-B/32 全链路（L1/L2 文本 + L3 视觉均用 CLIP），实现零训练部署。

### 6.2 VLM 后端

| 后端 | 场景 | 质量 | 成本 | 隐私 |
|------|------|------|------|------|
| Ollama (LLaVA) | 本地/离线部署 | 中等 | 免费（算力成本） | 高（数据不出本机） |
| OpenAI GPT-4o | 最高质量需求 | 最高 | 高（约 $0.01/帧） | 低 |
| Stub（调试） | 开发/单元测试 | N/A | 零 | N/A |

### 6.3 答案生成后端

| 后端 | 模式 | 能否处理图像 | 推荐场景 |
|------|------|------------|---------|
| DeepSeek | 纯文本 | ❌（仅文字上下文） | 低成本、文字描述已足够 |
| OpenAI GPT-4o | 多模态 | ✅ | 需要精确视觉识别的问题 |
| Ollama LLaVA | 多模态（本地） | ✅ | 本地部署、数据隐私 |

### 6.4 视频处理

- **帧提取**：OpenCV（`cv2.VideoCapture`）—— 成熟稳定，支持所有主流视频格式
- **图像处理**：PIL/Pillow —— 轻量易用的图像缩放与格式转换
- **存储格式**：JPEG（帧图像，有损但高压缩率）+ NumPy `.npy`（嵌入矩阵，高效）+ JSON（元数据，可读性强）

---

## 7. 数据结构设计

### 7.1 三层节点层次

```python
HierarchicalSemanticPyramid
│  video_path: str
│  video_duration: float
│  embed_dim: int
│
└── List[SegmentNode]                    # L1 全局叙事节点，N1 个
      │  seg_idx: int
      │  start_time, end_time: float
      │  summary: str                    # VLM 生成的宏观叙事摘要
      │  text_embedding: np.ndarray[D]  # CLIP 文本嵌入
      │
      └── List[ClipNode]                # L2 片段级语义节点，N2 个/段
            │  clip_idx: int
            │  start_time, end_time: float
            │  caption: str             # VLM 生成的精细描述
            │  text_embedding: np.ndarray[D]
            │
            └── List[FrameNode]         # L3 帧级视觉节点，N3 帧/片段
                  frame_idx: int
                  timestamp: float
                  frame_path: str       # 磁盘上的 JPEG 路径
                  visual_embedding: np.ndarray[D]  # Proj(CLIP(frame))
```

### 7.2 序列化文件布局

```
pyramid_cache/<video_name>/
├── metadata.json              # 金字塔元数据（视频信息、维度、时间参数）
├── build_summary.json         # 构建过程摘要（耗时、统计信息）
├── l1_embeddings.npy          # L1 嵌入矩阵，shape [N1, D]
├── l1_metadata.json           # L1 节点时间/文本信息列表
├── l2_embeddings_0.npy        # L1[0] 段的 L2 嵌入矩阵，shape [N2_0, D]
├── l2_metadata_0.json         # L1[0] 段的 L2 节点信息
├── l3_embeddings_0_0.npy      # L1[0].L2[0] 片段的 L3 嵌入矩阵，shape [N3_00, D]
├── l3_metadata_0_0.json       # L1[0].L2[0] 片段的帧节点信息（含路径）
├── ... (更多 l2/l3 文件)
└── frames/                    # 所有提取的帧图像
      ├── l1_000_rep_000000.jpg
      ├── l2_000_0000_rep_000000.jpg
      ├── l3_000_0000_000000.jpg
      └── ...
```

### 7.3 检索结果数据结构

```python
@dataclass
class RetrievalTrace:
    query: str                          # 原始查询文本
    query_embedding: np.ndarray[D]     # 查询嵌入向量
    k1_star: int                        # Phase 1 选中的 L1 段索引
    k2_star: int                        # Phase 2 选中的 L2 片段索引
    k3_star: int                        # Phase 3 选中的 L3 帧索引
    l1_scores: np.ndarray[N1]          # Phase 1 Softmax 分数分布
    l2_scores: np.ndarray[N2]          # Phase 2 Softmax 分数分布
    l3_scores: np.ndarray[N3]          # Phase 3 Softmax 分数分布
    z0, z1, z2: np.ndarray[D]          # 各阶段潜在状态向量
    segment_summary: str                # 选中 L1 段的摘要文本
    clip_caption: str                   # 选中 L2 片段的描述文本
    target_frame: FrameNode            # 目标帧节点
    target_timestamp: float            # 目标帧的绝对时间戳（秒）
```

---

## 8. 工作流程规划

### 8.1 离线预处理流程（一次性）

```
输入：原始视频文件 video.mp4

步骤 1：获取视频基础信息
  - 使用 cv2.VideoCapture 读取总帧数和帧率（PyAV 兜底）
  - 计算视频总时长 duration

步骤 2：时间分段（两种模式，任选其一）
  ┌─ 模式 A（默认）：固定步长切分
  │   - L1 分段：每 600s（10min）一个宏观事件段，最多 50 段
  │   - L2 分段：每段内每 20s 一个短片段，最多 60 个/段
  │
  └─ 模式 B（★v1.1，use_qwen_scene_detection=True）：Qwen-VL 场景感知切分
      a. 以 scene_detection_probe_fps（默认 0.5fps）采样全局探测帧
      b. QwenSceneDetector 以滑动窗口批量调用 Qwen-VL，返回场景边界时间戳
      c. segment_video_smart：合并过短段（< l1_min_duration）、
         拆分过长段（> l1_max_duration），生成语义对齐的 L1 分段
      d. L2 仍采用固定步长

步骤 3：L1 节点构建（循环每个 L1 段）
  a. 以低帧率（0.1fps）提取代表帧，保存为 JPEG
  b. 均匀采样 ≤6 帧送入 VLM，生成宏观叙事摘要 S_i
  c. 对 S_i 进行 CLIP 文本编码，得到 text_embedding [D]
  d. 创建 SegmentNode，存储时间信息、摘要文本和嵌入向量

步骤 4：L2 节点构建（循环每个 L2 片段）
  a. 以低帧率提取代表帧
  b. 均匀采样 ≤4 帧送入 VLM，生成精细片段描述 D_{i,j}
  c. 对 D_{i,j} 进行 CLIP 文本编码
  d. 创建 ClipNode

步骤 5：L3 节点构建（同时进行）
  a. 以 1fps 提取高密度候选帧
  ┌─ 步骤 5b（★v1.1，use_qwen_keyframe_scoring=True）：Qwen-VL 关键帧筛选
  │   - 加载 clip 内所有候选帧图像
  │   - QwenKeyframeScorer 一次调用 Qwen-VL，返回 Top-K 帧索引（重要性排序）
  │   - 按时序重排，保留 Top-K 帧替代全部候选帧
  └─ 步骤 5b（默认）：截断到 l3_max_frames_per_clip
  c. 批量送入 CLIP 图像编码器，得到视觉嵌入 [N3, 512]
  d. 通过 VisualProjectionLayer 投影到文本潜空间
  e. 创建 FrameNode（含帧路径和视觉嵌入）

步骤 6：保存金字塔
  - 序列化所有嵌入矩阵为 .npy 文件
  - 序列化文本元数据为 JSON 文件
  - 写入 metadata.json 记录配置和统计信息
  - 写入 build_summary.json 记录构建耗时和节点统计
```

### 8.2 在线推理流程（每次查询）

```
输入：用户查询 q（自然语言文本）

步骤 0：初始化
  - 加载金字塔（从磁盘）
  - q_embed = CLIP_text(q)，L2 归一化
  - z_0 = q_embed（以查询语义初始化潜在状态）

步骤 1：Phase 1 粗粒度路由（文本对文本）
  - query_state = normalize(q + z_0)
  - scores_L1 = Softmax((query_state · M_L1ᵀ) / √D)
  - k1* = argmax(scores_L1)
  - z_1 = normalize(z_0 + M_L1[k1*])
  → 定位到最相关的宏观事件段

步骤 2：Phase 2 细粒度聚焦（文本对文本）
  - query_state = normalize(q + z_1)
  - scores_L2 = Softmax((query_state · M_L2[k1*]ᵀ) / √D)
  - k2* = argmax(scores_L2)
  - z_2 = normalize(z_1 + M_L2[k1*][k2*])
  → 锁定到具体的视频片段

步骤 3：Phase 3 视觉定位（文本对图像，跨模态）
  - query_state = normalize(q + z_2)
  - scores_L3 = Softmax((query_state · M_L3[k1*,k2*]ᵀ) / √D)
  - k3* = argmax(scores_L3)
  → 精确定位到包含视觉证据的关键帧

步骤 4：答案生成
  - F_target = Frame[k1*, k2*, k3*]（从磁盘加载 JPEG）
  - context = {摘要 S_{k1*}, 描述 D_{k1*,k2*}, 时间戳}
  - answer = VLM(q, F_target, context)
```

---

## 9. 文件目录结构

```
video_tree_trmRAG/
│
├── __init__.py                  # 包入口，导出公共 API
├── config.py                    # 配置数据类
├── video_pyramid.py             # HSP 数据结构与持久化
├── video_indexer.py             # 视频处理与特征提取
├── visual_projection.py         # 视觉特征投影层
├── video_tree_trm.py            # 三阶段递归检索引擎（核心）
├── answer_generator.py          # VLM 答案生成
├── pipeline.py                  # 端到端推理管线
│
├── build_pyramid.py             # 离线金字塔构建 CLI 工具
├── run_videoqa.py               # 在线视频问答 CLI 工具
│
├── requirements_video.txt       # Python 依赖清单
│
├── PLAN.md                      # 本文档：代码编写规划
├── CODE_EXPLANATION.md          # 代码思路详细解释
└── USAGE_GUIDE.md               # 使用指南
```

---

## 10. 开发阶段与里程碑

### Phase 0：基础设施搭建（已完成）

- [x] **配置系统**（`config.py`）：设计 5 层嵌套配置数据类，支持 YAML 加载
- [x] **金字塔数据结构**（`video_pyramid.py`）：定义三层节点 + HSP 容器 + 序列化接口

### Phase 1：预处理模块（已完成）

- [x] **帧提取器**（`VideoFrameExtractor`）：基于 OpenCV，支持时间区间、帧率配置、断点续传
- [x] **时间分段函数**（`segment_video`）：两级时间切分，支持最大节点数限制
- [x] **VLM 描述生成器**（`VLMDescriptionGenerator`）：Ollama/OpenAI/Qwen/Stub 四后端，指数退避重试
- [x] **CLIP 特征提取器**（`CLIPFeatureExtractor`）：批量图像和文本编码，延迟加载模型
- [x] **Ollama 文本嵌入器**（`OllamaTextEmbedder`）：调用 nomic-embed-text API

### Phase 1.1：Qwen-VL 智能增强（已完成，v1.1）

- [x] **Qwen-VL 场景检测器**（`QwenSceneDetector`）：滑动窗口批量调用 Qwen-VL，检测场景边界时间戳
- [x] **语义感知切分函数**（`segment_video_smart`）：基于边界时间戳的动态 L1 分段（含合并/拆分逻辑）
- [x] **Qwen-VL 关键帧打分器**（`QwenKeyframeScorer`）：对 L3 候选帧评分，保留 Top-K 高信息密度帧
- [x] **Config 字段扩展**（`PyramidConfig`）：新增 7 个 Qwen 增强相关配置字段
- [x] **Pipeline 集成**（`VideoQAPipeline.build_pyramid`）：Qwen 组件按开关字段惰性初始化，向后兼容
- [x] **CLI 参数扩展**（`build_pyramid.py`）：新增 7 个 `--use_qwen_*` 系列参数

### Phase 2：嵌入对齐（已完成）

- [x] **视觉投影层**（`VisualProjectionLayer`）：可学习线性投影，支持同维度残差结构和恒等初始化
- [x] **投影管理器**（`ProjectionManager`）：封装权重加载、批量投影、numpy 接口

### Phase 3：检索引擎（已完成）

- [x] **Tree-TRM 注意力核**（`tree_trm_attention`）：缩放点积 + Softmax + argmax
- [x] **潜在状态更新函数**（`update_latent_state`）：additive/replace/gated 三种模式
- [x] **三阶段检索主体**（`VideoTreeTRM.retrieve`）：Phase 1/2/3 完整实现
- [x] **检索轨迹记录**（`RetrievalTrace`）：存储所有中间状态，支持可解释性分析

### Phase 4：答案生成（已完成）

- [x] **DeepSeek 纯文本模式**：低成本文本问答
- [x] **OpenAI GPT-4o 多模态模式**：高质量视觉问答
- [x] **Ollama LLaVA 本地模式**：本地视觉问答

### Phase 5：集成与工具（已完成）

- [x] **端到端管线**（`VideoQAPipeline`）：统一入口，支持完整流程和仅推理模式
- [x] **金字塔构建 CLI**（`build_pyramid.py`）：完整参数配置，支持所有后端
- [x] **VideoQA 推理 CLI**（`run_videoqa.py`）：单/批量问答，JSON 输出，检索轨迹显示

### 后续拓展方向（规划中）

- [ ] **训练投影层**：收集视频-文本对齐数据，端到端微调 `VisualProjectionLayer`
- [ ] **Top-K 多路径检索**：在每个阶段保留 Top-K 候选，最终 rerank（`QwenKeyframeScorer` 已提供 reranking 雏形）
- [ ] **音频模态融合**：提取语音转文字（ASR），增强 L2 节点语义
- [ ] **字幕/OCR 融合**：视频字幕作为辅助 L2 描述来源
- [ ] **Qwen-VL Reranking**：利用 `QwenKeyframeScorer` 对 Phase 3 Top-3 候选帧做二次精排
- [ ] **并行 VLM 调用**：使用 `ThreadPoolExecutor` 并行化 L1/L2 的 VLM 描述生成
- [ ] **EvalKit**：基于 EgoSchema/ActivityNet-QA/Video-MME 等 VideoQA 基准的自动评测脚本

---

## 11. 性能基准与评估计划

### 11.1 时间复杂度分析

设视频时长 $T$ 秒，各层参数如下：
- $N_1 = \lceil T / t_{L1} \rceil$：L1 段数量（$t_{L1}=600$s）
- $N_2 \approx N_1 \cdot \lceil t_{L1} / t_{L2} \rceil$：L2 总片段数（$t_{L2}=20$s，每段约 30 片段）
- $N_3 \approx N_2 \cdot t_{L2} \cdot \text{fps}_{L3}$：L3 总帧数（$\text{fps}_{L3}=1$）

**预处理阶段**（离线，一次性）：
$$T_{preprocess} = O(N_3) \cdot t_{CLIP} + O(N_1 + N_2) \cdot t_{VLM}$$
- 典型 2 小时电影：$N_1=12$，$N_2=360$，$N_3=7200$
- CLIP 编码 7200 帧（批量）：约 36s
- VLM 调用 372 次（stub/ollama/openai 差异巨大）：0s / ~2h / ~37min

**推理阶段**（在线，每次查询）：
$$T_{inference} = O(N_1 + N_2^{(k1^*)} + N_3^{(k1^*,k2^*)}) \cdot t_{dot} + t_{VLM}^{answer}$$
- 仅对 $N_1 + N_2^{(k1^*)} + N_3^{(k1^*,k2^*)}$ 个节点计算点积（~几十个）
- 推理时不调用 CLIP 图像编码（已预计算），仅需 CLIP 文本编码（~5ms）
- 总推理时间：**< 100ms**（不含答案生成 VLM 调用）

### 11.2 存储空间估算

| 数据 | 计算 | 2h 视频估算 |
|------|------|------------|
| JPEG 帧图像（L3，336px，1fps） | $N_3 \times$ ~10KB | ~72MB |
| L1/L2 嵌入矩阵（512-dim float32） | $(N_1+N_2) \times 512 \times 4$ bytes | ~750KB |
| L3 嵌入矩阵（512-dim float32） | $N_3 \times 512 \times 4$ bytes | ~15MB |
| JSON 元数据 | 约 100 bytes/节点 | ~100KB |
| **合计** | — | **~88MB** |

---

## 12. 已知局限与未来拓展方向

### 12.1 当前局限

1. **固定三层结构**：对于极短（< 10分钟）或极长（> 10小时）视频，三层可能不是最优划分
2. **硬选择（argmax）**：推理时使用 argmax，不可微分，无法端到端训练整个检索链
3. **投影层未预训练**：默认使用恒等初始化，跨模态对齐依赖 CLIP 原始空间，存在模态偏差
4. **无音频/字幕模态**：目前仅处理视觉帧，忽略了视频的语音和字幕信息
5. **顺序/串行处理**：预处理中 VLM 调用为串行，耗时较长

### 12.2 拓展路线图

```
V1.0：基础三层视觉 RAG（固定步长切分 + 均匀帧采样）
    ↓
V1.1（当前）：Qwen-VL 双重增强
             ├─ 场景感知 L1 切分（QwenSceneDetector + segment_video_smart）
             └─ L3 关键帧精选（QwenKeyframeScorer，预期 +10-15% 准确率）
    ↓
V1.2：引入 Top-K 多路径检索 + Qwen-VL 最终 Rerank
    ↓
V1.3：训练 VisualProjectionLayer（对比学习）
    ↓
V1.4：融合 ASR 字幕，L2 节点同时包含视觉描述和语音转录
    ↓
V2.0：可微分软选择（Soft Selection），支持端到端梯度反传
    ↓
V2.1：自适应层数（动态 H），根据查询复杂度自动决定检索深度
    ↓
V3.0：多视频跨文件检索（多视频金字塔合并）
```

---

*文档版本：v1.1.0 | 最后更新：2026-03-04 | 新增：Phase 1.1 Qwen-VL 智能增强模块、更新架构路线图*

