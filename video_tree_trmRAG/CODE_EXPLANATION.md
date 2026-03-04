# Video-Tree-TRM：代码思路详细解释文档

> 本文档对 `video_tree_trmRAG/` 目录下所有核心模块的实现思路、算法细节和关键设计决策进行深入解读，适合需要理解、修改或拓展本系统的开发者阅读。

---

## 目录

1. [系统总览：从文本 RAG 到视频 RAG 的迁移思路](#1-系统总览)
2. [config.py：配置系统设计](#2-configpy-配置系统设计)
3. [video_pyramid.py：分层语义金字塔数据结构](#3-video_pyramidpy-分层语义金字塔数据结构)
4. [video_indexer.py：视频原始素材处理](#4-video_indexerpy-视频原始素材处理)
   - 4.1 `VideoFrameExtractor`：帧提取设计
   - 4.2 `segment_video` / `segment_video_smart`：两级时间分段
   - 4.3 `QwenSceneDetector`：Qwen-VL 智能场景检测
   - 4.4 `QwenKeyframeScorer`：Qwen-VL 关键帧打分筛选
   - 4.5 `VLMDescriptionGenerator`：VLM 描述生成
   - 4.6 `sample_representative_frames`：均匀代表帧采样
5. [visual_projection.py：跨模态特征对齐](#5-visual_projectionpy-跨模态特征对齐)
6. [video_tree_trm.py：三阶段递归检索引擎（核心）](#6-video_tree_trmpy-三阶段递归检索引擎核心)
7. [answer_generator.py：答案生成模块](#7-answer_generatorpy-答案生成模块)
8. [pipeline.py：端到端推理管线](#8-pipelinepy-端到端推理管线)
9. [关键算法详解：Tree-TRM 注意力机制](#9-关键算法详解tree-trm-注意力机制)
10. [模态切换机制解析](#10-模态切换机制解析)
11. [潜在状态演化分析](#11-潜在状态演化分析)
12. [设计决策总结](#12-设计决策总结)

---

## 1. 系统总览

### 1.1 核心迁移思路

原始 Tree-TRM 是一个**文本 RAG 系统**，其记忆库是由文本块（Chunk）嵌入组成的向量矩阵。迁移到长视频检索需要解决的核心问题是：

**如何将时序结构的视频内容表示为可被 Tree-TRM 递归遍历的记忆结构？**

解决方案是引入**分层语义金字塔（Hierarchical Semantic Pyramid, HSP）**，将视频的时间维度映射为树形层次结构：

```
时间轴  ─────────────────────────────────────────────►
              宏观叙事（10-20min/段）      ← L1：文本摘要嵌入
          ┌──────────┬──────────┐
          场景①    场景②    场景③     ← L2：文本描述嵌入
        ┌─┴─┐    ┌─┴─┐    ┌─┴─┐
        帧帧帧    帧帧帧    帧帧帧     ← L3：视觉嵌入（CLIP+Proj）
```

这样，Tree-TRM 的三个递归阶段自然对应三个粒度层：
- **Phase 1**（H=1）：在 L1 层的文本摘要空间中路由 → "这个问题属于哪个宏观事件段？"
- **Phase 2**（H=2）：在 L2 层的文本描述空间中聚焦 → "在该事件段的哪个具体片段？"
- **Phase 3**（H=3）：在 L3 层的视觉特征空间中定位 → "具体是哪一帧画面？"

### 1.2 模块职责链

```
视频文件
  │
  ▼
VideoFrameExtractor（video_indexer.py）
  │ 帧提取（JPEG）
  ▼
QwenSceneDetector [可选]（video_indexer.py）    ← v1.1 新增
  │ 探测帧 → Qwen-VL → 场景边界时间戳
  ▼
segment_video_smart / segment_video（video_indexer.py）
  │ L1/L2 时间分段（语义感知 或 固定步长）
  ▼
VLMDescriptionGenerator（video_indexer.py）
  │ L1摘要 + L2描述（VLM生成文字）
  ▼
CLIPFeatureExtractor / OllamaTextEmbedder（video_indexer.py）
  │ L1/L2文本嵌入 + L3视觉嵌入（原始候选帧）
  ▼
QwenKeyframeScorer [可选]（video_indexer.py）   ← v1.1 新增
  │ 对 L3 候选帧打分 → 保留 Top-K 关键帧
  ▼
VisualProjectionLayer（visual_projection.py）
  │ 视觉特征 → 文本潜空间（Proj 算子）
  ▼
HierarchicalSemanticPyramid（video_pyramid.py）
  │ 三层结构组织、持久化
  ▼
VideoTreeTRM（video_tree_trm.py）
  │ 三阶段递归检索：k1*, k2*, k3*
  ▼
AnswerGenerator（answer_generator.py）
  │ VLM 答案生成（含关键帧图像）
  ▼
最终答案
```

---

## 2. `config.py`：配置系统设计

### 2.1 设计思路

采用**嵌套数据类（Nested Dataclasses）**组织配置，每个子系统对应一个独立的配置类：

```python
VideoTreeTRMConfig          # 根配置
  ├── PyramidConfig         # 金字塔构建参数
  ├── EmbeddingConfig       # 嵌入模型参数  
  ├── VLMConfig             # VLM 描述生成参数
  ├── AnswerConfig          # 答案生成参数
  └── RetrievalConfig       # 检索引擎参数
```

### 2.2 `PyramidConfig` 新增 Qwen 增强字段（v1.1）

为支持 Qwen-VL 智能场景切分与关键帧筛选，`PyramidConfig` 新增以下字段分为两组：

**场景检测组（L1 语义切分）**：

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `use_qwen_scene_detection` | bool | False | 开关：启用 Qwen-VL 替代固定步长切分 |
| `scene_detection_probe_fps` | float | 0.5 | 探测帧率（帧/秒）；越高边界越精确，API 调用越多 |
| `scene_detection_batch_size` | int | 8 | 每批发送帧数（相邻批次重叠 1 帧） |
| `l1_min_duration` | float | 60.0 | 合并过短段的阈值（秒） |
| `l1_max_duration` | float | 1200.0 | 拆分过长段的阈值（秒） |

**关键帧筛选组（L3 信息密度优化）**：

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `use_qwen_keyframe_scoring` | bool | False | 开关：启用 Qwen-VL 对 L3 帧打分筛选 |
| `qwen_keyframe_keep_ratio` | float | 0.5 | 每 clip 内保留帧比例（0.0~1.0） |

两组字段均在 `VLMConfig.qwen_api_key` 为空时自动降级，不影响原有功能。

### 2.3 关键设计点

**`EmbeddingConfig.embed_dim` 属性**：

```python
@property
def embed_dim(self) -> int:
    if self.text_backend == "ollama":
        return self.ollama_embed_dim  # 768
    _clip_dims = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, ...}
    return _clip_dims.get(self.clip_model, 512)
```

这个设计确保：无论使用 CLIP 还是 Ollama 作为文本后端，`embed_dim` 总是返回统一的嵌入空间维度 $D$，整个系统只需引用 `cfg.embedding.embed_dim` 即可，不需要在各模块中重复判断维度。

**`from_dict` 方法的健壮性处理**：

```python
valid_fields = {f.name for f in sub_cls.__dataclass_fields__.values()}
filtered = {k: v for k, v in value.items() if k in valid_fields}
kwargs[key] = sub_cls(**filtered)
```

通过过滤未知字段，保证从字典（或 YAML）加载时不会因配置文件版本不匹配而崩溃，提升了系统的向前兼容性。

---

## 3. `video_pyramid.py`：分层语义金字塔数据结构

### 3.1 节点层次设计

三层节点形成严格的**树形父子关系**：

```
SegmentNode（L1）           ← 粗粒度，文字摘要
  └─ List[ClipNode]（L2）  ← 中粒度，文字描述
        └─ List[FrameNode]（L3）  ← 细粒度，原始视觉
```

每个节点除了存储文本/视觉内容外，都携带**时间元数据**（`start_time`, `end_time`），这保证了检索结果可以被精确映射回视频时间轴。

### 3.2 嵌入矩阵访问接口

```python
M_L1 = pyramid.get_l1_embeddings()           # [N1, D]
M_L2 = pyramid.get_l2_embeddings(k1=2)        # [N2, D]
M_L3 = pyramid.get_l3_embeddings(k1=2, k2=5)  # [N3, D]
```

这三个方法的设计允许 `VideoTreeTRM` 检索引擎按需**懒加载**各层嵌入矩阵，而不是在初始化时将所有嵌入都加载到内存。对于含数百个 L2 节点的大型金字塔，这显著减少了内存占用。

### 3.3 序列化策略

序列化格式将**嵌入矩阵**（`.npy`，二进制，高效）和**文本/元数据**（`.json`，可读）分开存储：

```
pyramid_cache/movie/
├── l1_embeddings.npy       # 紧凑二进制，快速 np.load
├── l1_metadata.json        # 人类可读，含摘要文本
├── l2_embeddings_0.npy     # 按 L1 段索引分文件
└── l3_embeddings_0_0.npy   # 按 (L1, L2) 索引分文件
```

分文件的好处：加载时可以**按需加载**特定段的 L2/L3 嵌入，未来可扩展为内存映射（`np.memmap`）以支持超大规模视频库。

---

## 4. `video_indexer.py`：视频原始素材处理

### 4.1 `VideoFrameExtractor`：帧提取设计

**核心实现逻辑**：

```python
timestamps = np.arange(start_sec, end_sec, interval)
for ts in timestamps:
    frame_pos = int(ts * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, bgr = cap.read()
```

关键点：使用 `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)` 精确跳转到目标帧，而不是逐帧遍历。这对于稀疏采样（如 0.1fps）非常高效，避免了解码所有中间帧。

**断点续传机制**：

```python
if not os.path.exists(fpath):
    img.save(fpath, "JPEG", quality=self.jpeg_quality)
```

若帧图像文件已存在则跳过保存，这在预处理被中断后重新运行时能自动跳过已完成的帧，无需重复计算。

### 4.2 `segment_video` / `segment_video_smart`：两级时间分段

**固定步长版本 `segment_video`**（默认）：

```python
l1_starts = np.arange(0.0, video_duration, l1_duration)[:l1_max]
for l1_s in l1_starts:
    l1_e = min(l1_s + l1_duration, video_duration)
    l2_starts = np.arange(l1_s, l1_e, l2_duration)[:l2_max_per_seg]
    ...
```

使用 `np.arange` 生成等间隔时间点，截断到最大节点数，末尾段/片段通过 `min` 保证不超过视频时长。

**语义感知版本 `segment_video_smart`**（Qwen 增强）：

```python
def segment_video_smart(video_duration, boundary_timestamps,
                         l1_min_duration, l1_max_duration, ...):
    # 1. 候选边界 = 场景检测点 + 首尾
    # 2. 贪心合并（过短段：< l1_min_duration 合并到前段）
    # 3. 均匀拆分（过长段：> l1_max_duration 均匀拆为 n 段）
    # 4. 截断到 l1_max
    # 5. 每段内固定步长生成 L2
```

与 `segment_video` 的本质区别：L1 边界由 Qwen-VL 检测的场景转换点决定，而非固定步长。这保证了每个 L1 段内的内容语义连贯（不会"切断"一场正在进行的戏剧场景），使 L1 摘要更准确，Phase 1 路由命中率更高。

### 4.3 `QwenSceneDetector`：Qwen-VL 智能场景检测

**核心问题**：固定时长切分可能将同一场景切成两段（如一场对话被 600s 边界截断），或将多个不同场景合并到同一段。`QwenSceneDetector` 通过视觉内容感知解决这一问题。

**工作原理（滑动窗口）**：

```
全部探测帧（0.5fps）: [F0, F1, F2, F3, F4, F5, F6, F7, ...]
                        └──────batch_size=4──────┘
                           Qwen-VL: "哪对相邻帧之间有场景切换？"
                           → JSON: [1, 3]（即 F1→F2 和 F3→F4 切换）
                                         步长=3（重叠1帧）
                                         └──────batch_size=4──────┘
                                              Qwen-VL: [...]
```

```python
class QwenSceneDetector:
    _PROMPT = (
        "...Identify which consecutive frame pairs show a SIGNIFICANT scene change..."
        "Respond ONLY with a valid JSON array of 0-based frame indices after which..."
    )
    
    def detect_boundaries(self, frames_with_timestamps, batch_size=8):
        # 滑动窗口，步长 = batch_size - 1（首尾重叠1帧保证连续性）
        while start < n - 1:
            local_indices = self._call_qwen(batch_imgs)  # → [2, 5] 等
            for local_idx in local_indices:
                boundary_timestamps.append(midpoint(ts[idx], ts[idx+1]))
            start += step
        return sorted(set(boundary_timestamps))
```

**关键设计点**：
1. **结构化输出**：Prompt 要求 Qwen-VL 返回纯 JSON 数组，用 `re.search(r'\[.*?\]', ...)` 鲁棒解析
2. **重叠帧**：相邻批次共享 1 帧，确保批次边界处的场景切换不被遗漏
3. **边界取中点**：场景边界时间 = 两帧时间戳的均值，定位更精确
4. **零温度**：`temperature=0.0` 保证输出确定性，避免随机性引入错误检测
5. **兜底降级**：调用失败时返回 `[]`，`segment_video_smart` 退化为只用首尾作边界，等价于 `segment_video`

**API 调用次数估算（2h 视频，0.5fps，batch=8）**：
- 探测帧数 = 3600 帧；批次数 = 3600 / 7 ≈ 515 次调用
- 与 L1/L2 VLM 描述生成（约 372 次）量级相当

### 4.4 `QwenKeyframeScorer`：Qwen-VL 关键帧打分筛选

**核心问题**：以 1fps 从 20s 的 L2 clip 中提取 20 帧，其中大量帧可能是静止背景、闭眼瞬间或无关过渡帧，这些帧作为 L3 节点会稀释有效信号，增加 Phase 3 检索噪声。

**工作原理**：

```python
class QwenKeyframeScorer:
    _PROMPT_TMPL = (
        "You are analyzing {n} frames...Rate each frame's importance..."
        "Respond ONLY with a valid JSON array of frame indices (0-based) sorted by importance..."
        "containing exactly the top {top_k} frame indices..."
    )
    
    def filter_top_k(self, frames_meta, images, top_k):
        if len(images) <= top_k:
            return frames_meta, images  # 无需筛选
        
        ranked = self._call_qwen(images, top_k)  # → [4, 12, 7, 1, ...]（重要性降序）
        if not ranked:
            indices = uniform_downsample(n, top_k)  # 兜底
        else:
            indices = sorted(ranked[:top_k])  # 按时序重排，保持时间连续性
        return frames_meta[indices], images[indices]
```

**一次调用解决问题**：不同于场景检测需要多次调用，每个 L2 clip 只需**一次 API 调用**，将该 clip 内所有帧送入 Qwen-VL 并获取排序结果。

**`pipeline.py` 中的集成逻辑**：

```python
if qwen_keyframe_scorer is not None and len(l3_frames_meta) > 1:
    top_k = max(1, round(len(valid_imgs) * pyr_cfg.qwen_keyframe_keep_ratio))
    filtered_meta, filtered_imgs = qwen_keyframe_scorer.filter_top_k(
        valid_meta, valid_imgs_for_score, top_k
    )
    # 用筛选后的帧替换，后续 CLIP 编码和投影正常进行
```

**预期提升**：筛选后 L3 节点全为高信息密度关键帧（动作高峰、视觉变化剧烈处），Phase 3 的检索信噪比显著提升（预期 +10-15% 准确率）。

### 4.5 `VLMDescriptionGenerator`：VLM 描述生成

支持四种后端的统一接口：

| 后端 | 用途 | 实现方式 |
|------|------|---------|
| `stub` | 开发调试 | 直接返回占位文本，不发起任何网络请求 |
| `ollama` | 本地部署 | POST `/api/chat`，图像通过 `base64` 字段传输 |
| `openai` | 云端高质量 | POST `/v1/chat/completions`，图像通过 `image_url` 传输 |
| `qwen` | 阿里云百炼 | OpenAI 兼容接口，支持多图，图像先于文本传入 |

**指数退避重试**：

```python
for attempt in range(retries + 1):
    try:
        return self._call_backend(...)
    except Exception:
        time.sleep(2 ** attempt)  # 1s, 2s, 4s...
```

### 4.6 `sample_representative_frames`：均匀代表帧采样

```python
indices = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
```

使用**等比间隔采样**（包含首尾帧），确保无论段有多少帧，发给 VLM 的帧都能均匀覆盖整个时间段，避免集中在某一时刻。

---

## 5. `visual_projection.py`：跨模态特征对齐

### 5.1 为什么需要投影层？

CLIP 虽然在训练时强制使图像编码和文本编码在同一向量空间内对齐，但仍存在**模态内偏差（Modality Gap）**：

- 文本嵌入和图像嵌入在余弦相似度空间中各自聚集成不同的"簇"
- 直接用文本查询检索图像嵌入时，相似度分布可能偏向某些方向

投影层 $\text{Proj}(\cdot)$ 通过可学习参数，进一步校正这种偏差，使视觉特征更适合与文本查询进行跨模态匹配。

### 5.2 网络结构设计

**同维度情形**（`visual_dim == text_dim`，如 CLIP 全链路）：

```
输入 x [D]
  │
  └─► LayerNorm → fc1 → GELU → Dropout → fc2
                                              └──► 残差连接 → LayerNorm → 输出 [D]
       x ─────────────────────────────────────────►
```

- 残差连接：保证网络退化时（fc1, fc2 → 零矩阵时）输出等于输入，即恒等变换
- 恒等初始化（`_identity_init`）：将 fc1 和 fc2 初始化为单位矩阵，使未经训练时的投影层不改变原始 CLIP 特征，可以即插即用

**跨维度情形**（`visual_dim ≠ text_dim`，如 CLIP 视觉 + Ollama 文本）：

```
输入 x [visual_dim]
  └─► fc1 → LayerNorm → GELU → Dropout → fc2 → LayerNorm → 输出 [text_dim]
```

无残差连接（维度不同无法直接相加），使用 Xavier 均匀初始化，需要在对齐数据上进行微调。

### 5.3 L2 归一化的重要性

```python
if normalize:
    out = F.normalize(out, p=2, dim=-1)
```

投影后强制进行 L2 归一化，目的是让视觉嵌入与文本嵌入处于同一"单位超球面"上，使点积直接等价于余弦相似度：

$$\text{cos\_sim}(a, b) = a \cdot b \quad \text{（当 } \|a\|=\|b\|=1 \text{）}$$

这样 Tree-TRM 检索时的缩放点积就能正确反映语义相似度。

---

## 6. `video_tree_trm.py`：三阶段递归检索引擎（核心）

### 6.1 `RetrievalTrace`：可解释性设计

`RetrievalTrace` 记录了检索过程中的**所有中间状态**：

```python
@dataclass
class RetrievalTrace:
    k1_star, k2_star, k3_star: int      # 各阶段的选择结果
    l1_scores, l2_scores, l3_scores     # 各阶段的完整得分分布
    z0, z1, z2                          # 各阶段前后的潜在状态
    segment_summary, clip_caption       # 选中节点的文本内容
    target_frame, target_timestamp      # 最终定位的帧
```

通过保存得分分布（而不仅仅是 argmax 结果），可以分析模型的不确定性：
- 若最高得分与次高得分相差悬殊 → 模型高度确信
- 若最高得分与次高得分接近 → 存在多个候选位置，结果不稳定

### 6.2 `tree_trm_attention`：核心计算核

```python
def tree_trm_attention(query_state, memory_matrix, temperature=1.0):
    scale = math.sqrt(D) * temperature
    raw_scores = memory_matrix @ query_state / scale  # [N]
    
    # 数值稳定的 Softmax（减去最大值防止 exp 溢出）
    raw_scores = raw_scores - raw_scores.max()
    exp_scores = np.exp(raw_scores)
    scores = exp_scores / (exp_scores.sum() + 1e-9)
    
    k_star = int(np.argmax(scores))
    soft_retrieved = (scores[:, None] * memory_matrix).sum(axis=0)
    
    return scores, soft_retrieved, k_star
```

**为什么要除以 $\sqrt{D}$？**

随着维度 $D$ 增大，两个随机单位向量的点积的方差也会增大（约为 $D$），导致 Softmax 的输入分布越来越尖锐，梯度消失。除以 $\sqrt{D}$ 将点积的方差归一化到 $O(1)$，保持 Softmax 分布的平滑性。

**同时返回 `soft_retrieved`（软检索向量）的意义**：

虽然推理时使用 `argmax`（硬选择），但保留软检索向量是为了未来支持**可微分训练**：在反向传播中，软选择（加权求和）是可微的，而 argmax 不可微。

### 6.3 `update_latent_state`：三种状态更新策略

| 模式 | 公式 | 特性 |
|------|------|------|
| `additive`（默认） | $z_{h+1} = \text{norm}(z_h + M_{Lh}[k_h^*])$ | 保留历史语义，渐进累积上下文 |
| `replace` | $z_{h+1} = M_{Lh}[k_h^*]$ | 完全替换，只保留最新层信息 |
| `gated` | $z_{h+1} = 0.5 \cdot z_h + 0.5 \cdot M_{Lh}[k_h^*]$ | 折衷方案，固定门控 |

**`additive` 模式的数学直觉**：

初始状态 $z_0 = q$（查询嵌入），经过三次加性更新后：

$$z_2 = \text{norm}(\text{norm}(q + M_{L1}[k_1^*]) + M_{L2}[k_1^*, k_2^*])$$

每次更新都将选中节点的语义"注入"到状态中，使状态逐步从"我要找什么"演变为"我已找到这里，接下来要找更具体的"。

### 6.4 三阶段实现细节

**Phase 1 — 粗粒度路由（H=1，文本对文本）**：

```python
query_state = normalize(q + z0)  # q+z0 = 2q（当 z0=q 时）
scores_l1 = softmax((query_state · M_L1^T) / √D)
k1_star = argmax(scores_l1)
z1 = normalize(z0 + M_L1[k1_star])
```

当 `init_state_mode="query"` 时，$z_0 = q$，因此第一阶段的查询状态实际上是 $\text{norm}(2q)$，等价于 $q$（归一化后相同）。这意味着第一阶段是**纯粹基于原始查询语义**的检索。

**Phase 2 — 细粒度聚焦（H=2，文本对文本）**：

```python
query_state = normalize(q + z1)
# z1 = normalize(q + M_L1[k1_star])，已包含 L1 层的上下文
scores_l2 = softmax((query_state · M_L2_k1^T) / √D)
k2_star = argmax(scores_l2)
z2 = normalize(z1 + M_L2_k1[k2_star])
```

此时的查询状态 $q + z_1$ 融合了原始查询和 L1 层选中节点的语义，相当于"在语义上已经知道是哪个宏观事件段，现在在该范围内寻找更具体的片段"。

**Phase 3 — 视觉定位（H=3，文本对图像）**：

```python
query_state = normalize(q + z2)
# z2 包含 q、L1摘要语义、L2描述语义的叠加
scores_l3 = softmax((query_state · M_L3_k1_k2^T) / √D)
k3_star = argmax(scores_l3)
```

关键在于：$M_{L3}^{(k_1^*, k_2^*)}$ 是视觉嵌入矩阵（经过 `Proj(CLIP(·))` 投影），而 `query_state` 是文本空间的向量。由于投影层已将视觉特征对齐到文本潜空间，这个点积在语义上是有意义的跨模态相似度计算。

---

## 7. `answer_generator.py`：答案生成模块

### 7.1 高信噪比（SNR）的体现

Video-Tree-TRM 的三阶段检索将搜索范围从"整个视频"收窄到"单帧图像"，实现了高信噪比（High SNR）：

- **低信噪比**（直接送 100+ 帧给 VLM）：VLM 需要自己理解哪一帧最相关，容易被无关内容干扰
- **高信噪比**（送单张精确关键帧给 VLM）：VLM 只需聚焦于一帧中的具体视觉细节

```python
# 多模态模式（OpenAI/Ollama）：将关键帧图像作为视觉证据直接传入 VLM
content = [
    {"type": "text", "text": prompt},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
]
```

### 7.2 三种后端的适用场景

```python
if self.backend == "deepseek":
    return self._call_deepseek(prompt)  # 纯文本，VLM描述作为视觉代理
elif self.backend == "openai":
    return self._call_openai(prompt, frame_b64)  # 多模态，真实视觉输入
elif self.backend == "ollama":
    return self._call_ollama(prompt, frame_b64)  # 多模态，本地部署
```

**DeepSeek 纯文本模式**的工作原理：通过精心构造的提示词，将 L1 摘要 + L2 描述 + 时间戳信息作为"文字版视觉证据"传入 LLM，依赖预处理时 VLM 生成的描述来间接回答视觉问题。这是对 API 成本与答案质量的一种折衷。

---

## 8. `pipeline.py`：端到端推理管线

### 8.1 延迟初始化（Lazy Initialization）

```python
def _get_clip_extractor(self) -> CLIPFeatureExtractor:
    if self._clip_extractor is None:
        self._clip_extractor = CLIPFeatureExtractor(...)
    return self._clip_extractor
```

所有重量级组件（CLIP 模型、VLM 客户端、投影层）都采用延迟初始化：
- `build_pyramid` 模式：加载 CLIP + VLM
- `run_from_pyramid` 模式：只加载 CLIP（不需要 VLM）
- 仅文本嵌入模式：不加载 CLIP 图像编码器

### 8.2 两种使用模式

**模式 1：完整流程**

```
run(video_path, query)
  └─► build_pyramid(video_path)  # 离线预处理（可缓存）
        └─► retrieve(query)      # 三阶段检索
              └─► generate(trace)  # 答案生成
```

**模式 2：仅推理（预处理已完成）**

```
run_from_pyramid(pyramid_dir, query)
  └─► load_pyramid(pyramid_dir)  # 从磁盘加载（毫秒级）
        └─► retrieve(query)
              └─► generate(trace)
```

模式 2 适合**生产环境**：预处理一次，多次查询共享同一金字塔，检索延迟极低（< 100ms）。

### 8.3 Qwen 智能增强分支（v1.1 新增）

`build_pyramid()` 在原有流程基础上新增两条可选分支，由 `PyramidConfig` 中的开关字段控制：

**分支 1：Qwen-VL 场景感知 L1 切分**

```python
# pipeline.py 中的切分逻辑
if qwen_scene_detector is not None:
    # Step 1: 全局探测帧采样（0.5fps）
    probe_meta = frame_extractor.extract(video, fps=probe_fps, ...)
    # Step 2: Qwen-VL 批量场景检测
    boundaries = qwen_scene_detector.detect_boundaries(probe_images_ts)
    # Step 3: 基于边界的语义感知切分
    segments_info = segment_video_smart(duration, boundaries, ...)
else:
    segments_info = segment_video(duration, ...)  # 原始固定步长
```

**分支 2：Qwen-VL L3 关键帧筛选**

```python
# 在 L3 帧提取后、CLIP 编码前插入
if qwen_keyframe_scorer is not None and len(l3_frames_meta) > 1:
    top_k = max(1, round(n * pyr_cfg.qwen_keyframe_keep_ratio))
    filtered_meta, filtered_imgs = qwen_keyframe_scorer.filter_top_k(
        valid_meta, valid_imgs, top_k
    )
    # filtered_imgs 直接送入 CLIPFeatureExtractor
```

两个 `QwenSceneDetector` / `QwenKeyframeScorer` 实例均在 `qwen_api_key` 为空时设为 `None`，此时流程自动退化为原始逻辑，**不影响现有功能**。

### 8.4 缓存检查逻辑

```python
if not force_rebuild and HierarchicalSemanticPyramid.exists(save_dir):
    logger.info("检测到已有金字塔缓存，直接加载。")
    pyramid = HierarchicalSemanticPyramid.load(save_dir)
    return pyramid
```

构建金字塔（尤其是 VLM 调用）非常耗时，缓存检查避免了重复处理同一视频。`force_rebuild` 标志提供了强制重建的逃生出口。

---

## 9. 关键算法详解：Tree-TRM 注意力机制

### 9.1 与标准 Transformer 注意力的关系

Video-Tree-TRM 的注意力机制是标准 Transformer 多头注意力的**简化单头版本**：

| | Transformer Self-Attention | Tree-TRM Attention |
|---|---|---|
| 查询 (Q) | $X W_Q$ | $q + z_h$（预先加法融合） |
| 键 (K) | $X W_K$ | $M_{Lh}$（已归一化的记忆矩阵） |
| 值 (V) | $X W_V$ | $M_{Lh}$（键值合一，仅用于软检索） |
| 输出 | $\text{softmax}(QK^T/\sqrt{d})V$ | $\text{argmax}(\text{softmax}(QK^T/\sqrt{d}))$ |

关键差异：
1. 推理时使用 **argmax 硬选择**而非加权求和，实现精确的"跳转"式检索
2. 查询是**查询嵌入与潜在状态的加法融合**，而非矩阵投影
3. 记忆矩阵在不同层是**不同模态**的（文本 vs 视觉）

### 9.2 缩放因子的作用

$$\text{score}(q, k_i) = \frac{(q + z_h) \cdot k_i}{\sqrt{D} \cdot \tau}$$

- $\sqrt{D}$：防止高维空间中点积过大导致 Softmax 饱和
- $\tau$（温度参数）：控制选择的"确定性"
  - $\tau < 1$（如 0.5）：Softmax 更尖锐，模型更倾向于选择最高分的节点（更确定）
  - $\tau > 1$（如 2.0）：Softmax 更平滑，多个候选节点得分接近（更模糊）

---

## 10. 模态切换机制解析

Video-Tree-TRM 的核心创新之一是**无缝的模态切换**（Modality Switching）。

### 10.1 统一向量空间的构建

预处理阶段通过以下方式确保所有节点嵌入处于同一向量空间：

```
文本查询 ──► CLIP text encoder ──► 向量 [D]
                                        ↑ 同一空间
L1 摘要  ──► CLIP text encoder ──► 向量 [D]
L2 描述  ──► CLIP text encoder ──► 向量 [D]
                                        ↑ 投影对齐
L3 帧    ──► CLIP image encoder ──► VisualProjectionLayer ──► 向量 [D]
```

CLIP 的训练目标（对比学习：图文对的点积最大化）天然保证了文本编码器和图像编码器的输出在同一向量空间内，投影层进一步精化这种对齐。

### 10.2 切换点的语义意义

Phase 2 → Phase 3 的切换点是一个**认知层面的转变**：

- Phase 1 & 2：检索范围由"宏观叙事"到"具体片段"，是**语义空间的导航**
- Phase 3：不再追问"这是什么场景"，而是追问"哪一帧包含我要找的视觉证据"，是**视觉空间的验证**

`z_2` 在此时携带了"我已知道是哪个时间段、哪个片段，且该片段描述了什么"的语义预期，这一预期与视觉帧直接进行相似度匹配，完成跨模态"锚定"。

---

## 11. 潜在状态演化分析

### 11.1 状态向量的语义演化

以查询"What is written on the door poster?"为例，追踪 $z$ 的语义演化：

| 阶段 | 状态向量 | 语义含义 |
|------|---------|---------|
| $z_0 = q$ | 查询嵌入 | "门上海报写的什么" |
| $z_1 = \text{norm}(q + M_{L1}[k_1^*])$ | 查询 + L1摘要 | "门上海报写的什么"（在已知是"学校场景"这一宏观背景下） |
| $z_2 = \text{norm}(z_1 + M_{L2}[k_1^*, k_2^*])$ | 前述 + L2描述 | 在已知"争吵并摔门"这一具体动作上下文下，期待看到"门"和"海报"的视觉证据 |

$z_2$ 积累了从粗到细的语义上下文，它与 L3 视觉嵌入的相似度匹配能够更精确地定位包含门和海报的帧。

### 11.2 归一化的必要性

每次加法更新后进行 L2 归一化（`normalize=True`）：

```python
new_z = z + selected_embed
norm = np.linalg.norm(new_z)
new_z = new_z / norm
```

不归一化的问题：随着加性累积，向量模长会不断增大，导致下一阶段的 query_state 的各维度数值过大，影响 Softmax 计算的数值稳定性。归一化将所有状态向量保持在单位超球面上，确保每阶段的检索得分可比较。

---

## 12. 设计决策总结

### 12.1 为何选择三层结构而不是两层或四层？

| | 两层 | 三层（当前） | 四层 |
|---|---|---|---|
| 粒度覆盖 | 粗 → 帧（跨度大） | 粗 → 中 → 帧（合理过渡） | 过于细碎 |
| VLM 调用次数 | 较少 | 适中 | 极多 |
| 检索精度 | 中等 | 高 | 可能过拟合 |

三层对应人类认知的自然层次：事件段 → 行为片段 → 视觉瞬间，且与视频时间层次（分钟级、秒级、帧级）天然对齐。

### 12.2 为何 L3 使用视觉嵌入而非继续使用文本？

L3 层保留视觉嵌入（而不是对每帧用 VLM 生成描述再文本嵌入）的原因：

1. **细节保真**：帧级的像素信息（海报文字、人脸表情、物体细节）在文本化时会大量丢失
2. **计算效率**：CLIP 图像编码仅需约 5ms/帧，而 VLM 描述生成需要数十秒/帧
3. **可扩展性**：1fps 的 2 小时视频有 7200 帧，7200 次 VLM 调用不可接受

### 12.3 `(q + z_h)` vs 拼接 `[q; z_h]`

选择加法而非拼接的原因：
- 保持向量维度 $D$ 不变，与记忆矩阵维度匹配（可直接计算点积）
- 在归一化后，加法等价于在超球面上"插值"：状态向量始终位于查询语义和选中节点语义之间
- 实现简单，无需额外投影层

### 12.4 argmax vs Top-K vs Beam Search

当前实现使用 **argmax（贪心）**：

- 优点：实现简单，确定性强，延迟最低
- 缺点：可能陷入次优路径（若第一层选错了，后续所有层都在错误分支上检索）

未来拓展方向（已在 PLAN.md 的 V1.1 规划中）：
- **Top-K 多路径**：每层保留 $K$ 个候选，最终取所有路径中得分最高的叶节点
- **Beam Search**：以 Beam Width 控制搜索宽度，权衡精度与计算量

---

*文档版本：v1.1.0 | 最后更新：2026-03-04 | 新增：QwenSceneDetector、QwenKeyframeScorer、segment_video_smart、PyramidConfig Qwen 字段*

