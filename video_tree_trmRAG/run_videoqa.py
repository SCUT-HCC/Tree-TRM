#!/usr/bin/env python3
"""
run_videoqa.py — Video-Tree-TRM 端到端视频问答推理脚本
======================================================
该脚本是 Video-Tree-TRM 系统的**推理入口**，支持两种使用模式：

  模式 A：完整流程（预处理 + 检索 + 答案生成）
    python run_videoqa.py \\
      --video movie.mp4 \\
      --query "What is written on the door poster?" \\
      --vlm_backend ollama --answer_backend ollama

  模式 B：仅推理（已有金字塔缓存，跳过预处理）
    python run_videoqa.py \\
      --pyramid_dir pyramid_cache/movie \\
      --query "What is written on the door poster?" \\
      --answer_backend deepseek --deepseek_api_key sk-xxx

  批量问答（对同一视频提出多个问题）：
    python run_videoqa.py \\
      --pyramid_dir pyramid_cache/movie \\
      --query_file questions.txt \\
      --answer_backend openai --openai_api_key sk-xxx \\
      --output_json results.json

  从 YAML 配置文件运行：
    python run_videoqa.py \\
      --config config/video_tree_trm.yaml \\
      --video movie.mp4 \\
      --query "Who appears at timestamp 42 minutes?"
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# 确保可从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_tree_trmRAG.config import VideoTreeTRMConfig
from video_tree_trmRAG.pipeline import VideoQAPipeline, VideoQAResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-Tree-TRM：长视频问答推理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 输入模式（二选一） ──────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--video",
        default=None,
        help="原始视频文件路径（模式 A：完整流程）。",
    )
    input_group.add_argument(
        "--pyramid_dir",
        default=None,
        help="已构建的金字塔缓存目录（模式 B：仅推理）。",
    )

    # ── 问题输入（二选一） ─────────────────────────────────────────────
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        default=None,
        help="单个问题文本。",
    )
    query_group.add_argument(
        "--query_file",
        default=None,
        help="批量问题文件路径（每行一个问题，UTF-8 编码）。",
    )

    # ── YAML 配置文件（可选，覆盖其他参数） ────────────────────────────
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件路径（若提供，将覆盖同名命令行参数）。",
    )

    # ── 金字塔构建参数（模式 A 生效） ──────────────────────────────────
    parser.add_argument(
        "--pyramid_save_dir",
        default=None,
        help="金字塔保存目录（模式 A 用，None 则自动生成）。",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="强制重新构建金字塔（忽略缓存）。",
    )
    parser.add_argument("--l1_duration", type=float, default=600.0,
                        help="L1 段时长（秒）。")
    parser.add_argument("--l2_duration", type=float, default=20.0,
                        help="L2 片段时长（秒）。")
    parser.add_argument("--l3_fps", type=float, default=1.0,
                        help="L3 帧率（fps）。")

    # ── VLM 后端（预处理阶段）──────────────────────────────────────────
    parser.add_argument(
        "--vlm_backend",
        default="stub",
        choices=["ollama", "openai", "qwen", "stub"],
        help="VLM 描述生成后端（用于构建金字塔）。stub 模式不调用真实 VLM。qwen 为阿里云百炼，无需本地部署。",
    )
    parser.add_argument("--vlm_model", default="llava",
                        help="VLM 模型名称（Ollama：llava；OpenAI：gpt-4o；Qwen 无需此参数，用 --qwen_vlm_model）。")
    parser.add_argument("--ollama_url",
                        default="http://localhost:11434/api/chat",
                        help="Ollama Chat API 地址。")
    parser.add_argument("--openai_api_key", default="",
                        help="OpenAI API Key（vlm_backend=openai 时使用）。")
    parser.add_argument("--qwen_api_key", default="",
                        help="阿里云百炼 API Key（vlm_backend=qwen 或 answer_backend=qwen 时使用）。")
    parser.add_argument("--qwen_vlm_model", default="qwen-vl-plus",
                        help="千问视觉模型（用于构建金字塔）：qwen-vl-plus / qwen-vl-max。")
    parser.add_argument("--qwen_answer_model", default="qwen-vl-plus",
                        help="千问答案生成模型：qwen-vl-plus / qwen-vl-max / qwen-plus / qwen-turbo。")

    # ── 嵌入配置 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--text_backend",
        default="clip",
        choices=["clip", "ollama"],
        help="文本嵌入后端。",
    )
    parser.add_argument("--clip_model", default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP 模型变体。")
    parser.add_argument("--clip_device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="CLIP 推理设备。")
    parser.add_argument("--projection_checkpoint", default=None,
                        help="视觉投影层预训练权重路径（None 使用恒等初始化）。")

    # ── 答案生成后端 ───────────────────────────────────────────────────
    parser.add_argument(
        "--answer_backend",
        default="ollama",
        choices=["deepseek", "openai", "ollama", "qwen", "stub"],
        help="答案生成后端。qwen 为阿里云百炼（多模态，推荐）；stub 为测试占位模式。",
    )
    parser.add_argument("--deepseek_api_key", default="",
                        help="DeepSeek API Key。")
    parser.add_argument("--deepseek_model", default="deepseek-chat",
                        help="DeepSeek 模型名称。")
    parser.add_argument("--answer_openai_key", default="",
                        help="用于答案生成的 OpenAI API Key。")
    parser.add_argument("--answer_openai_model", default="gpt-4o",
                        help="用于答案生成的 OpenAI 模型名称。")
    parser.add_argument("--answer_ollama_model", default="llava",
                        help="用于答案生成的 Ollama 模型名称（需支持视觉）。")

    # ── 检索配置 ──────────────────────────────────────────────────────
    parser.add_argument("--softmax_temperature", type=float, default=1.0,
                        help="Softmax 温度系数（τ）。")
    parser.add_argument("--state_update_mode", default="additive",
                        choices=["additive", "replace", "gated"],
                        help="潜在状态更新策略。")

    # ── 输出 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_json",
        default=None,
        help="将结果保存为 JSON 文件路径（None 则仅打印到终端）。",
    )
    parser.add_argument(
        "--show_trace",
        action="store_true",
        help="打印详细检索轨迹（L1/L2/L3 分数分布）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细运行日志。",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config Construction
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> VideoTreeTRMConfig:
    """根据命令行参数（或 YAML 文件）构建系统配置。"""
    if args.config and os.path.isfile(args.config):
        logger.info(f"从 YAML 配置文件加载：{args.config}")
        cfg = VideoTreeTRMConfig.from_yaml(args.config)
        # 命令行参数可覆盖部分配置
        if args.answer_backend:
            cfg.answer.backend = args.answer_backend
        if args.deepseek_api_key:
            cfg.answer.deepseek_api_key = args.deepseek_api_key
        if args.answer_openai_key:
            cfg.answer.openai_api_key = args.answer_openai_key
        return cfg

    return VideoTreeTRMConfig.from_dict(
        {
            "pyramid": {
                "l1_segment_duration": args.l1_duration,
                "l2_clip_duration": args.l2_duration,
                "l3_fps": args.l3_fps,
            },
            "embedding": {
                "clip_model": args.clip_model,
                "clip_device": args.clip_device,
                "text_backend": args.text_backend,
                "projection_checkpoint": args.projection_checkpoint,
            },
            "vlm": {
                "backend": args.vlm_backend,
                "ollama_chat_url": args.ollama_url,
                "ollama_vlm_model": args.vlm_model,
                "openai_api_key": args.openai_api_key,
                "openai_vlm_model": args.vlm_model if args.vlm_backend == "openai" else "gpt-4o",
                "qwen_api_key": args.qwen_api_key,
                "qwen_vlm_model": args.qwen_vlm_model,
            },
            "answer": {
                "backend": args.answer_backend,
                "deepseek_api_key": args.deepseek_api_key,
                "deepseek_model": args.deepseek_model,
                "openai_api_key": args.answer_openai_key,
                "openai_model": args.answer_openai_model,
                "ollama_model": args.answer_ollama_model,
                "qwen_api_key": args.qwen_api_key,
                "qwen_answer_model": args.qwen_answer_model,
            },
            "retrieval": {
                "softmax_temperature": args.softmax_temperature,
                "state_update_mode": args.state_update_mode,
            },
            "verbose": args.verbose,
        }
    )


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------

def print_result(result: VideoQAResult, show_trace: bool = False) -> None:
    """格式化打印单个 VideoQA 结果。"""
    print("\n" + "=" * 70)
    print(f"📹  视频    : {result.video_name}")
    print(f"❓  问题    : {result.query}")
    print(f"✅  答案    : {result.answer}")
    print("-" * 70)

    if result.trace and result.trace.is_valid():
        t = result.trace
        print(f"🔍  检索轨迹:")
        print(f"   Phase 1 (粗粒度路由)  → L1[{t.k1_star}] : {t.segment_summary[:60]}...")
        print(f"   Phase 2 (细粒度聚焦)  → L2[{t.k2_star}] : {t.clip_caption[:60]}...")
        print(f"   Phase 3 (视觉定位)    → L3[{t.k3_star}] : "
              f"时间戳 {t.target_timestamp:.1f}s")
        if t.target_frame:
            print(f"   目标帧路径  : {t.target_frame.frame_path}")

        if show_trace:
            print("\n   L1 分数分布:", end=" ")
            if t.l1_scores is not None:
                scores_str = ", ".join(
                    [f"[{i}]{s:.3f}" for i, s in enumerate(t.l1_scores)]
                )
                print(scores_str[:120])
            print("   L2 分数分布:", end=" ")
            if t.l2_scores is not None:
                scores_str = ", ".join(
                    [f"[{i}]{s:.3f}" for i, s in enumerate(t.l2_scores)]
                )
                print(scores_str[:120])
    else:
        print("⚠️  检索未完整完成。")

    print(f"⏱️  耗时     : {result.elapsed_sec:.2f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 初始化日志
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 参数校验
    if args.video is None and args.pyramid_dir is None:
        logger.error("❌ 必须指定 --video（完整流程）或 --pyramid_dir（仅推理）之一。")
        sys.exit(1)

    # 加载问题
    if args.query:
        queries = [args.query]
    else:
        query_path = Path(args.query_file)
        if not query_path.exists():
            logger.error(f"❌ 问题文件不存在：{args.query_file}")
            sys.exit(1)
        queries = [
            line.strip()
            for line in query_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info(f"从文件加载了 {len(queries)} 个问题：{args.query_file}")

    # 构建配置
    config = build_config(args)

    # 创建 Pipeline
    pipeline = VideoQAPipeline(config)

    # ── 执行 VideoQA ──────────────────────────────────────────────────────
    results = []
    t_total = time.time()

    if args.video:
        # 模式 A：完整流程
        logger.info(f"📽️  模式 A：完整流程（含预处理），视频={args.video}")

        if len(queries) == 1:
            result = pipeline.run(
                video_path=args.video,
                query=queries[0],
                pyramid_dir=args.pyramid_save_dir,
                force_rebuild=args.force_rebuild,
            )
            results.append(result)
            print_result(result, show_trace=args.show_trace)
        else:
            # 多问题：先构建金字塔，再批量问答
            pyramid = pipeline.build_pyramid(
                video_path=args.video,
                save_dir=args.pyramid_save_dir,
                force_rebuild=args.force_rebuild,
            )
            # 利用已构建的金字塔批量推理
            pyr_dir = args.pyramid_save_dir or os.path.join(
                config.pyramid.cache_dir,
                os.path.splitext(os.path.basename(args.video))[0],
            )
            results = pipeline.run_batch(pyramid_dir=pyr_dir, queries=queries)
            for r in results:
                print_result(r, show_trace=args.show_trace)

    else:
        # 模式 B：仅推理
        logger.info(f"⚡ 模式 B：仅推理（加载已有金字塔），目录={args.pyramid_dir}")

        if len(queries) == 1:
            result = pipeline.run_from_pyramid(
                pyramid_dir=args.pyramid_dir,
                query=queries[0],
            )
            results.append(result)
            print_result(result, show_trace=args.show_trace)
        else:
            results = pipeline.run_batch(
                pyramid_dir=args.pyramid_dir,
                queries=queries,
            )
            for r in results:
                print_result(r, show_trace=args.show_trace)

    total_elapsed = time.time() - t_total
    logger.info(
        f"\n✅ 全部完成！共处理 {len(results)} 个问题，总耗时 {total_elapsed:.1f}s"
        f"（平均 {total_elapsed/max(len(results),1):.1f}s/问题）"
    )

    # ── 保存结果 ──────────────────────────────────────────────────────────
    if args.output_json:
        output_data = [r.to_dict() for r in results]
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存至：{args.output_json}")
        print(f"\n📄 结果已保存至：{args.output_json}")

    # ── 统计摘要 ──────────────────────────────────────────────────────────
    n_success = sum(1 for r in results if r.success)
    print(f"\n📊 成功率：{n_success}/{len(results)} ({100*n_success/max(len(results),1):.1f}%)")


if __name__ == "__main__":
    main()

