#!/usr/bin/env python3
"""
build_pyramid.py — 分层语义金字塔（HSP）离线构建脚本
======================================================
该脚本对应 Video-Tree-TRM 工作流的**预处理阶段**（离线）。

功能：
  - 从原始视频文件提取帧（L3），切分时间段（L1/L2）
  - 调用 VLM（Ollama LLaVA / OpenAI GPT-4o）为每个 L1/L2 段生成文字描述
  - 使用 CLIP（或 Ollama nomic-embed-text）编码文本嵌入（L1/L2）
  - 使用 CLIP 图像编码器 + 可学习投影层编码视觉嵌入（L3）
  - 将完整金字塔序列化保存到磁盘，供推理时快速加载

用法示例：
  # 最简用法（使用 Ollama LLaVA + stub 测试）
  python build_pyramid.py --video movie.mp4 --vlm_backend stub

  # 使用 Ollama LLaVA 生成真实描述
  python build_pyramid.py \\
    --video movie.mp4 \\
    --vlm_backend ollama \\
    --vlm_model llava \\
    --output_dir pyramid_cache/movie

  # 使用 OpenAI GPT-4o 生成高质量描述
  python build_pyramid.py \\
    --video movie.mp4 \\
    --vlm_backend openai \\
    --openai_api_key sk-xxx \\
    --l1_duration 600 \\
    --l2_duration 20 \\
    --l3_fps 1.0
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# 确保可以从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_tree_trmRAG.config import (
    EmbeddingConfig,
    PyramidConfig,
    VideoTreeTRMConfig,
    VLMConfig,
)
from video_tree_trmRAG.pipeline import VideoQAPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-Tree-TRM：分层语义金字塔（HSP）离线构建工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 必需参数 ───────────────────────────────────────────────────────
    parser.add_argument(
        "--video",
        required=True,
        help="输入视频文件路径（支持 mp4/mkv/avi 等 OpenCV 支持的格式）",
    )

    # ── 输出 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        default=None,
        help="金字塔保存目录。默认：pyramid_cache/<视频名>",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="强制重建金字塔（即使缓存已存在）",
    )

    # ── 金字塔结构 ────────────────────────────────────────────────────
    parser.add_argument(
        "--l1_duration",
        type=float,
        default=600.0,
        help="L1 宏观事件段时长（秒）。例如 600 = 10 分钟",
    )
    parser.add_argument(
        "--l2_duration",
        type=float,
        default=20.0,
        help="L2 短片段时长（秒）。例如 20 = 20 秒",
    )
    parser.add_argument(
        "--l3_fps",
        type=float,
        default=1.0,
        help="L3 关键帧提取帧率（fps）",
    )
    parser.add_argument(
        "--l1_max_frames_vlm",
        type=int,
        default=6,
        help="发送给 VLM 生成 L1 摘要的最大代表帧数",
    )
    parser.add_argument(
        "--l2_max_frames_vlm",
        type=int,
        default=4,
        help="发送给 VLM 生成 L2 描述的最大代表帧数",
    )
    parser.add_argument(
        "--l3_max_frames",
        type=int,
        default=30,
        help="每个 L2 片段内最大 L3 帧数",
    )
    parser.add_argument(
        "--max_frame_size",
        type=int,
        default=336,
        help="帧图像的最大长边（像素）",
    )

    # ── VLM 配置 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--vlm_backend",
        default="stub",
        choices=["ollama", "openai", "qwen", "stub"],
        help=(
            "VLM 后端选择：\n"
            "  stub   - 不调用真实 VLM，用于快速调试（不生成真实描述）\n"
            "  ollama - 使用本地 Ollama 服务（需先运行 ollama serve）\n"
            "  openai - 使用 OpenAI GPT-4o API（需要 API Key）\n"
            "  qwen   - 使用阿里云百炼千问视觉大模型（无需本地部署，推荐！）"
        ),
    )
    parser.add_argument(
        "--ollama_url",
        default="http://localhost:11434/api/chat",
        help="Ollama Chat API 地址",
    )
    parser.add_argument(
        "--vlm_model",
        default="llava",
        help="VLM 模型名称（Ollama：llava/bakllava；OpenAI：gpt-4o；Qwen：qwen-vl-plus/qwen-vl-max）",
    )
    parser.add_argument(
        "--openai_api_key",
        default="",
        help="OpenAI API Key（当 vlm_backend=openai 时必填）",
    )
    parser.add_argument(
        "--qwen_api_key",
        default="",
        help="阿里云百炼 API Key（当 vlm_backend=qwen 时必填）。在 https://bailian.console.aliyun.com/ 获取",
    )
    parser.add_argument(
        "--qwen_vlm_model",
        default="qwen-vl-plus",
        help="千问视觉模型（qwen-vl-plus / qwen-vl-max / qwen-vl-ocr）",
    )

    # ── 嵌入配置 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--text_backend",
        default="clip",
        choices=["clip", "ollama"],
        help=(
            "文本嵌入后端：\n"
            "  clip   - 使用 CLIP text encoder（与 L3 视觉嵌入共享联合空间，推荐）\n"
            "  ollama - 使用 Ollama nomic-embed-text（768-dim，文本理解更强）"
        ),
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP 模型变体",
    )
    parser.add_argument(
        "--clip_device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="CLIP 推理设备",
    )
    parser.add_argument(
        "--projection_checkpoint",
        default=None,
        help="投影层预训练权重路径（.pt），None 则使用恒等初始化",
    )

    # ── 其他 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细构建日志",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 检查视频文件
    if not os.path.isfile(args.video):
        logger.error(f"视频文件不存在：{args.video}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = args.output_dir or os.path.join("pyramid_cache", video_name)

    logger.info("=" * 60)
    logger.info("Video-Tree-TRM | 分层语义金字塔（HSP）构建")
    logger.info("=" * 60)
    logger.info(f"视频路径     : {args.video}")
    logger.info(f"输出目录     : {output_dir}")
    logger.info(f"L1 段时长    : {args.l1_duration}s")
    logger.info(f"L2 片段时长  : {args.l2_duration}s")
    logger.info(f"L3 帧率      : {args.l3_fps} fps")
    logger.info(f"VLM 后端     : {args.vlm_backend}")
    logger.info(f"文本嵌入     : {args.text_backend}")
    logger.info(f"CLIP 模型    : {args.clip_model}")
    logger.info("=" * 60)

    if args.vlm_backend == "stub":
        logger.warning(
            "⚠️  当前使用 stub 模式！VLM 不会生成真实描述，仅用于调试。"
            "实际部署时请改用 qwen/ollama/openai 后端。"
        )
    elif args.vlm_backend == "qwen" and not args.qwen_api_key:
        logger.error("❌ 使用 qwen 后端时必须提供 --qwen_api_key。")
        sys.exit(1)

    # 构建配置
    config = VideoTreeTRMConfig.from_dict(
        {
            "pyramid": {
                "l1_segment_duration": args.l1_duration,
                "l2_clip_duration": args.l2_duration,
                "l3_fps": args.l3_fps,
                "l1_max_frames_for_vlm": args.l1_max_frames_vlm,
                "l2_max_frames_for_vlm": args.l2_max_frames_vlm,
                "l3_max_frames_per_clip": args.l3_max_frames,
                "max_frame_long_side": args.max_frame_size,
                "cache_dir": "pyramid_cache",
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
            "verbose": args.verbose,
        }
    )

    # 运行构建
    t0 = time.time()
    pipeline = VideoQAPipeline(config)

    try:
        pyramid = pipeline.build_pyramid(
            video_path=args.video,
            save_dir=output_dir,
            force_rebuild=args.force_rebuild,
        )
        elapsed = time.time() - t0
        stats = pyramid.stats()

        logger.info("")
        logger.info("✅ 金字塔构建成功！")
        logger.info(f"   耗时    : {elapsed:.1f}s")
        logger.info(f"   L1 节点 : {stats['n_segments']}")
        logger.info(f"   L2 节点 : {stats['n_clips']}")
        logger.info(f"   L3 节点 : {stats['n_frames']}")
        logger.info(f"   保存于  : {output_dir}")

        # 保存构建摘要
        summary = {
            "video": args.video,
            "output_dir": output_dir,
            "elapsed_sec": round(elapsed, 2),
            "stats": stats,
            "config": {
                "l1_duration": args.l1_duration,
                "l2_duration": args.l2_duration,
                "l3_fps": args.l3_fps,
                "vlm_backend": args.vlm_backend,
                "text_backend": args.text_backend,
                "clip_model": args.clip_model,
            },
        }
        summary_path = os.path.join(output_dir, "build_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"   构建摘要: {summary_path}")

    except FileNotFoundError as e:
        logger.error(f"❌ 文件错误：{e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"❌ 依赖缺失：{e}")
        logger.error("请先安装依赖：pip install -r video_tree_trmRAG/requirements_video.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 构建失败：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

