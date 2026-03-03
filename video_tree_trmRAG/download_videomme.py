#!/usr/bin/env python3
"""
Video-MME 数据集下载 & Video-Tree-TRM 评测脚本
================================================

功能：
  1. 通过 hf-mirror.com 镜像下载 Video-MME 元数据（问题/答案/YouTube链接）
  2. 用 yt-dlp 下载对应 YouTube 视频
  3. 调用 Video-Tree-TRM Pipeline 构建金字塔 + 运行问答推理
  4. 将预测答案与标准答案对比，输出准确率

用法示例：
  # 下载并测试 5 个长视频（推荐首次运行）
  python download_videomme.py \
    --qwen_api_key "sk-xxx" \
    --duration long \
    --num_videos 5 \
    --output_dir /tmp/videomme

  # 只下载视频（不运行推理）
  python download_videomme.py \
    --download_only \
    --duration long \
    --num_videos 10 \
    --output_dir /tmp/videomme

  # 只运行推理（视频已提前下载好）
  python download_videomme.py \
    --inference_only \
    --qwen_api_key "sk-xxx" \
    --output_dir /tmp/videomme
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ── 设置 HuggingFace 国内镜像（必须在 import datasets 前设置）──────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: 下载 Video-MME 元数据
# ---------------------------------------------------------------------------

def download_metadata(duration: str, num_videos: int, output_dir: str) -> list:
    """从 HuggingFace（镜像）下载 Video-MME 问答元数据。

    Args:
        duration:   视频时长筛选，"short" | "medium" | "long" | "all"
        num_videos: 最多下载的视频数量
        output_dir: 保存目录

    Returns:
        样本列表，每项包含 videoID / url / question / options / answer
    """
    logger.info("=" * 60)
    logger.info("Step 1: 下载 Video-MME 元数据")
    logger.info(f"  镜像地址 : https://hf-mirror.com")
    logger.info(f"  时长筛选 : {duration}")
    logger.info(f"  视频数量 : {num_videos}")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请先安装 datasets：pip install datasets")
        sys.exit(1)

    logger.info("正在加载 Video-MME 元数据（约几秒）...")
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    logger.info(f"✅ 元数据加载完成，共 {len(ds)} 条问答")

    # 按时长筛选
    if duration != "all":
        ds = ds.filter(lambda x: x["duration"] == duration)
        logger.info(f"筛选 duration={duration} 后剩余 {len(ds)} 条问答")

    # 按视频去重，每视频取全部3道题
    video_qa: dict = {}
    for item in ds:
        vid = item["videoID"]
        if vid not in video_qa:
            video_qa[vid] = {
                "videoID": vid,
                "url": item["url"],
                "duration": item["duration"],
                "domain": item["domain"],
                "sub_category": item["sub_category"],
                "questions": [],
            }
        video_qa[vid]["questions"].append({
            "question_id": item["question_id"],
            "task_type":   item["task_type"],
            "question":    item["question"],
            "options":     item["options"],
            "answer":      item["answer"],
        })

    samples = list(video_qa.values())[:num_videos]
    logger.info(f"选取 {len(samples)} 个视频（共 {sum(len(s['questions']) for s in samples)} 道题）")

    # 保存到本地
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"元数据已保存：{meta_path}")

    return samples


# ---------------------------------------------------------------------------
# Step 2: 下载视频
# ---------------------------------------------------------------------------

def download_videos(samples: list, video_dir: str) -> dict:
    """用 yt-dlp 批量下载 YouTube 视频。

    Args:
        samples:   元数据列表
        video_dir: 视频保存目录

    Returns:
        {videoID: 本地路径} 字典（下载失败的不包含）
    """
    logger.info("=" * 60)
    logger.info("Step 2: 下载 YouTube 视频")
    logger.info(f"  保存目录 : {video_dir}")
    logger.info("=" * 60)

    os.makedirs(video_dir, exist_ok=True)

    # 检查 yt-dlp
    result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("未找到 yt-dlp，请先安装：pip install yt-dlp")
        sys.exit(1)
    logger.info(f"yt-dlp 版本：{result.stdout.strip()}")

    video_paths: dict = {}
    for i, sample in enumerate(samples):
        vid  = sample["videoID"]
        url  = sample["url"]
        out  = os.path.join(video_dir, f"{vid}.mp4")

        if os.path.isfile(out) and os.path.getsize(out) > 100_000:
            logger.info(f"[{i+1}/{len(samples)}] 跳过（已存在）: {vid}")
            video_paths[vid] = out
            continue

        logger.info(f"[{i+1}/{len(samples)}] 下载: {vid} | {url}")
        cmd = [
            "yt-dlp",
            # 优先 720p mp4，节省空间和时间
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--retries", "3",
            "--fragment-retries", "3",
            "--no-warnings",
            "-o", out,
            url,
        ]
        t0 = time.time()
        ret = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0

        if ret.returncode == 0 and os.path.isfile(out):
            size_mb = os.path.getsize(out) / 1024 / 1024
            logger.info(f"  ✅ 下载完成：{size_mb:.1f} MB，耗时 {elapsed:.0f}s")
            video_paths[vid] = out
        else:
            logger.warning(f"  ❌ 下载失败（可能被限制或已下架）：{vid}")

    logger.info(f"\n下载完成：{len(video_paths)}/{len(samples)} 个视频成功")
    return video_paths


# ---------------------------------------------------------------------------
# Step 3: 构建金字塔 + 推理
# ---------------------------------------------------------------------------

def run_inference(
    samples: list,
    video_paths: dict,
    output_dir: str,
    qwen_api_key: str,
    qwen_vlm_model: str,
    qwen_answer_model: str,
    l1_duration: float,
    l2_duration: float,
    l3_fps: float,
) -> list:
    """对每个视频构建金字塔并运行 Video-Tree-TRM 问答推理。

    Args:
        samples:           元数据列表
        video_paths:       {videoID: 本地路径}
        output_dir:        输出根目录
        qwen_api_key:      百炼 API Key
        qwen_vlm_model:    金字塔构建用 VLM 模型
        qwen_answer_model: 答案生成用 VLM 模型
        l1/l2 duration:    金字塔参数

    Returns:
        results 列表，每项含 videoID / question / prediction / ground_truth / correct
    """
    # 添加项目根目录到路径
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from video_tree_trmRAG.config import VideoTreeTRMConfig
    from video_tree_trmRAG.pipeline import VideoQAPipeline

    logger.info("=" * 60)
    logger.info("Step 3: 构建金字塔 & 推理")
    logger.info("=" * 60)

    pyramid_dir = os.path.join(output_dir, "pyramids")
    os.makedirs(pyramid_dir, exist_ok=True)

    cfg = VideoTreeTRMConfig.from_dict({
        "pyramid": {
            "l1_segment_duration": l1_duration,
            "l2_clip_duration":    l2_duration,
            "l3_fps":              l3_fps,
        },
        "embedding": {
            "clip_model":   "ViT-B/32",
            "clip_device":  "cuda",
            "text_backend": "clip",
        },
        "vlm": {
            "backend":        "qwen",
            "qwen_api_key":   qwen_api_key,
            "qwen_vlm_model": qwen_vlm_model,
        },
        "answer": {
            "backend":           "qwen",
            "qwen_api_key":      qwen_api_key,
            "qwen_answer_model": qwen_answer_model,
        },
        "verbose": True,
    })
    pipeline = VideoQAPipeline(cfg)

    all_results = []

    for i, sample in enumerate(samples):
        vid = sample["videoID"]
        if vid not in video_paths:
            logger.warning(f"[{i+1}] 跳过（视频未下载）: {vid}")
            continue

        video_path  = video_paths[vid]
        pyr_dir     = os.path.join(pyramid_dir, vid)

        logger.info(f"\n{'='*50}")
        logger.info(f"[{i+1}/{len(samples)}] 处理视频: {vid}")
        logger.info(f"  视频路径 : {video_path}")
        logger.info(f"  问题数量 : {len(sample['questions'])} 道")

        # 构建金字塔（若已存在则跳过）
        pyr_meta = os.path.join(pyr_dir, "pyramid.npz")
        if os.path.isfile(pyr_meta):
            logger.info("  金字塔已存在，直接加载")
        else:
            logger.info("  开始构建分层语义金字塔...")
            try:
                pipeline.build_pyramid(video_path=video_path, save_dir=pyr_dir)
            except Exception as e:
                logger.error(f"  ❌ 金字塔构建失败: {e}")
                continue

        # 对该视频的每道题运行推理
        for q_item in sample["questions"]:
            question    = q_item["question"]
            options     = q_item["options"]
            ground_truth = q_item["answer"]  # "A" / "B" / "C" / "D"

            # 将选项拼入问题（帮助 VLM 输出选项字母）
            options_text = "\n".join(options)
            full_query = (
                f"{question}\n\nOptions:\n{options_text}\n\n"
                f"Please answer with ONLY the option letter (A, B, C, or D)."
            )

            logger.info(f"\n  ❓ 问题: {question[:70]}...")
            logger.info(f"     答案: {ground_truth}")

            try:
                result = pipeline.run_from_pyramid(
                    pyramid_dir=pyr_dir,
                    query=full_query,
                )
                prediction = result.answer.strip()
                # 提取首字母作为选项（模型可能返回详细解释）
                pred_letter = prediction[0].upper() if prediction else "?"
                correct = pred_letter == ground_truth

                logger.info(f"     预测: {pred_letter} ({'✅' if correct else '❌'})")
                logger.info(f"     原始输出: {prediction[:100]}...")

                all_results.append({
                    "videoID":       vid,
                    "question_id":   q_item["question_id"],
                    "task_type":     q_item["task_type"],
                    "question":      question,
                    "options":       options,
                    "ground_truth":  ground_truth,
                    "prediction":    pred_letter,
                    "raw_output":    prediction[:300],
                    "correct":       correct,
                    "timestamp":     result.trace.target_timestamp if result.trace else -1,
                })

            except Exception as e:
                logger.error(f"  ❌ 推理失败: {e}")
                all_results.append({
                    "videoID":      vid,
                    "question_id":  q_item["question_id"],
                    "ground_truth": ground_truth,
                    "prediction":   "?",
                    "correct":      False,
                    "error":        str(e),
                })

    return all_results


# ---------------------------------------------------------------------------
# Step 4: 统计结果
# ---------------------------------------------------------------------------

def compute_metrics(results: list, output_dir: str) -> None:
    """计算并打印准确率统计，保存结果 JSON。"""
    if not results:
        logger.warning("无有效结果可统计。")
        return

    total   = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    acc     = correct / total * 100

    # 按 task_type 分类统计
    task_stats: dict = {}
    for r in results:
        t = r.get("task_type", "Unknown")
        if t not in task_stats:
            task_stats[t] = {"total": 0, "correct": 0}
        task_stats[t]["total"] += 1
        if r.get("correct"):
            task_stats[t]["correct"] += 1

    print("\n" + "=" * 60)
    print("📊  Video-MME 评测结果 —— Video-Tree-TRM")
    print("=" * 60)
    print(f"  总准确率 : {correct}/{total} = {acc:.1f}%")
    print()
    print("  按题型分类：")
    for task, stat in sorted(task_stats.items()):
        t_acc = stat["correct"] / stat["total"] * 100
        print(f"    {task:<35} {stat['correct']:>3}/{stat['total']:>3} = {t_acc:.1f}%")
    print("=" * 60)

    # 保存完整结果
    result_path = os.path.join(output_dir, "evaluation_results.json")
    summary = {
        "total":   total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "by_task": {
            t: {
                "accuracy": round(s["correct"] / s["total"] * 100, 2),
                **s,
            }
            for t, s in task_stats.items()
        },
        "details": results,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📄 完整结果已保存：{result_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-MME 下载 & Video-Tree-TRM 评测一体化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="/tmp/videomme",
                        help="所有输出的根目录（元数据/视频/金字塔/结果）")
    parser.add_argument("--duration", default="long",
                        choices=["short", "medium", "long", "all"],
                        help="视频时长筛选：short(<2min) / medium(4-15min) / long(>30min)")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="下载并测试的视频数量（建议首次先用3-5个）")

    # 模式控制
    parser.add_argument("--download_only",  action="store_true",
                        help="只下载视频，不运行推理")
    parser.add_argument("--inference_only", action="store_true",
                        help="只运行推理（视频和元数据已提前下载好）")

    # Qwen API
    parser.add_argument("--qwen_api_key", default="",
                        help="阿里云百炼 API Key（推理时必填）")
    parser.add_argument("--qwen_vlm_model", default="qwen-vl-plus",
                        help="构建金字塔用的 VLM（qwen-vl-plus / qwen-vl-max）")
    parser.add_argument("--qwen_answer_model", default="qwen-vl-plus",
                        help="答案生成用的模型（qwen-vl-plus / qwen-vl-max / qwen-plus）")

    # 金字塔参数
    parser.add_argument("--l1_duration", type=float, default=300.0,
                        help="L1 段时长（秒），长视频建议 300-600")
    parser.add_argument("--l2_duration", type=float, default=20.0,
                        help="L2 片段时长（秒）")
    parser.add_argument("--l3_fps",      type=float, default=0.5,
                        help="L3 帧率（fps），0.5 表示每2秒一帧")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = os.path.join(args.output_dir, "videos")
    meta_path = os.path.join(args.output_dir, "metadata.json")

    # ── 加载或下载元数据 ──────────────────────────────────────────────
    if args.inference_only and os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            samples = json.load(f)
        logger.info(f"从本地加载元数据：{len(samples)} 个视频")
    else:
        samples = download_metadata(args.duration, args.num_videos, args.output_dir)

    if args.download_only:
        download_videos(samples, video_dir)
        logger.info("✅ 视频下载完成！使用 --inference_only 运行推理。")
        return

    # ── 下载视频 ─────────────────────────────────────────────────────
    if not args.inference_only:
        video_paths = download_videos(samples, video_dir)
    else:
        # 推理模式：扫描已有视频
        video_paths = {}
        for s in samples:
            p = os.path.join(video_dir, f"{s['videoID']}.mp4")
            if os.path.isfile(p):
                video_paths[s["videoID"]] = p
        logger.info(f"发现 {len(video_paths)} 个本地视频")

    if not video_paths:
        logger.error("❌ 没有可用视频，退出。请先运行 --download_only 下载视频。")
        sys.exit(1)

    # ── 检查 API Key ──────────────────────────────────────────────────
    if not args.qwen_api_key:
        logger.error("❌ 推理需要 --qwen_api_key，请提供阿里云百炼 API Key。")
        sys.exit(1)

    # ── 运行推理 ─────────────────────────────────────────────────────
    results = run_inference(
        samples         = [s for s in samples if s["videoID"] in video_paths],
        video_paths     = video_paths,
        output_dir      = args.output_dir,
        qwen_api_key    = args.qwen_api_key,
        qwen_vlm_model  = args.qwen_vlm_model,
        qwen_answer_model = args.qwen_answer_model,
        l1_duration     = args.l1_duration,
        l2_duration     = args.l2_duration,
        l3_fps          = args.l3_fps,
    )

    # ── 统计结果 ─────────────────────────────────────────────────────
    compute_metrics(results, args.output_dir)


if __name__ == "__main__":
    main()

