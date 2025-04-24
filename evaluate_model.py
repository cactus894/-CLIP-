import json
import csv
import asyncio
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, auc
from backend.app.services.video_service import VideoProcessor

dataset_path = Path("data/val_1.json")
video_root = Path("data/videos")
output_dir = Path("data/results")
num_samples = 100

processor = VideoProcessor()

# 配置日志记录格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("eval.log"), logging.StreamHandler()]
)


def calculate_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """计算预测时间段与真实时间段之间的 IoU（交并比）"""
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0.0, inter_end - inter_start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union != 0 else 0.0


async def process_single_video(
    video_path: Path,
    sentences: List[str],
    timestamps: List[Tuple[float, float]]
) -> List[Tuple]:
    """处理单个视频及其对应的所有句子，返回匹配结果"""
    results = []
    if not video_path.exists():
        logging.warning(f"视频文件不存在: {video_path}")
        return results

    try:
        for sent, gt_ts in zip(sentences, timestamps):
            # 异步处理视频，获取预测时间段列表
            await processor.process_video(str(video_path), sent)
            pred_segments = processor.segments

            # 寻找每个句子与真实时间段 IoU 最大的预测结果
            best_iou = 0.0
            best_pred = (0.0, 0.0)
            for pred_ts in pred_segments:
                current_iou = calculate_iou(pred_ts, gt_ts)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred = pred_ts

            if best_iou > 0:
                results.append((str(video_path), sent, *gt_ts, *best_pred, best_iou))

    except Exception as e:
        logging.error(f"处理视频失败 {video_path}: {str(e)}")

    return results
    # eg. results = [
    #     ("videos/video1.mp4", "This is a sample sentence.", 0.0, 2.0, 0.5, 1.5, 0.5),
    #     ("videos/video1.mp4", "Another description.", 3.0, 5.0, 3.0, 4.0, 0.5)
    # ]


def calculate_metrics(results: List[Tuple]) -> Dict:
    """根据所有匹配结果计算性能指标，包括 IoU、Precision、Recall、F1"""
    ious = [item[-1] for item in results]  # 从 results 列表中的每个元素提取最后一个值，即best_iou
    thresholds = [0.3, 0.5, 0.7]
    metrics = {
        "mean_iou": np.mean(ious),  # 使用 NumPy 库计算一系列交并比的平均值
        "recall@0.5": sum(iou >= 0.5 for iou in ious) / len(ious),  # 在阈值为 0.5 的情况下的召回率
    }

    # 不同阈值下的精确度、召回率、F1
    for t in thresholds:
        tp = sum(iou >= t for iou in ious)   # 真阳性
        fp = len(ious) - tp                  # 假阳性
        fn = len(ious) - tp                  # 假阴性

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 更新指标字典，update 方法用于在 metrics 字典中添加或更新键值对
        metrics.update({
            f"precision@{t}": precision,
            f"recall@{t}": recall,
            f"f1@{t}": f1
        })

    return metrics


def visualize_results(metrics: Dict, ious: List[float]):
    """可视化结果图表，包括 F1 分数柱状图 和 IoU 分布直方图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置中文字体（Windows 默认路径）
    font_path = "C:/Windows/Fonts/simhei.ttf"
    font_prop = fm.FontProperties(fname=font_path)

    thresholds = [0.3, 0.5, 0.7]

    # F1 分数柱状图
    f1_scores = [metrics[f"f1@{t}"] for t in thresholds]
    plt.figure(figsize=(10, 6))
    plt.bar([str(t) for t in thresholds], f1_scores, color='skyblue')
    plt.title("不同IoU阈值下的F1分数", fontproperties=font_prop)
    plt.xlabel("IoU阈值", fontproperties=font_prop)
    plt.ylabel("F1分数", fontproperties=font_prop)
    plt.savefig(output_dir / "f1_scores.png")
    plt.close()

    # IoU 分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=20, edgecolor='black', alpha=0.7)
    plt.title("IoU分布直方图", fontproperties=font_prop)
    plt.xlabel("IoU", fontproperties=font_prop)
    plt.ylabel("频数", fontproperties=font_prop)
    plt.savefig(output_dir / "iou_distribution.png")
    plt.close()

    # 绘制 PR 曲线
    y_true = [1 if iou >= 0.5 else 0 for iou in ious]  # 标记真实标签
    precision, recall, _ = precision_recall_curve(y_true, ious)  # 使用 iou 作为模型预测的分数
    pr_auc = auc(recall, precision)  # 计算 PR 曲线下的面积
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.fill_between(recall, precision, color='lightblue', alpha=0.3)
    plt.title(f"PR 曲线 (AUC = {pr_auc:.2f})", fontproperties=font_prop)
    plt.xlabel("Recall", fontproperties=font_prop)
    plt.ylabel("Precision", fontproperties=font_prop)
    plt.savefig(output_dir / "pr_curve.png")
    plt.close()


async def evaluate_model() -> None:
    """模型评估主函数：执行全部处理并输出结果"""
    results = []

    # 加载 JSON 格式的标注数据，使用 json.load 将其转换为 Python 字典
    with open(dataset_path, 'r') as f:
        data: Dict = json.load(f)

    # 遍历字典 data 中的所有键值对，vid 是键（视频名），item 是值（包含 sentences 和 timestamps）
    limit = min(num_samples, len(data)) if num_samples else len(data)
    for i, (vid, item) in enumerate(tqdm(data.items(), total=limit, desc="Evaluating")):
        if num_samples and i >= num_samples:
            break

        video_path = video_root / f"{vid}.mp4"

        # 串行处理每个视频
        single_video_result = await process_single_video(
            video_path,
            item['sentences'],
            item['timestamps']
        )

        # 添加当前视频的所有结果
        results.extend(single_video_result)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存处理结果为 CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "sentence", "gt_start", "gt_end",
                        "pred_start", "pred_end", "iou"])
        writer.writerows(results)

    # 计算并输出评估指标
    metrics = calculate_metrics(results)
    logging.info(f"评估指标: {metrics}")

    # 生成图表
    visualize_results(metrics, [x[-1] for x in results])

if __name__ == '__main__':
    asyncio.run(evaluate_model())
