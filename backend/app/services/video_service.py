import uuid
import numpy as np
from fastapi import HTTPException
from pathlib import Path
from typing import Tuple, List
import torch
import clip
from PIL import Image
import ffmpeg
from backend.app.config import settings
from asyncio import to_thread
import shutil


# 封装路径构造函数
def generate_temp_file_path(filename_prefix: str, ext: str = "mp4") -> Path:
    return Path(settings.temp_dir) / f"{filename_prefix}_{uuid.uuid4().hex}.{ext}"


# 该类负责加载CLIP模型，提取视频帧，分析帧与文本匹配度，计算符合语义的片段
class VideoProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(settings.clip_model, device=self.device)  # 加载CLIP模型
        self.model.eval()  # 使CLIP进入推理模式，提高计算稳定性
        self.segments = []

    # 异步函数async，表示该函数的执行不会阻塞后面代码的执行
    async def process_video(
            self,
            video_path: str,  # 待处理视频的路径
            text_query: str,  # 要匹配的文本
            min_duration: float = 3.0,  # 最小片段时长，默认3秒
            threshold: float = None  # 匹配相似度阈值
    ) -> List[Path]:
        try:
            # 分帧处理，await用于等待异步方法的执行，extract_frames方法返回存放提取出的帧的目录路径
            frame_dir = await self.extract_frames(video_path)

            # 特征提取与匹配
            segments = await self.analyze_frames(frame_dir, text_query, min_duration,
                                                 threshold or settings.threshold)
            self.segments = segments

            # 清理帧目录
            await self.cleanup_directory(frame_dir)

            if not segments:
                return []

            # 为所有片段生成视频文件
            output_paths = []
            for start, end in segments:
                clip_path = await self.crop_video(video_path, start, end)
                output_paths.append(clip_path)

            return output_paths  # 返回所有裁剪后的文件路径

        except Exception as e:
            raise HTTPException(500, f"视频处理失败: {str(e)}")  # 发生异常时抛出HTTPException(500)

    async def crop_video(self, input_path: str, start: float, end: float) -> Path:
        # 生成输出文件路径
        output_path = generate_temp_file_path("output")

        (
            ffmpeg.input(input_path, ss=start, to=end)
            .output(str(output_path), c='copy')  # 使用流拷贝避免重新编码
            .overwrite_output()  # 如果 output_path 已存在，则覆盖原文件
            .run_async(pipe_stdout=True, pipe_stderr=True)  # 异步执行 FFmpeg 进程，不阻塞 Python 代码
        ).communicate()  # 等待 FFmpeg 进程执行完成，并读取标准输出和错误信息

        return output_path

    async def extract_frames(self, video_path: str) -> str:
        frame_dir = generate_temp_file_path("frames")  # 定义一个路径对象

        # 创建目录，parents=True表示自动创建父目录，exist_ok=True表示若目录存在不会报错
        frame_dir.mkdir(parents=True, exist_ok=True)

        # 使用FFmpeg提取关键帧
        (
            ffmpeg
            .input(video_path)  # 读取指定的视频文件
            .filter('fps', fps=settings.frame_rate)  # 设置帧率，决定每秒提取多少帧
            .output(str(frame_dir / 'frame_%04d.jpg'), start_number=0)  # 指定输出路径和文件名格式，%04d表示4位零填充的数字
            .overwrite_output()  # 如果输出文件已存在，强制覆盖，不会报错
            .run_async(pipe_stdout=True, pipe_stderr=True)  # 异步运行FFmpeg进程，不会阻塞Python代码执行
        ).communicate()  # 等待进程完成，并清空缓冲区，防止进程挂起

        return str(frame_dir)  # 返回存放帧的目录路径

    async def analyze_frames(
            self,
            frame_dir: str,
            text_query: str,
            min_duration: float,
            threshold: float
    ) -> List[Tuple[float, float]]:
        """分析帧并定位最佳片段"""
        # 文本编码
        text = clip.tokenize([text_query]).to(self.device)  # 转换成CLIP模型可理解的格式，将数据移动到GPU或CPU
        text_features = self.model.encode_text(text)  # 通过CLIP编码文本得到向量

        # 归一化文本特征向量
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 帧处理
        frame_files = sorted(Path(frame_dir).glob("*.jpg"))
        similarities = []  # 存储每一帧与文本的相似度

        for frame_path in frame_files:

            # 读取帧图像并进行预处理
            image = await to_thread(Image.open, frame_path)
            image = self.preprocess(image).unsqueeze(0).to(self.device)

            # 禁用梯度计算，避免占用额外的显存
            with torch.no_grad():

                # 通过CLIP编码图像，得到图像的向量表示
                image_features = self.model.encode_image(image)

                # 归一化图像特征向量
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 计算图像向量和文本向量的点积，得到相似度张量返回给CPU，并转换成float值
                similarity = (image_features @ text_features.T).cpu().item()

            # 将相似度存入similarities列表
            similarities.append(similarity)

        # 时间片段检测
        return await to_thread(self._find_segments, np.array(similarities),
                               settings.frame_rate, min_duration, threshold)

    def _find_segments(self, similarities: np.ndarray, fps: float, min_duration: float, threshold: float):
        """滑动窗口算法定位片段"""
        window_size = int(fps * min_duration)  # 计算滑动窗口大小
        smoothed = np.convolve(similarities, np.ones(5) / 5, mode='same')  # 通过卷积操作对相似度数组进行平滑处理，减小噪声影响

        segments = []  # 存储最终检测到的时间段
        start_idx = None  # 跟踪当前片段的索引若相似度超过阈值则记录开始帧，相似度低于阈值记录结束帧

        for i in range(len(smoothed)):
            if smoothed[i] > threshold and start_idx is None:
                start_idx = i
            elif smoothed[i] < threshold and start_idx is not None:
                end_idx = i
                duration = (end_idx - start_idx) / fps
                if duration >= min_duration:
                    segments.append((
                        start_idx / fps,
                        end_idx / fps
                    ))
                start_idx = None

        return segments

    async def cleanup_directory(self, dir_path: str):
        """异步清理目录"""
        await to_thread(lambda: shutil.rmtree(dir_path, ignore_errors=True))
