import os
import json
import subprocess
from pathlib import Path

VAL_JSON_PATH = "data/val_1.json"       # 请根据你实际位置修改
SAVE_DIR = "data/videos"                # 视频保存路径
NUM_VIDEOS = 20                         # 测试前 20 个视频

# 创建保存路径
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# 读取 JSON
with open(VAL_JSON_PATH, "r") as f:
    val_data = json.load(f)


# 视频ID提取
video_ids = list(val_data.keys())[:NUM_VIDEOS]


# 下载视频
for vid in video_ids:

    # YouTube 视频链接（标准格式）
    video_url = f"https://www.youtube.com/watch?v={vid.replace('v_', '')}"
    save_path = os.path.join(SAVE_DIR, f"{vid}.mp4")

    if os.path.exists(save_path):
        print(f"[✓] 已存在: {vid}")
        continue

    print(f"[↓] 下载中: {vid}")

    # yt-dlp 命令构造
    command = [
        "yt-dlp",
        "-f", "mp4",
        "-o", save_path,
        video_url
    ]

    try:
        subprocess.run(command, check=True)
        print(f"[✓] 成功: {vid}")
    except subprocess.CalledProcessError:
        print(f"[✗] 下载失败: {vid}")
