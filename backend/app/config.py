from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    app_name: str = "Video Processor"
    max_file_size: int = 500_000_000  # 500MB
    temp_dir: str = "./tmp"
    clip_model: str = "ViT-B/32"
    frame_rate: float = 3.0
    threshold: float = 0.25

    class Config:
        env_file = ".env"


settings = Settings()
os.makedirs(settings.temp_dir, exist_ok=True)  # 如果目录不存在，则创建目录
