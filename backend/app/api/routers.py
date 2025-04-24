from fastapi import APIRouter, UploadFile, File, Form
from backend.app.services.video_service import VideoProcessor, generate_temp_file_path
import aiofiles
import os
import zipfile
import io
from fastapi.responses import StreamingResponse

# 定义路由集合，将多个路由组织到一个模块中
router = APIRouter()

# 实例化VideoProcessor类
processor = VideoProcessor()


@router.post("/process")
async def process_video(
        video: UploadFile = File(...),
        text: str = Form(...),
        min_duration: float = Form(3.0),
        threshold: float = Form(None)
):
    # 保存上传文件
    upload_path = generate_temp_file_path("upload")
    async with aiofiles.open(upload_path, "wb") as buffer:
        await buffer.write(await video.read())
    output_paths = []

    try:
        # 处理视频
        output_paths = await processor.process_video(str(upload_path), text, min_duration, threshold)
        if not output_paths:
            return {"status": "fail", "message": "未找到匹配片段"}

        # 创建内存ZIP文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for clip_path in output_paths:
                zip_file.write(clip_path, arcname=clip_path.name)

        # 重置缓冲区指针
        zip_buffer.seek(0)

        # 返回视频文件流
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=clips.zip",
                "X-Clip-Count": str(len(output_paths))
            }
        )

    finally:
        # 清理所有临时文件
        for path in [upload_path] + output_paths:
            if path and path.exists():
                os.remove(path)
