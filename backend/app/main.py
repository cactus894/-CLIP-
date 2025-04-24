from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.app.api import routers
from backend.app.config import settings
import uvicorn

# 定义一个Web应用实例
app = FastAPI(title=settings.app_name)

# 添加路由，将video.py中定义的路由注册到app中
app.include_router(routers.router, prefix="/api/v1")

# 静态文件服务，可以通过http://localhost:8000/results来访问存储在settings.temp_dir目录下的文件
app.mount("/results", StaticFiles(directory=settings.temp_dir), name="results")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
