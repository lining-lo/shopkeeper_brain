"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc: 知识库FastAPI入口，提供前端静态页面、导入任务接口
"""
import os
import uvicorn
from starlette.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_local_page_dir


def register_routes(app):
    @app.get("/import")
    async def import_page():
        return FileResponse(os.path.join(get_local_page_dir(), "import.html"))

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        return ""

    @app.get("/status/{task_id}")
    async def get_status(task_id: str):
        return ""


def create_app() -> FastAPI:
    app = FastAPI(description="知识库导入", version="v1.0")
    app.mount("/front", StaticFiles(directory=get_local_page_dir()))
    register_routes(app)
    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
