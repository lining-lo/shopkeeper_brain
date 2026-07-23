"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc: 知识库FastAPI入口，提供前端静态页面、导入任务接口
"""
import os
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_local_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.import_file_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_routes(app):
    @app.get("/import")
    async def import_page():
        "导入页面接口"
        return FileResponse(os.path.join(get_local_page_dir(), "import.html"))

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                                   service: ImportFileService = Depends(get_import_file_service)):
        """上传文件接口"""

        # 1.文件上传，双写
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        # 2.开启langchain流程
        background_tasks.add_task(service.run_langgraph_import, task_id, file_dir, import_file_path)

        return UploadResponse(task_id=task_id, message="文件上传成功，请等待导入流程完成")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        """获取任务状态接口，每间隔1.5秒执行一次"""
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)

def create_app() -> FastAPI:
    """
    创建并初始化FastAPI应用实例（工厂模式）
    执行流程：初始化日志 -> 创建应用实例 -> 注册跨域中间件 -> 挂载前端静态资源 -> 注册业务路由
    :return: 配置完成的FastAPI实例
    """
    # 初始化全局日志配置
    setup_logging()

    # 实例化FastAPI应用
    app = FastAPI(description="知识库导入服务", version="v1.0")

    # 注册跨域中间件，允许前端页面跨域调用接口
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源域名访问，开发环境使用
        allow_credentials=False,  # 关闭Cookie/凭证传递（搭配*域名不能开启）
        allow_methods=["*"],  # 允许全部HTTP请求方法
        allow_headers=["*"],  # 允许全部请求头
    )

    # 挂载前端静态资源目录，访问路径 /front/xxx.html
    app.mount("/front", StaticFiles(directory=get_local_page_dir()), name="front_static")

    # 注册所有业务接口路由（文件上传、页面访问、任务状态查询）
    register_routes(app)

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
