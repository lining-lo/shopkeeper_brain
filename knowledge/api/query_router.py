"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc: 知识库FastAPI入口，提供前端静态页面、对话查询、SSE流式接口、会话历史管理
"""
import asyncio
import os
import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_local_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_task_result


def register_routes(app):
    """注册全部业务路由"""
    @app.get("/chat")
    async def chat_page():
        """聊天页面接口，返回前端chat.html"""
        return FileResponse(os.path.join(get_local_page_dir(), "chat.html"))

    @app.post("/query", response_model=StreamSubmitResponse | QueryResponse)
    async def query(request: QueryRequest, background_tasks: BackgroundTasks,
                    service: QueryService = Depends(get_query_service)):
        """
        用户提问统一入口
        自动区分【流式SSE模式】 / 【同步一次性返回模式】
        :param request: 用户提问请求体
        :param background_tasks: FastAPI内置后台任务（用于流式场景）
        :param service: 查询服务实例
        """
        query = request.query
        # 不存在session_id则新建会话
        session_id = request.session_id or service.create_session_id()
        is_stream = request.is_stream
        task_id = service.create_task_id()

        if is_stream:
            # 流式分支：提前创建SSE消息队列
            create_sse_queue(task_id)
            # 后台执行查询流程图，接口直接返回task_id，前端再去连接 /stream/{task_id} 接收数据流
            # ⚠️参数顺序注意：和run_query_graph形参保持一致 task_id, session_id, query, is_stream
            background_tasks.add_task(service.run_query_graph, task_id, session_id, query, is_stream)
            return StreamSubmitResponse(
                message="提交查询请求",
                session_id=session_id,
                task_id=task_id
            )
        else:
            # 非流式分支：同步阻塞执行，等待完整推理结束再返回答案
            # run_query_graph包含同步重型LLM/向量检索，使用线程池隔离，防止阻塞全局事件循环
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, service.run_query_graph, task_id, session_id, query, is_stream)
            # 执行完成后读取存储的最终回答
            answer = get_task_result(task_id, "answer")
            return QueryResponse(
                message="查询成功",
                session_id=session_id,
                answer=answer,
                task_id=task_id
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, reqest: Request):
        """
        SSE长连接流式接口
        前端拿到task_id后请求此地址，持续接收LLM增量输出、进度事件
        """
        return StreamingResponse(
            sse_generator(task_id, reqest),
            media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        """根据会话ID，读取历史聊天记录"""
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        """清空指定会话全部聊天历史"""
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def create_app() -> FastAPI:
    """
    创建并初始化FastAPI应用实例【工厂模式】
    执行流程：初始化日志 -> 创建应用实例 -> 注册跨域中间件 -> 挂载前端静态资源 -> 注册业务路由
    :return: 配置完成的FastAPI实例
    """
    # 初始化全局日志配置
    setup_logging()

    # 实例化FastAPI应用
    app = FastAPI(description="知识库查询服务", version="v1.0")

    # 注册跨域中间件，允许前端页面跨域调用接口
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源域名访问，仅开发环境使用；生产环境限制域名
        allow_credentials=False,  # 搭配allow_origins=["*"] 不能开启凭证
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载前端静态资源目录，访问示例：/front/chat.html
    app.mount("/front", StaticFiles(directory=get_local_page_dir()), name="front_static")

    # 注册所有业务接口路由
    register_routes(app)

    return app


if __name__ == "__main__":
    # 启动服务，监听所有网卡，端口8001
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)