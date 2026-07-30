"""
  @Author:lining-lo
  @Time:2026/7/30
  @Desc:FastAPI SSE大模型流式输出工具
        基于生产者消费者模型，利用task_id绑定独立队列实现多任务隔离；
        完成消息中转、SSE协议封装、连接监控、队列资源自动清理。
        支持进度、流式增量、最终结果三类事件推送。
"""
import asyncio
import json
import logging
import queue
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request

"""
    前端请求 /start-task?task_id=task_001
            ↓
    create_sse_queue("task_001") 创建队列存入全局字典
            ↓
    开启后台LLM推理任务【生产者】
            │
            │ LLM每生成一段文字
            ▼
    push_sse_event("task_001", DELTA, 文字数据)
            │
            ▼
    消息放入 task_001 对应的Queue队列
    
    # 另外一条链路（浏览器同时打开SSE长连接）
    前端打开 /sse/stream?task_id=task_001
            ↓
    sse_generator 启动循环【消费者】
            ↓
    不断从 task_001 队列取消息
            ↓
    _sse_pack 组装SSE标准格式
            ↓
    yield 文本推送到浏览器，实现打字效果
    
    # 用户关闭浏览器
    循环退出 → finally执行 → remove_sse_queue 销毁队列
"""

class SSEEvent:
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"  # LLM 流式输出增量
    FINAL = "final"  # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
# {
#     "task_123": queue.Queue(),
#     "task_456": queue.Queue()
# }
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    # 获取队列对象（队列 阻塞队列(非阻塞队列)）

    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    """移除指定任务的队列
    不存在 key 默认返回 None
    """
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    # event: delta \n data: {"content": "你好世界"} \n\n
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 队列存在
    if stream_queue:
        # 3. 将事件推送到队列
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """
    流式输出结果的消费者
    1. 从sse队列中获取结果
    2. 封装队列中的数据以及事件类型为sse协议的数据包格式
    3. 将封装好的数据包yield出去
    Args:
        task_id: 任务id

    Returns:
        AsyncGenerator:异步生成器对象

    """

    # 1. 根据任务id 获取任务队列对象
    sse_queue = _task_stream.get(task_id)

    # 2. 校验
    if sse_queue is None:
        return

    loop = asyncio.get_event_loop()

    # 3. 让当前线程一直从队列中获取数据【如果队列一旦有数据，就直接获取，如果队列没有数据，等一会，在问一下】
    try:
        while True:

            # 3.1 判断前端sse连接是否关闭（主动探测）--->FastApI:可以感知到：request
            if await request.is_disconnected():
                return
            try:
                # 3.2 从队列中获取(阻塞队列---)为了让事件循环不阻塞，
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                # 3.3 获取事件类型
                event_type = msg.get('event')
                # 3.4 获取事件数据
                event_data = msg.get('data')
                # 3.5 打包返回
                yield _sse_pack(event_type, event_data)  # 打包并且通过yield返回
            except queue.Empty:
                logging.info(f"队列为空...请稍等")
                continue
    except  (ConnectionResetError, BrokenPipeError) as e:
        # 客户端中断 关闭了窗口或者浏览器
        return

    except asyncio.CancelledError:
        # 服务端中断 协程被取消 重新抛出，让外层知道它被成功取消()
        raise
    finally:
        remove_sse_queue(task_id)
