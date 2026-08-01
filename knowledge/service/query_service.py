"""
  @Author:lining-lo
  @Time:2026/8/1
  @Desc: 知识库查询核心服务
         封装LangGraph查询流程图执行、任务状态管理、会话历史CRUD
"""
import logging
import uuid
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, get_task_result

logger = logging.getLogger(__name__)


class QueryService:
    """
    知识库查询业务服务
    职责：
    1. 执行LangGraph查询主流程（向量检索、Rerank、LLM回答生成）
    2. 管理任务生命周期状态（处理中/成功/失败）
    3. 任务ID、会话ID生成
    4. 聊天历史记录查询、清空
    """

    def run_query_graph(self, task_id, session_id, query, is_stream):
        """
        启动完整知识库查询流水线
        :param task_id: 当前查询任务唯一标识
        :param session_id: 用户会话ID，区分多轮对话上下文
        :param query: 用户原始提问
        :param is_stream: 是否开启SSE流式输出
        注意：当前使用 invoke() 阻塞执行，仅获取最终结果；流式消息依靠内部SSE工具推送队列
        """
        try:
            # 更新任务状态：开始处理
            update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)
            # 初始化图运行状态（全局上下文，流转于各个Node节点）
            state = {
                "original_query": query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }
            # 执行LangGraph完整链路，同步阻塞直到全部节点执行完毕
            # invoke：一次性执行全部流程，仅返回最终state，无法捕获中间流式输出
            query_app.invoke(state)

            # 全部节点正常执行完成，标记任务成功
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"启动查询流程失败：{e}")
            # 异常捕获，标记任务失败，防止任务永久卡在处理中
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)

    def create_task_id(self):
        """生成任务唯一ID，一次提问对应一个task_id"""
        return str(uuid.uuid4())

    def create_session_id(self):
        """生成会话唯一ID，一个用户对话窗口对应一个session_id，用于多轮历史关联"""
        return str(uuid.uuid4())

    def get_answer(self, task_id: str) -> str:
        """根据task_id读取任务最终回答结果"""
        return get_task_result(task_id, "answer", "")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        查询会话聊天历史
        :param session_id: 用户会话标识
        :param limit: 最大返回条数，默认50条
        :return: 格式化后的历史消息列表
        """
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        # 格式化Mongo原始文档，对外输出统一结构
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        """
        清空指定会话全部聊天历史
        :param session_id: 用户会话ID
        :return: 删除文档数量
        """
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)
