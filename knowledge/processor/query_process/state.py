# knowledge/processor/query_process/state.py

"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """查询流程图状态。

    包含整个查询流程中传递的所有数据。
    """
    session_id: str              # 会话ID
    task_id: str                  # 任务ID
    message_id: str               # 消息ID
    original_query: str           # 原始查询
    embedding_chunks: list        # 向量检索结果
    hyde_embedding_chunks: list   # HyDE检索结果
    rrf_chunks: list              # RRF融合后的切片
    web_search_docs: list         # 网页搜索结果
    reranked_docs: list           # 重排序后的文档
    prompt: str                   # 提示词
    answer: str                   # 答案
    item_names: List[str]         # 商品名称
    rewritten_query: str          # 重写查询
    history: list                 # 历史对话
    is_stream: bool               # 是否流式输出


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。
    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本。"""
    return copy.deepcopy(DEFAULT_STATE)


# 兼容旧版变量名
graph_default_state = DEFAULT_STATE