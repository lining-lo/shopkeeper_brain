"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:商品名确认节点
        读取聊天历史，调用LLM提取问题中的商品名称，清洗JSON格式；
        支持向量相似度对齐筛选，将商品、优化问句写入Graph状态，用于后续Milvus检索过滤
"""
import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import get_recent_messages


class ItemNameExtractor:
    """LLM商品名称识别，清洗LLM返回结果"""

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        调用LLM进行商品名称识别
        :param original_query: 原始问题   "万用表怎么测量电阻"
        :param history_text: 历史对话内容 10条  最近5轮对话
        :return:
            {
                "item_names": ["RS-12 万用表"],
                "rewritten_query": "RS-12 万用表怎么测量电阻"
            }
        """

        result = {"item_names": [], "rewritten_query": original_query}  # 默认结果

        llm_client = AIClients.get_llm_openai(True)
        if llm_client is None:
            return result

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=original_query, history_text=history_text)

        llm_response: AIMessage = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        llm_content = llm_response.content.strip()
        if llm_content is None:
            return result

        # 清洗LLM返回结果
        try:
            clean_llm_content_json = self._clean_parse(llm_content)

            result["item_names"] = clean_llm_content_json.get("item_names", [])
            result["rewritten_query"] = clean_llm_content_json.get("rewritten_query", original_query)
            return result
        except Exception as e:
            self.logger.error(f"LLM返回结果清洗失败：{str(e)}")
            return result

    def _clean_parse(self, llm_content: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            # 2.1 清洗 item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in rwa_item_names if raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    """向量化查询,商品名称对齐评分，分数差过滤"""

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name


class ItemNameConfirmNode(BaseNode):
    name: str = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self.item_name_aligner = ItemNameAligner(self.logger, self.name)
        self.item_name_extractor = ItemNameExtractor(self.logger, self.name)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.获取mongodb历史会话
        # {
        #     "_id": {
        #         "$oid": "6a48cc0f82601667381db7a5"
        #     },
        #     "session_id": "sess-f41icdzoh87mr5zeel1",
        #     "role": "user",
        #     "text": "万用表怎么测量电阻",
        #     "rewritten_query": "万用表怎么测量电阻",
        #     "item_names": [],
        #     "ts": 1783155727.498414
        # }
        # {
        #     "_id": {
        #         "$oid": "6a48cc0f82601667381db7a6"
        #     },
        #     "session_id": "sess-f41icdzoh87mr5zeel1",
        #     "role": "assistant",
        #     "text": "我不确定您指的是哪款产品。, 您是在询问以下产品吗: RS PRO RS-12 数字万用表、HAK 180 扫描仪？",
        #     "rewritten_query": "万用表怎么测量电阻",
        #     "item_names": [],
        #     "ts": 1783155727.52313
        # }
        session_id: str = state.get("session_id")
        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id)

        # 2.LLM获取商品名称
        original_query: str = state.get("original_query")

        history_text = ""
        for message in history_messages:
            history_text += f"{message["role"]}:{message['text']}\n"

        extract_item_names: Dict[str, Any] = self.item_name_extractor.extract_item_name(original_query, history_text)
        state["item_names"] = extract_item_names["item_names"]
        state["rewritten_query"] = extract_item_names["rewritten_query"]

        return state


if __name__ == "__main__":
    item_name_confirmed_node = ItemNameConfirmNode()
    init_state = {
        "session_id": "sess-f41icdzoh87mr5zeel1",
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node(init_state)
    print(llm_result)
