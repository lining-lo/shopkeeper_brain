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
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pymilvus import AnnSearchRequest

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
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

    def __init__(self, logger, node_name, config):
        self.logger = logger
        self.node_name = node_name
        self.config = config

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 向量匹配
        search_results: List[Dict[str, Any]] = self._match_vector(item_names)

        #  评分对齐
        # confirmed 高置信商品  >=0.7
        # options  中等置信商品，需要用户选择  >=0.6         < 0.7
        confirmed, options = self._item_name_score_align(search_results)

        # 分数差过滤，最高商品分数与其他商品分数差值大于 0.15   ,去掉分太低商品。
        if len(confirmed) > 1:  # (仅当confirmed长度大于1)
            confirmed = self._item_name_score_filter(confirmed, search_results)

        return confirmed, options

    def _match_vector(self, item_names) -> List[Dict[str, Any]]:
        """向量匹配： 根据LLM识别商品名称，从Milvus中查询相似商品名称（hybrid混合向量检索）
        args: 商品名称列表
        return:  search_results: List[Dict[str, Any]]
            [
                {
                    "extracted_name":extracted_name,
                    "matches": [
                        {"item_name":"商品A1", "score":0.9},
                        {"item_name":"商品A2", "score":0.6},
                    ]
                },
                {
                    "extracted_name":extracted_name,
                    "matches": [
                        {"item_name":"商品B1", "score":0.9},
                        {"item_name":"商品B2", "score":0.6},
                    ]
                }
            ]
        """
        search_results: List[Dict[str, Any]] = []  # 默认结果

        # 1.获取milvus客户端对象
        milvus_client = StorageClients.get_milvus_client()
        if not milvus_client:
            return search_results

        # 2.获取向量转换模型  BGE-M3
        embedding_model = AIClients.get_bge_m3_client()
        if not embedding_model:
            return search_results

        # 3.将商品名称转换为向量,调用工具类，批量转换
        # item_names_embeddings ={
        #    "dense": [[],[]],
        #    "sparse": [{},{}]
        # }
        item_names_embeddings = generate_bge_m3_hybrid_vectors(embedding_model, item_names, True)

        # 4.循环并装填：查询mivlus库中商品名称匹配。混合检索。
        for index, extract_item_name in enumerate(item_names):
            dense = item_names_embeddings["dense"][index]
            sparse = item_names_embeddings["sparse"][index]

            # 4.1 构建查询对象
            hybrid_search_requests: List[AnnSearchRequest] = create_hybrid_search_requests(dense, sparse)

            # 4.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.item_name_collection,  # kb_item_names_v1
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["item_name"]  # 只查询kb_item_names_v1集合中item_name字段
            )

            # .4.3 封装数据，装填列表
            search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]}
                    for h in hybrid_search_result[0] if hybrid_search_result
                ]
            }
            search_results.append(search_result)
        return search_results

    def _item_name_score_align(self, search_results) -> Tuple[List[str], List[str]]:
        """评分对齐
            规则：
                1.   score >=0.7  高分区
                2.   0.6  <= score < 0.7  中分区
                3.   score < 0.6  低分区
            args: search_results   向量检索，根据商品名称，进行向量混合检索得到列表：
                [
                    {
                        "extracted_name":extracted_name,
                        "matches": [
                            {"item_name":"商品A1", "score":0.9},
                            {"item_name":"商品A2", "score":0.8},
                        ]
                    },
                    {
                        "extracted_name":extracted_name,
                        "matches": [
                            {"item_name":"商品A1", "score":0.65},
                            {"item_name":"商品B2", "score":0.6},
                        ]
                    }
                ]
            return :
                confirmed: List[str] 高分区商品名称列表 [:3]  确认商品名称列表，立即三路检索
                options: List[str] 中分区商品名称列表    选项商品名称列表[:3]，需要让用户选择商品名，再次进行问题解答
        """
        confirmed: List[str] = []
        options: List[str] = []

        # 向量匹配的列表
        for search_result in search_results:

            # 每一个商品名称检索的类别
            extracted_name = search_result["extracted_name"]
            matches = sorted(search_result["matches"], key=lambda x: x["score"], reverse=True)  # 分数降序

            # 获取高分区列表
            high = [match["item_name"] for match in matches if match["score"] >= 0.7]

            if high:
                # 找到同名的则退出。没找到同名的返回None
                samename = next((item_name for item_name in high if item_name == extracted_name), None)
                if samename:
                    if samename not in confirmed:
                        confirmed.append(samename)
                elif len(high) == 1:
                    if high[0] not in confirmed:
                        confirmed.append(high[0])
                else:
                    for h in high[:3]:
                        if h not in options and h not in confirmed:
                            options.append(h)
            else:
                mid = [match["item_name"] for match in matches
                       if 0.6 <= match["score"] < 0.7
                       and match["item_name"] not in options
                       and match["item_name"] not in confirmed]
                if mid:
                    for m in mid[:3]:
                        options.append(m)

        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed, search_results):
        """
        分数差过滤
        在高分区中，获取最高分。最高分减去 每一个分数，差值小于 0.15 保留；  否则丢弃
        """
        high_confirmed_item_score = {}  # 所有高置信的名称和分字典
        for search_result in search_results:
            extracted_name = search_result["extracted_name"]
            matches = search_result["matches"]
            for match in matches:
                item_name = match["item_name"]
                score = match["score"]
                if item_name in confirmed:
                    high_confirmed_item_score[item_name] = max(high_confirmed_item_score.get(item_name, 0), score)

        sorted_high_confirmed_item_score = sorted(high_confirmed_item_score.items(), key=lambda x: x[1], reverse=True)
        high_score = sorted_high_confirmed_item_score[0][1]  # 最高分

        # 保留高置信组中 与最高分差值小于0.15的商品名称
        confirmed = [item_name for item_name, score in high_confirmed_item_score.items() if high_score - score <= 0.15]
        return confirmed


class ItemNameConfirmNode(BaseNode):
    name: str = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self.item_name_aligner = ItemNameAligner(self.logger, self.name, self.config)
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
        item_names: List[str] = extract_item_names["item_names"]
        rewritten_query: str = extract_item_names["rewritten_query"]

        # 3.根据商品名称在Milvus中匹配与筛选
        if item_names:
            confirmed, options = self.item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 4.决策更新状态
        self._decide(state, confirmed, options, rewritten_query)

        # 5.历史回填(仅当confirmed有值)
        for history in history_messages:
            history["item_names"] = confirmed
        state["history"] = history_messages  # 给后续节点使用

        return state

    def _decide(self, state, confirmed, options, rewritten_query):
        """
        如果confirmed存在，直接确认，进行三路检索
        如果options存在,生成state["answer"] 提示用户选择
        否则：生成state["answer"] 抱歉
        :param state:
        :param confirmed: 确认商品名称列表
        :param options: 待选商品列表，需要用户来选择
        :param rewritten_query: 查询问题  被LLM润色后问题【指代消除,润色】
        :return: None
        """
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}？"
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == "__main__":
    item_name_confirmed_node = ItemNameConfirmNode()
    init_state = {
        "session_id": "sess-f41icdzoh87mr5zeel1",
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        # "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
        "original_query": "HAK180的介质规格有哪些?"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node(init_state)
    print(llm_result)
