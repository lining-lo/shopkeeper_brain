"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:HyDE假设性文档增强检索节点
        接收重写问句与目标商品列表，通过LLM根据用户问题生成高质量假设性参考文档；
        拼接【原始问句+假设文档】作为检索文本，利用BGE-M3生成稠密+稀疏混合向量；
        结合商品过滤条件执行Milvus混合加权检索，强化语义召回能力，解决纯问句短文本检索不准、语义缺失问题，
        输出增强召回知识库片段，供给后续重排与问答节点使用。
"""
from typing import Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class HyDeSearchNode(BaseNode):
    name = "search_embedding_hyde"

    def process(self, state: QueryGraphState) -> Dict:
        """
        :param state: rewritten_query,item_names
        :return:
            {
                "hyde_embedding_chunks": []
            }
        """
        # 1.参数校验
        validated_rewritten_query, validated_item_names = self._validate_input(state)

        # 2.生成假设性文档  根据问题 -> LLM  ->  200-300字的问题答案
        hyde_document = self._generate_hyde_documents(validated_rewritten_query, validated_item_names)

        # 3.拼接  问题 + 假设性文档  =>  向量化
        embedding_query_str = f"{validated_rewritten_query}\n{hyde_document}"

        # 2.获取客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to get milvus client: {e}")
            return {}

        try:
            embedding_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"Failed to get bge_m3_client: {e}")
            return {}
        try:
            # 3.查询向量化
            embedding_hybrid_result = generate_bge_m3_hybrid_vectors(embedding_model, [embedding_query_str])

            # 4.构建查询条件
            expr, expr_params = item_names_filter(validated_item_names)  # 标量查询条件
            # 5.创建混合检索请求
            search_requests = create_hybrid_search_requests(  # 混合查询条件
                dense_vector=embedding_hybrid_result["dense"][0],
                sparse_vector=embedding_hybrid_result["sparse"][0],
                expr=expr,
                expr_params=expr_params
            )

            # 6.执行混合检索
            hybrid_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["chunk_id", "item_name", "content", "title"]
            )

            # 7.回填数据
            return {
                "hyde_embedding_chunks": hybrid_result[0]
            }
        except Exception as e:
            self.logger.error(f"Failed to execute vector search: {e}")
            return {}

    def _validate_input(self, state) -> Tuple[str, list]:
        # 1.参数校验
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"Invalid rewritten_query: {rewritten_query}")
            raise StateFieldError(self.name, "rewritten_query", str)

        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"Invalid item_names: {item_names}")
            raise StateFieldError(self.name, "item_names", str)
        return rewritten_query, item_names

    def _generate_hyde_documents(self, validated_rewritten_query, validated_item_names) -> str:
        """
        生成假设性文档  根据问题 -> LLM  ->  200-300字的问题答案
        :param validated_rewritten_query: 用户问题
        :param validated_item_names: 商品名称列表
        :return: str  假设性文档
        """
        # 1.获取LLM客户端
        try:
            # 不用非得返回json格式结果
            llm_client = AIClients.get_llm_openai(False)
        except ConnectionError as e:
            self.logger.error(f"Failed to get LLM client: {e}")
            return ""

        # 2.调用LLM   重点：提示词
        system_prompt = f"您是一位{validated_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names=validated_item_names,
                                                       rewritten_query=validated_rewritten_query)
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if not llm_response or not llm_response.content.strip():
            self.logger.info("LLM生成假设性文档内容为空")
            return ""

        # 3.解析结果并返回
        llm_content = llm_response.content.strip()

        return llm_content


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")

"""
============================================================
开始测试: HyDE 检索节点 (HydeSearchNode)
============================================================
【输入状态】:
  查询: RS-12 数字万用表如何测量直流电压？
  商品: ['RS-12 数字万用表']
------------------------------------------------------------
2026-07-24 11:17:12 - query.search_embedding_hyde - INFO - --- search_embedding_hyde 开始 ---
2026-07-24 11:17:14 - knowledge.utils.client.base - INFO - ChatOpenAI LLM 客户端初始化成功
2026-07-24 11:17:57 - langsmith.client - WARNING - Failed to get info from https://api.smith.langchain.com: LangSmithConnectionError('Connection error caused failure to GET /info in LangSmith API. Please confirm your internet connection. ConnectTimeout(MaxRetryError("HTTPSConnectionPool(host=\'api.smith.langchain.com\', port=443): Max retries exceeded with url: /info (Caused by ConnectTimeoutError(<HTTPSConnection(host=\'api.smith.langchain.com\', port=443) at 0x1dd428314c0>, \'Connection to api.smith.langchain.com timed out. (connect timeout=10.0)\'))"))\nContent-Length: None\nAPI Key: lsv2_********************************************ee')
2026-07-24 11:17:57 - langsmith.client - WARNING - Run compression is not enabled. Please update to the latest version of LangSmith. Falling back to regular multipart ingestion.
2026-07-24 11:18:38 - openai._base_client - INFO - Retrying request to /chat/completions in 0.459133 seconds
2026-07-24 11:18:42 - langsmith.client - WARNING - Failed to multipart ingest runs: Connection error caused failure to POST https://api.smith.langchain.com/runs/multipart in LangSmith API. Please confirm your internet connection. ConnectTimeout(MaxRetryError("HTTPSConnectionPool(host='api.smith.langchain.com', port=443): Max retries exceeded with url: /runs/multipart (Caused by ConnectTimeoutError(<HTTPSConnection(host='api.smith.langchain.com', port=443) at 0x1dd42831e80>, 'Connection to api.smith.langchain.com timed out. (connect timeout=3)'))"))
Content-Length: 3515
API Key: lsv2_********************************************eetrace=019f9220-8b26-73d2-93db-b041a9f2abcb,id=019f9220-8b26-73d2-93db-b041a9f2abcb
2026-07-24 11:18:44 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-24 11:18:48 - FlagEmbedding.finetune.embedder.encoder_only.m3.runner - INFO - loading existing colbert_linear and sparse_linear---------
2026-07-24 11:18:48 - knowledge.utils.client.base - INFO - bge_m3客户端初始化成功
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
2026-07-24 11:18:56 - knowledge.utils.milvus_util - INFO - Milvus 混合搜索完成，共处理 1 个查询，总计找到 5 个结果
2026-07-24 11:18:56 - query.search_embedding_hyde - INFO - --- search_embedding_hyde 完成 ---

【HyDE 检索结果】: 5 条
  [1] chunk_id=467775438085177893 item_name=RS-12 数字万用表 distance=0.8031659126281738
      内容: ## 直流电流测量- 1

注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。
![直流电流测量接线示意图（10A档位）](http://1...
  [2] chunk_id=467775438085177891 item_name=RS-12 数字万用表 distance=0.4799593687057495
      内容: ## 直流电压测量

注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。
1. 将功能转盘置于V DC的位置。
2. 将黑色表笔插入负极COM...
  [3] chunk_id=467775438085177892 item_name=RS-12 数字万用表 distance=0.4688156247138977
      内容: ## 交流电压测量

警告：谨防触电。
若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应...
  [4] chunk_id=467775438085177894 item_name=RS-12 数字万用表 distance=0.4616103172302246
      内容: ## 直流电流测量- 2

![万用表RS-12直流10A电流测量接线示意图](http://192.168.6.160:9000/knowledge-base...
  [5] chunk_id=467775438085177895 item_name=RS-12 数字万用表 distance=0.4586062431335449
      内容: ## 电阻测量

警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。
1. 将功能转盘置于最高电阻Ω位置.
2. 将黑色表笔插入负极COM...
------------------------------------------------------------
测试完成
"""
