"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:Rerank重排序节点
        融合本地RRF检索片段与web搜索结果，使用重排模型计算相关性打分，
        排序后执行动态断崖截断，剔除低相关噪音文档
"""
from typing import List, Dict, Any
from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    name = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        对 rrf结果集(向量检索 + HyDE检索) + Web-MCP结果集  进行重排序(利用reranker模型进行重新打分(交叉编码器))。
        :param state:
            rrf_chunks
                "rrf_chunks": [
                    {"chunk_id": "local_1",
                    "title": "主板维修手册",
                     "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
                    {"chunk_id": "local_2",
                    "title": "闲聊",
                     "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
                ]
            web_search_docs
                "web_search_docs": [
                    {"url": "https://example.com/repair",
                    "title": "短路查修指南",
                     "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
                    {"url": "https://example.com/news",
                    "title": "科技新闻",
                     "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
                ]
        :return:
            reranked_results : [
                    {"source":"web",
                    "url": "https://example.com/repair",
                    "title": "主板维修手册",
                     "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
                    {"source":"local",
                    "chunk_id": "local_2",
                    "title": "闲聊",
                     "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
                ]
        """

        # 1.获取查询文本 - 问题
        question: str = state.get("rewritten_query", "") or state.get("original_query", "")

        # 2.合并本地RRF结果 和  合并网络搜索结果
        multi_merge_docs = self._multi_merge_docs(state)

        # 3. 构建reranker模型的输入  (交叉编码器，  [[query-content],[query-content],[query-content]])
        # 4.Reranker 计算得分
        reranked_docs = self._reranker_docs(question, multi_merge_docs)

        # 5.动态断崖检测截断
        cutoff_docs = self._dynamic_cliff_cutoff(reranked_docs)

        # 6.更新状态返回结果
        state["reranked_docs"] = cutoff_docs
        return state

    def _multi_merge_docs(self, state):
        """
        将rrf和web-mcp数据统一格式化，合并。
        :param state:
        :return:
        """
        rrf_chunks = state.get("rrf_chunks", [])
        web_search_docs = state.get("web_search_docs", [])

        final_docs = []

        # 收集rrf文档
        for doc in rrf_chunks:

            if not isinstance(doc, dict):
                continue

            content = doc.get("content", "")
            if not content:
                continue
            title = doc.get("title", "")
            chunk_id = doc.get("chunk_id", "")
            format_doc = self.format_doc(source="local", content=content, title=title, chunk_id=chunk_id)
            final_docs.append(format_doc)

        # 收集web mcp文档
        for doc in web_search_docs:

            if not isinstance(doc, dict):
                continue

            content = doc.get("content", "") or doc.get("snippet", "")
            if not content:
                continue
            title = doc.get("title", "")
            url = doc.get("url", "")
            format_doc = self.format_doc(source="web", content=content, title=title, url=url)
            final_docs.append(format_doc)

        return final_docs

    def format_doc(self, source, content, title, chunk_id: str = "", url: str = ""):
        return {
            "source": source,
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url
        }

    def _reranker_docs(self, question: str, multi_merge_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        # 3. 构建reranker模型的输入
             pairs = [
                 ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
                 ["什么是万用表？", "今天天气很好"]
             ]
        # 4.Reranker 计算得分
            scores = reranker.compute_score(pairs,normalize=True )
        :param question:
        :param multi_merge_docs:
        :return:
        """
        # 3. 构建reranker模型的输入
        pairs = [[question, doc.get("content")] for doc in multi_merge_docs]

        # 4.Reranker 计算得分
        try:
            bge_m3_rerank_client = AIClients.get_bge_m3_rerank_client()
            # reranker_scores:List[float] = bge_m3_rerank_client.compute_score(pairs)
            reranker_scores = bge_m3_rerank_client.compute_score(pairs, normalize=True)
            # [5.06640625, -4.7421875, 3.962890625, -9.4765625]     没有进行归一化打分结果
            # API本身提供归一化处理： [0,1)
            # [0.993734466224161, 0.008644177936723585, 0.9813464782682386, 7.662101864956481e-05]
            print(reranker_scores)

            # 将分数合并到doc对象中
            doc_merge_score_list = [{**doc, "score": score} for doc, score in zip(multi_merge_docs, reranker_scores)]

            # 对文档进行排序，按照score倒数排序
            doc_sorted_list = sorted(doc_merge_score_list, key=lambda x: x["score"], reverse=True)
            print(doc_sorted_list)
            # 返回结果
            return doc_sorted_list

        except Exception as e:
            self.logger.error(f"reranker模型生成相关性得分失败了:{e}")
            # raise RerankError(f"reranker模型生成相关性得分失败了", self.name, e)
            return [{**doc, "score": None} for doc in multi_merge_docs]

    def _dynamic_cliff_cutoff(self, reranked_docs):
        """
        动态断崖检测截断。
            目的，全部文档返回，可能有部分文档分值较低，属于噪音；需要去除噪音文档。
                将相邻文档分数差值比较大的位置找到，叫做截断处。可以从断崖处进行截断。只保留断崖处之前的文档。后面文档属于噪音就不要了。
                第一断崖 VS 最大断崖  => 取决于业务场景。
        :param reranked_docs:
            需要被断崖检测的列表。

            【重排序结果】:
            [1] score=0.9937 | local | 主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。...
            [2] score=0.9813 | web   | 主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。...
            [3] score=0.0086 | local | 今天中午去吃猪脚饭吧，这块主板外观很漂亮。...
            [4] score=0.0001 | web   | 苹果发布新款手机，A系列芯片性能提升20%。...

            rerank_max_top_k = self.config.rerank_max_top_k   # 10
            rerank_min_top_k = self.config.rerank_min_top_k   # 3
            rerank_gap_abs = self.config.rerank_gap_abs  # 0.15  差值阈值
        :return:
            截断后的列表。即：断崖处之前的高分文档。
            至少返回rerank_min_top_k个文档
        """

        rerank_max_top_k = self.config.rerank_max_top_k  # 10
        rerank_min_top_k = self.config.rerank_min_top_k  # 3
        rerank_gap_abs = self.config.rerank_gap_abs  # 0.15  差值阈值
        upper_bound = min(rerank_max_top_k, len(reranked_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)

        # 无需断崖检测，直接返回
        if upper_bound <= lower_bound:
            return reranked_docs

        cut_off = upper_bound  # 断崖位置默认值

        max_gap = 0.0  # 记录最大断崖位置差值
        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i]["score"]
            next_score = reranked_docs[i + 1]["score"]
            gap = current_score - next_score  # 差值
            if gap > rerank_gap_abs and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
        cut_off = max(cut_off, lower_bound)
        return reranked_docs[:cut_off]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }
    # mock_state = {
    #     "rewritten_query": "怎么测这块主板的短路问题？",
    #     "rrf_chunks": [
    #         {"chunk_id": "local_1", "title": "主板维修手册",
    #          "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"}
    #     ]
    # }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'] or [])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'] or [])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
