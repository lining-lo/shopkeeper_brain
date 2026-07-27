"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:RRF融合节点,融合向量检索、HyDE假设文档检索多路召回结果，
        通过倒数排名公式计算综合得分，对重复命中文档加权提权，
        抹平不同检索策略的分数分布差异，实现精准、稳定的多路结果重排序，
        最终输出高质量融合切片列表，供给后续大模型问答、重排节点使用
"""
from typing import List, Dict, Any, Tuple
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        RRF 倒数融合排序
        :param state:
                embedding_chunks  向量检索结果
                    [
                        {
                          "chunk_id": 467775438085177895,
                          "distance": 0.7380014657974243,
                          "entity": {
                            "chunk_id": 467775438085177895,
                            "item_name": "RS-12 数字万用表",
                            "content": "## 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n1. 将功能转盘置于最高电阻Ω位置.\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n![万用表电阻测量操作示意图（表笔连接被测电阻，显示屏显示读数）](http://192.168.6.160:9000/knowledge-base-files/万用表RS-12的使用/dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)",
                            "title": "## 电阻测量"
                          }
                        }
                    ]
                hyde_embedding_chunks HyDE（假设性文档）检索结果
                    [
                        {
                            'chunk_id': 467775438085177891,
                            'distance': 0.8033741116523743,
                            'entity':
                                {
                                'chunk_id': 467775438085177891,
                                'title': '## 直流电压测量',
                                'item_name': 'RS-12 数字万用表',
                                'content': '## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n1. 将功能转盘置于V DC的位置。\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。'
                                }
                        },
                    ]
        :return: rrf_chunks  RRF融合后的切片
        """
        # 1.获取各路检索结果
        embedding_chunks: list = state.get("embedding_chunks", [])
        hyde_embedding_chunks: list = state.get("hyde_embedding_chunks", [])

        # 2.格式规整化  只要每路中的entity部分
        embedding_chunks_entities: List[Dict[str, Any]] = self._normalize_input(embedding_chunks)
        hyde_embedding_chunks_entities: List[Dict[str, Any]] = self._normalize_input(hyde_embedding_chunks)

        # 3.设置各路权重
        rrf_inputs = [(embedding_chunks_entities, 1.0), (hyde_embedding_chunks_entities, 1.0)]

        # 4.RRF 计算
        # self.config.rrf_k   平滑参数，默认60
        # self.config.rrf_max_results   最大结果数，默认10   排序后截取前10个文档
        rrf_chunks = self._rrf(rrf_inputs, self.config.rrf_k, self.config.rrf_max_results)

        # 6.更新状态返回
        state["rrf_chunks"] = [entity for entity, _ in rrf_chunks]  # 只要文档，不要分

        return state

    def _normalize_input(self, input_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        各路列表数据规整化。只保留entity部分
        :param embedding_chunks:
        :return:
        """
        if not input_chunks:
            return []
        normalized_output = []
        for input_chunk in input_chunks:
            entity = input_chunk.get("entity", {})
            if not entity:
                continue
            normalized_output.append(entity)
        return normalized_output

    def _rrf(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], rrf_k: int = 60, rrf_max_results: int = 5) -> \
            List[Tuple[Dict[str, Any], float]]:
        """
        倒数融合排序RRF
        :param rrf_inputs:  待排序的多路数据，每路设置好权重
            [
                (List[Dict[str,Any]], float),
                (List[Dict[str,Any]], float),
            ]
        :param rrf_k: 平滑参数 默认60
        :return: 排好序的列表
            [
                (entity, score),(entity, score)
            ]
        """
        rrf_scores = {}  # 记录每个chunk_id的score
        rrf_doc = {}  # 记录每个文档的数据
        for entity_list, weight in rrf_inputs:
            for index, entity in enumerate(entity_list, 1):  # 索引序号从1开始。而不是从0开始了。
                chunk_id = entity.get("chunk_id")
                if not chunk_id:
                    continue

                # 公式      score =  weight/ (k + ranker_i)
                score = weight / (rrf_k + index)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
                rrf_doc.setdefault(chunk_id, entity)  # 只在第一次出现会存储。多路都有你，也只会记录一次。

        # 按score排序
        data_result = [(rrf_doc.get(chunk_id), score) for chunk_id, score in rrf_scores.items() or []]
        data_result_sorted = sorted(data_result, key=lambda x: x[1], reverse=True)
        print(data_result_sorted)
        # 取前几名
        rrf_chunks = data_result_sorted[:rrf_max_results] if rrf_max_results else data_result_sorted
        return rrf_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")
