"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:向量检索节点
        使用BGE-M3生成稠密稀疏混合向量，按商品过滤Milvus知识库，
        多路向量加权融合检索，返回匹配文档，供给后续RAG使用
"""
import json
from typing import Dict, Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import execute_hybrid_search_query, create_hybrid_search_requests, item_names_filter


class VectorSearchNode(BaseNode):
    name: str = "search_embedding"

    def process(self, state: QueryGraphState) -> Dict:
        """
        :param rewritten_query:重写查询  ->  "万用表如何测量电压"
        :param item_names: 商品名称列表  ->  ["RS-12 数字万用表"]
        :return: embedding_chunks:向量检索结果  ->
        [
            {
                "chunk_id": 467775438085177895,
                "distance": 0.7380014657974243,
                "entity": {
                    "chunk_id": 467775438085177895,
                    "item_name": "RS-12 数字万用表",
                    "content": "## 电阻测量\n\n警告：为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。",
                    "title": "## 电阻测量"
                }
            },
            {
                "chunk_id": 467775438085177884,
                "distance": 0.7050827741622925,
                "entity": {
                    "chunk_id": 467775438085177884,
                    "item_name": "RS-12 数字万用表",
                    "content": "## 规格- 3\n\n- 【电阻】(对应功能)：量程为200kΩ，分辨率为0.1kΩ",
                    "title": "## 规格- 3"
                }
            }
        ]
        """
        # 1.校验参数
        validated_rewritten_query, validated_item_names = self._validate_input(state)

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
            embedding_hybrid_result = generate_bge_m3_hybrid_vectors(embedding_model, [validated_rewritten_query])

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
                "embedding_chunks": hybrid_result[0]
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


if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
"""
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
{
  "chunk_id": 467775438085177884,
  "distance": 0.7050827741622925,
  "entity": {
    "chunk_id": 467775438085177884,
    "item_name": "RS-12 数字万用表",
    "content": "## 规格- 3\n\n- 【电阻】(对应功能)：量程为200kΩ，分辨率为0.1kΩ，精确度为± (0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为2000kΩ，分辨率为1kΩ，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为9V，分辨率为10mV，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为1.5V，分辨率为1mV，精确度为± (1.0% reading + 2 digits)。",
    "title": "## 规格- 3"
  }
}
{
  "chunk_id": 467775438085177892,
  "distance": 0.42116472125053406,
  "entity": {
    "chunk_id": 467775438085177892,
    "item_name": "RS-12 数字万用表",
    "content": "## 交流电压测量\n\n警告：谨防触电。\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n![交流电压测量时表笔正确连接被测电路的示意图](http://192.168.6.160:9000/knowledge-base-files/万用表RS-12的使用/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg)\n1. 将功能转盘置于V AC的位置。\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n3. 将表笔尖端接触被测物。\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。",
    "title": "## 交流电压测量"
  }
}
{
  "chunk_id": 467775438085177891,
  "distance": 0.41844987869262695,
  "entity": {
    "chunk_id": 467775438085177891,
    "item_name": "RS-12 数字万用表",
    "content": "## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n1. 将功能转盘置于V DC的位置。\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。",
    "title": "## 直流电压测量"
  }
}
{
  "chunk_id": 467775438085177893,
  "distance": 0.4167860746383667,
  "entity": {
    "chunk_id": 467775438085177893,
    "item_name": "RS-12 数字万用表",
    "content": "## 直流电流测量- 1\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n![直流电流测量接线示意图（10A档位）](http://192.168.6.160:9000/knowledge-base-files/万用表RS-12的使用/8eb1e59b1e3f5e200f6d947da47dcd767fe061b91e72b1ba5325869677dcdad2.jpg)\n1. 将黑色表笔插入负极COM端口。\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n6. 接通电源。\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。",
    "title": "## 直流电流测量- 1"
  }
}
"""
