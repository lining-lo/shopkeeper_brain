"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:
"""
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RerankNode(BaseNode):
    name: str = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        return state
