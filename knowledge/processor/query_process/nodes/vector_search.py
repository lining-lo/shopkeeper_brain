"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:
"""
from typing import Dict

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class VectorSearchNode(BaseNode):
    name: str = "search_embedding"

    def process(self, state: QueryGraphState) -> Dict:
        return {}
