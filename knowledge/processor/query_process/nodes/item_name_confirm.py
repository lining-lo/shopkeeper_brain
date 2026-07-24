"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:
"""
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class ItemNameConfirmNode(BaseNode):
    name: str = "item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        return state
