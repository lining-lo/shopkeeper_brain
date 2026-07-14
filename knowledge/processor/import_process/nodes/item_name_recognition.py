from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class ItemNameRecognitionNode(BaseNode):
    name: str = "item_name_rec_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        return state