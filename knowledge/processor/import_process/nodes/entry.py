from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name: str = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        return state