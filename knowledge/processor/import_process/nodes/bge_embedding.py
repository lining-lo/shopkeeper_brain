from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class BgeEmbeddingChunksNode(BaseNode):
    name: str = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        return state