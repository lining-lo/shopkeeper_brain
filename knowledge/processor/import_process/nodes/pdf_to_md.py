from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name: str = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        return state


