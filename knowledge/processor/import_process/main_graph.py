"""
  @Author:lining-lo
  @Time:2026/7/13
  @Desc:使用LangGraph构建文档导入工作流
"""
import json
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def import_router(state: ImportGraphState) -> str:
    """
    入口节点后的路由逻辑
    根据文件类型决定走PDF转换分支还是直接处理MD分支
    :param state: 当前图状态
    :return: 下一个节点名称
    """
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return END


def create_import_graph() -> CompiledStateGraph:
    """
    创建导入流程图
    :return: 编译后的StateGraph实例
    流程结构：
        entry_node
              │
              ├── (PDF) ──> pdf_to_md_node ──┐
              │                              │
              └── (MD) ─────────────────────>├──> md_img_node
                                              │
                                              v
                                      document_split_node
                                              │
                                              v
                                      item_name_rec_node
                                              │
                                              v
                                        bge_embedding_node
                                              │
                                              v
                                        import_milvus_node
                                              │
                                              v
                                             END
    """

    # 1.定义状态图
    graph_pipeline = StateGraph(ImportGraphState)

    # 2.定义节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode()
    }

    # 2.1 添加入口节点
    graph_pipeline.set_entry_point("entry_node")

    # 2.2 添加所有节点到图中
    for key, value in nodes.items():
        graph_pipeline.add_node(key, value)

    # 3. 定义边（条件边，顺序边）

    # 3.1 条件边： 入口节点后根据文件类型路由
    # 定义路由映射表（routing map），将路由函数的返回值映射到实际的下一个节点。
    # 键（Key）：import_router 函数可能返回的值
    # 值（Value）：实际要跳转到的目标节点名称
    graph_pipeline.add_conditional_edges(
        "entry_node", import_router,
        {
            "md_img_node": "md_img_node",
            "pdf_to_md_node": "pdf_to_md_node",
            END: END
        }
    )

    # 3.2 顺序边：
    graph_pipeline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pipeline.add_edge("md_img_node", "document_split_node")
    graph_pipeline.add_edge("document_split_node", "item_name_rec_node")
    graph_pipeline.add_edge("item_name_rec_node", "bge_embedding_node")
    graph_pipeline.add_edge("bge_embedding_node", "import_milvus_node")
    graph_pipeline.add_edge("import_milvus_node", END)

    # 4.编译图
    return graph_pipeline.compile()


kb_import_graph_app = create_import_graph()  # CompiledStateGraph


def run_import_graph(import_file_path: str, file_dir: str) -> dict:
    """
    便捷函数：运行导入流程
    :param import_file_path:
    :param file_dir:
    :return: 最终状态字典
    """
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    init_state = create_default_state(**state)
    final_state = None

    for event in kb_import_graph_app.stream(init_state):
        for node_name, state in event.items():
            final_state = state

    return final_state


if __name__ == "__main__":
    setup_logging()

    import_file_path = r"D:\查重_简洁报告单.pdf"
    file_dir = r"D:\资料"

    final_state = run_import_graph(import_file_path, file_dir)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))
    print("-" * 50)
    kb_import_graph_app.get_graph().print_ascii()
