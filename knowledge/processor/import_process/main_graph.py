"""
导入流程主图

使用LangGraph构建文档导入工作流
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
    # ** state 是 Python 的解包操作，将字典展开为关键字参数
    # 相当于：create_default_state(import_file_path=..., file_dir=...)
    # "is_pdf_read_enabled": False,  # 默认值
    # "is_md_read_enabled": False,  # 默认值
    # "md_path": None,  # 默认值
    init_state = create_default_state(**state)
    final_state = None
    # 启动  LangGraph 工作流并流式执行（streaming execution）。
    # kb_import_graph_app: 已编译的 LangGraph 应用对象
    # .stream(state): 以流式方式执行工作流，每完成一个节点就 yield 一次结果
    # event: 每次迭代返回的事件数据，包含当前节点的名称和更新后的状态
    # 优势：可以实时看到每个节点的执行进度，而不是等待全部完成
    for event in kb_import_graph_app.stream(state):
        for node_name, state in event.items():  # 注意：这里使用了嵌套循环，因为一个事件可能包含多个节点的更新
            final_state = state  # 循环结束后，final_state 保存的是最后一个节点执行后的状态

    return final_state  # 返回工作流执行完成后的最终状态。


if __name__ == "__main__":
    setup_logging()

    import_file_path = r"D:\workspace\python\PythonProject\shopkeeper_brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用.pdf"
    file_dir = r"D:\workspace\python\PythonProject\shopkeeper_brain\knowledge\processor\import_process\temp_dir"

    final_state = run_import_graph(import_file_path, file_dir)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))
    print("-" * 50)
    kb_import_graph_app.get_graph().print_ascii()
