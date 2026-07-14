"""

导入流程状态类型定义



定义完整的状态结构和辅助函数

"""

from typing import TypedDict, List

import copy


class ImportGraphState(TypedDict, total=False):
    """

    导入流程图状态



    包含整个导入流程中传递的所有数据

    """

    # ==================== 任务标识 ====================

    task_id: str  # 任务 ID，用于任务追踪(web交互的时候用到，实时看到节点的处理日志)

    # ==================== 控制标志 ====================

    is_md_read_enabled: bool  # 是否启用 MD 读取

    is_pdf_read_enabled: bool  # 是否启用 PDF 读取   pdf需要转md

    # ==================== 路径信息 ====================

    # D:\workspace\python\PythonProject\shopkeeper_brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用.pdf
    import_file_path: str  # 导入文件路径,pdf所在路径，文件及路径的全路径。

    # D:\workspace\python\PythonProject\shopkeeper_brain\knowledge\processor\import_process\temp_dir\
    file_dir: str  # 导入(出)文件目录，pdf转换为md存放的目录，父目录级别

    # D:\workspace\python\PythonProject\shopkeeper_brain\knowledge\processor\import_process\temp_dir\
    pdf_path: str  # PDF 文件路径，不含文件名称的pdf目录

    # D:\workspace\python\PythonProject\shopkeeper_brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用_pipeline\auto\万用表RS-12的使用.md
    md_path: str  # 转换后Markdown 文件路径，具体的文件及目录全路径。

    # ==================== 文件信息 ====================

    file_title: str  # 文件标题（不含扩展名）

    item_name: str  # 识别出的商品/产品名称(方便程序员用)

    # ==================== 处理中间数据 ====================

    md_content: str  # Markdown 文档内容

    chunks: List  # 文档切片列表

    # ==================== 默认状态 ====================


GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "item_name": "",

}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
