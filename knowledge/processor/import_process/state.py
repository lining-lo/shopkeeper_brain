"""
  @Author:lining-lo
  @Time:2026/7/13
  @Desc:导入流程状态类型定义
"""
from typing import TypedDict, List
import copy


class ImportGraphState(TypedDict, total=False):
    """
      文档导入流程全局状态
      工作流节点之间传递的全部共享数据
    """
    task_id: str  # 任务唯一ID，用于日志追踪、进度展示
    is_md_read_enabled: bool  # 标识：源文件是否为 Markdown
    is_pdf_read_enabled: bool  # 标识：源文件是否为 PDF（PDF 需要先转 MD）

    import_file_path: str  # 原始上传文件完整路径
    file_dir: str  # 原始文件所在父目录

    pdf_path: str  # PDF 文件完整路径
    md_path: str  # Markdown 文件完整路径（原始MD / PDF转换后的MD）
    file_title: str  # 文件名称（剔除后缀）

    item_name: str  # 从文档识别得到的产品/商品名称
    md_content: str  # Markdown 完整文本内容
    chunks: List  # 文档文本切片结果（用于后续向量化入库）


GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,

    "import_file_path": "",
    "file_dir": "",

    "pdf_path": "",
    "md_path": "",
    "file_title": "",

    "item_name": "",
    "md_content": "",
    "chunks": [],

}


def create_default_state(**overrides) -> ImportGraphState:
    """
      创建导入流程默认状态，支持自定义字段覆盖
      Args:
        **overrides: 自定义覆盖的状态字段
      Returns:
        ImportGraphState：新的流程状态字典
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
