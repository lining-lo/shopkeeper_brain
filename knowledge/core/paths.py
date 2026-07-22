"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc: 项目路径常量定义，统一管理知识库模块下各类资源目录，避免硬编码绝对路径
"""
import os

# 知识库模块根目录绝对路径
KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 临时文件存储根目录（切块、导出备份、中间缓存文件存放）
LOCAL_BSSE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端静态页面目录，存放html页面
LOCAL_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")


def get_local_base_dir() -> str:
    """获取临时文件根目录绝对路径"""
    return LOCAL_BSSE_DIR


def get_local_page_dir() -> str:
    """获取前端页面资源目录绝对路径"""
    return LOCAL_PAGE_DIR
