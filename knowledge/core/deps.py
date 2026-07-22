"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc:专业创建对象，并进行缓存管理。可以实现对象共享。
"""
from functools import cache, lru_cache
from knowledge.service.import_file_service import ImportFileService


@cache  # 缓存满了，不清理。  OOM 问题
# @lru_cache # 缓存满了，可以根据lru(最近最少使用)策略进行缓存清理
def get_import_file_service() -> ImportFileService:
    return ImportFileService()
