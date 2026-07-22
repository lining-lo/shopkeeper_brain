"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc:任务状态查询接口响应实体，封装LangGraph任务运行状态信息
"""
from typing import List
from pydantic import BaseModel, Field

class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    status: str = Field(..., description="任务状态")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")


