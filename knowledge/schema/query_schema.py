"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc:向量知识库问答接口请求、响应Pydantic模型定义
"""
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(..., description="会话ID")
    is_stream: bool = Field(..., description="是否流式输出")


class QueryResponse(BaseModel):
    """查询响应模型"""
    message: str
    session_id: str
    answer: str
    task_id: str


class StreamSubmitResponse(BaseModel):
    """流式提交响应模型"""
    message: str
    session_id: str
    task_id: str
