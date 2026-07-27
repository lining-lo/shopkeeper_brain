"""
  @Author:lining-lo
  @Time:2026/7/27
  @Desc:从ModelScope下载bge-reranker-large重排序模型到本地
"""
from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large",
                              local_dir="D:\\ai_models\\modelscope_cache\\models\\BAAI\\bge-reranker-large")

print(local_dir)
