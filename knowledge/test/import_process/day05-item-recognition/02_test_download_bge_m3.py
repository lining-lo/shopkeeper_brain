"""
  @Author:lining-lo
  @Time:2026/7/19
  @Desc:下载bge-m3模型
"""
from modelscope import snapshot_download

# 从ModelScope下载BGE-M3混合嵌入模型至指定本地目录
local_dir = snapshot_download(
    model_id="BAAI/bge-m3",
    local_dir=r"D:\ai_models\modelscope_cache\models\BAAI\bge-m3"
)
# 打印模型本地存放路径
print(local_dir)
