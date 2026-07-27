"""
  @Author:lining-lo
  @Time:2026/7/27
  @Desc:交叉编码器重排模型基本使用
"""
from FlagEmbedding import FlagReranker

# 使用交叉编码器进行相关性得分。
# reranker = FlagReranker(
#     #model_name_or_path="BAAI/bge-reranker-large",
#     model_name_or_path="D:\\ai_models\\modelscope_cache\\models\\BAAI\\bge-reranker-large",
#     device="cuda:0",      # GPU 加速
#     use_fp16=True       # 半精度推理
# )
# [7.8984375, -9.484375]


reranker = FlagReranker(
    model_name_or_path="D:\\ai_models\\modelscope_cache\\models\\BAAI\\bge-reranker-large",
    device="cpu",
    use_fp16=False
)
# [7.8957743644714355, -9.480398178100586]

# 计算相关性得分
#  给Reranker模型计算相关性得分 ：[CLS]什么是万用表？[SEP]万用表是一种测量电压、电流、电阻的仪器[SEP]
pairs = [
    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
    ["什么是万用表？", "今天天气很好"]
]
scores = reranker.compute_score(pairs)
print(scores)
# 输出: [0.9234, 0.0156]  高分 = 高相关
