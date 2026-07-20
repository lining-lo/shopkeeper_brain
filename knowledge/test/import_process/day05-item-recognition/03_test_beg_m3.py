"""
  @Author:lining-lo
  @Time:2026/7/19
  @Desc:加载本地BGE-M3混合嵌入模型，单文本同时生成稠密向量、CSR稀疏矩阵
  并将稀疏矩阵转换为Milvus支持的字典格式稀疏向量，用于混合检索入库测试
"""
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 加载本地BGE-M3混合嵌入模型
bge_m3 = BGEM3EmbeddingFunction(
    model_name=r"D:\ai_models\modelscope_cache\models\BAAI\bge-m3",
    device="cpu",
    use_fp16=False,
)

# 2. 输入文本，同时生成稠密向量、稀疏矩阵
embeddings = bge_m3.encode_documents(["RS-12 数字万用表"])

# 3. 拆分稠密向量与稀疏矩阵
dense_vector = embeddings["dense"][0].tolist()  # 稠密向量 List[float] 长度1024
sparse_matrix = embeddings["sparse"]  # CSR格式稀疏矩阵

# 4. 解析CSR矩阵，转为Milvus可入库的字典稀疏向量
start_idx = sparse_matrix.indptr[0]
end_idx = sparse_matrix.indptr[1]
token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
weights = sparse_matrix.data[start_idx:end_idx].tolist()
sparse_vector = dict(zip(token_ids, weights))  # 格式 {token_id: 权重}

# 打印稀疏向量拆解信息
print(start_idx)
print(end_idx)
print(token_ids)
print(weights)
print(sparse_vector)
