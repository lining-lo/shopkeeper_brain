"""
  @Author:lining-lo
  @Time:2026/7/14
  @Desc:切片混合向量化节点
        分批读取带商品名的切片，批量生成稠密+稀疏向量回填切片；
        批量推理提速，异常自动填充空向量兜底，输出带双向量的切片供Milvus入库
"""
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    name: str = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "参数校验")
        # 1.校验数据
        chunks, embedding_batch_size = self._validate_get_inputs(state)
        self.logger.info(f"待向量化切片总数：{len(chunks)}，批次大小：{embedding_batch_size}")

        # 2.分批次向量化
        chunks_len = len(chunks)
        all_processed_chunks = []
        for i in range(0, chunks_len, embedding_batch_size):
            batch_chunks = chunks[i:i + embedding_batch_size]
            self.log_step("step2", f"处理批次：{i + 1} ~ {min(i + embedding_batch_size, chunks_len)} / {chunks_len}")
            # 执行向量化
            handled_batch = self._process_batch_chunks(batch_chunks, i, chunks_len)
            all_processed_chunks.extend(handled_batch)

        # 3.数据回填
        state["chunks"] = all_processed_chunks

        # 4.数据备份
        self._backup_chunks(state, chunks)

        return state

    def _validate_get_inputs(self, state) -> Tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks", None)
        # 配置带默认兜底
        embedding_batch_size = getattr(self.config, "embedding_batch_size", 8)

        # 修正校验逻辑：为空 OR 不是列表则报错
        if not chunks or not isinstance(chunks, list):
            self.logger.error(f"chunks不存在或格式非法，必须为非空列表")
            raise StateFieldError(self.name, "chunks", expected_type=list)

        if not embedding_batch_size or embedding_batch_size <= 0:
            self.logger.error(f"embedding_batch_size不存在或小于等于零")
            raise ValidationError(f"embedding_batch_size不存在或小于等于零", self.name)

        return chunks, embedding_batch_size

    def _process_batch_chunks(self, batch_chunks: List[Dict[str, Any]], start_idx: int, total_len: int) -> List[
        Dict[str, Any]]:
        # 批量收集文本，一次性推理，充分利用GPU并行
        text_list = []
        for chunk in batch_chunks:
            item_name = chunk.get("item_name", "")
            content = chunk.get("content", "")
            text = f"{item_name}\n{content}".strip()
            text_list.append(text)

        # 批量编码
        dense_list, sparse_dict_list = self._batch_embedding(text_list)

        # 向量回填对应切片
        handled_batch = []
        for idx, chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = dense_list[idx]
            chunk["sparse_vector"] = sparse_dict_list[idx]
            handled_batch.append(chunk)
        return handled_batch

    def _batch_embedding(self, text_list: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """整批文本统一向量化，返回稠密列表、稀疏字典列表"""
        # 初始化兜底空向量
        default_dense = [0.0] * 1024
        default_sparse = {}
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"BGE-M3模型客户端初始化失败: {str(e)}")
            return ([default_dense for _ in text_list], [default_sparse for _ in text_list])

        try:
            embeddings = bge_m3_client.encode_documents(text_list)
            dense_all = embeddings["dense"]
            sparse_csr = embeddings["sparse"]
            sparse_result = []

            # 批量解析CSR矩阵
            indptr = sparse_csr.indptr
            indices = sparse_csr.indices
            data = sparse_csr.data
            for i in range(len(text_list)):
                s = indptr[i]
                e = indptr[i + 1]
                token_ids = indices[s:e].tolist()
                weights = data[s:e].tolist()
                sparse_result.append(dict(zip(token_ids, weights)))

            dense_list = [vec.tolist() for vec in dense_all]
            return dense_list, sparse_result

        except Exception as e:
            self.logger.error(f"BGE-M3批量向量化失败: {str(e)}")
            return ([default_dense for _ in text_list], [default_sparse for _ in text_list])

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        md_path_str = state.get("md_path", "")
        # 先判断md_path是否为空，为空直接跳过
        if not md_path_str:
            self.logger.debug("未设置 md_path，跳过备份")
            return

        md_path = Path(md_path_str)
        local_dir = md_path.parent  # 取auto目录

        try:
            # Path自带创建目录
            local_dir.mkdir(exist_ok=True)
            # Path拼接路径
            output_path = local_dir / "chunks_new_embedding.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    with open('D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单_new.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    with open('D:\\资料\\查重_简洁报告单\\auto\\chunks_new.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    init = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": "D:\\查重_简洁报告单.pdf",
        "file_dir": "D:\\资料",
        "pdf_path": "D:\\查重_简洁报告单.pdf",
        "md_path": "D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md",
        "file_title": "查重_简洁报告单",
        "item_name": "RS PRO RS-12 数字万用表",
        "md_content": md_content,
        "chunks": chunks
    }

    state = create_default_state(**init)
    node = BgeEmbeddingChunksNode()

    print(node(state))

"""
文件备份位置：D:\\资料\\查重_简洁报告单\\auto\\chunks_new_embedding.json

打印结果：
{
    "is_pdf_read_enabled": true,
    "is_md_read_enabled": false,
    "import_file_path": "D:\\查重_简洁报告单.pdf",
    "file_dir": "D:\\资料",
    "pdf_path": "D:\\查重_简洁报告单.pdf",
    "md_path": "D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md",
    "file_title": "查重_简洁报告单",
    "item_name": "RS PRO RS-12 数字万用表",
    "md_content": "\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。",
    "chunks": [
        {
            "title": "查重_简洁报告单",
            "parent_title": "查重_简洁报告单",
            "file_title": "查重_简洁报告单",
            "content": "查重_简洁报告单\n\n![RS PRO品牌标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)\n使用说明书\nRS-12\n编号: 123-1939\n数字万用表",
            "item_name": "RS PRO RS-12 数字万用表",
            "dense_vector": [-0.04930327460169792, 0.017536623403429985,-0.08282745629549026,0.00011999431444564834...],
            "sparse_vector": {"5": 0.04015567526221275, "6": 0.09334199130535126,"7": 0.022246405482292175,"9": 0.05974582955241203...}
        },
        {
            "title": "## 安全手册",
            "parent_title": "## 安全手册",
            "file_title": "查重_简洁报告单",
            "content": "## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。",
            "item_name": "RS PRO RS-12 数字万用表",
            "dense_vector": [-0.04930327460169792, 0.017536623403429985,-0.08282745629549026,0.00011999431444564834...],
            "sparse_vector": {"5": 0.04015567526221275, "6": 0.09334199130535126,"7": 0.022246405482292175,"9": 0.05974582955241203...}
        },
        ...
    ]
}
"""
