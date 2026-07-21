"""
  @Author:lining-lo
  @Time:2026/7/20
  @Desc:商品名称识别节点，从文档切片上下文调用LLM提取商品型号名称；
  使用BGE-M3生成稠密+稀疏混合向量存入Milvus，
  并将识别出的商品名回填至所有切片与流程状态，提供后续商品检索能力
"""
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name: str = "item_name_rec_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.校验参数，获取file_title,chunks,item_name_chunk_k,item_name_chunk_size
        # config对象中属性不合法： ValidationError
        # state对象中属性不合法： StateFieldError
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._valiedate_state(state)

        # 2.构建识别商品名称上下文，取前3个切片，限制字符数2500。
        item_name_recognition_context: str = self._prepare_context(chunks, item_name_chunk_k, item_name_chunk_size)

        # 3.调用LLM,获取客户端AIClients，识别商品名称。 提示词模板；  降级：file_title
        item_name = self._recognition_item_name(file_title, item_name_recognition_context)

        # 4.BGE-M3向量化
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5.存入milvus数据库
        self.insert_milvus(file_title, item_name, dense_vector, sparse_vector)

        # 6.回填 将item_name回填每一个chunk对象中。也同时回填state对象中。
        self._fill_item_name(item_name, state, chunks)

        # 7.数据备份
        self._backup_chunks(state, chunks)
        return state

    def _valiedate_state(self, state) -> Tuple[str, List[Dict[str, Any]], int, int]:
        """
        1.校验参数，获取file_title,chunks,item_name_chunk_k,item_name_chunk_size
        :param state: ImportGraphState
            必须提供：file_title,chunks
        :return: Dict[str,Any]
        """
        file_title = state['file_title']
        chunks = state['chunks']
        item_name_chunk_k = self.config.item_name_chunk_k  # 3
        item_name_chunk_size = self.config.item_name_chunk_size  # 25000

        if not file_title:
            self.logger.error(f"file_title不存在")
            raise StateFieldError(self.name, "file_title", expected_type=str)
        if not chunks and not isinstance(chunks, list):
            self.logger.error(f"chunks不存在")
            raise StateFieldError(self.name, "chunks", expected_type=list)

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            self.logger.error(f"item_name_chunk_k不存在或小于等于零")
            raise ValidationError(f"item_name_chunk_k不存在或小于等于零", self.name)
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            self.logger.error(f"item_name_chunk_size不存在或小于等于零")
            raise ValidationError(f"item_name_chunk_size不存在或小于等于零", self.name)

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_context(self, chunks, item_name_chunk_k, item_name_chunk_size) -> str:
        """
        通过3个切片的内容组装LLM提示词模板参数。
        :param chunks: List[Dict[str, Any]] 切片列表
        :param item_name_chunk_k: 提取上下文的切片数量限制
        :param item_name_chunk_size: LLM提示词模板参数长度的限制。
        :return: 用于提取商品名称的上下文参考内容。
        """
        # 取前三个chunk
        chunks = chunks[0:item_name_chunk_k]
        final_context = []
        total = 0
        for index, chunk in enumerate(chunks):

            if not chunk or not isinstance(chunk, dict):
                continue

            chunk_content = chunk.get("content", "")

            content = f"【切片-{index + 1}】\n" + chunk_content
            total += len(content)
            if total < item_name_chunk_size:
                final_context.append(content)

        return "\n\n".join(final_context)

    def _recognition_item_name(self, file_title, item_name_recognition_context) -> str:
        """
        3.调用LLM,获取客户端AIClients，识别商品名称。 提示词模板；  降级：file_title
        :param file_title: 文件名称
        :param item_name_recognition_context: 识别商品名称的上下文参考内容
        :return: 商品名称
        """

        # 获取LLM模型客户端对象
        llm_client = AIClients.get_llm_openai(False)  # 返回是文本，不是json

        # 用户提示词模板
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,
                                                            context=item_name_recognition_context)

        # 调用LLM模型
        llm_response = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        # 处理LLM模型返回结果失败，降级处理
        if not llm_response or llm_response.content == "UNKNOWN":
            self.logger.error(f"LLM模型调用失败,降级处理,用文件名称作为商品名称")
            return file_title

        # 获取响应内容
        response_dict = json.loads(llm_response.content.strip())
        item_name = response_dict.get("item_name")

        return item_name

    def _embedding_item_name(self, item_name) -> Tuple[List[float], Dict[int, float]]:
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"BGE-M3模型客户端初始化失败", e)
            return None, None  # 降级处理

        try:
            embeddings = bge_m3_client.encode_documents([item_name])

            dense_vector = embeddings["dense"][0].tolist()  # List[float], 长度 1024
            sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵
            start_idx = sparse_matrix.indptr[0]  # indptr = [0,5]
            end_idx = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()  # indices = [11,22,33,44,55]
            weights = sparse_matrix.data[start_idx:end_idx].tolist()  # indices = [111,222,333,444,555]
            sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]

            # 返回 稠密向量，稀疏向量
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE-M3向量化商品名称失败", e)
            return None, None

    def insert_milvus(self, file_title, item_name, dense_vector, sparse_vector):
        # 存储到Milvus   StorageClients获取客户端，创建集合，数据保存;  存储失败跳过，并记录日志。
        # 标量字段：pk, file_title ,item_name
        # 向量字段：dense_vector,sparse_vector
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端初始化失败", e)
            return

        try:
            item_name_collection = self.config.item_name_collection
            if not milvus_client.has_collection(item_name_collection):
                self.logger.info(f"{item_name_collection}不存在,正在创建...")
                self._create_item_name_collection(item_name_collection, milvus_client)
                self.logger.info(f"{item_name_collection}集合创建成功")
        except Exception as e:
            self.logger.error(f"{item_name_collection}集合创建失败", e)

        try:
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            self.logger.info(f"正在保存数据...={data}")
            milvus_client.insert(item_name_collection, [data])
            self.logger.info(f"保存数据成功...")
        except Exception as e:
            self.logger.error(f"保存商品名称数据失败", e)

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True,
                         max_length=100)  # 主键
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # 标量字段
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)  # 标量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)  # 稠密向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")  # 模糊近似查询   余弦相似度  只考虑方向，夹角，不考虑长度
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX",
                              metric_type="IP")  # IP内积  方向和长度都考虑 (查询更精准)    如果归一化后，与COSINE相似

        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name
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
            output_path = local_dir / "chunks_new.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")



if __name__ == '__main__':
    setup_logging()

    with open('D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单_new.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    with open('D:\\资料\\查重_简洁报告单\\auto\\chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    init = {
        'file_dir': 'D:\\资料',
        'file_title': '查重_简洁报告单',
        'import_file_path': 'D:\\查重_简洁报告单.pdf',
        'is_md_read_enabled': False,
        'is_pdf_read_enabled': True,
        'md_content': md_content,
        'md_path': 'D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md',
        'pdf_path': 'D:\\查重_简洁报告单.pdf',
        "chunks": chunks
    }

    state = create_default_state(**init)
    node = ItemNameRecognitionNode()

    print(node(state))

"""
文件备份位置：D:\\资料\\查重_简洁报告单\\auto\\chunks_new.json

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
            "item_name": "RS PRO RS-12 数字万用表"
        },
        {
            "title": "## 安全手册",
            "parent_title": "## 安全手册",
            "file_title": "查重_简洁报告单",
            "content": "## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。",
            "item_name": "RS PRO RS-12 数字万用表"
        },
        ...
    ]
}
"""