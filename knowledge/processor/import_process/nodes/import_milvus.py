"""
  @Author:lining-lo
  @Time:2026/7/21
  @Desc:Milvus向量入库节点，基于门面+建造者模式实现知识库切片数据入库能力
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple
from pymilvus import MilvusClient, CollectionSchema, DataType
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, StateFieldError, MilvusError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.utils.client.storage_clients import StorageClients, logger


# ================================================================== #
#                        标量字段规范                                   #
# ================================================================== #
@dataclass(frozen=True)
class ScalarFieldSpec:
    """标量字段规范"""
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 预定义的标量字段（复用）
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


# ================================================================== #
#                        建造者：Schema 构建                           #
# ================================================================== #
class _MilvusSchemaBuilder:
    """职责：专门负责构建Milvus集合Schema"""

    @staticmethod
    def build(dim: int) -> CollectionSchema:
        logger.info("开始构建Milvus集合Schema...")
        schema = MilvusClient.create_schema(enable_dynamic_field=True)

        # 主键
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        # 稠密向量
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        # 稀疏向量
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 遍历预定义标量字段
        for field_spec in _SCALAR_FIELDS:
            kwargs = {
                "field_name": field_spec.field_name,
                "datatype": field_spec.datatype
            }
            if field_spec.max_length is not None:
                kwargs["max_length"] = field_spec.max_length
            schema.add_field(**kwargs)

        return schema


# ================================================================== #
#                        建造者：索引构建                               #
# ================================================================== #
class _MilvusIndexBuilder:
    """职责：负责构建Milvus集合索引参数"""

    @staticmethod
    def build():
        index_params = MilvusClient.prepare_index_params()

        # 稠密向量索引：余弦相似度
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 稀疏向量索引：内积IP
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        logger.info("索引参数构建完成")
        return index_params


# ================================================================== #
#                        插入器：数据插入与回填                          #
# ================================================================== #
class _MilvusInserter:
    """职责：将数据插入到Milvus 以及 回填chunk_id"""

    def __init__(self, client: MilvusClient, collection_name: str, node_name: str):
        self._client = client
        self._collection_name = collection_name
        self._node_name = node_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始插入{len(chunks)}块到Milvus，集合：{self._collection_name}")
        try:
            inserted_result = self._client.insert(
                collection_name=self._collection_name,
                data=chunks
            )
        except Exception as e:
            logger.error("Milvus数据插入发生异常", exc_info=e)
            raise MilvusError("Milvus数据插入失败", self._node_name) from e

        inserted_count = inserted_result.get('insert_count', 0)
        ids = inserted_result.get('ids', [])

        # 校验ID数量匹配
        if len(chunks) != len(ids):
            raise MilvusError(
                f"插入返回ID数量与chunk数量不匹配，输入:{len(chunks)}, 返回ids:{len(ids)}",
                self._node_name
            )

        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}条记录，并且回填chunk_id到chunk中")
        return chunks

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, cid in zip(chunks, ids):
            chunk["chunk_id"] = cid


# ================================================================== #
#                        门面：主节点                                   #
# ================================================================== #
class ImportMilvusNode(BaseNode):
    name: str = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.数据校验
        validated_chunks, collection_name, dim = self._validate_get_inputs(state)

        # 2.获取milvus客户端并确保集合存在
        milvus_client = self._ensure_has_collection(collection_name, dim)

        # 3.插入数据到milvus
        milvus_inserter = _MilvusInserter(milvus_client, collection_name, self.name)
        final_chunks = milvus_inserter.insert(validated_chunks)

        # 4.回填数据到state
        state["chunks"] = final_chunks

        # 5.数据备份
        self._backup_chunks(state, final_chunks)

        return state

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str, int]:
        self.logger.info("开始校验参数")
        collection_name = self.config.chunks_collection
        chunks = state.get("chunks")

        if not collection_name:
            self.logger.error("config.chunks_collection 配置为空")
            raise ValidationError("config.chunks_collection不存在", self.name)

        if not isinstance(chunks, list):
            self.logger.error(f"chunks 类型非法，实际类型: {type(chunks)}")
            raise StateFieldError(self.name, field_name="chunks", expected_type=list)

        validated_chunks = []
        expect_dim: Optional[int] = None
        for idx, chunk in enumerate(chunks):
            sparse = chunk.get("sparse_vector")
            dense = chunk.get("dense_vector")

            if not sparse or not dense:
                if not sparse and not dense:
                    self.logger.error(f"第{idx + 1}个chunk：稠密、稀疏向量全部缺失，跳过该切片")
                elif not sparse:
                    self.logger.error(f"第{idx + 1}个chunk：缺失sparse_vector，跳过该切片")
                else:
                    self.logger.error(f"第{idx + 1}个chunk：缺失dense_vector，跳过该切片")
                continue

            # 校验稠密向量维度统一
            current_dim = len(dense)
            if expect_dim is None:
                expect_dim = current_dim
            elif current_dim != expect_dim:
                self.logger.error(f"第{idx + 1}个chunk稠密向量维度不一致，预期{expect_dim}，当前{current_dim}，跳过")
                continue

            validated_chunks.append(chunk)

        if not validated_chunks:
            raise ValidationError("所有chunk均无有效混合向量，无数据可入库", self.name)

        self.logger.info(f"校验完成，有效切块数量：{len(validated_chunks)}，稠密向量维度：{expect_dim}")
        return validated_chunks, collection_name, expect_dim

    def _ensure_has_collection(self, collection_name: str, dim: int) -> MilvusClient:
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error("Milvus 客户端初始化失败", exc_info=e)
            raise MilvusError("Milvus 客户端初始化失败", self.name) from e

        try:
            if not milvus_client.has_collection(collection_name):
                self.logger.info(f"{collection_name}不存在,正在创建...")
                schema = _MilvusSchemaBuilder.build(dim)
                index_params = _MilvusIndexBuilder.build()
                milvus_client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params
                )
                self.logger.info(f"{collection_name}集合创建成功")
            # 加载集合
            milvus_client.load_collection(collection_name)
        except Exception as e:
            self.logger.error(f"{collection_name}集合创建/加载失败", exc_info=e)
            raise MilvusError(f"集合 {collection_name} 创建或加载异常", self.name) from e

        return milvus_client

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
            output_path = local_dir / "chunks_new_milvus.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == "__main__":
    setup_logging()

    with open('D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单_new.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    with open('D:\\资料\\查重_简洁报告单\\auto\\chunks_new_embedding.json', 'r', encoding='utf-8') as f:
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
    node = ImportMilvusNode()

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
            "chunk_id": 467833860584112384
            "title": "查重_简洁报告单",
            "parent_title": "查重_简洁报告单",
            "file_title": "查重_简洁报告单",
            "content": "查重_简洁报告单\n\n![RS PRO品牌标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)\n使用说明书\nRS-12\n编号: 123-1939\n数字万用表",
            "item_name": "RS PRO RS-12 数字万用表",
            "dense_vector": [-0.04930327460169792, 0.017536623403429985,-0.08282745629549026,0.00011999431444564834...],
            "sparse_vector": {"5": 0.04015567526221275, "6": 0.09334199130535126,"7": 0.022246405482292175,"9": 0.05974582955241203...}
        },
        {
            "chunk_id": 467833860584112385
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
