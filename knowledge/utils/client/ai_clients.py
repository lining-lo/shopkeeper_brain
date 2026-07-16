"""
  @Author:lining-lo
  @Time:2026/7/15
  @Desc:
    AI模型客户端统一管理器；
    包含DashScope兼容OpenAI(VLM)、LangChain ChatOpenAI文本大模型、
    BGE-M3向量化模型、BGE重排序模型；
    继承BaseClientManager实现线程安全懒加载单例，内置环境变量校验与统一异常处理；
    LLM区分普通文本模式与JSON结构化输出两套独立实例。
"""
import threading
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker
from knowledge.utils.client.base import BaseClientManager, logger
import logging
from typing import Optional


class AIClients(BaseClientManager):
    """
    AI 模型类客户端： OpenAI(VLM)
    """
    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    @classmethod
    def get_openai(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_openai)

    @classmethod
    def _create_openai(cls) -> OpenAI:
        try:
            api_key = cls._require_env("DASHSCOPE_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"OpenAI API 创建成功:{base_url}")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI API 创建失败:{e}")
            raise ConnectionError(f"OpenAI连接失败:{e}") from e

    """
    LLM客户端：
    """
    _openai_llm_text_client: Optional[ChatOpenAI] = None
    _openai_llm_text_lock = threading.Lock()

    _openai_llm_json_client: Optional[ChatOpenAI] = None
    _openai_llm_json_lock = threading.Lock()

    @classmethod
    def get_llm_openai(cls, response_format: bool = True) -> ChatOpenAI:
        if response_format:
            # _create_llm_openai(response_format) 加了 () 立即执行，应改为 lambda: 延迟执行
            return cls._get_or_create("_openai_llm_json_client", cls._openai_llm_json_lock,
                                      lambda: cls._create_llm_openai(response_format))
        else:
            return cls._get_or_create("_openai_llm_text_client", cls._openai_llm_text_lock,
                                      lambda: cls._create_llm_openai(response_format))

    @classmethod
    def _create_llm_openai(cls, response_format) -> ChatOpenAI:
        try:
            api_key = cls._require_env("DASHSCOPE_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0,
                model_kwargs=model_kwargs
            )
            logger.info(f"ChatOpenAI LLM 客户端初始化成功")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI LLM 客户端初始化失败:{e}")
            raise ConnectionError(f"ChatOpenAI LLM 连接失败:{e}") from e

    """
    BGE-M3客户端：
    """
    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            fp16 = cls._require_env("BGE_FP16")

            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16,
            )
            logger.info(f"bge_m3客户端初始化成功")
            return bge_m3_ef
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3客户端初始化失败:{e}")
            raise ConnectionError(f"bge_m3客户端创建失败:{e}") from e

    """
    BGE-M3重排序模型客户端：
    """
    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    @classmethod
    def get_bge_m3_rerank_client(cls) -> FlagReranker:
        return cls._get_or_create("_bge_m3_rerank_client", cls._bge_m3_rerank_lock, cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls) -> FlagReranker:
        try:
            model_name_or_path = cls._require_env("BGE_RERANKER_LARGE")
            device = cls._require_env("BGE_DEVICE")
            fp16_str = cls._require_env("BGE_FP16")
            fp16 = fp16_str.lower() in ("true", "1")

            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                # model_name_or_path="D:\\ai_models\\modelscope_cache\\models\\BAAI\\BAAI\\bge-reranker-large",
                device=device,  # GPU 加速
                use_fp16=fp16  # 半精度推理
            )
            logger.info(f"bge_m3_rerank客户端初始化成功")
            return reranker
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3_rerank客户端初始化失败:{e}")
            raise ConnectionError(f"bge_m3_rerank客户端创建失败:{e}") from e


if __name__ == "__main__":
    print(AIClients.get_bge_m3_rerank_client())
