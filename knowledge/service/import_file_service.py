"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc:文件上传业务层
        处理文件上传 双写（上传文件保存本地，上传文件保存 minio）
        开启 langgraph 流程，完成文件导入 milvus [7 个步骤]
"""
import datetime
import logging
import os
import shutil
import uuid
from typing import Tuple
from dotenv import load_dotenv

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task

# 加载配置文件
load_dotenv()
# 获取日志实例
logger = logging.getLogger(__name__)


class ImportFileService:
    def process_upload_file(self, file) -> Tuple[str, str, str]:
        """处理文件上传 双写（上传文件保存本地，上传文件保存 minio）"""
        # 1.生成任务id
        task_id = self._get_task_id()

        # 2.设置任务状态 -> 开始
        add_running_task(task_id, "upload_file")

        # 3.文件写入到本地
        file_dir = self._get_path_with_date()
        import_file_path = self._save_upload_file_to_local(file, file_dir)

        # 4.文件上传至minio
        self._save_upload_file_to_minio(import_file_path, file.filename)

        # 5.设置任务状态 -> 结束
        add_done_task(task_id, "upload_file")

        # 6.返回数据
        return task_id, file_dir, import_file_path

    def run_langgraph_import(self, task_id, file_dir, import_file_path):
        """开启 langgraph 流程，完成文件导入 milvus [7 个步骤]"""
        pass

    def _get_task_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _save_upload_file_to_local(self, file, file_dir) -> str:
        """将上传的文件保存到本地"""
        import_file_path = os.path.join(file_dir, file.filename)

        try:
            os.makedirs(file_dir, exist_ok=True)
            with open(import_file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logging.error(f"上传文件到本地失败：{e}")
            raise FileProcessingError(f"上传文件到本地失败：{e}")

        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path, filename):
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            logging.error(f"创建minio客户端失败：{e}")
            raise MinioError(f"创建minio客户端失败：{e}")

        try:
            bucket = os.getenv('MINIO_BUCKET_NAME')
            object_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{filename}"
            minio_client.fput_object(bucket, object_name, import_file_path)
        except Exception as e:
            logging.error(f"上传文件到minio失败：{e}")
            raise MinioError(f"上传文件到minio失败：{e}")

    def _get_path_with_date(self) -> str:
        return os.path.join(get_local_base_dir(), datetime.datetime.now().strftime('%Y%m%d'))
