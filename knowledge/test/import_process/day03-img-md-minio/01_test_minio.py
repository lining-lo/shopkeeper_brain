"""
  @Author:lining-lo
  @Time:2026/7/15
  @Desc:minio文件上传
"""
import os
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

# 初始化客户端
client = Minio(
    os.environ.get("MINIO_ENDPOINT"),
    access_key=os.environ.get("MINIO_ACCESS_KEY"),
    secret_key=os.environ.get("MINIO_SECRET_KEY"),
    secure=False  # 是否使用https
)

if not client.bucket_exists("mybucket"):
    client.make_bucket("mybucket")

# 上传文件
client.fput_object(
    bucket_name="mybucket",
    object_name="images/p1.png",  # Object名称含路径
    file_path="d:/temp/p1.png",  # 上传的本地文件路径
    content_type="image/png"  # MIME类型
)

# 上传后访问地址
url = f"http://192.168.6.160:9000/mybucket/images/p1.png"
print(url) # 桶的权限要开放  private -> public

