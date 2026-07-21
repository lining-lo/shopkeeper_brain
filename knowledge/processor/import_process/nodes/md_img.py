"""
  @Author:lining-lo
  @Time:2026/7/16
  @Desc:MD图片处理节点，解析Markdown本地图片并提取上下文，支持VLM图摘要、MinIO上传与文档链接替换
    功能链路：读取Markdown文档 → 扫描images目录图片 → 提取每张图片上下文信息（临近标题、图片前后正文） →
            调用VLM视觉模型结合上下文生成图片描述摘要 → 图片上传MinIO、替换MD本地图片路径为云端URL、文档备份；
    分层设计：MdFileHandler文件读写、ImageScanner上下文解析、VLMSummarizer摘要、ImageUploader上传处理，职责隔离。
"""
import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from threading import Lock
from typing import Tuple, List, Dict, Deque
from langchain_openai import OpenAI
from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


# ── 数据模型 ──

@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str  # 最近的章节标题
    pre_text: str  # 图片上方的正文内容
    post_text: str  # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str  # 图片文件名，如 "abc123.jpg"
    path: str  # 图片完整路径
    context: ImageContext  # 在 MD 中的上下文


# -------1.读取 & 备份-------------
class MdFileHandler:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """读取MD文件内容，返回MD内容、MD路径、图片路径"""
        self.logger.info("读取MD文件内容，返回MD内容、MD路径、图片路径")
        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(node_name=self.node_name,
                                  field_name="md_path",
                                  expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError("md_path文件不存在", self.node_name)

        with open(md_path_obj, 'r', encoding='utf-8') as f:
            md_content = f.read()

        images_dir = md_path_obj.parent / "images"
        print(type(images_dir))
        return md_content, md_path_obj, images_dir

    def backup(self, md_path_obj: Path, new_md_content: str):
        """
        备份替换image摘要和地址后的  新MD文档
        :param md_path_obj: 旧md文件路径
        :param new_md_content: 新md文档内容
        """
        self.logger.info("备份新MD文档")
        # 万用表RS-12的使用.md  ->   万用表RS-12的使用_new.md
        backup_path = md_path_obj.with_name(md_path_obj.stem + "_new.md")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(new_md_content)


# -------2.图片上下文-------------
class ImageScanner:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_content: str, images_dir: Path, image_extensions: set[str], context_length: int = 200) -> \
            List[ImageInfo]:
        # 扫描MD文档，找到每一个图片的上下文信息，返回一个图片信息列表
        # :param md_content: 完整md文档字符串
        # :param images_dir: 图片所在路径  E:\temp_dir\万用表RS-12的使用\auto\images
        # :param image_extensions: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        # :param context_length: 上文或下文最大长度
        # :return:List[ImageInfo]
        image_info_list: List[ImageInfo] = []

        for image_path in images_dir.iterdir():

            if not image_path.is_file():  # 不是文件跳过
                continue

            if not image_path.suffix in image_extensions:  # 不是合法扩展名跳过
                # raise ValidationError(f"图片{image_path} - 后缀格式{image_path.suffix}错误", self.node_name)
                continue

            # 查找图片上下文,如果表格图片，在md_content中可能找不到，返回None上下文ImageContext对象
            ctx = self._find_context(md_content, image_path.name, context_length)
            if ctx is None:
                continue

            image_info_list.append(ImageInfo(name=image_path.name, path=image_path, context=ctx))

        return image_info_list

    # 三种写法都行
    # def _find_context(self, image_path, md_content, context_length) -> Optional[ImageContext] :
    # def _find_context(self, image_path, md_content, context_length) -> ImageContext | None :
    # def _find_context(self, image_path, md_content, context_length) -> ImageContext or None :

    def _find_context(
            self, md_content: str, img_name: str, max_chars: int = 200
    ) -> ImageContext | None:
        """返回图片在 MD 中第一次出现位置的上下文，找不到返回 None。"""
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        # 将md文档拆成一行一行的字符串元素集合
        md_lines = md_content.split("\n")

        for line_idx, line in enumerate(md_lines):
            if not pattern.search(line):
                continue

            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._find_heading_above(md_lines, line_idx)
            pre_content = md_lines[prev_boundary + 1: line_idx]
            img_pre = self._extract_limited_context(pre_content, max_chars, direction="front")

            # 向下：找下一个标题，取图片到标题之间的内容作为下文
            next_boundary = self._find_heading_below(md_lines, line_idx)
            post_content = md_lines[line_idx + 1: next_boundary]
            img_post = self._extract_limited_context(post_content, max_chars, direction="end")

            return ImageContext(
                heading=prev_title,
                pre_text=img_pre,
                post_text=img_post,
            )

        return None

    @staticmethod
    def _find_heading_above(
            md_lines: List[str], from_idx: int
    ) -> Tuple[str, int]:
        """从 from_idx 向上查找最近的标题。"""
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    @staticmethod
    def _find_heading_below(md_lines: List[str], from_idx: int) -> int:
        """从 from_idx 向下查找下一个标题。"""
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    @staticmethod
    def _extract_limited_context(
            lines: List[str], max_chars: int, direction: str
    ) -> str:
        """按段落分割，按 direction 方向贪心装填，保持段落完整性。"""
        current_paragraph: List[str] = []
        paragraphs: List[str] = []

        for line in lines:
            # line.strip(): 去除字符串首尾的空白字符(空格、制表符、换行符等)
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()  # 就近原则

        total = 0
        selected: List[str] = []
        for para in paragraphs:
            if (total + len(para) > max_chars) and selected:  # 至少有个段落
                break
            selected.append(para)
            total += len(para)

        if direction == "front":
            selected.reverse()  # 与原文顺序一致，利于VLM

        return "\n\n".join(selected)  # 折行并空一行


# -------3.生成摘要-------------
class VLMSummarizer:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name
        self.stampts_deque: Deque[float] = deque()
        self.lock: Lock = Lock()

    def summarize_all(self, document_name: str, imageinfo_list: List[ImageInfo], vl_model: str,
                      requests_per_minute: int = 5) -> Dict[str, str]:
        """
        摘要生成，返回图片信息列表
        :param document_name: 文件名称 不带扩展名称
        :param imageinfo_list: 图片信息列表（图片标题、上文、下文）
        :param vl_model: VLML视觉语言模型名称
        :param requests_per_minute: 调用模型频率限制
        :return:
            {
                '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': '万用表RS-12直流电流测量接线示意图（10A档位）',
                '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': '蜂鸣器功能符号指示',
                '115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg': '中文说明书标识',
            }
        """
        summaries: Dict[str, str] = {}
        # stampts_deque: Deque[float] = deque()   不能每次请求都创建一个新的队列。多个请求共用一个队列。

        try:
            openai_client = AIClients.get_openai()
        except Exception as e:
            # 降级处理
            self.logger.error(f"获取OpenAI客户端失败: {e}")
            for image_info in imageinfo_list:
                summaries[image_info.name] = "默认图片摘要"
            self.logger.info(f"降级生成 {len(summaries)} 张图片摘要")
            return summaries

        for image_info in imageinfo_list:
            # 限流  requests_per_minute
            self._enforce_rate_limit(requests_per_minute)

            # 调用VLM获取摘要
            summary = self._summarize_one(image_info, openai_client, vl_model, document_name)
            summaries[image_info.name] = summary
        self.logger.info(f"生成 {len(summaries)} 张图片摘要")
        return summaries

    def _summarize_one(self, image_info: ImageInfo, openai_client: OpenAI, vl_model: str, document_name: str) -> str:
        """
        调用VLM获取摘要
        :param image_info: 图片上下文信息
        :param openai_client: VLM模型客户端对象  单例
        :param vl_model:   VLM模型名称  Qwen3-vl-flash
        :param document_name: 文件名称 不带扩展名称   用于生成图片摘要的提示词内容
        :return:
            图片摘要
        """
        parts = [p for p in (image_info.context.heading, image_info.context.pre_text, image_info.context.post_text) if
                 p]
        final_context = "\n".join(parts)

        # 图片转 Base64
        with open(image_info.path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # 调用 VLM
        response = openai_client.chat.completions.create(
            model=vl_model,  # 视觉模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",  # 告诉 API：这是一段文字
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",  # 告诉 API：这是一张图片
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        summary = response.choices[0].message.content
        return summary

    def _enforce_rate_limit(self, requests_per_minute, window: int = 60):
        """
        限流处理：
        :param stampts_deque: 存放每次请求时间戳  1970年1月1日00:00:00
        :param requests_per_minute:   每分钟请求次数限制 默认10
        """
        with self.lock:
            now = time.time()  # 获取当前时间戳

            while self.stampts_deque and now - self.stampts_deque[0] > window:
                self.stampts_deque.popleft()  # 删除队首元素

            if len(self.stampts_deque) >= requests_per_minute:
                sleep_duration = window - (now - self.stampts_deque[0])
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                now = time.time()
                while self.stampts_deque and now - self.stampts_deque[0] > window:
                    self.stampts_deque.popleft()  # 删除队首元素

            self.stampts_deque.append(now)


# -------4.图片存储(minio) & 替换-------------
class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, document_name: str, md_content: str, imageinfo_list: List[ImageInfo],
                           summaries: Dict[str, str], minio_bucket: str,
                           minio_endpoint: str):
        """
        上传图片到Minio  和   替换md 中 图片摘要和路径
        :param document_name: 文件名称 不带扩展名称
        :param images_dir: 图片目录
        :param md_content: md文件内容
        :param imageinfo_list: 图片信息列表(上下文)
        :param summaries: 图片和摘要的字典
        :param minio_bucket: 图片上传到指定Minio中的桶
        :param minio_endpoint: Minio地址
        :return: new_md_content  替换后的md内容
        """
        # 图片名称 对应 远程图片URL 字典
        remote_urls: Dict[str, str] = self._upload_all(imageinfo_list, document_name, minio_bucket, minio_endpoint)

        # 摘要和路径替换  返回新的md内容
        new_md_content = self._replace_in_md(remote_urls=remote_urls,
                                             md_content=md_content,
                                             imageinfo_list=imageinfo_list,
                                             summaries=summaries)

        return new_md_content

    def _upload_all(self, imageinfo_list: List[ImageInfo], document_name: str, minio_bucket: str,
                    minio_endpoint: str) -> Dict[str, str]:
        """
        图片上传到minio
        :param imageinfo_list: 图片信息列表
        :param document_name:  文件名称
        :param minio_bucket:  桶
        :param minio_endpoint:  minio地址
        :return:
            remote_urls: Dict[str, str]  图片名称 对应 远程图片URL 字典
             {
                '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': 'http://192.168.6.160:9000/knowledge-base-files/万用表RS-12的使用/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg',
                '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': 'http://192.168.6.160:9000/knowledge-base-files/万用表RS-12的使用/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg',
            }
        """
        remote_urls: Dict[str, str] = {}

        # 1.获取Minio服务器连接
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            # 主降级处理，当连接获取不到，所有图片都降级处理
            for image_info in imageinfo_list:
                remote_urls[image_info.name] = image_info.path
            return remote_urls

        # 2.循环上传图片
        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(bucket_name=minio_bucket,  # 上传指定的桶名称
                                         object_name=f"{document_name}/{image_info.name}",  # 上传后图片名称（带目录名称）
                                         file_path=image_info.path)  # 本地图片
                remote_url = f"{minio_endpoint}/{minio_bucket}/{document_name}/{image_info.name}"
                remote_urls[image_info.name] = remote_url  # 上传成功，设置为远程地址
            except Exception as e:
                # 二级降级处理  只针对上传失败某个图片进行降级,设置为本地图片地址
                remote_urls[image_info.name] = image_info.path
        return remote_urls

    def _replace_in_md(self, remote_urls: Dict[str, str], md_content: str, imageinfo_list: List[ImageInfo],
                       summaries: Dict[str, str]) -> str:
        """
        图片的 摘要和路径  替换  返回新的md内容

            利用正则表达式查找MD中的图片：  ![]()
                ![](images/e92ffd955b1ca1fd14290da681a763771c958cbcf0a73a332107f471f96c29b2.jpg)


            summaries = {
                '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': '万用表RS-12直流电流测量接线示意图（10A档位）',
                '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': '蜂鸣器功能符号指示'
            }
            remote_urls = {
                '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': 'http://192.168.6.160:9000/bucket/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg',
                '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': 'http://192.168.6.160:9000/bucket/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg'
            }

            ![万用表RS-12直流电流测量接线示意图（10A档位）](http://192.168.6.160:9000/bucket/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)
            ![蜂鸣器功能符号指示](http://192.168.6.160:9000/bucket/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)

        :param remote_urls:  图片名称 对应 远程图片URL 字典
        :param md_content:  md文件内容
        :param imageinfo_list:  图片信息列表
        :param summaries:  图片名和摘要  字典
        :return: 新md文档内容
        """

        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match):
            original_url = match.group(2).strip()  # 匹配到的图片本地地址
            image_url_md = Path(original_url).name  # 图片在md中的文件名称，带扩展名
            for image_name, summary in summaries.items():
                if image_name == image_url_md:
                    return f"![{summary}]({remote_urls[image_name]})"  # 匹配内容被替换后结果
            return match.group(0)  # 正则表达式匹配到的原文

        return pattern.sub(replacer, md_content)


class MarkDownImageNode(BaseNode):
    """
    上传MD或PDF转换为MD ,需要对MD进行图片处理：
        1.图片上传 MinIO   url
        2.图片描述  VLM（视觉语言模型）  千问
        3.图片上传进度   限流
        4.图片替换  摘要+路径
        5.备份
    """

    name = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.image_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.image_uploader = ImageUploader(self.logger, self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.文件处理： 获取整个MD内容，，获取图片路径
        md_content, md_path_obj, images_dir = self.md_file_handler.read_md(state)

        if not images_dir.exists():
            state['md_content'] = md_content
            return state

        # 2.获取图片上下文
        imageinfo_list: List[ImageInfo] = self.image_scanner.scan_img_dir(md_content, images_dir,
                                                                          image_extensions=self.config.image_extensions,
                                                                          # {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
                                                                          context_length=self.config.img_content_length)  # img_content_length: int = 200  # 图片上下文最大长度

        # 3.vlm(视觉语言模型  图生文)生成图片摘要
        summaries: Dict[str, str] = self.vlm_summarizer.summarize_all(document_name=md_path_obj.stem,
                                                                      imageinfo_list=imageinfo_list,
                                                                      vl_model=self.config.vl_model,
                                                                      requests_per_minute=self.config.requests_per_minute)

        # 4.上传图片到Minio,替换MD中图片的路径，插入摘要信息
        # 替换图片路径和摘要的md文件内容。
        new_md_content: str = self.image_uploader.upload_and_replace(document_name=md_path_obj.stem,
                                                                     md_content=md_content,
                                                                     imageinfo_list=imageinfo_list,
                                                                     summaries=summaries,
                                                                     minio_bucket=self.config.minio_bucket,
                                                                     minio_endpoint=self.config.get_minio_base_url())
        state['md_content'] = new_md_content

        # 5.备份替换后的MD文档
        self.md_file_handler.backup(md_path_obj, new_md_content)

        return state


if __name__ == "__main__":
    setup_logging()

    init = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": "D:\\查重_简洁报告单.pdf",
        "file_dir": "D:\\资料",
        "pdf_path": "D:\\查重_简洁报告单.pdf",
        "file_title": "查重_简洁报告单",
        "md_path": "D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md"
    }

    state = create_default_state(**init)
    node = MarkDownImageNode()

    print(node(state))

"""
文件备份位置：D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单_new.md

打印结果：
    {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": "D:\\查重_简洁报告单.pdf",
        "file_dir": "D:\\资料",
        "pdf_path": "D:\\查重_简洁报告单.pdf",
        "md_path": "D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md",
        "file_title": "查重_简洁报告单",
        "md_content": "![RS PRO品牌标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)..."
    }
"""
