"""
  @Author:lining-lo
  @Time:2026/7/16
  @Desc:MD图片处理节点，解析Markdown本地图片并提取上下文，支持VLM图摘要、MinIO上传与文档链接替换
    功能链路：读取Markdown文档 → 扫描images目录图片 → 提取每张图片上下文信息（临近标题、图片前后正文） →
            调用VLM视觉模型结合上下文生成图片描述摘要 → 图片上传MinIO、替换MD本地图片路径为云端URL、文档备份；
    分层设计：MdFileHandler文件读写、ImageScanner上下文解析、VLMSummarizer摘要、ImageUploader上传处理，职责隔离。
"""
import re
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from pprint import pprint
from typing import Tuple, List, Optional

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


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

        return md_content, md_path_obj, images_dir

    def backup(self):
        pass


# -------2.图片上下文-------------
class ImageScanner:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_content:str, images_dir: Path, image_extensions: set[str], context_length: int = 200) -> List[
        ImageInfo]:
        """
        扫描MD文档，找到每一个图片的上下文信息，返回一个图片信息列表
        :param md_content: 完整md文档字符串
        :param images_dir: 图片所在路径
        :param image_extensions: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        :param context_length: 上文或下文最大长度
        :return:List[ImageInfo]
        """
        image_info_list: List[ImageInfo] = []

        for image_path in images_dir.iterdir():

            if not image_path.is_file():  # 不是文件跳过
                continue

            if not image_path.suffix in image_extensions:  # 不是合法扩展名跳过
                # raise ValidationError(f"图片{image_path} - 后缀格式{image_path.suffix}错误", self.node_name)
                continue

            #查找图片上下文,如果表格图片，在md_content中可能找不到，返回None上下文ImageContext对象
            ctx = self._find_context(md_content,image_path.name,context_length)
            if ctx is None:
                continue

            image_info_list.append(ImageInfo(name=image_path.name,path=image_path,context=ctx))

        return image_info_list

    # 三种写法都行
    #def _find_context(self, image_path, md_content, context_length) -> Optional[ImageContext] :
    # def _find_context(self, image_path, md_content, context_length) -> ImageContext | None :
    # def _find_context(self, image_path, md_content, context_length) -> ImageContext or None :

    def _find_context(
            self, md_content: str, img_name: str, max_chars: int = 200
    ) -> ImageContext | None:
        """返回图片在 MD 中第一次出现位置的上下文，找不到返回 None。"""
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        #将md文档拆成一行一行的字符串元素集合
        md_lines = md_content.split("\n")

        for line_idx, line in enumerate(md_lines):
            if not pattern.search(line):
                continue

            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._find_heading_above( md_lines, line_idx )
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

    def summarizer_all(self):
        pass


# -------4.图片存储(minio) & 替换-------------
class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self):
        pass


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
        print(imageinfo_list)
        # 3.vlm生成图片摘要

        # 4.上传图片到Minio,替换MD中图片的路径，插入摘要信息

        # 5.备份替换后的MD文档

        return state



if __name__ == "__main__":
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

    pprint(node(state))
