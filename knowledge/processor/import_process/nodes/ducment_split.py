"""
  @Author:lining-lo
  @Time:2026/7/18
  @Desc:文档切割节点：对md文档分层拆分，输出结构化知识库切片
        参数校验 → 按标题分层初分 → 超长拆分/过短合并 → 组装标准分片 → 日志+备份
        识别1-6级标题、自动维护层级父标题、忽略代码块内标题、表格转纯文本、控制切片长度
"""
import json
import os
import re
from pathlib import Path
from typing import List, Any, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.获取输入参数并且校验：  md_content,file_title,max_content_length,min_content_length
        md_content, file_title, max_content_length, min_content_length = self._validate_get_input(state)

        # 2.按照标题切分
        sections: List[dict[str, Any]] = self._split_by_title(md_content, file_title)

        # 3.长切短合
        final_sections = self._split_long_merge_short(sections, max_content_length, min_content_length)

        # 4.组装切片
        final_chunks = self._assemble_chunk(final_sections)

        # 5.日志记录
        self._log_summary(md_content, final_chunks, max_content_length)

        # 6.数据备份
        self._backup_chunks(state, final_chunks)

        # 状态值设置
        state["chunks"] = final_chunks
        return state

    # ==================================================================================
    #          1.获取并校验参数
    # ==================================================================================
    def _validate_get_input(self, state):
        self.log_step("step1", "数据获取并校验")
        # 1.获取输入参数并且校验：  md_content,file_title,max_content_length,min_content_length
        md_content = state.get("md_content")  # 整个md文档内容
        file_title = state.get("file_title")  # 文件名称
        max_content_length = self.config.max_content_length  # 最大切片阈值  大于这个阈值需要二次切分
        min_content_length = self.config.min_content_length  # 最小切片阈值  小于这个阈值要合并

        if not md_content:
            raise ValidationError(f"切分的文档内容不存在!", self.name)

        if not file_title:
            raise ValidationError(f"切分的文档名称不存在!", self.name)

        if max_content_length < 0:
            raise ValidationError(f"参数错误：max_content_length不能小于零", self.name)

        if min_content_length < 0:
            raise ValidationError(f"参数错误：min_content_length不能小于零", self.name)

        if max_content_length < min_content_length:
            raise ValidationError(f"参数错误：max_content_length不能小于min_content_length值", self.name)

        return md_content, file_title, max_content_length, min_content_length

    # ==================================================================================
    #          2.用标题初切
    # ==================================================================================
    def _split_by_title(self, md_content: str, file_title: str) -> List[dict[str, Any]]:
        """
            按照标题切分：按\n获取文档行的集合。遍历，判断是否存在代码围挡，正则表达式识别标题；构建section
                sections = [{
                        "title": "## 安全手册",
                        "parent_title": "# 手册",
                        "file_title": "万用表手册",
                        "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                    },
                    {
                        "title": "## 安全手册",
                        "parent_title": "# 手册",
                        "file_title": "万用表手册",
                        "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                    }
                ]
        :param md_content:
        :param file_title:
        :return:
        """

        self.log_step("step2", "根据标题切分")
        # 变量声明区域
        sections: List[Dict[str, Any]] = []
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)$")
        body_lines = []
        title = ""
        parent_title = ""
        # 存放层级标题
        # ["","","","","","",""]
        # ["","# 标题","## 标题","### 标题","#### 标题","##### 标题","###### 标题"]
        hierarchy = [""] * 7  # 标题等级： 7个长度，第一个（索引0位置）不用  等级、层级
        current_level = 0
        is_fence = False  # 是否在代码围挡内容

        # 1.获取所有行集合
        # lines: List[str] = md_content.split("\n")
        lines: List[str] = [p.strip() for p in md_content.split("\n") if p.strip()]

        def _flush():
            """专业封装section"""
            body = "\n".join(body_lines)
            if body:

                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    parent_title = hierarchy[i] if hierarchy[i] else ""
                    if parent_title:
                        break

                # 降级处理： 当前标题没有父标题时，用当前标题作为父标题。比用文件名作为父标题好一些（信息更精准些）。
                if not parent_title:
                    parent_title = title if title else file_title

                # 封装section
                section = {
                    "title": title if title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title,
                    "body": body
                }
                sections.append(section)

        # 2.遍历
        for line in lines:

            # 判断#是否在代码围挡中
            if line.strip().startswith("~~~") or line.strip().startswith("```"):
                is_fence = not is_fence

            match = heading_re.match(line)

            if match and not is_fence:  # 匹配标题
                # 封装section
                _flush()
                title = line
                level = len(match.group(1))  # 获取#数量
                current_level = level
                # hierarchy = [""] * 7  不能这么处理，否则存在多个同级标题时，后面标题可能无法获取父标题。
                hierarchy[level] = line
                for i in range(level + 1, 7, 1):
                    hierarchy[i] = ""  # 清除上个章节遗留标题
                body_lines = []  # 清除缓存，删除上个章节body内容。接下来存我们这个章节的内容。
            else:  # 不是标题,装箱
                body_lines.append(line)

        _flush()

        return sections  # 按照标题进行初切结果列表

    # ==================================================================================
    #          3.长切短合
    # ==================================================================================
    def _split_long_merge_short(self, sections: List[dict[str, Any]], max_content_length: int = 1000,
                                min_content_length: int = 200) -> List[dict[str, Any]]:

        self.log_step("step3", "长切短合:")
        # 1.长的超过阈值max_content_length，需要二次切分
        current_sections = []
        for section in sections:
            # section不大于阈值，放在列表中返回
            # section大于阈值，切分完成后放列表返回。
            sub_sections: List[dict[str, Any]] = self._split_long_section(section, max_content_length)
            current_sections.extend(sub_sections)  # 把你的列表元素都放在我的列表里。

        # 2.短的小于阈值min_content_length，进行合并
        final_sections = self._merge_short_section(current_sections, min_content_length)
        return final_sections

    # ==================================================================================
    #          3.1 长切
    # ==================================================================================
    def _split_long_section(self, section: dict[str, Any], max_content_length) -> List[Dict[str, Any]]:
        """
        长章节再次切分：
            如果文档长度大于max_content_length，则进行二次切分
            RecursiveCharacterTextSplitter：递归字符文本切分器
        :param section:  章节信息
        :param max_content_length: 最大切片阈值
        :return: 切与不切的列表
        """
        self.log_step("step3.1", "长内容二次切分")
        # 1.获取section对象属性
        title = section.get("title", "")
        parent_title = section.get("parent_title", "")
        file_title = section.get("file_title", "")
        body = section.get("body", "")

        # 2.判断表格
        # 利用工具类，将表格降维处理
        if "<table>" in body:
            self.logger.info(f"对表格进行降维处理")
            body = MarkdownTableLinearizer.process(body)
            section['body'] = body

        # 3.对标题校验长度，超过50截断
        if len(title) > 50:
            title = title[:50]

        # 4.拼接标题前缀
        title_prefix = f"{title}\n\n"
        # 5.计算标题前缀长度 + 内容长度
        content_length = len(title_prefix) + len(body)

        # 6.判断是否需要切分
        if content_length <= max_content_length:
            return [section]

        # 7.计算body可用长度，判断长度是否小于等于0
        body_available_length = max_content_length - len(title_prefix)
        if body_available_length <= 0:
            return [section]

        # 8.需要继续切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_content_length,
            chunk_overlap=0,
            keep_separator=False,
            # separators=["[SEP]","的",""]
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        )
        #	判断切分数量结果，要么返回唯一原元素列表，要么返回多个新元素列表
        texts: List[str] = splitter.split_text(body)

        sub_section = []

        for index, text in enumerate(texts):
            sub_section.append({
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
                "body": text,
                "part": index + 1
            })

        # 9.返回
        return sub_section

    # ==================================================================================
    #          3.2 短合
    # ==================================================================================
    def _merge_short_section(self, current_sections: List[dict[str, Any]], min_content_length: int = 100) -> List[
        dict[str, Any]]:
        """
        贪心累加算法
            1.累加过程可能超过阈值，可接受
            2.最后一个小块，别忘记处理

        :param current_sections: 当前章节列表，有小的块就需要被合并
        :param min_content_length: 最小内容长度阈值，小于它就需要合并下一个内容到当前内容中。
        :return:
        """
        self.log_step("step3.2", "短内容合并")
        final_sections = []  # 合并章节内容后的列表
        current_section = current_sections[0]  # 当前章节section对象

        for next_section in current_sections[1:]:

            # 同一个大章节下，两个小章节，可以合并
            same_title = current_section.get("parent_title") == next_section.get("parent_title")
            if same_title and len(current_section.get("body")) < min_content_length:
                current_section["body"] = current_section["body"].rstrip() + "\n\n" + next_section["body"].lstrip()
                current_section["title"] = current_section["parent_title"]
                current_section["part"] = 0
            else:
                final_sections.append(current_section)  # 遇到不同父标题，把之前同一个标题合并内容封箱。
                current_section = next_section  # 重置

        # 添加最后一部分
        final_sections.append(current_section)

        # 4. 对所有 section 的 part 做处理
        part_counter = {}
        result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

            result.append(final_section)

        return result

    # ==================================================================================
    #          4.组装切片
    # ==================================================================================
    def _assemble_chunk(self, final_sections) -> List[Dict[str, Any]]:
        """
        :param final_sections:
            sections = [{
                        "title": "## 安全手册",
                        "parent_title": "# 手册",
                        "file_title": "万用表手册",
                        "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                    },
                    {
                        "title": "## 安全手册",
                        "parent_title": "# 手册",
                        "file_title": "万用表手册",
                        "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                    }
                ]
        :return:
             chunks = [{
                        "title": "## 安全手册",
                        "parent_title": "# 手册",
                        "file_title": "万用表手册",
                        "content": "title" + "body"
                    },
                    {
                        "title": "## 安全手册",
                        "parent_title": "# 手册",
                        "file_title": "万用表手册",
                        "content": "title" + "body"
                    }
                ]
        """
        self.log_step("step4", "组装切片")
        final_chunks: List[Dict[str, Any]] = []

        for section in final_sections:
            chunk = {
                "title": section["title"],
                "parent_title": section["parent_title"],
                "file_title": section["file_title"],
                "content": section["title"] + "\n\n" + section["body"]
            }

            # 3. 判断 part 是否存在
            if "part" in section:
                chunk['part'] = section.get('part')

            final_chunks.append(chunk)
        return final_chunks

    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

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
            output_path = local_dir / "chunks.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    with open('D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单_new.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    init = {
        'file_dir': 'D:\\资料',
        'file_title': '查重_简洁报告单',
        'import_file_path': 'D:\\查重_简洁报告单.pdf',
        'is_md_read_enabled': False,
        'is_pdf_read_enabled': True,
        'md_content': md_content,
        'md_path': 'D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md',
        'pdf_path': 'D:\\查重_简洁报告单.pdf',
    }

    state = create_default_state(**init)
    node = DocumentSplitNode()

    print(node(state))

"""
文件备份位置：D:\\资料\\查重_简洁报告单\\auto\\chunks.json

打印结果：
{
    "is_pdf_read_enabled": True,
    "is_md_read_enabled": False,
    "import_file_path": "D:\\查重_简洁报告单.pdf",
    "file_dir": "D:\\资料",
    "pdf_path": "D:\\查重_简洁报告单.pdf",
    "md_path": "D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md",
    "file_title": "查重_简洁报告单",
    "md_content": "\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表..."
    "chunks": [
        {
            "title": "查重_简洁报告单",
            "parent_title": "查重_简洁报告单",
            "file_title": "查重_简洁报告单",
            "content": "查重_简洁报告单\n\n![RS PRO品牌标识]..."
        },
        {
            "title": "## 安全手册",
            "parent_title": "## 安全手册",
            "file_title": "查重_简洁报告单",
            "content": "## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册..."
        }
        ...
        ]
}
"""
