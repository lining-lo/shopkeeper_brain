"""
  @Author:lining-lo
  @Time:2026/7/18
  @Desc:文档切割节点
"""
import json
import os
import re
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
        lines: List[str] = md_content.split("\n")

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
                    parent_title = parent_title if parent_title else title

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
                "body": f"{title}-{index + 1} {text}",
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
            2.孤儿小块，别忘记处理

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
            else:
                final_sections.append(current_section)  # 遇到不同父标题，把之前同一个标题合并内容封箱。
                current_section = next_section  # 重置

        # 添加最后一部分
        final_sections.append(current_section)

        return final_sections

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
            final_chunks.append({
                "title": section["title"],
                "parent_title": section["parent_title"],
                "file_title": section["file_title"],
                "content": section["title"] + "\n\n" + section["body"]
            })

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

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)  # json.dump() 具有写操作能力    注意： 不是json.dumps()
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()
    init = {
        'file_dir': 'D:\\资料',
        'file_title': '查重_简洁报告单',
        'import_file_path': 'D:\\查重_简洁报告单.pdf',
        'is_md_read_enabled': False,
        'is_pdf_read_enabled': True,
        'md_content': '![RS PRO '
                      '品牌标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)\n'
                      '\n'
                      '使用说明书\n'
                      '\n'
                      'RS-12\n'
                      '\n'
                      '编号: 123-1939\n'
                      '\n'
                      '数字万用表\n'
                      '\n'
                      '![中文操作界面标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/81735c16d6175e1dd624407b3448d8bf2039d8e22b1f9cc3941e533706a070a9.jpg)\n'
                      '\n'
                      'CE\n'
                      '\n'
                      '![RS-12数字万用表面板结构与功能标识图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/f179d9399297a15b5d4e764602734c25302eec0b528b231f0e455ca9c76dce0b.jpg)\n'
                      '\n'
                      '## 安全手册\n'
                      '\n'
                      '为了您的安全，请在使用本仪表之前仔细阅读该手册:\n'
                      '\n'
                      '使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n'
                      '\n'
                      '<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 '
                      'colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td '
                      'rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 '
                      'colspan=1>交/直流电压</td><td rowspan=1 '
                      'colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 '
                      'colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V, '
                      '200Vrms 用于200mV量程</td></tr><tr><td rowspan=1 '
                      'colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA '
                      '250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td '
                      'rowspan=1 colspan=1>10A 250V '
                      '快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 '
                      'colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms, '
                      '最多15秒</td></tr></table>\n'
                      '\n'
                      '2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n'
                      '\n'
                      '3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n'
                      '\n'
                      '4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n'
                      '\n'
                      '5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n'
                      '\n'
                      '6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n'
                      '\n'
                      '7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n'
                      '\n'
                      '## 安全标识\n'
                      '\n'
                      '![通用安全警告标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/6ed1c2d2c192fe7422f77d0eb13133a4f4b01ea3e738ce2017bc5df75185e1a0.jpg)\n'
                      '\n'
                      '表明此操作须参照说明书进行。\n'
                      '\n'
                      'WARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n'
                      '\n'
                      'CAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n'
                      '\n'
                      '![最大值标识（MAX）警示符号](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/4033d9da04e1ffceb8382efdf1c281a8fbddf7ebe238b3aa131ff2f0c43fbeb0.jpg)\n'
                      '\n'
                      '请勿连接到500VAC或VDC的电路上。\n'
                      '\n'
                      '![闪电警示符号：表示危险电压，需避免接触以防致命伤害或设备损坏](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/2219ca75130874ec766983013be86c7afd233a4a1b3a5188990261156f1f2cc5.jpg)\n'
                      '\n'
                      '表明此端口可能出现危险电压。\n'
                      '\n'
                      '![双绝缘保护标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/6b7cc68242e90cec872c128c48320467c4ea101b9973420a5a2a2c46c1a7d489.jpg)\n'
                      '\n'
                      '双绝缘保护。\n'
                      '\n'
                      '## 控制与端口\n'
                      '\n'
                      '1.LCD液晶显示\n'
                      '\n'
                      '2.功能选择转盘\n'
                      '\n'
                      '3.10A端口\n'
                      '\n'
                      '4.COM端口\n'
                      '\n'
                      '5.正极端口\n'
                      '\n'
                      '6.数据保持按键\n'
                      '\n'
                      '7.背光按键\n'
                      '\n'
                      '![万用表各部件标识图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d6946861c4592804bd8d7e75b58029565712d4dc58f855e374bf0fcf370c91dd.jpg)\n'
                      '\n'
                      '## 功能符号指示\n'
                      '\n'
                      '•))) 蜂鸣指示\n'
                      '\n'
                      '![二极管测试指示符号](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/e92ffd955b1ca1fd14290da681a763771c958cbcf0a73a332107f471f96c29b2.jpg)\n'
                      '\n'
                      '二极管测试指示\n'
                      '\n'
                      'µ micro (电流范围)\n'
                      '\n'
                      'm milli ( 电压/电流范围)\n'
                      '\n'
                      'k kilo (电阻范围)\n'
                      '\n'
                      'VDC 直流电压\n'
                      '\n'
                      'VAC 交流电流\n'
                      '\n'
                      'ADC 直流电流\n'
                      '\n'
                      'BAT 电池电量不足指示\n'
                      '\n'
                      '## 规格\n'
                      '\n'
                      '<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 '
                      'colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 '
                      'colspan=1>精确度</td></tr><tr><td rowspan=5 '
                      'colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td '
                      'rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>± (0.5% '
                      'reading + 2 digits)</td></tr><tr><td rowspan=1 '
                      'colspan=1>2000mV</td><td rowspan=1 '
                      'colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td '
                      'rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 '
                      'colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td '
                      'rowspan=2 colspan=1>± (0.8% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td '
                      'rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 '
                      'colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td '
                      'rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (1.2% '
                      'reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 '
                      'colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td '
                      'rowspan=4 colspan=1>直流电流</td><td rowspan=1 '
                      'colspan=1>2000μA</td><td rowspan=1 colspan=1>1μA</td><td '
                      'rowspan=2 colspan=1>± (1.0% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td '
                      'rowspan=1 colspan=1>10μA</td></tr><tr><td rowspan=1 '
                      'colspan=1>200mA</td><td rowspan=1 colspan=1>100μA</td><td '
                      'rowspan=1 colspan=1>± (1.2% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td '
                      'rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% '
                      'reading + 2 digits)</td></tr><tr><td rowspan=5 '
                      'colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td '
                      'rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>± (0.8% '
                      'reading + 2 digits)</td></tr><tr><td rowspan=1 '
                      'colspan=1>2000Ω</td><td rowspan=1 '
                      'colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td '
                      'rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 '
                      'colspan=1>200kΩ</td><td rowspan=1 '
                      'colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 '
                      'colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td '
                      'rowspan=1 colspan=1>± (1.0% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td '
                      'rowspan=1 colspan=1>9V</td><td rowspan=1 '
                      'colspan=1>10mV</td><td rowspan=2 colspan=1>± (1.0% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td '
                      'rowspan=1 colspan=1>1mV</td></tr></table>\n'
                      '\n'
                      '注意: 精确度规格由两种因素组成。  \n'
                      '● (% reading) –测量电路的精确度。  \n'
                      '● (+ digits) –数位转换器条码的精确度。  \n'
                      '注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n'
                      '\n'
                      '## 技术指标说明\n'
                      '\n'
                      '二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n'
                      '\n'
                      '短路蜂鸣测试 若电阻小于30时产生蜂鸣\n'
                      '\n'
                      '电池测试电流 9V (6mA)；1.5V (100mA)\n'
                      '\n'
                      '输入阻抗 >1MΩ\n'
                      '\n'
                      '交流电压频宽 45Hz～450Hz\n'
                      '\n'
                      'DCA电压跌路测试 200mV\n'
                      '\n'
                      '显示 3 ½ 数位，2000位液晶显示，1.1”数位\n'
                      '\n'
                      '超量程提示 以“1”表示\n'
                      '\n'
                      '极性 自动(正极无显示);负极显示(-)\n'
                      '\n'
                      '测量率 正常情况下每秒2次\n'
                      '\n'
                      '低电池提示 电池电压不足时，显示BAT符号\n'
                      '\n'
                      '电池 一粒9V (NEDA 1604) 电池\n'
                      '\n'
                      '保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n'
                      '\n'
                      '操作环境 32°F～122°F (0°C～50°C)\n'
                      '\n'
                      '储存温度 -4°F～140°F (-20°C～60°C)\n'
                      '\n'
                      '相对湿度 <70% 操作, <80% 储存\n'
                      '\n'
                      '室内使用,最高海拔 7000英尺(2000米)\n'
                      '\n'
                      '重量 255g\n'
                      '\n'
                      '尺寸 150mm x 70mm x 48mm\n'
                      '\n'
                      '安全认证 室内使用，符合过电压类别II\n'
                      '\n'
                      '污染级别 2\n'
                      '\n'
                      '## 电池安装\n'
                      '\n'
                      '警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n'
                      '\n'
                      '1. 把表笔与仪表断开。\n'
                      '\n'
                      '2. 用螺丝刀拧开电池后盖上的螺母。\n'
                      '\n'
                      '3. 正确安装电池，正负极应一致。\n'
                      '\n'
                      '4. 盖上电池后盖并拧紧螺丝钉。\n'
                      '\n'
                      '警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n'
                      '\n'
                      '注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n'
                      '\n'
                      '## 操作指导\n'
                      '\n'
                      '## 数值保持按键HOLD\n'
                      '\n'
                      '保持键允许仪表固定测量值以供参考：\n'
                      '\n'
                      '1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n'
                      '\n'
                      '2. 再次按下“DATA HOLD”键 切换至正常操作\n'
                      '\n'
                      '## 背光灯键（BACKLIGHT）\n'
                      '\n'
                      '1. 按下背光灯键开启背光灯。\n'
                      '\n'
                      '2. 再次按背光灯键关闭背光灯。\n'
                      '\n'
                      '警告：小心触电，高压电流十分危险，应小心操作。\n'
                      '\n'
                      '1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n'
                      '\n'
                      '2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n'
                      '\n'
                      '注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n'
                      '\n'
                      '## 测量非接触交流电压\n'
                      '\n'
                      '警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n'
                      '\n'
                      '1. 让其探头靠近或插入火线的输出插座孔时。\n'
                      '\n'
                      '2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n'
                      '\n'
                      '注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n'
                      '\n'
                      '注意: '
                      '此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n'
                      '\n'
                      '## 直流电压测量\n'
                      '\n'
                      '注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n'
                      '\n'
                      '1. 将功能转盘置于V DC的位置。\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n'
                      '\n'
                      '3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n'
                      '\n'
                      '4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n'
                      '\n'
                      '## 交流电压测量\n'
                      '\n'
                      '警告：谨防触电。\n'
                      '\n'
                      '若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n'
                      '\n'
                      '注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n'
                      '\n'
                      '![交流电压测量操作示意图（表笔连接与读数显示）](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg)\n'
                      '\n'
                      '1. 将功能转盘置于V AC的位置。\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n'
                      '\n'
                      '3. 将表笔尖端接触被测物。\n'
                      '\n'
                      '4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n'
                      '\n'
                      '在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n'
                      '\n'
                      '## 直流电流测量\n'
                      '\n'
                      '注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n'
                      '\n'
                      '![直流电流测量接线示意图（10A档位）](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/8eb1e59b1e3f5e200f6d947da47dcd767fe061b91e72b1ba5325869677dcdad2.jpg)\n'
                      '\n'
                      '1. 将黑色表笔插入负极COM端口。\n'
                      '\n'
                      '2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n'
                      '\n'
                      '3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n'
                      '\n'
                      '4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n'
                      '\n'
                      '5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n'
                      '\n'
                      '6. 接通电源。\n'
                      '\n'
                      '7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA '
                      'DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n'
                      '\n'
                      '![直流10A电流测量接线示意图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/950918b0a12b239de83d45093fa5e6258bfa3d33848f927ecf0993f3210bf3d9.jpg)\n'
                      '\n'
                      '## 电阻测量\n'
                      '\n'
                      '警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n'
                      '\n'
                      '1. 将功能转盘置于最高电阻Ω位置.\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n'
                      '\n'
                      '3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n'
                      '\n'
                      '4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n'
                      '\n'
                      '![数字万用表电阻测量操作示意图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)\n'
                      '\n'
                      '## 短路蜂鸣测试\n'
                      '\n'
                      '警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n'
                      '\n'
                      '1. 将功能键转盘置于 位置。\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n'
                      '\n'
                      '3. 把表笔与被测物体相接触。\n'
                      '\n'
                      '4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n'
                      '\n'
                      '## 二极管测试\n'
                      '\n'
                      '1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n'
                      '\n'
                      '2. 将功能转盘置于 位置。\n'
                      '\n'
                      '3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 '
                      '0V，开路时会在两种极性上显示“1”符号。\n'
                      '\n'
                      '## 电池测试\n'
                      '\n'
                      '1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n'
                      '\n'
                      '2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n'
                      '\n'
                      '3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n'
                      '\n'
                      '4. 在显示屏上读取数值。\n'
                      '\n'
                      '<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 '
                      'colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 '
                      'colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V '
                      '电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 '
                      'colspan=1>7.2 至 8.2V</td><td rowspan=1 '
                      'colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V '
                      '电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 '
                      'colspan=1>1.22 至 1.35V</td><td rowspan=1 '
                      'colspan=1>&lt;1.22V</td></tr></table>\n'
                      '\n'
                      '## 更换电池\n'
                      '\n'
                      '警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n'
                      '\n'
                      '1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n'
                      '\n'
                      '2. 按下面的步骤安装电池。\n'
                      '\n'
                      '3. 妥善处理废电池。\n'
                      '\n'
                      '警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n'
                      '\n'
                      '## 更换保险丝\n'
                      '\n'
                      '警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n'
                      '\n'
                      '1. 把表笔与仪表及其它被测物断开。\n'
                      '\n'
                      '2. 用螺丝刀拧开保险丝门上的螺母。\n'
                      '\n'
                      '3. 轻轻取出废旧的保险丝。\n'
                      '\n'
                      '4. 装入新的保险丝。\n'
                      '\n'
                      '5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V '
                      '快速熔断保险丝用于10A的量程。\n'
                      '\n'
                      '6. 盖回后盖，拧紧螺钉。\n'
                      '\n'
                      '警告: 为防触电，在保险盖盖紧前请勿操作仪表。',
        'md_path': 'D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md',
        'pdf_path': 'D:\\查重_简洁报告单.pdf',
    }
    state = create_default_state(**init)
    node = DocumentSplitNode()
    node(state)
