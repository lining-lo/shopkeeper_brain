"""
  @Author:lining-lo
  @Time:2026/7/14
  @Desc:pdf转md节点，用MinerU将pdf转化为md
"""
import json
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 用MinerU 将上传pdf文件转换为MorkDown
        # :param state:
        #         state = {
        #             "is_pdf_read_enabled": True,
        #             "is_md_read_enabled": False,
        #             "import_file_path":"E:\doc\万用表RS-12的使用.pdf",          # Path("E:\doc\万用表RS-12的使用.pdf")
        #             "file_dir":"E:\temp_dir"
        #         }
        # :return:

        # 1.参数校验
        import_file_path_obj, file_dir_path_obj = self._validate_state_inputs_path(state)

        # 2.用mineru命令将pdf转换为md
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_path_obj)
        if processed_code == 0:
            self.logger.info(f"PDF转MD成功!{import_file_path_obj} - 输出：{file_dir_path_obj}")
        else:
            self.logger.error("PDF转MD失败")
            raise PdfConversionError("PDF转MD失败", self.name)

        # 3.获取md文件路径（完整）
        md_path = self._get_md_paths(import_file_path_obj, file_dir_path_obj)

        # 4.修改状态并返回
        state['md_path'] = md_path

        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """校验输入参数：文件路径，输出目录"""
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        if not import_file_path:
            raise ValidationError("输入文件路径为空", self.name)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise FileProcessingError("输入文件路径不存在", self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent  # 如果输出路径不存在，就用输入文件的父路径作为输出路径。

        file_dir_path_obj = Path(file_dir)

        return import_file_path_obj, file_dir_path_obj

    def _execute_mineru(self, import_file_path_obj, file_dir_path_obj) -> int:
        proc = subprocess.Popen(
            args=[
                "mineru",
                "-p",
                rf"{import_file_path_obj}",
                "-o",
                rf"{file_dir_path_obj}",
                "--source=local",
                "--backend",
                "pipeline"
            ],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        processed_code = proc.wait()

        return processed_code

    def _get_md_paths(self, import_file_path_obj: Path, file_dir_path_obj: Path) -> str:
        """生成md文件路径"""
        file_name = import_file_path_obj.stem  # 不带扩展名

        md_path = file_dir_path_obj / file_name / "auto" / f"{file_name}.md"

        return str(md_path)


if __name__ == "__main__":
    setup_logging()

    state = {
        "import_file_path": r"D:\查重_简洁报告单.pdf",
        "pdf_path": r"D:\temp_out"
    }
    node = PdfToMdNode()
    result = node(state)
    print(result)
