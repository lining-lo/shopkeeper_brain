"""
  @Author:lining-lo
  @Time:2026/7/13
  @Desc:入口节点,根据上传文件后缀，来决定走哪个分支节点：[.pdf,.md]
"""
from pathlib import Path
from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


class EntryNode(BaseNode):
    name = "entry_node"

    # 实现父类中继承的方法，每个子节点都需要处理自己那一步任务
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口节点
            .pdf -> pdf_to_md_node -> md_img_node
            .md -> md_img_node
        :param state:
            import_file_path = "E:\\doc\\万用表RS-12的使用.pdf"
            或
            import_file_path = "E:\\doc\\万用表RS-12的使用.md"

            file_dir = "E:\\temp_dir"
        :return:
            {
                "import_file_path" : "E:\\doc\\万用表RS-12的使用.pdf",
                "file_dir" : "E:\\temp_dir",
                "is_pdf_enabled=True",
                "is_md_enabled=False"
            }
        """
        self.log_step("step1", "获取文件或目录")
        # 1.获取上传文件以及路径
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        self.log_step("step2", "校验文件是否存在")
        # 2.判断上传文件是否为空
        if not import_file_path:
            raise ValidationError(f"import_file_path的值为空", self.name)

        # 3.获取文件后缀，并变成小写
        path = Path(import_file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在：{import_file_path}", self.name)

        suffix = path.suffix.lower()

        # 4.不同文件类型做不同状态修改
        self.log_step("step3", "判断文件类型")
        if suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == ".md":
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            raise ValidationError(f"不支持的文件类型：{import_file_path} ->类型： {suffix}", self.name)

        # 5.获取文件标题   不带扩展名
        file_name = path.stem  # 不带扩展名的文件名

        # 6.更新状态，并返回状态
        state['file_title'] = file_name

        return state


if __name__ == "__main__":
    setup_logging()

    init = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": "D:\查重_简洁报告单.pdf",
        "file_dir": "D:\资料"
    }
    state = create_default_state(**init)
    node = EntryNode()
    processed_state = node(state)

    print(processed_state)

"""
打印结果：
    {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": "D:\\查重_简洁报告单.pdf",
        "file_dir": "D:\\资料",
        "pdf_path": "D:\\查重_简洁报告单.pdf",
        "file_title": "查重_简洁报告单"
    }
"""
