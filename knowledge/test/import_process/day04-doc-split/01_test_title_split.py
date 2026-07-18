"""
  @Author:lining-lo
  @Time:2026/7/18
  @Desc:使用正则表达式切割文档标题
"""
import re

# 匹配 1-6 级标题
heading_re = re.compile(r"^\s*#{1,6}\s+.+")

# 示例
lines = [
    "# 第一章",  # ✓ 匹配
    "## 1.1 概述",  # ✓ 匹配
    "  ### 缩进标题",  # ✓ 匹配（允许前导空格）
    "正文内容",  # ✗ 不匹配
    "#标签",  # ✗ 不匹配（# 后需要空格）
    "####### 七级",  # ✗ 不匹配（最多6级）
]

for line in lines:
    if heading_re.match(line):
        print(f"标题: {line}")
