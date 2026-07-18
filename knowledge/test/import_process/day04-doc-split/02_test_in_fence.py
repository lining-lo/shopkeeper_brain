"""
  @Author:lining-lo
  @Time:2026/7/18
  @Desc:切割文档标题代码围栏检测
"""
# 示例
import re

lines = [
    "# 第一章",  # ✓ 匹配
    "## 1.1 概述",  # ✓ 匹配
    "  ### 缩进标题",  # ✓ 匹配（允许前导空格）
    "~~~# 正文内容~~~",           # ✗ 不匹配
    "```# 正文内容```",  # ✗ 不匹配
    "#标签",  # ✗ 不匹配（# 后需要空格）
    "####### 七级",  # ✗ 不匹配（最多6级）
]

in_fence = False  # 是否在代码块内

heading_re = re.compile(r"^\s*#{1,6}\s+.+")

for line in lines:
    # 检测代码围栏边界（``` 或 ~~~）
    if line.strip().startswith("```") or line.strip().startswith("~~~"):
        in_fence = not in_fence  # 切换状态

    # 只有不在代码块内时才识别标题
    is_heading = (not in_fence) and heading_re.match(line)
    if is_heading:
        print(line)
