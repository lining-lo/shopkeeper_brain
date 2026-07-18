"""
  @Author:lining-lo
  @Time:2026/7/18
  @Desc:句子切分
"""
import re

# 中英文句子结束标点
sentence_pattern = r"(?<=[。！？；.!?;])\s*"

text = "这是第一句。这是第二句！第三句？"
sentences = re.split(sentence_pattern, text)

# ['这是第一句', '这是第二句', '第三句', '']
print(sentences)