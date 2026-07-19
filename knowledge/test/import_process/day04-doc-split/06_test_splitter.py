"""
  @Author:lining-lo
  @Time:2026/7/19
  @Desc:内容分割
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "苹果的颜色是红色的[SEP]香蕉的颜色是黄色的[SEP]橘子的颜色是橙色的"

splitter_false = RecursiveCharacterTextSplitter(
    chunk_size=2,  # 单个切分内容最大长度
    chunk_overlap=0,  # 切分内容重叠部分长度
    keep_separator=False,  # 是否保留分隔符
    separators=["[SEP]","的",""]
)
print(splitter_false.split_text(text))