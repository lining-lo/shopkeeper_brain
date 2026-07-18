"""
  @Author:lining-lo
  @Time:2026/7/17
  @Desc:使用正则表达式查找并替换内容
"""
import re
from pathlib import Path

md_content = "这是一张图片 ![](./images/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg) 和另一张 ![](./images/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)"

pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

summaries = {
    '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': '万用表RS-12直流电流测量接线示意图（10A档位）',
    '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': '蜂鸣器功能符号指示'
}
remote_urls = {
    '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': 'http://192.168.6.160:9000/bucket/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg',
    '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': 'http://192.168.6.160:9000/bucket/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg'
}


def replacer(match: re.Match) -> str:
    original_path = match.group(2).strip()
    file_name_in_md = Path(original_path).name
    for img_name, summary in summaries.items():
        if img_name == file_name_in_md:
            return f"![{summary}]({remote_urls[img_name]})"
    return match.group(0)

# group(0) — 表示整个匹配结果（即完整匹配到的字符串），不是任何一个捕获组。
# group(1) — 第 1 个捕获组 (.*?)，即 [] 中的内容（图片的 alt 文本）。
# group(2) — 第 2 个捕获组 (.*?)，即 () 中的内容（图片的 URL/路径）。

result = pattern.sub(replacer, md_content)
print(result)
