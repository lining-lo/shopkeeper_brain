"""
  @Author:lining-lo
  @Time:2026/7/18
  @Desc:文本装箱算法
"""
from typing import List

def pack_paragraphs(paragraphs: List[str], max_length: int) -> List[str]:
    """将段落装箱到多个 chunk 中"""
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 计算加入当前段落后的长度 +2 是 \n\n 两个换行字符
        new_length = len(current_chunk) + len(para) + 2

        if new_length <= max_length:
            # 可以装入当前块
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            # 当前块满了，存入结果，新开块
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    # 循环结束，把最后一块加入列表
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------- 测试示例 ----------------------
if __name__ == "__main__":
    # 测试段落列表
    test_paras = [
        "## 安全标识",
        "表明此操作须参照说明书进行。",
        "WARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。",
        "CAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。",
        "请勿连接到500VAC或VDC的电路上。",
        "表明此端口可能出现危险电压。",
        "双绝缘保护。"
    ]

    # 限制单块最大字符长度 120
    result = pack_paragraphs(test_paras, max_length=120)

    print("===== 分块结果 =====")
    for idx, chunk in enumerate(result, 1):
        print(f"\n【Chunk {idx}】长度={len(chunk)}")
        print(repr(chunk))
