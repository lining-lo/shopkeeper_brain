"""
  @Author:lining-lo
  @Time:2026/7/15
  @Desc:使用千问视觉模型识别图片内容
"""
import os

from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 图片转 Base64
with open("D:/temp/p1.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

print(base64_image)

# 调用 VLM
response = client.chat.completions.create(
    model="qwen3.6-plus",  # 视觉模型
    # model="qwen3-vl-plus", #视觉模型
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",  # 告诉 API：这是一段文字
                    "text": "请描述这张图片的内容"
                },
                {
                    "type": "image_url",  # 告诉 API：这是一张图片
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=100
)

summary = response.choices[0].message.content
print(summary)
