"""
  @Author:lining-lo
  @Time:2026/7/22
  @Desc:Pydantic基础示例，展示模型定义、参数校验、序列化常用API
"""
import datetime

from pydantic import BaseModel, Field


class Student(BaseModel):
    name: str = Field(..., description="姓名")
    age: int = Field(..., description="年龄")
    score: float = Field(..., description="成绩")
    birthday: datetime.date = Field(datetime.date(2000, 5, 5), description="出生日期")


# 1.正常创建
s1 = Student(name="张三", age=18, score=99.0)
print(s1)
# name='张三' age=18 score=99.0 birthday=datetime.date(2000, 5, 5)

# 2.转成字典
print(s1.model_dump())
# {'name': '张三', 'age': 18, 'score': 99.0, 'birthday': datetime.date(2000, 5, 5)}

# 3.转成json字符串
print(s1.model_dump_json())
# {"name":"张三","age":18,"score":99.0,"birthday":"2000-05-05"}

# 4.自动类型转换
s2 = Student(name="李四", age="22", score=99.0)
print(type(s2.age))
# <class 'int'>

# 5.类型校验失败直接报错
try:
    s3 = Student(name="王五", age="八十八", score=69.0)
except Exception as e:
    print(e)
# 1 validation error for Student
# age
#   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='八十八', input_type=str]
#     For further information visit https://errors.pydantic.dev/2.13/v/int_parsing

# 6.缺少必填字段直接报错
try:
    s4 = Student(name="赵六", score=69.0)
except Exception as e:
    print(e)
# 1 validation error for Student
# age
#   Field required [type=missing, input_value={'name': '赵六', 'score': 69.0}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.13/v/missing

# 7.多余字段自动丢弃
s5 = Student(name="王二麻子", age="16", score=88.0, hobby="rap")
print(s5.model_dump())
# {'name': '王二麻子', 'age': 16, 'score': 88.0, 'birthday': datetime.date(2000, 5, 5)}

# 8.从字典创建模拟后端返回
data = {"name": '猪大肠', "age": 19, "score": 40, "email": "zhudachang@test.com"}
s6 = Student(**data)
print(s6.model_dump())
# {'name': '猪大肠', 'age': 19, 'score': 40.0, 'birthday': datetime.date(2000, 5, 5)}
