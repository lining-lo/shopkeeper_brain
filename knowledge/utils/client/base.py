import logging
import os
import threading


logger = logging.getLogger(__name__)



class BaseClientManager:
    """
    客户端管理器基类，提供：
    _require_env(): 环境变量检验
    _get_or_create(): 双重检查锁模板方法
    """

    @staticmethod
    def _require_env(key: str) -> str:
        """
        读取必需的环境变量，缺失立即抛异常
        :param key:
        :return:
        """
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"缺少必须得环境变量:{key}")
        return value

    @classmethod
    def _get_or_create(cls,attr_name:str,lock:threading.Lock,factory):
        """
        双重检查锁模板方法。
        factory是一个工厂方法对象（不加括号传入），只有确认需要创建时才调用。
        这就是延迟执行 ---  把“创建”这个动作延迟到真正需要的那一刻。
        :param attr_name:
        :param lock:
        :param factory:
        :return:
        """
        #第一次检查（无锁，快速返回）
        instance = getattr(cls,attr_name,None)
        if instance is not None:
            return instance

        with lock:
            #第二次检查（有锁，防止并发重复创建）
            instance = getattr(cls,attr_name,None)
            if instance is not None:
                return instance
            instance = factory()
            setattr(cls,attr_name,instance)
            return instance


