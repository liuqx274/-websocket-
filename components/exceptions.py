""" 定义程序执行时的异常 """
from components.logger import get_logger  # 导入日志模块

logging = get_logger()


class SelfError(Exception):
    def __init__(self, message: str):
        self.message = message
        logging.error(f"{message}")
        super().__init__(self.message)

    def __str__(self):
        return self.message
