"""
    这是一个websocket项目的日志配置文件，本组件通过调用 get_logger函数 获得一个日志记录器。
    from components.logger import get_logger as init_logger  # 导入日志模块
    日志文件存在 logs文件夹中，日志文件的保存格式为  日期.log  ps. 2025-03-10.log
"""

# logger.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# 1️⃣ 定义日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # 如果目录不存在，则创建

# 2️⃣ 获取当前日志文件路径
log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# 3️⃣ 用于存储全局 logger 实例的变量
_logger_instance = None


def get_logger():
    """
    出于安全性考虑，我将所有的 logger 实例的名称都固定为 WebSocketServer
    以防止创建不同的 logger 实例时 ，对 全局logger实例的存储  _logger_instance 发送覆盖
    :return: 日志记录器
    """
    global _logger_instance

    # 如果已经有实例，直接返回
    if _logger_instance is not None:
        return _logger_instance

    # 创建新的 logger 实例
    logger = logging.getLogger("WebSocketServer")
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler（如果已经配置过，就不再重复添加）
    if not logger.handlers:
        # 创建文件处理器，按天分割日志文件
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # 将创建的 logger 实例保存到 _logger_instance 中
    _logger_instance = logger

    return logger
