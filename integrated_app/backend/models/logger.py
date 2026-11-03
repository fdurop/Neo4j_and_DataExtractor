# models/logger.py
import logging
import uuid
from flask import g, has_request_context

class RequestIdFilter(logging.Filter):
    """为每条日志添加 request_id"""
    def filter(self, record):
        if has_request_context() and hasattr(g, 'request_id'):
            record.request_id = g.request_id
        else:
            record.request_id = "N/A"
        return True

def get_logger(name: str = __name__, log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(request_id)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdFilter())
        logger.addHandler(console_handler)

    return logger

# 全局默认 logger
logger = get_logger()
