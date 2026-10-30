"""
日志配置模块
统一管理项目的日志记录功能
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class LoggerSetup:
    """日志设置类"""
    
    _initialized = False
    _log_file_path = None
    
    @classmethod
    def setup_logger(cls, name, log_dir=os.path.join("extraction", "logs"), level=logging.INFO):
        # 可选的日志目录配置：
        # log_dir="logs"                    # 项目根目录下的 logs 文件夹
        # log_dir="extraction/logs"         # 默认运行日志
        # log_dir="extraction/output/logs"  # 自定义相对目录
        """
        设置并返回一个logger实例
        
        Args:
            name: logger名称（通常是模块名）
            log_dir: 日志文件目录
            level: 日志级别
            
        Returns:
            logger实例
        """
        logger = logging.getLogger(name)
        
        # 避免重复添加handler
        if logger.handlers:
            return logger
        
        logger.setLevel(level)
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名（第一次初始化时确定）
        if not cls._initialized:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._log_file_path = os.path.join(log_dir, f"processing_{timestamp}.log")
            cls._initialized = True
        
        # 创建文件handler
        file_handler = logging.FileHandler(
            cls._log_file_path, 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # 控制台只显示警告和错误
        
        # 创建格式化器
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # 添加handler到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @classmethod
    def get_log_file_path(cls):
        """获取当前日志文件路径"""
        return cls._log_file_path
    
    @classmethod
    def log_session_start(cls, logger):
        """记录会话开始"""
        logger.info("="*80)
        logger.info("新处理会话开始")
        logger.info(f"日志文件: {cls._log_file_path}")
        logger.info("="*80)
    
    @classmethod
    def log_session_end(cls, logger, success=True):
        """记录会话结束"""
        logger.info("="*80)
        if success:
            logger.info("处理会话成功完成")
        else:
            logger.error("处理会话异常结束")
        logger.info(f"完整日志已保存至: {cls._log_file_path}")
        logger.info("="*80)


def get_logger(module_name):
    """
    便捷函数：获取logger实例
    
    Args:
        module_name: 模块名称
        
    Returns:
        logger实例
    """
    return LoggerSetup.setup_logger(module_name)


# 日志装饰器：记录函数执行
def log_execution(logger=None):
    """
    装饰器：记录函数的执行情况
    
    Usage:
        @log_execution(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            logger.debug(f"开始执行: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"成功完成: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"执行失败: {func.__name__}, 错误: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试日志功能
    test_logger = get_logger("test")
    LoggerSetup.log_session_start(test_logger)
    
    test_logger.debug("这是调试信息")
    test_logger.info("这是一般信息")
    test_logger.warning("这是警告信息")
    test_logger.error("这是错误信息")
    
    LoggerSetup.log_session_end(test_logger)
    
    print(f"\n日志已保存至: {LoggerSetup.get_log_file_path()}")
