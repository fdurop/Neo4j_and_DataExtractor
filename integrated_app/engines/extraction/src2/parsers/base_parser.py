"""
基础解析器抽象类
定义所有文档解析器的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseParser(ABC):
    """文档解析器基类"""
    
    def __init__(self, logger=None):
        """
        初始化解析器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.results = []
    
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析结果字典，包含：
            {
                "filename": str,
                "pages": List[Dict],  # 页面/幻灯片列表
                "metadata": Dict,     # 文档元数据
                "success": bool
            }
        """
        pass
    
    @abstractmethod
    def get_page_count(self, file_path: str) -> int:
        """
        获取文档页数/幻灯片数
        
        Args:
            file_path: 文件路径
            
        Returns:
            页数
        """
        pass
    
    @abstractmethod
    def extract_page_content(self, file_path: str, page_num: int) -> Dict[str, Any]:
        """
        提取指定页面的内容
        
        Args:
            file_path: 文件路径
            page_num: 页码（从0开始）
            
        Returns:
            页面内容字典，包含：
            {
                "page_num": int,
                "text": str,          # 文本内容
                "images": List[str],  # 图像路径列表
                "raw_page": Any       # 原始页面对象（供特定处理使用）
            }
        """
        pass
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        获取文档元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            元数据字典
        """
        return {
            "filename": file_path,
            "format": self.__class__.__name__.replace('Parser', '').lower()
        }
    
    def validate_file(self, file_path: str) -> bool:
        """
        验证文件是否有效
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否有效
        """
        import os
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def log_info(self, message: str):
        """记录信息日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")
    
    def log_error(self, message: str):
        """记录错误日志"""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"[ERROR] {message}")
    
    def log_warning(self, message: str):
        """记录警告日志"""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"[WARN] {message}")
