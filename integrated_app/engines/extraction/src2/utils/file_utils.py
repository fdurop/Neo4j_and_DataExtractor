"""
文件工具类
"""

import os
from typing import List, Optional


class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def get_files(directory: str, extensions: List[str] = None, exclude_temp: bool = True) -> List[str]:
        """
        获取目录下的文件
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名列表（如 ['.pdf', '.pptx']）
            exclude_temp: 是否排除临时文件
            
        Returns:
            文件路径列表
        """
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            # 排除临时文件
            if exclude_temp and filename.startswith('~$'):
                continue
            
            # 过滤扩展名
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                files.append(file_path)
        
        return files
    
    @staticmethod
    def ensure_dir(directory: str):
        """确保目录存在"""
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def get_filename_without_ext(file_path: str) -> str:
        """获取不含扩展名的文件名"""
        return os.path.splitext(os.path.basename(file_path))[0]
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """获取文件扩展名"""
        return os.path.splitext(file_path)[1].lower()
    
    @staticmethod
    def is_pdf(file_path: str) -> bool:
        """判断是否为PDF文件"""
        return FileUtils.get_file_extension(file_path) == '.pdf'
    
    @staticmethod
    def is_pptx(file_path: str) -> bool:
        """判断是否为PPTX文件"""
        return FileUtils.get_file_extension(file_path) == '.pptx'
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小（字节）"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0
