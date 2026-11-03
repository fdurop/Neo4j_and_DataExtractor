"""
文档解析器模块 - 适配不同文件格式
提供统一的接口解析PDF、PPTX等格式
"""

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .pptx_parser import PPTXParser

__all__ = [
    'BaseParser',
    'PDFParser',
    'PPTXParser'
]
