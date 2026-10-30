"""
文本处理模块
功能：文本分析、关键信息提取、专业术语识别
"""

import re
from collections import Counter
from math import ceil
from typing import List, Dict, Any, Optional


class TextProcessor:
    def __init__(self, logger=None):
        """
        初始化文本处理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.professional_terms = set()
    
    def process_text(
        self,
        text: str,
        page_num: int = 0,
        image_paths: List[str] = None,
        repeated_lines: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        处理文本内容
        
        Args:
            text: 原始文本
            page_num: 页码
            image_paths: 相关图像路径列表
            
        Returns:
            处理后的文本数据
        """
        if image_paths is None:
            image_paths = []
        
        cleaned_text = self._clean_text(text, repeated_lines=repeated_lines)
        result = {
            "page_num": page_num,
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "key_points": self._extract_key_points(cleaned_text),
            "technical_terms": self._extract_technical_terms(cleaned_text),
            "has_images": len(image_paths) > 0,
            "image_count": len(image_paths),
            "image_paths": image_paths,
            "text_processing": {
                "cleaning_policy": "line_cleanup_v1",
                "term_method": "regex_candidates_v1",
                "term_scope": "page",
                "term_review_status": "pending",
                "terms_are_verified_concepts": False,
                "key_points_method": "line_heuristic_not_verified_learning_objectives",
                "source_text_field": "raw_text",
                "processed_text_field": "cleaned_text",
            },
            "metadata": {
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.split('\n'))
            }
        }
        
        return result
    
    def _clean_text(self, text: str, repeated_lines: Optional[set] = None) -> str:
        """
        清理文本（去除多余空白、特殊字符等）
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        repeated_lines = repeated_lines or set()
        retained = []
        for raw_line in str(text or "").splitlines():
            line = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f\ue000-\uf8ff]', '', raw_line)
            line = re.sub(r'\s+', ' ', line).strip()
            normalized = self._normalize_line(line)
            if not line or normalized in repeated_lines:
                continue
            if re.fullmatch(r'\d{1,3}\s*[-–—]\s*\d{0,3}', line):
                continue
            line = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', line)
            line = re.sub(r'\s+', ' ', line).strip()
            if line:
                retained.append(line)
        return "\n".join(retained)

    @classmethod
    def find_repeated_lines(cls, page_texts: List[str], ratio: float = 0.6) -> set:
        """Find document-level headers/footers repeated on most pages."""
        if len(page_texts) < 3:
            return set()
        counts = Counter()
        for text in page_texts:
            page_lines = {
                cls._normalize_line(line)
                for line in str(text or "").splitlines()
                if cls._normalize_line(line)
            }
            counts.update(page_lines)
        threshold = max(3, ceil(len(page_texts) * ratio))
        return {line for line, count in counts.items() if count >= threshold and len(line) <= 120}

    @staticmethod
    def _normalize_line(line: str) -> str:
        line = re.sub(r'[\ue000-\uf8ff]', '', str(line or ""))
        return re.sub(r'\s+', ' ', line).strip().casefold()
    
    def _extract_key_points(self, text: str) -> List[str]:
        """
        提取文本中的关键点
        
        Args:
            text: 文本内容
            
        Returns:
            关键点列表
        """
        key_points = []
        
        # 提取标题式内容（通常是关键点）
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # 短句、包含数字编号、或以特定标点结尾的可能是关键点
            if line and (
                (len(line) < 100 and re.match(r'^\d+[\.\)、]', line)) or  # 数字编号
                line.endswith(('：', ':', '。')) or  # 特定标点
                (len(line) < 50 and line[0].isupper())  # 短句且首字母大写
            ):
                key_points.append(line)
        
        return key_points[:10]  # 最多返回10个关键点
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """
        提取专业术语
        
        Args:
            text: 文本内容
            
        Returns:
            专业术语列表
        """
        terms = []
        stop_words = {
            "On", "Off", "Low", "High", "Forward", "Average", "Input", "Output",
            "Power", "Control", "Service", "Routine", "Pass", "Filter", "Moving",
        }
        allow_words = {"Arduino", "Uno", "UNO", "MEGA", "Leonardo"}

        text_without_email = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', text)

        english_candidates = re.findall(r'\b[A-Za-z][A-Za-z0-9_+-]{1,}\b', text_without_email)
        for term in english_candidates:
            if term in stop_words:
                continue
            if term in allow_words:
                terms.append(term)
                continue
            if term.isupper() and len(term) >= 2:
                terms.append(term)
                continue
            if re.search(r'\d', term) and len(term) >= 3:
                terms.append(term)

        chinese_patterns = [
            r'[\u4e00-\u9fa5]{2,10}(?:系统|算法|方法|模型|理论|技术|协议|标准)',
            r'[\u4e00-\u9fa5]{2,10}(?:控制|检测|识别|分析|处理)',
            r'[\u4e00-\u9fa5]{2,10}(?:电机|驱动器|中断|滑台|传感器|执行器)',
        ]
        for pattern in chinese_patterns:
            terms.extend(re.findall(pattern, text_without_email))

        unique_terms = sorted({term.strip() for term in terms if self._is_valid_term(term)})
        self.professional_terms.update(unique_terms)

        return unique_terms[:20]

    @staticmethod
    def _is_valid_term(term: str) -> bool:
        term = (term or "").strip()
        if len(term) < 2 or len(term) > 30:
            return False
        if re.fullmatch(r'[A-Za-z]', term):
            return False
        if re.fullmatch(r'[A-Z]{1,3}\d{1,2}[+-]?', term):
            return False
        if re.fullmatch(r'\d+', term):
            return False
        if re.search(r'[\u4e00-\u9fa5]', term) and len(term) < 3:
            return False
        return True
    
    def get_text_context(self, full_text: str, target_text: str, context_length: int = 100) -> str:
        """
        获取目标文本的上下文
        
        Args:
            full_text: 完整文本
            target_text: 目标文本片段
            context_length: 上下文长度
            
        Returns:
            包含上下文的文本
        """
        try:
            index = full_text.find(target_text)
            if index == -1:
                return target_text
            
            start = max(0, index - context_length)
            end = min(len(full_text), index + len(target_text) + context_length)
            
            context = full_text[start:end]
            return context
        except Exception as e:
            if self.logger:
                self.logger.error(f"获取上下文失败: {e}")
            return target_text
    
    def detect_content_type(self, text: str) -> Dict[str, bool]:
        """
        检测文本内容类型
        
        Args:
            text: 文本内容
            
        Returns:
            内容类型标记字典
        """
        return {
            "has_code": self._has_code(text),
            "has_formula": self._has_formula(text),
            "has_table": self._has_table(text),
            "has_url": bool(re.search(r'https?://\S+', text)),
            "has_email": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        }
    
    def _has_code(self, text: str) -> bool:
        """检测是否包含代码"""
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python函数
            r'function\s+\w+\s*\(',  # JavaScript函数
            r'class\s+\w+',  # 类定义
            r'import\s+\w+',  # 导入语句
            r'#include\s*<',  # C/C++包含
            r'public\s+class',  # Java类
        ]
        return any(re.search(pattern, text) for pattern in code_indicators)
    
    def _has_formula(self, text: str) -> bool:
        """检测是否包含数学公式"""
        formula_indicators = [
            r'\$.*?\$',  # LaTeX行内公式
            r'\\\[.*?\\\]',  # LaTeX块公式
            r'\\begin\{equation\}',  # LaTeX方程环境
            r'[∫∑∏√±≤≥≠≈∞∂∇]',  # 数学符号
            r'\b[a-z]\s*=\s*[\d\w\+\-\*/\(\)]+',  # 简单方程
        ]
        return any(re.search(pattern, text) for pattern in formula_indicators)
    
    def _has_table(self, text: str) -> bool:
        """检测是否包含表格（基于文本特征）"""
        lines = text.split('\n')
        
        # 检测连续多行包含分隔符（表格特征）
        separator_lines = 0
        for line in lines:
            if re.search(r'[\|\t]{2,}', line):
                separator_lines += 1
        
        return separator_lines >= 3  # 至少3行包含分隔符
