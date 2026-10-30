"""
公式提取模块
功能：从文本和图像中提取数学公式
"""

import re
import os
from typing import List, Dict, Any, Optional
from PIL import Image
from .formula_validator import FormulaValidator


class FormulaExtractor:
    def __init__(self, ocr_engine=None, deepseek_wrapper=None, logger=None):
        """
        初始化公式提取器
        
        Args:
            ocr_engine: OCR引擎（如PaddleOCR）
            deepseek_wrapper: DeepSeek-VL包装器
            logger: 日志记录器
        """
        self.ocr_engine = ocr_engine
        self.deepseek_wrapper = deepseek_wrapper
        self.logger = logger
        self.validator = FormulaValidator()
    
    def extract_formulas_from_text(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        从文本中提取公式
        
        Args:
            text: 文本内容
            context: 上下文信息
            
        Returns:
            公式列表
        """
        formulas = []
        
        # Extract explicit LaTeX first, then simple equations. Validation and
        # deduplication happen once after all page sources are collected.
        latex_formulas = self._extract_latex_formulas(text)
        formulas.extend(latex_formulas)
        
        # 2. 提取简单数学表达式
        simple_formulas = self._extract_simple_formulas(text)
        formulas.extend(simple_formulas)
        
        return formulas
    
    def _extract_latex_formulas(self, text: str) -> List[Dict[str, Any]]:
        """提取LaTeX格式的公式"""
        formulas = []
        occupied = []

        # Parse block formulas first so their inner $...$ is not emitted again.
        block_pattern = r'\$\$(.+?)\$\$'
        for match in re.finditer(block_pattern, text, re.DOTALL):
            occupied.append(match.span())
            formulas.append({
                "type": "latex_block",
                "content": match.group(1),
                "latex": match.group(1),
                "position": match.span(),
                "source_type": "text_latex_block",
                "extraction_method": "text_latex",
            })

        # 行内公式 $...$
        inline_pattern = r'(?<!\$)\$([^\n$]+)\$(?!\$)'
        for match in re.finditer(inline_pattern, text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            formulas.append({
                "type": "latex_inline",
                "content": match.group(1),
                "latex": match.group(1),
                "position": match.span(),
                "source_type": "text_latex_inline",
                "extraction_method": "text_latex",
            })
        
        # LaTeX环境
        env_pattern = r'\\begin\{(equation|align|gather)\}(.*?)\\end\{\1\}'
        for match in re.finditer(env_pattern, text, re.DOTALL):
            formulas.append({
                "type": f"latex_{match.group(1)}",
                "content": match.group(2),
                "latex": match.group(0),
                "position": match.span(),
                "source_type": "text_latex_environment",
                "extraction_method": "text_latex",
            })
        
        return formulas
    
    def _extract_simple_formulas(self, text: str) -> List[Dict[str, Any]]:
        """提取简单数学表达式"""
        formulas = []
        protected_spans = []
        protected_patterns = (
            r'\$\$.+?\$\$',
            r'(?<!\$)\$[^\n$]+\$(?!\$)',
            r'\\begin\{(?:equation|align|gather)\}.*?\\end\{(?:equation|align|gather)\}',
        )
        for pattern in protected_patterns:
            protected_spans.extend(match.span() for match in re.finditer(pattern, text, re.DOTALL))

        # 匹配简单方程式：a = b + c
        equation_pattern = r'\b([a-zA-Z_]\w*)\s*=\s*([^\n\r;；，。$]{1,160})'
        for match in re.finditer(equation_pattern, text):
            if any(match.start() < end and match.end() > start for start, end in protected_spans):
                continue
            right_side = match.group(2).strip()
            is_function_call = re.search(r'[A-Za-z_]\w*\s*\(', right_side) is not None
            has_arithmetic = re.search(r'[A-Za-z0-9_)\]]\s*[+\-*/^]\s*[A-Za-z0-9_(\[]', right_side) is not None
            looks_like_code = any(token in match.group(0) for token in [';', '==', '!=', '++', '--'])
            if has_arithmetic and not is_function_call and not looks_like_code:
                formulas.append({
                    "type": "equation",
                    "content": match.group(0),
                    "latex": match.group(0),
                    "variable": match.group(1),
                    "expression": right_side.strip(),
                    "position": match.span(),
                    "source_type": "text_simple_equation",
                    "extraction_method": "text_rule",
                })
        
        return formulas

    def should_analyze_image(self, description: str) -> bool:
        """Conservatively route likely formula images to the specialist API."""
        description = str(description or "")
        keywords = (
            "公式", "方程", "数学表达式", "计算式", "积分", "微分", "导数",
            "矩阵", "分式", "根号", "formula", "equation", "latex",
        )
        if any(keyword.lower() in description.lower() for keyword in keywords):
            return True
        return bool(re.search(r"[A-Za-zα-ωΑ-Ω]\s*[=≈<>]\s*[^，。；\n]{2,80}", description))

    def extract_formulas_from_vlm_image(
        self,
        image_path: str,
        context: str = "",
        image_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Use the configured VLM and retain all formulas returned for one image."""
        if not self.deepseek_wrapper:
            return []
        try:
            response = self.deepseek_wrapper.recognize_formula(image_path, context=context)
            raw_formulas = response.get("formulas") if isinstance(response, dict) else []
            if not raw_formulas and isinstance(response, dict) and response.get("latex"):
                raw_formulas = [response]
            source_metadata = {
                "source_image": image_path,
                "source_type": "image_vlm",
                "extraction_method": f"{getattr(self.deepseek_wrapper, 'provider', 'vlm')}_api_formula",
                "backend": "api",
                "provider": getattr(self.deepseek_wrapper, "provider", None),
                "model": getattr(self.deepseek_wrapper, "model", None),
                "image_index": image_index,
                "raw_response": response.get("raw_response", "") if isinstance(response, dict) else "",
                "parse_error": response.get("parse_error") if isinstance(response, dict) else None,
                "partial_response": response.get("partial_response", False) if isinstance(response, dict) else False,
                "review_reason": response.get("review_reason") if isinstance(response, dict) else None,
            }
            return [{**source_metadata, **formula} for formula in raw_formulas if isinstance(formula, dict)]
        except Exception as exc:
            if self.logger:
                self.logger.error(f"公式视觉识别失败: {exc}")
            return [{
                "source_image": image_path,
                "source_type": "image_vlm",
                "extraction_method": "vlm_formula",
                "latex": "",
                "api_error": str(exc),
                "review_required": True,
                "review_reason": "api_failed",
                "image_index": image_index,
            }]

    def finalize_page(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize, validate, and deduplicate all formulas collected for a page."""
        return self.validator.validate_many(formulas)
    
    def extract_formulas_from_image(self, image_path: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        从图像中提取公式
        
        Args:
            image_path: 图像路径
            context: 上下文信息
            
        Returns:
            提取的公式信息
        """
        result = {
            "image_path": image_path,
            "formulas": [],
            "method": "none",
            "success": False
        }
        
        # 1. 优先使用DeepSeek-VL识别公式
        if self.deepseek_wrapper:
            try:
                formula = self._recognize_with_deepseek(image_path, context)
                if formula:
                    result["formulas"].append(formula)
                    result["method"] = "deepseek"
                    result["success"] = True
                    return result
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DeepSeek公式识别失败: {e}")
        
        # 2. 备用OCR识别
        if self.ocr_engine:
            try:
                formula = self._recognize_with_ocr(image_path)
                if formula:
                    result["formulas"].append(formula)
                    result["method"] = "ocr"
                    result["success"] = True
                    return result
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"OCR公式识别失败: {e}")
        
        return result
    
    def _recognize_with_deepseek(self, image_path: str, context: str = "") -> Optional[Dict[str, Any]]:
        """使用DeepSeek-VL识别公式"""
        if not self.deepseek_wrapper:
            return None
        
        prompt = """请识别图片中的数学公式，并以LaTeX格式输出。
要求：
1. 公式部分用LaTeX格式表示
2. 说明公式中各符号的含义
3. 简要说明公式的物理或数学意义

格式：
LaTeX: [公式]
符号说明: [说明]
意义: [意义]
"""
        
        response = self.deepseek_wrapper.generate_description(image_path, prompt)
        
        # 解析响应
        formula_data = {
            "latex": "",
            "symbols": {},
            "meaning": "",
            "raw_response": response
        }
        
        # 提取LaTeX部分
        latex_match = re.search(r'LaTeX:\s*(.+?)(?:\n|$)', response)
        if latex_match:
            formula_data["latex"] = latex_match.group(1).strip()
        
        # 提取意义部分
        meaning_match = re.search(r'意义:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        if meaning_match:
            formula_data["meaning"] = meaning_match.group(1).strip()
        
        return formula_data if formula_data["latex"] else None
    
    def _recognize_with_ocr(self, image_path: str) -> Optional[Dict[str, Any]]:
        """使用OCR识别公式"""
        if not self.ocr_engine:
            return None
        
        try:
            result = self.ocr_engine.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                return None
            
            # 提取所有文本
            texts = [line[1][0] for line in result[0]]
            full_text = ' '.join(texts)
            
            return {
                "content": full_text,
                "confidence": sum(line[1][1] for line in result[0]) / len(result[0]),
                "raw_ocr": result
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"OCR识别失败: {e}")
            return None
    
    def is_formula_image(self, image_path: str) -> bool:
        """
        判断图像是否可能包含公式
        
        Args:
            image_path: 图像路径
            
        Returns:
            是否为公式图像
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # 简单启发式判断
            # 公式图像通常：宽度适中、高度较小、横纵比大
            aspect_ratio = width / height if height > 0 else 0
            
            is_formula = (
                100 < width < 1000 and  # 宽度适中
                20 < height < 200 and   # 高度较小
                aspect_ratio > 2        # 横向长条形
            )
            
            return is_formula
        except Exception:
            return False
