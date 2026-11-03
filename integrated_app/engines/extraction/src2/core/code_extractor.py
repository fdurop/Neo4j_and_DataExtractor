"""
代码提取模块
功能：从文本中识别和提取代码片段
"""

import re
from typing import List, Dict, Any, Optional


class CodeExtractor:
    def __init__(self, logger=None):
        """
        初始化代码提取器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        
        # 支持的编程语言关键词
        self.language_keywords = {
            'python': ['def ', 'import ', 'class ', 'if __name__', 'print(', 'return '],
            'java': ['public class', 'private ', 'public static void main', 'System.out'],
            'cpp': [
                '#include', 'int main(', 'std::', 'cout <<', 'namespace ',
                'void setup(', 'void loop(', 'pinMode(', 'digitalWrite(',
                'attachInterrupt('
            ],
            'c': ['#include', 'int main(', 'printf(', 'scanf('],
            'javascript': ['function ', 'const ', 'let ', 'var ', 'console.log', '=>'],
            'matlab': ['function ', 'end', 'plot(', 'disp(', '%.', 'clear all'],
            'r': ['<-', 'library(', 'function(', 'data.frame', 'ggplot'],
        }
    
    def extract_code_from_text(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        从文本中提取代码块
        
        Args:
            text: 文本内容
            context: 上下文信息
            
        Returns:
            代码块列表
        """
        code_blocks = []
        
        # 1. 提取Markdown代码块
        markdown_blocks = self._extract_markdown_code(text)
        code_blocks.extend(markdown_blocks)
        
        # 2. 提取缩进代码块
        indented_blocks = self._extract_indented_code(text)
        code_blocks.extend(indented_blocks)
        
        # 3. 检测单行代码
        if not code_blocks:
            inline_code = self._detect_inline_code(text)
            if inline_code:
                code_blocks.append(inline_code)
        
        return code_blocks
    
    def _extract_markdown_code(self, text: str) -> List[Dict[str, Any]]:
        """提取Markdown格式的代码块"""
        code_blocks = []
        
        # 匹配 ```language ... ```
        pattern = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            
            code_blocks.append({
                "type": "markdown",
                "language": language,
                "code": code,
                "position": match.span(),
                "lines": len(code.split('\n'))
            })
        
        return code_blocks
    
    def _extract_indented_code(self, text: str) -> List[Dict[str, Any]]:
        """提取缩进的代码块"""
        code_blocks = []
        lines = text.split('\n')
        
        current_block = []
        in_code_block = False
        
        for i, line in enumerate(lines):
            # 检测是否为代码行（4空格或1制表符缩进）
            if line.startswith('    ') or line.startswith('\t'):
                current_block.append(line.lstrip())
                in_code_block = True
            else:
                if in_code_block and len(current_block) >= 3:
                    # 至少3行才认为是代码块
                    code = '\n'.join(current_block)
                    language = self._detect_language(code)
                    
                    code_blocks.append({
                        "type": "indented",
                        "language": language,
                        "code": code,
                        "lines": len(current_block),
                        "start_line": i - len(current_block)
                    })
                
                current_block = []
                in_code_block = False
        
        # 处理文件末尾的代码块
        if in_code_block and len(current_block) >= 3:
            code = '\n'.join(current_block)
            language = self._detect_language(code)
            code_blocks.append({
                "type": "indented",
                "language": language,
                "code": code,
                "lines": len(current_block)
            })
        
        return code_blocks
    
    def _detect_inline_code(self, text: str) -> Optional[Dict[str, Any]]:
        """检测没有 Markdown 标记的连续代码，优先避免把课件正文当代码。"""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return None

        code_patterns = [
            r'^\s*#include\s*[<"]',
            r'^\s*(?:def|class|import|from)\s+',
            r'^\s*(?:public|private|protected|static)\s+',
            r'^\s*(?:void|int|float|double|char|bool|byte|const|volatile)\b.*[;{]$',
            r'^\s*(?:if|else|for|while|switch|return)\s*(?:\(|\b)',
            r'^\s*[A-Za-z_]\w*\s*\([^;]*\)\s*;\s*$',
            r'^\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*=.*;\s*$',
            r'^\s*[{}]\s*$',
        ]
        code_lines = [
            line for line in lines
            if any(re.search(pattern, line) for pattern in code_patterns)
        ]

        # A few API names in explanatory prose are not a code block.
        if len(code_lines) < 3 or len(code_lines) / len(lines) < 0.35:
            return None

        code = '\n'.join(code_lines)
        language = self._detect_language(code)
        if language == 'unknown':
            return None
        return {
            "type": "inline",
            "language": language,
            "code": code,
            "lines": len(code_lines),
        }
    
    def _detect_language(self, code: str) -> str:
        """
        检测代码语言
        
        Args:
            code: 代码文本
            
        Returns:
            语言名称
        """
        # 统计各语言关键词出现次数
        scores = {}
        
        for language, keywords in self.language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in code)
            if score > 0:
                scores[language] = score
        
        if not scores:
            return 'unknown'
        
        # 返回得分最高的语言
        return max(scores, key=scores.get)
    
    def get_file_extension(self, language: str) -> str:
        """
        获取语言对应的文件扩展名
        
        Args:
            language: 语言名称
            
        Returns:
            文件扩展名
        """
        extensions = {
            'python': 'py',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'javascript': 'js',
            'typescript': 'ts',
            'html': 'html',
            'css': 'css',
            'matlab': 'm',
            'r': 'r',
            'sql': 'sql',
            'bash': 'sh',
            'shell': 'sh',
            'powershell': 'ps1',
        }
        
        return extensions.get(language.lower(), 'txt')
    
    def analyze_code(self, code: str, language: str = 'unknown') -> Dict[str, Any]:
        """
        分析代码内容
        
        Args:
            code: 代码文本
            language: 语言类型
            
        Returns:
            分析结果
        """
        analysis = {
            "language": language,
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            "has_functions": False,
            "has_classes": False,
            "has_imports": False,
            "has_comments": False
        }
        
        # 检测代码特征
        if language == 'python':
            analysis["has_functions"] = 'def ' in code
            analysis["has_classes"] = 'class ' in code
            analysis["has_imports"] = 'import ' in code or 'from ' in code
            analysis["has_comments"] = '#' in code
        elif language in ['java', 'cpp', 'c', 'javascript']:
            analysis["has_functions"] = re.search(r'\w+\s*\([^)]*\)\s*\{', code) is not None
            analysis["has_classes"] = 'class ' in code
            analysis["has_imports"] = '#include' in code or 'import ' in code
            analysis["has_comments"] = '//' in code or '/*' in code
        
        return analysis
