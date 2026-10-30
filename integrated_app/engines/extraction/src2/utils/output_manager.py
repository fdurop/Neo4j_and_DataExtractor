"""
输出管理器
统一管理所有类型数据的输出
"""

import os
import json
import csv
import shutil
from typing import Dict, Any, List
from datetime import datetime
from PIL import Image

from .knowledge_exporter import KnowledgeExporter
from .path_config import extraction_path, resolve_workspace_path, workspace_relative


def _relpath(value):
    return workspace_relative(value)


_PATH_KEYS = {
    "path", "paths", "file_path", "source_path", "source_image",
    "original_path", "enhanced_path", "image_path", "image_paths",
    "page_image_path", "review_image_path", "metadata_path", "matrix_path",
    "recognition_image_path",
    "img_path",
}


def _normalize_paths(value, key=None):
    if isinstance(value, dict):
        return {item_key: _normalize_paths(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_paths(item, key) for item in value]
    if isinstance(value, str):
        # Content that looks like a filesystem path is still source evidence.
        # Normalize explicit path fields only, matching the formal exporter.
        if key in _PATH_KEYS:
            return _relpath(value)
    return value


class OutputManager:
    """统一的输出管理器"""
    
    def __init__(
        self,
        base_dir: str = extraction_path("output"),
        logger=None,
        user_id: str = None,
        username: str = None,
        course_id: str = None,
        course_name: str = None,
        job_id: str = None,
    ):
        """
        初始化输出管理器
        
        Args:
            base_dir: 输出基础目录
            logger: 日志记录器
        """
        self.base_dir = base_dir
        self.debug_dir = os.path.join(base_dir, "debug")
        self.kg_dir = os.path.join(base_dir, "kg_data")
        self.ex_items_dir = os.path.join(base_dir, "ex_items")
        self.formula_review_dir = os.path.join(base_dir, "formula_review_items")
        self.table_review_dir = os.path.join(base_dir, "table_review_items")
        self.logger = logger
        self.knowledge_exporter = KnowledgeExporter(
            base_dir=self.kg_dir,
            logger=logger,
            user_id=user_id,
            username=username,
            course_id=course_id,
            course_name=course_name,
            job_id=job_id,
        )
        
        # 创建输出目录结构
        self.dirs = {
            "text": os.path.join(self.debug_dir, "text"),
            "images": os.path.join(self.debug_dir, "images"),
            "formulas": os.path.join(self.debug_dir, "formulas"),
            "tables": os.path.join(self.debug_dir, "tables"),
            "code": os.path.join(self.debug_dir, "code"),
            "logs": os.path.join(self.debug_dir, "logs"),
            "document_metadata": os.path.join(self.debug_dir, "document_metadata"),
            "ex_items": self.ex_items_dir,
            "formula_review_items": self.formula_review_dir,
            "table_review_items": self.table_review_dir,
        }
        
        self._create_directories()
    
    def _create_directories(self):
        """创建所有输出目录"""
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def load_existing(self) -> Dict[str, int]:
        """Load the formal checkpoint and return restored item counts."""
        return self.knowledge_exporter.load_existing()
    
    def save_text(self, data: Dict[str, Any], filename: str, page_num: int):
        """
        保存文本数据
        
        Args:
            data: 文本数据字典
            filename: 文件名（不含扩展名）
            page_num: 页码
        """
        try:
            output_file = os.path.join(
                self.dirs["text"],
                f"{filename}_page_{page_num+1}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(data), f, ensure_ascii=False, indent=2)
            self.knowledge_exporter.add_text(data, filename, page_num)
            
            if self.logger:
                self.logger.debug(f"保存文本: {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存文本失败: {e}")
    
    def save_image_metadata(self, data: Dict[str, Any], filename: str, page_num: int, img_index: int):
        """
        保存图像元数据
        
        Args:
            data: 图像元数据字典
            filename: 文件名
            page_num: 页码
            img_index: 图像索引
        """
        try:
            # 使用与图片一致的命名格式，确保排序时在一起
            output_file = os.path.join(
                self.dirs["images"],
                f"{filename}_slide_{page_num+1}_img_{img_index+1}_metadata.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(data), f, ensure_ascii=False, indent=2)
            if data.get("review_required"):
                self._save_review_item(data, filename, page_num, img_index)
            self.knowledge_exporter.add_image(data, filename, page_num, img_index)
            
            if self.logger:
                self.logger.debug(f"保存图像元数据: {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存图像元数据失败: {e}")

    def _save_review_item(self, data: Dict[str, Any], filename: str, page_num: int, img_index: int):
        """Save an image and metadata in the legacy frontend's ex_items format."""
        base_name = f"{filename}_slide_{page_num+1}_img_{img_index+1}"
        preview_file = os.path.join(self.ex_items_dir, f"{base_name}_enhanced.png")
        metadata_file = os.path.join(self.ex_items_dir, f"{base_name}_metadata.json")

        source_path = data.get("enhanced_path") or data.get("original_path") or data.get("image_path")
        if source_path and not os.path.isabs(str(source_path)):
            source_path = str(resolve_workspace_path(source_path))

        preview_saved = False
        if source_path and os.path.exists(source_path):
            try:
                with Image.open(source_path) as image:
                    if image.mode == "RGBA":
                        background = Image.new("RGB", image.size, "white")
                        background.paste(image, mask=image.getchannel("A"))
                        image = background
                    elif image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(preview_file, format="PNG")
                preview_saved = True
            except Exception:
                try:
                    if str(source_path).lower().endswith(".png"):
                        shutil.copy2(source_path, preview_file)
                        preview_saved = True
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(f"待审核图片预览保存失败: {exc}")

        review_data = dict(data)
        review_data.update({
            "review_required": True,
            "review_status": data.get("review_status") or "pending",
            "review_reason": data.get("review_reason") or "unknown",
            "source_document": filename,
            "page_num": page_num + 1,
            "image_index": img_index + 1,
            "review_image_path": _relpath(preview_file) if preview_saved else "",
        })
        if preview_saved:
            review_data["enhanced_path"] = _relpath(preview_file)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(_normalize_paths(review_data), f, ensure_ascii=False, indent=2)

        if self.logger:
            self.logger.warning(
                "图片进入人工审核队列: %s reason=%s",
                metadata_file,
                review_data["review_reason"],
            )
    
    def save_formulas(self, formulas: List[Dict[str, Any]], filename: str, page_num: int):
        """
        保存公式数据（同时保存JSON和CSV）
        
        Args:
            formulas: 公式列表
            filename: 文件名
            page_num: 页码
        """
        if not formulas:
            return
        
        try:
            # 1. 保存JSON格式
            json_file = os.path.join(
                self.dirs["formulas"],
                f"{filename}_page_{page_num+1}_formulas.json"
            )
            
            normalized_formulas = []
            for idx, formula in enumerate(formulas):
                item = dict(formula)
                item.setdefault("formula_id", f"{filename}_page_{page_num+1}_formula_{idx+1}")
                item.setdefault("page_num", page_num + 1)
                normalized_formulas.append(item)

            output_data = {
                "page": page_num + 1,
                "formula_count": len(normalized_formulas),
                "accepted_count": sum(not item.get("review_required") for item in normalized_formulas),
                "review_count": sum(bool(item.get("review_required")) for item in normalized_formulas),
                "formulas": normalized_formulas,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(output_data), f, ensure_ascii=False, indent=2)
            
            # 2. 保存CSV格式
            csv_file = os.path.join(
                self.dirs["formulas"],
                f"{filename}_page_{page_num+1}_formulas.csv"
            )
            
            with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                # 定义CSV字段
                fieldnames = [
                    '序号', 'LaTeX公式', '名称', '描述', '来源图像', '提取方法',
                    '置信度', '验证状态', '是否需审核'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for idx, formula in enumerate(normalized_formulas, 1):
                    writer.writerow({
                        '序号': idx,
                        'LaTeX公式': formula.get('latex', ''),
                        '名称': formula.get('name', ''),
                        '描述': formula.get('description', ''),
                        '来源图像': _relpath(formula.get('source_image', '')),
                        '提取方法': formula.get('extraction_method', ''),
                        '置信度': formula.get('confidence', ''),
                        '验证状态': formula.get('validation_status', ''),
                        '是否需审核': formula.get('review_required', False),
                    })

            for idx, formula in enumerate(normalized_formulas):
                if formula.get("review_required"):
                    self._save_modality_review_item(
                        formula, self.formula_review_dir, "formula", filename, page_num, idx
                    )

            accepted = [item for item in normalized_formulas if not item.get("review_required")]
            
            if self.logger:
                self.logger.info(
                    f"保存 {len(normalized_formulas)} 个公式，正式={len(accepted)}，"
                    f"待审核={len(normalized_formulas) - len(accepted)}: JSON={json_file}, CSV={csv_file}"
                )
            if accepted:
                self.knowledge_exporter.add_formulas(accepted, filename, page_num)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存公式失败: {e}")
    
    def save_tables(self, tables: List[Dict[str, Any]], filename: str, page_num: int):
        """
        保存表格数据（同时保存JSON和CSV）
        
        Args:
            tables: 表格列表
            filename: 文件名
            page_num: 页码
        """
        if not tables:
            return

        try:
            normalized_tables = []
            for table_idx, original_table in enumerate(tables):
                table = dict(original_table)
                table.setdefault("table_id", f"{filename}_page_{page_num+1}_table_{table_idx+1}")
                table.setdefault("page_num", page_num + 1)
                normalized_tables.append(table)
                # 1. 保存JSON格式
                json_file = os.path.join(
                    self.dirs["tables"],
                    f"{filename}_page_{page_num+1}_table_{table_idx+1}.json"
                )
                
                # 移除DataFrame对象（不能序列化）
                table_copy = table.copy()
                if 'dataframe' in table_copy:
                    del table_copy['dataframe']
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(_normalize_paths(table_copy), f, ensure_ascii=False, indent=2)
                
                # 2. 保存CSV格式
                csv_file = os.path.join(
                    self.dirs["tables"],
                    f"{filename}_page_{page_num+1}_table_{table_idx+1}.csv"
                )
                
                if table.get("csv"):
                    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                        f.write(str(table["csv"]))
                elif table.get("json"):
                    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                        fieldnames = list(table['json'][0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(table['json'])

                if table.get("review_required"):
                    self._save_modality_review_item(
                        table, self.table_review_dir, "table", filename, page_num, table_idx
                    )
            
            if self.logger:
                self.logger.info(f"保存 {len(tables)} 个表格 (JSON + CSV)")
            accepted = [table for table in normalized_tables if not table.get("review_required")]
            if accepted:
                self.knowledge_exporter.add_tables(accepted, filename, page_num)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存表格失败: {e}")

    def _save_modality_review_item(
        self,
        item: Dict[str, Any],
        review_dir: str,
        kind: str,
        filename: str,
        page_num: int,
        index: int,
    ):
        """Persist one formula/table review item and an optional PNG preview."""
        base_name = f"{filename}_page_{page_num+1}_{kind}_{index+1}"
        metadata_file = os.path.join(review_dir, f"{base_name}.json")
        preview_file = os.path.join(review_dir, f"{base_name}.png")
        source_path = item.get("source_image")
        if source_path and not os.path.isabs(str(source_path)):
            source_path = str(resolve_workspace_path(source_path))

        review_item = dict(item)
        if source_path and os.path.exists(source_path):
            try:
                with Image.open(source_path) as image:
                    if image.mode == "RGBA":
                        background = Image.new("RGB", image.size, "white")
                        background.paste(image, mask=image.getchannel("A"))
                        image = background
                    elif image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(preview_file, format="PNG")
                review_item["review_image_path"] = _relpath(preview_file)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"{kind}待审核预览保存失败: {exc}")

        review_item.update({
            "review_required": True,
            "review_status": item.get("review_status") or "pending",
            "source_document": filename,
            "page_num": page_num + 1,
        })
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(_normalize_paths(review_item), f, ensure_ascii=False, indent=2, default=str)
    
    def save_code(self, code_blocks: List[Dict[str, Any]], filename: str, page_num: int):
        """
        保存代码数据（只保存2个文件：纯源代码文件 + JSON元数据）
        
        Args:
            code_blocks: 代码块列表
            filename: 文件名
            page_num: 页码
        """
        if not code_blocks:
            return

        try:
            start_index = self.knowledge_exporter.next_code_index(filename, page_num)
            for idx, code_block in enumerate(code_blocks):
                absolute_index = start_index + idx
                language = code_block.get('language', 'txt').lower()
                
                # 根据语言确定文件扩展名
                language_extensions = {
                    "python": ".py",
                    "java": ".java",
                    "javascript": ".js",
                    "c++": ".cpp",
                    "cpp": ".cpp",
                    "c": ".c",
                    "go": ".go",
                    "rust": ".rs",
                    "typescript": ".ts",
                    "php": ".php",
                    "ruby": ".rb",
                    "swift": ".swift",
                    "kotlin": ".kt",
                    "matlab": ".m",
                    "r": ".r",
                    "sql": ".sql",
                    "html": ".html",
                    "css": ".css",
                    "arduino": ".ino",
                    "ino": ".ino",
                    "shell": ".sh",
                    "bash": ".sh",
                    "txt": ".txt"
                }
                
                # 如果语言无法识别或为unknown，统一使用.txt
                if language == "unknown" or language not in language_extensions:
                    ext = ".txt"
                else:
                    ext = language_extensions.get(language, ".txt")
                
                # 1. 保存纯源代码文件（不添加任何额外内容）
                code_file = os.path.join(
                    self.dirs["code"],
                    f"{filename}_page_{page_num+1}_code_{absolute_index+1}{ext}"
                )
                
                # 只写入纯代码，不添加注释或其他内容
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code_block.get('code', ''))
                
                # 2. 保存JSON元数据（包含所有辅助信息）
                metadata_file = os.path.join(
                    self.dirs["code"],
                    f"{filename}_page_{page_num+1}_code_{absolute_index+1}_metadata.json"
                )
                
                metadata = {
                    "source_image": _relpath(code_block.get('source_image', '')),
                    "extraction_method": code_block.get('extraction_method', ''),
                    "language": language,
                    "description": code_block.get('description', ''),
                    "code_file": os.path.basename(code_file),
                    "page": page_num + 1,
                    "index": absolute_index + 1,
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": code_block.get('raw_response', ''),
                    "backend": code_block.get('backend', ''),
                    "provider": code_block.get('provider', ''),
                    "model": code_block.get('model', ''),
                }
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(_normalize_paths(metadata), f, ensure_ascii=False, indent=2)
                
                if self.logger:
                    self.logger.info(f"保存代码: {os.path.basename(code_file)} + metadata.json")
            self.knowledge_exporter.add_code(
                code_blocks,
                filename,
                page_num,
                start_index=start_index,
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存代码失败: {e}")
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str):
        """
        保存文档元数据
        
        Args:
            metadata: 元数据字典
            filename: 文件名
        """
        try:
            output_file = os.path.join(self.dirs["document_metadata"], f"{filename}_metadata.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(metadata), f, ensure_ascii=False, indent=2)
            self.knowledge_exporter.register_document(filename, metadata)
            # Refresh structured JSON after each document. Paid embeddings are
            # generated once, after every input document has been processed.
            self.knowledge_exporter.finalize(build_vectors=False)
            
            if self.logger:
                self.logger.info(f"保存元数据: {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存元数据失败: {e}")
    
    def save_professional_terms(self, terms: set, filename: str = "professional_terms_library"):
        """
        保存专业术语库
        
        Args:
            terms: 术语集合
            filename: 文件名
        """
        try:
            output_file = os.path.join(self.debug_dir, f"{filename}.json")
            
            terms_data = {
                "total_terms": len(terms),
                "terms": sorted(list(terms)),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(terms_data, f, ensure_ascii=False, indent=2)
            
            if self.logger:
                self.logger.info(f"保存专业术语库: {output_file} ({len(terms)} 个术语)")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存专业术语库失败: {e}")
    
    def _get_code_extension(self, language: str) -> str:
        """获取代码文件扩展名"""
        extensions = {
            'python': 'py',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'javascript': 'js',
            'typescript': 'ts',
            'matlab': 'm',
            'r': 'r',
            'sql': 'sql',
        }
        return extensions.get(language.lower(), 'txt')
    
    def finalize(self):
        """Write the final handoff files and build embeddings exactly once."""
        self.knowledge_exporter.finalize(build_vectors=True)

    def get_output_summary(self) -> Dict[str, Any]:
        """
        获取输出摘要
        
        Returns:
            输出统计信息
        """
        summary = {}
        
        for name, dir_path in self.dirs.items():
            try:
                files = os.listdir(dir_path)
                summary[name] = {
                    "count": len(files),
                    "path": dir_path
                }
            except Exception:
                summary[name] = {
                    "count": 0,
                    "path": dir_path
                }
        summary["kg_data"] = {
            "count": len(os.listdir(self.kg_dir)) if os.path.exists(self.kg_dir) else 0,
            "path": self.kg_dir,
        }
        summary["debug"] = {
            "count": len(os.listdir(self.debug_dir)) if os.path.exists(self.debug_dir) else 0,
            "path": self.debug_dir,
        }
        return summary
