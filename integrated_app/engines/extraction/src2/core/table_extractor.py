"""
表格提取模块
功能：从PDF、PPTX中提取表格数据
"""

import re
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from .table_validator import TableValidator
from .table_image_preprocessor import TableImagePreprocessor


class TableExtractor:
    def __init__(self, logger=None, vlm_client=None):
        """
        初始化表格提取器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.vlm_client = vlm_client
        self.validator = TableValidator()
        self.image_preprocessor = TableImagePreprocessor()
    
    def extract_tables_from_pdf_page(self, page, page_text: str = "", method: str = "auto") -> List[Dict[str, Any]]:
        """
        从PDF页面提取表格
        
        Args:
            page: PyMuPDF页面对象
            page_text: 页面文本
            method: 提取方法 (auto/camelot/pdfplumber/text)
            
        Returns:
            表格列表
        """
        candidates = []

        if method == "auto":
            candidates.extend(self._extract_with_pymupdf(page))
        if method in {"auto", "camelot"}:
            candidates.extend(self._extract_with_camelot(page))
        if method in {"auto", "pdfplumber"}:
            candidates.extend(self._extract_with_pdfplumber(page))
        if method in {"auto", "text"} and not candidates:
            candidates.extend(self._extract_from_text(page_text))

        self._mark_diagonal_text_overlaps(page, candidates)
        validated = self.validator.validate_many(candidates)
        accepted = [table for table in validated if not table.get("review_required")]
        return accepted or validated
    
    def extract_tables_from_pptx_shape(self, shape) -> Optional[Dict[str, Any]]:
        """
        从PPTX形状中提取表格
        
        Args:
            shape: python-pptx表格形状
            
        Returns:
            表格数据字典
        """
        try:
            if not hasattr(shape, 'table'):
                return None
            
            table = shape.table
            rows = len(table.rows)
            cols = len(table.columns)
            
            # 提取表格数据
            data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    row_data.append(cell_text)
                data.append(row_data)
            
            # Preserve the original cell matrix; use the first row as the
            # default header while retaining the raw matrix for review.
            if data:
                if len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                else:
                    df = pd.DataFrame(data)
                
                return {
                    "type": "pptx_table",
                    "rows": rows,
                    "cols": cols,
                    "data": data,
                    "headers": [data[0]] if len(data) > 1 else [],
                    "cells": data[1:] if len(data) > 1 else data,
                    "raw_cells": data,
                    "source_type": "pptx_native_table",
                    "extraction_method": "python_pptx",
                    "dataframe": df,
                    "csv": df.to_csv(index=False),
                    "json": df.to_dict(orient='records')
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"PPTX表格提取失败: {e}")
            return None

    def should_analyze_image(self, description: str) -> bool:
        description = str(description or "")
        keywords = ("表格", "数据表", "统计表", "对比表", "参数表", "真值表", "table", "tabular")
        return any(keyword.lower() in description.lower() for keyword in keywords)

    def extract_tables_from_vlm_image(
        self,
        image_path: str,
        context: str = "",
        image_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.vlm_client:
            return []
        candidates = []
        last_error = None
        try:
            for attempt, variant in enumerate(self.image_preprocessor.variants(image_path), 1):
                try:
                    response = self.vlm_client.recognize_table(variant.path, context=context)
                except Exception as exc:
                    last_error = exc
                    # A rotated copy cannot repair a network/API failure. Stop here so
                    # one timeout does not multiply into three paid retry sequences.
                    break
                if not response.get("has_table", True):
                    continue

                raw_tables = response.get("tables")
                if not isinstance(raw_tables, list):
                    # Backward compatibility for responses produced by the old prompt.
                    raw_tables = [response]
                raw_tables = [item for item in raw_tables if isinstance(item, dict)]
                visible_count = response.get("visible_table_count")
                coverage_complete = response.get("coverage_complete")
                attempt_tables = []
                for region_index, raw_table in enumerate(raw_tables):
                    table = self.validator.validate({
                        **raw_table,
                        "source_image": image_path,
                        "recognition_image_path": variant.path,
                        "source_type": "image_vlm_table",
                        "extraction_method": f"{getattr(self.vlm_client, 'provider', 'vlm')}_api_table",
                        "backend": "api",
                        "provider": getattr(self.vlm_client, "provider", None),
                        "model": getattr(self.vlm_client, "model", None),
                        "image_index": image_index,
                        "table_region_index": region_index + 1,
                        "visible_table_count": visible_count,
                        "extracted_table_count": len(raw_tables),
                        "coverage_complete": coverage_complete,
                        "raw_response": response.get("raw_response", ""),
                        "preprocessing": {
                            "rotation": variant.rotation,
                            "scale": variant.scale,
                            "attempt": attempt,
                        },
                    })
                    candidates.append(table)
                    attempt_tables.append(table)
                if attempt_tables and all(not table.get("review_required") for table in attempt_tables):
                    if self.logger:
                        self.logger.info(
                            "表格图像识别通过: source=%s tables=%s rotation=%s scale=%s attempt=%s",
                            image_path,
                            len(attempt_tables),
                            variant.rotation,
                            variant.scale,
                            attempt,
                        )
                    return attempt_tables
                review_issues = {
                    issue
                    for table in attempt_tables
                    for issue in table.get("validation_issues", [])
                }
                if attempt_tables and review_issues == {"multiple_tables_require_review"}:
                    # The extraction is structurally complete; rotation would only
                    # repeat a paid request. Keep all regions for teacher review.
                    return attempt_tables

            if candidates:
                # Keep every region from the strongest preprocessing attempt so
                # teacher review can repair an omitted or uncertain table.
                attempt_numbers = {
                    (item.get("preprocessing") or {}).get("attempt", 1)
                    for item in candidates
                }
                best_attempt = max(
                    attempt_numbers,
                    key=lambda number: (
                        max(
                            int(item.get("extracted_table_count") or 0)
                            for item in candidates
                            if (item.get("preprocessing") or {}).get("attempt", 1) == number
                        ),
                        max(
                            float(item.get("confidence") or 0.0)
                            for item in candidates
                            if (item.get("preprocessing") or {}).get("attempt", 1) == number
                        ),
                        -number,
                    ),
                )
                return [
                    item for item in candidates
                    if (item.get("preprocessing") or {}).get("attempt", 1) == best_attempt
                ]
        except Exception as exc:
            last_error = exc

        if last_error:
            if self.logger:
                self.logger.error(f"表格视觉识别失败: {last_error}")
            return [{
                "source_image": image_path,
                "source_type": "image_vlm_table",
                "extraction_method": "vlm_table",
                "headers": [],
                "cells": [],
                "api_error": str(last_error),
                "review_required": True,
                "review_reason": "api_failed",
                "image_index": image_index,
            }]
        return [{
            "source_image": image_path,
            "source_type": "image_vlm_table",
            "extraction_method": "vlm_table_preprocessed",
            "headers": [],
            "cells": [],
            "review_required": True,
            "review_reason": "table_not_detected_after_rotation",
            "image_index": image_index,
        }]

    def finalize_page(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.validator.validate_many(tables)

    def _extract_with_pymupdf(self, page) -> List[Dict[str, Any]]:
        """Use PyMuPDF's native table finder when available."""
        try:
            finder = page.find_tables()
            result = []
            for table in getattr(finder, "tables", []):
                matrix = table.extract() or []
                if not matrix:
                    continue
                result.append({
                    "type": "pymupdf",
                    "source_type": "pdf_native_table",
                    "extraction_method": "pymupdf_find_tables",
                    "headers": [matrix[0]] if len(matrix) > 1 else [],
                    "cells": matrix[1:] if len(matrix) > 1 else matrix,
                    "raw_cells": matrix,
                    "bbox": list(table.bbox) if getattr(table, "bbox", None) else None,
                    "bbox_coordinate_system": "page_top_left",
                    "content_verified": False,
                })
            return result
        except Exception as exc:
            if self.logger:
                self.logger.debug(f"PyMuPDF表格提取失败: {exc}")
            return []

    def _mark_diagonal_text_overlaps(self, page, candidates):
        """Keep possible overlay contamination for review, without deleting text."""
        try:
            diagonal_lines = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    dx, dy = line.get("dir", (1.0, 0.0))
                    # Horizontal and vertical labels are ordinary table content.
                    if abs(dx) <= 0.1 or abs(dy) <= 0.1:
                        continue
                    text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if text.strip() and len(line.get("bbox", ())) == 4:
                        diagonal_lines.append({
                            "text": text,
                            "bbox": list(line["bbox"]),
                            "direction": [dx, dy],
                            "source_page_no": page.number + 1,
                        })
        except Exception as exc:
            if self.logger:
                self.logger.debug(f"PDF斜向文字风险检查失败: {exc}")
            return

        for candidate in candidates:
            if candidate.get("source_type") != "pdf_native_table":
                continue
            bbox = candidate.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            overlaps = [
                line for line in diagonal_lines
                if min(bbox[2], line["bbox"][2]) > max(bbox[0], line["bbox"][0])
                and min(bbox[3], line["bbox"][3]) > max(bbox[1], line["bbox"][1])
            ]
            if overlaps:
                candidate["review_required"] = True
                candidate["review_reason"] = "diagonal_text_overlaps_table"
                candidate["diagonal_text_overlaps"] = overlaps
    
    def _extract_with_camelot(self, page) -> List[Dict[str, Any]]:
        """使用Camelot提取表格"""
        tmp_name = None
        try:
            import camelot
            import tempfile
            import fitz

            fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            doc = fitz.open()
            doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
            doc.save(tmp_name)
            doc.close()

            result = []
            for flavor in ("lattice", "stream"):
                try:
                    tables = camelot.read_pdf(tmp_name, pages='1', flavor=flavor)
                    for table in tables:
                        matrix = table.df.fillna("").astype(str).values.tolist()
                        source_bbox = list(getattr(table, "_bbox", [])) or None
                        page_height = page.rect.height
                        bbox = (
                            [source_bbox[0], page_height - source_bbox[3],
                             source_bbox[2], page_height - source_bbox[1]]
                            if source_bbox and len(source_bbox) == 4 else None
                        )
                        result.append({
                            "type": "camelot",
                            "source_type": "pdf_native_table",
                            "extraction_method": f"camelot_{flavor}",
                            "method": flavor,
                            "accuracy": table.accuracy,
                            "headers": [matrix[0]] if len(matrix) > 1 else [],
                            "cells": matrix[1:] if len(matrix) > 1 else matrix,
                            "raw_cells": matrix,
                            "bbox": bbox,
                            "bbox_coordinate_system": "page_top_left",
                            "source_bbox": source_bbox,
                            "source_bbox_coordinate_system": "bottom_left",
                        })
                except Exception as exc:
                    if self.logger:
                        self.logger.debug(f"Camelot {flavor}提取失败: {exc}")
            return result
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Camelot提取失败: {e}")
            return []
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    
    def _extract_with_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """使用pdfplumber提取表格"""
        tmp_name = None
        try:
            import pdfplumber
            import tempfile
            import fitz

            fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            single_doc = fitz.open()
            single_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
            single_doc.save(tmp_name)
            single_doc.close()

            with pdfplumber.open(tmp_name) as pdf:
                tables = pdf.pages[0].find_tables()
                result = []
                for table in tables:
                    matrix = table.extract()
                    if matrix:
                        result.append({
                            "type": "pdfplumber",
                            "source_type": "pdf_native_table",
                            "extraction_method": "pdfplumber",
                            "headers": [matrix[0]] if len(matrix) > 1 else [],
                            "cells": matrix[1:] if len(matrix) > 1 else matrix,
                            "raw_cells": matrix,
                            "bbox": list(table.bbox),
                            "bbox_coordinate_system": "page_top_left",
                            "accuracy": 0.86,
                        })
            return result
        except Exception as e:
            if self.logger:
                self.logger.debug(f"pdfplumber提取失败: {e}")
            return []
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    
    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中检测并提取表格"""
        tables = []
        
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            # 检测表格行（包含多个分隔符）
            if re.search(r'[\|\t]{2,}', line) or re.search(r'\s{3,}', line):
                table_lines.append(line)
                in_table = True
            else:
                if in_table and len(table_lines) >= 2:
                    # 处理检测到的表格
                    table_data = self._parse_text_table(table_lines)
                    if table_data:
                        tables.append(table_data)
                table_lines = []
                in_table = False
        
        # 处理文件末尾的表格
        if in_table and len(table_lines) >= 2:
            table_data = self._parse_text_table(table_lines)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _parse_text_table(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """解析文本表格"""
        try:
            # 尝试不同的分隔符
            separators = ['|', '\t', r'\s{2,}']
            
            for sep in separators:
                rows = []
                for line in lines:
                    if sep == r'\s{2,}':
                        # 使用多个空格分隔
                        cells = re.split(r'\s{2,}', line.strip())
                    else:
                        cells = [c.strip() for c in line.split(sep)]
                    
                    if len(cells) > 1:
                        rows.append(cells)
                
                if len(rows) >= 2 and len(set(len(r) for r in rows)) == 1:
                    # 所有行列数相同，可能是有效表格
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    return {
                        "type": "text_table",
                        "source_type": "pdf_text_table",
                        "extraction_method": "text_spacing_rule",
                        "separator": sep,
                        "headers": [rows[0]],
                        "cells": rows[1:],
                        "dataframe": df,
                        "csv": df.to_csv(index=False),
                        "json": df.to_dict(orient='records')
                    }
            
            return None
        except Exception as e:
            if self.logger:
                self.logger.debug(f"文本表格解析失败: {e}")
            return None
