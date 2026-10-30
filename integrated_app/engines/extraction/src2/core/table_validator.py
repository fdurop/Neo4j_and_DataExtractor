"""Normalization, quality scoring, and deduplication for extracted tables."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List


class TableValidator:
    """Convert table candidates to one schema and reject weak structures."""

    def normalize(self, table: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(table)
        headers = item.get("headers") or []
        cells = item.get("cells") or item.get("data") or []

        if isinstance(headers, str):
            headers = [part.strip() for part in re.split(r"[,，|]", headers) if part.strip()]
        if headers and isinstance(headers, list) and not isinstance(headers[0], list):
            headers = [headers]

        if not cells and isinstance(item.get("json"), list) and item["json"]:
            records = item["json"]
            if isinstance(records[0], dict):
                keys = list(records[0].keys())
                if not headers:
                    headers = [keys]
                cells = [[record.get(key, "") for key in keys] for record in records]

        normalized_cells = []
        for row in cells if isinstance(cells, list) else []:
            if isinstance(row, dict):
                row = list(row.values())
            if isinstance(row, (list, tuple)):
                normalized_cells.append(["" if value is None else str(value).strip() for value in row])

        normalized_headers = []
        for row in headers if isinstance(headers, list) else []:
            if isinstance(row, (list, tuple)):
                normalized_headers.append(["" if value is None else str(value).strip() for value in row])

        target_cols = max([len(row) for row in normalized_headers + normalized_cells] or [0])
        merged_cells = item.get("merged_cells") if isinstance(item.get("merged_cells"), list) else []
        auto_repairs = []
        for row_index, row in enumerate(normalized_headers):
            if len(row) == target_cols:
                continue
            spans = sorted(
                [
                    span for span in merged_cells
                    if isinstance(span, (list, tuple)) and len(span) == 4
                    and span[0] == row_index and span[2] == row_index
                ],
                key=lambda span: span[1],
            )
            labels = [value for value in row if value]
            if spans and len(labels) == len(spans) and all(0 <= span[1] < target_cols for span in spans):
                expanded = [""] * target_cols
                for label, span in zip(labels, spans):
                    expanded[span[1]] = label
                normalized_headers[row_index] = expanded
                auto_repairs.append({"type": "expanded_merged_header", "row": row_index})

        item["headers"] = normalized_headers
        item["cells"] = normalized_cells
        item["merged_cells"] = merged_cells
        item["auto_repairs"] = auto_repairs
        item["rows"] = len(normalized_cells)
        item["cols"] = max(
            [len(row) for row in normalized_headers + normalized_cells] or [0]
        )
        item["title"] = str(item.get("title") or item.get("description") or "").strip()
        item["description"] = str(item.get("description") or item.get("title") or "").strip()
        item["units"] = item.get("units") if isinstance(item.get("units"), dict) else {}
        item["footnotes"] = item.get("footnotes") if isinstance(item.get("footnotes"), list) else []
        item["uncertain_cells"] = (
            item.get("uncertain_cells") if isinstance(item.get("uncertain_cells"), list) else []
        )
        item.setdefault("source_type", item.get("type") or item.get("extraction_method") or "unknown")
        item.setdefault("extraction_method", item.get("method") or item["source_type"])
        item["json"] = self._records(normalized_headers, normalized_cells)
        item["csv"] = self._csv_text(normalized_headers, normalized_cells)
        return item

    def validate(self, table: Dict[str, Any]) -> Dict[str, Any]:
        item = self.normalize(table)
        rows = item["cells"]
        header_rows = item["headers"]
        all_rows = header_rows + rows
        issues: List[str] = []

        if item.get("review_required") and item.get("review_reason"):
            issues.append(str(item["review_reason"]))

        if len(rows) < 1 or item["cols"] < 2:
            issues.append("table_too_small")
        lengths = [len(row) for row in all_rows if row]
        consistency = 1.0 if not lengths else sum(length == item["cols"] for length in lengths) / len(lengths)
        item["row_lengths"] = lengths
        values = [cell for row in all_rows for cell in row]
        non_empty_ratio = sum(bool(str(cell).strip()) for cell in values) / max(1, len(values))
        if consistency < 0.85:
            issues.append("inconsistent_columns")
        elif any(length != item["cols"] for length in lengths):
            issues.append("ragged_rows")
        if non_empty_ratio < 0.45:
            issues.append("too_many_empty_cells")
        if item["uncertain_cells"]:
            issues.append("uncertain_cells")

        if item.get("backend") == "api":
            visible_count = item.get("visible_table_count")
            extracted_count = item.get("extracted_table_count")
            if item.get("coverage_complete") is not True:
                issues.append("table_coverage_unverified")
            try:
                visible_count = int(visible_count)
                extracted_count = int(extracted_count)
                if visible_count != extracted_count:
                    issues.append("table_coverage_mismatch")
                if visible_count > 1:
                    issues.append("multiple_tables_require_review")
            except (TypeError, ValueError):
                issues.append("table_count_unverified")

        source_confidence = item.get("accuracy", item.get("confidence"))
        try:
            source_confidence = float(source_confidence) / 100.0 if float(source_confidence) > 1 else float(source_confidence)
        except (TypeError, ValueError):
            source_confidence = 0.86 if item.get("backend") != "api" else 0.82
        score = 0.35 * consistency + 0.30 * non_empty_ratio + 0.35 * source_confidence
        score -= 0.18 * len(issues)
        score = max(0.0, min(1.0, score))
        pass_threshold = 0.90 if item.get("backend") == "api" else 0.70
        review_required = bool(issues) or score < pass_threshold
        item.update(
            {
                "confidence": round(score, 4),
                "validation_issues": issues,
                "validation_status": "review" if review_required else "passed",
                "review_required": review_required,
                "review_reason": issues[0] if issues else ("low_confidence" if review_required else None),
                "review_status": "pending" if review_required else None,
                "indexable": bool(rows) and not review_required,
            }
        )
        return item

    def validate_many(self, tables: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for table in tables or []:
            item = self.validate(table)
            duplicate_index = self._find_duplicate(selected, item)
            if duplicate_index is None:
                selected.append(item)
            elif item["confidence"] > selected[duplicate_index]["confidence"]:
                selected[duplicate_index] = item
        return selected

    def _find_duplicate(self, selected: List[Dict[str, Any]], candidate: Dict[str, Any]):
        candidate_tokens = self._tokens(candidate)
        for index, existing in enumerate(selected):
            existing_tokens = self._tokens(existing)
            union = candidate_tokens | existing_tokens
            similarity = len(candidate_tokens & existing_tokens) / len(union) if union else 0.0
            if similarity >= 0.88:
                return index
        return None

    @staticmethod
    def _tokens(item: Dict[str, Any]):
        raw = json.dumps([item.get("headers"), item.get("cells")], ensure_ascii=False)
        return set(re.findall(r"[A-Za-z0-9_.%+-]+|[\u4e00-\u9fff]{1,4}", raw.lower()))

    @staticmethod
    def _records(headers: List[List[str]], cells: List[List[str]]) -> List[Dict[str, str]]:
        if not cells:
            return []
        columns = headers[-1] if headers else [f"column_{i+1}" for i in range(max(map(len, cells)))]
        columns = [column or f"column_{i+1}" for i, column in enumerate(columns)]
        return [
            {columns[i] if i < len(columns) else f"column_{i+1}": value for i, value in enumerate(row)}
            for row in cells
        ]

    @staticmethod
    def _csv_text(headers: List[List[str]], cells: List[List[str]]) -> str:
        import csv
        import io

        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(headers + cells)
        return stream.getvalue()
