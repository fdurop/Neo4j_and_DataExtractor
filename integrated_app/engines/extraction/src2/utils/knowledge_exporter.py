"""
Unified knowledge export for multimodal GraphRAG.

The extractor still writes its original per-modality files. This exporter adds a
stable handoff layer for graph construction and vector retrieval:

- course_manifest.json
- documents.json
- pages.json
- content_list.json
- multimodal_nodes.json
- vectors/vector_index.json
- vectors/vector_matrix.npy
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .vector_index import build_vector_index
from .embedding_text_cleaner import clean_embedding_text, compose_embedding_text
from .path_config import extraction_path, workspace_relative
from .teaching_handoff import HANDOFF_FILES, HANDOFF_VERSION, build_teaching_handoff


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+", " ", json.dumps(value, ensure_ascii=False, default=str)).strip()


def _short_hash(*parts: Any) -> str:
    raw = "|".join(_clean(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, set):
        return [str(v) for v in sorted(value) if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _relpath(value: Any) -> str:
    return workspace_relative(value)


def _relpath_list(value: Any) -> List[str]:
    return [_relpath(item) for item in _safe_list(value)]


class KnowledgeExporter:
    def __init__(
        self,
        base_dir: str = extraction_path("output"),
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        course_id: Optional[str] = None,
        course_name: Optional[str] = None,
        job_id: Optional[str] = None,
        logger=None,
    ):
        self.base_dir = base_dir
        self.logger = logger
        self.user_id = user_id or os.getenv("EXTRACTION_USER_ID", "default_user")
        self.username = username or os.getenv("EXTRACTION_USERNAME", self.user_id)
        self.course_id = course_id or os.getenv("EXTRACTION_COURSE_ID", "default_course")
        self.course_name = course_name or os.getenv("EXTRACTION_COURSE_NAME", "default_course")
        self.job_id = job_id or os.getenv("EXTRACTION_JOB_ID", "")
        self.created_at = datetime.now().isoformat()

        self.documents: Dict[str, Dict[str, Any]] = {}
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.content_items: Dict[str, Dict[str, Any]] = {}
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def load_existing(self) -> Dict[str, int]:
        """Restore a document-level checkpoint from an existing run directory."""
        manifest = self._read_json("course_manifest.json", {})
        documents = self._read_json("documents.json", [])
        pages = self._read_json("pages.json", [])
        content_items = self._read_json("content_list.json", [])
        nodes = self._read_json("multimodal_nodes.json", [])

        if not isinstance(manifest, dict):
            raise ValueError("course_manifest.json must contain an object")
        for filename, value in (
            ("documents.json", documents),
            ("pages.json", pages),
            ("content_list.json", content_items),
            ("multimodal_nodes.json", nodes),
        ):
            if not isinstance(value, list):
                raise ValueError(f"{filename} must contain a list")

        # IDs depend on course_id, so restore identity before processing more files.
        self.user_id = manifest.get("user_id") or self.user_id
        self.username = manifest.get("username") or self.username
        self.course_id = manifest.get("course_id") or self.course_id
        self.course_name = manifest.get("course_name") or self.course_name
        self.job_id = manifest.get("job_id") or self.job_id
        self.created_at = manifest.get("created_at") or self.created_at

        self.documents = {
            item["document_id"]: item for item in documents if item.get("document_id")
        }
        self.pages = {item["page_id"]: item for item in pages if item.get("page_id")}
        self.content_items = {
            item["element_id"]: item for item in content_items if item.get("element_id")
        }
        self.nodes = {item["node_id"]: item for item in nodes if item.get("node_id")}

        return {
            "documents": len(self.documents),
            "pages": len(self.pages),
            "content_items": len(self.content_items),
            "nodes": len(self.nodes),
        }

    def document_id(self, filename: str) -> str:
        return f"doc_{_short_hash(self.user_id, self.course_id, self.job_id, filename)}"

    def page_id(self, filename: str, page_num: int) -> str:
        return f"page_{_short_hash(self.user_id, self.course_id, self.job_id, filename, page_num + 1)}"

    def element_id(self, filename: str, page_num: int, kind: str, index: int) -> str:
        return f"{kind}_{_short_hash(self.user_id, self.course_id, self.job_id, filename, page_num + 1, kind, index + 1)}"

    def embedding_id(self, node_id: str) -> str:
        return f"emb_{node_id}"

    def register_document(self, filename: str, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        doc_id = self.document_id(filename)
        total_pages = metadata.get("total_pages") or metadata.get("total_slides") or metadata.get("page_count")
        self.documents[doc_id] = {
            "document_id": doc_id,
            "user_id": self.user_id,
            "username": self.username,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "job_id": self.job_id,
            "file_name": metadata.get("file_name") or filename,
            "file_type": metadata.get("file_type") or metadata.get("extension") or "",
            "file_path": _relpath(metadata.get("file_path") or ""),
            "page_count": total_pages,
            "chapter_title": metadata.get("chapter_title") or filename,
            "document_order": metadata.get("document_order"),
            "metadata": self._normalize_paths(metadata),
        }
        return doc_id

    def register_page(
        self,
        filename: str,
        page_num: int,
        raw_text: Optional[str] = None,
        images: Optional[List[str]] = None,
        page_image_path: str = "",
        source_text: Optional[str] = None,
    ):
        doc_id = self.document_id(filename)
        page_id = self.page_id(filename, page_num)
        existing = self.pages.get(page_id, {})
        # None means a modality is only registering its parent page. An empty
        # string is an explicitly cleaned empty page, not a request to restore
        # uncleaned text or an earlier extraction's text.
        merged_raw_text = raw_text if raw_text is not None else existing.get("raw_text", "")
        merged_images = _relpath_list(images) if images is not None else existing.get("image_paths", [])
        merged_title = self._guess_title(merged_raw_text) if raw_text is not None else existing.get("title", "")
        merged_summary = self._summary(merged_raw_text, 220) if raw_text is not None else existing.get("page_summary", "")
        merged_page_image_path = _relpath(page_image_path) or existing.get("page_image_path", "")
        if raw_text is not None or source_text is not None:
            merged_source_text = source_text
            source_text_available = source_text is not None
        else:
            merged_source_text = existing.get("source_text")
            source_text_available = existing.get("source_text_available", False)
        self.pages[page_id] = {
            "page_id": page_id,
            "user_id": self.user_id,
            "username": self.username,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "job_id": self.job_id,
            "document_id": doc_id,
            "page_no": page_num + 1,
            "title": merged_title,
            "raw_text": merged_raw_text,
            "raw_text_policy": "cleaned_page_text_legacy_field",
            "source_text": merged_source_text,
            "source_text_available": source_text_available,
            "page_summary": merged_summary,
            "page_image_path": merged_page_image_path,
            "image_paths": merged_images,
            "prev_page_id": self.page_id(filename, page_num - 1) if page_num > 0 else existing.get("prev_page_id"),
            "next_page_id": self.page_id(filename, page_num + 1) if raw_text or images or page_image_path or not existing else existing.get("next_page_id"),
            "embedding_id": self.embedding_id(page_id),
            "embedding_text": compose_embedding_text(
                [merged_title, merged_raw_text[:800]], node_type="Page", max_chars=1000
            ),
        }
        self._add_node(
            node_id=page_id,
            node_type="Page",
            name=merged_title or f"{filename} page {page_num + 1}",
            summary=merged_summary,
            description=merged_raw_text[:1000],
            document_id=doc_id,
            page_id=page_id,
            source_path=merged_page_image_path,
            embedding_text=self.pages[page_id]["embedding_text"],
            extra={"page_no": page_num + 1},
        )
        return page_id

    def add_text(self, data: Dict[str, Any], filename: str, page_num: int):
        # A successfully cleaned empty string must stay empty. Falling back
        # with `or raw_text` reintroduced filtered headers into the formal KG.
        text = data.get("cleaned_text") if "cleaned_text" in data else data.get("raw_text", "")
        text = text if isinstance(text, str) else ""
        source_text = data.get("raw_text") if isinstance(data.get("raw_text"), str) else None
        page_id = self.register_page(
            filename,
            page_num,
            raw_text=text,
            images=data.get("image_paths") or [],
            source_text=source_text,
        )
        if not text.strip():
            # Support reprocessing an existing checkpoint whose text has now
            # been filtered out. Do not leave the old indexed TextChunk behind.
            stale_id = self.element_id(filename, page_num, "text", 0)
            self.content_items.pop(stale_id, None)
            self.nodes.pop(stale_id, None)
            return None
        element_id = self.element_id(filename, page_num, "text", 0)
        key_points = data.get("key_points") or []
        terms = data.get("technical_terms") or []
        embedding_text = compose_embedding_text(
            [text, key_points, terms], node_type="TextChunk", max_chars=1800
        )
        item = self._base_item(element_id, "text", filename, page_num, page_id)
        item.update(
            {
                "text": text,
                "raw_text": data.get("raw_text", ""),
                "source_text_available": source_text is not None,
                "text_processing": data.get("text_processing", {}),
                "key_points": key_points,
                "technical_terms": terms,
                "embedding_id": self.embedding_id(element_id),
                "embedding_text": embedding_text,
            }
        )
        self.content_items[element_id] = item
        self._add_node(
            element_id,
            "TextChunk",
            name=self._guess_title(text) or f"text page {page_num + 1}",
            summary=self._summary(text, 180),
            description=text,
            document_id=self.document_id(filename),
            page_id=page_id,
            source_path="",
            embedding_text=embedding_text,
            extra={"technical_terms": terms, "key_points": key_points},
        )
        return element_id

    def add_image(self, data: Dict[str, Any], filename: str, page_num: int, img_index: int):
        element_id = self.element_id(filename, page_num, "figure", img_index)
        if data.get("review_required"):
            # Failed re-extraction must not leave an older description indexed.
            self.content_items.pop(element_id, None)
            self.nodes.pop(element_id, None)
            if self.logger:
                self.logger.warning(
                    "Skip review-required image node: %s page=%s image=%s reason=%s",
                    filename,
                    page_num + 1,
                    img_index + 1,
                    data.get("review_reason"),
                )
            return None
        page_id = self.register_page(filename, page_num)
        description = _clean(data.get("description"))
        if not description and str(data.get("method") or "").lower() in {"", "none", "unknown"}:
            self.content_items.pop(element_id, None)
            self.nodes.pop(element_id, None)
            if self.logger:
                self.logger.warning(
                    "Skip non-indexable image node: %s page=%s image=%s status=%s error=%s",
                    filename,
                    page_num + 1,
                    img_index + 1,
                    data.get("status"),
                    data.get("error") or data.get("api_error"),
                )
            return None
        image_path = _relpath(data.get("original_path") or data.get("image_path") or data.get("enhanced_path") or "")
        embedding_description = clean_embedding_text(
            description, node_type="Figure", max_chars=420
        )
        embedding_text = compose_embedding_text(
            [
                data.get("caption", ""),
                embedding_description,
            ],
            node_type="Figure",
            max_chars=500,
        )
        item = self._base_item(element_id, "image", filename, page_num, page_id)
        item.update(
            {
                "img_path": image_path,
                "enhanced_path": _relpath(data.get("enhanced_path")),
                "caption": data.get("caption", ""),
                "description": description,
                "embedding_description": embedding_description,
                "method": data.get("method"),
                "backend": data.get("backend"),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "status": data.get("status"),
                "error": data.get("error"),
                "api_error": data.get("api_error"),
                "indexable": data.get("indexable", bool(description)),
                "embedding_id": self.embedding_id(element_id),
                "embedding_text": embedding_text,
                "embedding_text_policy": "cleaned_v1",
                "description_policy": data.get("description_policy"),
                "context_policy": data.get("context_policy"),
                "semantic_review_flags": data.get("semantic_review_flags", []),
                "response_metadata": data.get("response_metadata", {}),
            }
        )
        self.content_items[element_id] = item
        self._add_node(
            element_id,
            "Figure",
            name=data.get("caption") or f"figure page {page_num + 1}-{img_index + 1}",
            summary=self._summary(description, 160),
            description=description,
            document_id=self.document_id(filename),
            page_id=page_id,
            source_path=image_path,
            embedding_text=embedding_text,
            extra={
                "method": data.get("method"),
                "backend": data.get("backend"),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "status": data.get("status"),
                "api_error": data.get("api_error"),
                "enhanced_path": _relpath(data.get("enhanced_path")),
                "indexable": data.get("indexable", bool(description)),
                "embedding_description": embedding_description,
                "embedding_text_policy": "cleaned_v1",
                "description_policy": data.get("description_policy"),
                "context_policy": data.get("context_policy"),
                "semantic_review_flags": data.get("semantic_review_flags", []),
                "response_metadata": data.get("response_metadata", {}),
            },
        )
        return element_id

    def add_formulas(self, formulas: List[Dict[str, Any]], filename: str, page_num: int):
        ids = []
        page_id = self.register_page(filename, page_num)
        accepted = [formula for formula in (formulas or []) if not formula.get("review_required")]
        for idx, formula in enumerate(accepted):
            element_id = self.element_id(filename, page_num, "formula", idx)
            latex = _clean(formula.get("latex"))
            description = _clean(formula.get("meaning") or formula.get("description") or formula.get("text"))
            formula_name = _clean(formula.get("name"))
            symbols = formula.get("symbols") or []
            conditions = formula.get("conditions") or []
            source_image = _relpath(formula.get("source_image"))
            embedding_text = compose_embedding_text(
                [formula_name, latex, description, symbols, conditions],
                node_type="Formula",
                max_chars=1600,
            )
            item = self._base_item(element_id, "equation", filename, page_num, page_id)
            item.update(
                {
                    "latex": latex,
                    "name": formula_name,
                    "text": description,
                    "symbols": symbols,
                    "conditions": conditions,
                    "source_image": source_image,
                    "extraction_method": formula.get("extraction_method", ""),
                    "backend": formula.get("backend"),
                    "provider": formula.get("provider"),
                    "model": formula.get("model"),
                    "confidence": formula.get("confidence"),
                    "validation_status": formula.get("validation_status"),
                    "embedding_id": self.embedding_id(element_id),
                    "embedding_text": embedding_text,
                }
            )
            self.content_items[element_id] = item
            self._add_node(
                element_id,
                "Formula",
                name=formula_name or latex[:80] or f"formula page {page_num + 1}-{idx + 1}",
                summary=self._summary(description or latex, 160),
                description=description,
                document_id=self.document_id(filename),
                page_id=page_id,
            source_path=source_image,
            embedding_text=embedding_text,
            extra={
                "latex": latex,
                "symbols": symbols,
                "conditions": conditions,
                "extraction_method": formula.get("extraction_method", ""),
                "backend": formula.get("backend"),
                "provider": formula.get("provider"),
                "model": formula.get("model"),
                "confidence": formula.get("confidence"),
                "validation_status": formula.get("validation_status"),
            },
            )
            ids.append(element_id)
        return ids

    def add_tables(self, tables: List[Dict[str, Any]], filename: str, page_num: int):
        ids = []
        page_id = self.register_page(filename, page_num)
        accepted = [table for table in (tables or []) if not table.get("review_required")]
        for idx, table in enumerate(accepted):
            element_id = self.element_id(filename, page_num, "table", idx)
            table_body = table.get("csv") or table.get("content") or _clean(table.get("data") or table.get("json"))
            description = _clean(table.get("description"))
            headers = _clean(table.get("headers"))
            title = _clean(table.get("title"))
            source_image = _relpath(table.get("source_image"))
            embedding_text = compose_embedding_text(
                [title, headers, description, table.get("units"), table_body[:1500]],
                node_type="Table",
                max_chars=2000,
            )
            item = self._base_item(element_id, "table", filename, page_num, page_id)
            item.update(
                {
                    "table_body": table_body,
                    "headers": table.get("headers", ""),
                    "cells": table.get("cells", []),
                    "title": title,
                    "units": table.get("units", {}),
                    "footnotes": table.get("footnotes", []),
                    "description": description,
                    "source_image": source_image,
                    "extraction_method": table.get("extraction_method") or table.get("type", ""),
                    "backend": table.get("backend"),
                    "provider": table.get("provider"),
                    "model": table.get("model"),
                    "confidence": table.get("confidence"),
                    "validation_status": table.get("validation_status"),
                    "embedding_id": self.embedding_id(element_id),
                    "embedding_text": embedding_text,
                }
            )
            self.content_items[element_id] = item
            self._add_node(
                element_id,
                "Table",
                name=title or f"table page {page_num + 1}-{idx + 1}",
                summary=self._summary(description or table_body, 160),
                description=description or table_body,
                document_id=self.document_id(filename),
            page_id=page_id,
            source_path=source_image,
            embedding_text=embedding_text,
            extra={
                "headers": table.get("headers", ""),
                "cells": table.get("cells", []),
                "units": table.get("units", {}),
                "footnotes": table.get("footnotes", []),
                "rows": table.get("rows"),
                "cols": table.get("cols"),
                "extraction_method": table.get("extraction_method") or table.get("type", ""),
                "backend": table.get("backend"),
                "provider": table.get("provider"),
                "model": table.get("model"),
                "confidence": table.get("confidence"),
                "validation_status": table.get("validation_status"),
            },
            )
            ids.append(element_id)
        return ids

    def next_code_index(self, filename: str, page_num: int) -> int:
        """Return the first unused zero-based code index for a page."""
        page_id = self.page_id(filename, page_num)
        index = 0
        while self.element_id(filename, page_num, "code", index) in self.content_items:
            index += 1
        return index

    def add_code(
        self,
        code_blocks: List[Dict[str, Any]],
        filename: str,
        page_num: int,
        start_index: int = 0,
    ):
        ids = []
        page_id = self.register_page(filename, page_num)
        for idx, code in enumerate(code_blocks or []):
            absolute_index = start_index + idx
            element_id = self.element_id(filename, page_num, "code", absolute_index)
            code_text = code.get("code", "")
            description = _clean(code.get("description"))
            language = code.get("language", "txt")
            source_image = _relpath(code.get("source_image"))
            embedding_text = compose_embedding_text(
                [language, description, code_text[:1500]], node_type="CodeBlock", max_chars=2000
            )
            item = self._base_item(element_id, "code", filename, page_num, page_id)
            item.update(
                {
                    "code": code_text,
                    "language": language,
                    "description": description,
                    "source_image": source_image,
                    "extraction_method": code.get("extraction_method", ""),
                    "backend": code.get("backend"),
                    "provider": code.get("provider"),
                    "model": code.get("model"),
                    "embedding_id": self.embedding_id(element_id),
                    "embedding_text": embedding_text,
                }
            )
            self.content_items[element_id] = item
            self._add_node(
                element_id,
                "CodeBlock",
                name=f"{language} code page {page_num + 1}-{absolute_index + 1}",
                summary=self._summary(description or code_text, 160),
                description=description or code_text,
                document_id=self.document_id(filename),
            page_id=page_id,
            source_path=source_image,
            embedding_text=embedding_text,
            extra={
                "language": language,
                "extraction_method": code.get("extraction_method", ""),
                "backend": code.get("backend"),
                "provider": code.get("provider"),
                "model": code.get("model"),
            },
        )
            ids.append(element_id)
        return ids

    def finalize(self, build_vectors: bool = True):
        self._fix_page_links()
        os.makedirs(self.base_dir, exist_ok=True)
        teaching_handoff = build_teaching_handoff(
            self.documents.values(), self.pages.values(), self.content_items.values()
        )

        course_manifest = {
            "user_id": self.user_id,
            "username": self.username,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "documents": sorted(self.documents.keys()),
            "output_schema": "multimodal_graphrag_extraction_v1",
            "optional_extensions": {HANDOFF_VERSION: HANDOFF_FILES},
            "path_base": "extraction-main",
            "path_format": "workspace_relative_posix",
        }

        self._write_json("course_manifest.json", course_manifest)
        self._write_json("documents.json", list(self.documents.values()))
        self._write_json("pages.json", sorted(self.pages.values(), key=lambda x: (x["document_id"], x["page_no"])))
        self._write_json("content_list.json", list(self.content_items.values()))
        self._write_json("multimodal_nodes.json", list(self.nodes.values()))
        self._write_json("relationships.json", self._build_relationships())
        self._write_json("schema.json", self._build_schema())
        for key, filename in HANDOFF_FILES.items():
            self._write_json(filename, teaching_handoff[key])

        if build_vectors:
            vector_records = list(self.nodes.values())
            vector_summary = build_vector_index(
                vector_records,
                os.path.join(self.base_dir, "vectors"),
                logger=self.logger,
            )
        else:
            vector_summary = {
                "status": "pending",
                "backend": "not_built",
                "model": None,
                "dim": 0,
                "count": 0,
            }
        self._write_json("knowledge_export_summary.json", {
            "user_id": self.user_id,
            "username": self.username,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "job_id": self.job_id,
            "documents": len(self.documents),
            "pages": len(self.pages),
            "content_items": len(self.content_items),
            "nodes": len(self.nodes),
            "vectors": vector_summary,
            "teaching_handoff": {
                "schema": HANDOFF_VERSION,
                "files": HANDOFF_FILES,
                "counts": teaching_handoff["quality_report"]["counts"],
            },
            "updated_at": datetime.now().isoformat(),
        })
        if self.logger:
            state = "with vectors" if build_vectors else "without vectors"
            self.logger.info("Unified knowledge export completed %s", state)

    def _build_relationships(self) -> List[Dict[str, Any]]:
        relationships: List[Dict[str, Any]] = []

        for doc in self.documents.values():
            relationships.append(
                {
                    "relationship_id": f"rel_course_contains_{doc['document_id']}",
                    "source_id": self.course_id,
                    "target_id": doc["document_id"],
                    "relationship_type": "COURSE_CONTAINS_DOCUMENT",
                    "weight": 1.0,
                    "properties": {"document_order": doc.get("document_order")},
                }
            )

        for page in self.pages.values():
            relationships.append(
                {
                    "relationship_id": f"rel_doc_contains_{page['page_id']}",
                    "source_id": page["document_id"],
                    "target_id": page["page_id"],
                    "relationship_type": "DOCUMENT_CONTAINS_PAGE",
                    "weight": 1.0,
                    "properties": {"page_no": page.get("page_no")},
                }
            )
            if page.get("prev_page_id"):
                relationships.append(
                    {
                        "relationship_id": f"rel_prev_{page['page_id']}",
                        "source_id": page["prev_page_id"],
                        "target_id": page["page_id"],
                        "relationship_type": "NEXT_PAGE",
                        "weight": 1.0,
                        "properties": {},
                    }
                )

        for item in self.content_items.values():
            relationships.append(
                {
                    "relationship_id": f"rel_page_contains_{item['element_id']}",
                    "source_id": item["page_id"],
                    "target_id": item["element_id"],
                    "relationship_type": "PAGE_CONTAINS_CONTENT",
                    "weight": 1.0,
                    "properties": {"content_type": item.get("type")},
                }
            )

        return relationships

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "schema_name": "multimodal_graphrag_extraction_v1",
            "identity_fields": [
                "user_id", "username", "course_id", "course_name", "job_id"
            ],
            "path_base": "extraction-main",
            "path_format": "workspace_relative_posix",
            "extraction_root": "extraction",
            "node_files": {
                "documents": "documents.json",
                "pages": "pages.json",
                "content_items": "content_list.json",
                "graph_nodes": "multimodal_nodes.json",
            },
            "relationship_file": "relationships.json",
            "vector_files": {
                "metadata": "vectors/vector_index.json",
                "matrix": "vectors/vector_matrix.npy",
            },
            "node_types": [
                "Course",
                "Document",
                "Page",
                "TextChunk",
                "Figure",
                "Table",
                "Formula",
                "CodeBlock",
                "Entity",
                "Concept",
            ],
            "relationship_types": [
                "COURSE_CONTAINS_DOCUMENT",
                "DOCUMENT_CONTAINS_PAGE",
                "NEXT_PAGE",
                "PAGE_CONTAINS_CONTENT",
                "MENTIONS_CONCEPT",
                "RELATED_TO",
            ],
            "required_downstream_files": [
                "course_manifest.json",
                "documents.json",
                "pages.json",
                "content_list.json",
                "multimodal_nodes.json",
                "relationships.json",
                "vectors/vector_index.json",
                "vectors/vector_matrix.npy",
            ],
            "optional_extensions": {
                HANDOFF_VERSION: {
                    "files": HANDOFF_FILES,
                    "candidate_semantics": "page_scoped_unreviewed_mentions_not_concept_nodes",
                    "offset_unit": "unicode_code_point",
                    "offset_interval": "[start,end)",
                    "original_text_field": "pages.json.source_text_when_available",
                    "legacy_page_raw_text": "cleaned_text_not_original_text",
                    "relations": "No semantic or prerequisite relations are inferred by this extension.",
                }
            },
        }

    def _base_item(self, element_id: str, kind: str, filename: str, page_num: int, page_id: str) -> Dict[str, Any]:
        return {
            "element_id": element_id,
            "type": kind,
            "user_id": self.user_id,
            "username": self.username,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "job_id": self.job_id,
            "document_id": self.document_id(filename),
            "page_id": page_id,
            "page_idx": page_num,
            "page_no": page_num + 1,
        }

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        summary: str,
        description: str,
        document_id: str,
        page_id: str,
        source_path: str,
        embedding_text: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "user_id": self.user_id,
            "username": self.username,
            "course_id": self.course_id,
            "course_name": self.course_name,
            "job_id": self.job_id,
            "document_id": document_id,
            "source_page_id": page_id,
            "name": name or node_id,
            "summary": summary or "",
            "description": description or "",
            "source_path": _relpath(source_path),
            "related_concepts": [],
            "embedding_id": self.embedding_id(node_id),
            "embedding_text": embedding_text or " ".join([name or "", summary or "", description or ""]).strip(),
            "created_at": self.created_at,
            **(extra or {}),
        }

    def _write_json(self, filename: str, data: Any):
        path = os.path.join(self.base_dir, filename)
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._normalize_paths(data), f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)

    def _read_json(self, filename: str, default: Any) -> Any:
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _fix_page_links(self):
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for page in self.pages.values():
            by_doc.setdefault(page["document_id"], []).append(page)
        for pages in by_doc.values():
            pages.sort(key=lambda x: x["page_no"])
            for idx, page in enumerate(pages):
                page["prev_page_id"] = pages[idx - 1]["page_id"] if idx > 0 else None
                page["next_page_id"] = pages[idx + 1]["page_id"] if idx < len(pages) - 1 else None

    @staticmethod
    def _guess_title(text: str) -> str:
        for line in (text or "").splitlines():
            line = _clean(line)
            if 2 <= len(line) <= 80:
                return line
        return ""

    @staticmethod
    def _summary(text: str, limit: int) -> str:
        text = _clean(text)
        return text[:limit]

    @staticmethod
    def _normalize_paths(value: Any, key: str = None) -> Any:
        if isinstance(value, dict):
            return {
                item_key: KnowledgeExporter._normalize_paths(item, item_key)
                for item_key, item in value.items()
            }
        if isinstance(value, list):
            return [KnowledgeExporter._normalize_paths(item, key) for item in value]
        path_keys = {
            "path", "paths", "file_path", "source_path", "source_image",
            "original_path", "enhanced_path", "image_path", "image_paths",
            "page_image_path", "review_image_path", "metadata_path", "matrix_path",
            "recognition_image_path",
            "img_path",
        }
        if isinstance(value, str):
            # Only declared path fields are paths. Quotes, code, formulas and
            # source text can also begin with '/' or a Windows drive literal;
            # rewriting them would invalidate both evidence and span offsets.
            if key in path_keys:
                return _relpath(value)
        return value
