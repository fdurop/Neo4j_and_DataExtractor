"""Offline, page-scoped evidence for downstream teaching graph review.

These records describe mentions and source structure, not verified concepts,
global synonyms, semantic hierarchies, or learning prerequisites. No model or
embedding call is made here. Offsets always address an explicitly named field.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


HANDOFF_VERSION = "page_evidence_v1"
HANDOFF_FILES = {
    "page_contexts": "page_contexts.json",
    "term_candidates": "term_candidates.json",
    "quality_report": "extraction_quality_report.json",
}
_IDENTITY_FIELDS = ("user_id", "username", "course_id", "course_name", "job_id")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _span(start: int, end: int) -> Dict[str, Any]:
    return {"start": start, "end": end, "unit": "unicode_code_point", "interval": "[start,end)"}


def _line_hints(text: str, element_id: str) -> Dict[str, Any]:
    """Keep exact source lines; never rewrite a heading as a learning goal."""
    headings = []
    objectives = []
    position = 0
    title = None
    objective_block = False
    heading_pattern = re.compile(r"^(?:第.{1,12}[章节]|\d+(?:\.\d+)*[.、)）\s]|[一二三四五六七八九十]+[、.])")
    objective_pattern = re.compile(
        r"^(?:学习目标|教学目标|课程目标|本[章节课单元]{1,3}目标|Learning objectives?|Learning outcomes?)(?:\s*[:：]|\s*$)",
        re.IGNORECASE,
    )
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = position + len(line) - len(line.lstrip())
        position += len(line)
        if not stripped:
            objective_block = False
            continue
        record = {
            "text": stripped,
            "evidence_id": element_id,
            "evidence_file": "content_list.json",
            "evidence_field": "text",
            "span": _span(start, start + len(stripped)),
            "review_status": "pending",
        }
        if title is None and 2 <= len(stripped) <= 80:
            title = {**record, "method": "first_short_line_heuristic"}
        if len(stripped) <= 100 and heading_pattern.match(stripped):
            headings.append({**record, "method": "numbered_heading_heuristic"})
        if objective_pattern.match(stripped):
            objective_block = True
            objectives.append({**record, "method": "explicit_objective_heading"})
        elif objective_block:
            # Only retain directly following objective-like lines, not the rest
            # of a page just because it once contained an objectives heading.
            if re.match(r"^(?:[-•·]|\d+[.、)）]|[一二三四五六七八九十]+[、.]|掌握|了解|理解|能够|学会)", stripped):
                objectives.append({**record, "method": "line_after_objective_heading"})
            else:
                objective_block = False
    return {"title_candidate": title, "heading_candidates": headings, "objective_candidates": objectives}


def _candidate_kind(term: str) -> str:
    # Shape is a review hint, never a decision to hide a teaching concept.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+./-]*", term):
        return "identifier_or_abbreviation" if re.search(r"[0-9_]", term) else "latin_term_or_abbreviation"
    return "phrase_mention"


def _term_candidates(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = _text(item.get("text"))
    terms = item.get("technical_terms") or []
    if not isinstance(terms, list):
        return []
    candidates = []
    seen = set()
    for value in terms:
        if not isinstance(value, str) or not value.strip() or value in seen:
            continue
        seen.add(value)
        pattern = re.escape(value)
        # Avoid PWM inside PWM_MODE, without treating adjacent Chinese text
        # as an English word boundary. Matching remains case-sensitive.
        if re.match(r"[A-Za-z0-9_]", value):
            pattern = r"(?<![A-Za-z0-9_])" + pattern
        if re.search(r"[A-Za-z0-9_]$", value):
            pattern += r"(?![A-Za-z0-9_])"
        occurrences = []
        for match in re.finditer(pattern, text):
            quote_start = max(0, match.start() - 70)
            quote_end = min(len(text), match.end() + 70)
            occurrences.append({
                "span": _span(match.start(), match.end()),
                "quote": text[quote_start:quote_end],
                "quote_span": _span(quote_start, quote_end),
            })
        if not occurrences:
            continue
        candidate_id = "term_" + _sha256(item["element_id"] + "\0" + value)[:20]
        candidates.append({
            **{field: item.get(field) for field in _IDENTITY_FIELDS},
            "candidate_id": candidate_id,
            "term": value,
            "document_id": item.get("document_id"),
            "page_id": item.get("page_id"),
            "page_no": item.get("page_no"),
            "evidence_id": item["element_id"],
            "evidence_file": "content_list.json",
            "evidence_field": "text",
            "text_sha256": _sha256(text),
            "occurrences": occurrences,
            "candidate_kind": _candidate_kind(value),
            "method": "existing_regex_technical_term_exact_match",
            "scope": "page",
            "teaching_role": "pending",
            "review_status": "pending",
            "aliases": [],
        })
    return candidates


def _material_reference(item: Dict[str, Any]) -> Dict[str, Any]:
    kind = item.get("type", "unknown")
    evidence_kind = {
        "text": "extracted_text",
        "image": "image_description",
        "formula": "formula_transcription",
        "equation": "formula_transcription",
        "table": "table_transcription",
        "code": "code_transcription",
    }.get(kind, "unknown")
    return {
        "evidence_id": item["element_id"],
        "content_type": kind,
        "evidence_kind": evidence_kind,
        "source_path": item.get("source_image") or item.get("img_path") or "",
        "backend": item.get("backend"),
        "provider": item.get("provider"),
        "model": item.get("model"),
        "extraction_method": item.get("extraction_method") or item.get("method"),
        "description_policy": item.get("description_policy"),
        "context_policy": item.get("context_policy"),
        "semantic_review_flags": item.get("semantic_review_flags", []),
        "validation_status": item.get("validation_status"),
        # Extraction acceptance is not a teacher's semantic approval.
        "teaching_review_status": "pending",
    }


def build_teaching_handoff(
    documents: Iterable[Dict[str, Any]],
    pages: Iterable[Dict[str, Any]],
    content_items: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rebuild optional files from formal records, including old checkpoints."""
    documents = list(documents)
    documents_by_id = {document["document_id"]: document for document in documents}
    pages = sorted(pages, key=lambda page: (page["document_id"], page["page_no"]))
    items = list(content_items)
    page_ids = {page["page_id"] for page in pages}
    by_page = defaultdict(list)
    warnings = []
    for item in items:
        by_page[item.get("page_id")].append(item)
        if item.get("page_id") not in page_ids:
            warnings.append({"code": "material_without_page", "evidence_id": item.get("element_id")})

    contexts = []
    candidates = []
    for page in pages:
        document = documents_by_id.get(page["document_id"], {})
        materials = sorted(by_page[page["page_id"]], key=lambda item: item["element_id"])
        text_items = [item for item in materials if item.get("type") == "text"]
        text_contexts = []
        page_candidates = []
        for item in text_items:
            text = _text(item.get("text"))
            hints = _line_hints(text, item["element_id"])
            source_available = item.get("source_text_available") is True and isinstance(item.get("raw_text"), str)
            text_contexts.append({
                "evidence_id": item["element_id"],
                "text_field": "content_list.json.text",
                "text_sha256": _sha256(text),
                "source_text_available": source_available,
                "source_text_field": "content_list.json.raw_text" if source_available else None,
                "source_text_sha256": _sha256(item["raw_text"]) if source_available else None,
                "key_point_candidates": item.get("key_points") or [],
                "key_points_policy": "line_heuristics_not_verified_learning_objectives",
                **hints,
            })
            page_candidates.extend(_term_candidates(item))
        candidates.extend(page_candidates)
        flags = []
        if not any(_text(item.get("text")).strip() for item in text_items):
            flags.append("no_text_evidence")
        if not materials:
            flags.append("no_formal_content")
        if text_items and not all(context["source_text_available"] for context in text_contexts):
            flags.append("original_text_unavailable")
        if not page_candidates:
            flags.append("no_regex_term_candidates")
        contexts.append({
            **{field: page.get(field) for field in _IDENTITY_FIELDS},
            "context_schema": HANDOFF_VERSION,
            "page_id": page["page_id"],
            "document_id": page["document_id"],
            "page_no": page["page_no"],
            "source_document_path": document.get("file_path", ""),
            "source_document_name": document.get("file_name", ""),
            "source_text_available": page.get("source_text_available") is True and isinstance(page.get("source_text"), str),
            "source_text_field": "pages.json.source_text" if page.get("source_text_available") is True and isinstance(page.get("source_text"), str) else None,
            "source_text_sha256": _sha256(page["source_text"]) if page.get("source_text_available") is True and isinstance(page.get("source_text"), str) else None,
            "title": page.get("title", ""),
            "title_policy": "heuristic_navigation_hint_not_verified_page_topic",
            "prev_page_id": page.get("prev_page_id"),
            "next_page_id": page.get("next_page_id"),
            "page_order_policy": "source_order_not_prerequisite",
            "page_image_path": page.get("page_image_path", ""),
            "text_contexts": text_contexts,
            "materials": [_material_reference(item) for item in materials],
            "term_candidate_ids": [candidate["candidate_id"] for candidate in page_candidates],
            "review_flags": flags,
            "semantic_review_status": "pending",
        })

    known_documents = set(documents_by_id)
    for page in pages:
        if page["document_id"] not in known_documents:
            warnings.append({"code": "page_without_document", "page_id": page["page_id"]})
    for document in documents:
        count = document.get("page_count")
        actual = {page["page_no"] for page in pages if page["document_id"] == document["document_id"]}
        if isinstance(count, int) and count >= 0:
            missing = sorted(set(range(1, count + 1)) - actual)
            if missing:
                warnings.append({"code": "missing_pages", "document_id": document["document_id"], "page_numbers": missing})

    report = {
        "report_schema": HANDOFF_VERSION,
        "scope": "formal_export_only_excludes_rejected_or_review_only_materials",
        "method": "deterministic_structure_audit_not_accuracy_evaluation",
        "counts": {
            "documents": len(documents), "pages": len(pages), "content_items": len(items),
            "page_scoped_term_candidates": len(candidates),
            "content_types": dict(sorted(Counter(item.get("type", "unknown") for item in items).items())),
            "page_review_flags": dict(sorted(Counter(flag for context in contexts for flag in context["review_flags"]).items())),
            "structural_warnings": len(warnings),
            "material_semantic_review_flags": dict(sorted(Counter(
                flag for item in items for flag in item.get("semantic_review_flags", [])
            ).items())),
        },
        "warnings": warnings,
        "limits": [
            "Term candidates are incomplete regex mentions, not verified teaching concepts.",
            "Image descriptions are generated observations, not verbatim slide text.",
            "Page order and co-occurrence do not establish prerequisites or semantic hierarchy.",
            "No global synonyms, teaching importance, or prerequisite relations are inferred here.",
        ],
    }
    return {"page_contexts": contexts, "term_candidates": candidates, "quality_report": report}
