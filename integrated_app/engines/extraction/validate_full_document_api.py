"""Prepare a fresh full-PDF validation, then execute only with an injected client.

Preparing does not load API credentials or use network/database/vector services.
Every page retains native text and a rendered image. Embedded images are audited
individually; omitting an individual call never deletes the full-page evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src2"))
from core.image_processor import ImageProcessor
from core.text_processor import TextProcessor
from core.table_extractor import TableExtractor
from core.formula_validator import FormulaValidator
from core.table_validator import TableValidator
from utils.image_filters import should_keep_image
from utils.knowledge_exporter import KnowledgeExporter

PREPARE_VERSION = "full_pdf_native_and_page_visual_v2_course_identity"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _provenance():
    files = ["src2/core/text_processor.py", "src2/core/image_processor.py",
             "src2/utils/image_filters.py", "src2/utils/knowledge_exporter.py"]
    return {name: sha256_file(ROOT / name) for name in files}


def _prepare_fingerprint():
    source = "\n".join(inspect.getsource(function) for function in
                       (sha256_file, _provenance, _save_embedded, _small_margin,
                        _build_course_identity, prepare_document))
    return hashlib.sha256((PREPARE_VERSION + source).encode("utf-8")).hexdigest()


def _save_embedded(document, info, destination):
    xref, smask = info[0], info[1]
    if smask:
        base, mask = fitz.Pixmap(document, xref), fitz.Pixmap(document, smask)
        pixmap = fitz.Pixmap(base, mask)
        pixmap.save(destination)
    else:
        image = document.extract_image(xref)
        if not image.get("image"):
            raise ValueError("Empty embedded image bytes")
        # Use a decoded PNG so transparent masks/unsupported container formats
        # cannot silently change the image presented to ImageProcessor.
        pixmap = fitz.Pixmap(image["image"])
        if pixmap.colorspace and pixmap.colorspace.n > 3:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        pixmap.save(destination)


def _small_margin(asset, page):
    if not asset["rects"]:
        return False
    width, height = page["page_size"]
    for x0, y0, x1, y1 in asset["rects"]:
        if (x1 - x0) * (y1 - y0) / max(1, width * height) > .015:
            return False
        if not ((x0 < .12 * width or x1 > .88 * width)
                and (y0 < .16 * height or y1 > .84 * height)):
            return False
    return True


def validate_prepared(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("prepare_version") != PREPARE_VERSION or manifest.get("prepare_fingerprint") != _prepare_fingerprint():
        raise ValueError("Prepared extraction algorithm changed; use a new validation run.")
    if manifest.get("source_code_provenance") != _provenance():
        raise ValueError("Prepared extraction source files changed; use a new validation run.")
    if sha256_file(manifest["pdf"]) != manifest["pdf_sha256"]:
        raise ValueError("Source PDF changed; prepared evidence is not reusable.")
    for task in manifest["tasks"]:
        if sha256_file(task["path"]) != task["image_sha256"]:
            raise ValueError("Prepared visual asset changed: " + task["task_id"])
    if len(manifest["prepared_pages"]) != manifest["original_page_count"]:
        raise ValueError("Prepared page coverage is incomplete.")
    return manifest


def _build_course_identity(pdf, output, *, course_name=None, course_id=None, job_id=None, pdf_title=None):
    """Derive a generic identity; callers can group several PDFs explicitly."""
    for key, value in (("course_name", course_name), ("course_id", course_id), ("job_id", job_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(key + " must be a nonempty string when supplied")
    if pdf_title is None:
        with fitz.open(pdf) as document:
            pdf_title = (document.metadata or {}).get("title", "")
    name = course_name.strip() if course_name is not None else (str(pdf_title or "").strip() or Path(pdf).stem)
    generated_id = "course_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return {"course_name": name, "course_id": course_id.strip() if course_id is not None else generated_id,
            "job_id": job_id.strip() if job_id is not None else Path(output).parent.name}


def _manifest_course_identity(manifest, output):
    """Read legacy identity from its own prepared export, without rewriting it."""
    identity = manifest.get("course_identity")
    if identity is not None:
        if not isinstance(identity, dict) or any(not isinstance(identity.get(key), str) or not identity[key].strip()
                                                 for key in ("course_name", "course_id", "job_id")):
            raise ValueError("Manifest course_identity is invalid")
        return {key: identity[key] for key in ("course_name", "course_id", "job_id")}
    documents_path = Path(output) / "kg_data_text_prepared" / "documents.json"
    if documents_path.is_file():
        documents = json.loads(documents_path.read_text(encoding="utf-8"))
        if isinstance(documents, list):
            for document in documents:
                if (isinstance(document, dict) and document.get("file_name") == Path(manifest["pdf"]).name
                        and all(isinstance(document.get(key), str) and document[key].strip()
                                for key in ("course_name", "course_id", "job_id"))):
                    return {key: document[key] for key in ("course_name", "course_id", "job_id")}
    return _build_course_identity(manifest["pdf"], output)


def prepare_document(pdf, output, scale=2.0, *, course_name=None, course_id=None, job_id=None):
    pdf, output = Path(pdf).resolve(), Path(output).resolve()
    if not pdf.is_file():
        raise ValueError("Input PDF does not exist.")
    if (output / "manifest.json").exists():
        manifest = validate_prepared(output)
        if Path(manifest["pdf"]).resolve() != pdf:
            raise ValueError("Output already belongs to a different source PDF")
        identity = _manifest_course_identity(manifest, output)
        for key, value in (("course_name", course_name), ("course_id", course_id), ("job_id", job_id)):
            if value is not None and (not isinstance(value, str) or value.strip() != identity[key]):
                raise ValueError("Existing run has a different " + key + "; use a new output directory")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output contains files but no matching prepared manifest; not overwriting.")
    if not .5 <= scale <= 3:
        raise ValueError("Render scale must be between .5 and 3.")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    text_processor, native_tables = TextProcessor(), TableExtractor()
    pages, occurrences, preparation_errors = [], [], []
    with fitz.open(pdf) as document:
        identity = _build_course_identity(pdf, output, course_name=course_name, course_id=course_id,
                                          job_id=job_id, pdf_title=(document.metadata or {}).get("title", ""))
        repeated = text_processor.find_repeated_lines([page.get_text() for page in document])
        for page_index, page in enumerate(document):
            number = page_index + 1
            preview = assets / ("page_%02d.png" % number)
            page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(preview)
            record = {"page_no": number, "page_size": [page.rect.width, page.rect.height],
                      "preview": str(preview), "preview_sha256": sha256_file(preview),
                      "drawing_count": len(page.get_drawings()), "embedded_images": [],
                      "native_table_candidates": [], "preparation_status": "success"}
            raw_text = page.get_text()
            try:
                candidates = native_tables._extract_with_pymupdf(page)
                native_tables._mark_diagonal_text_overlaps(page, candidates)
                record["native_table_candidates"] = native_tables.validator.validate_many(candidates)
            except Exception as error:
                record["native_table_error"] = str(error)
            for image_index, info in enumerate(page.get_images(full=True)):
                asset = {"image_index": image_index, "xref": info[0], "dimensions": [info[2], info[3]],
                         "rects": [[float(v) for v in rect] for rect in page.get_image_rects(info[0])],
                         "path": str(assets / ("page_%02d_xref_%s.png" % (number, info[0]))),
                         "page_no": number, "source_kind": "embedded_image"}
                try:
                    _save_embedded(document, info, asset["path"])
                    asset["image_sha256"] = sha256_file(asset["path"])
                    keep, reason, stats = should_keep_image(asset["path"])
                    asset.update(filter_keep=keep, filter_reason=reason, filter_stats=stats,
                                 preparation_status="success")
                except Exception as error:
                    asset.update(preparation_status="failed", error=str(error), individual_call=False,
                                 individual_call_reason="image_extraction_failed_full_page_evidence_retained")
                    preparation_errors.append({"page_no": number, "image_index": image_index, "error": str(error)})
                record["embedded_images"].append(asset)
                occurrences.append(asset)
            record["text_result"] = text_processor.process_text(
                raw_text, page_index, [a["path"] for a in record["embedded_images"] if a["preparation_status"] == "success"],
                repeated_lines=repeated)
            record["table_specialist_planned"] = bool(record["native_table_candidates"] or re.search(r"\bphase\b|接线端子状态|驱动时序|相位序列", raw_text, re.I))
            record["formula_specialist_planned"] = bool(re.search(r"\bsin\b|\bcos\b|\bClarke\b|\bPark\b|数学表达式|矩阵|[=＝]", raw_text, re.I))
            pages.append(record)
    hashes = defaultdict(set)
    for asset in occurrences:
        if asset.get("image_sha256"):
            hashes[asset["image_sha256"]].add(asset["page_no"])
    tasks, dedup = [], {}
    for page in pages:
        number = page["page_no"]
        whole_id = "page_%02d_visual" % number
        tasks.append({"task_id": whole_id, "operation": "image_description", "source_kind": "full_page",
                      "page_no": number, "path": page["preview"], "image_sha256": page["preview_sha256"],
                      "reason": "all_page_visual_coverage_including_vector_graphics_and_small_fragments",
                      "occurrences": [{"page_no": number, "image_index": 0, "source_kind": "full_page"}]})
        page["page_visual_task_id"] = whole_id
        fragmented = len(page["embedded_images"]) >= 12
        for asset in page["embedded_images"]:
            asset["full_page_coverage_task_id"] = whole_id
            if asset["preparation_status"] != "success":
                continue
            if not asset["rects"]:
                reason = "unused_pdf_resource_not_drawn"
            elif fragmented:
                reason = "fragmented_composition_covered_by_full_page_task"
            elif len(hashes[asset["image_sha256"]]) >= 3 and _small_margin(asset, page):
                reason = "repeated_small_margin_asset_covered_by_full_page_task"
            elif not asset["filter_keep"]:
                reason = "individual_" + asset["filter_reason"] + "_covered_by_full_page_task"
            else:
                reason = None
            asset["individual_call"] = reason is None
            asset["individual_call_reason"] = reason or "retained_teaching_material_candidate"
            if reason:
                continue
            digest = asset["image_sha256"]
            if digest not in dedup:
                task_id = "embedded_" + digest[:20]
                task = {"task_id": task_id, "operation": "image_description", "source_kind": "embedded_image",
                        "page_no": number, "path": asset["path"], "image_sha256": digest, "occurrences": []}
                tasks.append(task)
                dedup[digest] = task
            dedup[digest]["occurrences"].append({"page_no": number, "image_index": asset["image_index"] + 1,
                                                "source_kind": "embedded_image", "path": asset["path"]})
            asset["task_id"] = dedup[digest]["task_id"]
        for operation, planned in (("table", page["table_specialist_planned"]), ("formula", page["formula_specialist_planned"])):
            if planned:
                tasks.append({"task_id": "page_%02d_%s" % (number, operation), "operation": operation,
                              "source_kind": "full_page", "page_no": number, "path": page["preview"],
                              "image_sha256": page["preview_sha256"], "occurrences": [{"page_no": number}],
                              "reason": "native_layout_or_text_heuristic_specialist_check"})
    manifest = {
        "fresh_run": True, "prepare_version": PREPARE_VERSION, "prepare_fingerprint": _prepare_fingerprint(),
        "course_identity": identity,
        "source_code_provenance": _provenance(), "prepared_at": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf), "pdf_sha256": sha256_file(pdf), "original_page_count": len(pages),
        "pages": [page["page_no"] for page in pages], "prepared_pages": pages, "tasks": tasks,
        "preparation_errors": preparation_errors, "planned_requests": len(tasks),
        "request_counts_by_operation": dict(Counter(task["operation"] for task in tasks)),
        "embedded_occurrence_count": len(occurrences),
        "individual_image_exclusion_reasons": dict(Counter(asset["individual_call_reason"] for asset in occurrences if not asset.get("individual_call"))),
        "historical_descriptions_reused": False, "api_requests_sent": 0, "mode": "prepare_only",
        "coverage_note": "Every PDF page retains raw/cleaned text and a fresh page render; exclusions only omit an individual embedded-image call. No semantic logo/decorative verdict is inferred from repetition alone.",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "source_code_provenance.json", manifest["source_code_provenance"])
    exporter = KnowledgeExporter(base_dir=str(output / "kg_data_text_prepared"), **identity)
    filename = pdf.stem + "_pdf"
    exporter.register_document(filename, {"file_name": pdf.name, "file_path": str(pdf), "file_type": "pdf",
                                         "total_pages": len(pages), "validation_subset": False, "fresh_run": True})
    for page in pages:
        exporter.add_text(page["text_result"], filename, page["page_no"] - 1)
        exporter.register_page(filename, page["page_no"] - 1, page_image_path=page["preview"])
    exporter.finalize(build_vectors=False)
    return manifest


EXECUTION_VERSION = "full_pdf_visual_handoff_v2_course_identity"
COMPLETED_TASK_STATUSES = {"success", "review_required", "no_content"}


def _execution_provenance():
    files = ["validate_full_document_api.py", "src2/core/table_validator.py",
             "src2/core/formula_validator.py", "validation_vision_client.py"]
    return {name: sha256_file(ROOT / name) for name in files if (ROOT / name).exists()}


def _structured_result(task, client):
    """One specialist call only; malformed/empty positives are not successes."""
    operation = task["operation"]
    response = (client.recognize_table(task["path"], context="") if operation == "table"
                else client.recognize_formula(task["path"], context=""))
    if not isinstance(response, dict) or response.get("parse_error"):
        raise ValueError("Specialist response is not valid structured JSON")
    flag, key = ("has_table", "tables") if operation == "table" else ("has_formula", "formulas")
    if type(response.get(flag)) is not bool or not isinstance(response.get(key), list):
        raise ValueError("Specialist response has invalid presence flag or collection")
    items = response[key]
    if response[flag] != bool(items) or any(not isinstance(item, dict) for item in items):
        raise ValueError("Specialist presence flag contradicts extracted collection")
    if operation == "table":
        count = response.get("visible_table_count")
        if type(count) is not int or count < 0 or type(response.get("coverage_complete")) is not bool:
            raise ValueError("Table coverage fields are missing or invalid")
        if not items and (count != 0 or response["coverage_complete"] is not True):
            raise ValueError("Empty table result does not establish complete absence")
        for item in items:
            if (not isinstance(item.get("cells"), list) or not item["cells"]
                    or any(not isinstance(row, list) for row in item["cells"])
                    or not any(str(cell or "").strip() for row in item["cells"] for cell in row)):
                raise ValueError("Table response has no usable cell matrix")
    else:
        for item in items:
            if not isinstance(item.get("latex"), str) or not item["latex"].strip():
                raise ValueError("Formula response has no usable LaTeX")
    candidates = []
    for index, item in enumerate(items):
        candidate = copy.deepcopy(item)
        candidate.update(source_image=task["path"], source_kind=task["source_kind"],
                         source_type="image_vlm", extraction_method="fresh_full_page_vlm",
                         backend="api", provider=getattr(client, "provider", None),
                         model=getattr(client, "model", None), task_id=task["task_id"],
                         image_index=index, raw_response=response.get("raw_response"))
        if operation == "table":
            candidate.update(visible_table_count=response["visible_table_count"],
                             extracted_table_count=len(items), coverage_complete=response["coverage_complete"])
        candidates.append(candidate)
    validator = TableValidator() if operation == "table" else FormulaValidator()
    validated = validator.validate_many(candidates)
    return {"response": response, "candidates": validated,
            "indexable_count": sum(bool(item.get("indexable")) for item in validated),
            "review_count": sum(bool(item.get("review_required")) for item in validated)}


def _export_execution(output, manifest, records):
    """Rebuild handoff from this run only, retaining every native-text page."""
    pdf = Path(manifest["pdf"])
    filename = pdf.stem + "_pdf"
    exporter = KnowledgeExporter(base_dir=str(output / "kg_data"), **_manifest_course_identity(manifest, output))
    exporter.register_document(filename, {"file_name": pdf.name, "file_path": str(pdf), "file_type": "pdf",
                                         "total_pages": manifest["original_page_count"],
                                         "validation_subset": False, "fresh_run": True,
                                         "pdf_sha256": manifest["pdf_sha256"]})
    def annotate(element_id, task):
        if element_id is None:
            return
        provenance = {"source_kind": task["source_kind"], "task_id": task["task_id"],
                      "description_scope": "entire_page" if task["source_kind"] == "full_page" else "embedded_image_only",
                      "fresh_run": True, "source_image_sha256": task["image_sha256"],
                      "same_run_occurrence_reuse": len(task["occurrences"]) > 1}
        exporter.nodes[element_id].update(provenance)
        exporter.content_items[element_id].update(provenance)
    for page in manifest["prepared_pages"]:
        exporter.add_text(page["text_result"], filename, page["page_no"] - 1)
        exporter.register_page(filename, page["page_no"] - 1, page_image_path=page["preview"])
    for task in manifest["tasks"]:
        record = records.get(task["task_id"])
        if not record or record.get("status") not in COMPLETED_TASK_STATUSES:
            continue
        result = record["result"]
        if task["operation"] == "image_description":
            for occurrence in task["occurrences"]:
                image = copy.deepcopy(result)
                image.update(original_path=occurrence.get("path") or task["path"],
                             source_kind=task["source_kind"], task_id=task["task_id"],
                             fresh_run=True, same_run_occurrence_reuse=len(task["occurrences"]) > 1)
                image.setdefault("metadata", {})["validation"] = {
                    "fresh_run": True, "source_kind": task["source_kind"], "task_id": task["task_id"],
                    "page_no": occurrence["page_no"], "image_sha256": task["image_sha256"],
                    "same_run_occurrence_reuse": len(task["occurrences"]) > 1,
                    "description_scope": "entire_page" if task["source_kind"] == "full_page" else "embedded_image_only"}
                element_id = exporter.add_image(image, filename, occurrence["page_no"] - 1, occurrence["image_index"])
                annotate(element_id, task)
        elif task["operation"] == "table":
            for element_id in exporter.add_tables(result["candidates"], filename, task["page_no"] - 1):
                annotate(element_id, task)
        else:
            for element_id in exporter.add_formulas(result["candidates"], filename, task["page_no"] - 1):
                annotate(element_id, task)
    exporter.finalize(build_vectors=False)
    return exporter.document_id(filename)


def execute_prepared(output, vision_client, *, include_specialists=False, retry_failed=False):
    """Execute an audited plan with the injected budgeted client, stopping on failure.

    No credentials are loaded here. A same-run successful checkpoint can be
    replayed without a request. A failed task requires explicit retry_failed.
    Review-required observations complete a request but never enter formal data.
    """
    output = Path(output).resolve()
    manifest = validate_prepared(output)
    selected = [task for task in manifest["tasks"]
                if include_specialists or task["operation"] == "image_description"]
    identity = {"execution_version": EXECUTION_VERSION, "pdf_sha256": manifest["pdf_sha256"],
                "course_identity": _manifest_course_identity(manifest, output),
                "prepare_fingerprint": manifest["prepare_fingerprint"],
                "execution_source_provenance": _execution_provenance(),
                "provider": getattr(vision_client, "provider", None),
                "model": getattr(vision_client, "model", None)}
    checkpoint_path = output / "task_results.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        **identity, "fresh_run": True, "historical_descriptions_reused": False, "records": {}}
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise ValueError("Extraction checkpoint belongs to another PDF, algorithm or model")
    records = checkpoint.get("records")
    if not isinstance(records, dict):
        raise ValueError("Extraction checkpoint has invalid records")
    for task in selected:
        previous = records.get(task["task_id"])
        if previous and (previous.get("image_sha256") != task["image_sha256"]
                         or previous.get("operation") != task["operation"]):
            raise ValueError("Checkpoint task does not match prepared evidence")
    execution_path = output / "execution_manifest.json"
    execution = {**identity, "fresh_run": True, "pdf": manifest["pdf"],
                 "mode": "executing", "status": "incomplete", "all_required_tasks_completed": False,
                 "include_specialists": include_specialists, "original_page_count": manifest["original_page_count"],
                 "pages": manifest["pages"], "selected_task_ids": [task["task_id"] for task in selected],
                 "planned_requests": len(selected), "historical_descriptions_reused": False,
                 "prepare_manifest": str(output / "manifest.json"), "task_results": str(checkpoint_path),
                 "source_code_provenance": _execution_provenance(),
                 "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(execution_path, execution)
    processor = ImageProcessor(vlm_client=vision_client)
    stopped = None
    checkpoint_cache_hits = 0
    for index, task in enumerate(selected):
        if getattr(vision_client, "last_fatal_error", None):
            stopped = {"task_id": task["task_id"], "reason": "client_fatal_error"}
            break
        previous = records.get(task["task_id"])
        if previous and previous.get("status") in COMPLETED_TASK_STATUSES:
            previous["checkpoint_reused_within_run"] = True
            checkpoint_cache_hits += 1
            continue
        if previous and not retry_failed:
            stopped = {"task_id": task["task_id"], "reason": "previous_failure_requires_explicit_retry"}
            break
        record = {key: copy.deepcopy(task[key]) for key in
                  ("task_id", "operation", "source_kind", "page_no", "image_sha256")}
        record.update(status="started", started_at=datetime.now(timezone.utc).isoformat())
        if previous:
            record["previous_attempts"] = previous.get("previous_attempts", []) + [previous]
        records[task["task_id"]] = record
        write_json(checkpoint_path, checkpoint)
        try:
            if task["operation"] == "image_description":
                result = processor.process_image(task["path"], "")
                record["result"] = result
                if (result.get("status") != "success" or not str(result.get("description") or "").strip()
                        or result.get("error") or result.get("api_error")):
                    raise ValueError("Image description did not produce a successful nonempty observation")
                record["status"] = "review_required" if result.get("review_required") else "success"
            else:
                result = _structured_result(task, vision_client)
                record["result"] = result
                record["status"] = ("no_content" if not result["candidates"] else
                                    "review_required" if result["review_count"] else "success")
            record["response_metadata"] = copy.deepcopy(getattr(vision_client, "last_response_metadata", {}) or {})
            if getattr(vision_client, "last_fatal_error", None):
                raise ValueError("Budgeted client reported a fatal response")
            if record["response_metadata"].get("finish_reason") != "stop":
                raise ValueError("Vision response was truncated, filtered or did not finish normally")
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__,
                          error="Required visual task failed; see retained result and budgeted API log")
            stopped = {"task_id": task["task_id"], "reason": "required_task_failed", "error_type": type(error).__name__}
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(checkpoint_path, checkpoint)
        print(f"full_document task {index + 1}/{len(selected)} {task['task_id']}: {record['status']}", flush=True)
        if stopped:
            break
    write_json(checkpoint_path, checkpoint)
    document_id = _export_execution(output, manifest, records)
    completed = [task for task in selected if records.get(task["task_id"], {}).get("status") in COMPLETED_TASK_STATUSES]
    pending = [task["task_id"] for task in selected if task not in completed]
    complete = not pending and not manifest["preparation_errors"] and not stopped
    reviews = [{"task_id": task["task_id"], "page_no": task["page_no"], "operation": task["operation"],
                "result": records[task["task_id"]].get("result")}
               for task in selected if records.get(task["task_id"], {}).get("status") in {"review_required", "failed"}]
    write_json(output / "review_candidates.json", {"fresh_run": True, "items": reviews,
               "native_table_candidates": [{"page_no": page["page_no"], "tables": page["native_table_candidates"]}
                                            for page in manifest["prepared_pages"] if page["native_table_candidates"]],
               "note": "Native candidates and model-review items are retained for audit, not automatically confirmed."})
    coverage = [{"page_no": page["page_no"], "text_retained": True,
                 "page_visual_task_id": page["page_visual_task_id"],
                 "page_visual_status": records.get(page["page_visual_task_id"], {}).get("status", "pending")}
                for page in manifest["prepared_pages"]]
    execution.update(mode="executed", status="success" if complete else "incomplete",
                     all_required_tasks_completed=complete, document_id=document_id,
                     completed_task_count=len(completed), pending_task_ids=pending,
                     task_status_counts=dict(Counter(records.get(task["task_id"], {}).get("status", "pending") for task in selected)),
                     page_coverage=coverage, stopped=stopped, checkpoint_cache_hits=checkpoint_cache_hits,
                     client_cache_hits=getattr(vision_client, "cache_hits", None),
                     kg_data=str(output / "kg_data"), finished_at=datetime.now(timezone.utc).isoformat(),
                     scope_note="Fresh full-page and embedded-image observations remain distinct; review items are excluded from formal elements. No vectors or database writes.")
    write_json(execution_path, execution)
    return execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--course-name")
    parser.add_argument("--course-id")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    manifest = prepare_document(args.pdf, args.output, args.scale, course_name=args.course_name,
                                course_id=args.course_id, job_id=args.job_id)
    print(json.dumps({key: manifest[key] for key in
                      ("pdf", "original_page_count", "planned_requests", "request_counts_by_operation",
                       "embedded_occurrence_count", "individual_image_exclusion_reasons", "api_requests_sent")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
