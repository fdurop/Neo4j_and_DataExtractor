"""Validate a fresh PDF extraction with an injected, shared-budget JSON client.

CLI prepares only. Paid execution is the explicit run_graph_phase() library
call; this module never reads API keys or creates an HTTP/model/database client.
Use a new output directory for each repair so earlier results stay reviewable.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import inspect
import json
import time
from collections import Counter
from pathlib import Path

import all_builder
from graph_quality import export_teacher_bundle, prepare_knowledge


FILES = {"manifest": "course_manifest.json", "documents": "documents.json", "pages": "pages.json",
         "nodes": "multimodal_nodes.json", "relationships": "relationships.json"}
SOURCE_TYPE = "fresh_original_pdf_extraction"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_fresh_document(source, document_id, expected_pages=None):
    source = Path(source).resolve()
    formal = {key: read(source / filename) for key, filename in FILES.items()}
    documents = [d for d in formal["documents"] if d.get("document_id") == document_id]
    if len(documents) != 1:
        raise ValueError("Select exactly one document from the new extraction")
    pages = [p for p in formal["pages"] if p.get("document_id") == document_id]
    count = documents[0].get("page_count")
    page_ids = {p.get("page_id") for p in pages}
    if expected_pages is not None and (not isinstance(expected_pages, int) or isinstance(expected_pages, bool) or expected_pages < 1):
        raise ValueError("expected_pages must be a positive integer or None")
    if (not isinstance(count, int) or isinstance(count, bool) or count < 1
            or (expected_pages is not None and count != expected_pages)
            or len(pages) != count or len(page_ids) != count or None in page_ids
            or {p.get("page_no") for p in pages} != set(range(1, count + 1))):
        raise ValueError("Fresh document must contain each expected page exactly once")
    nodes = [n for n in formal["nodes"] if all_builder._get_page_id_from_node(n) in page_ids]
    node_ids = [n.get("node_id") for n in nodes]
    if None in node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("Fresh evidence nodes require unique node_id values")
    allowed = page_ids | set(node_ids) | {document_id, formal["manifest"].get("course_id")}
    return {"manifest": formal["manifest"], "documents": documents, "pages": pages, "nodes": nodes,
            "relationships": [r for r in formal["relationships"]
                              if r.get("source_id") in allowed and r.get("target_id") in allowed]}


def check_provenance(provenance):
    if not isinstance(provenance, dict) or provenance.get("source_type") != SOURCE_TYPE:
        raise ValueError("Explicit fresh PDF extraction provenance is required")
    for key in ("source_pdf", "source_pdf_sha256", "extraction_manifest", "extraction_run_id"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError("Missing fresh provenance field: " + key)
    pdf, manifest = Path(provenance["source_pdf"]).resolve(), Path(provenance["extraction_manifest"]).resolve()
    pdf_hash = sha(pdf)
    if pdf_hash != provenance["source_pdf_sha256"]:
        raise ValueError("Source PDF does not match the declared hash")
    declared = read(manifest)
    manifest_hash = declared.get("pdf_sha256") or declared.get("source_pdf_sha256")
    if manifest_hash != pdf_hash:
        raise ValueError("Extraction manifest does not identify this exact original PDF")
    if declared.get("mode") == "prepare_only" or declared.get("execute") is False:
        raise ValueError("A preparation-only extraction cannot be declared a fresh API result")
    if "historical" in str(declared.get("source_type", "")).lower():
        raise ValueError("Historical extraction materials cannot be labelled fresh")
    return {"source_type": SOURCE_TYPE, "source_pdf": str(pdf), "source_pdf_sha256": pdf_hash,
            "extraction_manifest": str(manifest), "extraction_manifest_sha256": sha(manifest),
            "extraction_run_id": provenance["extraction_run_id"],
            "fresh_api_completion_evidence": "See upstream extraction manifest and shared API ledger; PDF hash alone is not proof of a paid call."}


def prepare_graph_phase(kg_data_dir, document_id, output_dir, provenance, *, expected_pages=None,
                        max_stage_requests=35, synonym_pair_limit=None):
    if not isinstance(max_stage_requests, int) or isinstance(max_stage_requests, bool) or not 1 <= max_stage_requests <= 35:
        raise ValueError("Graph stage request cap must be between 1 and 35")
    if synonym_pair_limit is not None and (not isinstance(synonym_pair_limit, int) or isinstance(synonym_pair_limit, bool) or synonym_pair_limit < 0):
        raise ValueError("synonym_pair_limit must be nonnegative or None")
    source, output = Path(kg_data_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Fresh source and graph output must be separate directories")
    source_provenance = check_provenance(provenance)
    formal = select_fresh_document(source, document_id, expected_pages)
    hashes = {filename: sha(source / filename) for filename in FILES.values()}
    # The injected budget client may already own api_results.json here. No old
    # graph, scope, review view or extraction artifacts may be overwritten.
    if output.exists() and any(p.name != "api_results.json" for p in output.iterdir()):
        raise FileExistsError("Use a new graph output directory; prior results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    input_dir = output / "input_kg_data"
    input_dir.mkdir()
    for key, filename in FILES.items():
        write(input_dir / filename, formal[key])
    scope = {**source_provenance, "document_id": document_id, "page_count": len(formal["pages"]),
             "material_count": len(formal["nodes"]), "source_kg_data": str(source), "input_file_sha256": hashes,
             "graph_output": str(output), "max_stage_requests": max_stage_requests, "synonym_pair_limit": synonym_pair_limit,
             "new_images_uploaded_in_graph_phase": False, "fresh_extraction_upstream": True,
             "database_writes": False, "embedding_requests": 0, "teacher_review_status": "pending",
             "budget_policy": "Only the injected shared-budget client may issue requests; no independent budget or credential here."}
    write(output / "validation_scope.json", scope)
    code_files = [Path(__file__), *[Path(__file__).with_name(n) for n in
                  ("all_builder.py", "graph_quality.py", "relation_review.py", "teacher_workbench.html")]]
    write(output / "code_provenance.json", {p.name: sha(p) for p in code_files})
    return formal, scope


class StageClient:
    """Defensive call cap in addition to the real client's yuan/HTTP ledger."""
    def __init__(self, delegate, maximum):
        if not callable(getattr(delegate, "chat_json", None)):
            raise TypeError("Inject a shared-budget client with chat_json")
        self.delegate, self.maximum, self.calls = delegate, maximum, 0
        self.prompt_sha256 = []
        self.model = getattr(delegate, "model", "unknown")
        self.url = getattr(delegate, "url", "unknown")
        self.cache_only = getattr(delegate, "cache_only", False)

    @property
    def remaining_requests(self):
        remaining = self.maximum - self.calls
        shared = getattr(self.delegate, "remaining_requests", remaining)
        return max(0, min(remaining, shared)) if isinstance(shared, int) else max(0, remaining)

    def chat_json(self, prompt, **kwargs):
        if self.calls >= self.maximum:
            raise RuntimeError("Graph stage request cap reached; no extra request sent")
        self.calls += 1
        self.prompt_sha256.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        try:
            return self.delegate.chat_json(prompt, **kwargs)
        except Exception:
            # Raw client errors may contain headers/credentials. The shared
            # client's sanitized ledger owns detailed diagnostics.
            raise RuntimeError("Graph request failed; inspect the shared budget client's sanitized ledger") from None


def graph_config(output):
    output = Path(output)
    defaults = inspect.signature(all_builder.build_and_store_databases).parameters
    settings = {field.name: defaults[field.name].default for field in dataclasses.fields(all_builder._PipelineConfig)}
    settings.update(kg_data_dir=str(output / "input_kg_data"), debug_dir=str(output / "no_debug_terms"),
                    work_dir=str(output / "work"), write_neo4j=False, write_pinecone=False,
                    deepseek_api_key="", resume=True, force_reextract=False, reuse_final_concepts=False,
                    page_node_batch_size=1000, data_version="fresh_pdf_shared_budget_validation_v1")
    return all_builder._PipelineConfig(**settings)


def current_task_contract(formal, config, client, synonym_pair_limit=None):
    """Use the exact production page fingerprint policy, without sending prompts."""
    policy = all_builder._knowledge_build_fingerprint(config, client)
    build = hashlib.sha256(json.dumps({"policy": policy, "synthesize_descriptions": False,
        "synonym_pair_limit": synonym_pair_limit}, sort_keys=True).encode("utf-8")).hexdigest()
    terms = all_builder._load_professional_terms(config)
    tasks = []
    for task in all_builder._group_nodes_by_page(formal, config.page_node_batch_size):
        prompt = all_builder._build_page_prompt(task, terms)
        fingerprint = hashlib.sha256(json.dumps({"prompt": prompt, "build_fingerprint": build,
            "model": getattr(client, "model", config.deepseek_model),
            "endpoint": getattr(client, "url", config.deepseek_base_url)},
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        tasks.append({"task_id": task["task_id"], "page_id": task["page_id"], "batch_index": task["batch_index"],
            "node_ids": [all_builder._normalize_text(n.get("node_id") or n.get("id")) for n in task["nodes"]],
            "task_fingerprint": fingerprint, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return {"policy_fingerprint": policy, "build_fingerprint": build, "model": getattr(client, "model", "unknown"),
            "formal_source_hash": formal["source_hash"], "tasks": tasks}


def directory_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def clone_resume_partial(resume_from, output, scope, config, contract, client):
    """Copy only fingerprint-verified successful pages from a terminal same-run phase.

    The old phase is immutable. HTTP caches, failed responses, old final graphs,
    merge decisions and graphs from another extraction batch are never imported.
    """
    previous, output = Path(resume_from).resolve(), Path(output).resolve()
    if previous == output or previous.parent != output.parent:
        raise ValueError("Resume source must be a different sibling phase in this exact run")
    snapshot = directory_hashes(previous)
    if not snapshot or "summary.json" not in snapshot:
        raise ValueError("Resume source has no terminal summary; do not clone a running phase")
    old_summary, old_scope = read(previous / "summary.json"), read(previous / "validation_scope.json")
    if old_summary.get("status") != "incomplete" or old_summary.get("input_files_unchanged") is not True:
        raise ValueError("Resume requires an incomplete terminal phase with unchanged extraction inputs")
    if Path(old_scope.get("graph_output", "")).resolve() != previous:
        raise ValueError("Resume source was moved or belongs to another saved phase")
    same_fields = ("source_type", "source_pdf_sha256", "extraction_manifest_sha256", "extraction_run_id",
                   "document_id", "page_count", "material_count", "input_file_sha256", "synonym_pair_limit")
    if any(old_scope.get(key) != scope.get(key) for key in same_fields):
        raise ValueError("Resume source differs in extraction batch, PDF, selected input, or merge policy")
    for key in ("source_pdf", "source_kg_data", "extraction_manifest"):
        if Path(old_scope.get(key, "")).resolve() != Path(scope[key]).resolve():
            raise ValueError("Resume source paths do not identify the exact current extraction run")
    if old_summary.get("model") != contract["model"]:
        raise ValueError("Resume source model differs from the injected client")
    # Strict code-level check supplements task-level model/endpoint/prompt hashes.
    # The runner may evolve to add this recovery operation; graph policy may not.
    old_code = read(previous / "code_provenance.json")
    for filename in ("all_builder.py", "relation_review.py"):
        if old_code.get(filename) != sha(Path(__file__).with_name(filename)):
            raise ValueError("Resume page extraction/review algorithm differs from the prior phase")
    export_code_changes = {filename: {"previous_sha256": old_code.get(filename),
        "current_sha256": sha(Path(__file__).with_name(filename))}
        for filename in ("graph_quality.py", "teacher_workbench.html")
        if old_code.get(filename) != sha(Path(__file__).with_name(filename))}
    for filename in FILES.values():
        if sha(previous / "input_kg_data" / filename) != sha(output / "input_kg_data" / filename):
            raise ValueError("Resume selected input bytes differ")
    ledger = getattr(client, "ledger", None)
    ledger_path = getattr(ledger, "path", None)
    if ledger_path is not None:
        old_api = read(previous / "api_results.json")
        if not isinstance(old_api, list) or any(r.get("budget_ledger_path") != str(ledger_path) for r in old_api):
            raise ValueError("Resume source API records belong to another shared budget ledger")
    partial = previous / "work" / Path(config.partial_path).name
    expected = {task["task_id"]: task for task in contract["tasks"]}
    latest = {}
    for raw_line in partial.read_bytes().splitlines(keepends=True):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        task = expected.get(record.get("task_id"))
        if task is None or any(record.get(key) != task[key] for key in ("page_id", "batch_index", "node_ids", "task_fingerprint")):
            raise ValueError("Resume partial task differs from actual current prompts, model, endpoint, or input nodes")
        if record.get("status") not in {"success", "error"}:
            raise ValueError("Resume partial task has no explicit success/error status")
        if not isinstance(record.get("concepts"), list) or not isinstance(record.get("relationships"), list):
            raise ValueError("Resume partial task contains malformed page results")
        latest[record["task_id"]] = (record, raw_line)
    reused = [task for task in contract["tasks"] if task["task_id"] in latest and latest[task["task_id"]][0]["status"] == "success"]
    copied = b"".join(latest[task["task_id"]][1].rstrip(b"\r\n") + b"\n" for task in reused)
    if directory_hashes(previous) != snapshot:
        raise ValueError("Resume source changed during verification; wait for the original worker to finish")
    Path(config.partial_path).write_bytes(copied)
    details = {"resume_from": str(previous), "previous_terminal_status": old_summary["status"],
        "same_extraction_run_id": scope["extraction_run_id"], "verified_previous_file_sha256": snapshot,
        "reused_task_ids": [task["task_id"] for task in reused],
        "reused_page_ids": list(dict.fromkeys(task["page_id"] for task in reused)),
        "page_task_ids_to_request": [task["task_id"] for task in contract["tasks"] if task not in reused],
        "verified_current_build_fingerprint": contract["build_fingerprint"],
        "copied_partial_sha256": sha(config.partial_path), "old_api_cache_copied": False,
        "non_page_export_code_changes": export_code_changes,
        "note": "Reused pages were generated earlier in this same fresh extraction batch. This phase sends only missing/failed page prompts plus any synonym review requests; its shared budget records account only for new requests."}
    write(output / "resume_provenance.json", details)
    return details


def phase_request_provenance(client, before, contract, stage_client, resume):
    records = getattr(client, "records", [])
    start = before.get("record_count")
    new_records = records[start:] if isinstance(records, list) and isinstance(start, int) else []
    digest_to_task = {task["prompt_sha256"]: task["task_id"] for task in contract["tasks"]}
    http_records = [{"budget_record_id": record["budget_record_id"], "task_id": digest_to_task.get(record.get("prompt_sha256")),
                     "status": record.get("status"), "finish_reason": record.get("finish_reason")}
                    for record in new_records if record.get("budget_record_id")]
    return {"reused_task_ids": resume["reused_task_ids"] if resume else [],
        "reused_page_ids": resume["reused_page_ids"] if resume else [],
        "page_task_delegate_calls_this_phase": [digest_to_task[digest] for digest in stage_client.prompt_sha256 if digest in digest_to_task],
        "new_http_budget_record_count": len(http_records), "new_http_budget_records": http_records,
        "http_accounting_available": all(record.get("budget_record_id") or record.get("replayed") is True for record in new_records) if new_records else True,
        "note": "Delegate calls may include cache hits or pre-HTTP failures. HTTP budget records and the shared ledger, not delegate-call count, determine new paid attempts and cost."}


def client_usage(client):
    function = getattr(client, "stage_usage", None)
    usage = function() if callable(function) else {}
    records = getattr(client, "records", [])
    if not isinstance(usage, dict):
        usage = {}
    return {"record_count": len(records) if isinstance(records, list) else None,
            "usage": {key: usage.get(key, sum((r.get("usage") or {}).get(key, 0) for r in records)
                      if isinstance(records, list) else 0)
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}


def graph_metrics(graph, formal=None):
    prepared, audit = prepare_knowledge(graph)
    concepts, relations = prepared["concepts"], prepared["relationships"]
    metrics = {"concept_count": len(concepts), "relationship_count": len(relations),
               "hierarchy_candidates": len(prepared["hierarchy_candidates"]), "quarantined_records": len(prepared["review_items"]),
               "concepts_without_evidence": audit["concepts_without_evidence"],
               "alias_name_conflicts": audit["alias_name_conflicts"], "shared_aliases": audit["shared_aliases"],
               "hierarchy_cycle_nodes": audit["hierarchy_cycle_nodes"], "prerequisite_cycle_nodes": audit["prerequisite_cycle_nodes"],
               "relationship_types": dict(Counter(r.get("relation") for r in relations)),
               "suggested_roles": dict(Counter(c.get("suggested_teaching_role", "pending") for c in concepts)),
               "page_observation_count": sum(len(c.get("page_observations", [])) for c in concepts),
               "model_concepts_marked_approved": sum(c.get("review_status") == "approved" or c.get("role_review_status") == "approved" for c in concepts),
               "relationships_marked_approved": sum(r.get("review_status") == "approved" for r in relations),
               "evidence_validation_available": formal is not None,
               "note": "Structural/source diagnostics only. Counts, roles and absence of cycles do not measure semantic accuracy or learning effect."}
    if formal is not None:
        pages = {p["page_id"]: p for p in formal["pages"]}
        evidence_pages = {n["node_id"]: all_builder._get_page_id_from_node(n) for n in formal["nodes"]}
        evidence_pages.update({pid: pid for pid in pages})
        errors = []
        for concept in concepts:
            for pid in concept.get("source_page_ids", []):
                if pid not in pages:
                    errors.append({"concept": concept["canonical_name"], "unknown_page_id": pid})
            for eid in concept.get("evidence_node_ids", []):
                if eid not in evidence_pages:
                    errors.append({"concept": concept["canonical_name"], "unknown_evidence_id": eid})
            for observation in concept.get("page_observations", []):
                pid = observation.get("page_id")
                if pid not in pages:
                    errors.append({"concept": concept["canonical_name"], "observation_unknown_page_id": pid})
                for eid in observation.get("evidence_node_ids", []):
                    if evidence_pages.get(eid) != pid:
                        errors.append({"concept": concept["canonical_name"], "observation_page_id": pid, "wrong_evidence_id": eid})
        for relation in relations:
            for eid in relation.get("evidence_node_ids", []):
                if eid not in evidence_pages:
                    errors.append({"relation": [relation["source"], relation["target"], relation["relation"]], "unknown_evidence_id": eid})
        metrics.update(source_page_count=len(pages), source_consistency_errors=errors)
    return metrics


def run_graph_phase(kg_data_dir, document_id, output_dir, client, provenance, *, expected_pages=None,
                    max_stage_requests=35, synonym_pair_limit=None, baseline_graph=None, baseline_formal=None, resume_from=None,
                    replay_page_task_order=None):
    stage_client = StageClient(client, max_stage_requests)
    if replay_page_task_order is not None and (not stage_client.cache_only or resume_from is not None):
        raise ValueError("An explicit replay order requires a cache-only full response rebuild without partial reuse")
    formal, scope = prepare_graph_phase(kg_data_dir, document_id, output_dir, provenance,
        expected_pages=expected_pages, max_stage_requests=max_stage_requests, synonym_pair_limit=synonym_pair_limit)
    output = Path(output_dir).resolve()
    config = graph_config(output)
    Path(config.work_dir).mkdir()
    loaded = all_builder._load_formal_data(config)
    if replay_page_task_order is not None:
        tasks = all_builder._group_nodes_by_page(loaded, config.page_node_batch_size)
        expected = {task["task_id"] for task in tasks}
        if (not isinstance(replay_page_task_order, list) or len(replay_page_task_order) != len(expected)
                or set(replay_page_task_order) != expected):
            raise ValueError("Replay order must contain every current page task exactly once")
        by_task = {task["task_id"]: task for task in tasks}
        page_order = list(dict.fromkeys(by_task[task_id]["page_id"] for task_id in replay_page_task_order))
        pages_by_id = {page["page_id"]: page for page in loaded["pages"]}
        loaded["pages"] = [pages_by_id[page_id] for page_id in page_order]
        if [t["task_id"] for t in all_builder._group_nodes_by_page(loaded, config.page_node_batch_size)] != replay_page_task_order:
            raise ValueError("Replay task order cannot be represented as this document's complete page sequence")
        scope["replay_page_task_order"] = replay_page_task_order
        scope["replay_order_note"] = "Only in-memory page scheduling follows the original successful response completion order so synonym prompts reproduce exactly; selected source files and their byte hashes are unchanged."
        write(output / "validation_scope.json", scope)
    contract = current_task_contract(loaded, config, stage_client, synonym_pair_limit)
    write(output / "page_task_contract.json", contract)
    resume = clone_resume_partial(resume_from, output, scope, config, contract, client) if resume_from is not None else None
    before, started = client_usage(client), time.perf_counter()
    summary = {**scope, "status": "incomplete", "model": stage_client.model,
               "resume_provenance": resume,
               "cost_accounting": "The shared API budget ledger is authoritative; this runner adds no independent yuan estimate."}
    try:
        graph, report = all_builder._build_knowledge_layer(loaded, config, client=stage_client,
            synthesize_descriptions=False, synonym_pair_limit=synonym_pair_limit)
        exports = export_teacher_bundle(graph, loaded, output / "teacher_view")
        metrics = graph_metrics(graph, formal)
        write(output / "quality_metrics.json", metrics)
        if baseline_graph is not None:
            baseline = read(baseline_graph) if isinstance(baseline_graph, (str, Path)) else baseline_graph
            write(output / "quality_comparison.json", {"current": metrics, "baseline": graph_metrics(baseline, baseline_formal),
                "note": "Source extraction provenance differs. This is a diagnostic comparison without teacher-labelled ground truth; output counts are not accuracy scores."})
        summary.update(status="success", graph_report=report, exports=exports, metrics=metrics)
    except Exception as error:
        summary.update(error_type=type(error).__name__,
                       error="Graph phase did not complete. Inspect saved partial tasks and the shared client's sanitized ledger.")
    after = client_usage(client)
    summary.update(stage_chat_calls=stage_client.calls, seconds=round(time.perf_counter()-started, 3),
                   client_accounting_before=before, client_accounting_after=after,
                   page_task_provenance=phase_request_provenance(client, before, contract, stage_client, resume),
                   new_usage={key: after["usage"][key]-before["usage"][key] for key in before["usage"]})
    summary["input_files_unchanged"] = all(sha(Path(kg_data_dir) / name) == digest for name, digest in scope["input_file_sha256"].items())
    if not summary["input_files_unchanged"]:
        summary["status"] = "incomplete"
        summary["error"] = "Extraction input changed during validation; do not use the result as a fixed-input comparison."
    if resume is not None:
        summary["resume_source_files_unchanged"] = directory_hashes(Path(resume["resume_from"])) == resume["verified_previous_file_sha256"]
        if not summary["resume_source_files_unchanged"]:
            summary["status"] = "incomplete"
            summary["error"] = "Original phase changed during recovery; the saved result needs provenance review."
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare a fresh unit graph validation; never sends API requests from CLI")
    parser.add_argument("--kg-data", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--expected-pages", type=int, help="Optional exact page-count constraint; otherwise infer the complete set from the selected document")
    parser.add_argument("--max-stage-requests", type=int, default=35)
    parser.add_argument("--synonym-pair-limit", type=int)
    args = parser.parse_args()
    _, scope = prepare_graph_phase(args.kg_data, args.document_id, args.output, read(args.provenance),
        expected_pages=args.expected_pages, max_stage_requests=args.max_stage_requests, synonym_pair_limit=args.synonym_pair_limit)
    print(json.dumps({"status": "prepared_only", "scope": scope}, ensure_ascii=False))


if __name__ == "__main__":
    main()
