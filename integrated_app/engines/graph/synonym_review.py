"""Validate offline synonym suggestions and export a pending teacher-review queue.

This module only reads the explicitly supplied JSON files.  It has no API,
network, credential, graph-writing, approval, or synonym-merging integration.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DECISIONS = ("same_proposal", "distinct", "related_not_synonym", "uncertain")
CROSSCHECK_DECISIONS = {
    "keep_proposal": "same_proposal",
    "downgrade_to_related": "related_not_synonym",
    "downgrade_to_uncertain": "uncertain",
}
REVIEWER = "codex_offline_review"
REVIEW_STATUS = "pending"
NOTICE = "Codex离线复核建议，不是教师审核；所有条目待教师确认，不得自动批准或合并。"
REPORT_NAMES = ("synonym_review.json", "synonym_review.md")


class ReviewValidationError(ValueError):
    """An input cannot safely form an evidence-backed review queue."""


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewValidationError(f"{context} must be a non-empty string")
    return value


def _read_arrays(paths: Iterable[str | Path], label: str) -> tuple[list[dict], list[dict]]:
    """Read only named files and hash the exact bytes used for validation."""
    rows: list[dict] = []
    sources: list[dict] = []
    for supplied_path in paths:
        path = Path(supplied_path).resolve()
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReviewValidationError(f"{label} file {path}: invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise ReviewValidationError(f"{label} file {path} must contain a JSON array")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ReviewValidationError(f"{label} file {path}, item {index + 1} must be an object")
            rows.append(item)
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    if not sources:
        raise ReviewValidationError(f"at least one {label} file is required")
    return rows, sources


def _validate_citations(pair_id: str, references: Any, evidence: dict[str, dict], required: bool) -> list[dict]:
    if not isinstance(references, list):
        raise ReviewValidationError(f"{pair_id} evidence_refs must be an array")
    if required and not references:
        raise ReviewValidationError(f"{pair_id} non-uncertain decision or crosscheck requires at least one citation")
    enriched = []
    for reference in references:
        if not isinstance(reference, dict):
            raise ReviewValidationError(f"{pair_id} citations must be objects")
        evidence_id = _text(reference.get("evidence_id"), f"{pair_id} citation evidence_id")
        source = evidence.get(evidence_id)
        if source is None:
            raise ReviewValidationError(f"{pair_id} citation evidence_id does not belong to this pair: {evidence_id}")
        quote = _text(reference.get("quote"), f"{pair_id} citation quote")
        if len(quote) > 180:
            raise ReviewValidationError(f"{pair_id} citation quote exceeds 180 characters")
        if quote not in source["text"]:
            raise ReviewValidationError(f"{pair_id} citation quote must be an exact contiguous substring of {evidence_id}")
        enriched.append({
            "evidence_id": evidence_id,
            "quote": quote,
            "page_id": source.get("page_id"),
            "page_no": source.get("page_no"),
            "kind": source.get("kind"),
        })
    return enriched


def build_review_report(
    batch_paths: Iterable[str | Path], decision_paths: Iterable[str | Path],
    crosscheck_path: str | Path | None = None,
) -> dict:
    """Validate complete pair coverage and exact citations without changing inputs.

    A ``same_proposal`` decision is only a suggestion.  Any incoming approval,
    reviewer, or merge fields are discarded when constructing the output.
    """
    batches, batch_sources = _read_arrays(batch_paths, "batch")
    decisions, decision_sources = _read_arrays(decision_paths, "decision")
    pairs: dict[str, dict] = {}
    evidence_by_pair: dict[str, dict[str, dict]] = {}
    for pair in batches:
        pair_id = _text(pair.get("pair_id"), "batch pair_id")
        if pair_id in pairs:
            raise ReviewValidationError(f"duplicate batch pair_id: {pair_id}")
        _text(pair.get("left"), f"{pair_id} batch left")
        _text(pair.get("right"), f"{pair_id} batch right")
        evidence = pair.get("evidence")
        if not isinstance(evidence, list):
            raise ReviewValidationError(f"{pair_id} evidence must be an array")
        indexed: dict[str, dict] = {}
        for entry in evidence:
            if not isinstance(entry, dict):
                raise ReviewValidationError(f"{pair_id} evidence entries must be objects")
            evidence_id = _text(entry.get("evidence_id"), f"{pair_id} evidence_id")
            if evidence_id in indexed:
                raise ReviewValidationError(f"{pair_id} duplicate evidence_id: {evidence_id}")
            if not isinstance(entry.get("text"), str):
                raise ReviewValidationError(f"{pair_id} evidence {evidence_id} text must be a string")
            indexed[evidence_id] = entry
        pairs[pair_id] = pair
        evidence_by_pair[pair_id] = indexed
    if not pairs:
        raise ReviewValidationError("batch files contain no candidate pairs")

    reviewed: dict[str, dict] = {}
    for suggestion in decisions:
        pair_id = _text(suggestion.get("pair_id"), "decision pair_id")
        if pair_id in reviewed:
            raise ReviewValidationError(f"duplicate decision pair_id: {pair_id}")
        if pair_id not in pairs:
            raise ReviewValidationError(f"unknown decision pair_id: {pair_id}")
        pair = pairs[pair_id]
        for side in ("left", "right"):
            if suggestion.get(side) != pair[side]:
                raise ReviewValidationError(f"{pair_id} {side} must exactly match the batch")
        decision = suggestion.get("decision")
        if not isinstance(decision, str) or decision not in DECISIONS:
            raise ReviewValidationError(f"{pair_id} decision must be one of {', '.join(DECISIONS)}")
        reason = _text(suggestion.get("reason"), f"{pair_id} reason")
        enriched_refs = _validate_citations(pair_id, suggestion.get("evidence_refs", []),
                                            evidence_by_pair[pair_id], decision != "uncertain")
        reviewed[pair_id] = {
            "pair_id": pair_id,
            "left": pair["left"],
            "right": pair["right"],
            "decision": decision,
            "reason": reason,
            "evidence_refs": enriched_refs,
            "review_status": REVIEW_STATUS,
            "reviewer": REVIEWER,
        }
    missing = sorted(set(pairs) - set(reviewed))
    if missing:
        raise ReviewValidationError(f"missing decisions for pair_id: {', '.join(missing)}")

    initial_proposals = {pair_id for pair_id, item in reviewed.items() if item["decision"] == "same_proposal"}
    crosschecks: dict[str, dict] = {}
    crosscheck_sources = []
    if crosscheck_path is not None:
        crosscheck_rows, crosscheck_sources = _read_arrays([crosscheck_path], "crosscheck")
        for crosscheck in crosscheck_rows:
            pair_id = _text(crosscheck.get("pair_id"), "crosscheck pair_id")
            if pair_id in crosschecks:
                raise ReviewValidationError(f"duplicate crosscheck pair_id: {pair_id}")
            if pair_id not in initial_proposals:
                raise ReviewValidationError(f"crosscheck pair_id is not an initial same_proposal: {pair_id}")
            for side in ("left", "right"):
                if side in crosscheck and crosscheck[side] != pairs[pair_id][side]:
                    raise ReviewValidationError(f"{pair_id} crosscheck {side} must exactly match the batch")
            verdict = crosscheck.get("verdict")
            if not isinstance(verdict, str) or verdict not in CROSSCHECK_DECISIONS:
                raise ReviewValidationError(f"{pair_id} crosscheck verdict must be one of {', '.join(CROSSCHECK_DECISIONS)}")
            crosschecks[pair_id] = {
                "pair_id": pair_id,
                "verdict": verdict,
                "reason": _text(crosscheck.get("reason"), f"{pair_id} crosscheck reason"),
                "evidence_refs": _validate_citations(pair_id, crosscheck.get("evidence_refs", []),
                                                     evidence_by_pair[pair_id], required=True),
                "review_status": REVIEW_STATUS,
                "reviewer": REVIEWER,
            }
        missing_crosschecks = sorted(initial_proposals - set(crosschecks))
        if missing_crosschecks:
            raise ReviewValidationError(f"missing crosschecks for initial same_proposal pair_id: {', '.join(missing_crosschecks)}")
    items = []
    for pair_id in pairs:
        initial = reviewed[pair_id]
        crosscheck = crosschecks.get(pair_id)
        item = {
            **initial,
            "initial_decision": initial["decision"],
            "initial_assessment": initial,
            "crosscheck": crosscheck,
        }
        if crosscheck is not None:
            item.update(decision=CROSSCHECK_DECISIONS[crosscheck["verdict"]],
                        reason=crosscheck["reason"], evidence_refs=crosscheck["evidence_refs"])
        items.append(item)
    return {
        "metadata": {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": REVIEWER,
            "review_status": REVIEW_STATUS,
            "notice": NOTICE,
            "teacher_approved": False,
            "offline_only": True,
            "api_calls": 0,
            "network_used": False,
            "automatic_merge_allowed": False,
            "input_graph_modified": False,
            "crosscheck_applied": crosscheck_path is not None,
            "input_files": {"batches": batch_sources, "decisions": decision_sources, "crosscheck": crosscheck_sources},
        },
        "summary": {
            "total_pairs": len(items),
            "decision_counts": {decision: sum(item["decision"] == decision for item in items) for decision in DECISIONS},
            "initial_decision_counts": {decision: sum(item["initial_decision"] == decision for item in items) for decision in DECISIONS},
            "crosscheck_count": len(crosschecks),
            "crosscheck_verdict_counts": {verdict: sum(item["verdict"] == verdict for item in crosschecks.values()) for verdict in CROSSCHECK_DECISIONS},
            "pending_count": len(items),
            "teacher_approved_count": 0,
            "citation_count": sum(len(item["evidence_refs"]) for item in items),
        },
        "items": items,
    }


def _markdown_text(value: Any) -> str:
    value = html.escape(str(value), quote=False).replace("\\", "\\\\")
    return re.sub(r"([`*_{}\[\]()#+.!|>~-])", r"\\\1", value).replace("\n", " ")


def _quote_block(quote: str) -> str:
    # A source quotation may itself contain Markdown fences.
    longest = max((len(run) for run in re.findall(r"`+", quote)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{quote}\n{fence}"


def render_review_markdown(report: dict) -> str:
    """Render every suggestion with its exact source excerpts for a teacher."""
    lines = ["# 同义候选离线复核建议", "", NOTICE, "",
             f"复核者：{REVIEWER}。所有条目状态：pending。API调用：0；不修改输入图。", "",
             f"候选共 {report['summary']['total_pairs']} 对，引用 {report['summary']['citation_count']} 条。", "",
             "| 决策建议 | 数量 |", "| --- | ---: |"]
    for decision, count in report["summary"]["decision_counts"].items():
        lines.append(f"| {decision} | {count} |")
    lines.extend(["", "## 输入文件与SHA256", ""])
    for category, sources in report["metadata"]["input_files"].items():
        for source in sources:
            lines.append(f"- {category}：{_markdown_text(source['path'])}；SHA256：{source['sha256']}")
    for index, item in enumerate(report["items"], 1):
        lines.extend(["", f"## {index}. {_markdown_text(item['pair_id'])}", "",
                      f"概念：{_markdown_text(item['left'])} ↔ {_markdown_text(item['right'])}", "",
                      f"决策建议：{item['decision']}；状态：pending；复核者：{REVIEWER}；教师审核：未确认。", "",
                      f"理由：{_markdown_text(item['reason'])}", ""])
        if item["crosscheck"] is not None:
            initial = item["initial_assessment"]
            lines.extend([f"交叉审查：{item['crosscheck']['verdict']}。初次建议：{initial['decision']}。", "",
                          f"初次理由：{_markdown_text(initial['reason'])}", ""])
            for reference_index, reference in enumerate(initial["evidence_refs"], 1):
                source = "；".join(f"{key}={_markdown_text(reference.get(key))}" for key in ("evidence_id", "page_id", "page_no", "kind"))
                lines.extend([f"初次证据 {reference_index}：{source}", "", _quote_block(reference["quote"]), ""])
        if not item["evidence_refs"]:
            lines.append("引用：无；证据不足，保留待教师复核。")
        for reference_index, reference in enumerate(item["evidence_refs"], 1):
            source = "；".join(f"{key}={_markdown_text(reference.get(key))}" for key in ("evidence_id", "page_id", "page_no", "kind"))
            lines.extend([f"证据 {reference_index}：{source}", "", _quote_block(reference["quote"]), ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_review_reports(
    batch_paths: Iterable[str | Path], decision_paths: Iterable[str | Path], output_dir: str | Path,
    crosscheck_path: str | Path | None = None,
) -> dict:
    """Write JSON and Markdown only after all input validation succeeds."""
    report = build_review_report(batch_paths, decision_paths, crosscheck_path)
    directory = Path(output_dir).resolve()
    input_paths = {Path(source["path"]) for sources in report["metadata"]["input_files"].values() for source in sources}
    if any((directory / name).resolve() in input_paths for name in REPORT_NAMES):
        raise ReviewValidationError("output reports must not overwrite an input file")
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_review_markdown(report)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REPORT_NAMES[0]).write_text(json_text, encoding="utf-8")
    (directory / REPORT_NAMES[1]).write_text(markdown_text, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate offline synonym suggestions and export a pending teacher-review queue; never approve or merge.")
    parser.add_argument("--batches", nargs="+", required=True, type=Path, help="Candidate batch JSON arrays")
    parser.add_argument("--decisions", nargs="+", required=True, type=Path, help="Offline decision JSON arrays")
    parser.add_argument("--crosscheck", type=Path, help="Optional crosscheck JSON array covering every initial same_proposal, with no extra pairs")
    parser.add_argument("--output", required=True, type=Path, help="Directory for synonym_review.json and synonym_review.md")
    args = parser.parse_args(argv)
    try:
        report = generate_review_reports(args.batches, args.decisions, args.output, args.crosscheck)
    except (ReviewValidationError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "summary": report["summary"], "review_status": REVIEW_STATUS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
