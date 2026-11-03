"""Explicit real HTTP acceptance of a PDF, persistent graph edits and a question."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import time
import uuid
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8769")
    parser.add_argument("--course-name")
    parser.add_argument("--question")
    parser.add_argument("--execute-paid", action="store_true", required=True)
    args = parser.parse_args()
    output = ROOT / "runtime/verification" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    result = {"author": "昕煜 冷", "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "source_pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(), "checks": []}
    session = requests.Session()
    session.trust_env = False

    def save():
        (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def request(method, path, **kwargs):
        response = session.request(method, args.base_url + path, timeout=160, **kwargs)
        data = response.json()
        if not response.ok or data.get("success") is False:
            raise RuntimeError(f"HTTP {response.status_code}: " + str(data.get("error", "请求失败")))
        return data

    def passed(name, value=True):
        if not value:
            raise RuntimeError("验收失败：" + name)
        result["checks"].append({"name": name, "passed": True})
        save()

    try:
        identity = "integration_" + uuid.uuid4().hex[:10]
        password = secrets.token_urlsafe(24)
        request("POST", "/register", json={"username": identity, "password": password,
                "email": identity + "@example.invalid", "role": "teacher"})
        request("POST", "/login", json={"username": identity, "password": password})
        # This local-only handoff is ignored by Git and never copied to reports.
        (ROOT / "runtime/demo_login.local.json").write_text(json.dumps({"username": identity, "password": password,
                    "base_url": args.base_url}, ensure_ascii=False), encoding="utf-8")
        passed("real_teacher_registration_and_session")
        name = args.course_name or args.pdf.stem + " 联调 " + output.name
        response = request("POST", "/api/courses", json={"name": name})
        course = response.get("course", response)
        course_id = course.get("course_id") or course["id"]
        result.update(course_id=course_id, course_name=name, graph_url=args.base_url + "/knowledge-graph/?course_id=" + course_id)
        passed("new_independent_course")
        bad = session.post(args.base_url + "/api/upload", data={"course_id": course_id, "course_name": name},
                           files={"file": ("broken.pdf", b"invalid pdf", "application/pdf")}, timeout=20)
        passed("broken_pdf_rejected_before_model_calls", bad.status_code == 400)
        with args.pdf.open("rb") as source:
            uploaded = request("POST", "/api/upload", data={"course_id": course_id, "course_name": name},
                               files={"file": (args.pdf.name, source, "application/pdf")})
        job_id = uploaded.get("job_id") or uploaded["job"]["id"]
        result["job_id"] = job_id
        save()
        print(json.dumps({"verification": str(output), "course_id": course_id, "job_id": job_id}, ensure_ascii=False), flush=True)
        previous = None
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            state = request("GET", "/api/jobs/" + job_id)["job"]
            marker = (state["status"], state["stage"], state["progress"], state.get("message"))
            result["latest_job"] = state
            save()
            if marker != previous:
                print(json.dumps({key: state.get(key) for key in ("status", "stage", "progress", "message", "error")}, ensure_ascii=False), flush=True)
                previous = marker
            if state["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(2)
        if state["status"] != "succeeded":
            raise RuntimeError("真实任务未完成：" + str(state.get("error") or state["status"]))
        passed("fresh_pdf_extraction_graph_database_publication")
        query = {"course_id": course_id, "course_name": name}
        graph = request("GET", "/api/graph", params=query)
        passed("database_graph_readable_via_app", bool(graph.get("nodes")))
        passed("graph_nodes_course_scoped", all(node.get("properties", {}).get("course_id") in {None, course_id} for node in graph["nodes"]))
        concepts = [node for node in graph["nodes"] if node.get("type") == "concept"]
        ids = {node["id"] for node in concepts}
        result["graph_counts"] = {"concepts": len(concepts), "semantic_relations": sum(link.get("source") in ids and link.get("target") in ids for link in graph.get("links", [])),
                                  "nodes": len(graph["nodes"]), "links": len(graph.get("links", [])),
                                  "hierarchy_candidates": len(graph.get("hierarchy_candidates", []))}
        node = concepts[0]
        original = node.get("description", "")
        description = original + " [temporary persistence verification]"
        request("PUT", "/api/graph/node", params=query, json={"course_name": name, "id": node["id"], "description": description})
        edited = request("GET", "/api/graph", params=query)
        passed("edited_description_survives_reload", any(n["id"] == node["id"] and n["description"] == description for n in edited["nodes"]))
        request("PUT", "/api/graph/node", params=query, json={"course_name": name, "id": node["id"], "description": original})
        restored = request("GET", "/api/graph", params=query)
        passed("temporary_edit_restored_without_teacher_approval", any(n["id"] == node["id"] and n["description"] == original for n in restored["nodes"]))
        response = session.get(args.base_url + state["source_url"], timeout=20)
        passed("source_pdf_download_hash_matches_upload", hashlib.sha256(response.content).hexdigest() == result["source_pdf_sha256"])
        if args.question:
            answered = request("POST", "/api/qa/", json={"course_id": course_id, "course_name": name, "question": args.question})
            passed("course_question_returns_checked_sources", bool(answered.get("answer") and answered.get("sources") and answered.get("model_called")))
            (output / "question_result.json").write_text(json.dumps(answered, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime = request("GET", "/api/runtime")
        result["budget"] = runtime.get("budget")
        result["status"] = "success"
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)[:1000]
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"status": result["status"], "report": str(output / "report.json"), "checks": len(result["checks"]),
                      "budget": result.get("budget"), "error": result.get("error")}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
