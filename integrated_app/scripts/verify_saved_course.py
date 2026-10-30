"""Verify an existing course after restart without invoking a model."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8769")
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--login-file", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads(args.login_file.read_text(encoding="utf-8"))
    session = requests.Session()
    session.trust_env = False
    report = {"author": "昕煜 冷", "started_at": datetime.now(timezone.utc).isoformat(),
              "course_id": args.course_id, "checks": [], "model_requests": 0}

    def get(path, **kwargs):
        response = session.get(args.base_url + path, timeout=160, **kwargs)
        response.raise_for_status()
        return response

    def check(name, passed):
        if not passed:
            raise RuntimeError("Verification failed: " + name)
        report["checks"].append({"name": name, "passed": True})

    try:
        response = session.post(args.base_url + "/login", json={key: identity[key] for key in ("username", "password")}, timeout=20)
        check("login_after_restart", response.ok and response.json().get("success") is not False)
        before = get("/api/runtime").json()["budget"]
        check("database_mode_is_neo4j", get("/api/runtime").json()["database_mode"] == "neo4j")
        graph = get("/api/graph", params={"course_id": args.course_id}).json()
        nodes = graph.get("nodes", [])
        check("published_graph_readable_after_restart", bool(nodes))
        check("graph_nodes_stay_in_course", all(n.get("properties", {}).get("course_id") in {None, args.course_id} for n in nodes))
        jobs = get("/api/jobs", params={"course_id": args.course_id}).json()["jobs"]
        check("successful_job_remains_terminal", bool(jobs) and all(j["status"] == "succeeded" for j in jobs))
        check("source_pdf_hash_matches", hashlib.sha256(get(jobs[0]["source_url"]).content).hexdigest() == args.source_sha256)
        items = get("/api/repair/items", params={"course_id": args.course_id}).json()["items"]
        check("material_review_returns_course_items", bool(items) and all(i["course_id"] == args.course_id for i in items))
        check("materials_resolve_source_pages", all(isinstance(i.get("page_no"), int) and i["page_no"] > 0 and "#page=" in i.get("source_url", "") for i in items))
        images = [i for i in items if i.get("image_url")]
        check("material_image_available", bool(images))
        asset = get(images[0]["image_url"])
        check("image_endpoint_returns_valid_image_type", asset.headers.get("Content-Type", "").startswith("image/") and len(asset.content) > 100)
        after = get("/api/runtime").json()["budget"]
        check("verification_did_not_make_model_requests", before["request_count"] == after["request_count"])
        report.update(status="success", graph_counts={"nodes": len(nodes), "links": len(graph.get("links", []))},
                      material_count=len(items), image_count=len(images), budget=after)
    except Exception as error:
        report.update(status="failed", error=str(error)[:1000])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["status"] == "success" else 1)


if __name__ == "__main__":
    main()
