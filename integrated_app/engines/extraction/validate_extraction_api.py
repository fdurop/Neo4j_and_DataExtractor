"""Bounded paid vision smoke run on explicitly selected PDF pages.

Without --execute this only prepares local evidence. No embeddings or graph
database are touched. Each output directory is exclusive to preserve history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import fitz

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src2"))

from core.image_processor import ImageProcessor
from core.text_processor import TextProcessor
from utils.config_loader import load_vlm_config
from utils.knowledge_exporter import KnowledgeExporter
from utils.vlm_client import ApiVLMClient


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class BoundedVisionClient(ApiVLMClient):
    def __init__(self, config, maximum, output):
        super().__init__({**config, "retry_count": 0})
        self.maximum = maximum
        self.output = output
        self.records = []

    def _chat_with_image(self, image_path, prompt, **kwargs):
        if len(self.records) >= self.maximum:
            raise RuntimeError("Validation request budget exhausted; no request sent.")
        # Also override modality-specific retry settings supplied by callers.
        kwargs["retry_count"] = 0
        record = {
            "attempt": len(self.records) + 1,
            "operation": kwargs.get("operation"),
            "image": str(Path(image_path).resolve()),
            "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "provider": self.provider,
            "requested_model": self.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "started",
        }
        self.records.append(record)
        write_json(self.output / "api_calls.json", self.records)
        started = time.perf_counter()
        try:
            answer = super()._chat_with_image(image_path, prompt, **kwargs)
            record.update(status="success", returned_text=answer)
            return answer
        except Exception as error:
            message = str(error)
            if self.api_key:
                message = message.replace(self.api_key, "[REDACTED]")
            record.update(status="error", error=message)
            raise RuntimeError(message) from None
        finally:
            record.update(
                seconds=round(time.perf_counter() - started, 3),
                usage=self.last_usage,
                response_metadata=self.last_response_metadata,
            )
            write_json(self.output / "api_calls.json", self.records)
            print(f"Request {record['attempt']}/{self.maximum}: {record['status']} ({record['seconds']}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=6)
    parser.add_argument("--execute", action="store_true", help="Send selected images and cleaned page context to configured Qwen API")
    args = parser.parse_args()
    if not args.pdf.is_file() or args.output.exists():
        parser.error("Input PDF must exist and output directory must be new.")
    if not 1 <= args.max_requests <= 12:
        parser.error("This smoke runner permits 1 to 12 requests per run.")
    pages = sorted(set(args.pages))
    config = load_vlm_config()
    if args.execute and (
        config.get("provider", "qwen").lower() != "qwen"
        or urlparse(config.get("base_url", "")).hostname != "dashscope.aliyuncs.com"
        or not config.get("api_key")
    ):
        parser.error("Expected configured Qwen/DashScope credentials; no request sent.")
    args.output.mkdir(parents=True)
    assets = args.output / "assets"
    assets.mkdir()
    text_processor = TextProcessor()
    prepared = []
    with fitz.open(args.pdf) as document:
        if not pages or min(pages) < 1 or max(pages) > len(document):
            parser.error("Page numbers must be within the PDF's 1-based range.")
        repeated = text_processor.find_repeated_lines([page.get_text() for page in document])
        page_count = len(document)
        for number in pages:
            page = document[number - 1]
            preview = assets / f"page_{number}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(preview)
            images = []
            excluded = []
            for index, image in enumerate(page.get_images(full=True)):
                if image[2] * image[3] < 40000:
                    excluded.append({"image_index": index, "reason": "smoke_sample_small_image", "dimensions": [image[2], image[3]]})
                    continue
                extracted = document.extract_image(image[0])
                image_path = assets / f"page_{number}_image_{index}.{extracted['ext']}"
                image_path.write_bytes(extracted["image"])
                images.append({"index": index, "path": str(image_path.resolve())})
            text = text_processor.process_text(page.get_text(), number - 1, [image["path"] for image in images], repeated_lines=repeated)
            prepared.append({"page_no": number, "preview": str(preview.resolve()), "text_result": text, "images": images, "excluded_images": excluded})

    manifest = {
        "pdf": str(args.pdf.resolve()),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "pages": pages,
        "original_page_count": page_count,
        "selected_image_count": sum(len(page["images"]) for page in prepared),
        "max_requests": args.max_requests,
        "mode": "execute" if args.execute else "prepare_only",
        "scope": "description_and_formal_handoff_no_embeddings_no_graph_database",
        "source_code_sha256": hashlib.sha256((ROOT / "src2/core/image_processor.py").read_bytes()).hexdigest(),
        "prepared_pages": prepared,
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Prepared pages {pages}, {manifest['selected_image_count']} selected images.", flush=True)
    if not args.execute:
        return
    if manifest["selected_image_count"] > args.max_requests:
        parser.error("Sample has more images than the request budget. Nothing sent.")
    client = BoundedVisionClient(config, args.max_requests, args.output)
    processor = ImageProcessor(vlm_client=client)
    exporter = KnowledgeExporter(base_dir=str(args.output / "kg_data"), course_id="motor_validation", course_name="直流电机原理", job_id=args.output.name)
    filename = args.pdf.stem + "_pdf"
    exporter.register_document(filename, {
        "file_name": args.pdf.name, "file_path": str(args.pdf.resolve()), "file_type": "pdf",
        "total_pages": page_count, "selected_page_numbers": pages, "validation_subset": True,
    })
    results = []
    for page in prepared:
        number = page["page_no"]
        exporter.add_text(page["text_result"], filename, number - 1)
        exporter.register_page(filename, number - 1, page_image_path=page["preview"])
        for image in page["images"]:
            result = processor.process_image(image["path"], page["text_result"]["cleaned_text"])
            exporter.add_image(result, filename, number - 1, image["index"])
            results.append({"page_no": number, "image_index": image["index"], "result": result})
            write_json(args.output / "image_results.json", results)
    exporter.finalize(build_vectors=False)
    write_json(args.output / "summary.json", {
        "requests_attempted": len(client.records),
        "requests_succeeded": sum(record["status"] == "success" for record in client.records),
        "total_tokens": sum(record.get("usage", {}).get("total_tokens", 0) for record in client.records),
        "seconds": round(sum(record["seconds"] for record in client.records), 3),
        "human_review_status": "pending",
        "limits": "Selected large embedded images only. This is not a full PDF extraction or graph quality evaluation. Missing-page audit warnings are expected for this explicit subset.",
    })


if __name__ == "__main__":
    main()
