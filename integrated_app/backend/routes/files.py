"""PDF uploads, durable processing state and course-scoped file access."""
from pathlib import Path
import tempfile

from flask import Blueprint, abort, current_app, g, jsonify, request, send_file
from backend.routes.auth import require_login, require_teacher

files_bp = Blueprint("files", __name__)


def jobs():
    return current_app.extensions["job_service"]


def selected_course(values):
    return current_app.extensions["course_store"].resolve(values.get("course_id"), values.get("course_name"))


def file_record(job):
    return {"id": job["id"], "job_id": job["id"], "filename": job["original_filename"], "size": job["size"],
            "course_id": job["course_id"], "course_name": job["course_name"], "status": job["status"],
            "source_url": job["source_url"], "page_count": job["page_count"], "created_at": job["created_at"]}


@files_bp.post("/upload")
@require_teacher
def upload_file():
    course = selected_course(request.form)
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        abort(400, description="请选择PDF课件。")
    filename = uploaded.filename.strip()
    if (len(filename) > 240 or filename in {".", "..", ".pdf"} or "/" in filename or "\\" in filename
            or "\x00" in filename or not filename.lower().endswith(".pdf")):
        abort(400, description="仅支持文件名有效的PDF课件。")
    incoming = Path(current_app.config["APP_DATA_DIR"]) / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(prefix="upload-", suffix=".pdf", dir=incoming, delete=False)
    staged = Path(descriptor.name)
    descriptor.close()
    try:
        uploaded.save(staged)
        if not staged.stat().st_size:
            abort(400, description="PDF文件为空。")
        try:
            import fitz
            # In-memory parsing also avoids a Windows handle remaining open
            # after the parser rejects a corrupt file.
            with fitz.open(stream=staged.read_bytes(), filetype="pdf") as document:
                if document.needs_pass:
                    abort(400, description="请先解除PDF密码保护。")
                page_count = document.page_count
                if not 1 <= page_count <= int(current_app.config["MAX_PDF_PAGES"]):
                    abort(400, description=f"PDF页数需为1至{current_app.config['MAX_PDF_PAGES']}页。")
        except (ImportError, OSError, ValueError, RuntimeError):
            abort(400, description="无法读取PDF，请检查文件是否完整。")
        record = jobs().create(course, filename, staged, page_count, g.user.id)
    finally:
        if staged.exists():
            staged.unlink()
    return jsonify(success=True, job_id=record["id"], job=record, file=file_record(record))


@files_bp.get("/jobs")
@require_teacher
def list_jobs():
    course_id = request.args.get("course_id")
    if course_id or request.args.get("course_name"):
        course_id = selected_course(request.args)["id"]
    return jsonify(success=True, jobs=jobs().list(course_id))


@files_bp.get("/jobs/<job_id>")
@require_teacher
def get_job(job_id):
    return jsonify(success=True, job=jobs().get(job_id))


@files_bp.post("/jobs/<job_id>/cancel")
@require_teacher
def cancel_job(job_id):
    return jsonify(success=True, job=jobs().cancel(job_id))


@files_bp.get("/files")
@require_teacher
def list_files():
    course_id = request.args.get("course_id")
    if course_id or request.args.get("course_name"):
        course_id = selected_course(request.args)["id"]
    return jsonify(success=True, files=[file_record(job) for job in jobs().list(course_id)])


@files_bp.get("/files/<file_id>/")
@require_teacher
def file_info(file_id):
    return jsonify(file_record(jobs().get(file_id)))


@files_bp.get("/file-content/<file_id>/")
@files_bp.get("/jobs/<file_id>/source")
@require_login
def file_content(file_id):
    job = jobs().get(file_id, private=True)
    return send_file(job["input_path"], as_attachment=False, download_name=job["original_filename"], mimetype="application/pdf")


@files_bp.delete("/files/<file_id>")
@require_teacher
def delete_file(file_id):
    # Hide a finished upload; keep its source, paid results and audit record.
    jobs().hide(file_id)
    return jsonify(success=True, archived=True)


@files_bp.get("/file-status")
@require_teacher
def file_status():
    course_id = request.args.get("course_id")
    if course_id or request.args.get("course_name"):
        course_id = selected_course(request.args)["id"]
    return jsonify(jobs().list(course_id))


@files_bp.post("/cancel/<job_id>")
@require_teacher
def cancel_legacy(job_id):
    return jsonify(success=True, job=jobs().cancel(job_id))
