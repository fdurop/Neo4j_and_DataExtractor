"""Lightweight application factory; providers load only in explicit jobs."""
from pathlib import Path
import os
import secrets
import uuid

from flask import Flask, g, jsonify, session
from werkzeug.exceptions import HTTPException

from backend.models.db import db
from backend.models.user_model import User


def create_app(config=None):
    app = Flask(__name__)
    supplied = dict(config or {})
    data_dir = Path(supplied.get("APP_DATA_DIR") or os.getenv("APP_DATA_DIR") or
                    Path(__file__).resolve().parents[1] / "runtime").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    secret_path = data_dir / ".session_key"
    secret = supplied.get("SECRET_KEY")
    if not secret:
        if not secret_path.exists():
            try:
                with secret_path.open("x", encoding="ascii") as handle:
                    handle.write(secrets.token_hex(32))
            except FileExistsError:
                pass
        secret = secret_path.read_text(encoding="ascii").strip()
    app.config.from_mapping(
        APP_DATA_DIR=str(data_dir), SECRET_KEY=secret,
        SQLALCHEMY_DATABASE_URI="sqlite:///" + (data_dir / "users.db").as_posix(),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        KG_GRAPH_PATH=str(data_dir / "graphs.json"),
        MAX_CONTENT_LENGTH=64 * 1024 * 1024, MAX_PDF_PAGES=200,
        JOB_WORKERS=1, AUTO_START_JOBS=True, ALLOW_TEACHER_REGISTRATION=True,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
    )
    app.config.update(supplied)
    # load_settings intentionally returns None when no explicit secret is set.
    # Preserve the generated, persistent key rather than disabling sessions.
    app.config["SECRET_KEY"] = secret
    app.config["APP_DATA_DIR"] = str(data_dir)
    Path(app.config["KG_GRAPH_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    with app.app_context():
        db.create_all()
    from backend.services.course_store import CourseStore
    from backend.services.jobs import JobService
    from backend.routes.auth import auth_bp
    from backend.routes.courses import courses_bp
    from backend.routes.files import files_bp
    from backend.routes.question import qa_bp
    from backend.routes.knowledge_graph import kg_bp
    from backend.routes.repair import repair_bp
    app.extensions["course_store"] = CourseStore(data_dir / "courses.json")
    app.extensions["job_service"] = JobService(app, data_dir / "jobs")
    app.register_blueprint(auth_bp)
    app.register_blueprint(courses_bp, url_prefix="/api")
    app.register_blueprint(files_bp, url_prefix="/api")
    app.register_blueprint(qa_bp, url_prefix="/api/qa")
    app.register_blueprint(kg_bp, url_prefix="/api")
    app.register_blueprint(repair_bp, url_prefix="/api")

    @app.before_request
    def request_context():
        g.request_id = str(uuid.uuid4())
        user_id = session.get("user_id")
        g.user = db.session.get(User, user_id) if isinstance(user_id, int) else None
        if g.user is not None:
            session["role"] = g.user.role
        elif user_id is not None:
            session.clear()

    @app.get("/health")
    def health():
        return jsonify(status="ok", application="integrated_teaching_assistant")

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(success=False, error=error.description, request_id=g.get("request_id")), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.error("Request %s failed (%s)", g.get("request_id"), type(error).__name__)
        return jsonify(success=False, error="服务处理失败，请查看任务状态或检查运行配置。",
                       request_id=g.get("request_id")), 500
    return app
