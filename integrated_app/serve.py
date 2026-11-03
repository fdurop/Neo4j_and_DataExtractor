"""Run the complete local app at one origin; no provider call occurs on startup."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app_config import load_settings, configure_http_proxy


def build_application(config=None):
    config = dict(config or load_settings())
    configure_http_proxy(config)
    from backend.app import create_app
    from backend.services.pipeline import run_job
    from backend.services.graph_repository import GraphRepository
    repository = GraphRepository(config)
    config.update(PIPELINE_RUNNER=run_job, PIPELINE_PUBLISHER=repository.publish)
    backend = create_app(config)
    backend.extensions["graph_repository"] = repository

    @backend.get("/api/runtime")
    def runtime_status():
        from flask import jsonify
        path = Path(config["APP_DATA_DIR"]) / "api_budget.json"
        budget = None
        if path.exists():
            sys.path.insert(0, str(ROOT / "engines/extraction"))
            from validation_budget import ValidationBudget
            budget = ValidationBudget(path, config["API_MAX_COST_YUAN"], config["API_MAX_REQUESTS"], config["QWEN_MODEL"]).snapshot()
        return jsonify(model_configured=bool(config.get("DASHSCOPE_API_KEY")),
                       paid_pipeline_enabled=config.get("ENABLE_PAID_PIPELINE", False),
                       database_mode=config.get("DATABASE_MODE"), budget=budget)

    sys.path.insert(0, str(ROOT / "frontend"))
    os.environ["DJANGO_SETTINGS_MODULE"] = "fdu_frontend.settings"
    from django.conf import settings
    settings.SECRET_KEY = backend.config["SECRET_KEY"]
    settings.ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
    from django.core.wsgi import get_wsgi_application
    from django.contrib.staticfiles.handlers import StaticFilesHandler
    frontend = StaticFilesHandler(get_wsgi_application())

    def application(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET")
        use_backend = path.startswith("/api/") or path == "/health" or (
            method != "GET" and path.rstrip("/") in {"/login", "/register", "/logout"})
        if use_backend and path.rstrip("/") in {"/login", "/register", "/logout"}:
            environ["PATH_INFO"] = path.rstrip("/")
        return (backend if use_backend else frontend)(environ, start_response)

    return application, backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_settings(args.config)
    from backend.services.instance_lock import InstanceLock
    with InstanceLock(config["APP_DATA_DIR"]):
        application, backend = build_application(config)
        from werkzeug.serving import make_server
        server = make_server("127.0.0.1", args.port or config["APP_PORT"], application, threaded=True)
        print(json.dumps({"status": "ready", "url": "http://127.0.0.1:" + str(server.server_port),
                          "database_mode": config["DATABASE_MODE"], "paid_pipeline_enabled": config["ENABLE_PAID_PIPELINE"]}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            backend.extensions["job_service"].shutdown()
            backend.extensions["graph_repository"].close()


if __name__ == "__main__":
    main()
