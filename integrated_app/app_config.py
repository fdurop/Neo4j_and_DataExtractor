"""Explicit local configuration; importing this module does not read secrets."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


def read_env(path):
    values = {}
    path = Path(path)
    if path.is_file():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_settings(env_path=None):
    path = Path(env_path or os.getenv("APP_CONFIG_FILE", ROOT / "config/local.env")).resolve()
    values = read_env(path)
    names = {"APP_DATA_DIR", "APP_SECRET_KEY", "APP_PORT", "DATABASE_MODE", "NEO4J_URI", "NEO4J_USER",
             "NEO4J_PASSWORD", "NEO4J_DATABASE", "NEO4J_CA_BUNDLE", "DASHSCOPE_API_KEY", "QWEN_MODEL",
             "ENABLE_PAID_PIPELINE", "API_MAX_COST_YUAN", "API_MAX_REQUESTS", "MAX_PDF_PAGES"}
    values.update({key: os.environ[key] for key in names if key in os.environ})
    data = Path(values.get("APP_DATA_DIR") or ROOT / "runtime")
    if not data.is_absolute():
        data = ROOT / data
    mode = values.get("DATABASE_MODE", "local")
    if mode not in {"local", "neo4j"}:
        raise ValueError("DATABASE_MODE must be local or neo4j")
    return {
        "APP_ROOT": str(ROOT), "APP_CONFIG_FILE": str(path), "APP_DATA_DIR": str(data.resolve()),
        "SECRET_KEY": values.get("APP_SECRET_KEY"), "APP_PORT": int(values.get("APP_PORT", "8769")),
        "DATABASE_MODE": mode, "NEO4J_URI": values.get("NEO4J_URI", ""),
        "NEO4J_USER": values.get("NEO4J_USER", ""), "NEO4J_PASSWORD": values.get("NEO4J_PASSWORD", ""),
        "NEO4J_DATABASE": values.get("NEO4J_DATABASE") or None,
        "NEO4J_CA_BUNDLE": values.get("NEO4J_CA_BUNDLE") or None,
        "DASHSCOPE_API_KEY": values.get("DASHSCOPE_API_KEY", ""),
        "QWEN_MODEL": values.get("QWEN_MODEL", "qwen3.7-plus"),
        "ENABLE_PAID_PIPELINE": values.get("ENABLE_PAID_PIPELINE", "0").lower() in {"1", "true", "yes"},
        "API_MAX_COST_YUAN": values.get("API_MAX_COST_YUAN", "3"),
        "API_MAX_REQUESTS": int(values.get("API_MAX_REQUESTS", "100")),
        "MAX_PDF_PAGES": int(values.get("MAX_PDF_PAGES", "35")),
        "KG_GRAPH_PATH": str(data.resolve() / "graphs.json"),
        "SESSION_COOKIE_HTTPONLY": True, "SESSION_COOKIE_SAMESITE": "Lax", "REQUIRE_AUTH": True,
    }


def vision_config(settings):
    return {"provider": "qwen", "api_key": settings["DASHSCOPE_API_KEY"], "model": settings["QWEN_MODEL"],
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "timeout_seconds": 120, "max_tokens": 4096, "temperature": 0.1, "retry_count": 0,
            "enable_thinking": False, "image_detail": "auto", "enabled": True}
