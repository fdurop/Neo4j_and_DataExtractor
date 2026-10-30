"""Shared path rules for running extraction from the workspace root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


EXTRACTION_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = EXTRACTION_DIR.parent
EXTRACTION_NAME = EXTRACTION_DIR.name


def extraction_path(*parts: str) -> str:
    """Return a workspace-relative path below the extraction directory."""
    return str(Path(EXTRACTION_NAME).joinpath(*parts))


def resolve_workspace_path(value: Any) -> Path:
    """Resolve new workspace-relative paths and legacy extraction-relative paths."""
    path = Path(str(value).replace("/", os.sep))
    if path.is_absolute():
        return path.resolve()

    first = path.parts[0].lower() if path.parts else ""
    if first in {"input", "output", "config", "logs", "models", "src2"}:
        return (EXTRACTION_DIR / path).resolve()
    return (WORKSPACE_DIR / path).resolve()


def workspace_relative(value: Any) -> str:
    """Serialize a path relative to extraction-main using forward slashes."""
    if not value:
        return ""
    try:
        path = resolve_workspace_path(value)
        return os.path.relpath(path, WORKSPACE_DIR).replace(os.sep, "/")
    except Exception:
        return str(value).replace("\\", "/")
