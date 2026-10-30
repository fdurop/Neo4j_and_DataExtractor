"""Atomic course catalogue with stable identity and explicit archive records."""
import json
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
from threading import RLock
import unicodedata
import uuid
from werkzeug.exceptions import BadRequest, Conflict, NotFound


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CourseStore:
    def __init__(self, path):
        self.path, self.lock = Path(path), RLock()

    def list(self):
        with self.lock:
            courses = read_json(self.path, [])
            if not isinstance(courses, list):
                raise ValueError("Course catalogue must be a list")
            return [dict(course) for course in courses if not course.get("archived")]

    def resolve(self, course_id=None, course_name=None):
        if not course_id and not course_name:
            raise BadRequest("请选择课程。")
        for course in self.list():
            if (course_id and course["id"] == course_id) or (not course_id and course["name"] == course_name):
                if course_name and course["name"] != course_name:
                    raise BadRequest("课程编号与课程名称不一致。")
                return course
        raise NotFound("课程不存在。")

    def add(self, name, owner_id=None):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
            raise BadRequest("课程名称长度需要为1至160个字符。")
        name = unicodedata.normalize("NFC", name.strip())
        with self.lock:
            courses = read_json(self.path, [])
            if any(item["name"] == name and not item.get("archived") for item in courses):
                raise Conflict("课程名称已存在。")
            identity = "course_" + uuid.uuid4().hex
            record = {"id": identity, "course_id": identity, "name": name, "created_at": now(), "owner_id": owner_id}
            courses.append(record)
            atomic_json(self.path, courses)
            return dict(record)

    def archive(self, identity):
        with self.lock:
            courses = read_json(self.path, [])
            for item in courses:
                if item["id"] == identity and not item.get("archived"):
                    item.update(archived=True, archived_at=now())
                    atomic_json(self.path, courses)
                    return
        raise NotFound("课程不存在。")
