"""Cookie sessions for the loopback integration application."""
from functools import wraps
from flask import Blueprint, abort, current_app, g, jsonify, request, session
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash
from backend.models.db import db
from backend.models.user_model import User

auth_bp = Blueprint("auth", __name__)


def require_login(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            abort(401, description="请先登录。")
        return function(*args, **kwargs)
    return wrapped


def require_teacher(function):
    @wraps(function)
    @require_login
    def wrapped(*args, **kwargs):
        if g.user.role != "teacher":
            abort(403, description="此操作需要教师身份。")
        return function(*args, **kwargs)
    return wrapped


def public_user(user):
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role}


@auth_bp.post("/register")
def register():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="请求必须为JSON对象。")
    username, email, password = data.get("username"), data.get("email"), data.get("password")
    role = data.get("role", "student")
    if not isinstance(username, str) or not 1 <= len(username.strip()) <= 64:
        abort(400, description="用户名长度需要为1至64个字符。")
    if not isinstance(email, str) or "@" not in email or len(email) > 128:
        abort(400, description="请填写有效邮箱。")
    if not isinstance(password, str) or not 8 <= len(password) <= 256:
        abort(400, description="密码长度需要为8至256个字符。")
    if role not in {"student", "teacher"}:
        abort(400, description="角色无效。")
    if role == "teacher" and not current_app.config["ALLOW_TEACHER_REGISTRATION"]:
        abort(403, description="教师注册未开放。")
    user = User(username=username.strip(), email=email.strip(), password_hash=generate_password_hash(password), role=role)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        abort(409, description="用户名或邮箱已注册。")
    return jsonify(success=True, username=user.username, role=user.role, user=public_user(user))


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        abort(400, description="请求必须为JSON对象。")
    username, password = data.get("username"), data.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        abort(400, description="缺少用户名或密码。")
    user = User.query.filter_by(username=username.strip()).first()
    if not user or not check_password_hash(user.password_hash, password):
        abort(401, description="用户名或密码错误。")
    session.clear()
    session["user_id"] = user.id
    session["role"] = user.role
    session["username"] = user.username
    return jsonify(success=True, username=user.username, role=user.role, user=public_user(user))


@auth_bp.post("/logout")
def logout():
    session.clear()
    return jsonify(success=True)


@auth_bp.get("/api/session")
def session_info():
    user = g.get("user")
    return jsonify(success=True, authenticated=user is not None, user=public_user(user) if user else None)
