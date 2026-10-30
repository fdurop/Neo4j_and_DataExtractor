"""Build retrieval text without changing human-inspection extraction data."""

from __future__ import annotations

import re
from typing import Any, Iterable


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s，。；、“”]+|www\.[^\s，。；、“”]+", re.IGNORECASE)
_SOCIAL_HANDLE = re.compile(r"(?<![\w.])@[^\s，。；、“”]+")
_NOISE_CLAUSE = re.compile(
    r"(?:^|[。；;\n])[^。；;\n]{0,100}"
    r"(?:校徽|水印|二维码|电子邮件地址|联系邮箱|联系方式)"
    r"[^。；;\n]{0,120}(?=$|[。；;\n])",
    re.IGNORECASE,
)
_RELEVANCE_BOILERPLATE = re.compile(
    r"(?:该|此)(?:图片|图像|实物图|示意图|照片)?[^。；\n]{0,80}"
    r"(?:与课程主题相关|对应课程涉及|属于课程涉及|课程知识相关)[^。；\n]*[。；]?"
)
_ATTRIBUTION_CLAUSE = re.compile(
    r"(?:^|[。；;\n])[^。；;\n]{0,40}"
    r"(?:左|右)?(?:上|下)?角[^。；;\n]{0,100}"
    r"(?:叠加|标注|印有|显示)[^。；;\n]{0,100}"
    r"(?:网址|字幕|账号|来源|版权)[^。；;\n]{0,80}(?=$|[。；;\n])",
    re.IGNORECASE,
)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        values = [_string(item) for item in value]
        return " ".join(item for item in values if item.strip())
    return str(value)


def clean_embedding_text(text: Any, node_type: str = "", max_chars: int | None = None) -> str:
    value = _string(text).replace("\x00", " ")
    value = _EMAIL.sub(" ", value)
    value = _URL.sub(" ", value)
    value = _SOCIAL_HANDLE.sub(" ", value)

    if node_type.lower() in {"figure", "image"}:
        value = re.sub(r"^\s*\[?COURSE[_ -]?IRRELEVANT\]?\s*", "", value, flags=re.I)
        value = _NOISE_CLAUSE.sub(" ", value)
        value = _RELEVANCE_BOILERPLATE.sub(" ", value)
        value = _ATTRIBUTION_CLAUSE.sub(" ", value)
        value = re.sub(
            r"(?:右|左)?(?:上|下)?角(?:印有|显示|带有)[^。；\n]{0,60}(?:Logo|标志)[。；]?",
            " ",
            value,
            flags=re.I,
        )

    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"\s+([，。；：,.%;:])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" ;；")
    if max_chars and len(value) > max_chars:
        cut = value[:max_chars]
        boundary = max(cut.rfind("。"), cut.rfind("；"), cut.rfind(" "))
        value = cut[:boundary] if boundary >= int(max_chars * 0.65) else cut
    return value.strip()


def compose_embedding_text(
    parts: Iterable[Any], node_type: str = "", max_chars: int | None = None
) -> str:
    unique = []
    seen = set()
    for part in parts:
        cleaned = clean_embedding_text(part, node_type=node_type)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            unique.append(cleaned)
            seen.add(key)
    return clean_embedding_text(" ".join(unique), node_type=node_type, max_chars=max_chars)
