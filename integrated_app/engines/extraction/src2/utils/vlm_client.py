from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .config_loader import load_vlm_config


DEFAULT_QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


class ApiVLMClient:
    """OpenAI-compatible vision-language API client used by extraction."""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.provider = str(config.get("provider") or "qwen").strip().lower()
        self.provider_name = f"{self.provider}_api"
        self.api_key = str(config.get("api_key") or "").strip()
        is_deepseek = self.provider == "deepseek"
        self.model = str(config.get("model") or (
            "deepseek-flash" if is_deepseek else "qwen-vl-plus"
        )).strip()
        self.base_url = str(config.get("base_url") or (
            DEFAULT_DEEPSEEK_URL if is_deepseek else DEFAULT_QWEN_URL
        )).strip().rstrip("/")
        if is_deepseek and not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"
        self.timeout_seconds = int(config.get("timeout_seconds") or 120)
        self.retry_count = max(0, int(config.get("retry_count", 2)))
        self.max_tokens = int(config.get("max_tokens") or 700)
        self.temperature = float(config.get("temperature", 0.1))
        self.image_detail = str(config.get("image_detail") or "auto")
        self.formula_max_tokens = int(config.get("formula_max_tokens") or 2000)
        self.table_max_tokens = int(config.get("table_max_tokens") or 3000)
        self.table_timeout_seconds = int(config.get("table_timeout_seconds") or 150)
        self.table_retry_count = int(config.get("table_retry_count") or 1)
        thinking_value = config.get("enable_thinking", False)
        self.enable_thinking = str(thinking_value).lower() in {"1", "true", "yes", "on"}
        self.last_usage: Dict[str, Any] = {}
        self.last_response_metadata: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    def generate_description(self, image_path: str, prompt: Optional[str] = None) -> str:
        # Default prompt policy: visible_evidence_v2. Custom prompts are caller-owned.
        prompt = prompt or (
            "请对这张教学材料图片做忠实、可检索的中文描述。第一句写“类型：示意图/照片/"
            "表格/公式/代码/其他”中的一种，再记录关键可见文字、组件、连线、坐标轴或步骤关系。"
            "保留对象类别限定词、设备型号、端口号和关键符号，不将不同对象改写成同一泛称；"
            "图中局部示例的名称与说明应分别保留，不因页面主题而替换。"
            "只描述图片中可见且可确认的信息，不判定教学重要性，不推断先修、同义或层级关系；"
            "不要根据常识补充原理、用途或课程结论，不确定内容明确写“不确定”。"
            "不用Markdown，普通图片控制在350个汉字以内，优先保留对象名称与区分信息；"
            "若存在表格、公式或代码，只指出其存在、主题和必要关键标记，完整内容由专用识别流程处理。"
        )
        return self._chat_with_image(
            image_path, prompt, max_tokens=min(self.max_tokens, 500), operation="image_description"
        )

    def recognize_formula(self, image_path: str, context: str = "") -> Dict[str, Any]:
        prompt = (
            "识别图片中实际可见的数学、物理或工程公式，只输出合法JSON，不要输出Markdown代码块。"
            "严格保留下标、上标、分数、根号、积分上下限、矩阵、希腊字母、数字和正负号；不得补写不可见内容。"
            "英文字母大小写必须逐字符忠实于图片，不能因上下文把小写c改成大写C；无法确认大小写时加入uncertain_symbols。"
            "不要把孤立的正负号、数值阈值、尺寸标注、变量标签、公式标题或一条公式中的局部项单独当作完整公式。"
            "无法确认的字符在LaTeX中写为?，并加入uncertain_symbols。JSON格式："
            '{"has_formula":true,"formulas":[{"latex":"","uncertain_symbols":[],"is_complete_formula":true,"confidence":0.0}]}. '
            "没有公式时返回 {\"has_formula\":false,\"formulas\":[]}。"
            "每个公式只允许返回latex、uncertain_symbols、is_complete_formula、confidence四个字段；不要解释公式含义、"
            "不要展开符号列表。同一条跨行公式应合并为一个latex字符串。"
            f"页面上下文仅用于消歧，不得用于补写公式：{(context or '')[:800]}"
        )
        response = self._chat_with_image(
            image_path, prompt, max_tokens=self.formula_max_tokens, operation="formula"
        )
        parsed = self._parse_json_object(response)
        if parsed is not None:
            parsed["raw_response"] = response
            parsed.setdefault("formulas", [])
            return parsed

        recovered = self._recover_formula_objects(response)
        if recovered:
            return {
                "has_formula": True,
                "formulas": recovered,
                "parse_error": "truncated_json",
                "partial_response": True,
                "review_required": True,
                "review_reason": "partial_json_recovered",
                "raw_response": response,
            }

        # Preserve an unparseable response for manual review instead of treating
        # arbitrary prose as valid LaTeX.
        latex_match = re.search(r"LaTeX\s*[:：]\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return {
            "has_formula": bool(latex_match),
            "formulas": [{
                "latex": latex_match.group(1).strip() if latex_match else "",
                "confidence": 0.35,
                "uncertain_symbols": ["json_parse_failed"],
            }],
            "parse_error": "invalid_json",
            "raw_response": response,
        }

    def recognize_code(self, image_path: str) -> Dict[str, Any]:
        prompt = (
            "请判断图片中是否包含程序代码。若没有代码，只回答：无代码。\n"
            "若有代码，请按以下格式回答：\n"
            "语言: [编程语言]\n"
            "```[语言]\n"
            "[完整代码]\n"
            "```\n"
            "功能: [功能说明]"
        )
        response = self._chat_with_image(image_path, prompt, max_tokens=700, operation="code")
        lower = response.lower()
        result = {
            "code": "",
            "language": "txt",
            "description": "",
            "raw_response": response,
            "has_code": False,
        }
        if "无代码" in response or "no code" in lower:
            return result

        import re

        lang_match = re.search(r"语言\s*[:：]\s*(.+)", response)
        if lang_match:
            result["language"] = lang_match.group(1).strip()
        code_match = re.search(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            if code_match.group(1).strip():
                result["language"] = code_match.group(1).strip()
            result["code"] = code_match.group(2).strip()
        desc_match = re.search(r"功能\s*[:：]\s*(.+)", response, re.DOTALL)
        if desc_match:
            result["description"] = desc_match.group(1).strip()
        result["has_code"] = len(result["code"]) >= 10
        return result

    def recognize_table(self, image_path: str, context: str = "") -> Dict[str, Any]:
        prompt = (
            "识别图片中实际可见的全部表格，只输出合法JSON，不要输出Markdown代码块。"
            "先独立清点视觉上分离的表格区域，再逐表转录；一张图片存在两张或更多表格时不得合并或遗漏。"
            "严格保留原始行列、空单元格、数字、小数点、百分号、正负号和单位，不得推测被遮挡内容。"
            "多级表头用二维headers表示，无法确认的单元格填null并在uncertain_cells记录[row,col]。"
            "visible_table_count表示图片中看到的表格区域数；只有所有区域和单元格都已处理时coverage_complete才为true。JSON格式："
            '{"has_table":true,"visible_table_count":2,"coverage_complete":true,"tables":['
            '{"title":"","headers":[[""]],"cells":[[""]],"merged_cells":[],"units":{},'
            '"footnotes":[],"uncertain_cells":[],"confidence":0.0}]}. '
            "没有表格时返回 {\"has_table\":false,\"visible_table_count\":0,"
            "\"coverage_complete\":true,\"tables\":[]}。"
            f"页面上下文仅用于消歧：{(context or '')[:800]}"
        )
        response = self._chat_with_image(
            image_path,
            prompt,
            max_tokens=self.table_max_tokens,
            timeout_seconds=self.table_timeout_seconds,
            retry_count=self.table_retry_count,
            operation="table",
        )
        parsed = self._parse_json_object(response)
        if parsed is not None:
            parsed["raw_response"] = response
            return parsed
        return {
            "has_table": True,
            "headers": [],
            "cells": [],
            "confidence": 0.0,
            "parse_error": "invalid_json",
            "raw_response": response,
        }

    @staticmethod
    def _parse_json_object(response: str) -> Optional[Dict[str, Any]]:
        text = str(response or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _recover_formula_objects(response: str) -> List[Dict[str, Any]]:
        """Recover complete objects from a truncated top-level formulas array."""
        text = str(response or "")
        match = re.search(r'"formulas"\s*:\s*\[', text)
        if not match:
            return []

        recovered = []
        start = None
        depth = 0
        in_string = False
        escaped = False
        for index in range(match.end(), len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        item = json.loads(text[start:index + 1])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        item = None
                    if isinstance(item, dict) and "latex" in item:
                        uncertain = list(item.get("uncertain_symbols") or [])
                        if "partial_json_response" not in uncertain:
                            uncertain.append("partial_json_response")
                        item["uncertain_symbols"] = uncertain
                        item["confidence"] = min(float(item.get("confidence") or 0.5), 0.65)
                        recovered.append(item)
                    start = None
            elif char == "]" and depth == 0:
                break
        return recovered

    def _chat_with_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        retry_count: Optional[int] = None,
        operation: str = "vision",
    ) -> str:
        if not self.available:
            raise RuntimeError("VLM API is not configured. Please fill config/vlm_api.yaml or set env vars.")
        image_url = self._image_to_data_url(image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": self.image_detail},
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "enabled" if self.enable_thinking else "disabled"}
        elif self.provider == "qwen":
            payload["enable_thinking"] = self.enable_thinking
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.last_usage = {}
        self.last_response_metadata = {}
        last_error: Exception | None = None
        request_timeout = timeout_seconds or self.timeout_seconds
        request_retries = self.retry_count if retry_count is None else retry_count
        for attempt in range(request_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=request_timeout,
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                self.last_response_metadata = {
                    "response_id": data.get("id"),
                    "model": data.get("model"),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                }
                usage = data.get("usage") or {}
                self.last_usage = dict(usage)
                details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
                if self.logger:
                    self.logger.info(
                        "VLM API success: operation=%s model=%s image=%s prompt_tokens=%s "
                        "completion_tokens=%s reasoning_tokens=%s total_tokens=%s",
                        operation,
                        self.model,
                        image_path,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        details.get("reasoning_tokens"),
                        usage.get("total_tokens"),
                    )
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("VLM API returned no visible answer content.")
                return content.strip()
            except Exception as exc:
                last_error = exc
                if self.logger:
                    self.logger.warning(
                        "VLM API attempt failed: operation=%s model=%s image=%s "
                        "attempt=%s/%s error=%s",
                        operation,
                        self.model,
                        image_path,
                        attempt + 1,
                        request_retries + 1,
                        exc,
                    )
                if attempt < request_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"VLM API request failed: {last_error}")

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def create_vlm_client(logger=None) -> Optional[ApiVLMClient]:
    config = load_vlm_config()
    enabled = config.get("enabled", True)
    if str(enabled).lower() in {"0", "false", "no", "off"}:
        return None
    client = ApiVLMClient(config, logger=logger)
    if not client.available:
        if logger:
            logger.warning("VLM API is enabled but api_key/model/base_url is incomplete.")
        return None
    return client
