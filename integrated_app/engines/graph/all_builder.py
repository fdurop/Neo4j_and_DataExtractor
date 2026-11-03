import os
import re
import json
import time
import random
import hashlib
import inspect
import unicodedata

from dataclasses import dataclass, field
from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import requests

from neo4j_runtime import GraphDatabase, COURSE_SNAPSHOT_FIELDS, run_idempotent_write
from graph_quality import prepare_knowledge, export_teacher_bundle
import relation_review
from relation_review import partition_relationships
from course_incremental import concept_provenance, course_audit_records

# ============================================================
# 1. 环境变量加载
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_file(
        env_path: Optional[str] = None,
        override: bool = False
) -> None:
    target_path = env_path or os.path.join(
        BASE_DIR,
        "key.env"
    )

    if not os.path.isfile(target_path):
        print(
            f"⚠️ 未找到环境变量文件：{target_path}\n"
            "将尝试直接读取系统环境变量。"
        )
        return

    with open(
            target_path,
            "r",
            encoding="utf-8-sig"
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if (
                    not line
                    or line.startswith("#")
                    or "=" not in line
            ):
                continue

            key, _, value = line.partition("=")

            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key:
                continue

            if override:
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)


_load_env_file(override=True)

# ============================================================
# 2. 默认配置
# ============================================================
DEFAULT_KG_DATA_DIR = os.getenv(
    "KG_DATA_DIR",
    (
        r"C:\Users\Lin\Desktop\extraction\output"
        r"\qwen_qwen3.7-plus_7\kg_data"
    )
)

DEFAULT_DEBUG_DIR = os.getenv(
    "DEBUG_DIR",
    (
        r"C:\Users\Lin\Desktop\extraction\output"
        r"\qwen_qwen3.7-plus_7\debug"
    )
)

DEFAULT_WORK_DIR = os.getenv(
    "TEMP_DIR",
    (
        r"C:\Users\Lin\Desktop"
        r"\学习文件--杂项。\复芏计划项目相关\暂存"
    )
)

DEFAULT_DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    ""
)

DEFAULT_DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"
).rstrip("/")

# 说明：DEEPSEEK_MODEL 由 key.env 直接控制，本次已在 key.env
# 中设置为 flash 版本模型，此处逻辑本身无需改动。
DEFAULT_DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat"
)

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI_2", "")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER_2", "")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_2", "")

DEFAULT_PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    ""
)

DEFAULT_PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "multimodal-rag-index-v2"
)

DEFAULT_PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    ""
)

DEFAULT_PINECONE_DIMENSION = int(
    os.getenv("PINECONE_DIMENSION", "1024")
)

DEFAULT_PINECONE_METRIC = os.getenv(
    "PINECONE_METRIC",
    "cosine"
)

DEFAULT_PINECONE_CLOUD = os.getenv(
    "PINECONE_CLOUD",
    "aws"
)

DEFAULT_PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1"
)

DEFAULT_DATA_VERSION = os.getenv(
    "DATA_VERSION",
    "graph-rag-v2"
)

# ============================================================
# 3. 图谱常量
# ============================================================
LAYER_MATERIAL = 0

CONCEPT_LAYERS = (1, 2, 3)

LAYER_CORE = 1
LAYER_SECONDARY = 2
LAYER_LEAF = 3

# ------------------------------------------------------------
# 【本次改动 A】关系类型精简：16 → 11
#
# 删除的 6 个关系及其吸收去向：
#   HAS_PART      → 由 PART_OF 承担（方向改为 子→父，统一挂靠语义）
#   CONFIGURED_BY → 吸收进 HAS_PARAMETER
#   INPUTS        → 吸收进新增的 FLOWS_TO
#   OUTPUTS       → 吸收进新增的 FLOWS_TO
#   AFFECTS       → 吸收进 RELATED_TO
#   APPLIES_TO    → 吸收进 IMPLEMENTS
#
# 新增的 1 个关系：
#   FLOWS_TO：信号/能量/数据流向，方向约定为 source → target
#             （即"从……流向……"，与 INPUTS/OUTPUTS 的双向表达统一）
# ------------------------------------------------------------
ALLOWED_RELATION_TYPES = {
    "IS_A",
    "PART_OF",
    "HAS_PARAMETER",
    "CONTROLS",
    "DRIVES",
    "CONNECTS_TO",
    "FLOWS_TO",
    "IMPLEMENTS",
    "REQUIRES",
    "PREREQUISITE_OF",
    "RELATED_TO"
}

RELATION_DEFINITIONS = {
    "IS_A": "源概念是目标概念的一种或一个具体实例。",
    "PART_OF": "源概念是目标概念的组成部分（子→父方向）。",
    "HAS_PARAMETER": "源概念具有目标参数或由目标进行配置/设置。",
    "CONTROLS": "源概念控制目标概念（信号层控制）。",
    "DRIVES": "源概念驱动目标概念（功率/执行层驱动）。",
    "CONNECTS_TO": "源概念与目标概念存在明确物理或电气连接。",
    "FLOWS_TO": "源概念的信号、能量或数据流向目标概念（source→target）。",
    "IMPLEMENTS": "源概念实现目标功能、方法、原理，或应用于目标场景。",
    "REQUIRES": "源概念需要目标概念才能运行或成立。",
    "PREREQUISITE_OF": "源概念是理解目标概念的前置知识。",
    "RELATED_TO": "二者相关（包含相互影响关系），但材料不足以支持更明确的关系。"
}

RELATION_BASE_WEIGHTS = {
    "CONTROLS": 1.00,
    "DRIVES": 1.00,
    "HAS_PARAMETER": 0.97,
    "FLOWS_TO": 0.93,
    "IS_A": 0.95,
    "CONNECTS_TO": 0.92,
    "PART_OF": 0.90,
    "IMPLEMENTS": 0.85,
    "REQUIRES": 0.82,
    "PREREQUISITE_OF": 0.78,
    "RELATED_TO": 0.35
}

# ------------------------------------------------------------
# 历史改动：_LABEL_LIKE_PATTERN 正则扩展
# 新增四段匹配：
#   ① 字母+数字（原有）：D5、A1、IN3、PB2
#   ② 数字+字母：0V、5V、3V3
#   ③ 字母+符号结尾：DC+、A-、V+
#   ④ 纯2~4位大写字母缩写：GND、COM、NC、NO、VCC
# 白名单：PID、PWM、USB 等正经概念不受惩罚。
# ------------------------------------------------------------
_LABEL_LIKE_PATTERN = re.compile(
    r"^("
    r"[A-Za-z]{1,4}[-_]?\d{1,4}"       # D5、A1、IN3、PB2
    r"|\d{1,2}[A-Za-z]{1,3}"           # 0V、5V、3V3
    r"|[A-Za-z]{1,4}[+\-]"             # DC+、A-、V+
    r"|[A-Z]{2,4}"                     # GND、COM、NC、NO、VCC
    r")$"
)

# 白名单：命中正则但属于正经概念，不施加惩罚。
_LABEL_PATTERN_WHITELIST: Set[str] = {
    "PID", "PWM", "USB", "SPI", "I2C", "CAN",
    "ADC", "DAC", "DMA", "MCU", "CPU", "GPU",
    "ROM", "RAM", "RTC", "IRQ", "ISR", "API",
    "SDK", "HAL", "BSP", "PCB", "EMC", "EMI",
    "ESD", "FFT", "PLL", "VCO", "LDO", "MOSFET",
}

_CATEGORY_POSITIVE = {"原理", "方法", "设备", "现象"}
_CATEGORY_NEGATIVE = {"引脚", "参数", "信号"}

_PARAMETER_CATEGORY_HINTS = {"参数", "引脚"}

_STEP_ORDER_HINTS = [
    ("第一步", 1), ("第二步", 2), ("第三步", 3),
    ("第四步", 4), ("第五步", 5), ("第六步", 6),
    ("首先", 1), ("然后", 2), ("接着", 3),
    ("其次", 2), ("最后", 99)
]


# ============================================================
# 4. 内部配置对象
# ============================================================
@dataclass
class _PipelineConfig:
    kg_data_dir: str
    debug_dir: str
    work_dir: str

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_dimension: int
    pinecone_metric: str
    pinecone_cloud: str
    pinecone_region: str

    data_version: str

    resume: bool
    force_reextract: bool
    reuse_final_concepts: bool

    write_neo4j: bool
    write_pinecone: bool

    concept_confidence: float
    relation_confidence: float
    max_relations_per_pair: int
    page_node_batch_size: int

    metadata_similarity_threshold: float
    metadata_safe_evidence_count: int
    metadata_safe_page_count: int

    tier_weight_evidence: float
    tier_weight_doc_breadth: float
    tier_weight_page_breadth: float
    tier_label_penalty: float
    tier_category_bonus: float
    tier_category_penalty: float
    tier_threshold_core: float
    tier_threshold_secondary: float

    # 【本次改动 B】新增字段：IDF 泛在惩罚倍率
    # 用于压制"到处出现的引脚编号/背景词"获得虚高 Tier 分数的问题。
    # 惩罚公式：score -= tier_ubiquity_penalty * (page_ratio ** 2)
    # 其中 page_ratio = 该概念出现的页数 / 全课程总页数。
    tier_ubiquity_penalty: float

    parent_overlap_threshold: float

    @property
    def partial_path(self) -> str:
        return os.path.join(
            self.work_dir,
            "concepts_partial_v2.jsonl"
        )

    @property
    def raw_concepts_path(self) -> str:
        return os.path.join(
            self.work_dir,
            "concepts_raw_v2.json"
        )

    @property
    def final_concepts_path(self) -> str:
        return os.path.join(
            self.work_dir,
            "concepts_graph_v2.json"
        )

    @property
    def quality_report_path(self) -> str:
        return os.path.join(
            self.work_dir,
            "graph_quality_report_v2.json"
        )

    @property
    def pipeline_report_path(self) -> str:
        return os.path.join(
            self.work_dir,
            "database_pipeline_report_v2.json"
        )


# ============================================================
# 5. 通用内部工具
# ============================================================
def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str
        )

    return str(value).strip()


def _safe_float(
        value: Any,
        default: float = 0.0
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default

    return max(0.0, min(1.0, number))


def _unique_strings(values: Any) -> List[str]:
    if values is None:
        return []

    if not isinstance(values, list):
        values = [values]

    result = []
    seen = set()

    for value in values:
        text = _normalize_text(value)

        if not text or text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.isfile(path):
        if default is not None:
            return default

        raise FileNotFoundError(f"未找到文件：{path}")

    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def _write_json(path: str, data: Any) -> None:
    output_directory = os.path.dirname(
        os.path.abspath(path)
    )

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str
        )


def _safe_remove(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def _validate_neo4j_property_map(properties: Dict[str, Any]) -> None:
    """Reject values the property store cannot represent; None explicitly removes a key."""
    def scalar_supported(value):
        return (type(value) in (str, bool)
                or type(value) is int and -(2 ** 63) <= value < 2 ** 63
                or type(value) is float and bool(np.isfinite(value)))

    for key, value in properties.items():
        if not isinstance(key, str):
            raise ValueError("Neo4j property names must be strings.")
        if value is None or scalar_supported(value):
            continue
        if (isinstance(value, list) and all(scalar_supported(item) for item in value)
                and len({type(item) for item in value}) <= 1):
            continue
        raise ValueError(f"Unsupported Neo4j property type/value for {key!r}; nested data must be JSON encoded.")


def _neo4j_safe_props(
        properties: Dict[str, Any]
) -> Dict[str, Any]:
    result = {}

    for key, value in (properties or {}).items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            _validate_neo4j_property_map({key: value})
            result[key] = value
            continue

        if isinstance(value, list):
            if all(
                    isinstance(item, (str, int, float, bool))
                    for item in value
            ):
                _validate_neo4j_property_map({key: value})
                result[key] = value

    return result


def _review_status(value: Any) -> str:
    """Missing or unknown legacy states never imply a teacher approval."""
    return value if value in ("approved", "rejected", "pending") else "pending"


def _neo4j_review_metadata(record: Dict[str, Any], *, concept=False) -> Dict[str, Any]:
    """Use JSON for nested observations: Neo4j properties cannot store list[map]."""
    properties = {"review_status": _review_status(record.get("review_status"))}
    if concept:
        observations = record.get("page_observations", [])
        if not isinstance(observations, list) or not all(isinstance(item, dict) for item in observations):
            raise ValueError("Concept page_observations must be a list of objects.")
        properties.update(
            page_observations_json=json.dumps(observations, ensure_ascii=False, sort_keys=True, allow_nan=False),
            suggested_teaching_role=record.get("suggested_teaching_role")
            if record.get("suggested_teaching_role") in ("learning", "detail", "context_dependent") else "pending",
            teaching_role=record.get("teaching_role")
            if record.get("teaching_role") in ("learning", "detail", "context_dependent") else "pending",
            role_review_status=_review_status(record.get("role_review_status")),
        )
    return properties


def _neo4j_concept_id(course_id: str, name: str) -> str:
    """A shared term in two courses is not the same reviewed teaching record."""
    identity = json.dumps([course_id, name], ensure_ascii=False, separators=(",", ":"))
    return "concept_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _neo4j_audit_json(records, label):
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} must be a list of objects")
    return json.dumps(records, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _restore_neo4j_review_metadata(properties: Dict[str, Any], *, concept=False) -> Dict[str, Any]:
    """Decode data only, keep corrupt storage visible, and support old databases."""
    restored = dict(properties)
    restored["review_status"] = _review_status(restored.get("review_status"))
    if concept:
        raw = restored.get("page_observations_json", restored.get("page_observations", []))
        try:
            observations = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(observations, list) or not all(isinstance(item, dict) for item in observations):
                raise ValueError("expected a list of objects")
            # json.loads accepts NaN/Infinity by default; such payloads must not
            # become apparently healthy metadata or break a subsequent export.
            json.dumps(observations, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError, RecursionError):
            observations = []
            restored["metadata_restore_warnings"] = ["invalid_page_observations_json"]
        restored["page_observations"] = observations
        restored["role_review_status"] = _review_status(restored.get("role_review_status"))
        for field in ("teaching_role", "suggested_teaching_role"):
            if restored.get(field) not in ("learning", "detail", "context_dependent"):
                restored[field] = "pending"
    return restored


def _neo4j_review_row(properties: Dict[str, Any], record: Dict[str, Any], *, concept=False) -> Dict[str, Any]:
    """Separate machine content from review decisions for guarded MERGE updates."""
    props = {key: value for key, value in properties.items()
             if key not in {"review_status", "teaching_role", "role_review_status"}}
    metadata = _neo4j_review_metadata(record, concept=concept)
    status = metadata.pop("review_status")
    role_props = {}
    if concept:
        role_props = {key: metadata.pop(key) for key in ("teaching_role", "role_review_status")}
    props.update(metadata)
    _validate_neo4j_property_map(props)
    candidate = dict(props, review_status=status, **role_props)
    return {"content_props": props, "review_status": status, "role_props": role_props,
            "model_candidate_json": json.dumps(candidate, ensure_ascii=False, sort_keys=True, allow_nan=False)}


def _neo4j_review_merge_clause(variable: str, *, concept=False) -> str:
    """A model import cannot replace a stored approved/rejected teacher decision.

    Keep its content/role and leave the incoming snapshot available for review.
    This importer intentionally cannot revise an existing final teacher decision.
    """
    role_lock = (
        f"CASE WHEN {variable}.role_review_status IN ['approved', 'rejected'] THEN true ELSE false END"
        if concept else "false"
    )
    role_update = (
        f", {variable} += CASE WHEN review_locked OR role_locked THEN {{}} ELSE row.role_props END"
        if concept else ""
    )
    role_difference = (
        f"""OR ((review_locked OR role_locked) AND
            CASE WHEN row.role_props.teaching_role <> 'pending' THEN
                coalesce({variable}.teaching_role <> row.role_props.teaching_role, true)
            WHEN row.content_props.suggested_teaching_role <> 'pending' THEN
                coalesce({variable}.teaching_role <> row.content_props.suggested_teaching_role, true)
            ELSE false END)""" if concept else ""
    )
    provenance_update = f", {variable} += row.provenance_props" if concept else ""
    return f"""
    WITH {variable}, row,
         CASE WHEN {variable}.review_status IN ['approved', 'rejected'] THEN true ELSE false END AS review_locked,
         {role_lock} AS role_locked
    WITH {variable}, row, review_locked, role_locked,
         (review_locked AND any(key IN keys(row.content_props)
             WHERE ({variable}[key] IS NULL AND row.content_props[key] IS NOT NULL)
                OR ({variable}[key] IS NOT NULL AND row.content_props[key] IS NULL)
                OR coalesce({variable}[key] <> row.content_props[key], false)))
         {role_difference} AS model_update_pending
    SET {variable} += CASE WHEN review_locked THEN {{}} ELSE row.content_props END,
        {variable}.review_status = CASE WHEN review_locked THEN {variable}.review_status ELSE row.review_status END
        {role_update},
        {variable}.model_candidate_json = row.model_candidate_json,
        {variable}.model_update_pending = model_update_pending,
        {variable}.latest_model_data_version = $data_version,
        {variable}.data_version = CASE WHEN review_locked THEN {variable}.data_version ELSE $data_version END
        {provenance_update}
    """


def _cypher_safe_name(value: Any, fallback: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_]",
        "",
        _normalize_text(value)
    )

    if not cleaned:
        return fallback

    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"

    return cleaned


def _normalize_concept_key(name: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        _normalize_text(name)
    )

    text = text.casefold()

    text = re.sub(
        r"[\s_\-—–·,，。.;；:：/\\()（）\[\]【】]+",
        "",
        text
    )

    return text


def _concept_name_similarity(
        name_a: str,
        name_b: str
) -> float:
    key_a = _normalize_concept_key(name_a)
    key_b = _normalize_concept_key(name_b)

    if not key_a or not key_b:
        return 0.0

    if key_a == key_b:
        return 1.0

    if key_a in key_b or key_b in key_a:
        shorter = min(len(key_a), len(key_b))
        longer = max(len(key_a), len(key_b))
        return 0.75 + 0.20 * shorter / longer

    return SequenceMatcher(None, key_a, key_b).ratio()


def _unordered_pair_key(
        source: str,
        target: str
) -> Tuple[str, str]:
    return tuple(
        sorted([
            _normalize_text(source),
            _normalize_text(target)
        ])
    )


def _calculate_file_hash(paths: List[str]) -> str:
    digest = hashlib.sha256()

    for path in sorted(paths):
        digest.update(
            os.path.basename(path).encode("utf-8")
        )

        if not os.path.isfile(path):
            continue

        with open(path, "rb") as file:
            while True:
                block = file.read(1024 * 1024)

                if not block:
                    break

                digest.update(block)

    return digest.hexdigest()


# ============================================================
# 6. DeepSeek 客户端
# ============================================================
class _DeepSeekClient:
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.model = model
        print(
            "[DeepSeek Client 已初始化] "
            f"实际生效的请求模型为: {self.model}"
        )
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_json(
            self,
            prompt: str,
            max_retries: int = 4,
            timeout: int = 120
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY。")

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.05,
                        "response_format": {
                            "type": "json_object"
                        }
                    },
                    timeout=timeout
                )

                if response.status_code == 429:
                    wait_seconds = min(60, 15 * attempt)
                    print(
                        "⚠️ DeepSeek 触发限流，"
                        f"等待 {wait_seconds} 秒。"
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()

                choice = response.json()["choices"][0]
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise ValueError(f"Incomplete or filtered model response: {choice.get('finish_reason')}")
                content = choice["message"]["content"]

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    json_match = re.search(
                        r"\{.*\}",
                        content,
                        re.DOTALL
                    )

                    if not json_match:
                        raise ValueError(
                            "模型响应中没有合法 JSON。"
                        )

                    parsed = json.loads(json_match.group())

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "模型返回 JSON 顶层不是对象。"
                    )

                return parsed

            except Exception as error:
                last_error = error
                print(
                    "⚠️ DeepSeek 调用失败 "
                    f"({attempt}/{max_retries})：{error}"
                )

                if attempt < max_retries:
                    time.sleep(min(20, 3 * attempt))

        raise RuntimeError(
            f"DeepSeek API 多次调用失败：{last_error}"
        )


# ============================================================
# 7. 正式数据读取与验证
# ============================================================
def _load_formal_data(
        config: _PipelineConfig
) -> Dict[str, Any]:
    manifest_path = os.path.join(
        config.kg_data_dir, "course_manifest.json"
    )
    documents_path = os.path.join(
        config.kg_data_dir, "documents.json"
    )
    pages_path = os.path.join(
        config.kg_data_dir, "pages.json"
    )
    nodes_path = os.path.join(
        config.kg_data_dir, "multimodal_nodes.json"
    )
    relationships_path = os.path.join(
        config.kg_data_dir, "relationships.json"
    )

    formal_data = {
        "manifest": _read_json(manifest_path, {}),
        "documents": _read_json(documents_path),
        "pages": _read_json(pages_path),
        "nodes": _read_json(nodes_path),
        "relationships": _read_json(relationships_path)
    }

    formal_data["source_hash"] = _calculate_file_hash([
        manifest_path,
        documents_path,
        pages_path,
        nodes_path,
        relationships_path
    ])

    return formal_data


def _get_page_id_from_node(node: Dict[str, Any]) -> str:
    return _normalize_text(
        node.get("source_page_id")
        or node.get("page_id")
    )


def _validate_formal_data(
        formal_data: Dict[str, Any]
) -> Dict[str, Any]:
    documents = formal_data["documents"]
    pages = formal_data["pages"]
    nodes = formal_data["nodes"]
    relationships = formal_data["relationships"]

    for name, value in [
        ("documents.json", documents),
        ("pages.json", pages),
        ("multimodal_nodes.json", nodes),
        ("relationships.json", relationships)
    ]:
        if not isinstance(value, list):
            raise ValueError(f"{name} 顶层必须是列表。")

    document_ids = {
        _normalize_text(
            item.get("document_id") or item.get("id")
        )
        for item in documents
        if _normalize_text(
            item.get("document_id") or item.get("id")
        )
    }

    page_ids = {
        _normalize_text(
            item.get("page_id") or item.get("id")
        )
        for item in pages
        if _normalize_text(
            item.get("page_id") or item.get("id")
        )
    }

    node_ids = set()
    duplicate_node_ids = []
    orphan_nodes = []
    node_type_counts = defaultdict(int)

    for node in nodes:
        node_id = _normalize_text(
            node.get("node_id") or node.get("id")
        )

        if not node_id:
            continue

        if node_id in node_ids:
            duplicate_node_ids.append(node_id)

        node_ids.add(node_id)

        node_type = _normalize_text(
            node.get("node_type")
        ) or "Unknown"

        node_type_counts[node_type] += 1

        page_id = _get_page_id_from_node(node)

        if page_id and page_id not in page_ids:
            orphan_nodes.append({
                "node_id": node_id,
                "page_id": page_id
            })

    if duplicate_node_ids:
        raise ValueError(
            "检测到重复素材节点 ID："
            + "、".join(sorted(set(duplicate_node_ids))[:20])
        )

    course_id = _normalize_text(formal_data.get("manifest", {}).get("course_id"))
    all_known_ids = document_ids | page_ids | node_ids | ({course_id} if course_id else set())

    missing_relationship_endpoints = []

    for relationship in relationships:
        source_id = _normalize_text(
            relationship.get("source_id")
        )
        target_id = _normalize_text(
            relationship.get("target_id")
        )

        if source_id and source_id not in all_known_ids:
            missing_relationship_endpoints.append(source_id)

        if target_id and target_id not in all_known_ids:
            missing_relationship_endpoints.append(target_id)

    report = {
        "document_count": len(document_ids),
        "page_count": len(page_ids),
        "material_node_count": len(node_ids),
        "node_type_counts": dict(
            sorted(node_type_counts.items())
        ),
        "orphan_nodes": orphan_nodes,
        "missing_relationship_endpoints": sorted(
            set(missing_relationship_endpoints)
        ),
        "source_hash": formal_data.get("source_hash", "")
    }

    print("\n📋 正式数据校验完成：")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return report


# ============================================================
# 8. 按页面组织知识抽取任务
# ============================================================
def _load_professional_terms(
        config: _PipelineConfig
) -> List[str]:
    path = os.path.join(
        config.debug_dir,
        "professional_terms_library.json"
    )

    if not os.path.isfile(path):
        return []

    data = _read_json(path, [])

    if isinstance(data, dict):
        terms = data.get("terms", [])
    else:
        terms = data

    return _unique_strings(terms)[:100]


def _smart_node_text(
        node: Dict[str, Any],
        limit: int = 2200
) -> str:
    node_type = _normalize_text(node.get("node_type"))

    if node_type == "Table":
        content = (
                node.get("embedding_text")
                or node.get("description")
                or node.get("summary")
                or ""
        )
    elif node_type == "Formula":
        content = (
                node.get("description")
                or node.get("latex")
                or node.get("embedding_text")
                or ""
        )
    elif node_type == "CodeBlock":
        content = (
                node.get("description")
                or node.get("content")
                or node.get("embedding_text")
                or ""
        )
    else:
        content = (
                node.get("description")
                or node.get("raw_text")
                or node.get("embedding_text")
                or node.get("summary")
                or ""
        )

    text = _normalize_text(content)

    if len(text) > limit:
        return text[:limit] + "……"

    return text


def _group_nodes_by_page(
        formal_data: Dict[str, Any],
        page_node_batch_size: int
) -> List[Dict[str, Any]]:
    page_map = {}

    for page in formal_data["pages"]:
        page_id = _normalize_text(
            page.get("page_id") or page.get("id")
        )

        if page_id:
            page_map[page_id] = page

    nodes_by_page = defaultdict(list)

    for node in formal_data["nodes"]:
        page_id = _get_page_id_from_node(node)

        if page_id:
            nodes_by_page[page_id].append(node)

    tasks = []

    for page_id, page in page_map.items():
        page_nodes = nodes_by_page.get(page_id, [])

        if not page_nodes:
            continue

        page_nodes = sorted(
            page_nodes,
            key=lambda item: (
                _normalize_text(item.get("node_type")),
                _normalize_text(
                    item.get("node_id") or item.get("id")
                )
            )
        )

        for start in range(
                0, len(page_nodes), page_node_batch_size
        ):
            batch_index = start // page_node_batch_size

            tasks.append({
                "task_id": f"{page_id}::{batch_index}",
                "page_id": page_id,
                "page": page,
                "batch_index": batch_index,
                "nodes": page_nodes[
                    start:start + page_node_batch_size
                ]
            })

    return tasks


def _load_processed_task_ids(partial_path: str, expected_fingerprints=None) -> Set[str]:
    result = set()

    if not os.path.isfile(partial_path):
        return result

    with open(partial_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = _normalize_text(record.get("task_id"))

            current = expected_fingerprints is None or record.get("task_fingerprint") == expected_fingerprints.get(task_id)
            if task_id and record.get("status", "success") == "success" and current:
                result.add(task_id)
            elif task_id:
                result.discard(task_id)

    return result


def _append_partial_result(
        partial_path: str,
        record: Dict[str, Any]
) -> None:
    with open(partial_path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def _build_page_prompt(
        task: Dict[str, Any],
        professional_terms: List[str]
) -> str:
    page = task["page"]
    page_id = task["page_id"]

    document_id = _normalize_text(page.get("document_id"))

    page_no = (
        page.get("page_no")
        if page.get("page_no") is not None
        else page.get("page_number", "")
    )

    page_summary = _normalize_text(page.get("page_summary") or page.get("summary"))

    page_text = _normalize_text(
        page.get("raw_text") or page.get("description") or page.get("text_content")
    )

    node_ids = []
    material_sections = []

    for node in task["nodes"]:
        node_id = _normalize_text(
            node.get("node_id") or node.get("id")
        )

        node_type = _normalize_text(
            node.get("node_type")
        ) or "Unknown"

        node_ids.append(node_id)

        review_flags = node.get("semantic_review_flags") or []
        if not isinstance(review_flags, list):
            review_flags = ["invalid_semantic_review_flags_requires_review"]
        review_flags = [flag for flag in review_flags if isinstance(flag, str)]
        description_scope = _normalize_text(node.get("description_scope") or node.get("source_kind")) or "unspecified"

        material_sections.append(
            f"[Node ID: {node_id}]\n"
            f"Type: {node_type}\n"
            f"Description scope: {description_scope}\n"
            f"Unreviewed visual claim flags: {json.dumps(review_flags, ensure_ascii=False)}\n"
            f"Content:\n"
            f"{_smart_node_text(node)}"
        )

    relation_definitions = "\n".join(
        f"- {name}: {description}"
        for name, description in RELATION_DEFINITIONS.items()
    )

    return f"""
你是一个负责构建可追溯教学知识图谱的专家。

当前只能分析同一页面中的材料。不得根据常识虚构页面中没有出现的事实。
图像素材描述是未确认的模型观察，不等同于原始课件正文。Description scope 为 entire_page/full_page
时表示整页观察，不能把该页其他位置的内容归给某张局部图。Unreviewed visual claim flags 只表示
需要复核，不是错误定论；带方向或磁极警示的描述，在没有明确正文证据且不能确认标签归属时，
不得用来建立具体 N/S 极性、磁场方向或转向的事实关系。可以保留清晰出现的概念并保持待审。
不能仅凭静态照片推断开关闭合、回路连通或设备正在运行；没有明确图示或文字依据就不建立这类关系。

【页面信息】
page_id：{page_id}
document_id：{document_id}
page_no：{page_no}
页面摘要：{page_summary}
页面正文：{page_text[:3000]}

【专业术语参考】
{"、".join(professional_terms)}

【页面素材】
{chr(10).join(material_sections)}

【允许引用的证据节点 ID】
{json.dumps(node_ids, ensure_ascii=False)}

【概念抽取要求】

每个概念必须包含：

1. canonical_name：完整、正式、可跨页面复用的标准名称；
2. aliases：材料中出现的别名、缩写或中英文名称；
3. description：仅根据本页材料得到的简洁描述；
4. category：设备、部件、信号、参数、方法、原理、软件、现象或其他；
5. evidence_node_ids：直接支持该概念的节点 ID；
6. confidence：0 到 1 之间的置信度。
7. teaching_role：learning（本页学习对象）、detail（局部标号/素材细节）或 pending；
8. learning_objective：本页材料支持的解释、比较、计算或应用目标，细节或无法确定时留空。

没有有效证据的概念不要输出。
不得把相关但不同的对象合并。
不要为了凑数量提取无意义词语。
不得把词性作为教学知识点的判断标准。设备类型可以是学习对象，标号也可能有教学意义。
别名仅作为本页的称呼建议；不同设备类型、泛称与具体类型、整体与部件不得互为同义词。
正文页面主题不能覆盖局部图片中不同的例子。图片描述是已有模型观察，不是课件逐字原文。
同名概念在不同页面可有不同教学角色，请以本页语境判断，不用频次代替教学重要性。

【关系抽取要求】

允许的关系类型：

{relation_definitions}

每条关系必须包含：

1. source；
2. target；
3. relation；
4. description；
5. evidence_node_ids；
6. confidence。

约束：

- source 和 target 必须出现在本次 concepts 中；
- 没有直接证据的关系不要输出；
- 优先选择明确关系，避免滥用 RELATED_TO；
- 不输出语义重复的正反关系；
- FLOWS_TO 表示"从 source 流向 target"，请仔细判断方向；
- 同一对概念最多输出一种关系。
- HAS_PART 是整体到部分，PART_OF 是部分到整体，不输出重复的正反关系。
- IS_A 只能表示种属或实例关系，不能表示组成、配置、类比或数学表达；不能因“A由B组成”就输出 A IS_A B。
- REQUIRES 表示设备运行或命题成立所需的依赖，不表示学习先修；学习先修只能按 PREREQUISITE_OF 的教学依据判断。
- 否定句不得提取成正向依赖；“A不需要B”不能输出 A REQUIRES B。不确定极性时不生成该关系，保留证据供人工审核。
- PREREQUISITE_OF 必须有理解后项依赖前项的教学依据，目录顺序与同页共现均不足以证明。

【输出格式】

{{
  "concepts": [
    {{
      "canonical_name": "标准名称",
      "aliases": ["别名"],
      "description": "概念描述",
      "category": "类别",
      "teaching_role": "learning或detail或pending",
      "learning_objective": "有本页依据的目标或空字符串",
      "evidence_node_ids": ["节点ID"],
      "confidence": 0.9
    }}
  ],
  "relationships": [
    {{
      "source": "源概念",
      "target": "目标概念",
      "relation": "关系类型",
      "description": "关系说明",
      "evidence_node_ids": ["节点ID"],
      "confidence": 0.9
    }}
  ]
}}

只输出合法 JSON。
"""


def _is_non_concept_metadata(
        canonical_name: str,
        evidence_count: int,
        task: Dict[str, Any],
        formal_data: Dict[str, Any],
        config: _PipelineConfig
) -> Tuple[bool, str]:
    manifest = formal_data.get("manifest", {})
    page = task.get("page", {})

    reference_names = _unique_strings([
        manifest.get("course_name"),
        manifest.get("course_id"),
        page.get("summary"),
        page.get("title"),
        page.get("document_name")
    ])

    document_id = _normalize_text(page.get("document_id"))

    if document_id:
        for document in formal_data.get("documents", []):
            did = _normalize_text(
                document.get("document_id")
                or document.get("id")
            )

            if did == document_id:
                reference_names += _unique_strings([
                    document.get("file_name"),
                    document.get("document_name"),
                    document.get("name")
                ])
                break

    best_similarity = 0.0

    for reference_name in reference_names:
        similarity = _concept_name_similarity(
            canonical_name,
            reference_name
        )

        best_similarity = max(best_similarity, similarity)

    if best_similarity < config.metadata_similarity_threshold:
        return False, ""

    if evidence_count >= config.metadata_safe_evidence_count:
        return False, "saved_by_evidence"

    return True, (
        f"similar_to_title(sim={best_similarity:.2f})"
    )


def _relation_quality_score(
        relationship: Dict[str, Any]
) -> float:
    relation_type = _normalize_text(
        relationship.get("relation")
    ).upper()

    base_weight = RELATION_BASE_WEIGHTS.get(
        relation_type, 0.3
    )

    confidence = _safe_float(
        relationship.get("confidence"), 0.5
    )

    evidence_count = len(
        _unique_strings(
            relationship.get("evidence_node_ids")
        )
    )

    evidence_bonus = min(
        0.08, 0.02 * max(0, evidence_count - 1)
    )

    return base_weight * confidence + evidence_bonus


def _limit_relations_per_pair(
        relationships: List[Dict[str, Any]],
        maximum: int
) -> List[Dict[str, Any]]:
    groups = defaultdict(list)

    for relationship in relationships:
        pair_key = _unordered_pair_key(
            relationship.get("source", ""),
            relationship.get("target", "")
        )

        if (
                not pair_key[0]
                or not pair_key[1]
                or pair_key[0] == pair_key[1]
        ):
            continue

        groups[pair_key].append(relationship)

    retained = []

    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda item: (
                _relation_quality_score(item),
                len(item.get("evidence_node_ids", []))
            ),
            reverse=True
        )

        selected = []
        seen_signatures = set()

        for relationship in ranked:
            signature = (
                relationship.get("source"),
                relationship.get("target"),
                relationship.get("relation")
            )

            if signature in seen_signatures:
                continue

            seen_signatures.add(signature)
            selected.append(relationship)

            if len(selected) >= maximum:
                break

        retained.extend(selected)

    return retained


def _validate_page_result(
        task: Dict[str, Any],
        response: Dict[str, Any],
        config: _PipelineConfig,
        formal_data: Dict[str, Any],
        metadata_filter_log: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_node_ids = {
        _normalize_text(
            node.get("node_id") or node.get("id")
        )
        for node in task["nodes"]
    }

    page_id = task["page_id"]

    raw_concepts = response.get("concepts", [])
    raw_relationships = response.get("relationships", [])

    if not isinstance(raw_concepts, list):
        raw_concepts = []

    if not isinstance(raw_relationships, list):
        raw_relationships = []

    concepts = []
    local_names = set()

    for item in raw_concepts:
        if not isinstance(item, dict):
            continue

        name = _normalize_text(
            item.get("canonical_name") or item.get("name")
        )

        if not name:
            continue

        evidence_ids = [
            evidence_id
            for evidence_id in _unique_strings(
                item.get("evidence_node_ids")
            )
            if evidence_id in valid_node_ids
        ]

        if not evidence_ids:
            print(f"  ⚠️ 删除无有效证据概念：{name}")
            continue

        is_metadata, reason = _is_non_concept_metadata(
            name,
            len(evidence_ids),
            task,
            formal_data,
            config
        )
        # Similarity to a chapter title is not evidence that a teaching subject
        # is administrative metadata. Keep model-proposed learning subjects with
        # direct text evidence for review, rather than silently deleting them.
        text_evidence = any(
            _normalize_text(node.get("node_id") or node.get("id")) in evidence_ids
            and node.get("node_type") == "TextChunk" for node in task["nodes"]
        )
        title_review_reason = ""
        if (is_metadata and text_evidence and item.get("teaching_role") == "learning"
                and _normalize_text(item.get("learning_objective"))
                and item.get("category") in {"设备", "部件", "信号", "参数", "方法", "原理", "软件", "现象"}):
            title_review_reason = reason
            metadata_filter_log.append({"canonical_name": name, "page_id": page_id,
                "evidence_count": len(evidence_ids), "reason": reason,
                "stage": "page_prefilter", "action": "retained_for_review"})
            is_metadata = False

        if is_metadata:
            print(f"  🗑️ 过滤疑似元数据概念：{name}（{reason}）")
            metadata_filter_log.append({
                "canonical_name": name,
                "page_id": page_id,
                "evidence_count": len(evidence_ids),
                "reason": reason,
                "stage": "page_prefilter"
            })
            continue

        confidence = _safe_float(item.get("confidence"), 0.5)

        if confidence < config.concept_confidence:
            continue

        aliases = [
            alias
            for alias in _unique_strings(item.get("aliases"))
            if _normalize_concept_key(alias)
               != _normalize_concept_key(name)
        ]

        concepts.append({
            "canonical_name": name,
            "name": name,
            # Model alias suggestions are observations, never global synonyms.
            "aliases": [],
            "description": _normalize_text(
                item.get("description")
            ),
            "category": _normalize_text(
                item.get("category")
            ) or "未分类",
            "evidence_node_ids": evidence_ids,
            "source_page_ids": [page_id],
            "confidence": confidence,
            "teaching_role": "pending",
            "review_status": "pending",
            "page_observations": [{
                "page_id": page_id,
                "observed_name": name,
                "description": _normalize_text(item.get("description")),
                "aliases_suggested_in_context": aliases,
                "suggested_teaching_role": item.get("teaching_role") if item.get("teaching_role") in {"learning", "detail"} else "pending",
                "learning_objective": _normalize_text(item.get("learning_objective")),
                "evidence_node_ids": evidence_ids,
                "figure_only_evidence": all(
                    next((n.get("node_type") for n in task["nodes"] if _normalize_text(n.get("node_id") or n.get("id")) == ref), "") == "Figure"
                    for ref in evidence_ids
                ),
                "metadata_similarity_warning": title_review_reason,
            }],
        })

        local_names.add(name)

    relationships = []

    for item in raw_relationships:
        if not isinstance(item, dict):
            continue

        source = _normalize_text(item.get("source"))
        target = _normalize_text(item.get("target"))

        relation_type = _normalize_text(
            item.get("relation")
        ).upper()

        if not source or not target or source == target:
            continue

        if (
                source not in local_names
                or target not in local_names
        ):
            continue

        if relation_type not in ALLOWED_RELATION_TYPES:
            continue

        evidence_ids = [
            evidence_id
            for evidence_id in _unique_strings(
                item.get("evidence_node_ids")
            )
            if evidence_id in valid_node_ids
        ]

        if not evidence_ids:
            continue

        confidence = _safe_float(item.get("confidence"), 0.5)

        if confidence < config.relation_confidence:
            continue

        relationships.append({
            "source": source,
            "target": target,
            "relation": relation_type,
            "description": _normalize_text(
                item.get("description")
            ),
            "evidence_node_ids": evidence_ids,
            "source_page_ids": [page_id],
            "confidence": confidence,
            "review_status": "pending",
        })

    relationships = _limit_relations_per_pair(
        relationships,
        config.max_relations_per_pair
    )

    return concepts, relationships


def _load_partial_results(
        partial_path: str,
        expected_fingerprints=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    concepts = []
    relationships = []

    if not os.path.isfile(partial_path):
        return concepts, relationships

    latest_records = {}
    with open(partial_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            latest_records[record.get("task_id") or f"legacy_{len(latest_records)}"] = record

    for record in latest_records.values():
        if expected_fingerprints is not None:
            task_id = record.get("task_id")
            if task_id not in expected_fingerprints or record.get("task_fingerprint") != expected_fingerprints[task_id]:
                continue
        if record.get("status", "success") == "success":
            concepts.extend(record.get("concepts", []))
            relationships.extend(
                record.get("relationships", [])
            )

    return concepts, relationships


# ============================================================
# 9. 概念标准化和同义词合并
# ============================================================
def _merge_page_observations(group):
    observations, seen = [], set()
    for item in group:
        for observation in item.get("page_observations", []):
            key = json.dumps(observation, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                observations.append(observation)
    roles = {o.get("suggested_teaching_role", "pending") for o in observations}
    return {
        "page_observations": observations,
        "suggested_teaching_role": next(iter(roles)) if len(roles) == 1 else "context_dependent" if roles else "pending",
        "teaching_role": "pending",
        "review_status": "pending",
    }


def _candidate_aliases(concept):
    return _unique_strings(concept.get("aliases", []) + [
        alias for observation in concept.get("page_observations", [])
        for alias in observation.get("aliases_suggested_in_context", [])
    ])


class _UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item_a: str, item_b: str) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _premerge_exact_concepts(
        concepts: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    groups = defaultdict(list)

    for concept in concepts:
        # Exact premerge must not erase case or signs: A, a, A-, A/B and AB
        # can refer to distinct symbols. Fuzzy normalization is candidate-only.
        key = unicodedata.normalize("NFC", _normalize_text(
            concept.get("canonical_name", "")
        ))

        if key:
            groups[key].append(concept)

    merged = []
    original_mapping = {}

    for group in groups.values():
        representative_item = max(
            group,
            key=lambda item: (
                item.get("confidence", 0),
                len(item.get("canonical_name", ""))
            )
        )

        representative_name = representative_item[
            "canonical_name"
        ]

        all_names = _unique_strings([
            item["canonical_name"] for item in group
        ])

        aliases = _unique_strings(
            [
                alias
                for item in group
                for alias in item.get("aliases", [])
            ]
            + all_names
        )

        aliases = [
            alias
            for alias in aliases
            if alias != representative_name
        ]

        descriptions = _unique_strings([
            item.get("description") for item in group
        ])

        categories = _unique_strings([
            item.get("category") for item in group
        ])

        evidence_ids = _unique_strings([
            evidence_id
            for item in group
            for evidence_id in item.get(
                "evidence_node_ids", []
            )
        ])

        page_ids = _unique_strings([
            page_id
            for item in group
            for page_id in item.get(
                "source_page_ids", []
            )
        ])

        for name in all_names:
            original_mapping[name] = representative_name

        merged.append({
            "canonical_name": representative_name,
            "name": representative_name,
            "aliases": aliases,
            "description": (
                descriptions[0] if descriptions else ""
            ),
            "source_descriptions": descriptions,
            "category": (
                categories[0] if categories else "未分类"
            ),
            "evidence_node_ids": evidence_ids,
            "source_page_ids": page_ids,
            **_merge_page_observations(group),
            "confidence": max(
                item.get("confidence", 0) for item in group
            )
        })

    return merged, original_mapping


def _generate_synonym_candidates(
        concepts: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    candidates = []
    seen = set()

    for index_a in range(len(concepts)):
        concept_a = concepts[index_a]

        names_a = [
            concept_a["canonical_name"]
        ] + _candidate_aliases(concept_a)

        keys_a = {
            _normalize_concept_key(name)
            for name in names_a
            if _normalize_concept_key(name)
        }

        for index_b in range(index_a + 1, len(concepts)):
            concept_b = concepts[index_b]

            names_b = [
                concept_b["canonical_name"]
            ] + _candidate_aliases(concept_b)

            keys_b = {
                _normalize_concept_key(name)
                for name in names_b
                if _normalize_concept_key(name)
            }

            alias_overlap = bool(keys_a & keys_b)

            similarity = max(
                _concept_name_similarity(name_a, name_b)
                for name_a in names_a
                for name_b in names_b
            )

            if not alias_overlap and similarity < 0.72:
                continue

            pair = tuple(sorted([
                concept_a["canonical_name"],
                concept_b["canonical_name"]
            ]))

            if pair in seen:
                continue

            seen.add(pair)
            candidates.append(pair)

    return candidates


def _synonym_reason_conflicts_with_merge(reason: Any) -> bool:
    """Only a clear, unconditional contrary verdict blocks a model MERGE.

    Keep uncertain/conditional/example/negated wording for human review rather
    than guessing its meaning. This never converts the decision to DISTINCT.
    """
    if not isinstance(reason, str):
        return False
    verdict = re.compile(
        r"(?:应为|应该(?:判定为|判断为|判为|归为|是)?|最终(?:判断|判定|结论)(?:为|是|[:：])?|"
        r"判定为|所以是|因此(?:应为|判定为|是)|final\s+(?:decision|verdict)\s*(?:is|:)|therefore\s+(?:is|classify\s+as))"
        r"\s*[\"'“”]*\s*(MERGE|RELATED|DISTINCT)(?![A-Z_])", re.IGNORECASE)
    uncertain = re.compile(r"如果|假如|假设|若|例如|举例|示例|比如|可能|也许|或许|似乎|倾向|不确定|"
                           r"不是|不应|不要|不能|不可|并非|未判|\b(?:if|maybe|perhaps|example|not|never)\b", re.IGNORECASE)
    last_verdict = None
    for sentence in re.findall(r"[^。！？!?；;\n]+(?:[。！？!?；;\n]+|$)", reason):
        if uncertain.search(sentence) or re.search(r"[？?]\s*$", sentence) or re.match(r"\s*(?:不|否)[，,:：]", sentence):
            continue
        matches = list(verdict.finditer(sentence))
        if matches:
            last_verdict = matches[-1].group(1).upper()
    return last_verdict in {"RELATED", "DISTINCT"}


def _adjudicate_synonyms(
        client: _DeepSeekClient,
        concepts: List[Dict[str, Any]],
        candidates: List[Tuple[str, str]],
        review_log: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, str]:
    concept_map = {
        item["canonical_name"]: item for item in concepts
    }

    union_find = _UnionFind(list(concept_map.keys()))

    preferred_names = {}
    accepted_pairs = set()
    rejected_pairs = set()
    batch_size = 30

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]

        payload = []

        for name_a, name_b in batch:
            payload.append({
                "concept_a": {
                    "name": name_a,
                    "aliases": concept_map[name_a].get(
                        "aliases", []
                    ),
                    "description": concept_map[name_a].get(
                        "description", ""
                    ),
                    "category": concept_map[name_a].get(
                        "category", ""
                    ),
                    "page_observations": concept_map[name_a].get("page_observations", [])[:6],
                },
                "concept_b": {
                    "name": name_b,
                    "aliases": concept_map[name_b].get(
                        "aliases", []
                    ),
                    "description": concept_map[name_b].get(
                        "description", ""
                    ),
                    "category": concept_map[name_b].get(
                        "category", ""
                    ),
                    "page_observations": concept_map[name_b].get("page_observations", [])[:6],
                }
            })

        prompt = f"""
请判断以下知识图谱概念是否属于同一个概念。

可选结论：

- MERGE：完全是同一个概念；
- DISTINCT：不是同一个概念；
- RELATED：相关，但不能合并。

判断规则：

1. 缩写与完整名称可以合并。
2. 中英文名称指向同一对象时可以合并。
3. 上位概念和下位概念不能合并。
4. 设备、部件、参数、信号和功能不能相互合并。
5. 不确定时不要合并。
6. preferred_name 必须从 concept_a.name 或 concept_b.name 中选择。
7. MERGE 必须表示可在所有课程上下文中相互替换，不只是本页指向同一实例。
8. 泛称与具体类型、不同工作原理的设备、整体与部件必须 DISTINCT 或 RELATED。
9. 单字符符号、图中占位名称或共享泛化别名不能单独作为合并依据。
10. 为每个结论提供 confidence；不确定时不要 MERGE。

候选：

{json.dumps(payload, ensure_ascii=False, indent=2)}

输出：

{{
  "decisions": [
    {{
      "concept_a": "名称A",
      "concept_b": "名称B",
      "decision": "MERGE或DISTINCT或RELATED",
      "preferred_name": "名称A或名称B",
      "confidence": 0.95,
      "reason": "理由"
    }}
  ]
}}

只输出合法 JSON。
"""

        response = client.chat_json(prompt)

        decisions = response.get("decisions", [])

        if not isinstance(decisions, list):
            decisions = []

        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            name_a = _normalize_text(decision.get("concept_a"))
            name_b = _normalize_text(decision.get("concept_b"))

            action = _normalize_text(
                decision.get("decision")
            ).upper()

            preferred_name = _normalize_text(
                decision.get("preferred_name")
            )

            pair = frozenset({name_a, name_b})
            requested = {frozenset(p) for p in batch}
            # Only decisions for this batch are valid. A model must not inject
            # arbitrary unions or silently add a pair that was never requested.
            if pair not in requested or len(pair) != 2:
                if review_log is not None:
                    review_log.append(dict(decision, applied=False, guard_reason="unrequested_pair"))
                continue
            confidence = _safe_float(decision.get("confidence"), 0.0)
            key_a, key_b = _normalize_concept_key(name_a), _normalize_concept_key(name_b)
            # Real unit validation produced MERGE(U相电流, 电流). Nested names
            # may be a subtype/part/symbol expansion; do not let a confident
            # model turn that ambiguity into a global synonym automatically.
            name_needs_review = (key_a == key_b and name_a != name_b) or (
                key_a != key_b and bool(key_a and key_b) and (key_a in key_b or key_b in key_a))
            reason_conflict = action == "MERGE" and _synonym_reason_conflicts_with_merge(decision.get("reason"))
            accepted = (action == "MERGE" and confidence >= 0.9
                        and preferred_name in pair and bool(_normalize_text(decision.get("reason")))
                        and not name_needs_review and not reason_conflict)
            if accepted:
                accepted_pairs.add(pair)
                preferred_names[pair] = preferred_name
            else:
                rejected_pairs.add(pair)
            if review_log is not None:
                review_log.append(dict(decision, applied=False, pair_accepted=accepted,
                    guard_reason=("reason_decision_conflict_requires_review" if reason_conflict else
                        "nested_or_symbol_variant_requires_review" if action == "MERGE" and name_needs_review else ""),
                    **({"review_status": "pending", "review_required": True} if reason_conflict else {})))

        if start + batch_size < len(candidates):
            time.sleep(random.uniform(1.0, 2.0))

    accepted_pairs -= rejected_pairs
    # Complete-link merging prevents A~B and B~C from forcing A~C. In the old
    # union-find implementation a generic alias could collapse an entire family.
    for pair in sorted(accepted_pairs, key=lambda p: tuple(sorted(p))):
        name_a, name_b = sorted(pair)
        members_a = [n for n in concept_map if union_find.find(n) == union_find.find(name_a)]
        members_b = [n for n in concept_map if union_find.find(n) == union_find.find(name_b)]
        if all(a == b or frozenset({a, b}) in accepted_pairs for a in members_a for b in members_b):
            union_find.union(name_a, name_b)

    if review_log is not None:
        for record in review_log:
            a, b = record.get("concept_a"), record.get("concept_b")
            if a in concept_map and b in concept_map:
                record["applied"] = union_find.find(a) == union_find.find(b)

    groups = defaultdict(list)

    for name in concept_map:
        groups[union_find.find(name)].append(name)

    mapping = {}

    for group_names in groups.values():
        preferred_candidates = []

        for pair, preferred_name in preferred_names.items():
            if pair.issubset(set(group_names)):
                preferred_candidates.append(preferred_name)

        if preferred_candidates:
            representative = max(
                preferred_candidates,
                key=lambda name: (
                    len(name),
                    concept_map[name].get("confidence", 0)
                )
            )
        else:
            representative = max(
                group_names,
                key=lambda name: (
                    len(name),
                    concept_map[name].get("confidence", 0),
                    len(concept_map[name].get(
                        "description", ""
                    ))
                )
            )

        for name in group_names:
            mapping[name] = representative

    return mapping


def _merge_concepts_by_mapping(
        concepts: List[Dict[str, Any]],
        mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    groups = defaultdict(list)

    for concept in concepts:
        old_name = concept["canonical_name"]
        target_name = mapping.get(old_name, old_name)
        groups[target_name].append(concept)

    result = []

    for target_name, group in groups.items():
        all_names = _unique_strings([
            item["canonical_name"] for item in group
        ])

        aliases = _unique_strings(
            [
                alias
                for item in group
                for alias in item.get("aliases", [])
            ]
            + all_names
        )

        aliases = [
            alias for alias in aliases
            if alias != target_name
        ]

        descriptions = _unique_strings([
            description
            for item in group
            for description in (
                item.get("source_descriptions")
                or [item.get("description")]
            )
        ])

        categories = _unique_strings([
            item.get("category") for item in group
        ])

        evidence_ids = _unique_strings([
            evidence_id
            for item in group
            for evidence_id in item.get(
                "evidence_node_ids", []
            )
        ])

        page_ids = _unique_strings([
            page_id
            for item in group
            for page_id in item.get(
                "source_page_ids", []
            )
        ])

        result.append({
            "canonical_name": target_name,
            "name": target_name,
            "aliases": aliases,
            "description": (
                descriptions[0] if descriptions else ""
            ),
            "source_descriptions": descriptions,
            "category": (
                categories[0] if categories else "未分类"
            ),
            "evidence_node_ids": evidence_ids,
            "source_page_ids": page_ids,
            "evidence_count": len(evidence_ids),
            **_merge_page_observations(group),
            "confidence": max(
                item.get("confidence", 0) for item in group
            )
        })

    return result


def _synthesize_descriptions(
        client: _DeepSeekClient,
        concepts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result = []
    batch_size = 20

    for start in range(0, len(concepts), batch_size):
        batch = concepts[start:start + batch_size]

        payload = [
            {
                "canonical_name": item["canonical_name"],
                "aliases": item.get("aliases", []),
                "category": item.get("category", ""),
                "source_descriptions": item.get(
                    "source_descriptions", []
                )[:10]
            }
            for item in batch
        ]

        prompt = f"""
请统一整理以下知识图谱概念。

规则：

1. 不改变 canonical_name。
2. 不添加来源描述中没有依据的具体事实。
3. 合并重复内容，删除竖线拼接。
4. 如果来源描述涉及不同场景，只保留共同、可靠的定义。
5. description 应简洁清晰。
6. category 使用通用、简洁的类别。
7. aliases 只能从输入别名中选择，不得发明。

输入：

{json.dumps(payload, ensure_ascii=False, indent=2)}

输出：

{{
  "concepts": [
    {{
      "canonical_name": "原名称",
      "description": "统一描述",
      "category": "类别",
      "aliases": ["别名"]
    }}
  ]
}}

只输出合法 JSON。
"""

        response = client.chat_json(prompt)

        cleaned_map = {
            _normalize_text(item.get("canonical_name")): item
            for item in response.get("concepts", [])
            if isinstance(item, dict)
        }

        for concept in batch:
            name = concept["canonical_name"]
            cleaned = cleaned_map.get(name, {})

            description = _normalize_text(
                cleaned.get("description")
            )

            category = _normalize_text(
                cleaned.get("category")
            )

            allowed_aliases = set(
                concept.get("aliases", [])
            )

            cleaned_aliases = [
                alias
                for alias in _unique_strings(
                    cleaned.get("aliases")
                )
                if alias in allowed_aliases
                   and alias != name
            ]

            if description:
                concept["description"] = description

            if category:
                concept["category"] = category

            if cleaned_aliases:
                concept["aliases"] = cleaned_aliases

            concept.pop("source_descriptions", None)

            result.append(concept)

        if start + batch_size < len(concepts):
            time.sleep(random.uniform(1.0, 2.0))

    return result


def _merge_duplicate_relationships(
        relationships: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    groups = defaultdict(list)

    for relationship in relationships:
        key = (
            relationship["source"],
            relationship["target"],
            relationship["relation"]
        )
        groups[key].append(relationship)

    result = []

    for (source, target, relation_type), group in (
            groups.items()
    ):
        descriptions = _unique_strings([
            item.get("description") for item in group
        ])

        evidence_ids = _unique_strings([
            evidence_id
            for item in group
            for evidence_id in item.get(
                "evidence_node_ids", []
            )
        ])

        page_ids = _unique_strings([
            page_id
            for item in group
            for page_id in item.get(
                "source_page_ids", []
            )
        ])

        # 保留 auto_generated 标记：
        # 只要 group 里有一条是自动挂靠生成的，合并后仍标记为 True
        is_auto_generated = any(
            item.get("auto_generated", False)
            for item in group
        )
        review_states = {_review_status(item.get("review_status")) for item in group}
        merged_review_status = next(iter(review_states)) if len(review_states) == 1 else "pending"

        result.append({
            "source": source,
            "target": target,
            "relation": relation_type,
            "description": (
                descriptions[0] if descriptions else ""
            ),
            "evidence_node_ids": evidence_ids,
            "source_page_ids": page_ids,
            "evidence_count": len(evidence_ids),
            "confidence": max(
                item.get("confidence", 0) for item in group
            ),
            "base_weight": RELATION_BASE_WEIGHTS.get(
                relation_type, 0.35
            ),
            "auto_generated": is_auto_generated,
            "review_status": merged_review_status,
            "review_required": any(item.get("review_required", False) for item in group)
                               or merged_review_status == "pending",
        })

    return result


def _remap_relationships(
        relationships: List[Dict[str, Any]],
        exact_mapping: Dict[str, str],
        synonym_mapping: Dict[str, str],
        concepts: List[Dict[str, Any]],
        config: _PipelineConfig
) -> List[Dict[str, Any]]:
    valid_names = {
        item["canonical_name"] for item in concepts
    }

    result = []

    for relationship in relationships:
        original_source = relationship.get("source")
        original_target = relationship.get("target")

        exact_source = exact_mapping.get(
            original_source, original_source
        )
        exact_target = exact_mapping.get(
            original_target, original_target
        )

        source = synonym_mapping.get(
            exact_source, exact_source
        )
        target = synonym_mapping.get(
            exact_target, exact_target
        )

        relation_type = _normalize_text(
            relationship.get("relation")
        ).upper()

        if (
                source not in valid_names
                or target not in valid_names
                or source == target
        ):
            continue

        if relation_type not in ALLOWED_RELATION_TYPES:
            continue

        evidence_ids = _unique_strings(
            relationship.get("evidence_node_ids")
        )

        if not evidence_ids:
            continue

        confidence = _safe_float(
            relationship.get("confidence"), 0.5
        )

        if confidence < config.relation_confidence:
            continue

        result.append({
            "source": source,
            "target": target,
            "relation": relation_type,
            "description": _normalize_text(
                relationship.get("description")
            ),
            "evidence_node_ids": evidence_ids,
            "source_page_ids": _unique_strings(
                relationship.get("source_page_ids")
            ),
            "confidence": confidence,
            # LLM 抽取的关系，auto_generated 固定为 False
            "auto_generated": False,
            "review_status": relationship.get("review_status", "pending"),
        })

    result = _merge_duplicate_relationships(result)

    return _limit_relations_per_pair(
        result, config.max_relations_per_pair
    )


# ============================================================
# 9B. 跨页元数据复核 + Tier 打分 + 父子挂靠
# ============================================================
def _recover_cross_page_metadata(
        concepts: List[Dict[str, Any]],
        metadata_filter_log: List[Dict[str, Any]],
        config: _PipelineConfig
) -> int:
    if not metadata_filter_log:
        return 0

    page_count_by_name = defaultdict(set)

    for record in metadata_filter_log:
        name_key = _normalize_concept_key(
            record.get("canonical_name", "")
        )
        page_id = _normalize_text(record.get("page_id"))
        if name_key and page_id:
            page_count_by_name[name_key].add(page_id)

    recovered = 0

    for record in metadata_filter_log:
        name_key = _normalize_concept_key(
            record.get("canonical_name", "")
        )
        cross_page = len(
            page_count_by_name.get(name_key, set())
        )

        if cross_page >= config.metadata_safe_page_count:
            record["cross_page_count"] = cross_page
            record["recover_hint"] = True
            recovered += 1
        else:
            record["cross_page_count"] = cross_page
            record["recover_hint"] = False

    return recovered


def _compute_concept_layer(
        concepts: List[Dict[str, Any]],
        formal_data: Dict[str, Any],
        config: _PipelineConfig
) -> Dict[str, int]:
    page_to_document = {}

    for page in formal_data.get("pages", []):
        page_id = _normalize_text(
            page.get("page_id") or page.get("id")
        )
        document_id = _normalize_text(
            page.get("document_id")
        )
        if page_id and document_id:
            page_to_document[page_id] = document_id

    # 【本次改动 D】计算全课程总页数，作为 IDF 惩罚的分母基准
    total_pages = len({
        _normalize_text(page.get("page_id") or page.get("id"))
        for page in formal_data.get("pages", [])
        if _normalize_text(page.get("page_id") or page.get("id"))
    })

    distribution = {1: 0, 2: 0, 3: 0}
    scores = []

    for concept in concepts:
        evidence_count = len(
            _unique_strings(
                concept.get("evidence_node_ids")
            )
        )

        page_ids = _unique_strings(
            concept.get("source_page_ids")
        )

        document_ids = _unique_strings([
            page_to_document[page_id]
            for page_id in page_ids
            if page_id in page_to_document
        ])

        score = 0.0
        score += config.tier_weight_evidence * evidence_count
        score += (
            config.tier_weight_doc_breadth
            * len(document_ids)
        )
        score += (
            config.tier_weight_page_breadth
            * len(page_ids)
        )

        # 【本次改动 D】IDF 平方惩罚：
        # 概念出现在全课程越大比例的页面中，惩罚呈平方增长。
        # 连续函数，无需额外的硬阈值开关。
        if total_pages > 0:
            page_ratio = len(page_ids) / total_pages
            score -= (
                config.tier_ubiquity_penalty
                * (page_ratio ** 2)
            )

        name = _normalize_text(
            concept.get("canonical_name")
        )

        # 使用扩展后的正则 + 白名单豁免
        if (
            _LABEL_LIKE_PATTERN.match(name)
            and name.upper() not in _LABEL_PATTERN_WHITELIST
        ):
            score -= config.tier_label_penalty

        category = _normalize_text(
            concept.get("category")
        )

        if category in _CATEGORY_POSITIVE:
            score += config.tier_category_bonus
        elif category in _CATEGORY_NEGATIVE:
            score -= config.tier_category_penalty

        if score >= config.tier_threshold_core:
            layer = LAYER_CORE
        elif score >= config.tier_threshold_secondary:
            layer = LAYER_SECONDARY
        else:
            layer = LAYER_LEAF

        concept["layer"] = layer
        # Compatibility field `layer` is an importance tier, not taxonomy depth.
        concept["importance_level"] = layer
        concept["tier_score"] = round(score, 4)
        concept["source_document_ids"] = document_ids

        distribution[layer] += 1
        scores.append(score)

    if scores:
        print(
            "  📊 Tier 打分分布："
            f"L1={distribution[1]} "
            f"L2={distribution[2]} "
            f"L3={distribution[3]}；"
            f"score范围 [{min(scores):.2f}, "
            f"{max(scores):.2f}]"
        )

    return distribution


def _evidence_overlap(
        child_evidence: Set[str],
        parent_evidence: Set[str]
) -> Tuple[float, Set[str]]:
    if not child_evidence:
        return 0.0, set()

    intersection = child_evidence & parent_evidence
    overlap = len(intersection) / len(child_evidence)
    return overlap, intersection


def _link_parents_for_tier(
        child_concepts: List[Dict[str, Any]],
        parent_pool: List[Dict[str, Any]],
        config: _PipelineConfig
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    auto_relations = []
    orphans = []

    parent_evidence_cache = [
        (
            parent,
            set(_unique_strings(
                parent.get("evidence_node_ids")
            ))
        )
        for parent in parent_pool
    ]

    for child in child_concepts:
        child_name = child["canonical_name"]
        child_evidence = set(_unique_strings(
            child.get("evidence_node_ids")
        ))

        best_parent = None
        best_overlap = 0.0
        best_intersection = set()

        for parent, parent_evidence in parent_evidence_cache:
            if parent["canonical_name"] == child_name:
                continue

            overlap, intersection = _evidence_overlap(
                child_evidence, parent_evidence
            )

            if (
                    overlap > best_overlap
                    and overlap >= config.parent_overlap_threshold
            ):
                best_overlap = overlap
                best_parent = parent
                best_intersection = intersection

        if best_parent is None:
            orphans.append(child)
            continue

        child_category = _normalize_text(
            child.get("category")
        )

        # 【本次改动 E】关系类型与方向修正：
        # - 参数类子概念：HAS_PARAMETER，方向为 父→子（不变，语义正确）
        # - 其他子概念：PART_OF（原 HAS_PART 已删除），
        #   方向改为 子→父（source=子, target=父），
        #   与 IS_A / PREREQUISITE_OF 的"子指向父"方向习惯保持一致。
        if child_category in _PARAMETER_CATEGORY_HINTS:
            relation_type = "HAS_PARAMETER"
            relation_source = best_parent["canonical_name"]
            relation_target = child_name
        else:
            relation_type = "PART_OF"
            relation_source = child_name
            relation_target = best_parent["canonical_name"]

        auto_relations.append({
            "source": relation_source,
            "target": relation_target,
            "relation": relation_type,
            "description": (
                f"自动挂靠：{child_name} 从属于 "
                f"{best_parent['canonical_name']}"
                f"（证据包含度 {best_overlap:.2f}）。"
            ),
            "evidence_node_ids": sorted(best_intersection),
            "source_page_ids": _unique_strings(
                child.get("source_page_ids")
            ),
            "confidence": round(best_overlap, 4),
            "auto_generated": True
        })

    return auto_relations, orphans


def _link_concept_parents(
        concepts: List[Dict[str, Any]],
        config: _PipelineConfig
) -> Tuple[List[Dict[str, Any]], int]:
    tier1 = [c for c in concepts if c.get("layer") == 1]
    tier2 = [c for c in concepts if c.get("layer") == 2]
    tier3 = [c for c in concepts if c.get("layer") == 3]

    relations_b1, _ = _link_parents_for_tier(
        tier2, tier1, config
    )

    parent_pool_b2 = tier1 + tier2
    relations_b2, orphans = _link_parents_for_tier(
        tier3, parent_pool_b2, config
    )

    # Absence of an evidence-overlap parent does not make a detail more important.
    promoted = 0
    for orphan in orphans:
        orphan["parent_review_required"] = True

    auto_relations = relations_b1 + relations_b2

    print(
        f"  🔗 自动挂靠：B1={len(relations_b1)} 条，"
        f"B2={len(relations_b2)} 条；"
        f"孤儿提升={promoted} 个"
    )

    return auto_relations, promoted


def _infer_step_order(
        auto_relations: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]]
) -> None:
    description_by_name = {
        c["canonical_name"]: _normalize_text(
            c.get("description")
        )
        for c in concepts
    }

    for relation in auto_relations:
        # 注意：PART_OF 的 target 是父概念，source 是子概念，
        # 而步骤顺序信息应该看子概念（PART_OF 中的 source；
        # HAS_PARAMETER 中的 target）的描述。
        relation_type = relation.get("relation")

        if relation_type == "HAS_PARAMETER":
            child_name = relation.get("target")
        else:
            child_name = relation.get("source")

        text = description_by_name.get(child_name, "")

        for hint, order_value in _STEP_ORDER_HINTS:
            if hint in text:
                relation["order"] = order_value
                break


# ============================================================
# 10. 构建知识层
# ============================================================
def _knowledge_build_fingerprint(config, client):
    # No secrets enter this hash or the reports. Source, model, algorithm and
    # thresholds must all match before a final graph can be reused.
    functions = (_build_page_prompt, _validate_page_result, _is_non_concept_metadata,
        _premerge_exact_concepts, _merge_concepts_by_mapping, _adjudicate_synonyms, _synonym_reason_conflicts_with_merge,
        _compute_concept_layer, _link_concept_parents, _merge_duplicate_relationships,
        _knowledge_build_fingerprint)
    fields = {key: value for key, value in vars(config).items()
        if key.startswith(("tier_", "metadata_")) or key in {
            "concept_confidence", "relation_confidence", "parent_overlap_threshold",
            "max_relations_per_pair", "page_node_batch_size", "data_version"}}
    payload = {"functions": [inspect.getsource(function) for function in functions], "settings": fields,
        "model": getattr(client, "model", config.deepseek_model),
        "endpoint": getattr(client, "url", config.deepseek_base_url),
        "relations": RELATION_DEFINITIONS,
        "relationship_review_policy": inspect.getsource(relation_review)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _quarantine_negated_requirements(data):
    """Apply the shared review rule even when a caller supplies an old graph."""
    usable, review_items = partition_relationships(data.get("relationships", []))
    result = dict(data, relationships=usable)
    result["review_items"] = list(data.get("review_items", [])) + review_items
    return result


def _build_knowledge_layer(
        formal_data: Dict[str, Any],
        config: _PipelineConfig,
        client=None,
        synthesize_descriptions: bool = True,
        synonym_pair_limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    build_fingerprint = hashlib.sha256(json.dumps({
        "policy": _knowledge_build_fingerprint(config, client),
        "synthesize_descriptions": synthesize_descriptions,
        "synonym_pair_limit": synonym_pair_limit,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    if (
            config.reuse_final_concepts
            and not config.force_reextract
            and os.path.isfile(config.final_concepts_path)
    ):
        print(
            "♻️ 检测到已有最终知识层文件，校验来源与版本："
            f"{config.final_concepts_path}"
        )

        final_data = _read_json(config.final_concepts_path)
        if (formal_data.get("source_hash") and final_data.get("source_hash") == formal_data["source_hash"]
                and final_data.get("build_fingerprint") == build_fingerprint):
            # Old automatic parent guesses are still quarantined on read.
            final_data = _quarantine_negated_requirements(final_data)
            final_data, cache_audit = prepare_knowledge(final_data)
            return final_data, {"loaded_from_cache": True, "hierarchy_audit": cache_audit,
                "concept_count": len(final_data.get("concepts", [])),
                "relationship_count": len(final_data.get("relationships", []))}
        previous_hash = hashlib.sha256(json.dumps(final_data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        previous_path = config.final_concepts_path + f".previous_{previous_hash}.json"
        if not os.path.exists(previous_path):
            _write_json(previous_path, final_data)
        print("⚠️ 缓存的来源、模型或算法版本不匹配，保留旧图并重新处理。")

    if client is None and not config.deepseek_api_key:
        raise ValueError(
            "需要重新抽取知识层，但未找到 DEEPSEEK_API_KEY。"
        )

    if config.force_reextract:
        _safe_remove(config.partial_path)
        _safe_remove(config.raw_concepts_path)
        _safe_remove(config.final_concepts_path)
        print("🧹 已清理旧知识抽取缓存，开始重新抽取。")

    if client is None:
        client = _DeepSeekClient(
            config.deepseek_api_key,
            config.deepseek_base_url,
            config.deepseek_model
        )

    terms = _load_professional_terms(config)

    tasks = _group_nodes_by_page(
        formal_data,
        config.page_node_batch_size
    )
    task_prompts = {task["task_id"]: _build_page_prompt(task, terms) for task in tasks}
    task_fingerprints = {
        task_id: hashlib.sha256(json.dumps({"prompt": prompt,
            "build_fingerprint": build_fingerprint,
            "model": getattr(client, "model", config.deepseek_model),
            "endpoint": getattr(client, "url", config.deepseek_base_url)},
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        for task_id, prompt in task_prompts.items()
    }

    processed_task_ids = (
        _load_processed_task_ids(config.partial_path, task_fingerprints)
        if config.resume
        else set()
    )

    if not config.resume:
        _safe_remove(config.partial_path)

    print(
        f"\n🧠 共生成 {len(tasks)} 个页面级知识抽取任务。"
    )

    metadata_filter_log: List[Dict[str, Any]] = []
    failed_tasks = []

    for task_index, task in enumerate(tasks, start=1):
        if task["task_id"] in processed_task_ids:
            print(
                f"⏭️ 跳过已处理任务 "
                f"{task_index}/{len(tasks)}："
                f"{task['task_id']}"
            )
            continue

        print(
            f"⏳ 处理任务 {task_index}/{len(tasks)}："
            f"{task['task_id']}"
        )

        try:
            task_error = None
            response = client.chat_json(
                task_prompts[task["task_id"]]
            )

            concepts, relationships = _validate_page_result(
                task,
                response,
                config,
                formal_data,
                metadata_filter_log
            )

        except Exception as error:
            print(f"❌ 当前页面任务失败：{error}")
            concepts = []
            relationships = []
            task_error = str(error)
            failed_tasks.append({"task_id": task["task_id"], "error": task_error})

        _append_partial_result(
            config.partial_path,
            {
                "task_id": task["task_id"],
                "page_id": task["page_id"],
                "batch_index": task["batch_index"],
                "task_fingerprint": task_fingerprints[task["task_id"]],
                "node_ids": [
                    _normalize_text(
                        node.get("node_id")
                        or node.get("id")
                    )
                    for node in task["nodes"]
                ],
                "concepts": concepts,
                "relationships": relationships,
                "status": "error" if task_error else "success",
                "error": task_error,
            }
        )

        if task_index < len(tasks) and not getattr(client, "cache_only", False):
            time.sleep(random.uniform(1.2, 2.5))

    raw_concepts, raw_relationships = _load_partial_results(
        config.partial_path, task_fingerprints
    )

    _write_json(
        config.raw_concepts_path,
        {
            "concepts": raw_concepts,
            "relationships": raw_relationships
        }
    )

    if failed_tasks:
        _write_json(config.quality_report_path, {"status": "incomplete", "failed_tasks": failed_tasks})
        raise RuntimeError(f"{len(failed_tasks)} page task(s) failed; saved for resume, not a complete graph")

    # Preserve the entire original page relation before remapping or duplicate
    # merging can replace its description with another record's explanation.
    raw_relationships, relationship_review_items = partition_relationships(raw_relationships)

    print("🧹 合并完全同名概念……")

    (
        premerged_concepts,
        exact_mapping
    ) = _premerge_exact_concepts(raw_concepts)

    print("🔎 生成疑似同义词候选……")

    candidates = _generate_synonym_candidates(
        premerged_concepts
    )

    print(f"共发现 {len(candidates)} 组疑似同义概念。")
    all_candidate_count = len(candidates)
    pending_synonym_candidates = []
    remaining_requests = getattr(client, "remaining_requests", None)
    if isinstance(remaining_requests, int):
        affordable_pairs = max(0, remaining_requests) * 30
        synonym_pair_limit = min(synonym_pair_limit, affordable_pairs) if synonym_pair_limit is not None else affordable_pairs
    if synonym_pair_limit is not None and len(candidates) > synonym_pair_limit:
        pending_synonym_candidates = candidates[synonym_pair_limit:]
        candidates = candidates[:synonym_pair_limit]
    _write_json(os.path.join(config.work_dir, "pending_synonym_candidates.json"), pending_synonym_candidates)

    synonym_review = []
    if candidates:
        synonym_mapping = _adjudicate_synonyms(
            client,
            premerged_concepts,
            candidates,
            synonym_review
        )
    else:
        synonym_mapping = {
            item["canonical_name"]: item["canonical_name"]
            for item in premerged_concepts
        }

    merged_concepts = _merge_concepts_by_mapping(
        premerged_concepts,
        synonym_mapping
    )

    _write_json(os.path.join(config.work_dir, "synonym_decisions_v3.json"), synonym_review)

    print("✍️ 统一整理 Concept 描述……")

    final_concepts = _synthesize_descriptions(client, merged_concepts) if synthesize_descriptions else merged_concepts

    print("🔁 跨页元数据复核……")
    recovered_metadata_count = _recover_cross_page_metadata(
        final_concepts,
        metadata_filter_log,
        config
    )

    print("🏷️ 阶段 A：Tier 打分……")
    layer_distribution = _compute_concept_layer(
        final_concepts,
        formal_data,
        config
    )

    print("🔗 阶段 B/C：父子挂靠……")
    auto_relations, orphan_promoted_count = (
        _link_concept_parents(final_concepts, config)
    )

    _infer_step_order(auto_relations, final_concepts)

    layer_distribution = {1: 0, 2: 0, 3: 0}
    for concept in final_concepts:
        layer_distribution[
            concept.get("layer", LAYER_LEAF)
        ] += 1

    final_relationships = _remap_relationships(
        raw_relationships,
        exact_mapping,
        synonym_mapping,
        final_concepts,
        config
    )

    print("🧩 阶段 E：合并自动挂靠边……")
    valid_names = {
        c["canonical_name"] for c in final_concepts
    }

    cleaned_auto_relations = []
    for relation in auto_relations:
        if (
                relation["source"] in valid_names
                and relation["target"] in valid_names
                and relation["source"] != relation["target"]
        ):
            cleaned_auto_relations.append(relation)

    # Candidate parent links must not compete with factual semantic relations
    # during per-pair pruning. Otherwise a high-overlap guess can displace one.
    combined_relationships = final_relationships

    combined_relationships = _merge_duplicate_relationships(
        combined_relationships
    )

    final_relationships = _limit_relations_per_pair(
        combined_relationships,
        config.max_relations_per_pair
    )

    auto_parented_count = len(cleaned_auto_relations)

    final_data = {
        "schema_version": "3.0",
        "semantic_review_status": "pending",
        "data_version": config.data_version,
        "source_hash": formal_data.get("source_hash", ""),
        "build_fingerprint": build_fingerprint,
        "relation_limit_per_pair": (
            config.max_relations_per_pair
        ),
        "relation_base_weights": RELATION_BASE_WEIGHTS,
        "concepts": final_concepts,
        "relationships": final_relationships,
        "review_items": relationship_review_items,
        "hierarchy_candidates": [dict(r, review_status="pending", review_required=True,
            candidate_reason="evidence_overlap_does_not_prove_semantic_hierarchy")
            for r in cleaned_auto_relations]
    }

    final_data = _quarantine_negated_requirements(final_data)
    final_data, hierarchy_audit = prepare_knowledge(final_data)
    final_relationships = final_data["relationships"]

    _write_json(config.final_concepts_path, final_data)

    report = {
        "loaded_from_cache": False,
        "status": "success",
        "description_synthesis": "model" if synthesize_descriptions else "source_descriptions_preserved",
        "hierarchy_audit": hierarchy_audit,
        "page_task_count": len(tasks),
        "raw_concept_count": len(raw_concepts),
        "premerged_concept_count": len(premerged_concepts),
        "synonym_candidate_count": len(candidates),
        "synonym_candidate_total": all_candidate_count,
        "synonym_candidate_pending_count": len(pending_synonym_candidates),
        "concept_count": len(final_concepts),
        "raw_relationship_count": len(raw_relationships),
        "relationship_count": len(final_relationships),
        "layer_distribution": {
            str(k): v for k, v in layer_distribution.items()
        },
        "excluded_metadata_count": len([
            r for r in metadata_filter_log
            if not r.get("recover_hint")
            and r.get("action") != "retained_for_review"
        ]),
        "title_similarity_retained_for_review": sum(r.get("action") == "retained_for_review" for r in metadata_filter_log),
        "metadata_filter_should_have_saved_count": (
            recovered_metadata_count
        ),
        "auto_parented_count": auto_parented_count,
        "orphan_promoted_count": orphan_promoted_count,
        "metadata_filter_log": metadata_filter_log[:200],
        "final_file": config.final_concepts_path
    }

    _write_json(config.quality_report_path, report)

    return final_data, report

# ============================================================
# 11. Neo4j 双层图谱入库
# ============================================================
def _upload_to_neo4j(
    formal_data: Dict[str, Any],
    knowledge_data: Dict[str, Any],
    config: _PipelineConfig
) -> Dict[str, Any]:
    if not config.neo4j_uri:
        raise ValueError("未找到 NEO4J_URI。")

    if not config.neo4j_user:
        raise ValueError("未找到 NEO4J_USER。")

    if not config.neo4j_password:
        raise ValueError("未找到 NEO4J_PASSWORD。")

    course_id = _normalize_text(formal_data["manifest"].get("course_id")) or "default_course"
    _, upload_review = partition_relationships(knowledge_data.get("relationships", []))
    audit_items = list(knowledge_data.get("review_items", [])) + upload_review
    audit_items = list({json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False): item
                       for item in audit_items}.values())
    review_items_json = _neo4j_audit_json(audit_items, "review_items")
    hierarchy_candidates_json = _neo4j_audit_json(knowledge_data.get("hierarchy_candidates", []), "hierarchy_candidates")

    print("\n" + "=" * 60)
    print("🚀 开始写入 Neo4j 双层图谱")
    print("=" * 60)

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password)
    )

    def run_query(query, parameters=None):
        run_idempotent_write(driver, query, parameters)

    def run_unwind(query, rows, label, chunk_size=100):
        if not rows:
            print(f"  - {label}：无数据")
            return

        written = 0

        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]

            run_idempotent_write(driver, query, {
                "batch": chunk, "data_version": config.data_version,
                "course_id": course_id,
            })

            written += len(chunk)

        print(f"  ✓ {label}：{written} 条")

    try:
        driver.verify_connectivity()

        # Old global-name constraints need an explicit migration, never an
        # automatic destructive schema change during a model import.
        with driver.session() as session:
            legacy = list(session.run("""
                SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties
                WHERE 'Concept' IN labelsOrTypes AND properties = ['name']
                RETURN name
            """))
            if legacy:
                raise ValueError("Legacy Concept.name uniqueness requires an explicit course-scoped migration before import")
            input_ids = _unique_strings([record.get("document_id") or record.get("id")
                                        for record in formal_data["documents"]] +
                                       [record.get("page_id") or record.get("id") for record in formal_data["pages"]] +
                                       [record.get("node_id") or record.get("id") for record in formal_data["nodes"]])
            conflicts = list(session.run("""
                MATCH (n) WHERE n.id IN $input_ids AND NOT n:Course AND NOT n:Concept
                  AND (n.course_id IS NULL OR n.course_id <> $course_id)
                RETURN n.id AS id LIMIT 1
            """, {"input_ids": input_ids, "course_id": course_id}))
            if conflicts:
                raise ValueError("Material IDs already belong to another or unscoped course; import stopped before writes")
            previous_course = session.run("""
                MATCH (course:Course {id: $course_id})
                RETURN course.review_items_json AS review_items,
                       course.hierarchy_candidates_json AS hierarchy_candidates
            """, {"course_id": course_id}).single()
            if previous_course:
                audit_items = course_audit_records(previous_course["review_items"], audit_items)
                review_items_json = _neo4j_audit_json(audit_items, "review_items")
                hierarchy_candidates_json = _neo4j_audit_json(course_audit_records(
                    previous_course["hierarchy_candidates"], knowledge_data.get("hierarchy_candidates", []),
                    candidates=True,
                ), "hierarchy_candidates")

        # ==================================================
        # 11.1 创建约束和索引
        # ==================================================
        print("🧱 [1/7] 创建约束和索引……")

        schema_queries = [
            """
            CREATE CONSTRAINT course_id_unique
            IF NOT EXISTS
            FOR (n:Course)
            REQUIRE n.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT document_id_unique
            IF NOT EXISTS
            FOR (n:Document)
            REQUIRE n.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT page_id_unique
            IF NOT EXISTS
            FOR (n:Page)
            REQUIRE n.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT concept_id_unique
            IF NOT EXISTS
            FOR (n:Concept)
            REQUIRE n.id IS UNIQUE
            """,
            """
            CREATE INDEX concept_layer_index
            IF NOT EXISTS
            FOR (n:Concept)
            ON (n.layer)
            """,
            """
            CREATE INDEX concept_category_index
            IF NOT EXISTS
            FOR (n:Concept)
            ON (n.category)
            """,
            """
            CREATE INDEX page_document_index
            IF NOT EXISTS
            FOR (n:Page)
            ON (n.document_id)
            """
        ]

        for query in schema_queries:
            run_query(query)

        material_labels = sorted({
            _cypher_safe_name(
                node.get("node_type"),
                "MaterialNode"
            )
            for node in formal_data["nodes"]
        })

        for material_label in material_labels:
            constraint_name = (
                f"{material_label.lower()}_id_unique"
            )

            constraint_name = _cypher_safe_name(
                constraint_name,
                "material_node_id_unique"
            )

            run_query(
                f"""
                CREATE CONSTRAINT {constraint_name}
                IF NOT EXISTS
                FOR (n:{material_label})
                REQUIRE n.id IS UNIQUE
                """
            )

        # ==================================================
        # 11.2 Course 与 Document
        # ==================================================
        print("📚 [2/7] 写入 Course 和 Document……")

        manifest = formal_data["manifest"]

        course_id = _normalize_text(
            manifest.get("course_id")
        ) or "default_course"

        course_name = _normalize_text(
            manifest.get("course_name")
        ) or course_id

        run_query(
            """
            MERGE (course:Course {id: $course_id})
            SET course.course_name = $course_name,
                course.name = $course_name,
                course.course_id = $course_id,
                course.hierarchy_candidates_json = $hierarchy_candidates_json,
                course.review_items_json = $review_items_json,
                course.layer = $layer,
                course.data_version = $data_version
            """,
            {
                "course_id": course_id,
                "course_name": course_name,
                "hierarchy_candidates_json": hierarchy_candidates_json,
                "review_items_json": review_items_json,
                "layer": LAYER_MATERIAL,
                "data_version": config.data_version
            }
        )

        document_rows = []

        for document in formal_data["documents"]:
            document_id = _normalize_text(
                document.get("document_id")
                or document.get("id")
            )

            if not document_id:
                continue

            document_course_id = (
                _normalize_text(document.get("course_id"))
                or course_id
            )

            properties = _neo4j_safe_props(document)
            properties["document_id"] = document_id
            properties["course_id"] = document_course_id

            document_rows.append({
                "id": document_id,
                "course_id": document_course_id,
                "props": properties
            })

        run_unwind(
            f"""
            UNWIND $batch AS row
            MERGE (document:Document {{id: row.id}})
            SET document += row.props,
                document.layer = {LAYER_MATERIAL},
                document.data_version = $data_version
            """,
            document_rows,
            "Document 节点"
        )

        run_unwind(
            """
            UNWIND $batch AS row
            MATCH (course:Course {id: row.course_id})
            MATCH (document:Document {id: row.id})
            MERGE (course)-[relation:CONTAINS]->(document)
            SET relation.data_version = $data_version,
                relation.structure_source = 'derived'
            """,
            document_rows,
            "Course→Document"
        )

        # ==================================================
        # 11.3 Page 与素材节点
        # ==================================================
        print("📄 [3/7] 写入 Page 和多模态素材节点……")

        page_rows = []
        page_to_document: Dict[str, str] = {}
        page_to_course: Dict[str, str] = {}

        for page in formal_data["pages"]:
            page_id = _normalize_text(
                page.get("page_id") or page.get("id")
            )

            document_id = _normalize_text(
                page.get("document_id")
            )

            page_course_id = (
                _normalize_text(page.get("course_id"))
                or course_id
            )

            if not page_id:
                continue

            properties = _neo4j_safe_props(page)
            properties["page_id"] = page_id
            properties["course_id"] = page_course_id

            if document_id:
                properties["document_id"] = document_id
                page_to_document[page_id] = document_id

            page_to_course[page_id] = page_course_id

            page_rows.append({
                "id": page_id,
                "document_id": document_id,
                "course_id": page_course_id,
                "props": properties
            })

        run_unwind(
            f"""
            UNWIND $batch AS row
            MERGE (page:Page {{id: row.id}})
            SET page += row.props,
                page.layer = {LAYER_MATERIAL},
                page.data_version = $data_version
            """,
            page_rows,
            "Page 节点"
        )

        page_document_rows = [
            row for row in page_rows if row["document_id"]
        ]

        run_unwind(
            """
            UNWIND $batch AS row
            MATCH (document:Document {id: row.document_id})
            MATCH (page:Page {id: row.id})
            MERGE (document)-[relation:CONTAINS]->(page)
            SET relation.data_version = $data_version,
                relation.structure_source = 'derived'
            """,
            page_document_rows,
            "Document→Page"
        )

        material_ids: Set[str] = set()
        material_to_page: Dict[str, str] = {}
        material_nodes_by_type = defaultdict(list)

        for node in formal_data["nodes"]:
            node_id = _normalize_text(
                node.get("node_id") or node.get("id")
            )

            if not node_id:
                continue

            node_type = _cypher_safe_name(
                node.get("node_type"),
                "MaterialNode"
            )

            page_id = _get_page_id_from_node(node)

            document_id = _normalize_text(
                node.get("document_id")
            )

            if not document_id and page_id:
                document_id = page_to_document.get(
                    page_id, ""
                )

            node_course_id = (
                _normalize_text(node.get("course_id"))
                or page_to_course.get(page_id, course_id)
            )

            properties = _neo4j_safe_props(node)
            properties["node_id"] = node_id
            properties["node_type"] = node_type
            properties["course_id"] = node_course_id

            if page_id:
                properties["source_page_id"] = page_id
                material_to_page[node_id] = page_id

            if document_id:
                properties["document_id"] = document_id

            material_nodes_by_type[node_type].append({
                "id": node_id,
                "page_id": page_id,
                "document_id": document_id,
                "course_id": node_course_id,
                "props": properties
            })

            material_ids.add(node_id)

        for node_type, rows in (
            material_nodes_by_type.items()
        ):
            run_unwind(
                f"""
                UNWIND $batch AS row
                MERGE (material:{node_type} {{id: row.id}})
                SET material += row.props,
                    material.layer = {LAYER_MATERIAL},
                    material.data_version = $data_version
                """,
                rows,
                f"{node_type} 节点"
            )

        page_material_rows = [
            {"id": node_id, "page_id": page_id}
            for node_id, page_id
            in material_to_page.items()
        ]

        run_unwind(
            f"""
            UNWIND $batch AS row
            MATCH (page:Page {{id: row.page_id}})
            MATCH (material {{id: row.id}})
            WHERE material.layer = {LAYER_MATERIAL}
              AND NOT material:Page
              AND NOT material:Document
              AND NOT material:Course
            MERGE (page)-[relation:CONTAINS]->(material)
            SET relation.data_version = $data_version,
                relation.structure_source = 'derived'
            """,
            page_material_rows,
            "Page→素材节点"
        )

        # ==================================================
        # 11.4 正式素材层关系
        # ==================================================
        print("🔗 [4/7] 写入正式素材层关系……")

        material_relationships_by_type = defaultdict(list)

        for relationship in formal_data["relationships"]:
            source_id = _normalize_text(
                relationship.get("source_id")
            )
            target_id = _normalize_text(
                relationship.get("target_id")
            )

            relationship_type = _cypher_safe_name(
                relationship.get("relationship_type"),
                "RELATED_TO"
            )

            if not source_id or not target_id:
                continue

            material_relationships_by_type[
                relationship_type
            ].append({
                "source_id": source_id,
                "target_id": target_id,
                "props": _neo4j_safe_props(relationship)
            })

        for relationship_type, rows in (
            material_relationships_by_type.items()
        ):
            run_unwind(
                f"""
                UNWIND $batch AS row
                MATCH (source {{id: row.source_id}})
                MATCH (target {{id: row.target_id}})
                MERGE (source)-[
                    relation:{relationship_type}
                ]->(target)
                SET relation += row.props,
                    relation.data_version = $data_version
                """,
                rows,
                f"素材关系 {relationship_type}"
            )

        # ==================================================
        # 11.5 Concept 知识层
        # ==================================================
        print("🧠 [5/7] 写入 Concept 知识层……")

        raw_concepts = knowledge_data.get("concepts", [])

        if not isinstance(raw_concepts, list):
            raise ValueError(
                "knowledge_data['concepts'] 必须是列表。"
            )

        valid_page_ids = {row["id"] for row in page_rows}

        concept_rows = []
        valid_concept_names = set()
        dropped_concept_count = 0

        for concept in raw_concepts:
            concept_name = _normalize_text(
                concept.get("canonical_name")
                or concept.get("name")
            )

            if not concept_name:
                dropped_concept_count += 1
                continue

            evidence_node_ids = [
                evidence_id
                for evidence_id in _unique_strings(
                    concept.get("evidence_node_ids")
                )
                if evidence_id in material_ids
            ]

            if not evidence_node_ids:
                print(
                    "  ⚠️ 跳过无有效证据 Concept："
                    f"{concept_name}"
                )
                dropped_concept_count += 1
                continue

            derived_page_ids = [
                material_to_page[evidence_id]
                for evidence_id in evidence_node_ids
                if evidence_id in material_to_page
            ]

            source_page_ids = _unique_strings(
                concept.get("source_page_ids")
            )

            source_page_ids = _unique_strings(
                source_page_ids + derived_page_ids
            )

            source_page_ids = [
                page_id
                for page_id in source_page_ids
                if page_id in valid_page_ids
            ]

            source_document_ids = _unique_strings([
                page_to_document[page_id]
                for page_id in source_page_ids
                if page_id in page_to_document
            ])

            aliases = [
                alias
                for alias in _unique_strings(
                    concept.get("aliases")
                )
                if alias != concept_name
            ]

            confidence = _safe_float(
                concept.get("confidence"), 0.5
            )

            concept_layer = concept.get("layer")

            if concept_layer not in CONCEPT_LAYERS:
                concept_layer = LAYER_LEAF

            concept_rows.append({
                "id": _neo4j_concept_id(course_id, concept_name),
                "name": concept_name,
                "canonical_name": concept_name,
                "aliases": aliases,
                "description": _normalize_text(
                    concept.get("description")
                ),
                "category": _normalize_text(
                    concept.get("category")
                ) or "未分类",
                "confidence": confidence,
                "evidence_node_ids": evidence_node_ids,
                "evidence_count": len(evidence_node_ids),
                "source_page_ids": source_page_ids,
                "source_document_ids": source_document_ids,
                "course_id": course_id,
                "layer": concept_layer,
                "importance_level": concept.get("importance_level", concept_layer),
                "tier_score": concept.get("tier_score", 0.0),
                "promoted_from_tier3": bool(
                    concept.get("promoted_from_tier3", False)
                )
            })
            concept_rows[-1].update(_neo4j_review_row(concept_rows[-1], concept, concept=True))

            valid_concept_names.add(concept_name)

        # A PDF is an additive contribution to the course. Read only the
        # matching concepts' provenance, without using it to overwrite reviewed
        # text/roles. The guarded MERGE below still owns those teacher locks.
        with driver.session() as session:
            previous_concepts = {record["id"]: dict(record["properties"])
                                 for record in session.run("""
                MATCH (concept:Concept)
                WHERE concept.course_id = $course_id AND concept.id IN $ids
                RETURN concept.id AS id, properties(concept) AS properties
            """, {"course_id": course_id, "ids": [row["id"] for row in concept_rows]})}
        for row in concept_rows:
            row["provenance_props"] = concept_provenance(previous_concepts.get(row["id"], {}), row["content_props"])
            candidate = json.loads(row["model_candidate_json"])
            candidate.update(row["provenance_props"])
            row["model_candidate_json"] = json.dumps(candidate, ensure_ascii=False, sort_keys=True, allow_nan=False)

        run_unwind(
            f"""
            UNWIND $batch AS row
            MERGE (concept:Concept {{id: row.content_props.id}})
            {_neo4j_review_merge_clause('concept', concept=True)}
            """,
            concept_rows,
            "Concept 节点"
        )

        if dropped_concept_count:
            print(
                "  ⚠️ 共过滤 "
                f"{dropped_concept_count} 个无效 Concept"
            )

        # ==================================================
        # 11.6 HAS_EVIDENCE 跨层关系
        # ==================================================
        print("🌉 [6/7] 建立 HAS_EVIDENCE 证据关系……")

        evidence_rows = []

        for concept_row in concept_rows:
            for evidence_node_id in (
                concept_row["evidence_node_ids"]
            ):
                evidence_rows.append({
                    "concept_name": concept_row["name"],
                    "concept_id": concept_row["id"],
                    "evidence_node_id": evidence_node_id
                })

        # A label-free id lookup can scan the whole shared database per row.
        # Group by material label to use the id uniqueness indexes created above.
        material_labels_by_id = {
            _normalize_text(node.get("node_id") or node.get("id")):
                _cypher_safe_name(node.get("node_type"), "MaterialNode")
            for node in formal_data["nodes"]
        }
        evidence_by_label = defaultdict(list)
        for row in evidence_rows:
            evidence_by_label[material_labels_by_id[row["evidence_node_id"]]].append(row)
        for material_label, rows in evidence_by_label.items():
            run_unwind(
                f"""
                UNWIND $batch AS row
                MATCH (concept:Concept {{id:row.concept_id}})
                MATCH (material:{material_label} {{id:row.evidence_node_id}})
                WHERE concept.course_id=$course_id AND material.course_id=$course_id
                  AND material.layer={LAYER_MATERIAL}
                  AND NOT material:Page AND NOT material:Document AND NOT material:Course
                MERGE (concept)-[evidence:HAS_EVIDENCE]->(material)
                SET evidence.data_version=$data_version
                """, rows, "HAS_EVIDENCE " + material_label)

        # ==================================================
        # 11.7 Concept 语义关系
        # ==================================================
        print("🕸️ [7/7] 写入 Concept 语义关系……")

        raw_knowledge_relationships = (
            knowledge_data.get("relationships", [])
        )

        if not isinstance(
            raw_knowledge_relationships, list
        ):
            raise ValueError(
                "knowledge_data['relationships'] "
                "必须是列表。"
            )

        # Uploading a cached graph directly must not bypass semantic review.
        raw_knowledge_relationships, upload_relationship_review_items = partition_relationships(raw_knowledge_relationships)

        validated_relationships = []
        dropped_relationship_count = 0

        for relationship in raw_knowledge_relationships:
            source = _normalize_text(
                relationship.get("source")
            )
            target = _normalize_text(
                relationship.get("target")
            )

            relationship_type = _normalize_text(
                relationship.get("relation")
            ).upper()

            if (
                not source
                or not target
                or source == target
            ):
                dropped_relationship_count += 1
                continue

            if (
                source not in valid_concept_names
                or target not in valid_concept_names
            ):
                dropped_relationship_count += 1
                continue

            if (
                relationship_type
                not in ALLOWED_RELATION_TYPES
            ):
                dropped_relationship_count += 1
                continue

            evidence_node_ids = [
                evidence_id
                for evidence_id in _unique_strings(
                    relationship.get("evidence_node_ids")
                )
                if evidence_id in material_ids
            ]

            if not evidence_node_ids:
                dropped_relationship_count += 1
                continue

            source_page_ids = _unique_strings(
                relationship.get("source_page_ids")
            )

            source_page_ids = _unique_strings(
                source_page_ids
                + [
                    material_to_page[evidence_id]
                    for evidence_id in evidence_node_ids
                    if evidence_id in material_to_page
                ]
            )

            confidence = _safe_float(
                relationship.get("confidence"), 0.5
            )

            base_weight = RELATION_BASE_WEIGHTS.get(
                relationship_type, 0.35
            )

            is_auto_generated = bool(
                relationship.get("auto_generated", False)
            )

            validated_relationships.append({
                "source": source,
                "target": target,
                "relation": relationship_type,
                "description": _normalize_text(
                    relationship.get("description")
                ),
                "evidence_node_ids": evidence_node_ids,
                "source_page_ids": source_page_ids,
                "evidence_count": len(evidence_node_ids),
                "confidence": confidence,
                "base_weight": base_weight,
                "auto_generated": is_auto_generated,
                "review_status": _review_status(relationship.get("review_status")),
                "review_required": bool(relationship.get("review_required", False)),
            })

        validated_relationships = (
            _merge_duplicate_relationships(
                validated_relationships
            )
        )

        validated_relationships = (
            _limit_relations_per_pair(
                validated_relationships,
                config.max_relations_per_pair
            )
        )

        relationships_by_type = defaultdict(list)

        for relationship in validated_relationships:
            relationship["source_id"] = _neo4j_concept_id(course_id, relationship["source"])
            relationship["target_id"] = _neo4j_concept_id(course_id, relationship["target"])
            relationship.update(_neo4j_review_row(
                {key: value for key, value in relationship.items()
                 if key not in {"source", "target", "relation", "review_status"}}, relationship))
            relationships_by_type[
                relationship["relation"]
            ].append(relationship)

        for relationship_type, rows in (
            relationships_by_type.items()
        ):
            safe_relationship_type = _cypher_safe_name(
                relationship_type,
                "RELATED_TO"
            )

            run_unwind(
                f"""
                UNWIND $batch AS row
                MATCH (source:Concept {{
                    id: row.source_id
                }})
                MATCH (target:Concept {{
                    id: row.target_id
                }})
                MERGE (source)-[
                    relation:{safe_relationship_type}
                ]->(target)
                {_neo4j_review_merge_clause('relation')}
                """,
                rows,
                f"Concept 关系 {relationship_type}"
            )

        if dropped_relationship_count:
            print(
                "  ⚠️ 共过滤 "
                f"{dropped_relationship_count} 条无效知识关系"
            )

        # ==================================================
        # 11.8 Neo4j 入库统计
        # ==================================================
        with driver.session() as session:
            total_nodes = session.run(
                "MATCH (node) RETURN count(node) AS count"
            ).single()["count"]

            total_relationships = session.run(
                """
                MATCH ()-[relationship]->()
                RETURN count(relationship) AS count
                """
            ).single()["count"]

            concept_count = session.run(
                """
                MATCH (concept:Concept)
                RETURN count(concept) AS count
                """
            ).single()["count"]

            material_count = session.run(
                f"""
                MATCH (node)
                WHERE node.layer = {LAYER_MATERIAL}
                RETURN count(node) AS count
                """
            ).single()["count"]

            evidence_count = session.run(
                """
                MATCH ()-[evidence:HAS_EVIDENCE]->()
                RETURN count(evidence) AS count
                """
            ).single()["count"]

            knowledge_relationship_count = session.run(
                """
                MATCH (:Concept)-[relationship]->(:Concept)
                WHERE type(relationship) <> 'HAS_EVIDENCE'
                RETURN count(relationship) AS count
                """
            ).single()["count"]

            auto_generated_count = session.run(
                """
                MATCH (:Concept)-[r]->(:Concept)
                WHERE r.auto_generated = true
                RETURN count(r) AS count
                """
            ).single()["count"]

        report = {
            "status": "success",
            "total_node_count": total_nodes,
            "material_node_count": material_count,
            "concept_count": concept_count,
            "total_relationship_count": (
                total_relationships
            ),
            "knowledge_relationship_count": (
                knowledge_relationship_count
            ),
            "evidence_relationship_count": evidence_count,
            "auto_generated_relationship_count": (
                auto_generated_count
            ),
            "written_concept_count": len(concept_rows),
            "written_knowledge_relationship_count": len(
                validated_relationships
            ),
            "quarantined_relationship_count": len(audit_items),
            "relationship_review_items": audit_items,
            "data_version": config.data_version
        }

        print("\n📊 Neo4j 入库报告：")
        print(
            json.dumps(report, ensure_ascii=False, indent=2)
        )

        return report

    finally:
        driver.close()


# ============================================================
# 12. Pinecone 向量文件读取与验证
# ============================================================
def _load_vector_files(
    config: _PipelineConfig
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    matrix_path = os.path.join(
        config.kg_data_dir, "vectors", "vector_matrix.npy"
    )
    index_path = os.path.join(
        config.kg_data_dir, "vectors", "vector_index.json"
    )

    if not os.path.isfile(matrix_path):
        raise FileNotFoundError(
            f"未找到向量矩阵：{matrix_path}"
        )

    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"未找到向量索引：{index_path}"
        )

    matrix = np.load(matrix_path, allow_pickle=False)

    index_data = _read_json(index_path)

    if isinstance(index_data, dict):
        items = (
            index_data.get("items")
            or index_data.get("vectors")
            or []
        )
        top_metadata = {
            key: value
            for key, value in index_data.items()
            if key not in {"items", "vectors"}
        }
    elif isinstance(index_data, list):
        items = index_data
        top_metadata = {}
    else:
        raise ValueError(
            "vector_index.json 顶层必须是字典或列表。"
        )

    if not isinstance(items, list):
        raise ValueError(
            "vector_index.json 的向量条目必须是列表。"
        )

    if matrix.ndim != 2:
        raise ValueError(
            "vector_matrix.npy 必须是二维矩阵。"
        )

    if matrix.shape[0] != len(items):
        raise ValueError(
            "向量矩阵行数与索引条目数不一致："
            f"{matrix.shape[0]} != {len(items)}"
        )

    if matrix.shape[1] != config.pinecone_dimension:
        raise ValueError(
            "向量维度与 Pinecone 配置不一致："
            f"{matrix.shape[1]} != "
            f"{config.pinecone_dimension}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("向量矩阵存在 NaN 或无穷值。")

    seen_embedding_ids = set()
    duplicate_ids = []

    for row_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"向量索引第 {row_index} 条不是字典。"
            )

        embedding_id = _normalize_text(
            item.get("embedding_id") or item.get("id")
        )

        if not embedding_id:
            raise ValueError(
                f"向量索引第 {row_index} 条缺少 "
                "embedding_id。"
            )

        if embedding_id in seen_embedding_ids:
            duplicate_ids.append(embedding_id)

        seen_embedding_ids.add(embedding_id)

    if duplicate_ids:
        raise ValueError(
            "检测到重复 embedding_id："
            + "、".join(sorted(set(duplicate_ids))[:20])
        )

    print("\n📐 向量数据校验通过：")
    print(f"- 向量数量：{matrix.shape[0]}")
    print(f"- 向量维度：{matrix.shape[1]}")

    return matrix, items, top_metadata


def _build_vector_source_lookup(
    formal_data: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    manifest = formal_data["manifest"]

    default_course_id = (
        _normalize_text(manifest.get("course_id"))
        or "default_course"
    )

    document_map = {}

    for document in formal_data["documents"]:
        document_id = _normalize_text(
            document.get("document_id")
            or document.get("id")
        )

        if document_id:
            document_map[document_id] = document

    page_map = {}

    for page in formal_data["pages"]:
        page_id = _normalize_text(
            page.get("page_id") or page.get("id")
        )

        if page_id:
            page_map[page_id] = page

    lookup = {
        default_course_id: {
            "node_id": default_course_id,
            "node_type": "Course",
            "course_id": default_course_id,
            "document_id": "",
            "page_id": "",
            "page_no": -1
        }
    }

    for document_id, document in document_map.items():
        lookup[document_id] = {
            "node_id": document_id,
            "node_type": "Document",
            "course_id": (
                _normalize_text(document.get("course_id"))
                or default_course_id
            ),
            "document_id": document_id,
            "page_id": "",
            "page_no": -1
        }

    for page_id, page in page_map.items():
        document_id = _normalize_text(
            page.get("document_id")
        )

        page_no = (
            page.get("page_no")
            if page.get("page_no") is not None
            else page.get("page_number", -1)
        )

        lookup[page_id] = {
            "node_id": page_id,
            "node_type": "Page",
            "course_id": (
                _normalize_text(page.get("course_id"))
                or default_course_id
            ),
            "document_id": document_id,
            "page_id": page_id,
            "page_no": page_no
        }

    for node in formal_data["nodes"]:
        node_id = _normalize_text(
            node.get("node_id") or node.get("id")
        )

        if not node_id:
            continue

        page_id = _get_page_id_from_node(node)
        page = page_map.get(page_id, {})

        document_id = _normalize_text(
            node.get("document_id")
            or page.get("document_id")
        )

        page_no = (
            page.get("page_no")
            if page.get("page_no") is not None
            else page.get("page_number", -1)
        )

        lookup[node_id] = {
            "node_id": node_id,
            "node_type": (
                _normalize_text(node.get("node_type"))
                or "Unknown"
            ),
            "course_id": (
                _normalize_text(node.get("course_id"))
                or default_course_id
            ),
            "document_id": document_id,
            "page_id": page_id,
            "page_no": page_no
        }

    return lookup


def _build_pinecone_records(
    matrix: np.ndarray,
    vector_items: List[Dict[str, Any]],
    formal_data: Dict[str, Any],
    config: _PipelineConfig
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    lookup = _build_vector_source_lookup(formal_data)

    records = []
    skipped = []
    node_type_counts = defaultdict(int)
    records_with_page = 0

    for row_index, item in enumerate(vector_items):
        status = _normalize_text(
            item.get("status")
        ).lower()

        if status in {
            "failed", "error", "invalid", "skipped"
        }:
            skipped.append({
                "row_index": row_index,
                "reason": f"status={status}"
            })
            continue

        embedding_id = _normalize_text(
            item.get("embedding_id") or item.get("id")
        )

        node_id = _normalize_text(item.get("node_id"))

        if not embedding_id or not node_id:
            skipped.append({
                "row_index": row_index,
                "reason": "缺少 embedding_id 或 node_id"
            })
            continue

        source = lookup.get(node_id)

        if not source:
            skipped.append({
                "row_index": row_index,
                "node_id": node_id,
                "reason": "正式数据中不存在该节点"
            })
            continue

        node_type = (
            source.get("node_type")
            or _normalize_text(item.get("node_type"))
            or "Unknown"
        )

        page_id = _normalize_text(source.get("page_id"))
        document_id = _normalize_text(
            source.get("document_id")
        )

        course_id = (
            _normalize_text(source.get("course_id"))
            or "default_course"
        )

        page_no = source.get("page_no", -1)

        if page_no is None:
            page_no = -1

        try:
            page_no = int(page_no)
        except (TypeError, ValueError):
            page_no = -1

        embedding_text = _normalize_text(
            item.get("embedding_text")
            or item.get("text")
        )

        metadata = {
            "embedding_id": embedding_id,
            "node_id": node_id,
            "node_type": node_type,
            "content_kind": node_type.lower(),
            "page_id": page_id,
            "source_page_id": page_id,
            "document_id": document_id,
            "course_id": course_id,
            "page_no": page_no,
            "text": embedding_text[:1800],
            "data_version": config.data_version
        }

        records.append({
            "id": embedding_id,
            "values": (
                matrix[row_index]
                .astype(np.float32)
                .tolist()
            ),
            "metadata": metadata
        })

        node_type_counts[node_type] += 1

        if page_id:
            records_with_page += 1

    report = {
        "matrix_vector_count": int(matrix.shape[0]),
        "prepared_vector_count": len(records),
        "skipped_vector_count": len(skipped),
        "node_type_counts": dict(
            sorted(node_type_counts.items())
        ),
        "records_with_page_id": records_with_page,
        "records_without_page_id": (
            len(records) - records_with_page
        ),
        "skipped_preview": skipped[:30]
    }

    return records, report


# ============================================================
# 13. Pinecone 索引创建与入库
# ============================================================
def _get_pinecone_description_value(
    description: Any,
    field: str,
    default: Any = None
) -> Any:
    if isinstance(description, dict):
        return description.get(field, default)

    return getattr(description, field, default)


def _ensure_pinecone_index(
    client: "Pinecone",
    config: _PipelineConfig,
    create_if_missing: bool
) -> None:
    from pinecone import ServerlessSpec

    existing_indexes = client.list_indexes().names()

    if config.pinecone_index_name not in existing_indexes:
        if not create_if_missing:
            raise ValueError(
                "Pinecone 索引不存在，且"
                "create_pinecone_index=False："
                f"{config.pinecone_index_name}"
            )

        print(
            "⏳ 创建 Pinecone 索引："
            f"{config.pinecone_index_name}"
        )

        client.create_index(
            name=config.pinecone_index_name,
            dimension=config.pinecone_dimension,
            metric=config.pinecone_metric,
            spec=ServerlessSpec(
                cloud=config.pinecone_cloud,
                region=config.pinecone_region
            )
        )

        while True:
            description = client.describe_index(
                config.pinecone_index_name
            )

            status = _get_pinecone_description_value(
                description, "status", {}
            )

            if isinstance(status, dict):
                ready = status.get("ready", False)
            else:
                ready = getattr(status, "ready", False)

            if ready:
                break

            time.sleep(1)

        print("✅ Pinecone 索引创建完成。")
        return

    description = client.describe_index(
        config.pinecone_index_name
    )

    existing_dimension = _get_pinecone_description_value(
        description, "dimension"
    )

    existing_metric = _normalize_text(
        _get_pinecone_description_value(
            description, "metric", ""
        )
    ).lower()

    if (
        existing_dimension is not None
        and int(existing_dimension)
        != config.pinecone_dimension
    ):
        raise ValueError(
            "现有 Pinecone 索引维度不兼容："
            f"{existing_dimension} != "
            f"{config.pinecone_dimension}。\n"
            "请使用新的索引名称。"
        )

    if (
        existing_metric
        and existing_metric
        != config.pinecone_metric.lower()
    ):
        raise ValueError(
            "现有 Pinecone 索引 Metric 不兼容："
            f"{existing_metric} != "
            f"{config.pinecone_metric}。\n"
            "请使用新的索引名称。"
        )

    print(
        "✅ Pinecone 索引配置检查通过："
        f"{config.pinecone_index_name}"
    )


def _upload_to_pinecone(
    formal_data: Dict[str, Any],
    config: _PipelineConfig,
    create_if_missing: bool
) -> Dict[str, Any]:
    from pinecone import Pinecone

    if not config.pinecone_api_key:
        raise ValueError("未找到 PINECONE_API_KEY。")

    print("\n" + "=" * 60)
    print("🚀 开始写入 Pinecone 向量数据库")
    print("=" * 60)

    (
        matrix,
        vector_items,
        vector_file_metadata
    ) = _load_vector_files(config)

    records, preparation_report = _build_pinecone_records(
        matrix,
        vector_items,
        formal_data,
        config
    )

    if not records:
        raise ValueError("没有可上传的有效向量。")

    print("\n📊 向量准备报告：")
    print(
        json.dumps(
            preparation_report,
            ensure_ascii=False,
            indent=2
        )
    )

    client = Pinecone(api_key=config.pinecone_api_key)

    _ensure_pinecone_index(
        client, config, create_if_missing
    )

    index = client.Index(config.pinecone_index_name)

    batch_size = 100
    uploaded = 0

    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]

        if config.pinecone_namespace:
            index.upsert(
                vectors=chunk,
                namespace=config.pinecone_namespace
            )
        else:
            index.upsert(vectors=chunk)

        uploaded += len(chunk)

        print(
            "  ✓ 已上传 "
            f"{uploaded}/{len(records)} 条向量"
        )

    time.sleep(2)

    statistics = index.describe_index_stats()

    report = {
        "status": "success",
        "index_name": config.pinecone_index_name,
        "namespace": config.pinecone_namespace,
        "dimension": config.pinecone_dimension,
        "metric": config.pinecone_metric,
        "uploaded_vector_count": uploaded,
        "data_version": config.data_version,
        "vector_file_metadata": vector_file_metadata,
        "preparation": preparation_report,
        "index_statistics": str(statistics)
    }

    print("\n🎉 Pinecone 向量入库完成。")

    return report


# ============================================================
# 14. Neo4j 图谱导出内部工具
# ============================================================
def _frontend_node_type(
    labels: List[str],
    properties: Dict[str, Any]
) -> str:
    priority = [
        "Course", "Document", "Page", "Concept",
        "Figure", "Table", "Formula", "CodeBlock",
        "TextChunk", "MaterialNode"
    ]

    node_type = next(
        (label for label in priority if label in labels),
        properties.get("node_type")
        or (labels[0] if labels else "Unknown")
    )

    node_type = re.sub(
        r"(.)([A-Z][a-z]+)", r"\1_\2", str(node_type)
    )

    node_type = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", node_type
    )

    return node_type.lower()


def _frontend_symbol_size(labels: List[str]) -> int:
    sizes = {
        "Course": 80, "Document": 62, "Page": 52,
        "Concept": 42, "Figure": 35, "Table": 35,
        "Formula": 35, "CodeBlock": 35,
        "TextChunk": 26, "MaterialNode": 28
    }

    for label, size in sizes.items():
        if label in labels:
            return size

    return 30


def _frontend_node_name(
    node: Dict[str, Any],
    document_map: Dict[str, Dict[str, Any]]
) -> str:
    properties = node["properties"]
    labels = node["labels"]

    node_id = _normalize_text(properties.get("id"))

    if "Course" in labels:
        return _normalize_text(
            properties.get("course_name")
            or properties.get("name")
            or node_id
        )

    if "Document" in labels:
        return _normalize_text(
            properties.get("file_name")
            or properties.get("document_name")
            or properties.get("name")
            or node_id
        )

    if "Page" in labels:
        document_id = _normalize_text(
            properties.get("document_id")
        )

        document = document_map.get(document_id, {})

        document_name = _normalize_text(
            document.get("file_name")
            or document.get("document_name")
            or document_id
        )

        page_no = (
            properties.get("page_no")
            if properties.get("page_no") is not None
            else properties.get("page_number")
        )

        if document_name and page_no is not None:
            return f"{document_name} - 第{page_no}页"

    if "Concept" in labels:
        return _normalize_text(
            properties.get("name") or node_id
        )

    name = _normalize_text(
        properties.get("name")
        or properties.get("summary")
        or properties.get("description")
        or properties.get("raw_text")
        or node_id
    )

    return name[:100]


def _find_center_node(
    nodes: Dict[str, Dict[str, Any]],
    document_map: Dict[str, Dict[str, Any]],
    center_node_id: Optional[str],
    center_node_name: Optional[str]
) -> Optional[str]:
    if center_node_id:
        normalized_id = _normalize_text(center_node_id)

        matches = [
            element_id
            for element_id, node in nodes.items()
            if _normalize_text(
                node["properties"].get("id")
            ) == normalized_id
        ]

        if not matches:
            raise ValueError(
                f"未找到 ID 为「{center_node_id}」的节点。"
            )

        if len(matches) > 1:
            raise ValueError(
                f"节点 ID 不唯一：{center_node_id}"
            )

        return matches[0]

    if not center_node_name:
        return None

    normalized_name = _normalize_text(center_node_name)

    matches = []

    for element_id, node in nodes.items():
        properties = node["properties"]

        candidate_names = {
            _normalize_text(properties.get("name")),
            _normalize_text(
                properties.get("canonical_name")
            ),
            _normalize_text(
                properties.get("course_name")
            ),
            _normalize_text(properties.get("file_name")),
            _frontend_node_name(node, document_map)
        }

        aliases = properties.get("aliases", [])

        if isinstance(aliases, list):
            candidate_names.update(
                _normalize_text(alias) for alias in aliases
            )

        if normalized_name in candidate_names:
            matches.append(element_id)

    if not matches:
        raise ValueError(
            f"未找到名称为「{center_node_name}」的节点。"
        )

    if len(matches) > 1:
        candidates = [
            {
                "id": nodes[element_id][
                    "properties"
                ].get("id"),
                "name": _frontend_node_name(
                    nodes[element_id], document_map
                ),
                "type": _frontend_node_type(
                    nodes[element_id]["labels"],
                    nodes[element_id]["properties"]
                )
            }
            for element_id in matches
        ]

        raise ValueError(
            "中心节点名称匹配到多个节点，"
            "请使用 center_node_id：\n"
            + json.dumps(
                candidates, ensure_ascii=False, indent=2
            )
        )

    return matches[0]


def _load_graph_from_neo4j(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    course_id: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError("Neo4j连接配置不完整。")
    from graph_readback import read_pages
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        driver.verify_connectivity()
        node_records = read_pages(driver, """
            MATCH (node)
            WHERE ($course_id IS NULL OR node.course_id = $course_id)
              AND elementId(node) > $after
            RETURN elementId(node) AS cursor, elementId(node) AS element_id,
                labels(node) AS labels,
                CASE WHEN node:Course THEN node {.*, review_graph_json: null,
                    model_graph_json: null, previous_review_graph_json: null,
                    review_revision: null, review_updated_at: null}
                ELSE properties(node) END AS properties
            ORDER BY cursor LIMIT $limit
        """, course_id)
        nodes = {
            record["element_id"]: {
                "labels": list(record["labels"]),
                "properties": _restore_neo4j_review_metadata(
                    dict(record["properties"]), concept="Concept" in record["labels"]
                ) if "Concept" in record["labels"] else {k: v for k, v in dict(record["properties"]).items()
                    if "Course" not in record["labels"] or k not in COURSE_SNAPSHOT_FIELDS}
            } for record in node_records
        }
        concept_ids = {identity for identity, node in nodes.items() if "Concept" in node["labels"]}
        relationship_records = read_pages(driver, """
            MATCH (source)-[relationship]->(target)
            WHERE ($course_id IS NULL OR (source.course_id = $course_id AND target.course_id = $course_id))
              AND elementId(relationship) > $after
            RETURN elementId(relationship) AS cursor,
                elementId(source) AS source, elementId(target) AS target,
                type(relationship) AS relation, properties(relationship) AS properties
            ORDER BY cursor LIMIT $limit
        """, course_id, batch_size=400)
        relationships = [
            {"source": record["source"], "target": record["target"], "relation": record["relation"],
             "properties": _restore_neo4j_review_metadata(dict(record["properties"]))
                if record["source"] in concept_ids and record["target"] in concept_ids
                else dict(record["properties"])}
            for record in relationship_records
        ]
        return nodes, relationships
    finally:
        driver.close()


def _bfs_expand(
    center_element_id: str,
    nodes: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    radius: int
) -> Set[str]:
    adjacency = defaultdict(set)

    for relationship in relationships:
        adjacency[relationship["source"]].add(
            relationship["target"]
        )
        adjacency[relationship["target"]].add(
            relationship["source"]
        )

    visited = {center_element_id}
    frontier = {center_element_id}

    for _ in range(radius):
        next_frontier = set()

        for element_id in frontier:
            for neighbor_id in adjacency.get(
                element_id, set()
            ):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.add(neighbor_id)

        if not next_frontier:
            break

        frontier = next_frontier

    return visited


def _filter_by_layer_and_course(
    nodes: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    allowed_layers: Optional[Set[int]],
    course_id: Optional[str]
) -> Set[str]:
    normalized_course_id = (
        _normalize_text(course_id) if course_id else None
    )

    kept_ids = set()

    for element_id, node in nodes.items():
        properties = node["properties"]

        if allowed_layers is not None:
            layer = properties.get("layer")

            if layer not in allowed_layers:
                continue

        if normalized_course_id:
            node_course_id = _normalize_text(
                properties.get("course_id")
            )

            if node_course_id != normalized_course_id:
                continue

        kept_ids.add(element_id)

    return kept_ids


def _format_frontend_graph(
    nodes: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    keep_ids: Set[str]
) -> Dict[str, Any]:
    document_map = {
        _normalize_text(
            node["properties"].get("id")
        ): node["properties"]
        for node in nodes.values()
        if "Document" in node["labels"]
    }

    frontend_nodes = []

    for element_id in keep_ids:
        node = nodes[element_id]
        properties = node["properties"]
        labels = node["labels"]

        frontend_nodes.append({
            "id": _normalize_text(properties.get("id")),
            "name": _frontend_node_name(
                node, document_map
            ),
            "type": _frontend_node_type(
                labels, properties
            ),
            "labels": labels,
            "layer": properties.get("layer"),
            "category": properties.get("category"),
            "description": properties.get(
                "description", ""
            ),
            "symbolSize": _frontend_symbol_size(labels),
            "properties": properties
        })

    frontend_links = []

    for relationship in relationships:
        source_id = relationship["source"]
        target_id = relationship["target"]

        if (
            source_id not in keep_ids
            or target_id not in keep_ids
        ):
            continue

        source_properties = nodes[source_id]["properties"]
        target_properties = nodes[target_id]["properties"]

        frontend_links.append({
            "source": _normalize_text(
                source_properties.get("id")
            ),
            "target": _normalize_text(
                target_properties.get("id")
            ),
            "relation": relationship["relation"],
            "properties": relationship["properties"]
        })

    course_ids = {_normalize_text(nodes[key]["properties"].get("course_id")) for key in keep_ids}
    review_items, hierarchy_candidates = [], []
    for element_id, node in nodes.items():
        if "Course" not in node["labels"]:
            continue
        course = node["properties"]
        course_id = _normalize_text(course.get("course_id") or course.get("id"))
        if element_id not in keep_ids and course_id not in course_ids:
            continue
        for field, destination in (("review_items_json", review_items), ("hierarchy_candidates_json", hierarchy_candidates)):
            records = json.loads(course.get(field, "[]"))
            _neo4j_audit_json(records, field)
            if field == "hierarchy_candidates_json":
                records = [dict(record, source=_neo4j_concept_id(course_id, record["source"]),
                                target=_neo4j_concept_id(course_id, record["target"])) for record in records]
            destination.extend(records)
    return {
        "nodes": frontend_nodes,
        "links": frontend_links,
        "review_items": review_items,
        "hierarchy_candidates": hierarchy_candidates,
        "node_count": len(frontend_nodes),
        "link_count": len(frontend_links)
    }


# ============================================================
# 15. 对外公开入口一：构建并写入数据库
# ============================================================
def build_and_store_databases(
    kg_data_dir: str = DEFAULT_KG_DATA_DIR,
    debug_dir: str = DEFAULT_DEBUG_DIR,
    work_dir: str = DEFAULT_WORK_DIR,

    deepseek_api_key: str = DEFAULT_DEEPSEEK_API_KEY,
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL,

    neo4j_uri: str = DEFAULT_NEO4J_URI,
    neo4j_user: str = DEFAULT_NEO4J_USER,
    neo4j_password: str = DEFAULT_NEO4J_PASSWORD,

    pinecone_api_key: str = DEFAULT_PINECONE_API_KEY,
    pinecone_index_name: str = (
        DEFAULT_PINECONE_INDEX_NAME
    ),
    pinecone_namespace: str = (
        DEFAULT_PINECONE_NAMESPACE
    ),
    pinecone_dimension: int = (
        DEFAULT_PINECONE_DIMENSION
    ),
    pinecone_metric: str = DEFAULT_PINECONE_METRIC,
    pinecone_cloud: str = DEFAULT_PINECONE_CLOUD,
    pinecone_region: str = DEFAULT_PINECONE_REGION,
    create_pinecone_index: bool = True,

    data_version: str = DEFAULT_DATA_VERSION,

    resume: bool = True,
    force_reextract: bool = False,
    reuse_final_concepts: bool = True,

    write_neo4j: bool = True,
    write_pinecone: bool = True,

    concept_confidence: float = 0.6,
    relation_confidence: float = 0.6,

    # 【本次改动 F】默认值由 2 改为 1：
    # 关系精简后语义更清晰，同一对概念保留一条最优关系即可，
    # 减少图谱中的冗余边。
    max_relations_per_pair: int = 1,

    page_node_batch_size: int = 12,

    metadata_similarity_threshold: float = 0.82,
    metadata_safe_evidence_count: int = 3,
    metadata_safe_page_count: int = 2,

    tier_weight_evidence: float = 1.0,
    tier_weight_doc_breadth: float = 3.0,
    tier_weight_page_breadth: float = 1.5,
    tier_label_penalty: float = 25.0,
    tier_category_bonus: float = 2.0,
    tier_category_penalty: float = 2.0,
    tier_threshold_core: float = 8.0,
    tier_threshold_secondary: float = 3.0,

    # 【本次改动 F】新增参数：IDF 泛在惩罚倍率
    # 建议范围 5.0 ~ 12.0；数值越大，对"到处出现"的背景词打击越重。
    tier_ubiquity_penalty: float = 8.0,

    parent_overlap_threshold: float = 0.6,
    llm_client=None,
    synthesize_descriptions: bool = True,
    synonym_pair_limit: Optional[int] = None,
) -> Dict[str, Any]:
    config = _PipelineConfig(
        kg_data_dir=kg_data_dir,
        debug_dir=debug_dir,
        work_dir=work_dir,

        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_model=deepseek_model,

        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,

        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        pinecone_dimension=pinecone_dimension,
        pinecone_metric=pinecone_metric,
        pinecone_cloud=pinecone_cloud,
        pinecone_region=pinecone_region,

        data_version=data_version,

        resume=resume,
        force_reextract=force_reextract,
        reuse_final_concepts=reuse_final_concepts,

        write_neo4j=write_neo4j,
        write_pinecone=write_pinecone,

        concept_confidence=concept_confidence,
        relation_confidence=relation_confidence,
        max_relations_per_pair=max_relations_per_pair,
        page_node_batch_size=page_node_batch_size,

        metadata_similarity_threshold=(
            metadata_similarity_threshold
        ),
        metadata_safe_evidence_count=(
            metadata_safe_evidence_count
        ),
        metadata_safe_page_count=(
            metadata_safe_page_count
        ),

        tier_weight_evidence=tier_weight_evidence,
        tier_weight_doc_breadth=tier_weight_doc_breadth,
        tier_weight_page_breadth=tier_weight_page_breadth,
        tier_label_penalty=tier_label_penalty,
        tier_category_bonus=tier_category_bonus,
        tier_category_penalty=tier_category_penalty,
        tier_threshold_core=tier_threshold_core,
        tier_threshold_secondary=(
            tier_threshold_secondary
        ),

        # 【本次改动 F】新增传参
        tier_ubiquity_penalty=tier_ubiquity_penalty,

        parent_overlap_threshold=(
            parent_overlap_threshold
        )
    )

    os.makedirs(config.work_dir, exist_ok=True)

    pipeline_start_time = time.time()

    print("=" * 60)
    print("🏗️ 知识图谱构建与数据库写入流水线启动")
    print("=" * 60)

    print("\n📂 [阶段1/4] 读取并校验正式数据……")

    formal_data = _load_formal_data(config)

    validation_report = _validate_formal_data(formal_data)

    print("\n🧠 [阶段2/4] 构建知识层（抽取/合并/打分/挂靠）……")

    knowledge_data, knowledge_report = (
        _build_knowledge_layer(formal_data, config, client=llm_client,
            synthesize_descriptions=synthesize_descriptions, synonym_pair_limit=synonym_pair_limit)
    )

    teacher_exports = export_teacher_bundle(
        knowledge_data, formal_data, os.path.join(config.work_dir, "teacher_view")
    )

    neo4j_report = None
    pinecone_report = None

    if write_neo4j:
        print("\n🗄️ [阶段3/4] 写入 Neo4j……")

        neo4j_report = _upload_to_neo4j(
            formal_data, knowledge_data, config
        )
    else:
        print("\n⏭️ [阶段3/4] 跳过 Neo4j 写入（已禁用）。")

    if write_pinecone:
        print("\n📌 [阶段4/4] 写入 Pinecone……")

        pinecone_report = _upload_to_pinecone(
            formal_data, config, create_pinecone_index
        )
    else:
        print("\n⏭️ [阶段4/4] 跳过 Pinecone 写入（已禁用）。")

    elapsed_seconds = round(
        time.time() - pipeline_start_time, 2
    )

    pipeline_report = {
        "status": "success",
        "elapsed_seconds": elapsed_seconds,
        "data_version": data_version,
        "validation_report": validation_report,
        "knowledge_report": knowledge_report,
        "teacher_exports": teacher_exports,
        "neo4j_report": neo4j_report,
        "pinecone_report": pinecone_report
    }

    _write_json(config.pipeline_report_path, pipeline_report)

    print("\n" + "=" * 60)
    print(
        "🎉 全流程完成，用时 "
        f"{elapsed_seconds} 秒。"
    )
    print(
        "📄 汇总报告已写入："
        f"{config.pipeline_report_path}"
    )
    print("=" * 60)

    return pipeline_report


# ============================================================
# 16. 对外公开入口二：从 Neo4j 导出前端图谱
# ============================================================
def export_graph_from_neo4j(
    output_path: str,
    center_node_id: Optional[str] = None,
    center_node_name: Optional[str] = None,
    radius: int = 2,
    allowed_layers: Optional[List[int]] = None,
    course_id: Optional[str] = None,
    neo4j_uri: str = DEFAULT_NEO4J_URI,
    neo4j_user: str = DEFAULT_NEO4J_USER,
    neo4j_password: str = DEFAULT_NEO4J_PASSWORD
) -> Dict[str, Any]:
    print("=" * 60)
    print("🌐 从 Neo4j 导出前端知识图谱")
    print("=" * 60)

    nodes, relationships = _load_graph_from_neo4j(
        neo4j_uri, neo4j_user, neo4j_password, course_id=course_id
    )

    print(
        f"\n📥 已从 Neo4j 读取 {len(nodes)} 个节点、"
        f"{len(relationships)} 条关系。"
    )

    document_map = {
        _normalize_text(
            node["properties"].get("id")
        ): node["properties"]
        for node in nodes.values()
        if "Document" in node["labels"]
    }

    layer_filter_set = (
        set(allowed_layers)
        if allowed_layers is not None
        else None
    )

    layer_course_keep_ids = _filter_by_layer_and_course(
        nodes, relationships, layer_filter_set, course_id
    )

    center_element_id = _find_center_node(
        nodes,
        document_map,
        center_node_id,
        center_node_name
    )

    if center_element_id is not None:
        if center_element_id not in layer_course_keep_ids:
            raise ValueError(
                "中心节点被 layer/course_id 过滤条件排除，"
                "请检查过滤参数。"
            )

        bfs_keep_ids = _bfs_expand(
            center_element_id,
            nodes,
            relationships,
            radius
        )

        keep_ids = bfs_keep_ids & layer_course_keep_ids
        keep_ids.add(center_element_id)
    else:
        keep_ids = layer_course_keep_ids

    frontend_graph = _format_frontend_graph(
        nodes, relationships, keep_ids
    )

    frontend_graph["center_node_id"] = (
        _normalize_text(
            nodes[center_element_id]["properties"].get("id")
        )
        if center_element_id is not None
        else None
    )
    frontend_graph["radius"] = (
        radius if center_element_id is not None else None
    )
    frontend_graph["allowed_layers"] = allowed_layers
    frontend_graph["course_id"] = course_id

    _write_json(output_path, frontend_graph)

    print(
        "\n✅ 导出完成："
        f"{frontend_graph['node_count']} 个节点、"
        f"{frontend_graph['link_count']} 条关系。"
    )
    print(f"📄 已写入：{output_path}")
    print("=" * 60)

    return frontend_graph


# ============================================================
# 17. 模块公开接口
# ============================================================
__all__ = [
    "build_and_store_databases",
    "export_graph_from_neo4j",
    "ALLOWED_RELATION_TYPES",
    "RELATION_DEFINITIONS",
    "RELATION_BASE_WEIGHTS"
]


# ============================================================
# 18. 脚本直接运行入口
# ============================================================
if __name__ == "__main__":
    build_and_store_databases(
        resume=True,
        force_reextract=False,
        reuse_final_concepts=True,
        write_neo4j=True,
        write_pinecone=True,
        create_pinecone_index=True
    )
