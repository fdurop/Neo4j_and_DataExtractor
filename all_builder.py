import os
import re
import json
import time
import random
import hashlib
import unicodedata

from dataclasses import dataclass, field
from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import requests

from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec

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


_load_env_file()

# ============================================================
# 2. 默认配置
# ============================================================
DEFAULT_KG_DATA_DIR = os.getenv(
    "KG_DATA_DIR",
    (
        r"C:\Users\Lin\extraction\output"
        r"\qwen_qwen3.7-plus_6\kg_data"
    )
)

DEFAULT_DEBUG_DIR = os.getenv(
    "DEBUG_DIR",
    (
        r"C:\Users\Lin\extraction\output"
        r"\qwen_qwen3.7-plus_6\debug"
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

DEFAULT_DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat"
)

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

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

ALLOWED_RELATION_TYPES = {
    "IS_A",
    "PART_OF",
    "HAS_PART",
    "HAS_PARAMETER",
    "CONFIGURED_BY",
    "CONTROLS",
    "DRIVES",
    "CONNECTS_TO",
    "INPUTS",
    "OUTPUTS",
    "AFFECTS",
    "IMPLEMENTS",
    "REQUIRES",
    "PREREQUISITE_OF",
    "APPLIES_TO",
    "RELATED_TO"
}

RELATION_DEFINITIONS = {
    "IS_A": "源概念是目标概念的一种或一个具体实例。",
    "PART_OF": "源概念是目标概念的组成部分。",
    "HAS_PART": "源概念包含目标概念作为组成部分。",
    "HAS_PARAMETER": "源概念具有目标参数。",
    "CONFIGURED_BY": "源概念由目标概念进行配置或设置。",
    "CONTROLS": "源概念控制目标概念。",
    "DRIVES": "源概念驱动目标概念。",
    "CONNECTS_TO": "源概念与目标概念存在明确连接。",
    "INPUTS": "源概念接收目标概念作为输入。",
    "OUTPUTS": "源概念产生或输出目标概念。",
    "AFFECTS": "源概念会影响目标概念。",
    "IMPLEMENTS": "源概念实现目标功能、方法或原理。",
    "REQUIRES": "源概念需要目标概念才能运行或成立。",
    "PREREQUISITE_OF": "源概念是理解目标概念的前置知识。",
    "APPLIES_TO": "源概念应用于目标对象或场景。",
    "RELATED_TO": "二者相关，但材料不足以支持更明确的关系。"
}

RELATION_BASE_WEIGHTS = {
    "CONTROLS": 1.00,
    "DRIVES": 1.00,
    "HAS_PARAMETER": 0.97,
    "CONFIGURED_BY": 0.97,
    "INPUTS": 0.95,
    "OUTPUTS": 0.95,
    "IS_A": 0.95,
    "CONNECTS_TO": 0.92,
    "PART_OF": 0.90,
    "HAS_PART": 0.90,
    "IMPLEMENTS": 0.88,
    "AFFECTS": 0.85,
    "REQUIRES": 0.82,
    "PREREQUISITE_OF": 0.78,
    "APPLIES_TO": 0.75,
    "RELATED_TO": 0.35
}

# ------------------------------------------------------------
# 改动 1：_LABEL_LIKE_PATTERN 正则扩展
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


def _neo4j_safe_props(
        properties: Dict[str, Any]
) -> Dict[str, Any]:
    result = {}

    for key, value in (properties or {}).items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            result[key] = value
            continue

        if isinstance(value, list):
            if all(
                    isinstance(item, (str, int, float, bool))
                    for item in value
            ):
                result[key] = value

    return result


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

                content = (
                    response.json()
                    ["choices"][0]["message"]["content"]
                )

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

    all_known_ids = document_ids | page_ids | node_ids

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


def _load_processed_task_ids(partial_path: str) -> Set[str]:
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

            if task_id:
                result.add(task_id)

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

    page_summary = _normalize_text(page.get("summary"))

    page_text = _normalize_text(
        page.get("description") or page.get("text_content")
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

        material_sections.append(
            f"[Node ID: {node_id}]\n"
            f"Type: {node_type}\n"
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

没有有效证据的概念不要输出。
不得把相关但不同的对象合并。
不要为了凑数量提取无意义词语。

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
- 同一对概念最多输出两种关系。

【输出格式】

{{
  "concepts": [
    {{
      "canonical_name": "标准名称",
      "aliases": ["别名"],
      "description": "概念描述",
      "category": "类别",
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
            "aliases": aliases,
            "description": _normalize_text(
                item.get("description")
            ),
            "category": _normalize_text(
                item.get("category")
            ) or "未分类",
            "evidence_node_ids": evidence_ids,
            "source_page_ids": [page_id],
            "confidence": confidence
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
            "confidence": confidence
        })

    relationships = _limit_relations_per_pair(
        relationships,
        config.max_relations_per_pair
    )

    return concepts, relationships


def _load_partial_results(
        partial_path: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    concepts = []
    relationships = []

    if not os.path.isfile(partial_path):
        return concepts, relationships

    with open(partial_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            concepts.extend(record.get("concepts", []))
            relationships.extend(
                record.get("relationships", [])
            )

    return concepts, relationships


# ============================================================
# 9. 概念标准化和同义词合并
# ============================================================
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
        key = _normalize_concept_key(
            concept.get("canonical_name", "")
        )

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
        ] + concept_a.get("aliases", [])

        keys_a = {
            _normalize_concept_key(name)
            for name in names_a
            if _normalize_concept_key(name)
        }

        for index_b in range(index_a + 1, len(concepts)):
            concept_b = concepts[index_b]

            names_b = [
                concept_b["canonical_name"]
            ] + concept_b.get("aliases", [])

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


def _adjudicate_synonyms(
        client: _DeepSeekClient,
        concepts: List[Dict[str, Any]],
        candidates: List[Tuple[str, str]]
) -> Dict[str, str]:
    concept_map = {
        item["canonical_name"]: item for item in concepts
    }

    union_find = _UnionFind(list(concept_map.keys()))

    preferred_names = {}
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
                    )
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
                    )
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
            name_a = _normalize_text(decision.get("concept_a"))
            name_b = _normalize_text(decision.get("concept_b"))

            action = _normalize_text(
                decision.get("decision")
            ).upper()

            preferred_name = _normalize_text(
                decision.get("preferred_name")
            )

            if (
                    action != "MERGE"
                    or name_a not in concept_map
                    or name_b not in concept_map
            ):
                continue

            union_find.union(name_a, name_b)

            if preferred_name in {name_a, name_b}:
                preferred_names[
                    frozenset({name_a, name_b})
                ] = preferred_name

        if start + batch_size < len(candidates):
            time.sleep(random.uniform(1.0, 2.0))

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


# ------------------------------------------------------------
# 改动 3：_merge_duplicate_relationships
# 新增 auto_generated 字段：只要 group 中有一条为 True 则为 True
# ------------------------------------------------------------
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

        # 改动 3：保留 auto_generated 标记
        # 只要 group 里有一条是自动挂靠生成的，合并后仍标记为 True
        is_auto_generated = any(
            item.get("auto_generated", False)
            for item in group
        )

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
            "auto_generated": is_auto_generated
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
            "auto_generated": False
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

        name = _normalize_text(
            concept.get("canonical_name")
        )

        # 改动 1：使用扩展后的正则 + 白名单豁免
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


# ------------------------------------------------------------
# 改动 2：_link_parents_for_tier
# PART_OF → HAS_PART（父节点包含子节点，方向正确）
# ------------------------------------------------------------
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

        if child_category in _PARAMETER_CATEGORY_HINTS:
            relation_type = "HAS_PARAMETER"
        else:
            # 改动 2：修正为 HAS_PART（父→子，语义正确）
            relation_type = "HAS_PART"

        auto_relations.append({
            "source": best_parent["canonical_name"],
            "target": child_name,
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

    promoted = 0

    for orphan in orphans:
        orphan["layer"] = LAYER_SECONDARY
        orphan["promoted_from_tier3"] = True
        promoted += 1

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
        child_name = relation.get("target")
        text = description_by_name.get(child_name, "")

        for hint, order_value in _STEP_ORDER_HINTS:
            if hint in text:
                relation["order"] = order_value
                break


# ============================================================
# 10. 构建知识层
# ============================================================
def _build_knowledge_layer(
        formal_data: Dict[str, Any],
        config: _PipelineConfig
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if (
            config.reuse_final_concepts
            and not config.force_reextract
            and os.path.isfile(config.final_concepts_path)
    ):
        print(
            "♻️ 检测到已有最终知识层文件，直接复用："
            f"{config.final_concepts_path}"
        )

        final_data = _read_json(config.final_concepts_path)

        report = {
            "loaded_from_cache": True,
            "concept_count": len(
                final_data.get("concepts", [])
            ),
            "relationship_count": len(
                final_data.get("relationships", [])
            )
        }

        return final_data, report

    if not config.deepseek_api_key:
        raise ValueError(
            "需要重新抽取知识层，但未找到 DEEPSEEK_API_KEY。"
        )

    if config.force_reextract:
        _safe_remove(config.partial_path)
        _safe_remove(config.raw_concepts_path)
        _safe_remove(config.final_concepts_path)
        print("🧹 已清理旧知识抽取缓存，开始重新抽取。")

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

    processed_task_ids = (
        _load_processed_task_ids(config.partial_path)
        if config.resume
        else set()
    )

    if not config.resume:
        _safe_remove(config.partial_path)

    print(
        f"\n🧠 共生成 {len(tasks)} 个页面级知识抽取任务。"
    )

    metadata_filter_log: List[Dict[str, Any]] = []

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
            response = client.chat_json(
                _build_page_prompt(task, terms)
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

        _append_partial_result(
            config.partial_path,
            {
                "task_id": task["task_id"],
                "page_id": task["page_id"],
                "batch_index": task["batch_index"],
                "node_ids": [
                    _normalize_text(
                        node.get("node_id")
                        or node.get("id")
                    )
                    for node in task["nodes"]
                ],
                "concepts": concepts,
                "relationships": relationships
            }
        )

        if task_index < len(tasks):
            time.sleep(random.uniform(1.2, 2.5))

    raw_concepts, raw_relationships = _load_partial_results(
        config.partial_path
    )

    _write_json(
        config.raw_concepts_path,
        {
            "concepts": raw_concepts,
            "relationships": raw_relationships
        }
    )

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

    if candidates:
        synonym_mapping = _adjudicate_synonyms(
            client,
            premerged_concepts,
            candidates
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

    print("✍️ 统一整理 Concept 描述……")

    final_concepts = _synthesize_descriptions(
        client,
        merged_concepts
    )

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

    combined_relationships = (
        final_relationships + cleaned_auto_relations
    )

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
        "data_version": config.data_version,
        "source_hash": formal_data.get("source_hash", ""),
        "relation_limit_per_pair": (
            config.max_relations_per_pair
        ),
        "relation_base_weights": RELATION_BASE_WEIGHTS,
        "concepts": final_concepts,
        "relationships": final_relationships
    }

    _write_json(config.final_concepts_path, final_data)

    report = {
        "loaded_from_cache": False,
        "page_task_count": len(tasks),
        "raw_concept_count": len(raw_concepts),
        "premerged_concept_count": len(premerged_concepts),
        "synonym_candidate_count": len(candidates),
        "concept_count": len(final_concepts),
        "raw_relationship_count": len(raw_relationships),
        "relationship_count": len(final_relationships),
        "layer_distribution": {
            str(k): v for k, v in layer_distribution.items()
        },
        "excluded_metadata_count": len([
            r for r in metadata_filter_log
            if not r.get("recover_hint")
        ]),
        # P2 改动：字段名更明确，避免误读为"已救回图里"
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

    print("\n" + "=" * 60)
    print("🚀 开始写入 Neo4j 双层图谱")
    print("=" * 60)

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password)
    )

    def run_query(query, parameters=None):
        with driver.session() as session:
            session.run(query, parameters or {}).consume()

    def run_unwind(query, rows, label, chunk_size=500):
        if not rows:
            print(f"  - {label}：无数据")
            return

        written = 0

        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]

            with driver.session() as session:
                session.run(
                    query,
                    {
                        "batch": chunk,
                        "data_version": config.data_version
                    }
                ).consume()

            written += len(chunk)

        print(f"  ✓ {label}：{written} 条")

    try:
        driver.verify_connectivity()

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
            CREATE CONSTRAINT concept_name_unique
            IF NOT EXISTS
            FOR (n:Concept)
            REQUIRE n.name IS UNIQUE
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
                course.layer = $layer,
                course.data_version = $data_version
            """,
            {
                "course_id": course_id,
                "course_name": course_name,
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
                "id": concept_name,
                "name": concept_name,
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
                "tier_score": concept.get("tier_score", 0.0),
                "promoted_from_tier3": bool(
                    concept.get("promoted_from_tier3", False)
                )
            })

            valid_concept_names.add(concept_name)

        run_unwind(
            """
            UNWIND $batch AS row
            MERGE (concept:Concept {name: row.name})
            SET concept.id = row.id,
                concept.canonical_name = row.name,
                concept.aliases = row.aliases,
                concept.description = row.description,
                concept.category = row.category,
                concept.confidence = row.confidence,
                concept.evidence_node_ids =
                    row.evidence_node_ids,
                concept.evidence_count = row.evidence_count,
                concept.source_page_ids =
                    row.source_page_ids,
                concept.source_document_ids =
                    row.source_document_ids,
                concept.course_id = row.course_id,
                concept.layer = row.layer,
                concept.tier_score = row.tier_score,
                concept.promoted_from_tier3 =
                    row.promoted_from_tier3,
                concept.data_version = $data_version
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
                    "evidence_node_id": evidence_node_id
                })

        run_unwind(
            f"""
            UNWIND $batch AS row
            MATCH (concept:Concept {{
                name: row.concept_name
            }})
            MATCH (material {{id: row.evidence_node_id}})
            WHERE material.layer = {LAYER_MATERIAL}
              AND NOT material:Page
              AND NOT material:Document
              AND NOT material:Course
            MERGE (concept)-[
                evidence:HAS_EVIDENCE
            ]->(material)
            SET evidence.data_version = $data_version
            """,
            evidence_rows,
            "HAS_EVIDENCE"
        )

        # ==================================================
        # 11.7 Concept 语义关系
        # 改动 3b：append 时补读 auto_generated 字段，
        #          Cypher SET 时补写入 auto_generated
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

            # 改动 3b：读取 auto_generated 字段
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
                # 改动 3b：透传 auto_generated
                "auto_generated": is_auto_generated
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
                    name: row.source
                }})
                MATCH (target:Concept {{
                    name: row.target
                }})
                MERGE (source)-[
                    relation:{safe_relationship_type}
                ]->(target)
                SET relation.description =
                        row.description,
                    relation.evidence_node_ids =
                        row.evidence_node_ids,
                    relation.source_page_ids =
                        row.source_page_ids,
                    relation.evidence_count =
                        row.evidence_count,
                    relation.confidence =
                        row.confidence,
                    relation.base_weight =
                        row.base_weight,
                    relation.auto_generated =
                        row.auto_generated,
                    relation.data_version =
                        $data_version
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

            # 验证查询：确认 auto_generated 边已写入
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
    client: Pinecone,
    config: _PipelineConfig,
    create_if_missing: bool
) -> None:
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
    neo4j_password: str
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if not neo4j_uri:
        raise ValueError("未找到 NEO4J_URI。")

    if not neo4j_user:
        raise ValueError("未找到 NEO4J_USER。")

    if not neo4j_password:
        raise ValueError("未找到 NEO4J_PASSWORD。")

    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            node_records = session.run(
                """
                MATCH (node)
                RETURN
                    elementId(node) AS element_id,
                    labels(node) AS labels,
                    properties(node) AS properties
                """
            )

            nodes = {
                record["element_id"]: {
                    "labels": list(record["labels"]),
                    "properties": dict(record["properties"])
                }
                for record in node_records
            }

            relationship_records = session.run(
                """
                MATCH (source)-[relationship]->(target)
                RETURN
                    elementId(source) AS source,
                    elementId(target) AS target,
                    type(relationship) AS relation,
                    properties(relationship) AS properties
                """
            )

            relationships = [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "properties": dict(record["properties"])
                }
                for record in relationship_records
            ]

        return nodes, relationships

    finally:
        driver.close()


# ============================================================
# 15. 公开函数一：文件到数据库的一键式处理
# ============================================================
def build_and_store_databases(
    kg_data_dir: str = DEFAULT_KG_DATA_DIR,
    debug_dir: str = DEFAULT_DEBUG_DIR,
    work_dir: str = DEFAULT_WORK_DIR,
    resume: bool = True,
    force_reextract: bool = False,
    reuse_final_concepts: bool = True,
    write_neo4j: bool = True,
    write_pinecone: bool = True,
    create_pinecone_index: bool = True,
    pinecone_index_name: str = DEFAULT_PINECONE_INDEX_NAME,
    pinecone_namespace: str = DEFAULT_PINECONE_NAMESPACE,
    concept_confidence: float = 0.55,
    relation_confidence: float = 0.55,
    max_relations_per_pair: int = 2,
    page_node_batch_size: int = 12,
    data_version: str = DEFAULT_DATA_VERSION,
    metadata_similarity_threshold: float = 0.85,
    metadata_safe_evidence_count: int = 5,
    metadata_safe_page_count: int = 3,
    tier_weight_evidence: float = 1.0,
    tier_weight_doc_breadth: float = 3.0,
    tier_weight_page_breadth: float = 1.5,
    tier_label_penalty: float = 25.0,
    tier_category_bonus: float = 2.0,
    tier_category_penalty: float = 2.0,
    tier_threshold_core: float = 10.0,
    tier_threshold_secondary: float = 4.0,
    parent_overlap_threshold: float = 0.6,
    deepseek_api_key: Optional[str] = None,
    deepseek_base_url: Optional[str] = None,
    deepseek_model: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_dimension: int = DEFAULT_PINECONE_DIMENSION,
    pinecone_metric: str = DEFAULT_PINECONE_METRIC,
    pinecone_cloud: str = DEFAULT_PINECONE_CLOUD,
    pinecone_region: str = DEFAULT_PINECONE_REGION
) -> Dict[str, Any]:
    """
    从 extraction 正式输出文件开始，一键完成全流程。

    升级点：
      - 素材层 layer=0，知识层 Concept layer∈{1,2,3}（tier 打分）；
      - 非概念元数据过滤（含安全阀）；
      - 概念父子自动挂靠 + 孤儿提升。

    默认调用：
        report = build_and_store_databases()

    本函数不会删除 Neo4j，也不会删除 Pinecone 索引。
    """
    if max_relations_per_pair <= 0:
        raise ValueError(
            "max_relations_per_pair 必须大于 0。"
        )

    if page_node_batch_size <= 0:
        raise ValueError(
            "page_node_batch_size 必须大于 0。"
        )

    if not (0 <= concept_confidence <= 1):
        raise ValueError(
            "concept_confidence 必须位于 0 到 1 之间。"
        )

    if not (0 <= relation_confidence <= 1):
        raise ValueError(
            "relation_confidence 必须位于 0 到 1 之间。"
        )

    if not (0 <= parent_overlap_threshold <= 1):
        raise ValueError(
            "parent_overlap_threshold 必须位于 0 到 1 之间。"
        )

    if not (0 <= metadata_similarity_threshold <= 1):
        raise ValueError(
            "metadata_similarity_threshold "
            "必须位于 0 到 1 之间。"
        )

    os.makedirs(work_dir, exist_ok=True)

    config = _PipelineConfig(
        kg_data_dir=os.path.abspath(kg_data_dir),
        debug_dir=os.path.abspath(debug_dir),
        work_dir=os.path.abspath(work_dir),
        deepseek_api_key=(
            deepseek_api_key
            if deepseek_api_key is not None
            else DEFAULT_DEEPSEEK_API_KEY
        ),
        deepseek_base_url=(
            deepseek_base_url
            if deepseek_base_url is not None
            else DEFAULT_DEEPSEEK_BASE_URL
        ),
        deepseek_model=(
            deepseek_model
            if deepseek_model is not None
            else DEFAULT_DEEPSEEK_MODEL
        ),
        neo4j_uri=(
            neo4j_uri
            if neo4j_uri is not None
            else DEFAULT_NEO4J_URI
        ),
        neo4j_user=(
            neo4j_user
            if neo4j_user is not None
            else DEFAULT_NEO4J_USER
        ),
        neo4j_password=(
            neo4j_password
            if neo4j_password is not None
            else DEFAULT_NEO4J_PASSWORD
        ),
        pinecone_api_key=(
            pinecone_api_key
            if pinecone_api_key is not None
            else DEFAULT_PINECONE_API_KEY
        ),
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
        tier_threshold_secondary=tier_threshold_secondary,
        parent_overlap_threshold=parent_overlap_threshold
    )

    print("\n" + "=" * 70)
    print("🌟 启动文件到数据库的一键式构建流程")
    print("=" * 70)
    print(f"正式数据目录：{config.kg_data_dir}")
    print(f"工作目录：{config.work_dir}")
    print(
        "Neo4j："
        + ("启用" if config.write_neo4j else "跳过")
    )
    print(
        "Pinecone："
        + ("启用" if config.write_pinecone else "跳过")
    )

    started_at = time.time()

    formal_data = _load_formal_data(config)

    formal_validation = _validate_formal_data(formal_data)

    if (
        config.reuse_final_concepts
        and not config.force_reextract
        and os.path.isfile(config.final_concepts_path)
    ):
        cached_data = _read_json(
            config.final_concepts_path, {}
        )

        cached_source_hash = _normalize_text(
            cached_data.get("source_hash")
        )

        current_source_hash = _normalize_text(
            formal_data.get("source_hash")
        )

        if (
            cached_source_hash
            and current_source_hash
            and cached_source_hash != current_source_hash
        ):
            print(
                "⚠️ 检测到 extraction 正式输入已经变化，"
                "旧知识层缓存将不再复用。"
            )
            config.force_reextract = True

    knowledge_data, knowledge_report = (
        _build_knowledge_layer(formal_data, config)
    )

    if config.write_neo4j:
        neo4j_report = _upload_to_neo4j(
            formal_data, knowledge_data, config
        )
    else:
        neo4j_report = {"status": "skipped"}

    if config.write_pinecone:
        pinecone_report = _upload_to_pinecone(
            formal_data,
            config,
            create_if_missing=create_pinecone_index
        )
    else:
        pinecone_report = {"status": "skipped"}

    elapsed_seconds = round(time.time() - started_at, 2)

    report = {
        "status": "success",
        "elapsed_seconds": elapsed_seconds,
        "configuration": {
            "kg_data_dir": config.kg_data_dir,
            "debug_dir": config.debug_dir,
            "work_dir": config.work_dir,
            "data_version": config.data_version,
            "pinecone_index_name": (
                config.pinecone_index_name
            ),
            "pinecone_namespace": (
                config.pinecone_namespace
            ),
            "max_relations_per_pair": (
                config.max_relations_per_pair
            ),
            "concept_confidence": (
                config.concept_confidence
            ),
            "relation_confidence": (
                config.relation_confidence
            ),
            "parent_overlap_threshold": (
                config.parent_overlap_threshold
            ),
            "tier_threshold_core": (
                config.tier_threshold_core
            ),
            "tier_threshold_secondary": (
                config.tier_threshold_secondary
            )
        },
        "formal_data_validation": formal_validation,
        "knowledge_extraction": knowledge_report,
        "neo4j": neo4j_report,
        "pinecone": pinecone_report,
        "files": {
            "knowledge_graph": (
                config.final_concepts_path
            ),
            "knowledge_quality_report": (
                config.quality_report_path
            ),
            "pipeline_report": (
                config.pipeline_report_path
            )
        }
    }

    _write_json(config.pipeline_report_path, report)

    print("\n" + "=" * 70)
    print("✅ 文件到数据库的一键式流程执行完成")
    print("=" * 70)
    print(f"总耗时：{elapsed_seconds} 秒")
    print(
        "Concept 数量："
        f"{knowledge_report.get('concept_count', 0)}"
    )
    print(
        "知识关系数量："
        f"{knowledge_report.get('relationship_count', 0)}"
    )
    print(
        "运行报告："
        f"{config.pipeline_report_path}"
    )

    return report


# ============================================================
# 16. 公开函数二：从 Neo4j 导出图谱
# ============================================================
def export_graph_from_neo4j(
    output_path: str,
    center_node_id: Optional[str] = None,
    center_node_name: Optional[str] = None,
    radius: int = 0,
    layer: int = -1,
    course_id: Optional[str] = None,
    include_properties: bool = True,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> Dict[str, Any]:
    """
    从 Neo4j 导出前端可使用的图谱 JSON。

    layer 图层筛选（升级后语义，注意与旧版不同）：

      - -1：不过滤，知识层和素材层全都要（等价旧版的 0）；
      -  0：仅素材层（Course/Document/Page/Material）；
      -  1：仅核心知识点（Concept, layer=1）；
      -  2：仅附属知识点（Concept, layer=2）；
      -  3：仅叶子知识点（Concept, layer=3）。

    radius：
      - radius=0：未指定中心节点时导出全部符合条件节点；
        指定中心节点时导出其完整连通分量。
      - radius>0：从中心节点向外扩展指定跳数。
    """
    # ======================================================
    # 16.1 参数校验
    # ======================================================
    if not _normalize_text(output_path):
        raise ValueError("output_path 不能为空。")

    if not isinstance(radius, int):
        raise TypeError("radius 必须是整数。")

    if radius < 0:
        raise ValueError("radius 不能小于 0。")

    if layer not in {-1, 0, 1, 2, 3}:
        raise ValueError(
            "layer 只能是 -1、0、1、2 或 3。"
        )

    if (
        not center_node_id
        and not center_node_name
        and radius > 0
    ):
        raise ValueError(
            "radius 大于 0 时，必须指定 "
            "center_node_id 或 center_node_name。"
        )

    target_neo4j_uri = (
        neo4j_uri
        if neo4j_uri is not None
        else DEFAULT_NEO4J_URI
    )

    target_neo4j_user = (
        neo4j_user
        if neo4j_user is not None
        else DEFAULT_NEO4J_USER
    )

    target_neo4j_password = (
        neo4j_password
        if neo4j_password is not None
        else DEFAULT_NEO4J_PASSWORD
    )

    print("\n" + "=" * 60)
    print("📤 开始从 Neo4j 导出图谱")
    print("=" * 60)

    # ======================================================
    # 16.2 从 Neo4j 读取完整图数据
    # ======================================================
    nodes, relationships = _load_graph_from_neo4j(
        target_neo4j_uri,
        target_neo4j_user,
        target_neo4j_password
    )

    if not nodes:
        raise ValueError("Neo4j 中没有可导出的节点。")

    document_map = {
        _normalize_text(
            node["properties"].get("id")
        ): node["properties"]
        for node in nodes.values()
        if "Document" in node["labels"]
    }

    # ======================================================
    # 16.3 查找中心节点
    # ======================================================
    center_element_id = _find_center_node(
        nodes,
        document_map,
        center_node_id,
        center_node_name
    )

    if center_element_id is not None and layer != -1:
        center_layer = nodes[
            center_element_id
        ]["properties"].get("layer")

        if center_layer != layer:
            center_display_name = _frontend_node_name(
                nodes[center_element_id], document_map
            )

            raise ValueError(
                f"中心节点「{center_display_name}」"
                f"位于第 {center_layer} 层，"
                f"与指定的 layer={layer} 不一致。"
            )

    # ======================================================
    # 16.4 构建无向邻接表
    # ======================================================
    adjacency: Dict[str, Set[str]] = defaultdict(set)

    for relationship in relationships:
        source = relationship["source"]
        target = relationship["target"]

        if source not in nodes or target not in nodes:
            continue

        adjacency[source].add(target)
        adjacency[target].add(source)

    # ======================================================
    # 16.5 根据中心节点和半径确定初始范围
    # ======================================================
    node_distances: Dict[str, int] = {}

    if center_element_id is None:
        scope = set(nodes.keys())

        for element_id in scope:
            node_distances[element_id] = -1
    else:
        scope = {center_element_id}
        node_distances[center_element_id] = 0

        traversal_queue = deque([center_element_id])

        while traversal_queue:
            current = traversal_queue.popleft()
            current_distance = node_distances[current]

            if radius > 0 and current_distance >= radius:
                continue

            for neighbor in adjacency.get(current, set()):
                if neighbor in scope:
                    continue

                scope.add(neighbor)
                node_distances[neighbor] = (
                    current_distance + 1
                )
                traversal_queue.append(neighbor)

    # ======================================================
    # 16.6 按课程过滤
    # ======================================================
    normalized_course_id = _normalize_text(course_id)

    if normalized_course_id:
        matching_course_nodes = {
            element_id
            for element_id, node in nodes.items()
            if (
                "Course" in node["labels"]
                and _normalize_text(
                    node["properties"].get("id")
                ) == normalized_course_id
            )
        }

        if not matching_course_nodes:
            raise ValueError(
                "未找到课程 ID 为 "
                f"「{normalized_course_id}」的 Course 节点。"
            )

        course_related_nodes = set()

        for element_id, node in nodes.items():
            properties = node["properties"]

            node_id = _normalize_text(
                properties.get("id")
            )

            node_course_id = _normalize_text(
                properties.get("course_id")
            )

            if node_course_id == normalized_course_id:
                course_related_nodes.add(element_id)
            elif (
                "Course" in node["labels"]
                and node_id == normalized_course_id
            ):
                course_related_nodes.add(element_id)

        compatibility_queue = deque(
            matching_course_nodes
        )
        compatibility_seen = set(matching_course_nodes)

        while compatibility_queue:
            current = compatibility_queue.popleft()

            for neighbor in adjacency.get(current, set()):
                if neighbor in compatibility_seen:
                    continue

                neighbor_node = nodes[neighbor]
                neighbor_properties = (
                    neighbor_node["properties"]
                )

                neighbor_course_id = _normalize_text(
                    neighbor_properties.get("course_id")
                )

                if (
                    neighbor_course_id
                    and neighbor_course_id
                    != normalized_course_id
                ):
                    continue

                compatibility_seen.add(neighbor)
                course_related_nodes.add(neighbor)
                compatibility_queue.append(neighbor)

        scope &= course_related_nodes

        if not scope:
            raise ValueError(
                "应用 course_id 过滤后，没有剩余节点。"
            )

        if (
            center_element_id is not None
            and center_element_id not in scope
        ):
            raise ValueError(
                "指定的中心节点不属于课程 "
                f"「{normalized_course_id}」。"
            )

    # ======================================================
    # 16.7 按图层过滤
    # ======================================================
    if layer == -1:
        retained = set(scope)
    else:
        retained = {
            element_id
            for element_id in scope
            if nodes[element_id][
                "properties"
            ].get("layer") == layer
        }

    if not retained:
        raise ValueError(
            "应用中心节点、课程和图层筛选后，没有剩余节点。"
        )

    # ======================================================
    # 16.8 校验节点前端 ID
    # ======================================================
    frontend_id_to_element_id = {}

    for element_id in retained:
        node_id = _normalize_text(
            nodes[element_id]["properties"].get("id")
        )

        if not node_id:
            raise ValueError(
                "检测到缺少 id 属性的节点，"
                f"Neo4j elementId：{element_id}"
            )

        if node_id in frontend_id_to_element_id:
            existing_element_id = (
                frontend_id_to_element_id[node_id]
            )

            raise ValueError(
                "检测到重复前端节点 ID："
                f"{node_id}\n"
                "对应 Neo4j elementId："
                f"{existing_element_id}、{element_id}"
            )

        frontend_id_to_element_id[node_id] = element_id

    # ======================================================
    # 16.9 构造前端节点
    # ======================================================
    output_nodes = []

    for element_id in retained:
        node = nodes[element_id]
        labels = node["labels"]
        properties = node["properties"]

        node_id = _normalize_text(properties.get("id"))

        output_node = {
            "id": node_id,
            "name": _frontend_node_name(
                node, document_map
            ),
            "type": _frontend_node_type(
                labels, properties
            ),
            "symbolSize": _frontend_symbol_size(labels),
            "layer": properties.get("layer", 0)
        }

        if center_element_id is not None:
            output_node["distance"] = node_distances.get(
                element_id, -1
            )
            output_node["is_center"] = (
                element_id == center_element_id
            )

        description = _normalize_text(
            properties.get("description")
            or properties.get("summary")
            or properties.get("raw_text")
        )

        if description:
            output_node["description"] = description

        aliases = properties.get("aliases")

        if isinstance(aliases, list):
            cleaned_aliases = _unique_strings(aliases)

            if cleaned_aliases:
                output_node["aliases"] = cleaned_aliases

        confidence = properties.get("confidence")

        if confidence is not None:
            output_node["confidence"] = confidence

        category = _normalize_text(
            properties.get("category")
        )

        if category:
            output_node["category"] = category

        source_page_ids = properties.get(
            "source_page_ids"
        )

        if source_page_ids:
            output_node["source_page_ids"] = (
                source_page_ids
            )

        source_page_id = _normalize_text(
            properties.get("source_page_id")
        )

        if source_page_id:
            output_node["source_page_id"] = source_page_id

        document_id = _normalize_text(
            properties.get("document_id")
        )

        if document_id:
            output_node["document_id"] = document_id

        node_course_id = _normalize_text(
            properties.get("course_id")
        )

        if node_course_id:
            output_node["course_id"] = node_course_id

        if include_properties:
            output_node["labels"] = list(labels)
            output_node["properties"] = dict(properties)

        output_nodes.append(output_node)

    # ======================================================
    # 16.10 构造前端关系
    # ======================================================
    output_links = []
    relationship_id_counter = defaultdict(int)

    for relationship in relationships:
        source_element_id = relationship["source"]
        target_element_id = relationship["target"]

        if (
            source_element_id not in retained
            or target_element_id not in retained
        ):
            continue

        source_id = _normalize_text(
            nodes[source_element_id][
                "properties"
            ].get("id")
        )

        target_id = _normalize_text(
            nodes[target_element_id][
                "properties"
            ].get("id")
        )

        relation_type = _normalize_text(
            relationship.get("relation")
        ) or "RELATED_TO"

        relationship_signature = (
            source_id,
            target_id,
            relation_type
        )

        relationship_id_counter[
            relationship_signature
        ] += 1

        relationship_sequence = (
            relationship_id_counter[
                relationship_signature
            ]
        )

        link_id = (
            f"{source_id}::{relation_type}::"
            f"{target_id}::{relationship_sequence}"
        )

        relationship_properties = relationship.get(
            "properties", {}
        )

        output_link = {
            "id": link_id,
            "source": source_id,
            "target": target_id,
            "relation": relation_type
        }

        description = _normalize_text(
            relationship_properties.get("description")
        )

        if description:
            output_link["description"] = description

        confidence = relationship_properties.get(
            "confidence"
        )

        if confidence is not None:
            output_link["confidence"] = confidence

        base_weight = relationship_properties.get(
            "base_weight"
        )

        if base_weight is not None:
            output_link["base_weight"] = base_weight

        evidence_count = relationship_properties.get(
            "evidence_count"
        )

        if evidence_count is not None:
            output_link["evidence_count"] = evidence_count

        evidence_node_ids = relationship_properties.get(
            "evidence_node_ids"
        )

        if evidence_node_ids:
            output_link["evidence_node_ids"] = (
                evidence_node_ids
            )

        source_page_ids = relationship_properties.get(
            "source_page_ids"
        )

        if source_page_ids:
            output_link["source_page_ids"] = (
                source_page_ids
            )

        # auto_generated 透传到前端，便于可视化区分
        auto_generated = relationship_properties.get(
            "auto_generated"
        )

        if auto_generated is not None:
            output_link["auto_generated"] = auto_generated

        if include_properties:
            output_link["properties"] = dict(
                relationship_properties
            )

        output_links.append(output_link)

    # ======================================================
    # 16.11 确定中心节点导出信息
    # ======================================================
    exported_center_node_id = None
    exported_center_node_name = None

    if center_element_id is not None:
        center_node = nodes[center_element_id]

        exported_center_node_id = _normalize_text(
            center_node["properties"].get("id")
        )

        exported_center_node_name = _frontend_node_name(
            center_node, document_map
        )

    # ======================================================
    # 16.12 排序，保证重复导出结果稳定
    # ======================================================
    output_nodes.sort(
        key=lambda item: (
            item.get("layer", 0),
            item.get("type", ""),
            item.get("name", ""),
            item.get("id", "")
        )
    )

    output_links.sort(
        key=lambda item: (
            item.get("source", ""),
            item.get("target", ""),
            item.get("relation", ""),
            item.get("id", "")
        )
    )

    # ======================================================
    # 16.13 统计节点类型、图层和关系类型
    # ======================================================
    node_type_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    relationship_type_counts = defaultdict(int)
    auto_generated_link_count = 0

    for node in output_nodes:
        node_type_counts[
            node.get("type", "unknown")
        ] += 1

        layer_counts[
            str(node.get("layer", 0))
        ] += 1

    for link in output_links:
        relationship_type_counts[
            link.get("relation", "RELATED_TO")
        ] += 1

        if link.get("auto_generated"):
            auto_generated_link_count += 1

    # ======================================================
    # 16.14 生成最终结果
    # ======================================================
    result = {
        "metadata": {
            "schema_version": "3.0",
            "node_count": len(output_nodes),
            "link_count": len(output_links),
            "auto_generated_link_count": (
                auto_generated_link_count
            ),
            "center_node_id": exported_center_node_id,
            "center_node_name": (
                exported_center_node_name
            ),
            "requested_center_node_id": center_node_id,
            "requested_center_node_name": (
                center_node_name
            ),
            "radius": radius,
            "layer": layer,
            "course_id": (
                normalized_course_id or None
            ),
            "include_properties": include_properties,
            "node_type_counts": dict(
                sorted(node_type_counts.items())
            ),
            "layer_counts": dict(
                sorted(layer_counts.items())
            ),
            "relationship_type_counts": dict(
                sorted(relationship_type_counts.items())
            )
        },
        "nodes": output_nodes,
        "links": output_links
    }

    # ======================================================
    # 16.15 写入 JSON
    # ======================================================
    absolute_output_path = os.path.abspath(output_path)

    output_directory = os.path.dirname(
        absolute_output_path
    )

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    _write_json(absolute_output_path, result)

    print("\n" + "=" * 60)
    print("✅ Neo4j 图谱导出完成")
    print("=" * 60)
    print(f"节点数量：{len(output_nodes)}")
    print(f"关系数量：{len(output_links)}")
    print(f"自动挂靠关系数量：{auto_generated_link_count}")

    if exported_center_node_id:
        print(
            "中心节点："
            f"{exported_center_node_name} "
            f"({exported_center_node_id})"
        )

    print(f"输出文件：{absolute_output_path}")

    return result


# ============================================================
# 17. 模块公开接口
# ============================================================
__all__ = [
    "build_and_store_databases",
    "export_graph_from_neo4j"
]


# ============================================================
# 18. 主程序入口
# ============================================================
if __name__ == "__main__":
    try:
        pipeline_report = build_and_store_databases()

        export_output_path = os.path.join(
            DEFAULT_WORK_DIR,
            "frontend_graph_v2.json"
        )

        export_result = export_graph_from_neo4j(
            output_path=export_output_path,
            radius=0,
            layer=-1,
            include_properties=True
        )
        # layer 升级后语义（注意和旧版不同）：
        #   -1 = 不过滤，知识层+素材层全都要；
        #    0 = 只要素材层（Page、多模态数据等）；
        #    1 = 只要核心知识点（Concept layer=1）；
        #    2 = 只要附属知识点（Concept layer=2）；
        #    3 = 只要叶子知识点（Concept layer=3）。

        print("\n" + "=" * 70)
        print("🎉 全部任务执行完成")
        print("=" * 70)
        print(
            "数据库流程状态："
            f"{pipeline_report.get('status', 'unknown')}"
        )
        print(
            "导出节点数量："
            f"{export_result['metadata']['node_count']}"
        )
        print(
            "导出关系数量："
            f"{export_result['metadata']['link_count']}"
        )
        print(
            "自动挂靠关系数量："
            f"{export_result['metadata']['auto_generated_link_count']}"
        )
        print(f"前端图谱文件：{export_output_path}")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了程序运行。")
        raise SystemExit(130)

    except Exception as error:
        print("\n" + "=" * 70)
        print("❌ 流水线执行失败")
        print("=" * 70)
        print(f"{type(error).__name__}: {error}")
        raise