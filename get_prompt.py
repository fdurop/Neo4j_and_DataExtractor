# 修复一些bug
# 优化传回的形式
# 修复表格问题


import os
import json
import yaml
import glob
import requests

from typing import List, Dict, Any, Tuple, Optional
from pinecone import Pinecone
from neo4j import GraphDatabase


# ==========================================
# ⚙️ 1. 环境变量与配置加载
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KEY_ENV_PATH = os.path.join(BASE_DIR, "key.env")
YAML_CONFIG_PATH = os.path.join(BASE_DIR, "vlm_api.yaml")

KG_DATA_DIR = (
    r"C:\Users\Lin\extraction\output"
    r"\qwen_qwen3.7-plus_6\kg_data"
)

DEBUG_IMAGES_DIR = (
    r"C:\Users\Lin\extraction\output"
    r"\qwen_qwen3.7-plus_6\debug\images"
)


def load_env_file(
    env_path: Optional[str] = None,
    override: bool = False
) -> None:
    """
    读取 key.env，并将其中的配置写入环境变量。

    参数：
    - env_path：自定义环境变量文件路径；
    - override：是否覆盖系统中已存在的同名环境变量。
    """
    target_path = env_path or KEY_ENV_PATH

    if not os.path.isfile(target_path):
        print(f"⚠️ 未找到环境变量文件：{target_path}")
        return

    with open(target_path, "r", encoding="utf-8-sig") as file:
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


def load_yaml_config(
    yaml_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    读取 vlm_api.yaml。

    如果文件不存在，返回兼容的空配置结构，
    后续将使用 key.env 和默认配置。
    """
    target_path = yaml_path or YAML_CONFIG_PATH

    if not os.path.isfile(target_path):
        print(
            f"⚠️ 未找到 YAML 配置文件：{target_path}\n"
            "将尝试使用 key.env 和默认配置。"
        )
        return {
            "embedding": {},
            "vlm": {}
        }

    with open(
        target_path,
        "r",
        encoding="utf-8-sig"
    ) as file:
        loaded_config = yaml.safe_load(file) or {}

    if not isinstance(loaded_config, dict):
        raise ValueError(
            f"YAML 配置文件格式错误，应为字典结构：{target_path}"
        )

    loaded_config.setdefault("embedding", {})
    loaded_config.setdefault("vlm", {})

    if not isinstance(loaded_config["embedding"], dict):
        loaded_config["embedding"] = {}

    if not isinstance(loaded_config["vlm"], dict):
        loaded_config["vlm"] = {}

    return loaded_config


def get_nested_config(
    data: Dict[str, Any],
    section: str,
    key: str,
    default: Any = None
) -> Any:
    """安全读取嵌套配置。"""
    section_data = data.get(section, {})

    if not isinstance(section_data, dict):
        return default

    value = section_data.get(key)

    if value is None or value == "":
        return default

    return value


# 先加载 key.env，再加载 YAML。
load_env_file()
config = load_yaml_config()


# ==========================================
# ⚙️ 2. Embedding 配置
# ==========================================
EMBEDDING_API_KEY = (
    get_nested_config(
        config,
        "embedding",
        "api_key"
    )
    or get_nested_config(
        config,
        "vlm",
        "api_key"
    )
    or os.getenv("DASHSCOPE_API_KEY")
)

EMBEDDING_BASE_URL = str(
    get_nested_config(
        config,
        "embedding",
        "base_url",
        "https://dashscope.aliyuncs.com/api/v1"
    )
).rstrip("/")

EMBEDDING_MODEL = get_nested_config(
    config,
    "embedding",
    "model",
    "text-embedding-v3"
)

EMBEDDING_DIMENSION = int(
    get_nested_config(
        config,
        "embedding",
        "dimension",
        1024
    )
)

EMBEDDING_TIMEOUT = int(
    get_nested_config(
        config,
        "embedding",
        "timeout",
        60
    )
)

EMBEDDING_URL = (
    f"{EMBEDDING_BASE_URL}"
    "/services/embeddings/text-embedding/text-embedding"
)


# ==========================================
# ⚙️ 3. Pinecone 与 Neo4j 配置
# ==========================================
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    ""
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "multimodal-rag-index"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    ""
)

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    ""
)

NEO4J_USER = os.getenv(
    "NEO4J_USER",
    ""
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    ""
)


def validate_runtime_config() -> None:
    """
    在执行检索前检查必要配置。
    避免程序运行到一半才出现难以判断的认证错误。
    """
    missing_items = []

    if not EMBEDDING_API_KEY:
        missing_items.append(
            "DASHSCOPE_API_KEY 或 YAML 中的 embedding.api_key"
        )

    if not PINECONE_API_KEY:
        missing_items.append("PINECONE_API_KEY")

    if not NEO4J_URI:
        missing_items.append("NEO4J_URI")

    if not NEO4J_USER:
        missing_items.append("NEO4J_USER")

    if not NEO4J_PASSWORD:
        missing_items.append("NEO4J_PASSWORD")

    if missing_items:
        missing_text = "\n".join(
            f"- {item}"
            for item in missing_items
        )

        raise ValueError(
            "缺少运行所需配置：\n"
            f"{missing_text}\n"
            f"请检查文件：{KEY_ENV_PATH}"
        )


# ==========================================
# 🧰 4. 通用辅助函数
# ==========================================
def normalize_text(value: Any) -> str:
    """
    将字符串、列表、字典等内容安全转换为文本。
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    return str(value).strip()


def get_match_value(
    match: Any,
    key: str,
    default: Any = None
) -> Any:
    """
    兼容 Pinecone 返回字典对象或 SDK 对象两种形式。
    """
    if isinstance(match, dict):
        return match.get(key, default)

    return getattr(match, key, default)


def choose_existing_image(
    image_info: Dict[str, Any]
) -> Optional[str]:
    """
    优先选择增强图，增强图不存在时选择原图。
    """
    candidate_paths = [
        image_info.get("enhanced"),
        image_info.get("original")
    ]

    for candidate_path in candidate_paths:
        if not candidate_path:
            continue

        normalized_path = os.path.normpath(
            str(candidate_path)
        )

        if os.path.isfile(normalized_path):
            return normalized_path

    return None


# ==========================================
# 📊 5. 表格格式化
# ==========================================
def clean_embedding_table_text(
    embedding_text: str,
    table_title: str
) -> str:
    """
    对 embedding_text 做轻量清理。

    注意：
    当前完整表格主要存储在 embedding_text 中，
    因此这里只删除少量重复标题，不尝试破坏原始表格结构。
    """
    text = normalize_text(embedding_text)

    if not text:
        return ""

    title = normalize_text(table_title)

    if title and text.startswith(title):
        remaining_text = text[len(title):].strip()

        if remaining_text:
            text = remaining_text

    return text


def get_table_content(
    node: Dict[str, Any]
) -> Tuple[str, str]:
    """
    按优先级读取表格正文。

    返回：
    - 表格正文；
    - 正文来源字段。

    长期推荐在抽取阶段保存 table_markdown 或 table_csv。
    当前数据库主要使用 embedding_text 保存完整表格。
    """
    table_title = normalize_text(
        node.get("name")
        or node.get("summary")
        or node.get("description")
        or "未命名表格"
    )

    candidate_fields = [
        "table_markdown",
        "markdown",
        "table_csv",
        "csv",
        "table_content",
        "content",
        "embedding_text"
    ]

    for field_name in candidate_fields:
        field_value = normalize_text(
            node.get(field_name)
        )

        if not field_value:
            continue

        if field_name == "embedding_text":
            field_value = clean_embedding_table_text(
                field_value,
                table_title
            )

        if field_value:
            return field_value, field_name

    return "", ""


def format_table_node(
    node: Dict[str, Any]
) -> str:
    """
    将 Neo4j Table 节点转换为适合大模型阅读的文本。

    输出内容包括：
    - 表格名称；
    - 行列数；
    - 识别置信度；
    - 完整表格正文；
    - 表格脚注；
    - 数据语义提醒。
    """
    table_title = normalize_text(
        node.get("name")
        or node.get("summary")
        or node.get("description")
        or "未命名表格"
    )

    table_content, content_source = get_table_content(
        node
    )

    rows = node.get("rows")
    cols = node.get("cols")
    confidence = node.get("confidence")
    validation_status = normalize_text(
        node.get("validation_status")
    )
    footnotes = node.get("footnotes", [])

    output_parts = [
        f"**表格名称**：{table_title}"
    ]

    metadata_items = []

    if rows is not None:
        metadata_items.append(
            f"行数：{rows}"
        )

    if cols is not None:
        metadata_items.append(
            f"列数：{cols}"
        )

    if confidence is not None:
        metadata_items.append(
            f"识别置信度：{confidence}"
        )

    if validation_status:
        metadata_items.append(
            f"校验状态：{validation_status}"
        )

    if content_source:
        metadata_items.append(
            f"正文来源：{content_source}"
        )

    if metadata_items:
        output_parts.append(
            "**表格元数据**："
            + "；".join(metadata_items)
        )

    if table_content:
        output_parts.append(
            "**完整表格正文**：\n"
            f"{table_content}"
        )
    else:
        output_parts.append(
            "**完整表格正文**："
            "当前节点中没有找到可用的表格正文。"
        )

    normalized_footnotes = []

    if isinstance(footnotes, list):
        for footnote in footnotes:
            footnote_text = normalize_text(
                footnote
            )

            if footnote_text:
                normalized_footnotes.append(
                    footnote_text
                )

    elif footnotes:
        footnote_text = normalize_text(
            footnotes
        )

        if footnote_text:
            normalized_footnotes.append(
                footnote_text
            )

    if normalized_footnotes:
        footnote_output = "\n".join(
            f"- {footnote}"
            for footnote in normalized_footnotes
        )

        output_parts.append(
            "**表格脚注**：\n"
            f"{footnote_output}"
        )

    output_parts.append(
        "**数据语义提醒**："
        "Peak 表示峰值电流；"
        "RMS 表示有效值电流；"
        "Pulse/rev 表示电机旋转一圈所需的脉冲数，"
        "不能直接当作细分倍数。"
    )

    return "\n\n".join(output_parts)


# ==========================================
# 🧠 6. Embedding 客户端
# ==========================================
class EmbeddingClient:
    @staticmethod
    def get_embedding(
        text: str
    ) -> List[float]:
        """
        调用阿里云 DashScope Embedding 接口，
        将查询文本转换为向量。
        """
        query_text = normalize_text(text)

        if not query_text:
            raise ValueError(
                "Embedding 输入文本不能为空。"
            )

        if not EMBEDDING_API_KEY:
            raise ValueError(
                "未找到 Embedding API Key。"
                "请在 key.env 中配置 DASHSCOPE_API_KEY。"
            )

        headers = {
            "Authorization": (
                f"Bearer {EMBEDDING_API_KEY}"
            ),
            "Content-Type": "application/json"
        }

        payload = {
            "model": EMBEDDING_MODEL,
            "input": {
                "texts": [query_text]
            },
            "parameters": {
                "text_type": "query",
                "dimension": EMBEDDING_DIMENSION
            }
        }

        try:
            response = requests.post(
                EMBEDDING_URL,
                headers=headers,
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )

        except requests.RequestException as error:
            raise RuntimeError(
                "调用 Embedding 接口失败："
                f"{error}"
            ) from error

        if not response.ok:
            response_preview = response.text[:1000]

            raise RuntimeError(
                "Embedding 接口返回错误：\n"
                f"HTTP 状态码：{response.status_code}\n"
                f"响应内容：{response_preview}"
            )

        try:
            response_data = response.json()
        except ValueError as error:
            raise RuntimeError(
                "Embedding 接口返回的内容不是有效 JSON：\n"
                f"{response.text[:1000]}"
            ) from error

        try:
            embedding = (
                response_data["output"]
                ["embeddings"][0]
                ["embedding"]
            )
        except (
            KeyError,
            IndexError,
            TypeError
        ) as error:
            raise RuntimeError(
                "无法从 Embedding 响应中解析向量：\n"
                + json.dumps(
                    response_data,
                    ensure_ascii=False,
                    indent=2,
                    default=str
                )
            ) from error

        if not isinstance(embedding, list):
            raise RuntimeError(
                "Embedding 响应中的向量格式不正确。"
            )

        if len(embedding) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Embedding 向量维度与配置不一致："
                f"期望 {EMBEDDING_DIMENSION}，"
                f"实际 {len(embedding)}。"
                "请确认 Pinecone 索引维度和 Embedding 配置一致。"
            )

        return embedding


# ==========================================
# 🖼️ 7. 本地图片解析
# ==========================================
class ImageResolver:
    def __init__(self):
        self.doc_map: Dict[str, Dict[str, Any]] = {}

        documents_path = os.path.join(
            KG_DATA_DIR,
            "documents.json"
        )

        if not os.path.isfile(documents_path):
            print(
                "⚠️ 未找到 documents.json："
                f"{documents_path}"
            )
            return

        try:
            with open(
                documents_path,
                "r",
                encoding="utf-8-sig"
            ) as file:
                documents = json.load(file)

        except (
            OSError,
            json.JSONDecodeError
        ) as error:
            print(
                "⚠️ documents.json 读取失败："
                f"{error}"
            )
            return

        if not isinstance(documents, list):
            print(
                "⚠️ documents.json 顶层不是列表，"
                "无法建立文档映射。"
            )
            return

        for document in documents:
            if not isinstance(document, dict):
                continue

            document_id = document.get(
                "document_id"
            )

            if document_id:
                self.doc_map[str(document_id)] = (
                    document
                )

    @staticmethod
    def find_enhanced_image(
        original_path: str
    ) -> Optional[str]:
        """
        根据原图路径查找增强图。
        """
        normalized_original = os.path.normpath(
            original_path
        )

        file_root, file_extension = os.path.splitext(
            normalized_original
        )

        enhanced_candidates = [
            f"{file_root}_enhanced{file_extension}",
            f"{file_root}_enhance{file_extension}"
        ]

        for candidate_path in enhanced_candidates:
            if os.path.isfile(candidate_path):
                return candidate_path

        return None

    def resolve_figure_images(
        self,
        document_id: str,
        page_no: int
    ) -> List[Dict[str, Optional[str]]]:
        """
        根据文档编号和页码解析对应的本地图片。
        """
        document_info = self.doc_map.get(
            str(document_id),
            {}
        )

        file_name = normalize_text(
            document_info.get("file_name")
        )

        file_type = normalize_text(
            document_info.get("file_type")
        ).lower()

        resolved_images: List[
            Dict[str, Optional[str]]
        ] = []

        seen_original_paths = set()

        metadata = document_info.get(
            "metadata",
            {}
        )

        if (
            file_type == "pptx"
            and isinstance(metadata, dict)
            and isinstance(
                metadata.get("image_mapping"),
                dict
            )
        ):
            image_mapping = metadata[
                "image_mapping"
            ]

            mapped_paths = image_mapping.get(
                str(page_no),
                []
            )

            if not isinstance(mapped_paths, list):
                mapped_paths = []

            for mapped_path in mapped_paths:
                if not mapped_path:
                    continue

                original_path = os.path.normpath(
                    str(mapped_path)
                )

                if (
                    original_path
                    in seen_original_paths
                ):
                    continue

                seen_original_paths.add(
                    original_path
                )

                enhanced_path = (
                    self.find_enhanced_image(
                        original_path
                    )
                )

                resolved_images.append({
                    "original": (
                        original_path
                        if os.path.isfile(
                            original_path
                        )
                        else None
                    ),
                    "enhanced": enhanced_path
                })

            if resolved_images:
                return resolved_images

        if not file_name:
            return resolved_images

        search_patterns = [
            os.path.join(
                DEBUG_IMAGES_DIR,
                f"{file_name}*p{page_no}_img*.*"
            ),
            os.path.join(
                DEBUG_IMAGES_DIR,
                f"{file_name}*slide_{page_no}_img*.*"
            )
        ]

        all_matches = []

        for pattern in search_patterns:
            all_matches.extend(
                glob.glob(pattern)
            )

        for matched_path in sorted(set(all_matches)):
            if not os.path.isfile(matched_path):
                continue

            lower_path = matched_path.lower()

            if (
                "enhance" in lower_path
                or lower_path.endswith(".json")
            ):
                continue

            original_path = os.path.normpath(
                matched_path
            )

            if original_path in seen_original_paths:
                continue

            seen_original_paths.add(
                original_path
            )

            enhanced_path = (
                self.find_enhanced_image(
                    original_path
                )
            )

            resolved_images.append({
                "original": original_path,
                "enhanced": enhanced_path
            })

        return resolved_images


# ==========================================
# 🕸️ 8. Neo4j 图谱检索
# ==========================================
class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(
                NEO4J_USER,
                NEO4J_PASSWORD
            )
        )

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def get_page_context(
        self,
        page_id: str,
        expand_depth: int
    ) -> Dict[str, Any]:
        """
        获取指定页面的：
        - 页面信息；
        - 多模态节点；
        - 知识图谱概念；
        - 知识图谱关系。
        """
        context = {
            "page_info": {},
            "multimodal_materials": [],
            "knowledge_graph": {
                "concepts": [],
                "relations": []
            }
        }

        with self.driver.session() as session:
            page_result = session.run(
                """
                MATCH (p:Page {id: $page_id})
                RETURN p
                """,
                page_id=page_id
            ).single()

            if page_result:
                context["page_info"] = dict(
                    page_result["p"].items()
                )

            material_result = session.run(
                """
                MATCH (m)
                WHERE m.source_page_id = $page_id
                  AND NOT 'Page' IN labels(m)
                  AND NOT 'Document' IN labels(m)
                RETURN labels(m)[0] AS node_type, m
                """,
                page_id=page_id
            )

            for record in material_result:
                node_data = dict(
                    record["m"].items()
                )

                node_data["_node_type"] = (
                    record["node_type"]
                )

                context[
                    "multimodal_materials"
                ].append(node_data)

            if expand_depth <= 0:
                return context

            safe_expand_depth = max(
                1,
                min(int(expand_depth), 5)
            )

            path_depth = safe_expand_depth - 1

            graph_query = f"""
                MATCH (m)
                WHERE m.source_page_id = $page_id
                   OR m.id = $page_id
                MATCH (m)<-[:HAS_EVIDENCE]-(start_c:Concept)
                OPTIONAL MATCH path =
                    (start_c)-[*0..{path_depth}]-(related_c:Concept)
                RETURN start_c, path
            """

            graph_result = session.run(
                graph_query,
                page_id=page_id
            )

            concepts_by_name: Dict[
                str,
                Dict[str, Any]
            ] = {}

            relations: List[
                Dict[str, Any]
            ] = []

            seen_relation_ids = set()

            for record in graph_result:
                start_concept = record[
                    "start_c"
                ]

                if start_concept:
                    start_name = normalize_text(
                        start_concept.get("name")
                    )

                    if (
                        start_name
                        and start_name
                        not in concepts_by_name
                    ):
                        concepts_by_name[
                            start_name
                        ] = dict(
                            start_concept.items()
                        )

                path = record["path"]

                if not path:
                    continue

                for node in path.nodes:
                    if "Concept" not in node.labels:
                        continue

                    concept_name = normalize_text(
                        node.get("name")
                    )

                    if (
                        concept_name
                        and concept_name
                        not in concepts_by_name
                    ):
                        concepts_by_name[
                            concept_name
                        ] = dict(
                            node.items()
                        )

                for relation in path.relationships:
                    relation_id = (
                        relation.element_id
                    )

                    if (
                        relation_id
                        in seen_relation_ids
                    ):
                        continue

                    seen_relation_ids.add(
                        relation_id
                    )

                    source_name = normalize_text(
                        relation.start_node.get(
                            "name"
                        )
                    )

                    target_name = normalize_text(
                        relation.end_node.get(
                            "name"
                        )
                    )

                    if not source_name:
                        source_name = normalize_text(
                            relation.start_node.get(
                                "id"
                            )
                        )

                    if not target_name:
                        target_name = normalize_text(
                            relation.end_node.get(
                                "id"
                            )
                        )

                    relations.append({
                        "source": source_name,
                        "target": target_name,
                        "relation_type": (
                            relation.type
                        ),
                        "description": normalize_text(
                            relation.get(
                                "description",
                                ""
                            )
                        )
                    })

            context["knowledge_graph"][
                "concepts"
            ] = list(
                concepts_by_name.values()
            )

            context["knowledge_graph"][
                "relations"
            ] = relations

        return context


# ==========================================
# 🔍 9. Pinecone 检索结果处理
# ==========================================
def extract_page_ids_from_matches(
    matches: Any,
    page_top_k: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    从 Pinecone 检索结果中按原始相关性顺序提取页面编号。

    同时保留每个页面第一次命中的分数和元数据。
    """
    page_ids: List[str] = []
    page_hits: List[Dict[str, Any]] = []
    seen_page_ids = set()

    for match in matches or []:
        metadata = get_match_value(
            match,
            "metadata",
            {}
        ) or {}

        if not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except (TypeError, ValueError):
                metadata = {}

        page_id = metadata.get("page_id")

        if not page_id:
            node_type = metadata.get(
                "node_type"
            )

            if node_type == "Page":
                page_id = metadata.get(
                    "node_id"
                )

        if not page_id:
            continue

        page_id = str(page_id)

        if page_id in seen_page_ids:
            continue

        seen_page_ids.add(page_id)
        page_ids.append(page_id)

        page_hits.append({
            "page_id": page_id,
            "score": get_match_value(
                match,
                "score",
                None
            ),
            "matched_node_id": (
                metadata.get("node_id")
            ),
            "matched_node_type": (
                metadata.get("node_type")
            )
        })

        if len(page_ids) >= page_top_k:
            break

    return page_ids, page_hits


# ==========================================
# 🚀 10. 检索素材主流程
# ==========================================
def retrieve_materials(
    query: str,
    vector_top_k: int = 10,
    page_top_k: int = 3,
    concept_expand_depth: int = 2
) -> Dict[str, Any]:
    """
    执行完整检索流程：

    1. 查询文本向量化；
    2. Pinecone 向量检索；
    3. 提取命中页面；
    4. Neo4j 获取页面素材和知识图谱；
    5. 解析对应本地图片；
    6. 返回统一素材包。
    """
    validate_runtime_config()

    query_text = normalize_text(query)

    if not query_text:
        raise ValueError(
            "查询内容不能为空。"
        )

    if vector_top_k <= 0:
        raise ValueError(
            "vector_top_k 必须大于 0。"
        )

    if page_top_k <= 0:
        raise ValueError(
            "page_top_k 必须大于 0。"
        )

    if concept_expand_depth < 0:
        raise ValueError(
            "concept_expand_depth 不能小于 0。"
        )

    print(
        f"🔍 正在为查询生成素材包：'{query_text}'"
    )

    query_vector = (
        EmbeddingClient.get_embedding(
            query_text
        )
    )

    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY
    )

    pinecone_index = pinecone_client.Index(
        PINECONE_INDEX_NAME
    )

    query_parameters = {
        "vector": query_vector,
        "top_k": vector_top_k,
        "include_metadata": True
    }

    if PINECONE_NAMESPACE:
        query_parameters["namespace"] = (
            PINECONE_NAMESPACE
        )

    search_result = pinecone_index.query(
        **query_parameters
    )

    if isinstance(search_result, dict):
        matches = search_result.get(
            "matches",
            []
        )
    else:
        matches = getattr(
            search_result,
            "matches",
            []
        )

    page_ids, page_hits = (
        extract_page_ids_from_matches(
            matches=matches,
            page_top_k=page_top_k
        )
    )

    final_package: Dict[str, Any] = {
        "query": query_text,
        "parameters": {
            "vector_top_k": vector_top_k,
            "page_top_k": page_top_k,
            "concept_expand_depth": (
                concept_expand_depth
            ),
            "embedding_model": (
                EMBEDDING_MODEL
            ),
            "embedding_dimension": (
                EMBEDDING_DIMENSION
            ),
            "pinecone_index": (
                PINECONE_INDEX_NAME
            ),
            "pinecone_namespace": (
                PINECONE_NAMESPACE
            )
        },
        "retrieval_summary": {
            "vector_match_count": len(
                matches or []
            ),
            "selected_page_count": len(
                page_ids
            ),
            "selected_page_ids": page_ids,
            "page_hits": page_hits
        },
        "pages": []
    }

    if not page_ids:
        print(
            "⚠️ Pinecone 没有返回可用的页面编号。"
        )
        return final_package

    graph_client = GraphRetriever()
    image_resolver = ImageResolver()

    page_hit_map = {
        hit["page_id"]: hit
        for hit in page_hits
    }

    try:
        for page_id in page_ids:
            page_context = (
                graph_client.get_page_context(
                    page_id=page_id,
                    expand_depth=(
                        concept_expand_depth
                    )
                )
            )

            page_info = page_context.get(
                "page_info",
                {}
            )

            document_id = page_info.get(
                "document_id"
            )

            page_no = page_info.get(
                "page_no"
            )

            resolved_images = []

            if (
                document_id is not None
                and page_no is not None
            ):
                resolved_images = (
                    image_resolver
                    .resolve_figure_images(
                        document_id=str(
                            document_id
                        ),
                        page_no=int(page_no)
                    )
                )

            page_package = {
                "page_id": page_id,
                "retrieval_hit": (
                    page_hit_map.get(
                        page_id,
                        {}
                    )
                ),
                "document_id": document_id,
                "page_no": page_no,
                "summary": normalize_text(
                    page_info.get("summary")
                ),
                "text_content": normalize_text(
                    page_info.get(
                        "description"
                    )
                ),
                "resolved_images": (
                    resolved_images
                ),
                "multimodal_nodes": (
                    page_context.get(
                        "multimodal_materials",
                        []
                    )
                ),
                "knowledge_graph": (
                    page_context.get(
                        "knowledge_graph",
                        {
                            "concepts": [],
                            "relations": []
                        }
                    )
                )
            }

            final_package["pages"].append(
                page_package
            )

    finally:
        graph_client.close()

    return final_package


# ==========================================
# 🧱 11. 知识图谱文本组装
# ==========================================
def build_knowledge_graph_text(
    pages: List[Dict[str, Any]]
) -> str:
    """
    提取并去重知识图谱中的概念和关系。
    """
    concepts_map: Dict[str, str] = {}
    relations_set = set()

    for page in pages:
        knowledge_graph = page.get(
            "knowledge_graph",
            {}
        )

        concepts = knowledge_graph.get(
            "concepts",
            []
        )

        relations = knowledge_graph.get(
            "relations",
            []
        )

        for concept in concepts:
            concept_name = normalize_text(
                concept.get("name")
            )

            concept_description = normalize_text(
                concept.get("description")
            )

            if (
                concept_name
                and concept_description
                and concept_name
                not in concepts_map
            ):
                concepts_map[
                    concept_name
                ] = concept_description

        for relation in relations:
            source = normalize_text(
                relation.get("source")
            )

            target = normalize_text(
                relation.get("target")
            )

            relation_type = normalize_text(
                relation.get(
                    "relation_type"
                )
            ) or "RELATED_TO"

            description = normalize_text(
                relation.get("description")
            )

            if not source or not target:
                continue

            relation_text = (
                f"- **{source}** ➔ "
                f"**{target}** "
                f"({relation_type})"
            )

            if description:
                relation_text += (
                    f"：{description}"
                )

            relations_set.add(
                relation_text
            )

    output_parts = [
        "### 1. 核心概念定义"
    ]

    if concepts_map:
        for concept_name in sorted(
            concepts_map.keys()
        ):
            output_parts.append(
                f"- **{concept_name}**："
                f"{concepts_map[concept_name]}"
            )
    else:
        output_parts.append(
            "无核心概念定义。"
        )

    output_parts.extend([
        "",
        "### 2. 概念间逻辑关系"
    ])

    if relations_set:
        output_parts.extend(
            sorted(relations_set)
        )
    else:
        output_parts.append(
            "无相关概念逻辑。"
        )

    return "\n".join(output_parts)


# ==========================================
# 📚 12. 多模态素材文本组装
# ==========================================
def build_materials_text(
    pages: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    将文本、表格、代码、图片说明组装为 Markdown，
    并返回需要上传给视觉模型的图片路径。

    表格节点会通过 format_table_node() 输出完整正文。
    """
    materials_parts: List[str] = []
    image_paths: List[str] = []

    seen_image_paths = set()
    next_image_number = 1

    for page_index, page in enumerate(
        pages,
        start=1
    ):
        document_id = page.get(
            "document_id"
        ) or "未知文档"

        page_no = page.get(
            "page_no"
        )

        if page_no is None:
            page_no = "未知页码"

        retrieval_hit = page.get(
            "retrieval_hit",
            {}
        )

        retrieval_score = retrieval_hit.get(
            "score"
        )

        materials_parts.append(
            f"### 参考页面 {page_index}"
        )

        materials_parts.append(
            f"- 文档编号：{document_id}"
        )

        materials_parts.append(
            f"- 页码：{page_no}"
        )

        if retrieval_score is not None:
            materials_parts.append(
                f"- 向量命中分数："
                f"{retrieval_score}"
            )

        materials_parts.append("")

        page_nodes = page.get(
            "multimodal_nodes",
            []
        )

        # 先解析该页面真正存在的图片，
        # 为当前页面的 Figure 节点建立编号。
        current_page_images = []

        for image_info in page.get(
            "resolved_images",
            []
        ):
            selected_image = (
                choose_existing_image(
                    image_info
                )
            )

            if not selected_image:
                continue

            if selected_image in seen_image_paths:
                continue

            seen_image_paths.add(
                selected_image
            )

            image_paths.append(
                selected_image
            )

            current_page_images.append({
                "number": next_image_number,
                "path": selected_image
            })

            next_image_number += 1

        page_figure_index = 0

        for node in page_nodes:
            node_type = normalize_text(
                node.get("_node_type")
                or node.get("node_type")
                or "Unknown"
            )

            if node_type == "TextChunk":
                text_content = normalize_text(
                    node.get("description")
                    or node.get("content")
                    or node.get(
                        "embedding_text"
                    )
                )

                if text_content:
                    materials_parts.append(
                        "**[文本]**"
                    )

                    materials_parts.append(
                        text_content
                    )

                    materials_parts.append("")

            elif node_type == "Table":
                table_text = format_table_node(
                    node
                )

                materials_parts.append(
                    "**[表格数据]**"
                )

                materials_parts.append(
                    table_text
                )

                materials_parts.append("")

            elif node_type == "CodeBlock":
                code_content = normalize_text(
                    node.get("content")
                    or node.get("description")
                    or node.get(
                        "embedding_text"
                    )
                )

                if not code_content:
                    continue

                language = normalize_text(
                    node.get("language")
                    or node.get(
                        "programming_language"
                    )
                    or "text"
                )

                materials_parts.append(
                    f"**[代码片段，语言："
                    f"{language}]**"
                )

                materials_parts.append(
                    code_content
                )

                materials_parts.append("")

            elif node_type == "Figure":
                figure_description = normalize_text(
                    node.get("description")
                    or node.get("summary")
                )

                if not figure_description:
                    continue

                if (
                    page_figure_index
                    < len(current_page_images)
                ):
                    image_number = (
                        current_page_images[
                            page_figure_index
                        ]["number"]
                    )

                    materials_parts.append(
                        f"**[图片说明，对应输入图片 "
                        f"{image_number}]**"
                    )
                else:
                    materials_parts.append(
                        "**[图片说明，未解析到对应"
                        "本地图片文件]**"
                    )

                materials_parts.append(
                    figure_description
                )

                materials_parts.append("")

                page_figure_index += 1

        # 如果页面有图片，但没有对应 Figure 节点，
        # 仍然明确列出图片编号。
        if (
            current_page_images
            and page_figure_index == 0
        ):
            for image_item in (
                current_page_images
            ):
                materials_parts.append(
                    f"**[输入图片 "
                    f"{image_item['number']}]**："
                    "该图片来自当前参考页面，"
                    "数据库中没有对应的 Figure 描述。"
                )

                materials_parts.append("")

    materials_text = "\n".join(
        materials_parts
    ).strip()

    return materials_text, image_paths


# ==========================================
# 📝 13. 最终 Prompt 组装
# ==========================================
def build_vlm_prompt(
    retrieved_data: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """
    将 retrieve_materials() 返回的素材包转换为：

    1. 适合多模态大模型阅读的文本 Prompt；
    2. 需要同时发送给模型的本地图片路径列表。
    """
    query = normalize_text(
        retrieved_data.get("query")
    )

    pages = retrieved_data.get(
        "pages",
        []
    )

    if not isinstance(pages, list):
        pages = []

    knowledge_graph_text = (
        build_knowledge_graph_text(
            pages
        )
    )

    materials_text, image_paths = (
        build_materials_text(
            pages
        )
    )

    final_prompt = f"""你是一个专业的机电系统教学助手。请严格根据下面提供的知识图谱、课件文本、完整表格和图片回答用户问题。

回答时必须遵守以下规则：

1. 优先依据参考课件中的明确内容，不得把推测写成确定事实。
2. 完整读取表格正文和表格脚注，不得只根据表格标题回答。
3. Peak 表示峰值电流，RMS 表示有效值电流，二者不得交换。
4. Pulse/rev 表示电机旋转一圈所需的脉冲数，不等同于细分倍数。
5. 如果需要计算每个脉冲对应的角度，应使用一圈角度除以 Pulse/rev。
6. 区分“步进电机驱动器原理样机”和“商品化步进电机驱动器”，不得将不同架构强行合并。
7. 不得将上位机、下位机和商品化驱动器直接等同。
8. 素材没有明确说明时，不得断言 ENA 的有效电平和失能后的电机状态。
9. 图片、表格、课件文本和知识图谱存在冲突时，应明确指出冲突，不得自行猜测。
10. 如果补充课件以外的通用知识，必须明确标注为“一般工程知识”。

## 知识图谱支持

{knowledge_graph_text}

---

## 参考课件素材

{materials_text if materials_text else "本次检索没有找到可用的参考课件素材。"}

---

## 用户问题

**{query}**

请综合上述信息给出准确、清晰的回答："""

    return final_prompt, image_paths


# ==========================================
# 🧪 14. 本地测试入口
# ==========================================
if __name__ == "__main__":
    TEST_QUERY = (
        "步进电机驱动器的工作原理是什么？"
    )

    retrieved_materials = retrieve_materials(
        query=TEST_QUERY,
        vector_top_k=10,
        page_top_k=2,
        concept_expand_depth=2
    )

    prompt_text, resolved_image_paths = (
        build_vlm_prompt(
            retrieved_materials
        )
    )

    print("\n" + "=" * 80)
    print("📦 检索返回的原始素材包")
    print("=" * 80)

    print(
        json.dumps(
            retrieved_materials,
            ensure_ascii=False,
            indent=2,
            default=str
        )
    )

    print("\n" + "=" * 80)
    print("🤖 最终发送给大模型的 Prompt")
    print("=" * 80)

    print(prompt_text)

    print("\n" + "=" * 80)
    print("🖼️ 需要随 Prompt 一起发送的图片")
    print("=" * 80)

    if not resolved_image_paths:
        print("本次检索没有解析到可用图片。")
    else:
        for image_index, image_path in enumerate(
            resolved_image_paths,
            start=1
        ):
            print(
                f"图片 {image_index}："
                f"{image_path}"
            )