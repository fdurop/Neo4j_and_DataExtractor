# 修复bug


import os
import json
import yaml
import glob
import requests
from typing import List, Dict, Any
from pinecone import Pinecone
from neo4j import GraphDatabase


# ==========================================
# ⚙️ 1. 配置加载与全局变量
# ==========================================
def load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

# 路径配置
KG_DATA_DIR = r"C:\Users\Lin\extraction\output\qwen_qwen3.7-plus_6\kg_data"
DEBUG_IMAGES_DIR = r"C:\Users\Lin\extraction\output\qwen_qwen3.7-plus_6\debug\images"


# 读取 YAML 配置
def load_yaml_config():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_api.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"未找到配置文件: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_yaml_config()
EMBEDDING_API_KEY = config['embedding']['api_key'] or config['vlm']['api_key']
EMBEDDING_URL = f"{config['embedding']['base_url']}/services/embeddings/text-embedding/text-embedding"

# 数据库凭证
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = "multimodal-rag-index"
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://fced1671.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "fced1671")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


# ==========================================
# 🧠 2. 核心功能客户端
# ==========================================
class EmbeddingClient:
    """调用 DashScope 获取 1024 维文本向量"""

    @staticmethod
    def get_embedding(text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config['embedding']['model'],
            "input": {"texts": [text]},
            "parameters": {
                "text_type": "query",
                "dimension": config['embedding']['dimension']
            }
        }
        response = requests.post(EMBEDDING_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['output']['embeddings'][0]['embedding']


class ImageResolver:
    """负责将节点映射到本地真实的图片文件"""

    def __init__(self):
        self.doc_map = {}
        docs_path = os.path.join(KG_DATA_DIR, "documents.json")
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for d in docs:
                    self.doc_map[d['document_id']] = d

    def resolve_figure_images(self, document_id: str, page_no: int) -> List[Dict[str, str]]:
        """根据文档ID和页码，找出该页所有的图片及其增强版"""
        doc_info = self.doc_map.get(document_id, {})
        file_name = doc_info.get('file_name', '')
        file_type = doc_info.get('file_type', '')

        resolved_images = []

        # 如果是 PPTX 且 JSON 中有 image_mapping，直接精准读取
        if file_type == 'pptx' and 'metadata' in doc_info and 'image_mapping' in doc_info['metadata']:
            mapping = doc_info['metadata']['image_mapping']
            img_paths = mapping.get(str(page_no), [])
            for path in img_paths:
                base_path = os.path.normpath(path)
                # 检查是否有增强版
                enhance_path = base_path.replace(".png", "_enhanced.png").replace(".jpeg", "_enhanced.jpeg")
                if not os.path.exists(enhance_path):
                    enhance_path = base_path.replace(".png", "_enhance.png")  # 兼容不同命名习惯

                resolved_images.append({
                    "original": base_path if os.path.exists(base_path) else None,
                    "enhanced": enhance_path if os.path.exists(enhance_path) else None
                })
            return resolved_images

        # 如果是 PDF 或没有 mapping，使用 glob 模糊匹配
        if file_name:
            # 匹配模式：课件名_p页码_img*.* 或 课件名_slide_页码_img*.*
            search_pattern1 = os.path.join(DEBUG_IMAGES_DIR, f"{file_name}*p{page_no}_img*.*")
            search_pattern2 = os.path.join(DEBUG_IMAGES_DIR, f"{file_name}*slide_{page_no}_img*.*")

            all_matches = glob.glob(search_pattern1) + glob.glob(search_pattern2)
            # 过滤掉已经是 enhanced 的文件，避免重复计算
            base_matches = [m for m in all_matches if "enhance" not in m.lower() and not m.endswith(".json")]

            for base_path in base_matches:
                name, ext = os.path.splitext(base_path)
                enhance_path = f"{name}_enhanced{ext}"
                if not os.path.exists(enhance_path):
                    enhance_path = f"{name}_enhance{ext}"

                resolved_images.append({
                    "original": base_path,
                    "enhanced": enhance_path if os.path.exists(enhance_path) else None
                })

        return resolved_images


class GraphRetriever:
    """Neo4j 图谱查询客户端"""

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_page_context(self, page_id: str, expand_depth: int) -> Dict[str, Any]:
        """获取指定 Page 的完整上下文（包含素材和 N 层知识点）"""
        context = {
            "page_info": {},
            "multimodal_materials": [],
            "knowledge_graph": {"concepts": {}, "relations": []}
        }

        with self.driver.session() as session:
            # 1. 获取 Page 基本信息
            page_res = session.run("MATCH (p:Page {id: $pid}) RETURN p", pid=page_id).single()
            if page_res:
                context["page_info"] = dict(page_res["p"].items())

            # 2. 获取该 Page 下辖的所有多模态素材节点 (修复：使用 source_page_id)
            mat_res = session.run("""
                MATCH (m) 
                WHERE m.source_page_id = $pid AND NOT 'Page' IN labels(m) AND NOT 'Document' IN labels(m)
                RETURN labels(m)[0] AS type, m
            """, pid=page_id)

            for record in mat_res:
                node_data = dict(record["m"].items())
                node_data["_node_type"] = record["type"]
                context["multimodal_materials"].append(node_data)

            # 3. 拓展知识点（N 层 BFS）
            if expand_depth > 0:
                # 修复：知识点可能挂在素材节点(source_page_id)上，也可能直接挂在Page节点(id)上
                kg_query = f"""
                    MATCH (m)
                    WHERE m.source_page_id = $pid OR m.id = $pid
                    MATCH (m)<-[:HAS_EVIDENCE]-(start_c:Concept)
                    OPTIONAL MATCH path = (start_c)-[*0..{expand_depth - 1}]-(related_c:Concept)
                    RETURN start_c, path
                """
                kg_res = session.run(kg_query, pid=page_id)

                concepts_dict = {}
                relations_list = []
                seen_rels = set()

                for record in kg_res:
                    # 记录起点 Concept
                    sc = record["start_c"]
                    if sc and sc["name"] not in concepts_dict:
                        concepts_dict[sc["name"]] = dict(sc.items())

                    # 解析路径中的节点和关系
                    path = record["path"]
                    if path:
                        for node in path.nodes:
                            if "Concept" in node.labels and node["name"] not in concepts_dict:
                                concepts_dict[node["name"]] = dict(node.items())

                        for rel in path.relationships:
                            rel_id = rel.element_id
                            if rel_id not in seen_rels:
                                seen_rels.add(rel_id)
                                relations_list.append({
                                    "source": rel.start_node["name"],
                                    "target": rel.end_node["name"],
                                    "relation_type": rel.type,
                                    "description": rel.get("description", "")
                                })

                context["knowledge_graph"]["concepts"] = list(concepts_dict.values())
                context["knowledge_graph"]["relations"] = relations_list

        return context


# ==========================================
# 🚀 3. 顶层流水线组装
# ==========================================
def retrieve_materials(query: str, vector_top_k: int = 10, page_top_k: int = 3, concept_expand_depth: int = 2) -> Dict[str, Any]:
    """
    检索并打包所有相关素材（不调用 VLM 生成答案）

    :param query: 用户查询的问题
    :param vector_top_k: Pinecone 向量检索召回的初始候选节点数（扩大候选集）
    :param page_top_k: 最终保留的独立页面数量（按页去重截断）
    :param concept_expand_depth: 知识点图谱向外拓展的层数 (1表示只查直接关联的知识点，2表示再拓展一层关系)
    :return: 结构化的素材包字典
    """
    print(f"🔍 正在为查询生成素材包: '{query}'")

    # 1. 向量化 Query
    print("  [1/4] 正在调用 DashScope 获取 Query 向量...")
    query_vector = EmbeddingClient.get_embedding(query)

    # 2. Pinecone 检索 (扩大候选集)
    print(f"  [2/4] 正在 Pinecone 中检索 Top-{vector_top_k} 候选向量节点...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    search_res = index.query(vector=query_vector, top_k=vector_top_k, include_metadata=True)

    # 提取命中的 page_id (按页去重截断)
    hit_page_ids = set()
    for match in search_res['matches']:
        meta = match.get('metadata', {})
        # 如果命中本身就是 Page，直接取 node_id；如果是素材，取其 page_id
        pid = meta.get('page_id') or (meta.get('node_id') if meta.get('node_type') == 'Page' else None)
        if pid:
            hit_page_ids.add(pid)
            # 达到期望的独立页面数量后提前结束
            if len(hit_page_ids) >= page_top_k:
                break

    print(f"  -> 从 {len(search_res['matches'])} 个候选向量中，成功截取 {len(hit_page_ids)} 个独立页面 (Pages)")

    # 3. Neo4j 图谱反查与拓展
    print(f"  [3/4] 正在 Neo4j 中反查多模态素材并拓展 {concept_expand_depth} 层知识点...")
    graph_client = GraphRetriever()
    image_resolver = ImageResolver()

    final_package = {
        "query": query,
        "parameters": {
            "vector_top_k": vector_top_k,
            "page_top_k": page_top_k,
            "concept_expand_depth": concept_expand_depth
        },
        "pages": []
    }

    for pid in hit_page_ids:
        # 获取图谱上下文
        context = graph_client.get_page_context(pid, expand_depth=concept_expand_depth)
        page_info = context["page_info"]

        # 4. 解析图片物理路径
        resolved_images = []
        if "document_id" in page_info and "page_no" in page_info:
            resolved_images = image_resolver.resolve_figure_images(
                document_id=page_info["document_id"],
                page_no=page_info["page_no"]
            )

        # 组装单页数据
        page_package = {
            "page_id": pid,
            "document_id": page_info.get("document_id"),
            "page_no": page_info.get("page_no"),
            "summary": page_info.get("summary", ""),
            "text_content": page_info.get("description", ""),
            "resolved_images": resolved_images,
            "multimodal_nodes": context["multimodal_materials"],
            "knowledge_graph": context["knowledge_graph"]
        }
        final_package["pages"].append(page_package)

    graph_client.close()
    print("  [4/4] ✅ 素材包组装完成！")

    return final_package


# ==========================================
# 🧪 4. 本地测试入口
# ==========================================
if __name__ == "__main__":
    # 测试查询
    test_query = "步进电机驱动器的工作原理是什么？"

    # 调用检索函数 (候选向量取 10，最终保留 2 个页面)
    materials = retrieve_materials(
        query=test_query,
        vector_top_k=10,
        page_top_k=2,
        concept_expand_depth=2
    )

    # 打印结果预览
    print("\n" + "=" * 50)
    print("📦 最终素材包结构预览:")
    print("=" * 50)
    print(json.dumps(materials, ensure_ascii=False, indent=2))

    # 将完整结果保存到本地以便检查
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieved_materials_debug.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(materials, f, ensure_ascii=False, indent=2)
    print(f"\n💾 完整素材包已保存至: {output_path}")