# 修复一些bug
# 优化传回的形式


import os
import json
import yaml
import glob
import requests
from typing import List, Dict, Any, Tuple
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

KG_DATA_DIR = r"C:\Users\Lin\extraction\output\qwen_qwen3.7-plus_6\kg_data"
DEBUG_IMAGES_DIR = r"C:\Users\Lin\extraction\output\qwen_qwen3.7-plus_6\debug\images"


def load_yaml_config():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_api.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"未找到配置文件: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_yaml_config()
EMBEDDING_API_KEY = config['embedding']['api_key'] or config['vlm']['api_key']
EMBEDDING_URL = f"{config['embedding']['base_url']}/services/embeddings/text-embedding/text-embedding"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = "multimodal-rag-index"
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://fced1671.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "fced1671")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


# ==========================================
# 🧠 2. 核心功能客户端
# ==========================================
class EmbeddingClient:
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
    def __init__(self):
        self.doc_map = {}
        docs_path = os.path.join(KG_DATA_DIR, "documents.json")
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for d in docs:
                    self.doc_map[d['document_id']] = d

    def resolve_figure_images(self, document_id: str, page_no: int) -> List[Dict[str, str]]:
        doc_info = self.doc_map.get(document_id, {})
        file_name = doc_info.get('file_name', '')
        file_type = doc_info.get('file_type', '')

        resolved_images = []

        if file_type == 'pptx' and 'metadata' in doc_info and 'image_mapping' in doc_info['metadata']:
            mapping = doc_info['metadata']['image_mapping']
            img_paths = mapping.get(str(page_no), [])
            for path in img_paths:
                base_path = os.path.normpath(path)
                enhance_path = base_path.replace(".png", "_enhanced.png").replace(".jpeg", "_enhanced.jpeg")
                if not os.path.exists(enhance_path):
                    enhance_path = base_path.replace(".png", "_enhance.png")

                resolved_images.append({
                    "original": base_path if os.path.exists(base_path) else None,
                    "enhanced": enhance_path if os.path.exists(enhance_path) else None
                })
            return resolved_images

        if file_name:
            search_pattern1 = os.path.join(DEBUG_IMAGES_DIR, f"{file_name}*p{page_no}_img*.*")
            search_pattern2 = os.path.join(DEBUG_IMAGES_DIR, f"{file_name}*slide_{page_no}_img*.*")

            all_matches = glob.glob(search_pattern1) + glob.glob(search_pattern2)
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
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_page_context(self, page_id: str, expand_depth: int) -> Dict[str, Any]:
        context = {
            "page_info": {},
            "multimodal_materials": [],
            "knowledge_graph": {"concepts": {}, "relations": []}
        }

        with self.driver.session() as session:
            page_res = session.run("MATCH (p:Page {id: $pid}) RETURN p", pid=page_id).single()
            if page_res:
                context["page_info"] = dict(page_res["p"].items())

            mat_res = session.run("""
                MATCH (m) 
                WHERE m.source_page_id = $pid AND NOT 'Page' IN labels(m) AND NOT 'Document' IN labels(m)
                RETURN labels(m)[0] AS type, m
            """, pid=page_id)

            for record in mat_res:
                node_data = dict(record["m"].items())
                node_data["_node_type"] = record["type"]
                context["multimodal_materials"].append(node_data)

            if expand_depth > 0:
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
                    sc = record["start_c"]
                    if sc and sc["name"] not in concepts_dict:
                        concepts_dict[sc["name"]] = dict(sc.items())

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
# 🚀 3. 检索与 Prompt 组装流水线
# ==========================================
def retrieve_materials(query: str, vector_top_k: int = 10, page_top_k: int = 3, concept_expand_depth: int = 2) -> Dict[
    str, Any]:
    print(f"🔍 正在为查询生成素材包: '{query}'")
    query_vector = EmbeddingClient.get_embedding(query)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    search_res = index.query(vector=query_vector, top_k=vector_top_k, include_metadata=True)

    hit_page_ids = set()
    for match in search_res['matches']:
        meta = match.get('metadata', {})
        pid = meta.get('page_id') or (meta.get('node_id') if meta.get('node_type') == 'Page' else None)
        if pid:
            hit_page_ids.add(pid)
            if len(hit_page_ids) >= page_top_k:
                break

    graph_client = GraphRetriever()
    image_resolver = ImageResolver()

    final_package = {
        "query": query,
        "parameters": {"vector_top_k": vector_top_k, "page_top_k": page_top_k,
                       "concept_expand_depth": concept_expand_depth},
        "pages": []
    }

    for pid in hit_page_ids:
        context = graph_client.get_page_context(pid, expand_depth=concept_expand_depth)
        page_info = context["page_info"]

        resolved_images = []
        if "document_id" in page_info and "page_no" in page_info:
            resolved_images = image_resolver.resolve_figure_images(
                document_id=page_info["document_id"],
                page_no=page_info["page_no"]
            )

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
    return final_package


def build_vlm_prompt(retrieved_data: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    将检索到的 JSON 素材包“拍扁”，转换为大模型易于理解的 Markdown 纯文本。
    包含详细的节点定义、关系解释和多模态素材。
    """
    query = retrieved_data["query"]
    pages = retrieved_data["pages"]

    # 1. 提取去重的知识点定义与关系
    concepts_map = {}
    relations_set = set()

    for page in pages:
        # 收集节点定义
        for concept in page["knowledge_graph"]["concepts"]:
            name = concept["name"]
            desc = concept.get("description", "").strip()
            if name not in concepts_map and desc:
                concepts_map[name] = desc

        # 收集关系解释
        for rel in page["knowledge_graph"]["relations"]:
            source = rel['source']
            target = rel['target']
            rel_type = rel['relation_type']
            desc = rel.get('description', '').strip()

            # 格式化关系描述
            relation_str = f"- **{source}** ➔ **{target}** ({rel_type}): {desc}"
            relations_set.add(relation_str)

    # 组装图谱文本
    kg_text = "### 1. 核心概念定义\n"
    if concepts_map:
        for name, desc in concepts_map.items():
            kg_text += f"- **{name}**: {desc}\n"
    else:
        kg_text += "无核心概念定义。\n"

    kg_text += "\n### 2. 概念间逻辑关系\n"
    if relations_set:
        kg_text += "\n".join(relations_set) + "\n"
    else:
        kg_text += "无相关概念逻辑。\n"

    # 2. 提取多模态素材文本与图片路径
    materials_text = ""
    image_paths = []
    img_counter = 1

    for i, page in enumerate(pages):
        materials_text += f"### 📄 参考页面 {i + 1} (文档: {page['document_id']}, 第 {page['page_no']} 页)\n"

        # 提取该页的文本块、表格、代码等
        for node in page["multimodal_nodes"]:
            node_type = node.get("_node_type", "Unknown")
            desc = node.get("description", "").strip()
            if not desc:
                continue

            if node_type == "TextChunk":
                materials_text += f"**[文本]**: {desc}\n\n"
            elif node_type == "Table":
                materials_text += f"**[表格数据]**: {desc}\n\n"
            elif node_type == "CodeBlock":
                materials_text += f"**[代码片段]**: {desc}\n\n"
            elif node_type == "Figure":
                materials_text += f"**[图片说明]**: {desc} *(对应输入图片 {img_counter})*\n\n"
                img_counter += 1  # 仅用于占位提示

        # 收集该页的真实图片路径
        for img in page["resolved_images"]:
            path = img.get("enhanced") or img.get("original")
            if path and os.path.exists(path):
                image_paths.append(path)

    # 3. 组装最终 Prompt
    final_prompt = f"""你是一个专业的机电系统教学助手。请根据以下提供的【知识图谱逻辑】和【参考素材】，准确、清晰地回答用户的问题。

## 🧠 知识图谱支持
{kg_text}
---
## 📚 参考课件素材 (包含文本、表格与图片说明)
{materials_text if materials_text else "无相关参考素材。"}
---
## 🗣️ 用户问题
**{query}**

请综合上述信息给出解答："""

    return final_prompt, image_paths


# ==========================================
# 🧪 4. 本地测试入口
# ==========================================
if __name__ == "__main__":
    test_query = "步进电机驱动器的工作原理是什么？"

    # 1. 检索素材
    materials = retrieve_materials(
        query=test_query,
        vector_top_k=10,
        page_top_k=2,
        concept_expand_depth=2
    )

    # 2. 组装 Prompt
    prompt_text, image_paths = build_vlm_prompt(materials)

    # 3. 打印最终喂给大模型的内容
    print("\n" + "=" * 60)
    print("🤖 最终喂给大模型的 Prompt 如下：")
    print("=" * 60)
    print(prompt_text)

    print("\n" + "=" * 60)
    print("🖼️ 需要随 Prompt 一起上传给 VLM 的图片路径：")
    print("=" * 60)
    for idx, path in enumerate(image_paths, 1):
        print(f"图片 {idx}: {path}")