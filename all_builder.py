import os
import json
import time
import numpy as np
import requests
import re
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase


# ==========================================
# ⚙️ 全局配置区
# ==========================================
# 1. 自动加载环境变量
def load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.env")
    if not os.path.exists(env_path):
        print(f"⚠️ 未找到 {env_path}，将尝试从系统环境变量读取")
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def neo4j_safe_props(obj):
    """Neo4j 属性只支持标量或标量数组，过滤掉嵌套 dict 等非法属性值。"""
    result = {}
    for key, value in (obj or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
            result[key] = value
    return result


# 2. 路径配置
KG_DATA_DIR = r"C:\Users\Lin\Desktop\学习文件--杂项。\复芏计划项目相关\输出结果形式\qwen_qwen3.7-plus_1\qwen_qwen3.7-plus_1\kg_data"
DEBUG_DIR = r"C:\Users\Lin\Desktop\学习文件--杂项。\复芏计划项目相关\输出结果形式\qwen_qwen3.7-plus_1\qwen_qwen3.7-plus_1\debug"
TEMP_DIR = r"C:\Users\Lin\Desktop\学习文件--杂项。\复芏计划项目相关\暂存"

os.makedirs(TEMP_DIR, exist_ok=True)

# 3. 数据库配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = "multimodal-rag-index"

NEO4J_URI = "neo4j+s://fced1671.databases.neo4j.io"
NEO4J_USER = "fced1671"
NEO4J_PASSWORD = "lrM6dVuLrf2UhnF7wS8SDTaFwlb15QHy0VbBWL6AnEI"


# ==========================================
# 🧠 模块一：DeepSeek 知识点抽取 (KG Builder)
# ==========================================
class DeepSeekClient:
    def __init__(self, api_key, base_url):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.url = f"{base_url}/chat/completions"

    def chat(self, prompt: str, max_retries=3) -> str:
        for attempt in range(max_retries):
            try:
                res = requests.post(
                    self.url,
                    headers=self.headers,
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"}  # 强制 JSON 输出
                    },
                    timeout=60
                )
                res.raise_for_status()
                return res.json()['choices'][0]['message']['content']
            except Exception as e:
                print(f"⚠️ DeepSeek API 调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(5)
        return "{}"


def build_knowledge_concepts():
    print("\n" + "=" * 50)
    print("🧠 开始执行任务一：大模型双层知识点抽取")
    print("=" * 50)

    if not DEEPSEEK_API_KEY:
        print("❌ 未找到 DEEPSEEK_API_KEY，请检查 key.env")
        return {"concepts": [], "relationships": []}

    client = DeepSeekClient(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)

    # 1. 读取专业术语库
    terms_path = os.path.join(DEBUG_DIR, "professional_terms_library.json")
    terms = []
    if os.path.exists(terms_path):
        with open(terms_path, 'r', encoding='utf-8') as f:
            terms_data = json.load(f)
        # 该文件顶层是 {"total_terms":..., "terms":[...], ...}
        terms = terms_data.get("terms", []) if isinstance(terms_data, dict) else terms_data
    terms_str = ", ".join(str(t) for t in terms[:50]) + " 等..."  # 截取部分作为提示

    # 2. 读取多模态节点
    nodes_path = os.path.join(KG_DATA_DIR, "multimodal_nodes.json")
    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)

    all_concepts = []
    all_relations = []

    # 3. 分批处理 (每 15 个 Node 一批，防止超长)
    batch_size = 15
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i + batch_size]
        print(f"⏳ 正在处理第 {i // batch_size + 1} 批素材节点 (共 {len(nodes) // batch_size + 1} 批)...")

        # 构造输入文本
        input_text = ""
        for n in batch_nodes:
            desc = n.get('description', '') or n.get('raw_text', '')
            input_text += f"[Node ID: {n['node_id']}] Type: {n['node_type']}\nContent: {desc[:300]}...\n\n"

        prompt = f"""
        你是一个机电系统领域的知识图谱构建专家。请从以下多模态素材（图片描述、文本块）中提取核心教学知识点（Concept）。
        参考专业术语库：{terms_str}

        【素材列表】：
        {input_text}

        【任务要求】：
        1. 提取核心概念（如：输入整形、低通滤波、急停开关等）。
        2. 必须在 evidence_node_ids 中填入支撑该概念的素材 Node ID（从上面的素材列表中找）。
        3. 提取概念之间的关系，关系类型仅限：PREREQUISITE_OF, APPLIES_TO, DEFINES, IS_DEFINED_AS, RELATED_TO。

        【输出格式】（必须是合法的 JSON）：
        {{
            "concepts": [
                {{"name": "概念名称", "description": "概念描述", "evidence_node_ids": ["node_id_1", "node_id_2"]}}
            ],
            "relationships": [
                {{"source": "概念A", "target": "概念B", "relation": "RELATED_TO", "description": "关系说明"}}
            ]
        }}
        """

        response_text = client.chat(prompt)
        try:
            # 提取 JSON
            json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
            data = json.loads(json_str if json_str else response_text)
            all_concepts.extend(data.get('concepts', []))
            all_relations.extend(data.get('relationships', []))
        except Exception as e:
            print(f"❌ 解析 JSON 失败: {e}")

    # 4. 去重合并逻辑 (复用老代码逻辑)
    print("🧹 正在进行知识点去重与合并...")
    unique_concepts = {}
    for c in all_concepts:
        name = (c.get('name') or '').strip()
        if not name:
            continue
        if name not in unique_concepts:
            unique_concepts[name] = c
        else:
            # 合并 evidence_node_ids
            unique_concepts[name]['evidence_node_ids'] = list(set(
                unique_concepts[name].get('evidence_node_ids', []) + c.get('evidence_node_ids', [])
            ))

    unique_relations = []
    seen_rels = set()
    for r in all_relations:
        source = (r.get('source') or '').strip()
        target = (r.get('target') or '').strip()
        relation = (r.get('relation') or '').strip()
        if not source or not target or not relation:
            continue
        key = (source, target, relation)
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relations.append({
                "source": source,
                "target": target,
                "relation": relation,
                "description": r.get('description', '')
            })

    final_data = {
        "concepts": list(unique_concepts.values()),
        "relationships": unique_relations
    }

    # 5. 保存到本地暂存区
    save_path = os.path.join(TEMP_DIR, "concepts_graph.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 知识点抽取完成！共 {len(unique_concepts)} 个概念，{len(unique_relations)} 条关系。")
    print(f"💾 已保存至本地：{save_path}")

    return final_data


# ==========================================
# 🕸️ 模块二：Neo4j 双层图谱入库
# ==========================================
def upload_to_neo4j(concepts_data):
    print("\n" + "=" * 50)
    print("🚀 开始执行任务二：双层图数据入库 (Neo4j)")
    print("=" * 50)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def run_query(query, parameters=None):
        with driver.session() as session:
            session.run(query, parameters)

    # 0. 建课程根节点，素材层关系中会用到 default_course 作为端点
    manifest_path = os.path.join(KG_DATA_DIR, "course_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        run_query("MERGE (n:Course {id: $id}) SET n.course_name = $name",
                  {"id": manifest.get("course_id", "default_course"),
                   "name": manifest.get("course_name", "")})

    # 1. 导入底层：素材层 (Document, Page, Multimodal Nodes)
    print("📦 [1/4] 正在导入底层素材节点...")

    # 导入 Documents
    with open(os.path.join(KG_DATA_DIR, "documents.json"), 'r', encoding='utf-8') as f:
        for doc in json.load(f):
            run_query("MERGE (n:Document {id: $id}) SET n += $props",
                      {"id": doc["document_id"], "props": neo4j_safe_props(doc)})

    # 导入 Pages
    with open(os.path.join(KG_DATA_DIR, "pages.json"), 'r', encoding='utf-8') as f:
        for page in json.load(f):
            run_query("MERGE (n:Page {id: $id}) SET n += $props",
                      {"id": page["page_id"], "props": neo4j_safe_props(page)})

    # 导入 Nodes (Figure, TextChunk, etc.)
    with open(os.path.join(KG_DATA_DIR, "multimodal_nodes.json"), 'r', encoding='utf-8') as f:
        for node in json.load(f):
            label = node["node_type"]
            run_query(f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                      {"id": node["node_id"], "props": neo4j_safe_props(node)})

    # 导入素材层关系
    print("🔗 [2/4] 正在建立素材层物理关系...")
    with open(os.path.join(KG_DATA_DIR, "relationships.json"), 'r', encoding='utf-8') as f:
        for rel in json.load(f):
            rel_type = rel["relationship_type"]
            run_query(f"""
                MATCH (s {{id: $source_id}}), (t {{id: $target_id}})
                MERGE (s)-[r:{rel_type}]->(t)
            """, {"source_id": rel["source_id"], "target_id": rel["target_id"]})

    # 2. 导入顶层：知识层 (Concepts)
    print("🧠 [3/4] 正在导入顶层知识点节点与逻辑关系...")
    for concept in concepts_data.get('concepts', []):
        run_query("""
            MERGE (c:Concept {name: $name})
            SET c.description = $desc, c.id = $name
        """, {"name": concept['name'], "desc": concept.get('description', '')})

    for rel in concepts_data.get('relationships', []):
        rel_type = rel['relation']
        run_query(f"""
            MATCH (s:Concept {{name: $source}}), (t:Concept {{name: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.description = $desc
        """, {"source": rel['source'], "target": rel['target'], "desc": rel.get('description', '')})

    # 3. 缝合双层：跨层桥梁 (HAS_EVIDENCE)
    print("🌉 [4/4] 正在缝合双层图谱 (建立跨层证据链接)...")
    bridge_count = 0
    for concept in concepts_data.get('concepts', []):
        concept_name = concept['name']
        evidence_ids = concept.get('evidence_node_ids', [])
        for ev_id in evidence_ids:
            run_query("""
                MATCH (c:Concept {name: $c_name}), (m {id: $ev_id})
                MERGE (c)-[:HAS_EVIDENCE]->(m)
            """, {"c_name": concept_name, "ev_id": ev_id})
            bridge_count += 1

    print(f"🎉 图数据入库完成！成功建立 {bridge_count} 条跨层关联。")
    driver.close()


# ==========================================
# 🌲 模块三：Pinecone 向量入库
# ==========================================
def upload_to_pinecone():
    print("\n" + "=" * 50)
    print("🚀 开始执行任务三：向量数据入库 (Pinecone)")
    print("=" * 50)

    if not PINECONE_API_KEY:
        print("❌ 未找到 PINECONE_API_KEY，请检查 key.env")
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"⏳ 创建索引 '{PINECONE_INDEX_NAME}' (384维)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)

    index = pc.Index(PINECONE_INDEX_NAME)

    matrix_path = os.path.join(KG_DATA_DIR, "vectors", "vector_matrix.npy")
    index_path = os.path.join(KG_DATA_DIR, "vectors", "vector_index.json")

    vectors_matrix = np.load(matrix_path)
    with open(index_path, 'r', encoding='utf-8') as f:
        vectors_info = json.load(f)['items']

    upload_batch = []
    for i, info in enumerate(vectors_info):
        text_val = info["embedding_text"]
        upload_batch.append({
            "id": info["embedding_id"],
            "values": vectors_matrix[i].tolist(),
            "metadata": {
                "node_id": info["node_id"],
                "node_type": info["node_type"],
                "text": text_val[:1200] if len(text_val) > 1200 else text_val
            }
        })

    index.upsert(vectors=upload_batch)
    print(f"🎉 成功上传 {len(upload_batch)} 条向量数据到 Pinecone！")


# ==========================================
# 🚀 主程序入口
# ==========================================
if __name__ == "__main__":
    print("🌟 启动多模态双层图谱构建流水线...")

    # 1. 抽取知识点 (如果暂存区已经有，可以注释掉这行直接读本地)
    concepts_data = build_knowledge_concepts()

    # 如果想跳过抽取直接读本地文件，取消下面两行的注释：
    # with open(os.path.join(TEMP_DIR, "concepts_graph.json"), 'r', encoding='utf-8') as f:
    #     concepts_data = json.load(f)

    # 2. 图数据库入库
    upload_to_neo4j(concepts_data)

    # 3. 向量数据库入库
    upload_to_pinecone()

    print("\n" + "=" * 50)
    print("✨ All-in-One 任务圆满结束！你的双层教学知识图谱已准备就绪！")
    print("=" * 50)
