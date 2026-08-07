import os
import json
import time
import random
import numpy as np
import requests
import re
from typing import List, Dict, Set
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase

# ==========================================
# ⚙️ 全局配置区
# ==========================================
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

# 路径配置
KG_DATA_DIR = r"C:\Users\Lin\Desktop\学习文件--杂项。\复芏计划项目相关\输出结果形式\qwen_qwen3.7-plus_1\qwen_qwen3.7-plus_1\kg_data"
DEBUG_DIR = r"C:\Users\Lin\Desktop\学习文件--杂项。\复芏计划项目相关\输出结果形式\qwen_qwen3.7-plus_1\qwen_qwen3.7-plus_1\debug"
TEMP_DIR = r"C:\Users\Lin\Desktop\学习文件--杂项。\复芏计划项目相关\暂存"
os.makedirs(TEMP_DIR, exist_ok=True)

# 【优化⑦】所有密钥统一从 key.env 读取，不再硬编码
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = "multimodal-rag-index"
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://fced1671.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "fced1671")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# 【优化④】断点续跑用的中间文件（每批抽取完立即追加保存）
PARTIAL_SAVE_PATH = os.path.join(TEMP_DIR, "concepts_partial.jsonl")
FINAL_SAVE_PATH = os.path.join(TEMP_DIR, "concepts_graph.json")

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
                    self.url, headers=self.headers,
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=90
                )
                res.raise_for_status()
                return res.json()['choices'][0]['message']['content']
            except requests.exceptions.HTTPError as e:
                # 【优化⑤】429限流单独处理，等待更久
                if e.response is not None and e.response.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"⚠️ 触发限流，等待 {wait}秒...")
                    time.sleep(wait)
                else:
                    print(f"⚠️ HTTP错误 (尝试 {attempt+1}/{max_retries}): {e}")
                    time.sleep(5)
            except Exception as e:
                print(f"⚠️ DeepSeek API 调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(5)
        return "{}"


def _smart_truncate(node: Dict, text_limit=800, other_limit=500) -> str:
    """【优化②】根据节点类型智能截断长度，文本块给更多空间"""
    desc = node.get('description', '') or node.get('raw_text', '') or ''
    limit = text_limit if node.get('node_type') == 'TextChunk' else other_limit
    if len(desc) > limit:
        return desc[:limit] + "..."
    return desc


def _load_processed_node_ids() -> Set[str]:
    """【优化④】读取已完成批次涉及的 node_id，用于断点续跑判断"""
    processed = set()
    if os.path.exists(PARTIAL_SAVE_PATH):
        with open(PARTIAL_SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    batch_record = json.loads(line)
                    processed.update(batch_record.get('batch_node_ids', []))
                except Exception:
                    continue
    return processed


def _append_partial_result(batch_node_ids: List[str], concepts: List[Dict], relations: List[Dict]):
    """【优化④】每批抽取完立即追加落盘，防止 API 费用打水漂"""
    with open(PARTIAL_SAVE_PATH, 'a', encoding='utf-8') as f:
        record = {
            "batch_node_ids": batch_node_ids,
            "concepts": concepts,
            "relationships": relations
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_knowledge_concepts(resume: bool = True):
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
        terms = terms_data.get("terms", []) if isinstance(terms_data, dict) else terms_data
    terms_str = ", ".join(str(t) for t in terms[:50]) + " 等..."

    # 2. 读取多模态节点
    nodes_path = os.path.join(KG_DATA_DIR, "multimodal_nodes.json")
    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    valid_node_ids = {n['node_id'] for n in nodes}  # 【优化③】用于后续幻觉校验

    # 【优化④】断点续跑：跳过已处理过的节点所在批次
    processed_ids = _load_processed_node_ids() if resume else set()
    if processed_ids:
        print(f"♻️ 检测到断点续跑记录，已处理过 {len(processed_ids)} 个节点，将跳过对应批次")

    all_concepts = []
    all_relations = []
    batch_size = 15
    total_batches = (len(nodes) + batch_size - 1) // batch_size

    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i + batch_size]
        batch_ids = [n['node_id'] for n in batch_nodes]
        batch_no = i // batch_size + 1

        # 跳过已处理批次
        if processed_ids and all(bid in processed_ids for bid in batch_ids):
            print(f"⏭️ 第 {batch_no}/{total_batches} 批已处理过，跳过")
            continue

        print(f"⏳ 正在处理第 {batch_no}/{total_batches} 批素材节点...")

        input_text = ""
        for n in batch_nodes:
            desc = _smart_truncate(n)
            input_text += f"[Node ID: {n['node_id']}] Type: {n['node_type']}\nContent: {desc}\n\n"

        prompt = f"""
        你是一个机电系统领域的知识图谱构建专家。请从以下多模态素材（图片描述、文本块）中提取核心教学知识点（Concept）。
        参考专业术语库：{terms_str}

        【素材列表】：
        {input_text}

        【任务要求】：
        1. 提取核心概念（如：输入整形、低通滤波、急停开关等）。
        2. evidence_node_ids 必须**严格**从上面素材列表的 [Node ID: ...] 中原样复制，禁止编造不存在的 ID。
        3. 提取概念之间的关系，关系类型仅限：PREREQUISITE_OF, APPLIES_TO, DEFINES, IS_DEFINED_AS, RELATED_TO。
        4. 如果某素材不构成任何有意义的知识点，直接忽略，不要为了凑数编造。

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
        batch_concepts, batch_relations = [], []
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            json_str = json_match.group() if json_match else response_text
            data = json.loads(json_str)
            batch_concepts = data.get('concepts', [])
            batch_relations = data.get('relationships', [])

            # 【优化③】幻觉校验：过滤掉引用了不存在 node_id 的证据
            for c in batch_concepts:
                original_ev = c.get('evidence_node_ids', [])
                valid_ev = [eid for eid in original_ev if eid in valid_node_ids]
                dropped = len(original_ev) - len(valid_ev)
                if dropped > 0:
                    print(f"  ⚠️ 概念「{c.get('name')}」有 {dropped} 个幻觉 evidence_id 被过滤")
                c['evidence_node_ids'] = valid_ev

            all_concepts.extend(batch_concepts)
            all_relations.extend(batch_relations)

        except Exception as e:
            print(f"❌ 解析 JSON 失败: {e}")

        # 【优化④】立即落盘，防止中途崩溃丢失已抽取的结果
        _append_partial_result(batch_ids, batch_concepts, batch_relations)

        # 【优化⑤】批次间限流保护
        if i + batch_size < len(nodes):
            delay = random.uniform(1.5, 3)
            time.sleep(delay)

    # 3. 从 partial 文件重新汇总全部结果（包含本次和历史断点续跑的）
    all_concepts, all_relations = [], []
    if os.path.exists(PARTIAL_SAVE_PATH):
        with open(PARTIAL_SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                all_concepts.extend(record.get('concepts', []))
                all_relations.extend(record.get('relationships', []))

    # 4. 去重合并逻辑（【优化⑥】合并时拼接多条描述，而非丢弃）
    print("🧹 正在进行知识点去重与合并...")
    unique_concepts = {}
    for c in all_concepts:
        name = (c.get('name') or '').strip()
        if not name:
            continue
        if name not in unique_concepts:
            unique_concepts[name] = {
                "name": name,
                "description": c.get('description', ''),
                "evidence_node_ids": list(set(c.get('evidence_node_ids', [])))
            }
        else:
            existing = unique_concepts[name]
            existing['evidence_node_ids'] = list(set(
                existing['evidence_node_ids'] + c.get('evidence_node_ids', [])
            ))
            new_desc = c.get('description', '').strip()
            if new_desc and new_desc not in existing['description']:
                existing['description'] = (existing['description'] + " | " + new_desc)[:500]

    # 【优化③】关系幻觉校验：source/target 必须存在于 concepts 列表中
    valid_concept_names = set(unique_concepts.keys())
    unique_relations = []
    seen_rels = set()
    dropped_rel_count = 0
    for r in all_relations:
        source = (r.get('source') or '').strip()
        target = (r.get('target') or '').strip()
        relation = (r.get('relation') or '').strip()
        if not source or not target or not relation:
            continue
        if source not in valid_concept_names or target not in valid_concept_names:
            dropped_rel_count += 1
            continue
        key = (source, target, relation)
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relations.append({
                "source": source, "target": target,
                "relation": relation, "description": r.get('description', '')
            })

    if dropped_rel_count > 0:
        print(f"⚠️ 过滤掉 {dropped_rel_count} 条引用了不存在概念的幻觉关系")

    final_data = {
        "concepts": list(unique_concepts.values()),
        "relationships": unique_relations
    }

    with open(FINAL_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 知识点抽取完成！共 {len(unique_concepts)} 个概念，{len(unique_relations)} 条关系。")
    print(f"💾 已保存至本地：{FINAL_SAVE_PATH}")

    return final_data


# ==========================================
# 🕸️ 模块二：Neo4j 双层图谱入库（批量优化版）
# ==========================================
def upload_to_neo4j(concepts_data):
    print("\n" + "=" * 50)
    print("🚀 开始执行任务二：双层图数据入库 (Neo4j)")
    print("=" * 50)

    if not NEO4J_PASSWORD:
        print("❌ 未找到 NEO4J_PASSWORD，请检查 key.env")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 【优化①】统一改成"一个 Session + UNWIND 批量提交"，大幅减少网络往返
    def run_unwind(query: str, batch_list: List[Dict], label: str = ""):
        if not batch_list:
            return
        with driver.session() as session:
            session.run(query, {"batch": batch_list})
        print(f"  ✓ {label}: {len(batch_list)} 条已批量写入")

    def run_single(query, parameters=None):
        with driver.session() as session:
            session.run(query, parameters)

    # 0. 课程根节点
    manifest_path = os.path.join(KG_DATA_DIR, "course_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        run_single("MERGE (n:Course {id: $id}) SET n.course_name = $name",
                  {"id": manifest.get("course_id", "default_course"),
                   "name": manifest.get("course_name", "")})

    # 1. 批量导入 Document / Page / Multimodal Nodes
    print("📦 [1/4] 正在批量导入底层素材节点...")

    with open(os.path.join(KG_DATA_DIR, "documents.json"), 'r', encoding='utf-8') as f:
        docs = [{"id": d["document_id"], "props": neo4j_safe_props(d)} for d in json.load(f)]
    run_unwind("""
        UNWIND $batch AS row
        MERGE (n:Document {id: row.id})
        SET n += row.props
    """, docs, "Document 节点")

    with open(os.path.join(KG_DATA_DIR, "pages.json"), 'r', encoding='utf-8') as f:
        pages = [{"id": p["page_id"], "props": neo4j_safe_props(p)} for p in json.load(f)]
    run_unwind("""
        UNWIND $batch AS row
        MERGE (n:Page {id: row.id})
        SET n += row.props
    """, pages, "Page 节点")

    # 多模态节点按 node_type 分组批量导入（因为动态 Label 无法用参数化，需要分组）
    with open(os.path.join(KG_DATA_DIR, "multimodal_nodes.json"), 'r', encoding='utf-8') as f:
        all_nodes = json.load(f)
    nodes_by_type: Dict[str, List[Dict]] = {}
    for n in all_nodes:
        label = n["node_type"]
        nodes_by_type.setdefault(label, []).append(
            {"id": n["node_id"], "props": neo4j_safe_props(n)}
        )
    for label, batch in nodes_by_type.items():
        run_unwind(f"""
            UNWIND $batch AS row
            MERGE (n:{label} {{id: row.id}})
            SET n += row.props
        """, batch, f"{label} 节点")

    # 2. 批量导入素材层关系（按 relationship_type 分组）
    print("🔗 [2/4] 正在批量建立素材层物理关系...")
    with open(os.path.join(KG_DATA_DIR, "relationships.json"), 'r', encoding='utf-8') as f:
        all_rels = json.load(f)
    rels_by_type: Dict[str, List[Dict]] = {}
    for r in all_rels:
        rtype = r["relationship_type"]
        rels_by_type.setdefault(rtype, []).append(
            {"source_id": r["source_id"], "target_id": r["target_id"]}
        )
    for rtype, batch in rels_by_type.items():
        run_unwind(f"""
            UNWIND $batch AS row
            MATCH (s {{id: row.source_id}}), (t {{id: row.target_id}})
            MERGE (s)-[r:{rtype}]->(t)
        """, batch, f"{rtype} 关系")

    # 3. 批量导入知识层 Concept 节点
    print("🧠 [3/4] 正在批量导入顶层知识点节点与逻辑关系...")
    concept_batch = [
        {"name": c['name'], "desc": c.get('description', '')}
        for c in concepts_data.get('concepts', [])
    ]
    run_unwind("""
        UNWIND $batch AS row
        MERGE (c:Concept {name: row.name})
        SET c.description = row.desc, c.id = row.name
    """, concept_batch, "Concept 节点")

    # 知识层关系按 relation 分组批量导入
    rel_batch_by_type: Dict[str, List[Dict]] = {}
    for r in concepts_data.get('relationships', []):
        rel_batch_by_type.setdefault(r['relation'], []).append(
            {"source": r['source'], "target": r['target'], "desc": r.get('description', '')}
        )
    for rtype, batch in rel_batch_by_type.items():
        run_unwind(f"""
            UNWIND $batch AS row
            MATCH (s:Concept {{name: row.source}}), (t:Concept {{name: row.target}})
            MERGE (s)-[r:{rtype}]->(t)
            SET r.description = row.desc
        """, batch, f"Concept关系-{rtype}")

    # 4. 批量缝合双层：HAS_EVIDENCE
    print("🌉 [4/4] 正在批量缝合双层图谱 (建立跨层证据链接)...")
    bridge_batch = []
    for concept in concepts_data.get('concepts', []):
        for ev_id in concept.get('evidence_node_ids', []):
            bridge_batch.append({"c_name": concept['name'], "ev_id": ev_id})
    run_unwind("""
        UNWIND $batch AS row
        MATCH (c:Concept {name: row.c_name}), (m {id: row.ev_id})
        MERGE (c)-[:HAS_EVIDENCE]->(m)
    """, bridge_batch, "跨层证据链接")

    # 【优化⑨】最终统计报告
    print("\n📊 正在生成入库质量报告...")
    with driver.session() as session:
        total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        concept_count = session.run("MATCH (c:Concept) RETURN count(c) AS c").single()["c"]
        evidence_count = session.run("MATCH ()-[r:HAS_EVIDENCE]->() RETURN count(r) AS c").single()["c"]

    print("=" * 50)
    print(f"  图谱总节点数: {total_nodes}")
    print(f"  图谱总关系数: {total_rels}")
    print(f"  知识点(Concept)数量: {concept_count}")
    print(f"  跨层证据链接数量: {evidence_count}")
    print("=" * 50)

    driver.close()
    print("🎉 图数据入库完成！")


# ==========================================
# 🌲 模块三：Pinecone 向量入库（分批优化版）
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

    # 【优化⑧】分批上传，每批 100 条，避免未来数据量增长后单次请求过大
    chunk_size = 100
    total_uploaded = 0
    for i in range(0, len(upload_batch), chunk_size):
        chunk = upload_batch[i:i + chunk_size]
        index.upsert(vectors=chunk)
        total_uploaded += len(chunk)
        print(f"  ✓ 已上传 {total_uploaded}/{len(upload_batch)} 条向量")

    print(f"🎉 成功上传 {total_uploaded} 条向量数据到 Pinecone！")


# ==========================================
# 🚀 主程序入口
# ==========================================
if __name__ == "__main__":
    print("🌟 启动多模态双层图谱构建流水线...")

    # 1. 抽取知识点（默认开启断点续跑；若想强制重新抽取，删除 TEMP_DIR 下的 concepts_partial.jsonl 即可）
    concepts_data = build_knowledge_concepts(resume=True)

    # 如果想跳过抽取直接读本地已生成的最终文件，取消下面两行注释：
    # with open(FINAL_SAVE_PATH, 'r', encoding='utf-8') as f:
    #     concepts_data = json.load(f)

    # 2. 图数据库入库
    upload_to_neo4j(concepts_data)

    # 3. 向量数据库入库
    upload_to_pinecone()

    print("\n" + "=" * 50)
    print("✨ All-in-One 任务圆满结束！你的双层教学知识图谱已准备就绪！")
    print("=" * 50)